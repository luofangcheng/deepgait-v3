"""FTIR footprint analysis — v2.1 red-background pipeline.

Supports FTIR systems with:
- **Red backlight runway** (R ≫ G, B) — animal body blocks red light,
  paw contacts reflect green light (green intensity ∝ pressure).
- Two-stage detection: red-channel background subtraction → green
  channel enhancement within body silhouette.

Builds on the v1 :mod:`deepgait3.core._legacy.footprint` module by adding:

1. **Red-background body/paw segmentation** (deepgait v2.1)
   Stage 1: ``bg_R - frame_R > body_threshold`` → body mask.
   Stage 2: ``frame_G - bg_G > paw_threshold`` within body → paw mask.

2. **Union-find 4-paw grouping** (kb/09 §5.2)
   After connected-components labelling, individual toes/fingers are
   merged into one ``FootMask`` when their centroid distance is below
   ``max_finger_distance_px``.

3. **L/R + F/H classification** (kb/09 §5.3)
   Each ``FootMask`` centroid is projected onto the body axis
   (``DistanceFromLine``). The (parallel, perpendicular) components map
   to quadrants: RF / RH / LF / LH.
   Body axis can be estimated automatically from paw centroids via PCA.

4. **Cross-frame paw identity tracking** (deepgait v2.1)
   :class:`PawTracker` uses Hungarian algorithm to maintain consistent
   paw identities across frames, reducing label flicker.

5. **Per-frame :class:`FootprintSequence` output** (kb/09 §5.1)
   - ``area_px``, ``area_mm²``
   - ``intensity_mean / max / min / total`` (from green channel = pressure)
   - ``centroid`` (x, y)
   - ``bbox`` (x, y, w, h)
   - ``hull_area_px``
   - ``is_in_stance`` (area > threshold)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Legacy HSV thresholds (retained for backward compatibility with green-
# illuminated CatWalk-style systems; NOT used in the default red-background
# pipeline).
DEFAULT_HSV_LOWER: Tuple[int, int, int] = (35, 50, 30)
DEFAULT_HSV_UPPER: Tuple[int, int, int] = (85, 255, 255)

# Red-background system defaults (deepgait v2.1).
#   body_threshold: min red-channel drop for animal body detection.
#   paw_threshold:  min green-channel rise for paw contact detection.
DEFAULT_BODY_THRESHOLD: int = 40    # bg_R - frame_R > 40 → body pixel
DEFAULT_PAW_THRESHOLD: int = 12     # absolute green > 12 → paw pixel (bg G≈13)
DEFAULT_BG_OFFSET: int = 15         # legacy (used by BackgroundModel)
DEFAULT_MIN_AREA_PX: int = 50
DEFAULT_MAX_FINGER_DISTANCE_PX: int = 25   # MouseWalker: ~3 mm @ 8 px/mm
DEFAULT_IN_STANCE_AREA_PX: int = 30        # area threshold to mark as in_stance

# Tracking: max centroid distance (px) for same-paw matching between frames.
DEFAULT_TRACKING_MAX_DISTANCE_PX: float = 50.0
# Number of frames to keep in the tracking history.
DEFAULT_TRACKING_HISTORY: int = 5


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FootMask:
    """One merged paw footprint (after union-find grouping)."""
    label: int                         # unique id within the frame
    area_px: int
    area_mm2: float
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]    # x, y, w, h
    hull_area_px: int
    intensity_mean: float
    intensity_max: int
    intensity_min: int
    intensity_total: float
    is_in_stance: bool
    matched_paw: Optional[str] = None  # one of {LF, RF, LH, RH} or None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "area_px": self.area_px,
            "area_mm2": self.area_mm2,
            "centroid": list(self.centroid),
            "bbox": list(self.bbox),
            "hull_area_px": self.hull_area_px,
            "intensity_mean": self.intensity_mean,
            "intensity_max": self.intensity_max,
            "intensity_min": self.intensity_min,
            "intensity_total": self.intensity_total,
            "is_in_stance": self.is_in_stance,
            "matched_paw": self.matched_paw,
        }


@dataclass(slots=True)
class FootprintSequence:
    """One frame's worth of foot contacts, classified LF/RF/LH/RH."""
    timestamp_ms: float = 0.0
    fps: int = 100
    feet: Dict[str, FootMask] = field(default_factory=dict)  # paw → FootMask
    n_feet: int = 0
    # Every detected foot BEFORE classification (and before the 4-quadrant
    # gate). Live accumulators iterate this so faint/uncategorised contacts
    # are not silently dropped. Kept separate from ``feet`` so downstream
    # gait metrics (which expect exactly LF/RF/LH/RH) are unaffected.
    all_feet: List[FootMask] = field(default_factory=list)

    @property
    def in_stance(self) -> Dict[str, bool]:
        return {paw: fm.is_in_stance for paw, fm in self.feet.items()}

    @property
    def areas_px(self) -> Dict[str, int]:
        return {paw: fm.area_px for paw, fm in self.feet.items()}

    @property
    def intensity_totals(self) -> Dict[str, float]:
        return {paw: fm.intensity_total for paw, fm in self.feet.items()}

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "fps": self.fps,
            "n_feet": self.n_feet,
            "feet": {p: fm.to_dict() for p, fm in self.feet.items()},
            "all_feet": [fm.to_dict() for fm in self.all_feet],
        }


# ---------------------------------------------------------------------------
# BackgroundModel — MouseWalker style |frame - median| ≤ BGoffset
# ---------------------------------------------------------------------------
class BackgroundModel:
    """Per-channel median background model over a rolling frame window.

    MouseWalker (kb/09 §5.1) computes the background as the per-channel
    median of the most recent N frames. A pixel is "foreground" when
    ``|frame - median| > BGoffset`` for any channel. This is robust to
    per-channel lighting drift and slow LED intensity changes.
    """

    def __init__(
        self,
        window: int = 30,
        bg_offset: int = DEFAULT_BG_OFFSET,
        warmup_frames: int = 5,
    ) -> None:
        if window < 3:
            raise ValueError("window must be >= 3")
        if bg_offset < 0:
            raise ValueError("bg_offset must be >= 0")
        self.window = int(window)
        self.bg_offset = int(bg_offset)
        self.warmup_frames = int(warmup_frames)
        self._buffer: deque = deque(maxlen=self.window)
        self._median: Optional[np.ndarray] = None

    @property
    def is_ready(self) -> bool:
        return self._median is not None

    @property
    def n_frames_seen(self) -> int:
        return len(self._buffer)

    def get_median(self) -> Optional[np.ndarray]:
        """Return the current background median image (BGR uint8).

        Returns ``None`` before the model is ready.
        """
        return self._median

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Add a frame to the rolling window and return the new median."""
        if frame.dtype != np.uint8:
            raise ValueError(f"frame must be uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"frame must be H×W×3, got {frame.shape}")
        self._buffer.append(frame.astype(np.int16))
        if len(self._buffer) >= self.warmup_frames:
            self._median = np.median(np.stack(self._buffer, axis=0),
                                      axis=0).astype(np.uint8)
        return self._median if self._median is not None else np.zeros_like(frame)

    def foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a binary mask of pixels deviating from the background."""
        if not self.is_ready:
            raise RuntimeError("BackgroundModel: needs more frames to be ready")
        diff = np.abs(frame.astype(np.int16) - self._median.astype(np.int16))
        # Per-channel max deviation > offset → foreground.
        return (diff.max(axis=2) > self.bg_offset).astype(np.uint8) * 255

    def reset(self) -> None:
        self._buffer.clear()
        self._median = None


# ---------------------------------------------------------------------------
# Union-find group — MouseWalker PartOfFoot
# ---------------------------------------------------------------------------
class _UnionFind:
    """Tiny union-find for grouping foot fingers."""
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out


def group_fingers_into_feet(
    blobs: List[Tuple[int, Tuple[int, int, int, int], Tuple[float, float], int]],
    max_finger_distance_px: int = DEFAULT_MAX_FINGER_DISTANCE_PX,
) -> List[List[Tuple[int, Tuple[int, int, int, int], Tuple[float, float], int]]]:
    """Merge connected-component blobs whose centroids are close enough.

    Parameters
    ----------
    blobs : list of (label, bbox, centroid, area_px)
        Output of cv2.connectedComponentsWithStats (per-component slice).
    max_finger_distance_px : int
        Two blobs whose centroid distance is below this are merged into
        the same paw. Mirrors MouseWalker's ``MaxFingerDistance`` knob.

    Returns
    -------
    list of "foot groups", each a list of the input blobs that belong
    to the same paw (single-blob paws are wrapped in a 1-element group).
    """
    if not blobs:
        return []
    uf = _UnionFind(len(blobs))
    centroids = np.array([b[2] for b in blobs], dtype=float)
    n = len(blobs)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centroids[i] - centroids[j])
            if d <= max_finger_distance_px:
                uf.union(i, j)
    groups = uf.groups()
    return [ [blobs[i] for i in idxs] for idxs in groups.values() ]


# ---------------------------------------------------------------------------
# merge_group_into_FootMask — turn a union-find group into one FootMask
# ---------------------------------------------------------------------------
def _merge_group(
    group: List[Tuple[int, Tuple[int, int, int, int], Tuple[float, float], int]],
    intensity_frame: np.ndarray,
    px_per_mm: Optional[float],
    in_stance_threshold_px: int,
    new_label: int,
) -> FootMask:
    """Combine multiple finger blobs into a single FootMask."""
    # Bounding box is the union.
    x0 = min(b[1][0] for b in group)
    y0 = min(b[1][1] for b in group)
    x1 = max(b[1][0] + b[1][2] for b in group)
    y1 = max(b[1][1] + b[1][3] for b in group)
    bbox = (x0, y0, x1 - x0, y1 - y0)

    # Area-weighted centroid.
    total_area = sum(b[3] for b in group)
    if total_area == 0:
        centroid = (float(x0), float(y0))
    else:
        cx = sum(b[2][0] * b[3] for b in group) / total_area
        cy = sum(b[2][1] * b[3] for b in group) / total_area
        centroid = (float(cx), float(cy))

    # Intensity stats over the merged footprint.
    pad = 1
    h, w = intensity_frame.shape[:2]
    rx0, ry0 = max(x0 - pad, 0), max(y0 - pad, 0)
    rx1, ry1 = min(x1 + pad, w), min(y1 + pad, h)
    roi = intensity_frame[ry0:ry1, rx0:rx1]
    if intensity_frame.ndim == 3:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.size else roi
    else:
        gray_roi = roi
    footprint_mask = np.zeros((ry1 - ry0, rx1 - rx0), dtype=np.uint8)
    for b in group:
        bx0 = b[1][0] - rx0
        by0 = b[1][1] - ry0
        bx1 = bx0 + b[1][2]
        by1 = by0 + b[1][3]
        footprint_mask[by0:by1, bx0:bx1] = 255
    intensity_values = gray_roi[footprint_mask > 0]
    if intensity_values.size > 0:
        intensity_mean = float(np.mean(intensity_values))
        intensity_max = int(np.max(intensity_values))
        intensity_min = int(np.min(intensity_values))
        intensity_total = float(np.sum(intensity_values))
    else:
        intensity_mean = intensity_max = intensity_min = 0
        intensity_total = 0.0

    # Convex hull area.
    contours, _ = cv2.findContours(footprint_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = np.vstack(contours)
        hull = cv2.convexHull(all_pts)
        hull_area = int(cv2.contourArea(hull))
    else:
        hull_area = int(total_area)

    area_mm2 = total_area / (px_per_mm ** 2) if px_per_mm else float(total_area)
    return FootMask(
        label=new_label,
        area_px=int(total_area),
        area_mm2=area_mm2,
        centroid=centroid,
        bbox=bbox,
        hull_area_px=hull_area,
        intensity_mean=intensity_mean,
        intensity_max=intensity_max,
        intensity_min=intensity_min,
        intensity_total=intensity_total,
        is_in_stance=int(total_area) >= in_stance_threshold_px,
        matched_paw=None,
    )


# ---------------------------------------------------------------------------
# Dynamic body axis estimation from paw centroids
# ---------------------------------------------------------------------------
def estimate_body_axis(
    centroids: List[Tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Estimate the body (walking) axis from a set of paw centroids using PCA.

    When fewer than 2 centroids are available, falls back to a horizontal
    axis centred in the frame (the common walking direction on a linear
    runway).

    Parameters
    ----------
    centroids : list of (x, y)
        Paw centroid positions (can be from one or multiple frames).
    frame_width, frame_height : int
        Frame dimensions for fallback axis placement.

    Returns
    -------
    (p1, p2) : ((x1, y1), (x2, y2))
        Two points defining the body axis.  p1 → p2 points in the
        estimated walking direction.
    """
    if len(centroids) < 2:
        # Fallback: horizontal axis centred in frame.
        cy = frame_height / 2.0
        return ((frame_width * 0.1, cy), (frame_width * 0.9, cy))

    pts = np.array(centroids, dtype=np.float64)
    mean = pts.mean(axis=0)
    centred = pts - mean
    cov = centred.T @ centred / (len(pts) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Primary axis = eigenvector with largest eigenvalue.
    principal = eigenvectors[:, np.argmax(eigenvalues)]

    # Project all points onto the principal axis to find extent.
    proj = centred @ principal
    t_min, t_max = float(proj.min()), float(proj.max())
    # Add 10% padding.
    pad = (t_max - t_min) * 0.1 if t_max > t_min else frame_width * 0.1
    t_min -= pad
    t_max += pad
    p1 = (float(mean[0] + t_min * principal[0]),
          float(mean[1] + t_min * principal[1]))
    p2 = (float(mean[0] + t_max * principal[0]),
          float(mean[1] + t_max * principal[1]))
    return (p1, p2)


# ---------------------------------------------------------------------------
# PawTracker — cross-frame identity tracking via Hungarian algorithm
# ---------------------------------------------------------------------------
class PawTracker:
    """Maintain consistent paw identities across frames.

    Uses the Hungarian (Kuhn-Munkres) algorithm to match paw centroids
    between consecutive frames based on Euclidean distance.  Maintains a
    short history to resolve temporary occlusions.
    """

    def __init__(
        self,
        max_distance_px: float = DEFAULT_TRACKING_MAX_DISTANCE_PX,
        history_frames: int = DEFAULT_TRACKING_HISTORY,
    ) -> None:
        self.max_distance_px = max_distance_px
        self.history_frames = history_frames
        # paw_name → deque of (centroid_x, centroid_y) over recent frames
        self._history: Dict[str, deque] = {}
        # paw_name → FootMask from the most recent frame
        self._last_feet: Dict[str, FootMask] = {}
        self._next_id: int = 0

    @staticmethod
    def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
        """Simple greedy Hungarian-style assignment for small matrices (≤ 6×6).

        For the 4-paw case the cost matrix is tiny; a full Hungarian
        implementation is unnecessary.  Greedy minimum-cost matching with
        duplicate row/col guards is sufficient.
        """
        n_rows, n_cols = cost.shape
        if n_rows == 0 or n_cols == 0:
            return []
        assignments: List[Tuple[int, int]] = []
        used_cols: set = set()
        # Sort all (row, col, cost) by cost ascending.
        candidates = sorted(
            [(r, c, cost[r, c]) for r in range(n_rows) for c in range(n_cols)],
            key=lambda x: x[2],
        )
        rows_assigned: set = set()
        for r, c, _ in candidates:
            if r not in rows_assigned and c not in used_cols:
                assignments.append((r, c))
                rows_assigned.add(r)
                used_cols.add(c)
        return assignments

    def update(self, feet: Dict[str, FootMask]) -> Dict[str, FootMask]:
        """Match incoming ``feet`` to tracked identities.

        Parameters
        ----------
        feet : dict of ``{tentative_name: FootMask}``
            Output from ``classify_feet_lfrh`` (or raw labels).

        Returns
        -------
        dict of ``{tracked_name: FootMask}`` with consistent identities.
        Paws that could not be matched retain their input names.
        """
        if not feet:
            # Update history with empty frame (paws may have lifted).
            return {}

        new_names = list(feet.keys())
        old_names = list(self._history.keys())

        if not old_names:
            # First frame: initialise history.
            for name, fm in feet.items():
                self._history[name] = deque(
                    [fm.centroid], maxlen=self.history_frames,
                )
                self._last_feet[name] = fm
            return dict(feet)

        # Build cost matrix: Euclidean distance between old (tracked) and
        # new centroids.
        n_old = len(old_names)
        n_new = len(new_names)
        cost = np.full((n_new, n_old), self.max_distance_px * 2.0, dtype=float)
        for j, old_name in enumerate(old_names):
            old_centroid = self._history[old_name][-1]
            for i, new_name in enumerate(new_names):
                c = np.asarray(feet[new_name].centroid, dtype=float)
                d = float(np.linalg.norm(np.array(old_centroid) - c))
                cost[i, j] = d

        assignments = self._hungarian(cost)
        matched_new: set = set()
        matched_old: set = set()
        out: Dict[str, FootMask] = {}

        for i_new, j_old in assignments:
            if cost[i_new, j_old] > self.max_distance_px:
                continue  # too far — treat as new paw
            new_name = new_names[i_new]
            old_name = old_names[j_old]
            fm = feet[new_name]
            # Carry forward the old identity.
            try:
                object.__setattr__(fm, "matched_paw", old_name)
            except Exception:
                pass
            out[old_name] = fm
            self._history[old_name].append(fm.centroid)
            self._last_feet[old_name] = fm
            matched_new.add(i_new)
            matched_old.add(j_old)

        # Unmatched new paws → assign fresh identity.
        for i, name in enumerate(new_names):
            if i in matched_new:
                continue
            fm = feet[name]
            self._history[name] = deque(
                [fm.centroid], maxlen=self.history_frames,
            )
            self._last_feet[name] = fm
            out[name] = fm

        # Paws that disappeared this frame: keep in history but not in output.
        return out

    def reset(self) -> None:
        self._history.clear()
        self._last_feet.clear()


# ---------------------------------------------------------------------------
# Classification — L/R + F/H via DistanceFromLine (kb/09 §5.3)
# ---------------------------------------------------------------------------
def classify_feet_lfrh(
    feet: List[FootMask],
    body_axis_p1: Tuple[float, float],
    body_axis_p2: Tuple[float, float],
) -> Dict[str, FootMask]:
    """Assign each :class:`FootMask` to LF / RF / LH / RH using the
    MouseWalker quadrant rule.

    Improved v2.1: each quadrant independently picks the foot **closest
    to the body midpoint** (most central), rather than the most outward.
    This prevents misassignment when one side has multiple candidates.

    Parameters
    ----------
    feet : list of FootMask
        Detected feet in the current frame.
    body_axis_p1, body_axis_p2 : (x, y)
        Two points along the body axis (e.g. estimated from centroids).

    Returns
    -------
    dict mapping ``"LF" | "RF" | "LH" | "RH"`` to :class:`FootMask`.
    Each quadrant gets at most one foot. Unmatched feet are dropped.

    References
    ----------
    kb/09 §5.3 — MouseWalker DistanceFromLine quadrant classification.
    """
    if not feet:
        return {}
    p1 = np.asarray(body_axis_p1, dtype=float)
    p2 = np.asarray(body_axis_p2, dtype=float)
    axis_vec = p2 - p1
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 1e-6:
        return {}
    axis_dir = axis_vec / axis_len

    # Compute (parallel, perpendicular) for each foot.
    body_mid_point = p1 + axis_dir * (axis_len / 2.0)
    candidates: List[Tuple[FootMask, float, float]] = []
    for fm in feet:
        c = np.asarray(fm.centroid, dtype=float)
        d = c - p1
        parallel = float(np.dot(d, axis_dir))
        # Perpendicular: 2D cross-product (z-component).
        perp = float(axis_dir[0] * d[1] - axis_dir[1] * d[0])
        candidates.append((fm, parallel, perp))

    # Split into fore/hind pools by which side of midpoint.
    body_mid_par = axis_len / 2.0
    fore_pool: List[Tuple[FootMask, float, float]] = []
    hind_pool: List[Tuple[FootMask, float, float]] = []
    for fm, par, perp in candidates:
        if par > body_mid_par:
            fore_pool.append((fm, par, perp))
        else:
            hind_pool.append((fm, par, perp))

    out: Dict[str, FootMask] = {}

    def _assign_best(
        pool: List[Tuple[FootMask, float, float]],
        slot: str,
        perp_sign: int,
    ) -> None:
        """Pick the foot closest to body midpoint on the specified side."""
        if not pool:
            return
        # Filter by perpendicular sign.
        same_side = [(fm, par, perp) for fm, par, perp in pool
                     if (perp > 0) == (perp_sign > 0)]
        if not same_side:
            return
        # Pick the one with parallel offset closest to body_mid_par
        # (most central, not most outward).
        same_side.sort(key=lambda x: abs(x[1] - body_mid_par))
        out[slot] = same_side[0][0]

    # Perpendicular > 0 = right side, < 0 = left side.
    _assign_best(fore_pool, "LF", perp_sign=-1)
    _assign_best(fore_pool, "RF", perp_sign=+1)
    _assign_best(hind_pool, "LH", perp_sign=-1)
    _assign_best(hind_pool, "RH", perp_sign=+1)

    # Tag matched_paw in-place.
    for paw, fm in out.items():
        try:
            fm.matched_paw = paw
        except Exception:
            object.__setattr__(fm, "matched_paw", paw)
    return out


# ---------------------------------------------------------------------------
# Body-outline streak rejection (green_diff / canny modes)
# ---------------------------------------------------------------------------
# A real paw contact is a compact blob whose bounding-box aspect ratio
# (longest ÷ shortest side) is typically 0.5–3.5. When belly fur grazes
# the glass as the animal walks, the faint green signal forms a long thin
# ribbon along the body silhouette with aspect ≫ 5. We reject those.
# A splayed paw during propulsion can momentarily reach aspect ~4, so the
# threshold sits at 5.0 to avoid false rejection.
STREAK_MAX_ASPECT: float = 5.0


def _reject_body_streaks(feet: List[FootMask]) -> List[FootMask]:
    """Drop foot blobs that look like body-silhouette drag marks.

    A paw contact is a compact, roughly square-ish blob. When belly fur
    grazes the glass as the animal walks, the resulting green signal
    forms a long, thin ribbon along the body outline — not a paw. We
    reject blobs whose bounding-box aspect ratio (longest side ÷
    shortest side) exceeds :data:`STREAK_MAX_ASPECT`.

    Aspect alone is the most reliable discriminator because it is immune
    to the morphology fill-holes step (which can inflate solidity). A
    splayed paw rarely exceeds aspect 4; a body-drag streak is typically
    aspect 10–50.
    """
    kept: List[FootMask] = []
    for fm in feet:
        x, y, bw, bh = fm.bbox
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect <= STREAK_MAX_ASPECT:
            kept.append(fm)
    return kept


# ---------------------------------------------------------------------------
# Top-level pipeline — analyze_frame_v2
# ---------------------------------------------------------------------------
def analyze_frame_v2(
    frame: np.ndarray,
    *,
    background: Optional[BackgroundModel] = None,
    body_axis: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    px_per_mm: Optional[float] = None,
    # -- red-background two-stage parameters (v2.1) --
    use_red_background: bool = True,
    body_threshold: int = DEFAULT_BODY_THRESHOLD,
    paw_threshold: int = DEFAULT_PAW_THRESHOLD,
    # -- detection mode (v2.2): how paw mask is computed --
    #   "body_and_green" : legacy — paw = body_mask & (G > paw_threshold).
    #                      Strict; drops light touches that don't block red.
    #   "green_diff"     : MouseWalker-style — paw = (G - bg_G) > paw_threshold.
    #                      Decoupled from body; recovers faint paw contacts.
    #   "canny"          : PrAnCER-style — Canny edges on green-diff, then
    #                      filled. Robust to low absolute brightness.
    detection_mode: str = "green_diff",
    canny_low: int = 30,
    canny_high: int = 80,
    # -- body-proximity constraint (v2.2.1) --
    # When non-zero, a green_diff pixel is rejected if it is farther than
    # this many pixels from the body silhouette (dilated by ~this radius).
    # Kills tail drag and belly-fur streaks while keeping paws adjacent to
    # the body. 0 disables the filter (default for backward compatibility).
    body_proximity_px: int = 0,
    # -- legacy HSV parameters (retained for compatibility) --
    hsv_lower: Tuple[int, int, int] = DEFAULT_HSV_LOWER,
    hsv_upper: Tuple[int, int, int] = DEFAULT_HSV_UPPER,
    # -- shared parameters --
    min_area_px: int = DEFAULT_MIN_AREA_PX,
    max_finger_distance_px: int = DEFAULT_MAX_FINGER_DISTANCE_PX,
    in_stance_threshold_px: int = DEFAULT_IN_STANCE_AREA_PX,
    bg_offset: int = DEFAULT_BG_OFFSET,
) -> FootprintSequence:
    """Full v2.2 footprint pipeline for one frame.

    **Red-background mode** (``use_red_background=True``, default)::

        1. Red-channel background subtraction → body mask.
           ``bg_R - frame_R > body_threshold`` (animal blocks red light).
        2. Paw detection — depends on ``detection_mode``:
           - "body_and_green": paw = body_mask & (G > paw_threshold).
               Strict legacy behaviour; drops faint contacts.
           - "green_diff":    paw = (G - bg_G) > paw_threshold.
               Decoupled from body; recovers light/fur touches.
           - "canny":         Canny edges on |G - bg_G|, then fill holes.
               Boundary-based; robust to low absolute brightness.
        3. Morphology: dilate → fill holes → erode (MouseWalker CleanPIC
           order), so faint toe spots are *bridged* not eroded away.
        4. Connected components → individual finger blobs.
        5. Union-find grouping by centroid distance.
        6. Per-foot intensity + hull area + is_in_stance.
           Intensity extracted from green channel (= pressure proxy).
        7. (Optional) L/R + F/H classification via body axis.

    **Legacy HSV mode** (``use_red_background=False``)::

        Identical to the v1/v2 pipeline: background-subtract, HSV green
        segmentation, morphology, CC, union-find, classify.

    Returns
    -------
    :class:`FootprintSequence` with per-paw foot masks (when body axis
    is provided) or a raw list otherwise. ``all_feet`` always holds every
    detected foot (pre-classification), so callers that want faint
    contacts (e.g. the live accumulator) can iterate it directly.
    """
    # ── 1. Background model warmup ──────────────────────────────────────
    if background is not None:
        if not background.is_ready:
            background.update(frame)
            return FootprintSequence()

    # ── 2. Foreground segmentation ──────────────────────────────────────
    if use_red_background and background is not None and background.is_ready:
        # ----- Red-background detection -----
        bg_median = background.get_median()  # uint8 BGR
        if bg_median is None:
            return FootprintSequence()

        # Stage 1: body mask — where red channel drops significantly.
        # Kept for classification + (optionally) constraining paws.
        red_diff = bg_median[:, :, 2].astype(np.int16) - frame[:, :, 2].astype(np.int16)
        body_mask = (red_diff > body_threshold)

        # Stage 2: paw mask — computed per ``detection_mode``.
        mode = (detection_mode or "green_diff").lower()
        green_diff = frame[:, :, 1].astype(np.int16) - bg_median[:, :, 1].astype(np.int16)

        if mode == "body_and_green":
            # Legacy strict gate: must be on the body AND bright enough.
            paw_mask = body_mask & (frame[:, :, 1] > paw_threshold)
            combined = paw_mask.astype(np.uint8) * 255
        elif mode == "canny":
            # PrAnCER-style: edges on the green-difference image, then
            # fill the enclosed regions. Captures the *boundary* of a
            # contact regardless of its absolute brightness, so faint
            # fur touches (low G, but a visible edge) are recovered.
            diff_u8 = np.clip(np.abs(green_diff), 0, 255).astype(np.uint8)
            edges = cv2.Canny(diff_u8, canny_low, canny_high)
            # Dilate edges so findContours/fillHoles yields solid blobs.
            dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.dilate(edges, dil_k)
            combined = edges
            # Gate with green_diff > 0 so noise-only edges are removed.
            weak = (green_diff > 0).astype(np.uint8) * 255
            combined = cv2.bitwise_and(combined, weak)
        else:  # "green_diff" (default)
            # MouseWalker-style: paw = green rises above background.
            # Decoupled from body_mask, so light/fur contacts that do
            # NOT block the red backlight are still detected.
            paw_mask = green_diff > paw_threshold
            combined = paw_mask.astype(np.uint8) * 255

        # Stage 3: body-proximity constraint (v2.2.1).
        # Paws are always adjacent to the body. Tail drag, belly-fur
        # streaks and background noise appear FAR from the body silhouette
        # and accumulate over time into long ribbons. Reject green pixels
        # that are more than body_proximity_px away from the (dilated)
        # body mask.
        if body_proximity_px > 0 and body_mask.any():
            prox_dilate = cv2.dilate(
                body_mask.astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * body_proximity_px + 1,) * 2),
                iterations=1,
            )
            combined = cv2.bitwise_and(combined, prox_dilate)

    elif not use_red_background:
        # ----- Legacy HSV pipeline (green-illuminated CatWalk-style) -----
        if background is not None and background.is_ready:
            fg_mask = background.foreground_mask(frame)
        else:
            fg_mask = np.full(frame.shape[:2], 255, dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
        combined = cv2.bitwise_and(hsv_mask, fg_mask)
    else:
        # No background model available — cannot segment.
        return FootprintSequence()

    # ── 3. Morphology (MouseWalker CleanPIC order) ─────────────────────
    # CleanPIC: remove tiny noise, then DILATE → fill holes → ERODE.
    # This bridges faint toe spots into a single pad rather than eroding
    # them away (the old open+close did the opposite and lost faint toes).
    # NOTE: fill-holes is only applied in red-background modes where paws
    # form a single contiguous pad (FTIR). In HSV legacy mode, paws are
    # separate blobs that must NOT be merged via fill-holes.
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_k)
    if use_red_background:
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.dilate(cleaned, dil_k)
        # Fill interior holes (toes within a pad). Use a contour-based
        # approach instead of the flood-fill-from-border trick to avoid
        # merging separate blobs that are far apart.
        _contours, _hierarchy = cv2.findContours(
            cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
        )
        for i, cnt in enumerate(_contours):
            if _hierarchy is not None and _hierarchy[0][i][3] >= 0:
                # This contour has a parent → it is a hole; fill it.
                cv2.drawContours(cleaned, [cnt], -1, 255, -1)
        ero_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.erode(cleaned, ero_k)
    else:
        # HSV legacy: simple close to merge nearby fingers only.
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)

    # ── 4. Connected components ────────────────────────────────────────
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8,
    )
    blobs = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        blobs.append((i, (x, y, w, h), (cx, cy), area))

    # ── 5. Union-find grouping ─────────────────────────────────────────
    foot_groups = group_fingers_into_feet(blobs, max_finger_distance_px)

    # ── 6. Per-foot Mask with green-channel intensity ──────────────────
    # For red-background systems, use the GREEN channel as intensity
    # (proportional to paw pressure).  For legacy systems, use grayscale.
    if use_red_background:
        intensity_frame = frame[:, :, 1]  # green channel only
    else:
        intensity_frame = frame

    feet: List[FootMask] = []
    for new_label, group in enumerate(foot_groups):
        fm = _merge_group(
            group, intensity_frame, px_per_mm, in_stance_threshold_px,
            new_label=new_label,
        )
        feet.append(fm)

    # ── 6b. Reject body-outline streaks (green_diff / canny modes) ──────
    # When the animal's belly fur grazes the glass it produces a faint
    # green signal along the body silhouette. As the animal walks, this
    # leaves a long thin streak in the accumulator that is NOT a paw.
    # Paw contacts are compact (aspect ≈ 1–3); body drag is elongated
    # (aspect > 4). Filter on solidity + aspect ratio. Only applied in
    # the decoupled modes where the contamination can occur.
    if use_red_background and mode != "body_and_green":
        feet = _reject_body_streaks(feet)

    # ── 7. Classify ────────────────────────────────────────────────────
    # ``all_feet`` keeps EVERY detected foot (including faint/uncategorised
    # ones) so live accumulators can render them. ``feet`` holds only the
    # LF/RF/LH/RH assignments used by downstream gait metrics.
    seq = FootprintSequence()
    seq.all_feet = list(feet)
    if body_axis is not None:
        seq.feet = classify_feet_lfrh(
            feet, body_axis[0], body_axis[1],
        )
    else:
        seq.feet = {f"foot_{fm.label}": fm for fm in feet}
    seq.n_feet = len(seq.feet)
    return seq


def recall_score(
    detected: Dict[str, FootMask],
    ground_truth: Dict[str, Tuple[float, float]],
    match_radius_px: float = 30.0,
) -> float:
    """Compute footprint-detection recall (0..1).

    A ground-truth paw is "recalled" if at least one detected foot has a
    centroid within ``match_radius_px`` of it. The score is the fraction
    of ground-truth paws recalled.

    Acceptance gate (DEVELOPMENT_PLAN §6.2 W6): recall >= 0.95.
    """
    if not ground_truth:
        return 1.0
    detected_centroids = [np.asarray(fm.centroid, dtype=float)
                          for fm in detected.values()]
    recalled = 0
    for gt_name, gt_xy in ground_truth.items():
        gt = np.asarray(gt_xy, dtype=float)
        for c in detected_centroids:
            if np.linalg.norm(gt - c) <= match_radius_px:
                recalled += 1
                break
    return recalled / len(ground_truth)


__all__ = [
    "DEFAULT_HSV_LOWER",
    "DEFAULT_HSV_UPPER",
    "DEFAULT_BODY_THRESHOLD",
    "DEFAULT_PAW_THRESHOLD",
    "DEFAULT_BG_OFFSET",
    "DEFAULT_MIN_AREA_PX",
    "DEFAULT_MAX_FINGER_DISTANCE_PX",
    "DEFAULT_IN_STANCE_AREA_PX",
    "DEFAULT_TRACKING_MAX_DISTANCE_PX",
    "DEFAULT_TRACKING_HISTORY",
    "FootMask",
    "FootprintSequence",
    "BackgroundModel",
    "PawTracker",
    "group_fingers_into_feet",
    "estimate_body_axis",
    "classify_feet_lfrh",
    "analyze_frame_v2",
    "recall_score",
]