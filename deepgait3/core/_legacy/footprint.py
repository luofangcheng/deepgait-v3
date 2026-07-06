"""FTIR footprint analysis: green-light segmentation + connected components.

Physics (FTIR):
    PMMA light guide (n=1.49) + green LED (525 nm) → total internal reflection.
    Paw contact (skin n≈1.40) changes critical angle → light escapes at
    contact area.  Contact area ∝ footprint area; contact pressure ∝ intensity.

Pipeline:
    1. HSV green segmentation  → binary mask
    2. Morphological open/close  → clean mask
    3. Connected components      → per-footprint blobs
    4. Nearest-neighbor match to DLC paw positions → assign footprint to paw

Output per footprint:
    - area_px, area_mm² (with px_per_mm calibration)
    - bounding box (x, y, w, h)
    - centroid (cx, cy)
    - major/minor axis lengths (ellipse fit)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class Footprint:
    """A single detected footprint (one paw contact)."""
    label: int
    area_px: int
    area_mm2: float
    bbox: tuple[int, int, int, int]   # x, y, w, h
    centroid: tuple[float, float]     # cx, cy
    major_axis: float
    minor_axis: float
    angle_deg: float                  # ellipse orientation
    matched_paw: str | None = None    # e.g. "RightFore"


def segment_green(
    frame: np.ndarray,
    hsv_lower: tuple[int, int, int] = (35, 50, 30),
    hsv_upper: tuple[int, int, int] = (85, 255, 255),
) -> np.ndarray:
    """HSV green-channel segmentation.

    Args:
        frame: BGR image (H×W×3) from OpenCV.
        hsv_lower/upper: HSV threshold tuple (H:0-179, S:0-255, V:0-255).

    Returns:
        Binary mask (H×W, uint8) of green regions.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    return mask


def clean_mask(
    mask: np.ndarray,
    open_kernel: int = 3,
    close_kernel: int = 5,
) -> np.ndarray:
    """Morphological open (remove noise) + close (fill holes).

    Args:
        mask: binary mask.
        open_kernel: kernel size for opening.
        close_kernel: kernel size for closing.

    Returns:
        Cleaned binary mask.
    """
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_k)
    return cleaned


def extract_footprints(
    mask: np.ndarray,
    px_per_mm: float | None = None,
    min_area_px: int = 50,
) -> list[Footprint]:
    """Connected-component analysis on a cleaned binary mask.

    Args:
        mask: cleaned binary mask.
        px_per_mm: pixels per mm for real-world area conversion. None = keep px.
        min_area_px: discard components smaller than this.

    Returns:
        List of Footprint objects, sorted by area descending.
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    footprints: list[Footprint] = []
    for i in range(1, n_labels):  # skip background label 0
        area_px = int(stats[i, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])

        # Ellipse fit for major/minor axis (needs ≥5 contour points)
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and len(contours[0]) >= 5:
            ellipse = cv2.fitEllipse(contours[0])
            (ex, ey), (major, minor), angle = ellipse
            major_axis = max(major, minor)
            minor_axis = min(major, minor)
            angle_deg = angle
        else:
            major_axis = minor_axis = angle_deg = 0.0

        area_mm2 = area_px / (px_per_mm ** 2) if px_per_mm else float(area_px)
        footprints.append(Footprint(
            label=i,
            area_px=area_px,
            area_mm2=area_mm2,
            bbox=(x, y, w, h),
            centroid=(cx, cy),
            major_axis=major_axis,
            minor_axis=minor_axis,
            angle_deg=angle_deg,
        ))

    footprints.sort(key=lambda f: f.area_px, reverse=True)
    return footprints


def match_footprints_to_paws(
    footprints: list[Footprint],
    paw_positions: dict[str, tuple[float, float]],
) -> list[Footprint]:
    """Assign each footprint to the nearest DLC paw position.

    Args:
        footprints: detected footprints (centroids known).
        paw_positions: dict mapping paw name -> (x, y) from DLC pose.

    Returns:
        Footprints with ``matched_paw`` field populated.  Unmatched footprints
        keep ``None``.  Each paw gets at most one footprint (the nearest).
    """
    if not paw_positions or not footprints:
        return footprints

    # Build distance matrix: footprints × paws
    paw_names = list(paw_positions.keys())
    paw_xy = np.array([paw_positions[n] for n in paw_names], dtype=float)
    fp_xy = np.array([f.centroid for f in footprints], dtype=float)

    if fp_xy.size == 0 or paw_xy.size == 0:
        return footprints

    # Pairwise distances (N_fp × N_paws)
    dists = np.sqrt(((fp_xy[:, None, :] - paw_xy[None, :, :]) ** 2).sum(axis=2))

    # Greedy nearest-neighbor assignment: closest pair first, then next
    matched_fp: set[int] = set()
    matched_paw: set[int] = set()
    result = [Footprint(**{k: getattr(f, k) for k in Footprint.__dataclass_fields__}) for f in footprints]

    # Flatten and sort by distance
    flat = [(int(dists[i, j]), i, j) for i in range(len(footprints)) for j in range(len(paw_names))]
    flat.sort(key=lambda t: t[0])

    for d, fp_idx, paw_idx in flat:
        if fp_idx in matched_fp or paw_idx in matched_paw:
            continue
        result[fp_idx].matched_paw = paw_names[paw_idx]
        matched_fp.add(fp_idx)
        matched_paw.add(paw_idx)

    return result


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------

def analyze_frame(
    frame: np.ndarray,
    paw_positions: dict[str, tuple[float, float]] | None = None,
    px_per_mm: float | None = None,
    hsv_lower: tuple[int, int, int] = (35, 50, 30),
    hsv_upper: tuple[int, int, int] = (85, 255, 255),
    min_area_px: int = 50,
) -> list[Footprint]:
    """Full pipeline: segment -> clean -> extract -> optionally match to paws.

    Args:
        frame: BGR image.
        paw_positions: optional DLC paw positions for matching.
        px_per_mm: optional calibration.
        hsv_lower/upper: green segmentation thresholds.
        min_area_px: minimum component area.

    Returns:
        List of Footprint objects.
    """
    mask = segment_green(frame, hsv_lower, hsv_upper)
    mask = clean_mask(mask)
    footprints = extract_footprints(mask, px_per_mm, min_area_px)
    if paw_positions is not None:
        footprints = match_footprints_to_paws(footprints, paw_positions)
    return footprints
