"""Bi-scale temporal footprint detector (Stage 1 v0.5 prototype).

Uses two thresholds:
- **strong_threshold** finds the main footprint body.
- **weak_threshold** finds small/weak blobs (toes) in the neighborhood
  of an already-confirmed footprint.

This rescues the small green toes that single-threshold approaches drop,
while keeping noise rejection intact (weak blobs OUTSIDE a confirmed
footprint's neighborhood are discarded).

This is a research prototype: results feed back into the autoresearch
loop to tune the four hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from deepgait3.core.pawprint.scoring import compute_exg
from deepgait3.core.pawprint.scoring_detection import detect_blobs_from_score


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedBlob:
    """One blob at one frame, with its score."""
    cx_px: float
    cy_px: float
    area_px: int
    bbox: tuple
    score_mean: float   # mean score inside the blob's mask
    is_strong: bool     # passed strong threshold alone
    frame_idx: int


@dataclass
class FootprintCandidate:
    """One footprint being built across frames."""
    track_id: int
    strong_blobs: list[DetectedBlob] = field(default_factory=list)
    weak_blobs: list[DetectedBlob] = field(default_factory=list)  # assigned toes
    first_frame: int = -1
    last_frame: int = -1
    closed: bool = False

    @property
    def n_strong_frames(self) -> int:
        return len(self.strong_blobs)

    @property
    def n_weak_frames(self) -> int:
        return len(self.weak_blobs)

    @property
    def duration_frames(self) -> int:
        if self.first_frame < 0:
            return 0
        return self.last_frame - self.first_frame + 1


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class BiScaleDetector:
    """Two-threshold per-frame detector + simple temporal tracker.

    Parameters (all are searched by ``autoresearch.py``):
        strong_threshold:        pixels with score >= this are footprint body
        weak_threshold:          pixels with score >= this are toe candidates
        toe_max_distance_px:      a weak blob within this distance (px) of a
                                 strong footprint is assigned as a toe
        toe_min_overlap_frames:   a weak blob must appear in >= N frames
                                 inside the footprint's lifetime to be kept
        strong_min_area_px:       min blob area for strong blobs
        weak_min_area_px:         min blob area for weak blobs (toes are tiny)
        max_gap_frames:           close track if no strong detection for N frames
    """

    def __init__(self,
                 strong_threshold: int = 80,
                 weak_threshold: int = 20,
                 toe_max_distance_px: float = 24.0,
                 toe_min_overlap_frames: int = 1,
                 strong_min_area_px: int = 20,
                 weak_min_area_px: int = 3,
                 max_gap_frames: int = 3,
                 score_fn=None):
        # Default score function: MExG (Modified Excess Green Index).
        # Outperforms ExG by ~3% F1 on real GT (test1.mp4 + data-test1.xlsx).
        # See examples/v07_algo_a_b_gt.py for the comparison.
        if score_fn is None:
            from .scoring import compute_mexg
            score_fn = compute_mexg
        self.score_fn = score_fn
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.toe_max_distance_px = toe_max_distance_px
        self.toe_min_overlap_frames = toe_min_overlap_frames
        self.strong_min_area_px = strong_min_area_px
        self.weak_min_area_px = weak_min_area_px
        self.max_gap_frames = max_gap_frames
        self.score_fn = score_fn

        self.candidates: dict[int, FootprintCandidate] = {}
        self.closed_candidates: list[FootprintCandidate] = []
        self.next_track_id = 0

        # Stats
        self._n_strong_blobs_total = 0
        self._n_weak_assigned_total = 0
        self._n_weak_unassigned_total = 0

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> None:
        score = self.score_fn(frame_bgr)
        strong_blobs, weak_blobs = self._detect_two_scales(frame_bgr, score, frame_idx)

        # 1. Update active candidates by matching strong blobs (IoU-like).
        self._stale_close(frame_idx)
        self._match_strong_blobs(strong_blobs, frame_idx)

        # 2. Try to assign each weak blob to an active candidate (toe recovery).
        for wb in weak_blobs:
            owner = self._find_owner(wb, frame_idx)
            if owner is not None:
                wb.frame_idx = frame_idx
                owner.weak_blobs.append(wb)
                self._n_weak_assigned_total += 1
            else:
                self._n_weak_unassigned_total += 1

    # ------------------------------------------------------------------
    def finalize(self) -> list[FootprintCandidate]:
        self.closed_candidates.extend(self.candidates.values())
        for c in self.closed_candidates:
            c.closed = True
        self.candidates.clear()
        return list(self.closed_candidates)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _detect_two_scales(self, frame_bgr: np.ndarray,
                            score: np.ndarray, frame_idx: int) -> tuple[list[DetectedBlob], list[DetectedBlob]]:
        """Run strong + weak threshold, return both blob lists."""
        _mask_s, strong_dicts = detect_blobs_from_score(
            score, threshold=self.strong_threshold,
            min_area=self.strong_min_area_px,
        )
        _mask_w, weak_dicts = detect_blobs_from_score(
            score, threshold=self.weak_threshold,
            min_area=self.weak_min_area_px,
        )
        # Compute mean score for each blob (signal strength indicator)
        strong = []
        for b in strong_dicts:
            mask = self._blob_mask(b, score.shape)
            mean_s = float(score[mask].mean()) if mask.any() else 0.0
            strong.append(DetectedBlob(
                cx_px=b["cx"], cy_px=b["cy"], area_px=b["area"],
                bbox=b["bbox"], score_mean=mean_s, is_strong=True, frame_idx=frame_idx,
            ))
        weak = []
        # Exclude weak blobs whose bbox contains any strong blob's centroid.
        strong_centers = [(b["cx"], b["cy"]) for b in strong_dicts]
        for b in weak_dicts:
            cx, cy = b["cx"], b["cy"]
            if any(self._dist_px(cx, cy, scx, scy) < 6.0 for scx, scy in strong_centers):
                # Weak blob is INSIDE a strong footprint, not a separate toe.
                continue
            mask = self._blob_mask(b, score.shape)
            mean_s = float(score[mask].mean()) if mask.any() else 0.0
            weak.append(DetectedBlob(
                cx_px=cx, cy_px=cy, area_px=b["area"],
                bbox=b["bbox"], score_mean=mean_s, is_strong=False, frame_idx=frame_idx,
            ))
        self._n_strong_blobs_total += len(strong)
        return strong, weak

    def _blob_mask(self, blob_dict: dict, shape: tuple[int, int]) -> np.ndarray:
        x0, y0, x1, y1 = blob_dict["bbox"]
        H, W = shape
        m = np.zeros((H, W), dtype=bool)
        m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
        return m

    def _dist_px(self, x1, y1, x2, y2) -> float:
        return float(np.hypot(x1 - x2, y1 - y2))

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------
    def _stale_close(self, frame_idx: int) -> None:
        to_close = [
            tid for tid, c in self.candidates.items()
            if c.last_frame >= 0 and frame_idx - c.last_frame > self.max_gap_frames
        ]
        for tid in to_close:
            c = self.candidates.pop(tid)
            self.closed_candidates.append(c)
            c.closed = True

    def _match_strong_blobs(self, strong_blobs: list[DetectedBlob],
                              frame_idx: int) -> None:
        """Greedy nearest-neighbor match of strong blobs to active candidates."""
        if not strong_blobs:
            return
        if not self.candidates:
            for sb in strong_blobs:
                self._new_candidate(sb, frame_idx)
            return

        # Build cost matrix: distance from each blob to each active centroid.
        cand_ids = list(self.candidates.keys())
        cand_cents = []
        for tid in cand_ids:
            c = self.candidates[tid]
            if c.strong_blobs:
                last = c.strong_blobs[-1]
                cand_cents.append((last.cx_px, last.cy_px))
            else:
                cand_cents.append((0.0, 0.0))
        used_blobs = set()
        # Greedy: for each candidate, find nearest unused strong blob
        # within ~50 px (large enough to allow movement).
        for i, tid in enumerate(cand_ids):
            cx, cy = cand_cents[i]
            best_j, best_d = -1, float("inf")
            for j, sb in enumerate(strong_blobs):
                if j in used_blobs:
                    continue
                d = self._dist_px(cx, cy, sb.cx_px, sb.cy_px)
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 50.0:
                sb = strong_blobs[best_j]
                sb.frame_idx = frame_idx
                self.candidates[tid].strong_blobs.append(sb)
                self.candidates[tid].last_frame = frame_idx
                used_blobs.add(best_j)
        # Unmatched → new candidates
        for j, sb in enumerate(strong_blobs):
            if j not in used_blobs:
                self._new_candidate(sb, frame_idx)

    def _new_candidate(self, sb: DetectedBlob, frame_idx: int) -> None:
        sb.frame_idx = frame_idx
        c = FootprintCandidate(
            track_id=self.next_track_id,
            first_frame=frame_idx,
            last_frame=frame_idx,
        )
        c.strong_blobs.append(sb)
        self.candidates[self.next_track_id] = c
        self.next_track_id += 1

    def _find_owner(self, wb: DetectedBlob, frame_idx: int) -> Optional[FootprintCandidate]:
        """A weak blob belongs to the nearest active candidate if close enough
        AND that candidate has had >= toe_min_overlap_frames of activity.

        The overlap requirement is checked globally (across the candidate's
        lifetime), not per-frame.
        """
        if not self.candidates:
            return None
        best, best_d = None, float("inf")
        for c in self.candidates.values():
            if c.n_strong_frames < self.toe_min_overlap_frames:
                continue
            # Use median centroid of recent strong blobs
            recent = c.strong_blobs[-5:] if c.strong_blobs else []
            if not recent:
                continue
            cx = np.median([b.cx_px for b in recent])
            cy = np.median([b.cy_px for b in recent])
            d = self._dist_px(cx, cy, wb.cx_px, wb.cy_px)
            if d < best_d:
                best_d = d
                best = c
        if best is not None and best_d <= self.toe_max_distance_px:
            return best
        return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @property
    def stats(self) -> dict:
        n_valid = sum(1 for c in self.closed_candidates
                       if c.duration_frames >= 5)
        n_short = sum(1 for c in self.closed_candidates
                       if c.duration_frames <= 3)
        return {
            "n_strong_blobs_total": self._n_strong_blobs_total,
            "n_weak_assigned_total": self._n_weak_assigned_total,
            "n_weak_unassigned_total": self._n_weak_unassigned_total,
            "n_valid_tracks": n_valid,
            "n_short_tracks": n_short,
            "total_toes_recovered": self._n_weak_assigned_total,
        }


__all__ = ["BiScaleDetector", "DetectedBlob", "FootprintCandidate"]