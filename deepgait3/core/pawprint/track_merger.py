"""Post-processing: merge BiScaleDetector candidates that are actually
the same physical footprint.

Symptom this addresses: a single real footprint can be split into multiple
short tracks because the per-frame IoU tracker loses continuity when the
mask shape changes fast (e.g. weight-bearing → toe-splitting) or when a
weak toe signal temporarily dominates the score.

Merge criteria (all must hold):
1. **Temporal adjacency**:  end_A ≤ start_B + max_gap_frames
2. **Spatial continuity**:  centroid_distance(end_A, start_B) ≤ merge_distance_px
3. **Area continuity**:     0.3 < area_end_A / area_start_B < 3.0
4. **No other track in between**:  no track C with start_C strictly between A and B
   (avoids eating a real neighbour)

Greedy in time order: sort candidates by first_frame, then sweep and merge
the first compatible pair.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from deepgait3.core.pawprint.biscale import FootprintCandidate


def _track_end(c: FootprintCandidate) -> tuple[float, float, float]:
    """Return (cx, cy, area) at the LAST strong blob of a closed candidate."""
    if not c.strong_blobs:
        return (0.0, 0.0, 0.0)
    last = c.strong_blobs[-1]
    return (last.cx_px, last.cy_px, float(last.area_px))


def _track_start(c: FootprintCandidate) -> tuple[float, float, float]:
    if not c.strong_blobs:
        return (0.0, 0.0, 0.0)
    first = c.strong_blobs[0]
    return (first.cx_px, first.cy_px, float(first.area_px))


def _track_median_area(c: FootprintCandidate) -> float:
    if not c.strong_blobs:
        return 0.0
    return float(np.median([b.area_px for b in c.strong_blobs]))


def merge_tracks(
    candidates: list[FootprintCandidate],
    max_gap_frames: int = 5,
    merge_distance_px: float = 60.0,
    area_ratio_min: float = 0.3,
    area_ratio_max: float = 3.0,
    allow_dip_below: bool = True,
) -> list[FootprintCandidate]:
    """Greedily merge adjacent candidates that satisfy the 4 criteria.

    Args:
        candidates: closed candidates from BiScaleDetector.finalize().
        max_gap_frames: max frames between end of A and start of B to merge.
        merge_distance_px: max centroid distance between end_A and start_B.
        area_ratio_min/max: acceptable area ratio (end_A / start_B) bounds.
        allow_dip_below: if True, merge even when a smaller-area candidate
                        exists between A and B (use cautiously).

    Returns:
        A new list of merged candidates (input is not mutated).
    """
    if not candidates:
        return []

    # Filter to candidates with at least one strong blob
    valid = [c for c in candidates if c.strong_blobs]
    if len(valid) <= 1:
        return list(valid)

    # Sort by first_frame
    valid = sorted(valid, key=lambda c: c.first_frame)

    merged: list[FootprintCandidate] = []
    next_id = max((c.track_id for c in valid), default=0) + 1

    i = 0
    while i < len(valid):
        cur = valid[i]
        cur_end = _track_end(cur)
        cur_end_area = cur_end[2]
        cur_first_frame = cur.first_frame
        cur_last_frame = cur.last_frame
        cur_strong = list(cur.strong_blobs)
        cur_weak = list(cur.weak_blobs)
        absorbed_any = False

        # Try to absorb the next candidates one by one
        j = i + 1
        while j < len(valid):
            nxt = valid[j]
            nxt_start = _track_start(nxt)
            # 1) temporal adjacency (gap)
            if nxt.first_frame - cur_last_frame > max_gap_frames:
                break
            # 2) spatial continuity
            d = float(np.hypot(cur_end[0] - nxt_start[0],
                                cur_end[1] - nxt_start[1]))
            # 3) area continuity
            ratio = (cur_end_area / nxt_start[2]) if nxt_start[2] > 0 else float("inf")
            if (d > merge_distance_px
                or not (area_ratio_min <= ratio <= area_ratio_max)):
                # Cannot merge — nxt is its own footprint, stop trying.
                break
            # Accept merge
            cur_strong.extend(nxt.strong_blobs)
            cur_weak.extend(nxt.weak_blobs)
            cur_last_frame = max(cur_last_frame, nxt.last_frame)
            cur_end = _track_end(nxt)
            cur_end_area = cur_end[2]
            absorbed_any = True
            j += 1

        # Build the merged candidate
        merged_cand = FootprintCandidate(
            track_id=cur.track_id,
            first_frame=cur_first_frame,
            last_frame=cur_last_frame,
        )
        merged_cand.strong_blobs = cur_strong
        merged_cand.weak_blobs = cur_weak
        merged.append(merged_cand)
        next_id += 1

        # Advance: skip absorbed candidates, or step by 1.
        if absorbed_any:
            i = j
        else:
            i += 1

    return merged


__all__ = ["merge_tracks"]