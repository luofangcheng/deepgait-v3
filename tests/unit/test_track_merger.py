"""Unit tests for ``merge_tracks`` — the post-processing step that
combines BiScaleDetector candidates split across multiple short tracks.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepgait3.core.pawprint.biscale import DetectedBlob, FootprintCandidate
from deepgait3.core.pawprint.track_merger import merge_tracks


def _make_blob(cx: float, cy: float, area: int, frame: int) -> DetectedBlob:
    return DetectedBlob(
        cx_px=cx, cy_px=cy, area_px=area, bbox=(int(cx), int(cy),
                                                  int(cx) + 10, int(cy) + 10),
        score_mean=200.0, is_strong=True, frame_idx=frame,
    )


def _make_candidate(track_id: int, start: int, end: int, cx_start: float,
                     cy_start: float, cx_end: float, cy_end: float,
                     area: int = 100) -> FootprintCandidate:
    """Build a candidate with strong blobs at every frame from start..end,
    moving linearly from (cx_start, cy_start) to (cx_end, cy_end)."""
    c = FootprintCandidate(track_id=track_id, first_frame=start, last_frame=end)
    for f in range(start, end + 1):
        t = (f - start) / max(end - start, 1)
        cx = cx_start + (cx_end - cx_start) * t
        cy = cy_start + (cy_end - cy_start) * t
        c.strong_blobs.append(_make_blob(cx, cy, area, f))
    return c


# ---------------------------------------------------------------------------
# 1) Two adjacent same-footprint tracks → merge
# ---------------------------------------------------------------------------

def test_two_close_tracks_merge():
    """A track ending at (100, 200) frame 10 followed by a track starting at
    (105, 205) frame 13 must merge into one (gap 3 frames, dist 7 px)."""
    a = _make_candidate(0, start=1, end=10, cx_start=50, cy_start=200,
                         cx_end=100, cy_end=200, area=100)
    b = _make_candidate(1, start=13, end=20, cx_start=105, cy_start=205,
                         cx_end=150, cy_end=200, area=100)
    merged = merge_tracks([a, b], max_gap_frames=5, merge_distance_px=60.0)
    assert len(merged) == 1
    m = merged[0]
    assert m.first_frame == 1
    assert m.last_frame == 20
    assert len(m.strong_blobs) == 18  # 10 + 8


# ---------------------------------------------------------------------------
# 2) Two distant tracks → do NOT merge
# ---------------------------------------------------------------------------

def test_distant_tracks_do_not_merge():
    """Two tracks > 60 px apart stay separate."""
    a = _make_candidate(0, start=1, end=10, cx_start=100, cy_start=100,
                         cx_end=200, cy_end=100, area=100)
    b = _make_candidate(1, start=12, end=20, cx_start=900, cy_start=900,
                         cx_end=1000, cy_end=900, area=100)
    merged = merge_tracks([a, b], max_gap_frames=5, merge_distance_px=60.0)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# 3) Tracks too far apart in time → do NOT merge
# ---------------------------------------------------------------------------

def test_too_far_apart_in_time_do_not_merge():
    """Gap > max_gap_frames → no merge even if spatially close."""
    a = _make_candidate(0, start=1, end=5, cx_start=100, cy_start=100,
                         cx_end=100, cy_end=100, area=100)
    b = _make_candidate(1, start=20, end=25, cx_start=100, cy_start=100,
                         cx_end=100, cy_end=100, area=100)
    merged = merge_tracks([a, b], max_gap_frames=5, merge_distance_px=60.0)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# 4) Area ratio out of bounds → do NOT merge
# ---------------------------------------------------------------------------

def test_area_ratio_out_of_bounds_do_not_merge():
    """Track B starts much smaller than track A ends (footprint sudden
    disappearance) → not merged."""
    a = _make_candidate(0, start=1, end=10, cx_start=100, cy_start=100,
                         cx_end=100, cy_end=100, area=500)
    b = _make_candidate(1, start=12, end=20, cx_start=105, cy_start=100,
                         cx_end=105, cy_end=100, area=50)   # 10× smaller
    merged = merge_tracks([a, b], max_gap_frames=5, merge_distance_px=60.0)
    assert len(merged) == 2


# ---------------------------------------------------------------------------
# 5) Three short tracks of the same footprint → all merge into one
# ---------------------------------------------------------------------------

def test_three_short_tracks_all_merge():
    """A scenario like real GT shows: 1 footprint → 3 short tracks.
    After merging, only 1 long track remains."""
    a = _make_candidate(0, start=1, end=4, cx_start=100, cy_start=100,
                         cx_end=100, cy_end=100, area=80)
    b = _make_candidate(1, start=6, end=8, cx_start=102, cy_start=101,
                         cx_end=104, cy_end=102, area=80)
    c = _make_candidate(2, start=10, end=13, cx_start=106, cy_start=103,
                         cx_end=110, cy_end=104, area=80)
    merged = merge_tracks([a, b, c], max_gap_frames=5, merge_distance_px=60.0)
    assert len(merged) == 1
    m = merged[0]
    assert m.first_frame == 1
    assert m.last_frame == 13
    assert len(m.strong_blobs) == 11  # 4 + 3 + 4


# ---------------------------------------------------------------------------
# 6) Empty / single-track input
# ---------------------------------------------------------------------------

def test_empty_returns_empty():
    assert merge_tracks([]) == []


def test_single_track_unchanged():
    a = _make_candidate(0, start=1, end=10, cx_start=100, cy_start=100,
                         cx_end=200, cy_end=200, area=100)
    merged = merge_tracks([a])
    assert len(merged) == 1
    assert merged[0].track_id == a.track_id


# ---------------------------------------------------------------------------
# 7) Track with no strong blobs is filtered out
# ---------------------------------------------------------------------------

def test_track_with_no_strong_blobs_filtered():
    """A candidate with no strong blobs should be dropped during merging."""
    a = _make_candidate(0, start=1, end=10, cx_start=100, cy_start=100,
                         cx_end=200, cy_end=100, area=100)
    empty = FootprintCandidate(track_id=1, first_frame=11, last_frame=12)
    merged = merge_tracks([a, empty])
    assert len(merged) == 1
    assert merged[0].track_id == 0


# ---------------------------------------------------------------------------
# 8) Doesn't mutate input
# ---------------------------------------------------------------------------

def test_input_not_mutated():
    a = _make_candidate(0, start=1, end=10, cx_start=100, cy_start=100,
                         cx_end=200, cy_end=100, area=100)
    b = _make_candidate(1, start=12, end=20, cx_start=205, cy_start=100,
                         cx_end=300, cy_end=100, area=100)
    n_a_before = len(a.strong_blobs)
    n_b_before = len(b.strong_blobs)
    merge_tracks([a, b], max_gap_frames=5, merge_distance_px=60.0)
    assert len(a.strong_blobs) == n_a_before
    assert len(b.strong_blobs) == n_b_before