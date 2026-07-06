"""Unit tests for ``BiScaleDetector`` — the dual-threshold Stage 1 v0.5
algorithm.

These tests pin down the four invariants that matter for toe recovery:

1. **Strong detection alone** works as a v0.4.2-style detector (baseline).
2. **Weak blob near a confirmed footprint** gets assigned as a toe.
3. **Weak blob far from any footprint** stays unassigned (noise gating).
4. **Track continuity** — a strong blob within ``~50 px`` of the previous
   frame's centroid extends the existing track rather than starting a new one.
5. **Track closing** — no strong detection for > ``max_gap_frames`` closes
   the track.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepgait3.core.pawprint.biscale import (
    BiScaleDetector,
    DetectedBlob,
    FootprintCandidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _green_blob_frame(h: int = 384, w: int = 1920,
                       cx: int = 200, cy: int = 100,
                       main_radius: int = 12, main_intensity: int = 200,
                       toe_xy: tuple[int, int] | None = None,
                       toe_radius: int = 4, toe_intensity: int = 45) -> np.ndarray:
    """Build a BGR frame with one strong green blob + optional weak toe.

    Toe intensity is kept low so that ExG = 2*G - R - B stays below the
    default ``strong_threshold=100`` (ExG at intensity=45 → 70, at 30 → 40).
    """
    img = np.full((h, w, 3), 10, dtype=np.uint8)
    # Main footprint
    yy, xx = np.ogrid[:h, :w]
    main_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= main_radius ** 2
    img[main_mask] = (10, main_intensity, 10)
    # Optional toe (small, weak, adjacent)
    if toe_xy is not None:
        tcx, tcy = toe_xy
        toe_mask = (xx - tcx) ** 2 + (yy - tcy) ** 2 <= toe_radius ** 2
        img[toe_mask] = (10, toe_intensity, 10)
    return img


# ---------------------------------------------------------------------------
# 1) Baseline — strong-only detection still works
# ---------------------------------------------------------------------------

def test_strong_only_detection_finds_main():
    """A clear green blob is detected as a strong footprint, no false toes."""
    det = BiScaleDetector(strong_threshold=100, weak_threshold=5,
                          toe_min_overlap_frames=1)
    frame = _green_blob_frame(cx=300, cy=150, main_radius=12)
    for i in range(1, 6):
        det.process_frame(frame, i)
    cands = det.finalize()
    assert len(cands) == 1, f"expected 1 candidate, got {len(cands)}"
    c = cands[0]
    assert c.n_strong_frames == 5
    assert c.n_weak_frames == 0  # no weak blob nearby → no toes


# ---------------------------------------------------------------------------
# 2) Toe recovery — weak blob near footprint gets assigned
# ---------------------------------------------------------------------------

def test_weak_blob_near_footprint_is_toe():
    """A weak blob within 24 px of the main footprint is recovered as a toe."""
    det = BiScaleDetector(strong_threshold=100, weak_threshold=5,
                          toe_max_distance_px=24.0,
                          toe_min_overlap_frames=1)
    # toe_xy = (320, 150) = 20 px from main at (300, 150) — within 24 px
    frame = _green_blob_frame(cx=300, cy=150, main_radius=12,
                                toe_xy=(320, 150), toe_radius=4,
                                toe_intensity=45)
    for i in range(1, 6):
        det.process_frame(frame, i)
    cands = det.finalize()
    assert len(cands) == 1
    c = cands[0]
    assert c.n_strong_frames == 5
    # Toe (ExG=70) is above weak_threshold=5 but below strong=100
    assert c.n_weak_frames >= 1, f"toe not recovered: n_weak={c.n_weak_frames}"


# ---------------------------------------------------------------------------
# 3) Noise gating — weak blob far from any footprint stays unassigned
# ---------------------------------------------------------------------------

def test_far_weak_blob_is_noise():
    """A weak blob far from any footprint does NOT become a toe."""
    det = BiScaleDetector(strong_threshold=100, weak_threshold=5,
                          toe_max_distance_px=24.0,
                          toe_min_overlap_frames=1)
    frame = _green_blob_frame(cx=300, cy=150, main_radius=12,
                                # toe 800 px away — way outside 24 px radius
                                toe_xy=(1100, 150), toe_radius=4,
                                toe_intensity=45)
    for i in range(1, 6):
        det.process_frame(frame, i)
    cands = det.finalize()
    assert len(cands) == 1
    c = cands[0]
    assert c.n_weak_frames == 0, f"noise leaked as toe: {c.n_weak_frames}"


# ---------------------------------------------------------------------------
# 4) Track continuity — moving footprint stays as one track
# ---------------------------------------------------------------------------

def test_moving_footprint_single_track():
    """A blob moving smoothly across frames must produce ONE track, not many."""
    det = BiScaleDetector(strong_threshold=100, weak_threshold=5,
                          max_gap_frames=3)
    for i in range(1, 11):
        cx = 200 + i * 20  # moves 20 px right each frame (within 50 px match radius)
        frame = _green_blob_frame(cx=cx, cy=150, main_radius=12)
        det.process_frame(frame, i)
    cands = det.finalize()
    assert len(cands) == 1, f"moving footprint split into {len(cands)} tracks"
    assert cands[0].n_strong_frames == 10


# ---------------------------------------------------------------------------
# 5) Track closing — gap > max_gap_frames forces closure
# ---------------------------------------------------------------------------

def test_gap_closes_track():
    """If no strong detection for > max_gap_frames, the track closes."""
    det = BiScaleDetector(strong_threshold=100, weak_threshold=5,
                          max_gap_frames=2)
    frame = _green_blob_frame(cx=300, cy=150)
    for i in range(1, 4):  # 3 frames of detection
        det.process_frame(frame, i)
    # 4 empty frames — exceeds max_gap_frames=2
    for i in range(4, 8):
        blank = np.full((384, 1920, 3), 5, dtype=np.uint8)
        det.process_frame(blank, i)
    # New footprint at different position
    frame2 = _green_blob_frame(cx=1500, cy=150)
    for i in range(8, 12):
        det.process_frame(frame2, i)
    cands = det.finalize()
    # Should have 2 tracks: first one closed during the gap, second one fresh
    assert len(cands) >= 2, f"expected ≥2 tracks, got {len(cands)}"
    # The first track must have closed with last_frame ≤ 3
    first = min(cands, key=lambda c: c.first_frame)
    assert first.first_frame == 1
    assert first.last_frame == 3


# ---------------------------------------------------------------------------
# 6) Score integration
# ---------------------------------------------------------------------------

def test_stats_counts_valid_tracks():
    """Only tracks with >= 5 frames count as 'valid'."""
    det = BiScaleDetector(strong_threshold=100, weak_threshold=5)
    # Short track: 3 frames
    for i in range(1, 4):
        det.process_frame(_green_blob_frame(cx=200, cy=150), i)
    # Long track: 8 frames
    for i in range(4, 12):
        det.process_frame(_green_blob_frame(cx=1500, cy=150), i)
    cands = det.finalize()
    stats = det.stats
    assert stats["n_valid_tracks"] == 1
    assert stats["n_short_tracks"] == 1


# ---------------------------------------------------------------------------
# 7) Multi-toe case
# ---------------------------------------------------------------------------

def test_multiple_toes_assigned_to_same_footprint():
    """Two weak blobs near the same footprint should both be assigned."""
    det = BiScaleDetector(strong_threshold=100, weak_threshold=5,
                          toe_max_distance_px=24.0,
                          toe_min_overlap_frames=1)
    # Build a frame with main + 2 toes on either side (intensity=40 → ExG=60)
    frame = _green_blob_frame(cx=300, cy=150, main_radius=12)
    yy, xx = np.ogrid[:384, :1920]
    frame[((xx - 320) ** 2 + (yy - 150) ** 2) <= 16] = (10, 40, 10)
    frame[((xx - 280) ** 2 + (yy - 150) ** 2) <= 16] = (10, 40, 10)
    for i in range(1, 6):
        det.process_frame(frame, i)
    cands = det.finalize()
    assert len(cands) == 1
    assert cands[0].n_weak_frames >= 2, f"expected ≥2 toes, got {cands[0].n_weak_frames}"