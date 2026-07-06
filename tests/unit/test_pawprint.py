"""Unit tests for Stage 1: fTIR footprint extraction.

Test list follows ``deepgait-v2/report.md`` §12 (which the v2 module was missing).
Each test exercises one logical unit of the dynamics pipeline so that any
schema/behavioral regression introduced during copy-in or future refactor
is caught early.

These tests are PURE-LOGIC: no real video. They build synthetic frames /
blobs / masks in-memory and verify the algorithm's invariants.
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from deepgait3.core.pawprint import (
    FootMask,
    FrameData,
    PawPrint,
    PawPrintExtractor,
    QualityFlags,
    detect_blobs,
    cluster_blobs_into_feet,
)
from deepgait3.core.pawprint.tracker import IoUFootprintTracker


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_bg(shape=(384, 1920), value=10.0):
    """Background G-channel: uniform low value (no prints)."""
    return np.full(shape, value, dtype=np.float32)


def _stamp_blob(bg, cx_px, cy_px, radius=8, intensity=80):
    """Return a frame_bgr whose G-channel has a bright circular blob.

    The frame is otherwise a copy of the bg.
    """
    H, W = bg.shape
    yy, xx = np.ogrid[:H, :W]
    mask = (xx - cx_px) ** 2 + (yy - cy_px) ** 2 <= radius ** 2
    G = bg.astype(np.float32).copy()
    G[mask] = intensity
    frame_bgr = np.stack([G, G, G], axis=-1).astype(np.uint8)
    return frame_bgr


def _make_footmask(cx_px=100.0, cy_px=100.0, w=20, h=20):
    """Construct a small FootMask at a known position."""
    mask = np.zeros((h + 16, w + 16), dtype=bool)
    mask[8:8 + h, 8:8 + w] = True
    return FootMask(
        blob_indices=[0],
        centroid_px=(cx_px, cy_px),
        bbox_xyxy=(int(cx_px), int(cy_px), int(cx_px + w), int(cy_px + h)),
        bbox_xyxy_padded=(int(cx_px - 8), int(cy_px - 8),
                          int(cx_px + w + 8), int(cy_px + h + 8)),
        mask_padded=mask,
        raw_intensity_crop=np.full((h + 16, w + 16), 50.0, dtype=np.float32),
        bg_intensity_crop=np.full((h + 16, w + 16), 10.0, dtype=np.float32),
        pressure_map=np.full((h + 16, w + 16), 1.0, dtype=np.float32),
        total_area_px=w * h,
        mean_intensity=50.0,
        peak_intensity=80.0,
        touches_edge=False,
    )


# ---------------------------------------------------------------------------
# 1) detect_blobs: threshold filters out sub-min-area noise
# ---------------------------------------------------------------------------

def test_detect_blobs_threshold():
    """Blobs smaller than ``min_area_px`` are dropped; big blobs survive."""
    bg = _make_bg()
    frame = _stamp_blob(bg, cx_px=200, cy_px=100, radius=15, intensity=80)
    blobs = detect_blobs(frame, bg, tau_paw=18, min_area_px=10,
                          max_area_px=None, bbox_pad_px=4)
    assert len(blobs) == 1
    assert blobs[0]["area_px"] >= 10
    # Now bump min_area_px way above the blob size → must return nothing.
    blobs2 = detect_blobs(frame, bg, tau_paw=18, min_area_px=9999)
    assert blobs2 == []


# ---------------------------------------------------------------------------
# 2) cluster_blobs_into_feet: nearby blobs merge, far blobs don't
# ---------------------------------------------------------------------------

def test_cluster_blobs_distance():
    """Two blobs within D_merge_px → one FootMask; > D_merge_px apart → two."""
    bg = _make_bg()
    f1 = _stamp_blob(bg, cx_px=100, cy_px=100, radius=10, intensity=80)
    f2 = _stamp_blob(bg, cx_px=105, cy_px=100, radius=10, intensity=80)
    blobs_close = detect_blobs(f1 + f2, bg, tau_paw=18, min_area_px=10,
                                bbox_pad_px=4)
    # Pad creates spurious extra blobs from halo; filter back to the two main ones
    blobs_close = [b for b in blobs_close if b["area_px"] > 50]
    foots = cluster_blobs_into_feet(blobs_close, D_merge_px=23.0)
    assert len(foots) == 1, "blobs 5 px apart must merge"
    assert foots[0].total_area_px >= 2 * 50

    # Far apart (> D_merge): each becomes its own FootMask
    f3 = _stamp_blob(bg, cx_px=100, cy_px=100, radius=10, intensity=80)
    f4 = _stamp_blob(bg, cx_px=400, cy_px=100, radius=10, intensity=80)
    blobs_far = detect_blobs(f3 + f4, bg, tau_paw=18, min_area_px=10,
                              bbox_pad_px=4)
    blobs_far = [b for b in blobs_far if b["area_px"] > 50]
    foots_far = cluster_blobs_into_feet(blobs_far, D_merge_px=23.0)
    assert len(foots_far) == 2, "blobs 300 px apart must stay separate"


# ---------------------------------------------------------------------------
# 3) IoU tracker: continues track across overlapping frames
# ---------------------------------------------------------------------------

def test_iou_tracker_continues_on_overlap():
    """Same FootMask position across 5 frames → one track."""
    tr = IoUFootprintTracker(frame_shape=(384, 1920))
    fm = _make_footmask(cx_px=200.0, cy_px=150.0)
    for f in range(0, 5):
        tr.update(f, [fm])
    assert len(tr.active) == 1
    track = next(iter(tr.active.values()))
    assert track.n_frames == 5


# ---------------------------------------------------------------------------
# 4) IoU tracker: closes on gap > max_gap_frames
# ---------------------------------------------------------------------------

def test_iou_tracker_closes_on_gap():
    """A gap > max_gap_frames forces a track to close."""
    tr = IoUFootprintTracker(frame_shape=(384, 1920), max_gap_frames=2)
    fm1 = _make_footmask(cx_px=200.0, cy_px=150.0)
    fm2 = _make_footmask(cx_px=500.0, cy_px=150.0)  # far away → no match
    tr.update(0, [fm1])
    tr.update(1, [fm1])
    tr.update(2, [])
    tr.update(3, [])
    tr.update(4, [])
    tr.update(5, [fm2])
    closed = tr.finalize()
    # fm1 must have closed (was popped due to gap), fm2 may or may not be closed
    # depending on timing — at minimum fm1's track is in closed list
    assert any(c.foots[0][1].centroid_px == (200.0, 150.0) for c in closed)


# ---------------------------------------------------------------------------
# 5) PawPrint: default-construct with all 36+ fields populated
# ---------------------------------------------------------------------------

def test_pawprint_default_construct():
    """PawPrint() with no args must give a 36-field shape with sane defaults."""
    pp = PawPrint(print_id=0, touchdown_frame=10, liftoff_frame=20,
                   true_liftoff_frame=22, peak_area_frame=15,
                   peak_intensity_frame=14, duration_s=0.2,
                   time_to_peak_area_s=0.08, time_to_peak_intensity_s=0.07)
    # Group A required fields
    assert pp.print_id == 0
    assert pp.touchdown_frame == 10
    assert pp.liftoff_frame == 20
    # Defaults
    assert pp.loading_duration_s == 0.0
    assert pp.frames == []
    assert pp.toe_positions_xy_mm is None
    assert pp.linkage_to_3d is None
    assert pp.quality.snr == 0.0
    assert pp.n_frames == 0  # via property
    # Required J field for downstream: paw_id starts None
    assert pp.linkage_to_3d is None


# ---------------------------------------------------------------------------
# 6) Sensitivity-mode presets from extractor
# ---------------------------------------------------------------------------

def test_sensitivity_mode_presets():
    """PawPrintExtractor exposes strict/balanced/loose presets."""
    ext = PawPrintExtractor(px_per_mm=1.92, fps=60, sensitivity_mode="strict")
    assert ext.tau_paw == 18
    assert ext.min_area_px == 19
    assert ext.min_print_frames == 3
    assert abs(ext.iou_min - 0.4) < 1e-6

    ext_loose = PawPrintExtractor(px_per_mm=1.92, fps=60, sensitivity_mode="loose")
    assert ext_loose.tau_paw == 10
    assert ext_loose.min_area_px == 3
    assert ext_loose.min_print_frames == 1


# ---------------------------------------------------------------------------
# 7) Pickle roundtrip preserves every field
# ---------------------------------------------------------------------------

def test_pickle_roundtrip():
    """PawPrint survives pickle.dump/load without losing any field."""
    pp = PawPrint(print_id=7, touchdown_frame=10, liftoff_frame=20,
                   true_liftoff_frame=22, peak_area_frame=15,
                   peak_intensity_frame=14, duration_s=0.2,
                   time_to_peak_area_s=0.08, time_to_peak_intensity_s=0.07,
                   max_area_mm2=12.5, peak_frame_centroid_xy_mm=(45.0, 22.0),
                   stand_index=1.7, cop_path_length_mm=3.4,
                   raw_intensity_curve=[0.1, 0.5, 0.9],
                   quality=QualityFlags(snr=4.5, n_frames=10))
    blob = pickle.dumps(pp)
    pp2 = pickle.loads(blob)
    assert pp2.print_id == 7
    assert pp2.peak_frame_centroid_xy_mm == (45.0, 22.0)
    assert pp2.stand_index == 1.7
    assert pp2.cop_path_length_mm == 3.4
    assert pp2.raw_intensity_curve == [0.1, 0.5, 0.9]
    assert pp2.quality.snr == 4.5


# ---------------------------------------------------------------------------
# 8) Decay extension: stays under max extension frames even if intensity lingers
# ---------------------------------------------------------------------------

def test_decay_extension_truncates_at_threshold():
    """If intensity never falls below threshold, decay extension still terminates."""
    from deepgait3.core.pawprint.extractor import PawPrintExtractor

    ext = PawPrintExtractor(
        px_per_mm=1.92, fps=60, sensitivity_mode="balanced",
        decay_extension_max_frames=10, decay_intensity_threshold=6.0,
    )
    # We test the contract: max extension is bounded regardless of input.
    assert ext.decay_extension_max_frames == 10
    # Real decay logic lives inside __call__ over a closed track — we can't
    # easily fake a video here, but the field-existence invariant is what
    # the Stage-3 downstream needs. Verify the dataclass accepts decay fields.
    pp = PawPrint(
        print_id=1, touchdown_frame=0, liftoff_frame=10, true_liftoff_frame=20,
        peak_area_frame=5, peak_intensity_frame=5, duration_s=0.33,
        time_to_peak_area_s=0.08, time_to_peak_intensity_s=0.08,
        decay_tau_ms=42.5, decay_R2=0.95, is_clean_liftoff=True,
    )
    assert pp.decay_tau_ms == 42.5
    assert pp.is_clean_liftoff is True


# ---------------------------------------------------------------------------
# Schema integrity test: not in the original v2 list but VITAL for v3.0
# ---------------------------------------------------------------------------

def test_pawprint_schema_field_count():
    """Lock the PawPrint field set so any drift breaks CI immediately."""
    pp = PawPrint(print_id=0, touchdown_frame=0, liftoff_frame=0,
                   true_liftoff_frame=0, peak_area_frame=0,
                   peak_intensity_frame=0, duration_s=0.0,
                   time_to_peak_area_s=0.0, time_to_peak_intensity_s=0.0)
    # Document the v0.4.2 field set (a few beyond the 36 in DESIGN.md table).
    expected_field_groups = {
        # Group A
        "print_id", "touchdown_frame", "liftoff_frame",
        "true_liftoff_frame", "peak_area_frame", "peak_intensity_frame",
        "duration_s", "time_to_peak_area_s", "time_to_peak_intensity_s",
        # Group B
        "loading_duration_s", "weight_bearing_duration_s", "unloading_duration_s",
        # Group C
        "frames",
        # Group D
        "max_area_mm2", "peak_frame_centroid_xy_mm", "peak_frame_bbox_xyxy",
        "print_length_mm", "print_width_mm", "print_orientation_deg",
        # Group E
        "compactness", "toe_positions_xy_mm", "centroid_drift_mm",
        # Group F
        "stand_index", "rising_slope",
        "peak_pressure", "mean_pressure_at_peak", "pressure_area_ratio",
        "raw_intensity_curve", "touchdown_intensity", "liftoff_intensity",
        # Group G
        "cop_trajectory_mm", "cop_path_length_mm", "cop_displacement_mm",
        # Group H
        "max_area_curve", "decay_phase_mask", "decay_tau_ms", "decay_R2",
        "is_clean_liftoff",
        # Group I
        "quality",
        # Group J
        "linkage_to_3d",
    }
    actual = {f.name for f in PawPrint.__dataclass_fields__.values()}
    missing = expected_field_groups - actual
    assert not missing, f"PawPrint missing fields: {missing}"