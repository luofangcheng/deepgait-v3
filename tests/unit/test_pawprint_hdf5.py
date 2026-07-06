"""Unit tests for the HDF5 PawPrint database.

Verifies round-trip integrity: write a synthetic PawPrint[] + TrialMeta,
read it back, confirm every field lands in the right HDF5 location and
survives a fresh process (we open a second h5py.File handle on the same
on-disk file).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from deepgait3.core.io.pawprint_hdf5 import (
    DetectorInfo,
    PawPrintH5Reader,
    PawPrintH5Writer,
    TrialMeta,
)
from deepgait3.core.pawprint import FrameData, PawPrint, QualityFlags


def _make_peak_frame(h: int = 20, w: int = 20) -> FrameData:
    raw = np.full((h, w), 50.0, dtype=np.float32)
    raw[5:15, 5:15] = 200.0  # bright square in the middle
    mask = np.zeros((h, w), dtype=bool)
    mask[5:15, 5:15] = True
    pressure = mask.astype(np.float32) * 100.0
    return FrameData(
        frame=42, time_s=0.7,
        bbox_xyxy=(5, 5, 15, 15),
        bbox_xyxy_padded=(0, 0, 20, 20),
        raw_intensity_crop=raw,
        bg_intensity_crop=np.full((h, w), 10.0, dtype=np.float32),
        paw_mask=mask,
        pressure_map=pressure,
        centroid_xy_mm=(10.0, 10.0),
        area_mm2=42.0,
        mean_intensity_in_mask=150.0,
        mean_pressure=100.0,
        peak_intensity=200.0,
        peak_pressure=120.0,
    )


def _make_synthetic_pawprints(n: int = 3) -> list[PawPrint]:
    out = []
    for i in range(n):
        pp = PawPrint(
            print_id=i,
            touchdown_frame=10 + i * 50,
            liftoff_frame=20 + i * 50,
            true_liftoff_frame=22 + i * 50,
            peak_area_frame=15 + i * 50,
            peak_intensity_frame=14 + i * 50,
            duration_s=0.2,
            time_to_peak_area_s=0.08,
            time_to_peak_intensity_s=0.07,
            max_area_mm2=12.5 + i,
            peak_frame_centroid_xy_mm=(100.0 + i * 50, 50.0),
            peak_frame_bbox_xyxy=(80, 40, 120, 60),
            stand_index=1.7 + i * 0.1,
            cop_trajectory_mm=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            cop_path_length_mm=10.0 + i,
            raw_intensity_curve=[0.1, 0.5, 0.9, 0.3],
            max_area_curve=[5.0, 8.0, 10.0, 12.0],
            decay_phase_mask=[False, False, True, True],
            decay_tau_ms=42.0 + i,
            decay_R2=0.95,
            is_clean_liftoff=True,
            quality=QualityFlags(snr=4.5, n_frames=4),
        )
        # Attach 4 frames so timeseries + images can be written
        pp.frames = [_make_peak_frame() for _ in range(4)]
        # Override the peak area frame in one of them
        pp.frames[2].frame = pp.peak_area_frame
        out.append(pp)
    return out


# ---------------------------------------------------------------------------
# round-trip tests
# ---------------------------------------------------------------------------

def test_write_then_read_meta(tmp_path: Path):
    h5_path = tmp_path / "trial.h5"
    meta = TrialMeta(trial_id="2026-06-25_mouse03_baseline",
                      animal_id="mouse03", treatment="baseline",
                      fps=60, px_per_mm=1.92, walkway_length_mm=1000.0,
                      ftir_video_path="ftir.mp4", timestamp="2026-06-25T14:00",
                      notes="first trial")
    with PawPrintH5Writer(h5_path, meta) as w:
        w.set_detector(DetectorInfo(algorithm="exg", threshold=80,
                                     min_area_px=5, sensitivity_mode="balanced"))
        for pp in _make_synthetic_pawprints(3):
            w.add_print(pp)

    with PawPrintH5Reader(h5_path) as r:
        assert r.meta["trial_id"] == "2026-06-25_mouse03_baseline"
        assert r.meta["fps"] == 60
        assert r.meta["px_per_mm"] == 1.92
        assert r.detector["algorithm"] == "exg"
        assert r.detector["threshold"] == 80


def test_write_then_read_print_metadata(tmp_path: Path):
    h5_path = tmp_path / "trial.h5"
    with PawPrintH5Writer(h5_path, TrialMeta(trial_id="t1")) as w:
        for pp in _make_synthetic_pawprints(2):
            w.add_print(pp)

    with PawPrintH5Reader(h5_path) as r:
        m = r.print_meta(1)
        # 36+ fields stored under /pawprints/print_001/meta
        assert m["print_id"] == 1
        assert m["touchdown_frame"] == 60
        assert m["liftoff_frame"] == 70
        assert m["peak_area_frame"] == 65
        assert m["max_area_mm2"] == 13.5
        assert m["peak_frame_centroid_x_mm"] == 150.0
        assert m["peak_frame_centroid_y_mm"] == 50.0
        assert m["stand_index"] == pytest.approx(1.8, abs=1e-4)
        assert m["decay_tau_ms"] == pytest.approx(43.0, abs=1e-4)
        assert bool(m["is_clean_liftoff"]) is True
        # Stage 2 fields still empty
        assert m["paw_id"] == ""


def test_write_then_read_timeseries(tmp_path: Path):
    h5_path = tmp_path / "trial.h5"
    with PawPrintH5Writer(h5_path, TrialMeta(trial_id="t1")) as w:
        w.add_print(_make_synthetic_pawprints(1)[0])

    with PawPrintH5Reader(h5_path) as r:
        times = r.timeseries(0, "time_s")
        areas = r.timeseries(0, "area_mm2")
        decay = r.timeseries(0, "decay_phase")
        # 4 frames were attached
        assert times.shape == (4,)
        assert areas.shape == (4,)
        assert decay.shape == (4,)
        assert decay.dtype == bool
        assert decay[-1] == True  # last frame marked as decay phase


def test_write_then_read_image(tmp_path: Path):
    h5_path = tmp_path / "trial.h5"
    with PawPrintH5Writer(h5_path, TrialMeta(trial_id="t1")) as w:
        w.add_print(_make_synthetic_pawprints(1)[0])

    with PawPrintH5Reader(h5_path) as r:
        bgr = r.image(0, "peak_frame_bgr")
        mask = r.image(0, "peak_mask")
        pmap = r.image(0, "peak_pressure_map")
        assert bgr.ndim == 2  # single channel raw intensity (stored as float32)
        assert bgr.shape == (20, 20)
        assert mask.shape == (20, 20)
        assert mask.dtype == bool
        assert mask[10, 10] == True   # center of paw
        assert mask[0, 0] == False    # outside paw
        assert pmap.shape == (20, 20)


def test_write_then_read_flat_index(tmp_path: Path):
    """The /index/ flat arrays let us query across prints without iterating."""
    h5_path = tmp_path / "trial.h5"
    pps = _make_synthetic_pawprints(5)
    with PawPrintH5Writer(h5_path, TrialMeta(trial_id="t1")) as w:
        for pp in pps:
            w.add_print(pp)

    with PawPrintH5Reader(h5_path) as r:
        ids = r.flat_index("print_ids")
        max_areas = r.flat_index("max_areas_mm2")
        durations = r.flat_index("durations_s")
        centroids = r.flat_index("print_centroids_x_mm")
        assert ids.shape == (5,)
        assert np.array_equal(ids, np.array([0, 1, 2, 3, 4]))
        assert np.allclose(max_areas, [12.5, 13.5, 14.5, 15.5, 16.5])
        assert np.allclose(durations, [0.2] * 5)
        assert np.allclose(centroids, [100.0, 150.0, 200.0, 250.0, 300.0])


def test_read_via_iterator(tmp_path: Path):
    h5_path = tmp_path / "trial.h5"
    with PawPrintH5Writer(h5_path, TrialMeta(trial_id="t1")) as w:
        for pp in _make_synthetic_pawprints(3):
            w.add_print(pp)

    with PawPrintH5Reader(h5_path) as r:
        records = list(r)
        assert len(records) == 3
        assert [m["print_id"] for m in records] == [0, 1, 2]


def test_paw_id_none_round_trips_as_empty_string(tmp_path: Path):
    """Stage 1 doesn't set paw_id; the database stores it as '' until Stage 2."""
    h5_path = tmp_path / "trial.h5"
    with PawPrintH5Writer(h5_path, TrialMeta(trial_id="t1")) as w:
        w.add_print(_make_synthetic_pawprints(1)[0])

    with PawPrintH5Reader(h5_path) as r:
        m = r.print_meta(0)
        assert m["paw_id"] == ""
        assert np.isnan(m["match_distance_mm"])


def test_quality_fields_round_trip(tmp_path: Path):
    h5_path = tmp_path / "trial.h5"
    with PawPrintH5Writer(h5_path, TrialMeta(trial_id="t1")) as w:
        pp = _make_synthetic_pawprints(1)[0]
        pp.quality.touches_edge = True
        pp.quality.saturated_pixels_pct = 12.3
        pp.quality.snr = 7.8
        w.add_print(pp)

    with PawPrintH5Reader(h5_path) as r:
        # Quality is under /pawprints/print_000/quality (not meta)
        with h5py.File(h5_path, "r") as f:
            q = f["/pawprints/print_000/quality"]
            assert q.attrs["touches_edge"] is True or q.attrs["touches_edge"] == 1
            assert q.attrs["saturated_pixels_pct"] == pytest.approx(12.3, abs=1e-4)
            assert q.attrs["snr"] == pytest.approx(7.8, abs=1e-4)