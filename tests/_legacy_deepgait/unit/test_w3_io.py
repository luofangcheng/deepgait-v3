"""Unit tests for Phase 1 W3 — Layer 2 I/O.

Covers:
    deepgait/io/h5_writer.py     — TrialH5Writer
    deepgait/io/h5_reader.py     — TrialH5Reader
    deepgait/io/bids_exporter.py — BidsExporter
    deepgait/io/nwb_exporter.py  — NwbExporter (skipped when pynwb missing)

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.1 W3):
    "H5 write/read round-trips consistently"
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest


# Computed once at import time — True iff pynwb is importable.
def _pynwb_present() -> bool:
    import importlib.util

    return importlib.util.find_spec("pynwb") is not None


_PYNWB_PRESENT = _pynwb_present()


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def small_trial_arrays():
    """Synthetic but schema-correct trial data, ~3 frames, 4 cameras."""
    rng = np.random.default_rng(7)
    n_frames = 3
    cameras = {
        "left":  {"pose_2d": rng.random((n_frames, 12, 3)).astype(np.float32)},
        "right": {"pose_2d": rng.random((n_frames, 12, 3)).astype(np.float32)},
        "top":   {"pose_2d": rng.random((n_frames, 12, 3)).astype(np.float32)},
        "bottom_ftir": {
            "ftir_intensity": rng.random((n_frames, 16, 16)).astype(np.float32),
        },
    }
    pose_3d = {
        "positions": rng.random((n_frames, 12, 3)).astype(np.float32),
        "reprojection_error": rng.random((n_frames, 12)).astype(np.float32),
        "in_stance": rng.random((n_frames, 4)) > 0.5,
        "confidence": rng.random((n_frames, 12)).astype(np.float32),
    }
    gait_per_paw = {
        "LF": {
            "stance_duration": np.array([0.05, 0.06, 0.07], np.float32),
            "swing_duration":  np.array([0.10, 0.11, 0.12], np.float32),
            "stride_length":   np.array([2.5, 2.6, 2.7], np.float32),
        },
        "RF": {
            "stance_duration": np.array([0.05, 0.06, 0.07], np.float32),
            "swing_duration":  np.array([0.10, 0.11, 0.12], np.float32),
            "stride_length":   np.array([2.5, 2.6, 2.7], np.float32),
        },
    }
    coordination = {"regularity_index": 95.0, "symmetry_l_r": 0.97}
    summary = {"mean_stride_length": 2.6, "n_strides": 3, "speed_cm_s": 26.0}
    return {
        "n_frames": n_frames,
        "cameras": cameras,
        "pose_3d": pose_3d,
        "gait_per_paw": gait_per_paw,
        "coordination": coordination,
        "summary": summary,
    }


def _write_full_trial(writer, data):
    """Drive every writer method with the fixture data."""
    from deepgait3.io.h5_writer import TrialMetadata

    writer.write_metadata(TrialMetadata(
        animal_id="M001", species="mouse", strain="C57BL/6",
        operator="alice", device_serial="DONGLE-0001",
        config_hash="abc123",
    ))
    for role, payload in data["cameras"].items():
        if "pose_2d" in payload:
            writer.write_pose_2d(role, payload["pose_2d"])
        if "ftir_intensity" in payload:
            writer.write_ftir_intensity(
                role, payload["ftir_intensity"],
                fps=100, exposure_us=5000.0,
            )
    writer.write_pose_3d(**data["pose_3d"])
    writer.write_gait_metrics(
        data["gait_per_paw"], coordination=data["coordination"],
        summary=data["summary"],
    )


# =============================================================================
# TrialH5Writer — schema validation
# =============================================================================
class TestH5WriterValidation:
    def test_pose_2d_shape_validation(self, tmp_path):
        from deepgait3.io.h5_writer import TrialH5Writer

        with TrialH5Writer(tmp_path / "x.h5") as w:
            w.write_metadata(animal_id="M001")
            with pytest.raises(ValueError):
                w.write_pose_2d("left", np.zeros((10, 5, 3), np.float32))
            with pytest.raises(ValueError):
                w.write_pose_2d("left", np.zeros((10, 12, 2), np.float32))

    def test_pose_3d_shape_consistency(self, tmp_path):
        from deepgait3.io.h5_writer import TrialH5Writer

        with TrialH5Writer(tmp_path / "x.h5") as w:
            w.write_metadata(animal_id="M001")
            with pytest.raises(ValueError):
                # reprojection_error has wrong N
                w.write_pose_3d(
                    positions=np.zeros((10, 12, 3), np.float32),
                    reprojection_error=np.zeros((9, 12), np.float32),
                    in_stance=np.zeros((10, 4), np.bool_),
                    confidence=np.zeros((10, 12), np.float32),
                )

    def test_video_shape_validation(self, tmp_path):
        from deepgait3.io.h5_writer import TrialH5Writer

        with TrialH5Writer(tmp_path / "x.h5") as w:
            w.write_metadata(animal_id="M001")
            with pytest.raises(ValueError):
                # Missing channel dim
                w.write_camera_video("left", np.zeros((10, 64, 64), np.uint8),
                                     fps=100, exposure_us=5000.0)

    def test_context_manager_creates_file(self, tmp_path):
        from deepgait3.io.h5_writer import TrialH5Writer

        p = tmp_path / "trial.h5"
        with TrialH5Writer(p) as w:
            w.write_metadata(animal_id="M001")
        assert p.is_file()
        assert p.stat().st_size > 0


# =============================================================================
# H5 round-trip — W3 acceptance gate
# =============================================================================
class TestH5RoundTrip:
    def test_full_round_trip_preserves_all_data(self, tmp_path, small_trial_arrays):
        """W3 acceptance: write → read must be byte-identical."""
        from deepgait3.io.h5_writer import TrialH5Writer
        from deepgait3.io.h5_reader import TrialH5Reader

        path = tmp_path / "trial.h5"
        with TrialH5Writer(path) as w:
            _write_full_trial(w, small_trial_arrays)

        with TrialH5Reader(path) as r:
            trial = r.read_all()

        # metadata
        assert trial.metadata["animal_id"] == "M001"
        assert trial.metadata["species"] == "mouse"
        assert trial.metadata["strain"] == "C57BL/6"
        assert trial.metadata["operator"] == "alice"
        assert trial.metadata["device_serial"] == "DONGLE-0001"
        assert trial.metadata["config_hash"] == "abc123"

        # cameras
        for role in ("left", "right", "top"):
            cam = trial.cameras[role]
            np.testing.assert_allclose(
                cam.pose_2d, small_trial_arrays["cameras"][role]["pose_2d"],
            )
        ftir = trial.cameras["bottom_ftir"]
        np.testing.assert_allclose(
            ftir.ftir_intensity,
            small_trial_arrays["cameras"]["bottom_ftir"]["ftir_intensity"],
        )
        assert ftir.fps == 100
        assert ftir.exposure_us == 5000.0

        # pose_3d
        p3d_in = small_trial_arrays["pose_3d"]
        np.testing.assert_allclose(trial.pose_3d.positions, p3d_in["positions"])
        np.testing.assert_allclose(
            trial.pose_3d.reprojection_error, p3d_in["reprojection_error"]
        )
        np.testing.assert_array_equal(trial.pose_3d.in_stance, p3d_in["in_stance"])
        np.testing.assert_allclose(trial.pose_3d.confidence, p3d_in["confidence"])

        # gait
        for paw, metrics in small_trial_arrays["gait_per_paw"].items():
            for name, values in metrics.items():
                np.testing.assert_allclose(
                    trial.gait_per_paw[paw][name], values,
                    err_msg=f"mismatch in {paw}.{name}",
                )
        assert trial.coordination == pytest.approx(
            small_trial_arrays["coordination"]
        )
        assert trial.summary == pytest.approx(small_trial_arrays["summary"])

    def test_dtypes_preserved(self, tmp_path, small_trial_arrays):
        """dtype preservation is part of the round-trip contract."""
        from deepgait3.io.h5_writer import TrialH5Writer
        from deepgait3.io.h5_reader import TrialH5Reader

        path = tmp_path / "dtypes.h5"
        with TrialH5Writer(path) as w:
            _write_full_trial(w, small_trial_arrays)
        with TrialH5Reader(path) as r:
            trial = r.read_all()
        assert trial.pose_3d.positions.dtype == np.float32
        assert trial.pose_3d.in_stance.dtype == np.bool_
        for cam in trial.cameras.values():
            if cam.pose_2d is not None:
                assert cam.pose_2d.dtype == np.float32

    def test_read_missing_file_raises(self, tmp_path):
        from deepgait3.io.h5_reader import TrialH5Reader

        with pytest.raises(FileNotFoundError):
            with TrialH5Reader(tmp_path / "nope.h5") as r:
                r.read_metadata()

    def test_schema_version_stamped_on_file(self, tmp_path):
        from deepgait3.io.h5_writer import TrialH5Writer, SCHEMA_VERSION
        import h5py

        path = tmp_path / "v.h5"
        with TrialH5Writer(path) as w:
            w.write_metadata(animal_id="M001")
        with h5py.File(path, "r") as f:
            ver = f.attrs["schema_version"]
            ver = ver.decode() if isinstance(ver, bytes) else ver
            assert ver == SCHEMA_VERSION

    def test_calibration_round_trip(self, tmp_path):
        from deepgait3.io.h5_writer import TrialH5Writer
        from deepgait3.io.h5_reader import TrialH5Reader

        cams = {
            "left": {
                "K": np.eye(3, dtype=np.float64),
                "dist": np.zeros(5, dtype=np.float64),
                "rvec": np.zeros(3, dtype=np.float64),
                "tvec": np.zeros(3, dtype=np.float64),
            },
        }
        path = tmp_path / "calib.h5"
        with TrialH5Writer(path) as w:
            w.write_metadata(animal_id="M001")
            w.write_calibration(
                cams,
                charuco={"rows": 5, "cols": 7, "square_size": 25},
                reprojection_error_rms=0.42,
            )
        with TrialH5Reader(path) as r:
            trial = r.read_all()
        np.testing.assert_allclose(trial.calibration["left"]["K"], np.eye(3))
        assert trial.calibration["left"]["dist"].shape == (5,)

    def test_audit_and_licensing_round_trip(self, tmp_path):
        from deepgait3.io.h5_writer import TrialH5Writer
        from deepgait3.io.h5_reader import TrialH5Reader

        path = tmp_path / "audit.h5"
        with TrialH5Writer(path) as w:
            w.write_metadata(animal_id="M001")
            w.write_audit(
                log_lines=["start", "frame 1", "frame 2"],
                hmac_chain=np.zeros((4,), dtype=np.uint8),
            )
            w.write_licensing(
                dongle_id="DONGLE-0001",
                license_type="single",
                features_bitmap=0xFF,
                last_check_timestamp="2026-06-19T00:00:00",
            )
        with TrialH5Reader(path) as r:
            trial = r.read_all()
        assert trial.audit_log == ["start", "frame 1", "frame 2"]
        assert trial.audit_hmac_chain.shape == (4,)
        assert trial.licensing["dongle_id"] == "DONGLE-0001"
        assert trial.licensing["features_bitmap"] == 0xFF


# =============================================================================
# BidsExporter
# =============================================================================
class TestBidsExporter:
    def test_export_creates_bids_layout(self, tmp_path, small_trial_arrays):
        from deepgait3.io.bids_exporter import BidsExporter, BidsExportSpec

        # Synthesize a tiny video file so the exporter has something to copy.
        video_path = tmp_path / "left.mp4"
        video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # tiny stub

        spec = BidsExportSpec(
            animal_id="M001",
            species="mouse",
            trial_id="trial_001",
            experiment_date="2026-06-19T10:00:00",
            operator="alice",
            app_version="2.0.0",
            n_frames=small_trial_arrays["n_frames"],
            fps=100,
            cameras={
                "left": {
                    "video": video_path,
                    "pose_2d": small_trial_arrays["cameras"]["left"]["pose_2d"],
                },
            },
            gait_per_paw=small_trial_arrays["gait_per_paw"],
            coordination=small_trial_arrays["coordination"],
            summary=small_trial_arrays["summary"],
        )
        out = BidsExporter(tmp_path / "bids").export(spec)
        assert out.is_dir()
        assert (out / "dataset_description.json").is_file()
        assert (out / "participants.tsv").is_file()
        assert (out / "README.md").is_file()
        # subject + session layout
        sub = out / "sub-M001"
        assert sub.is_dir()
        sess = next((sub / d for d in sub.iterdir() if d.name.startswith("ses-")))
        assert (sess / "sub-M001_cam-left_video.mp4").is_file()
        assert (sess / "sub-M001_cam-left_video.json").is_file()
        beh = sess / "behav"
        assert (beh / "sub-M001_cam-left_pose.tsv.gz").is_file()
        # per-paw metrics
        for paw in spec.gait_per_paw:
            assert (beh / f"sub-M001_paw-{paw}_metrics.tsv.gz").is_file()
        assert (beh / "sub-M001_coordination.json").is_file()
        assert (beh / "sub-M001_summary.json").is_file()

    def test_dataset_description_is_valid_json(self, tmp_path, small_trial_arrays):
        import json as _json
        from deepgait3.io.bids_exporter import BidsExporter, BidsExportSpec

        spec = BidsExportSpec(
            animal_id="M002", species="rat", trial_id="t2",
            experiment_date="2026-06-19T10:00:00",
            operator="bob", app_version="2.0.0",
            n_frames=10, fps=100, cameras={},
        )
        out = BidsExporter(tmp_path / "b").export(spec)
        desc = _json.loads((out / "dataset_description.json").read_text())
        assert desc["Name"]
        assert desc["GeneratedBy"][0]["Version"] == "2.0.0"


# =============================================================================
# NwbExporter
# =============================================================================
class TestNwbExporter:
    def test_export_raises_when_pynwb_missing(self, tmp_path, monkeypatch):
        """When pynwb is unavailable, export() must raise NwbUnavailable,
        NOT crash on import. This is the W3 robustness contract."""
        from deepgait3.io import nwb_exporter
        from deepgait3.io.nwb_exporter import NwbExporter, NwbExportSpec, NwbUnavailable

        monkeypatch.setattr(nwb_exporter, "PYNWB_AVAILABLE", False)
        spec = NwbExportSpec(
            animal_id="M001", species="mouse", strain="C57BL/6",
            operator="alice", experiment_date=datetime.now(timezone.utc),
            trial_id="t1", fps=100,
        )
        with pytest.raises(NwbUnavailable):
            NwbExporter(tmp_path / "x.nwb").export(spec)

    @pytest.mark.skipif(not _PYNWB_PRESENT, reason="pynwb not installed")
    def test_export_writes_nwb_when_pynwb_available(self, tmp_path):
        # Only runs when pynwb is actually installed.
        from deepgait3.io.nwb_exporter import (
            NwbExporter, NwbExportSpec, PYNWB_AVAILABLE,
        )
        if not PYNWB_AVAILABLE:
            pytest.skip("pynwb not installed")
        rng = np.random.default_rng(1)
        spec = NwbExportSpec(
            animal_id="M001", species="mouse", strain="C57BL/6",
            operator="alice", experiment_date=datetime.now(timezone.utc),
            trial_id="t1", fps=100,
            pose_3d_positions=rng.random((5, 12, 3)).astype(np.float32),
            pose_3d_confidence=rng.random((5, 12)).astype(np.float32),
            in_stance=rng.random((5, 4)) > 0.5,
            gait_summary={"mean_stride_length": 2.6},
        )
        out = NwbExporter(tmp_path / "trial.nwb").export(spec)
        assert out.is_file()
        assert out.stat().st_size > 0
