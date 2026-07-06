"""Unit tests for FTIR → gait metrics pipeline (gait_ftir.py v2.1),
background model (background_model.py), and run detector (run_detector.py).

Run with::

    pytest tests/unit/test_gait_ftir.py -v
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mock_seq(LF=False, RF=False, LH=False, RH=False):
    """Create a minimal mock object that looks like a FootprintSequence."""
    return MockSeq({
        p: MockFoot(bool(v)) for p, v in
        (("LF", LF), ("RF", RF), ("LH", LH), ("RH", RH))
    })


class MockSeq:
    def __init__(self, feet):
        self.feet = feet


class MockFoot:
    def __init__(self, is_in_stance=False):
        self.is_in_stance = is_in_stance


def synthetic_in_stance(n_frames=200, fps=100, pattern=(15, 10)):
    """Generate LF=RF=LH=RH stance arrays (all paws in sync) with
    alternating stance/swing pattern."""
    arr = np.zeros(n_frames, dtype=np.int8)
    pos = 0
    while pos < n_frames:
        for length in pattern:
            if pos < n_frames:
                arr[pos:pos+length] = 1
                pos += length
            if pos < n_frames:
                pos += pattern[1] if length == pattern[0] else pattern[0]
    return {p: arr.copy() for p in ("LF", "RF", "LH", "RH")}


# ---------------------------------------------------------------------------
# gait_ftir unit tests
# ---------------------------------------------------------------------------
class TestTemporalStance:
    def test_build_empty(self):
        from deepgait3.core._legacy.gait_ftir import build_temporal_stance
        out = build_temporal_stance([], n_frames=0)
        assert list(out.keys()) == ["LF", "RF", "LH", "RH"]
        for v in out.values():
            assert v.shape == (0,)

    def test_build_with_frames(self):
        from deepgait3.core._legacy.gait_ftir import build_temporal_stance
        seqs = []
        for i in range(10):
            m = mock_seq(LF=(i % 3 == 0), RF=False, LH=(i % 2 == 0), RH=True)
            seqs.append(m)
        out = build_temporal_stance(seqs)
        assert len(out["LF"]) == 10
        assert out["LF"][0] == 1  # 0 % 3 == 0
        assert out["LF"][3] == 1
        assert out["RF"][0] == 0
        assert all(out["RH"] == 1)


class TestDetectRunBoundaries:
    def test_no_stance(self):
        from deepgait3.core._legacy.gait_ftir import detect_run_boundaries
        s, e = detect_run_boundaries({
            "LF": np.zeros(100, dtype=np.int8),
            "RF": np.zeros(100, dtype=np.int8),
            "LH": np.zeros(100, dtype=np.int8),
            "RH": np.zeros(100, dtype=np.int8),
        })
        assert s == 0 and e == 0

    def test_single_run(self):
        from deepgait3.core._legacy.gait_ftir import detect_run_boundaries
        arr = np.zeros(200, dtype=np.int8)
        arr[50:150] = 1
        s, e = detect_run_boundaries(
            {k: arr.copy() for k in ("LF", "RF", "LH", "RH")}
        )
        assert s == 50
        assert e == 150


class TestStanceSegments:
    def test_empty(self):
        from deepgait3.core._legacy.gait_ftir import compute_stance_segments
        assert compute_stance_segments(np.zeros(10, dtype=np.int8)) == []

    def test_single_segment(self):
        from deepgait3.core._legacy.gait_ftir import compute_stance_segments
        arr = np.zeros(20, dtype=np.int8)
        arr[5:15] = 1
        segs = compute_stance_segments(arr)
        assert segs == [(5, 15)]

    def test_multiple_segments(self):
        from deepgait3.core._legacy.gait_ftir import compute_stance_segments
        arr = np.zeros(50, dtype=np.int8)
        arr[2:8] = 1
        arr[20:30] = 1
        arr[40:45] = 1
        segs = compute_stance_segments(arr)
        assert len(segs) == 3
        assert segs[0] == (2, 8)
        assert segs[2] == (40, 45)


class TestPerStepMetrics:
    def test_basic(self):
        from deepgait3.core._legacy.gait_ftir import compute_per_step_metrics
        arr = np.zeros(50, dtype=np.int8)
        arr[10:20] = 1  # stance 1: 10 frames
        arr[30:38] = 1  # stance 2: 8 frames
        # Centroid X moves rightward
        cx = np.arange(50, dtype=float)
        intensity = np.ones(50, dtype=float) * 100
        steps = compute_per_step_metrics(arr, intensity, cx, fps=100, px_per_mm=1.0)
        assert len(steps) == 2
        assert steps[0]["stand_s"] == pytest.approx(0.10, abs=0.01)
        assert steps[0]["stance_frames"] == 10
        # step_length from stance 1 start (10) to stance 2 start (30) = 20 px = 20 mm
        assert steps[0]["step_length_mm"] == pytest.approx(20.0, abs=0.1)

    def test_braking_propulsion(self):
        from deepgait3.core._legacy.gait_ftir import compute_per_step_metrics
        arr = np.zeros(30, dtype=np.int8)
        arr[5:15] = 1
        # Intensity rising then falling (peak at index 8 = frame 13)
        intensity = np.array([0,0,0,0,0, 1,3,5,8,5, 3,2,1,0,0] + [0]*15, dtype=float)
        cx = np.arange(30, dtype=float)
        steps = compute_per_step_metrics(arr, intensity, cx, fps=100)
        assert len(steps) == 1
        assert steps[0]["braking_s"] > 0
        assert steps[0]["propulsion_s"] > 0
        assert 0 < steps[0]["braking_index"] < 1
        assert 0 < steps[0]["propulsion_index"] < 1


class TestPerPawAggregates:
    def test_all_keys_present(self):
        from deepgait3.core._legacy.gait_ftir import compute_per_paw_aggregates
        arr = np.zeros(100, dtype=np.int8)
        arr[10:40] = 1
        arr[60:80] = 1
        intensity = np.random.rand(100) * 200
        cx = np.arange(100, dtype=float) * 0.5
        m = compute_per_paw_aggregates(arr, intensity, cx, None, fps=100, px_per_mm=2.0)
        expected_keys = [
            "max_contact_area_cm2", "total_contact_s", "max_contact_s",
            "step_cycle_s", "stride_length_cm", "step_length_cm",
            "avg_stance_s", "avg_swing_s", "duty_cycle_pct", "swing_phase_pct",
            "avg_braking_s", "avg_propulsion_s", "braking_index", "propulsion_index",
            "avg_swing_speed_cm_s", "avg_instantaneous_speed_cm_s",
            "total_speed_cm_s", "peak_swing_speed_cm_s",
            "max_intensity", "avg_max_intensity", "total_auc", "n_steps",
        ]
        for k in expected_keys:
            assert k in m, f"missing key {k}"

    def test_empty_signal(self):
        from deepgait3.core._legacy.gait_ftir import compute_per_paw_aggregates
        m = compute_per_paw_aggregates(
            np.zeros(3, dtype=np.int8), None, None, None, 100,
        )
        assert m["avg_stance_s"] == 0.0
        assert m["avg_swing_s"] == 0.0
        assert m["n_steps"] == 0

    def test_stance_calculation(self):
        from deepgait3.core._legacy.gait_ftir import compute_per_paw_aggregates
        arr = np.array([0,0,1,1,1,0,0,0,1,1,1,1,0,0], dtype=np.int8)
        intensity = np.ones_like(arr, dtype=float) * 100
        cx = np.arange(len(arr), dtype=float)
        m = compute_per_paw_aggregates(arr, intensity, cx, None, fps=100)
        # stand = mean([3,4]) / 100 = 0.035, swing = 3 / 100 = 0.03
        assert 0.02 < (m["avg_stance_s"] + m["avg_swing_s"]) < 0.08
        assert 40 < m["duty_cycle_pct"] <= 70


class TestDualStance:
    def test_no_overlap(self):
        from deepgait3.core._legacy.gait_ftir import compute_dual_stance
        in_stance = {
            "LF": np.zeros(100, dtype=np.int8),
            "RF": np.zeros(100, dtype=np.int8),
            "LH": np.zeros(100, dtype=np.int8),
            "RH": np.zeros(100, dtype=np.int8),
        }
        result = compute_dual_stance(in_stance, fps=100)
        assert result == {} or result.get("total_active_frames", 0) == 0

    def test_perfect_overlap(self):
        from deepgait3.core._legacy.gait_ftir import compute_dual_stance
        arr = np.ones(100, dtype=np.int8)
        in_stance = {p: arr.copy() for p in ("LF", "RF", "LH", "RH")}
        result = compute_dual_stance(in_stance, fps=100)
        assert result["total_active_frames"] == 100
        assert result["dual_LF_LH_pct"] == 100.0
        assert result["dual_LF_RH_pct"] == 100.0


class TestCoordination:
    def test_alternating(self):
        from deepgait3.core._legacy.gait_ftir import compute_coordination
        # LF and RH alternate
        lf = np.zeros(50, dtype=np.int8)
        rh = np.zeros(50, dtype=np.int8)
        for i in range(0, 50, 10):
            lf[i:i+5] = 1
            rh[i+5:i+10] = 1
        in_stance = {"LF": lf, "RH": rh}
        result = compute_coordination(in_stance, fps=100)
        # When LF starts stance, RH is NOT in stance (they alternate)
        assert "homologous_stance_pct" not in result  # no LH/RH pair
        # diagonal_LH_RF also missing since no LH/RF


class TestCatwalkMetrics:
    def test_full_pipeline(self):
        from deepgait3.core._legacy.gait_ftir import compute_catwalk_equivalent_metrics
        in_stance = synthetic_in_stance(n_frames=200, fps=100,
                                        pattern=([15, 8] * 5))
        intensity = {p: np.random.rand(200) * 200 for p in ("LF", "RF", "LH", "RH")}
        cx = {p: np.arange(200, dtype=float) for p in ("LF", "RF", "LH", "RH")}
        result = compute_catwalk_equivalent_metrics(
            in_stance, intensity, centroids_x=cx, fps=100,
        )
        # Per-paw
        for paw in ("LF", "RF", "LH", "RH"):
            assert f"{paw}_n_steps" in result
            assert f"{paw}_avg_stance_s" in result
            assert f"{paw}_duty_cycle_pct" in result
        # Run-level
        assert result["run_duration_s"] > 0
        assert result["n_steps"] > 0
        # Body speed
        assert "body_speed_cm_s" in result
        # Coordination
        assert "homologous_stance_pct" in result

    def test_legacy_wrapper(self):
        from deepgait3.core._legacy.gait_ftir import compute_ftir_gait_metrics
        in_stance = synthetic_in_stance(n_frames=100, fps=100,
                                        pattern=([20, 10] * 3))
        intensity = {p: np.random.rand(100) * 200 for p in ("LF", "RF", "LH", "RH")}
        result = compute_ftir_gait_metrics(in_stance, intensity, fps=100)
        # Legacy wrapper should still produce run_duration_s and n_steps
        assert result["run_duration_s"] > 0
        assert result["n_steps"] > 0


# ---------------------------------------------------------------------------
# background_model unit tests
# ---------------------------------------------------------------------------
class TestRollingMedianBackground:
    def test_not_ready_initially(self):
        from deepgait3.core._legacy.background_model import RollingMedianBackground
        bg = RollingMedianBackground(warmup_frames=30)
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert bg.ready is False

    def test_warmup_and_foreground(self):
        from deepgait3.core._legacy.background_model import RollingMedianBackground
        bg = RollingMedianBackground(warmup_frames=5, update_every=1, window_size=10)
        bg_img = np.full((80, 60, 3), 100, dtype=np.uint8)
        for _ in range(10):
            bg.update(bg_img)
        assert bg.ready is True
        fg = bg.foreground_mask(bg_img)
        assert fg.sum() == 0
        fg_img = bg_img.copy()
        fg_img[30:40, 20:30] = 200
        fg2 = bg.foreground_mask(fg_img)
        assert fg2.sum() > 0


# ---------------------------------------------------------------------------
# run_detector unit tests
# ---------------------------------------------------------------------------
class TestRunDetector:
    def test_waits_then_runs(self):
        from deepgait3.core._legacy.run_detector import RunDetector, RunDetectorState
        d = RunDetector(fps=100, min_frames_for_run=5, empty_frames_to_finish=5)
        assert d.state == RunDetectorState.WAITING
        r = d.process_frame(mock_seq(LF=False, RF=False, LH=False, RH=False))
        assert r is None
        assert d.state == RunDetectorState.WAITING
        r = d.process_frame(mock_seq(LF=True, RF=False, LH=False, RH=False))
        assert r is None
        assert d.state == RunDetectorState.RUNNING

    def test_finishes_after_empty_frames(self):
        from deepgait3.core._legacy.run_detector import RunDetector
        d = RunDetector(fps=100, min_frames_for_run=3, empty_frames_to_finish=4)
        for _ in range(10):
            d.process_frame(mock_seq(LF=True, RF=False, LH=False, RH=False))
        result = None
        for _ in range(4):
            result = d.process_frame(mock_seq(LF=False, RF=False, LH=False, RH=False))
        assert result is not None, "4th empty frame should trigger finish"
        assert result.n_frames > 0

    def test_too_short_ignored(self):
        from deepgait3.core._legacy.run_detector import RunDetector
        d = RunDetector(fps=100, min_frames_for_run=10, empty_frames_to_finish=3)
        d.process_frame(mock_seq(LF=True))
        d.process_frame(mock_seq(LF=True))
        for _ in range(4):
            r = d.process_frame(mock_seq(LF=False))
        assert r is None

    def test_multi_run(self):
        from deepgait3.core._legacy.run_detector import RunDetector
        d = RunDetector(fps=100, min_frames_for_run=3, empty_frames_to_finish=4)
        results = []
        for _ in range(8):
            r = d.process_frame(mock_seq(LF=True))
            if r: results.append(r)
        for _ in range(4):
            r = d.process_frame(mock_seq(LF=False))
            if r: results.append(r)
        for _ in range(6):
            r = d.process_frame(mock_seq(RF=True))
            if r: results.append(r)
        for _ in range(4):
            r = d.process_frame(mock_seq(RF=False))
            if r: results.append(r)
        assert len(results) >= 1, f"expected at least 1 run, got {len(results)}"

    def test_finish_force(self):
        from deepgait3.core._legacy.run_detector import RunDetector
        d = RunDetector(fps=100, min_frames_for_run=3, empty_frames_to_finish=99)
        for _ in range(12):
            d.process_frame(mock_seq(LF=True))
        result = d.finish()
        assert result is not None
        assert result.n_frames > 0
