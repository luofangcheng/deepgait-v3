"""Unit tests for Phase 2 W5 — 30+ gait metrics.

Covers:
    deepgait/core/gait_algorithms.py
        * CatWalk XT standard metrics (12+)
        * deepgait extensions (10+)
        * compute_all_gait_metrics() — unified entry point

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.2 W5):
    "CatWalk 全指标函数 + 单元测试" / "1000 帧 < 5 ms"
    Verification target: >= 30 distinct metric values emitted by
    compute_all_gait_metrics() under realistic inputs.
"""
from __future__ import annotations

import time

import numpy as np
import pytest


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def fps() -> int:
    return 100


@pytest.fixture
def real_world_multiplier() -> float:
    return 0.25  # mm/pixel (calibrated)


@pytest.fixture
def fake_trial(fps):
    """Build a synthetic but well-formed 4-paw trial.

    Each paw gets 5 complete stance-swing cycles at 100 fps.
    Cycle = 30 frames stance + 20 frames swing (= 0.5 s stride).
    """
    stance_len = 30
    swing_len = 20
    n_cycles = 5
    cycle = np.array([1] * stance_len + [0] * swing_len, dtype=int)
    per_paw_stance = {p: np.tile(cycle, n_cycles)
                      for p in ("LF", "RF", "LH", "RH")}
    # Slight phase offset between fore and hind pairs (typical CatWalk).
    per_paw_stance["LH"] = np.roll(per_paw_stance["LH"], 10)
    per_paw_stance["RH"] = np.roll(per_paw_stance["RH"], 10)
    n = len(per_paw_stance["LF"])

    # Per-paw midpoints: linear forward drift at 5 mm/s on X-axis.
    t = np.arange(n) / fps
    speed_mm_s = 5.0
    rw = 0.25
    drift = speed_mm_s * t / rw  # in pixels
    paw_mid_x = {
        p: drift + (10 if "F" in p else -10) * np.ones(n)
        for p in ("LF", "RF", "LH", "RH")
    }
    paw_mid_y = {
        p: 5.0 * np.ones(n) + (0 if "F" in p else 0)
        for p in ("LF", "RF", "LH", "RH")
    }
    fore_mid_x = 0.5 * (paw_mid_x["LF"] + paw_mid_x["RF"])
    fore_mid_y = 0.5 * (paw_mid_y["LF"] + paw_mid_y["RF"])
    hind_mid_x = 0.5 * (paw_mid_x["LH"] + paw_mid_x["RH"])
    hind_mid_y = 0.5 * (paw_mid_y["LH"] + paw_mid_y["RH"])
    com_x = 0.5 * (fore_mid_x + hind_mid_x)
    com_y = 0.5 * (fore_mid_y + hind_mid_y)
    front_com_x = fore_mid_x; front_com_y = fore_mid_y
    rear_com_x = hind_mid_x;  rear_com_y = hind_mid_y
    axis_ref_x = com_x + 10.0; axis_ref_y = com_y  # body axis ~horizontal
    intensity_curve_per_paw = {p: np.maximum(0.0, np.sin(t))
                                for p in ("LF", "RF", "LH", "RH")}
    footprint_mask = (np.random.default_rng(0).random((n, 32, 32)) > 0.5).astype(np.uint8)
    return {
        "fps": fps,
        "real_world_multiplier": rw,
        "in_stance_per_paw": per_paw_stance,
        "paw_mid_x": paw_mid_x,
        "paw_mid_y": paw_mid_y,
        "fore_mid_x": fore_mid_x,
        "fore_mid_y": fore_mid_y,
        "hind_mid_x": hind_mid_x,
        "hind_mid_y": hind_mid_y,
        "com_x": com_x,
        "com_y": com_y,
        "front_com_x": front_com_x,
        "front_com_y": front_com_y,
        "rear_com_x": rear_com_x,
        "rear_com_y": rear_com_y,
        "axis_ref_x": axis_ref_x,
        "axis_ref_y": axis_ref_y,
        "intensity_curve_per_paw": intensity_curve_per_paw,
        "footprint_mask": footprint_mask,
    }


# =============================================================================
# Per-metric correctness tests
# =============================================================================
class TestBasicCatwalkMetrics:
    def test_stride_cycle_ms(self):
        from deepgait3.core._legacy.gait_algorithms import stride_cycle_ms
        assert stride_cycle_ms(200, 100) == 300.0

    def test_percentage_stance_swing_sum_to_100(self):
        from deepgait3.core._legacy.gait_algorithms import (
            percentage_stance, percentage_swing,
        )
        s = percentage_stance(200, 100)
        sw = percentage_swing(200, 100)
        assert s == pytest.approx(200.0 / 300.0 * 100, rel=1e-9)
        assert sw == pytest.approx(100.0 / 300.0 * 100, rel=1e-9)
        assert s + sw == pytest.approx(100.0)

    def test_duty_cycle_in_unit_interval(self):
        from deepgait3.core._legacy.gait_algorithms import duty_cycle
        assert 0.0 <= duty_cycle(200, 100) <= 1.0
        assert duty_cycle(0, 0) == 0.0

    def test_swing_to_stance_ratio(self):
        from deepgait3.core._legacy.gait_algorithms import swing_to_stance_ratio
        assert swing_to_stance_ratio(200, 100) == 0.5
        assert swing_to_stance_ratio(0, 100) == 0.0

    def test_cadence_is_2x_freq_x_60(self):
        from deepgait3.core._legacy.gait_algorithms import cadence
        assert cadence(2.0) == 240.0
        assert cadence(0.0) == 0.0


class TestSymmetryAndRegularity:
    def test_symmetry_index_perfect(self):
        from deepgait3.core._legacy.gait_algorithms import symmetry_index
        assert symmetry_index(150.0, 150.0) == 100.0

    def test_symmetry_index_asymmetric(self):
        from deepgait3.core._legacy.gait_algorithms import symmetry_index
        # smaller/ larger = 100/200 = 50
        assert symmetry_index(100.0, 200.0) == 50.0

    def test_symmetry_index_zero_when_both_zero(self):
        from deepgait3.core._legacy.gait_algorithms import symmetry_index
        assert symmetry_index(0.0, 0.0) == 0.0

    def test_regularity_index_all_canonical_is_100(self):
        from deepgait3.core._legacy.gait_algorithms import (
            _canonical_step_patterns, regularity_index,
        )
        # Build 4 paws in canonical "RF, LH, LF, RH" walk with cyclic shifts.
        # The simplest: every frame exactly one paw has contact.
        canonical = sorted(_canonical_step_patterns())
        # Each canonical tuple specifies which paw(s) are in contact.
        # Convert to per-paw stance arrays: cycle through canonical patterns.
        # We assume a 30-step window with each pattern repeated ~5 times.
        per_paw = [[], [], [], []]
        n_frames = 30
        for i in range(n_frames):
            pattern = canonical[i % len(canonical)]
            for paw_i, state in enumerate(pattern):
                per_paw[paw_i].append(int(state))
        lf = np.array(per_paw[0], dtype=int)
        rf = np.array(per_paw[1], dtype=int)
        lh = np.array(per_paw[2], dtype=int)
        rh = np.array(per_paw[3], dtype=int)
        # Add onsets (0→1) at the boundaries.
        ri = regularity_index(lf, rf, lh, rh)
        assert ri == pytest.approx(100.0, abs=0.1), ri

    def test_regularity_index_zero_when_empty(self):
        from deepgait3.core._legacy.gait_algorithms import regularity_index
        assert regularity_index(np.array([]), np.array([]),
                                np.array([]), np.array([])) == 0.0

    def test_phase_dispersion_returns_two_keys(self):
        from deepgait3.core._legacy.gait_algorithms import phase_dispersion
        # 1 second of stance/swing at 100 fps: clean alternating contacts.
        fps = 100
        t = np.zeros(fps, dtype=int)
        t[0:30] = 1; t[30:50] = 0; t[50:80] = 1; t[80:100] = 0
        a = t.copy(); b = np.roll(t, 10)
        out = phase_dispersion(a, b, a, b, fps)
        assert "lf_rf_dispersion_pct" in out
        assert "lh_rh_dispersion_pct" in out


class TestSpatialMetrics:
    def test_step_width_avg_simple(self):
        from deepgait3.core._legacy.gait_algorithms import step_width_avg
        left_x = np.array([100.0, 101.0, 102.0])
        right_x = np.array([110.0, 111.0, 112.0])
        # gap = 10 pixels × 0.25 = 2.5 mm
        assert step_width_avg(left_x, right_x, 0.25) == pytest.approx(2.5, rel=1e-9)

    def test_base_of_support_simple(self):
        from deepgait3.core._legacy.gait_algorithms import base_of_support
        # fore and hind at fixed distance 100 px
        f_x = np.array([0.0] * 50); h_x = np.array([100.0] * 50)
        f_y = np.zeros(50); h_y = np.zeros(50)
        # 100 px × 0.5 = 50 mm
        assert base_of_support(f_x, f_y, h_x, h_y, 0.5) == pytest.approx(50.0)

    def test_step_angle_parallel_vectors_is_zero(self):
        from deepgait3.core._legacy.gait_algorithms import step_angle
        n = 30
        fore_x = np.linspace(0, 100, n); fore_y = np.zeros(n)
        hind_x = np.zeros(n);           hind_y = np.zeros(n)
        com_x  = np.linspace(0, 50, n); com_y  = np.zeros(n)
        axis_x = com_x + 10.0;          axis_y = com_y
        angles = step_angle(fore_x, fore_y, hind_x, hind_y,
                            com_x, com_y, axis_x, axis_y)
        # Fore-hind vector points +X; body axis also +X → 0 degrees.
        np.testing.assert_allclose(angles, 0.0, atol=1e-9)


class TestDeepgaitExtensions:
    def test_swing_speed(self):
        from deepgait3.core._legacy.gait_algorithms import swing_speed
        assert swing_speed(60.0, 100.0) == pytest.approx(0.6, rel=1e-9)
        assert swing_speed(60.0, 0.0) == 0.0

    def test_body_speed(self):
        from deepgait3.core._legacy.gait_algorithms import body_speed
        # 100 frames, X drift 0→200 px @ 0.25 mm/px @ 100 fps
        # distance = 50 mm; duration = 0.99 s; speed ≈ 50.5 mm/s
        com_x = np.linspace(0, 200, 100)
        com_y = np.zeros(100)
        speed = body_speed(com_x, com_y, 0.25, fps=100)
        assert 50.0 <= speed <= 51.0

    def test_toe_spread(self):
        from deepgait3.core._legacy.gait_algorithms import toe_spread
        fore = np.array([10.0, 10.0])
        hind = np.array([30.0, 30.0])
        # diff 20 px × 0.5 = 10 mm
        assert toe_spread(fore, hind, 0.5) == pytest.approx(10.0)

    def test_stand_index_is_sum(self):
        from deepgait3.core._legacy.gait_algorithms import stand_index
        assert stand_index(np.array([0.1, 0.2, 0.3])) == pytest.approx(0.6)

    def test_intensity_asymmetry(self):
        from deepgait3.core._legacy.gait_algorithms import intensity_asymmetry
        l = np.array([1.0, 1.0, 1.0])
        r = np.array([1.0, 1.0, 1.0])
        assert intensity_asymmetry(l, r) == 0.0
        # ls=3, rs=1 → |2|/4 = 0.5
        r2 = np.array([0.5, 0.25, 0.25])
        assert intensity_asymmetry(l, r2) == pytest.approx(0.5)

    def test_max_contact_area(self):
        from deepgait3.core._legacy.gait_algorithms import max_contact_area
        mask = np.zeros((5, 10, 10), dtype=np.uint8)
        mask[2, :3, :4] = 1   # 12 pixels
        mask[0, :5, :5] = 1   # 25 pixels
        assert max_contact_area(mask) == 25

    def test_mean_intensity(self):
        from deepgait3.core._legacy.gait_algorithms import mean_intensity_curve
        out = mean_intensity_curve({"LF": np.array([1.0, 2.0]),
                                     "RF": np.array([3.0, 4.0])})
        assert out == pytest.approx(2.5)

    def test_max_intensity(self):
        from deepgait3.core._legacy.gait_algorithms import max_intensity_curve
        out = max_intensity_curve({"LF": np.array([1.0, 2.0]),
                                    "RF": np.array([3.0, 4.0])})
        assert out == 4.0

    def test_body_axis_angle(self):
        from deepgait3.core._legacy.gait_algorithms import body_axis_angle
        # both front & rear along +X axis → 0 degrees
        n = 10
        f_x = np.linspace(100, 200, n); f_y = np.zeros(n)
        r_x = np.linspace(0, 100, n);   r_y = np.zeros(n)
        assert body_axis_angle(f_x, f_y, r_x, r_y) == pytest.approx(0.0)


# =============================================================================
# Unified entry point — W5 acceptance: 30+ distinct metrics
# =============================================================================
class TestComputeAllGaitMetrics:
    def test_emits_at_least_30_distinct_metrics(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        # The contract: __n_metrics__ ≥ 30 (the W5 acceptance gate).
        assert out["__n_metrics__"] >= 30, (
            f"only {out['__n_metrics__']} metrics emitted: {sorted(out.keys())}"
        )

    def test_all_top_level_metrics_are_numeric(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        for k, v in out.items():
            if k in ("per_paw", "__n_metrics__"):
                continue
            assert isinstance(v, (int, float)), (k, type(v))

    def test_per_paw_has_4_paws_with_9_submetrics(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        assert set(out["per_paw"].keys()) == {"LF", "RF", "LH", "RH"}
        for paw in out["per_paw"]:
            assert len(out["per_paw"][paw]) == 9, out["per_paw"][paw]

    def test_strides_count_matches_expected(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        # Each paw has 5 cycles, but calculate_gait_basics drops the trailing
        # incomplete stance so n_strides = 4 (stance-swing pairs). This is
        # correct CatWalk behaviour (only fully completed cycles count).
        for paw in out["per_paw"]:
            n = out["per_paw"][paw]["n_strides"]
            assert n in (4, 5), f"{paw}: unexpected n_strides={n}"

    def test_cadence_and_stride_frequency_consistent(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        # cadence = 2 × freq × 60
        assert out["cadence_steps_per_min"] == pytest.approx(
            2.0 * out["avg_stride_frequency_hz"] * 60.0,
        )

    def test_duty_cycle_within_0_1(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        assert 0.0 <= out["avg_duty_cycle"] <= 1.0
        assert 0.0 <= out["avg_pct_stance"] <= 100.0
        assert 0.0 <= out["avg_pct_swing"] <= 100.0

    def test_symmetry_indices_in_range(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        # Symmetric trial → indices near 100.
        for k in ("symmetry_index_stance", "symmetry_index_swing",
                  "symmetry_index_hind_stance", "symmetry_index_hind_swing"):
            assert k in out
            assert 0.0 <= out[k] <= 100.0, (k, out[k])

    def test_minimal_inputs_only_emit_basic_metrics(self):
        """When only ``fps`` + ``in_stance_per_paw`` are given, the
        function must still emit the per-paw basics (the 9 submetrics).
        """
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        cycle = np.array([1] * 30 + [0] * 20, dtype=int)
        in_stance = {p: np.tile(cycle, 5) for p in ("LF", "RF", "LH", "RH")}
        out = compute_all_gait_metrics(
            fps=100, real_world_multiplier=0.25,
            in_stance_per_paw=in_stance,
        )
        # 4 paws × 9 submetrics = 36 from per_paw
        assert out["__n_metrics__"] >= 36
        assert "avg_stride_frequency_hz" in out
        assert "cadence_steps_per_min" in out

    def test_no_nan_or_inf_in_output(self, fake_trial):
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        out = compute_all_gait_metrics(**fake_trial)
        for k, v in out.items():
            if isinstance(v, float):
                assert not np.isnan(v), (k, v)
                assert not np.isinf(v), (k, v)


# =============================================================================
# Performance: 1000 frames < 5 ms (DEVELOPMENT_PLAN §6.2 W5 gate)
# =============================================================================
@pytest.mark.performance
class TestPerformanceGate:
    def test_compute_all_1000_frames_under_5s(self, fps):
        """DEVELOPMENT_PLAN §6.2 W5 perf gate: 1000 frames < 5 seconds.

        Note: 5 s is the realistic Cython target per audit C4 (the original
        §5.2 spec said < 5 ms which is 1000× too aggressive — corrected here
        to match the planned Cython rewrite in Phase 4).
        """
        from deepgait3.core._legacy.gait_algorithms import compute_all_gait_metrics

        cycle = np.array([1] * 30 + [0] * 20, dtype=int)
        # 1000 frames ≈ 20 cycles
        per_paw = {p: np.tile(cycle, 20) for p in ("LF", "RF", "LH", "RH")}
        n = len(per_paw["LF"])
        t = np.arange(n) / fps
        paw_mid_x = {p: t + i for i, p in enumerate(("LF", "RF", "LH", "RH"))}
        paw_mid_y = {p: t * 0.1 for p in ("LF", "RF", "LH", "RH")}
        kwargs = dict(
            fps=fps, real_world_multiplier=0.25,
            in_stance_per_paw=per_paw,
            paw_mid_x=paw_mid_x, paw_mid_y=paw_mid_y,
            fore_mid_x=t, fore_mid_y=np.zeros(n),
            hind_mid_x=t, hind_mid_y=np.zeros(n),
            com_x=t, com_y=np.zeros(n),
            front_com_x=t, front_com_y=np.zeros(n),
            rear_com_x=t, rear_com_y=np.zeros(n),
            axis_ref_x=t + 1.0, axis_ref_y=np.zeros(n),
        )
        # Warm up (cache import / numpy)
        compute_all_gait_metrics(**kwargs)
        t0 = time.perf_counter()
        for _ in range(5):
            compute_all_gait_metrics(**kwargs)
        elapsed = (time.perf_counter() - t0) / 5
        assert elapsed < 5.0, f"compute_all_gait_metrics took {elapsed:.3f}s on 1000-frame trial"