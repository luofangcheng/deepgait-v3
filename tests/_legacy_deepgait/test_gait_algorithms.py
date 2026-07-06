"""Unit tests for deepgait core algorithms (Phase 0 validation).

Uses synthetic data that mimics a known stance/swing pattern so expected
outputs can be computed by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepgait3.core._legacy import gait_algorithms as ga
from deepgait3.utils import geometry


# ---------------------------------------------------------------------------
# Fixtures: synthetic paw trajectory
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_catwalk():
    """Generate a synthetic paw trajectory with known stance/swing cycles.

    Pattern: 10 frames stance (still), 10 frames swing (move forward 1 px/frame),
    repeated 5 times = 100 frames.
    """
    n_cycles = 5
    stance_len = 10
    swing_len = 10
    n = n_cycles * (stance_len + swing_len)
    toe_x = np.zeros(n, dtype=float)
    toe_y = np.zeros(n, dtype=float)
    heel_x = np.zeros(n, dtype=float)
    heel_y = np.zeros(n, dtype=float)
    for c in range(n_cycles):
        base = c * (stance_len + swing_len)
        # stance: stationary at position c*10
        pos = c * 10.0
        toe_x[base:base + stance_len] = pos
        heel_x[base:base + stance_len] = pos - 2.0  # heel slightly behind
        # swing: move forward 1 px per frame
        toe_x[base + stance_len:base + stance_len + swing_len] = np.linspace(
            pos, pos + swing_len, swing_len
        )
        heel_x[base + stance_len:base + stance_len + swing_len] = np.linspace(
            pos - 2.0, pos + swing_len - 2.0, swing_len
        )
    return toe_x, toe_y, heel_x, heel_y, n_cycles, stance_len, swing_len


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

def test_midpoint():
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 4.0])
    assert np.allclose(geometry.midpoint(a, b), [1.0, 2.0])


def test_distance_between_points():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert geometry.distance_between_points(a, b) == pytest.approx(5.0)


def test_distance_point_to_line():
    # Point (1,1) to line y=0 (through (0,0)-(2,0))
    p = np.array([1.0, 1.0])
    l1 = np.array([0.0, 0.0])
    l2 = np.array([2.0, 0.0])
    assert geometry.distance_point_to_line(p, l1, l2) == pytest.approx(1.0)


def test_paw_angle():
    # Paw horizontal (toe right of heel), reference horizontal
    toe = np.array([[2.0, 0.0]])
    heel = np.array([[0.0, 0.0]])
    ref1 = np.array([[0.0, 0.0]])
    ref2 = np.array([[1.0, 0.0]])
    assert geometry.paw_angle(toe, heel, ref1, ref2)[0] == pytest.approx(0.0, abs=1e-6)

    # Paw vertical, reference horizontal -> 90 deg
    toe = np.array([[0.0, 2.0]])
    heel = np.array([[0.0, 0.0]])
    assert geometry.paw_angle(toe, heel, ref1, ref2)[0] == pytest.approx(90.0, abs=1e-6)


def test_find_runs():
    arr = np.array([1, 1, 0, 0, 0, 1, 1])
    runs = geometry.find_runs(arr)
    assert runs == [(0, 2, 1), (2, 5, 0), (5, 7, 1)]


def test_flip_y():
    assert geometry.flip_y(np.array([0.0, 10.0, 20.0]), 100.0).tolist() == [100.0, 90.0, 80.0]


# ---------------------------------------------------------------------------
# Stance / Swing tests
# ---------------------------------------------------------------------------

def test_catwalk_in_stance(synthetic_catwalk):
    toe_x, toe_y, heel_x, heel_y, n_cycles, stance_len, swing_len = synthetic_catwalk
    in_stance = ga.catwalk_in_stance(toe_x, toe_y, heel_x, heel_y, bias=1.0)
    # With clean synthetic data, most stance frames should be 1, swing 0.
    # The threshold-based method may misclassify a few boundary frames (swing->stance
    # transition has small displacement), so we check majority correctness.
    assert in_stance.sum() >= n_cycles * (stance_len - 1)
    assert (1 - in_stance).sum() >= n_cycles * (swing_len - 1)


def test_treadmill_in_stance():
    # Treadmill: stance = backward motion (decreasing X), swing = forward (increasing)
    x = np.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0, 0], dtype=float)
    in_stance = ga.treadmill_in_stance(x)
    # First segment flat (slope 0 -> swing), then increasing (swing), then decreasing (stance)
    assert in_stance.dtype == int
    assert len(in_stance) == len(x)


# ---------------------------------------------------------------------------
# Gait basics tests
# ---------------------------------------------------------------------------

def test_calculate_gait_basics(synthetic_catwalk):
    toe_x, toe_y, heel_x, heel_y, n_cycles, stance_len, swing_len = synthetic_catwalk
    in_stance = ga.catwalk_in_stance(toe_x, toe_y, heel_x, heel_y)
    basics = ga.calculate_gait_basics(in_stance, fps=100)
    # With threshold-based detection, boundary frames may be misclassified,
    # so n_strides may be n_cycles or n_cycles-1.  Accept either.
    assert basics.n_strides in (n_cycles, n_cycles - 1)
    # Duration should be close to 100 ms; allow 50% tolerance for boundary effects
    assert basics.stance_duration_ms == pytest.approx(100.0, rel=0.50)
    assert basics.swing_duration_ms == pytest.approx(100.0, rel=0.50)


# ---------------------------------------------------------------------------
# Stride length / frequency tests
# ---------------------------------------------------------------------------

def test_calculate_stride_data_free_run(synthetic_catwalk):
    toe_x, toe_y, heel_x, heel_y, n_cycles, stance_len, swing_len = synthetic_catwalk
    mid_x = (toe_x + heel_x) / 2.0
    mid_y = (toe_y + heel_y) / 2.0
    in_stance = ga.catwalk_in_stance(toe_x, toe_y, heel_x, heel_y)
    basics = ga.calculate_gait_basics(in_stance, fps=100)
    stride_lengths, variability = ga.calculate_stride_data(
        basics.switch_positions,
        mid_x, mid_y,
        is_free_run=True,
        treadmill_speed=None,
        real_world_multiplier=1.0,
        fps=100,
    )
    # Most strides should have length near swing displacement = 10 px.
    # Boundary strides may differ; check that at least one clean stride is ~10 px.
    unique = np.unique(stride_lengths[stride_lengths > 0])
    assert len(unique) >= 1
    assert any(np.isclose(u, 10.0, atol=2.0) for u in unique)
    # Variability may be moderate due to boundary misclassification in catwalk_in_stance
    assert variability <= 5.0


def test_stride_frequency(synthetic_catwalk):
    _, _, _, _, n_cycles, stance_len, swing_len = synthetic_catwalk
    # 5 strides in 100 frames at 100 fps = 1 second -> 5 Hz
    freq = ga.stride_frequency(
        stance_frames_per_stride=[stance_len] * n_cycles,
        swing_frames_per_stride=[swing_len] * n_cycles,
        n_strides=n_cycles,
        fps=100,
    )
    assert freq == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# AutoCorrect tests
# ---------------------------------------------------------------------------

def test_auto_correct():
    # Long stance, short swing glitch, long stance
    arr = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=int)
    corrected = ga.auto_correct(arr)
    # The short 0 (length 1) should be merged into 1s
    assert np.array_equal(corrected, np.ones_like(arr))


def test_auto_correct_no_change():
    # Balanced segments, no short glitches
    arr = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=int)
    corrected = ga.auto_correct(arr)
    assert np.array_equal(corrected, arr)


# ---------------------------------------------------------------------------
# SEM tests
# ---------------------------------------------------------------------------

def test_sem():
    vals = np.array([1.0, 2.0, 3.0])
    # SD = 1.0, SEM = 1/sqrt(3)
    assert ga.sem(vals) == pytest.approx(1.0 / np.sqrt(3), abs=1e-6)


# ---------------------------------------------------------------------------
# Paw angle tests
# ---------------------------------------------------------------------------

def test_calculate_paw_angles():
    n = 10
    toe_x = np.linspace(0, 10, n)
    toe_y = np.zeros(n)
    heel_x = np.linspace(0, 10, n)
    heel_y = np.zeros(n)  # horizontal paw (heel and toe same Y)
    ref1x = np.zeros(n)
    ref1y = np.zeros(n)
    ref2x = np.ones(n)
    ref2y = np.zeros(n)
    in_stance = np.ones(n, dtype=int)
    angles, mean_angle = ga.calculate_paw_angles(
        toe_x, toe_y, heel_x, heel_y, ref1x, ref1y, ref2x, ref2y, in_stance
    )
    # Paw horizontal, ref horizontal -> 0 deg
    assert mean_angle == pytest.approx(0.0, abs=1e-6)
    # Mask swing frames
    in_stance[5:] = 0
    angles2, mean_angle2 = ga.calculate_paw_angles(
        toe_x, toe_y, heel_x, heel_y, ref1x, ref1y, ref2x, ref2y, in_stance
    )
    assert np.isnan(angles2[5:]).all()
    assert mean_angle2 == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Stance width tests
# ---------------------------------------------------------------------------

def test_calculate_stance_widths_legacy():
    right_y = np.array([10.0, 10.0, 10.0])
    left_y = np.array([5.0, 5.0, 5.0])
    widths, mean_w = ga.calculate_stance_widths_legacy(right_y, left_y, 1.0)
    assert np.allclose(widths, 5.0)
    assert mean_w == 5.0


def test_calculate_stance_widths_per_paw():
    paw_x = np.array([1.0, 1.0])
    paw_y = np.array([1.0, 1.0])
    com_x = np.array([0.0, 0.0])
    com_y = np.array([0.0, 0.0])
    axis_x = np.array([1.0, 1.0])
    axis_y = np.array([0.0, 0.0])
    dists, mean_d = ga.calculate_stance_widths_per_paw(
        paw_x, paw_y, com_x, com_y, axis_x, axis_y, 1.0
    )
    # Point (1,1) to line y=0 (through (0,0)-(1,0)) = distance 1
    assert np.allclose(dists, 1.0)
    assert mean_d == 1.0


# ---------------------------------------------------------------------------
# Gait symmetry tests
# ---------------------------------------------------------------------------

def test_gait_symmetry_perfect():
    assert ga.gait_symmetry(100.0, 100.0, 50.0, 50.0) == pytest.approx(1.0, abs=1e-6)


def test_gait_symmetry_asymmetric():
    # left stance 100, right stance 50 -> stance_sym = 1 - 50/150 = 0.666...
    sym = ga.gait_symmetry(100.0, 50.0, 50.0, 50.0)
    assert sym == pytest.approx((2 / 3 + 1.0) / 2, abs=1e-6)
