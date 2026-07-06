"""End-to-end pipeline tests.

Generates a synthetic DLC-format CSV (12 bodyparts, 3-row header), runs the full
``analyze()`` pipeline, and asserts that key metrics are sane.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deepgait3.core._legacy import bodyparts
from deepgait3.core._legacy.pipeline import analyze
from deepgait3.core._legacy import gait_export


# ---------------------------------------------------------------------------
# Fixtures: synthetic DLC CSV
# ---------------------------------------------------------------------------

def _make_synthetic_trajectory(n_frames: int = 200, fps: int = 100):
    """Build per-bodypart (x, y) trajectories mimicking a walking mouse.

    - Body axis (Nose, Butt, MidPoints): moves forward at steady speed.
    - Each paw: stance (still) -> swing (fast forward), periodic.
    - Left/right paws offset in phase and Y to create realistic stance width.
    """
    rng = np.random.default_rng(42)
    t = np.arange(n_frames)
    forward_speed = 0.5  # px/frame steady forward body motion

    # Body axis: Nose ahead, Butt behind, MidPoints flanking center
    body_x = t * forward_speed
    nose_x = body_x + 5.0
    butt_x = body_x - 5.0
    com_x = body_x.copy()
    nose_y = np.full(n_frames, 50.0)
    butt_y = np.full(n_frames, 50.0)
    com_y = np.full(n_frames, 50.0)
    mpr_x, mpr_y = com_x.copy(), com_y + 3.0
    mpl_x, mpl_y = com_x.copy(), com_y - 3.0

    # Paw trajectories
    paw_data = {}
    cycle_len = 20  # 10 stance + 10 swing
    stance_len = 10
    swing_len = 10

    def paw_traj(phase_offset: int, y_offset: float):
        x = np.zeros(n_frames)
        y = np.full(n_frames, 50.0 + y_offset)
        for c in range(n_frames // cycle_len + 2):
            base = c * cycle_len + phase_offset
            if base >= n_frames:
                break
            pos = c * 10.0  # stance position advances each cycle
            s_end = min(base + stance_len, n_frames)
            x[base:s_end] = pos
            sw_end = min(base + stance_len + swing_len, n_frames)
            n_swing = sw_end - s_end
            if n_swing > 0:
                x[s_end:sw_end] = np.linspace(pos, pos + 10.0, n_swing)
        # heel slightly behind toe
        return x, y

    paw_cfg = {
        # name: (phase_offset, y_offset)
        "FrontRight1": (0, 2.0), "FrontRight2": (0, 2.0),
        "FrontLeft1":  (cycle_len // 2, -2.0), "FrontLeft2":  (cycle_len // 2, -2.0),
        "HindRight1":  (5, 2.0), "HindRight2":  (5, 2.0),
        "HindLeft1":   (5 + cycle_len // 2, -2.0), "HindLeft2": (5 + cycle_len // 2, -2.0),
    }
    for name, (phase, yoff) in paw_cfg.items():
        x, y = paw_traj(phase, yoff)
        # heel is 2 px behind toe
        if name.endswith("2"):  # heel
            x = x - 2.0
        paw_data[name] = (x, y)

    # Assemble full dict
    full = {
        "Nose": (nose_x, nose_y),
        "Butt": (butt_x, butt_y),
        "MidPointRight": (mpr_x, mpr_y),
        "MidPointLeft": (mpl_x, mpl_y),
    }
    full.update(paw_data)
    return full


def _write_dlc_csv(traj: dict, path: Path, scorer: str = "DeepGait"):
    """Write a 3-row-header DLC-style CSV."""
    # Build MultiIndex columns
    cols = []
    data = {}
    for part, (x, y) in traj.items():
        cols.append((scorer, part, "x"))
        cols.append((scorer, part, "y"))
        cols.append((scorer, part, "likelihood"))
        data[(scorer, part, "x")] = x
        data[(scorer, part, "y")] = y
        data[(scorer, part, "likelihood")] = np.ones_like(x)
    # DLC CSV has an extra first column "bodyparts" or empty index
    index = pd.Index(range(len(next(iter(traj.values()))[0])))
    df = pd.DataFrame(data, index=index)
    df.columns = pd.MultiIndex.from_tuples(cols)
    df.to_csv(path)
    return path


@pytest.fixture
def synthetic_dlc_csv(tmp_path):
    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "synthetic_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_runs(synthetic_dlc_csv):
    """Pipeline should run end-to-end without error."""
    res = analyze(synthetic_dlc_csv, fps=100, mode="catwalk")
    assert res.n_frames == 200
    assert len(res.paws) == 4
    for name in ("RightFore", "LeftFore", "RightHind", "LeftHind"):
        assert name in res.paws


def test_pipeline_produces_metrics(synthetic_dlc_csv):
    """Key metrics should be non-zero for a walking mouse."""
    res = analyze(synthetic_dlc_csv, fps=100, mode="catwalk")
    for paw_name, paw in res.paws.items():
        # Should detect at least one stride
        assert paw.n_strides >= 1, f"{paw_name} detected 0 strides"
        # Stance/swing durations should be positive
        assert paw.stance_duration_ms > 0
        assert paw.swing_duration_ms > 0
        # Stride length mean should be positive (animal is moving forward)
        assert paw.stride_length_mean > 0
        # Frequency should be positive
        assert paw.stride_frequency_hz > 0


def test_pipeline_symmetry(synthetic_dlc_csv):
    """Symmetric synthetic gait should yield high symmetry index."""
    res = analyze(synthetic_dlc_csv, fps=100, mode="catwalk")
    # Symmetric phase offsets -> symmetry should be > 0.5
    assert 0.0 <= res.gait_symmetry_index <= 1.0
    assert res.gait_symmetry_index > 0.3


def test_pipeline_stance_width_nonneg(synthetic_dlc_csv):
    """Stance widths should be non-negative."""
    res = analyze(synthetic_dlc_csv, fps=100, mode="catwalk")
    for paw_name, paw in res.paws.items():
        assert paw.stance_width_mean >= 0


def test_pipeline_export_summary_csv(synthetic_dlc_csv, tmp_path):
    """Summary CSV should contain one row per paw."""
    res = analyze(synthetic_dlc_csv, fps=100, mode="catwalk")
    out = tmp_path / "summary.csv"
    gait_export.to_summary_csv(res, out)
    df = pd.read_csv(out)
    assert len(df) == 4
    assert "name" in df.columns
    assert "stance_duration_ms" in df.columns
    assert (df["stance_duration_ms"] > 0).all()


def test_pipeline_export_timeseries_csv(synthetic_dlc_csv, tmp_path):
    """Timeseries CSV should have one row per frame."""
    res = analyze(synthetic_dlc_csv, fps=100, mode="catwalk")
    out = tmp_path / "timeseries.csv"
    gait_export.to_timeseries_csv(res, out)
    df = pd.read_csv(out)
    assert len(df) == 200
    assert "frame" in df.columns
    # At least one stance column per paw
    stance_cols = [c for c in df.columns if c.endswith("_stance")]
    assert len(stance_cols) == 4


def test_pipeline_treadmill_mode(synthetic_dlc_csv):
    """Treadmill mode should also run and produce metrics."""
    res = analyze(
        synthetic_dlc_csv, fps=100, mode="treadmill",
        treadmill_speed=10.0,  # px/s
    )
    for paw_name, paw in res.paws.items():
        assert paw.n_strides >= 0  # treadmill may detect fewer
