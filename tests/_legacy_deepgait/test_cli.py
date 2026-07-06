"""End-to-end CLI tests via subprocess.

Invokes ``python -m deepgait`` as a real process to verify argument parsing,
exit codes, stdout formatting, and file exports.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv

# Project root (parent of the deepgait/ package) — ensures subprocess can import deepgait3.deepgait
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


@pytest.fixture
def dlc_csv(tmp_path):
    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    return csv_path


def _run(*args: str, cwd: str | Path | None = None) -> subprocess.CompletedProcess:
    """Run ``python -m deepgait <args>`` and capture output.

    Sets PYTHONPATH to the project root so the deepgait package is importable
    regardless of the subprocess working directory.
    """
    cmd = [sys.executable, "-m", "deepgait", *args]
    env = {**os.environ, "PYTHONPATH": _PROJECT_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)


# ---------------------------------------------------------------------------
# info subcommand
# ---------------------------------------------------------------------------

def test_cli_info(dlc_csv):
    proc = _run("info", str(dlc_csv))
    assert proc.returncode == 0, proc.stderr
    assert "Frames: 200" in proc.stdout
    assert "Bodyparts (12)" in proc.stdout
    # All bodypart names should appear
    for name in ["Nose", "Butt", "FrontRight1", "HindLeft2"]:
        assert name in proc.stdout


def test_cli_info_missing_file():
    proc = _run("info", "/tmp/__nonexistent_deepgait__.csv")
    assert proc.returncode == 2
    assert "not found" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# analyze subcommand
# ---------------------------------------------------------------------------

def test_cli_analyze_default(dlc_csv, tmp_path):
    """Default analyze: catwalk mode, auto output paths."""
    proc = _run("analyze", str(dlc_csv), cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    # Summary table in stdout
    assert "RightFore" in proc.stdout
    assert "Symmetry index" in proc.stdout
    assert "200 frames analyzed" in proc.stdout
    # Default output files created alongside input
    assert (dlc_csv.with_suffix(".summary.csv")).is_file()
    assert (dlc_csv.with_suffix(".timeseries.csv")).is_file()


def test_cli_analyze_excel_export(dlc_csv, tmp_path):
    """-o report.xlsx should create a two-sheet workbook."""
    out = tmp_path / "report.xlsx"
    proc = _run("analyze", str(dlc_csv), "-o", str(out))
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    # Verify Excel content
    xl = pd.ExcelFile(out)
    assert "summary" in xl.sheet_names
    assert "timeseries" in xl.sheet_names
    summary_df = xl.parse("summary")
    assert len(summary_df) == 4  # 4 paws


def test_cli_analyze_treadmill_requires_speed(dlc_csv):
    """Treadmill mode without --treadmill-speed should fail with exit 2."""
    proc = _run("analyze", str(dlc_csv), "--mode", "treadmill")
    assert proc.returncode == 2
    assert "treadmill-speed" in proc.stderr.lower()


def test_cli_analyze_treadmill_with_speed(dlc_csv, tmp_path):
    """Treadmill mode with speed should succeed."""
    proc = _run(
        "analyze", str(dlc_csv),
        "--mode", "treadmill", "--treadmill-speed", "20",
        "--multiplier", "0.05",
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr


def test_cli_analyze_missing_file():
    proc = _run("analyze", "/tmp/__nonexistent_deepgait__.csv")
    assert proc.returncode == 2
    assert "not found" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# DLC subcommand tests
# ---------------------------------------------------------------------------

def test_cli_dlc_help():
    """dlc --help should list all subcommands."""
    proc = _run("dlc", "--help")
    assert proc.returncode == 0, proc.stderr
    for cmd in ["init-project", "extract", "label", "train", "run"]:
        assert cmd in proc.stdout


def test_cli_dlc_init_project_graceful_fallback(tmp_path):
    """init-project without DLC should still create a config.yaml."""
    proc = _run(
        "dlc", "init-project", "deepgait", "alice",
        str(tmp_path / "fake.mp4"),
        "--work-dir", str(tmp_path),
        "--fps", "100",
        "--width", "640",
        "--height", "480",
    )
    assert proc.returncode == 0, proc.stderr
    # DLC not installed warning should be in stderr
    assert "not installed" in proc.stderr.lower() or "wrote config" in proc.stderr.lower()
    # Config file should exist
    proj_dir = list(tmp_path.glob("deepgait-alice-*"))
    assert len(proj_dir) == 1
    cfg = proj_dir[0] / "config.yaml"
    assert cfg.is_file()


def test_cli_dlc_run_requires_videos():
    """dlc run without --videos should fail."""
    proc = _run("dlc", "run", "/tmp/fake_config.yaml")
    assert proc.returncode == 2
    assert "required" in proc.stderr.lower() or "videos" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# FTIR subcommand tests
# ---------------------------------------------------------------------------

def test_cli_ftir_help():
    """ftir --help should list options."""
    proc = _run("ftir", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "image" in proc.stdout.lower()
    assert "--px-per-mm" in proc.stdout


def test_cli_ftir_analyze_image(tmp_path):
    """ftir should analyze a synthetic FTIR image and produce CSV."""
    import cv2
    import numpy as np
    from tests._legacy_deepgait.test_ftir import make_synthetic_ftir_frame
    img = tmp_path / "ftir.png"
    frame = make_synthetic_ftir_frame(noise=False)
    cv2.imwrite(str(img), frame)

    proc = _run("ftir", str(img), "--px-per-mm", "10", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "Footprints detected: 4" in proc.stdout
    # CSV should be created
    csv = img.with_suffix(".ftir.csv")
    assert csv.is_file()
    content = csv.read_text()
    assert "paw,area_px" in content
    assert "blob_" in content


def test_cli_ftir_missing_image():
    proc = _run("ftir", "/tmp/__nonexistent_ftir__.png")
    assert proc.returncode == 2
    assert "not found" in proc.stderr.lower()


# ---------------------------------------------------------------------------
# Argument parser behavior
# ---------------------------------------------------------------------------

def test_cli_no_subcommand():
    """No subcommand should fail with usage error (exit 2)."""
    proc = _run()
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower() or "required" in proc.stderr.lower()


def test_cli_help():
    """--help should exit 0 and mention subcommands."""
    proc = _run("--help")
    assert proc.returncode == 0
    assert "analyze" in proc.stdout
    assert "info" in proc.stdout


def test_cli_analyze_help():
    """analyze --help should list key options."""
    proc = _run("analyze", "--help")
    assert proc.returncode == 0
    for opt in ["--fps", "--mode", "--multiplier", "--output"]:
        assert opt in proc.stdout
