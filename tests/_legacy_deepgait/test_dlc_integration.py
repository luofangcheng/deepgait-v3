"""DeepLabCut integration tests (require conda dlc env with DLC installed).

These tests are skipped if DLC is not importable.  To run them, use the
conda dlc environment:

    /home/luofangcheng/miniforge3/envs/dlc/bin/python -m pytest \\
        deepgait/tests/test_dlc_integration.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest


# Try to import DLC at module load; skip everything if not available.
try:
    import deeplabcut  # noqa: F401
    DLC_AVAILABLE = True
except ImportError:
    DLC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DLC_AVAILABLE,
    reason="DLC not importable (run with conda dlc env to enable)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_video(path: Path, n_frames: int = 30, fps: int = 30) -> None:
    """Write a small synthetic video (BGR frames) for DLC tests."""
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    for i in range(n_frames):
        # Simple moving gradient so DLC has something to track
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        x = int((i / n_frames) * 280) + 20
        cv2.circle(frame, (x, 120), 20, (0, 255, 0), -1)  # green dot = "paw"
        writer.write(frame)
    writer.release()


def _dlc_create_project(tmp_path: Path) -> tuple[str, list[str]]:
    """Create a DLC project + synthetic video. Returns (config_path, video_paths)."""
    import deeplabcut

    video = tmp_path / "mouse.mp4"
    _make_synthetic_video(video)

    cfg = deeplabcut.create_new_project(
        project="dlc_test",
        experimenter="tester",
        videos=[str(video)],
        working_directory=str(tmp_path),
        copy_videos=False,
    )
    return cfg, [str(video)]


# ---------------------------------------------------------------------------
# Test 1: create_new_project actually works
# ---------------------------------------------------------------------------

def test_dlc_create_new_project(tmp_path):
    """DLC's create_new_project should succeed and return a config path."""
    cfg, videos = _dlc_create_project(tmp_path)
    assert Path(cfg).is_file()
    assert Path(cfg).name == "config.yaml"
    # Project directory should exist with standard DLC structure
    proj_dir = Path(cfg).parent
    assert (proj_dir / "videos").is_dir()
    assert any(p.name == "mouse.mp4" for p in (proj_dir / "videos").iterdir())


# ---------------------------------------------------------------------------
# Test 2: deepgait dlc_workflow.create_project wraps DLC
# ---------------------------------------------------------------------------

def test_dlc_workflow_create_project_real_dlc(tmp_path):
    """dlc_workflow.create_project should use DLC and overwrite with deepgait config."""
    from deepgait3.core._legacy import dlc_config, dlc_workflow

    video = tmp_path / "mouse.mp4"
    _make_synthetic_video(video)
    spec = dlc_config.ProjectSpec(
        project="deepgait_dlc",
        experimenter="tester",
        videos=[str(video)],
        working_directory=str(tmp_path),
    )
    cfg_path = dlc_workflow.create_project(spec)
    assert cfg_path.is_file()

    # The config should be deepgait's (with 12 bodyparts), not DLC's default
    import yaml
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["bodyparts"] == dlc_config.default_bodyparts()
    assert "Nose" in cfg["bodyparts"]
    assert len(cfg["bodyparts"]) == 12


# ---------------------------------------------------------------------------
# Test 3: extract_frames with real DLC
# ---------------------------------------------------------------------------

def test_dlc_extract_frames(tmp_path):
    """DLC extract_frames should create a labeled-data directory with images."""
    import deeplabcut
    cfg, _ = _dlc_create_project(tmp_path)
    # DLC 3.0 supports: 'automatic', 'manual', 'match'
    deeplabcut.extract_frames(cfg, mode="automatic", algo="kmeans", userfeedback=False)
    # Should have created labeled-data/<videoname>/ with some images
    proj = Path(cfg).parent
    labeled = proj / "labeled-data"
    assert labeled.is_dir()
    subdirs = [sd for sd in labeled.iterdir() if sd.is_dir()]
    assert len(subdirs) >= 1
    # At least one subdir should contain images
    has_images = any(
        any(p.suffix.lower() in (".png", ".jpg") for p in sd.iterdir())
        for sd in subdirs
    )
    assert has_images
