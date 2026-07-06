"""Tests for DLC workflow helpers (config, path discovery, lazy import).

These run without DeepLabCut installed — we only test the pure-Python logic.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from deepgait3.core._legacy import bodyparts, dlc_config, dlc_results, dlc_workflow


# ---------------------------------------------------------------------------
# dlc_config tests
# ---------------------------------------------------------------------------

def test_default_bodyparts_is_vgl_12():
    parts = dlc_config.default_bodyparts()
    assert parts == bodyparts.BODYPARTS_12
    assert len(parts) == 12
    assert "Nose" in parts
    assert "HindLeft2" in parts


def test_build_config_dict_basic():
    spec = dlc_config.ProjectSpec(
        project="deepgait",
        experimenter="alice",
        videos=["/data/mouse1.mp4"],
        working_directory="/tmp/deepgait_proj",
    )
    cfg = dlc_config.build_config_dict(spec)
    assert cfg["Task"] == "deepgait"
    assert cfg["scorer"] == "alice"
    assert cfg["bodyparts"] == bodyparts.BODYPARTS_12
    assert cfg["engine"] == "pytorch"
    assert cfg["numframes2pick"] == 20
    assert cfg["TrainingFraction"] == [0.8]
    assert cfg["default_net_type"] == "resnet_50"
    assert cfg["batch_size"] == 8
    # video_sets should contain the resolved video path
    assert any("mouse1.mp4" in k for k in cfg["video_sets"])


def test_build_config_dict_custom_bodyparts():
    custom = ["nose", "tail"]
    spec = dlc_config.ProjectSpec(
        project="test", experimenter="x",
        videos=["/data/v.mp4"], working_directory="/tmp",
        bodyparts=custom,
    )
    cfg = dlc_config.build_config_dict(spec)
    assert cfg["bodyparts"] == custom


def test_build_config_dict_crop():
    spec = dlc_config.ProjectSpec(
        project="test", experimenter="x",
        videos=["/data/v.mp4"], working_directory="/tmp",
        crop=(100, 500, 50, 400),
    )
    cfg = dlc_config.build_config_dict(spec)
    assert cfg["cropping"] is True
    assert cfg["x1"] == 100
    assert cfg["x2"] == 500
    assert cfg["y1"] == 50
    assert cfg["y2"] == 400
    crop_str = list(cfg["video_sets"].values())[0]["crop"]
    assert crop_str == "100,500,50,400"


def test_build_config_dict_no_crop_uses_video_dims():
    spec = dlc_config.ProjectSpec(
        project="test", experimenter="x",
        videos=["/data/v.mp4"], working_directory="/tmp",
        video_width=1920, video_height=1080,
    )
    cfg = dlc_config.build_config_dict(spec)
    assert cfg["cropping"] is False
    assert cfg["x2"] == 1920
    assert cfg["y2"] == 1080


def test_write_config_creates_file(tmp_path):
    spec = dlc_config.ProjectSpec(
        project="deepgait", experimenter="alice",
        videos=["/data/v.mp4"], working_directory=str(tmp_path),
    )
    cfg_path = dlc_config.write_config(spec)
    assert cfg_path.is_file()
    assert cfg_path.name == "config.yaml"
    # Verify it's valid YAML with expected fields
    with cfg_path.open() as f:
        loaded = yaml.safe_load(f)
    assert loaded["Task"] == "deepgait"
    assert loaded["scorer"] == "alice"
    assert loaded["bodyparts"] == bodyparts.BODYPARTS_12
    assert loaded["engine"] == "pytorch"


def test_project_dir_name_format():
    spec = dlc_config.ProjectSpec(
        project="deepgait", experimenter="alice",
        videos=[], working_directory="/tmp",
    )
    name = dlc_config.project_dir_name(spec)
    assert name.startswith("deepgait-alice-")


# ---------------------------------------------------------------------------
# dlc_results tests
# ---------------------------------------------------------------------------

def test_find_dlc_outputs_finds_all(tmp_path):
    """DLC-style output files next to a video should all be discovered."""
    video = tmp_path / "mouse.mp4"
    video.touch()
    # Create DLC-style outputs
    (tmp_path / "mouseDLC_resnet50_2026-06-16.h5").touch()
    (tmp_path / "mouseDLC_resnet50_2026-06-16.csv").touch()
    (tmp_path / "mouseDLC_resnet50_2026-06-16_meta.json").touch()
    (tmp_path / "mouseDLC_resnet50_2026-06-16_filtered.h5").touch()
    (tmp_path / "mouseDLC_resnet50_2026-06-16_filtered.csv").touch()

    out = dlc_results.find_dlc_outputs(video)
    assert out.video == video
    assert out.h5 is not None
    assert out.csv is not None
    assert out.meta is not None
    assert out.filtered_h5 is not None
    assert out.filtered_csv is not None
    assert out.has_results


def test_find_dlc_outputs_best_csv_prefers_filtered(tmp_path):
    video = tmp_path / "mouse.mp4"
    video.touch()
    (tmp_path / "mouse_raw.csv").touch()
    (tmp_path / "mouse_filtered.csv").touch()
    out = dlc_results.find_dlc_outputs(video)
    assert out.best_csv is not None
    assert out.best_csv.name == "mouse_filtered.csv"


def test_find_dlc_outputs_falls_back_to_raw(tmp_path):
    video = tmp_path / "mouse.mp4"
    video.touch()
    (tmp_path / "mouse_data.csv").touch()
    out = dlc_results.find_dlc_outputs(video)
    assert out.best_csv is not None
    assert out.best_csv.name == "mouse_data.csv"
    assert out.filtered_csv is None


def test_find_dlc_outputs_empty(tmp_path):
    video = tmp_path / "mouse.mp4"
    video.touch()
    out = dlc_results.find_dlc_outputs(video)
    assert not out.has_results
    assert out.best_csv is None
    assert out.h5 is None
    assert out.csv is None


def test_find_all_dlc_outputs_multiple_videos(tmp_path):
    (tmp_path / "mouse1.mp4").touch()
    (tmp_path / "mouse2.avi").touch()
    (tmp_path / "not_a_video.txt").touch()
    outputs = dlc_results.find_all_dlc_outputs(tmp_path)
    assert len(outputs) == 2
    video_names = {o.video.name for o in outputs}
    assert video_names == {"mouse1.mp4", "mouse2.avi"}


# ---------------------------------------------------------------------------
# dlc_workflow lazy-import tests
# ---------------------------------------------------------------------------

def test_dlc_workflow_imports_without_dlc():
    """Module should import even when DLC is not installed."""
    # This passes as long as the import at top of file succeeded.
    assert hasattr(dlc_workflow, "create_project")
    assert hasattr(dlc_workflow, "analyze_video_gait")


def test_require_dlc_raises_when_missing():
    """_require_dlc should raise DLCNotInstalledError when DLC absent."""
    with pytest.raises(dlc_workflow.DLCNotInstalledError) as exc_info:
        dlc_workflow._require_dlc()
    assert "not installed" in str(exc_info.value).lower()


def test_create_project_falls_back_without_dlc(tmp_path):
    """create_project should fall back to config-only when DLC is missing."""
    spec = dlc_config.ProjectSpec(
        project="test", experimenter="x",
        videos=["/data/v.mp4"], working_directory=str(tmp_path),
    )
    # Should NOT raise — graceful fallback writes config
    cfg_path = dlc_workflow.create_project(spec)
    assert cfg_path.is_file()
    assert cfg_path.name == "config.yaml"


def test_analyze_videos_raises_without_dlc():
    """analyze_videos should raise clearly when DLC is missing."""
    with pytest.raises(dlc_workflow.DLCNotInstalledError):
        dlc_workflow.analyze_videos("config.yaml", ["v.mp4"])


def test_analyze_video_gait_raises_without_dlc():
    """analyze_video_gait should raise clearly when DLC is missing."""
    with pytest.raises(dlc_workflow.DLCNotInstalledError):
        dlc_workflow.analyze_video_gait("config.yaml", ["v.mp4"])
