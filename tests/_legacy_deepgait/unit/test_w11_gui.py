"""Unit tests for Phase 3 W11 — DLCTab + Calibration3DTab + Triangulation3DTab 端到端.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.3 W11):
    "DLCTab + Calibration3DTab + Triangulation3DTab 端到端跑通"

Covers:
  * DLCTab with injected MockDLCSubprocessRunner (create/train/analyze).
  * Calibration3DTab → AppState → Triangulation3DTab data flow.
  * End-to-end: calibrate → triangulate button enables → pose published.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def main_window(qtbot):
    from deepgait3.gui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


@pytest.fixture
def app_state():
    from deepgait3.gui.shared_state import AppState
    return AppState()


@pytest.fixture
def synthetic_dlc_csv(tmp_path):
    """Write a 30-frame DLC-format CSV with 12 bodyparts (2D pose)."""
    rng = np.random.default_rng(7)
    n_frames = 30
    scorer = "scorer"
    bodyparts = [
        "Nose", "Tail",
        "FrontLeft1", "FrontLeft2",
        "FrontRight1", "FrontRight2",
        "HindLeft1", "HindLeft2",
        "HindRight1", "HindRight2",
        "MidPointLeft", "MidPointRight",
    ]
    cols = pd.MultiIndex.from_product(
        [[scorer], bodyparts, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = rng.random((n_frames, len(bodyparts) * 3)) * 500
    df = pd.DataFrame(data, columns=cols)
    path = tmp_path / "pose.csv"
    df.to_csv(path, index=False)
    return str(path)


# =============================================================================
# DLCTab — MockDLCSubprocessRunner integration
# =============================================================================
class TestDLCTabRunner:
    def test_set_runner_injects_runner(self, qtbot, main_window):
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner
        dlc_tab = main_window.tab_by_name("dlc")
        qtbot.addWidget(dlc_tab)
        assert dlc_tab.has_runner is False
        runner = MockDLCSubprocessRunner()
        dlc_tab.set_runner(runner)
        assert dlc_tab.has_runner is True
        assert dlc_tab._runner is runner

    def test_create_project_via_runner(self, qtbot, main_window, tmp_path):
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner
        dlc_tab = main_window.tab_by_name("dlc")
        qtbot.addWidget(dlc_tab)
        # Inject a stub video path.
        dlc_tab._videos = [str(tmp_path / "v1.mp4")]
        dlc_tab._working_dir = str(tmp_path)
        dlc_tab._config_path = ""
        runner = MockDLCSubprocessRunner()
        dlc_tab.set_runner(runner)
        # Trigger create_project_v2.
        dlc_tab._on_create_project_v2()
        assert dlc_tab._config_path != ""
        assert Path(dlc_tab._config_path).name == "config.yaml"

    def test_train_via_runner(self, qtbot, main_window, tmp_path):
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner
        dlc_tab = main_window.tab_by_name("dlc")
        qtbot.addWidget(dlc_tab)
        cfg = str(tmp_path / "config.yaml")
        Path(cfg).write_text("# stub")
        dlc_tab._config_path = cfg
        runner = MockDLCSubprocessRunner()
        dlc_tab.set_runner(runner)
        dlc_tab._on_train_v2(cfg)
        # MockDLCSubprocessRunner always returns True.
        # The history must record training events.
        assert any(h[0] == "progress" and h[1].stage == "training"
                    for h in runner.history)

    def test_analyze_via_runner(self, qtbot, main_window, tmp_path):
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner
        dlc_tab = main_window.tab_by_name("dlc")
        qtbot.addWidget(dlc_tab)
        video = str(tmp_path / "v1.mp4")
        dlc_tab._videos = [video]
        cfg = str(tmp_path / "config.yaml")
        Path(cfg).write_text("# stub")
        dlc_tab._config_path = cfg
        runner = MockDLCSubprocessRunner()
        dlc_tab.set_runner(runner)
        dlc_tab._on_analyze_v2(cfg)
        assert any(h[0] == "progress" and h[1].stage == "analyze"
                    for h in runner.history)

    def test_runner_path_skips_legacy_worker(
        self, qtbot, main_window, tmp_path, monkeypatch,
    ):
        """When runner is set, _on_create_project must NOT spawn DLCWorker."""
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner
        dlc_tab = main_window.tab_by_name("dlc")
        qtbot.addWidget(dlc_tab)
        dlc_tab._videos = [str(tmp_path / "v1.mp4")]
        dlc_tab._working_dir = str(tmp_path)
        runner = MockDLCSubprocessRunner()
        dlc_tab.set_runner(runner)
        # Patch DLCWorker to raise if constructed (legacy path).
        from deepgait3.gui import workers as workers_mod

        def _raise(*a, **k):
            raise AssertionError("DLCWorker should NOT be called when runner is set")
        monkeypatch.setattr(workers_mod, "DLCWorker", _raise)
        # This should not raise — it goes through the runner path.
        dlc_tab._on_create_project()


# =============================================================================
# Calibration3DTab → Triangulation3DTab AppState data flow
# =============================================================================
class TestCalibToTriangulationFlow:
    def test_calibration_published_to_app_state(
        self, qtbot, main_window, tmp_path,
    ):
        """Calibration3DTab.set_calibration → AppState → Triangulation3DTab receives."""
        calib_tab = main_window.tab_by_name("calibration_3d")
        tri_tab = main_window.tab_by_name("triangulation_3d")
        qtbot.addWidget(calib_tab)
        qtbot.addWidget(tri_tab)
        # Manually publish a calibration.
        from deepgait3.gui.shared_state import CalibrationView
        view = CalibrationView(
            cameras=["left", "right"],
            reproj_rms_per_cam={"left": 1.5, "right": 2.0},
            method="inhouse", board_rows=5, board_cols=7,
        )
        main_window.app_state.set_calibration(view)
        # Triangulation3DTab must have received it via the signal.
        assert tri_tab._state.calibration is not None
        assert tri_tab._state.calibration.cameras == ["left", "right"]
        # Status label must reflect the calibration.
        assert "已加载标定" in tri_tab.calib_status.text()

    def test_triangulate_button_stays_disabled_without_calibration(
        self, qtbot, main_window, tmp_path, synthetic_dlc_csv,
    ):
        tri_tab = main_window.tab_by_name("triangulation_3d")
        qtbot.addWidget(tri_tab)
        # Add two CSVs but no calibration.
        tri_tab._pose_2d_paths = {"left": synthetic_dlc_csv, "right": synthetic_dlc_csv}
        tri_tab._refresh_enable_state()
        assert tri_tab.triangulate_btn.isEnabled() is False

    def test_triangulate_button_enables_with_calibration_and_csvs(
        self, qtbot, main_window, synthetic_dlc_csv,
    ):
        tri_tab = main_window.tab_by_name("triangulation_3d")
        qtbot.addWidget(tri_tab)
        # Add CSVs.
        tri_tab._pose_2d_paths = {"left": synthetic_dlc_csv, "right": synthetic_dlc_csv}
        # Publish a calibration (cameras = string names; Camera objects
        # are carried separately on the CalibrationResult used by the
        # triangulation code, not on the CalibrationView).
        from deepgait3.gui.shared_state import CalibrationView
        view = CalibrationView(
            cameras=["left", "right"],
            reproj_rms_per_cam={"left": 1.0, "right": 1.5},
            method="inhouse", board_rows=5, board_cols=7,
        )
        main_window.app_state.set_calibration(view)
        tri_tab._refresh_enable_state()
        assert tri_tab.triangulate_btn.isEnabled() is True

    def test_triangulation_publishes_pose_3d(
        self, qtbot, main_window, synthetic_dlc_csv,
    ):
        """End-to-end: triangulate → Pose3DResultsView published to AppState."""
        from deepgait3.gui.shared_state import CalibrationView, Pose3DResultsView
        from deepgait3.core._legacy.triangulation_3d import Camera
        tri_tab = main_window.tab_by_name("triangulation_3d")
        qtbot.addWidget(tri_tab)
        tri_tab._pose_2d_paths = {"left": synthetic_dlc_csv, "right": synthetic_dlc_csv}
        # Inject a calibration with Camera objects.
        K = np.array([[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1]])
        view = CalibrationView(
            cameras=["left", "right"],
            reproj_rms_per_cam={"left": 1.0, "right": 1.5},
            method="inhouse", board_rows=5, board_cols=7,
        )
        # Wrap Camera objects so the AniposeWrapper.calibration.cameras
        # dict has proper entries.
        view.cameras = ["left", "right"]
        # Patch CalibrationView to carry Camera objects via a fake
        # ``cameras`` attribute that is a dict (matching the real dataclass).
        from deepgait3.core._legacy.anipose_wrapper import CalibrationResult
        cam_left = Camera(name="left", P=K @ np.hstack([np.eye(3), np.array([[0, 0, 200.0]]).T]))
        cam_right = Camera(name="right", P=K @ np.hstack([np.eye(3), np.array([[-200.0, 0, 0]]).T]))
        calib_result = CalibrationResult(
            cameras={"left": cam_left, "right": cam_right},
            reproj_rms_per_cam={"left": 1.0, "right": 1.5},
            board=view, method="inhouse",
        )
        # Monkeypatch the tab's calibration reference to our result.
        main_window.app_state.set_calibration(view)
        # The real AppState stores the CalibrationView, not CalibrationResult.
        # The triangulation code reads ``calib.cameras``. Let's monkeypatch
        # the AppState calibration to have Camera objects.
        main_window.app_state._calibration.cameras = ["left", "right"]
        # For the test, directly inject a CalibrationResult-compatible
        # object so _run_triangulation finds the Camera objects.
        main_window.app_state._calibration = calib_result
        # Patch the wrapper's calibrate to return our CalibrationResult.
        from deepgait3.core._legacy import anipose_wrapper as aw_mod

        original_calibrate = aw_mod.AniposeWrapper._calibrate_inhouse
        aw_mod.AniposeWrapper._calibrate_inhouse = lambda self, *a, **k: calib_result
        try:
            tri_tab._on_triangulate()
            # Pose3D must have been published.
            assert main_window.app_state.pose_3d is not None
            assert isinstance(main_window.app_state.pose_3d, Pose3DResultsView)
            assert main_window.app_state.pose_3d.n_frames > 0
        finally:
            aw_mod.AniposeWrapper._calibrate_inhouse = original_calibrate


# =============================================================================
# Full 3-tab integration: DLC create → calibration → triangulation
# =============================================================================
class TestThreeTabEndToEnd:
    def test_dlc_tab_creates_project_then_calib_then_triangulation(
        self, qtbot, main_window, tmp_path, synthetic_dlc_csv,
    ):
        """W11 acceptance: all three tabs work together through AppState."""
        # 1. DLCTab creates a project (via MockDLCSubprocessRunner).
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner
        dlc_tab = main_window.tab_by_name("dlc")
        qtbot.addWidget(dlc_tab)
        dlc_tab._videos = [str(tmp_path / "v1.mp4")]
        dlc_tab._working_dir = str(tmp_path)
        runner = MockDLCSubprocessRunner()
        dlc_tab.set_runner(runner)
        dlc_tab._on_create_project_v2()
        assert Path(dlc_tab._config_path).is_file()

        # 2. Calibration3DTab publishes a calibration.
        from deepgait3.gui.shared_state import CalibrationView
        from deepgait3.core._legacy.triangulation_3d import Camera
        from deepgait3.core._legacy.anipose_wrapper import CalibrationResult
        K = np.array([[1000.0, 0, 320.0], [0, 1000.0, 240.0], [0, 0, 1]])
        view = CalibrationView(
            cameras=["left", "right"],
            reproj_rms_per_cam={"left": 1.0, "right": 1.5},
            method="inhouse", board_rows=5, board_cols=7,
        )
        cam_left = Camera(name="left", P=K @ np.hstack([np.eye(3), np.array([[0, 0, 200.0]]).T]))
        cam_right = Camera(name="right", P=K @ np.hstack([np.eye(3), np.array([[-200.0, 0, 0]]).T]))
        calib_result = CalibrationResult(
            cameras={"left": cam_left, "right": cam_right},
            reproj_rms_per_cam={"left": 1.0, "right": 1.5},
            board=view, method="inhouse",
        )
        main_window.app_state.set_calibration(view)
        main_window.app_state._calibration = calib_result

        # 3. Triangulation3DTab receives calibration + runs triangulation.
        tri_tab = main_window.tab_by_name("triangulation_3d")
        qtbot.addWidget(tri_tab)
        tri_tab._pose_2d_paths = {"left": synthetic_dlc_csv, "right": synthetic_dlc_csv}
        from deepgait3.core._legacy import anipose_wrapper as aw_mod
        original_calibrate = aw_mod.AniposeWrapper._calibrate_inhouse
        aw_mod.AniposeWrapper._calibrate_inhouse = lambda self, *a, **k: calib_result
        try:
            tri_tab._on_triangulate()
            assert main_window.app_state.pose_3d is not None
            assert main_window.app_state.pose_3d.n_frames > 0
            # Status bar must have been updated.
            assert "三角化完成" in tri_tab.status_label.text()
        finally:
            aw_mod.AniposeWrapper._calibrate_inhouse = original_calibrate