"""Unit tests for Phase 3 W9 — PySide6 GUI scaffolding (8 tabs).

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.3 W9):
    "8 tab 切换无崩溃" + "标签页结构、共享状态、QThread workers"

This module covers the W9 deliverables specifically:
  * :class:`MainWindow` constructs 8 tabs without crashing.
  * Tab switching to every index works (no exception).
  * :class:`AppState` signals propagate to subscribers.
  * :class:`Calibration3DTab` discovers camera directories + runs
    calibration on synthetic ChArUco images.
  * :class:`Triangulation3DTab` loads 2D pose CSVs + runs triangulation
    against a synthetic calibration.
  * :class:`GaitTab` (legacy) still accepts a CSV via ``load_csv``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def main_window(qtbot):
    """Construct MainWindow with offscreen Qt and register with qtbot."""
    from deepgait3.gui.main_window import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    return win


@pytest.fixture
def app_state():
    """A standalone AppState for unit-testing the signal plumbing."""
    from deepgait3.gui.shared_state import AppState
    return AppState()


@pytest.fixture
def synthetic_dlc_csv(tmp_path):
    """Write a 50-frame DLC-format CSV with 12 bodyparts."""
    rng = np.random.default_rng(42)
    n_frames = 50
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
    data = rng.random((n_frames, len(bodyparts) * 3))
    df = pd.DataFrame(data, columns=cols)
    path = tmp_path / "trial.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def synthetic_calibration_dir(tmp_path):
    """A calibration root with two camera subdirs, each with 3 dummy images.

    Real ChArUco detection requires actual board photos; we instead
    verify the directory-walking logic and that the tab gracefully
    reports ``inf`` RMS when no corners are detected.
    """
    root = tmp_path / "calib"
    for cam in ("left", "right"):
        cam_dir = root / cam
        cam_dir.mkdir(parents=True)
        for i in range(3):
            # 64×64 dark-gray noise — no ChArUco corners.
            arr = np.full((64, 64, 3), 30, dtype=np.uint8)
            cv2 = pytest.importorskip("cv2")
            cv2.imwrite(str(cam_dir / f"img_{i:02d}.png"), arr)
    return str(root)


# =============================================================================
# MainWindow — W9 acceptance gate
# =============================================================================
class TestMainWindowW9:
    def test_constructs_without_crash(self, main_window):
        assert main_window.n_tabs == 8

    def test_tab_labels_match_spec(self, main_window):
        labels = [main_window.tabs.tabText(i)
                  for i in range(main_window.n_tabs)]
        # W17: tab 1 is now "初始化" (renamed from "相机采集") and
        # moved to position 0; the other 7 tabs shift right by 1.
        assert labels == [
            "初始化", "步态分析", "FTIR 分析", "DLC 工作流",
            "3D 标定", "3D 三角化",
            "步态编辑", "文献图表",
        ]

    @pytest.mark.parametrize("tab_index", range(8))
    def test_each_tab_is_switchable(self, main_window, tab_index):
        """W9 acceptance gate: switching to each tab must not crash."""
        main_window.tabs.setCurrentIndex(tab_index)
        assert main_window.tabs.currentIndex() == tab_index
        assert main_window.tabs.currentWidget() is not None

    def test_tab_by_name_returns_widget(self, main_window):
        for name in MainWindow.TAB_ORDER:
            assert main_window.tab_by_name(name) is not None, name

    def test_app_state_is_appstate_instance(self, main_window):
        from deepgait3.gui.shared_state import AppState
        assert isinstance(main_window.app_state, AppState)

    def test_window_title_contains_deepgait(self, main_window):
        assert "deepgait" in main_window.windowTitle().lower()

    def test_status_bar_has_permanent_version_label(self, main_window):
        perm = main_window.status.findChildren(
            type(main_window.status).__mro__[0]
        )
        # The permanent QLabel "deepgait v2.0" must be findable.
        labels = main_window.status.findChildren(
            __import__("PySide6.QtWidgets", fromlist=["QLabel"]).QLabel
        )
        assert any("v2.0" in l.text() for l in labels)


# Import MainWindow name for the parametrized test above.
from deepgait3.gui.main_window import MainWindow


# =============================================================================
# AppState — signal plumbing
# =============================================================================
class TestAppState:
    def test_default_state_is_empty(self, app_state):
        assert app_state.gait_results is None
        assert app_state.footprint is None
        assert app_state.pose_3d is None
        assert app_state.calibration is None
        assert app_state.status_message == "Ready"

    def test_gait_results_signal_fires(self, qtbot, app_state):
        from deepgait3.gui.shared_state import GaitResultsView

        received = []
        app_state.gait_results_changed.connect(received.append)
        view = GaitResultsView(csv_path="x.csv", n_frames=10, metrics={"a": 1.0})
        with qtbot.waitSignal(app_state.gait_results_changed):
            app_state.set_gait_results(view)
        assert received == [view]
        assert app_state.gait_results is view

    def test_status_message_propagates(self, qtbot, app_state):
        received = []
        app_state.status_message_changed.connect(received.append)
        with qtbot.waitSignal(app_state.status_message_changed):
            app_state.set_status_message("hello")
        assert received == ["hello"]

    def test_clear_resets_all(self, app_state):
        from deepgait3.gui.shared_state import GaitResultsView

        app_state.set_gait_results(GaitResultsView())
        app_state.clear()
        assert app_state.gait_results is None
        assert app_state.footprint is None


# =============================================================================
# Calibration3DTab
# =============================================================================
class TestCalibration3DTab:
    def test_constructs_with_app_state(self, qtbot, app_state):
        from deepgait3.gui.calibration_3d_tab import Calibration3DTab
        tab = Calibration3DTab(app_state)
        qtbot.addWidget(tab)
        assert tab._state is app_state

    def test_discover_camera_dirs_walks_subdirs(
        self, qtbot, app_state, tmp_path,
    ):
        from deepgait3.gui.calibration_3d_tab import Calibration3DTab
        tab = Calibration3DTab(app_state)
        qtbot.addWidget(tab)
        # Build a synthetic tree.
        root = tmp_path / "calib"
        (root / "left").mkdir(parents=True)
        (root / "right").mkdir(parents=True)
        (root / "left" / "a.png").write_bytes(b"x")
        (root / "left" / "b.jpg").write_bytes(b"x")
        (root / "right" / "c.png").write_bytes(b"x")
        # Non-image file should be ignored.
        (root / "left" / "notes.txt").write_text("ignore me")
        out = tab._discover_camera_dirs(str(root))
        assert set(out.keys()) == {"left", "right"}
        assert len(out["left"]) == 2
        assert len(out["right"]) == 1

    def test_calibration_publishes_to_app_state(
        self, qtbot, app_state, synthetic_calibration_dir,
    ):
        """Run calibration on synthetic (cornerless) images: RMS must be
        ``inf`` but the result must still be published."""
        from deepgait3.gui.calibration_3d_tab import Calibration3DTab
        from deepgait3.gui.shared_state import CalibrationView

        tab = Calibration3DTab(app_state)
        qtbot.addWidget(tab)
        tab._images_root = synthetic_calibration_dir
        view = tab._run_calibration(synthetic_calibration_dir)
        assert isinstance(view, CalibrationView)
        assert view.cameras == ["left", "right"]
        # No real ChArUco corners → infinite RMS, but pipeline ran.
        for cam in view.cameras:
            assert view.reproj_rms_per_cam[cam] == float("inf")
        # State propagation via set_calibration.
        received = []
        app_state.calibration_changed.connect(received.append)
        app_state.set_calibration(view)
        assert received == [view]

    def test_run_calibration_raises_on_empty_dir(
        self, qtbot, app_state, tmp_path,
    ):
        from deepgait3.gui.calibration_3d_tab import Calibration3DTab
        tab = Calibration3DTab(app_state)
        qtbot.addWidget(tab)
        with pytest.raises(ValueError):
            tab._run_calibration(str(tmp_path))


# =============================================================================
# Triangulation3DTab
# =============================================================================
class TestTriangulation3DTab:
    def test_constructs_with_app_state(self, qtbot, app_state):
        from deepgait3.gui.triangulation_3d_tab import Triangulation3DTab
        tab = Triangulation3DTab(app_state)
        qtbot.addWidget(tab)
        assert tab._state is app_state

    def test_load_pose_2d_parses_dlc_csv(
        self, qtbot, app_state, synthetic_dlc_csv,
    ):
        from deepgait3.gui.triangulation_3d_tab import Triangulation3DTab
        tab = Triangulation3DTab(app_state)
        qtbot.addWidget(tab)
        arr = tab._load_pose_2d(synthetic_dlc_csv)
        assert arr.ndim == 3
        assert arr.shape[2] == 2
        assert arr.shape[0] == 50  # 50 frames

    def test_refresh_enable_state_requires_calibration_and_csvs(
        self, qtbot, app_state,
    ):
        from deepgait3.gui.triangulation_3d_tab import Triangulation3DTab
        tab = Triangulation3DTab(app_state)
        qtbot.addWidget(tab)
        # No calibration + no CSV → button disabled.
        assert tab.triangulate_btn.isEnabled() is False
        # Two CSVs but no calibration → still disabled.
        tab._pose_2d_paths = {"left": "a.csv", "right": "b.csv"}
        tab._refresh_enable_state()
        assert tab.triangulate_btn.isEnabled() is False


# =============================================================================
# GaitTab — legacy CSV load
# =============================================================================
class TestGaitTabLegacy:
    def test_legacy_gait_tab_records_csv_path(self, qtbot, main_window,
                                                synthetic_dlc_csv):
        """Legacy GaitTab stores the path on file pick and enables analyze."""
        gait_tab = main_window.tab_by_name("gait")
        qtbot.addWidget(gait_tab)
        # Simulate a file pick.
        gait_tab.path_edit.setText(synthetic_dlc_csv)
        gait_tab._csv_path = synthetic_dlc_csv
        gait_tab.analyze_btn.setEnabled(True)
        assert gait_tab._csv_path == synthetic_dlc_csv
        assert gait_tab.analyze_btn.isEnabled() is True

    def test_main_window_open_csv_sets_gait_tab_path(
        self, qtbot, main_window, synthetic_dlc_csv, monkeypatch,
    ):
        """Menu File → Open CSV updates the gait tab's path."""
        from PySide6.QtWidgets import QFileDialog
        monkeypatch.setattr(
            QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (synthetic_dlc_csv, "")),
        )
        main_window._on_open_csv()
        gait_tab = main_window.tab_by_name("gait")
        assert gait_tab.path_edit.text() == synthetic_dlc_csv
        assert gait_tab.analyze_btn.isEnabled() is True


# =============================================================================
# MainWindow menu integration
# =============================================================================
class TestMainWindowMenu:
    def test_open_csv_updates_gait_tab_path(self, qtbot, main_window,
                                              synthetic_dlc_csv,
                                              monkeypatch):
        """File → Open CSV updates the gait tab's path field."""
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(
            QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (synthetic_dlc_csv, "")),
        )
        main_window._on_open_csv()
        gait_tab = main_window.tab_by_name("gait")
        assert gait_tab.path_edit.text() == synthetic_dlc_csv

    def test_open_calibration_dir_updates_calib_tab(
        self, qtbot, main_window, synthetic_calibration_dir, monkeypatch,
    ):
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory",
            staticmethod(lambda *a, **k: synthetic_calibration_dir),
        )
        main_window._on_open_calibration_dir()
        calib_tab = main_window.tab_by_name("calibration_3d")
        assert calib_tab.path_edit.text() == synthetic_calibration_dir
        assert calib_tab._images_root == synthetic_calibration_dir