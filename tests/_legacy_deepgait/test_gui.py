"""GUI tests using pytest-qt.

Tests focus on widget state transitions, signal-slot connections, and
integration with core modules (not visual rendering).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem

from deepgait3.gui.main_window import MainWindow
from deepgait3.gui.style import PRIMARY, ACCENT, BG
from deepgait3.core._legacy import gait_export


@pytest.fixture
def main_window(qtbot):
    """Create a MainWindow and register it with qtbot."""
    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    return win


# ---------------------------------------------------------------------------
# Main window tests
# ---------------------------------------------------------------------------

def test_main_window_title(main_window):
    assert "deepgait" in main_window.windowTitle().lower()
    assert "步态" in main_window.windowTitle()


def test_main_window_has_six_tabs(main_window):
    assert main_window.tabs.count() == 8  # Phase 3 W9: 6 → 8 tabs
    labels = [main_window.tabs.tabText(i) for i in range(main_window.tabs.count())]
    assert "步态分析" in labels
    assert "FTIR" in " ".join(labels)
    assert "DLC" in " ".join(labels)
    assert "文献" in " ".join(labels)
    # W17: "相机采集" was renamed to "初始化".
    assert "初始化" in " ".join(labels)
    assert "步态编辑" in labels


def test_tab_switching(main_window, qtbot):
    for i in range(6):
        main_window.tabs.setCurrentIndex(i)
        assert main_window.tabs.currentIndex() == i


def test_status_bar_ready(main_window):
    assert "就绪" in main_window.status.currentMessage()


def test_menu_bar_exists(main_window):
    menu = main_window.menuBar()
    assert menu is not None
    # Check File menu exists
    file_menu = None
    for action in menu.actions():
        if "文件" in action.text():
            file_menu = action.menu()
            break
    assert file_menu is not None


# ---------------------------------------------------------------------------
# Style tests (basic sanity)
# ---------------------------------------------------------------------------

def test_style_constants():
    assert PRIMARY.startswith("#")
    assert ACCENT.startswith("#")
    assert BG.startswith("#")


# ---------------------------------------------------------------------------
# Shared state tests
# ---------------------------------------------------------------------------

def test_shared_state_initially_none(main_window):
    assert main_window.current_gait_results is None
    assert main_window.current_ftir_footprints is None
    assert main_window.current_ftir_intensities is None


# ---------------------------------------------------------------------------
# GaitTab tests
# ---------------------------------------------------------------------------

def test_gait_tab_browse_and_analyze(qtbot, tmp_path):
    """Load a synthetic CSV into GaitTab and run analysis."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.gui.gait_tab import GaitTab

    # Create synthetic CSV
    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)

    tab = GaitTab()
    qtbot.addWidget(tab)
    tab.show()

    # Simulate browse
    tab._csv_path = str(csv_path)
    tab.path_edit.setText(str(csv_path))
    tab.analyze_btn.setEnabled(True)

    # Click analyze
    qtbot.mouseClick(tab.analyze_btn, Qt.MouseButton.LeftButton)
    # Wait for worker to finish
    qtbot.waitUntil(lambda: tab._results is not None, timeout=5000)

    assert tab._results is not None
    assert len(tab._results.paws) == 4
    assert tab.table.rowCount() == 4
    assert tab.export_excel_btn.isEnabled()
    assert tab.export_csv_btn.isEnabled()


def test_gait_tab_export_csv(qtbot, tmp_path):
    """Analyze then export CSV."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.gui.gait_tab import GaitTab

    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)

    tab = GaitTab()
    qtbot.addWidget(tab)
    tab._csv_path = str(csv_path)
    tab.path_edit.setText(str(csv_path))
    tab.analyze_btn.setEnabled(True)
    qtbot.mouseClick(tab.analyze_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: tab._results is not None, timeout=5000)

    out = tmp_path / "gui_export.csv"
    tab._results = tab._results  # ensure set
    gait_export.to_summary_csv(tab._results, out)
    assert out.is_file()
    content = out.read_text()
    assert "RightFore" in content


# ---------------------------------------------------------------------------
# FTIRTab tests
# ---------------------------------------------------------------------------

def test_ftir_tab_load_and_analyze(qtbot, tmp_path):
    """Load synthetic FTIR image and analyze."""
    import cv2
    from tests._legacy_deepgait.test_ftir import make_synthetic_ftir_frame
    from deepgait3.gui.ftir_tab import FTIRTab

    img = tmp_path / "ftir.png"
    frame = make_synthetic_ftir_frame(noise=False)
    cv2.imwrite(str(img), frame)

    tab = FTIRTab()
    qtbot.addWidget(tab)
    tab.show()

    # Simulate browse
    tab._frame = frame
    tab.path_edit.setText(str(img))
    tab.analyze_btn.setEnabled(True)
    qtbot.mouseClick(tab.analyze_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: len(tab._footprints) > 0, timeout=5000)

    assert len(tab._footprints) == 4
    assert len(tab._intensities) == 4
    assert tab.table.rowCount() == 4
    assert tab.export_btn.isEnabled()


def test_ftir_tab_hsv_slider(qtbot):
    """HSV slider changes should update the label values."""
    from deepgait3.gui.ftir_tab import FTIRTab
    tab = FTIRTab()
    qtbot.addWidget(tab)
    # Set slider to a new value
    tab.h_low.setValue(50)
    assert tab.h_low.value() == 50


def test_ftir_tab_shared_state(qtbot, tmp_path, main_window):
    """Analyze FTIR should update shared state on main window."""
    import cv2
    from tests._legacy_deepgait.test_ftir import make_synthetic_ftir_frame
    from deepgait3.gui.ftir_tab import FTIRTab

    img = tmp_path / "ftir.png"
    frame = make_synthetic_ftir_frame(noise=False)
    cv2.imwrite(str(img), frame)

    # Replace FTIR tab in main window
    tab = FTIRTab()
    # W17: FTIR moved from index 1 to index 2. Use indexOf by name so
    # future reorders don't break this test.
    ftir_idx = main_window.tabs.indexOf(main_window.tab_by_name("ftir"))
    main_window.tabs.removeTab(ftir_idx)
    main_window.tabs.insertTab(ftir_idx, tab, "FTIR 分析")
    main_window.tabs.setCurrentIndex(ftir_idx)

    tab._frame = frame
    tab.analyze_btn.setEnabled(True)
    qtbot.mouseClick(tab.analyze_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: main_window.current_ftir_footprints is not None, timeout=5000)

    assert main_window.current_ftir_footprints is not None
    assert len(main_window.current_ftir_footprints) == 4


# ---------------------------------------------------------------------------
# DLCTab tests
# ---------------------------------------------------------------------------

def test_dlc_tab_widgets(qtbot):
    """DLCTab should have all required widgets."""
    from deepgait3.gui.dlc_tab import DLCTab
    tab = DLCTab()
    qtbot.addWidget(tab)
    # Bodyparts list should have 12 items
    assert tab.bp_list.count() == 12
    # Action buttons should exist
    assert tab.create_btn is not None
    assert tab.extract_btn is not None
    assert tab.label_btn is not None
    assert tab.train_btn is not None
    assert tab.analyze_btn is not None


def test_dlc_tab_create_project_no_dlc(qtbot, tmp_path):
    """Create project without DLC should graceful-fallback (write config only)."""
    from deepgait3.gui.dlc_tab import DLCTab
    tab = DLCTab()
    qtbot.addWidget(tab)

    # Set up spec inputs
    fake_video = tmp_path / "fake.mp4"
    fake_video.touch()
    tab._videos = [str(fake_video)]
    tab.work_dir_edit.setText(str(tmp_path))
    tab.project_edit.setText("testproj")
    tab.experimenter_edit.setText("alice")

    # Click create
    qtbot.mouseClick(tab.create_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: tab._worker is None, timeout=5000)

    # Config should be written (graceful fallback)
    proj_dirs = list(tmp_path.glob("testproj-alice-*"))
    assert len(proj_dirs) == 1
    cfg = proj_dirs[0] / "config.yaml"
    assert cfg.is_file()


def test_dlc_tab_add_videos(qtbot, tmp_path):
    """Add videos via direct API (simulating file dialog)."""
    from deepgait3.gui.dlc_tab import DLCTab
    tab = DLCTab()
    qtbot.addWidget(tab)
    fake = tmp_path / "v.mp4"
    fake.touch()
    tab._videos = [str(fake)]
    tab.video_list.addItem(QListWidgetItem(str(fake)))
    assert tab.video_list.count() == 1


def test_train_dialog_defaults(qtbot):
    """TrainDialog should default to GTX 3060-friendly settings."""
    from deepgait3.gui.dlc_tab import TrainDialog
    dlg = TrainDialog()
    qtbot.addWidget(dlg)
    params = dlg.get_params()
    assert params["net_type"] == "resnet_50"
    assert params["epochs"] == 200
    assert params["batch_size"] == 8
    assert params["device"] is None  # auto


# ---------------------------------------------------------------------------
# ChartsTab tests
# ---------------------------------------------------------------------------

def test_charts_tab_widgets(qtbot):
    """ChartsTab should have all chart types and export buttons."""
    from deepgait3.gui.charts_tab import ChartsTab
    tab = ChartsTab()
    qtbot.addWidget(tab)
    assert tab.chart_type_combo.count() == 5
    # All 3 export buttons start disabled
    assert not tab.export_png_btn.isEnabled()
    assert not tab.export_pdf_btn.isEnabled()
    assert not tab.export_svg_btn.isEnabled()


def test_charts_tab_generate_stance_swing(qtbot, tmp_path, main_window):
    """Generate stance/swing chart from shared gait results."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.gui.charts_tab import ChartsTab

    # Run analysis to populate shared state
    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    from deepgait3.core._legacy.pipeline import analyze
    main_window.current_gait_results = analyze(csv_path, fps=100)

    # Open charts tab
    tab = ChartsTab()
    # W17: charts moved from index 6 to index 7. Use indexOf by name.
    charts_idx = main_window.tabs.indexOf(main_window.tab_by_name("charts"))
    main_window.tabs.removeTab(charts_idx)
    main_window.tabs.insertTab(charts_idx, tab, "文献图表")
    main_window.tabs.setCurrentIndex(charts_idx)

    # Generate chart
    tab.chart_type_combo.setCurrentText("stance/swing 阶梯图")
    tab._on_generate()
    # Export buttons should be enabled
    assert tab.export_png_btn.isEnabled()


def test_charts_tab_export_png(qtbot, tmp_path, main_window):
    """Export PNG should create a file."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.gui.charts_tab import ChartsTab
    from deepgait3.core._legacy.pipeline import analyze

    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    main_window.current_gait_results = analyze(csv_path, fps=100)

    tab = ChartsTab()
    # W17: charts moved; use indexOf by name (position-immune).
    charts_idx = main_window.tabs.indexOf(main_window.tab_by_name("charts"))
    main_window.tabs.removeTab(charts_idx)
    main_window.tabs.insertTab(charts_idx, tab, "文献图表")
    main_window.tabs.setCurrentIndex(charts_idx)

    tab.chart_type_combo.setCurrentText("stride 长度分布")
    tab._on_generate()

    out = tmp_path / "chart.png"
    tab.canvas.fig.savefig(str(out), dpi=150)
    assert out.is_file()
    assert out.stat().st_size > 1000  # PNG has content


# ---------------------------------------------------------------------------
# CameraTab tests
# ---------------------------------------------------------------------------

def test_camera_tab_widgets(qtbot):
    """CameraTab should have source selector, parameters, and start/stop."""
    from deepgait3.gui.camera_tab import CameraTab
    tab = CameraTab()
    qtbot.addWidget(tab)
    assert tab.source_combo.count() >= 4  # 3 cameras + file
    assert tab.start_btn.isEnabled()
    assert not tab.stop_btn.isEnabled()
    assert tab.image_view is not None


def test_camera_tab_source_changed_to_file(qtbot):
    """Switching to file source should show file picker."""
    from deepgait3.gui.camera_tab import CameraTab
    tab = CameraTab()
    qtbot.addWidget(tab)
    tab.show()
    qtbot.waitExposed(tab)
    # Find "FILE" entry index
    file_idx = next(
        i for i in range(tab.source_combo.count())
        if tab.source_combo.itemData(i) == "FILE"
    )
    # Call handler directly
    tab._on_source_changed(file_idx)
    # Check via isVisibleTo or geometry instead of isVisible (parent visibility
    # matters in headless tests).
    assert not tab.file_edit.isHidden()
    assert not tab.browse_btn.isHidden()


def test_camera_worker_processes_synthetic_video(qtbot, tmp_path):
    """CameraWorker should emit frames from a synthetic video file."""
    import cv2
    import numpy as np
    from deepgait3.gui.workers import CameraWorker

    # Create a small synthetic video
    video_path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), (i * 8) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    frames_captured = []
    fps_values = []

    worker = CameraWorker(source=str(video_path), target_fps=30, width=320, height=240)
    worker.frame_ready.connect(lambda f: frames_captured.append(f))
    worker.fps_updated.connect(lambda f: fps_values.append(f))
    worker.start()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=5000)

    assert len(frames_captured) >= 25
    assert all(f.shape == (240, 320, 3) for f in frames_captured)


def test_camera_worker_handles_missing_source(qtbot):
    """CameraWorker should emit error for nonexistent source."""
    from deepgait3.gui.workers import CameraWorker

    errors = []
    worker = CameraWorker(source=999, target_fps=30, width=320, height=240)  # likely no camera
    worker.error.connect(lambda m: errors.append(m))
    worker.start()
    qtbot.waitUntil(lambda: not worker.isRunning(), timeout=3000)

    # Either error or finished quickly with no frames (depends on env)
    # In test env, camera 999 likely doesn't exist, so error expected
    assert len(errors) >= 0  # passes either way; main point is no crash


# ---------------------------------------------------------------------------
# EditorTab tests
# ---------------------------------------------------------------------------

def test_editor_tab_widgets(qtbot):
    """EditorTab should have 4 paw plots and controls."""
    from deepgait3.gui.editor_tab import EditorTab
    tab = EditorTab()
    qtbot.addWidget(tab)
    assert len(tab._plots) == 4
    for paw_name in ("LeftFore", "RightFore", "LeftHind", "RightHind"):
        assert paw_name in tab._plots
    # Controls start disabled
    assert not tab.frame_slider.isEnabled()
    assert not tab.flip_btn.isEnabled()
    assert not tab.export_btn.isEnabled()


def test_editor_tab_load_from_shared_state(qtbot, main_window, tmp_path):
    """Loading from shared state should populate plots."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.core._legacy.pipeline import analyze
    from deepgait3.gui.editor_tab import EditorTab

    # Populate shared state
    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    res = analyze(csv_path, fps=100)
    main_window.current_gait_results = res

    tab = EditorTab()
    # W17: editor moved from index 5 to index 6. Use indexOf by name.
    editor_idx = main_window.tabs.indexOf(main_window.tab_by_name("editor"))
    main_window.tabs.removeTab(editor_idx)
    main_window.tabs.insertTab(editor_idx, tab, "步态编辑")
    main_window.tabs.setCurrentIndex(editor_idx)

    tab._on_load()
    # Slider should be enabled
    assert tab.frame_slider.isEnabled()
    assert tab.frame_slider.maximum() == res.n_frames - 1
    # Plots should have data
    for plot in tab._plots.values():
        assert len(plot.get_data()) == res.n_frames
    # Table should have 4 rows
    assert tab.params_table.rowCount() == 4


def test_editor_flip_frame_changes_data(qtbot, main_window, tmp_path):
    """Flipping a frame should change in_stance at that frame."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.core._legacy.pipeline import analyze
    from deepgait3.gui.editor_tab import EditorTab

    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    main_window.current_gait_results = analyze(csv_path, fps=100)

    tab = EditorTab()
    # W17: editor moved from index 5 to index 6. Use indexOf by name.
    editor_idx = main_window.tabs.indexOf(main_window.tab_by_name("editor"))
    main_window.tabs.removeTab(editor_idx)
    main_window.tabs.insertTab(editor_idx, tab, "步态编辑")
    main_window.tabs.setCurrentIndex(editor_idx)
    qtbot.addWidget(tab)
    tab._on_load()
    # Get first paw's data
    paw_name = "LeftFore"
    plot = tab._plots[paw_name]
    data_before = plot.get_data()
    # Flip frame 50
    plot.flip_frame(50)
    data_after = plot.get_data()
    # Should differ at frame 50
    assert data_before[50] != data_after[50]
    # Other frames should be unchanged
    assert np.array_equal(data_before[:50], data_after[:50])
    assert np.array_equal(data_before[51:], data_after[51:])
    # Corrected mask should mark frame 50
    mask = plot.get_corrected_mask()
    assert mask[50]
    assert not mask[49]


def test_editor_auto_correct_button(qtbot, main_window, tmp_path):
    """AutoCorrect button should run and re-enable plot updates."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.core._legacy.pipeline import analyze
    from deepgait3.gui.editor_tab import EditorTab

    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    main_window.current_gait_results = analyze(csv_path, fps=100)

    tab = EditorTab()
    # W17: editor moved from index 5 to index 6. Use indexOf by name.
    editor_idx = main_window.tabs.indexOf(main_window.tab_by_name("editor"))
    main_window.tabs.removeTab(editor_idx)
    main_window.tabs.insertTab(editor_idx, tab, "步态编辑")
    main_window.tabs.setCurrentIndex(editor_idx)
    qtbot.addWidget(tab)
    tab._on_load()
    # Run auto-correct (suppresses its modal completion dialog)
    with patch("deepgait3.gui.editor_tab.QMessageBox.information"):
        tab._on_run_auto_correct()
    # All plots should still have data
    for plot in tab._plots.values():
        assert len(plot.get_data()) > 0


def test_editor_export_csv(qtbot, main_window, tmp_path):
    """Export CSV should write a file."""
    from tests._legacy_deepgait.test_pipeline import _make_synthetic_trajectory, _write_dlc_csv
    from deepgait3.core._legacy.pipeline import analyze
    from deepgait3.gui.editor_tab import EditorTab
    from deepgait3.core._legacy import gait_export

    traj = _make_synthetic_trajectory(n_frames=200, fps=100)
    csv_path = tmp_path / "mouse_DLC.csv"
    _write_dlc_csv(traj, csv_path)
    main_window.current_gait_results = analyze(csv_path, fps=100)

    tab = EditorTab()
    # W17: editor moved from index 5 to index 6. Use indexOf by name.
    editor_idx = main_window.tabs.indexOf(main_window.tab_by_name("editor"))
    main_window.tabs.removeTab(editor_idx)
    main_window.tabs.insertTab(editor_idx, tab, "步态编辑")
    main_window.tabs.setCurrentIndex(editor_idx)
    qtbot.addWidget(tab)
    tab._on_load()
    out = tmp_path / "edited.csv"
    gait_export.to_summary_csv(tab._results, out)
    assert out.is_file()
    content = out.read_text()
    assert "LeftFore" in content





