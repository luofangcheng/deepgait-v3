"""Regression tests for the W17 「初始化」 tab refactor.

Covers the new ``InitializationTab`` (``deepgait3.gui.initialization_tab``),
the extended ``ICamera`` abstract API, the ``CameraConfigView`` shared
state, and the JSON preset persistence layer.

Run with::

    pytest tests/unit/test_initialization_tab.py -v
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Qt setup
# ---------------------------------------------------------------------------
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    return app


# ---------------------------------------------------------------------------
# Test ICamera abstract API extension
# ---------------------------------------------------------------------------
class TestICameraNewAbstractMethods:
    """Verify all 4 implementations satisfy the extended ICamera ABC."""

    def test_icamera_lists_18_abstract_methods(self):
        from deepgait3.hardware.camera.base import ICamera
        expected = {
            "open", "close", "grab_one",
            "start_continuous", "stop_continuous",
            "configure_hardware_trigger",
            "set_exposure_us", "set_gain_db", "set_roi",
            "get_serial", "get_model",
            "set_brightness", "set_contrast", "set_pixel_format", "set_fps",
            "get_supported_features", "snapshot_config", "restore_config",
        }
        actual = set(ICamera.__abstractmethods__)
        assert expected == actual, (
            f"Missing: {expected - actual}; Extra: {actual - expected}"
        )

    def test_mockcamera_satisfies_abc(self):
        from deepgait3.hardware.camera.base import ICamera
        from deepgait3.hardware.camera.multi_cam import MockCamera
        m = MockCamera(serial="M1", fps=100, width=640, height=480)
        # If any abstract method is missing, instantiation would fail
        # (ICamera.__init__ would complain).
        assert isinstance(m, ICamera)

    def test_default_features_dict_has_all_keys(self):
        from deepgait3.hardware.camera.base import ICamera
        feat = ICamera._default_features()
        for key in ("brightness", "contrast", "exposure_us", "gain_db",
                    "fps", "pixel_format", "roi"):
            assert key in feat, f"missing feature key: {key}"


# ---------------------------------------------------------------------------
# Test MockCamera full parameter set
# ---------------------------------------------------------------------------
class TestMockCameraExtendedAPI:
    def setup_method(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera
        self.cam = MockCamera(serial="M1", fps=100, width=640, height=480)

    def test_set_brightness_valid(self):
        self.cam.set_brightness(50)
        snap = self.cam.snapshot_config()
        assert snap["brightness"] == 50

    def test_set_brightness_out_of_range(self):
        with pytest.raises(ValueError, match="brightness"):
            self.cam.set_brightness(-200)
        with pytest.raises(ValueError, match="brightness"):
            self.cam.set_brightness(200)

    def test_set_contrast_valid(self):
        self.cam.set_contrast(-30)
        assert self.cam.snapshot_config()["contrast"] == -30

    def test_set_contrast_out_of_range(self):
        with pytest.raises(ValueError, match="contrast"):
            self.cam.set_contrast(-200)

    def test_set_fps_valid(self):
        self.cam.set_fps(250)
        assert self.cam.snapshot_config()["fps"] == 250

    def test_set_fps_out_of_range(self):
        with pytest.raises(ValueError, match="fps"):
            self.cam.set_fps(0)
        with pytest.raises(ValueError, match="fps"):
            self.cam.set_fps(501)

    def test_set_pixel_format_valid(self):
        self.cam.set_pixel_format("Mono8")
        assert self.cam.snapshot_config()["pixel_format"] == "Mono8"

    def test_set_pixel_format_empty(self):
        with pytest.raises(ValueError, match="pixel_format"):
            self.cam.set_pixel_format("")

    def test_get_supported_features_structure(self):
        feat = self.cam.get_supported_features()
        # brightness: (lo, hi, default, step)
        lo, hi, default, step = feat["brightness"]
        assert lo <= default <= hi
        assert step > 0
        # fps: (lo, hi, default)
        lo, hi, default = feat["fps"]
        assert lo <= default <= hi
        # pixel_format: list of strings
        assert isinstance(feat["pixel_format"], list)
        assert all(isinstance(s, str) for s in feat["pixel_format"])

    def test_snapshot_then_restore_roundtrip(self):
        """Save snapshot to one camera, restore to another — values match."""
        self.cam.set_brightness(75)
        self.cam.set_contrast(-50)
        self.cam.set_fps(200)
        self.cam.set_pixel_format("BayerRG8")
        self.cam.set_roi(0, 0, 1280, 720)
        snap = self.cam.snapshot_config()
        from deepgait3.hardware.camera.multi_cam import MockCamera
        cam2 = MockCamera(serial="M2", fps=100, width=640, height=480)
        cam2.restore_config(snap)
        snap2 = cam2.snapshot_config()
        assert snap2["brightness"] == 75
        assert snap2["contrast"] == -50
        assert snap2["fps"] == 200
        assert snap2["pixel_format"] == "BayerRG8"


# ---------------------------------------------------------------------------
# Test CameraConfigView + AppState signal
# ---------------------------------------------------------------------------
class TestCameraConfigViewSharedState:
    def test_view_default_values(self):
        from deepgait3.gui.shared_state import CameraConfigView
        v = CameraConfigView(role="left", serial="SN-1")
        assert v.role == "left"
        assert v.width == 640
        assert v.height == 480
        assert v.fps == 100
        assert v.online is False

    def test_view_as_dict_has_all_keys(self):
        from deepgait3.gui.shared_state import CameraConfigView
        v = CameraConfigView(role="left", serial="SN-1",
                              width=1280, height=720, fps=200,
                              brightness=30, contrast=-10,
                              pixel_format="Mono8",
                              roi=(100, 50, 1180, 670), online=True)
        d = v.as_dict()
        for k in ("role", "serial", "width", "height", "fps",
                  "exposure_us", "gain_db", "brightness", "contrast",
                  "pixel_format", "roi", "online"):
            assert k in d

    def test_appstate_camera_config_signal_fires(self, qapp):
        from deepgait3.gui.shared_state import AppState, CameraConfigView
        s = AppState()
        captured = []
        s.camera_config_changed.connect(lambda v: captured.append(v.role))
        s.set_camera_config(CameraConfigView(role="left", serial="X"))
        s.set_camera_config(CameraConfigView(role="right", serial="Y"))
        assert captured == ["left", "right"]
        assert s.get_camera_config("right").serial == "Y"

    def test_appstate_camera_config_overwrite(self, qapp):
        from deepgait3.gui.shared_state import AppState, CameraConfigView
        s = AppState()
        s.set_camera_config(CameraConfigView(role="top", serial="A", fps=100))
        s.set_camera_config(CameraConfigView(role="top", serial="A", fps=200))
        assert s.get_camera_config("top").fps == 200


# ---------------------------------------------------------------------------
# Test InitializationTab widget structure
# ---------------------------------------------------------------------------
class TestInitializationTabStructure:
    def setup_method(self):
        from deepgait3.gui.shared_state import AppState
        from deepgait3.gui.initialization_tab import InitializationTab
        self.state = AppState()
        self.tab = InitializationTab(app_state=self.state)

    def test_has_two_sub_tabs(self):
        # W17.1: 上下分体布局, count() 由 splitter 提供, 兼容接口由
        # _SplitterAsTabsShim 提供 (返回标签列表)。
        assert self.tab.n_sub_tabs() == 2
        assert self.tab.sub_tabs.tabText(0) == "多相机配置"
        assert self.tab.sub_tabs.tabText(1) == "实时预览 / 录制"

    def test_layout_is_split_view_not_tab_widget(self):
        """W18.1: 单层布局 — config_panel 是唯一的顶层 widget。
        不再使用 QSplitter; sub_splitter 设为 None。
        """
        from PySide6.QtWidgets import QSplitter
        from PySide6.QtCore import Qt
        # W18.1: sub_splitter is None (removed).
        assert self.tab.sub_splitter is None
        # config_panel (MultiCameraConfigPanel) 是唯一可见 widget.
        assert self.tab.config_panel is not None
        # n_sub_tabs 由 shim 提供, 仍返回 2
        assert self.tab.n_sub_tabs() == 2
        # sub_tabs shim still works
        assert self.tab.sub_tabs.count() == 2
        assert self.tab.sub_tabs.tabText(0) == "多相机配置"
        assert self.tab.sub_tabs.tabText(1) == "实时预览 / 录制"

    def test_has_four_camera_groups(self):
        groups = self.tab.config_groups()
        assert set(groups.keys()) == {"bottom", "left", "right", "top"}
        for role, grp in groups.items():
            assert grp.role == role

    def test_camera_group_has_all_widgets(self):
        grp = self.tab.config_groups()["left"]
        for attr in ("serial_label", "status_badge",
                      "brightness_slider", "contrast_slider",
                      "exposure_spin", "gain_spin",
                      "width_spin", "height_spin", "fps_spin",
                      "pixel_combo",
                      "roi_x", "roi_y", "roi_w", "roi_h",
                      "apply_btn"):
            assert hasattr(grp, attr), f"group missing {attr}"

    def test_collect_params_roundtrip(self):
        grp = self.tab.config_groups()["left"]
        grp.brightness_slider.setValue(40)
        grp.contrast_slider.setValue(-15)
        grp.fps_spin.setValue(180)
        grp.exposure_spin.setValue(8000.0)
        params = grp.collect_params()
        assert params["brightness"] == 40
        assert params["contrast"] == -15
        assert params["fps"] == 180
        assert params["exposure_us"] == 8000.0

    def test_populate_from_roundtrip(self):
        grp = self.tab.config_groups()["right"]
        snapshot = {
            "width": 1280, "height": 720,
            "fps": 200, "exposure_us": 12345.6,
            "gain_db": 5.0,
            "brightness": 33, "contrast": -22,
            "pixel_format": "BayerRG8",
            "x": 10, "y": 20, "roi_w": 1024, "roi_h": 600,
        }
        grp.populate_from(snapshot)
        params = grp.collect_params()
        assert params["width"] == 1280
        assert params["fps"] == 200
        assert params["brightness"] == 33
        assert params["pixel_format"] == "BayerRG8"
        assert params["x"] == 10

    def test_backward_compat_widget_proxies(self):
        """The old CameraTab put these directly on the tab; we proxy them."""
        for attr in ("source_combo", "file_edit", "browse_btn",
                      "width_spin", "height_spin", "fps_spin",
                      "record_check", "start_btn", "stop_btn",
                      "fps_label", "image_view"):
            assert hasattr(self.tab, attr), f"missing proxy: {attr}"


# ---------------------------------------------------------------------------
# Test apply_to() pushes to camera
# ---------------------------------------------------------------------------
class TestApplyToCamera:
    def test_apply_to_mockcamera_calls_all_setters(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera
        from deepgait3.gui.initialization_tab import InitializationTab
        from deepgait3.gui.shared_state import AppState
        state = AppState()
        tab = InitializationTab(app_state=state)
        cam = MockCamera(serial="M1", fps=100, width=640, height=480)
        grp = tab.config_groups()["left"]
        grp.set_camera(cam)
        grp.brightness_slider.setValue(50)
        grp.contrast_slider.setValue(20)
        grp.fps_spin.setValue(200)
        grp.pixel_combo.setCurrentText("Mono8")
        errs = grp.apply_to(cam)
        assert errs == []
        snap = cam.snapshot_config()
        assert snap["brightness"] == 50
        assert snap["contrast"] == 20
        assert snap["fps"] == 200
        assert snap["pixel_format"] == "Mono8"

    def test_apply_to_none_returns_error(self):
        from deepgait3.gui.initialization_tab import InitializationTab
        from deepgait3.gui.shared_state import AppState
        tab = InitializationTab(app_state=AppState())
        grp = tab.config_groups()["bottom"]
        errs = grp.apply_to(None)
        assert len(errs) == 1
        assert "未绑定" in errs[0]


# ---------------------------------------------------------------------------
# Test preset save/load
# ---------------------------------------------------------------------------
class TestPresetSaveLoad:
    def setup_method(self, tmp_path=None):
        from deepgait3.gui.shared_state import AppState
        from deepgait3.gui.initialization_tab import (
            InitializationTab, _save_preset, _load_preset, _preset_dir,
        )
        self.state = AppState()
        self.tab = InitializationTab(app_state=self.state)
        self._save_preset = _save_preset
        self._load_preset = _load_preset
        self._preset_dir = _preset_dir

    def test_save_and_load_roundtrip(self, tmp_path):
        # Redirect preset dir to tmp_path.
        with mock.patch("deepgait3.gui.initialization_tab._preset_dir",
                         return_value=tmp_path):
            configs = {
                "left": {"width": 1280, "height": 720, "fps": 200,
                          "brightness": 30, "contrast": -10,
                          "pixel_format": "Mono8",
                          "x": 0, "y": 0, "roi_w": 1280, "roi_h": 720},
                "right": {"width": 640, "height": 480, "fps": 100,
                           "brightness": 0, "contrast": 0,
                           "pixel_format": "BGR8",
                           "x": 0, "y": 0, "roi_w": 640, "roi_h": 480},
            }
            path = self._save_preset("lab1", configs)
            assert path.exists()
            assert path.suffix == ".json"
            loaded = self._load_preset("lab1")
            assert loaded == configs

    def test_load_missing_raises_filenotfound(self, tmp_path):
        with mock.patch("deepgait3.gui.initialization_tab._preset_dir",
                         return_value=tmp_path):
            with pytest.raises(FileNotFoundError):
                self._load_preset("nonexistent")

    def test_load_malformed_raises_valueerror(self, tmp_path):
        with mock.patch("deepgait3.gui.initialization_tab._preset_dir",
                         return_value=tmp_path):
            bad = tmp_path / "bad.json"
            bad.write_text('{"not_a_preset": true}', encoding="utf-8")
            with pytest.raises(ValueError, match="not look like a deepgait"):
                self._load_preset("bad")

    def test_install_preset_populates_widgets(self, tmp_path):
        with mock.patch("deepgait3.gui.initialization_tab._preset_dir",
                         return_value=tmp_path):
            configs = {
                "left": {"width": 1024, "height": 768, "fps": 150,
                          "brightness": 50, "contrast": 0,
                          "pixel_format": "BayerRG8",
                          "x": 0, "y": 0, "roi_w": 1024, "roi_h": 768},
            }
            self._save_preset("test1", configs)
            self.tab.install_preset("test1")
            grp = self.tab.config_groups()["left"]
            assert grp.width_spin.value() == 1024
            assert grp.fps_spin.value() == 150
            assert grp.brightness_slider.value() == 50
            assert grp.pixel_combo.currentText() == "BayerRG8"
            # Unspecified roles (right/top/bottom) keep defaults.
            grp_r = self.tab.config_groups()["right"]
            assert grp_r.width_spin.value() == 640
            assert grp_r.fps_spin.value() == 100


# ---------------------------------------------------------------------------
# Test MainWindow integration
# ---------------------------------------------------------------------------
class TestMainWindowIntegration:
    def test_main_window_has_initialization_tab_at_position_0(self, qapp):
        from deepgait3.gui.main_window import MainWindow
        w = MainWindow()
        assert w.n_tabs == 8
        assert w.tabs.tabText(0) == "初始化"

    def test_main_window_old_key_still_works(self, qapp):
        from deepgait3.gui.main_window import MainWindow
        w = MainWindow()
        # Legacy code used tab_by_name("camera"); should still resolve.
        new = w.tab_by_name("initialization")
        old = w.tab_by_name("camera")
        assert new is old
        assert new is not None

    def test_main_window_passes_appstate_to_init_tab(self, qapp):
        from deepgait3.gui.main_window import MainWindow
        w = MainWindow()
        init_tab = w.tab_by_name("initialization")
        assert init_tab.app_state is w.app_state

    def test_appstate_publishes_camera_config_on_apply(self, qapp):
        from deepgait3.gui.main_window import MainWindow
        from deepgait3.gui.initialization_tab import (
            _save_preset, _load_preset,
        )
        from deepgait3.gui.shared_state import CameraConfigView
        w = MainWindow()
        captured = []
        w.app_state.camera_config_changed.connect(
            lambda v: captured.append((v.role, v.fps, v.brightness))
        )
        # Manually publish (simulating _on_apply_all) for the left camera.
        w.app_state.set_camera_config(CameraConfigView(
            role="left", serial="SN-L", width=1280, height=720, fps=200,
            brightness=42, online=True,
        ))
        assert ("left", 200, 42) in captured


# ---------------------------------------------------------------------------
# W18: per-camera preview pane
# ---------------------------------------------------------------------------
class TestW18PerCameraPreviewPane:
    """W18: each camera group has a CameraPreviewPane with ▶/■/录制."""

    def setup_method(self):
        from deepgait3.gui.shared_state import AppState
        from deepgait3.gui.initialization_tab import InitializationTab
        self.state = AppState()
        self.tab = InitializationTab(app_state=self.state)

    def test_each_group_has_preview_pane(self):
        # W18.1: preview pane no longer embedded in CameraConfigGroup;
        # it lives on config_panel.preview_panes().
        for role, pane in self.tab.config_panel.preview_panes().items():
            assert pane.role == role

    def test_preview_pane_has_start_stop_record(self):
        from deepgait3.gui.initialization_tab import CameraPreviewPane
        for pane in self.tab.config_panel.preview_panes().values():
            assert isinstance(pane, CameraPreviewPane)
            assert pane.start_btn is not None
            assert pane.stop_btn is not None
            assert pane.record_check is not None
            # Initial state: stop disabled
            assert pane.start_btn.isEnabled() is True
            assert pane.stop_btn.isEnabled() is False
            assert pane.is_running() is False

    def test_preview_pane_start_without_camera_warns(self, qapp, monkeypatch):
        """▶ 按钮在没有 camera 时弹警告,不崩溃。"""
        from PySide6.QtWidgets import QMessageBox
        pane = self.tab.config_panel.preview_panes()["left"]
        # 模拟 QMessageBox.warning 不阻塞
        warning_calls = []
        monkeypatch.setattr(QMessageBox, "warning",
                            staticmethod(lambda *a, **k: warning_calls.append(a)))
        pane._on_start()
        assert len(warning_calls) == 1
        # 状态没变
        assert pane.is_running() is False

    def test_preview_pane_starts_and_stops_with_mock(self, qapp, monkeypatch):
        """绑定 MockCamera → ▶ 启动 → ■ 停止, 状态正确切换。"""
        from deepgait3.hardware.camera.multi_cam import MockCamera
        from deepgait3.gui.initialization_tab import CameraPreviewPane
        pane = self.tab.config_panel.preview_panes()["left"]
        cam = MockCamera(serial="M1", fps=100, width=320, height=240)
        pane.set_camera(cam)
        pane._on_start()
        # 因为 QTimer 是非阻塞的, 立即 check is_running 应为 True
        assert pane.is_running() is True
        assert pane.start_btn.isEnabled() is False
        assert pane.stop_btn.isEnabled() is True
        # 停止
        pane._on_stop()
        assert pane.is_running() is False
        assert pane.start_btn.isEnabled() is True
        assert pane.stop_btn.isEnabled() is False

    def test_preview_pane_set_camera_stops_existing(self, qapp):
        """运行时换相机应先停止再绑。"""
        from deepgait3.hardware.camera.multi_cam import MockCamera
        pane = self.tab.config_panel.preview_panes()["right"]
        cam1 = MockCamera(serial="M1", fps=100, width=320, height=240)
        pane.set_camera(cam1)
        pane._on_start()
        assert pane.is_running() is True
        # 换相机
        cam2 = MockCamera(serial="M2", fps=100, width=320, height=240)
        pane.set_camera(cam2)
        assert pane.is_running() is False
        assert pane._camera is cam2

    def test_layout_4_rows_vertical(self, qapp):
        """W18.1: 2 层水平 — 配置 + 预览都在 QScrollArea 内的 QVBoxLayout 中。"""
        from PySide6.QtWidgets import QGridLayout, QScrollArea, QVBoxLayout
        from deepgait3.gui.initialization_tab import CameraConfigGroup, CameraPreviewPane
        config_panel = self.tab.config_panel
        # 找到 QScrollArea → cam_widget → QVBoxLayout → 两个 QGridLayout
        scroll = None
        for i in range(config_panel.layout().count()):
            item = config_panel.layout().itemAt(i)
            w = item.widget()
            if isinstance(w, QScrollArea):
                scroll = w
                break
        assert scroll is not None, "no QScrollArea"
        cam_widget = scroll.widget()
        assert cam_widget is not None
        cam_layout = cam_widget.layout()
        assert isinstance(cam_layout, QVBoxLayout)
        # cam_layout 包含 2 个 QGridLayout (配置 + 预览)
        grids = []
        for i in range(cam_layout.count()):
            item = cam_layout.itemAt(i)
            lo = item.layout()
            if isinstance(lo, QGridLayout):
                grids.append(lo)
        assert len(grids) >= 2, f"expected 2 grids, found {len(grids)}"
        # 第一个 grid: 4 CameraConfigGroup
        config_grid = grids[0]
        assert config_grid.columnCount() == 4
        for c in range(4):
            w = config_grid.itemAtPosition(0, c).widget()
            assert isinstance(w, CameraConfigGroup)
        # 第二个 grid: 4 CameraPreviewPane
        preview_grid = grids[1]
        assert preview_grid.columnCount() == 4
        for c in range(4):
            w = preview_grid.itemAtPosition(0, c).widget()
            assert isinstance(w, CameraPreviewPane)
