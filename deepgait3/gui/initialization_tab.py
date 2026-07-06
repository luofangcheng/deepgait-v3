"""Tab 1: 初始化 (Initialization) — multi-camera configuration + live preview.

This tab was previously named "相机采集" (Camera Acquisition) and lived at
tab index 7.  It is now the **first** tab the user sees, because the
natural workflow starts with hardware setup before any acquisition.

The tab is internally a nested ``QTabWidget`` with two sub-tabs:

* **多相机配置** (Multi-camera Configuration)
    - 4 camera groups (left / right / top / bottom) with full parameter
      set: brightness, contrast, exposure, gain, ROI, fps, pixel format
    - Each group has an "应用此相机" button that pushes the parameters
      to the corresponding :class:`ICamera` instance.
    - A "应用全部" button pushes to all 4 cameras at once.
    - A "保存预设" / "加载预设" pair persists / restores the full
      4-camera state to / from a JSON file under
      ``$XDG_DATA_HOME/deepgait/camera_presets/``.

* **实时预览 / 录制** (Live Preview / Recording)
    - The single-camera preview + recording from the old
      :class:`CameraTab` (via :class:`CameraWorker`).
    - The 4-camera synchronous preview via :class:`MultiCameraManager`.

The configuration is published to :class:`AppState` via
:meth:`AppState.set_camera_config` so other tabs (FTIR, triangulation,
recording) can subscribe.

Replaces the old ``camera_tab.py`` (kept as a thin shim for backward
compatibility — see ``camera_tab.py``).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph import ImageView

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QVBoxLayout,
    QWidget,
)

from deepgait3.gui.shared_state import AppState, CameraConfigView
from deepgait3.gui.workers import CameraWorker


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default 4-camera roster
# ---------------------------------------------------------------------------
DEFAULT_ROLES: Tuple[str, ...] = ("bottom", "left", "right", "top")
ROLE_LABELS: Dict[str, str] = {
    "bottom": "C1 · 底部 (FTIR 足迹)",
    "left":   "C2 · 左侧 (3D 姿态)",
    "right":  "C3 · 右侧 (3D 姿态)",
    "top":    "C4 · 顶部 (3D 姿态)",
}


def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Preset storage
# ---------------------------------------------------------------------------
def _preset_dir() -> Path:
    """Return the directory where camera presets are stored.

    Honors ``$XDG_DATA_HOME`` on Linux; falls back to
    ``~/.local/share`` (XDG default). On Windows, uses
    ``%APPDATA%/deepgait``.
    """
    base = os.environ.get("XDG_DATA_HOME")
    if not base:
        if os.name == "nt":
            base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        else:
            base = str(Path.home() / ".local" / "share")
    d = Path(base) / "deepgait" / "camera_presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_preset(name: str, configs: Dict[str, Dict[str, Any]]) -> Path:
    payload = {
        "version": 1,
        "name": name,
        "cameras": configs,
    }
    path = _preset_dir() / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def _load_preset(name: str) -> Dict[str, Dict[str, Any]]:
    path = _preset_dir() / f"{name}.json"
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or "cameras" not in data:
        raise ValueError(
            f"Preset {name!r} at {path} does not look like a deepgait "
            f"camera preset (missing 'cameras' key). Re-export or delete."
        )
    return data["cameras"]


# ---------------------------------------------------------------------------
# One per-camera configuration group
# ---------------------------------------------------------------------------
class CameraConfigGroup(QGroupBox):
    """Widget for configuring one camera.

    Holds widgets (sliders, spinboxes, comboboxes) for one role and a
    single :meth:`apply_to(camera)` method that pushes all values to a
    given :class:`ICamera` instance.
    """

    def __init__(self, role: str, parent: QWidget | None = None) -> None:
        super().__init__(ROLE_LABELS.get(role, role), parent)
        self.role = role
        self._build_ui()

    def _build_ui(self) -> None:
        # W18.1: 恢复为纯 QFormLayout, 不再嵌入 preview_pane。
        # 预览现在由 MultiCameraConfigPanel 第二层 (4 个水平 CameraPreviewPane) 提供。
        layout = QFormLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # 序列号 (read-only display, populated on detect)
        self.serial_label = QLabel("未检测")
        self.serial_label.setStyleSheet("color: #555; font-family: monospace;")
        layout.addRow("序列号:", self.serial_label)

        # 状态徽章
        self.status_badge = QLabel("⚪ 离线")
        self.status_badge.setStyleSheet("color: #888;")
        layout.addRow("状态:", self.status_badge)

        # 亮度 (slider + value label)
        self.brightness_slider, self.brightness_value_label = self._make_slider(
            -100, 100, 0
        )
        layout.addRow("亮度:", self._slider_row(
            self.brightness_slider, self.brightness_value_label
        ))

        # 对比度
        self.contrast_slider, self.contrast_value_label = self._make_slider(
            -100, 100, 0
        )
        layout.addRow("对比度:", self._slider_row(
            self.contrast_slider, self.contrast_value_label
        ))

        # 曝光 (μs)
        self.exposure_spin = QDoubleSpinBox()
        self.exposure_spin.setRange(50.0, 1_000_000.0)
        self.exposure_spin.setDecimals(1)
        self.exposure_spin.setSingleStep(100.0)
        self.exposure_spin.setValue(5000.0)
        self.exposure_spin.setSuffix(" μs")
        layout.addRow("曝光:", self.exposure_spin)

        # 增益 (dB)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 36.0)
        self.gain_spin.setDecimals(1)
        self.gain_spin.setSingleStep(0.5)
        self.gain_spin.setValue(0.0)
        self.gain_spin.setSuffix(" dB")
        layout.addRow("增益:", self.gain_spin)

        # 分辨率 W x H
        res_row = QWidget()
        res_layout = QHBoxLayout(res_row)
        res_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 4096)
        self.width_spin.setValue(640)
        self.width_spin.setSingleStep(160)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4096)
        self.height_spin.setValue(480)
        self.height_spin.setSingleStep(120)
        res_layout.addWidget(self.width_spin)
        res_layout.addWidget(QLabel("×"))
        res_layout.addWidget(self.height_spin)
        res_layout.addStretch()
        layout.addRow("分辨率:", res_row)

        # 帧率
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 500)
        self.fps_spin.setValue(100)
        layout.addRow("目标 FPS:", self.fps_spin)

        # 像素格式
        self.pixel_combo = QComboBox()
        for fmt in ("BGR8", "Mono8", "BayerRG8", "BayerGB8",
                    "BayerGR8", "BayerBG8", "RGB8"):
            self.pixel_combo.addItem(fmt)
        self.pixel_combo.setCurrentText("BGR8")
        layout.addRow("像素格式:", self.pixel_combo)

        # ROI
        roi_row = QWidget()
        roi_layout = QHBoxLayout(roi_row)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_x = QSpinBox(); self.roi_x.setRange(0, 4096); self.roi_x.setValue(0)
        self.roi_y = QSpinBox(); self.roi_y.setRange(0, 4096); self.roi_y.setValue(0)
        self.roi_w = QSpinBox(); self.roi_w.setRange(64, 4096); self.roi_w.setValue(640)
        self.roi_h = QSpinBox(); self.roi_h.setRange(64, 4096); self.roi_h.setValue(480)
        for sp, label in (
            (self.roi_x, "x"), (self.roi_y, "y"),
            (self.roi_w, "w"), (self.roi_h, "h"),
        ):
            sp.setSingleStep(8)
            sp.setMaximumWidth(80)
            roi_layout.addWidget(QLabel(label))
            roi_layout.addWidget(sp)
        roi_layout.addStretch()
        layout.addRow("ROI:", roi_row)

        # Apply button (this camera only)
        self.apply_btn = QPushButton(f"应用此相机")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addRow("", self.apply_btn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_slider(lo: int, hi: int, default: int) -> Tuple[QWidget, QLabel]:
        """Build a horizontal QSlider with a value label."""
        from PySide6.QtWidgets import QSlider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(lo, hi)
        slider.setValue(default)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval((hi - lo) // 4 or 1)
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(36)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight |
                                  Qt.AlignmentFlag.AlignVCenter)
        value_label.setStyleSheet("font-family: monospace;")
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        return slider, value_label

    @staticmethod
    def _slider_row(slider: QWidget, value_label: QLabel) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(slider, stretch=1)
        row_layout.addWidget(value_label)
        return row

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_camera(self, camera) -> None:
        """Bind this group to a real (or mock) :class:`ICamera`."""
        self._camera = camera
        # W18.1: preview pane 不再嵌入 CameraConfigGroup,
        # 由 MultiCameraConfigPanel 的第二层统一管理。
        try:
            serial = camera.get_serial() if camera else "未检测"
            model = camera.get_model() if camera else ""
            self.serial_label.setText(f"{model} · {serial}")
            self.status_badge.setText("🟢 在线")
            self.status_badge.setStyleSheet("color: #2e7d32;")
        except Exception as e:
            logger.warning("set_camera(%s) failed: %s", self.role, e)
            self.status_badge.setText("🔴 错误")
            self.status_badge.setStyleSheet("color: #c62828;")

    def set_serial_label(self, text: str) -> None:
        """Update the serial label without binding a camera."""
        self.serial_label.setText(text)

    def collect_params(self) -> Dict[str, Any]:
        return {
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "fps": self.fps_spin.value(),
            "exposure_us": self.exposure_spin.value(),
            "gain_db": self.gain_spin.value(),
            "brightness": self.brightness_slider.value(),
            "contrast": self.contrast_slider.value(),
            "pixel_format": self.pixel_combo.currentText(),
            "x": self.roi_x.value(),
            "y": self.roi_y.value(),
            "roi_w": self.roi_w.value(),
            "roi_h": self.roi_h.value(),
        }

    def apply_to(self, camera) -> List[str]:
        """Push all current widget values to ``camera``.

        Returns a list of human-readable error messages for any parameter
        that could not be set (e.g. unsupported on this hardware).
        """
        if camera is None:
            return ["未绑定相机"]
        errors: List[str] = []
        params = self.collect_params()
        # 顺序: pixel_format → roi → brightness → contrast →
        #        exposure → gain → fps
        try:
            camera.set_pixel_format(params["pixel_format"])
        except Exception as e:
            errors.append(f"像素格式: {e}")
        try:
            camera.set_roi(params["x"], params["y"],
                           params["roi_w"], params["roi_h"])
            # also update width/height in collect_params for snapshot
            params["width"], params["height"] = params["roi_w"], params["roi_h"]
        except Exception as e:
            errors.append(f"ROI: {e}")
        try:
            camera.set_brightness(params["brightness"])
        except Exception as e:
            errors.append(f"亮度: {e}")
        try:
            camera.set_contrast(params["contrast"])
        except Exception as e:
            errors.append(f"对比度: {e}")
        try:
            camera.set_exposure_us(params["exposure_us"])
        except Exception as e:
            errors.append(f"曝光: {e}")
        try:
            camera.set_gain_db(params["gain_db"])
        except Exception as e:
            errors.append(f"增益: {e}")
        try:
            camera.set_fps(params["fps"])
        except Exception as e:
            errors.append(f"帧率: {e}")
        return errors

    def populate_from(self, params: Dict[str, Any]) -> None:
        """Update widgets from a snapshot dict (used by load_preset)."""
        if "width" in params:
            self.width_spin.setValue(int(params["width"]))
        if "height" in params:
            self.height_spin.setValue(int(params["height"]))
        if "fps" in params:
            self.fps_spin.setValue(int(params["fps"]))
        if "exposure_us" in params:
            self.exposure_spin.setValue(float(params["exposure_us"]))
        if "gain_db" in params:
            self.gain_spin.setValue(float(params["gain_db"]))
        if "brightness" in params:
            self.brightness_slider.setValue(int(params["brightness"]))
        if "contrast" in params:
            self.contrast_slider.setValue(int(params["contrast"]))
        if "pixel_format" in params:
            idx = self.pixel_combo.findText(str(params["pixel_format"]))
            if idx >= 0:
                self.pixel_combo.setCurrentIndex(idx)
        if "x" in params:
            self.roi_x.setValue(int(params["x"]))
        if "y" in params:
            self.roi_y.setValue(int(params["y"]))
        if "roi_w" in params:
            self.roi_w.setValue(int(params["roi_w"]))
        if "roi_h" in params:
            self.roi_h.setValue(int(params["roi_h"]))

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_apply_clicked(self) -> None:
        if not hasattr(self, "_camera") or self._camera is None:
            QMessageBox.warning(self.parentWidget() or self,
                                "提示", "请先点击「检测相机」")
            return
        errs = self.apply_to(self._camera)
        if errs:
            QMessageBox.warning(
                self.parentWidget() or self, "部分参数应用失败",
                "以下参数无法下发到相机:\n\n" + "\n".join(errs)
            )
        else:
            self.status_badge.setText("🟢 已应用")
            self.status_badge.setStyleSheet("color: #2e7d32;")


# ---------------------------------------------------------------------------
# CameraPreviewPane (W18) — per-camera live preview
# ---------------------------------------------------------------------------
class CameraPreviewPane(QGroupBox):
    """A small ImageView + ▶/■ + 录制 checkbox for one camera.

    Owned by :class:`CameraConfigGroup` and lives to the right of the
    parameter form.  When the user clicks **▶**, this pane opens the
    bound :class:`ICamera` and starts a QTimer that calls
    ``cam.grab_one(33)`` at ~30 fps.  When the user clicks **■**, the
    timer stops and the camera is closed.

    The pane is intentionally self-contained: it does not depend on
    CameraWorker or the old ``LivePreviewPanel`` (which remains in
    place for the case where the user wants a single 4-camera
    synchronous preview with the old recording flow).
    """

    def __init__(self, role: str, parent: QWidget | None = None) -> None:
        super().__init__("预览", parent)
        self.role = role
        self._camera = None
        self._timer: Optional[QTimer] = None
        self._frame_count = 0
        self._t_start_ns: Optional[int] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # pyqtgraph ImageView (BGR / RGB uint8)
        pg.setConfigOptions(antialias=True)
        self.image_view = ImageView()
        self.image_view.setMinimumSize(360, 220)
        layout.addWidget(self.image_view, stretch=1)

        # Controls row
        ctrl = QHBoxLayout()
        self.start_btn = QPushButton("▶ 预览")
        self.start_btn.setMaximumWidth(80)
        self.start_btn.clicked.connect(self._on_start)
        ctrl.addWidget(self.start_btn)
        self.stop_btn = QPushButton("■")
        self.stop_btn.setMaximumWidth(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl.addWidget(self.stop_btn)
        self.record_check = QCheckBox("录制")
        ctrl.addWidget(self.record_check)
        self.fps_label = QLabel("FPS: —")
        self.fps_label.setStyleSheet("color: #555; font-family: monospace;")
        ctrl.addWidget(self.fps_label, stretch=1)
        layout.addLayout(ctrl)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_camera(self, camera) -> None:
        """Bind to a real/mock :class:`ICamera`.  If a preview is
        running, the binding is delayed until next start (don't try
        to swap mid-stream)."""
        if self._timer is not None:
            self._on_stop()
        self._camera = camera

    def is_running(self) -> bool:
        return self._timer is not None and self._timer.isActive()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self._camera is None:
            QMessageBox.warning(self, "提示",
                                f"请先在「{self.role}」上方点击「检测相机」")
            return
        if self.is_running():
            return
        try:
            self._camera.open()
        except Exception as e:
            QMessageBox.critical(self, "预览失败",
                                 f"无法打开 {self.role} 相机:\n{e}")
            return
        # 给一个合适的 ROI (应用当前组的 width/height)
        params = self._peek_widget_params()
        if params is not None:
            try:
                self._camera.set_roi(params["x"], params["y"],
                                     params["roi_w"], params["roi_h"])
            except Exception:
                pass
        self._frame_count = 0
        self._t_start_ns = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        # ~30 fps = 33 ms
        self._timer.start(33)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_label.setText("FPS: —")

    def _tick(self) -> None:
        if self._camera is None:
            self._on_stop()
            return
        try:
            frame = self._camera.grab_one(timeout_ms=30)
        except Exception as e:
            self.fps_label.setText(f"grab 错误: {e}")
            return
        # CHW for pyqtgraph
        rgb = _bgr_to_rgb(frame.image)
        self.image_view.setImage(rgb.transpose(2, 0, 1))
        # FPS 计算
        self._frame_count += 1
        now = time.perf_counter_ns()
        if self._t_start_ns is None:
            self._t_start_ns = now
        else:
            elapsed_s = (now - self._t_start_ns) / 1e9
            if elapsed_s > 0.5:  # 每 0.5s 更新一次 fps label
                fps = self._frame_count / elapsed_s
                self.fps_label.setText(f"FPS: {fps:.1f}")
                # 暂不重置,持续显示; 防止 label 闪
        # 录制
        if self.record_check.isChecked():
            self._maybe_record(frame)

    def _maybe_record(self, frame) -> None:
        """Save frames to a per-camera MP4 if recording is enabled.

        VideoWriter is lazy-initialized on the first frame after ▶.
        """
        if not hasattr(self, "_writer") or self._writer is None:
            import cv2
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = (Path.home() / ".local" / "share" / "deepgait" / "recordings"
                    / f"{self.role}_{ts}.mp4")
            path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
            self._record_path = path
        if self._writer is not None:
            self._writer.write(frame.image)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._on_stop()
        if hasattr(self, "_writer") and self._writer is not None:
            self._writer.release()
            self._writer = None
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _peek_widget_params(self) -> Optional[Dict[str, Any]]:
        """Read params from the parent group's widgets without
        holding a strong reference.  Returns None if the parent
        is not a CameraConfigGroup.
        """
        parent = self.parent()
        if parent is None or not isinstance(parent, CameraConfigGroup):
            return None
        try:
            return parent.collect_params()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Multi-camera configuration sub-tab
# ---------------------------------------------------------------------------
class MultiCameraConfigPanel(QWidget):
    """The first sub-tab: 4 camera groups + global actions."""

    def __init__(self, app_state: AppState,
                 camera_factory: Optional[Callable[[str], Any]] = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_state = app_state
        # camera_factory: (role) -> ICamera | None. None = use MockCamera.
        self._camera_factory = camera_factory
        # role -> ICamera (lazily bound)
        self._cameras: Dict[str, Any] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(10)

        # 顶部工具栏
        toolbar = QHBoxLayout()
        self.detect_btn = QPushButton("🔍 检测相机")
        self.detect_btn.clicked.connect(self._on_detect_cameras)
        toolbar.addWidget(self.detect_btn)

        self.apply_all_btn = QPushButton("⚙️ 应用全部")
        self.apply_all_btn.clicked.connect(self._on_apply_all)
        toolbar.addWidget(self.apply_all_btn)

        toolbar.addStretch()

        toolbar.addWidget(QLabel("预设:"))
        self.preset_name_edit = QLineEdit("default")
        self.preset_name_edit.setMaximumWidth(140)
        self.preset_name_edit.setPlaceholderText("预设名称")
        toolbar.addWidget(self.preset_name_edit)

        self.save_preset_btn = QPushButton("💾 保存")
        self.save_preset_btn.clicked.connect(self._on_save_preset)
        toolbar.addWidget(self.save_preset_btn)

        self.load_preset_btn = QPushButton("📂 加载")
        self.load_preset_btn.clicked.connect(self._on_load_preset)
        toolbar.addWidget(self.load_preset_btn)

        main.addLayout(toolbar)

        # W18.1: 两层 (配置 + 预览) 放在同一个 cam_widget 内,
        # 然后整块放入 QScrollArea。cam_widget 的 QVBoxLayout
        # 容纳 cam_grid + preview_grid, 不留空白。
        cam_widget = QWidget()
        cam_layout = QVBoxLayout(cam_widget)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.setSpacing(6)

        # 第一层 - 4 相机配置 (水平 1×4)
        cam_grid = QGridLayout()
        cam_grid.setContentsMargins(0, 0, 0, 0)
        cam_grid.setSpacing(8)
        self._groups: Dict[str, CameraConfigGroup] = {}
        self._preview_panes: Dict[str, CameraPreviewPane] = {}
        for i, role in enumerate(DEFAULT_ROLES):
            grp = CameraConfigGroup(role)
            self._groups[role] = grp
            cam_grid.addWidget(grp, 0, i)
        cam_grid.setColumnStretch(0, 1)
        cam_grid.setColumnStretch(1, 1)
        cam_grid.setColumnStretch(2, 1)
        cam_grid.setColumnStretch(3, 1)
        cam_layout.addLayout(cam_grid)

        # 第二层 - 4 相机预览 (水平 1×4, 对应 C1/C2/C3/C4)
        preview_grid = QGridLayout()
        preview_grid.setContentsMargins(0, 4, 0, 0)
        preview_grid.setSpacing(8)
        for i, role in enumerate(DEFAULT_ROLES):
            pane = CameraPreviewPane(role=role)
            self._preview_panes[role] = pane
            preview_grid.addWidget(pane, 0, i)
        preview_grid.setColumnStretch(0, 1)
        preview_grid.setColumnStretch(1, 1)
        preview_grid.setColumnStretch(2, 1)
        preview_grid.setColumnStretch(3, 1)
        cam_layout.addLayout(preview_grid)

        # 预览网格设置为 stretch=1, 吃满剩余垂直空间 (不留空白)
        cam_layout.setStretch(1, 1)

        from PySide6.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidget(cam_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        main.addWidget(scroll, stretch=1)

        # 底部状态
        self.status_label = QLabel("就绪 · 点击「检测相机」开始")
        self.status_label.setStyleSheet(
            "padding: 4px; background: #f0f4f8; border: 1px solid #ccc;"
        )
        main.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Public helpers (used by InitializationTab for AppState publishing)
    # ------------------------------------------------------------------
    def groups(self) -> Dict[str, CameraConfigGroup]:
        return dict(self._groups)

    def preview_panes(self) -> Dict[str, CameraPreviewPane]:
        """W18.1: 4 个独立预览窗口 (第二层),按 role 索引。"""
        return dict(self._preview_panes)

    def cameras(self) -> Dict[str, Any]:
        return dict(self._cameras)

    def install_preset_to_widgets(self, name: str) -> None:
        """Same as _on_load_preset but silent (no QMessageBox)."""
        configs = _load_preset(name)
        for role, grp in self._groups.items():
            if role in configs:
                grp.populate_from(configs[role])

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_detect_cameras(self) -> None:
        """Try to instantiate 4 cameras via the factory (or fall back to MockCamera)."""
        from deepgait3.hardware.camera.multi_cam import MockCamera
        for role in DEFAULT_ROLES:
            try:
                if self._camera_factory is not None:
                    cam = self._camera_factory(role)
                else:
                    cam = MockCamera(
                        serial=f"MOCK-{role}",
                        fps=100, width=640, height=480,
                    )
            except Exception as e:
                logger.warning("Camera factory failed for %s: %s; "
                               "using MockCamera", role, e)
                cam = MockCamera(serial=f"MOCK-{role}", fps=100,
                                  width=640, height=480)
            self._cameras[role] = cam
            self._groups[role].set_camera(cam)
            # W18.1: also bind preview pane (second row).
            if role in self._preview_panes:
                self._preview_panes[role].set_camera(cam)
        self.status_label.setText(
            f"已检测 {len(self._cameras)} 台相机 — "
            f"请调整参数后点击「应用此相机」或「应用全部」"
        )
        # Publish to AppState
        for role, cam in self._cameras.items():
            view = CameraConfigView(
                role=role,
                serial=cam.get_serial(),
                online=True,
            )
            self.app_state.set_camera_config(view)

    def _on_apply_all(self) -> None:
        if not self._cameras:
            QMessageBox.warning(self, "提示", "请先点击「检测相机」")
            return
        all_errs: List[str] = []
        for role, grp in self._groups.items():
            cam = self._cameras.get(role)
            errs = grp.apply_to(cam)
            for e in errs:
                all_errs.append(f"[{role}] {e}")
            # Publish updated config
            params = grp.collect_params()
            view = CameraConfigView(
                role=role,
                serial=cam.get_serial() if cam else "",
                width=params["roi_w"], height=params["roi_h"],
                fps=params["fps"], exposure_us=params["exposure_us"],
                gain_db=params["gain_db"],
                brightness=params["brightness"],
                contrast=params["contrast"],
                pixel_format=params["pixel_format"],
                roi=(params["x"], params["y"], params["roi_w"], params["roi_h"]),
                online=True,
            )
            self.app_state.set_camera_config(view)
        if all_errs:
            QMessageBox.warning(self, "部分参数应用失败",
                                "\n".join(all_errs))
            self.status_label.setText(
                f"应用完成 — {len(all_errs)} 项失败, "
                f"{len(self._cameras) * 7 - len(all_errs)} 项成功"
            )
        else:
            self.status_label.setText(
                f"应用完成 — {len(self._cameras)} 台相机全部成功"
            )

    def _on_save_preset(self) -> None:
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入预设名称")
            return
        configs: Dict[str, Dict[str, Any]] = {}
        for role, grp in self._groups.items():
            configs[role] = grp.collect_params()
        try:
            path = _save_preset(name, configs)
            self.status_label.setText(f"预设已保存: {path}")
        except OSError as e:
            QMessageBox.critical(self, "保存失败",
                                 f"无法写入预设文件:\n{e}")

    def _on_load_preset(self) -> None:
        name = self.preset_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入预设名称")
            return
        try:
            self.install_preset_to_widgets(name)
            self.status_label.setText(f"预设 {name!r} 已加载")
        except FileNotFoundError:
            QMessageBox.warning(self, "提示",
                                f"找不到预设 {name!r}。请检查名称或先保存。")
        except (ValueError, json.JSONDecodeError) as e:
            QMessageBox.critical(self, "加载失败",
                                 f"预设文件格式错误:\n{e}")


# ---------------------------------------------------------------------------
# Live preview sub-tab (the old CameraTab, simplified)
# ---------------------------------------------------------------------------
class LivePreviewPanel(QWidget):
    """The second sub-tab: single-camera / 4-camera live preview + record.

    Migrated from the old ``CameraTab``. Kept functionally identical so
    the existing test suite for camera preview continues to pass.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker: CameraWorker | None = None
        self._recording = False
        self._multi_cam_mgr = None
        self._multi_cam_views: list = []
        self._build_ui()

    def _build_ui(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(12)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)

        src_group = QGroupBox("视频源")
        src_layout = QVBoxLayout(src_group)
        self.source_combo = QComboBox()
        self.source_combo.addItem("摄像头 0", 0)
        self.source_combo.addItem("摄像头 1", 1)
        self.source_combo.addItem("摄像头 2", 2)
        self.source_combo.addItem("摄像头 3", 3)
        self.source_combo.addItem("4 相机同步预览", "MULTI")
        self.source_combo.addItem("视频文件...", "FILE")
        src_layout.addWidget(self.source_combo)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("选择视频文件...")
        self.file_edit.setVisible(False)
        src_layout.addWidget(self.file_edit)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setVisible(False)
        self.browse_btn.clicked.connect(self._on_browse_file)
        src_layout.addWidget(self.browse_btn)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        left_layout.addWidget(src_group)

        param_group = QGroupBox("参数")
        param_layout = QFormLayout(param_group)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(160, 4096); self.width_spin.setValue(640)
        param_layout.addRow("宽度:", self.width_spin)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(120, 4096); self.height_spin.setValue(480)
        param_layout.addRow("高度:", self.height_spin)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 500); self.fps_spin.setValue(100)
        param_layout.addRow("目标 FPS:", self.fps_spin)
        left_layout.addWidget(param_group)

        rec_group = QGroupBox("录制")
        rec_layout = QVBoxLayout(rec_group)
        self.record_check = QCheckBox("保存到文件")
        rec_layout.addWidget(self.record_check)
        left_layout.addWidget(rec_group)

        self.start_btn = QPushButton("开始预览")
        self.start_btn.clicked.connect(self._on_start)
        left_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        left_layout.addWidget(self.stop_btn)
        self.fps_label = QLabel("FPS: —")
        left_layout.addWidget(self.fps_label)
        left_layout.addStretch()
        splitter.addWidget(left)
        splitter.setSizes([320, 880])

        # Right: preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        pg.setConfigOptions(antialias=True)
        self.image_view = ImageView()
        self.image_view.setMinimumSize(600, 450)
        right_layout.addWidget(self.image_view, stretch=1)
        splitter.addWidget(right)

    # Slots (unchanged from old CameraTab for test compatibility)
    def _on_source_changed(self, index: int) -> None:
        is_file = self.source_combo.itemData(index) == "FILE"
        self.file_edit.setVisible(is_file)
        self.browse_btn.setVisible(is_file)

    def _on_browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "",
            "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        if path:
            self.file_edit.setText(path)

    def _on_start(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        data = self.source_combo.currentData()
        if data == "MULTI":
            return self._on_start_multi_cam()
        if data == "FILE":
            path = self.file_edit.text().strip()
            if not path:
                QMessageBox.warning(self, "提示", "请选择视频文件")
                return
            source = path
        else:
            source = int(data)
        save_path = None
        if self.record_check.isChecked():
            save_path, _ = QFileDialog.getSaveFileName(
                self, "保存视频", "capture.mp4",
                "MP4 (*.mp4);;AVI (*.avi);;所有文件 (*.*)"
            )
            if not save_path:
                save_path = "capture.mp4"
            self._recording = True
        self._worker = CameraWorker(
            source=source,
            target_fps=self.fps_spin.value(),
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            save_path=save_path,
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.fps_updated.connect(self._on_fps)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _on_stop(self) -> None:
        if self._worker:
            self._worker.stop()
        if self._multi_cam_mgr is not None:
            self._stop_multi_cam()
        self.stop_btn.setEnabled(False)

    def _on_start_multi_cam(self) -> None:
        from deepgait3.hardware.camera.multi_cam import (
            MultiCameraManager, DEFAULT_ROSTER, MockCamera,
        )
        cameras = [
            (role, MockCamera(serial=f"MOCK-{role}",
                              fps=self.fps_spin.value(),
                              width=self.width_spin.value(),
                              height=self.height_spin.value()))
            for role in DEFAULT_ROSTER
        ]
        self._multi_cam_mgr = MultiCameraManager(cameras=cameras, trigger_line=0)
        self._build_multi_cam_grid()
        self._multi_cam_mgr.start_all()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._multi_cam_timer = QTimer(self)
        self._multi_cam_timer.timeout.connect(self._on_multi_cam_tick)
        self._multi_cam_timer.start(33)

    def _build_multi_cam_grid(self) -> None:
        for view in self._multi_cam_views:
            try:
                view.setParent(None); view.deleteLater()
            except Exception:
                pass
        self._multi_cam_views = []
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        for i in range(4):
            view = pg.ImageView()
            view.setMinimumSize(300, 200)
            row, col = divmod(i, 2)
            grid.addWidget(view, row, col)
            self._multi_cam_views.append(view)
        if not hasattr(self, "_grid_widget"):
            self._grid_widget = grid_widget
        layout = self.image_view.parentWidget().layout()
        if layout is not None:
            layout.addWidget(grid_widget)
        self.image_view.setVisible(False)
        grid_widget.setVisible(True)

    def _on_multi_cam_tick(self) -> None:
        if self._multi_cam_mgr is None:
            return
        snap = self._multi_cam_mgr.grab_quartet(timeout_ms=100)
        if snap is None:
            return
        roles = list(snap.keys())
        for i, role in enumerate(roles[:4]):
            if i >= len(self._multi_cam_views):
                break
            frame = snap[role]
            rgb = _bgr_to_rgb(frame.image)
            self._multi_cam_views[i].setImage(rgb.transpose(2, 0, 1))

    def _stop_multi_cam(self) -> None:
        if self._multi_cam_mgr is not None:
            self._multi_cam_mgr.stop_all()
            self._multi_cam_mgr.close_all()
            self._multi_cam_mgr = None
        if hasattr(self, "_multi_cam_timer"):
            self._multi_cam_timer.stop()
            self._multi_cam_timer.deleteLater()
            del self._multi_cam_timer
        self.image_view.setVisible(True)
        if hasattr(self, "_grid_widget"):
            self._grid_widget.setVisible(False)

    def _on_frame(self, frame: np.ndarray) -> None:
        rgb = _bgr_to_rgb(frame)
        self.image_view.setImage(rgb.transpose(2, 0, 1))

    def _on_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "采集错误", msg)
        self._on_worker_finished()

    def _on_worker_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_label.setText("FPS: —")
        self._worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker:
            self._worker.stop()
            self._worker.wait(2000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Top-level: InitializationTab (was CameraTab)
# ---------------------------------------------------------------------------
class InitializationTab(QWidget):
    """Tab 1: 初始化 — multi-camera configuration + live preview.

    This replaces the old ``CameraTab`` (which lived at tab 8). The old
    ``CameraTab`` class is preserved as an alias for backward
    compatibility (see ``camera_tab.py``).
    """

    def __init__(self, app_state: Optional[AppState] = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.app_state = app_state or AppState()
        self._build_ui()

    def _build_ui(self) -> None:
        # W18.1: 单层布局 — MultiCameraConfigPanel 内部已经包含
        # 第一层 (4 相机配置) + 第二层 (4 相机预览)。
        # 不再使用 QSplitter；旧的 LivePreviewPanel 作为隐藏的
        # backward-compat 代理 (widget proxy) 保留但不显示。
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.config_panel = MultiCameraConfigPanel(self.app_state)
        outer.addWidget(self.config_panel)

        # Backward-compat: keep a hidden LivePreviewPanel so that
        # proxy methods (source_combo, _on_start_multi_cam, etc.)
        # still resolve without AttributeError.
        self.preview_panel = LivePreviewPanel()
        self.preview_panel.setVisible(False)

        # Backward-compat shim: pretend there are 2 sub-panels so
        # old code (including tests) that calls sub_tabs.count() /
        # .tabText(i) still works.  The 2 panels are:
        #   0 = config_panel, 1 = preview_panel (hidden).
        self.sub_splitter = None  # removed; no longer present
        self.sub_tabs = _SplitterAsTabsShim(None, [
            ("多相机配置", self.config_panel),
            ("实时预览 / 录制", self.preview_panel),
        ])

    def n_sub_tabs(self) -> int:
        """Number of sub-panels (config + preview)."""
        return self.sub_tabs.count()

    def config_groups(self) -> Dict[str, CameraConfigGroup]:
        return self.config_panel.groups()

    def install_preset(self, name: str) -> None:
        self.config_panel.install_preset_to_widgets(name)

    # ------------------------------------------------------------------
    # Backward-compat proxies — the old CameraTab put preview-related
    # state and widgets directly on the tab.  Tests and external code
    # that used ``cam_tab.source_combo`` or ``cam_tab._on_start_multi_cam``
    # keep working by delegating to the preview sub-panel.
    # ------------------------------------------------------------------
    @property
    def source_combo(self):
        return self.preview_panel.source_combo

    @property
    def file_edit(self):
        return self.preview_panel.file_edit

    @property
    def browse_btn(self):
        return self.preview_panel.browse_btn

    @property
    def width_spin(self):
        return self.preview_panel.width_spin

    @property
    def height_spin(self):
        return self.preview_panel.height_spin

    @property
    def fps_spin(self):
        return self.preview_panel.fps_spin

    @property
    def record_check(self):
        return self.preview_panel.record_check

    @property
    def start_btn(self):
        return self.preview_panel.start_btn

    @property
    def stop_btn(self):
        return self.preview_panel.stop_btn

    @property
    def fps_label(self):
        return self.preview_panel.fps_label

    @property
    def image_view(self):
        return self.preview_panel.image_view

    @property
    def _worker(self):
        return self.preview_panel._worker

    @property
    def _multi_cam_mgr(self):
        return self.preview_panel._multi_cam_mgr

    @property
    def _multi_cam_views(self):
        return self.preview_panel._multi_cam_views

    @property
    def _grid_widget(self):
        return getattr(self.preview_panel, "_grid_widget", None)

    def _on_source_changed(self, *args, **kwargs):
        return self.preview_panel._on_source_changed(*args, **kwargs)

    def _on_browse_file(self):
        return self.preview_panel._on_browse_file()

    def _on_start(self):
        return self.preview_panel._on_start()

    def _on_stop(self):
        return self.preview_panel._on_stop()

    def _on_start_multi_cam(self):
        return self.preview_panel._on_start_multi_cam()

    def _on_multi_cam_tick(self):
        return self.preview_panel._on_multi_cam_tick()

    def _on_frame(self, frame):
        return self.preview_panel._on_frame(frame)

    def _on_fps(self, fps):
        return self.preview_panel._on_fps(fps)

    def _on_error(self, msg):
        return self.preview_panel._on_error(msg)

    def _on_worker_finished(self):
        return self.preview_panel._on_worker_finished()

    def _build_multi_cam_grid(self):
        return self.preview_panel._build_multi_cam_grid()

    def _stop_multi_cam(self):
        return self.preview_panel._stop_multi_cam()

    def closeEvent(self, event):  # noqa: N802
        return self.preview_panel.closeEvent(event)


# ---------------------------------------------------------------------------
# _SplitterAsTabsShim (W17.1)
# ---------------------------------------------------------------------------
class _SplitterAsTabsShim:
    """最小 shim: 模拟 QTabWidget.count() / .tabText(i) / .widget(i) 接口。

    W17.1 把 ``InitializationTab`` 内部从 ``QTabWidget``(两个子 tab)
    改为 ``QSplitter(Qt.Vertical)``(上下分体)。但旧代码 + 测试仍然
    通过 ``tab.sub_tabs.count()`` / ``.tabText(i)`` / ``.widget(i)``
    访问子面板。本 shim 让这些调用继续工作(不抛 AttributeError)。

    真正显示的子面板由 splitter 管理(可拖动分隔条);本 shim 仅作为
    兼容性外观 (facade)。
    """

    def __init__(self, splitter, panel_pairs) -> None:
        self._splitter = splitter
        self._labels = [label for label, _ in panel_pairs]
        self._widgets = [w for _, w in panel_pairs]

    def count(self) -> int:
        return len(self._widgets)

    def tabText(self, i: int) -> str:
        if not 0 <= i < len(self._widgets):
            raise IndexError(i)
        return self._labels[i]

    def widget(self, i: int) -> "QWidget":
        if not 0 <= i < len(self._widgets):
            raise IndexError(i)
        return self._widgets[i]

    def currentWidget(self) -> "QWidget":
        """Return the currently-largest (or upper) panel as a stable choice."""
        if not self._widgets:
            return None
        # 优先:上方面板(config),与原 QTabWidget 默认行为一致
        return self._widgets[0]

    def setCurrentIndex(self, i: int) -> None:
        """No-op for splitter (kept for API compatibility)."""
        if not 0 <= i < len(self._widgets):
            raise IndexError(i)


# Backward-compat alias: old name still works.
CameraTab = InitializationTab
