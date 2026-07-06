"""Tab 2: 步态分析 (Gait Analysis) — FTIR footprint → gait metrics.

Supports two modes:

* **视频文件模式**: load one or more videos, fill animal-ID table,
  auto-detect runs, compute CatWalk-equivalent metrics.
* **实时 C1 模式**: connect to the bottom (FTIR) camera C1, enter
  animal ID, start live monitoring.  First footprint → run begins;
  animal leaves → run ends automatically.

Layout
------

* Top toolbar (mode selector + load / connect / start / stop)
* Left panel — real-time parameter display (QFormLayout)
* Right panel — footprint accumulation image (GraphicsLayoutWidget + ImageItem)
* Bottom — 4-paw stance/swing sequence chart (pyqtgraph PlotWidget)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotDataItem

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QScrollArea,
)

from deepgait3.core._legacy import footprint_v2 as fp2
from deepgait3.core._legacy.background_model import RollingMedianBackground
from deepgait3.core._legacy.run_detector import RunDetector, RunResult
from deepgait3.core._legacy import gait_ftir, gait_algorithms as ga


logger = logging.getLogger(__name__)

# Paw colors — CatWalk-style palette (matches reference sequence.svg)
PAW_COLORS: Dict[str, str] = {
    "RF": "#F44336",   # red
    "LF": "#FFC107",   # yellow
    "RH": "#E91E63",   # pink
    "LH": "#2196F3",   # blue
}
PAW_LABELS: Dict[str, str] = {
    "RF": "右前", "LF": "左前", "RH": "右后", "LH": "左后",
}
PAW_ORDER = ["RF", "LF", "RH", "LH"]
# Sequence-chart row order (top→bottom): hind limbs first, then fore.
# Y positions: RH=3.5, LH=2.5, RF=1.5, LF=0.5
SEQ_ROW_ORDER = ["RH", "LH", "RF", "LF"]


# ---------------------------------------------------------------------------
# 4-Paw stance/swing sequence chart (CatWalk-style)
# ---------------------------------------------------------------------------
class StanceSwingSequenceChart(pg.PlotWidget):
    """CatWalk-style 4-row stance/swing chart.

    Each stance period is drawn as a **solid colored rectangular block**
    with row height ``ROW_HEIGHT``.  Swing periods show as gaps.
    Row order top→bottom: RH, LH, RF, LF (hind limbs on top).

    The X-axis is **time (seconds)** = frame_idx / fps, so this chart
    shares a single time base with the video preview and footprint
    accumulation panels.

    The X axis is **time (seconds)**.  X_max = total run duration
    (n_frames / fps), so the chart always spans the full run length.
    """
    ROW_HEIGHT = 0.8             # vertical thickness of each stance block
    ROW_Y = {"RH": 3.5, "LH": 2.5, "RF": 1.5, "LF": 0.5}
    current_frame_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fps: float = 100.0   # updated by GaitTab.set_fps()
        self.setBackground("#FAFAFA")
        self.showGrid(x=True, y=False, alpha=0.2)
        self.setLabel("bottom", "Time (s)")
        self.getAxis("left").setTicks([
            [(self.ROW_Y[p], p) for p in SEQ_ROW_ORDER],
        ])
        self.setYRange(0.0, 4.0)
        # Default X range: 5s; will be auto-set to n_frames/fps on update.
        self.setXRange(0, 5.0)
        self._blocks: list = []           # active stance blocks
        self._n_frames = 0
        self._current_frame: int = 0

        # Vertical cursor line for current time
        self._cursor = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen("#000000", width=1, style=Qt.PenStyle.DashLine),
        )
        self.addItem(self._cursor)

        # Click to seek
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_fps(self, fps: float) -> None:
        """Update fps (X axis unit: frames → seconds)."""
        if fps > 0:
            self._fps = float(fps)

    def update_stance(self, in_stance: Dict[str, np.ndarray]) -> None:
        """Refresh the chart: draw one solid block per stance run."""
        # Remove old blocks
        for blk in self._blocks:
            self.removeItem(blk)
        self._blocks.clear()
        self._n_frames = 0
        for paw in SEQ_ROW_ORDER:
            arr = in_stance.get(paw)
            if arr is None or len(arr) == 0:
                continue
            self._n_frames = max(self._n_frames, len(arr))
            color = pg.mkColor(PAW_COLORS[paw])
            y_center = self.ROW_Y[paw]
            j, n = 0, len(arr)
            while j < n:
                if arr[j]:
                    start = j
                    while j < n and arr[j]:
                        j += 1
                    end = j - 1
                    self._add_block(start, end, y_center, color)
                else:
                    j += 1
        # X_max = total run duration (n_frames / fps).  The X range
        # always matches the source video length.
        if self._n_frames > 0 and self._fps > 0:
            total_s = self._n_frames / self._fps
            self.setXRange(0, max(total_s, 0.5), padding=0.02)
        self._update_cursor()

    def _add_block(self, start_frame: int, end_frame: int,
                   y_center: float, color) -> None:
        """Add one solid rectangular stance block (time axis)."""
        y_top = y_center + self.ROW_HEIGHT / 2
        y_bot = y_center - self.ROW_HEIGHT / 2
        # Convert frame indices to time (seconds) for the X axis.
        s = start_frame / self._fps
        e = end_frame / self._fps
        x_poly = [s, e, e, s, s]
        y_poly = [y_bot, y_bot, y_top, y_top, y_bot]
        fill = pg.PlotCurveItem(
            x=np.array(x_poly, dtype=float),
            y=np.array(y_poly, dtype=float),
            fillLevel=y_bot,
            pen=pg.mkPen(color, width=1),
            brush=pg.mkBrush(color),
        )
        self.addItem(fill)
        self._blocks.append(fill)

    def set_current_frame(self, frame_idx: int) -> None:
        """Move the vertical cursor to the given frame (time = frame/fps)."""
        self._current_frame = max(0, min(frame_idx,
                                         max(self._n_frames - 1, 0)))
        self._update_cursor()

    def _update_cursor(self) -> None:
        # X axis is time in seconds: position = frame / fps
        t = self._current_frame / self._fps if self._fps > 0 else 0
        self._cursor.setPos(t)

    def _on_click(self, event) -> None:
        """Click on chart → seek to that time (convert to frame)."""
        pos = event.scenePos()
        vb = self.plotItem.vb
        if vb is None:
            return
        mapped = vb.mapSceneToView(pos)
        t = mapped.x()
        frame_idx = int(t * self._fps) if self._fps > 0 else 0
        if 0 <= frame_idx < self._n_frames:
            self.set_current_frame(frame_idx)
            self.current_frame_changed.emit(frame_idx)

    def reset(self) -> None:
        for blk in self._blocks:
            self.removeItem(blk)
        self._blocks.clear()
        self._n_frames = 0
        self._current_frame = 0
        self._cursor.setPos(0)
        self.setXRange(0, 5.0)


# ---------------------------------------------------------------------------
# Real-time parameter panel
# ---------------------------------------------------------------------------
class RealTimeParamPanel(QWidget):
    """Left panel: read-only metrics in QFormLayout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: Dict[str, QLabel] = {}
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Run-level
        for key, label in (
            ("run_duration_s", "Run Duration (s)"),
            ("run_avg_speed_mm_s", "Avg Speed (mm/s)"),
            ("run_max_variation_pct", "Max Variation (%)"),
            ("n_steps", "Steps"),
            ("cadence_steps_per_min", "Cadence (steps/min)"),
            ("body_speed_mean_mm_s", "Body Speed Mean (mm/s)"),
            ("body_speed_variation_mm_s", "Body Speed Var (mm/s)"),
        ):
            lbl = QLabel("—")
            lbl.setStyleSheet("font-family: monospace;")
            self._labels[key] = lbl
            layout.addRow(label + ":", lbl)

        # Per-paw (shown in a QTabWidget)
        self.paw_tabs = QTabWidget()
        for paw in PAW_ORDER:
            paw_panel = QWidget()
            paw_layout = QFormLayout(paw_panel)
            paw_labels = {}
            for key, label in (
                ("stand_s", "Stand (s)"),
                ("swing_s", "Swing (s)"),
                ("swing_speed_mm_s", "Swing Speed (mm/s)"),
                ("step_cycle_s", "Step Cycle (s)"),
                ("duty_cycle_pct", "Duty Cycle (%)"),
                ("single_stance_s", "Single Stance (s)"),
                ("initial_dual_stance_s", "Init Dual Stance (s)"),
                ("terminal_dual_stance_s", "Term Dual Stance (s)"),
                ("max_contact_at_pct", "Max Contact at (%)"),
                ("max_intensity_at_pct", "Max Intensity at (%)"),
                ("stand_index", "Stand Index"),
            ):
                lbl = QLabel("—")
                lbl.setStyleSheet("font-family: monospace;")
                paw_labels[key] = lbl
                paw_layout.addRow(label + ":", lbl)
            self.paw_tabs.addTab(paw_panel, f"{PAW_LABELS[paw]} ({paw})")
            setattr(self, f"_paw_{paw}_labels", paw_labels)
        layout.addRow(self.paw_tabs)

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Refresh all labels from a flat metrics dict."""
        for key, lbl in self._labels.items():
            if key in metrics:
                lbl.setText(str(metrics[key]))
        for paw in PAW_ORDER:
            pw_labels = getattr(self, f"_paw_{paw}_labels")
            for key, lbl in pw_labels.items():
                full_key = f"{paw}_{key}"
                if full_key in metrics:
                    lbl.setText(str(metrics[full_key]))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Layer 4 (Inter-Limb Coordination) was removed in W20.  The cross-
# correlation X-axis (lag frames) is in a different domain from the
# time-synchronised panels, so it is no longer displayed.
# ---------------------------------------------------------------------------
# GaitFTIRWorker — background video/live processing
# ---------------------------------------------------------------------------
class GaitFTIRWorker(QThread):
    """Process video frames in background; real-time signals to GaitTab UI."""
    frame_processed = Signal(object, object, int)  # sequence, frame_bgr, idx
    run_completed = Signal(object, object, object, object, object, object)  # result, in_stance, intensity, centroids_x, area_px, pressure
    error_msg = Signal(str)
    progress = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_path: str = ""
        self.animal_id: str = ""
        self._abort: bool = False
        self.detected_fps: float = 100.0   # set from video metadata
        # Tunable thresholds (set before calling run_video).
        self.body_threshold: int = 30
        self.paw_threshold: int = 12
        self.min_area_px: int = 20

    def run_video(self, video_path: str, animal_id: str) -> None:
        self.video_path = video_path
        self.animal_id = animal_id
        self._abort = False
        self.start()

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        """Process video frames in background thread."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_msg.emit(f"无法打开视频: {self.video_path}")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 100.0
        self.detected_fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress.emit(f"开始处理 {self.video_path} ({total_frames} 帧, {fps:.0f} fps)")

        bg = RollingMedianBackground(warmup_frames=30)
        detector = RunDetector(fps=fps)
        detector.animal_id = self.animal_id

        # Data collection across frames.
        all_in_stance: Dict[str, List[int]] = {p: [] for p in PAW_ORDER}
        all_intensity: Dict[str, List[float]] = {p: [] for p in PAW_ORDER}
        all_centroids_x: Dict[str, List[float]] = {p: [] for p in PAW_ORDER}
        all_area_px: Dict[str, List[float]] = {p: [] for p in PAW_ORDER}
        # Per-paw pressure tracking: (frame, x, y, area, intensity) per detection.
        all_pressure: Dict[str, List[Dict]] = {p: [] for p in PAW_ORDER}
        # Accumulated centroids for dynamic body axis estimation.
        accum_centroids: List[Tuple[float, float]] = []
        frame_idx = 0

        while not self._abort:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_bgr is None:
                continue
            frame_idx += 1

            try:
                bg.update(frame_bgr)
                h, w = frame_bgr.shape[:2]

                # Dynamic body axis from accumulated paw centroids.
                if len(accum_centroids) >= 6:
                    body_axis = fp2.estimate_body_axis(accum_centroids[-30:], w, h)
                else:
                    body_axis = ((int(w * 0.2), int(h * 0.5)),
                                 (int(w * 0.8), int(h * 0.5)))

                seq = fp2.analyze_frame_v2(
                    frame_bgr, background=bg,
                    body_axis=body_axis,
                    use_red_background=True,
                    body_threshold=self.body_threshold,
                    paw_threshold=self.paw_threshold,
                    min_area_px=self.min_area_px,
                )
            except Exception as e:
                # Silent skip for individual frame errors
                continue

            # Collect centroid for dynamic body axis.
            for fm in seq.feet.values():
                if fm.is_in_stance:
                    accum_centroids.append(fm.centroid)
            # Keep sliding window of recent centroids.
            if len(accum_centroids) > 60:
                accum_centroids = accum_centroids[-60:]

            # Store stance / intensity / centroids / area data.
            for paw in PAW_ORDER:
                foot = seq.feet.get(paw)
                stance = int(foot is not None and foot.is_in_stance)
                all_in_stance[paw].append(stance)
                all_intensity[paw].append(
                    float(foot.intensity_max) if foot else 0.0
                )
                all_centroids_x[paw].append(
                    float(foot.centroid[0]) if foot else 0.0
                )
                all_area_px[paw].append(
                    float(foot.area_px) if foot else 0.0
                )
                if foot is not None and foot.is_in_stance:
                    all_pressure[paw].append({
                        "frame": frame_idx,
                        "x": float(foot.centroid[0]),
                        "y": float(foot.centroid[1]),
                        "area": float(foot.area_px),
                        "intensity": float(foot.intensity_max),
                    })

            self.frame_processed.emit(seq, frame_bgr, frame_idx - 1)

            result = detector.process_frame(seq)
            if result is not None:
                self.run_completed.emit(
                    result, all_in_stance, all_intensity,
                    all_centroids_x, all_area_px, all_pressure,
                )
                all_in_stance = {p: [] for p in PAW_ORDER}
                all_intensity = {p: [] for p in PAW_ORDER}
                all_centroids_x = {p: [] for p in PAW_ORDER}
                all_area_px = {p: [] for p in PAW_ORDER}
                all_pressure = {p: [] for p in PAW_ORDER}

            QThread.msleep(1)

        result = detector.finish()
        if result is not None:
            self.run_completed.emit(
                result, all_in_stance, all_intensity,
                all_centroids_x, all_area_px, all_pressure,
            )
        cap.release()
        self.progress.emit(f"处理完成 ({frame_idx} 帧)")


# ---------------------------------------------------------------------------
# Video batch table
# ---------------------------------------------------------------------------
class VideoBatchTable(QTableWidget):
    """Table for batch video + animal ID entries."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["动物编号", "视频文件", "状态", ""])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

    def add_row(self, animal_id: str = "", video_path: str = "") -> None:
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(animal_id))
        self.setItem(row, 1, QTableWidgetItem(video_path))
        self.setItem(row, 2, QTableWidgetItem("待分析"))
        remove_btn = QPushButton("✕")
        remove_btn.setMaximumWidth(30)
        remove_btn.clicked.connect(lambda: self._remove_row(row))
        self.setCellWidget(row, 3, remove_btn)

    def _remove_row(self, row: int) -> None:
        self.removeRow(row)

    def get_entries(self) -> List[Dict[str, str]]:
        entries = []
        for r in range(self.rowCount()):
            aid = self.item(r, 0).text().strip() if self.item(r, 0) else ""
            path = self.item(r, 1).text().strip() if self.item(r, 1) else ""
            if aid and path:
                entries.append({"animal_id": aid, "video_path": path})
        return entries


# ---------------------------------------------------------------------------
# Top-level GaitTab
# ---------------------------------------------------------------------------
class GaitTab(QWidget):
    """步态分析 tab — FTIR footprint → gait metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bg_model: Optional[RollingMedianBackground] = None
        self._run_detector: Optional[RunDetector] = None
        self._live_timer: Optional[QTimer] = None
        self._camera = None
        self._accumulated_image: Optional[np.ndarray] = None
        self._fps: float = 100.0
        self._is_running: bool = False
        self._is_paused: bool = False
        self._worker: Optional[GaitFTIRWorker] = None
        self._all_in_stance: Dict[str, List[int]] = {}
        self.param_panel = RealTimeParamPanel()  # hidden, used by _on_run_completed
        self._build_ui()

    def _build_ui(self) -> None:
        # Wrap the entire GaitTab in a QScrollArea so the 4 panels
        # (each 5:1 aspect) can be larger than the tab window without
        # causing the tab to render blank.  setWidgetResizable=True
        # ensures the inner container fills the scroll viewport
        # horizontally so the panels expand to full width.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        container = QWidget()
        main = QVBoxLayout(container)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)
        scroll.setWidget(container)
        outer.addWidget(scroll)
        self._scroll_area = scroll  # keep reference

        # ================================================================
        # Layer 1: 控制面板 (data loading + frame nav + start/pause/stop)
        # ================================================================
        layer1 = QGroupBox("① 控制面板")
        l1 = QVBoxLayout(layer1)
        l1.setSpacing(4)

        # -- toolbar (mode + load) --
        tbar = QHBoxLayout()
        tbar.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["视频文件", "实时 C1"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        tbar.addWidget(self.mode_combo)
        tbar.addStretch()
        l1.addLayout(tbar)

        # -- video mode: batch table + load buttons --
        self.video_controls = QWidget()
        vc_l = QVBoxLayout(self.video_controls)
        vc_l.setContentsMargins(0, 0, 0, 0)
        vc_l.setSpacing(4)
        self.batch_table = VideoBatchTable()
        self.batch_table.setMaximumHeight(140)
        vc_l.addWidget(self.batch_table)
        vb_row = QHBoxLayout()
        self.load_video_btn = QPushButton("加载视频")
        self.load_video_btn.clicked.connect(self._on_load_video)
        vb_row.addWidget(self.load_video_btn)
        self.load_folder_btn = QPushButton("加载文件夹")
        self.load_folder_btn.clicked.connect(self._on_load_folder)
        vb_row.addWidget(self.load_folder_btn)
        self.add_empty_btn = QPushButton("添加空行")
        self.add_empty_btn.clicked.connect(lambda: self.batch_table.add_row())
        vb_row.addWidget(self.add_empty_btn)
        vb_row.addStretch()
        vc_l.addLayout(vb_row)
        l1.addWidget(self.video_controls)

        # -- live C1 mode --
        self.live_controls = QWidget()
        lc_l = QHBoxLayout(self.live_controls)
        lc_l.setContentsMargins(0, 0, 0, 0)
        lc_l.addWidget(QLabel("动物编号:"))
        self.animal_id_edit = QLineEdit()
        self.animal_id_edit.setPlaceholderText("例: C57-001")
        self.animal_id_edit.setMaximumWidth(140)
        self.animal_id_edit.textChanged.connect(self._update_button_state)
        lc_l.addWidget(self.animal_id_edit)
        self.connect_c1_btn = QPushButton("连接 C1")
        self.connect_c1_btn.clicked.connect(self._on_connect_c1)
        lc_l.addWidget(self.connect_c1_btn)
        self.c1_status = QLabel("未连接")
        self.c1_status.setStyleSheet("color: #888;")
        lc_l.addWidget(self.c1_status)
        lc_l.addStretch()
        l1.addWidget(self.live_controls)
        self.live_controls.hide()

        # -- frame info --
        info_row = QHBoxLayout()
        self.frame_info_label = QLabel("")
        self.frame_info_label.setStyleSheet(
            "color: #aaa; font-family: monospace; font-size: 11px;"
        )
        info_row.addWidget(self.frame_info_label)
        info_row.addStretch()
        l1.addLayout(info_row)

        # -- frame slider (VisualGaitLab-style frame navigation) --
        slider_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_frame_slider_moved)
        slider_row.addWidget(self.frame_slider)
        self.frame_label = QLabel("Frame: —")
        self.frame_label.setMinimumWidth(100)
        self.frame_label.setStyleSheet("font-family: monospace; color: #555;")
        slider_row.addWidget(self.frame_label)
        l1.addLayout(slider_row)

        # -- threshold controls (collapsible) --
        self.threshold_group = QGroupBox("⚙ 阈值调节")
        self.threshold_group.setCheckable(True)
        self.threshold_group.setChecked(False)
        th_l = QHBoxLayout(self.threshold_group)
        th_l.setSpacing(8)

        th_l.addWidget(QLabel("身体:"))
        self.body_threshold_spin = QSpinBox()
        self.body_threshold_spin.setRange(10, 100)
        self.body_threshold_spin.setValue(30)
        self.body_threshold_spin.setToolTip("红色通道下降阈值 (bg_R - frame_R)\n值越小，身体区域越大")
        th_l.addWidget(self.body_threshold_spin)

        th_l.addWidget(QLabel("爪印:"))
        self.paw_threshold_spin = QSpinBox()
        self.paw_threshold_spin.setRange(5, 50)
        self.paw_threshold_spin.setValue(12)
        self.paw_threshold_spin.setToolTip("绿色通道绝对值阈值 (G > N)\n背景G≈13，值越小爪印越多")
        th_l.addWidget(self.paw_threshold_spin)

        th_l.addWidget(QLabel("最小面积:"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(5, 200)
        self.min_area_spin.setValue(20)
        self.min_area_spin.setToolTip("最小连通区域 (px)\n过滤噪点")
        th_l.addWidget(self.min_area_spin)

        th_l.addStretch()
        l1.addWidget(self.threshold_group)

        # -- control buttons (start / pause / stop) --
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始分析")
        self.start_btn.setStyleSheet(
            "QPushButton:enabled { background-color: #2e7d32; color: white; }"
        )
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        ctrl_row.addWidget(self.start_btn)
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        ctrl_row.addWidget(self.pause_btn)
        self.stop_btn = QPushButton("⏹ 结束")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addStretch()
        l1.addLayout(ctrl_row)

        # -- status bar --
        self.status_label = QLabel("就绪 — 请加载视频或连接 C1 相机")
        self.status_label.setStyleSheet("color: #555; font-style: italic;")
        l1.addWidget(self.status_label)
        main.addWidget(layer1, stretch=0)

        # ================================================================
        # Layer 2: 视频预览 (camera view, spatial X axis 0-1920 px)
        # ================================================================
        layer2 = QGroupBox("② 视频预览")
        l2 = QVBoxLayout(layer2)
        l2.setContentsMargins(4, 4, 4, 4)
        self.video_preview = QLabel("视频预览区")
        self.video_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_preview.setStyleSheet(
            "background: #111; color: #555; border: 1px solid #444; "
            "font-size: 14px;"
        )
        # Width expands to fill the layer; height = width / 5 (FTIR
        # walkway native 1920×384).  See _apply_panel_aspect().
        self.video_preview.setMinimumHeight(60)
        self.video_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        l2.addWidget(self.video_preview)
        # (No time-cursor overlay — the X axis of video_preview is the
        # 5:1 walk-way camera view, not a time axis.)
        main.addWidget(layer2, stretch=0)

        # ================================================================
        # Layer 3: 脚印累积图 (spatial X axis 0-1920 px on the walkway)
        # ================================================================
        layer3 = QGroupBox("③ 脚印累积图")
        l3 = QVBoxLayout(layer3)
        l3.setContentsMargins(4, 4, 4, 4)
        # Use a plain QLabel + QPixmap.  Width-responsive / 5:1-height.
        self.footprint_view = QLabel()
        self.footprint_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.footprint_view.setStyleSheet(
            "background: #000000; border: 1px solid #444;"
        )
        self.footprint_view.setMinimumHeight(60)
        self.footprint_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.footprint_view.setScaledContents(False)
        l3.addWidget(self.footprint_view)
        # (No time-cursor overlay — footprint X axis is the spatial
        #  position 0-1920 px on the walkway, not a time axis.)
        main.addWidget(layer3, stretch=1)

        # ================================================================
        # Layer 4: 四肢序列图 (time X axis 0..n_frames/fps seconds)
        # ================================================================
        layer4 = QGroupBox("④ 四肢序列图")
        l4 = QVBoxLayout(layer4)
        l4.setContentsMargins(4, 4, 4, 4)
        self.seq_chart = StanceSwingSequenceChart()
        self.seq_chart.current_frame_changed.connect(self._on_chart_frame_changed)
        # Responsive — same width-fills / 5:1-height pattern as preview.
        self.seq_chart.setMinimumHeight(60)
        self.seq_chart.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        l4.addWidget(self.seq_chart)
        main.addWidget(layer4, stretch=1)

        # ================================================================
        # Layer 4 (Inter-Limb Coordination) was removed in W20 — the
        # X-axis (lag frames) is in a different domain from the
        # time-synchronised panels, so it is no longer displayed.
        # ================================================================

        # ================================================================
        # Run Summary (CatWalk-style, auto-hides after 5s)
        # ================================================================
        self.run_summary = QGroupBox("Run 汇总")
        self.run_summary.setStyleSheet(
            "QGroupBox { border: 2px solid #4caf50; border-radius: 4px; "
            "margin-top: 6px; padding-top: 14px; }"
        )
        self.run_summary.hide()
        rs_layout = QHBoxLayout(self.run_summary)
        self.run_summary_labels: Dict[str, QLabel] = {}
        for key, label in (
            ("run_duration_s", "Duration"),
            ("n_steps", "Steps"),
            ("LH_stride_length_cm", "LH Stride"),
            ("body_speed_cm_s", "Speed"),
        ):
            row = QVBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size: 10px; color: #888;")
            val = QLabel("—")
            val.setStyleSheet("font-size: 16px; font-weight: bold; font-family: monospace; color: #2e7d32;")
            row.addWidget(lbl)
            row.addWidget(val)
            rs_layout.addLayout(row)
            self.run_summary_labels[key] = val
        main.addWidget(self.run_summary)
        self._summary_timer = QTimer(self)
        self._summary_timer.setSingleShot(True)
        self._summary_timer.timeout.connect(lambda: self.run_summary.hide())

    # ------------------------------------------------------------------
    # Responsive sizing: keep 4 panels at 5:1 aspect ratio
    # ------------------------------------------------------------------
    _PANEL_ASPECT = 5.0  # width / height (1920 / 384)

    def resizeEvent(self, event) -> None:
        """Re-size the 4 panels to keep 5:1 aspect ratio on window resize."""
        super().resizeEvent(event)
        self._apply_panel_aspect()

    def _apply_panel_aspect(self) -> None:
        """Set height = width / 5 on each of the 4 panels.

        The outer QScrollArea handles vertical overflow — we never
        compress the panel to fit the visible area, so the 5:1
        aspect ratio is always preserved.
        """
        for panel in (self.video_preview, self.seq_chart,
                      self.footprint_view):
            w = panel.width()
            if w > 0:
                target_h = max(int(w / self._PANEL_ASPECT), 60)
                if abs(panel.height() - target_h) > 2:
                    panel.setFixedHeight(target_h)

    def showEvent(self, event) -> None:
        """Apply aspect ratio after first show (when widgets have real size)."""
        super().showEvent(event)
        self._apply_panel_aspect()

    # ------------------------------------------------------------------
    # Mode switch
    # ------------------------------------------------------------------
    def _on_mode_changed(self, index: int) -> None:
        is_video = (self.mode_combo.currentText() == "视频文件")
        self.video_controls.setVisible(is_video)
        self.live_controls.setVisible(not is_video)
        self._update_button_state()

    # ------------------------------------------------------------------
    # Button state logic: 有视频 + 有动物编号 → 才能开始
    # ------------------------------------------------------------------
    def _update_button_state(self) -> None:
        if self._is_running and self._is_paused:
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            return
        ready = self._data_is_ready()
        running = self._is_running and not self._is_paused
        self.start_btn.setEnabled(ready and not running)
        self.pause_btn.setEnabled(running)
        self.stop_btn.setEnabled(self._is_running)

    def _data_is_ready(self) -> bool:
        if self.mode_combo.currentText() == "视频文件":
            # 只要至少有一行填了视频路径即可
            for r in range(self.batch_table.rowCount()):
                path_item = self.batch_table.item(r, 1)
                if path_item and path_item.text().strip():
                    return True
            return False
        else:
            return bool(self.animal_id_edit.text().strip())

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    def _on_load_video(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "",
            "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        for p in paths:
            self.batch_table.add_row(video_path=p)
        self._update_button_state()

    def _on_load_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择视频文件夹")
        if not folder:
            return
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for p in Path(folder).glob(ext):
                self.batch_table.add_row(video_path=str(p))
        self._update_button_state()

    # ------------------------------------------------------------------
    # Live C1
    # ------------------------------------------------------------------
    def _on_connect_c1(self) -> None:
        """Reach into the InitializationTab and grab the C1 camera."""
        try:
            main_win = self.window()
            init_tab = main_win.tab_by_name("initialization")
            cam_dict = init_tab.config_panel.cameras()
            if "bottom" not in cam_dict:
                QMessageBox.warning(self, "提示",
                    "C1 (底部) 相机未检测。请先在「初始化」tab 点击「检测相机」。")
                return
            self._camera = cam_dict["bottom"]
            self._fps = float(
                init_tab.config_panel.groups()["bottom"].fps_spin.value()
            )
            self.c1_status.setText("🟢 C1 已连接")
            self.c1_status.setStyleSheet("color: #2e7d32;")
        except Exception as e:
            logger.warning("connect C1 failed: %s", e)
            QMessageBox.critical(self, "连接失败", f"无法连接 C1 相机:\n{e}")

    # ------------------------------------------------------------------
    # Start / Pause / Stop
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self._is_running and not self._is_paused:
            return
        if self._is_paused:
            self._is_paused = False
            self.status_label.setText("分析运行中…")
        else:
            if not self._data_is_ready():
                QMessageBox.warning(self, "提示", "请先加载数据并填写动物编号。")
                return
            mode = self.mode_combo.currentText()
            if mode == "视频文件":
                entries = []
                for r in range(self.batch_table.rowCount()):
                    aid = self.batch_table.item(r, 0).text().strip() if self.batch_table.item(r, 0) else ""
                    path = self.batch_table.item(r, 1).text().strip() if self.batch_table.item(r, 1) else ""
                    if path:
                        entries.append({"animal_id": aid or "Unknown", "video_path": path})
                if not entries:
                    QMessageBox.warning(self, "提示", "请先加载视频文件。")
                    return
                self._start_video_batch(entries)
            else:
                animal_id = self.animal_id_edit.text().strip()
                if self._camera is None:
                    self._on_connect_c1()
                self._start_live(animal_id)
        self._update_button_state()

    def _on_pause(self) -> None:
        if not self._is_running or self._is_paused:
            return
        self._is_paused = True
        self.status_label.setText("已暂停 — 点击 ▶ 继续")
        self._update_button_state()

    def _on_stop(self) -> None:
        self._is_running = False
        self._is_paused = False
        if self._worker is not None:
            self._worker.abort()
            self._worker.wait(2000)
            self._worker = None
        if self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None
        self.status_label.setText("已停止")
        self._update_button_state()

    # ------------------------------------------------------------------
    # Frame slider + chart seek (VisualGaitLab-style)
    # ------------------------------------------------------------------
    def _on_frame_slider_moved(self, value: int) -> None:
        """Slider changed → update chart cursor + label."""
        self.seq_chart.set_current_frame(value)
        self.frame_label.setText(f"Frame: {value}")
        self._update_frame_preview()

    def _on_chart_frame_changed(self, frame_idx: int) -> None:
        """Click on chart → sync slider + label."""
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText(f"Frame: {frame_idx}")
        self._update_frame_preview()

    def _update_frame_preview(self) -> None:
        """Show current frame image in the video_preview QLabel (slider-driven)."""
        if not hasattr(self, "_current_frame_bgr") or self._current_frame_bgr is None:
            return
        frame_idx = self.frame_slider.value()
        self._show_frame_preview(self._current_frame_bgr, frame_idx)

    # Fixed preview target size — native video frame size (1920×384 for
    # Width/height of the live video preview QLabel, derived from the
    # panel's current size (responsive 5:1).
    def _show_frame_preview(self, img: np.ndarray, frame_idx: int) -> None:
        """Display a BGR frame in the video preview area (preserve aspect)."""
        h, w = img.shape[:2]
        # Always scale to the LIVE widget size — responsive design.
        max_w = max(self.video_preview.width(), 1)
        max_h = max(self.video_preview.height(), 1)
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)
        if scale < 1.0:
            img = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h2, w2, c2 = rgb.shape
        from PySide6.QtGui import QImage
        qimage = QImage(rgb.tobytes(), w2, h2, w2 * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        # Use scaled() with KeepAspectRatio to prevent deformation
        self.video_preview.setPixmap(
            pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        )
        self.frame_info_label.setText(f"帧 #{frame_idx}  |  {w}×{h}")

    # ------------------------------------------------------------------
    # Video batch mode
    # ------------------------------------------------------------------
    def _start_video_batch(self, entries: List[Dict[str, str]]) -> None:
        self._is_running = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.seq_chart.reset()
        self._accumulated_image = None
        self._live_in_stance: Dict[str, List[int]] = {p: [] for p in PAW_ORDER}
        # Allow the preview QLabel to re-size to the new video's aspect
        # ratio on its first frame.
        self._preview_size_locked = False
        # Process first entry via background worker
        entry = entries[0]
        self._worker = GaitFTIRWorker(self)
        # Read threshold values from GUI controls.
        self._worker.body_threshold = self.body_threshold_spin.value()
        self._worker.paw_threshold = self.paw_threshold_spin.value()
        self._worker.min_area_px = self.min_area_spin.value()
        self._worker.frame_processed.connect(self._on_worker_frame)
        self._worker.run_completed.connect(self._on_worker_run_completed)
        self._worker.error_msg.connect(self._on_worker_error)
        self._worker.progress.connect(self.status_label.setText)
        self._worker.finished.connect(self._on_worker_finished)
        self.status_label.setText(f"分析中: {entry['video_path']}")
        self._worker.run_video(entry["video_path"], entry["animal_id"])

    def _on_worker_frame(self, seq, frame_bgr, frame_idx: int) -> None:
        """Real-time: update footprint + preview + sequence chart."""
        self._current_frame_bgr = frame_bgr
        self._update_footprint_image(seq, frame_bgr)
        # Accumulate per-paw stance for the live sequence chart.
        if not hasattr(self, "_live_in_stance") or not self._live_in_stance:
            self._live_in_stance: Dict[str, List[int]] = {
                p: [] for p in PAW_ORDER
            }
        for paw in PAW_ORDER:
            foot = seq.feet.get(paw)
            self._live_in_stance[paw].append(
                int(foot is not None and foot.is_in_stance)
            )
        # Update slider (throttled — every 3rd frame to avoid repaint storm).
        if frame_idx % 3 == 0:
            if self.frame_slider.maximum() < frame_idx:
                self.frame_slider.setMaximum(frame_idx)
            self.frame_slider.setValue(frame_idx)
            # Refresh sequence chart + video preview
            in_stance_arr = {p: np.array(v, dtype=np.int8)
                             for p, v in self._live_in_stance.items()}
            self.seq_chart.update_stance(in_stance_arr)
            self.seq_chart.set_current_frame(frame_idx)
            self._show_frame_preview(frame_bgr, frame_idx)
            # Time-cursor overlays (video_preview + footprint_view) are
            # updated automatically via slider's valueChanged signal.

    def _on_worker_run_completed(
        self, result, in_stance_lists, intensity_lists,
        centroids_x_lists=None, area_px_lists=None,
        pressure_lists=None,
    ) -> None:
        """A run completed inside the worker."""
        # Sync fps from the worker (read from video metadata) and
        # push to the seq_chart so its time-axis matches the video.
        if self._worker is not None:
            self._fps = self._worker.detected_fps
            self.seq_chart.set_fps(self._fps)
        self._on_run_completed(
            result, in_stance_lists, intensity_lists,
            centroids_x_lists, area_px_lists, pressure_lists,
        )

    def _on_worker_error(self, msg: str) -> None:
        QMessageBox.critical(self, "错误", msg)
        self._on_worker_finished()

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._on_stop()

    # ------------------------------------------------------------------
    # Live C1 mode
    # ------------------------------------------------------------------
    def _start_live(self, animal_id: str) -> None:
        self._is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        bg = RollingMedianBackground(warmup_frames=30)
        detector = RunDetector(fps=self._fps)
        detector.animal_id = animal_id
        self._bg_model = bg
        self._run_detector = detector
        self._accumulated_image = None
        self.seq_chart.reset()
        self._all_in_stance: Dict[str, List[int]] = {p: [] for p in PAW_ORDER}
        self._all_intensity: Dict[str, List[float]] = {p: [] for p in PAW_ORDER}

        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._live_tick)
        self._live_timer.start(33)  # ~30 fps

    def _live_tick(self) -> None:
        if not self._is_running or self._camera is None:
            return
        try:
            frame_info = self._camera.grab_one(timeout_ms=30)
        except Exception as e:
            logger.warning("live grab failed: %s", e)
            return
        frame_bgr = frame_info.image
        self._bg_model.update(frame_bgr)
        h, w = frame_bgr.shape[:2]
        try:
            seq = fp2.analyze_frame_v2(
                frame_bgr, background=self._bg_model,
                body_axis=((int(w*0.3), int(h*0.5)),
                           (int(w*0.7), int(h*0.5))),
                use_red_background=True,
                body_threshold=self.body_threshold_spin.value(),
                paw_threshold=self.paw_threshold_spin.value(),
                min_area_px=self.min_area_spin.value(),
            )
        except Exception as e:
            logger.debug("live analyze_frame_v2 failed: %s", e)
            return
        for paw in PAW_ORDER:
            foot = seq.feet.get(paw)
            stance = int(foot is not None and foot.is_in_stance)
            self._all_in_stance[paw].append(stance)
            self._all_intensity[paw].append(
                float(foot.intensity_max) if foot else 0.0
            )
        self._update_footprint_image(seq, frame_bgr)
        # Update sequence chart
        in_stance_arr = {p: np.array(v, dtype=np.int8)
                         for p, v in self._all_in_stance.items()}
        self.seq_chart.update_stance(in_stance_arr)
        # Live mode — set X range to cover all accumulated frames.
        n_live = len(self._all_in_stance[PAW_ORDER[0]])
        if n_live > 0:
            self.seq_chart.set_current_frame(n_live - 1)
            # Live X range = n_frames / fps (sliding window)
            self.seq_chart.setXRange(
                0, max(n_live / max(self._fps, 1), 0.5), padding=0.02,
            )
            # (Time-cursor overlays removed in W20; X axis of
            #  video_preview and footprint_view is spatial, not time.)
        # Check run completion
        result = self._run_detector.process_frame(seq)
        if result is not None:
            self._on_run_completed(result, self._all_in_stance,
                                   self._all_intensity)
            self._all_in_stance = {p: [] for p in PAW_ORDER}
            self._all_intensity = {p: [] for p in PAW_ORDER}

    # ------------------------------------------------------------------
    # Run completed handler
    # ------------------------------------------------------------------
    def _on_run_completed(
        self,
        result: RunResult,
        in_stance_lists: Dict[str, List[int]],
        intensity_lists: Dict[str, List[float]],
        centroids_x_lists: Dict[str, List[float]] | None = None,
        area_px_lists: Dict[str, List[float]] | None = None,
        pressure_lists: Dict[str, List[Dict]] | None = None,
    ) -> None:
        in_stance = {p: np.array(v, dtype=np.int8)
                     for p, v in in_stance_lists.items()}
        intensity = {p: np.array(v, dtype=np.float64)
                     for p, v in intensity_lists.items()}
        centroids_x = None
        area_px = None
        if centroids_x_lists:
            centroids_x = {p: np.array(v, dtype=np.float64)
                           for p, v in centroids_x_lists.items()}
        if area_px_lists:
            area_px = {p: np.array(v, dtype=np.float64)
                       for p, v in area_px_lists.items()}
        # Store pressure data for XYZ visualization (future use).
        self._last_pressure_data = pressure_lists

        metrics = gait_ftir.compute_catwalk_equivalent_metrics(
            in_stance, intensity,
            centroids_x=centroids_x,
            area_px_curves=area_px,
            fps=self._fps,
        )
        self.param_panel.update_metrics(metrics)
        self.seq_chart.update_stance(in_stance)
        # Enable slider range
        n = max(len(v) for v in in_stance.values())
        if n > 0:
            self.frame_slider.setRange(0, n - 1)
            self.frame_slider.setEnabled(True)
        # Run summary (show, auto-hide 5s)
        for k, lbl in self.run_summary_labels.items():
            if k in metrics:
                lbl.setText(str(metrics[k]))
        self.run_summary.show()
        self._summary_timer.start(5000)
        logger.info("Run completed: animal=%s frames=%d-%d",
                    result.animal_id, result.start_frame, result.end_frame)

    # ------------------------------------------------------------------
    # Footprint accumulation image — 动态压力累积
    # ------------------------------------------------------------------
    # Black background.  Green footprints are painted with intensity
    # proportional to the paw's pressure (green channel value).  Each
    # pixel keeps its MAXIMUM green intensity across all frames, so
    # the final footprint shows the deepest contact point (peak stance).
    _FOOTPRINT_BG_COLOR = np.array([0, 0, 0], dtype=np.uint8)        # BGR black

    def _update_footprint_image(self, seq, frame_bgr: np.ndarray = None) -> None:
        """Paint detected paws onto the accumulation image with pressure-weighted intensity.

        Instead of uniform green, each pixel's green value comes from the
        actual frame green channel (pressure proxy).  ``np.maximum`` ensures
        only the strongest contact at each pixel is retained, producing a
        natural "light → deep" gradient as the paw presses down.
        """
        if frame_bgr is None:
            return
        fh, fw = frame_bgr.shape[:2]
        if self._accumulated_image is None or \
                self._accumulated_image.shape[:2] != (fh, fw):
            self._accumulated_image = np.tile(
                self._FOOTPRINT_BG_COLOR, (fh, fw, 1)
            )
        self._current_frame_bgr = frame_bgr.copy()

        # Build a per-frame "pressure image": green channel where paw is
        # detected, zero elsewhere.
        pressure_frame = np.zeros((fh, fw), dtype=np.uint8)
        green_channel = frame_bgr[:, :, 1]  # G channel = pressure

        for paw in PAW_ORDER:
            foot = seq.feet.get(paw)
            if foot is None or not foot.is_in_stance:
                continue
            x, y, w, h = foot.bbox
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(fw, x + w), min(fh, y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            # Copy the green channel values within the bbox as pressure.
            paw_green = green_channel[y0:y1, x0:x1].copy()
            # Only keep pixels that are actually "paw" (green > 0).
            # Use the current paw_threshold as a minimum.
            paw_mask = paw_green >= self.paw_threshold_spin.value()
            paw_green[~paw_mask] = 0
            # Keep the maximum pressure at each pixel across frames.
            current = pressure_frame[y0:y1, x0:x1]
            pressure_frame[y0:y1, x0:x1] = np.maximum(current, paw_green)

        # Paint pressure onto accumulated image (BGR: G channel = pressure).
        # Use maximum so later deeper contacts overwrite lighter ones.
        acc_g = self._accumulated_image[:, :, 1]
        np.maximum(acc_g, pressure_frame, out=acc_g)

        rgb = cv2.cvtColor(self._accumulated_image, cv2.COLOR_BGR2RGB)
        # Render the accumulated image to the QLabel via QPixmap, scaled
        # with KeepAspectRatio to preserve the 1920×384 (5:1) shape.
        h, w, c = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, w * c, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        target_w = max(self.footprint_view.width(), 1)
        target_h = max(self.footprint_view.height(), 1)
        self.footprint_view.setPixmap(
            pix.scaled(target_w, target_h,
                       Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
        )

    def reset(self) -> None:
        self._on_stop()
        self._accumulated_image = None
        self.seq_chart.reset()
        self.c1_status.setText("未连接")
        self.c1_status.setStyleSheet("color: #888;")
        self.status_label.setText("就绪 — 请加载视频或连接 C1 相机")

    # ------------------------------------------------------------------
    # Backward-compat proxy properties — old GaitTab (DLC CSV-based)
    # exposed these widgets directly.  Tests and external code that
    # referenced ``gait_tab.paw_selector``, ``gait_tab.table``, etc.
    # keep working via these proxies (they return simple placeholder
    # or equivalent widgets).
    # ------------------------------------------------------------------
    @property
    def paw_selector(self):
        if not hasattr(self, "_paw_selector_proxy"):
            self._paw_selector_proxy = QComboBox()
            self._paw_selector_proxy.addItems(PAW_ORDER)
        return self._paw_selector_proxy

    @property
    def table(self):
        if not hasattr(self, "_table_proxy"):
            self._table_proxy = QTableWidget()
        return self._table_proxy

    @property
    def browse_btn(self):
        return self.load_video_btn

    @property
    def analyze_btn(self):
        return self.start_btn

    @property
    def path_edit(self):
        if not hasattr(self, "_path_edit_proxy"):
            self._path_edit_proxy = QLineEdit()
        return self._path_edit_proxy

    @property
    def chart(self):
        """Return matplotlib-based chart (may be None in new tab)."""
        return None

    @property
    def realtime_chart(self):
        """Return the pyqtgraph-based 4-paw sequence chart."""
        return self.seq_chart

    @property
    def chart_type_combo(self):
        if not hasattr(self, "_chart_type_combo_proxy"):
            self._chart_type_combo_proxy = QComboBox()
        return self._chart_type_combo_proxy

    @property
    def fps_spin(self):
        if not hasattr(self, "_fps_spin_proxy"):
            self._fps_spin_proxy = QSpinBox()
            self._fps_spin_proxy.setValue(100)
        return self._fps_spin_proxy

    @property
    def mode_combo_proxy(self):
        return self.mode_combo

    @property
    def progress(self):
        if not hasattr(self, "_progress_proxy"):
            from PySide6.QtWidgets import QProgressBar
            self._progress_proxy = QProgressBar()
        return self._progress_proxy

    @property
    def export_excel_btn(self):
        if not hasattr(self, "_export_excel_btn_proxy"):
            self._export_excel_btn_proxy = QPushButton("Export Excel")
        return self._export_excel_btn_proxy

    @property
    def export_csv_btn(self):
        if not hasattr(self, "_export_csv_btn_proxy"):
            self._export_csv_btn_proxy = QPushButton("Export CSV")
        return self._export_csv_btn_proxy

    def _on_browse(self):
        self._on_load_video()

    def _on_analyze(self):
        self._on_start()


# ---------------------------------------------------------------------------
# Backward-compat exports (used by editor_tab.py, old tests, etc.)
# ---------------------------------------------------------------------------
_TABLE_COLUMNS = [
    "Paw", "Stance(ms)", "Swing(ms)", "Strides",
    "SL_mean", "SL_var", "Freq(Hz)", "Angle°", "Width", "Symmetry",
]

PAW_COLORS_DICT = {
    "LeftFore": "#2e7d32",
    "RightFore": "#c62828",
    "LeftHind": "#6a1b9a",
    "RightHind": "#1565c0",
}
