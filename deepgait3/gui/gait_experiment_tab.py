"""Tab 2: 新建实验 (New Experiment) — deepgait v2.0.

Real-time C1 camera acquisition or video file playback with live
footprint accumulation.  Records video to ``{project}/videos/{animal_id}.mp4``.

Supports:
- C1 camera connection (via InitializationTab)
- Video file as an alternative source
- Live preview (QLabel + KeepAspectRatio)
- Real-time footprint accumulation (RunningMedianBackground + analyze_frame_v2)
- Recording to project folder
- Animal ID table
- Threshold controls
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepgait3.core.pawprint.single_frame import detect_single_frame
from deepgait3.core._legacy.background_model import RollingMedianBackground

logger = logging.getLogger(__name__)

PAW_ORDER = ("LF", "RF", "LH", "RH")


# ---------------------------------------------------------------------------
# ExperimentEntryTable — animal ID list
# ---------------------------------------------------------------------------
class ExperimentEntryTable(QTableWidget):
    """Table for experiment entries: animal ID + status."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Animal ID", "Video Name", "Status"])
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive,
        )
        self.setColumnWidth(0, 140)
        self.setColumnWidth(1, 160)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)

    def add_entry(self, animal_id: str = "") -> int:
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(animal_id))
        self.setItem(row, 1, QTableWidgetItem("(auto)"))
        self.setItem(row, 2, QTableWidgetItem("Ready"))
        return row

    def remove_selected(self) -> None:
        for row in sorted(
            {i.row() for i in self.selectedIndexes()}, reverse=True,
        ):
            self.removeRow(row)

    def entries(self) -> List[Dict[str, str]]:
        out = []
        for row in range(self.rowCount()):
            aid = self.item(row, 0)
            out.append({
                "animal_id": aid.text().strip() if aid else "",
                "status": self.item(row, 2).text() if self.item(row, 2) else "Ready",
            })
        return out

    def set_status(self, row: int, status: str) -> None:
        if 0 <= row < self.rowCount():
            self.setItem(row, 2, QTableWidgetItem(status))

    def set_video_name(self, row: int, name: str) -> None:
        if 0 <= row < self.rowCount():
            self.setItem(row, 1, QTableWidgetItem(name))


# ---------------------------------------------------------------------------
# GaitExperimentTab
# ---------------------------------------------------------------------------
class GaitExperimentTab(QWidget):
    """Real-time acquisition + live footprint preview."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._app_state = None
        self._camera = None
        self._is_running = False
        self._is_paused = False
        self._frame_count = 0
        self._fps = 100.0
        self._current_row = -1

        # Background model + accumulated image
        self._bg_model: Optional[RollingMedianBackground] = None
        self._accumulated_image: Optional[np.ndarray] = None
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._source_mode = "camera"  # "camera" | "file"

        # Timers
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._live_tick)
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps_label)
        self._fps_counter = 0
        self._last_fps_time = time.perf_counter()

        self._build_ui()

    def set_app_state(self, app_state) -> None:
        self._app_state = app_state
        self._update_project_display()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setSpacing(4)
        main.setContentsMargins(6, 6, 6, 6)

        # -- Project info ----------------------------------------------------
        self.project_label = QLabel("当前项目: 未打开")
        main.addWidget(self.project_label)

        # -- Source selection ------------------------------------------------
        src_gb = QGroupBox("数据源")
        src_l = QHBoxLayout(src_gb)
        src_l.addWidget(QLabel("来源:"))
        self.radio_camera = QRadioButton("C1 相机")
        self.radio_camera.setChecked(True)
        self.radio_camera.toggled.connect(self._on_source_changed)
        src_l.addWidget(self.radio_camera)
        self.radio_file = QRadioButton("视频文件")
        self.radio_file.toggled.connect(self._on_source_changed)
        src_l.addWidget(self.radio_file)

        self.connect_btn = QPushButton("连接 C1")
        self.connect_btn.clicked.connect(self._on_connect_camera)
        src_l.addWidget(self.connect_btn)

        self.load_video_btn = QPushButton("加载视频")
        self.load_video_btn.clicked.connect(self._on_load_video)
        self.load_video_btn.setVisible(False)
        src_l.addWidget(self.load_video_btn)

        self.source_status = QLabel("未连接")
        src_l.addWidget(self.source_status)
        src_l.addStretch()
        main.addWidget(src_gb)

        # -- Entry table -----------------------------------------------------
        table_gb = QGroupBox("实验列表")
        tl = QVBoxLayout(table_gb)
        tl.setSpacing(2)
        tl.setContentsMargins(4, 8, 4, 4)
        # Create table first (needed by button callbacks), then add UI in order.
        self.entry_table = ExperimentEntryTable()
        self.entry_table.setMinimumHeight(60)
        self.entry_table.setMaximumHeight(260)
        # Buttons above table
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ 添加行")
        add_btn.clicked.connect(lambda: self.entry_table.add_entry())
        btn_row.addWidget(add_btn)
        del_btn = QPushButton("删除选中")
        del_btn.clicked.connect(self.entry_table.remove_selected)
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        tl.addLayout(btn_row)
        tl.addWidget(self.entry_table)
        main.addWidget(table_gb)

        # -- Control buttons -------------------------------------------------
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始采集")
        self.start_btn.clicked.connect(self._on_start)
        ctrl_row.addWidget(self.start_btn)
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        ctrl_row.addWidget(self.pause_btn)
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self.stop_btn)

        self.record_check = QCheckBox("录制视频")
        self.record_check.setChecked(True)
        ctrl_row.addWidget(self.record_check)
        ctrl_row.addStretch()

        self.status_label = QLabel("就绪")
        ctrl_row.addWidget(self.status_label)
        main.addLayout(ctrl_row)

        # -- Preview panels (vertical stack, 5:1 aspect) -------------------
        preview_gb = QGroupBox("实时预览")
        preview_l = QVBoxLayout(preview_gb)
        preview_l.setSpacing(4)

        # Video preview (top)
        video_box = QVBoxLayout()
        video_box.addWidget(QLabel("视频预览"))
        self.video_preview = QLabel()
        self.video_preview.setStyleSheet("background: black;")
        self.video_preview.setMinimumSize(960, 240)
        self.video_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.video_preview.setScaledContents(True)
        video_box.addWidget(self.video_preview, stretch=1)
        preview_l.addLayout(video_box, stretch=1)

        # Footprint accumulation (bottom) — with threshold controls above
        fp_box = QVBoxLayout()
        fp_header = QHBoxLayout()
        fp_header.addWidget(QLabel("脚印累积图"))
        fp_header.addSpacing(8)
        fp_header.addWidget(QLabel("绿色亮度阈值:"))
        self.tau_spin = QSpinBox()
        self.tau_spin.setRange(1, 80)
        self.tau_spin.setValue(10)
        self.tau_spin.setMaximumWidth(50)
        self.tau_spin.setToolTip("亮度阈值(tau)：绿光高于背景多少算脚印。越低越灵敏(7-15为佳)")
        fp_header.addWidget(self.tau_spin)
        fp_header.addWidget(QLabel("绿色面积阈值:"))
        self.paw_spin = QSpinBox()
        self.paw_spin.setRange(2, 600)
        self.paw_spin.setValue(10)
        self.paw_spin.setMaximumWidth(60)
        self.paw_spin.setToolTip("最小爪印面积(px)，越小越能保留轻微脚印")
        fp_header.addWidget(self.paw_spin)
        fp_header.addWidget(QLabel("显示:"))
        self.brightness_spin = QDoubleSpinBox()
        self.brightness_spin.setRange(0.5, 3.0)
        self.brightness_spin.setValue(1.0)
        self.brightness_spin.setSingleStep(0.1)
        self.brightness_spin.setDecimals(1)
        self.brightness_spin.setMaximumWidth(60)
        self.brightness_spin.setToolTip("累积图显示亮度倍率")
        fp_header.addWidget(self.brightness_spin)
        fp_header.addStretch()
        fp_box.addLayout(fp_header)
        self.footprint_view = QLabel()
        self.footprint_view.setStyleSheet("background: black;")
        self.footprint_view.setMinimumSize(960, 240)
        self.footprint_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.footprint_view.setScaledContents(True)
        fp_box.addWidget(self.footprint_view, stretch=1)
        preview_l.addLayout(fp_box, stretch=1)

        main.addWidget(preview_gb)

        # -- Frame info ------------------------------------------------------
        self.frame_label = QLabel("Frame: 0 | FPS: --")
        self.frame_label.setStyleSheet("font-family: monospace; color: #888;")
        main.addWidget(self.frame_label)

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------
    def _on_source_changed(self) -> None:
        self._source_mode = "camera" if self.radio_camera.isChecked() else "file"
        self.connect_btn.setVisible(self._source_mode == "camera")
        self.load_video_btn.setVisible(self._source_mode == "file")

    def _on_connect_camera(self) -> None:
        try:
            main_win = self.window()
            init_tab = main_win.tab_by_name("initialization")
            cam_dict = init_tab.config_panel.cameras()
            self._camera = cam_dict.get("bottom")  # C1
            if self._camera is None:
                QMessageBox.warning(self, "错误", "未找到 C1 相机，请在「初始化」Tab 中先检测相机。")
                return
            self._fps = float(
                init_tab.config_panel.groups().get("bottom", type(
                    "Dummy", (), {"fps_spin": type("D", (), {"value": lambda: 100})()},
                )()).fps_spin.value() or 100,
            )
            self.source_status.setText(f"● C1 已连接 ({self._fps:.0f} fps)")
            self.source_status.setStyleSheet("color: #2E7D32; font-weight: bold;")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"连接 C1 失败: {e}")

    def _on_load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not path:
            return
        self._video_path = path
        self.source_status.setText(f"● 视频: {Path(path).name}")
        self.source_status.setStyleSheet("color: #1565C0; font-weight: bold;")
        # Auto-add row if table is empty
        if self.entry_table.rowCount() == 0:
            self.entry_table.add_entry(Path(path).stem)
        # Fill first empty row
        name = Path(path).stem
        for row in range(self.entry_table.rowCount()):
            aid_item = self.entry_table.item(row, 0)
            if aid_item and not aid_item.text().strip():
                aid_item.setText(name)
                self.entry_table.set_video_name(row, name)
                break

    # ------------------------------------------------------------------
    # Acquisition control
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        # Select current row
        entries = self.entry_table.entries()
        row = self.entry_table.currentRow()
        if row < 0:
            row = 0
        if row >= len(entries):
            QMessageBox.warning(self, "提示", "请先添加实验行并填写动物编号。")
            return
        entry = entries[row]
        animal_id = entry["animal_id"]
        if not animal_id:
            QMessageBox.warning(self, "提示", "请填写动物编号。")
            return
        # Guard: only one source active
        if self._source_mode == "camera" and self._camera is None:
            QMessageBox.warning(self, "提示", "请先连接 C1 相机。")
            return
        if self._source_mode == "file" and not hasattr(self, "_video_path"):
            QMessageBox.warning(self, "提示", "请先加载视频文件。")
            return

        self._current_row = row
        self._is_running = True
        self._is_paused = False
        self._frame_count = 0
        self._accumulated_image = None
        self._fps_counter = 0
        self._last_fps_time = time.perf_counter()

        # Init background model
        self._bg_model = RollingMedianBackground(warmup_frames=30)

        # Init video writer for recording
        self._video_writer = None
        if self.record_check.isChecked():
            self._init_video_writer(animal_id)

        # Open video file if in file mode
        if self._source_mode == "file":
            self._video_capture = cv2.VideoCapture(getattr(self, "_video_path", ""))

        # Set video name in table
        video_name = f"{animal_id}.mp4"
        self.entry_table.set_video_name(row, video_name)
        self.entry_table.set_status(row, "采集中...")

        # UI state
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.status_label.setText(f"采集中: {animal_id}")

        # Start timers
        self._live_timer.start(33)  # ~30 fps
        self._fps_timer.start(1000)  # FPS update every 1s

    def _on_pause(self) -> None:
        self._is_paused = not self._is_paused
        self.pause_btn.setText("▶ 继续" if self._is_paused else "⏸ 暂停")

    def _on_stop(self) -> None:
        self._is_running = False
        self._live_timer.stop()
        self._fps_timer.stop()

        # Save debug image
        if self._accumulated_image is not None and self._accumulated_image[:,:,1].max() > 0:
            cv2.imwrite('/tmp/experiment_debug.png', self._accumulated_image)
            logger.info('Debug accum saved: G max=%d, nonzero=%d',
                        self._accumulated_image[:,:,1].max(),
                        (self._accumulated_image[:,:,1] > 0).sum())

        # Close video writer
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

        if hasattr(self, "_video_capture"):
            self._video_capture.release()

        # Update status
        if self._current_row >= 0:
            self.entry_table.set_status(self._current_row, f"完成 ({self._frame_count}帧)")

        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.status_label.setText("已停止")

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------
    def _live_tick(self) -> None:
        if not self._is_running or self._is_paused:
            return

        # Grab frame
        frame_bgr = None
        if self._source_mode == "camera" and self._camera is not None:
            try:
                info = self._camera.grab_one(timeout_ms=30)
                frame_bgr = info.image
            except Exception:
                return
        elif self._source_mode == "file":
            cap = getattr(self, "_video_capture", None)
            if cap is not None:
                ret, frame_bgr = cap.read()
                if not ret:
                    self._on_stop()
                    return

        if frame_bgr is None:
            return

        h, w = frame_bgr.shape[:2]
        self._frame_count += 1
        self._fps_counter += 1

        # Background model + footprint detection (pawprint tau-threshold)
        self._bg_model.update(frame_bgr)
        try:
            bg = self._bg_model.get_median()
            bg_G = bg[:, :, 1].astype(np.float32) if bg is not None else np.zeros((h, w), dtype=np.float32)
            body_axis = ((int(w * 0.2), int(h * 0.5)),
                         (int(w * 0.8), int(h * 0.5)))
            seq = detect_single_frame(
                frame_bgr, bg_G,
                tau_paw=self.tau_spin.value(),
                min_area_px=self.paw_spin.value(),
                D_merge_px=23.0,
                walkway_roi=(0, 0, w, h),
                bbox_pad_px=8,
                mouse_det=None,
                px_per_mm=None,
                body_axis=body_axis,
                in_stance_threshold_px=30,
            )
        except Exception:
            from deepgait3.core._legacy.footprint_v2 import FootprintSequence
            seq = FootprintSequence()

        # -- Update footprint accumulation -----------------------------------
        self._update_footprint(frame_bgr, seq)

        # -- Record video ----------------------------------------------------
        if self._video_writer is not None:
            self._video_writer.write(frame_bgr)

        # -- Update preview (every 3rd frame) --------------------------------
        if self._frame_count % 3 == 0:
            self._show_preview(frame_bgr)
            self._show_footprint()

    def _update_footprint(self, frame_bgr: np.ndarray, seq) -> None:
        """Paint detected paws onto accumulation image.

        Iterates ``seq.all_feet`` (every detected foot, including faint
        contacts that fail the 4-quadrant classification or the
        ``is_in_stance`` gate) so light/fur touches are not silently
        dropped. The green channel within each foot's bbox is max-merged
        into the accumulator; brightness is applied at display time.
        """
        h, w = frame_bgr.shape[:2]
        if self._accumulated_image is None:
            self._accumulated_image = np.zeros((h, w, 3), dtype=np.uint8)
        green = frame_bgr[:, :, 1].astype(np.uint8)
        # Pre-subtract the background green so only the contact signal
        # (above-background green) is accumulated — this keeps fur glints
        # from slowly raising the whole floor over time.
        bg = self._bg_model.get_median() if self._bg_model is not None else None
        if bg is not None:
            base = bg[:, :, 1].astype(np.int16)
            signal = np.clip(green.astype(np.int16) - base, 0, 255).astype(np.uint8)
        else:
            signal = green
        for foot in seq.all_feet:
            x, y, bw, bh = foot.bbox
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw), min(h, y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            paw_signal = signal[y0:y1, x0:x1]
            current = self._accumulated_image[y0:y1, x0:x1, 1]
            np.maximum(current, paw_signal, out=current)

        # Debug: log first few accumulations
        if self._frame_count <= 5:
            logger.info('Frame %d: all_feet=%d accum G max=%d nonzero=%d',
                        self._frame_count, len(seq.all_feet),
                        self._accumulated_image[:,:,1].max(),
                        (self._accumulated_image[:,:,1] > 0).sum())

    def _show_preview(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        pw = max(self.video_preview.width(), 480)
        ph = max(self.video_preview.height(), 96)
        # Keep 5:1 aspect ratio
        target_aspect = 5.0
        widget_aspect = pw / max(ph, 1)
        if widget_aspect > target_aspect:
            # Widget is wider → constrain by height
            nh = ph
            nw = int(nh * target_aspect)
        else:
            # Widget is taller → constrain by width
            nw = pw
            nh = int(nw / target_aspect)
        nw, nh = max(1, nw), max(1, nh)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (nw, nh))
        qimg = QImage(img.data, nw, nh, img.strides[0], QImage.Format.Format_RGB888)
        self.video_preview.setPixmap(QPixmap.fromImage(qimg))

    def _show_footprint(self) -> None:
        if self._accumulated_image is None:
            return
        display = self._accumulated_image.copy()
        g = display[:, :, 1]

        # ── Step 1: open morphology to break tail/belly drag streaks ──────
        # The accumulator uses max() which lets thin streaks (tail drag,
        # belly fur) build up over many frames into long ribbons that
        # contaminate the visual. An opening (erode → dilate) with a
        # kernel ~12 px disconnects these ribbons while preserving
        # compact paw blobs (which are typically > 15×15 px in the
        # accumulator). The tail/belly drag is typically < 8 px wide, so
        # erosion with a 12 px disk removes it completely.
        g_mask = (g > 5).astype(np.uint8) * 255
        if g_mask.any():
            open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            cleaned = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN, open_k)
            # Re-apply intensity (max of original and cleaned mask).
            g = cv2.bitwise_and(g, cleaned)

        # ── Step 2: render as green-on-black (mask-protected background) ──
        # Reference: /home/luofangcheng/Documents/ZCODE/tmp/extract_cumulative_paws.py L290-298
        # Black background NEVER changes.  Only masked paw pixels get green.
        g_f = g.astype(np.float32)
        mask = g > 0
        if mask.any():
            vals = g_f[mask]
            lo, hi = float(vals.min()), float(vals.max())
            norm = np.clip((g_f - lo) / (hi - lo if hi > lo else 1.0), 0.0, 1.0)
        else:
            norm = np.zeros_like(g_f)
        bright = self.brightness_spin.value()
        display[:, :, 0] = 0
        display[:, :, 1] = np.where(mask, (norm * 255.0 * bright).clip(0, 255).astype(np.uint8), 0)
        display[:, :, 2] = 0

        h, w = display.shape[:2]
        pw = max(self.footprint_view.width(), 480)
        ph = max(self.footprint_view.height(), 96)
        target_aspect = 5.0
        widget_aspect = pw / max(ph, 1)
        if widget_aspect > target_aspect:
            nh = ph
            nw = int(nh * target_aspect)
        else:
            nw = pw
            nh = int(nw / target_aspect)
        nw, nh = max(1, nw), max(1, nh)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (nw, nh))
        qimg = QImage(img.data, nw, nh, img.strides[0], QImage.Format.Format_RGB888)
        self.footprint_view.setPixmap(QPixmap.fromImage(qimg))

    def _update_fps_label(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_fps_time
        if dt > 0:
            fps = self._fps_counter / dt
            self.frame_label.setText(
                f"Frame: {self._frame_count} | FPS: {fps:.1f}",
            )
        self._fps_counter = 0
        self._last_fps_time = now

    # ------------------------------------------------------------------
    # Video writer
    # ------------------------------------------------------------------
    def _init_video_writer(self, animal_id: str) -> None:
        """Create cv2.VideoWriter targeting the project videos dir."""
        if self._app_state and self._app_state.project.is_valid:
            vdir = Path(self._app_state.project.project_path) / "videos"
        else:
            vdir = Path.home() / "deepgait_projects" / "default" / "videos"
        vdir.mkdir(parents=True, exist_ok=True)
        path = vdir / f"{animal_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        # Determine frame size from camera or default
        w, h = 1920, 384  # default FTIR size
        try:
            if self._source_mode == "file" and hasattr(self, "_video_capture"):
                cap = self._video_capture
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            pass
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(str(path), fourcc, self._fps, (w, h))
        logger.info("Recording to %s", path)

    # ------------------------------------------------------------------
    # Project display
    # ------------------------------------------------------------------
    def _update_project_display(self) -> None:
        if self._app_state and self._app_state.project.is_valid:
            p = self._app_state.project
            self.project_label.setText(
                f"当前项目: {p.project_name} ({p.project_path})",
            )
        else:
            self.project_label.setText("当前项目: 未打开")
