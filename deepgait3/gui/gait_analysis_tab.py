"""Tab 4: 数据分析 (Data Analysis) — deepgait v3.

Workflow:
  1. Load project videos → table populated with animal IDs + file paths
  2. Select a video → preview with live footprint accumulation
  3. Tune tau / area / brightness in the 参数微调 panel
  4. Click 视频分析 → batch-processes ALL videos with the tuned parameters
  5. Status column shows per-video progress

Algorithm: :func:`deepgait3.core.pawprint.single_frame.detect_single_frame`
(pawprint tau-threshold detection).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepgait3.core.pawprint.single_frame import detect_single_frame
from deepgait3.core._legacy import gait_export, gait_ftir, gait_pressure, foot_pattern
from deepgait3.core._legacy.background_model import RollingMedianBackground
from deepgait3.core._legacy.project_manager import animal_data_dir
from deepgait3.gui.gait_tab import PAW_COLORS, PAW_ORDER, VideoBatchTable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch analysis worker — pawprint-powered, parameter-injectable
# ---------------------------------------------------------------------------
class BatchAnalysisWorker(QThread):
    """Processes a list of videos sequentially with pawprint detection.

    Parameters are injected from the UI's 参数微调 panel so the user
    can preview before committing to a full batch run.
    """

    video_started = Signal(int, str)          # index, animal_id
    video_progress = Signal(int, int, int)    # index, frame, total_frames
    video_completed = Signal(int, str, dict)  # index, output_dir, metrics
    frame_processed = Signal(object, object, int)  # seq, frame_bgr, frame_idx
    all_completed = Signal(list)              # list of result dicts
    error_occurred = Signal(str)

    def __init__(
        self,
        entries: List[Dict[str, str]],
        output_base_dir: Path,
        tau_paw: float = 10.0,
        min_area_px: int = 10,
        px_per_mm: float = 3.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.entries = entries
        self.output_base_dir = Path(output_base_dir)
        self.tau_paw = tau_paw
        self.min_area_px = min_area_px
        self.px_per_mm = px_per_mm
        self._abort = False

    def run(self) -> None:
        results = []
        for idx, entry in enumerate(self.entries):
            if self._abort:
                break
            animal_id = entry.get("animal_id", f"unknown_{idx}")
            video_path = entry.get("video_path", "")
            if not video_path:
                continue
            self.video_started.emit(idx, animal_id)
            try:
                result = self._process_one(idx, animal_id, video_path)
                results.append(result)
                self.video_completed.emit(
                    idx, str(result.get("output_dir", "")),
                    result.get("metrics", {}),
                )
            except Exception as e:
                self.error_occurred.emit(f"{animal_id}: {e}")
                logger.exception("Batch worker error for %s", animal_id)
        self.all_completed.emit(results)

    # ------------------------------------------------------------------
    def _process_one(self, idx: int, animal_id: str, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        bg = RollingMedianBackground(warmup_frames=30)
        accum_centroids: List = []
        all_in_stance = {p: [] for p in PAW_ORDER}
        all_intensity = {p: [] for p in PAW_ORDER}
        all_centroids_x = {p: [] for p in PAW_ORDER}
        all_centroids_y = {p: [] for p in PAW_ORDER}
        all_area_px = {p: [] for p in PAW_ORDER}
        all_pressure: Dict[str, List[Dict]] = {p: [] for p in PAW_ORDER}

        fi = 0
        while not self._abort:
            ret, frame = cap.read()
            if not ret:
                break
            bg.update(frame)

            # Body axis: use PCA on recent centroids, fallback horizontal.
            if len(accum_centroids) >= 6:
                body_axis = self._estimate_body_axis(accum_centroids[-30:], w, h)
            else:
                body_axis = ((int(w * 0.2), int(h * 0.5)),
                             (int(w * 0.8), int(h * 0.5)))

            # Pawprint detection (replaces old fp2.analyze_frame_v2).
            bg_med = bg.get_median()
            if bg_med is not None and bg.is_ready:
                bg_G = bg_med[:, :, 1].astype(np.float32)
            else:
                bg_G = np.zeros((h, w), dtype=np.float32)

            seq = detect_single_frame(
                frame, bg_G,
                tau_paw=self.tau_paw,
                min_area_px=self.min_area_px,
                D_merge_px=23.0,
                walkway_roi=(0, 0, w, h),
                bbox_pad_px=8,
                mouse_det=None,
                body_axis=body_axis,
                in_stance_threshold_px=30,
            )

            for fm in seq.feet.values():
                if fm.is_in_stance:
                    accum_centroids.append(fm.centroid)
            if len(accum_centroids) > 60:
                accum_centroids = accum_centroids[-60:]

            for paw in PAW_ORDER:
                foot = seq.feet.get(paw)
                all_in_stance[paw].append(int(foot is not None and foot.is_in_stance))
                all_intensity[paw].append(float(foot.intensity_max) if foot else 0.0)
                all_centroids_x[paw].append(float(foot.centroid[0]) if foot else 0.0)
                all_centroids_y[paw].append(float(foot.centroid[1]) if foot else 0.0)
                all_area_px[paw].append(float(foot.area_px) if foot else 0.0)
                if foot is not None and foot.is_in_stance:
                    all_pressure[paw].append({
                        "frame": fi, "x": float(foot.centroid[0]),
                        "y": float(foot.centroid[1]),
                        "area": float(foot.area_px),
                        "intensity": float(foot.intensity_max),
                    })

            self.frame_processed.emit(seq, frame, fi)
            fi += 1
            if fi % 10 == 0:
                self.video_progress.emit(idx, fi, total_frames)
        cap.release()

        # --- metrics ---
        in_stance = {p: np.array(v, dtype=np.int8) for p, v in all_in_stance.items()}
        intensity = {p: np.array(v, dtype=np.float64) for p, v in all_intensity.items()}
        cx = {p: np.array(v, dtype=np.float64) for p, v in all_centroids_x.items()}
        cy = {p: np.array(v, dtype=np.float64) for p, v in all_centroids_y.items()}
        area_px = {p: np.array(v, dtype=np.float64) for p, v in all_area_px.items()}

        metrics = gait_ftir.compute_catwalk_equivalent_metrics(
            in_stance, intensity, centroids_x=cx, area_px_curves=area_px,
            fps=fps, px_per_mm=self.px_per_mm,
        )
        for paw in PAW_ORDER:
            pa = gait_pressure.compute_per_paw_pressure_aggregates(
                in_stance[paw], intensity[paw], area_px[paw],
                fps, self.px_per_mm,
            )
            for k, v in pa.items():
                metrics[f"{paw}_{k}"] = v
        try:
            stance_onsets = {
                p: np.where(np.diff(np.concatenate(([0], in_stance[p]))) == 1)[0]
                for p in PAW_ORDER
            }
            seq_r = foot_pattern.classify_step_sequence(stance_onsets)
            metrics.update({f"step_sequence_{k}": v for k, v in seq_r.items()})
            metrics["regularity_index"] = foot_pattern.compute_regularity_index(seq_r)
            bos = foot_pattern.compute_bos(cx, cy, in_stance, self.px_per_mm)
            metrics.update(bos)
            support = foot_pattern.compute_support_patterns(in_stance)
            metrics.update(support)
        except Exception:
            pass
        all_steps: Dict[str, List[Dict]] = {}
        for paw in PAW_ORDER:
            steps = gait_ftir.compute_per_step_metrics(
                in_stance[paw], intensity[paw], cx[paw], fps, self.px_per_mm,
            )
            all_steps[paw] = steps
        pressure_rows = gait_pressure.build_per_frame_pressure(
            fi, cx, cy, area_px, intensity, intensity,
        )
        out_dir = animal_data_dir(self.output_base_dir, animal_id)
        # accumulated image: simple peak-merge for export
        accumulated_image: Optional[np.ndarray] = None  # built during export
        exported = gait_export.export_all(
            out_dir, animal_id, metrics, all_steps, pressure_rows,
            accumulated_image=accumulated_image,
            in_stance=in_stance, intensity_curves=intensity,
            centroids_x=cx, centroids_y=cy, fps=fps,
        )
        return {
            "animal_id": animal_id, "video_path": video_path,
            "output_dir": str(out_dir), "metrics": metrics,
            "exported_files": [str(p) for p in exported],
            "n_frames": fi, "fps": fps,
        }

    @staticmethod
    def _estimate_body_axis(centroids, w, h):
        pts = np.array(centroids)
        if len(pts) < 3:
            return ((int(w * 0.2), int(h * 0.5)),
                    (int(w * 0.8), int(h * 0.5)))
        mean = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - mean, full_matrices=False)
        direction = vh[0]
        cx, cy = float(mean[0]), float(mean[1])
        half_len = max(w * 0.3, 100)
        p1 = (int(cx - direction[0] * half_len), int(cy - direction[1] * half_len))
        p2 = (int(cx + direction[0] * half_len), int(cy + direction[1] * half_len))
        return (p1, p2)

    def abort(self) -> None:
        self._abort = True


# ---------------------------------------------------------------------------
# GaitAnalysisTab
# ---------------------------------------------------------------------------
class GaitAnalysisTab(QWidget):
    """Batch video analysis with parameter preview + status tracking."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._app_state = None
        self._worker: Optional[BatchAnalysisWorker] = None
        self._is_running = False

        # Preview state (single-video parameter tuning)
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._preview_tick)
        self._preview_cap: Optional[cv2.VideoCapture] = None
        self._preview_bg: Optional[RollingMedianBackground] = None
        self._preview_accum: Optional[np.ndarray] = None
        self._preview_count = 0
        self._preview_fps = 60.0

        self._build_ui()

    def set_app_state(self, app_state) -> None:
        self._app_state = app_state

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setSpacing(8)
        main.setContentsMargins(8, 8, 8, 8)

        # -- Project info ----------------------------------------------------
        self.project_label = QLabel("当前项目: 未打开")
        main.addWidget(self.project_label)

        # -- Video batch table -----------------------------------------------
        table_gb = QGroupBox("视频列表")
        tl = QVBoxLayout(table_gb)
        self.batch_table = VideoBatchTable()
        self.batch_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.batch_table.setMaximumHeight(130)
        tl.addWidget(self.batch_table)
        btn_row = QHBoxLayout()
        load_project_btn = QPushButton("加载项目 videos")
        load_project_btn.clicked.connect(self._on_load_project_videos)
        btn_row.addWidget(load_project_btn)
        add_video_btn = QPushButton("增加视频")
        add_video_btn.clicked.connect(self._on_add_video)
        btn_row.addWidget(add_video_btn)
        del_video_btn = QPushButton("删除视频")
        del_video_btn.clicked.connect(self._on_delete_video)
        btn_row.addWidget(del_video_btn)
        btn_row.addStretch()
        tl.addLayout(btn_row)
        main.addWidget(table_gb)

        # -- Preview controls -------------------------------------------------
        preview_ctrl = QHBoxLayout()
        self.preview_btn = QPushButton("▶ 开始采集")
        self.preview_btn.clicked.connect(self._on_preview_video)
        preview_ctrl.addWidget(self.preview_btn)
        self.stop_preview_btn = QPushButton("■ 停止")
        self.stop_preview_btn.setEnabled(False)
        self.stop_preview_btn.clicked.connect(self._on_stop_preview)
        preview_ctrl.addWidget(self.stop_preview_btn)
        preview_ctrl.addStretch()
        self.preview_status = QLabel("选择视频后点击开始")
        preview_ctrl.addWidget(self.preview_status)
        main.addLayout(preview_ctrl)

        # -- 参数微调 (same layout as 新建实验 tab) ---------------------------
        tune_gb = QGroupBox("参数微调")
        tune_l = QVBoxLayout(tune_gb)
        tune_l.setSpacing(4)

        # Video preview (top, stretch=1)
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
        tune_l.addLayout(video_box, stretch=1)

        # Footprint accumulation (bottom, stretch=1) — with threshold controls
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
        tune_l.addLayout(fp_box, stretch=1)

        main.addWidget(tune_gb)

        # -- Frame info --------------------------------------------------------
        self.frame_label = QLabel("Frame: 0 | FPS: --")
        self.frame_label.setStyleSheet("font-family: monospace; color: #888;")
        main.addWidget(self.frame_label)

        # -- Batch control row ------------------------------------------------
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 视频分析")
        self.start_btn.clicked.connect(self._on_start_batch)
        ctrl_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addStretch()
        self.status_label = QLabel("就绪")
        ctrl_row.addWidget(self.status_label)
        main.addLayout(ctrl_row)
        main.addStretch()

    # ------------------------------------------------------------------
    # Slots — video loading
    # ------------------------------------------------------------------
    def _on_load_project_videos(self) -> None:
        if self._app_state is None or not self._app_state.project.is_valid:
            QMessageBox.warning(self, "提示", "请先在「新建项目」中打开一个项目。")
            return
        vdir = Path(self._app_state.project.project_path) / "videos"
        if not vdir.exists():
            QMessageBox.warning(self, "提示", f"项目 videos 目录不存在:\n{vdir}")
            return
        import glob as _g
        self.batch_table.setRowCount(0)
        for p in sorted(_g.iglob(f"{vdir}/*.mp4")):
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            name = Path(p).stem
            self.batch_table.setItem(row, 0, QTableWidgetItem(name))
            self.batch_table.setItem(row, 1, QTableWidgetItem(p))
            self.batch_table.setItem(row, 2, QTableWidgetItem("Ready"))

    def _on_add_video(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        for p in paths:
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            name = Path(p).stem
            self.batch_table.setItem(row, 0, QTableWidgetItem(name))
            self.batch_table.setItem(row, 1, QTableWidgetItem(p))
            self.batch_table.setItem(row, 2, QTableWidgetItem("Ready"))

    def _on_delete_video(self) -> None:
        row = self.batch_table.currentRow()
        if row >= 0:
            self.batch_table.removeRow(row)

    # ------------------------------------------------------------------
    # Single-video preview (参数微调)
    # ------------------------------------------------------------------
    def _on_preview_video(self) -> None:
        row = self.batch_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先在视频列表中选择一行。")
            return
        vid_item = self.batch_table.item(row, 1)
        if vid_item is None or not vid_item.text().strip():
            return
        video_path = vid_item.text().strip()

        # Release previous
        self._stop_preview_internal()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开视频:\n{video_path}")
            return
        self._preview_cap = cap
        self._preview_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        self._preview_bg = RollingMedianBackground(warmup_frames=30)
        self._preview_accum = None
        self._preview_count = 0

        self.preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(True)
        self.preview_status.setText("预览中...")
        self._preview_timer.start(33)  # ~30 fps

    def _on_stop_preview(self) -> None:
        self._stop_preview_internal()
        self.preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.preview_status.setText("已停止")

    def _stop_preview_internal(self) -> None:
        self._preview_timer.stop()
        if self._preview_cap is not None:
            self._preview_cap.release()
            self._preview_cap = None
        self._preview_bg = None
        self._preview_accum = None
        self._preview_count = 0

    def _preview_tick(self) -> None:
        """Single-video preview loop — same pattern as 新建实验 tab."""
        cap = self._preview_cap
        if cap is None:
            return
        ret, frame_bgr = cap.read()
        if not ret:
            # End of video — auto-stop, same as experiment tab.
            self._on_stop_preview()
            return

        h, w = frame_bgr.shape[:2]
        self._preview_count += 1

        # Background model + detection
        bg = self._preview_bg
        if bg is not None:
            bg.update(frame_bgr)
        try:
            bg_med = bg.get_median() if bg is not None else None
            if bg_med is not None and bg is not None and bg.is_ready:
                bg_G = bg_med[:, :, 1].astype(np.float32)
            else:
                bg_G = np.zeros((h, w), dtype=np.float32)
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
                body_axis=body_axis,
                in_stance_threshold_px=30,
            )
        except Exception:
            from deepgait3.core._legacy.footprint_v2 import FootprintSequence
            seq = FootprintSequence()

        # Accumulate footprints
        self._update_footprint(frame_bgr, seq)

        # Render every 3rd frame
        if self._preview_count % 3 == 0:
            self._show_preview(frame_bgr)
            self._show_footprint()

        self.frame_label.setText(
            f"Frame: {self._preview_count} | FPS: {self._preview_fps:.0f}"
        )

    # ------------------------------------------------------------------
    # Footprint accumulation (same as 新建实验 tab)
    # ------------------------------------------------------------------
    def _update_footprint(self, frame_bgr: np.ndarray, seq) -> None:
        h, w = frame_bgr.shape[:2]
        if self._preview_accum is None:
            self._preview_accum = np.zeros((h, w, 3), dtype=np.uint8)
        green = frame_bgr[:, :, 1].astype(np.uint8)
        bg = self._preview_bg.get_median() if self._preview_bg is not None else None
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
            current = self._preview_accum[y0:y1, x0:x1, 1]
            np.maximum(current, paw_signal, out=current)

    def _show_preview(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        pw = max(self.video_preview.width(), 480)
        ph = max(self.video_preview.height(), 96)
        target_aspect = 5.0
        widget_aspect = pw / max(ph, 1)
        if widget_aspect > target_aspect:
            nh = ph; nw = int(nh * target_aspect)
        else:
            nw = pw; nh = int(nw / target_aspect)
        nw, nh = max(1, nw), max(1, nh)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (nw, nh))
        qimg = QImage(img.data, nw, nh, img.strides[0], QImage.Format.Format_RGB888)
        self.video_preview.setPixmap(QPixmap.fromImage(qimg))

    def _show_footprint(self) -> None:
        if self._preview_accum is None:
            return
        display = self._preview_accum.copy()
        g = display[:, :, 1]
        # Opening morphology to break tail/belly streaks
        g_mask = (g > 5).astype(np.uint8) * 255
        if g_mask.any():
            open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            cleaned = cv2.morphologyEx(g_mask, cv2.MORPH_OPEN, open_k)
            g = cv2.bitwise_and(g, cleaned)
        # Green-on-black with mask-protected background
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
        display[:, :, 1] = np.where(
            mask, (norm * 255.0 * bright).clip(0, 255).astype(np.uint8), 0,
        )
        display[:, :, 2] = 0
        h, w = display.shape[:2]
        pw = max(self.footprint_view.width(), 480)
        ph = max(self.footprint_view.height(), 96)
        target_aspect = 5.0
        widget_aspect = pw / max(ph, 1)
        if widget_aspect > target_aspect:
            nh = ph; nw = int(nh * target_aspect)
        else:
            nw = pw; nh = int(nw / target_aspect)
        nw, nh = max(1, nw), max(1, nh)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (nw, nh))
        qimg = QImage(img.data, nw, nh, img.strides[0], QImage.Format.Format_RGB888)
        self.footprint_view.setPixmap(QPixmap.fromImage(qimg))

    # ------------------------------------------------------------------
    # Slots — batch analysis
    # ------------------------------------------------------------------
    def _on_start_batch(self) -> None:
        entries = self._collect_entries()
        if not entries:
            QMessageBox.warning(self, "提示", "请先加载项目videos。")
            return
        if self._app_state and self._app_state.project.is_valid:
            base_dir = Path(self._app_state.project.project_path) / "data"
        else:
            base_dir = Path.home() / "deepgait_output"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Stop preview if running
        self._stop_preview_internal()

        self._is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("批量分析中...")

        # Set all statuses to "等待"
        for row in range(self.batch_table.rowCount()):
            self.batch_table.setItem(row, 2, QTableWidgetItem("等待"))

        self._worker = BatchAnalysisWorker(
            entries, base_dir,
            tau_paw=self.tau_spin.value(),
            min_area_px=self.paw_spin.value(),
            px_per_mm=(
                self._app_state.project.px_per_mm
                if self._app_state and self._app_state.project.is_valid
                else 3.0
            ),
        )
        self._worker.video_started.connect(self._on_video_started)
        self._worker.video_progress.connect(self._on_video_progress)
        self._worker.video_completed.connect(self._on_video_completed)
        self._worker.all_completed.connect(self._on_all_completed)
        self._worker.error_occurred.connect(self._on_analysis_error)
        self._worker.start()

    def _collect_entries(self) -> List[Dict[str, str]]:
        entries = []
        for row in range(self.batch_table.rowCount()):
            aid_item = self.batch_table.item(row, 0)
            vid_item = self.batch_table.item(row, 1)
            if vid_item and vid_item.text().strip():
                entries.append({
                    "animal_id": aid_item.text().strip() if aid_item else f"unknown_{row}",
                    "video_path": vid_item.text().strip(),
                })
        return entries

    def _on_video_started(self, idx: int, animal_id: str) -> None:
        if 0 <= idx < self.batch_table.rowCount():
            self.batch_table.setItem(idx, 2, QTableWidgetItem("分析中..."))

    def _on_video_progress(self, idx: int, frame: int, total: int) -> None:
        pct = min(100, int(100 * frame / max(total, 1)))
        if 0 <= idx < self.batch_table.rowCount():
            self.batch_table.setItem(
                idx, 2, QTableWidgetItem(f"{frame}/{total} ({pct}%)"),
            )

    def _on_video_completed(self, idx: int, output_dir: str, metrics: dict) -> None:
        if 0 <= idx < self.batch_table.rowCount():
            self.batch_table.setItem(idx, 2, QTableWidgetItem("完成 ✓"))

    def _on_all_completed(self, results: list) -> None:
        self._is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"全部完成: {len(results)} 个视频")

    def _on_analysis_error(self, msg: str) -> None:
        logger.error("Batch analysis error: %s", msg)

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.abort()
            self._worker.wait(3000)
        self._is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已停止")

    # ------------------------------------------------------------------
    def update_project_display(self) -> None:
        if self._app_state and self._app_state.project.is_valid:
            p = self._app_state.project
            self.project_label.setText(
                f"当前项目: {p.project_name} ({p.project_path})",
            )
        else:
            self.project_label.setText("当前项目: 未打开")
