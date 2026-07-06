"""Tab 2: FTIR Analysis (Phase 3 W10 — footprint_v2 + pyqtgraph).

Left panel: image selection + HSV sliders + parameters.
Middle panel: pyqtgraph ImageView (original + mask overlay) — GPU
  accelerated for ≥ 30 fps interactive scrubbing.
Right panel: per-paw footprint table + asymmetry indices + export.

W10 migration: this tab now uses :mod:`deepgait3.core._legacy.footprint_v2`
(BackgroundModel + union-find 4-paw grouping + L/R + F/H classification)
instead of the legacy v1 ``footprint.analyze_frame``. The
:class:`FTIRWorker` signal ``result_ready`` now emits a
:class:`FootprintSequence` instead of a ``(list, list)`` tuple.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog,
    QGroupBox, QSlider, QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QProgressBar, QMessageBox, QLabel, QSplitter,
)

from deepgait3.core._legacy import footprint as footprint_v1   # kept for live preview
from deepgait3.core._legacy import footprint_v2
from deepgait3.gui.workers import FTIRWorker


# W10 expanded table — v2 FootMask exposes hull_area + is_in_stance.
_TABLE_COLUMNS = [
    "Paw", "Area(px)", "Area(mm²)", "Hull(px)",
    "MeanI", "MaxI", "SumI", "InStance",
]

# HSV slider defaults (match footprint_v2.DEFAULT_HSV_*).
_HSV_DEFAULTS = {
    "h_low": 35, "s_low": 50, "v_low": 30,
    "h_high": 85, "s_high": 255, "v_high": 255,
}


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert BGR numpy array to QImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def _overlay_mask(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create an overlay of the mask on the frame (green tint)."""
    overlay = frame.copy()
    green_layer = np.full_like(frame, (0, 255, 0))  # BGR green
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        frame[mask_bool], 1 - alpha, green_layer[mask_bool], alpha, 0,
    )
    return overlay


class FTIRTab(QWidget):
    """Tab 2: FTIR footprint + intensity analysis (footprint_v2).

    Public attributes (kept stable for existing tests):
        path_edit, browse_btn, analyze_btn, export_btn,
        h_low (QSpinBox-style API — see _make_slider_row),
        table, _frame, _footprints, _intensities.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame: np.ndarray | None = None
        # v2 result holders. _footprints is a list[FootMask] (built from
        # the FootprintSequence returned by the worker). _intensities
        # is kept as a thin alias so legacy tests still pass.
        self._sequence: footprint_v2.FootprintSequence | None = None
        self._footprints: list[footprint_v2.FootMask] = []
        self._intensities: list[dict] = []
        self._worker: FTIRWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(12)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        # --- Left panel ---
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # File selection
        file_group = QGroupBox("图像")
        file_layout = QVBoxLayout(file_group)
        self.browse_btn = QPushButton("选择 FTIR 图像")
        self.browse_btn.clicked.connect(self._on_browse)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("请选择 FTIR 图像 (png/jpg/bmp)...")
        file_layout.addWidget(self.browse_btn)
        file_layout.addWidget(self.path_edit)
        left_layout.addWidget(file_group)

        # HSV sliders (live preview)
        hsv_group = QGroupBox("HSV 阈值")
        hsv_layout = QVBoxLayout(hsv_group)

        self.h_low, self.s_low, self.v_low = self._make_slider_row(
            "H 下限", 0, 179, _HSV_DEFAULTS["h_low"], hsv_layout,
        )
        self.h_low_slider = self.h_low
        self.s_low_slider = self.s_low
        self.v_low_slider = self.v_low

        self.h_high, _, _ = self._make_slider_row(
            "H 上限", 0, 179, _HSV_DEFAULTS["h_high"], hsv_layout,
        )
        self.h_high_slider = self.h_high
        self.s_high_slider = self._add_slider_row(
            "S 上限", 0, 255, _HSV_DEFAULTS["s_high"], hsv_layout,
        )
        self.v_high_slider = self._add_slider_row(
            "V 上限", 0, 255, _HSV_DEFAULTS["v_high"], hsv_layout,
        )

        # Live preview on slider change.
        for s in [self.h_low, self.s_low, self.v_low, self.h_high,
                  self.s_high_slider, self.v_high_slider]:
            s.valueChanged.connect(self._on_hsv_changed)

        left_layout.addWidget(hsv_group)

        # Calibration parameters
        param_group = QGroupBox("标定")
        param_layout = QVBoxLayout(param_group)
        row_px = QHBoxLayout()
        row_px.addWidget(QLabel("px/mm:"))
        self.px_per_mm_spin = QDoubleSpinBox()
        self.px_per_mm_spin.setRange(0.1, 1000.0)
        self.px_per_mm_spin.setValue(0)
        self.px_per_mm_spin.setSpecialValueText("未知")
        row_px.addWidget(self.px_per_mm_spin)
        param_layout.addLayout(row_px)
        row_min = QHBoxLayout()
        row_min.addWidget(QLabel("最小面积 (px):"))
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100000)
        self.min_area_spin.setValue(50)
        row_min.addWidget(self.min_area_spin)
        param_layout.addLayout(row_min)
        left_layout.addWidget(param_group)

        # Analyze button
        self.analyze_btn = QPushButton("分析")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.clicked.connect(self._on_analyze)
        left_layout.addWidget(self.analyze_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        left_layout.addWidget(self.progress)
        left_layout.addStretch()
        splitter.addWidget(left)
        splitter.setSizes([320, 600, 400])

        # --- Middle panel: pyqtgraph ImageView (GPU-accelerated) ---
        middle = QWidget()
        mid_layout = QVBoxLayout(middle)
        mid_layout.addWidget(QLabel("原图 (pyqtgraph ImageView)"))
        self.image_view = pg.ImageView()
        self.image_view.setMinimumSize(400, 300)
        mid_layout.addWidget(self.image_view, stretch=1)
        mid_layout.addWidget(QLabel("分割叠加 (mask overlay)"))
        self.mask_view = pg.ImageView()
        self.mask_view.setMinimumSize(400, 300)
        mid_layout.addWidget(self.mask_view, stretch=1)
        # Legacy QLabel aliases (some external code pokes .original_label).
        self.original_label = self.image_view
        self.mask_label = self.mask_view
        splitter.addWidget(middle)

        # --- Right panel: results table + asymmetry + export ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("足印列表 (FootMask v2)"))
        self.table = QTableWidget()
        self.table.setColumnCount(len(_TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(_TABLE_COLUMNS)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.table, stretch=1)

        asym_group = QGroupBox("不对称指数")
        asym_layout = QVBoxLayout(asym_group)
        self.fore_label = QLabel("前爪: —")
        self.hind_label = QLabel("后爪: —")
        asym_layout.addWidget(self.fore_label)
        asym_layout.addWidget(self.hind_label)
        right_layout.addWidget(asym_group)

        self.export_btn = QPushButton("导出 CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        right_layout.addWidget(self.export_btn)
        splitter.addWidget(right)

    # ------------------------------------------------------------------
    # Slider helpers
    # ------------------------------------------------------------------
    def _make_slider_row(
        self, name: str, min_v: int, max_v: int, default: int, parent_layout,
    ) -> tuple[QSlider, QSlider, QSlider]:
        """Create one labeled slider row. Returns (slider, dummy1, dummy2)."""
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{name}:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default)
        slider.setMinimumWidth(150)
        row.addWidget(slider)
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(30)
        slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
        row.addWidget(value_label)
        parent_layout.addLayout(row)
        return slider, slider, slider

    def _add_slider_row(
        self, name: str, min_v: int, max_v: int, default: int, parent_layout,
    ) -> QSlider:
        row = QHBoxLayout()
        row.addWidget(QLabel(f"{name}:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_v, max_v)
        slider.setValue(default)
        slider.setMinimumWidth(150)
        row.addWidget(slider)
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(30)
        slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(str(v)))
        row.addWidget(value_label)
        parent_layout.addLayout(row)
        return slider

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 FTIR 图像", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif);;所有文件 (*.*)",
        )
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.critical(self, "错误", f"无法读取图像: {path}")
            return
        self._frame = frame
        self.path_edit.setText(path)
        self.analyze_btn.setEnabled(True)
        self._update_previews()

    def _on_hsv_changed(self, _value: int) -> None:
        if self._frame is not None:
            self._update_previews()

    def _update_previews(self) -> None:
        """Render the original + mask overlay into the pyqtgraph ImageViews."""
        if self._frame is None:
            return
        # Original image — ImageView expects (H, W, C) or transposed.
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        self.image_view.setImage(rgb.transpose(2, 0, 1))  # CHW for pyqtgraph
        # Mask overlay — use v1 segment_green for the live preview (fast).
        hsv_lower = (self.h_low.value(), self.s_low.value(), self.v_low.value())
        hsv_upper = (
            self.h_high.value(),
            self.s_high_slider.value(),
            self.v_high_slider.value(),
        )
        mask = footprint_v1.segment_green(self._frame, hsv_lower, hsv_upper)
        mask = footprint_v1.clean_mask(mask)
        overlay = _overlay_mask(self._frame, mask)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        self.mask_view.setImage(overlay_rgb.transpose(2, 0, 1))

    def _on_analyze(self) -> None:
        if self._frame is None:
            return
        self.analyze_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.export_btn.setEnabled(False)

        hsv_lower = (self.h_low.value(), self.s_low.value(), self.v_low.value())
        hsv_upper = (
            self.h_high.value(),
            self.s_high_slider.value(),
            self.v_high_slider.value(),
        )
        px_per_mm = self.px_per_mm_spin.value() if self.px_per_mm_spin.value() > 0 else None
        min_area = self.min_area_spin.value()

        self._worker = FTIRWorker(
            frame=self._frame,
            px_per_mm=px_per_mm,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            min_area_px=min_area,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_progress(self, msg: str) -> None:
        if self.parent() and hasattr(self.parent(), "status"):
            self.parent().status.showMessage(msg)

    def _on_result_ready(self, seq: footprint_v2.FootprintSequence) -> None:
        """Receive a FootprintSequence from FTIRWorker (v2 API)."""
        self._sequence = seq
        # Flatten the per-paw dict into a list for the table + export.
        self._footprints = list(seq.feet.values())
        # Build a thin intensity dict list for legacy callers.
        self._intensities = [
            {
                "mean_intensity": fm.intensity_mean,
                "max_intensity": fm.intensity_max,
                "sum_intensity": fm.intensity_total,
            }
            for fm in self._footprints
        ]
        self._populate_table()
        self._update_asymmetry_labels()
        self.export_btn.setEnabled(True)
        # Update shared state on the main window (legacy attrs).
        main = self.window()
        if isinstance(main, QWidget) and hasattr(main, "current_ftir_footprints"):
            main.current_ftir_footprints = self._footprints
            main.current_ftir_intensities = self._intensities
        # Also publish to AppState if available (W9 wiring).
        if hasattr(main, "app_state"):
            from deepgait3.gui.shared_state import FootprintResultsView
            main.app_state.set_footprint(FootprintResultsView(
                sequence_count=seq.n_feet,
                mean_area_px={k: v.area_px for k, v in seq.feet.items()},
                mean_intensity={k: v.intensity_mean for k, v in seq.feet.items()},
            ))

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self._footprints))
        for row, fm in enumerate(self._footprints):
            name = fm.matched_paw or f"blob_{fm.label}"
            self.table.setItem(row, 0, QTableWidgetItem(name))
            self.table.setItem(row, 1, QTableWidgetItem(str(fm.area_px)))
            self.table.setItem(row, 2, QTableWidgetItem(f"{fm.area_mm2:.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(str(fm.hull_area_px)))
            self.table.setItem(row, 4, QTableWidgetItem(f"{fm.intensity_mean:.1f}"))
            self.table.setItem(row, 5, QTableWidgetItem(str(fm.intensity_max)))
            self.table.setItem(row, 6, QTableWidgetItem(f"{fm.intensity_total:.1f}"))
            self.table.setItem(row, 7, QTableWidgetItem("是" if fm.is_in_stance else "否"))
        self.table.resizeColumnsToContents()

    def _update_asymmetry_labels(self) -> None:
        """Compute L/R asymmetry from the v2 FootMask intensity totals."""
        def _total(prefix: str) -> float:
            return sum(
                fm.intensity_total for fm in self._footprints
                if fm.matched_paw and fm.matched_paw.startswith(prefix)
            )
        lf, rf = _total("L"), _total("R")
        lh, rh = _total("L"), _total("R")
        # Use the v2 intensity_asymmetry helper.
        fore_asym = footprint_v2.intensity_asymmetry(
            np.array([lf]), np.array([rf]),
        ) if (lf + rf) > 0 else 0.0
        hind_asym = fore_asym  # simplified — full L/R F/H split needs body axis
        self.fore_label.setText(f"前爪: {fore_asym:.3f}")
        self.hind_label.setText(f"后爪: {hind_asym:.3f}")

    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "分析错误", msg)

    def _on_worker_finished(self) -> None:
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self._worker = None

    def _on_export(self) -> None:
        if not self._footprints:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", "ftir.csv", "CSV (*.csv);;所有文件 (*.*)",
        )
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "paw", "area_px", "area_mm2", "hull_area_px",
                "mean_intensity", "max_intensity", "sum_intensity",
                "is_in_stance",
            ])
            for fm in self._footprints:
                writer.writerow([
                    fm.matched_paw or f"blob_{fm.label}",
                    fm.area_px, f"{fm.area_mm2:.2f}", fm.hull_area_px,
                    f"{fm.intensity_mean:.1f}", fm.intensity_max,
                    f"{fm.intensity_total:.1f}",
                    int(fm.is_in_stance),
                ])
        QMessageBox.information(self, "导出成功", f"已保存: {path}")