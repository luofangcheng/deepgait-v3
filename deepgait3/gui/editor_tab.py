"""Tab 6: Gait Editor — manual stance/swing correction with slider.

Inspired by VGL's GaitWindow:
- 4 paw step plots (one per paw)
- QSlider to scrub through frames
- Click on any chart point to flip stance/swing at that frame
- AutoCorrect button to merge short spurious segments
- Real-time parameter recompute
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph import PlotWidget

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QFileDialog,
    QGroupBox, QLabel, QSlider, QSpinBox, QMessageBox, QTableWidget,
    QTableWidgetItem, QSplitter, QCheckBox, QDoubleSpinBox,
)

from deepgait3.core._legacy import pipeline as gait_pipeline
from deepgait3.core._legacy import results as core_results
from deepgait3.core._legacy import gait_algorithms as ga
from deepgait3.gui.gait_tab import _TABLE_COLUMNS


# Step plot colors (4 paws: left/right, fore/hind)
PAW_COLORS = {
    "LeftFore":  "#1976D2",   # blue
    "RightFore": "#D32F2F",   # red
    "LeftHind":  "#388E3C",   # green
    "RightHind": "#F57C00",   # orange
}


class PawStepPlot(PlotWidget):
    """A single paw's stance/swing step plot with click-to-flip support.

    Click on the plot flips the stance/swing at the clicked frame.
    """

    frame_clicked = Signal(int)   # emits frame index

    def __init__(self, paw_name: str) -> None:
        super().__init__()
        self.paw_name = paw_name
        self._data: np.ndarray = np.array([], dtype=int)
        self._corrected_mask: np.ndarray = np.array([], dtype=bool)
        self._curve: pg.PlotDataItem | None = None

        self.setMinimumHeight(120)
        self.setLabel("left", "Stance")
        self.setLabel("bottom", "Frame")
        self.setYRange(-0.1, 1.1)
        self.setMouseEnabled(x=True, y=False)
        color = PAW_COLORS.get(paw_name, "#333333")
        self._color = color
        self.getPlotItem().setTitle(paw_name, color=color)

        # Connect click
        self.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def set_data(self, in_stance: np.ndarray) -> None:
        """Set the in_stance array and refresh plot."""
        self._data = in_stance.copy()
        self._corrected_mask = np.zeros_like(in_stance, dtype=bool)
        self._refresh()

    def set_corrected_data(self, in_stance: np.ndarray, corrected_mask: np.ndarray) -> None:
        """Set externally corrected data (e.g., from AutoCorrect)."""
        self._data = in_stance.copy()
        self._corrected_mask = corrected_mask.copy()
        self._refresh()

    def flip_frame(self, frame: int) -> None:
        """Flip stance/swing at the given frame."""
        if 0 <= frame < len(self._data):
            self._data[frame] = 1 - self._data[frame]
            self._corrected_mask[frame] = True
            self._refresh()

    def get_data(self) -> np.ndarray:
        return self._data.copy()

    def get_corrected_mask(self) -> np.ndarray:
        return self._corrected_mask.copy()

    def _refresh(self) -> None:
        self.clear()
        if self._data.size == 0:
            return
        x = np.arange(len(self._data))
        # Build step data manually: alternate vertical and horizontal segments
        # so len(x_step) == len(y_step).
        n = len(self._data)
        x_step: list[float] = []
        y_step: list[float] = []
        for i in range(n):
            x_step.append(float(i))
            y_step.append(float(self._data[i]))
            if i < n - 1 and self._data[i] != self._data[i + 1]:
                # Insert midpoint with same y (horizontal segment ends here)
                x_step.append(float(i + 1))
                y_step.append(float(self._data[i]))
        self.plot(x_step, y_step, pen=pg.mkPen(self._color, width=1.5))
        # Fill stance regions
        if x_step:
            fill = pg.FillBetweenItem(
                pg.PlotDataItem(x_step, y_step),
                pg.PlotDataItem([0, n - 1], [0, 0]),
                brush=pg.mkBrush(self._color + "60"),
            )
            self.addItem(fill)
        # Mark corrected frames with red dots
        if self._corrected_mask.any():
            corr_x = x[self._corrected_mask]
            corr_y = self._data[self._corrected_mask]
            self.plot(corr_x, corr_y, pen=None, symbol="o",
                      symbolSize=8, symbolBrush="red", symbolPen=None)

    def _on_mouse_clicked(self, evt) -> None:
        if evt.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.getPlotItem().getViewBox()
        scene_pos = evt.scenePos()
        if not self.sceneBoundingRect().contains(scene_pos):
            return
        mouse_point = vb.mapSceneToView(scene_pos)
        frame = int(round(mouse_point.x()))
        if 0 <= frame < len(self._data):
            self.frame_clicked.emit(frame)

    def set_cursor(self, frame: int) -> None:
        """Draw a vertical line at the current frame."""
        self.removeItem(self._cursor) if hasattr(self, "_cursor") else None
        if not (0 <= frame < len(self._data)):
            return
        self._cursor = pg.InfiniteLine(
            pos=frame, angle=90, pen=pg.mkPen("yellow", width=2)
        )
        self.addItem(self._cursor)


class EditorTab(QWidget):
    """Tab 6: Gait editor for manual stance/swing correction."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results: core_results.GaitResults | None = None
        self._plots: dict[str, PawStepPlot] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(10)

        # --- Top: file / status bar ---
        top = QHBoxLayout()
        self.load_btn = QPushButton("加载步态结果")
        self.load_btn.clicked.connect(self._on_load)
        top.addWidget(self.load_btn)
        self.status_label = QLabel("未加载数据")
        top.addWidget(self.status_label, stretch=1)
        self.export_btn = QPushButton("导出 CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        top.addWidget(self.export_btn)
        main.addLayout(top)

        # --- Splitter: left = plots, right = params + auto-correct ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter, stretch=1)

        # --- Left: 2x2 grid of step plots ---
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(4)
        # Order: LeftFore, RightFore, LeftHind, RightHind
        paw_order = ["LeftFore", "RightFore", "LeftHind", "RightHind"]
        for i, paw in enumerate(paw_order):
            plot = PawStepPlot(paw)
            plot.frame_clicked.connect(self._on_frame_clicked)
            self._plots[paw] = plot
            grid.addWidget(plot, i // 2, i % 2)
        splitter.addWidget(grid_widget)

        # --- Right: parameters + auto-correct ---
        right = QWidget()
        right_layout = QVBoxLayout(right)

        # Frame scrubber
        scrub_group = QGroupBox("帧定位")
        scrub_layout = QVBoxLayout(scrub_group)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        scrub_layout.addWidget(self.frame_slider)
        self.frame_label = QLabel("帧: —")
        scrub_layout.addWidget(self.frame_label)
        # Flip button
        self.flip_btn = QPushButton("翻转当前帧 (stance ↔ swing)")
        self.flip_btn.setEnabled(False)
        self.flip_btn.clicked.connect(self._on_flip_current)
        scrub_layout.addWidget(self.flip_btn)
        right_layout.addWidget(scrub_group)

        # Auto-correct
        ac_group = QGroupBox("自动校正")
        ac_layout = QVBoxLayout(ac_group)
        ac_layout.addWidget(QLabel("阈值 = avg_segment_length / 4"))
        self.ac_check = QCheckBox("启用 AutoCorrect")
        self.ac_check.setChecked(True)
        ac_layout.addWidget(self.ac_check)
        self.run_ac_btn = QPushButton("运行 AutoCorrect")
        self.run_ac_btn.setEnabled(False)
        self.run_ac_btn.clicked.connect(self._on_run_auto_correct)
        ac_layout.addWidget(self.run_ac_btn)
        right_layout.addWidget(ac_group)

        # Parameters table
        self.params_table = QTableWidget()
        self.params_table.setColumnCount(len(_TABLE_COLUMNS))
        self.params_table.setHorizontalHeaderLabels(_TABLE_COLUMNS)
        self.params_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.params_table, stretch=1)

        # Recompute button
        self.recompute_btn = QPushButton("重算参数")
        self.recompute_btn.setEnabled(False)
        self.recompute_btn.clicked.connect(self._on_recompute)
        right_layout.addWidget(self.recompute_btn)

        splitter.addWidget(right)
        splitter.setSizes([700, 400])

    # -----------------------------------------------------------------------
    # Slots
    # -----------------------------------------------------------------------

    def _on_load(self) -> None:
        # Prefer the shared state (from GaitTab).  If not available, ask user.
        main = self.window()
        if isinstance(main, QWidget) and getattr(main, "current_gait_results", None) is not None:
            self._load_results(main.current_gait_results)
            return
        # Fallback: re-analyze from CSV
        path, _ = QFileDialog.getOpenFileName(
            self, "选择步态 CSV", "", "CSV (*.csv);;所有文件 (*.*)"
        )
        if not path:
            return
        try:
            res = gait_pipeline.analyze(path, fps=100, mode="catwalk")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析失败: {e}")
            return
        self._load_results(res)

    def _load_results(self, res: core_results.GaitResults) -> None:
        self._results = res
        self.status_label.setText(
            f"已加载: {res.n_frames} 帧 @ {res.fps} FPS"
        )
        # Enable controls
        self.frame_slider.setRange(0, res.n_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(True)
        self.flip_btn.setEnabled(True)
        self.run_ac_btn.setEnabled(True)
        self.recompute_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        # Populate plots
        for paw_name, paw in res.paws.items():
            if paw_name in self._plots:
                self._plots[paw_name].set_data(paw.in_stance)
        # Recompute params
        self._on_recompute()
        # Update shared state
        main = self.window()
        if isinstance(main, QWidget) and hasattr(main, "current_gait_results"):
            main.current_gait_results = res

    def _on_slider_changed(self, value: int) -> None:
        self.frame_label.setText(f"帧: {value} / {self.frame_slider.maximum()}")
        for plot in self._plots.values():
            plot.set_cursor(value)

    def _on_frame_clicked(self, frame: int) -> None:
        # Move slider to clicked frame (cascades to set_cursor)
        self.frame_slider.setValue(frame)
        # Also flip at that frame
        for plot in self._plots.values():
            plot.flip_frame(frame)
        self._on_recompute()

    def _on_flip_current(self) -> None:
        frame = self.frame_slider.value()
        for plot in self._plots.values():
            plot.flip_frame(frame)
        self._on_recompute()

    def _on_run_auto_correct(self) -> None:
        if not self._results:
            return
        for paw_name, plot in self._plots.items():
            data = plot.get_data()
            corrected = ga.auto_correct(data)
            mask = corrected != data  # mark changed frames
            plot.set_corrected_data(corrected, mask)
        self._on_recompute()
        QMessageBox.information(self, "完成", "AutoCorrect 已应用到所有爪子")

    def _on_recompute(self) -> None:
        """Recompute all per-paw metrics from current (possibly edited) data."""
        if not self._results:
            return
        for paw_name, paw in self._results.paws.items():
            new_data = self._plots[paw_name].get_data()
            paw.in_stance = new_data
            basics = ga.calculate_gait_basics(new_data, self._results.fps)
            paw.stance_duration_ms = basics.stance_duration_ms
            paw.swing_duration_ms = basics.swing_duration_ms
            paw.n_strides = basics.n_strides
            # Reuse existing stride lengths / angles / widths (frame-level data
            # unchanged by stance/swing edits)
        # Update table
        self._populate_table()

    def _populate_table(self) -> None:
        if not self._results:
            return
        self.params_table.setRowCount(len(self._results.paws))
        for row, (name, paw) in enumerate(self._results.paws.items()):
            self.params_table.setItem(row, 0, QTableWidgetItem(name))
            self.params_table.setItem(row, 1, QTableWidgetItem(f"{paw.stance_duration_ms:.1f}"))
            self.params_table.setItem(row, 2, QTableWidgetItem(f"{paw.swing_duration_ms:.1f}"))
            self.params_table.setItem(row, 3, QTableWidgetItem(str(paw.n_strides)))
            self.params_table.setItem(row, 4, QTableWidgetItem(f"{paw.stride_length_mean:.2f}"))
            self.params_table.setItem(row, 5, QTableWidgetItem(f"{paw.stride_length_variability:.2f}"))
            self.params_table.setItem(row, 6, QTableWidgetItem(f"{paw.stride_frequency_hz:.2f}"))
            self.params_table.setItem(row, 7, QTableWidgetItem(f"{paw.paw_angle_mean_deg:.1f}"))
            self.params_table.setItem(row, 8, QTableWidgetItem(f"{paw.stance_width_mean:.2f}"))
            sym = self._results.gait_symmetry_index if name in ("LeftFore", "LeftHind") else ""
            self.params_table.setItem(row, 9, QTableWidgetItem(
                f"{sym:.3f}" if sym != "" else ""))
        self.params_table.resizeColumnsToContents()

    def _on_export(self) -> None:
        if not self._results:
            return
        from deepgait3.core._legacy import gait_export
        path, _ = QFileDialog.getSaveFileName(self, "导出 CSV", "edited.csv", "CSV (*.csv)")
        if path:
            gait_export.to_summary_csv(self._results, path)
            QMessageBox.information(self, "导出成功", f"已保存: {path}")
