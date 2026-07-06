"""Tab 4: Publication Charts.

Generate publication-quality matplotlib charts from gait/FTIR results.
Charts: stance/swing step plot, stride histogram, paw angle time series,
        asymmetry bar, footprint scatter.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, QMessageBox,
    QLabel, QSplitter,
)

from deepgait3.core._legacy import results as core_results, intensity


CHART_TYPES = [
    "stance/swing 阶梯图",
    "stride 长度分布",
    "paw angle 时间序列",
    "不对称指数柱状图",
    "足印面积/强度散点图",
]

COLOR_SCHEMES = ["Set1", "Set2", "Paired", "Dark2", "tab10"]


class ChartsCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for publication charts."""

    def __init__(self) -> None:
        self.fig = Figure(figsize=(8, 5), dpi=100)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def clear_chart(self) -> None:
        self.ax.clear()
        self.draw()

    def plot_stance_swing_grid(
        self, res: core_results.GaitResults, color_scheme: str = "Set1",
        show_grid: bool = True, font_size: int = 12,
    ) -> None:
        """Plot 2x2 grid of stance/swing for all 4 paws."""
        import matplotlib.pyplot as plt
        self.fig.clear()
        paws = list(res.paws.values())
        cmap = plt.get_cmap(color_scheme)
        for i, paw in enumerate(paws):
            ax = self.fig.add_subplot(2, 2, i + 1)
            if paw.in_stance.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            t = np.arange(len(paw.in_stance)) / res.fps
            ax.fill_between(t, 0, paw.in_stance, step="mid", alpha=0.4, color=cmap(0))
            ax.step(t, paw.in_stance, where="mid", color=cmap(0), linewidth=1.5)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Time (s)", fontsize=font_size)
            ax.set_ylabel("Stance", fontsize=font_size)
            ax.set_title(f"{paw.name} (n={paw.n_strides})", fontsize=font_size + 1)
            if show_grid:
                ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=font_size - 1)
        self.fig.tight_layout()
        self.draw()

    def plot_stride_histogram(
        self, res: core_results.GaitResults, color_scheme: str = "Set1",
        font_size: int = 12, show_grid: bool = True,
    ) -> None:
        """Plot stride length histogram for each paw."""
        import matplotlib.pyplot as plt
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cmap = plt.get_cmap(color_scheme)
        for i, paw in enumerate(res.paws.values()):
            nonzero = paw.stride_lengths[paw.stride_lengths > 0]
            if nonzero.size > 0:
                ax.hist(nonzero, bins=15, alpha=0.6, label=paw.name, color=cmap(i), edgecolor="white")
        ax.set_xlabel("Stride Length", fontsize=font_size)
        ax.set_ylabel("Count", fontsize=font_size)
        ax.set_title("Stride Length Distribution", fontsize=font_size + 2)
        ax.legend(fontsize=font_size)
        if show_grid:
            ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=font_size - 1)
        self.fig.tight_layout()
        self.draw()

    def plot_paw_angle_timeseries(
        self, res: core_results.GaitResults, color_scheme: str = "Set1",
        font_size: int = 12, show_grid: bool = True,
    ) -> None:
        """Plot paw angle time series for all paws."""
        import matplotlib.pyplot as plt
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cmap = plt.get_cmap(color_scheme)
        for i, paw in enumerate(res.paws.values()):
            if paw.paw_angles.size > 0:
                t = np.arange(len(paw.paw_angles)) / res.fps
                ax.plot(t, paw.paw_angles, label=paw.name, color=cmap(i), linewidth=1.2, alpha=0.8)
        ax.set_xlabel("Time (s)", fontsize=font_size)
        ax.set_ylabel("Paw Angle (deg)", fontsize=font_size)
        ax.set_title("Paw Angle Time Series", fontsize=font_size + 2)
        ax.legend(fontsize=font_size)
        if show_grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=font_size - 1)
        self.fig.tight_layout()
        self.draw()

    def plot_asymmetry_bar(
        self, asymmetries: list[intensity.AsymmetryResult], color_scheme: str = "Set1",
        font_size: int = 12, show_grid: bool = True,
    ) -> None:
        """Bar chart of asymmetry indices for fore/hind pairs."""
        import matplotlib.pyplot as plt
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cmap = plt.get_cmap(color_scheme)
        if not asymmetries:
            ax.text(0.5, 0.5, "No asymmetry data", ha="center", va="center", transform=ax.transAxes)
            self.draw()
            return
        labels = [f"{a.pair[0]}\nvs\n{a.pair[1]}" for a in asymmetries]
        values = [a.asymmetry_index for a in asymmetries]
        x = np.arange(len(labels))
        ax.bar(x, values, color=cmap(0), edgecolor="white", alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Perfect symmetry")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=font_size)
        ax.set_ylabel("Asymmetry Index", fontsize=font_size)
        ax.set_title("L/R Asymmetry", fontsize=font_size + 2)
        ax.legend(fontsize=font_size)
        if show_grid:
            ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=font_size - 1)
        self.fig.tight_layout()
        self.draw()

    def plot_footprint_scatter(
        self, footprints: list, intensities: list, color_scheme: str = "Set1",
        font_size: int = 12, show_grid: bool = True,
    ) -> None:
        """Scatter plot: area (x) vs mean intensity (y) per footprint."""
        import matplotlib.pyplot as plt
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        cmap = plt.get_cmap(color_scheme)
        if not footprints:
            ax.text(0.5, 0.5, "No footprint data", ha="center", va="center", transform=ax.transAxes)
            self.draw()
            return
        for i, (fp, ir) in enumerate(zip(footprints, intensities)):
            name = fp.matched_paw or f"blob_{fp.label}"
            ax.scatter(fp.area_mm2, ir.mean_intensity, s=80, color=cmap(i),
                       label=name, edgecolors="white", linewidth=1.5)
        ax.set_xlabel("Print Area (mm²)", fontsize=font_size)
        ax.set_ylabel("Mean Intensity", fontsize=font_size)
        ax.set_title("Footprint Area vs Intensity", fontsize=font_size + 2)
        ax.legend(fontsize=font_size)
        if show_grid:
            ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=font_size - 1)
        self.fig.tight_layout()
        self.draw()


class ChartsTab(QWidget):
    """Tab 4: Publication-quality charts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        main = QHBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(12)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        # --- Left panel: controls ---
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Chart type
        type_group = QGroupBox("图表类型")
        type_layout = QVBoxLayout(type_group)
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(CHART_TYPES)
        type_layout.addWidget(self.chart_type_combo)
        left_layout.addWidget(type_group)

        # Parameters
        param_group = QGroupBox("参数")
        param_layout = QVBoxLayout(param_group)
        row_dpi = QHBoxLayout()
        row_dpi.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        row_dpi.addWidget(self.dpi_spin)
        param_layout.addLayout(row_dpi)
        row_scheme = QHBoxLayout()
        row_scheme.addWidget(QLabel("配色:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems(COLOR_SCHEMES)
        row_scheme.addWidget(self.scheme_combo)
        param_layout.addLayout(row_scheme)
        row_font = QHBoxLayout()
        row_font.addWidget(QLabel("字体大小:"))
        self.font_spin = QDoubleSpinBox()
        self.font_spin.setRange(6, 32)
        self.font_spin.setValue(12)
        row_font.addWidget(self.font_spin)
        param_layout.addLayout(row_font)
        self.grid_check = QCheckBox("显示网格")
        self.grid_check.setChecked(True)
        param_layout.addWidget(self.grid_check)
        self.sem_check = QCheckBox("显示 SEM 误差线")
        self.sem_check.setChecked(False)
        param_layout.addWidget(self.sem_check)
        left_layout.addWidget(param_group)

        # Buttons
        self.generate_btn = QPushButton("生成图表")
        self.generate_btn.clicked.connect(self._on_generate)
        left_layout.addWidget(self.generate_btn)
        self.export_png_btn = QPushButton("导出 PNG")
        self.export_png_btn.setEnabled(False)
        self.export_png_btn.clicked.connect(lambda: self._on_export("png"))
        left_layout.addWidget(self.export_png_btn)
        self.export_pdf_btn = QPushButton("导出 PDF")
        self.export_pdf_btn.setEnabled(False)
        self.export_pdf_btn.clicked.connect(lambda: self._on_export("pdf"))
        left_layout.addWidget(self.export_pdf_btn)
        self.export_svg_btn = QPushButton("导出 SVG")
        self.export_svg_btn.setEnabled(False)
        self.export_svg_btn.clicked.connect(lambda: self._on_export("svg"))
        left_layout.addWidget(self.export_svg_btn)
        left_layout.addStretch()
        splitter.addWidget(left)
        splitter.setSizes([300, 900])

        # --- Right panel: canvas ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.canvas = ChartsCanvas()
        right_layout.addWidget(self.canvas, stretch=1)
        splitter.addWidget(right)

    def _get_gait_results(self) -> core_results.GaitResults | None:
        main = self.window()
        if isinstance(main, QWidget) and hasattr(main, "current_gait_results"):
            return main.current_gait_results
        return None

    def _get_ftir_data(self) -> tuple[list, list] | None:
        main = self.window()
        if isinstance(main, QWidget):
            fps = getattr(main, "current_ftir_footprints", None)
            ints = getattr(main, "current_ftir_intensities", None)
            if fps is not None and ints is not None:
                return fps, ints
        return None

    def _on_generate(self) -> None:
        chart_type = self.chart_type_combo.currentText()
        scheme = self.scheme_combo.currentText()
        font_size = int(self.font_spin.value())
        show_grid = self.grid_check.isChecked()

        if chart_type == "stance/swing 阶梯图":
            res = self._get_gait_results()
            if not res:
                QMessageBox.warning(self, "提示", "请先在「步态分析」Tab 加载数据")
                return
            self.canvas.plot_stance_swing_grid(res, scheme, show_grid, font_size)
        elif chart_type == "stride 长度分布":
            res = self._get_gait_results()
            if not res:
                QMessageBox.warning(self, "提示", "请先在「步态分析」Tab 加载数据")
                return
            self.canvas.plot_stride_histogram(res, scheme, font_size, show_grid)
        elif chart_type == "paw angle 时间序列":
            res = self._get_gait_results()
            if not res:
                QMessageBox.warning(self, "提示", "请先在「步态分析」Tab 加载数据")
                return
            self.canvas.plot_paw_angle_timeseries(res, scheme, font_size, show_grid)
        elif chart_type == "不对称指数柱状图":
            res = self._get_gait_results()
            if not res:
                QMessageBox.warning(self, "提示", "请先在「步态分析」Tab 加载数据")
                return
            # Compute asymmetries from per-paw intensities/swing/stance
            asyms = [
                intensity.AsymmetryResult(
                    pair=(p.name, p.name), asymmetry_index=0.0, ratio=0.0
                )
                for p in res.paws.values()
            ]
            # Actually we need real asymmetries from intensity module
            ftir = self._get_ftir_data()
            if ftir:
                _, ints = ftir
                asyms = intensity.analyze_asymmetries(ints)
            self.canvas.plot_asymmetry_bar(asyms, scheme, font_size, show_grid)
        elif chart_type == "足印面积/强度散点图":
            ftir = self._get_ftir_data()
            if not ftir:
                QMessageBox.warning(self, "提示", "请先在「FTIR 分析」Tab 加载数据")
                return
            fps, ints = ftir
            self.canvas.plot_footprint_scatter(fps, ints, scheme, font_size, show_grid)

        self.export_png_btn.setEnabled(True)
        self.export_pdf_btn.setEnabled(True)
        self.export_svg_btn.setEnabled(True)

    def _on_export(self, fmt: str) -> None:
        ext = fmt
        path, _ = QFileDialog.getSaveFileName(
            self, f"导出 {fmt.upper()}", f"figure.{ext}",
            f"{fmt.upper()} (*.{ext});;所有文件 (*.*)"
        )
        if not path:
            return
        dpi = self.dpi_spin.value()
        try:
            self.canvas.fig.savefig(path, dpi=dpi, bbox_inches="tight")
            QMessageBox.information(self, "导出成功", f"已保存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
