"""Interactive paw pressure heatmap renderer — deepgait v2.0.

Renders per-paw 2D green-channel pressure patches as colour-mapped
heatmaps using pyqtgraph ``ImageItem`` + ``ColorMap`` + ``ColorBarItem``.

Usage::

    renderer = PressureHeatmapWidget()
    renderer.set_pressure_data(paw="LF", frame_idx=42, patch=patch_2d)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Colour map — black → green → yellow → red (CatWalk style)
# ---------------------------------------------------------------------------
PRESSURE_COLORMAP = pg.ColorMap(
    [0.0, 0.15, 0.35, 0.55, 0.75, 1.0],
    [
        (0,   0,   0),     # no pressure (black)
        (0,   60,  0),     # light touch (dark green)
        (0,   200, 50),    # contact (bright green)
        (180, 220, 0),     # moderate pressure (yellow-green)
        (255, 160, 0),     # high pressure (orange)
        (255, 20,  0),     # peak pressure (red)
    ],
)

PAW_ORDER = ("LF", "RF", "LH", "RH")


# ---------------------------------------------------------------------------
# PressureHeatmapWidget
# ---------------------------------------------------------------------------
class PressureHeatmapWidget(QWidget):
    """Interactive 2D pressure heatmap for one paw at a selected frame.

    Layout::

        ┌─────────────────────────────────────────┐
        │  Paw: [LF ▾]  Frame: [───slider───] [N] │
        ├────────────────────────────┬────────────┤
        │                            │            │
        │   ImageItem heatmap        │ ColorBar   │
        │   (black→red LUT)          │            │
        │                            │            │
        ├────────────────────────────┴────────────┤
        │  Intensity: 0           Peak: 255       │
        └─────────────────────────────────────────┘
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._patches: Dict[str, Dict[int, np.ndarray]] = {
            p: {} for p in PAW_ORDER
        }
        self._n_frames = 0
        self._current_paw = "LF"
        self._current_frame = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # -- Controls --------------------------------------------------------
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("四肢:"))
        self.paw_combo = QComboBox()
        self.paw_combo.addItems(PAW_ORDER)
        self.paw_combo.currentTextChanged.connect(self._on_paw_changed)
        ctrl.addWidget(self.paw_combo)

        ctrl.addWidget(QLabel(" Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        ctrl.addWidget(self.frame_slider)

        self.frame_label = QLabel("0/0")
        ctrl.addWidget(self.frame_label)
        layout.addLayout(ctrl)

        # -- Graphics view ---------------------------------------------------
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.graphics_widget.setBackground("k")

        # ImageItem (left)
        self.view_box = self.graphics_widget.addViewBox(row=0, col=0)
        self.view_box.setAspectLocked(False)
        self.view_box.invertY(True)
        self.image_item = pg.ImageItem()
        self.image_item.setColorMap(PRESSURE_COLORMAP)
        self.view_box.addItem(self.image_item)

        # ColorBarItem (right)
        self.colorbar = pg.ColorBarItem(
            values=(0, 255), colorMap=PRESSURE_COLORMAP,
            width=15, interactive=False,
        )
        self.graphics_widget.addItem(self.colorbar, row=0, col=1)

        # Intensity label
        self.intensity_label = pg.LabelItem(
            "Intensity: 0 — 255", color=(200, 200, 200),
            size="10pt",
        )
        self.intensity_label.setParentItem(self.view_box)
        self.intensity_label.anchor(
            itemPos=(0.5, 0.02), parentPos=(0.5, 0.02),
        )

        layout.addWidget(self.graphics_widget)

        # -- Info bar --------------------------------------------------------
        self.info_label = QLabel("未加载数据")
        self.info_label.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(self.info_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_pressure_data(
        self,
        paw: str,
        frame_idx: int,
        green_patch: np.ndarray,
    ) -> None:
        """Store a paw's pressure patch for a given frame."""
        if paw not in self._patches:
            return
        self._patches[paw][frame_idx] = green_patch.copy()
        self._n_frames = max(self._n_frames, frame_idx + 1)
        if self.frame_slider.maximum() < self._n_frames - 1:
            self.frame_slider.setMaximum(self._n_frames - 1)

    def set_bulk_data(
        self,
        all_patches: Dict[str, Dict[int, np.ndarray]],
        n_frames: int = 0,
    ) -> None:
        """Load pre-computed patch data for all paws."""
        for paw in PAW_ORDER:
            if paw in all_patches:
                self._patches[paw] = dict(all_patches[paw])
        self._n_frames = max(n_frames, max(
            (max(p.keys()) if p else -1) + 1 for p in self._patches.values()
        ))
        if self.frame_slider.maximum() < self._n_frames - 1:
            self.frame_slider.setMaximum(max(0, self._n_frames - 1))
        self._refresh()

    def set_current_frame(self, frame_idx: int) -> None:
        """Jump to a specific frame."""
        self._current_frame = max(0, min(frame_idx, self._n_frames - 1))
        self.frame_slider.setValue(self._current_frame)
        self._refresh()

    def clear(self) -> None:
        """Reset all data."""
        self._patches = {p: {} for p in PAW_ORDER}
        self._n_frames = 0
        self.frame_slider.setRange(0, 0)
        self.image_item.clear()
        self.info_label.setText("未加载数据")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        patch = self._patches.get(self._current_paw, {}).get(self._current_frame)
        if patch is None or patch.size == 0:
            self.image_item.clear()
            self.info_label.setText(
                f"{self._current_paw} @ frame {self._current_frame}: 无数据",
            )
            return

        self.image_item.setImage(patch, levels=(0, 255))
        self.colorbar.setLevels((0, 255))

        max_val = patch.max()
        mean_val = patch.mean()
        self.intensity_label.setText(
            f"Peak: {max_val:.0f}  |  Mean: {mean_val:.1f}",
        )
        self.info_label.setText(
            f"{self._current_paw} @ frame {self._current_frame}  "
            f"shape={patch.shape}  peak={max_val:.0f}  mean={mean_val:.1f}",
        )

    def _on_paw_changed(self, paw: str) -> None:
        self._current_paw = paw
        self._refresh()

    def _on_frame_changed(self, frame_idx: int) -> None:
        self._current_frame = frame_idx
        self.frame_label.setText(
            f"{frame_idx}/{max(0, self._n_frames - 1)}",
        )
        self._refresh()
