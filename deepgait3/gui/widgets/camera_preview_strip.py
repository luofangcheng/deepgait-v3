"""Four-camera 1×4 preview strip with per-camera FPS sink.

Reuses :class:`CameraPreviewPane` (defined in
:mod:`deepgait3.gui.initialization_tab`) as a pure visual sink:

* Each pane has its ``set_camera(None)`` called so the internal
  ``_tick`` timer is disabled — the host tab drives ``set_image``
  directly from its own :class:`MultiCameraManager` pull loop.
* The host tab pushes per-camera hardware FPS via :meth:`set_fps`
  (typically every 500 ms, alongside sync health).
* The individual ``start``/``stop``/``record`` buttons on each pane
  are hidden — acquisition is controlled at the tab level.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QWidget,
)

from deepgait3.gui.initialization_tab import CameraPreviewPane


class FourCamStrip(QWidget):
    """1×4 horizontal strip of 4 :class:`CameraPreviewPane` instances."""

    ROLES: tuple = ("left", "right", "top", "bottom")
    """Visual left-to-right order for the 4 cameras in the strip."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._panes: Dict[str, CameraPreviewPane] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        for role in self.ROLES:
            pane = CameraPreviewPane(role, self)
            # Strip the per-pane controls; the host tab owns them.
            pane.start_btn.setVisible(False)
            pane.stop_btn.setVisible(False)
            pane.record_check.setVisible(False)
            # Block the pane's internal _tick auto-pull loop.
            pane.set_camera(None)
            # Resize policy so the strip fills available width.
            pane.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
            )
            layout.addWidget(pane, stretch=1)
            self._panes[role] = pane

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_image(self, role: str, frame_bgr: np.ndarray) -> None:
        """Push a BGR frame into the named role's preview pane.

        Skips silently if ``role`` is unknown or the frame is empty.
        Mirrors the rendering path used by
        :class:`CameraPreviewPane._tick` so the appearance matches the
        initialisation-tab preview exactly.
        """
        pane = self._panes.get(role)
        if pane is None or frame_bgr is None:
            return
        if frame_bgr.size == 0:
            return
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # pyqtgraph ImageView wants CHW float32 / uint8 with autoscale.
            pane.image_view.setImage(rgb.transpose(2, 0, 1))
        except Exception:
            # ImageView occasionally throws on resize-while-update;
            # never let preview errors kill the live loop.
            pass

    def set_fps(self, role: str, fps: float) -> None:
        """Update the per-pane FPS label.

        Renders ``"—"`` for ``NaN`` (insufficient samples).
        """
        pane = self._panes.get(role)
        if pane is None:
            return
        if math.isnan(fps) or math.isinf(fps):
            pane.fps_label.setText("FPS: —")
        else:
            pane.fps_label.setText(f"FPS: {fps:.1f}")

    def reset(self) -> None:
        """Reset all FPS labels to ``'—'``.  Call on disconnect."""
        for pane in self._panes.values():
            pane.fps_label.setText("FPS: —")

    def panes(self) -> Dict[str, CameraPreviewPane]:
        """Return the underlying panes (read-only access)."""
        return dict(self._panes)


__all__ = ["FourCamStrip"]
