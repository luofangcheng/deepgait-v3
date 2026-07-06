"""Mouse body detector — finds the largest dark region in an FTIR frame.

The mouse silhouette is colder (darker) than the walkway background.
We threshold *bg_G − frame_G*, close small gaps, and return the largest
connected component as the tight mouse bbox together with an expanded
ROI (tight ± pad, clamped to frame boundaries).
"""
from __future__ import annotations
from typing import Optional, Tuple

import cv2
import numpy as np


class MouseDetector:
    """Stateless per-frame mouse body detector."""

    def __init__(
        self,
        *,
        dark_threshold: float = 5,
        close_kernel: int = 7,
        min_area_px: int = 3000,
        roi_pad: int = 50,
    ):
        self.dark_threshold = dark_threshold
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
        )
        self.min_area_px = min_area_px
        self.roi_pad = roi_pad

    def __call__(
        self,
        frame_bgr: np.ndarray,
        bg_green: np.ndarray,
        frame_idx: int,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], int]:
        """Detect mouse body in one frame.

        Returns
        -------
        tight : (x1, y1, x2, y2) or None
        expanded : (x1, y1, x2, y2) or None
        area_px : int
        """
        H, W = frame_bgr.shape[:2]
        G = frame_bgr[:, :, 1].astype(np.float32)
        dark = cv2.subtract(bg_green, G)

        mask = (dark >= self.dark_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)

        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best_idx, best_area = -1, 0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > best_area and area >= self.min_area_px:
                best_area, best_idx = area, i

        if best_idx < 0:
            return None, None, 0

        x = int(stats[best_idx, cv2.CC_STAT_LEFT])
        y = int(stats[best_idx, cv2.CC_STAT_TOP])
        w = int(stats[best_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
        tight = (x, y, x + w, y + h)

        ex1 = max(0, x - self.roi_pad)
        ey1 = max(0, y - self.roi_pad)
        ex2 = min(W, x + w + self.roi_pad)
        ey2 = min(H, y + h + self.roi_pad)
        expanded = (ex1, ey1, ex2, ey2)

        return tight, expanded, best_area


__all__ = ["MouseDetector"]
