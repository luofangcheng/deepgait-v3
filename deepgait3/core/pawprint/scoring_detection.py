"""Frame-level blob detection on a color-score map.

Ported from ``deepgait-v2/test_paw_detection.py::detect_blobs_from_score``
and exposed as a pure function so it can be reused for both the v3 v0.4.2
detector path and the four color-score algorithms in ``scoring.py``.

Returns a dict per blob: ``{"cx", "cy", "area", "bbox"}``.
"""
from __future__ import annotations

import numpy as np
import cv2


def detect_blobs_from_score(
    score_map: np.ndarray,
    threshold: float,
    min_area: int = 5,
    morphology: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Threshold + optional morphology + connected components.

    Args:
        score_map: H×W int or float array. Higher = more paw-like.
        threshold: Pixels with ``score > threshold`` are foreground.
        min_area: Connected components smaller than this are dropped.
        morphology: If True, apply 3×3 elliptical open+close to clean noise.

    Returns:
        mask: H×W uint8 binary mask (0 or 255).
        blobs: list of dicts ``{"cx": float, "cy": float, "area": int,
        "bbox": (x, y, x+w, y+h)}``.
    """
    mask = (score_map > threshold).astype(np.uint8) * 255
    if morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, _labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs: list[dict] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        blobs.append({
            "cx": float(cents[i, 0]),
            "cy": float(cents[i, 1]),
            "area": area,
            "bbox": (x, y, x + w, y + h),
        })
    return mask, blobs


__all__ = ["detect_blobs_from_score"]