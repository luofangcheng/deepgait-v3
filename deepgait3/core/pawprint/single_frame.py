"""Single-frame detection adapter: pawprint primitives → FootprintSequence.

Bridges the new ``core/pawprint`` detection pipeline (``detect_blobs``,
``cluster_blobs_into_feet``, ``MouseDetector``) to the ``FootprintSequence``
output type that the GUI tabs expect, so workers can swap implementations
with minimal changes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from deepgait3.core._legacy.footprint_v2 import FootMask, FootprintSequence

from .detection import detect_blobs
from .grouping import cluster_blobs_into_feet
from .mouse_detector import MouseDetector


def detect_single_frame(
    frame_bgr: np.ndarray,
    bg_G: np.ndarray,
    *,
    tau_paw: float = 10.0,
    min_area_px: int = 10,
    D_merge_px: float = 23.0,
    walkway_roi: Tuple[int, int, int, int] = (0, 15, 1920, 360),
    bbox_pad_px: int = 8,
    mouse_det: MouseDetector | None = None,
    px_per_mm: float | None = None,
    body_axis: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    in_stance_threshold_px: int = 30,
) -> FootprintSequence:
    """Run pawprint detection on a single BGR frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Input BGR image (uint8, H×W×3).
    bg_G : np.ndarray
        Green-channel background image (float32, H×W).  For single-image
        mode, use ``frame_bgr[:, :, 1].astype(np.float32)``.
    tau_paw : float
        Green-delta threshold for blobs (default 10).
    min_area_px : int
        Minimum blob area in pixels.
    D_merge_px : float
        Max centroid distance for merging finger blobs into one foot.
    walkway_roi : tuple
        (x, y, w, h) of the walkway region.
    bbox_pad_px : int
        Padding around each blob's tight bbox.
    mouse_det : MouseDetector or None
        Pre-constructed detector.  If provided, the expanded mouse ROI
        is used to filter footmasks far from the body.
    px_per_mm : float or None
        Calibration factor for mm² conversion.
    body_axis : tuple or None
        Midline + direction for L/R paw classification (not yet wired).
    in_stance_threshold_px : int
        Minimum area to classify a foot as in stance.

    Returns
    -------
    FootprintSequence
    """
    H, W = frame_bgr.shape[:2]

    # 1. Mouse ROI (optional)
    expanded_roi: Optional[Tuple[int, int, int, int]] = None
    if mouse_det is not None:
        _, expanded_roi, _ = mouse_det(frame_bgr, bg_G, 0)

    # 2. Detect blobs
    blobs = detect_blobs(
        frame_bgr, bg_G,
        tau_paw=tau_paw,
        min_area_px=min_area_px,
        walkway_roi=walkway_roi,
        bbox_pad_px=bbox_pad_px,
    )

    # 3. Cluster into feet
    paw_feet = cluster_blobs_into_feet(
        blobs,
        D_merge_px=D_merge_px,
        frame_shape=(H, W),
    )

    # 4. Clip to expanded mouse ROI
    if expanded_roi is not None:
        ex1, ey1, ex2, ey2 = expanded_roi
        paw_feet = [
            fm for fm in paw_feet
            if _intersects(fm.bbox_xyxy, (ex1, ey1, ex2, ey2))
        ]

    # 5. Convert pawprint FootMask → footprint_v2 FootMask
    all_feet: List[FootMask] = []
    mm2_factor = (1.0 / (px_per_mm * px_per_mm)) if px_per_mm else 1.0

    for idx, pf in enumerate(paw_feet):
        x1, y1, x2, y2 = pf.bbox_xyxy
        bbox = (x1, y1, x2 - x1, y2 - y1)
        area_px = int(pf.total_area_px)

        # Intensity stats from raw delta crop
        if pf.mask_padded is not None and pf.mask_padded.any():
            delta_vals = pf.raw_intensity_crop[pf.mask_padded]
            intensity_mean = float(np.mean(delta_vals))
            intensity_max = int(np.max(delta_vals))
            intensity_min = int(np.min(delta_vals))
            intensity_total = float(np.sum(delta_vals))
        else:
            intensity_mean = pf.mean_intensity
            intensity_max = int(pf.peak_intensity)
            intensity_min = 0
            intensity_total = 0.0

        # Hull area
        hull_area_px = area_px
        if pf.mask_padded is not None and pf.mask_padded.any():
            mask_u8 = pf.mask_padded.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                hull = cv2.convexHull(contours[0])
                hull_area_px = int(cv2.contourArea(hull))

        fm = FootMask(
            label=idx + 1,
            area_px=area_px,
            area_mm2=area_px * mm2_factor,
            centroid=pf.centroid_px,
            bbox=bbox,
            hull_area_px=hull_area_px,
            intensity_mean=intensity_mean,
            intensity_max=intensity_max,
            intensity_min=intensity_min,
            intensity_total=intensity_total,
            is_in_stance=area_px >= in_stance_threshold_px,
            matched_paw=None,
        )
        all_feet.append(fm)

    # 6. Classify feet (L/R, F/H) if body axis provided
    feet: dict[str, FootMask] = {}
    if body_axis is not None and all_feet:
        feet = _classify_feet(all_feet, body_axis, H)

    return FootprintSequence(
        feet=feet,
        n_feet=len(all_feet),
        all_feet=all_feet,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _intersects(
    bbox: Tuple[int, int, int, int],
    roi: Tuple[int, int, int, int],
) -> bool:
    """True if bbox (x1,y1,x2,y2) overlaps roi (x1,y1,x2,y2)."""
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = roi
    return not (bx2 <= rx1 or bx1 >= rx2 or by2 <= ry1 or by1 >= ry2)


def _classify_feet(
    all_feet: List[FootMask],
    body_axis: Tuple[Tuple[float, float], Tuple[float, float]],
    height: int,
) -> dict[str, FootMask]:
    """Classify feet as LF/RF/LH/RH using body axis.

    Simple approach: split left/right by cross-product with midline,
    and front/hind by vertical position relative to midline midpoint.
    """
    (mx1, my1), (mx2, my2) = body_axis
    mid_y = (my1 + my2) / 2.0

    result: dict[str, FootMask] = {}
    for fm in all_feet:
        cx, cy = fm.centroid
        # Left/right: sign of cross product (midline vector × centroid vector)
        cross = (mx2 - mx1) * (cy - my1) - (my2 - my1) * (cx - mx1)
        side = "L" if cross > 0 else "R"
        # Front/hind: above or below midline midpoint
        end = "F" if cy < mid_y else "H"
        paw = f"{side}{end}"
        # If paw already taken, keep the one with larger area
        if paw not in result or fm.area_px > result[paw].area_px:
            result[paw] = fm
            fm.matched_paw = paw

    return result
