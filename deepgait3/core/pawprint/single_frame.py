"""Single-frame YOLO detection — returns native FootMask list.

Thin wrapper around YoloPawDetector for GUI compatibility.
"""
from __future__ import annotations

from deepgait3.core.pawprint.models import FootMask
from deepgait3.core.pawprint.yolo_detector import YoloPawDetector

_detector: YoloPawDetector | None = None


def _get_detector() -> YoloPawDetector:
    global _detector
    if _detector is None:
        _detector = YoloPawDetector()
    return _detector


def detect_single_frame(
    frame_bgr, bg_G, *,
    min_area_px: int = 5, conf: float = 0.25,
    mouse_det=None,
    # accepted for compat, ignored
    tau_paw=0, D_merge_px=0, walkway_roi=None, bbox_pad_px=0,
    px_per_mm=None, body_axis=None, in_stance_threshold_px=0,
) -> list[FootMask]:
    det = _get_detector()
    det.conf = conf
    footmasks = det.detect_single(frame_bgr, bg_G, min_area_px=min_area_px)
    if mouse_det is not None:
        _, expanded, _ = mouse_det(frame_bgr, bg_G, 0)
        if expanded is not None and expanded != (0, 0, 0, 0):
            ex1, ey1, ex2, ey2 = expanded
            footmasks = [fm for fm in footmasks
                         if _intersects(fm.bbox_xyxy, (ex1, ey1, ex2, ey2))]
    return footmasks


def _intersects(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)
