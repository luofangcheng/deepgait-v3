"""LOW-threshold blob detector."""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2


def detect_blobs(frame_bgr, bg_G, *, tau_paw=18, min_area_px=10,
                 max_area_px=None, walkway_roi=(0, 15, 1920, 360),
                 bbox_pad_px=8):
    G = frame_bgr[:, :, 1].astype(np.float32)
    delta = cv2.subtract(G, bg_G)
    H, W = G.shape
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    x0, y0, w, h = walkway_roi
    roi_mask[y0:min(y0+h, H), x0:min(x0+w, W)] = 1
    binary = ((delta >= tau_paw) & (roi_mask > 0)).astype(np.uint8)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px: continue
        if max_area_px is not None and area > max_area_px: continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w_ = int(stats[i, cv2.CC_STAT_WIDTH]); h_ = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(cents[i, 0]), float(cents[i, 1])
        bbox_tight = (x, y, x+w_, y+h_)
        px1 = max(0, x - bbox_pad_px); py1 = max(0, y - bbox_pad_px)
        px2 = min(W, x+w_+bbox_pad_px); py2 = min(H, y+h_+bbox_pad_px)
        local_lbl_pad = lbl[py1:py2, px1:px2]
        mask_pad = (local_lbl_pad == i)
        delta_pad = delta[py1:py2, px1:px2].copy().astype(np.float32)
        bg_pad = bg_G[py1:py2, px1:px2].copy().astype(np.float32)
        blobs.append({
            "bbox_xyxy": bbox_tight,
            "bbox_xyxy_padded": (px1, py1, px2, py2),
            "mask_padded": mask_pad,
            "delta_padded": delta_pad,
            "bg_padded": bg_pad,
            "cx_px": cx, "cy_px": cy,
            "area_px": area,
            "touches_edge": (x==0 or y==0 or x+w_>=W or y+h_>=H),
        })
    return blobs


__all__ = ["detect_blobs"]
