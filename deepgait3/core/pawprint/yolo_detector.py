"""YOLOv8n-seg paw detector (ported from experiment/demo_video_gpu.py).

Batch GPU inference → FootMask conversion.  Mask resize on GPU via
torch interpolation.  Single GPU→CPU transfer per batch.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

from deepgait3.core.pawprint.models import FootMask

# ── model path ───────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / \
    "experiment" / "outputs" / "pawprint_yolo" / "weights" / "best.pt"

BATCH_SIZE = 16


class YoloPawDetector:
    """YOLOv8n-seg detector for fTIR pawprint segmentation."""

    def __init__(self, model_path: str | Path | None = None, conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path or _MODEL_PATH))
        self.conf = conf

    # ── public API ───────────────────────────────────────────────────────

    def detect_single(self, frame_bgr: np.ndarray, bg_G: np.ndarray,
                      *, min_area_px: int = 5) -> list[FootMask]:
        result = self.detect_batch([frame_bgr], bg_G, min_area_px=min_area_px)
        return result[0]

    def detect_batch(self, frames_bgr: list[np.ndarray], bg_G: np.ndarray,
                     *, min_area_px: int = 5) -> list[list[FootMask]]:
        h, w = frames_bgr[0].shape[:2]
        all_footmasks: list[list[FootMask]] = []

        for batch_start in range(0, len(frames_bgr), BATCH_SIZE):
            batch = frames_bgr[batch_start:batch_start + BATCH_SIZE]
            results_batch = self.model(batch, verbose=False, stream=False, conf=self.conf)

            for frame, results in zip(batch, results_batch):
                footmasks = self._results_to_footmasks(frame, results, bg_G, h, w, min_area_px)
                all_footmasks.append(footmasks)

        return all_footmasks

    # ── internals ────────────────────────────────────────────────────────

    def _results_to_footmasks(self, frame: np.ndarray, results, bg_G: np.ndarray,
                               h: int, w: int, min_area_px: int) -> list[FootMask]:
        footmasks: list[FootMask] = []
        if results.masks is None:
            return footmasks

        masks_np = results.masks.data.cpu().numpy()  # (N, mH, mW)
        # Resize masks to original resolution (cv2 — same as experiment)
        if masks_np.shape[1:] != (h, w):
            masks_np = np.stack([
                cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                for m in masks_np
            ])

        G = frame[:, :, 1].astype(np.float32)
        delta = G - bg_G

        for i in range(masks_np.shape[0]):
            mask = masks_np[i]
            mask_bin = (mask > 0.5).astype(np.uint8)
            area_px = int(mask_bin.sum())
            if area_px < min_area_px:
                continue
            moments = cv2.moments(mask_bin)
            cx = moments["m10"] / moments["m00"] if moments["m00"] > 0 else 0.0
            cy = moments["m01"] / moments["m00"] if moments["m00"] > 0 else 0.0
            ys, xs = np.where(mask_bin)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            pad = 2
            py1, py2 = max(0, y1 - pad), min(h, y2 + pad)
            px1, px2 = max(0, x1 - pad), min(w, x2 + pad)
            mask_crop = mask_bin[py1:py2, px1:px2].astype(bool)
            delta_crop = delta[py1:py2, px1:px2]
            in_mask = delta_crop[mask_crop]
            mean_intensity = float(in_mask.mean()) if in_mask.size > 0 else 0.0
            peak_intensity = float(in_mask.max()) if in_mask.size > 0 else 0.0
            pressure_map = 18.0 * np.maximum(delta_crop - 8.0, 0) ** 0.75
            fm = FootMask(
                blob_indices=[i], centroid_px=(cx, cy),
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_xyxy_padded=(px1, py1, px2, py2),
                mask_padded=mask_crop,
                raw_intensity_crop=delta_crop.astype(np.float32),
                bg_intensity_crop=np.zeros_like(delta_crop, dtype=np.float32),
                pressure_map=pressure_map.astype(np.float32),
                total_area_px=area_px,
                mean_intensity=mean_intensity, peak_intensity=peak_intensity,
                touches_edge=(x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1),
            )
            footmasks.append(fm)
        return footmasks
