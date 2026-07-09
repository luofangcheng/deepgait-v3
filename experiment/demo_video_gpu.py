"""GPU-accelerated pipeline: batch YOLO + torch masks → cumulative footprint.

Key optimizations vs demo_video.py:
  - Batch inference: process N frames per GPU forward pass
  - torch masks: resize + binarize on GPU, no cv2 per-mask resize
  - Single torch→numpy transfer per batch

Usage::

    python experiment/demo_video_gpu.py

Outputs experiment/outputs/video_demo_gpu/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepgait3.core.pawprint.tracker import IoUFootprintTracker
from deepgait3.core.pawprint.models import FootMask
from deepgait3.core.pawprint.cumulative import build_cumulative_union, render_overlay


BATCH_SIZE = 16  # frames per GPU forward pass (RTX 3060 12GB can handle this)


def yolo_batch_to_footmasks(
    frames_bgr: list[np.ndarray],
    results_batch,
    bg_G: np.ndarray,
    frame_offsets: list[int],
) -> list[list[FootMask]]:
    """Convert a batch of YOLO results to per-frame FootMask lists.

    Mask resizing is done with torch interpolation on GPU.
    """
    h, w = frames_bgr[0].shape[:2]
    all_footmasks: list[list[FootMask]] = [[] for _ in frames_bgr]

    for b, (frame, results) in enumerate(zip(frames_bgr, results_batch)):
        if results.masks is None:
            continue

        masks_t = results.masks.data  # (N, mH, mW) on GPU
        boxes = results.boxes
        if masks_t.dtype != torch.float32:
            masks_t = masks_t.float()

        # Batch-resize all masks to original resolution on GPU
        if masks_t.shape[1:] != (h, w):
            masks_t = torch.nn.functional.interpolate(
                masks_t.unsqueeze(0) if masks_t.dim() == 3 else masks_t,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            if masks_t.dim() == 4:
                masks_t = masks_t.squeeze(0)

        masks_np = masks_t.cpu().numpy()  # single GPU→CPU transfer per batch

        G = frame[:, :, 1].astype(np.float32)
        delta = G - bg_G

        for i in range(masks_np.shape[0]):
            mask = masks_np[i]
            mask_bin = (mask > 0.5).astype(np.uint8)
            area_px = int(mask_bin.sum())
            if area_px < 5:
                continue

            # Centroid from moments
            moments = cv2.moments(mask_bin)
            cx = moments["m10"] / moments["m00"] if moments["m00"] > 0 else 0.0
            cy = moments["m01"] / moments["m00"] if moments["m00"] > 0 else 0.0

            # Bbox
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
                blob_indices=[i],
                centroid_px=(cx, cy),
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_xyxy_padded=(px1, py1, px2, py2),
                mask_padded=mask_crop,
                raw_intensity_crop=delta_crop.astype(np.float32),
                bg_intensity_crop=np.zeros_like(delta_crop, dtype=np.float32),
                pressure_map=pressure_map.astype(np.float32),
                total_area_px=area_px,
                mean_intensity=mean_intensity,
                peak_intensity=peak_intensity,
                touches_edge=(x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1),
            )
            all_footmasks[b].append(fm)

    return all_footmasks


def build_cumulative_gpu(
    cumulative_intensity: np.ndarray,
) -> np.ndarray:
    """Black background, green max-intensity projection."""
    overlay = np.zeros((*cumulative_intensity.shape, 3), dtype=np.uint8)
    if cumulative_intensity.max() > 0:
        norm = (cumulative_intensity / cumulative_intensity.max() * 255).astype(np.uint8)
        overlay[:, :, 1] = norm
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU-accelerated video footprint pipeline")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pawprint_yolo/weights/best.pt")
    parser.add_argument("--video", type=Path,
                        default=Path("/home/luofangcheng/Documents/ZCODE/test1.mp4"))
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/video_demo_gpu")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(str(args.video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Video: {len(frames)} frames, {w}x{h}, {fps} FPS")
    print(f"Batch size: {args.batch_size}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Median background
    green_stack = np.stack([f[:, :, 1].astype(np.float32) for f in frames])
    bg_G = np.median(green_stack, axis=0)

    # Tracker
    tracker = IoUFootprintTracker(frame_shape=(h, w), iou_min=0.3, max_gap_frames=3)

    # Batch inference loop
    import time
    t0 = time.perf_counter()

    for batch_start in range(0, len(frames), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]

        # Single GPU forward pass for entire batch
        results_batch = model(batch_frames, verbose=False, stream=False)

        # Convert to FootMasks (GPU mask resize inside)
        all_footmasks = yolo_batch_to_footmasks(
            batch_frames, results_batch, bg_G,
            list(range(batch_start, batch_end)),
        )

        for i, footmasks in enumerate(all_footmasks):
            tracker.update(batch_start + i + 1, footmasks)

        elapsed = time.perf_counter() - t0
        print(f"  Batch {batch_start // args.batch_size + 1}: "
              f"frames {batch_start + 1}-{batch_end}, "
              f"{elapsed:.1f}s elapsed")

    total_time = time.perf_counter() - t0
    tracks = tracker.finalize()

    print(f"\n--- Results ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {len(frames) / total_time:.1f} FPS")
    print(f"Tracks: {len(tracks)}")
    print(f"Footprint detections: {sum(len(t.foots) for t in tracks)}")

    # Build cumulative images using GPU-accelerated track-union algorithm
    h, w = frames[0].shape[:2]
    cum_intensity = build_cumulative_union(tracks, (h, w), use_gpu=True)

    overlay = render_overlay(cum_intensity)
    cv2.imwrite(str(args.output / "cumulative_overlay.png"), overlay)
    cv2.imwrite(str(args.output / "cumulative_mask.png"),
                (cum_intensity > 0).astype(np.uint8) * 255)

    print(f"\nSaved to {args.output}")
    print(f"  cumulative_overlay.png  — black bg, green max-intensity footprints")


if __name__ == "__main__":
    main()
