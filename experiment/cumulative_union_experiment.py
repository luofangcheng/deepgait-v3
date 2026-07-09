"""Experiment: track-level union mask vs. pixel-wise max-merge for cumulative footprints.

This script does NOT modify existing experiment/ algorithms.  It builds a new
pipeline alongside demo_video_gpu.py and compares:

  1. OLD: pixel-wise max-merge across all frames (current demo_video_gpu.py)
  2. NEW: track-level OR mask + per-pixel max intensity inside the union
  3. NEW+CLOSED: same as NEW + morphological closing to bridge palm-toe gaps

Outputs are written to experiment/outputs/cumulative_union_experiment/.

Usage::

    cd /home/luofangcheng/Documents/ZCODE/deepgait-v3
    python experiment/cumulative_union_experiment.py
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepgait3.core.pawprint.tracker import IoUFootprintTracker, FootprintTrack
from deepgait3.core.pawprint.models import FootMask

BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Re-use the GPU batch conversion from demo_video_gpu.py
# ---------------------------------------------------------------------------

def yolo_batch_to_footmasks(
    frames_bgr: list[np.ndarray],
    results_batch,
    bg_G: np.ndarray,
) -> list[list[FootMask]]:
    """Convert batch YOLO results → per-frame FootMask lists."""
    h, w = frames_bgr[0].shape[:2]
    all_footmasks: list[list[FootMask]] = [[] for _ in frames_bgr]

    for b, (frame, results) in enumerate(zip(frames_bgr, results_batch)):
        if results.masks is None:
            continue

        masks_t = results.masks.data
        if masks_t.dtype != torch.float32:
            masks_t = masks_t.float()
        if masks_t.shape[1:] != (h, w):
            masks_t = torch.nn.functional.interpolate(
                masks_t.unsqueeze(0) if masks_t.dim() == 3 else masks_t,
                size=(h, w), mode="bilinear", align_corners=False,
            )
            if masks_t.dim() == 4:
                masks_t = masks_t.squeeze(0)

        masks_np = masks_t.cpu().numpy()
        G = frame[:, :, 1].astype(np.float32)
        delta = G - bg_G

        for i in range(masks_np.shape[0]):
            mask = masks_np[i]
            mask_bin = (mask > 0.5).astype(np.uint8)
            area_px = int(mask_bin.sum())
            if area_px < 5:
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


# ---------------------------------------------------------------------------
# OLD cumulative algorithm (pixel-wise max-merge)
# ---------------------------------------------------------------------------

def build_cumulative_old(
    tracks: list[FootprintTrack],
    shape: tuple[int, int],
) -> np.ndarray:
    """Reproduce demo_video_gpu.py cumulative_overlay logic."""
    h, w = shape
    cum_intensity = np.zeros((h, w), dtype=np.float32)

    for track in tracks:
        for _, fm in track.foots:
            py1, py2, px1, px2 = fm.bbox_xyxy_padded
            if px2 <= px1 or py2 <= py1:
                continue
            crop_h, crop_w = py2 - py1, px2 - px1

            intensity_crop = fm.raw_intensity_crop
            if intensity_crop.shape[:2] != (crop_h, crop_w):
                intensity_crop = cv2.resize(intensity_crop, (crop_w, crop_h))
            mask_crop = fm.mask_padded.astype(np.float32)
            if mask_crop.shape[:2] != (crop_h, crop_w):
                mask_crop = cv2.resize(mask_crop, (crop_w, crop_h))

            cum_intensity[py1:py2, px1:px2] = np.maximum(
                cum_intensity[py1:py2, px1:px2],
                intensity_crop * mask_crop,
            )

    return cum_intensity


# ---------------------------------------------------------------------------
# NEW cumulative algorithm (track-level union mask)
# ---------------------------------------------------------------------------

def _place_global_mask(fm: FootMask, H: int, W: int) -> np.ndarray:
    """Place a FootMask's mask_padded onto a full-frame boolean mask."""
    px1, py1, px2, py2 = fm.bbox_xyxy_padded
    if px2 <= px1 or py2 <= py1:
        return np.zeros((H, W), dtype=bool)
    full = np.zeros((H, W), dtype=bool)
    full[py1:py2, px1:px2] = fm.mask_padded
    return full


def _place_global_intensity(fm: FootMask, H: int, W: int) -> np.ndarray:
    """Place a FootMask's raw_intensity_crop onto a full-frame float image."""
    px1, py1, px2, py2 = fm.bbox_xyxy_padded
    if px2 <= px1 or py2 <= py1:
        return np.zeros((H, W), dtype=np.float32)

    ch, cw = py2 - py1, px2 - px1
    intensity = fm.raw_intensity_crop
    if intensity.shape[:2] != (ch, cw):
        intensity = cv2.resize(intensity, (cw, ch))
    mask = fm.mask_padded.astype(np.float32)
    if mask.shape[:2] != (ch, cw):
        mask = cv2.resize(mask, (cw, ch))

    full = np.zeros((H, W), dtype=np.float32)
    full[py1:py2, px1:px2] = np.maximum(intensity, 0) * mask
    return full


def build_cumulative_union(
    tracks: list[FootprintTrack],
    shape: tuple[int, int],
    closing_kernel: int | None = None,
) -> np.ndarray:
    """Track-level union mask + per-pixel max intensity.

    For each track:
      1. OR all frame masks to get the spatial footprint (union mask).
      2. Within the union mask, take the max intensity observed at each pixel.
      3. Optionally apply morphological closing to bridge palm-toe gaps.
    """
    H, W = shape
    cum_intensity = np.zeros((H, W), dtype=np.float32)

    for track in tracks:
        if not track.foots:
            continue

        # Build union mask and max-intensity image for this track
        union_mask = np.zeros((H, W), dtype=bool)
        track_intensity = np.zeros((H, W), dtype=np.float32)

        for _, fm in track.foots:
            global_mask = _place_global_mask(fm, H, W)
            global_intensity = _place_global_intensity(fm, H, W)

            union_mask |= global_mask
            track_intensity = np.maximum(track_intensity, global_intensity)

        # Optional morphological closing to bridge small gaps (palm ↔ toes)
        if closing_kernel and closing_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel),
            )
            union_mask_u8 = union_mask.astype(np.uint8) * 255
            closed_u8 = cv2.morphologyEx(union_mask_u8, cv2.MORPH_CLOSE, kernel)
            union_mask = closed_u8 > 0

        # Merge into global cumulative image
        cum_intensity[union_mask] = np.maximum(
            cum_intensity[union_mask], track_intensity[union_mask],
        )

    return cum_intensity


# ---------------------------------------------------------------------------
# Metrics comparison
# ---------------------------------------------------------------------------

@dataclass
class TrackMetrics:
    track_id: int
    n_frames: int
    old_area_px: int = 0
    union_area_px: int = 0
    closed_area_px: int = 0
    old_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    union_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    closed_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


def measure_track_metrics(
    tracks: list[FootprintTrack],
    shape: tuple[int, int],
    closing_kernel: int = 13,
) -> list[TrackMetrics]:
    """Compare per-track footprint geometry under old vs. union vs. closed."""
    H, W = shape
    results: list[TrackMetrics] = []

    for track in tracks:
        if not track.foots:
            continue

        m = TrackMetrics(track_id=track.track_id, n_frames=len(track.foots))

        # OLD: take the frame with max area as the representative
        peak_fm = max(track.foots, key=lambda f: f[1].total_area_px)[1]
        px1, py1, px2, py2 = peak_fm.bbox_xyxy
        m.old_area_px = peak_fm.total_area_px
        m.old_bbox = (px1, py1, px2, py2)

        # NEW: union mask
        union_mask = np.zeros((H, W), dtype=bool)
        for _, fm in track.foots:
            union_mask |= _place_global_mask(fm, H, W)
        m.union_area_px = int(union_mask.sum())
        ys, xs = np.where(union_mask)
        if xs.size:
            m.union_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # NEW+CLOSED
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel),
        )
        closed_u8 = cv2.morphologyEx(
            union_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel,
        )
        closed_mask = closed_u8 > 0
        m.closed_area_px = int(closed_mask.sum())
        ys, xs = np.where(closed_mask)
        if xs.size:
            m.closed_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        results.append(m)

    return results


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_overlay(cum_intensity: np.ndarray) -> np.ndarray:
    """Black background, green max-intensity projection."""
    overlay = np.zeros((*cum_intensity.shape, 3), dtype=np.uint8)
    if cum_intensity.max() > 0:
        norm = (cum_intensity / cum_intensity.max() * 255).astype(np.uint8)
        overlay[:, :, 1] = norm
    return overlay


def render_comparison(
    old_intensity: np.ndarray,
    union_intensity: np.ndarray,
    closed_intensity: np.ndarray,
) -> np.ndarray:
    """Side-by-side BGR comparison of the three algorithms."""
    old = render_overlay(old_intensity)
    uni = render_overlay(union_intensity)
    cls = render_overlay(closed_intensity)

    # Resize to same height for concatenation
    h = min(old.shape[0], uni.shape[0], cls.shape[0])
    old = cv2.resize(old, (int(old.shape[1] * h / old.shape[0]), h))
    uni = cv2.resize(uni, (int(uni.shape[1] * h / uni.shape[0]), h))
    cls = cv2.resize(cls, (int(cls.shape[1] * h / cls.shape[0]), h))

    gap = np.zeros((h, 20, 3), dtype=np.uint8)
    combined = np.concatenate([old, gap, uni, gap, cls], axis=1)

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "OLD: pixel max-merge", (10, 30), font, 0.7, (255, 255, 255), 2)
    x2 = old.shape[1] + 30
    cv2.putText(combined, "NEW: track union", (x2, 30), font, 0.7, (255, 255, 255), 2)
    x3 = x2 + uni.shape[1] + 20
    cv2.putText(combined, "NEW+CLOSED", (x3, 30), font, 0.7, (255, 255, 255), 2)

    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment: track-level union vs. pixel-wise max-merge",
    )
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pawprint_yolo/weights/best.pt")
    parser.add_argument("--video", type=Path,
                        default=Path("/home/luofangcheng/Documents/ZCODE/test1.mp4"))
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/cumulative_union_experiment")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--closing-kernel", type=int, default=13,
                        help="Morphological closing kernel size (0 to disable)")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(str(args.video))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Input: {args.video.name} — {len(frames)} frames, {w}x{h}, {fps:.0f} FPS")

    # Median background
    green_stack = np.stack([f[:, :, 1].astype(np.float32) for f in frames])
    bg_G = np.median(green_stack, axis=0)

    # Tracker
    tracker = IoUFootprintTracker(frame_shape=(h, w), iou_min=0.3, max_gap_frames=3)

    # Batch inference + tracking
    print("\nRunning YOLO batch inference...")
    t0 = time.perf_counter()
    for batch_start in range(0, len(frames), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]
        results_batch = model(batch_frames, verbose=False, stream=False)
        all_footmasks = yolo_batch_to_footmasks(batch_frames, results_batch, bg_G)
        for i, footmasks in enumerate(all_footmasks):
            tracker.update(batch_start + i + 1, footmasks)
    total_time = time.perf_counter() - t0
    tracks = tracker.finalize()

    print(f"\n--- YOLO done ({total_time:.1f}s, {len(frames)/total_time:.1f} FPS) ---")
    print(f"Tracks: {len(tracks)}")
    print(f"Total detections: {sum(len(t.foots) for t in tracks)}")

    # Build cumulative images with three methods
    print("\nBuilding cumulative images...")
    old_intensity = build_cumulative_old(tracks, (h, w))
    union_intensity = build_cumulative_union(tracks, (h, w), closing_kernel=None)
    closed_intensity = build_cumulative_union(
        tracks, (h, w),
        closing_kernel=args.closing_kernel if args.closing_kernel > 0 else None,
    )

    # Save individual results
    cv2.imwrite(str(args.output / "cumulative_old.png"), render_overlay(old_intensity))
    cv2.imwrite(str(args.output / "cumulative_union.png"), render_overlay(union_intensity))
    cv2.imwrite(str(args.output / "cumulative_union_closed.png"), render_overlay(closed_intensity))

    # Save comparison
    comparison = render_comparison(old_intensity, union_intensity, closed_intensity)
    cv2.imwrite(str(args.output / "comparison.png"), comparison)

    # Save raw masks for inspection
    cv2.imwrite(str(args.output / "mask_old.png"), (old_intensity > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(args.output / "mask_union.png"), (union_intensity > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(args.output / "mask_union_closed.png"), (closed_intensity > 0).astype(np.uint8) * 255)

    # Metrics comparison
    print("\n--- Per-track geometry comparison ---")
    metrics = measure_track_metrics(tracks, (h, w), closing_kernel=args.closing_kernel)
    print(f"{'track':>5} {'n':>3} {'old_area':>10} {'union_area':>12} {'closed_area':>13} "
          f"{'old_len':>8} {'union_len':>10} {'closed_len':>11}")
    for m in metrics[:20]:  # show first 20
        def bbox_len(bbox):
            return bbox[3] - bbox[1]
        print(f"{m.track_id:>5} {m.n_frames:>3} {m.old_area_px:>10} {m.union_area_px:>12} "
              f"{m.closed_area_px:>13} {bbox_len(m.old_bbox):>8} "
              f"{bbox_len(m.union_bbox):>10} {bbox_len(m.closed_bbox):>11}")

    # Summary statistics
    if metrics:
        old_areas = [m.old_area_px for m in metrics]
        union_areas = [m.union_area_px for m in metrics]
        closed_areas = [m.closed_area_px for m in metrics]
        print(f"\n--- Summary ---")
        print(f"Tracks analyzed: {len(metrics)}")
        print(f"Old area   — mean: {np.mean(old_areas):.0f}, median: {np.median(old_areas):.0f}")
        print(f"Union area — mean: {np.mean(union_areas):.0f}, median: {np.median(union_areas):.0f}")
        print(f"Closed area— mean: {np.mean(closed_areas):.0f}, median: {np.median(closed_areas):.0f}")
        print(f"Union/Old  — mean ratio: {np.mean(np.array(union_areas)/np.maximum(old_areas,1)):.2f}x")
        print(f"Closed/Old — mean ratio: {np.mean(np.array(closed_areas)/np.maximum(old_areas,1)):.2f}x")

    print(f"\nDone. Outputs in {args.output}")


if __name__ == "__main__":
    main()
