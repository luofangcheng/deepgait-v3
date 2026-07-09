"""YOLO detection on video → cumulative footprint visualization.

Usage::

    python experiment/demo_video.py [--video /home/luofangcheng/Documents/ZCODE/test1.mp4]

Outputs experiment/outputs/video_demo/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepgait3.core.pawprint.tracker import IoUFootprintTracker
from demo_pipeline import yolo_to_footmasks


def build_cumulative_maps(
    frames: list[np.ndarray],
    bg_G: np.ndarray,
    tracks: list,
    frame_shape: tuple,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cumulative footprint visualizations from tracked footprints.

    Returns
    -------
    cumulative_mask : (H, W) uint8
        Max-projection binary mask of all footprints.
    cumulative_intensity : (H, W) uint8
        Green-on-black max-projection of footprint intensities.
    cumulative_overlay : (H, W, 3) uint8
        White-background overlay with mouse center trail.
    """
    h, w = frame_shape
    cumulative_mask = np.zeros((h, w), dtype=np.uint8)
    cumulative_intensity = np.zeros((h, w), dtype=np.float32)

    for track in tracks:
        for frame_idx, fm in track.foots:
            px1, py1, px2, py2 = fm.bbox_xyxy_padded

            # Place mask at global position
            mask_global = np.zeros((h, w), dtype=np.uint8)
            crop_h = py2 - py1
            crop_w = px2 - px1
            mask_crop = fm.mask_padded.astype(np.uint8)
            if mask_crop.shape[:2] != (crop_h, crop_w):
                mask_crop = cv2.resize(mask_crop, (crop_w, crop_h))
            mask_global[py1:py2, px1:px2] = mask_crop

            # Max-projection mask
            cumulative_mask = np.maximum(cumulative_mask, mask_global)

            # Place intensity at global position
            intensity_global = np.zeros((h, w), dtype=np.float32)
            intensity_crop = fm.raw_intensity_crop
            if intensity_crop.shape[:2] != (crop_h, crop_w):
                intensity_crop = cv2.resize(intensity_crop, (crop_w, crop_h))
            # Only within mask
            intensity_crop = intensity_crop * (mask_crop > 0).astype(np.float32)
            intensity_global[py1:py2, px1:px2] = intensity_crop
            cumulative_intensity = np.maximum(cumulative_intensity, intensity_global)

    # Normalize intensity for display (green on black)
    if cumulative_intensity.max() > 0:
        intensity_display = (cumulative_intensity / cumulative_intensity.max() * 255).astype(np.uint8)
    else:
        intensity_display = np.zeros((h, w), dtype=np.uint8)

    # Black-background overlay: max green intensity projection (like GUI)
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if cumulative_intensity.max() > 0:
        norm_intensity = (cumulative_intensity / cumulative_intensity.max() * 255).astype(np.uint8)
        overlay[:, :, 1] = norm_intensity  # green channel = footprint intensity

    return cumulative_mask * 255, intensity_display, overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO video footprint accumulation")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pawprint_yolo/weights/best.pt")
    parser.add_argument("--video", type=Path,
                        default=Path("/home/luofangcheng/Documents/ZCODE/test1.mp4"))
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/video_demo")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}. Run train_yolo.py first.")
        return
    if not args.video.exists():
        print(f"Video not found: {args.video}")
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

    print(f"Video: {n_frames} frames, {w}x{h}, {fps} FPS")

    # Load all frames + compute background
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Loaded {len(frames)} frames")

    # Median background
    green_stack = np.stack([f[:, :, 1].astype(np.float32) for f in frames])
    bg_G = np.median(green_stack, axis=0)
    print("Background model ready")

    # YOLO detection + tracking
    tracker = IoUFootprintTracker(frame_shape=(h, w), iou_min=0.3, max_gap_frames=3)

    print("\nDetecting paws on each frame...")
    for idx, frame in enumerate(frames):
        results = model(frame, verbose=False)
        footmasks = yolo_to_footmasks(frame, results, bg_G)
        tracker.update(idx + 1, footmasks)

        if (idx + 1) % 30 == 0:
            print(f"  Frame {idx + 1}/{len(frames)}")

    tracks = tracker.finalize()
    print(f"\nTracks found: {len(tracks)}")
    total_footprints = sum(len(t.foots) for t in tracks)
    print(f"Total footprint detections: {total_footprints}")

    # Build cumulative maps
    print("\nBuilding cumulative footprint images...")
    cum_mask, cum_intensity, cum_overlay = build_cumulative_maps(frames, bg_G, tracks, (h, w))

    cv2.imwrite(str(args.output / "cumulative_mask.png"), cum_mask)
    cv2.imwrite(str(args.output / "cumulative_intensity.png"), cum_intensity)
    cv2.imwrite(str(args.output / "cumulative_overlay.png"), cum_overlay)
    print(f"Saved: cumulative_mask.png, cumulative_intensity.png, cumulative_overlay.png")

    # Also save a side-by-side comparison frame
    mid_frame = frames[len(frames) // 2]
    results = model(mid_frame, verbose=False)
    annotated = mid_frame.copy()
    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            mH, mW = mask.shape
            if mH != h or mW != w:
                mask = cv2.resize(mask, (w, h))
            mask_bin = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(args.output / "sample_detection.png"), annotated)

    print(f"\nDone. All results in {args.output}")


if __name__ == "__main__":
    main()
