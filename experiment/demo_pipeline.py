"""End-to-end demo: YOLO detection → FootMask → tracking → FootprintCycle.

Usage::

    python experiment/demo_pipeline.py

Outputs experiment/outputs/pipeline_demo/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepgait3.core.pawprint.models import FootMask
from deepgait3.core.pawprint.tracker import IoUFootprintTracker
from deepgait3.core.pawprint.cycle_builder import build_cycles


def yolo_to_footmasks(
    frame: np.ndarray,
    results,
    bg_G: np.ndarray | None = None,
) -> list[FootMask]:
    """Convert YOLOv8-seg results to the FootMask format expected by downstream pipeline.

    Parameters
    ----------
    frame : original BGR image (H, W, 3)
    results : ultralytics Results object from a single-frame inference
    bg_G : optional float32 median green-channel background (H, W), for intensity extraction

    Returns
    -------
    List[FootMask] compatible with IoUFootprintTracker and build_cycles.
    """
    h, w = frame.shape[:2]
    footmasks: list[FootMask] = []

    # Unwrap list if needed (single-frame model() returns [Results])
    r = results[0] if isinstance(results, list) else results
    if r.masks is None:
        return footmasks

    masks_np = r.masks.data.cpu().numpy()  # (N, mH, mW)
    boxes = r.boxes

    # Compute green-delta once (for intensity fields)
    G = frame[:, :, 1].astype(np.float32)
    if bg_G is not None:
        delta = G - bg_G
    else:
        delta = G  # fallback

    for i, mask in enumerate(masks_np):
        mH, mW = mask.shape
        if mH != h or mW != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        mask_bin = (mask > 0.5).astype(np.uint8)
        area_px = int(mask_bin.sum())
        if area_px < 5:
            continue

        # Compute centroid from binary mask moments
        moments = cv2.moments(mask_bin)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = 0.0, 0.0

        # Bounding box
        ys, xs = np.where(mask_bin)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        # Extract intensity crop from green-delta using the mask region
        pad = 2
        py1 = max(0, y1 - pad)
        py2 = min(h, y2 + pad)
        px1 = max(0, x1 - pad)
        px2 = min(w, x2 + pad)

        mask_crop = mask_bin[py1:py2, px1:px2].astype(bool)
        delta_crop = delta[py1:py2, px1:px2]

        # Intensity stats from within-mask pixels
        if mask_crop.sum() > 0:
            intensities = delta_crop[mask_crop]
            mean_intensity = float(intensities.mean())
            peak_intensity = float(intensities.max())
        else:
            mean_intensity = 0.0
            peak_intensity = 0.0

        # Pressure map (same formula as current cluster_blobs_into_feet)
        pressure_k, pressure_alpha, pressure_b = 18.0, 0.75, 8.0
        pressure_map = pressure_k * np.maximum(delta_crop - pressure_b, 0) ** pressure_alpha

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
        footmasks.append(fm)

    return footmasks


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO → tracker → cycles demo")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pawprint_yolo/weights/best.pt")
    parser.add_argument("--frames-dir", type=Path,
                        default=PROJECT_ROOT / "experiment/data/raw")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pipeline_demo")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--px-per-mm", type=float, default=1.92)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}. Run train_yolo.py first.")
        return

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    # Load frames
    frame_paths = sorted(args.frames_dir.glob("frame_*.png"))
    frames = []
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)

    print(f"Loaded {len(frames)} frames")

    # Compute median background (for green-delta intensity)
    green_stack = np.stack([f[:, :, 1].astype(np.float32) for f in frames])
    bg_G = np.median(green_stack, axis=0)
    print(f"Background computed, shape={bg_G.shape}")

    # Initialize tracker
    tracker = IoUFootprintTracker(
        frame_shape=frames[0].shape[:2],
        iou_min=0.3,
        max_gap_frames=3,
    )

    # Per-frame loop: YOLO → FootMask → tracker
    print("\nProcessing frames...")
    for idx, frame in enumerate(frames):
        results = model(frame, verbose=False)
        footmasks = yolo_to_footmasks(frame, results, bg_G)
        tracker.update(idx + 1, footmasks)

        if (idx + 1) % 20 == 0:
            print(f"  Frame {idx + 1}/{len(frames)}")

    # Finalize tracks and build cycles
    tracks = tracker.finalize()
    print(f"\nTracks: {len(tracks)}")

    cycles = build_cycles(tracks, fps=args.fps, px_per_mm=args.px_per_mm)
    print(f"Cycles: {len(cycles)}")

    # Summary
    total_frames = sum(c.n_frames for c in cycles)
    print(f"Total footprint frames: {total_frames}")
    print(f"Average cycle duration: {np.mean([c.duration_s for c in cycles]):.3f}s")
    print(f"Average max area: {np.mean([c.max_area_px for c in cycles]):.1f} px")

    # Save per-cycle summary
    summary_path = args.output / "cycles_summary.csv"
    with open(summary_path, "w") as f:
        f.write("cycle_id,touchdown_frame,liftoff_frame,duration_s,max_area_px,n_frames\n")
        for c in cycles:
            f.write(f"{c.cycle_id},{c.touchdown_frame},{c.liftoff_frame},"
                    f"{c.duration_s:.3f},{c.max_area_px},{c.n_frames}\n")
    print(f"\nSummary saved to {summary_path}")

    # Save per-cycle visualizations
    print("\nSaving cycle visualizations...")
    for c in cycles[:10]:  # first 10 cycles
        # Draw all frames in this cycle
        for fr in c.frames[:5]:  # first 5 frames per cycle
            frame_img = frames[fr.frame - 1].copy()
            x1, y1, x2, y2 = fr.bbox_x1, fr.bbox_y1, fr.bbox_x2, fr.bbox_y2
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame_img, f"cycle{c.cycle_id} f{fr.frame}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
            out_name = f"cycle_{c.cycle_id:03d}_frame_{fr.frame:04d}.png"
            cv2.imwrite(str(args.output / out_name), frame_img)

    print(f"Done. Results in {args.output}")


if __name__ == "__main__":
    main()
