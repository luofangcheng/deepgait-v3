"""Speed benchmark: current blob detection vs YOLOv8-seg.

Usage::

    python experiment/eval_speed.py [--model experiment/outputs/pawprint_yolo/weights/best.pt]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add project root for deepgait3 imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def benchmark_blob_pipeline(
    frames: list[np.ndarray],
    bg_G: np.ndarray,
    n_warmup: int = 10,
    n_bench: int = 100,
) -> dict:
    """Benchmark the current blob detection + clustering pipeline."""
    from deepgait3.core.pawprint.detection import detect_blobs
    from deepgait3.core.pawprint.grouping import cluster_blobs_into_feet

    # Warmup
    for i in range(min(n_warmup, len(frames))):
        blobs = detect_blobs(frames[i], bg_G, tau_paw=10, min_area_px=10)
        _ = cluster_blobs_into_feet(blobs, D_merge_px=23.0, frame_shape=frames[i].shape[:2])

    # Benchmark
    times = []
    for i in range(n_warmup, min(n_warmup + n_bench, len(frames))):
        t0 = time.perf_counter()
        blobs = detect_blobs(frames[i], bg_G, tau_paw=10, min_area_px=10)
        feet = cluster_blobs_into_feet(blobs, D_merge_px=23.0, frame_shape=frames[i].shape[:2])
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "name": "Blob+Cluster",
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


def benchmark_yolo(
    model,
    frames: list[np.ndarray],
    n_warmup: int = 10,
    n_bench: int = 100,
) -> dict:
    """Benchmark YOLOv8-seg inference."""
    # Warmup
    for i in range(min(n_warmup, len(frames))):
        _ = model(frames[i], verbose=False)

    # Benchmark
    times = []
    for i in range(n_warmup, min(n_warmup + n_bench, len(frames))):
        t0 = time.perf_counter()
        results = model(frames[i], verbose=False)
        # Simulate minimal post-processing: extract masks
        if results[0].masks is not None:
            _ = results[0].masks.xy  # polygon coordinates
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    return {
        "name": "YOLOv8n-seg",
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000.0 / np.mean(times)),
    }


def load_frames(frame_dir: Path, max_frames: int = 150) -> list[np.ndarray]:
    """Load frames from a directory of PNGs."""
    frames = []
    for p in sorted(frame_dir.glob("frame_*.png"))[:max_frames]:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Speed benchmark blob vs YOLO")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pawprint_yolo/weights/best.pt",
                        help="Path to trained YOLO model")
    parser.add_argument("--frames", type=Path,
                        default=PROJECT_ROOT / "experiment/data/raw",
                        help="Directory of test frames")
    parser.add_argument("--n-frames", type=int, default=150)
    args = parser.parse_args()

    print("Loading frames...")
    frames = load_frames(args.frames, args.n_frames)
    print(f"Loaded {len(frames)} frames, shape={frames[0].shape}")

    # Compute median background (same as current pipeline)
    print("Computing background...")
    green_frames = np.stack([f[:, :, 1].astype(np.float32) for f in frames])
    bg_G = np.median(green_frames, axis=0)

    # 1. Benchmark blob pipeline
    print("\n--- Blob Detection Pipeline ---")
    blob_result = benchmark_blob_pipeline(frames, bg_G)
    print(f"  Mean: {blob_result['mean_ms']:.2f} ms")
    print(f"  Std:  {blob_result['std_ms']:.2f} ms")
    print(f"  Min:  {blob_result['min_ms']:.2f} ms")
    print(f"  Max:  {blob_result['max_ms']:.2f} ms")
    print(f"  FPS:  {blob_result['fps']:.1f}")

    # 2. Benchmark YOLO
    if args.model.exists():
        print("\n--- YOLOv8n-seg ---")
        from ultralytics import YOLO
        model = YOLO(str(args.model))
        yolo_result = benchmark_yolo(model, frames)
        print(f"  Mean: {yolo_result['mean_ms']:.2f} ms")
        print(f"  Std:  {yolo_result['std_ms']:.2f} ms")
        print(f"  Min:  {yolo_result['min_ms']:.2f} ms")
        print(f"  Max:  {yolo_result['max_ms']:.2f} ms")
        print(f"  FPS:  {yolo_result['fps']:.1f}")

        # Comparison
        speedup = blob_result['mean_ms'] / yolo_result['mean_ms']
        print(f"\n=== Speedup: {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} ===")
    else:
        print(f"\nModel not found at {args.model}. Run train_yolo.py first.")


if __name__ == "__main__":
    main()
