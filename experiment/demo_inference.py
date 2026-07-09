"""Visual demo: YOLOv8-seg inference on fTIR frames.

Usage::

    python experiment/demo_inference.py [--model experiment/outputs/pawprint_yolo/weights/best.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PAW_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
]


def draw_results(frame: np.ndarray, results) -> np.ndarray:
    """Draw YOLO segmentation masks and boxes on frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()  # (N, mH, mW) — model-internal resolution
        boxes = results[0].boxes

        for i, mask in enumerate(masks):
            color = PAW_COLORS[i % len(PAW_COLORS)]

            # Resize mask from model resolution to original frame size
            mH, mW = mask.shape
            if mH != h or mW != w:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

            # Draw mask overlay
            mask_bin = (mask > 0.5).astype(np.uint8)
            overlay = np.zeros_like(frame)
            overlay[mask_bin == 1] = color
            annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

            # Draw contour
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2)

            # Draw bbox
            if boxes is not None and i < len(boxes):
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                cv2.putText(annotated, f"{conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 pawprint detection demo")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pawprint_yolo/weights/best.pt")
    parser.add_argument("--frames", type=Path,
                        default=PROJECT_ROOT / "experiment/data/raw")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/demo")
    parser.add_argument("--max-frames", type=int, default=30)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        print("Run train_yolo.py first.")
        return

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    frames = sorted(args.frames.glob("frame_*.png"))[:args.max_frames]
    print(f"Processing {len(frames)} frames...")

    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        results = model(img, verbose=False)
        annotated = draw_results(img, results)

        out_path = args.output / f"demo_{frame_path.stem}.png"
        cv2.imwrite(str(out_path), annotated)

        if i % 10 == 0:
            print(f"  {i + 1}/{len(frames)}")

    print(f"\nDone. Results saved to {args.output}")


if __name__ == "__main__":
    main()
