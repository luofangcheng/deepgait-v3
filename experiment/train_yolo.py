"""Train YOLOv8-seg on fTIR pawprint data.

Usage::

    python experiment/train_yolo.py

Output: experiment/outputs/pawprint_yolo.pt
"""
from __future__ import annotations

from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_yaml = project_root / "experiment" / "train" / "pawprint.yaml"
    output_dir = project_root / "experiment" / "outputs"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained YOLOv8n-seg (nano, best speed/accuracy tradeoff)
    model = YOLO("yolov8n-seg.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,           # RTX 3060
        workers=4,
        project=str(output_dir),
        name="pawprint_yolo",
        exist_ok=True,
        patience=20,        # early stopping if no improvement for 20 epochs
        save=True,
        save_period=10,     # checkpoint every 10 epochs
        val=True,
        # Segmentation-specific
        task="segment",
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        # Augmentation (mild — fTIR images are consistent)
        hsv_h=0.0,  # no hue (grayscale-ish)
        hsv_s=0.0,  # no saturation
        hsv_v=0.1,  # mild brightness jitter
        degrees=0,
        translate=0.05,
        scale=0.1,
        fliplr=0.5,
        mosaic=0.0,
        erasing=0.0,
    )

    # Evaluate on val set
    metrics = model.val()
    print(f"\nValidation results:")
    print(f"  mAP50-95 (bbox):  {metrics.box.map:.4f}")
    print(f"  mAP50-95 (mask):  {metrics.seg.map:.4f}")
    print(f"\nModel saved to: {output_dir / 'pawprint_yolo' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
