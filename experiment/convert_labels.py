"""Convert footprints.db bbox data to YOLO segmentation/detection labels.

Reads per-frame footprint bounding boxes from the SQLite database and
writes one ``.txt`` label file per frame in YOLO format.

Usage::

    python experiment/convert_labels.py \
        --db tests/fixtures/mouse_001/footprints.db \
        --frames experiment/data/raw \
        --out experiment/data/labels

Output format (YOLO detection, normalized):
    class_id cx cy w h
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def convert_db_to_yolo(
    db_path: Path,
    frames_dir: Path,
    out_dir: Path,
    *,
    img_width: int = 1920,
    img_height: int = 384,
    min_area_px: int = 5,
) -> int:
    """Extract per-frame bbox annotations and write YOLO label files.

    Returns the number of label files written.
    """
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # Get all footprint frames with bbox, grouped by frame number.
    cur.execute("""
        SELECT ff.frame, ff.bbox_x1, ff.bbox_y1, ff.bbox_x2, ff.bbox_y2,
               ff.area_px
        FROM footprint_frame ff
        ORDER BY ff.frame, ff.area_px DESC
    """)
    rows = cur.fetchall()
    db.close()

    # Group by frame
    frame_anns: dict[int, list[tuple[int, int, int, int]]] = {}
    for frame, x1, y1, x2, y2, area in rows:
        if area < min_area_px:
            continue
        frame_anns.setdefault(frame, []).append((x1, y1, x2, y2))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for frame_num, bboxes in frame_anns.items():
        frame_name = f"frame_{frame_num:04d}"
        label_path = out_dir / f"{frame_name}.txt"

        lines = []
        for x1, y1, x2, y2 in bboxes:
            # Clamp to image bounds
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))

            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 x4 y4
            # Use bbox corners as polygon approximation
            nx1 = x1 / img_width
            ny1 = y1 / img_height
            nx2 = x2 / img_width
            ny2 = y2 / img_height

            lines.append(f"0 {nx1:.6f} {ny1:.6f} {nx2:.6f} {ny1:.6f} "
                         f"{nx2:.6f} {ny2:.6f} {nx1:.6f} {ny2:.6f}")

        if lines:
            label_path.write_text("\n".join(lines))
            written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert footprints.db bbox data to YOLO labels"
    )
    parser.add_argument("--db", required=True, type=Path, help="Path to footprints.db")
    parser.add_argument("--frames", required=True, type=Path, help="Directory of raw frame PNGs")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for YOLO .txt labels")
    parser.add_argument("--img-width", type=int, default=1920)
    parser.add_argument("--img-height", type=int, default=384)
    parser.add_argument("--min-area-px", type=int, default=5, help="Minimum footprint area")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction of frames for training (rest for val)")
    args = parser.parse_args()

    n = convert_db_to_yolo(
        args.db, args.frames, args.out,
        img_width=args.img_width, img_height=args.img_height,
        min_area_px=args.min_area_px,
    )
    print(f"Wrote {n} label files to {args.out}")

    # Report frame -> label coverage
    frame_pngs = sorted(args.frames.glob("frame_*.png"))
    label_txts = sorted(args.out.glob("frame_*.txt"))
    print(f"Frames: {len(frame_pngs)}, Labels: {len(label_txts)}")
    if frame_pngs:
        coverage = len(label_txts) / len(frame_pngs) * 100
        print(f"Coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
