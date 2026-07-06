"""Cumulative footprint map v2: for each detected footprint track, extract the
frame where the footprint area is MAXIMUM (peak contact), then composite all
peak-contact footprints onto a single image. No labels, clean view.

Algorithm: BiScaleDetector(MExG, strong=80, weak=20).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.scoring import compute_mexg

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    # 1. Run detector
    print("Running BiScaleDetector(MExG, st=80, wt=20) ...")
    cap = cv2.VideoCapture(VIDEO)
    det = BiScaleDetector(strong_threshold=80, weak_threshold=20,
                           toe_max_distance_px=24.0, toe_min_overlap_frames=1)
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        frames.append(frame)
        det.process_frame(frame, idx)
    cap.release()
    det.finalize()
    n_frames = len(frames)
    H, W = frames[0].shape[:2]
    print(f"  {n_frames} frames, {len(det.closed_candidates)} candidates")

    # 2. For each candidate, find the peak-area strong blob
    peak_blobs: list[dict] = []  # {frame_idx, area, bbox, cx, cy}
    for c in det.closed_candidates:
        if c.n_strong_frames < 2:
            continue
        peak = max(c.strong_blobs, key=lambda b: b.area_px)
        peak_blobs.append({
            "frame_idx": peak.frame_idx,
            "area": peak.area_px,
            "bbox": peak.bbox,
            "cx": peak.cx_px,
            "cy": peak.cy_px,
        })
    print(f"  {len(peak_blobs)} peak-area footprints")

    # 3. Build composite image: extract each footprint's ExG mask at its peak frame,
    #    then overlay all onto a dark background
    composite = np.zeros((H, W), dtype=np.float32)
    footprint_count = np.zeros((H, W), dtype=np.float32)  # for normalization

    for pb in peak_blobs:
        fidx = pb["frame_idx"] - 1  # 0-indexed
        if fidx < 0 or fidx >= len(frames):
            continue
        frame = frames[fidx]
        score = compute_mexg(frame).astype(np.float32)
        # Binarize: only strong footprint pixels
        mask = (score > 80).astype(np.float32)
        composite += mask * score
        footprint_count += mask

    # Normalize: divide by number of footprints at each pixel
    footprint_count[footprint_count == 0] = 1.0
    avg_score = composite / footprint_count
    avg_score_8u = np.clip(avg_score / 2.0, 0, 255).astype(np.uint8)

    # Apply color map, background stays black
    color = cv2.applyColorMap(avg_score_8u, cv2.COLORMAP_HOT)
    color[avg_score_8u < 5] = 0

    # 4. Draw only the peak blob bounding box outlines (thin, subtle)
    for pb in peak_blobs:
        x1, y1, x2, y2 = pb["bbox"]
        # Thin green outline, no text
        cv2.rectangle(color, (x1, y1), (x2, y2), (0, 180, 0), 1)

    out_path = OUT_DIR / "footprint_peak_contact_composite.png"
    cv2.imwrite(str(out_path), color)
    print(f"  saved: {out_path}")

    # 5. Also save a clean version with no bbox at all
    color_clean = cv2.applyColorMap(avg_score_8u, cv2.COLORMAP_HOT)
    color_clean[avg_score_8u < 5] = 0
    out_clean = OUT_DIR / "footprint_peak_contact_clean.png"
    cv2.imwrite(str(out_clean), color_clean)
    print(f"  saved: {out_clean}")

    # 6. Peak area summary
    print(f"\nPeak-area footprints: {len(peak_blobs)}")
    areas = [p["area"] for p in peak_blobs]
    print(f"  Mean peak area: {np.mean(areas):.0f} px²")
    print(f"  Median peak area: {np.median(areas):.0f} px²")
    print(f"  Max peak area: {np.max(areas):.0f} px²")
    print(f"  Min peak area: {np.min(areas):.0f} px²")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())