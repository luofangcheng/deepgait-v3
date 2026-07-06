"""Cumulative footprint map: max-projection of MExG scores across all frames,
annotated with each detected footprint's peak-area bounding box.

Also generates a per-paw area-vs-time chart.

Outputs:
  /home/luofangcheng/Documents/ZCODE/tmp/footprint_max_projection.png
  /home/luofangcheng/Documents/ZCODE/tmp/footprint_area_timeseries.png
  /home/luofangcheng/Documents/ZCODE/tmp/footprint_per_paw_max_area.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.dataset import iter_frames
from deepgait3.core.pawprint.scoring import compute_mexg

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    print("Step 1: collecting frames + running detector ...")
    cap = cv2.VideoCapture(VIDEO)
    det = BiScaleDetector(strong_threshold=80, weak_threshold=20,
                           toe_max_distance_px=24.0, toe_min_overlap_frames=1)
    frames = []
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
    print(f"  {n_frames} frames, {len(det.closed_candidates)} candidates, "
          f"{det.stats['total_toes_recovered']} toes")

    # ---- Max-projection of MExG score ----
    print("Step 2: building max-projection ...")
    H, W = frames[0].shape[:2]
    max_score = np.zeros((H, W), dtype=np.int16)
    for f in frames:
        score = compute_mexg(f).astype(np.int16)
        np.maximum(max_score, score, out=max_score)

    # Normalize to 0-255 for visualization
    proj_8u = np.clip(max_score.astype(np.float32) / 3.0, 0, 255).astype(np.uint8)

    # ---- Annotate each candidate's peak area ----
    print("Step 3: annotating peak-area bounding boxes ...")
    per_paw_data: dict[str, list[dict]] = {}  # paw → list of {frame, area, cx, cy}

    proj_color = cv2.applyColorMap(proj_8u, cv2.COLORMAP_HOT)  # HOT colormap
    # Only color pixels with signal; background stays dark
    bg_mask = (max_score < 20)
    proj_color[bg_mask] = 0

    for c in det.closed_candidates:
        if c.n_strong_frames < 3:
            continue
        # Peak-area frame
        peak = max(c.strong_blobs, key=lambda b: b.area_px)
        x1, y1, x2, y2 = peak.bbox
        cx, cy = int(peak.cx_px), int(peak.cy_px)
        area = peak.area_px
        frame_idx = peak.frame_idx

        # Draw each peak footprint as a green circle + text
        cv2.circle(proj_color, (cx, cy), max(10, int(np.sqrt(area / np.pi) * 2)),
                   (0, 255, 0), 2)
        # Try to infer paw label from track position (simplistic: y < H/2 → front, else hind)
        label = f"f{frame_idx} a{area}px"
        cv2.putText(proj_color, label, (cx + 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Per-paw tracking (use coarse paw-like grouping by position)
        if cy < H // 2:
            paw_region = "upper"  # forelimb
        else:
            paw_region = "lower"  # hindlimb
        key = paw_region
        per_paw_data.setdefault(key, []).append({
            "frame": frame_idx,
            "area_px": area,
            "cx": cx,
            "cy": cy,
        })

    # ---- Per-paw area over time ----
    print("Step 4: computing per-paw area timeseries ...")
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = {"upper": "#ff7f0e", "lower": "#1f77b4"}
    labels = {"upper": "Fore paws (upper)", "lower": "Hind paws (lower)"}
    for paw_region, records in per_paw_data.items():
        records.sort(key=lambda r: r["frame"])
        frames_arr = [r["frame"] for r in records]
        areas_arr = [r["area_px"] for r in records]
        ax.plot(frames_arr, areas_arr, "o-", markersize=6, linewidth=1.5,
                color=colors.get(paw_region, "gray"),
                label=labels.get(paw_region, paw_region))
    ax.set_xlabel("frame index")
    ax.set_ylabel("peak footprint area (px²)")
    ax.set_title(f"Footprint peak area per track — BiScaleDetector(MExG) on test1.mp4")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    area_ts_path = OUT_DIR / "footprint_area_timeseries.png"
    fig.savefig(area_ts_path, dpi=120)
    print(f"  saved: {area_ts_path}")

    # ---- Save max-projection ----
    maxproj_path = OUT_DIR / "footprint_max_projection.png"
    cv2.imwrite(str(maxproj_path), proj_color)
    print(f"  saved: {maxproj_path}")

    # ---- Save per-paw CSV ----
    csv_path = OUT_DIR / "footprint_per_paw_max_area.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paw_region", "frame", "area_px", "cx", "cy"])
        for paw_region, records in per_paw_data.items():
            for r in records:
                w.writerow([paw_region, r["frame"], r["area_px"], r["cx"], r["cy"]])
    print(f"  saved: {csv_path}")

    # ---- Cumulative area over time ----
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    cumulative = []
    running = 0
    for f_idx in range(1, n_frames + 1):
        frame_candidates = [c for c in det.closed_candidates
                             if c.first_frame <= f_idx <= c.last_frame]
        # Sum area of active footprints at this frame
        area_this_frame = 0
        for c in frame_candidates:
            blobs_here = [b for b in c.strong_blobs if b.frame_idx == f_idx]
            if blobs_here:
                area_this_frame += max(b.area_px for b in blobs_here)
        running += area_this_frame
        cumulative.append(running)
    ax2.plot(range(1, n_frames + 1), cumulative, color="darkgreen", linewidth=1.5)
    ax2.set_xlabel("frame index")
    ax2.set_ylabel("cumulative footprint area (px²)")
    ax2.set_title("Cumulative footprint area over time")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    cum_path = OUT_DIR / "footprint_cumulative_area.png"
    fig2.savefig(cum_path, dpi=120)
    print(f"  saved: {cum_path}")

    # ---- Print summary ----
    print("\n=== Footprint Area Summary ===")
    print(f"Total prints detected: {len(det.closed_candidates)}")
    print(f"Valid prints (≥3 frames): {sum(1 for c in det.closed_candidates if c.n_strong_frames >= 3)}")
    all_areas = [b.area_px for c in det.closed_candidates for b in c.strong_blobs]
    if all_areas:
        print(f"Mean peak area: {np.mean(all_areas):.0f} px²")
        print(f"Max peak area:  {np.max(all_areas):.0f} px²")
        print(f"Median peak area: {np.median(all_areas):.0f} px²")
        print(f"Total cumulative area: {cumulative[-1]:.0f} px²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())