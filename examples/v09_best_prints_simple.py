"""Simple overlay: for each GT print, find the closest algorithm blob at that
frame and draw the bbox on the video frame. Saves per-print PNGs + a grid."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.dataset import iter_frames
from deepgait3.core.pawprint.gt_loaders import load_gt_from_xlsx

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
XLSX = "/home/luofangcheng/Documents/ZCODE/data-test1.xlsx"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    gt = load_gt_from_xlsx(XLSX, fps=60, coord_unit="px")
    print(f"GT: {len(gt)} prints")

    # Run detector
    print("Running BiScaleDetector(MExG, st=80, wt=20) ...")
    cap = cv2.VideoCapture(VIDEO)
    det = BiScaleDetector(strong_threshold=80, weak_threshold=20,
                           toe_max_distance_px=24.0, toe_min_overlap_frames=1)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        det.process_frame(frame, idx)
    cap.release()
    det.finalize()
    print(f"  candidates={len(det.closed_candidates)}, toes={det.stats['total_toes_recovered']}")

    # Build lookup: frame -> list of strong blob bboxes
    frame_blobs: dict[int, list[tuple]] = {}
    for c in det.closed_candidates:
        for b in c.strong_blobs:
            fid = b.frame_idx
            if fid <= 0:
                continue
            frame_blobs.setdefault(fid, []).append((c.track_id, b.bbox, b.area_px, b.cx_px, b.cy_px))

    # For each GT print, find the closest blob at GT frame (or ±3 frames)
    results = []
    for i, g in enumerate(gt):
        best_bbox = None
        best_dist = 999
        best_area = 0
        best_fid = -1
        for dt in range(-3, 4):
            fid = g.frame_idx + dt
            if fid in frame_blobs:
                for tid, bbox, area, cx, cy in frame_blobs[fid]:
                    # Pick the largest blob as representative
                    if area > best_area:
                        best_area = area
                        best_bbox = bbox
                        best_dist = abs(dt)
                        best_fid = fid

        status = "matched" if best_bbox is not None else "missed"
        results.append({
            "i": i, "frame": g.frame_idx, "paw": g.paw_id,
            "status": status, "match_frame": best_fid,
            "area": best_area, "dist_frames": best_dist,
        })

        # Draw overlay
        cap2 = cv2.VideoCapture(VIDEO)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, g.frame_idx - 1)
        ok, frame = cap2.read()
        cap2.release()
        if not ok:
            continue
        vis = frame.copy()
        if best_bbox is not None:
            x1, y1, x2, y2 = best_bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{g.paw_id} area={best_area}px dist={best_dist}frames",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(vis, f"{g.paw_id} NO MATCH", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(vis, f"GT frame {g.frame_idx}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_path = OUT_DIR / f"best_print_{i:02d}_{g.paw_id}_f{g.frame_idx:03d}.png"
        cv2.imwrite(str(out_path), vis)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(gt)}] {out_path.name} {status} area={best_area}")

    # Print table
    print("\n" + "=" * 70)
    print(f"{'#':>3} {'f':>4} {'paw':>4} {'status':>8} {'match_f':>7} {'area':>6} {'Δf':>3}")
    print("-" * 70)
    for r in results:
        mf = r["match_frame"] if r["match_frame"] > 0 else "-"
        print(f"{r['i']:>3} {r['frame']:>4} {r['paw']:>4} {r['status']:>8} "
              f"{str(mf):>7} {r['area']:>6} {r['dist_frames']:>3}")

    matched = [r for r in results if r["status"] == "matched"]
    print(f"\nMatched: {len(matched)}/{len(results)}")

    # Save CSV
    csv_path = OUT_DIR / "best_algo_per_print.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "frame", "paw", "status", "match_frame", "area", "dist_frames"])
        for r in results:
            w.writerow([r["i"], r["frame"], r["paw"], r["status"],
                         r["match_frame"], r["area"], r["dist_frames"]])
    print(f"CSV: {csv_path}")

    # Grid image (4×7)
    fig, axes = plt.subplots(5, 5, figsize=(22, 18))
    for i, r in enumerate(results):
        ax = axes[i // 5, i % 5]
        png = OUT_DIR / f"best_print_{i:02d}_{r['paw']}_f{r['frame']:03d}.png"
        if png.exists():
            img = cv2.imread(str(png))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        color = "green" if r["status"] == "matched" else "red"
        ax.set_title(f"#{i+1} {r['paw']} f={r['frame']} | {r['status']}", color=color, fontsize=9)
        ax.axis("off")
    fig.suptitle("BiScaleDetector(MExG) — 25 GT prints overlay (green=detected, red=missed)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    grid_path = OUT_DIR / "best_algorithm_prints_grid.png"
    fig.savefig(grid_path, dpi=100)
    print(f"Grid: {grid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())