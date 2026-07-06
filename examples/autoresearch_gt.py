"""GT-driven autoresearch: tune BiScaleDetector hyperparameters against
real ground truth (commercial Excel export).

For each (config, video) pair:
1. Run BiScaleDetector on the video.
2. Convert its candidate centroids (peak frame, peak centroid) into
   a list of predictions.
3. Compute precision / recall / F1 against the GT from the xlsx.
4. Score = F1 (with recall boost: matches with no GT and GT with no
   match both penalize the score).

Outputs (to /home/luofangcheng/Documents/ZCODE/tmp/):
- autoresearch_gt_<timestamp>.csv
- autoresearch_gt_top10.csv + .png
- autoresearch_gt_heatmap.png
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.benchmark import evaluate
from deepgait3.core.pawprint.dataset import collect_videos, iter_frames
from deepgait3.core.pawprint.gt_loaders import load_gt_from_xlsx, save_gt_json

OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")


def _predictions_from_detector(det: BiScaleDetector) -> list[dict]:
    """Convert BiScaleDetector.closed_candidates into a prediction list.

    Each candidate contributes one prediction:
        {frame_idx: peak_frame (mid of strong sequence),
         cx, cy:    median centroid over strong blobs}
    """
    preds = []
    for c in det.closed_candidates:
        if not c.strong_blobs:
            continue
        cx = float(np.median([b.cx_px for b in c.strong_blobs]))
        cy = float(np.median([b.cy_px for b in c.strong_blobs]))
        # peak frame = frame with the largest strong blob area
        peak = max(c.strong_blobs, key=lambda b: b.area_px)
        preds.append({
            "frame_idx": peak.frame_idx if peak.frame_idx > 0 else c.first_frame,
            "cx": cx,
            "cy": cy,
            "n_strong_frames": c.n_strong_frames,
            "n_weak_frames": c.n_weak_frames,
        })
    return preds


def _run_one(video_path: Path, gt: list, strong: int, weak: int,
              dist: float, overlap: int,
              match_dist_px: float = 40.0,
              match_frame_tol: int = 5,
              spatial_check: bool = True) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    det = BiScaleDetector(
        strong_threshold=strong, weak_threshold=weak,
        toe_max_distance_px=dist, toe_min_overlap_frames=overlap,
    )
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        det.process_frame(frame, idx)
    cap.release()
    det.finalize()
    preds = _predictions_from_detector(det)
    eval_result = evaluate(
        gt, preds,
        match_distance_px=match_dist_px,
        match_frame_tolerance=match_frame_tol,
        spatial_check=spatial_check,
    )
    return {
        "strong": strong, "weak": weak, "dist": dist, "overlap": overlap,
        "n_predictions": len(preds),
        "n_candidates": len(det.closed_candidates),
        "n_valid_candidates": sum(1 for c in det.closed_candidates if c.n_strong_frames >= 5),
        "n_toes_recovered": det.stats["total_toes_recovered"],
        "precision": eval_result.precision,
        "recall": eval_result.recall,
        "f1": eval_result.f1,
        "mean_error_px": eval_result.mean_error_px,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to fTIR mp4 video")
    ap.add_argument("gt_xlsx", help="path to commercial gait Excel export")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--coord-unit", default="px",
                     choices=["px", "cm"])
    ap.add_argument("--match-dist-px", type=float, default=40.0)
    ap.add_argument("--match-frame-tol", type=int, default=5)
    ap.add_argument("--no-spatial", action="store_true",
                     help="ignore pixel coordinates; match on frame only")
    ap.add_argument("--limit-configs", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"FAIL: video not found: {video_path}")
        return 1
    gt = load_gt_from_xlsx(args.gt_xlsx, fps=args.fps, coord_unit=args.coord_unit)
    if not gt:
        print("FAIL: no valid GT prints loaded")
        return 1
    print(f"Video: {video_path.name}")
    print(f"GT: {len(gt)} prints from {args.gt_xlsx}")
    print(f"Match criteria: dist <= {args.match_dist_px} px, |dt| <= {args.match_frame_tol} frames")
    print(f"GT print locations (frame, x, y, paw):")
    for g in gt[:8]:
        print(f"  {g.paw_id:>3} frame={g.frame_idx:>4} pos=({g.cx_px:.0f}, {g.cy_px:.0f})")
    if len(gt) > 8:
        print(f"  ... +{len(gt) - 8} more")

    # Search grid
    strong_thrs = [40, 60, 80, 100, 120]
    weak_thrs = [3, 5, 8, 12, 18]
    toe_dists = [6, 9, 12, 18, 24]
    toe_overlaps = [1, 2, 3, 5]
    configs = [
        (st, wt, td, to)
        for st in strong_thrs
        for wt in weak_thrs
        for td in toe_dists
        for to in toe_overlaps
        if st > wt
    ]
    if args.limit_configs:
        configs = configs[: args.limit_configs]
    print(f"\nGrid: {len(configs)} configs (strong × weak × dist × overlap)")

    results: list[dict] = []
    t0 = time.time()
    for i, (st, wt, td, to) in enumerate(configs):
        r = _run_one(video_path, gt, st, wt, td, to,
                      match_dist_px=args.match_dist_px,
                      match_frame_tol=args.match_frame_tol,
                      spatial_check=not args.no_spatial)
        results.append(r)
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed * len(configs) / (i + 1) - elapsed
            print(f"  [{i+1}/{len(configs)}] st={st} wt={wt} d={td:.0f} o={to} "
                  f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.3f} "
                  f"err={r['mean_error_px']:.1f}px "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
    total = time.time() - t0
    print(f"\nDone in {total:.0f}s")

    # Rank by F1
    results.sort(key=lambda r: r["f1"], reverse=True)

    # Top 10
    print("\n" + "=" * 100)
    print("TOP 10 BY F1 SCORE")
    print("=" * 100)
    print(f"{'rank':>4} {'strong':>7} {'weak':>5} {'dist':>5} {'overlap':>7} "
          f"{'cands':>6}{'valid':>6}{'toes':>5} "
          f"{'P':>6}{'R':>6}{'F1':>7}{'err':>6}")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>4} {r['strong']:>7} {r['weak']:>5} {r['dist']:>5.0f} "
              f"{r['overlap']:>7} {r['n_candidates']:>6}{r['n_valid_candidates']:>6}"
              f"{r['n_toes_recovered']:>5} "
              f"{r['precision']:>6.2f}{r['recall']:>6.2f}{r['f1']:>7.3f}"
              f"{r['mean_error_px']:>6.1f}")

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"autoresearch_gt_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "strong_thr", "weak_thr", "toe_max_dist_px",
                     "toe_min_overlap_frames", "n_predictions", "n_candidates",
                     "n_valid_candidates", "n_toes_recovered",
                     "precision", "recall", "f1", "mean_error_px"])
        for i, r in enumerate(results):
            w.writerow([i + 1, r["strong"], r["weak"], r["dist"], r["overlap"],
                         r["n_predictions"], r["n_candidates"],
                         r["n_valid_candidates"], r["n_toes_recovered"],
                         round(r["precision"], 4), round(r["recall"], 4),
                         round(r["f1"], 4), round(r["mean_error_px"], 2)])
    print(f"\nCSV: {csv_path}")

    best = results[0]
    best_path = OUT_DIR / f"autoresearch_gt_best_{ts}.json"
    best_path.write_text(json.dumps({
        "strong_threshold": best["strong"],
        "weak_threshold": best["weak"],
        "toe_max_distance_px": best["dist"],
        "toe_min_overlap_frames": best["overlap"],
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
        "mean_error_px": best["mean_error_px"],
        "n_toes_recovered": best["n_toes_recovered"],
    }, indent=2))
    print(f"Best config: {best_path}")

    # Heatmap: F1 as function of (strong, weak), averaged over (dist, overlap)
    grid = np.zeros((len(strong_thrs), len(weak_thrs)))
    cnt = np.zeros((len(strong_thrs), len(weak_thrs)))
    for r in results:
        i = strong_thrs.index(r["strong"])
        j = weak_thrs.index(r["weak"])
        grid[i, j] += r["f1"]
        cnt[i, j] += 1
    grid = np.divide(grid, cnt, out=np.zeros_like(grid), where=cnt > 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, cmap="viridis", origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(weak_thrs)))
    ax.set_xticklabels(weak_thrs)
    ax.set_yticks(range(len(strong_thrs)))
    ax.set_yticklabels(strong_thrs)
    ax.set_xlabel("weak_threshold")
    ax.set_ylabel("strong_threshold")
    bcfg = best
    ax.set_title(f"Avg F1 across (dist × overlap)\nbest: st={bcfg['strong']}, "
                  f"wt={bcfg['weak']}, d={bcfg['dist']:.0f}, o={bcfg['overlap']} "
                  f"(F1={bcfg['f1']:.3f})")
    for i in range(len(strong_thrs)):
        for j in range(len(weak_thrs)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < grid.max() * 0.7 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label="F1")
    fig.tight_layout()
    heatmap_path = OUT_DIR / f"autoresearch_gt_heatmap_{ts}.png"
    fig.savefig(heatmap_path, dpi=130)
    print(f"Heatmap: {heatmap_path}")

    # Top-10 bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = [f"st={r['strong']}/wt={r['weak']}\nd={r['dist']:.0f}/o={r['overlap']}\nF1={r['f1']:.3f}"
              for r in results[:10]]
    f1s = [r["f1"] for r in results[:10]]
    ax.barh(range(10), f1s, color="steelblue")
    ax.set_yticks(range(10))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.set_title(f"Top 10 configs vs real GT ({video_path.name}, {len(gt)} prints)")
    fig.tight_layout()
    bar_path = OUT_DIR / f"autoresearch_gt_top10_{ts}.png"
    fig.savefig(bar_path, dpi=130)
    print(f"Top-10 bars: {bar_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())