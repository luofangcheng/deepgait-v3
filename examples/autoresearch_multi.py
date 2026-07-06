"""Multi-video autoresearch: grid search BiScaleDetector on every video in a dataset.

Runs the same 4-hyperparameter sweep as ``autoresearch.py`` but across
all videos found under ``--root`` paths. Per-video + aggregate scores are
reported. Use this when you add more real videos — it tells you whether
the optimal params generalize across recordings.

Usage::

    python examples/autoresearch_multi.py /path/to/videos/
    python examples/autoresearch_multi.py video1.mp4 video2.mp4 path/to/dir/

Outputs (to /home/luofangcheng/Documents/ZCODE/tmp/):
- autoresearch_multi_<timestamp>.csv       per-config per-video scores
- autoresearch_multi_top10.csv             top-10 configs averaged across videos
- autoresearch_multi_top10.png             bar chart
- autoresearch_multi_heatmap.png           strong × weak heatmap (avg score)
- autoresearch_multi_per_video.png         per-video score distribution
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.dataset import collect_videos, iter_frames, video_info
from examples.autoresearch import RunResult, run_one

OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")


def _run_with_name(video_path: Path, strong: int, weak: int,
                    dist: float, overlap: int) -> tuple[RunResult, str]:
    """Wrap run_one with the video's short name as the tag."""
    r = run_one(str(video_path), strong, weak, dist, overlap)
    return r, video_path.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="+", help="videos or directories to scan")
    ap.add_argument("--limit-videos", type=int, default=None,
                     help="cap number of videos (for fast iteration)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos = collect_videos(args.roots)
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        print(f"FAIL: no videos found under {args.roots}")
        return 1
    print(f"Found {len(videos)} videos:")
    for v in videos:
        info = video_info(v)
        print(f"  {v.name}  ({info.n_frames} frames, "
              f"{info.width}×{info.height}, {info.fps:.1f}fps)")

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
    print(f"\nGrid size: {len(configs)} configs × {len(videos)} videos "
          f"= {len(configs) * len(videos)} runs")

    # results[video_name][config_tuple] = RunResult
    results: dict[str, dict[tuple, RunResult]] = {}
    t0 = time.time()
    for vi, video in enumerate(videos):
        print(f"\n[{vi+1}/{len(videos)}] {video.name}")
        results[video.name] = {}
        for ci, (st, wt, td, to) in enumerate(configs):
            r, name = _run_with_name(video, st, wt, td, to)
            results[video.name][(st, wt, td, to)] = r
            if (ci + 1) % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (len(configs) * len(videos)) / (
                    (vi * len(configs) + ci + 1)
                ) - elapsed
                print(f"    [{ci+1}/{len(configs)}] score={r.score:.1f} "
                      f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
    total = time.time() - t0
    print(f"\nFinished in {total:.0f}s")

    # Aggregate: per-config average score across videos
    config_scores: dict[tuple, list[float]] = defaultdict(list)
    for vname, cfg_results in results.items():
        for cfg, r in cfg_results.items():
            config_scores[cfg].append(r.score)
    avg_scores = {cfg: float(np.mean(scores)) for cfg, scores in config_scores.items()}
    ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

    # Top 10
    print("\n" + "=" * 80)
    print(f"TOP 10 CONFIGURATIONS (avg over {len(videos)} videos)")
    print("=" * 80)
    print(f"{'rank':>4} {'strong':>7} {'weak':>5} {'dist_px':>8} {'overlap':>7} "
          f"{'avg_score':>10}{'min':>8}{'max':>8}{'std':>8}")
    top10 = ranked[:10]
    for i, (cfg, avg) in enumerate(top10):
        scores = config_scores[cfg]
        print(f"{i+1:>4} {cfg[0]:>7} {cfg[1]:>5} {cfg[2]:>8.0f} {cfg[3]:>7} "
              f"{avg:>10.1f}{min(scores):>8.1f}{max(scores):>8.1f}"
              f"{float(np.std(scores)):>8.1f}")

    # ---- Save artifacts ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV per (config, video)
    csv_path = OUT_DIR / f"autoresearch_multi_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "strong_thr", "weak_thr", "toe_max_dist_px",
                     "toe_min_overlap_frames", "n_strong", "n_assigned",
                     "n_unassigned", "n_valid_tracks", "n_short_tracks",
                     "elapsed_s", "score"])
        for vname, cfg_results in results.items():
            for cfg, r in cfg_results.items():
                w.writerow([vname, cfg[0], cfg[1], cfg[2], cfg[3],
                             r.n_strong, r.n_assigned, r.n_unassigned,
                             r.n_valid_tracks, r.n_short_tracks,
                             round(r.elapsed_s, 3), round(r.score, 2)])
    print(f"\nPer-run CSV: {csv_path}")

    # Top 10 CSV
    top_path = OUT_DIR / f"autoresearch_multi_top10_{ts}.csv"
    with top_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "strong_thr", "weak_thr", "toe_max_dist_px",
                     "toe_min_overlap_frames", "avg_score", "min_score",
                     "max_score", "std_score"])
        for i, (cfg, avg) in enumerate(top10):
            scores = config_scores[cfg]
            w.writerow([i + 1, cfg[0], cfg[1], cfg[2], cfg[3],
                         round(avg, 2), round(min(scores), 2),
                         round(max(scores), 2), round(float(np.std(scores)), 2)])
    print(f"Top-10 CSV: {top_path}")

    # Best config JSON
    best_cfg, best_avg = top10[0]
    best_path = OUT_DIR / f"autoresearch_multi_best_{ts}.json"
    best_path.write_text(json.dumps({
        "strong_threshold": best_cfg[0],
        "weak_threshold": best_cfg[1],
        "toe_max_distance_px": best_cfg[2],
        "toe_min_overlap_frames": best_cfg[3],
        "avg_score": best_avg,
        "videos_scored": len(videos),
    }, indent=2))
    print(f"Best config: {best_path}")

    # Heatmap: avg score as function of (strong, weak), averaged over (dist, overlap, videos)
    grid = np.zeros((len(strong_thrs), len(weak_thrs)))
    for cfg, avg in avg_scores.items():
        i = strong_thrs.index(cfg[0])
        j = weak_thrs.index(cfg[1])
        grid[i, j] += avg
    grid /= (len(toe_dists) * len(toe_overlaps))
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(weak_thrs)))
    ax.set_xticklabels(weak_thrs)
    ax.set_yticks(range(len(strong_thrs)))
    ax.set_yticklabels(strong_thrs)
    ax.set_xlabel("weak_threshold")
    ax.set_ylabel("strong_threshold")
    bcfg = top10[0][0]
    ax.set_title(f"Avg score across {len(videos)} videos\n"
                  f"best = st={bcfg[0]}, wt={bcfg[1]}, "
                  f"d={bcfg[2]:.0f}, o={bcfg[3]} (avg={bcfg and top10[0][1]:.1f})")
    for i in range(len(strong_thrs)):
        for j in range(len(weak_thrs)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    color="white" if grid[i, j] < grid.max() * 0.7 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label="avg score")
    fig.tight_layout()
    heatmap_path = OUT_DIR / f"autoresearch_multi_heatmap_{ts}.png"
    fig.savefig(heatmap_path, dpi=130)
    print(f"Heatmap: {heatmap_path}")

    # Per-video score distribution for the best config
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [v.name[:30] for v in videos]
    scores_per_video = [results[v.name][top10[0][0]].score for v in videos]
    ax.barh(range(len(videos)), scores_per_video, color="steelblue")
    ax.set_yticks(range(len(videos)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("score (best config)")
    ax.set_title(f"Best config score per video (avg {np.mean(scores_per_video):.1f})")
    fig.tight_layout()
    pv_path = OUT_DIR / f"autoresearch_multi_per_video_{ts}.png"
    fig.savefig(pv_path, dpi=130)
    print(f"Per-video bar: {pv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())