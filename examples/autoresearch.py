"""Autoresearch: grid search over BiScaleDetector hyperparameters.

Sweeps 4 hyperparameters on test1.mp4 and reports the top-N
configurations by a heuristic score that rewards:

- Valid tracks (footprints that live >= 5 frames)
- Toes recovered (weak blobs successfully assigned to a footprint)
- Penalizes short tracks (likely noise)
- Penalizes unassigned weak blobs (noise that escaped filtering)

This is a heuristic — without ground truth we can only rank by
internal consistency, not by absolute correctness.

Outputs:
- /home/luofangcheng/Documents/ZCODE/tmp/autoresearch_results.csv
- /home/luofangcheng/Documents/ZCODE/tmp/autoresearch_top10.png
- /home/luofangcheng/Documents/ZCODE/tmp/autoresearch_best_config.json
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector

DEFAULT_VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")


@dataclass
class RunResult:
    strong_thr: int
    weak_thr: int
    toe_max_dist_px: float
    toe_min_overlap_frames: int
    n_strong: int
    n_assigned: int
    n_unassigned: int
    n_valid_tracks: int
    n_short_tracks: int
    elapsed_s: float
    score: float

    @staticmethod
    def compute_score(*, n_valid: int, n_assigned: int,
                       n_short: int, n_unassigned: int) -> float:
        """Heuristic without ground truth.

        - 10 pts per valid track (>= 5 frames)
        - +2 pts per assigned weak (toe recovered)
        - -3 pts per short track (likely noise)
        - -0.5 pts per unassigned weak (noise that escaped)
        """
        return 10.0 * n_valid + 2.0 * n_assigned - 3.0 * n_short - 0.5 * n_unassigned


def run_one(video_path: str, strong_thr: int, weak_thr: int,
             toe_max_dist_px: float, toe_min_overlap_frames: int) -> RunResult:
    """Run detector on full video with given config."""
    assert strong_thr > weak_thr, "strong_thr must exceed weak_thr"
    cap = cv2.VideoCapture(video_path)
    detector = BiScaleDetector(
        strong_threshold=strong_thr,
        weak_threshold=weak_thr,
        toe_max_distance_px=toe_max_dist_px,
        toe_min_overlap_frames=toe_min_overlap_frames,
    )
    t0 = time.time()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        detector.process_frame(frame, idx)
    detector.finalize()
    cap.release()
    elapsed = time.time() - t0
    s = detector.stats
    score = RunResult.compute_score(
        n_valid=s["n_valid_tracks"],
        n_assigned=s["n_weak_assigned_total"],
        n_short=s["n_short_tracks"],
        n_unassigned=s["n_weak_unassigned_total"],
    )
    return RunResult(
        strong_thr=strong_thr, weak_thr=weak_thr,
        toe_max_dist_px=toe_max_dist_px,
        toe_min_overlap_frames=toe_min_overlap_frames,
        n_strong=s["n_strong_blobs_total"],
        n_assigned=s["n_weak_assigned_total"],
        n_unassigned=s["n_weak_unassigned_total"],
        n_valid_tracks=s["n_valid_tracks"],
        n_short_tracks=s["n_short_tracks"],
        elapsed_s=elapsed,
        score=score,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = DEFAULT_VIDEO
    if not Path(video).exists():
        print(f"FAIL: {video} not found")
        return 1

    # Search grid
    strong_thrs = [40, 60, 80, 100, 120]
    weak_thrs = [3, 5, 8, 12, 18]
    toe_dists = [6, 9, 12, 18, 24]
    toe_overlaps = [1, 2, 3, 5]

    # Constraint: strong > weak always
    configs = [
        (st, wt, td, to)
        for st in strong_thrs
        for wt in weak_thrs
        for td in toe_dists
        for to in toe_overlaps
        if st > wt
    ]
    print(f"Grid size: {len(configs)} configs")

    results: list[RunResult] = []
    t0 = time.time()
    for i, (st, wt, td, to) in enumerate(configs):
        r = run_one(video, st, wt, td, to)
        results.append(r)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(configs)}] st={st} wt={wt} d={td} o={to} "
                  f"valid={r.n_valid_tracks} toes={r.n_assigned} "
                  f"score={r.score:.1f}  ({r.elapsed_s:.1f}s/run)")
    total = time.time() - t0
    print(f"\nDone in {total:.1f}s ({total/len(configs):.2f}s per config)")

    # Sort by score desc
    results.sort(key=lambda r: r.score, reverse=True)

    # Top 10
    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)
    print(f"{'rank':>4} {'strong':>7} {'weak':>5} {'dist_px':>8} {'overlap':>7} "
          f"{'valid':>6} {'toes':>5} {'short':>6} {'unassigned':>11} {'score':>7}")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>4} {r.strong_thr:>7} {r.weak_thr:>5} {r.toe_max_dist_px:>8.0f} "
              f"{r.toe_min_overlap_frames:>7} {r.n_valid_tracks:>6} "
              f"{r.n_assigned:>5} {r.n_short_tracks:>6} {r.n_unassigned:>11} "
              f"{r.score:>7.1f}")

    # Save CSV
    csv_path = OUT_DIR / "autoresearch_results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "strong_thr", "weak_thr", "toe_max_dist_px",
                     "toe_min_overlap_frames", "n_strong", "n_assigned",
                     "n_unassigned", "n_valid_tracks", "n_short_tracks",
                     "elapsed_s", "score"])
        for i, r in enumerate(results):
            w.writerow([i + 1, r.strong_thr, r.weak_thr, r.toe_max_dist_px,
                         r.toe_min_overlap_frames, r.n_strong, r.n_assigned,
                         r.n_unassigned, r.n_valid_tracks, r.n_short_tracks,
                         round(r.elapsed_s, 3), round(r.score, 2)])
    print(f"\nResults CSV: {csv_path}")

    # Best config JSON
    best = results[0]
    best_path = OUT_DIR / "autoresearch_best_config.json"
    best_path.write_text(json.dumps({
        "strong_threshold": best.strong_thr,
        "weak_threshold": best.weak_thr,
        "toe_max_distance_px": best.toe_max_dist_px,
        "toe_min_overlap_frames": best.toe_min_overlap_frames,
        "score": best.score,
        "n_valid_tracks": best.n_valid_tracks,
        "n_toes_recovered": best.n_assigned,
    }, indent=2))
    print(f"Best config: {best_path}")

    # Heatmap: score as function of strong_thr × weak_thr (averaged over dist, overlap)
    print("\nGenerating heatmap ...")
    avg_score = np.zeros((len(strong_thrs), len(weak_thrs)))
    cnt = np.zeros((len(strong_thrs), len(weak_thrs)))
    for r in results:
        i = strong_thrs.index(r.strong_thr)
        j = weak_thrs.index(r.weak_thr)
        avg_score[i, j] += r.score
        cnt[i, j] += 1
    avg_score = np.divide(avg_score, cnt, out=np.zeros_like(avg_score),
                            where=cnt > 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(avg_score, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(weak_thrs)))
    ax.set_xticklabels(weak_thrs)
    ax.set_yticks(range(len(strong_thrs)))
    ax.set_yticklabels(strong_thrs)
    ax.set_xlabel("weak_threshold")
    ax.set_ylabel("strong_threshold")
    ax.set_title(f"Avg score (avg over toe_max_dist × overlap)\nbest = "
                  f"st={best.strong_thr}, wt={best.weak_thr}, score={best.score:.1f}")
    for i in range(len(strong_thrs)):
        for j in range(len(weak_thrs)):
            ax.text(j, i, f"{avg_score[i, j]:.0f}", ha="center", va="center",
                    color="white" if avg_score[i, j] < avg_score.max() * 0.7 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    heatmap_path = OUT_DIR / "autoresearch_heatmap.png"
    fig.savefig(heatmap_path, dpi=130)
    print(f"Heatmap: {heatmap_path}")

    # Top-10 bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    top10 = results[:10]
    labels = [f"st={r.strong_thr}/wt={r.weak_thr}\nd={r.toe_max_dist_px:.0f}/o={r.toe_min_overlap_frames}"
              for r in top10]
    scores = [r.score for r in top10]
    ax.barh(range(10), scores, color="steelblue")
    ax.set_yticks(range(10))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("score")
    ax.set_title("Top 10 BiScaleDetector configurations")
    fig.tight_layout()
    bar_path = OUT_DIR / "autoresearch_top10.png"
    fig.savefig(bar_path, dpi=130)
    print(f"Top-10 bars: {bar_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())