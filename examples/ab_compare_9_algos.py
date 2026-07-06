"""A/B test of 9 vegetation-index algorithms on test1.mp4.

Compares all 9 algorithms in SCORING_ALGORITHMS using default thresholds
(taken from each algorithm's original publication), without any tuning.

For each algorithm:
1. Run per-frame score map.
2. Apply default threshold + morphology.
3. Per-frame blob count.
4. Aggregate stats: total blobs, noise frames (>6), empty frames.

Output (to /home/luofangcheng/Documents/ZCODE/tmp/):
- ab_compare_9_algos.png     bar chart of total blobs per algorithm
- ab_compare_9_algos.csv     raw stats
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.scoring import SCORING_ALGORITHMS
from deepgait3.core.pawprint.scoring_detection import detect_blobs_from_score

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")

# Default thresholds taken from each algorithm's original publication,
# so we compare "out of the box" without per-algorithm tuning.
DEFAULT_THRESHOLDS = {
    "exg": 80,           # test_paw_detection.py default
    "lab_astar": 10,     # test_paw_detection.py default
    "exgr": 50,          # test_paw_detection.py default
    "color_distance": 150,
    "cive": 0,           # CIVE is already low=green; we negated, so higher=green; threshold ~0 separates
    "vdvi": 300,         # range -1000..+1000, threshold ~300 = VDVI > 0.3
    "ngrdi": 300,        # range -1000..+1000, threshold ~300 = NGRDI > 0.3
    "mexg": 50,          # MExG, range ±766, threshold ~50
    "gli": 300,          # GLI, range -1000..+1000, threshold ~0.3
}


def _per_frame_blobs(video_path: str, algo_name: str, threshold: int) -> list[list[dict]]:
    score_fn = SCORING_ALGORITHMS[algo_name]
    cap = cv2.VideoCapture(video_path)
    blobs_per_frame: list[list[dict]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = score_fn(frame)
        _, blobs = detect_blobs_from_score(score, threshold=threshold, min_area=5)
        blobs_per_frame.append(blobs)
    cap.release()
    return blobs_per_frame


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name in SCORING_ALGORITHMS:
        thr = DEFAULT_THRESHOLDS[name]
        print(f"Running {name} (threshold={thr}) ...")
        blobs = _per_frame_blobs(VIDEO, name, thr)
        counts = np.array([len(b) for b in blobs], dtype=np.int32)
        results[name] = {
            "threshold": thr,
            "total_blobs": int(counts.sum()),
            "mean_per_frame": float(counts.mean()),
            "max_per_frame": int(counts.max()),
            "noise_frames": int((counts > 6).sum()),
            "empty_frames": int((counts == 0).sum()),
            "blobs_per_frame": counts.tolist(),
        }

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'algo':<18}{'thr':>6}{'total':>10}{'mean':>8}{'max':>6}"
          f"{'noise':>8}{'empty':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(f"{name:<18}{r['threshold']:>6}{r['total_blobs']:>10}"
              f"{r['mean_per_frame']:>8.2f}{r['max_per_frame']:>6}"
              f"{r['noise_frames']:>8}{r['empty_frames']:>8}")

    # CSV
    csv_path = OUT_DIR / "ab_compare_9_algos.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "threshold", "total_blobs", "mean_per_frame",
                     "max_per_frame", "noise_frames", "empty_frames"])
        for name, r in results.items():
            w.writerow([name, r["threshold"], r["total_blobs"],
                         round(r["mean_per_frame"], 2), r["max_per_frame"],
                         r["noise_frames"], r["empty_frames"]])
    print(f"\nCSV: {csv_path}")

    # Plot: bar chart + per-frame time series
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    names = list(results.keys())
    totals = [results[n]["total_blobs"] for n in names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    axes[0].barh(names, totals, color=colors)
    axes[0].set_xlabel("Total blobs detected")
    axes[0].set_title("9 vegetation-index algorithms on test1.mp4 (default thresholds)")
    for i, v in enumerate(totals):
        axes[0].text(v + 5, i, str(v), va="center", fontsize=9)
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3, axis="x")

    for name, r in results.items():
        axes[1].plot(r["blobs_per_frame"], label=name, linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("frame index")
    axes[1].set_ylabel("# blobs")
    axes[1].set_title("Blob count over time")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    plot_path = OUT_DIR / "ab_compare_9_algos.png"
    fig.savefig(plot_path, dpi=120)
    print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())