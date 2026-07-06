"""Compare all 9 algorithms on REAL GT (test1.mp4 + data-test1.xlsx).

For each algorithm:
1. Run BiScaleDetector-style processing using the algorithm as score_fn.
2. Compute F1 (time-only match ±5 frames), precision, recall.

This is the final A/B that tells us which algorithm to use in production.
"""
from __future__ import annotations

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

from deepgait3.core.pawprint.benchmark import evaluate
from deepgait3.core.pawprint.gt_loaders import load_gt_from_xlsx
from deepgait3.core.pawprint.scoring import SCORING_ALGORITHMS

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
XLSX = "/home/luofangcheng/Documents/ZCODE/data-test1.xlsx"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")

# Per-algorithm strong threshold (chosen by analogy to ExG's optimal 80)
# All algorithms here are HIGH=green, so use the same threshold scale:
# - ExG family (ExG, ExGR, MExG) range ±500, threshold ~80 = 16% of range
# - CIVE range ±30 (scaled to ±3000), threshold 0
# - VDVI/GLI range ±1000, threshold 300 = 30%
# - NGRDI range ±1000, threshold 300
# - Lab a* (negative=green, negated), threshold 10
# - ColorDist threshold 150

DEFAULT_THRESHOLDS = {
    "exg": 80, "lab_astar": 10, "exgr": 50, "color_distance": 150,
    "cive": 0, "vdvi": 300, "ngrdi": 300, "mexg": 80, "gli": 300,
}

WEAK_THRESHOLDS = {  # weak threshold = strong / 4 (analogous to BiScale)
    "exg": 20, "lab_astar": 3, "exgr": 12, "color_distance": 40,
    "cive": 0, "vdvi": 75, "ngrdi": 75, "mexg": 20, "gli": 75,
}


def _simple_detect(video_path: str, score_fn, threshold: int,
                    min_area: int = 10) -> list[list[dict]]:
    """Per-frame blob detection without tracking (for F1 eval)."""
    cap = cv2.VideoCapture(video_path)
    blobs_per_frame = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = score_fn(frame)
        binary = (score > threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        n, _lbl, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        blobs = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            blobs.append({
                "cx": float(cents[i, 0]),
                "cy": float(cents[i, 1]),
                "area": area,
            })
        blobs_per_frame.append(blobs)
    cap.release()
    return blobs_per_frame


def _predictions(blobs_per_frame: list[list[dict]]) -> list[dict]:
    """Each blob becomes a prediction (1 frame each)."""
    preds = []
    for fidx, blobs in enumerate(blobs_per_frame, start=1):
        for b in blobs:
            preds.append({
                "frame_idx": fidx,
                "cx": b["cx"],
                "cy": b["cy"],
            })
    return preds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_gt_from_xlsx(XLSX, fps=60, coord_unit="px")
    print(f"GT: {len(gt)} prints")

    rows = []
    for algo_name in SCORING_ALGORITHMS:
        score_fn = SCORING_ALGORITHMS[algo_name]
        thr = DEFAULT_THRESHOLDS[algo_name]
        print(f"\n{algo_name} (threshold={thr}) ...")
        t0 = time.time()
        blobs_per_frame = _simple_detect(VIDEO, score_fn, thr)
        preds = _predictions(blobs_per_frame)
        elapsed = time.time() - t0
        ev = evaluate(gt, preds, match_distance_px=40.0,
                       match_frame_tolerance=5, spatial_check=False)
        n_total = len(preds)
        rows.append({
            "algo": algo_name,
            "threshold": thr,
            "n_predictions": n_total,
            "precision": ev.precision,
            "recall": ev.recall,
            "f1": ev.f1,
            "n_tp": ev.n_tp, "n_fp": ev.n_fp, "n_fn": ev.n_fn,
            "elapsed_s": round(elapsed, 2),
        })
        print(f"  P={ev.precision:.3f} R={ev.recall:.3f} F1={ev.f1:.3f} "
              f"TP={ev.n_tp} FP={ev.n_fp} FN={ev.n_fn} ({elapsed:.1f}s)")

    # Sort by F1
    rows.sort(key=lambda r: r["f1"], reverse=True)

    print("\n" + "=" * 80)
    print("RANKING BY F1 (real GT, default thresholds)")
    print("=" * 80)
    print(f"{'rank':>4} {'algo':<18}{'thr':>6}{'n_preds':>10}{'P':>7}"
          f"{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}")
    for i, r in enumerate(rows):
        print(f"{i+1:>4} {r['algo']:<18}{r['threshold']:>6}"
              f"{r['n_predictions']:>10}{r['precision']:>7.3f}"
              f"{r['recall']:>7.3f}{r['f1']:>7.3f}"
              f"{r['n_tp']:>5}{r['n_fp']:>5}{r['n_fn']:>5}")

    # CSV
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"v07_algo_a_b_gt_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "algo", "threshold", "n_predictions",
                     "precision", "recall", "f1", "n_tp", "n_fp", "n_fn",
                     "elapsed_s"])
        for i, r in enumerate(rows):
            w.writerow([i + 1, r["algo"], r["threshold"], r["n_predictions"],
                         round(r["precision"], 4), round(r["recall"], 4),
                         round(r["f1"], 4), r["n_tp"], r["n_fp"], r["n_fn"],
                         r["elapsed_s"]])
    print(f"\nCSV: {csv_path}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [r["algo"] for r in rows]
    f1s = [r["f1"] for r in rows]
    colors = ["#2ca02c" if r["algo"] == "exg" else "#1f77b4" for r in rows]
    ax.barh(range(len(names)), f1s, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("F1 (real GT, default thresholds)")
    ax.set_title(f"9 algorithms on test1.mp4 + data-test1.xlsx (n_gt={len(gt)})")
    for i, v in enumerate(f1s):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    bar_path = OUT_DIR / f"v07_algo_a_b_gt_bar_{ts}.png"
    fig.savefig(bar_path, dpi=130)
    print(f"Bar chart: {bar_path}")

    # P/R scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    for r in rows:
        c = "#2ca02c" if r["algo"] == "exg" else "#1f77b4"
        ax.scatter(r["recall"], r["precision"], s=200, c=c, alpha=0.7,
                    edgecolors="black", linewidth=0.5)
        ax.annotate(r["algo"], (r["recall"], r["precision"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Precision vs Recall (9 algorithms)")
    ax.grid(alpha=0.3)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    fig.tight_layout()
    pr_path = OUT_DIR / f"v07_algo_a_b_gt_pr_{ts}.png"
    fig.savefig(pr_path, dpi=130)
    print(f"P/R chart: {pr_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())