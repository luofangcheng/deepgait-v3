"""Experiment Step 1 + 2: Category A (9 score × fixed threshold) vs Category B
(3 score × 4 adaptive threshold methods), evaluated on real GT.

Category A: 9 score functions × 1 threshold strategy (algorithm-default)
Category B: 3 score functions × 4 adaptive threshold strategies = 12 configs
Total: 21 configurations evaluated on test1.mp4 + data-test1.xlsx

This is the empirical foundation for deciding whether to train a CNN
(Category C — UnderPressure-style temporal model).
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
from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.dataset import iter_frames
from deepgait3.core.pawprint.gt_loaders import load_gt_from_xlsx
from deepgait3.core.pawprint.scoring import SCORING_ALGORITHMS

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
XLSX = "/home/luofangcheng/Documents/ZCODE/data-test1.xlsx"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")


# ---------------------------------------------------------------------------
# Threshold strategies
# ---------------------------------------------------------------------------

def _threshold_fixed(score: np.ndarray, value: int) -> np.ndarray:
    return (score > value).astype(np.uint8)


def _threshold_otsu(score: np.ndarray, **_kwargs) -> np.ndarray:
    """Otsu's method: histogram-based automatic threshold."""
    s8 = score.astype(np.int16)
    # cv2.threshold requires 8-bit. Clip to 0..255.
    s_min = int(s8.min()); s_max = int(s8.max())
    if s_max <= s_min:
        return np.zeros_like(score, dtype=np.uint8)
    norm = ((s8 - s_min) * 255 / (s_max - s_min)).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _threshold_triangle(score: np.ndarray, **_kwargs) -> np.ndarray:
    """Triangle method (geometric histogram peak detection)."""
    s8 = score.astype(np.int16)
    s_min = int(s8.min()); s_max = int(s8.max())
    if s_max <= s_min:
        return np.zeros_like(score, dtype=np.uint8)
    norm = ((s8 - s_min) * 255 / (s_max - s_min)).astype(np.uint8)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binary


def _threshold_percentile(score: np.ndarray, percentile: float = 95.0, **_kwargs) -> np.ndarray:
    """Top-percentile threshold: keep the top ``percentile``% brightest pixels."""
    p = np.percentile(score, percentile)
    return (score > p).astype(np.uint8)


THRESHOLD_METHODS = {
    "fixed": _threshold_fixed,
    "otsu": _threshold_otsu,
    "triangle": _threshold_triangle,
    "percentile95": lambda s, **kw: _threshold_percentile(s, percentile=95.0),
    "percentile97": lambda s, **kw: _threshold_percentile(s, percentile=97.0),
}

# Per-algorithm default fixed thresholds (Category A only)
DEFAULT_FIXED_THRESHOLDS = {
    "exg": 80, "lab_astar": 10, "exgr": 50, "color_distance": 150,
    "cive": 0, "vdvi": 300, "ngrdi": 300, "mexg": 80, "gli": 300,
}


def _morph_cleanup(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _blobs_from_binary(binary: np.ndarray, min_area: int = 5) -> list[dict]:
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
    return blobs


def _evaluate(score_fn_name: str, threshold_name: str, threshold_arg,
                gt: list, video_path: str, weak_ratio: float = 0.25) -> dict:
    """Run full video, produce predictions, evaluate vs GT (time-only match)."""
    score_fn = SCORING_ALGORITHMS[score_fn_name]
    thr_method = THRESHOLD_METHODS[threshold_name]

    cap = cv2.VideoCapture(video_path)
    blobs_per_frame: list[list[dict]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        score = score_fn(frame)
        binary = thr_method(score, value=threshold_arg) if threshold_name == "fixed" else thr_method(score)
        binary = _morph_cleanup(binary)
        blobs_per_frame.append(_blobs_from_binary(binary))
    cap.release()

    preds = []
    for fidx, blobs in enumerate(blobs_per_frame, start=1):
        for b in blobs:
            preds.append({"frame_idx": fidx, "cx": b["cx"], "cy": b["cy"]})

    ev = evaluate(gt, preds, match_distance_px=40.0,
                   match_frame_tolerance=5, spatial_check=False)
    n_total = len(preds)
    return {
        "category": "A" if threshold_name == "fixed" else "B",
        "score_fn": score_fn_name,
        "threshold": threshold_name,
        "threshold_value": str(threshold_arg),
        "n_preds": n_total,
        "precision": ev.precision,
        "recall": ev.recall,
        "f1": ev.f1,
        "n_tp": ev.n_tp,
        "n_fp": ev.n_fp,
        "n_fn": ev.n_fn,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = load_gt_from_xlsx(XLSX, fps=60, coord_unit="px")
    print(f"GT: {len(gt)} prints")

    # Category A: 9 score × fixed threshold
    print("\n=== Category A: 9 score × fixed threshold ===")
    cat_a = []
    for algo in SCORING_ALGORITHMS:
        r = _evaluate(algo, "fixed", DEFAULT_FIXED_THRESHOLDS[algo], gt, VIDEO)
        cat_a.append(r)
        print(f"  {algo:>18} (thr={r['threshold_value']:>4}): P={r['precision']:.3f} "
              f"R={r['recall']:.3f} F1={r['f1']:.3f}")

    # Category B: 3 score × 4 adaptive thresholds
    print("\n=== Category B: 3 score × 4 adaptive thresholds ===")
    cat_b = []
    for algo in ["exg", "mexg", "lab_astar"]:  # top performers from Cat A
        for thr_name in ["otsu", "triangle", "percentile95", "percentile97"]:
            r = _evaluate(algo, thr_name, 0, gt, VIDEO)
            cat_b.append(r)
            print(f"  {algo:>18} ({thr_name:>13}): P={r['precision']:.3f} "
                  f"R={r['recall']:.3f} F1={r['f1']:.3f}")

    # Combined
    all_results = cat_a + cat_b
    all_results.sort(key=lambda r: r["f1"], reverse=True)

    print("\n" + "=" * 90)
    print("TOP 10 (Category A + B combined, time-only F1)")
    print("=" * 90)
    print(f"{'rank':>4}{'cat':>5}{'algo':>18}{'threshold':>16}{'n_preds':>10}"
          f"{'P':>7}{'R':>7}{'F1':>7}")
    for i, r in enumerate(all_results[:10]):
        print(f"{i+1:>4}{r['category']:>5}{r['score_fn']:>18}"
              f"{r['threshold']:>16}{r['n_preds']:>10}"
              f"{r['precision']:>7.3f}{r['recall']:>7.3f}{r['f1']:>7.3f}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"experiment_cat_a_b_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "category", "score_fn", "threshold",
                     "threshold_value", "n_preds", "precision", "recall",
                     "f1", "n_tp", "n_fp", "n_fn"])
        for i, r in enumerate(all_results):
            w.writerow([i + 1, r["category"], r["score_fn"], r["threshold"],
                         r["threshold_value"], r["n_preds"],
                         round(r["precision"], 4), round(r["recall"], 4),
                         round(r["f1"], 4), r["n_tp"], r["n_fp"], r["n_fn"]])
    print(f"\nCSV: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [f"{r['category']}/{r['score_fn']}/{r['threshold']}" for r in all_results[:15]]
    f1s = [r["f1"] for r in all_results[:15]]
    colors = ["#1f77b4" if r["category"] == "A" else "#ff7f0e" for r in all_results[:15]]
    ax.barh(range(len(labels)), f1s, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("F1 (real GT)")
    ax.set_xlim(0, 1)
    ax.set_title(f"Top 15 configs on test1.mp4 (n_gt={len(gt)}) — Category A vs B")
    for i, v in enumerate(f1s):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    bar_path = OUT_DIR / f"experiment_cat_a_b_top15_{ts}.png"
    fig.savefig(bar_path, dpi=130)
    print(f"Bar chart: {bar_path}")

    # Best per category
    best_a = max(cat_a, key=lambda r: r["f1"])
    best_b = max(cat_b, key=lambda r: r["f1"])
    print(f"\n=== Category Summary ===")
    print(f"Best Category A: {best_a['score_fn']} (thr={best_a['threshold_value']}) F1={best_a['f1']:.3f}")
    print(f"Best Category B: {best_b['score_fn']} ({best_b['threshold']}) F1={best_b['f1']:.3f}")
    print(f"\n→ Implication for Category C (CNN): only worth training if F1 ≥ "
          f"{max(best_a['f1'], best_b['f1']) + 0.10:.3f} (current_best + 10pp margin)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())