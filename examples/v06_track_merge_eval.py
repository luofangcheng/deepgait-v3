"""v0.6 evaluation: BiScaleDetector + TrackMerger vs BiScaleDetector alone,
on real GT (commercial xlsx).

Compares precision / recall / F1 / n_candidates with and without the
post-processing merge step.

Outputs:
- /home/luofangcheng/Documents/ZCODE/tmp/v06_with_vs_without_merge.csv
- /home/luofangcheng/Documents/ZCODE/tmp/v06_comparison.png
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.benchmark import evaluate
from deepgait3.core.pawprint.dataset import iter_frames
from deepgait3.core.pawprint.gt_loaders import load_gt_from_xlsx
from deepgait3.core.pawprint.track_merger import merge_tracks

OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
XLSX = "/home/luofangcheng/Documents/ZCODE/data-test1.xlsx"
# Use the best params from the GT-driven autoresearch
BEST_STRONG = 80
BEST_WEAK = 3


def _predictions_from_candidates(candidates) -> list[dict]:
    preds = []
    for c in candidates:
        if not c.strong_blobs:
            continue
        cx = float(np.median([b.cx_px for b in c.strong_blobs]))
        cy = float(np.median([b.cy_px for b in c.strong_blobs]))
        peak = max(c.strong_blobs, key=lambda b: b.area_px)
        preds.append({
            "frame_idx": peak.frame_idx if peak.frame_idx > 0 else c.first_frame,
            "cx": cx, "cy": cy,
        })
    return preds


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_gt_from_xlsx(XLSX, fps=60, coord_unit="px")
    print(f"GT: {len(gt)} prints")

    # Run detector
    print(f"\nRunning BiScaleDetector(st={BEST_STRONG}, wt={BEST_WEAK}) ...")
    cap = cv2.VideoCapture(VIDEO)
    det = BiScaleDetector(
        strong_threshold=BEST_STRONG, weak_threshold=BEST_WEAK,
        toe_max_distance_px=24.0, toe_min_overlap_frames=1,
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

    n_raw = len(det.closed_candidates)
    print(f"  raw candidates: {n_raw}")

    # Sweep merge parameters and report F1 for each
    print("\n--- Without merge (baseline) ---")
    raw_preds = _predictions_from_candidates(det.closed_candidates)
    raw_eval = evaluate(gt, raw_preds, match_distance_px=40.0,
                          match_frame_tolerance=5, spatial_check=False)
    print(f"  n_preds={len(raw_preds)} P={raw_eval.precision:.3f} "
          f"R={raw_eval.recall:.3f} F1={raw_eval.f1:.3f}")

    print("\n--- Sweeping merge parameters ---")
    print(f"{'gap':>5}{'dist':>6}{'ratio_lo':>10}{'ratio_hi':>10}"
          f"{'#tracks':>9}{'P':>7}{'R':>7}{'F1':>7}")

    sweep_results = []
    for gap in [3, 5, 8, 12, 20]:
        for dist in [40, 60, 80, 120]:
            for r_lo, r_hi in [(0.3, 3.0), (0.5, 2.0), (0.7, 1.5)]:
                merged = merge_tracks(
                    det.closed_candidates,
                    max_gap_frames=gap,
                    merge_distance_px=dist,
                    area_ratio_min=r_lo,
                    area_ratio_max=r_hi,
                )
                merged_preds = _predictions_from_candidates(merged)
                ev = evaluate(gt, merged_preds, match_distance_px=40.0,
                               match_frame_tolerance=5, spatial_check=False)
                sweep_results.append({
                    "gap": gap, "dist": dist,
                    "ratio_lo": r_lo, "ratio_hi": r_hi,
                    "n_tracks": len(merged),
                    "precision": ev.precision, "recall": ev.recall,
                    "f1": ev.f1, "n_tp": ev.n_tp, "n_fp": ev.n_fp, "n_fn": ev.n_fn,
                })
                print(f"{gap:>5}{dist:>6}{r_lo:>10.2f}{r_hi:>10.2f}"
                      f"{len(merged):>9}{ev.precision:>7.3f}{ev.recall:>7.3f}"
                      f"{ev.f1:>7.3f}")

    sweep_results.sort(key=lambda r: r["f1"], reverse=True)

    print("\n" + "=" * 80)
    print("TOP 5 MERGE CONFIGURATIONS")
    print("=" * 80)
    print(f"{'rank':>4} {'gap':>5}{'dist':>6}{'ratio':>12}{'tracks':>8}"
          f"{'P':>7}{'R':>7}{'F1':>7}")
    for i, r in enumerate(sweep_results[:5]):
        ratio = f"{r['ratio_lo']:.1f}-{r['ratio_hi']:.1f}"
        print(f"{i+1:>4} {r['gap']:>5}{r['dist']:>6}{ratio:>12}"
              f"{r['n_tracks']:>8}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['f1']:>7.3f}")

    # Save CSV
    csv_path = OUT_DIR / "v06_with_vs_without_merge.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gap", "dist", "ratio_lo", "ratio_hi", "n_tracks",
                     "precision", "recall", "f1", "n_tp", "n_fp", "n_fn"])
        # Baseline first
        w.writerow(["NO_MERGE", "-", "-", "-", n_raw,
                     round(raw_eval.precision, 4), round(raw_eval.recall, 4),
                     round(raw_eval.f1, 4), raw_eval.n_tp, raw_eval.n_fp,
                     raw_eval.n_fn])
        for r in sweep_results:
            w.writerow([r["gap"], r["dist"], r["ratio_lo"], r["ratio_hi"],
                         r["n_tracks"], round(r["precision"], 4),
                         round(r["recall"], 4), round(r["f1"], 4),
                         r["n_tp"], r["n_fp"], r["n_fn"]])
    print(f"\nCSV: {csv_path}")

    # Visualization: P/R/F1 vs n_tracks
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    n_tracks_arr = np.array([r["n_tracks"] for r in sweep_results])
    p_arr = np.array([r["precision"] for r in sweep_results])
    r_arr = np.array([r["recall"] for r in sweep_results])
    f1_arr = np.array([r["f1"] for r in sweep_results])
    axes[0].scatter(n_tracks_arr, p_arr, alpha=0.6, color="C0")
    axes[0].axhline(raw_eval.precision, color="C0", linestyle="--",
                    label=f"baseline P={raw_eval.precision:.3f}")
    axes[0].set_xlabel("# merged tracks")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision vs n_tracks")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].scatter(n_tracks_arr, r_arr, alpha=0.6, color="C1")
    axes[1].axhline(raw_eval.recall, color="C1", linestyle="--",
                    label=f"baseline R={raw_eval.recall:.3f}")
    axes[1].set_xlabel("# merged tracks")
    axes[1].set_ylabel("Recall")
    axes[1].set_title("Recall vs n_tracks")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].scatter(n_tracks_arr, f1_arr, alpha=0.6, color="C2")
    axes[2].axhline(raw_eval.f1, color="C2", linestyle="--",
                    label=f"baseline F1={raw_eval.f1:.3f}")
    axes[2].set_xlabel("# merged tracks")
    axes[2].set_ylabel("F1")
    axes[2].set_title("F1 vs n_tracks")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"v0.5 baseline (n_tracks={n_raw}, F1={raw_eval.f1:.3f}) vs v0.6 merge sweep")
    fig.tight_layout()
    cmp_path = OUT_DIR / "v06_comparison.png"
    fig.savefig(cmp_path, dpi=130)
    print(f"Comparison: {cmp_path}")

    # Summary
    best = sweep_results[0]
    print(f"\n=== SUMMARY ===")
    print(f"v0.5 (no merge):   n_tracks={n_raw}  P={raw_eval.precision:.3f} "
          f"R={raw_eval.recall:.3f} F1={raw_eval.f1:.3f}")
    print(f"v0.6 (best merge): n_tracks={best['n_tracks']}  P={best['precision']:.3f} "
          f"R={best['recall']:.3f} F1={best['f1']:.3f}")
    print(f"  best params: gap={best['gap']}, dist={best['dist']}, "
          f"ratio=[{best['ratio_lo']:.1f}, {best['ratio_hi']:.1f}]")
    print(f"  ΔF1 = {best['f1'] - raw_eval.f1:+.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())