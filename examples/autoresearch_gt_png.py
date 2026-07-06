"""GT-PNG-driven autoresearch: tune BiScaleDetector against the embedded
footprint PNG masks.

For each (config, GT print) pair:
1. At GT frame_idx, run per-frame scoring with config params.
2. Find the footprint-shaped blob closest in area to the GT PNG.
3. Compute MAE and IoU between the algorithm's mask and the GT mask.

The aggregate score is ``median IoU - 0.5 * median MAE`` (higher is better).

Outputs to /home/luofangcheng/Documents/ZCODE/tmp/.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.dataset import iter_frames
from deepgait3.core.pawprint.gt_loaders import load_gt_from_xlsx
from deepgait3.core.pawprint.scoring import compute_exg

OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")


def _png_mask(png_path: Path, threshold: int = 30) -> np.ndarray:
    arr = np.asarray(Image.open(png_path))
    if arr.ndim == 3:
        G = arr[:, :, 1].astype(np.int16) - arr[:, :, 0].astype(np.int16) - arr[:, :, 2].astype(np.int16)
        return (G > threshold).astype(np.uint8)
    return (arr > threshold).astype(np.uint8)


def _crop_to_paw_region(mask: np.ndarray, max_pad_px: int = 0) -> np.ndarray:
    n, _lbl, stats, _cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    x = stats[biggest, cv2.CC_STAT_LEFT]
    y = stats[biggest, cv2.CC_STAT_TOP]
    w = stats[biggest, cv2.CC_STAT_WIDTH]
    h = stats[biggest, cv2.CC_STAT_HEIGHT]
    return mask[y:y + h, x:x + w]


def _resize_to_match(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    out = cv2.resize(src.astype(np.float32), (target_shape[1], target_shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return (out > 0.5).astype(np.uint8)


def _mae_iou(alg_crop: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    if alg_crop.shape != gt_mask.shape:
        alg_crop = _resize_to_match(alg_crop, gt_mask.shape)
    mae = float(np.abs(alg_crop.astype(np.int16) - gt_mask.astype(np.int16)).mean())
    inter = int(np.logical_and(alg_crop, gt_mask).sum())
    union = int(np.logical_or(alg_crop, gt_mask).sum())
    iou = inter / union if union > 0 else 0.0
    return mae, iou


def _evaluate_config(video_frames: list[np.ndarray], gt: list,
                      gt_pngs: list[Path], strong: int, weak: int) -> dict:
    """Compute per-print MAE/IoU for one (strong, weak) config."""
    maes, ious = [], []
    for g, png in zip(gt, gt_pngs):
        if g.frame_idx < 1 or g.frame_idx > len(video_frames):
            continue
        gt_mask = _png_mask(png)
        if gt_mask.sum() == 0:
            continue
        gt_paw = _crop_to_paw_region(gt_mask, max_pad_px=0)
        gt_area = max(int(gt_paw.sum()), 1)
        frame_bgr = video_frames[g.frame_idx - 1]
        score = compute_exg(frame_bgr)
        # strong blobs
        binary_strong = (score > strong).astype(np.uint8)
        binary_weak = (score > weak).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for b in (binary_strong, binary_weak):
            cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel, dst=b)
            cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, dst=b)
        n, _lbl, stats, cents = cv2.connectedComponentsWithStats(binary_strong, connectivity=8)
        # Find the strongest blob whose area is within 0.5×–3× of GT
        best_i, best_diff = -1, float("inf")
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 5:
                continue
            ratio = area / gt_area
            if 0.3 < ratio < 4.0:
                diff = abs(np.log(ratio))
                if diff < best_diff:
                    best_diff = diff
                    best_i = i
        if best_i < 0:
            maes.append(0.5); ious.append(0.0)
            continue
        bx = stats[best_i, cv2.CC_STAT_LEFT]
        by = stats[best_i, cv2.CC_STAT_TOP]
        bw = stats[best_i, cv2.CC_STAT_WIDTH]
        bh = stats[best_i, cv2.CC_STAT_HEIGHT]
        alg_crop = binary_strong[by:by + bh, bx:bx + bw]
        alg_paw = _crop_to_paw_region(alg_crop, max_pad_px=0)
        mae, iou = _mae_iou(alg_paw, gt_paw)
        maes.append(mae)
        ious.append(iou)
    if not maes:
        return {"median_mae": 0.5, "median_iou": 0.0, "score": -1.0,
                "mean_iou": 0.0, "mean_mae": 0.5, "n_prints": 0}
    median_mae = float(np.median(maes))
    median_iou = float(np.median(ious))
    return {
        "median_mae": median_mae,
        "median_iou": median_iou,
        "mean_mae": float(np.mean(maes)),
        "mean_iou": float(np.mean(ious)),
        "score": median_iou - 0.5 * median_mae,
        "n_prints": len(maes),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to fTIR mp4 video")
    ap.add_argument("gt_xlsx", help="path to commercial gait Excel export")
    ap.add_argument("--gt-images-dir", default="/home/luofangcheng/Documents/ZCODE/tmp/gt_images")
    ap.add_argument("--fps", type=int, default=60)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_gt_from_xlsx(args.gt_xlsx, fps=args.fps, coord_unit="px")
    gt_pngs = sorted(Path(args.gt_images_dir).glob("*.png"))[: len(gt)]
    print(f"GT: {len(gt)} prints, {len(gt_pngs)} PNG masks")

    print("Loading video frames ...")
    video_frames = [f for _, f in iter_frames(args.video)]
    print(f"  {len(video_frames)} frames loaded")

    strong_thrs = [40, 60, 80, 100, 120]
    weak_thrs = [3, 5, 8, 12, 18]
    print(f"\nGrid: {len(strong_thrs) * len(weak_thrs)} configs (no dist/overlap "
          f"since we evaluate per-frame, not per-track)")

    results = []
    t0 = time.time()
    for i, st in enumerate(strong_thrs):
        for j, wt in enumerate(weak_thrs):
            if st <= wt:
                continue
            r = _evaluate_config(video_frames, gt, gt_pngs, st, wt)
            results.append({"strong": st, "weak": wt, **r})
            elapsed = time.time() - t0
            done = i * len(weak_thrs) + j + 1
            eta = elapsed * len(strong_thrs) * len(weak_thrs) / done - elapsed
            print(f"  [{done}/{len(strong_thrs) * len(weak_thrs)}] "
                  f"st={st} wt={wt} med_iou={r['median_iou']:.3f} "
                  f"med_mae={r['median_mae']:.3f} score={r['score']:.3f} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    results.sort(key=lambda x: x["score"], reverse=True)
    print("\n" + "=" * 80)
    print(f"{'rank':>4} {'strong':>7} {'weak':>5} {'med_iou':>9}{'med_mae':>9}{'mean_iou':>10}{'mean_mae':>10}{'score':>8}")
    for i, r in enumerate(results[:10]):
        print(f"{i+1:>4} {r['strong']:>7} {r['weak']:>5} {r['median_iou']:>9.3f}"
              f"{r['median_mae']:>9.3f}{r['mean_iou']:>10.3f}{r['mean_mae']:>10.3f}"
              f"{r['score']:>8.3f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"autoresearch_gt_png_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "strong_thr", "weak_thr", "median_iou", "median_mae",
                     "mean_iou", "mean_mae", "score", "n_prints"])
        for i, r in enumerate(results):
            w.writerow([i + 1, r["strong"], r["weak"],
                         round(r["median_iou"], 4), round(r["median_mae"], 4),
                         round(r["mean_iou"], 4), round(r["mean_mae"], 4),
                         round(r["score"], 4), r["n_prints"]])
    print(f"\nCSV: {csv_path}")

    best = results[0]
    best_path = OUT_DIR / f"autoresearch_gt_png_best_{ts}.json"
    best_path.write_text(json.dumps({
        "strong_threshold": best["strong"],
        "weak_threshold": best["weak"],
        "median_iou": best["median_iou"],
        "median_mae": best["median_mae"],
        "score": best["score"],
    }, indent=2))
    print(f"Best config: {best_path}")

    # Heatmap: score as function of (strong, weak)
    grid = np.zeros((len(strong_thrs), len(weak_thrs)))
    for r in results:
        i = strong_thrs.index(r["strong"])
        j = weak_thrs.index(r["weak"])
        grid[i, j] = r["score"]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(grid, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(weak_thrs)))
    ax.set_xticklabels(weak_thrs)
    ax.set_yticks(range(len(strong_thrs)))
    ax.set_yticklabels(strong_thrs)
    ax.set_xlabel("weak_threshold")
    ax.set_ylabel("strong_threshold")
    b = best
    ax.set_title(f"GT-PNG mask score (med_iou - 0.5*med_mae)\n"
                  f"best: st={b['strong']}, wt={b['weak']} (score={b['score']:.3f}, "
                  f"med_iou={b['median_iou']:.3f})")
    for i in range(len(strong_thrs)):
        for j in range(len(weak_thrs)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="white" if grid[i, j] < grid.max() * 0.7 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    heatmap_path = OUT_DIR / f"autoresearch_gt_png_heatmap_{ts}.png"
    fig.savefig(heatmap_path, dpi=130)
    print(f"Heatmap: {heatmap_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())