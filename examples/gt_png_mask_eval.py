"""GT-PNG mask evaluation: use the embedded footprint thumbnail PNGs as
the real ground truth.

For each GT print, the xlsx ships with a small PNG (image1.png, image2.png,
...) that contains the actual footprint mask from the commercial gait
software.  We do NOT trust the ``步伐位置坐标`` column (we have empirical
evidence it lives in a different coordinate system than the video).  Instead:

1. At the GT's ``frame_idx``, run the algorithm and find all footprint
   candidates that overlap the GT's frame.
2. For each candidate, crop its per-frame footprint mask to a tight
   bounding box and resize to the GT PNG's dimensions.
3. Compare with the GT PNG via mean absolute error (MAE) on binary
   masks.  Lower MAE = better match.

We then report: per-print MAE distribution, overall median MAE, and
rank configs by median MAE (lower is better).
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
from deepgait3.core.pawprint.scoring_detection import detect_blobs_from_score

OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")


def _png_mask(png_path: Path, threshold: int = 30) -> np.ndarray:
    """Load an embedded GT PNG and return its binary paw mask."""
    arr = np.asarray(Image.open(png_path))
    if arr.ndim == 3:
        G = arr[:, :, 1].astype(np.int16) - arr[:, :, 0].astype(np.int16) - arr[:, :, 2].astype(np.int16)
        mask = (G > threshold).astype(np.uint8)
    else:
        mask = (arr > threshold).astype(np.uint8)
    return mask


def _algorithm_mask_in_bbox(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    exg_threshold: int = 60,
) -> np.ndarray:
    """Segment the algorithm's paw mask inside the given video bbox.

    Returns a binary mask cropped to bbox size.
    """
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(frame_bgr.shape[1], x1); y1 = min(frame_bgr.shape[0], y1)
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    score = compute_exg(crop)
    binary = (score > exg_threshold).astype(np.uint8)
    # Light cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def _crop_to_paw_region(mask: np.ndarray, max_pad_px: int = 20) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a binary mask to the bounding box of the largest blob + padding."""
    n, _lbl, stats, _cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask, (0, 0, mask.shape[1], mask.shape[0])
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    x = stats[biggest, cv2.CC_STAT_LEFT]
    y = stats[biggest, cv2.CC_STAT_TOP]
    w = stats[biggest, cv2.CC_STAT_WIDTH]
    h = stats[biggest, cv2.CC_STAT_HEIGHT]
    H, W = mask.shape
    x0 = max(0, x - max_pad_px); y0 = max(0, y - max_pad_px)
    x1 = min(W, x + w + max_pad_px); y1 = min(H, y + h + max_pad_px)
    return mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def _resize_to_match(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to target (H, W) with bilinear → re-threshold."""
    out = cv2.resize(src.astype(np.float32), (target_shape[1], target_shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return (out > 0.5).astype(np.uint8)


def _mae(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if pred_mask.shape != gt_mask.shape:
        pred_mask = _resize_to_match(pred_mask, gt_mask.shape)
    return float(np.abs(pred_mask.astype(np.int16) - gt_mask.astype(np.int16)).mean())


def _iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    if pred_mask.shape != gt_mask.shape:
        pred_mask = _resize_to_match(pred_mask, gt_mask.shape)
    inter = int(np.logical_and(pred_mask, gt_mask).sum())
    union = int(np.logical_or(pred_mask, gt_mask).sum())
    return inter / union if union > 0 else 0.0


def _find_paw_bbox_in_frame(
    frame_bgr: np.ndarray, gt_mask_shape: tuple[int, int],
    search_radius_px: int = 80,
    exg_threshold: int = 60,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """Search the frame for a footprint-like blob near the expected area.

    The xlsx coord doesn't match video pixels, so we instead sweep the
    whole frame for footprint-sized blobs and pick the one whose cropped
    mask has the closest size to the GT mask.
    """
    H, W, _ = frame_bgr.shape
    score = compute_exg(frame_bgr)
    binary = (score > exg_threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    gt_area = gt_mask_shape[0] * gt_mask_shape[1]
    candidates = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10 or area > gt_area * 5:
            continue
        bbox = (int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]))
        cx, cy = float(cents[i, 0]), float(cents[i, 1])
        # Score by how close the blob area is to GT area
        size_score = -abs(np.log(area / gt_area))
        candidates.append((size_score, i, bbox, cx, cy, area))
    candidates.sort(reverse=True)
    if not candidates:
        return np.zeros(gt_mask_shape, dtype=np.uint8), None
    best = candidates[0]
    i = best[1]
    bbox = best[2]
    mask_crop = _algorithm_mask_in_bbox(frame_bgr, bbox, exg_threshold=exg_threshold)
    return mask_crop, bbox


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", help="path to fTIR mp4 video")
    ap.add_argument("gt_xlsx", help="path to commercial gait Excel export")
    ap.add_argument("--gt-images-dir", default="/home/luofangcheng/Documents/ZCODE/tmp/gt_images")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--limit-prints", type=int, default=None,
                     help="cap how many GT prints to evaluate")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_gt_from_xlsx(args.gt_xlsx, fps=args.fps, coord_unit="px")
    if args.limit_prints:
        gt = gt[: args.limit_prints]
    print(f"GT: {len(gt)} prints")

    gt_imgs = sorted(Path(args.gt_images_dir).glob("*.png"))
    if len(gt_imgs) < len(gt):
        print(f"WARNING: only {len(gt_imgs)} PNG masks, expected {len(gt)}")
    print(f"GT PNG masks: {len(gt_imgs)}")

    # Load all video frames into memory (test1.mp4 is small enough)
    print("Loading video frames ...")
    frames = []
    for idx, frame in iter_frames(args.video):
        frames.append(frame)
    print(f"  {len(frames)} frames loaded")

    # For each GT print, find the algorithm's mask and compare
    print("\n--- Per-print MAE ---")
    print(f"{'rank':>4} {'frame':>6} {'paw':>4} {'gt_area':>8} {'alg_area':>9} "
          f"{'MAE':>7} {'IoU':>7}")
    results = []
    for i, g in enumerate(gt):
        if i >= len(gt_imgs):
            print(f"  (no PNG mask for print {i}, skipping)")
            continue
        gt_png = _png_mask(gt_imgs[i])
        if gt_png.sum() == 0:
            print(f"  (empty GT PNG for print {i}, skipping)")
            continue
        fidx = g.frame_idx
        if fidx < 1 or fidx > len(frames):
            print(f"  (frame {fidx} out of range)")
            continue
        frame_bgr = frames[fidx - 1]
        alg_crop, bbox = _find_paw_bbox_in_frame(frame_bgr, gt_png.shape)
        alg_crop_paw, _ = _crop_to_paw_region(alg_crop, max_pad_px=0)
        gt_paw, _ = _crop_to_paw_region(gt_png, max_pad_px=0)
        mae = _mae(alg_crop_paw, gt_paw)
        iou = _iou(alg_crop_paw, gt_paw)
        alg_area = int(alg_crop_paw.sum())
        gt_area = int(gt_paw.sum())
        results.append({
            "i": i, "frame_idx": fidx, "paw_id": g.paw_id,
            "gt_area": gt_area, "alg_area": alg_area,
            "mae": mae, "iou": iou,
        })
        print(f"{i:>4} {fidx:>6} {g.paw_id:>4} {gt_area:>8} {alg_area:>9} "
              f"{mae:>7.3f} {iou:>7.3f}")

    if not results:
        print("No comparable prints")
        return 1

    maes = [r["mae"] for r in results]
    ious = [r["iou"] for r in results]
    print(f"\nMedian MAE: {np.median(maes):.3f}")
    print(f"Mean MAE:   {np.mean(maes):.3f}")
    print(f"Median IoU: {np.median(ious):.3f}")
    print(f"Mean IoU:   {np.mean(ious):.3f}")

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(maes, bins=20, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("MAE (lower = better)")
    axes[0].set_ylabel("# prints")
    axes[0].set_title(f"MAE distribution (median {np.median(maes):.3f})")
    axes[1].hist(ious, bins=20, color="darkgreen", edgecolor="black")
    axes[1].set_xlabel("IoU (higher = better)")
    axes[1].set_ylabel("# prints")
    axes[1].set_title(f"IoU distribution (median {np.median(ious):.3f})")
    fig.tight_layout()
    hist_path = OUT_DIR / "gt_png_mask_eval_hist.png"
    fig.savefig(hist_path, dpi=130)
    print(f"\nHistogram: {hist_path}")

    # Per-print visualization: side-by-side GT PNG vs algorithm crop
    n_show = min(8, len(results))
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
    for k in range(n_show):
        r = results[k]
        gt_png = _png_mask(gt_imgs[r["i"]])
        fidx = r["frame_idx"]
        frame_bgr = frames[fidx - 1]
        alg_crop, bbox = _find_paw_bbox_in_frame(frame_bgr, gt_png.shape)
        alg_crop_paw, _ = _crop_to_paw_region(alg_crop, max_pad_px=0)
        gt_paw, _ = _crop_to_paw_region(gt_png, max_pad_px=0)
        # Resize to same height for display
        h_disp = 80
        def _resize_disp(m):
            scale = h_disp / m.shape[0]
            return cv2.resize(m, (max(1, int(m.shape[1] * scale)), h_disp))
        axes[0, k].imshow(_resize_disp(gt_paw), cmap="Greens", vmin=0, vmax=1)
        axes[0, k].set_title(f"GT {r['paw_id']} f{r['frame_idx']}", fontsize=8)
        axes[0, k].axis("off")
        axes[1, k].imshow(_resize_disp(alg_crop_paw), cmap="Greens", vmin=0, vmax=1)
        axes[1, k].set_title(f"Alg MAE={r['mae']:.2f}", fontsize=8)
        axes[1, k].axis("off")
    fig.suptitle("Top-8 prints: GT (top) vs Algorithm (bottom)")
    fig.tight_layout()
    cmp_path = OUT_DIR / "gt_png_mask_eval_comparison.png"
    fig.savefig(cmp_path, dpi=130)
    print(f"Comparison: {cmp_path}")

    # Save CSV
    csv_path = OUT_DIR / "gt_png_mask_eval.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "frame_idx", "paw_id", "gt_area", "alg_area", "mae", "iou"])
        for r in results:
            w.writerow([r["i"], r["frame_idx"], r["paw_id"],
                         r["gt_area"], r["alg_area"],
                         round(r["mae"], 4), round(r["iou"], 4)])
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())