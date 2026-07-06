"""Visualize the best algorithm's output: BiScaleDetector with MExG.

For each detected footprint:
1. Find its peak frame
2. Crop the bbox from the raw video frame
3. Overlay the algorithm's mask + the GT footprint mask (from xlsx PNG)
4. Save the overlay for visual inspection
5. Produce a summary grid (4×6) of all detected footprints
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.benchmark import evaluate
from deepgait3.core.pawprint.dataset import iter_frames
from deepgait3.core.pawprint.gt_loaders import load_gt_from_xlsx

VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
XLSX = "/home/luofangcheng/Documents/ZCODE/data-test1.xlsx"
GT_PNG_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp/gt_images")
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _png_mask(png_path: Path, threshold: int = 30) -> np.ndarray:
    arr = np.asarray(Image.open(png_path))
    if arr.ndim == 3:
        G = arr[:, :, 1].astype(np.int16) - arr[:, :, 0].astype(np.int16) - arr[:, :, 2].astype(np.int16)
        return (G > threshold).astype(np.uint8)
    return (arr > threshold).astype(np.uint8)


def _resize_to_match(src: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    out = cv2.resize(src.astype(np.float32), (target_shape[1], target_shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return (out > 0.5).astype(np.uint8)


def main() -> int:
    gt = load_gt_from_xlsx(XLSX, fps=60, coord_unit="px")
    print(f"GT: {len(gt)} prints")

    print("\nRunning BEST algorithm: BiScaleDetector(MExG, strong=80, weak=20) ...")
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

    n_cand = len(det.closed_candidates)
    n_valid = sum(1 for c in det.closed_candidates if c.n_strong_frames >= 5)
    print(f"  candidates: {n_cand}, valid (≥5 frames): {n_valid}")
    print(f"  toes recovered: {det.stats['total_toes_recovered']}")

    # GT-matched predictions (for visualization we use the closest blob at GT frame)
    gt_pngs = sorted(GT_PNG_DIR.glob("*.png"))[: len(gt)]

    print("\nGenerating per-print overlays ...")
    overlay_paths = []
    summary_records = []

    for i, (g, png) in enumerate(zip(gt, gt_pngs)):
        # Find the closest candidate at this GT frame
        best_c = None
        best_score = float("inf")
        for c in det.closed_candidates:
            if not c.strong_blobs:
                continue
            # Pick the strong blob closest to GT frame
            closest = min(c.strong_blobs, key=lambda b: abs(b.frame_idx - g.frame_idx))
            dist = abs(closest.frame_idx - g.frame_idx)
            if dist <= 5 and dist < best_score:
                best_score = dist
                best_c = c

        # Get the algorithm's mask at GT frame
        gt_paw_mask = _png_mask(png)
        if best_c is not None:
            # Find the strong blob at GT frame
            blobs_at_gt = [b for b in best_c.strong_blobs if b.frame_idx == g.frame_idx]
            if not blobs_at_gt:
                # use the peak-area blob in this candidate
                blobs_at_gt = [max(best_c.strong_blobs, key=lambda b: b.area_px)]
            blob = blobs_at_gt[0]
            bx1, by1, bx2, by2 = blob.bbox
            alg_paw_mask = np.zeros((by2 - by1, bx2 - bx1), dtype=np.uint8)
        else:
            alg_paw_mask = None
            bx1, by1, bx2, by2 = 0, 0, 0, 0

        # Compute metrics
        if alg_paw_mask is not None and alg_paw_mask.any():
            alg_resized = _resize_to_match(alg_paw_mask, gt_paw_mask.shape)
            inter = int(np.logical_and(alg_resized, gt_paw_mask).sum())
            union = int(np.logical_or(alg_resized, gt_paw_mask).sum())
            iou = inter / union if union > 0 else 0.0
            mae = float(np.abs(alg_resized.astype(np.int16) - gt_paw_mask.astype(np.int16)).mean())
            status = "matched" if best_c is not None else "missed"
        else:
            iou, mae, status = 0.0, 1.0, "no_blobs"

        summary_records.append({
            "i": i, "frame": g.frame_idx, "paw_id": g.paw_id,
            "status": status, "iou": iou, "mae": mae,
            "gt_area": int(gt_paw_mask.sum()),
            "alg_area": int(alg_paw_mask.sum()) if alg_paw_mask is not None else 0,
        })

        # Save a side-by-side overlay (frame + bbox + mask vs PNG)
        cap = cv2.VideoCapture(VIDEO)
        cap.set(cv2.CAP_PROP_POS_FRAMES, g.frame_idx - 1)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue

        # Draw bbox on frame
        vis = frame.copy()
        if alg_paw_mask is not None and alg_paw_mask.any():
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # green bbox
            cv2.putText(vis, f"alg IoU={iou:.2f}", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"{g.paw_id} frame={g.frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"GT=(n/a, coord mismatch)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        out_png = OUT_DIR / f"best_algo_print{i:02d}_{g.paw_id}_f{g.frame_idx:03d}.png"
        cv2.imwrite(str(out_png), vis)
        overlay_paths.append(out_png)

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(gt)}] saved {out_png.name} (status={status}, IoU={iou:.2f})")

    # Summary CSV
    import csv
    csv_path = OUT_DIR / "best_algo_per_print.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "frame", "paw_id", "status", "iou", "mae",
                     "gt_area", "alg_area"])
        for r in summary_records:
            w.writerow([r["i"], r["frame"], r["paw_id"], r["status"],
                         round(r["iou"], 3), round(r["mae"], 3),
                         r["gt_area"], r["alg_area"]])
    print(f"\nCSV: {csv_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'rank':>4} {'frame':>5} {'paw':>4} {'status':>10} {'IoU':>6} {'MAE':>6} {'gt_area':>8} {'alg_area':>8}")
    print("-" * 90)
    for i, r in enumerate(summary_records):
        print(f"{i+1:>4} {r['frame']:>5} {r['paw_id']:>4} {r['status']:>10} "
              f"{r['iou']:>6.3f} {r['mae']:>6.3f} {r['gt_area']:>8} {r['alg_area']:>8}")

    # Stats summary
    matched = [r for r in summary_records if r["status"] == "matched"]
    print(f"\nMatched: {len(matched)}/{len(summary_records)} prints")
    if matched:
        median_iou = float(np.median([r["iou"] for r in matched]))
        mean_iou = float(np.mean([r["iou"] for r in matched]))
        print(f"Median IoU: {median_iou:.3f}")
        print(f"Mean IoU:   {mean_iou:.3f}")

    # Create a grid visualization of all 25 prints
    print("\nGenerating grid visualization (5×5) ...")
    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    for i, (r, png_path) in enumerate(zip(summary_records, overlay_paths)):
        ax = axes[i // 5, i % 5]
        if png_path.exists():
            img = cv2.imread(str(png_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "no image", ha="center", va="center", transform=ax.transAxes)
        title = f"#{i+1} {r['paw_id']} f={r['frame']} | IoU={r['iou']:.2f}"
        color = "green" if r["iou"] >= 0.5 else ("orange" if r["iou"] >= 0.2 else "red")
        ax.set_title(title, color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Best Algorithm: BiScaleDetector(MExG) — 25 GT prints on test1.mp4",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    grid_path = OUT_DIR / "best_algo_grid.png"
    fig.savefig(grid_path, dpi=110)
    print(f"Grid: {grid_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())