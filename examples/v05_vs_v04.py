"""V0.4.2 (background-subtraction) vs V0.5 (BiScale ExG) comparison.

Runs both detectors on test1.mp4 and produces:
- side-by-side summary table
- frame snapshot with overlay for each detector
- per-frame blob-count time series
- candidate / toe-recovery comparison

Outputs to /home/luofangcheng/Documents/ZCODE/tmp/
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint import PawPrintExtractor
from deepgait3.core.pawprint.biscale import BiScaleDetector
from deepgait3.core.pawprint.scoring import compute_exg
from deepgait3.core.pawprint.scoring_detection import detect_blobs_from_score

DEFAULT_VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
OUT_DIR = Path("/home/luofangcheng/Documents/ZCODE/tmp")


def _v04_pipeline(video_path: str) -> tuple[list[list[dict]], int]:
    """V0.4.2 frame-by-frame using warmup-bg + relative G diff."""
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg_G = np.zeros((H, W), dtype=np.float32)
    n_warmup = 30
    for _ in range(n_warmup):
        ok, frame = cap.read()
        if not ok:
            break
        bg_G += frame[:, :, 1].astype(np.float32)
    bg_G /= max(n_warmup, 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    blobs_per_frame: list[list[dict]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx <= n_warmup:
            blobs_per_frame.append([])
            continue
        G = frame[:, :, 1].astype(np.float32)
        delta = cv2.subtract(G, bg_G)
        binary = (delta >= 18).astype(np.uint8)
        n, _lbl, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        blobs = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 10:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            blobs.append({
                "cx": float(cents[i, 0]),
                "cy": float(cents[i, 1]),
                "area": area,
                "bbox": (x, y, x + w, y + h),
                "kind": "main",
            })
        blobs_per_frame.append(blobs)
    cap.release()
    return blobs_per_frame, n_frames


def _v05_pipeline(video_path: str) -> tuple[list[list[dict]], BiScaleDetector, int]:
    """V0.5 BiScale detector: track strong blobs + weak (toe) blobs per frame."""
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det = BiScaleDetector(
        strong_threshold=100, weak_threshold=5,
        toe_max_distance_px=24.0, toe_min_overlap_frames=1,
    )
    blobs_per_frame: list[list[dict]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        # Pre-compute strong + weak detections for the snapshot
        score = compute_exg(frame)
        _ms, strong_dicts = detect_blobs_from_score(score, threshold=100, min_area=20)
        _mw, weak_dicts = detect_blobs_from_score(score, threshold=5, min_area=3)
        # Use BiScaleDetector to update tracks
        det.process_frame(frame, idx)
        # Combine for visualization: strong blobs + weak blobs already-assigned
        weak_centers_active = set()
        for c in det.candidates.values():
            for wb in c.weak_blobs:
                if wb.frame_idx == idx:
                    weak_centers_active.add((round(wb.cx_px, 1), round(wb.cy_px, 1)))
        all_blobs = []
        for b in strong_dicts:
            all_blobs.append({
                "cx": b["cx"], "cy": b["cy"], "area": b["area"],
                "bbox": b["bbox"], "kind": "main",
            })
        for b in weak_dicts:
            key = (round(b["cx"], 1), round(b["cy"], 1))
            kind = "toe" if key in weak_centers_active else "noise"
            all_blobs.append({
                "cx": b["cx"], "cy": b["cy"], "area": b["area"],
                "bbox": b["bbox"], "kind": kind,
            })
        blobs_per_frame.append(all_blobs)
    cap.release()
    det.finalize()
    return blobs_per_frame, det, n_frames


def _annotate(rgb: np.ndarray, blobs: list[dict], show_noise: bool = False) -> np.ndarray:
    """Color-code: main=magenta, toe=green, noise=red (if show_noise)."""
    img = rgb.copy()
    for b in blobs:
        if b["kind"] == "main":
            color = (255, 0, 255); label = ""
        elif b["kind"] == "toe":
            color = (0, 255, 0); label = "T"
        elif show_noise:
            color = (255, 0, 0); label = "N"
        else:
            continue
        cx, cy = int(b["cx"]), int(b["cy"])
        radius = max(6, int(np.sqrt(b["area"] / np.pi) * 1.5))
        cv2.circle(img, (cx, cy), radius, color, 2)
        if label:
            cv2.putText(img, label, (cx + radius + 2, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def main(video_path: str = DEFAULT_VIDEO) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(video_path).exists():
        print(f"FAIL: {video_path} not found")
        return 1
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path} — {n_frames} frames @ {fps:.1f}fps")
    cap.release()

    # ---- Run both pipelines ----
    print("\n[v0.4.2] bg-subtraction ...")
    t0 = time.time()
    v04_blobs, _ = _v04_pipeline(video_path)
    t04 = time.time() - t0
    print(f"  → {t04:.1f}s, {sum(len(b) for b in v04_blobs)} total blobs")

    print("\n[v0.5]  BiScale ExG (strong=100, weak=5, dist=24px, overlap=1) ...")
    t0 = time.time()
    v05_blobs, det, _ = _v05_pipeline(video_path)
    t05 = time.time() - t0
    print(f"  → {t05:.1f}s, {sum(len(b) for b in v05_blobs)} total blobs")

    stats = det.stats
    print(f"  → {stats['n_valid_tracks']} valid footprints (≥5 frames)")
    print(f"  → {stats['total_toes_recovered']} toes recovered")

    # ---- Summary table ----
    print("\n" + "=" * 80)
    print(f"{'detector':<20}{'total_blobs':>12}{'mean/frame':>13}{'max':>6}"
          f"{'noise_frames':>14}{'toes':>8}{'valid_tracks':>14}")
    print("-" * 80)
    v04_counts = np.array([len(b) for b in v04_blobs])
    v04_noise = int((v04_counts > 6).sum())
    print(f"{'v0.4.2 bg-sub':<20}{int(v04_counts.sum()):>12}{float(v04_counts.mean()):>13.2f}"
          f"{int(v04_counts.max()):>6}{v04_noise:>14}{0:>8}{0:>14}")
    v05_counts = np.array([len(b) for b in v05_blobs])
    v05_noise = int((v05_counts > 12).sum())  # BiScale sees more (toes + main)
    print(f"{'v0.5 BiScale ExG':<20}{int(v05_counts.sum()):>12}{float(v05_counts.mean()):>13.2f}"
          f"{int(v05_counts.max()):>6}{v05_noise:>14}"
          f"{stats['total_toes_recovered']:>8}{stats['n_valid_tracks']:>14}")

    # ---- Per-frame time series ----
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(v04_counts, label="v0.4.2 (bg-sub)", linewidth=1.0, alpha=0.8, color="C0")
    ax.plot(v05_counts, label="v0.5 (BiScale ExG)", linewidth=1.0, alpha=0.7, color="C1")
    ax.set_xlabel("frame index")
    ax.set_ylabel("# blobs detected")
    ax.set_title(f"Blob count per frame — {Path(video_path).name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ts_path = OUT_DIR / "v05_vs_v04_blobs_per_frame.png"
    fig.savefig(ts_path, dpi=120)
    print(f"\nTime series: {ts_path}")

    # ---- Snapshot overlays at 3 frames ----
    snap_frames = [n_frames // 4, n_frames // 2, 3 * n_frames // 4]
    cap = cv2.VideoCapture(video_path)
    fig, axes = plt.subplots(len(snap_frames), 2,
                              figsize=(18, 4.5 * len(snap_frames)))
    for row, fidx in enumerate(snap_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # v0.4.2
        ann04 = _annotate(rgb, v04_blobs[fidx])
        axes[row, 0].imshow(ann04)
        axes[row, 0].set_title(f"v0.4.2 bg-sub — frame {fidx}, {len(v04_blobs[fidx])} main blobs",
                                fontsize=10)
        axes[row, 0].axis("off")
        # v0.5 (with noise shown in red)
        ann05 = _annotate(rgb, v05_blobs[fidx], show_noise=True)
        toes_here = sum(1 for b in v05_blobs[fidx] if b["kind"] == "toe")
        axes[row, 1].imshow(ann05)
        axes[row, 1].set_title(f"v0.5 BiScale — frame {fidx}, "
                                f"{toes_here} toes + {sum(1 for b in v05_blobs[fidx] if b['kind'] == 'main')} main",
                                fontsize=10)
        axes[row, 1].axis("off")
    cap.release()
    fig.tight_layout()
    snap_path = OUT_DIR / "v05_vs_v04_snapshot_overlay.png"
    fig.savefig(snap_path, dpi=130)
    print(f"Snapshot overlays: {snap_path}")

    # ---- Track-length distribution ----
    track_lengths = [c.duration_frames for c in det.closed_candidates]
    if track_lengths:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(track_lengths, bins=range(0, max(track_lengths) + 5, 2),
                 color="steelblue", edgecolor="black", alpha=0.85)
        ax.set_xlabel("track length (frames)")
        ax.set_ylabel("# footprints")
        ax.set_title(f"Footprint track length distribution — {len(track_lengths)} total tracks")
        ax.axvline(5, color="red", linestyle="--", label="min valid = 5")
        ax.legend()
        fig.tight_layout()
        len_path = OUT_DIR / "v05_track_lengths.png"
        fig.savefig(len_path, dpi=130)
        print(f"Track length dist: {len_path}")

    # ---- JSON summary ----
    import json
    summary = {
        "video": video_path,
        "n_frames": n_frames,
        "fps": fps,
        "v04": {
            "total_blobs": int(v04_counts.sum()),
            "mean_per_frame": float(v04_counts.mean()),
            "max_per_frame": int(v04_counts.max()),
            "noise_frames": v04_noise,
            "elapsed_s": round(t04, 2),
        },
        "v05": {
            "total_blobs": int(v05_counts.sum()),
            "mean_per_frame": float(v05_counts.mean()),
            "max_per_frame": int(v05_counts.max()),
            "noise_frames": v05_noise,
            "valid_tracks": stats["n_valid_tracks"],
            "toes_recovered": stats["total_toes_recovered"],
            "elapsed_s": round(t05, 2),
        },
    }
    json_path = OUT_DIR / "v05_vs_v04_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON: {json_path}")

    print("\nDone. Open figures in your image viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO))