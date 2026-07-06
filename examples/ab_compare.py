"""A/B comparison: V3 v0.4.2 background-subtraction vs 4 color-score algorithms.

Runs all five detectors frame-by-frame on ``test1.mp4`` and emits a
side-by-side report:

- Per-frame blob count (time series)
- Total blob count over the full video
- Mean + max blob area
- Estimated "noise rate" = frames with > 6 blobs (likely over-segmentation)
- PNG snapshot showing one frame's detection overlay for each detector

Usage (from ``deepgait-v3/``):

    python examples/ab_compare.py [video_path] [frame_idx]

Defaults: video=~/Documents/ZCODE/test1.mp4, snapshot frame = middle of video.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path setup so the script runs from the repo root without editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint import PawPrintExtractor
from deepgait3.core.pawprint.scoring import SCORING_ALGORITHMS
from deepgait3.core.pawprint.scoring_detection import detect_blobs_from_score


DEFAULT_VIDEO = "/home/luofangcheng/Documents/ZCODE/test1.mp4"
DEFAULT_OUT_DIR = Path("/tmp/ab_compare_out")

# Per-algorithm default thresholds (taken from test_paw_detection.py defaults)
DEFAULT_THRESHOLDS = {
    "exg": 80,
    "lab_astar": 10,
    "exgr": 50,
    "color_distance": 150,
}


def _v3_baseline(video_path: str) -> list[list[dict]]:
    """V3 v0.4.2 background-subtraction pipeline, blob-per-frame output.

    We don't reuse PawPrintExtractor directly because it tracks+clusters
    footprints into PawPrint objects.  For a fair frame-level comparison
    we re-implement the detector step in isolation: warmup bg + per-frame
    detection, no tracking.
    """
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Warmup: average G channel of first 30 frames
    bg_G = np.zeros((H, W), dtype=np.float32)
    n_warmup = 30
    for i in range(n_warmup):
        ok, frame = cap.read()
        if not ok:
            break
        bg_G += frame[:, :, 1].astype(np.float32)
    bg_G /= max(n_warmup, 1)

    # Replay and detect
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
        binary = (delta >= 18).astype(np.uint8)  # tau_paw = 18 (balanced)
        n, _lbl, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        blobs = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 10:  # min_area_px (balanced)
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
            })
        blobs_per_frame.append(blobs)
    cap.release()
    return blobs_per_frame, n_frames, fps


def _color_score_detector(video_path: str, algo_name: str,
                           threshold: int, min_area: int = 5) -> list[list[dict]]:
    """Run one of the 4 color-score algorithms on every frame."""
    score_fn = SCORING_ALGORITHMS[algo_name]
    cap = cv2.VideoCapture(video_path)
    blobs_per_frame: list[list[dict]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        score = score_fn(frame)
        _, blobs = detect_blobs_from_score(score, threshold=threshold, min_area=min_area)
        blobs_per_frame.append(blobs)
    cap.release()
    return blobs_per_frame


def _summarize(name: str, blobs_per_frame: list[list[dict]]) -> dict:
    """Aggregate statistics for one detector."""
    total_blobs = sum(len(b) for b in blobs_per_frame)
    counts = np.array([len(b) for b in blobs_per_frame], dtype=np.int32)
    all_areas = [bb["area"] for b in blobs_per_frame for bb in b]
    return {
        "name": name,
        "total_blobs": int(total_blobs),
        "mean_per_frame": float(counts.mean()),
        "max_per_frame": int(counts.max()),
        "noise_frames": int((counts > 6).sum()),  # likely over-segmentation
        "empty_frames": int((counts == 0).sum()),
        "mean_area_px": float(np.mean(all_areas)) if all_areas else 0.0,
        "median_area_px": float(np.median(all_areas)) if all_areas else 0.0,
        "blobs_per_frame": counts.tolist(),
    }


def _make_overlay(rgb_frame: np.ndarray, blobs: list[dict], color, label: str) -> np.ndarray:
    """Annotate blobs on a copy of the frame."""
    img = rgb_frame.copy()
    for i, b in enumerate(blobs[:6]):
        cx, cy = int(b["cx"]), int(b["cy"])
        radius = max(8, int(np.sqrt(b["area"] / np.pi) * 1.5))
        cv2.circle(img, (cx, cy), radius, color, 2)
        cv2.putText(img, f"{label}{i}:{b['area']}", (cx + radius + 3, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def main(video_path: str = DEFAULT_VIDEO, snapshot_frame: int | None = None) -> int:
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists():
        print(f"FAIL: video not found: {video_path}")
        return 1

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("=" * 70)
    print(f"A/B comparison on {video_path}")
    print(f"  {n_frames} frames @ {fps:.1f} fps, {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    cap.release()

    if snapshot_frame is None:
        snapshot_frame = n_frames // 2

    # --------------------------------------------------------------
    # Run all 5 detectors
    # --------------------------------------------------------------
    print("\n[1/5] V3 v0.4.2 baseline (background-subtraction, tau=18) ...")
    v3_blobs, n_seen, _ = _v3_baseline(video_path)
    print(f"     → {sum(len(b) for b in v3_blobs)} total blobs across {n_seen} frames")

    color_results: dict[str, list[list[dict]]] = {}
    for name in SCORING_ALGORITHMS:
        thr = DEFAULT_THRESHOLDS[name]
        print(f"[+] {name} (threshold={thr}) ...", end=" ", flush=True)
        color_results[name] = _color_score_detector(video_path, name, thr)
        n = sum(len(b) for b in color_results[name])
        print(f"{n} blobs")

    # --------------------------------------------------------------
    # Summary table
    # --------------------------------------------------------------
    summary = {
        "v3_baseline": _summarize("v3_baseline", v3_blobs),
    }
    for name, blobs in color_results.items():
        summary[name] = _summarize(name, blobs)

    print("\n" + "=" * 70)
    print(f"{'detector':<20}{'total':>10}{'mean/frame':>12}{'max':>6}{'noise':>8}{'empty':>8}{'mean_area':>12}")
    print("-" * 70)
    for s in summary.values():
        print(f"{s['name']:<20}{s['total_blobs']:>10}{s['mean_per_frame']:>12.2f}"
              f"{s['max_per_frame']:>6}{s['noise_frames']:>8}{s['empty_frames']:>8}"
              f"{s['mean_area_px']:>12.1f}")

    # --------------------------------------------------------------
    # Visualization: time series + frame snapshot
    # --------------------------------------------------------------
    print("\n[*] Generating comparison figures ...")

    # (a) Blob count over time
    fig, ax = plt.subplots(figsize=(14, 5))
    for s in summary.values():
        ax.plot(s["blobs_per_frame"], label=s["name"], linewidth=1.0, alpha=0.8)
    ax.set_xlabel("frame index")
    ax.set_ylabel("# blobs detected")
    ax.set_title(f"Blob count per frame — {Path(video_path).name}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ts_path = DEFAULT_OUT_DIR / "blobs_per_frame.png"
    fig.savefig(ts_path, dpi=120)
    print(f"  saved: {ts_path}")

    # (b) Side-by-side snapshot at snapshot_frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, snapshot_frame)
    ok, frame_bgr = cap.read()
    cap.release()
    assert ok, "could not read snapshot frame"
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # We need a fresh pass for snapshot — re-run per-frame is cheap, store in dict
    # already populated above; use first one
    snap_v3 = v3_blobs[snapshot_frame] if snapshot_frame < len(v3_blobs) else []
    snap_color = {n: b[snapshot_frame] if snapshot_frame < len(b) else []
                   for n, b in color_results.items()}

    colors = {
        "v3_baseline": (255, 0, 255),
        "exg": (0, 255, 0),
        "lab_astar": (0, 255, 255),
        "exgr": (255, 255, 0),
        "color_distance": (255, 128, 0),
    }

    n_panels = 1 + len(snap_color)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.5 * n_panels))
    axes[0].imshow(_make_overlay(rgb, snap_v3, colors["v3_baseline"], "v3_"))
    axes[0].set_title(f"V3 v0.4.2 baseline (bg-sub) — {len(snap_v3)} blobs", fontsize=10)
    axes[0].axis("off")
    for i, (name, blobs) in enumerate(snap_color.items(), start=1):
        axes[i].imshow(_make_overlay(rgb, blobs, colors[name], f"{name[:3]}_"))
        thr = DEFAULT_THRESHOLDS[name]
        axes[i].set_title(f"{name} (threshold={thr}) — {len(blobs)} blobs", fontsize=10)
        axes[i].axis("off")
    fig.tight_layout()
    snap_path = DEFAULT_OUT_DIR / f"snapshot_frame{snapshot_frame:04d}.png"
    fig.savefig(snap_path, dpi=130)
    print(f"  saved: {snap_path}")

    # Save raw summary as JSON
    import json
    summary_json = {k: {kk: vv for kk, vv in v.items() if kk != "blobs_per_frame"}
                    for k, v in summary.items()}
    summary_json["blobs_per_frame"] = {k: v["blobs_per_frame"] for k, v in summary.items()}
    json_path = DEFAULT_OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"  saved: {json_path}")

    print("\nDone. Open the figures with eog or your image viewer.")
    return 0


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    snap = int(sys.argv[2]) if len(sys.argv) > 2 else None
    raise SystemExit(main(video, snap))