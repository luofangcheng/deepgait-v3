"""Optimized footprint extraction pipeline (YOLO + GPU batch + tracker).

Produces per-cycle and per-frame CSV files with all fields required
by the gait_metrics_module for full CatWalk-compatible gait analysis.

Pipeline
--------
  Video → Frames → Median BG → YOLO batch (GPU) → FootMask → IoU Tracker
  → FootprintCycle (16+ fields) → CSV export + cumulative viz

Usage::

    python experiment/extract_footprints.py --video /path/to/video.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepgait3.core.pawprint.tracker import IoUFootprintTracker, FootprintTrack
from deepgait3.core.pawprint.models import FootMask

BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Optimized FootprintCycle — all fields required by gait_metrics_module
# plus additional useful diagnostics.  Mirrors the 36-field PawPrint contract
# from dynamics_v0.4.2 but with v0.5.0 naming.
# ---------------------------------------------------------------------------

@dataclass
class FootprintRecord:
    """Per-frame record for a single footprint."""
    frame: int
    time_s: float
    centroid_x_mm: float
    centroid_y_mm: float
    area_mm2: float
    area_px: int
    mean_intensity: float
    peak_intensity: float


@dataclass
class ExtractedCycle:
    """One complete paw contact event (touchdown → liftoff).

    Contains all 16 fields required by gait_metrics_module, plus extra
    fields for downstream analysis and debugging.
    """
    # --- Identity (filled by Stage 2, empty here) ---
    cycle_id: int
    paw_id: str = ""                          # LF/RF/LH/RH — Stage 2 assigns

    # --- Temporal ---
    touchdown_frame: int = 0
    liftoff_frame: int = 0
    peak_area_frame: int = 0
    peak_intensity_frame: int = 0
    duration_s: float = 0.0
    loading_duration_s: float = 0.0            # TD → peak area
    weight_bearing_duration_s: float = 0.0     # frames > 80% peak intensity
    unloading_duration_s: float = 0.0           # peak area → liftoff

    # --- Spatial (mm) ---
    peak_centroid_x_mm: float = 0.0
    peak_centroid_y_mm: float = 0.0
    print_length_mm: float = 0.0                # bbox height at peak
    print_width_mm: float = 0.0                 # bbox width at peak
    max_area_mm2: float = 0.0
    max_area_px: int = 0

    # --- Intensity & Pressure ---
    mean_pressure_at_peak: float = 0.0
    peak_pressure: float = 0.0
    stand_index: float = 0.0                    # pressure * area (proxy)
    pressure_area_ratio: float = 0.0

    # --- Center of Pressure ---
    cop_path_length_mm: float = 0.0
    cop_displacement_mm: float = 0.0

    # --- Per-frame records ---
    frames: List[FootprintRecord] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def peak_frame_centroid_xy_mm(self) -> Tuple[float, float]:
        """For gait_metrics_module compatibility."""
        return (self.peak_centroid_x_mm, self.peak_centroid_y_mm)

    @property
    def linkage_to_3d(self):
        """For gait_metrics_module compatibility — returns self for paw_id access."""
        return self


# ---------------------------------------------------------------------------
# YOLO → FootMask (GPU batch)
# ---------------------------------------------------------------------------

def yolo_batch_to_footmasks(
    frames_bgr: list[np.ndarray],
    results_batch,
    bg_G: np.ndarray,
) -> list[list[FootMask]]:
    """Convert batch YOLO results → per-frame FootMask lists.  Mask resize on GPU."""
    h, w = frames_bgr[0].shape[:2]
    all_footmasks: list[list[FootMask]] = [[] for _ in frames_bgr]

    for b, (frame, results) in enumerate(zip(frames_bgr, results_batch)):
        if results.masks is None:
            continue

        masks_t = results.masks.data
        if masks_t.shape[1:] != (h, w):
            masks_t = torch.nn.functional.interpolate(
                masks_t.unsqueeze(0) if masks_t.dim() == 3 else masks_t,
                size=(h, w), mode="bilinear", align_corners=False,
            )
            if masks_t.dim() == 4:
                masks_t = masks_t.squeeze(0)

        masks_np = masks_t.cpu().numpy()
        G = frame[:, :, 1].astype(np.float32)
        delta = G - bg_G

        for i in range(masks_np.shape[0]):
            mask = masks_np[i]
            mask_bin = (mask > 0.5).astype(np.uint8)
            area_px = int(mask_bin.sum())
            if area_px < 5:
                continue

            moments = cv2.moments(mask_bin)
            cx = moments["m10"] / moments["m00"] if moments["m00"] > 0 else 0.0
            cy = moments["m01"] / moments["m00"] if moments["m00"] > 0 else 0.0

            ys, xs = np.where(mask_bin)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            pad = 2
            py1, py2 = max(0, y1 - pad), min(h, y2 + pad)
            px1, px2 = max(0, x1 - pad), min(w, x2 + pad)

            mask_crop = mask_bin[py1:py2, px1:px2].astype(bool)
            delta_crop = delta[py1:py2, px1:px2]
            in_mask = delta_crop[mask_crop]
            mean_intensity = float(in_mask.mean()) if in_mask.size > 0 else 0.0
            peak_intensity = float(in_mask.max()) if in_mask.size > 0 else 0.0

            pressure_map = 18.0 * np.maximum(delta_crop - 8.0, 0) ** 0.75

            fm = FootMask(
                blob_indices=[i],
                centroid_px=(cx, cy),
                bbox_xyxy=(x1, y1, x2, y2),
                bbox_xyxy_padded=(px1, py1, px2, py2),
                mask_padded=mask_crop,
                raw_intensity_crop=delta_crop.astype(np.float32),
                bg_intensity_crop=np.zeros_like(delta_crop, dtype=np.float32),
                pressure_map=pressure_map.astype(np.float32),
                total_area_px=area_px,
                mean_intensity=mean_intensity,
                peak_intensity=peak_intensity,
                touches_edge=(x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1),
            )
            all_footmasks[b].append(fm)

    return all_footmasks


# ---------------------------------------------------------------------------
# Track → ExtractedCycle (with all gait-metrics fields)
# ---------------------------------------------------------------------------

def build_extracted_cycles(
    tracks: list[FootprintTrack],
    fps: float,
    px_per_mm: float,
    intensity_threshold_frac: float = 0.8,
) -> list[ExtractedCycle]:
    """Convert FootprintTrack objects into gait-metrics-ready ExtractedCycle list."""
    cycles: list[ExtractedCycle] = []

    for track in tracks:
        if len(track.foots) < 2:   # need at least 2 frames per cycle
            continue

        frames: list[FootprintRecord] = []
        areas: list[float] = []
        intensities: list[float] = []
        centroids: list[Tuple[float, float]] = []

        for frame_idx, fm in track.foots:
            time_s = (frame_idx - 1) / fps
            area_mm2 = fm.total_area_px / (px_per_mm ** 2)
            cx_mm = fm.centroid_px[0] / px_per_mm
            cy_mm = fm.centroid_px[1] / px_per_mm

            fr = FootprintRecord(
                frame=frame_idx,
                time_s=round(time_s, 4),
                centroid_x_mm=round(cx_mm, 3),
                centroid_y_mm=round(cy_mm, 3),
                area_mm2=round(area_mm2, 3),
                area_px=fm.total_area_px,
                mean_intensity=round(fm.mean_intensity, 2),
                peak_intensity=round(fm.peak_intensity, 2),
            )
            frames.append(fr)
            areas.append(area_mm2)
            intensities.append(fm.mean_intensity)
            centroids.append((cx_mm, cy_mm))

        # Sort by frame
        frames.sort(key=lambda f: f.frame)
        n = len(frames)

        touchdown_frame = frames[0].frame
        liftoff_frame = frames[-1].frame
        duration_s = (liftoff_frame - touchdown_frame) / fps

        # Peak area
        peak_idx = np.argmax(areas)
        peak_time = frames[peak_idx].time_s
        max_area_mm2 = areas[peak_idx]
        max_area_px = frames[peak_idx].area_px
        peak_cx, peak_cy = centroids[peak_idx]

        # Peak intensity
        peak_intensity_idx = np.argmax(intensities)

        # 3-phase: loading = TD→peak_area, unloading = peak_area→liftoff
        loading_duration_s = peak_time - frames[0].time_s
        unloading_duration_s = frames[-1].time_s - peak_time

        # Weight-bearing: frames where intensity > threshold * peak_intensity
        peak_intensity_val = intensities[peak_intensity_idx]
        wb_threshold = peak_intensity_val * intensity_threshold_frac
        wb_count = sum(1 for v in intensities if v >= wb_threshold)
        weight_bearing_duration_s = wb_count / fps

        # Print dimensions at peak area
        peak_fm = track.foots[peak_idx][1]
        px1, py1, px2, py2 = peak_fm.bbox_xyxy
        print_length_mm = (py2 - py1) / px_per_mm
        print_width_mm = (px2 - px1) / px_per_mm

        # Pressure at peak
        mean_pressure_at_peak = peak_fm.mean_intensity
        peak_pressure = max(f.mean_intensity for f in frames)
        stand_index = mean_pressure_at_peak * max_area_mm2  # proxy
        pressure_area_ratio = mean_pressure_at_peak / max_area_mm2 if max_area_mm2 > 0 else 0.0

        # CoP (Center of Pressure) path from centroid trajectory
        cop_path = 0.0
        for i in range(1, n):
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            cop_path += np.sqrt(dx ** 2 + dy ** 2)
        cop_displacement = np.sqrt(
            (centroids[-1][0] - centroids[0][0]) ** 2 +
            (centroids[-1][1] - centroids[0][1]) ** 2
        )

        cycle = ExtractedCycle(
            cycle_id=0,  # assigned after sorting
            touchdown_frame=touchdown_frame,
            liftoff_frame=liftoff_frame,
            peak_area_frame=peak_idx,
            peak_intensity_frame=peak_intensity_idx,
            duration_s=round(duration_s, 4),
            loading_duration_s=round(loading_duration_s, 4),
            weight_bearing_duration_s=round(weight_bearing_duration_s, 4),
            unloading_duration_s=round(unloading_duration_s, 4),
            peak_centroid_x_mm=round(peak_cx, 3),
            peak_centroid_y_mm=round(peak_cy, 3),
            print_length_mm=round(print_length_mm, 3),
            print_width_mm=round(print_width_mm, 3),
            max_area_mm2=round(max_area_mm2, 3),
            max_area_px=max_area_px,
            mean_pressure_at_peak=round(mean_pressure_at_peak, 2),
            peak_pressure=round(peak_pressure, 2),
            stand_index=round(stand_index, 2),
            pressure_area_ratio=round(pressure_area_ratio, 4),
            cop_path_length_mm=round(cop_path, 3),
            cop_displacement_mm=round(cop_displacement, 3),
            frames=frames,
        )
        cycles.append(cycle)

    # Sort by touchdown frame and assign sequential cycle_id
    cycles.sort(key=lambda c: c.touchdown_frame)
    for i, c in enumerate(cycles):
        c.cycle_id = i + 1

    return cycles


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_csv(cycles: list[ExtractedCycle], output_dir: Path, fps: float) -> None:
    """Write per-cycle summary and per-frame detail CSVs."""
    # Per-cycle summary
    summary_path = output_dir / "cycles_summary.csv"
    with open(summary_path, "w") as f:
        fields = [
            "cycle_id", "touchdown_frame", "liftoff_frame", "duration_s",
            "peak_area_frame", "max_area_mm2", "max_area_px",
            "peak_centroid_x_mm", "peak_centroid_y_mm",
            "print_length_mm", "print_width_mm",
            "loading_duration_s", "weight_bearing_duration_s", "unloading_duration_s",
            "mean_pressure_at_peak", "peak_pressure", "stand_index", "pressure_area_ratio",
            "cop_path_length_mm", "cop_displacement_mm", "n_frames",
        ]
        f.write(",".join(fields) + "\n")
        for c in cycles:
            row = [
                c.cycle_id, c.touchdown_frame, c.liftoff_frame, c.duration_s,
                c.peak_area_frame, c.max_area_mm2, c.max_area_px,
                c.peak_centroid_x_mm, c.peak_centroid_y_mm,
                c.print_length_mm, c.print_width_mm,
                c.loading_duration_s, c.weight_bearing_duration_s, c.unloading_duration_s,
                c.mean_pressure_at_peak, c.peak_pressure, c.stand_index, c.pressure_area_ratio,
                c.cop_path_length_mm, c.cop_displacement_mm, c.n_frames,
            ]
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"  cycles_summary.csv — {len(cycles)} cycles")

    # Per-frame detail
    detail_path = output_dir / "footprints_detail.csv"
    with open(detail_path, "w") as f:
        f.write("cycle_id,frame,time_s,centroid_x_mm,centroid_y_mm,area_mm2,area_px,mean_intensity,peak_intensity\n")
        for c in cycles:
            for fr in c.frames:
                f.write(f"{c.cycle_id},{fr.frame},{fr.time_s},{fr.centroid_x_mm},{fr.centroid_y_mm},"
                        f"{fr.area_mm2},{fr.area_px},{fr.mean_intensity},{fr.peak_intensity}\n")
    total_records = sum(c.n_frames for c in cycles)
    print(f"  footprints_detail.csv — {total_records} records")


def export_cumulative(cycles: list[ExtractedCycle], shape: tuple, output_dir: Path) -> None:
    """Black-background green max-intensity cumulative footprint image."""
    h, w = shape
    cum_intensity = np.zeros((h, w), dtype=np.float32)
    # We need the original FootMask intensity crops — stored in the track.
    # For now, reconstruct from ExtractedCycle centroids as a simple overlay.
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for c in cycles:
        px = int(c.peak_centroid_x_mm * 1.92)  # approximate — needs real px_per_mm
        py = int(c.peak_centroid_y_mm * 1.92)
        px, py = max(0, min(w - 1, px)), max(0, min(h - 1, py))
        cv2.circle(overlay, (px, py), radius=8, color=(0, 255, 0), thickness=-1)

    cv2.imwrite(str(output_dir / "cumulative_overlay.png"), overlay)
    print(f"  cumulative_overlay.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract footprints from video (YOLO + GPU)")
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/pawprint_yolo/weights/best.pt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "experiment/outputs/extraction")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--px-per-mm", type=float, default=1.92,
                        help="Pixel-to-mm calibration")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}")
        return

    from ultralytics import YOLO
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(str(args.video))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"Input: {args.video.name} — {len(frames)} frames, {w}x{h}, {fps:.0f} FPS")
    print(f"Calibration: {args.px_per_mm} px/mm")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Background
    green_stack = np.stack([f[:, :, 1].astype(np.float32) for f in frames])
    bg_G = np.median(green_stack, axis=0)

    # Tracking
    tracker = IoUFootprintTracker(frame_shape=(h, w), iou_min=0.3, max_gap_frames=3)

    # Batch inference + tracking
    t0 = time.perf_counter()
    for batch_start in range(0, len(frames), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(frames))
        batch_frames = frames[batch_start:batch_end]
        results_batch = model(batch_frames, verbose=False, stream=False)
        all_footmasks = yolo_batch_to_footmasks(batch_frames, results_batch, bg_G)
        for i, footmasks in enumerate(all_footmasks):
            tracker.update(batch_start + i + 1, footmasks)

    total_time = time.perf_counter() - t0
    tracks = tracker.finalize()

    # Build cycles with full gait-metrics fields
    cycles = build_extracted_cycles(tracks, fps=args.fps, px_per_mm=args.px_per_mm)

    # Export
    print(f"\n--- Results ({total_time:.1f}s, {len(frames)/total_time:.1f} FPS) ---")
    print(f"  Tracks: {len(tracks)}")
    print(f"  Cycles: {len(cycles)}")
    print(f"  Avg duration: {np.mean([c.duration_s for c in cycles]):.3f}s")
    print(f"  Avg max area: {np.mean([c.max_area_mm2 for c in cycles]):.2f} mm²")

    export_csv(cycles, args.output, args.fps)

    # Cumulative
    from demo_video_gpu import build_cumulative_gpu
    cum_intensity = np.zeros((h, w), dtype=np.float32)
    for track in tracks:
        for _, fm in track.foots:
            py1, py2, px1, px2 = fm.bbox_xyxy_padded
            crop_h, crop_w = py2 - py1, px2 - px1
            intensity_crop = fm.raw_intensity_crop
            if intensity_crop.shape[:2] != (crop_h, crop_w):
                intensity_crop = cv2.resize(intensity_crop, (crop_w, crop_h))
            mask_crop = fm.mask_padded.astype(np.float32)
            if mask_crop.shape[:2] != (crop_h, crop_w):
                mask_crop = cv2.resize(mask_crop, (crop_w, crop_h))
            cum_intensity[py1:py2, px1:px2] = np.maximum(
                cum_intensity[py1:py2, px1:px2], intensity_crop * mask_crop)
    overlay = build_cumulative_gpu(cum_intensity)
    cv2.imwrite(str(args.output / "cumulative_overlay.png"), overlay)
    cv2.imwrite(str(args.output / "cumulative_mask.png"),
                (cum_intensity > 0).astype(np.uint8) * 255)
    print(f"  cumulative_overlay.png, cumulative_mask.png")
    print(f"\nDone. All outputs in {args.output}")


if __name__ == "__main__":
    main()
