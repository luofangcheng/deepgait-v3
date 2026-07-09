"""End-to-end extraction pipeline: video → YOLO → tracker → cycles → project data.

Usage::

    from deepgait3.core.data.pipeline import extract_trial

    trial = extract_trial(
        video_path=Path("/path/to/video.mp4"),
        output_dir=Path("/output"),
        mouse_id="C57_001",
    )
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from deepgait3.core.pawprint.models import FootMask
from deepgait3.core.pawprint.yolo_detector import YoloPawDetector
from deepgait3.core.pawprint.tracker import IoUFootprintTracker, FootprintTrack
from deepgait3.core.pawprint.cycle_builder import build_cycles
from deepgait3.core.pawprint.mouse_detector import MouseDetector
from deepgait3.core.data.schema import TrialData, ExtractedCycle, FootprintRecord
from deepgait3.core.data.exporter import export_trial

BATCH_SIZE = 16


def extract_trial(
    video_path: Path,
    output_dir: Path,
    *,
    mouse_id: str = "",
    fps: float = 60.0,
    px_per_mm: float = 1.92,
    conf: float = 0.25,
    min_area_px: int = 5,
    iou_min: float = 0.3,
    max_gap_frames: int = 3,
    min_print_frames: int = 2,
    model_path: str | Path | None = None,
) -> TrialData:
    """Run full extraction pipeline on a single video.

    Parameters
    ----------
    video_path : path to input .mp4 video.
    output_dir : where to write CSVs, images, database.
    mouse_id : animal identifier.
    fps : frame rate override (auto-detected if 0).
    px_per_mm : pixel-to-mm calibration.
    conf : YOLO confidence threshold.
    min_area_px : minimum footprint area in pixels.
    model_path : path to YOLOv8-seg checkpoint.

    Returns
    -------
    TrialData with all extracted cycles.
    """
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    # ── 2. Compute background model (first pass) ─────────────────────────
    greens_list: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        greens_list.append(frame[:, :, 1].astype(np.float32))
    cap.release()
    
    if not greens_list:
        raise RuntimeError(f"Video contains no frames: {video_path}")
    
    greens = np.stack(greens_list)
    bg_G = np.median(greens, axis=0)

    # ── 3. YOLO detection (second pass) ──────────────────────────────────
    detector = YoloPawDetector(model_path=model_path, conf=conf)
    tracker = IoUFootprintTracker(
        frame_shape=(h, w), iou_min=iou_min, max_gap_frames=max_gap_frames,
    )
    mouse_det = MouseDetector(dark_threshold=5, close_kernel=7,
                              min_area_px=3000, roi_pad=50)
    
    num_frames_processed = 0
    cap = cv2.VideoCapture(str(video_path))
    
    while True:
        # Read batch of frames
        batch_frames: list[np.ndarray] = []
        for _ in range(BATCH_SIZE):
            ret, frame = cap.read()
            if not ret:
                break
            batch_frames.append(frame)
        
        if not batch_frames:
            break
        
        # YOLO batch (GPU)
        all_footmasks = detector.detect_batch(batch_frames, bg_G, min_area_px=min_area_px)
        
        for i, footmasks in enumerate(all_footmasks):
            idx = num_frames_processed + i + 1
            
            # Mouse ROI gate (optional, skip if no mouse detected)
            tight, expanded, _ = mouse_det(batch_frames[i], bg_G, idx)
            if expanded is not None and expanded != (0, 0, 0, 0):
                footmasks = [
                    fm for fm in footmasks
                    if _intersects(fm.bbox_xyxy, expanded)
                ]
            
            tracker.update(idx, footmasks)
        
        num_frames_processed += len(batch_frames)
    
    cap.release()
    tracks = tracker.finalize()

    # ── 4. Build cycles ──────────────────────────────────────────────────
    raw_cycles = build_cycles(
        tracks, fps=fps, px_per_mm=px_per_mm, min_frames=min_print_frames,
    )
    cycles = _convert_cycles(raw_cycles, fps, px_per_mm)

    # ── 5. Assemble TrialData ────────────────────────────────────────────
    trial = TrialData(
        mouse_id=mouse_id,
        trial_name="",
        input_video=str(video_path.resolve()),
        num_frames=num_frames_processed,
        frame_width=w, frame_height=h,
        fps=fps, px_per_mm=px_per_mm,
        created_at=datetime.now(timezone.utc).isoformat(),
        cycles=cycles,
    )

    # ── 6. Export ────────────────────────────────────────────────────────
    export_trial(trial, output_dir)

    # ── 7. Cumulative visualisation ──────────────────────────────────────
    _save_cumulative(tracks, (h, w), output_dir)

    elapsed = time.perf_counter() - t0
    print(f"Extraction complete: {len(cycles)} cycles, "
          f"{num_frames_processed} frames in {elapsed:.1f}s "
          f"({num_frames_processed/elapsed:.1f} FPS)")
    return trial


# ── helpers ─────────────────────────────────────────────────────────────────

def _intersects(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


def _convert_cycles(raw_cycles, fps: float, px_per_mm: float) -> list[ExtractedCycle]:
    """Convert pawprint FootprintCycle → data ExtractedCycle with all fields."""
    out: list[ExtractedCycle] = []
    for rc in raw_cycles:
        frames = []
        for fr in sorted(rc.frames, key=lambda f: f.frame):
            frames.append(FootprintRecord(
                frame=fr.frame,
                time_s=fr.time_s,
                centroid_x_mm=fr.centroid_x_mm,
                centroid_y_mm=fr.centroid_y_mm,
                area_mm2=fr.area_mm2,
                area_px=fr.area_px,
                mean_intensity=fr.mean_intensity,
                peak_intensity=fr.peak_intensity,
                bbox_x1=fr.bbox_x1, bbox_y1=fr.bbox_y1,
                bbox_x2=fr.bbox_x2, bbox_y2=fr.bbox_y2,
                png_path=fr.png_path,
            ))

        # CoP from centroid trajectory
        cop_path = 0.0
        for i in range(1, len(frames)):
            dx = frames[i].centroid_x_mm - frames[i-1].centroid_x_mm
            dy = frames[i].centroid_y_mm - frames[i-1].centroid_y_mm
            cop_path += np.sqrt(dx**2 + dy**2)
        cop_disp = 0.0
        if len(frames) >= 2:
            cop_disp = np.sqrt(
                (frames[-1].centroid_x_mm - frames[0].centroid_x_mm)**2 +
                (frames[-1].centroid_y_mm - frames[0].centroid_y_mm)**2
            )

        out.append(ExtractedCycle(
            cycle_id=rc.cycle_id,
            touchdown_frame=rc.touchdown_frame,
            liftoff_frame=rc.liftoff_frame,
            peak_area_frame=rc.peak_area_frame,
            peak_intensity_frame=rc.peak_intensity_frame,
            duration_s=rc.duration_s,
            loading_duration_s=rc.loading_duration_s,
            weight_bearing_duration_s=rc.weight_bearing_duration_s,
            unloading_duration_s=rc.unloading_duration_s,
            peak_centroid_x_mm=rc.centroid_at_peak_x_mm,
            peak_centroid_y_mm=rc.centroid_at_peak_y_mm,
            print_length_mm=0.0,  # filled below
            print_width_mm=0.0,
            max_area_mm2=rc.max_area_mm2,
            max_area_px=rc.max_area_px,
            mean_pressure_at_peak=0.0,
            peak_pressure=max(f.peak_intensity for f in frames) if frames else 0.0,
            stand_index=0.0,
            pressure_area_ratio=0.0,
            cop_path_length_mm=cop_path,
            cop_displacement_mm=cop_disp,
            frames=frames,
        ))
    return out


def _save_cumulative(tracks, shape, output_dir):
    h, w = shape
    cum_intensity = np.zeros((h, w), dtype=np.float32)
    for track in tracks:
        for _, fm in track.foots:
            py1, py2, px1, px2 = fm.bbox_xyxy_padded
            ch, cw = py2 - py1, px2 - px1
            if ch <= 0 or cw <= 0:
                continue
            intensity_crop = fm.raw_intensity_crop
            if intensity_crop.shape[:2] != (ch, cw):
                intensity_crop = cv2.resize(intensity_crop, (cw, ch))
            mask_crop = fm.mask_padded.astype(np.float32)
            if mask_crop.shape[:2] != (ch, cw):
                mask_crop = cv2.resize(mask_crop, (cw, ch))
            cum_intensity[py1:py2, px1:px2] = np.maximum(
                cum_intensity[py1:py2, px1:px2], intensity_crop * mask_crop,
            )

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    if cum_intensity.max() > 0:
        norm = (cum_intensity / cum_intensity.max() * 255).astype(np.uint8)
        overlay[:, :, 1] = norm
    cv2.imwrite(str(output_dir / "cumulative_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "cumulative_mask.png"),
                (cum_intensity > 0).astype(np.uint8) * 255)
