"""Stage 1 pipeline — matches experiment/demo_video.py exactly.

Frame sequence → median bg → YOLO batch → tracker → cycles → visuals.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .models import TrialResult, MouseRoi
from .mouse_detector import MouseDetector
from .cycle_builder import build_cycles
from .db import create_db, save_trial
from .tracker import IoUFootprintTracker
from .yolo_detector import YoloPawDetector
from .cumulative import build_cumulative_union, render_overlay

DEFAULTS = dict(
    conf=0.25, min_area_px=5, batch_size=16,
    fps=60.0, px_per_mm=1.92,
    iou_min=0.3, max_gap_frames=3, min_print_frames=2,
    mouse_dark_threshold=5, mouse_close_kernel=7,
    mouse_min_area_px=3000, roi_pad=50,
)


class Stage1Pipeline:
    """Extract footprint cycles from frame sequence (YOLO GPU batch)."""

    def __init__(self, mouse_id: str = "", **kwargs):
        self.mouse_id = mouse_id
        for k, v in DEFAULTS.items():
            setattr(self, k, kwargs.get(k, v))
        self._detector: YoloPawDetector | None = None

    @property
    def detector(self) -> YoloPawDetector:
        if self._detector is None:
            self._detector = YoloPawDetector(conf=self.conf)
        return self._detector

    # ── main ─────────────────────────────────────────────────────────────

    def run(self, frame_dir: Path, output_dir: Path) -> TrialResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths, frames = _load_frames(frame_dir)
        H, W = frames[0].shape[:2]
        print(f"  [{self.mouse_id}] {len(frames)} frames, {W}x{H}")

        bg_G = _build_median_bg(frames)

        tracker = IoUFootprintTracker(
            frame_shape=(H, W), iou_min=self.iou_min,
            max_gap_frames=self.max_gap_frames, ref_window_frames=5,
        )
        mouse_det = MouseDetector(
            dark_threshold=self.mouse_dark_threshold,
            close_kernel=self.mouse_close_kernel,
            min_area_px=self.mouse_min_area_px, roi_pad=self.roi_pad,
        )
        mouse_rois: List[MouseRoi] = []

        # Batch detection loop (same as experiment/demo_video.py)
        for batch_start in range(0, len(frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]

            for i, frame in enumerate(batch_frames):
                idx = batch_start + i + 1
                tight, expanded, marea = mouse_det(frame, bg_G, idx)
                tag = (0, 0, 0, 0) if tight is None else tight
                eag = (0, 0, 0, 0) if expanded is None else expanded
                mouse_rois.append(MouseRoi(frame=idx, tight_xyxy=tag,
                                           expanded_xyxy=eag, area_px=marea or 0))

            all_footmasks = self.detector.detect_batch(
                batch_frames, bg_G, min_area_px=self.min_area_px)

            for i, (_, fm_list) in enumerate(zip(batch_frames, all_footmasks)):
                idx = batch_start + i + 1
                expanded = mouse_rois[idx - 1].expanded_xyxy
                if expanded != (0, 0, 0, 0):
                    fm_list = [fm for fm in fm_list
                               if _intersects(fm.bbox_xyxy, expanded)]
                tracker.update(idx, fm_list)

        tracks = tracker.finalize()
        cycles = build_cycles(tracks, fps=self.fps, px_per_mm=self.px_per_mm,
                              min_frames=self.min_print_frames)
        print(f"  [{self.mouse_id}] {len(cycles)} cycles")

        result = TrialResult(
            mouse_id=self.mouse_id, input_dir=str(frame_dir),
            num_frames=len(frames), frame_width=W, frame_height=H,
            fps=self.fps, px_per_mm=self.px_per_mm,
            roi_pad=self.roi_pad, tau_paw=0.0,
            mouse_rois=mouse_rois, cycles=cycles,
        )

        _save_per_print_pngs(output_dir, frames, result)
        db_path = output_dir / "footprints.db"
        conn = create_db(db_path)
        save_trial(conn, result)
        conn.close()

        # Cumulative (same as experiment/demo_video.py)
        _save_cumulative(tracks, output_dir, H, W)

        return result


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_frames(frame_dir: Path):
    paths = sorted(frame_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"no frame_*.png in {frame_dir}")
    frames = [cv2.imread(str(p)) for p in paths]
    return paths, frames


def _build_median_bg(frames):
    greens = np.stack([f[:, :, 1].astype(np.float32) for f in frames], axis=0)
    return np.median(greens, axis=0)


def _intersects(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


def _save_per_print_pngs(output_dir, frames, result):
    d = output_dir / "per_print"; d.mkdir(exist_ok=True)
    H, W = frames[0].shape[:2]
    fm = {i + 1: f for i, f in enumerate(frames)}
    total = 0
    for cycle in result.cycles:
        for fr in cycle.frames:
            if fr.frame not in fm:
                continue
            x1, y1, x2, y2 = fr.bbox_x1, fr.bbox_y1, fr.bbox_x2, fr.bbox_y2
            pad = 8
            px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
            px2, py2 = min(W, x2 + pad), min(H, y2 + pad)
            crop = fm[fr.frame][py1:py2, px1:px2]
            name = f"cycle_{cycle.cycle_id:04d}_frame_{fr.frame:04d}.png"
            cv2.imwrite(str(d / name), crop)
            fr.png_path = f"per_print/{name}"; total += 1
    print(f"  [{result.mouse_id}] {total} per_print PNGs")


def _save_cumulative(tracks, output_dir, H, W):
    """Save cumulative overlay using GPU-accelerated track-union algorithm."""
    cum = build_cumulative_union(tracks, (H, W), use_gpu=True)
    overlay = render_overlay(cum)
    cv2.imwrite(str(output_dir / "cumulative_overlay.png"), overlay)
    cv2.imwrite(str(output_dir / "cumulative_mask.png"),
                (cum > 0).astype(np.uint8) * 255)
