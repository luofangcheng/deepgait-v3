"""Stage-1 pipeline: frame sequence → cycles + database + visualisations.

Usage as a library::

    from deepgait3.core.pawprint.pipeline import Stage1Pipeline
    pipeline = Stage1Pipeline(mouse_id="C57_001", tau_paw=10, roi_pad=50, fps=60)
    result = pipeline.run(frame_dir, output_dir)

Or via CLI (see ``cli.py``)::

    deepgait3 stage1 /data/mouse_001
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .models import TrialResult, MouseRoi, FootprintCycle
from .mouse_detector import MouseDetector
from .cycle_builder import build_cycles
from .db import create_db, save_trial

# low-level detectors (unchanged from v2 dynamics)
from .detection import detect_blobs
from .grouping import cluster_blobs_into_feet
from .tracker import IoUFootprintTracker


# ── defaults (confirmed 2026-06-27 on test1-frames) ─────────────────────────

DEFAULTS = dict(
    tau_paw=10,
    min_area_px=10,
    D_merge_px=23.0,
    walkway_roi=(0, 15, 1920, 360),
    bbox_pad_px=8,
    mouse_dark_threshold=5,
    mouse_close_kernel=7,
    mouse_min_area_px=3000,
    roi_pad=50,
    fps=60.0,
    px_per_mm=1.92,
    iou_min=0.3,
    max_gap_frames=3,
    min_print_frames=2,
)


class Stage1Pipeline:
    """Extract footprint cycles from a frame sequence with mouse-ROI gating."""

    def __init__(self, mouse_id: str = "", **kwargs):
        self.mouse_id = mouse_id
        for k, v in DEFAULTS.items():
            setattr(self, k, kwargs.get(k, v))

    # ── main entry ──────────────────────────────────────────────────────

    def run(self, frame_dir: Path, output_dir: Path) -> TrialResult:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. load frames
        paths, frames = _load_frames(frame_dir)
        H, W = frames[0].shape[:2]
        print(f"  [{self.mouse_id}] loaded {len(frames)} frames, {W}x{H}")

        # 2. median background (static)
        bg_G = _build_median_bg(frames)

        # 3. per-frame: mouse → blobs → cluster → track (within ROI)
        tracker = IoUFootprintTracker(
            frame_shape=(H, W), iou_min=self.iou_min,
            max_gap_frames=self.max_gap_frames, ref_window_frames=5,
        )
        mouse_det = MouseDetector(
            dark_threshold=self.mouse_dark_threshold,
            close_kernel=self.mouse_close_kernel,
            min_area_px=self.mouse_min_area_px,
            roi_pad=self.roi_pad,
        )
        mouse_rois: List[MouseRoi] = []

        for idx, frame in enumerate(frames, start=1):
            tight, expanded, marea = mouse_det(frame, bg_G, idx)
            if tight is not None:
                mouse_rois.append(MouseRoi(
                    frame=idx, tight_xyxy=tight,
                    expanded_xyxy=expanded, area_px=marea,
                ))
            else:
                mouse_rois.append(MouseRoi(
                    frame=idx, tight_xyxy=(0,0,0,0),
                    expanded_xyxy=(0,0,0,0), area_px=0,
                ))

            blobs = detect_blobs(
                frame, bg_G, tau_paw=self.tau_paw,
                min_area_px=self.min_area_px,
                walkway_roi=self.walkway_roi,
                bbox_pad_px=self.bbox_pad_px,
            )
            footmasks = cluster_blobs_into_feet(
                blobs, D_merge_px=self.D_merge_px,
                frame_shape=(H, W),
            )

            # clip footmasks to expanded ROI
            if expanded is not None:
                ex1, ey1, ex2, ey2 = expanded
                footmasks = [
                    fm for fm in footmasks
                    if _intersects(fm.bbox_xyxy, expanded)
                ]

            tracker.update(idx, footmasks)

        tracks = tracker.finalize()

        # 4. build cycles
        cycles = build_cycles(
            tracks, fps=self.fps, px_per_mm=self.px_per_mm,
            min_frames=self.min_print_frames,
        )
        print(f"  [{self.mouse_id}] {len(cycles)} footprint cycles found")

        # 5. assemble TrialResult
        result = TrialResult(
            mouse_id=self.mouse_id,
            input_dir=str(frame_dir),
            num_frames=len(frames),
            frame_width=W, frame_height=H,
            fps=self.fps, px_per_mm=self.px_per_mm,
            roi_pad=self.roi_pad, tau_paw=self.tau_paw,
            mouse_rois=mouse_rois,
            cycles=cycles,
        )

        # 6. save per_print PNGs + set png_path (before DB so DB picks it up)
        _save_per_print_pngs(output_dir, frames, result)

        # 7. save to database
        db_path = output_dir / "footprints.db"
        conn = create_db(db_path)
        trial_id = save_trial(conn, result)
        conn.close()
        print(f"  [{self.mouse_id}] saved to {db_path} (trial_id={trial_id})")

        # 8. visualisations
        _save_visuals(output_dir, frames, bg_G, result, self.__dict__)

        return result


# ── helpers ─────────────────────────────────────────────────────────────────

def _load_frames(frame_dir: Path):
    paths = sorted(frame_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"no frame_*.png in {frame_dir}")
    frames = [cv2.imread(str(p)) for p in paths]
    for p, f in zip(paths, frames):
        if f is None:
            raise ValueError(f"failed to load {p}")
    return paths, frames


def _build_median_bg(frames):
    greens = np.stack([f[:,:,1].astype(np.float32) for f in frames], axis=0)
    return np.median(greens, axis=0)


def _intersects(bbox_a, bbox_b):
    """True if two xyxy bboxes overlap."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


# ── per_print PNG crop ──────────────────────────────────────────────────────

def _save_per_print_pngs(output_dir: Path, frames, result: TrialResult):
    """Crop and save each footprint frame as a PNG in per_print/."""
    per_print_dir = output_dir / "per_print"
    per_print_dir.mkdir(exist_ok=True)
    H, W = frames[0].shape[:2]
    frame_map = {i + 1: f for i, f in enumerate(frames)}  # 1-based
    total = 0
    for cycle in result.cycles:
        for fr in cycle.frames:
            if fr.frame not in frame_map:
                continue
            x1, y1, x2, y2 = fr.bbox_x1, fr.bbox_y1, fr.bbox_x2, fr.bbox_y2
            pad = 8
            px1 = max(0, x1 - pad); py1 = max(0, y1 - pad)
            px2 = min(W, x2 + pad); py2 = min(H, y2 + pad)
            crop = frame_map[fr.frame][py1:py2, px1:px2]
            name = f"cycle_{cycle.cycle_id:04d}_frame_{fr.frame:04d}.png"
            cv2.imwrite(str(per_print_dir / name), crop)
            fr.png_path = f"per_print/{name}"
            total += 1
    print(f"  [{result.mouse_id}] {total} per_print PNGs saved")


# ── visualisation ───────────────────────────────────────────────────────────

def _save_visuals(output_dir: Path, frames, bg_G, result: TrialResult, cfg: dict):
    H, W = frames[0].shape[:2]

    # cumulative mask
    cum_mask = _build_cumulative_mask(frames, bg_G, result, H, W)
    cv2.imwrite(str(output_dir / "cumulative_mask.png"),
                cum_mask.astype(np.uint8) * 255)

    # cumulative intensity (green footprints / black background)
    cum_intensity = _build_cumulative_intensity(frames, bg_G, result, H, W)
    out_bg_black = np.zeros((H, W, 3), dtype=np.uint8)
    if cum_mask.any():
        vals = cum_intensity[cum_mask]
        lo, hi = float(vals.min()), float(vals.max())
        denom = hi - lo if hi > lo else 1.0
        norm = np.clip((cum_intensity - lo) / denom, 0, 1)
        out_bg_black[:, :, 1] = np.where(cum_mask, (norm * 255).astype(np.uint8), 0)
    cv2.imwrite(str(output_dir / "cumulative_intensity.png"), out_bg_black)

    # white-bg + green footprints overlay
    out_white = np.full((H, W, 3), 255, dtype=np.uint8)
    if cum_mask.any():
        vals = cum_intensity[cum_mask]
        lo, hi = float(vals.min()), float(vals.max())
        denom = hi - lo if hi > lo else 1.0
        norm = np.where(cum_mask, np.clip((cum_intensity - lo) / denom, 0, 1), 0)
        gi = (norm * 255).astype(np.uint8)
        out_white[:, :, 0] = np.where(cum_mask, 0, 255)   # B
        out_white[:, :, 2] = np.where(cum_mask, 0, 255)   # R
        out_white[:, :, 1] = np.where(cum_mask, gi, 255)  # G
    # mouse centre trail (yellow dots)
    for roi in result.mouse_rois:
        ex1, ey1, ex2, ey2 = roi.expanded_xyxy
        cv2.circle(out_white, ((ex1 + ex2) // 2, (ey1 + ey2) // 2),
                   3, (0, 255, 255), -1)
    cv2.imwrite(str(output_dir / "cumulative_overlay.png"), out_white)

    # per-frame debug: save every Nth frame to keep volume reasonable
    per_dir = output_dir / "per_frame"
    per_dir.mkdir(exist_ok=True)
    mouse_det = MouseDetector(
        dark_threshold=cfg.get("mouse_dark_threshold", 5),
        close_kernel=cfg.get("mouse_close_kernel", 7),
        min_area_px=cfg.get("mouse_min_area_px", 3000),
        roi_pad=cfg.get("roi_pad", 50),
    )
    for idx, frame in enumerate(frames, start=1):
        tight, _, _ = mouse_det(frame, bg_G, idx)
        blobs = detect_blobs(frame, bg_G,
                             tau_paw=cfg.get("tau_paw", 10),
                             min_area_px=cfg.get("min_area_px", 10),
                             walkway_roi=cfg.get("walkway_roi", (0,15,1920,360)),
                             bbox_pad_px=cfg.get("bbox_pad_px", 8))
        feet = cluster_blobs_into_feet(blobs,
                                       D_merge_px=cfg.get("D_merge_px", 23.0),
                                       frame_shape=(H, W))
        vis = frame.copy()
        if tight is not None:
            e = (max(0, tight[0]-cfg.get("roi_pad", 50)),
                 max(0, tight[1]-cfg.get("roi_pad", 50)),
                 min(W, tight[2]+cfg.get("roi_pad", 50)),
                 min(H, tight[3]+cfg.get("roi_pad", 50)))
            cv2.rectangle(vis, (tight[0], tight[1]), (tight[2], tight[3]),
                          (255, 0, 0), 1)   # blue = tight
            cv2.rectangle(vis, (e[0], e[1]), (e[2], e[3]),
                          (0, 255, 0), 2)   # green = expanded
        for fm in feet:
            px1, py1, px2, py2 = fm.bbox_xyxy_padded
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 1)  # red
        cv2.imwrite(str(per_dir / f"frame_{idx:04d}_det.png"), vis)

    # mouse_roi.txt
    with open(output_dir / "mouse_roi.txt", "w") as fh:
        fh.write("# frame  tight_x1 tight_y1 tight_x2 tight_y2  "
                 "expanded_x1 expanded_y1 expanded_x2 expanded_y2  area_px\n")
        for roi in result.mouse_rois:
            t = roi.tight_xyxy
            e = roi.expanded_xyxy
            fh.write(f"{roi.frame}  {t[0]} {t[1]} {t[2]} {t[3]}  "
                     f"{e[0]} {e[1]} {e[2]} {e[3]}  {roi.area_px}\n")


def _build_cumulative_mask(frames, bg_G, result: TrialResult, H, W):
    cum_mask = np.zeros((H, W), dtype=bool)
    mouse_map = {r.frame: r for r in result.mouse_rois}
    for idx, frame in enumerate(frames, start=1):
        roi_info = mouse_map.get(idx)
        if roi_info is None or roi_info.expanded_xyxy == (0,0,0,0):
            continue
        ex1, ey1, ex2, ey2 = roi_info.expanded_xyxy
        blobs = detect_blobs(frame, bg_G,
                             tau_paw=result.tau_paw, min_area_px=10,
                             walkway_roi=(0,15,1920,360), bbox_pad_px=8)
        feet = cluster_blobs_into_feet(blobs, D_merge_px=23.0, frame_shape=(H,W))
        frame_mask = np.zeros((H, W), dtype=bool)
        for fm in feet:
            px1, py1, px2, py2 = fm.bbox_xyxy_padded
            full = np.zeros((H, W), dtype=bool)
            full[py1:py2, px1:px2] = fm.mask_padded
            full[:ey1, :] = False; full[ey2:, :] = False
            full[:, :ex1] = False; full[:, ex2:] = False
            frame_mask |= full
        cum_mask |= frame_mask
    return cum_mask


def _build_cumulative_intensity(frames, bg_G, result: TrialResult, H, W):
    cum_intensity = np.zeros((H, W), dtype=np.float32)
    mouse_map = {r.frame: r for r in result.mouse_rois}
    for idx, frame in enumerate(frames, start=1):
        roi_info = mouse_map.get(idx)
        if roi_info is None or roi_info.expanded_xyxy == (0,0,0,0):
            continue
        ex1, ey1, ex2, ey2 = roi_info.expanded_xyxy
        blobs = detect_blobs(frame, bg_G,
                             tau_paw=result.tau_paw, min_area_px=10,
                             walkway_roi=(0,15,1920,360), bbox_pad_px=8)
        feet = cluster_blobs_into_feet(blobs, D_merge_px=23.0, frame_shape=(H,W))
        # ROI mask
        roi_mask = np.zeros((H, W), dtype=bool)
        for fm in feet:
            px1, py1, px2, py2 = fm.bbox_xyxy_padded
            full = np.zeros((H, W), dtype=bool)
            full[py1:py2, px1:px2] = fm.mask_padded
            full[:ey1, :] = False; full[ey2:, :] = False
            full[:, :ex1] = False; full[:, ex2:] = False
            roi_mask |= full
        for fm in feet:
            px1, py1, px2, py2 = fm.bbox_xyxy_padded
            delta = np.maximum(fm.raw_intensity_crop, 0.0)
            region = cum_intensity[py1:py2, px1:px2]
            local = roi_mask[py1:py2, px1:px2]
            if local.any():
                np.maximum(region, delta, out=region, where=local)
    return cum_intensity


__all__ = ["Stage1Pipeline", "DEFAULTS"]
