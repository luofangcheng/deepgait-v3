"""v0.4.2 extractor with sensitivity_mode (strict/balanced/loose)."""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
import cv2

from .models import FrameData, PawPrint, QualityFlags, FootMask
from .detection import detect_blobs
from .grouping import cluster_blobs_into_feet
from .tracker import IoUFootprintTracker

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


SENSITIVITY_PRESETS = {
    "strict":   dict(tau_paw=18, min_area_px=19, min_print_frames=3, iou_min=0.4),
    "balanced": dict(tau_paw=18, min_area_px=10, min_print_frames=2, iou_min=0.3),
    "loose":    dict(tau_paw=10, min_area_px=3,  min_print_frames=1, iou_min=0.2),
}


class PawPrintExtractor:
    def __init__(self,
                 px_per_mm=1.92, fps=60,
                 sensitivity_mode="balanced",
                 tau_paw=None, min_area_px=None, max_area_px=None,
                 D_merge_px=23.0,
                 walkway_roi=(0, 15, 1920, 360),
                 iou_min=None, ref_window_frames=5, max_gap_frames=3,
                 min_print_frames=None,
                 pressure_curve_k=18.0, pressure_curve_alpha=0.75,
                 pressure_curve_b=8.0,
                 warmup_frames=30, bg_alpha=0.02, bbox_pad_px=8,
                 decay_extension_max_frames=15,
                 decay_intensity_threshold=6.0):
        # Apply sensitivity presets where user didn't override
        if sensitivity_mode in SENSITIVITY_PRESETS:
            preset = SENSITIVITY_PRESETS[sensitivity_mode]
            if tau_paw is None: tau_paw = preset["tau_paw"]
            if min_area_px is None: min_area_px = preset["min_area_px"]
            if iou_min is None: iou_min = preset["iou_min"]
            if min_print_frames is None: min_print_frames = preset["min_print_frames"]
        else:
            if tau_paw is None: tau_paw = 18
            if min_area_px is None: min_area_px = 10
            if iou_min is None: iou_min = 0.3
            if min_print_frames is None: min_print_frames = 2

        self.sensitivity_mode = sensitivity_mode
        self.px_per_mm = float(px_per_mm)
        self.fps = int(fps)
        self.tau_paw = int(tau_paw)
        self.min_area_px = int(min_area_px)
        self.max_area_px = max_area_px
        self.D_merge_px = float(D_merge_px)
        self.walkway_roi = walkway_roi
        self.iou_min = float(iou_min)
        self.ref_window_frames = int(ref_window_frames)
        self.max_gap_frames = int(max_gap_frames)
        self.min_print_frames = int(min_print_frames)
        self.k = pressure_curve_k
        self.alpha = pressure_curve_alpha
        self.b = pressure_curve_b
        self.warmup_frames = int(warmup_frames)
        self.bg_alpha = float(bg_alpha)
        self.bbox_pad = int(bbox_pad_px)
        self.decay_extension_max_frames = int(decay_extension_max_frames)
        self.decay_thr = float(decay_intensity_threshold)

    def _pressure_from_intensity(self, I):
        return self.k * np.power(np.maximum(I - self.b, 0.0), self.alpha)

    def __call__(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError("cannot open " + str(video_path))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracker = IoUFootprintTracker(
            frame_shape=(H, W), iou_min=self.iou_min,
            max_gap_frames=self.max_gap_frames,
            ref_window_frames=self.ref_window_frames,
        )
        bg_G = None
        warmup_buf = []
        frame_idx = 0
        bg_snapshots = {}
        all_frames_bgr = []

        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1
            all_frames_bgr.append(frame.copy())
            G = frame[:, :, 1].astype(np.float32)
            if bg_G is None:
                warmup_buf.append(G)
                if len(warmup_buf) >= self.warmup_frames:
                    bg_G = np.mean(np.stack(warmup_buf), axis=0)
                    warmup_buf = []
                continue
            blobs = detect_blobs(frame, bg_G,
                                  tau_paw=self.tau_paw,
                                  min_area_px=self.min_area_px,
                                  max_area_px=self.max_area_px,
                                  walkway_roi=self.walkway_roi,
                                  bbox_pad_px=self.bbox_pad)
            footmasks = cluster_blobs_into_feet(
                blobs, D_merge_px=self.D_merge_px,
                frame_shape=(H, W),
                pressure_k=self.k, pressure_alpha=self.alpha,
                pressure_b=self.b,
            )
            tracker.update(frame_idx, footmasks)
            mask_excl = np.zeros_like(G, dtype=bool)
            for fm in footmasks:
                px1, py1, px2, py2 = fm.bbox_xyxy_padded
                mask_excl[py1:py2, px1:px2] |= fm.mask_padded
            bg_G = np.where(mask_excl, bg_G, bg_G + self.bg_alpha * (G - bg_G))
            bg_snapshots[frame_idx] = bg_G.copy()
        cap.release()
        tracks = tracker.finalize()

        # Decay extension
        N_total = len(all_frames_bgr)
        for tr in tracks:
            if tr.n_frames == 0:
                tr._extension_frames = set()
                continue
            last_frame = tr.last_frame
            last_fm = tr.foots[-1][1]
            bbox = last_fm.bbox_xyxy_padded
            mask_ref = last_fm.mask_padded
            bg_ref = bg_snapshots.get(last_frame)
            if bg_ref is None:
                tr._extension_frames = set()
                continue
            ext_set = set()
            for offset in range(1, self.decay_extension_max_frames + 1):
                target = last_frame + offset
                if target > N_total: break
                frame = all_frames_bgr[target - 1]
                G = frame[:, :, 1].astype(np.float32)
                px1, py1, px2, py2 = bbox
                delta_local = G[py1:py2, px1:px2] - bg_ref[py1:py2, px1:px2]
                delta_local_clean = np.maximum(delta_local, 0.0)
                if mask_ref.any():
                    int_in_mask = float(delta_local_clean[mask_ref].mean())
                else:
                    int_in_mask = 0.0
                if int_in_mask < self.decay_thr: break
                still_above = (delta_local_clean > self.decay_thr) & mask_ref
                area_px = int(still_above.sum())
                pressure_local = self._pressure_from_intensity(delta_local_clean)
                ext_fm = FootMask(
                    blob_indices=[],
                    centroid_px=last_fm.centroid_px,
                    bbox_xyxy=last_fm.bbox_xyxy,
                    bbox_xyxy_padded=bbox,
                    mask_padded=mask_ref.copy(),
                    raw_intensity_crop=delta_local.copy().astype(np.float32),
                    bg_intensity_crop=bg_ref[py1:py2, px1:px2].copy(),
                    pressure_map=pressure_local.astype(np.float32),
                    total_area_px=area_px if area_px > 0 else last_fm.total_area_px,
                    mean_intensity=int_in_mask,
                    peak_intensity=float(delta_local_clean.max()),
                    touches_edge=False,
                )
                tr.foots.append((target, ext_fm))
                ext_set.add(target)
            tr._extension_frames = ext_set

        # Build PawPrints
        prints = []
        for tr in tracks:
            if tr.n_frames < self.min_print_frames: continue
            ext_set = getattr(tr, "_extension_frames", set())
            prints.append(self._build_pawprint(tr.track_id, tr.foots, ext_set))
        prints.sort(key=lambda p: p.touchdown_frame)
        for i, p in enumerate(prints):
            p.print_id = i
        return prints

    def _build_pawprint(self, print_id, track, ext_set):
        frames = []
        for (fi, fm) in track:
            mask = fm.mask_padded
            delta = fm.raw_intensity_crop
            bg = fm.bg_intensity_crop
            pmap = fm.pressure_map
            area_mm2 = float(fm.total_area_px) / (self.px_per_mm ** 2)
            if mask.any():
                mean_int = float(delta[mask].mean())
                mean_p = float(pmap[mask].mean())
                peak_int = float(delta.max())
                peak_p = float(pmap.max())
            else:
                mean_int = mean_p = peak_int = peak_p = 0.0
            frames.append(FrameData(
                frame=fi, time_s=fi / self.fps,
                bbox_xyxy=fm.bbox_xyxy,
                bbox_xyxy_padded=fm.bbox_xyxy_padded,
                raw_intensity_crop=delta.copy(),
                bg_intensity_crop=bg.copy(),
                paw_mask=mask.copy(), pressure_map=pmap.copy(),
                centroid_xy_mm=(fm.centroid_px[0] / self.px_per_mm,
                                fm.centroid_px[1] / self.px_per_mm),
                area_mm2=area_mm2,
                mean_intensity_in_mask=mean_int,
                mean_pressure=mean_p,
                peak_intensity=peak_int, peak_pressure=peak_p,
            ))

        areas = [f.area_mm2 for f in frames]
        peak_a_idx = int(np.argmax(areas))
        peak_a_f = frames[peak_a_idx]
        intensities = [f.mean_intensity_in_mask for f in frames]
        peak_i_idx = int(np.argmax(intensities))
        peak_i_f = frames[peak_i_idx]
        td = frames[0].frame
        lo = frames[-1].frame

        true_lo = td
        for f in frames:
            if f.mean_intensity_in_mask >= self.decay_thr:
                true_lo = f.frame

        dur = (true_lo - td) / self.fps
        max_int = max(intensities) if intensities else 0.0
        thr_wb = 0.8 * max_int
        n_wb = sum(1 for i in intensities if i >= thr_wb)
        load = (peak_i_f.frame - td) / self.fps
        unload = (true_lo - peak_i_f.frame) / self.fps
        wb = n_wb / self.fps
        max_area_curve = list(np.maximum.accumulate(areas))
        decay_phase_mask = [(f.frame in ext_set) for f in frames]

        decay_tau_ms = None
        decay_R2 = None
        if HAS_SCIPY and len(intensities) - peak_i_idx >= 3:
            try:
                y_fit = np.array(intensities[peak_i_idx:], dtype=float)
                x_fit = np.arange(len(y_fit), dtype=float)
                if y_fit[0] > y_fit[-1] and y_fit[0] > 0:
                    def model(t, A, tau, B):
                        return A * np.exp(-t / max(tau, 0.1)) + B
                    p0 = [max(y_fit[0] - y_fit[-1], 1.0), 5.0, max(y_fit[-1], 0.1)]
                    popt, _ = curve_fit(
                        model, x_fit, y_fit, p0=p0,
                        bounds=([0.0, 0.1, 0.0], [200.0, 30.0, 100.0]),
                        maxfev=2000,
                    )
                    A, tau, B = popt
                    if tau > 0 and A > 0:
                        decay_tau_ms = float(tau / self.fps * 1000.0)
                        y_pred = model(x_fit, *popt)
                        ss_res = float(np.sum((y_fit - y_pred) ** 2))
                        ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
                        decay_R2 = float(1.0 - ss_res / max(ss_tot, 1e-9)) if ss_tot > 0 else 0.0
            except Exception:
                pass

        is_clean = (decay_tau_ms is not None and decay_tau_ms < 50.0)

        return PawPrint(
            print_id=print_id,
            touchdown_frame=td,
            liftoff_frame=lo,
            true_liftoff_frame=true_lo,
            peak_area_frame=peak_a_f.frame,
            peak_intensity_frame=peak_i_f.frame,
            duration_s=float(dur),
            time_to_peak_area_s=(peak_a_f.frame - td) / self.fps,
            time_to_peak_intensity_s=(peak_i_f.frame - td) / self.fps,
            loading_duration_s=float(load),
            weight_bearing_duration_s=float(wb),
            unloading_duration_s=float(unload),
            frames=frames,
            max_area_mm2=float(max(areas)),
            peak_frame_centroid_xy_mm=peak_a_f.centroid_xy_mm,
            peak_frame_bbox_xyxy=peak_a_f.bbox_xyxy,
            peak_pressure=float(max(f.peak_pressure for f in frames)),
            mean_pressure_at_peak=float(peak_a_f.mean_pressure),
            raw_intensity_curve=[float(i) for i in intensities],
            max_area_curve=max_area_curve,
            decay_phase_mask=decay_phase_mask,
            decay_tau_ms=decay_tau_ms,
            decay_R2=decay_R2,
            is_clean_liftoff=is_clean,
            touchdown_intensity=float(intensities[0]),
            liftoff_intensity=float(intensities[-1]),
            quality=QualityFlags(n_frames=len(frames)),
        )


__all__ = ["PawPrintExtractor", "SENSITIVITY_PRESETS"]
