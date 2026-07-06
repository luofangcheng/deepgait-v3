"""Build FootprintCycle objects from IoU-tracked FootMask sequences.

Each track produced by ``IoUFootprintTracker`` represents a candidate
print.  This module:
  1. Filters tracks shorter than *min_frames*.
  2. Extracts per-frame scalars (area, intensity, centroid, …).
  3. Identifies peak-area / peak-intensity frames.
  4. Produces a ``FootprintCycle`` per track.
  5. Sorts cycles by touchdown frame and assigns sequential ``cycle_id``.
"""
from __future__ import annotations
from typing import List

import numpy as np

from .models import FootprintCycle, FrameRecord


# 80 % of peak intensity → weight-bearing threshold
_WB_THRESH = 0.8


def build_cycles(
    tracks,
    *,
    fps: float = 60.0,
    px_per_mm: float = 1.92,
    min_frames: int = 2,
    decay_thr: float = 6.0,
) -> List[FootprintCycle]:
    """Convert tracked footprint sequences into FootprintCycle list.

    Parameters
    ----------
    tracks : list of FootprintTrack
    fps : frame rate
    px_per_mm : pixels-per-mm calibration
    min_frames : minimum number of frames to keep a track
    decay_thr : intensity threshold used for true-liftoff estimation
    """
    cycles: List[FootprintCycle] = []

    for tr in tracks:
        if len(tr.foots) < min_frames:
            continue

        frames: List[FrameRecord] = []
        areas: List[float] = []
        intensities: List[float] = []

        for fi, fm in tr.foots:
            mask = fm.mask_padded
            delta = np.maximum(fm.raw_intensity_crop, 0.0)
            pmap = fm.pressure_map
            area_mm2 = float(fm.total_area_px) / (px_per_mm ** 2)

            if mask.any():
                mean_int = float(delta[mask].mean())
                mean_p = float(pmap[mask].mean())
                peak_int = float(delta.max())
                peak_p = float(pmap.max())
            else:
                mean_int = mean_p = peak_int = peak_p = 0.0

            areas.append(area_mm2)
            intensities.append(mean_int)

            px1, py1, px2, py2 = fm.bbox_xyxy
            cx_mm = fm.centroid_px[0] / px_per_mm
            cy_mm = fm.centroid_px[1] / px_per_mm

            frames.append(FrameRecord(
                frame=fi,
                time_s=fi / fps,
                area_mm2=area_mm2,
                area_px=fm.total_area_px,
                centroid_x_mm=cx_mm,
                centroid_y_mm=cy_mm,
                bbox_x1=px1, bbox_y1=py1, bbox_x2=px2, bbox_y2=py2,
                mean_intensity=mean_int,
                peak_intensity=peak_int,
                mean_pressure=mean_p,
                peak_pressure=peak_p,
                is_peak_area=False,
                is_peak_intensity=False,
            ))

        # Identify peaks
        if not frames:
            continue

        peak_a_idx = int(np.argmax(areas))
        frames[peak_a_idx].is_peak_area = True
        peak_i_idx = int(np.argmax(intensities))
        frames[peak_i_idx].is_peak_intensity = True

        td = frames[0].frame
        lo = frames[-1].frame

        # True liftoff — last frame where intensity is still above decay_thr
        true_lo = td
        for f in frames:
            if f.mean_intensity >= decay_thr:
                true_lo = f.frame

        dur = (true_lo - td) / fps
        max_int = max(intensities) if intensities else 0.0
        wb_thr = _WB_THRESH * max_int
        n_wb = sum(1 for v in intensities if v >= wb_thr)
        loading = (frames[peak_i_idx].frame - td) / fps
        unloading = (true_lo - frames[peak_i_idx].frame) / fps
        wb = n_wb / fps

        peak_a = frames[peak_a_idx]
        cycles.append(FootprintCycle(
            touchdown_frame=td,
            liftoff_frame=lo,
            peak_area_frame=peak_a.frame,
            peak_intensity_frame=frames[peak_i_idx].frame,
            duration_s=dur,
            max_area_mm2=max(areas),
            max_area_px=int(max(fm.total_area_px for _, fm in tr.foots)),
            centroid_at_peak_x_mm=peak_a.centroid_x_mm,
            centroid_at_peak_y_mm=peak_a.centroid_y_mm,
            bbox_at_peak_xyxy=(peak_a.bbox_x1, peak_a.bbox_y1,
                               peak_a.bbox_x2, peak_a.bbox_y2),
            loading_duration_s=loading,
            weight_bearing_duration_s=wb,
            unloading_duration_s=unloading,
            touchdown_intensity=float(intensities[0]),
            liftoff_intensity=float(intensities[-1]),
            is_clean_liftoff=(intensities[-1] < 3.0),
            n_frames=len(frames),
            frames=frames,
        ))

    # Sort by touchdown, assign sequential ids
    cycles.sort(key=lambda c: c.touchdown_frame)
    for i, c in enumerate(cycles, start=1):
        c.cycle_id = i

    return cycles


__all__ = ["build_cycles"]
