"""Pressure parameter calculation for FTIR gait analysis (CatWalk XT style).

Computes intensity-based metrics from per-frame paw green-channel patches.

References
----------
* Timotius et al. (2023) — CatWalk XT parameter review (Table 1)
* Hamers et al. (2001) — Stand Index definition
* kb/08_catwalk_metrics_spec.md — deepgait parameter spec
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Per-paw pressure patch extraction
# ---------------------------------------------------------------------------
def extract_paw_pressure_patch(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    green_threshold: int = 0,
) -> np.ndarray:
    """Extract the 2D green-channel patch for a paw from a BGR frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        BGR frame (H, W, 3).
    bbox : (x, y, w, h)
        Bounding box in pixel coordinates.
    green_threshold : int
        Minimum green value to keep (pixels below are set to 0).

    Returns
    -------
    2D uint8 array of shape (h, w) with green channel values.
    """
    x, y, w, h = bbox
    fh, fw = frame_bgr.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((max(h, 1), max(w, 1)), dtype=np.uint8)
    patch = frame_bgr[y0:y1, x0:x1, 1].copy().astype(np.uint8)
    if green_threshold > 0:
        patch[patch < green_threshold] = 0
    return patch


# ---------------------------------------------------------------------------
# Per-step pressure metrics
# ---------------------------------------------------------------------------
def compute_per_step_pressure(
    in_stance: np.ndarray,
    intensity_curve: np.ndarray,
    area_px_curve: np.ndarray,
    fps: float,
    px_per_mm: float = 1.0,
) -> List[Dict[str, float]]:
    """Compute per-stance-bout pressure metrics for one paw.

    Parameters
    ----------
    in_stance : 1D 0/1 array.
    intensity_curve : 1D float array (max green per frame in paw bbox).
    area_px_curve : 1D float array (paw area in px per frame).
    fps : frames per second.
    px_per_mm : pixel-to-mm conversion.

    Returns
    -------
    List of per-step dicts with keys:
        start_frame, end_frame, stance_frames, stand_s,
        max_intensity, mean_intensity, max_contact_area_cm2,
        print_area_cm2, stand_index.
    """
    from deepgait3.core._legacy.gait_ftir import compute_stance_segments
    segments = compute_stance_segments(in_stance)
    results: List[Dict[str, float]] = []
    for s, e in segments:
        seg_intensity = intensity_curve[s:e].astype(float)
        seg_area = area_px_curve[s:e].astype(float)
        if len(seg_intensity) == 0:
            continue

        max_i = float(np.max(seg_intensity))
        mean_i = float(np.mean(seg_intensity))
        max_area_px = float(np.max(seg_area))
        sum_area_px = float(np.sum(seg_area))
        stand_s = (e - s) / max(fps, 1.0)

        # Area conversions
        max_contact_cm2 = max_area_px / (px_per_mm ** 2) / 100.0
        print_area_cm2 = sum_area_px / fps / (px_per_mm ** 2) / 100.0

        # Stand Index (Hamers 2001)
        stand_index = mean_i * print_area_cm2 * stand_s

        results.append({
            "start_frame": s,
            "end_frame": e,
            "stance_frames": e - s,
            "stand_s": round(stand_s, 4),
            "max_intensity": round(max_i, 1),
            "mean_intensity": round(mean_i, 1),
            "max_contact_area_cm2": round(max_contact_cm2, 4),
            "print_area_cm2": round(print_area_cm2, 4),
            "stand_index": round(stand_index, 3),
        })
    return results


# ---------------------------------------------------------------------------
# Per-paw pressure aggregates
# ---------------------------------------------------------------------------
def compute_per_paw_pressure_aggregates(
    in_stance: np.ndarray,
    intensity_curve: np.ndarray,
    area_px_curve: np.ndarray,
    fps: float,
    px_per_mm: float = 1.0,
) -> Dict[str, float]:
    """Compute per-paw aggregated pressure metrics (CatWalk "Print" category).

    Returns flat dict with:
        max_contact_area_cm2, print_area_cm2, print_length_cm, print_width_cm,
        mean_intensity, max_intensity, max_contact_max_intensity,
        avg_stand_index, total_stand_index.
    """
    steps = compute_per_step_pressure(
        in_stance, intensity_curve, area_px_curve, fps, px_per_mm,
    )
    n = len(steps)
    if n == 0:
        return {
            "max_contact_area_cm2": 0.0, "print_area_cm2": 0.0,
            "print_length_cm": 0.0, "print_width_cm": 0.0,
            "mean_intensity": 0.0, "max_intensity": 0.0,
            "max_contact_max_intensity": 0.0,
            "avg_stand_index": 0.0, "total_stand_index": 0.0,
        }

    max_contact = max(s["max_contact_area_cm2"] for s in steps)
    total_print = sum(s["print_area_cm2"] for s in steps)
    mean_intensity = float(np.mean([s["mean_intensity"] for s in steps]))
    max_intensity = float(np.max([s["max_intensity"] for s in steps]))
    max_contact_max_i = float(np.max([
        s["max_intensity"] for s in steps
        if s["max_contact_area_cm2"] == max_contact
    ] or [0]))
    avg_stand_index = float(np.mean([s["stand_index"] for s in steps]))
    total_stand_index = float(np.sum([s["stand_index"] for s in steps]))

    return {
        "max_contact_area_cm2": round(max_contact, 4),
        "print_area_cm2": round(total_print, 4),
        "print_length_cm": 0.0,   # requires per-frame bbox tracking
        "print_width_cm": 0.0,    # requires per-frame bbox tracking
        "mean_intensity": round(mean_intensity, 1),
        "max_intensity": round(max_intensity, 1),
        "max_contact_max_intensity": round(max_contact_max_i, 1),
        "avg_stand_index": round(avg_stand_index, 3),
        "total_stand_index": round(total_stand_index, 3),
    }


# ---------------------------------------------------------------------------
# Per-frame pressure data (for CSV export + XYZ visualization)
# ---------------------------------------------------------------------------
def build_per_frame_pressure(
    n_frames: int,
    all_centroids_x: Dict[str, np.ndarray],
    all_centroids_y: Dict[str, np.ndarray],
    all_area_px: Dict[str, np.ndarray],
    all_intensity_max: Dict[str, np.ndarray],
    all_intensity_mean: Dict[str, np.ndarray],
    paw_names: Tuple[str, ...] = ("LF", "RF", "LH", "RH"),
) -> List[Dict[str, float]]:
    """Build per-frame pressure records for CSV export.

    Returns list of dicts, one per frame.
    """
    rows: List[Dict[str, float]] = []
    for fi in range(n_frames):
        row: Dict[str, float] = {"frame": fi}
        for paw in paw_names:
            row[f"{paw}_x"] = float(all_centroids_x.get(paw, [0])[fi]) if fi < len(all_centroids_x.get(paw, [])) else 0.0
            row[f"{paw}_y"] = float(all_centroids_y.get(paw, [0])[fi]) if fi < len(all_centroids_y.get(paw, [])) else 0.0
            row[f"{paw}_area_px"] = float(all_area_px.get(paw, [0])[fi]) if fi < len(all_area_px.get(paw, [])) else 0.0
            row[f"{paw}_intensity_max"] = float(all_intensity_max.get(paw, [0])[fi]) if fi < len(all_intensity_max.get(paw, [])) else 0.0
            row[f"{paw}_intensity_mean"] = float(all_intensity_mean.get(paw, [0])[fi]) if fi < len(all_intensity_mean.get(paw, [])) else 0.0
        rows.append(row)
    return rows
