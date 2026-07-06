"""FTIR 脚印 → gait metrics pipeline (CatWalk XT / MouseWalker style).

v2.1 — red-background pipeline with centroid-based speed estimation,
cross-paw dual stance timing, and coordination metrics.

Key functions
-------------
* :func:`build_temporal_stance` — convert frame-sequence footprints
  into per-paw 1D stance arrays.
* :func:`compute_stance_segments` — extract contiguous stance bouts per paw.
* :func:`compute_per_step_metrics` — per-step stride length, speed,
  swing speed, braking/propulsion indices.
* :func:`compute_per_paw_aggregates` — CatWalk "单爪步态数据" aggregated stats.
* :func:`compute_dual_stance` — cross-paw dual/triple stance timing.
* :func:`compute_coordination` — homologous / ipsilateral / diagonal
  coordination percentages.
* :func:`compute_catwalk_equivalent_metrics` — top-level: CatWalk XT
  "单爪步态数据" + "多爪统计数据" in a flat dict.
* :func:`compute_ftir_gait_metrics` — legacy wrapper (backward compatible).

All functions are pure (no side effects) and callable from offline
(video file) or online (live camera C1) paths.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Sequence


# ---------------------------------------------------------------------------
# 1. Temporal stance building
# ---------------------------------------------------------------------------
def build_temporal_stance(
    sequences: "Sequence[object]",
    n_frames: int | None = None,
    paw_names: Sequence[str] = ("LF", "RF", "LH", "RH"),
) -> Dict[str, np.ndarray]:
    """Convert a list of ``FootprintSequence`` into per-paw stance arrays.

    For each frame, if ``seq.feet[paw].is_in_stance`` is True → 1,
    else → 0.  Missing paws default to 0.
    """
    if n_frames is None:
        n_frames = len(sequences)
    out: Dict[str, np.ndarray] = {
        p: np.zeros(n_frames, dtype=np.int8) for p in paw_names
    }
    for i, seq in enumerate(sequences[:n_frames]):
        if seq is None or getattr(seq, "feet", None) is None:
            continue
        for paw in paw_names:
            foot = seq.feet.get(paw)
            if foot is not None and getattr(foot, "is_in_stance", False):
                out[paw][i] = 1
    return out


# ---------------------------------------------------------------------------
# 2. Stance segments extraction
# ---------------------------------------------------------------------------
def compute_stance_segments(
    in_stance: np.ndarray,
) -> List[Tuple[int, int]]:
    """Extract contiguous stance bout (start, end) frame indices.

    Returns list of (start_frame, end_frame_exclusive).
    """
    if len(in_stance) < 2:
        return []
    d = np.diff(np.concatenate([[0], in_stance.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
# 3. Per-step metrics (stride length, speed, swing speed, etc.)
# ---------------------------------------------------------------------------
def compute_per_step_metrics(
    in_stance: np.ndarray,
    intensity: np.ndarray | None,
    centroids_x: np.ndarray | None,
    fps: float,
    px_per_mm: float = 1.0,
) -> List[Dict[str, float]]:
    """Compute per-step (per-stance-bout) CatWalk metrics for one paw.

    Parameters
    ----------
    in_stance : 1D 0/1 array of length N.
    intensity : 1D float array (green channel intensity = pressure).
    centroids_x : 1D float array (X-coordinate of paw centroid).
    fps : frames per second.
    px_per_mm : pixel-to-mm conversion.

    Returns
    -------
    List of per-step dicts with keys:
        start_frame, end_frame, stance_frames, stand_s,
        stride_length_mm, step_length_mm, swing_s, step_cycle_s,
        instantaneous_speed_mm_s, swing_speed_mm_s,
        max_intensity, mean_intensity, area_under_curve,
        braking_s, propulsion_s, braking_index, propulsion_index.
    """
    segments = compute_stance_segments(in_stance)
    if len(segments) < 1:
        return []

    results: List[Dict[str, float]] = []
    for i_seg, (s, e) in enumerate(segments):
        step: Dict[str, float] = {
            "start_frame": s,
            "end_frame": e,
            "stance_frames": e - s,
            "stand_s": (e - s) / max(fps, 1.0),
        }

        # ---- stride / step length from centroid X ----
        if centroids_x is not None and len(centroids_x) > s:
            # Step length = distance to next stance of same paw (stride).
            if i_seg + 1 < len(segments):
                next_start = segments[i_seg + 1][0]
                if next_start < len(centroids_x):
                    dx = abs(centroids_x[next_start] - centroids_x[s])
                    step["stride_length_mm"] = round(dx / px_per_mm, 3)
                else:
                    step["stride_length_mm"] = 0.0
            else:
                step["stride_length_mm"] = 0.0

            # Step length (for display) = same as stride for per-paw.
            step["step_length_mm"] = step["stride_length_mm"]
        else:
            step["step_length_mm"] = 0.0
            step["stride_length_mm"] = 0.0

        # ---- swing duration (gap to next stance) ----
        if i_seg + 1 < len(segments):
            next_start = segments[i_seg + 1][0]
            swing_frames = next_start - e
            step["swing_frames"] = swing_frames
            step["swing_s"] = swing_frames / max(fps, 1.0)
        else:
            step["swing_frames"] = 0
            step["swing_s"] = 0.0

        # ---- step cycle ----
        step["step_cycle_s"] = step["stand_s"] + step["swing_s"]

        # ---- speeds ----
        if step["swing_s"] > 0 and step["stride_length_mm"] > 0:
            step["swing_speed_mm_s"] = round(
                step["stride_length_mm"] / step["swing_s"], 1,
            )
        else:
            step["swing_speed_mm_s"] = 0.0

        if step["step_cycle_s"] > 0 and step["step_length_mm"] > 0:
            step["instantaneous_speed_mm_s"] = round(
                step["step_length_mm"] / step["step_cycle_s"], 1,
            )
        else:
            step["instantaneous_speed_mm_s"] = 0.0

        # ---- intensity / braking / propulsion ----
        if intensity is not None and len(intensity) > e:
            seg_intensity = intensity[s:e].astype(float)
            step["max_intensity"] = float(np.max(seg_intensity)) if len(seg_intensity) else 0.0
            step["mean_intensity"] = float(np.mean(seg_intensity)) if len(seg_intensity) else 0.0
            step["area_under_curve"] = float(np.sum(seg_intensity)) / fps

            # Braking / propulsion: split stance at peak intensity.
            if len(seg_intensity) >= 2:
                peak_rel = int(np.argmax(seg_intensity))
                braking_frames = max(peak_rel, 0)
                propulsion_frames = max(len(seg_intensity) - peak_rel - 1, 0)
                step["braking_s"] = braking_frames / max(fps, 1.0)
                step["propulsion_s"] = propulsion_frames / max(fps, 1.0)
                total_phase = step["braking_s"] + step["propulsion_s"]
                if total_phase > 0:
                    step["braking_index"] = round(
                        step["braking_s"] / total_phase, 3,
                    )
                    step["propulsion_index"] = round(
                        step["propulsion_s"] / total_phase, 3,
                    )
                else:
                    step["braking_index"] = 0.0
                    step["propulsion_index"] = 0.0
            else:
                step["braking_s"] = 0.0
                step["propulsion_s"] = 0.0
                step["braking_index"] = 0.0
                step["propulsion_index"] = 0.0
        else:
            step["max_intensity"] = 0.0
            step["mean_intensity"] = 0.0
            step["area_under_curve"] = 0.0
            step["braking_s"] = 0.0
            step["propulsion_s"] = 0.0
            step["braking_index"] = 0.0
            step["propulsion_index"] = 0.0

        results.append(step)

    return results


# ---------------------------------------------------------------------------
# 4. Per-paw aggregate metrics (CatWalk "单爪步态数据")
# ---------------------------------------------------------------------------
def compute_per_paw_aggregates(
    in_stance: np.ndarray,
    intensity: np.ndarray | None,
    centroids_x: np.ndarray | None,
    area_px_curve: np.ndarray | None,
    fps: float,
    px_per_mm: float = 1.0,
) -> Dict[str, float]:
    """Compute CatWalk-style per-paw aggregated statistics.

    Returns a flat dict matching the "单爪步态数据" structure.
    """
    steps = compute_per_step_metrics(
        in_stance, intensity, centroids_x, fps, px_per_mm,
    )
    n_steps = len(steps)
    total_frames = len(in_stance)
    total_time = total_frames / max(fps, 1.0)

    if n_steps == 0:
        return _empty_paw_aggregates()

    # ---- timing aggregates ----
    stand_vals = [s["stand_s"] for s in steps]
    swing_vals = [s["swing_s"] for s in steps if s["swing_s"] > 0]
    cycle_vals = [s["step_cycle_s"] for s in steps if s["step_cycle_s"] > 0]

    avg_stand = float(np.mean(stand_vals)) if stand_vals else 0.0
    avg_swing = float(np.mean(swing_vals)) if swing_vals else 0.0
    avg_cycle = float(np.mean(cycle_vals)) if cycle_vals else 0.0

    # ---- stride / step length ----
    stride_vals = [s["stride_length_mm"] for s in steps if s["stride_length_mm"] > 0]
    step_vals = [s["step_length_mm"] for s in steps if s["step_length_mm"] > 0]

    avg_stride = float(np.mean(stride_vals)) if stride_vals else 0.0
    avg_step = float(np.mean(step_vals)) if step_vals else 0.0

    # ---- speed ----
    swing_speed_vals = [s["swing_speed_mm_s"] for s in steps if s["swing_speed_mm_s"] > 0]
    inst_speed_vals = [s["instantaneous_speed_mm_s"] for s in steps if s["instantaneous_speed_mm_s"] > 0]

    avg_swing_speed = float(np.mean(swing_speed_vals)) if swing_speed_vals else 0.0
    avg_inst_speed = float(np.mean(inst_speed_vals)) if inst_speed_vals else 0.0
    peak_swing_speed = float(np.max(swing_speed_vals)) if swing_speed_vals else 0.0
    # Total speed = total stride displacement / total time
    total_stride = sum(stride_vals)
    total_speed = total_stride / total_time if total_time > 0 else 0.0

    # ---- intensity ----
    intensity_vals = [s["max_intensity"] for s in steps if s["max_intensity"] > 0]
    mean_intensity_vals = [s["mean_intensity"] for s in steps if s["mean_intensity"] > 0]
    auc_vals = [s["area_under_curve"] for s in steps]

    max_intensity = float(np.max(intensity_vals)) if intensity_vals else 0.0
    avg_max_intensity = float(np.mean(intensity_vals)) if intensity_vals else 0.0

    # ---- braking / propulsion ----
    brake_vals = [s["braking_s"] for s in steps]
    prop_vals = [s["propulsion_s"] for s in steps]
    brake_idx_vals = [s["braking_index"] for s in steps]
    prop_idx_vals = [s["propulsion_index"] for s in steps]

    avg_brake = float(np.mean(brake_vals)) if brake_vals else 0.0
    avg_prop = float(np.mean(prop_vals)) if prop_vals else 0.0
    avg_brake_idx = float(np.mean(brake_idx_vals)) if brake_idx_vals else 0.0
    avg_prop_idx = float(np.mean(prop_idx_vals)) if prop_idx_vals else 0.0

    # ---- area (from bbox if available) ----
    if area_px_curve is not None and len(area_px_curve) > 0:
        stance_mask = in_stance > 0
        if stance_mask.any():
            max_area_px = float(np.max(area_px_curve[stance_mask]))
            avg_area_px = float(np.mean(area_px_curve[stance_mask]))
            # Convert to cm² (approximation: px_per_mm² → cm²)
            max_area_cm2 = max_area_px / (px_per_mm ** 2) / 100.0
            avg_area_cm2 = avg_area_px / (px_per_mm ** 2) / 100.0
        else:
            max_area_cm2 = 0.0
            avg_area_cm2 = 0.0
    else:
        max_area_cm2 = 0.0
        avg_area_cm2 = 0.0

    # ---- contact time ----
    total_contact_s = sum(stand_vals)
    contact_pct = (total_contact_s / total_time * 100.0) if total_time > 0 else 0.0
    max_contact_s = max(stand_vals) if stand_vals else 0.0

    # ---- duty cycle ----
    duty_cycle_pct = (avg_stand / avg_cycle * 100.0) if avg_cycle > 0 else 0.0
    swing_phase_pct = (avg_swing / avg_cycle * 100.0) if avg_cycle > 0 else 0.0

    return {
        # Contact
        "max_contact_area_cm2": round(max_area_cm2, 4),
        "avg_contact_area_cm2": round(avg_area_cm2, 4),
        "total_contact_s": round(total_contact_s, 4),
        "max_contact_s": round(max_contact_s, 4),
        "contact_time_pct": round(contact_pct, 1),
        # Step cycle
        "step_cycle_s": round(avg_cycle, 4),
        "stride_length_cm": round(avg_stride / 10.0, 3),
        "step_length_cm": round(avg_step / 10.0, 3),
        # Stance / swing
        "avg_stance_s": round(avg_stand, 4),
        "avg_swing_s": round(avg_swing, 4),
        "duty_cycle_pct": round(duty_cycle_pct, 1),
        "swing_phase_pct": round(swing_phase_pct, 1),
        # Braking / propulsion
        "avg_braking_s": round(avg_brake, 4),
        "avg_propulsion_s": round(avg_prop, 4),
        "braking_index": round(avg_brake_idx, 3),
        "propulsion_index": round(avg_prop_idx, 3),
        # Speed
        "avg_swing_speed_cm_s": round(avg_swing_speed / 10.0, 1),
        "avg_instantaneous_speed_cm_s": round(avg_inst_speed / 10.0, 1),
        "total_speed_cm_s": round(total_speed / 10.0, 1),
        "peak_swing_speed_cm_s": round(peak_swing_speed / 10.0, 1),
        # Intensity
        "max_intensity": round(max_intensity, 0),
        "avg_max_intensity": round(avg_max_intensity, 0),
        "total_auc": round(float(np.sum(auc_vals)), 2),
        # Counts
        "n_steps": n_steps,
    }


def _empty_paw_aggregates() -> Dict[str, float]:
    return {k: 0.0 for k in [
        "max_contact_area_cm2", "avg_contact_area_cm2",
        "total_contact_s", "max_contact_s", "contact_time_pct",
        "step_cycle_s", "stride_length_cm", "step_length_cm",
        "avg_stance_s", "avg_swing_s", "duty_cycle_pct", "swing_phase_pct",
        "avg_braking_s", "avg_propulsion_s", "braking_index", "propulsion_index",
        "avg_swing_speed_cm_s", "avg_instantaneous_speed_cm_s",
        "total_speed_cm_s", "peak_swing_speed_cm_s",
        "max_intensity", "avg_max_intensity", "total_auc",
    ]} | {"n_steps": 0}


# ---------------------------------------------------------------------------
# 5. Dual / Triple stance timing (cross-paw)
# ---------------------------------------------------------------------------
def compute_dual_stance(
    in_stance: Dict[str, np.ndarray],
    fps: float,
) -> Dict[str, float]:
    """Compute dual- and triple-stance timing percentages.

    Returns dict with:
        dual_LF_LH_pct, dual_RF_RH_pct, dual_LF_RH_pct, dual_RF_LH_pct,
        triple_LF_LH_RF_pct, triple_LF_LH_RH_pct, etc.
    """
    n = min(len(v) for v in in_stance.values()) if in_stance else 0
    if n == 0:
        return {}

    # Build per-frame stance bitmask.
    paws = list(in_stance.keys())
    mask = np.zeros(n, dtype=np.int32)
    for i, paw in enumerate(paws):
        mask += ((in_stance[paw] > 0).astype(np.int32) << i)

    total_active = int((mask > 0).sum())
    if total_active == 0:
        return {}

    # Helper: count frames where specific paw bits are set.
    def _pct(bit_pattern: int) -> float:
        hit = (mask.astype(np.int32) & bit_pattern) == bit_pattern
        count = int(hit.sum())
        return round(100.0 * count / total_active, 1)

    # Map paw names to bit positions.
    idx = {p: i for i, p in enumerate(paws)}
    result: Dict[str, float] = {}

    # Dual stance pairs.
    for (a, b, label) in [
        ("LF", "LH", "dual_LF_LH_pct"),
        ("RF", "RH", "dual_RF_RH_pct"),
        ("LF", "RH", "dual_LF_RH_pct"),
        ("RF", "LH", "dual_RF_LH_pct"),
    ]:
        if a in idx and b in idx:
            pattern = (1 << idx[a]) | (1 << idx[b])
            result[label] = _pct(pattern)

    # Triple stance.
    for (a, b, c, label) in [
        ("LF", "LH", "RF", "triple_LF_LH_RF_pct"),
        ("LF", "LH", "RH", "triple_LF_LH_RH_pct"),
        ("RF", "RH", "LF", "triple_RF_RH_LF_pct"),
        ("RF", "RH", "LH", "triple_RF_RH_LH_pct"),
    ]:
        if all(p in idx for p in (a, b, c)):
            pattern = (1 << idx[a]) | (1 << idx[b]) | (1 << idx[c])
            result[label] = _pct(pattern)

    result["total_active_frames"] = total_active
    return result


# ---------------------------------------------------------------------------
# 6. Coordination metrics (homologous / ipsilateral / diagonal)
# ---------------------------------------------------------------------------
def compute_coordination(
    in_stance: Dict[str, np.ndarray],
    fps: float,
) -> Dict[str, float]:
    """Compute CatWalk coordination percentages.

    For each pair of paws, measure the fraction of stance onsets
    of paw A that occur while paw B is also in stance.

    Returns dict with:
        homologous_stance_LH_RH_pct, ipsilateral_stance_LH_LF_pct,
        ipsilateral_stance_RH_RF_pct, diagonal_stance_LH_RF_pct,
        diagonal_stance_RH_LF_pct, and swing counterparts.
    """
    result: Dict[str, float] = {}
    if not in_stance:
        return result

    def _coordination(a: np.ndarray, b: np.ndarray) -> float:
        """% of a's stance onsets where b is also in stance."""
        a_onsets = np.where(np.diff(np.concatenate(([0], a))) == 1)[0]
        if len(a_onsets) == 0:
            return 0.0
        b_in_stance = b > 0
        count = sum(1 for t in a_onsets if t < len(b_in_stance) and b_in_stance[t])
        return round(100.0 * count / len(a_onsets), 1)

    def _swing_coordination(a: np.ndarray, b: np.ndarray) -> float:
        """% of a's swing onsets where b is also in swing."""
        a_onsets = np.where(np.diff(np.concatenate(([0], 1 - a))) == 1)[0]
        if len(a_onsets) == 0:
            return 0.0
        b_in_swing = ~(b > 0)
        count = sum(1 for t in a_onsets if t < len(b_in_swing) and b_in_swing[t])
        return round(100.0 * count / len(a_onsets), 1)

    pairs = [
        ("LH", "RH", "homologous"),
        ("LH", "LF", "ipsilateral_LH_LF"),
        ("RH", "RF", "ipsilateral_RH_RF"),
        ("LH", "RF", "diagonal_LH_RF"),
        ("RH", "LF", "diagonal_RH_LF"),
    ]

    for a, b, label in pairs:
        if a in in_stance and b in in_stance:
            arr_a = in_stance[a]
            arr_b = in_stance[b]
            result[f"{label}_stance_pct"] = _coordination(arr_a, arr_b)
            result[f"{label}_swing_pct"] = _swing_coordination(arr_a, arr_b)

    return result


# ---------------------------------------------------------------------------
# 7. Body speed from centroid movement
# ---------------------------------------------------------------------------
def compute_body_speed(
    centroids_x: Dict[str, np.ndarray] | None,
    fps: float,
    px_per_mm: float = 1.0,
) -> Dict[str, float]:
    """Estimate body speed from paw centroid X-coordinate drift.

    Uses the median X displacement across all paws as a proxy for
    body centre-of-mass movement.
    """
    if centroids_x is None or not centroids_x:
        return {"body_speed_mm_s": 0.0, "body_speed_cm_s": 0.0}

    speeds = []
    for paw, cx in centroids_x.items():
        if len(cx) < 2:
            continue
        valid = cx[cx > 0]
        if len(valid) < 2:
            continue
        total_dx = abs(float(valid[-1] - valid[0]))
        duration = (len(cx) - 1) / max(fps, 1.0)
        if duration > 0:
            speeds.append(total_dx / px_per_mm / duration)

    if speeds:
        body_speed_mm_s = float(np.median(speeds))
    else:
        body_speed_mm_s = 0.0

    return {
        "body_speed_mm_s": round(body_speed_mm_s, 1),
        "body_speed_cm_s": round(body_speed_mm_s / 10.0, 1),
    }


# ---------------------------------------------------------------------------
# 8. CatWalk-equivalent top-level metrics
# ---------------------------------------------------------------------------
def compute_catwalk_equivalent_metrics(
    in_stance: Dict[str, np.ndarray],
    intensity_curves: Dict[str, np.ndarray | None],
    centroids_x: Dict[str, np.ndarray | None] | None = None,
    area_px_curves: Dict[str, np.ndarray | None] | None = None,
    fps: float = 100.0,
    px_per_mm: float = 1.0,
    paw_names: Sequence[str] = ("LF", "RF", "LH", "RH"),
) -> Dict[str, float]:
    """Compute CatWalk XT-equivalent gait metrics from FTIR stance data.

    Output matches the structure of CatWalk's "单爪步态数据" + "多爪统计数据".

    Parameters
    ----------
    in_stance : {paw: 0/1 array}
    intensity_curves : {paw: float array or None}
    centroids_x : {paw: float array or None}, optional
        X-coordinate of paw centroid per frame.
    area_px_curves : {paw: float array or None}, optional
        Paw area in pixels per frame.
    fps : float
    px_per_mm : float
        Pixel-to-mm conversion factor.
    paw_names : sequence of paw labels

    Returns
    -------
    Flat dict with keys:
        {paw}_{metric} — per-paw aggregates
        body_speed_mm_s, body_speed_cm_s
        dual_*, triple_* — stance timing
        *_{stance,swing}_pct — coordination
    """
    result: Dict[str, float] = {}

    # ---- Per-paw aggregates ----
    for paw in paw_names:
        arr = in_stance.get(paw)
        if arr is None or len(arr) < 3:
            per = _empty_paw_aggregates()
        else:
            intensity = intensity_curves.get(paw) if intensity_curves else None
            cx = centroids_x.get(paw) if centroids_x else None
            area_px = area_px_curves.get(paw) if area_px_curves else None
            per = compute_per_paw_aggregates(
                arr, intensity, cx, area_px, fps, px_per_mm,
            )
        for k, v in per.items():
            result[f"{paw}_{k}"] = v

    # ---- Body speed ----
    if centroids_x is not None:
        result.update(compute_body_speed(centroids_x, fps, px_per_mm))

    # ---- Dual / triple stance ----
    result.update(compute_dual_stance(in_stance, fps))

    # ---- Coordination ----
    result.update(compute_coordination(in_stance, fps))

    # ---- Run-level summary ----
    total_frames = max(len(v) for v in in_stance.values()) if in_stance else 0
    result["run_duration_s"] = round(total_frames / max(fps, 1.0), 3)
    n_steps = sum(v.get("n_steps", 0) for v in [
        compute_per_paw_aggregates(
            in_stance.get(p), intensity_curves.get(p) if intensity_curves else None,
            centroids_x.get(p) if centroids_x else None,
            area_px_curves.get(p) if area_px_curves else None,
            fps, px_per_mm,
        ) for p in paw_names
    ])
    result["n_steps"] = n_steps

    return result


# ---------------------------------------------------------------------------
# 9. Legacy: compute_ftir_gait_metrics (backward compatible)
# ---------------------------------------------------------------------------
def compute_ftir_gait_metrics(
    in_stance: Dict[str, np.ndarray],
    intensity_curves: Dict[str, np.ndarray | None],
    fps: float,
    walkway_length_mm: float = 400.0,
    paw_names: Sequence[str] = ("LF", "RF", "LH", "RH"),
) -> Dict[str, float]:
    """Legacy wrapper — delegates to the new CatWalk-equivalent pipeline.

    Retained for backward compatibility with existing callers.
    """
    return compute_catwalk_equivalent_metrics(
        in_stance, intensity_curves,
        centroids_x=None, area_px_curves=None,
        fps=fps,
        paw_names=paw_names,
    )


# ---------------------------------------------------------------------------
# 10. Real-time in_stance update (incremental)
# ---------------------------------------------------------------------------
def update_temporal_stance(
    existing: Dict[str, np.ndarray],
    frame_sequence: object,
    frame_idx: int,
    paw_names: Sequence[str] = ("LF", "RF", "LH", "RH"),
) -> Dict[str, np.ndarray]:
    """Extend per-paw stance arrays with one new frame."""
    for paw in paw_names:
        foot = None
        feet = getattr(frame_sequence, "feet", None) or {}
        if feet:
            foot = feet.get(paw)
        val = int(foot is not None and getattr(foot, "is_in_stance", False))
        existing[paw][frame_idx] = val
    return existing


# ---------------------------------------------------------------------------
# 11. Run detection helpers
# ---------------------------------------------------------------------------
def detect_run_boundaries(
    in_stance: Dict[str, np.ndarray],
    min_frames_between_runs: int = 30,
) -> Tuple[int, int]:
    """Detect start/end frame indices of a run."""
    any_stance = np.zeros_like(next(iter(in_stance.values())), dtype=bool)
    for arr in in_stance.values():
        any_stance |= (arr > 0)
    active = np.where(any_stance)[0]
    if len(active) == 0:
        return 0, 0
    return int(active[0]), int(active[-1]) + 1
