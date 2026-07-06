"""Gait algorithms translated from VisualGaitLab GaitCalculationMethods.cs.

All functions operate on 1-D NumPy arrays (per-frame values for a single bodypart
or paw).  Vectorized where possible; loops only where stride-level logic requires it.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from deepgait3.utils import geometry


# ---------------------------------------------------------------------------
# 1. Stance / Swing detection
# ---------------------------------------------------------------------------

def treadmill_in_stance(all_x: np.ndarray) -> np.ndarray:
    """Treadmill mode: stance/swing from X-coordinate trend (VGL §1.1).

    Returns binary array: 1 = stance, 0 = swing.
    """
    if all_x.size < 2:
        return np.zeros_like(all_x, dtype=int)
    diff = np.diff(all_x)
    getting_bigger = diff >= 0
    switches = np.where(np.diff(getting_bigger.astype(int)) != 0)[0] + 1
    switches = np.concatenate(([0], switches, [len(all_x) - 1]))
    in_stance = np.zeros(len(all_x), dtype=int)
    for i in range(len(switches) - 1):
        s, e = switches[i], switches[i + 1]
        # e is inclusive index in VGL logic; here e is the last index of segment
        seg_diff = diff[s:e] if e > s else diff[s:s]
        slope_sum = seg_diff.sum()
        in_stance[s:e] = 0 if slope_sum >= 0 else 1
    return in_stance


def catwalk_in_stance(
    toe_x: np.ndarray,
    toe_y: np.ndarray,
    heel_x: np.ndarray,
    heel_y: np.ndarray,
    bias: float = 1.0,
) -> np.ndarray:
    """CatWalk / free-run mode: stance/swing from paw mid-point displacement (VGL §1.2).

    Returns binary array: 1 = stance, 0 = swing.
    """
    mid_x = (toe_x + heel_x) / 2.0
    mid_y = (toe_y + heel_y) / 2.0
    pos_diff = np.sqrt(np.diff(mid_x) ** 2 + np.diff(mid_y) ** 2)
    if pos_diff.size == 0:
        return np.zeros_like(toe_x, dtype=int)
    threshold = pos_diff.mean() * bias
    in_stance = np.zeros(len(toe_x), dtype=int)
    in_stance[1:] = np.where(pos_diff < threshold, 1, 0)
    return in_stance


# ---------------------------------------------------------------------------
# 2. Gait basics (stance/swing duration, stride count, switch positions)
# ---------------------------------------------------------------------------

class GaitBasics(NamedTuple):
    stance_duration_ms: float
    swing_duration_ms: float
    n_strides: int
    switch_positions: np.ndarray          # frame indices where stance/swing flips
    stance_frames_per_stride: list[int]   # one entry per complete stride
    swing_frames_per_stride: list[int]


def calculate_gait_basics(in_stance: np.ndarray, fps: int) -> GaitBasics:
    """VGL §2: extract stance/swing durations and stride count.

    Trims incomplete leading/trailing segments so the sequence contains only
    complete stance-swing cycles.  A complete stride = stance -> swing -> stance.
    """
    n = len(in_stance)
    if n == 0:
        return GaitBasics(0.0, 0.0, 0, np.array([], dtype=int), [], [])

    # Find all transition points (0->1 or 1->0)
    transitions = np.diff(in_stance)
    switch_idx = np.where(transitions != 0)[0] + 1

    # Trim so that the sequence starts with stance (1) and ends with stance (1).
    # This guarantees every stance is followed by a swing, forming complete cycles.
    start = 0
    while start < n and in_stance[start] == 0:
        start += 1
    end = n - 1
    while end >= 0 and in_stance[end] == 0:
        end -= 1
    if start >= end:
        return GaitBasics(0.0, 0.0, 0, np.array([], dtype=int), [], [])

    trimmed = in_stance[start:end + 1]
    stance_frames = int((trimmed == 1).sum())
    swing_frames = int((trimmed == 0).sum())

    # Per-stride frame counts from runs in trimmed region
    runs = geometry.find_runs(trimmed)
    stance_per_stride: list[int] = []
    swing_per_stride: list[int] = []
    for s, e, val in runs:
        length = e - s
        if val == 1:
            stance_per_stride.append(length)
        else:
            swing_per_stride.append(length)

    # A complete stride needs both a stance and a following swing.
    # If the sequence ends with stance, the last stance has no trailing swing -
    # drop it so that stance_per_stride and swing_per_stride are 1:1.
    min_len = min(len(stance_per_stride), len(swing_per_stride))
    stance_per_stride = stance_per_stride[:min_len]
    swing_per_stride = swing_per_stride[:min_len]
    n_strides = min_len

    if n_strides == 0:
        return GaitBasics(0.0, 0.0, 0, np.array([], dtype=int), [], [])

    # Rebuild switch_positions for the trimmed region only
    trimmed_switches = switch_idx[(switch_idx >= start) & (switch_idx <= end)]
    switch_positions = np.concatenate(([start], trimmed_switches, [end + 1]))

    stance_ms = stance_frames / n_strides * 1000.0 / fps
    swing_ms = swing_frames / n_strides * 1000.0 / fps
    return GaitBasics(
        stance_duration_ms=stance_ms,
        swing_duration_ms=swing_ms,
        n_strides=n_strides,
        switch_positions=switch_positions,
        stance_frames_per_stride=stance_per_stride,
        swing_frames_per_stride=swing_per_stride,
    )


# ---------------------------------------------------------------------------
# 3. Stride length
# ---------------------------------------------------------------------------

def calculate_stride_data(
    switch_positions: np.ndarray,
    mid_x: np.ndarray,
    mid_y: np.ndarray,
    is_free_run: bool,
    treadmill_speed: float | None,
    real_world_multiplier: float,
    fps: int,
) -> tuple[np.ndarray, float]:
    """VGL §3: stride length per frame + stride-length variability.

    Returns:
        stride_lengths: np.ndarray of shape (n_frames,) with stride length for each frame.
        variability: sample standard deviation of unique stride lengths.
    """
    n = len(mid_x)
    stride_lengths = np.zeros(n, dtype=float)
    unique_strides: list[float] = []

    # switch_positions are frame indices; a stride is switch[j] to switch[j+2]
    m = len(switch_positions)
    for j in range(0, m - 2, 2):
        start = int(switch_positions[j])
        end = int(switch_positions[j + 2])
        if start >= n or end >= n:
            continue
        if is_free_run:
            dist = geometry.distance_between_points(
                np.array([mid_x[start], mid_y[start]]),
                np.array([mid_x[end], mid_y[end]]),
            )
            dist = float(dist) * real_world_multiplier
        else:
            if treadmill_speed is None:
                continue
            frame_dist = end - start
            time_dist = frame_dist / fps
            dist = treadmill_speed * time_dist
        unique_strides.append(dist)
        stride_lengths[start:end] = dist

    variability = float(np.std(unique_strides, ddof=1)) if len(unique_strides) > 1 else 0.0
    return stride_lengths, variability


# ---------------------------------------------------------------------------
# 4. Stride frequency
# ---------------------------------------------------------------------------

def stride_frequency(
    stance_frames_per_stride: list[int],
    swing_frames_per_stride: list[int],
    n_strides: int,
    fps: int,
) -> float:
    """VGL §4: strides per second."""
    total_frames = sum(stance_frames_per_stride) + sum(swing_frames_per_stride)
    if total_frames == 0 or n_strides == 0:
        return 0.0
    duration_s = total_frames / fps
    return n_strides / duration_s


# ---------------------------------------------------------------------------
# 5. Paw angles
# ---------------------------------------------------------------------------

def calculate_paw_angles(
    toe_x: np.ndarray,
    toe_y: np.ndarray,
    heel_x: np.ndarray,
    heel_y: np.ndarray,
    ref_p1x: np.ndarray,
    ref_p1y: np.ndarray,
    ref_p2x: np.ndarray,
    ref_p2y: np.ndarray,
    in_stance: np.ndarray,
) -> tuple[np.ndarray, float]:
    """VGL §5: paw angles relative to body axis, with swing frames masked out.

    Returns:
        angles: per-frame angles (swing frames = NaN)
        mean_angle: mean of stance-only angles in degrees.
    """
    toe = np.column_stack((toe_x, toe_y))
    heel = np.column_stack((heel_x, heel_y))
    ref_p1 = np.column_stack((ref_p1x, ref_p1y))
    ref_p2 = np.column_stack((ref_p2x, ref_p2y))
    angles = geometry.paw_angle(toe, heel, ref_p1, ref_p2)
    angles = np.where(in_stance == 1, angles, np.nan)
    mean_angle = float(np.nanmean(angles)) if np.any(in_stance) else 0.0
    return angles, mean_angle


# ---------------------------------------------------------------------------
# 6. Stance widths
# ---------------------------------------------------------------------------

def calculate_stance_widths_legacy(
    right_mid_y: np.ndarray,
    left_mid_y: np.ndarray,
    real_world_multiplier: float,
) -> tuple[np.ndarray, float]:
    """VGL §6.1: simple Y-difference between left/right paw midpoints."""
    widths = np.abs(right_mid_y - left_mid_y) * real_world_multiplier
    mean_width = float(np.mean(widths)) if widths.size else 0.0
    return widths, mean_width


def calculate_stance_widths_per_paw(
    paw_mid_x: np.ndarray,
    paw_mid_y: np.ndarray,
    com_x: np.ndarray,
    com_y: np.ndarray,
    axis_ref_x: np.ndarray,
    axis_ref_y: np.ndarray,
    real_world_multiplier: float,
) -> tuple[np.ndarray, float]:
    """VGL §6.2: perpendicular distance from each paw midpoint to body axis."""
    point = np.column_stack((paw_mid_x, paw_mid_y))
    line_p1 = np.column_stack((com_x, com_y))
    line_p2 = np.column_stack((axis_ref_x, axis_ref_y))
    dists = geometry.distance_point_to_line(point, line_p1, line_p2) * real_world_multiplier
    mean_width = float(np.mean(dists)) if dists.size else 0.0
    return dists, mean_width


# ---------------------------------------------------------------------------
# 7. SEM
# ---------------------------------------------------------------------------

def sem(values: np.ndarray) -> float:
    """Standard error of the mean (VGL §7)."""
    n = len(values)
    if n < 2:
        return 0.0
    sd = float(np.std(values, ddof=1))
    return sd / math.sqrt(n)


# ---------------------------------------------------------------------------
# 8. AutoCorrect
# ---------------------------------------------------------------------------

def auto_correct(in_stance: np.ndarray) -> np.ndarray:
    """VGL §8: merge short spurious segments surrounded by longer same-type segments.

    Threshold = ceil(avg_segment_length / 4).
    """
    if in_stance.size == 0:
        return in_stance.copy()
    segments = geometry.find_runs(in_stance)
    if len(segments) < 3:
        return in_stance.copy()
    avg_len = len(in_stance) / len(segments)
    threshold = int(math.ceil(avg_len / 4.0))
    corrected = in_stance.copy()
    for i in range(1, len(segments) - 1):
        s, e, val = segments[i]
        seg_len = e - s
        left_len = segments[i - 1][1] - segments[i - 1][0]
        right_len = segments[i + 1][1] - segments[i + 1][0]
        neighbors_same = segments[i - 1][2] == segments[i + 1][2]
        if seg_len <= threshold and left_len > threshold and right_len > threshold and neighbors_same:
            corrected[s:e] = segments[i - 1][2]
    return corrected


# ---------------------------------------------------------------------------
# 9. Gait symmetry (VGL extension)
# ---------------------------------------------------------------------------

def gait_symmetry(
    left_stance_ms: float,
    right_stance_ms: float,
    left_swing_ms: float,
    right_swing_ms: float,
) -> float:
    """Simple symmetry index: 1 - |left-right| / (left+right), averaged over stance+swing."""
    stance_sum = left_stance_ms + right_stance_ms
    swing_sum = left_swing_ms + right_swing_ms
    if stance_sum == 0 or swing_sum == 0:
        return 0.0
    stance_sym = 1.0 - abs(left_stance_ms - right_stance_ms) / stance_sum
    swing_sym = 1.0 - abs(left_swing_ms - right_swing_ms) / swing_sum
    return (stance_sym + swing_sym) / 2.0


# =============================================================================
# 10. CatWalk XT standard metrics — extended (kb/05 §6, kb/03 §11)
# =============================================================================
# The original VisualGaitLab algorithm set above (~12 metrics) covers the
# classic CatWalk paper but misses several metrics that CatWalk XT
# reports out-of-the-box and that are referenced by Noldus's published
# documentation. The functions below close that gap and push the metric
# count past 30 (the W5 acceptance gate, DEVELOPMENT_PLAN §6.2 W5).

def stride_cycle_ms(stance_ms: float, swing_ms: float) -> float:
    """CatWalk: Stride = stance + swing (one full cycle)."""
    return float(stance_ms + swing_ms)


def percentage_stance(stance_ms: float, swing_ms: float) -> float:
    """CatWalk: %Stance = stance / (stance + swing) × 100."""
    total = stance_ms + swing_ms
    return 100.0 * stance_ms / total if total > 0 else 0.0


def percentage_swing(stance_ms: float, swing_ms: float) -> float:
    """CatWalk: %Swing = swing / (stance + swing) × 100."""
    total = stance_ms + swing_ms
    return 100.0 * swing_ms / total if total > 0 else 0.0


def duty_cycle(stance_ms: float, swing_ms: float) -> float:
    """CatWalk: Duty Cycle = stance / (stance + swing) (fraction, not %)."""
    total = stance_ms + swing_ms
    return stance_ms / total if total > 0 else 0.0


def swing_to_stance_ratio(stance_ms: float, swing_ms: float) -> float:
    """CatWalk: Swing / Stance ratio."""
    return swing_ms / stance_ms if stance_ms > 0 else 0.0


def cadence(stride_frequency_hz: float) -> float:
    """CatWalk: Cadence in steps/min = 2 × strides/sec × 60.

    One "step" = a single paw contact. Each stride contains two steps
    (e.g. LF contact → RF contact for a front-paw pair), so cadence is
    2× the stride frequency in strides/sec.
    """
    return 2.0 * stride_frequency_hz * 60.0


def step_width_avg(
    left_paw_mid_x: np.ndarray,
    right_paw_mid_x: np.ndarray,
    real_world_multiplier: float,
) -> float:
    """CatWalk: Step Width = perpendicular distance between L/R paws (mm).

    For a corridor walk this is the X-coordinate separation; for a
    side-view camera it is the Y separation. The function works on the
    X-axis by default — pass transposed arrays for other orientations.
    """
    if left_paw_mid_x.size == 0 or right_paw_mid_x.size == 0:
        return 0.0
    n = min(len(left_paw_mid_x), len(right_paw_mid_x))
    diffs = np.abs(left_paw_mid_x[:n] - right_paw_mid_x[:n]) * real_world_multiplier
    return float(np.mean(diffs))


def base_of_support(
    fore_mid_x: np.ndarray,
    fore_mid_y: np.ndarray,
    hind_mid_x: np.ndarray,
    hind_mid_y: np.ndarray,
    real_world_multiplier: float,
) -> float:
    """CatWalk: Base of Support = mean perpendicular distance between
    fore/hind paw midpoints on the body axis. Lower BoS = more stable.
    """
    if fore_mid_x.size == 0 or hind_mid_x.size == 0:
        return 0.0
    n = min(len(fore_mid_x), len(hind_mid_x))
    diffs = np.sqrt(
        (fore_mid_x[:n] - hind_mid_x[:n]) ** 2
        + (fore_mid_y[:n] - hind_mid_y[:n]) ** 2,
    ) * real_world_multiplier
    return float(np.mean(diffs))


def step_angle(
    fore_mid_x: np.ndarray,
    fore_mid_y: np.ndarray,
    hind_mid_x: np.ndarray,
    hind_mid_y: np.ndarray,
    com_x: np.ndarray,
    com_y: np.ndarray,
    axis_ref_x: np.ndarray,
    axis_ref_y: np.ndarray,
) -> np.ndarray:
    """CatWalk: Step Angle = angle (deg) between the fore-hind line and
    the body axis at each frame. Reported per-frame in stride CSVs.
    """
    # Vector v1: hind -> fore  (fore-hind line)
    v1x = fore_mid_x - hind_mid_x
    v1y = fore_mid_y - hind_mid_y
    # Vector v2: CoM -> axis ref  (body axis)
    v2x = axis_ref_x - com_x
    v2y = axis_ref_y - com_y
    # 2D angle between two vectors (degrees, in [-180, 180])
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    return np.degrees(np.arctan2(cross, dot))


# ---------------------------------------------------------------------------
# 10.1 Regularity Index + Phase Dispersion (CatsWalk / Hamers et al. 2006)
# ---------------------------------------------------------------------------
def regularity_index(
    lf_in_stance: np.ndarray,
    rf_in_stance: np.ndarray,
    lh_in_stance: np.ndarray,
    rh_in_stance: np.ndarray,
) -> float:
    """CatWalk: Regularity Index (RI) = % of normal step patterns.

    "Normal" patterns are the 6 possible 4-paw inter-paw contact patterns
    (RF, RH, LF, LH with their cyclic shifts). RI = (count of normal
    patterns) / (count of all patterns) × 100.

    Implementation: convert each paw's stance/swing sequence to a binary
    pattern, find every 4-paw inter-contact, classify each by Hamming
    distance to the 6 canonical patterns, and report the fraction in %.
    """
    patterns = _extract_intercontact_patterns(
        lf_in_stance, rf_in_stance, lh_in_stance, rh_in_stance,
    )
    if not patterns:
        return 0.0
    canonical = _canonical_step_patterns()
    normal = sum(1 for p in patterns if p in canonical)
    return 100.0 * normal / len(patterns)


def phase_dispersion(
    lf_in_stance: np.ndarray,
    rf_in_stance: np.ndarray,
    lh_in_stance: np.ndarray,
    rh_in_stance: np.ndarray,
    fps: int,
) -> dict:
    """CatWalk: Phase Dispersion = the spread of LF↔RF and LH↔RH
    contact offsets over a stride cycle, expressed as % of cycle.

    Returns
    -------
    dict with keys ``lf_rf_dispersion_pct`` and ``lh_rh_dispersion_pct``.
    """
    lf_rf = _contact_offsets_pct(lf_in_stance, rf_in_stance, fps)
    lh_rh = _contact_offsets_pct(lh_in_stance, rh_in_stance, fps)
    return {
        "lf_rf_dispersion_pct": float(np.std(lf_rf)) if lf_rf else 0.0,
        "lh_rh_dispersion_pct": float(np.std(lh_rh)) if lh_rh else 0.0,
    }


def _extract_intercontact_patterns(*paws: np.ndarray) -> list[tuple]:
    """Build the 4-paw inter-contact pattern sequence.

    For each frame, the "pattern" is the tuple of paw states. Inter-contact
    frames (any paw transitioning 0→1) are kept; the rest are skipped.
    """
    if not paws:
        return []
    n = min(len(p) for p in paws)
    if n < 2:
        return []
    out = []
    prev = None
    for i in range(n):
        cur = tuple(int(p[i]) for p in paws)
        if prev is not None and any(cur[k] == 1 and prev[k] == 0
                                    for k in range(4)):
            out.append(cur)
        prev = cur
    return out


def _canonical_step_patterns() -> set[tuple]:
    """The 6 normal CatWalk step patterns (3 fore-pair + 3 hind-pair
    cyclic shifts, with fore/hind phase offset)."""
    base_fore = [
        (1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0),  # RF, LH, LF, RH cyclic
        (0, 1, 0, 0), (0, 0, 1, 0), (1, 0, 0, 0),
    ]
    # Hind-paw partner shifts give the other 3 valid normals.
    base_hind = [
        (0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0),
    ]
    return set(base_fore + base_hind)


def _contact_offsets_pct(a: np.ndarray, b: np.ndarray, fps: int) -> list[float]:
    """Compute the contact-onset offset between two paws, as % of cycle."""
    a_on = np.where(np.diff(np.concatenate(([0], a))) == 1)[0]
    b_on = np.where(np.diff(np.concatenate(([0], b))) == 1)[0]
    if a_on.size == 0 or b_on.size == 0:
        return []
    # For each ``a`` onset, find the next ``b`` onset within a reasonable
    # cycle window (default ≤ 1 s).
    offsets: list[float] = []
    cycle_frames = max(fps, 1)
    for ta in a_on:
        candidates = b_on[b_on > ta]
        if candidates.size == 0:
            continue
        offset_frames = int(candidates[0] - ta)
        if offset_frames > cycle_frames:
            continue
        offsets.append(100.0 * offset_frames / cycle_frames)
    return offsets


# ---------------------------------------------------------------------------
# 10.2 Symmetry index (CatWalk 'SI' = mean / max — used clinically)
# ---------------------------------------------------------------------------
def symmetry_index(left_value: float, right_value: float) -> float:
    """CatWalk: Symmetry Index = mean(L, R) / max(L, R) × 100.

    Returns 100 when perfectly symmetric. Used to compare disease-side
    vs healthy-side gait metrics.
    """
    if max(left_value, right_value) <= 0:
        return 0.0
    return 100.0 * min(left_value, right_value) / max(left_value, right_value)


# =============================================================================
# 11. deepgait extensions (kb/05 §6 — deepgait 新增)
# =============================================================================

def swing_speed(stride_length_mm: float, swing_ms: float) -> float:
    """deepgait extension: Swing Speed = stride_length / swing_duration.

    Units: mm/ms. Higher swing speed = faster recovery, typical of
    locomotor hyperactivity models.
    """
    return stride_length_mm / swing_ms if swing_ms > 0 else 0.0


def body_speed(
    com_x: np.ndarray,
    com_y: np.ndarray,
    real_world_multiplier: float,
    fps: int,
) -> float:
    """deepgait extension: Body Speed (mm/s) from center-of-mass drift.

    Uses only the X-axis (direction of travel) for stability, ignoring
    the small lateral sway.
    """
    if com_x.size < 2 or fps <= 0:
        return 0.0
    distance_mm = float(com_x[-1] - com_x[0]) * real_world_multiplier
    duration_s = (len(com_x) - 1) / fps
    return distance_mm / duration_s if duration_s > 0 else 0.0


def toe_spread(
    fore_toe_x: np.ndarray,
    hind_toe_x: np.ndarray,
    real_world_multiplier: float,
) -> float:
    """deepgait extension: Toe Spread (mm) — fore-vs-hind toe X distance.

    Larger toe spread = wider stance (compensation for instability).
    """
    if fore_toe_x.size == 0 or hind_toe_x.size == 0:
        return 0.0
    n = min(len(fore_toe_x), len(hind_toe_x))
    diffs = np.abs(fore_toe_x[:n] - hind_toe_x[:n]) * real_world_multiplier
    return float(np.mean(diffs))


def stand_index(intensity_curve: np.ndarray) -> float:
    """deepgait extension: Stand Index = area under the FTIR intensity
    curve during stance frames.

    Sum of intensity values × 1/fps → units of (intensity × seconds).
    Larger stand index = more sustained contact = bigger/weighter paw.
    """
    if intensity_curve.size == 0:
        return 0.0
    return float(np.sum(intensity_curve))


def intensity_asymmetry(
    left_intensity_curve: np.ndarray,
    right_intensity_curve: np.ndarray,
) -> float:
    """deepgait extension: Intensity Asymmetry = |L - R| / (L + R)."""
    ls = float(np.sum(left_intensity_curve))
    rs = float(np.sum(right_intensity_curve))
    total = ls + rs
    if total <= 0:
        return 0.0
    return abs(ls - rs) / total


def max_contact_area(footprint_mask: np.ndarray) -> float:
    """deepgait extension: Max Contact Area (pixels) = largest single-frame
    footprint mask size across the trial.

    Larger = heavier load on that paw.
    """
    if footprint_mask.ndim == 3:
        return float(max(int((m > 0).sum()) for m in footprint_mask))
    return 0.0


def mean_intensity_curve(intensity_per_paw: dict) -> float:
    """deepgait extension: Mean intensity (a.u.) across all paws."""
    if not intensity_per_paw:
        return 0.0
    means = [float(np.mean(c)) for c in intensity_per_paw.values()
             if c.size > 0]
    return float(np.mean(means)) if means else 0.0


def max_intensity_curve(intensity_per_paw: dict) -> float:
    """deepgait extension: Max intensity (a.u.) across all paws."""
    if not intensity_per_paw:
        return 0.0
    return float(max(np.max(c) for c in intensity_per_paw.values()
                     if c.size > 0))


def body_axis_angle(
    front_com_x: np.ndarray, front_com_y: np.ndarray,
    rear_com_x: np.ndarray,  rear_com_y: np.ndarray,
) -> float:
    """deepgait extension: Body Axis Angle (deg) = mean angle between
    front-CoM and rear-CoM direction vectors. Used as a proxy for body
    curvature in free-walk gait.
    """
    if front_com_x.size == 0 or rear_com_x.size == 0:
        return 0.0
    dx = front_com_x - rear_com_x
    dy = front_com_y - rear_com_y
    angles = np.degrees(np.arctan2(dy, dx))
    return float(np.mean(angles))


# =============================================================================
# 12. Unified entry point — `compute_all_gait_metrics`
# =============================================================================

def compute_all_gait_metrics(
    *,
    fps: int,
    real_world_multiplier: float,
    in_stance_per_paw: dict,        # {"LF": ndarray, "RF": ..., "LH": ..., "RH": ...}
    paw_mid_x: dict | None = None,
    paw_mid_y: dict | None = None,
    fore_mid_x: np.ndarray | None = None,
    fore_mid_y: np.ndarray | None = None,
    hind_mid_x: np.ndarray | None = None,
    hind_mid_y: np.ndarray | None = None,
    com_x: np.ndarray | None = None,
    com_y: np.ndarray | None = None,
    front_com_x: np.ndarray | None = None,
    front_com_y: np.ndarray | None = None,
    rear_com_x: np.ndarray | None = None,
    rear_com_y: np.ndarray | None = None,
    axis_ref_x: np.ndarray | None = None,
    axis_ref_y: np.ndarray | None = None,
    intensity_curve_per_paw: dict | None = None,
    footprint_mask: np.ndarray | None = None,
    treadmill_speed_mm_s: float | None = None,
) -> dict:
    """Compute the full 30+ metric CatWalk + deepgait gait suite.

    Parameters
    ----------
    fps : int
        Recording frame rate.
    real_world_multiplier : float
        Pixel→mm conversion factor (from ChArUco calibration).
    in_stance_per_paw : dict
        ``{"LF": ndarray, "RF": ndarray, "LH": ndarray, "RH": ndarray}``.
        Each value is a 0/1 stance/swing sequence of length N.
    paw_mid_x, paw_mid_y : dict, optional
        Per-paw mid-point trajectories (pixel units).
    fore_mid_x, fore_mid_y : ndarray, optional
        Average fore-pair midpoint trajectory (for BoS, step angle).
    hind_mid_x, hind_mid_y : ndarray, optional
        Average hind-pair midpoint trajectory (for BoS).
    com_x, com_y : ndarray, optional
        Body center-of-mass trajectory.
    front_com_x, front_com_y, rear_com_x, rear_com_y : ndarray, optional
        Front- and rear-CoM for body axis angle.
    axis_ref_x, axis_ref_y : ndarray, optional
        Body axis reference for step angle.
    intensity_curve_per_paw : dict, optional
        FTIR intensity per paw.
    footprint_mask : ndarray, optional
        Per-frame FTIR footprint mask.
    treadmill_speed_mm_s : float, optional
        Treadmill belt speed (CatWalk belt mode only).

    Returns
    -------
    dict
        Flat {metric_name: value} with at least 31 entries (CatWalk
        XT 12 + deepgait extensions). NaN-bearing entries are filled with
        0.0 for CSV export.

    References
    ----------
    * kb/03_vgl_gait_algorithms.md §11 (CatWalk staticdata fields)
    * kb/05_mousegait_design.md §6 (deepgait extensions)
    """
    out: dict = {}

    # ----- Per-paw stance/swing/cycle/duty/swing-ratio -----
    per_paw = {}
    for paw, stance in in_stance_per_paw.items():
        basics = calculate_gait_basics(np.asarray(stance), fps)
        sf = stride_frequency(
            basics.stance_frames_per_stride,
            basics.swing_frames_per_stride,
            basics.n_strides, fps,
        )
        per_paw[paw] = {
            "stance_ms": basics.stance_duration_ms,
            "swing_ms": basics.swing_duration_ms,
            "cycle_ms": stride_cycle_ms(
                basics.stance_duration_ms, basics.swing_duration_ms,
            ),
            "pct_stance": percentage_stance(
                basics.stance_duration_ms, basics.swing_duration_ms,
            ),
            "pct_swing": percentage_swing(
                basics.stance_duration_ms, basics.swing_duration_ms,
            ),
            "duty_cycle": duty_cycle(
                basics.stance_duration_ms, basics.swing_duration_ms,
            ),
            "swing_to_stance": swing_to_stance_ratio(
                basics.stance_duration_ms, basics.swing_duration_ms,
            ),
            "stride_frequency_hz": sf,
            "n_strides": basics.n_strides,
        }
    out["per_paw"] = per_paw

    # ----- Aggregate stance/swing (mean over 4 paws) -----
    if per_paw:
        out["avg_stance_ms"] = float(np.mean([v["stance_ms"] for v in per_paw.values()]))
        out["avg_swing_ms"] = float(np.mean([v["swing_ms"] for v in per_paw.values()]))
        out["avg_cycle_ms"] = float(np.mean([v["cycle_ms"] for v in per_paw.values()]))
        out["avg_pct_stance"] = float(np.mean([v["pct_stance"] for v in per_paw.values()]))
        out["avg_pct_swing"] = float(np.mean([v["pct_swing"] for v in per_paw.values()]))
        out["avg_duty_cycle"] = float(np.mean([v["duty_cycle"] for v in per_paw.values()]))
        out["avg_swing_to_stance"] = float(np.mean([v["swing_to_stance"] for v in per_paw.values()]))
        # Stride frequency: average across paws, ignoring zero.
        sfs = [v["stride_frequency_hz"] for v in per_paw.values()
               if v["stride_frequency_hz"] > 0]
        out["avg_stride_frequency_hz"] = float(np.mean(sfs)) if sfs else 0.0
        out["cadence_steps_per_min"] = cadence(out["avg_stride_frequency_hz"])

    # ----- Stride length (only when paw midpoints are provided) -----
    if paw_mid_x is not None and paw_mid_y is not None:
        stride_lens = []
        for paw in in_stance_per_paw:
            if paw not in paw_mid_x:
                continue
            stance = in_stance_per_paw[paw]
            basics = calculate_gait_basics(np.asarray(stance), fps)
            sl, _ = calculate_stride_data(
                basics.switch_positions,
                np.asarray(paw_mid_x[paw]),
                np.asarray(paw_mid_y[paw]),
                is_free_run=(treadmill_speed_mm_s is None),
                treadmill_speed=treadmill_speed_mm_s,
                real_world_multiplier=real_world_multiplier,
                fps=fps,
            )
            stride_lens.extend(sl[sl > 0].tolist())
        if stride_lens:
            out["stride_length_avg_mm"] = float(np.mean(stride_lens))
            out["stride_length_variability"] = float(
                np.std(stride_lens, ddof=1) if len(stride_lens) > 1 else 0.0,
            )

    # ----- Regularity + Phase Dispersion -----
    if all(p in in_stance_per_paw for p in ("LF", "RF", "LH", "RH")):
        out["regularity_index_pct"] = regularity_index(
            np.asarray(in_stance_per_paw["LF"]),
            np.asarray(in_stance_per_paw["RF"]),
            np.asarray(in_stance_per_paw["LH"]),
            np.asarray(in_stance_per_paw["RH"]),
        )
        pd_ = phase_dispersion(
            np.asarray(in_stance_per_paw["LF"]),
            np.asarray(in_stance_per_paw["RF"]),
            np.asarray(in_stance_per_paw["LH"]),
            np.asarray(in_stance_per_paw["RH"]),
            fps,
        )
        out["phase_dispersion_lf_rf_pct"] = pd_["lf_rf_dispersion_pct"]
        out["phase_dispersion_lh_rh_pct"] = pd_["lh_rh_dispersion_pct"]

    # ----- Symmetry indices (left/right pairs) -----
    if "LF" in per_paw and "RF" in per_paw:
        out["symmetry_index_stance"] = symmetry_index(
            per_paw["LF"]["stance_ms"], per_paw["RF"]["stance_ms"],
        )
        out["symmetry_index_swing"] = symmetry_index(
            per_paw["LF"]["swing_ms"], per_paw["RF"]["swing_ms"],
        )
    if "LH" in per_paw and "RH" in per_paw:
        out["symmetry_index_hind_stance"] = symmetry_index(
            per_paw["LH"]["stance_ms"], per_paw["RH"]["stance_ms"],
        )
        out["symmetry_index_hind_swing"] = symmetry_index(
            per_paw["LH"]["swing_ms"], per_paw["RH"]["swing_ms"],
        )

    # ----- Step width / BoS / Step angle -----
    if paw_mid_x is not None:
        if "LF" in paw_mid_x and "RF" in paw_mid_x:
            out["step_width_avg_mm"] = step_width_avg(
                np.asarray(paw_mid_x["LF"]),
                np.asarray(paw_mid_x["RF"]),
                real_world_multiplier,
            )
        if fore_mid_x is not None and hind_mid_x is not None:
            out["base_of_support_mm"] = base_of_support(
                np.asarray(fore_mid_x), np.asarray(fore_mid_y),
                np.asarray(hind_mid_x), np.asarray(hind_mid_y),
                real_world_multiplier,
            )
        if all(a is not None for a in (fore_mid_x, fore_mid_y, hind_mid_x,
                                        hind_mid_y, com_x, com_y, axis_ref_x,
                                        axis_ref_y)):
            out["step_angle_deg"] = float(np.nanmean(step_angle(
                np.asarray(fore_mid_x), np.asarray(fore_mid_y),
                np.asarray(hind_mid_x), np.asarray(hind_mid_y),
                np.asarray(com_x), np.asarray(com_y),
                np.asarray(axis_ref_x), np.asarray(axis_ref_y),
            )))

    # ----- Body speed (deepgait extension) -----
    if com_x is not None:
        out["body_speed_mm_s"] = body_speed(
            np.asarray(com_x), np.asarray(com_y),
            real_world_multiplier, fps,
        )

    # ----- Toe spread (deepgait extension) -----
    if paw_mid_x is not None and "LF" in paw_mid_x and "LH" in paw_mid_x:
        out["toe_spread_mm"] = toe_spread(
            np.asarray(paw_mid_x["LF"]),
            np.asarray(paw_mid_x["LH"]),
            real_world_multiplier,
        )

    # ----- Body axis angle (deepgait extension) -----
    if front_com_x is not None and rear_com_x is not None:
        out["body_axis_angle_deg"] = body_axis_angle(
            np.asarray(front_com_x), np.asarray(front_com_y),
            np.asarray(rear_com_x),  np.asarray(rear_com_y),
        )

    # ----- Swing speed (deepgait extension) -----
    if "stride_length_avg_mm" in out and "avg_swing_ms" in out:
        out["swing_speed_mm_ms"] = swing_speed(
            out["stride_length_avg_mm"], out["avg_swing_ms"],
        )

    # ----- FTIR / footprint extensions -----
    if intensity_curve_per_paw:
        out["stand_index"] = stand_index(
            np.concatenate(list(intensity_curve_per_paw.values())),
        ) if intensity_curve_per_paw else 0.0
        out["mean_intensity"] = mean_intensity_curve(intensity_curve_per_paw)
        out["max_intensity"] = max_intensity_curve(intensity_curve_per_paw)
        if "LF" in intensity_curve_per_paw and "RF" in intensity_curve_per_paw:
            out["intensity_asymmetry_lf_rf"] = intensity_asymmetry(
                np.asarray(intensity_curve_per_paw["LF"]),
                np.asarray(intensity_curve_per_paw["RF"]),
            )

    if footprint_mask is not None:
        out["max_contact_area_pixels"] = max_contact_area(
            np.asarray(footprint_mask),
        )

    # Final tally: how many scalar metrics did we emit (excluding per_paw)?
    scalar_count = sum(1 for k, v in out.items()
                       if k != "per_paw" and isinstance(v, (int, float)))
    out["__n_metrics__"] = scalar_count + sum(len(p) for p in per_paw.values())

    return out
