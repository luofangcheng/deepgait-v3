"""End-to-end gait analysis pipeline.

Single entry point: :func:`analyze` — takes a DLC CSV path, returns a fully
populated :class:`GaitResults`.  Orchestrates stance/swing detection,
stride/timing extraction, angles, widths, and symmetry.

Design:
- NaN gaps from low-likelihood DLC frames are linearly interpolated *before*
  stance/swing detection (VGL behavior).  Only x,y are interpolated; the
  binary in_stance array is computed on the interpolated trajectory.
- Body axis reference points (Nose, Butt, MidPoints) are interpolated too.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np

from deepgait3.core._legacy import bodyparts, gait_algorithms as ga, gait_io, results
from deepgait3.utils import geometry


Mode = Literal["treadmill", "catwalk"]


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1-D array.

    Leading/trailing NaNs are filled with the nearest valid value.
    """
    a = arr.copy().astype(float)
    n = a.size
    if n == 0:
        return a
    valid = ~np.isnan(a)
    if not valid.any():
        return np.zeros_like(a)  # all-NaN -> treat as 0 (paw undetected)
    if valid.all():
        return a
    idx = np.arange(n)
    a[~valid] = np.interp(idx[~valid], idx[valid], a[valid])
    return a


def _interp_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate NaN gaps in x and y independently."""
    return _interpolate_nans(x), _interpolate_nans(y)


def analyze(
    csv_path: str | Path,
    fps: int = 100,
    mode: Mode = "catwalk",
    video_height: float | None = None,
    likelihood_threshold: float = 0.1,
    treadmill_speed: float | None = None,
    real_world_multiplier: float = 1.0,
    autocorrect: bool = True,
) -> results.GaitResults:
    """Run full gait analysis on a DLC CSV file.

    Args:
        csv_path: path to DLC CSV (e.g., *_filtered.csv).
        fps: frames per second of source video.
        mode: "treadmill" (X-trend) or "catwalk" (displacement threshold).
        video_height: pixel height for Y-flip; None = no flip.
        likelihood_threshold: DLC likelihood below this -> NaN.
        treadmill_speed: required if mode="treadmill" and stride length is needed
            (cm/s or px/s depending on multiplier convention).
        real_world_multiplier: px -> real units (e.g. cm/px).
        autocorrect: apply AutoCorrect to merge spurious short segments.

    Returns:
        GaitResults with per-paw and body-level metrics.
    """
    csv_path = Path(csv_path)
    data = gait_io.read_dlc_csv(
        csv_path,
        video_height=video_height,
        likelihood_threshold=likelihood_threshold,
    )
    n_frames = len(next(iter(data.values())))

    # Body axis (interpolated)
    nose_x, nose_y, butt_x, butt_y, com_x, com_y = gait_io.load_body_axis(data)
    nose_x, nose_y = _interp_xy(nose_x, nose_y)
    butt_x, butt_y = _interp_xy(butt_x, butt_y)
    com_x, com_y = _interp_xy(com_x, com_y)

    res = results.GaitResults(n_frames=n_frames, fps=fps, source_csv=str(csv_path))

    for paw in bodyparts.PAWS:
        pr = _analyze_paw(
            paw=paw,
            data=data,
            mode=mode,
            fps=fps,
            treadmill_speed=treadmill_speed,
            real_world_multiplier=real_world_multiplier,
            autocorrect=autocorrect,
            # Body axis reference for angles/widths
            ref_p1x=nose_x, ref_p1y=nose_y,
            ref_p2x=butt_x, ref_p2y=butt_y,
            com_x=com_x, com_y=com_y,
        )
        res.paws[paw.name] = pr

    # Body-level symmetry: average fore + hind L/R
    res.gait_symmetry_index = _compute_symmetry(res)
    return res


def _analyze_paw(
    paw: bodyparts.Paw,
    data: dict[str, np.ndarray],
    mode: Mode,
    fps: int,
    treadmill_speed: float | None,
    real_world_multiplier: float,
    autocorrect: bool,
    ref_p1x: np.ndarray, ref_p1y: np.ndarray,
    ref_p2x: np.ndarray, ref_p2y: np.ndarray,
    com_x: np.ndarray, com_y: np.ndarray,
) -> results.PawResults:
    """Run all per-paw analyses."""
    toe_x, toe_y, heel_x, heel_y = gait_io.load_paw_keypoints(data, paw)
    toe_x, toe_y = _interp_xy(toe_x, toe_y)
    heel_x, heel_y = _interp_xy(heel_x, heel_y)

    mid_x = (toe_x + heel_x) / 2.0
    mid_y = (toe_y + heel_y) / 2.0

    # 1. Stance/swing
    if mode == "treadmill":
        in_stance = ga.treadmill_in_stance(mid_x)
    else:
        in_stance = ga.catwalk_in_stance(toe_x, toe_y, heel_x, heel_y)
    if autocorrect:
        in_stance = ga.auto_correct(in_stance)

    # 2. Timing + strides
    basics = ga.calculate_gait_basics(in_stance, fps=fps)

    # 3. Stride length
    stride_lengths, variability = ga.calculate_stride_data(
        basics.switch_positions, mid_x, mid_y,
        is_free_run=(mode == "catwalk"),
        treadmill_speed=treadmill_speed,
        real_world_multiplier=real_world_multiplier,
        fps=fps,
    )
    nonzero = stride_lengths[stride_lengths > 0]
    stride_mean = float(np.mean(nonzero)) if nonzero.size else 0.0

    # 4. Stride frequency
    freq = ga.stride_frequency(
        basics.stance_frames_per_stride,
        basics.swing_frames_per_stride,
        basics.n_strides,
        fps,
    )

    # 5. Paw angles (relative to body axis Nose->Butt)
    angles, angle_mean = ga.calculate_paw_angles(
        toe_x, toe_y, heel_x, heel_y,
        ref_p1x, ref_p1y, ref_p2x, ref_p2y,
        in_stance,
    )

    # 6. Stance width (perpendicular distance from paw midpoint to body axis).
    #    Body axis line is defined by CoM and Nose (both per-frame).
    widths, width_mean = ga.calculate_stance_widths_per_paw(
        mid_x, mid_y, com_x, com_y,
        ref_p1x, ref_p1y,  # Nose as axis reference
        real_world_multiplier,
    )

    return results.PawResults(
        name=paw.name,
        side=paw.side,
        limb=paw.limb,
        stance_duration_ms=basics.stance_duration_ms,
        swing_duration_ms=basics.swing_duration_ms,
        n_strides=basics.n_strides,
        stride_length_mean=stride_mean,
        stride_length_variability=variability,
        stride_frequency_hz=freq,
        paw_angle_mean_deg=angle_mean,
        stance_width_mean=width_mean,
        in_stance=in_stance,
        stride_lengths=stride_lengths,
        paw_angles=angles,
    )


def _compute_symmetry(res: results.GaitResults) -> float:
    """Average symmetry across fore L/R and hind L/R pairs."""
    pairs = [
        ("LeftFore", "RightFore"),
        ("LeftHind", "RightHind"),
    ]
    scores: list[float] = []
    for left_name, right_name in pairs:
        if left_name not in res.paws or right_name not in res.paws:
            continue
        left = res.paws[left_name]
        right = res.paws[right_name]
        s = ga.gait_symmetry(
            left.stance_duration_ms, right.stance_duration_ms,
            left.swing_duration_ms, right.swing_duration_ms,
        )
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0
