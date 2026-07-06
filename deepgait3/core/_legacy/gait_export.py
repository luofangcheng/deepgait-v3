"""Gait analysis export — English CSV + PNG image output.

Writes all analysis results to ``{project}/data/{animal_id}/`` in
CatWalk-compatible format with English headers.  Retains legacy wrappers
for backward compatibility.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CSV export (new CatWalk-style)
# ---------------------------------------------------------------------------
def export_gait_metrics_csv(
    metrics: Dict[str, float],
    output_path: Path,
    paw_names: Sequence[str] = ("LF", "RF", "LH", "RH"),
) -> Path:
    """Write per-paw summary CSV with English headers."""
    path = Path(output_path)
    headers = [
        "paw", "n_steps", "avg_stance_s", "avg_swing_s", "duty_cycle_pct",
        "stride_length_cm", "step_length_cm",
        "avg_swing_speed_cm_s", "avg_instantaneous_speed_cm_s",
        "max_contact_area_cm2", "print_area_cm2",
        "mean_intensity", "max_intensity", "max_contact_max_intensity",
        "avg_stand_index", "total_stand_index", "cadence_steps_per_min",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for paw in paw_names:
            row: Dict[str, Any] = {"paw": paw}
            for col in headers[1:]:
                key = f"{paw}_{col}"
                row[col] = metrics.get(key, "")
            w.writerow(row)
    return path


def export_per_step_csv(
    all_steps: Dict[str, List[Dict]],
    output_path: Path,
) -> Path:
    """Write per-step detailed CSV."""
    path = Path(output_path)
    headers = [
        "paw", "step_num", "start_frame", "end_frame",
        "stance_frames", "stand_s", "swing_frames", "swing_s",
        "stride_length_cm", "swing_speed_cm_s", "instantaneous_speed_cm_s",
        "max_intensity", "mean_intensity", "max_contact_area_cm2",
        "braking_s", "propulsion_s", "braking_index", "propulsion_index",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for paw, steps in all_steps.items():
            for i, step in enumerate(steps):
                row = {"paw": paw, "step_num": i + 1}
                for col in headers[2:]:
                    row[col] = step.get(col, "")
                w.writerow(row)
    return path


def export_pressure_per_frame_csv(
    pressure_rows: List[Dict[str, float]],
    output_path: Path,
    paw_names: Sequence[str] = ("LF", "RF", "LH", "RH"),
) -> Path:
    """Write per-frame pressure CSV."""
    path = Path(output_path)
    headers = ["frame"]
    for paw in paw_names:
        headers += [
            f"{paw}_x", f"{paw}_y",
            f"{paw}_area_px", f"{paw}_intensity_max", f"{paw}_intensity_mean",
        ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in pressure_rows:
            flat: Dict[str, Any] = {}
            for k, v in row.items():
                if isinstance(v, float):
                    flat[k] = round(v, 4) if abs(v) < 100 else round(v, 2)
                else:
                    flat[k] = v
            w.writerow(flat)
    return path


# ---------------------------------------------------------------------------
# Image export (PNG via matplotlib Agg backend)
# ---------------------------------------------------------------------------
def _ensure_mpl() -> None:
    import os
    os.environ.setdefault("QT_API", "pyside6")
    import matplotlib
    matplotlib.use("Agg")


def export_max_footmap_png(
    accumulated_image: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Save the accumulated footprint image as PNG."""
    _ensure_mpl()
    import matplotlib.pyplot as plt
    path = Path(output_path)
    rgb = accumulated_image[..., ::-1]  # BGR → RGB
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(rgb)
    ax.set_title("Max Footprint Map", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    return path


def export_sequence_png(
    in_stance: Dict[str, np.ndarray],
    output_path: Path,
    fps: float = 100.0,
    dpi: int = 150,
) -> Path:
    """Save the limb sequence chart as PNG (CatWalk style)."""
    _ensure_mpl()
    import matplotlib.pyplot as plt
    path = Path(output_path)
    paw_colors = {"LF": "#FFC107", "RF": "#F44336", "LH": "#2196F3", "RH": "#E91E63"}
    row_order = ["RH", "LH", "RF", "LF"]
    fig, ax = plt.subplots(figsize=(12, 3))
    for row_idx, paw in enumerate(row_order):
        arr = in_stance.get(paw)
        if arr is None or len(arr) == 0:
            continue
        for s, e in _find_stance_segments(arr):
            t0, t1 = s / max(fps, 1.0), e / max(fps, 1.0)
            ax.barh(row_idx, t1 - t0, left=t0, height=0.7,
                    color=paw_colors.get(paw, "#888"), edgecolor="none")
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order)
    ax.set_xlabel("Time (s)")
    ax.set_title("Limb Sequence", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def export_pressure_profile_png(
    intensity_curves: Dict[str, np.ndarray],
    output_path: Path,
    fps: float = 100.0,
    dpi: int = 150,
) -> Path:
    """Save pressure-vs-frame line chart as PNG."""
    _ensure_mpl()
    import matplotlib.pyplot as plt
    path = Path(output_path)
    paw_colors = {"LF": "#FFC107", "RF": "#F44336", "LH": "#2196F3", "RH": "#E91E63"}
    fig, ax = plt.subplots(figsize=(12, 3))
    for paw in ("LF", "RF", "LH", "RH"):
        curve = intensity_curves.get(paw)
        if curve is None or len(curve) == 0:
            continue
        t = np.arange(len(curve)) / max(fps, 1.0)
        ax.plot(t, curve, color=paw_colors.get(paw, "#888"),
                label=paw, linewidth=0.8, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Max Intensity (a.u.)")
    ax.set_title("Pressure Profile", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def export_foot_pattern_png(
    centroids_x: Dict[str, np.ndarray],
    centroids_y: Dict[str, np.ndarray],
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Save foot pattern spatial distribution as PNG."""
    _ensure_mpl()
    import matplotlib.pyplot as plt
    path = Path(output_path)
    paw_colors = {"LF": "#FFC107", "RF": "#F44336", "LH": "#2196F3", "RH": "#E91E63"}
    markers = {"LF": "o", "RF": "s", "LH": "^", "RH": "v"}
    fig, ax = plt.subplots(figsize=(12, 3))
    for paw in ("LF", "RF", "LH", "RH"):
        cx = centroids_x.get(paw)
        cy = centroids_y.get(paw)
        if cx is None or len(cx) == 0:
            continue
        ax.scatter(cx, cy, c=paw_colors.get(paw, "#888"),
                   marker=markers.get(paw, "o"), label=paw, s=8, alpha=0.6)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.invert_yaxis()
    ax.set_title("Foot Pattern", fontsize=10)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _find_stance_segments(in_stance: np.ndarray) -> List[Tuple[int, int]]:
    if len(in_stance) < 2:
        return []
    d = np.diff(np.concatenate([[0], in_stance.astype(np.int8), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
# Top-level batch export
# ---------------------------------------------------------------------------
def export_all(
    output_dir: Path,
    animal_id: str,
    metrics: Dict[str, float],
    all_steps: Dict[str, List[Dict]],
    pressure_rows: List[Dict[str, float]],
    accumulated_image: Optional[np.ndarray] = None,
    in_stance: Optional[Dict[str, np.ndarray]] = None,
    intensity_curves: Optional[Dict[str, np.ndarray]] = None,
    centroids_x: Optional[Dict[str, np.ndarray]] = None,
    centroids_y: Optional[Dict[str, np.ndarray]] = None,
    fps: float = 100.0,
) -> List[Path]:
    """Export all CSV + PNG outputs for one animal. Returns list of paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    paths.append(export_gait_metrics_csv(
        metrics, output_dir / f"{animal_id}_gait_metrics.csv"))
    paths.append(export_per_step_csv(
        all_steps, output_dir / f"{animal_id}_per_step.csv"))
    paths.append(export_pressure_per_frame_csv(
        pressure_rows, output_dir / f"{animal_id}_pressure_per_frame.csv"))

    if accumulated_image is not None:
        try:
            paths.append(export_max_footmap_png(
                accumulated_image, output_dir / f"{animal_id}_max_footmap.png"))
        except Exception:
            pass
    if in_stance is not None:
        try:
            paths.append(export_sequence_png(
                in_stance, output_dir / f"{animal_id}_sequence.png", fps))
        except Exception:
            pass
    if intensity_curves is not None:
        try:
            paths.append(export_pressure_profile_png(
                intensity_curves, output_dir / f"{animal_id}_pressure_profile.png", fps))
        except Exception:
            pass
    if centroids_x is not None and centroids_y is not None:
        try:
            paths.append(export_foot_pattern_png(
                centroids_x, centroids_y, output_dir / f"{animal_id}_foot_pattern.png"))
        except Exception:
            pass
    return paths


# ---------------------------------------------------------------------------
# Legacy wrappers (backward compat — reduced functionality)
# ---------------------------------------------------------------------------
def to_summary_csv(res: object, path: str | Path) -> Path:
    """Legacy wrapper — delegates to pandas export when possible."""
    path = Path(path)
    try:
        from deepgait3.core._legacy import results as _r
        if isinstance(res, _r.GaitResults):
            import pandas as pd
            rows = res.summary_table()
            df = pd.DataFrame(rows)
            cols = [c for c in [
                "name","side","limb","stance_duration_ms","swing_duration_ms",
                "n_strides","stride_length_mean","stride_length_variability",
                "stride_frequency_hz","paw_angle_mean_deg","stance_width_mean",
            ] if c in df.columns]
            df[cols].to_csv(path, index=False)
            return path
    except Exception:
        pass
    # Fallback: write empty with headers
    path.write_text("name,side,limb,stance_duration_ms\n", encoding="utf-8")
    return path


def to_timeseries_csv(res: object, path: str | Path) -> Path:
    """Legacy wrapper."""
    path = Path(path)
    try:
        from deepgait3.core._legacy import results as _r
        if isinstance(res, _r.GaitResults):
            import pandas as pd
            n = res.n_frames
            df = pd.DataFrame({"frame": range(n)})
            for name, paw in res.paws.items():
                s = np.full(n, np.nan)
                if paw.in_stance.size:
                    s[:paw.in_stance.size] = paw.in_stance
                df[f"{name}_stance"] = s
            df.to_csv(path, index=False)
            return path
    except Exception:
        pass
    path.write_text("frame\n", encoding="utf-8")
    return path
