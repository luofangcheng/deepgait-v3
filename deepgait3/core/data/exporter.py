"""Data export: TrialData → CSV, JSON, HDF5.

Usage::

    from deepgait3.core.data.exporter import export_trial

    trial = TrialData(mouse_id="C57_001", ...)
    export_trial(trial, Path("/output/dir"))
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from deepgait3.core.data.schema import TrialData, ExtractedCycle, FootprintRecord


def export_trial(trial: TrialData, output_dir: Path) -> dict[str, Path]:
    """Export all trial data to the given directory.

    Returns a dict mapping format name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # CSV — cycles summary
    csv_path = output_dir / "cycles_summary.csv"
    _write_cycles_csv(trial.cycles, csv_path)
    written["cycles_csv"] = csv_path

    # CSV — per-frame detail
    detail_path = output_dir / "footprints_detail.csv"
    _write_detail_csv(trial.cycles, detail_path)
    written["detail_csv"] = detail_path

    # JSON — full structured export
    json_path = output_dir / "trial_data.json"
    _write_json(trial, json_path)
    written["json"] = json_path

    return written


# ── CSV writers ──────────────────────────────────────────────────────────────

_CYCLE_FIELDS = [
    "cycle_id", "paw_id",
    "touchdown_frame", "liftoff_frame", "peak_area_frame", "peak_intensity_frame",
    "duration_s", "loading_duration_s", "weight_bearing_duration_s", "unloading_duration_s",
    "peak_centroid_x_mm", "peak_centroid_y_mm",
    "print_length_mm", "print_width_mm",
    "max_area_mm2", "max_area_px",
    "mean_pressure_at_peak", "peak_pressure", "stand_index", "pressure_area_ratio",
    "cop_path_length_mm", "cop_displacement_mm",
    "n_frames",
]


def _write_cycles_csv(cycles: list[ExtractedCycle], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CYCLE_FIELDS)
        for c in cycles:
            w.writerow([
                c.cycle_id, c.paw_id,
                c.touchdown_frame, c.liftoff_frame,
                c.peak_area_frame, c.peak_intensity_frame,
                c.duration_s, c.loading_duration_s,
                c.weight_bearing_duration_s, c.unloading_duration_s,
                c.peak_centroid_x_mm, c.peak_centroid_y_mm,
                c.print_length_mm, c.print_width_mm,
                c.max_area_mm2, c.max_area_px,
                c.mean_pressure_at_peak, c.peak_pressure,
                c.stand_index, c.pressure_area_ratio,
                c.cop_path_length_mm, c.cop_displacement_mm,
                c.n_frames,
            ])


_DETAIL_FIELDS = [
    "cycle_id", "frame", "time_s",
    "centroid_x_mm", "centroid_y_mm",
    "area_mm2", "area_px",
    "mean_intensity", "peak_intensity",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
]


def _write_detail_csv(cycles: list[ExtractedCycle], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_DETAIL_FIELDS)
        for c in cycles:
            for fr in c.frames:
                w.writerow([
                    c.cycle_id, fr.frame, fr.time_s,
                    fr.centroid_x_mm, fr.centroid_y_mm,
                    fr.area_mm2, fr.area_px,
                    fr.mean_intensity, fr.peak_intensity,
                    fr.bbox_x1, fr.bbox_y1, fr.bbox_x2, fr.bbox_y2,
                ])


# ── JSON writer ──────────────────────────────────────────────────────────────

def _write_json(trial: TrialData, path: Path) -> None:
    data = {
        "mouse_id": trial.mouse_id,
        "trial_name": trial.trial_name,
        "input_video": trial.input_video,
        "num_frames": trial.num_frames,
        "frame_width": trial.frame_width,
        "frame_height": trial.frame_height,
        "fps": trial.fps,
        "px_per_mm": trial.px_per_mm,
        "created_at": trial.created_at,
        "n_cycles": trial.n_cycles,
        "cycles": [
            {
                "cycle_id": c.cycle_id,
                "paw_id": c.paw_id,
                "touchdown_frame": c.touchdown_frame,
                "liftoff_frame": c.liftoff_frame,
                "duration_s": c.duration_s,
                "max_area_mm2": c.max_area_mm2,
                "peak_centroid_x_mm": c.peak_centroid_x_mm,
                "peak_centroid_y_mm": c.peak_centroid_y_mm,
                "print_length_mm": c.print_length_mm,
                "print_width_mm": c.print_width_mm,
                "cop_path_length_mm": c.cop_path_length_mm,
                "n_frames": c.n_frames,
                "frames": [
                    {
                        "frame": fr.frame,
                        "time_s": fr.time_s,
                        "centroid_x_mm": fr.centroid_x_mm,
                        "centroid_y_mm": fr.centroid_y_mm,
                        "area_mm2": fr.area_mm2,
                        "mean_intensity": fr.mean_intensity,
                    }
                    for fr in c.frames
                ],
            }
            for c in trial.cycles
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
