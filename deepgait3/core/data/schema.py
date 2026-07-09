"""Data contracts for the data layer — trial-level and per-footprint records.

These dataclasses define the structured output of Stage 1 footprint
extraction and serve as the input to gait_metrics (Stage 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class FootprintRecord:
    """Per-frame measurement for a single footprint within one cycle.

    One row in ``footprints_detail.csv``.
    """
    frame: int
    time_s: float
    centroid_x_mm: float
    centroid_y_mm: float
    area_mm2: float
    area_px: int
    mean_intensity: float
    peak_intensity: float
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0
    png_path: str = ""


@dataclass
class ExtractedCycle:
    """One complete paw contact event (touchdown → liftoff).

    Contains all fields required by ``gait_metrics_module`` plus extras.
    One row in ``cycles_summary.csv``.
    """
    # Identity
    cycle_id: int
    paw_id: str = ""                          # LF/RF/LH/RH — Stage 2 fills this

    # Temporal
    touchdown_frame: int = 0
    liftoff_frame: int = 0
    peak_area_frame: int = 0
    peak_intensity_frame: int = 0
    duration_s: float = 0.0
    loading_duration_s: float = 0.0
    weight_bearing_duration_s: float = 0.0
    unloading_duration_s: float = 0.0

    # Spatial (mm)
    peak_centroid_x_mm: float = 0.0
    peak_centroid_y_mm: float = 0.0
    print_length_mm: float = 0.0
    print_width_mm: float = 0.0
    max_area_mm2: float = 0.0
    max_area_px: int = 0

    # Intensity / Pressure
    mean_pressure_at_peak: float = 0.0
    peak_pressure: float = 0.0
    stand_index: float = 0.0
    pressure_area_ratio: float = 0.0

    # Center of Pressure
    cop_path_length_mm: float = 0.0
    cop_displacement_mm: float = 0.0

    # Per-frame records
    frames: List[FootprintRecord] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    # gait_metrics_module compat
    @property
    def peak_frame_centroid_xy_mm(self) -> Tuple[float, float]:
        return (self.peak_centroid_x_mm, self.peak_centroid_y_mm)

    @property
    def linkage_to_3d(self):
        return self


@dataclass
class TrialData:
    """Top-level container for a single trial's extraction results."""
    mouse_id: str = ""
    trial_name: str = ""
    input_video: str = ""
    num_frames: int = 0
    frame_width: int = 0
    frame_height: int = 0
    fps: float = 60.0
    px_per_mm: float = 1.92
    created_at: str = ""
    cycles: List[ExtractedCycle] = field(default_factory=list)

    @property
    def n_cycles(self) -> int:
        return len(self.cycles)
