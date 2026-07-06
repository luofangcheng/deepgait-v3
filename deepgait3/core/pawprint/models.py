"""DeepGait v3 — Stage 1 data contracts (2026-06-27).

Replaces the v0.4.2 PawPrint schema.  Key simplifications:
  - No paw_id / Linkage3D (limb identification is Stage 2).
  - No heavy ndarray fields — only scalars, suitable for database storage.
  - Cycles are numbered by touchdown order (1, 2, 3, …) *within a trial*.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Legacy contracts (kept for extractor.py / tracker.py backward compat)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class FrameData:
    frame: int
    time_s: float
    bbox_xyxy: Tuple[int, int, int, int]
    bbox_xyxy_padded: Tuple[int, int, int, int]
    raw_intensity_crop: np.ndarray
    bg_intensity_crop: np.ndarray
    paw_mask: np.ndarray
    pressure_map: np.ndarray
    centroid_xy_mm: Tuple[float, float]
    area_mm2: float
    mean_intensity_in_mask: float
    mean_pressure: float
    peak_intensity: float
    peak_pressure: float


@dataclass(slots=True)
class QualityFlags:
    touches_edge: bool = False
    merged_with_neighbor: bool = False
    n_frames: int = 0
    min_area_below_thresh: bool = False
    saturated_pixels_pct: float = 0.0
    snr: float = 0.0


@dataclass(slots=True)
class Linkage3D:
    paw_id: Optional[str] = None
    ankle_3d_at_peak: Optional[Tuple[float, float, float]] = None
    ankle_2d_projection: Optional[Tuple[float, float]] = None
    match_distance_mm: Optional[float] = None


@dataclass(slots=True)
class FootMask:
    blob_indices: List[int]
    centroid_px: Tuple[float, float]
    bbox_xyxy: Tuple[int, int, int, int]
    bbox_xyxy_padded: Tuple[int, int, int, int]
    mask_padded: np.ndarray
    raw_intensity_crop: np.ndarray
    bg_intensity_crop: np.ndarray
    pressure_map: np.ndarray
    total_area_px: int
    mean_intensity: float
    peak_intensity: float
    touches_edge: bool = False


@dataclass(slots=True)
class PawPrint:
    print_id: int
    touchdown_frame: int
    liftoff_frame: int
    true_liftoff_frame: int
    peak_area_frame: int
    peak_intensity_frame: int
    duration_s: float
    time_to_peak_area_s: float
    time_to_peak_intensity_s: float
    loading_duration_s: float = 0.0
    weight_bearing_duration_s: float = 0.0
    unloading_duration_s: float = 0.0
    frames: List[FrameData] = field(default_factory=list)
    max_area_mm2: float = 0.0
    peak_frame_centroid_xy_mm: Tuple[float, float] = (0.0, 0.0)
    peak_frame_bbox_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)
    print_length_mm: float = 0.0
    print_width_mm: float = 0.0
    print_orientation_deg: float = 0.0
    compactness: float = 0.0
    toe_positions_xy_mm: Optional[List[Tuple[float, float]]] = None
    stand_index: float = 0.0
    rising_slope: float = 0.0
    cop_trajectory_mm: List[Tuple[float, float]] = field(default_factory=list)
    cop_path_length_mm: float = 0.0
    cop_displacement_mm: float = 0.0
    peak_pressure: float = 0.0
    mean_pressure_at_peak: float = 0.0
    pressure_area_ratio: float = 0.0
    raw_intensity_curve: List[float] = field(default_factory=list)
    max_area_curve: List[float] = field(default_factory=list)
    decay_phase_mask: List[bool] = field(default_factory=list)
    decay_tau_ms: Optional[float] = None
    decay_R2: Optional[float] = None
    is_clean_liftoff: bool = False
    touchdown_intensity: float = 0.0
    liftoff_intensity: float = 0.0
    centroid_drift_mm: float = 0.0
    mask_iou_with_peak: List[float] = field(default_factory=list)
    body_axis_at_touchdown_deg: Optional[float] = None
    quality: QualityFlags = field(default_factory=QualityFlags)
    linkage_to_3d: Optional[Linkage3D] = None

    @property
    def n_frames(self) -> int: return len(self.frames)
    @property
    def pressure_curve(self) -> List[float]: return [f.mean_pressure for f in self.frames]
    @property
    def area_curve(self) -> List[float]: return [f.area_mm2 for f in self.frames]
    @property
    def frame_indices(self) -> List[int]: return [f.frame for f in self.frames]


# ═══════════════════════════════════════════════════════════════════════════
# New v0.5.0 contracts
# ═══════════════════════════════════════════════════════════════════════════


# ── mouse body ──────────────────────────────────────────────────────────────

@dataclass
class MouseRoi:
    frame: int
    tight_xyxy: Tuple[int, int, int, int]       # x1, y1, x2, y2
    expanded_xyxy: Tuple[int, int, int, int]     # tight ± pad, clamped
    area_px: int


# ── per-frame record (DB-friendly, no ndarrays) ─────────────────────────────

@dataclass
class FrameRecord:
    frame: int
    time_s: float
    area_mm2: float = 0.0
    area_px: int = 0
    centroid_x_mm: float = 0.0
    centroid_y_mm: float = 0.0
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0
    mean_intensity: float = 0.0
    peak_intensity: float = 0.0
    mean_pressure: float = 0.0
    peak_pressure: float = 0.0
    is_peak_area: bool = False
    is_peak_intensity: bool = False
    png_path: str = ""                   # relative path to per_print/ crop


# ── a single footprint cycle ────────────────────────────────────────────────

@dataclass
class FootprintCycle:
    """A single paw-print from touchdown through peak-area to liftoff.

    *cycle_id* is the 1-based sequential number assigned after sorting all
    cycles by *touchdown_frame* — no limb identity is assigned here.
    """
    cycle_id: int = 0
    touchdown_frame: int = 0
    liftoff_frame: int = 0
    peak_area_frame: int = 0
    peak_intensity_frame: int = 0
    duration_s: float = 0.0
    max_area_mm2: float = 0.0
    max_area_px: int = 0
    centroid_at_peak_x_mm: float = 0.0
    centroid_at_peak_y_mm: float = 0.0
    bbox_at_peak_xyxy: Tuple[int, int, int, int] = (0, 0, 0, 0)
    loading_duration_s: float = 0.0
    weight_bearing_duration_s: float = 0.0
    unloading_duration_s: float = 0.0
    touchdown_intensity: float = 0.0
    liftoff_intensity: float = 0.0
    is_clean_liftoff: bool = False
    n_frames: int = 0
    frames: List[FrameRecord] = field(default_factory=list)

    @property
    def frame_indices(self) -> List[int]:
        return [f.frame for f in self.frames]


# ── trial-level container ───────────────────────────────────────────────────

@dataclass
class TrialResult:
    mouse_id: str
    input_dir: str
    num_frames: int = 0
    frame_width: int = 0
    frame_height: int = 0
    fps: float = 60.0
    px_per_mm: float = 1.92
    roi_pad: int = 50
    tau_paw: float = 10.0
    mouse_rois: List[MouseRoi] = field(default_factory=list)
    cycles: List[FootprintCycle] = field(default_factory=list)


__all__ = [
    "MouseRoi",
    "FrameRecord",
    "FootprintCycle",
    "TrialResult",
]
