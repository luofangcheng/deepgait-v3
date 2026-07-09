"""DeepGait v3 — Stage 1: fTIR footprint extraction (YOLOv8-seg).

Pipeline::

    1. YOLOv8n-seg batch inference (GPU) → FootMask per frame
    2. IoUFootprintTracker → FootprintTrack
    3. build_cycles → FootprintCycle
    4. SQLite database + cumulative visualisations
"""

from .models import MouseRoi, FrameRecord, FootprintCycle, TrialResult
from .tracker import IoUFootprintTracker, FootprintTrack
from .mouse_detector import MouseDetector
from .cycle_builder import build_cycles
from .db import create_db, save_trial
from .pipeline import Stage1Pipeline, DEFAULTS
from .single_frame import detect_single_frame
from .yolo_detector import YoloPawDetector

# v0.4.2 schema kept for Stage 2 wiring
from .models import FrameData, QualityFlags, Linkage3D, PawPrint, FootMask

from .cumulative import build_cumulative_union, build_cumulative_union_cpu, render_overlay
from .trim import TrimError, find_trim_range, trim_video

__version__ = "0.6.0"
__all__ = [
    "MouseRoi", "FrameRecord", "FootprintCycle", "TrialResult",
    "MouseDetector", "build_cycles", "create_db", "save_trial",
    "Stage1Pipeline", "DEFAULTS",
    "IoUFootprintTracker", "FootprintTrack",
    "detect_single_frame", "YoloPawDetector",
    "FrameData", "QualityFlags", "Linkage3D", "PawPrint", "FootMask",
    "build_cumulative_union", "build_cumulative_union_cpu", "render_overlay",
    "TrimError", "find_trim_range", "trim_video",
]
