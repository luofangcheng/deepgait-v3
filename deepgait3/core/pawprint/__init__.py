"""DeepGait v3 — Stage 1: fTIR footprint extraction.

New pipeline (2026-06-27):
  1. Mouse body detection → expanded ROI
  2. Low-threshold blob detection (tau_paw=10) within ROI
  3. Union-find clustering → IoU tracking → footprint cycles
  4. SQLite database + cumulative visualisations

Underlying blob/cluster/track primitives migrated from
``deepgait-v2/dynamics_v0.4.2/dynamics_v04_module/dynamics_v04/``.
"""

from .models import MouseRoi, FrameRecord, FootprintCycle, TrialResult
from .detection import detect_blobs
from .grouping import cluster_blobs_into_feet
from .tracker import IoUFootprintTracker, FootprintTrack
from .mouse_detector import MouseDetector
from .cycle_builder import build_cycles
from .db import create_db, save_trial
from .pipeline import Stage1Pipeline, DEFAULTS
from .single_frame import detect_single_frame

# v0.4.2 schema kept for Stage 2 wiring
from .models import FrameData, QualityFlags, Linkage3D, PawPrint, FootMask
from .extractor import PawPrintExtractor

__version__ = "0.5.0"
__all__ = [
    "MouseRoi", "FrameRecord", "FootprintCycle", "TrialResult",
    "MouseDetector", "build_cycles", "create_db", "save_trial",
    "Stage1Pipeline", "DEFAULTS",
    "detect_blobs", "cluster_blobs_into_feet",
    "IoUFootprintTracker", "FootprintTrack",
    "detect_single_frame",
    "FrameData", "QualityFlags", "Linkage3D", "PawPrint", "FootMask",
    "PawPrintExtractor",
]
