"""Shared application state for the deepgait GUI (Layer 5).

The 8 tabs of the main window need to read and write the same data
(gait results, FTIR footprints, 3D pose, calibration matrices). We
centralise the state in a single ``AppState`` object that:

* holds the most recent analysis outputs,
* emits Qt signals when fields change so other tabs can react,
* is QObject-derived so it can be used with Qt's signal/slot mechanism
  across threads (the workers in :mod:`deepgait3.gui.workers` post
  results back through this state).

CLOSED-SOURCE NOTE: this module compiles cleanly with PySide6 (LGPL)
and avoids the GPL-licensed PyQt6 bindings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Signal


# ---------------------------------------------------------------------------
# Plain data containers (no Qt dependency — easy to unit-test)
# ---------------------------------------------------------------------------
@dataclass
class GaitResultsView:
    """Compact view of a gait analysis result for GUI display."""
    csv_path: str = ""
    fps: int = 100
    n_frames: int = 0
    n_strides_per_paw: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    per_paw: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.metrics and not self.per_paw


@dataclass
class FootprintResultsView:
    """Most recent FTIR footprint sequence (per-paw masks + intensity)."""
    sequence_count: int = 0
    mean_area_px: Dict[str, float] = field(default_factory=dict)
    mean_intensity: Dict[str, float] = field(default_factory=dict)


@dataclass
class Pose3DResultsView:
    """Most recent 3D pose result."""
    n_frames: int = 0
    n_bodyparts: int = 0
    reproj_rms_px: float = 0.0
    source: str = ""                              # "inhouse" | "aniposelib"


@dataclass
class CalibrationView:
    """Most recent multi-camera calibration result."""
    cameras: List[str] = field(default_factory=list)
    reproj_rms_per_cam: Dict[str, float] = field(default_factory=dict)
    method: str = ""                              # "inhouse" | "aniposelib"
    board_rows: int = 0
    board_cols: int = 0


@dataclass
class CameraConfigView:
    """W17: Most recent multi-camera configuration (initialization tab).

    One view per camera (keyed by ``role`` + ``serial``); ``AppState`` keeps
    a dict of these so other tabs (FTIR, triangulation, recording) can
    subscribe to the active camera parameters without hard-coding fps.
    """
    role: str = "bottom"          # "left" | "right" | "top" | "bottom"
    serial: str = ""
    width: int = 640
    height: int = 480
    fps: int = 100
    exposure_us: float = 5000.0
    gain_db: float = 0.0
    brightness: int = 0
    contrast: int = 0
    pixel_format: str = "BGR8"
    # ROI: (x, y, width, height). 默认全幅。
    roi: tuple = (0, 0, 640, 480)
    online: bool = False          # 相机是否在线

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "serial": self.serial,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "exposure_us": self.exposure_us,
            "gain_db": self.gain_db,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "pixel_format": self.pixel_format,
            "roi": list(self.roi),
            "online": self.online,
        }


# ---------------------------------------------------------------------------
# QObject — the actual shared state with signals
# ---------------------------------------------------------------------------
class GaitProjectView:
    """Compact view of the current gait project for GUI display.

    (plain data — no Qt dependency)
    """
    def __init__(
        self,
        project_name: str = "",
        project_path: str = "",
        experimenter: str = "",
        px_per_mm: float = 3.0,
    ) -> None:
        self.project_name = project_name
        self.project_path = project_path
        self.experimenter = experimenter
        self.px_per_mm = px_per_mm

    @property
    def is_valid(self) -> bool:
        return bool(self.project_name and self.project_path)


class AppState(QObject):
    """Singleton-style shared state for the main window.

    Signals (Qt semantics, queued across threads automatically):
        * ``gait_results_changed`` — emitted when a new ``GaitResultsView``
          is installed via :meth:`set_gait_results`.
        * ``footprint_changed``  — FTIR footprint update.
        * ``pose_3d_changed``     — 3D triangulation result.
        * ``calibration_changed`` — ChArUco calibration finished.
        * ``project_changed``     — current gait project changed.
        * ``status_message_changed`` — global status bar text.
    """

    gait_results_changed = Signal(object)
    footprint_changed = Signal(object)
    pose_3d_changed = Signal(object)
    calibration_changed = Signal(object)
    project_changed = Signal(object)
    status_message_changed = Signal(str)
    camera_config_changed = Signal(object)   # W17: per-camera config

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._gait: Optional[GaitResultsView] = None
        self._footprint: Optional[FootprintResultsView] = None
        self._pose_3d: Optional[Pose3DResultsView] = None
        self._calibration: Optional[CalibrationView] = None
        self._project: GaitProjectView = GaitProjectView()
        self._camera_configs: Dict[str, CameraConfigView] = {}
        self._status: str = "Ready"

    # ---- getters --------------------------------------------------------
    @property
    def gait_results(self) -> Optional[GaitResultsView]:
        return self._gait

    @property
    def footprint(self) -> Optional[FootprintResultsView]:
        return self._footprint

    @property
    def pose_3d(self) -> Optional[Pose3DResultsView]:
        return self._pose_3d

    @property
    def calibration(self) -> Optional[CalibrationView]:
        return self._calibration

    @property
    def status_message(self) -> str:
        return self._status

    @property
    def camera_configs(self) -> Dict[str, CameraConfigView]:
        return dict(self._camera_configs)

    def get_camera_config(self, role: str) -> Optional[CameraConfigView]:
        return self._camera_configs.get(role)

    @property
    def project(self) -> GaitProjectView:
        return self._project

    # ---- setters (emit signals) -----------------------------------------
    def set_gait_results(self, value: GaitResultsView) -> None:
        self._gait = value
        self.gait_results_changed.emit(value)

    def set_footprint(self, value: FootprintResultsView) -> None:
        self._footprint = value
        self.footprint_changed.emit(value)

    def set_pose_3d(self, value: Pose3DResultsView) -> None:
        self._pose_3d = value
        self.pose_3d_changed.emit(value)

    def set_calibration(self, value: CalibrationView) -> None:
        self._calibration = value
        self.calibration_changed.emit(value)

    def set_camera_config(self, value: CameraConfigView) -> None:
        """Update config for one camera and notify subscribers."""
        self._camera_configs[value.role] = value
        self.camera_config_changed.emit(value)

    def set_project(self, value: GaitProjectView) -> None:
        self._project = value
        self.project_changed.emit(value)

    def set_status_message(self, message: str) -> None:
        self._status = message
        self.status_message_changed.emit(message)

    # ---- helpers --------------------------------------------------------
    def clear(self) -> None:
        """Reset all state (called on file close / new trial)."""
        self._gait = None
        self._footprint = None
        self._pose_3d = None
        self._calibration = None
        self._status = "Ready"
        # Emit one final signal so the UI can refresh.
        self.status_message_changed.emit(self._status)


__all__ = [
    "GaitResultsView",
    "FootprintResultsView",
    "Pose3DResultsView",
    "CalibrationView",
    "CameraConfigView",
    "AppState",
]