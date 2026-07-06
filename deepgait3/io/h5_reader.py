"""HDF5 trial reader for deepgait v2 (Layer 2).

Reads files written by :class:`deepgait3.io.h5_writer.TrialH5Writer` and
exposes them as plain NumPy arrays / Python dicts. The reader is the
*only* safe way to ingest a deepgait trial back into memory — it
validates schema version, coerces dtypes, and hides the on-disk layout
from downstream code (gait algorithms, GUI, exporters).

Designed for both:

* **Post-hoc analysis** — open a finished file and pull all arrays.
* **Live tailing** (when the writer used SWMR) — call :meth:`TrialH5Reader.open`
  with ``swmr=True`` and read growing datasets.

Acceptance gate (DEVELOPMENT_PLAN §6.1 W3):
    "H5 write/read round-trips consistently" — covered by
    ``tests/unit/test_w3_io.py::TestH5RoundTrip``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np

from .h5_writer import SCHEMA_VERSION


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strongly-typed read views
# ---------------------------------------------------------------------------
@dataclass
class Pose3D:
    """In-memory view of ``/pose_3d``."""
    positions: np.ndarray              # (N, 12, 3) float32
    reprojection_error: np.ndarray     # (N, 12) float32
    in_stance: np.ndarray              # (N, 4) bool
    confidence: np.ndarray             # (N, 12) float32


@dataclass
class CameraData:
    """In-memory view of one ``/cameras/<name>`` slot."""
    name: str
    video: Optional[np.ndarray] = None           # (N, H, W, 3) uint8
    pose_2d: Optional[np.ndarray] = None         # (N, 12, 3) float32
    ftir_intensity: Optional[np.ndarray] = None  # (N, H, W) float32
    fps: int = 0
    exposure_us: float = 0.0
    gain_db: float = 0.0


@dataclass
class Trial:
    """In-memory view of a whole trial file."""
    metadata: Dict[str, str] = field(default_factory=dict)
    cameras: Dict[str, CameraData] = field(default_factory=dict)
    pose_3d: Optional[Pose3D] = None
    gait_per_paw: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    coordination: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, float] = field(default_factory=dict)
    calibration: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    audit_log: List[str] = field(default_factory=list)
    audit_hmac_chain: Optional[np.ndarray] = None
    licensing: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# TrialH5Reader
# ---------------------------------------------------------------------------
class TrialH5Reader:
    """Read a deepgait v2 trial HDF5 file back into memory.

    Parameters
    ----------
    path : str | Path
        Trial file produced by :class:`TrialH5Writer`.
    swmr : bool
        If True, open in SWMR-reader mode so the file can be read while
        a writer is still appending. Default False.
    """

    def __init__(self, path: Union[str, Path], swmr: bool = False) -> None:
        self.path = Path(path)
        self._swmr = bool(swmr)
        self._h5: Optional[h5py.File] = None

    # ---- lifecycle --------------------------------------------------------
    def __enter__(self) -> "TrialH5Reader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._h5 is not None:
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"trial file not found: {self.path}")
        self._h5 = h5py.File(self.path, "r", libver="latest", swmr=self._swmr)
        self._check_schema()

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            raise RuntimeError("TrialH5Reader: file not open")
        return self._h5

    def _check_schema(self) -> None:
        ver = self.h5.attrs.get("schema_version")
        if ver is not None and ver.decode() if isinstance(ver, bytes) else ver != SCHEMA_VERSION:
            major = (ver or "").split(".")[0] if ver else ""
            if major != SCHEMA_VERSION.split(".")[0]:
                raise ValueError(
                    f"incompatible schema version: file={ver!r}, expected {SCHEMA_VERSION}. "
                    f"Run deepgait3.io.migration to upgrade."
                )

    # ---- top-level read ---------------------------------------------------
    def read_all(self) -> Trial:
        """Read the entire trial into a :class:`Trial` dataclass."""
        return Trial(
            metadata=self.read_metadata(),
            cameras=self.read_cameras(),
            pose_3d=self.read_pose_3d(),
            gait_per_paw=self.read_gait_per_paw(),
            coordination=self.read_coordination(),
            summary=self.read_summary(),
            calibration=self.read_calibration(),
            audit_log=self.read_audit_log(),
            audit_hmac_chain=self.read_audit_hmac_chain(),
            licensing=self.read_licensing(),
        )

    # ---- /metadata --------------------------------------------------------
    def read_metadata(self) -> Dict[str, str]:
        if "metadata" not in self.h5:
            return {}
        out = {}
        for k, v in self.h5["metadata"].attrs.items():
            out[k] = v.decode() if isinstance(v, bytes) else str(v)
        return out

    # ---- /cameras ---------------------------------------------------------
    def read_cameras(self) -> Dict[str, CameraData]:
        out: Dict[str, CameraData] = {}
        if "cameras" not in self.h5:
            return out
        for name in self.h5["cameras"]:
            out[name] = self.read_camera(name)
        return out

    def read_camera(self, name: str) -> CameraData:
        cam_path = f"cameras/{name}"
        if cam_path not in self.h5:
            raise KeyError(f"camera not found: {name}")
        cam_grp = self.h5[cam_path]
        cd = CameraData(name=name)
        if "video" in cam_grp:
            cd.video = cam_grp["video"][...]
        if "pose_2d" in cam_grp:
            cd.pose_2d = cam_grp["pose_2d"][...]
        if "ftir_intensity" in cam_grp:
            cd.ftir_intensity = cam_grp["ftir_intensity"][...]
        meta = cam_grp.get("metadata")
        if meta is not None:
            cd.fps = int(meta.attrs.get("fps", 0))
            cd.exposure_us = float(meta.attrs.get("exposure_us", 0.0))
            cd.gain_db = float(meta.attrs.get("gain_db", 0.0))
        return cd

    # ---- /pose_3d ---------------------------------------------------------
    def read_pose_3d(self) -> Optional[Pose3D]:
        if "pose_3d" not in self.h5:
            return None
        g = self.h5["pose_3d"]
        return Pose3D(
            positions=g["positions"][...],
            reprojection_error=g["reprojection_error"][...],
            in_stance=g["in_stance"][...],
            confidence=g["confidence"][...],
        )

    # ---- /gait_metrics ----------------------------------------------------
    def read_gait_per_paw(self) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        path = "gait_metrics/per_paw"
        if path not in self.h5:
            return out
        for paw in self.h5[path]:
            paw_grp = self.h5[f"{path}/{paw}"]
            out[paw] = {k: paw_grp[k][...] for k in paw_grp.keys()}
        return out

    def read_coordination(self) -> Dict[str, float]:
        path = "gait_metrics/coordination"
        if path not in self.h5:
            return {}
        return {k: float(v) for k, v in self.h5[path].attrs.items()}

    def read_summary(self) -> Dict[str, float]:
        path = "gait_metrics/summary"
        if path not in self.h5:
            return {}
        return {k: float(v) for k, v in self.h5[path].attrs.items()}

    # ---- /calibration -----------------------------------------------------
    def read_calibration(self) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        path = "calibration/cameras"
        if path not in self.h5:
            return out
        for cam_name in self.h5[path]:
            cam_grp = self.h5[f"{path}/{cam_name}"]
            out[cam_name] = {k: cam_grp[k][...] for k in cam_grp.keys()}
        return out

    # ---- /audit -----------------------------------------------------------
    def read_audit_log(self) -> List[str]:
        if "audit/log" not in self.h5:
            return []
        raw = self.h5["audit/log"][...]
        return [s.decode() if isinstance(s, bytes) else str(s) for s in raw]

    def read_audit_hmac_chain(self) -> Optional[np.ndarray]:
        if "audit/hmac_chain" not in self.h5:
            return None
        return self.h5["audit/hmac_chain"][...]

    # ---- /licensing -------------------------------------------------------
    def read_licensing(self) -> Dict[str, object]:
        if "licensing" not in self.h5:
            return {}
        attrs = self.h5["licensing"].attrs
        out: Dict[str, object] = {}
        for k, v in attrs.items():
            if isinstance(v, bytes):
                out[k] = v.decode()
            else:
                out[k] = v
        return out

    def __repr__(self) -> str:
        state = "open" if self._h5 is not None else "closed"
        return f"TrialH5Reader(path={self.path!s}, {state})"
