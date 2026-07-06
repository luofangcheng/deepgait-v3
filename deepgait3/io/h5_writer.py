"""HDF5 trial writer for deepgait v2 (Layer 2).

Implements the on-disk schema documented in ``kb/23_data_formats.md §2.2``.
The writer is the *single* entry point the recording pipeline uses to
serialise one trial to disk. Designed for incremental, real-time use:

* Datasets are created with explicit ``chunks`` and ``maxshape`` so the
  writer can ``append`` frames as they arrive from the camera bus.
* Optional SWMR (Single Writer / Multiple Reader) mode is enabled with
  ``swmr=True`` so an analyzer process can read growing files live
  (DEVELOPMENT_PLAN §5.2 ``io/h5_writer.py``).
* Raw video is written *only when explicitly requested* — most trials
  store just pose + metrics, which keeps the per-trial file at a few MB
  (per audit C2: H5 holds key info; raw video goes to a separate
  recording path). See ``write_camera_video`` docstring.

Acceptance gate (DEVELOPMENT_PLAN §6.1 W3):
    "H5 write/read round-trips consistently" — covered by
    ``tests/unit/test_w3_io.py::TestH5RoundTrip``.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import h5py
import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — mirror kb/23_data_formats.md §2.2
# ---------------------------------------------------------------------------
APP_VERSION = "2.0.0"
SCHEMA_VERSION = "2.0"

# Standard chunk shapes — chosen so each chunk ≈ 1 MB.
CHUNK_POSE_2D = (1000, 12, 3)
CHUNK_POSE_3D_POS = (1000, 12, 3)
CHUNK_POSE_3D_REPROJ = (1000, 12)
CHUNK_STANCE = (1000, 4)
CHUNK_CONF = (1000, 12)

COMPRESSION = "gzip"
COMPRESSION_OPTS = 6


def _clamp_chunk(chunk: tuple, shape: tuple) -> tuple:
    """Clamp a chunk shape so no axis exceeds the actual data shape.

    h5py rejects chunks larger than the dataset shape, so for small arrays
    (e.g. unit tests with 3 frames) we shrink the leading axis to fit.
    Production data (thousands of frames) uses the full chunk size.
    """
    return tuple(min(c, s) for c, s in zip(chunk, shape))


# ---------------------------------------------------------------------------
# Metadata spec
# ---------------------------------------------------------------------------
class TrialMetadata:
    """Strongly-typed container for the per-trial ``/metadata`` group."""

    __slots__ = (
        "animal_id", "species", "strain", "genotype",
        "experiment_date", "operator", "device_serial",
        "app_version", "config_hash", "schema_version",
    )

    def __init__(
        self,
        *,
        animal_id: str,
        species: str = "mouse",
        strain: str = "",
        genotype: str = "",
        experiment_date: Optional[str] = None,
        operator: str = "",
        device_serial: str = "",
        config_hash: str = "",
        app_version: str = APP_VERSION,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.animal_id = animal_id
        self.species = species
        self.strain = strain
        self.genotype = genotype
        self.experiment_date = experiment_date or datetime.now().isoformat()
        self.operator = operator
        self.device_serial = device_serial
        self.config_hash = config_hash
        self.app_version = app_version
        self.schema_version = schema_version

    def to_attrs(self) -> Dict[str, str]:
        return {
            "animal_id": self.animal_id,
            "species": self.species,
            "strain": self.strain,
            "genotype": self.genotype,
            "experiment_date": self.experiment_date,
            "operator": self.operator,
            "device_serial": self.device_serial,
            "app_version": self.app_version,
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
        }


# ---------------------------------------------------------------------------
# TrialH5Writer
# ---------------------------------------------------------------------------
class TrialH5Writer:
    """Serialise one trial to an HDF5 file (schema v2.0).

    Parameters
    ----------
    path : str | Path
        Destination file. Parent directories are created as needed.
    swmr : bool
        If True, open in SWMR-capable mode so a separate reader process
        can observe the file as it grows. Default False (most tests and
        the GUI run single-process).

    Example
    -------
    >>> with TrialH5Writer("trial_001.h5") as w:
    ...     w.write_metadata(animal_id="M001", operator="alice")
    ...     w.write_pose_2d("left", pose_2d_array)
    """

    def __init__(self, path: Union[str, Path], swmr: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._swmr = bool(swmr)
        self._h5: Optional[h5py.File] = None
        self._closed = False

    # ---- lifecycle --------------------------------------------------------
    def __enter__(self) -> "TrialH5Writer":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._h5 is not None:
            return
        # libver='latest' is required for SWMR support.
        self._h5 = h5py.File(
            self.path, "w", libver="latest",
        )
        # Stamp the file with a top-level schema version for migration.py.
        self._h5.attrs["schema_version"] = SCHEMA_VERSION
        self._h5.attrs["created_at"] = datetime.now().isoformat()
        if self._swmr:
            try:
                self._h5.swmr_mode = True
            except Exception:
                # SWMR is only relevant for live readers; degrade silently.
                logger.debug("SWMR mode not active (single-process writer)")

    def close(self) -> None:
        if self._h5 is None or self._closed:
            return
        try:
            self._h5.flush()
        finally:
            self._h5.close()
            self._h5 = None
            self._closed = True

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            raise RuntimeError("TrialH5Writer: file not open")
        return self._h5

    # ---- /metadata --------------------------------------------------------
    def write_metadata(
        self,
        meta: Union[TrialMetadata, Dict, None] = None,
        *,
        animal_id: Optional[str] = None,
        species: str = "mouse",
        strain: str = "",
        genotype: str = "",
        experiment_date: Optional[str] = None,
        operator: str = "",
        device_serial: str = "",
        config_hash: str = "",
        app_version: str = APP_VERSION,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        """Write the file-level ``/metadata`` group.

        Accepts three calling conventions:

        1. ``write_metadata(TrialMetadata(...))`` — explicit object.
        2. ``write_metadata({...})`` — dict of the same keys.
        3. ``write_metadata(animal_id=..., operator=..., ...)`` — kwargs
           (matches the signature sketched in kb/23_data_formats.md §2.3).
        """
        if meta is None:
            meta = TrialMetadata(
                animal_id=animal_id or "", species=species, strain=strain,
                genotype=genotype, experiment_date=experiment_date,
                operator=operator, device_serial=device_serial,
                config_hash=config_hash, app_version=app_version,
                schema_version=schema_version,
            )
        elif isinstance(meta, dict):
            meta = TrialMetadata(**meta)
        grp = self.h5.create_group("metadata")
        for k, v in meta.to_attrs().items():
            grp.attrs[k] = v

    # ---- /cameras/<name>/{video, pose_2d, ftir_intensity} -----------------
    def write_camera_video(
        self,
        camera_name: str,
        frames: np.ndarray,
        fps: int,
        exposure_us: float,
        gain_db: float = 0.0,
    ) -> None:
        """Write the raw video for one camera.

        NOTE: raw video is the dominant storage cost (~1.2 MB/frame at
        4MP BGR8, 4 cameras x 100 fps = ~480 MB/s). Per audit item C2
        (NFR-PERF-006 infeasibility), H5 holds pose + metrics only by
        default; this method is reserved for short calibration snippets
        or trials where the user explicitly requests H5 video. The
        production recording pipeline writes raw video to a separate
        SSD-RAID path via opencv VideoWriter.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"video must be (N, H, W, 3), got shape {frames.shape}"
            )
        cam = self.h5.require_group(f"cameras/{camera_name}")
        chunk = (min(100, frames.shape[0]), *frames.shape[1:])
        cam.create_dataset(
            "video", data=frames.astype(np.uint8),
            chunks=chunk, compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
        )
        self._write_camera_metadata(cam, fps, exposure_us, gain_db)

    def write_pose_2d(self, camera_name: str, pose: np.ndarray) -> None:
        """Write 2D pose for one camera. Shape (N, 12, 3) float32."""
        if pose.ndim != 3 or pose.shape[1:] != (12, 3):
            raise ValueError(
                f"pose_2d must be (N, 12, 3), got shape {pose.shape}"
            )
        cam = self.h5.require_group(f"cameras/{camera_name}")
        cam.create_dataset(
            "pose_2d", data=pose.astype(np.float32),
            chunks=_clamp_chunk(CHUNK_POSE_2D, pose.shape),
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )

    def write_ftir_intensity(
        self, camera_name: str, intensity: np.ndarray, fps: int,
        exposure_us: float, gain_db: float = 0.0,
    ) -> None:
        """Write the FTIR intensity map. Shape (N, H, W) float32."""
        if intensity.ndim != 3:
            raise ValueError(
                f"ftir_intensity must be (N, H, W), got shape {intensity.shape}"
            )
        cam = self.h5.require_group(f"cameras/{camera_name}")
        chunk = (min(100, intensity.shape[0]), *intensity.shape[1:])
        cam.create_dataset(
            "ftir_intensity", data=intensity.astype(np.float32),
            chunks=chunk, compression=COMPRESSION,
            compression_opts=COMPRESSION_OPTS,
        )
        self._write_camera_metadata(cam, fps, exposure_us, gain_db)

    def _write_camera_metadata(
        self, cam_grp: h5py.Group, fps: int, exposure_us: float, gain_db: float,
    ) -> None:
        meta = cam_grp.require_group("metadata")
        meta.attrs["fps"] = int(fps)
        meta.attrs["exposure_us"] = float(exposure_us)
        meta.attrs["gain_db"] = float(gain_db)

    # ---- /pose_3d ---------------------------------------------------------
    def write_pose_3d(
        self,
        positions: np.ndarray,
        reprojection_error: np.ndarray,
        in_stance: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        """Write the triangulated 3D pose stream.

        Parameters
        ----------
        positions : (N, 12, 3) float32
        reprojection_error : (N, 12) float32
        in_stance : (N, 4) bool  (LF, RF, LH, RH)
        confidence : (N, 12) float32
        """
        if positions.ndim != 3 or positions.shape[1:] != (12, 3):
            raise ValueError(f"positions must be (N, 12, 3), got {positions.shape}")
        if reprojection_error.shape != (positions.shape[0], 12):
            raise ValueError(
                f"reprojection_error must be (N, 12), got {reprojection_error.shape}"
            )
        if in_stance.shape != (positions.shape[0], 4):
            raise ValueError(f"in_stance must be (N, 4), got {in_stance.shape}")
        if confidence.shape != (positions.shape[0], 12):
            raise ValueError(f"confidence must be (N, 12), got {confidence.shape}")

        g = self.h5.require_group("pose_3d")
        g.create_dataset(
            "positions", data=positions.astype(np.float32),
            chunks=_clamp_chunk(CHUNK_POSE_3D_POS, positions.shape),
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )
        g.create_dataset(
            "reprojection_error", data=reprojection_error.astype(np.float32),
            chunks=_clamp_chunk(CHUNK_POSE_3D_REPROJ, reprojection_error.shape),
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )
        g.create_dataset(
            "in_stance", data=in_stance.astype(np.bool_),
            chunks=_clamp_chunk(CHUNK_STANCE, in_stance.shape),
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )
        g.create_dataset(
            "confidence", data=confidence.astype(np.float32),
            chunks=_clamp_chunk(CHUNK_CONF, confidence.shape),
            compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
        )

    # ---- /gait_metrics ----------------------------------------------------
    def write_gait_metrics(
        self,
        per_paw: Dict[str, Dict[str, np.ndarray]],
        coordination: Optional[Dict[str, float]] = None,
        summary: Optional[Dict[str, float]] = None,
    ) -> None:
        """Write the per-paw gait metrics, coordination indices, and summary.

        Parameters
        ----------
        per_paw : dict
            ``{"LF": {"stance_duration": ndarray, "swing_duration": ndarray, ...}, ...}``
        coordination : dict, optional
            Trial-level coordination indices (regularity_index, etc.).
        summary : dict, optional
            Trial-level aggregates (mean_stride_length, n_strides, speed_cm_s).
        """
        g = self.h5.require_group("gait_metrics")
        per_paw_grp = g.require_group("per_paw")
        for paw, metrics in per_paw.items():
            paw_grp = per_paw_grp.require_group(paw)
            for name, values in metrics.items():
                arr = np.asarray(values)
                paw_grp.create_dataset(
                    name, data=arr,
                    compression=COMPRESSION, compression_opts=COMPRESSION_OPTS,
                )
        if coordination:
            coord_grp = g.require_group("coordination")
            for k, v in coordination.items():
                coord_grp.attrs[k] = float(v)
        if summary:
            summ_grp = g.require_group("summary")
            for k, v in summary.items():
                summ_grp.attrs[k] = float(v)

    # ---- /calibration -----------------------------------------------------
    def write_calibration(
        self,
        cameras: Dict[str, Dict[str, np.ndarray]],
        charuco: Optional[Dict] = None,
        reprojection_error_rms: Optional[float] = None,
    ) -> None:
        """Write per-camera intrinsics/extrinsics.

        Parameters
        ----------
        cameras : dict
            ``{"left": {"K": (3,3), "dist": (5,), "rvec": (3,), "tvec": (3,)}, ...}``
        """
        calib = self.h5.require_group("calibration")
        cams_grp = calib.require_group("cameras")
        for cam_name, params in cameras.items():
            cam_grp = cams_grp.require_group(cam_name)
            for k, v in params.items():
                cam_grp.create_dataset(k, data=np.asarray(v))
        if charuco:
            board = calib.require_group("charuco_board")
            for k, v in charuco.items():
                board.attrs[k] = int(v) if isinstance(v, int) else v
        if reprojection_error_rms is not None:
            calib.attrs["reprojection_error_rms"] = float(reprojection_error_rms)

    # ---- /audit -----------------------------------------------------------
    def write_audit(self, log_lines: Iterable[str], hmac_chain: np.ndarray) -> None:
        """Write the HMAC-chained audit log (research integrity)."""
        audit = self.h5.require_group("audit")
        audit.create_dataset(
            "log", data=np.array(list(log_lines), dtype=h5py.string_dtype()),
        )
        audit.create_dataset("hmac_chain", data=np.asarray(hmac_chain))

    # ---- /licensing -------------------------------------------------------
    def write_licensing(
        self,
        dongle_id: str,
        license_type: str,
        features_bitmap: int,
        last_check_timestamp: str,
    ) -> None:
        lic = self.h5.require_group("licensing")
        lic.attrs["dongle_id"] = dongle_id
        lic.attrs["license_type"] = license_type
        lic.attrs["features_bitmap"] = int(features_bitmap)
        lic.attrs["last_check_timestamp"] = last_check_timestamp

    def __repr__(self) -> str:
        state = "open" if self._h5 is not None else "closed"
        return f"TrialH5Writer(path={self.path!s}, {state})"
