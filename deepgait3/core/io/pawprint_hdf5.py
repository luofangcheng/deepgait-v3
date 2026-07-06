"""HDF5 multi-dimensional PawPrint database.

See ``docs/PAW_DATABASE.md`` for the schema rationale.

Public API:
- ``PawPrintH5Writer``: write one trial's PawPrint[] to a single .h5 file.
- ``PawPrintH5Reader``: read back individual prints, timeseries, or images.
- ``TrialMeta``: dataclass for the text/numeric trial-level metadata.

Why HDF5:
- Memory-mapped access: open <100 ms, per-print read <10 ms.
- Image + numeric + text unified under one file.
- Reuses v1's existing h5py stack (``deepgait/io/h5_writer.py``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from deepgait3.core.pawprint import FrameData, PawPrint


# ---------------------------------------------------------------------------
# Trial-level metadata
# ---------------------------------------------------------------------------

@dataclass
class TrialMeta:
    """Text + scalar numeric attributes for one trial.

    Persisted as HDF5 root attributes (scalar strings/floats/ints).
    """
    trial_id: str
    animal_id: str = ""
    treatment: str = ""
    fps: int = 60
    px_per_mm: float = 1.92
    walkway_length_mm: float = 1000.0
    ftir_video_path: str = ""
    timestamp: str = ""
    notes: str = ""

    def to_h5(self, f: h5py.File) -> None:
        for k, v in asdict(self).items():
            f.attrs[k] = v


@dataclass
class DetectorInfo:
    """Provenance: which detector produced this trial's prints."""
    algorithm: str = "unknown"
    threshold: float = 0.0
    min_area_px: int = 0
    sensitivity_mode: str = ""

    def to_h5(self, f: h5py.File) -> None:
        grp = f.require_group("/detector")
        for k, v in asdict(self).items():
            grp.attrs[k] = v


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class PawPrintH5Writer:
    """Write a list of ``PawPrint`` to a single ``.h5`` file.

    Usage::

        with PawPrintH5Writer("trial.h5", TrialMeta(trial_id="...")) as w:
            w.set_detector(DetectorInfo(algorithm="exg", threshold=80))
            for pp in pawprints:
                w.add_print(pp)
        # file is closed on context exit
    """

    def __init__(self, path: str | Path, meta: TrialMeta,
                 compression: str = "gzip", compression_level: int = 4):
        self.path = Path(path)
        self.meta = meta
        self.compression = compression
        self.comp_level = compression_level
        self._f: h5py.File | None = None
        self._print_grp: h5py.Group | None = None
        self._index_grp: h5py.Group | None = None
        self._index_bufs: dict[str, list] = {}

    def __enter__(self) -> "PawPrintH5Writer":
        self._f = h5py.File(self.path, "w")
        self.meta.to_h5(self._f)
        self._print_grp = self._f.require_group("/pawprints")
        self._index_grp = self._f.require_group("/index")
        for key in (
            "print_ids", "paw_ids", "touchdown_frames", "liftoff_frames",
            "peak_frames", "max_areas_mm2", "durations_s",
            "print_centroids_x_mm",
        ):
            self._index_bufs[key] = []
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Flush index buffers into flat HDF5 datasets before closing.
        if self._f is not None and self._index_grp is not None:
            for k, v in self._index_bufs.items():
                if not v:
                    continue
                if k == "paw_ids":
                    # h5py requires fixed-length string dtype; pad to length 4.
                    arr = np.asarray(v, dtype="S4")
                else:
                    arr = np.asarray(v)
                ds = self._index_grp.create_dataset(
                    k, data=arr,
                    compression=self.compression,
                    compression_opts=self.comp_level,
                )
                ds.attrs["description"] = f"flat index of {k}"
        if self._f is not None:
            self._f.close()
        self._f = None

    # ------------------------------------------------------------------
    def set_detector(self, info: DetectorInfo) -> None:
        assert self._f is not None, "writer not entered"
        info.to_h5(self._f)

    def add_print(self, pp: PawPrint) -> None:
        assert self._f is not None, "writer not entered"
        grp = self._print_grp.require_group(f"print_{pp.print_id:03d}")
        self._write_meta(grp, pp)
        self._write_timeseries(grp, pp)
        self._write_images(grp, pp)
        self._write_quality(grp, pp)
        # Update flat index
        self._index_bufs["print_ids"].append(pp.print_id)
        self._index_bufs["paw_ids"].append(
            pp.linkage_to_3d.paw_id if pp.linkage_to_3d else ""
        )
        self._index_bufs["touchdown_frames"].append(pp.touchdown_frame)
        self._index_bufs["liftoff_frames"].append(pp.liftoff_frame)
        self._index_bufs["peak_frames"].append(pp.peak_area_frame)
        self._index_bufs["max_areas_mm2"].append(float(pp.max_area_mm2))
        self._index_bufs["durations_s"].append(float(pp.duration_s))
        self._index_bufs["print_centroids_x_mm"].append(
            float(pp.peak_frame_centroid_xy_mm[0])
        )

    # ------------------------------------------------------------------
    def _write_meta(self, grp: h5py.Group, pp: PawPrint) -> None:
        m = grp.require_group("meta")
        m.attrs["print_id"] = pp.print_id
        m.attrs["touchdown_frame"] = pp.touchdown_frame
        m.attrs["liftoff_frame"] = pp.liftoff_frame
        m.attrs["true_liftoff_frame"] = pp.true_liftoff_frame
        m.attrs["peak_area_frame"] = pp.peak_area_frame
        m.attrs["peak_intensity_frame"] = pp.peak_intensity_frame
        m.attrs["duration_s"] = float(pp.duration_s)
        m.attrs["time_to_peak_area_s"] = float(pp.time_to_peak_area_s)
        m.attrs["time_to_peak_intensity_s"] = float(pp.time_to_peak_intensity_s)
        m.attrs["loading_duration_s"] = float(pp.loading_duration_s)
        m.attrs["weight_bearing_duration_s"] = float(pp.weight_bearing_duration_s)
        m.attrs["unloading_duration_s"] = float(pp.unloading_duration_s)
        m.attrs["max_area_mm2"] = float(pp.max_area_mm2)
        m.attrs["peak_frame_centroid_x_mm"] = float(pp.peak_frame_centroid_xy_mm[0])
        m.attrs["peak_frame_centroid_y_mm"] = float(pp.peak_frame_centroid_xy_mm[1])
        m.attrs["peak_frame_bbox_xyxy"] = np.asarray(pp.peak_frame_bbox_xyxy, dtype=np.int32)
        m.attrs["print_length_mm"] = float(pp.print_length_mm)
        m.attrs["print_width_mm"] = float(pp.print_width_mm)
        m.attrs["print_orientation_deg"] = float(pp.print_orientation_deg)
        m.attrs["compactness"] = float(pp.compactness)
        m.attrs["centroid_drift_mm"] = float(pp.centroid_drift_mm)
        m.attrs["stand_index"] = float(pp.stand_index)
        m.attrs["rising_slope"] = float(pp.rising_slope)
        m.attrs["peak_pressure"] = float(pp.peak_pressure)
        m.attrs["mean_pressure_at_peak"] = float(pp.mean_pressure_at_peak)
        m.attrs["pressure_area_ratio"] = float(pp.pressure_area_ratio)
        m.attrs["touchdown_intensity"] = float(pp.touchdown_intensity)
        m.attrs["liftoff_intensity"] = float(pp.liftoff_intensity)
        m.attrs["cop_path_length_mm"] = float(pp.cop_path_length_mm)
        m.attrs["cop_displacement_mm"] = float(pp.cop_displacement_mm)
        # Optional decay fields — write even if None
        m.attrs["decay_tau_ms"] = float(pp.decay_tau_ms) if pp.decay_tau_ms is not None else float("nan")
        m.attrs["decay_R2"] = float(pp.decay_R2) if pp.decay_R2 is not None else float("nan")
        m.attrs["is_clean_liftoff"] = bool(pp.is_clean_liftoff)
        # Stage 2 fills these
        m.attrs["paw_id"] = pp.linkage_to_3d.paw_id if pp.linkage_to_3d else ""
        m.attrs["match_distance_mm"] = (
            float(pp.linkage_to_3d.match_distance_mm)
            if pp.linkage_to_3d and pp.linkage_to_3d.match_distance_mm is not None
            else float("nan")
        )

    def _write_timeseries(self, grp: h5py.Group, pp: PawPrint) -> None:
        ts = grp.require_group("timeseries")
        n = pp.n_frames
        # Allocate numpy arrays from FrameData lists
        frame_idx = np.array([f.frame for f in pp.frames], dtype=np.int32)
        time_s = np.array([f.time_s for f in pp.frames], dtype=np.float32)
        area_mm2 = np.array([f.area_mm2 for f in pp.frames], dtype=np.float32)
        mean_p = np.array([f.mean_pressure for f in pp.frames], dtype=np.float32)
        mean_i = np.array([f.mean_intensity_in_mask for f in pp.frames], dtype=np.float32)
        peak_i = np.array([f.peak_intensity for f in pp.frames], dtype=np.float32)
        cop_x = np.array([c[0] for c in pp.cop_trajectory_mm], dtype=np.float32) if pp.cop_trajectory_mm else np.zeros(n, dtype=np.float32)
        cop_y = np.array([c[1] for c in pp.cop_trajectory_mm], dtype=np.float32) if pp.cop_trajectory_mm else np.zeros(n, dtype=np.float32)
        decay_phase = np.asarray(pp.decay_phase_mask, dtype=bool) if pp.decay_phase_mask else np.zeros(n, dtype=bool)
        max_area_curve = np.asarray(pp.max_area_curve, dtype=np.float32) if pp.max_area_curve else np.zeros(n, dtype=np.float32)
        comp = self.compression
        lvl = self.comp_level
        for name, arr in [
            ("frame_idx", frame_idx), ("time_s", time_s),
            ("area_mm2", area_mm2), ("mean_pressure", mean_p),
            ("mean_intensity", mean_i), ("peak_intensity", peak_i),
            ("cop_x_mm", cop_x), ("cop_y_mm", cop_y),
            ("decay_phase", decay_phase), ("max_area_curve", max_area_curve),
        ]:
            ts.create_dataset(name, data=arr, compression=comp, compression_opts=lvl)

    def _write_images(self, grp: h5py.Group, pp: PawPrint) -> None:
        """Write peak-frame image, mask, pressure map.

        If no frames are stored, skip the images group.
        """
        if pp.n_frames == 0:
            return
        # Find the peak-area frame
        peak_idx = next(
            (i for i, f in enumerate(pp.frames) if f.frame == pp.peak_area_frame),
            0,
        )
        peak_fd = pp.frames[peak_idx]
        img = grp.require_group("images")
        comp = self.compression
        lvl = self.comp_level
        img.create_dataset(
            "peak_frame_bgr", data=peak_fd.raw_intensity_crop.astype(np.float32),
            compression=comp, compression_opts=lvl,
        )
        # mask + pressure_map are per-frame but typically same shape
        if peak_fd.paw_mask.shape == peak_fd.raw_intensity_crop.shape:
            img.create_dataset(
                "peak_mask", data=peak_fd.paw_mask.astype(bool),
                compression=comp, compression_opts=lvl,
            )
            img.create_dataset(
                "peak_pressure_map", data=peak_fd.pressure_map.astype(np.float32),
                compression=comp, compression_opts=lvl,
            )

    def _write_quality(self, grp: h5py.Group, pp: PawPrint) -> None:
        q = grp.require_group("quality")
        q.attrs["touches_edge"] = bool(pp.quality.touches_edge)
        q.attrs["merged_with_neighbor"] = bool(pp.quality.merged_with_neighbor)
        q.attrs["n_frames"] = int(pp.quality.n_frames)
        q.attrs["min_area_below_thresh"] = bool(pp.quality.min_area_below_thresh)
        q.attrs["saturated_pixels_pct"] = float(pp.quality.saturated_pixels_pct)
        q.attrs["snr"] = float(pp.quality.snr)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

class PawPrintH5Reader:
    """Read a ``.h5`` trial back.

    Supports both per-print access (``reader.print(7)``) and streaming
    over all prints (``for pp_meta in reader``).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._f = h5py.File(self.path, "r")

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self) -> "PawPrintH5Reader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def meta(self) -> dict:
        return {k: self._f.attrs[k] for k in self._f.attrs}

    @property
    def detector(self) -> dict:
        if "/detector" not in self._f:
            return {}
        return {k: self._f["/detector"].attrs[k] for k in self._f["/detector"].attrs}

    @property
    def print_ids(self) -> np.ndarray:
        return self._f["/index/print_ids"][:]

    def __iter__(self) -> Iterator[dict]:
        for pid in self.print_ids:
            yield self.print_meta(int(pid))

    def print_meta(self, print_id: int) -> dict:
        """Return all scalar metadata for one print as a flat dict."""
        grp = self._f[f"/pawprints/print_{print_id:03d}"]
        out: dict = {}
        for k, v in grp["meta"].attrs.items():
            out[k] = v
        return out

    def timeseries(self, print_id: int, key: str) -> np.ndarray:
        return self._f[f"/pawprints/print_{print_id:03d}/timeseries/{key}"][:]

    def image(self, print_id: int, key: str) -> np.ndarray:
        return self._f[f"/pawprints/print_{print_id:03d}/images/{key}"][:]

    def flat_index(self, key: str) -> np.ndarray:
        return self._f[f"/index/{key}"][:]


__all__ = [
    "TrialMeta",
    "DetectorInfo",
    "PawPrintH5Writer",
    "PawPrintH5Reader",
]