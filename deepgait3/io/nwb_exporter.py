"""NWB:N 2.5 exporter for deepgait v2 (Layer 2).

Exports a deepgait trial to the `Neurodata Without Borders
<https://www.nwb.org/>`_ format so the trial can be ingested by DANDI
and NWB-compatible neuroscience tooling. Per DEVELOPMENT_PLAN §5.2
(``io/nwb_exporter.py``) the implementation uses ``pynwb``.

If ``pynwb`` is not installed (developer laptop, CI, or a shipping
build that omits NWB support), :class:`NwbExporter.export` raises
``NwbUnavailable`` with a clear installation hint instead of crashing
on import. This lets :mod:`deepgait3.io` be imported anywhere.

Acceptance gate (DEVELOPMENT_PLAN §6.1 W3): exporters must run end-to-end.
Verified by ``tests/unit/test_w3_io.py::TestNwbExporter`` (skipped when
pynwb is unavailable).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pynwb availability probe
# ---------------------------------------------------------------------------
def _pynwb_available() -> bool:
    try:
        import pynwb  # noqa: F401
        return True
    except Exception:
        return False


PYNWB_AVAILABLE = _pynwb_available()


class NwbUnavailable(RuntimeError):
    """Raised when the user requests NWB export but pynwb is missing."""


# ---------------------------------------------------------------------------
# NwbExportSpec
# ---------------------------------------------------------------------------
@dataclass
class NwbExportSpec:
    """Inputs needed for an NWB export."""
    animal_id: str
    species: str                          # "mouse" or "rat"
    strain: str
    operator: str
    experiment_date: datetime
    trial_id: str
    fps: int
    pose_3d_positions: Optional[np.ndarray] = None     # (N, 12, 3) mm
    pose_3d_confidence: Optional[np.ndarray] = None    # (N, 12)
    in_stance: Optional[np.ndarray] = None             # (N, 4) bool
    gait_summary: Dict[str, float] = field(default_factory=dict)
    session_description: str = "deepgait v2 gait recording"


# ---------------------------------------------------------------------------
# NwbExporter
# ---------------------------------------------------------------------------
class NwbExporter:
    """Export a deepgait trial to NWB:N 2.5."""

    def __init__(self, out_path: Union[str, Path]) -> None:
        self.out_path = Path(out_path)

    def export(self, spec: NwbExportSpec) -> Path:
        """Write the NWB file. Raises :class:`NwbUnavailable` if pynwb missing."""
        if not PYNWB_AVAILABLE:
            raise NwbUnavailable(
                "pynwb is not installed. Install with "
                "`pip install -e .[nwb]` to enable NWB export."
            )
        # Import lazily so module-level import never fails on missing dep.
        from pynwb import NWBFile, NWBHDF5IO, TimeSeries
        from pynwb.file import Subject

        nwbfile = self._build_nwbfile(spec)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(str(self.out_path), mode="w") as io:
            io.write(nwbfile)
        return self.out_path

    # ---- builders ---------------------------------------------------------
    def _build_nwbfile(self, spec: NwbExportSpec):
        from pynwb import NWBFile, TimeSeries
        from pynwb.file import Subject

        session_start = spec.experiment_date
        if session_start.tzinfo is None:
            session_start = session_start.replace(tzinfo=timezone.utc)

        nwbfile = NWBFile(
            session_description=spec.session_description,
            identifier=spec.trial_id,
            session_start_time=session_start,
            experimenter=[spec.operator],
            lab="deepgait v2",
            institution="deepgait",
            experiment_description="Mouse/rat gait analysis",
            session_id=spec.trial_id,
        )
        nwbfile.subject = Subject(
            subject_id=spec.animal_id,
            species=spec.species,
            strain=spec.strain,
        )
        # 3D pose as a Behavior ProcessingModule with one TimeSeries per bodypart.
        if spec.pose_3d_positions is not None:
            from pynwb.behavior import Position, SpatialSeries

            behavior = nwbfile.create_processing_module(
                name="behavior",
                description="deepgait v2 pose + gait data",
            )
            pos = Position(name="Position")
            n_bp = spec.pose_3d_positions.shape[1]
            timestamps = np.arange(
                spec.pose_3d_positions.shape[0],
            ) / max(spec.fps, 1)
            for bp in range(n_bp):
                data = spec.pose_3d_positions[:, bp, :].astype(np.float64)
                pos.add_spatial_series(SpatialSeries(
                    name=f"bodypart_{bp:02d}",
                    data=data,
                    timestamps=timestamps,
                    reference_frame="walkway origin (mm)",
                    unit="mm",
                ))
            behavior.add(pos)

            if spec.pose_3d_confidence is not None:
                behavior.add(TimeSeries(
                    name="pose_confidence",
                    data=spec.pose_3d_confidence.astype(np.float64),
                    timestamps=timestamps,
                    unit="a.u.",
                ))
            if spec.in_stance is not None:
                behavior.add(TimeSeries(
                    name="in_stance",
                    data=spec.in_stance.astype(np.uint8),
                    timestamps=timestamps,
                    unit="boolean",
                ))

        # gait summary as scalar TimeSeries (one sample, timestamp 0).
        if spec.gait_summary:
            from pynwb import TimeSeries

            behavior = nwbfile.processing.get("behavior")
            if behavior is None:
                behavior = nwbfile.create_processing_module(
                    name="behavior",
                    description="deepgait v2 pose + gait data",
                )
            for k, v in spec.gait_summary.items():
                behavior.add(TimeSeries(
                    name=f"gait_summary_{k}",
                    data=np.array([float(v)], dtype=np.float64),
                    timestamps=np.array([0.0]),
                    unit="a.u.",
                    description=f"trial-level gait summary: {k}",
                ))
        return nwbfile

    def __repr__(self) -> str:
        return f"NwbExporter(out_path={self.out_path!s})"