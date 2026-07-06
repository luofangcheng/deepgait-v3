"""BIDS-like directory exporter for deepgait v2 (Layer 2).

Exports a deepgait trial to a BIDS-inspired directory layout so the
trial can be ingested by BIDS-compatible analysis pipelines (e.g.
[Movements](https://bids-specification.readthedocs.io/) for movement
data). deepgait does not claim full BIDS compliance — the layout is
BIDS-*like*: trial-level metadata goes in ``dataset_description.json``
+ ``participants.tsv``, the four camera videos become
``sub-<animal>_cam-<role>_video.mp4``, and pose/metrics are written
as gzipped TSV sidecars.

Implementation uses only the standard library (json, csv, gzip, shutil,
pathlib) per DEVELOPMENT_PLAN §5.2 — no extra dependency.

Acceptance gate (DEVELOPMENT_PLAN §6.1 W3): exporters must produce a
re-importable layout. Verified by ``tests/unit/test_w3_io.py``.
"""
from __future__ import annotations

import csv
import gzip
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np


logger = logging.getLogger(__name__)


# BIDS-inspired labels deepgait maps onto.
CAMERA_ROLE_TO_BIDS = {
    "bottom_ftir": "bottom",
    "bottom": "bottom",
    "left": "left",
    "right": "right",
    "top": "top",
}


# ---------------------------------------------------------------------------
# BidsExport spec
# ---------------------------------------------------------------------------
@dataclass
class BidsExportSpec:
    """Inputs needed for a BIDS-like export."""
    animal_id: str
    species: str
    trial_id: str
    experiment_date: str
    operator: str
    app_version: str
    n_frames: int
    fps: int
    # Per-camera pose + metrics. The exporter itself is data-shape agnostic
    # — it just writes whatever numpy arrays / dicts the caller passes in.
    cameras: Dict[str, Dict]  # {role: {"video": path|None, "pose_2d": ndarray|None}}
    gait_per_paw: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    coordination: Optional[Dict[str, float]] = None
    summary: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# BidsExporter
# ---------------------------------------------------------------------------
class BidsExporter:
    """Export a deepgait trial to a BIDS-like directory tree."""

    def __init__(self, out_dir: Union[str, Path]) -> None:
        self.out_dir = Path(out_dir)

    # ---- public API -------------------------------------------------------
    def export(self, spec: BidsExportSpec) -> Path:
        """Write the full BIDS-like layout. Returns the output root."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._write_dataset_description(spec)
        self._write_participants(spec)
        self._write_session(spec)
        self._write_videos(spec)
        self._write_pose(spec)
        self._write_gait(spec)
        self._write_readme(spec)
        return self.out_dir

    # ---- builders ---------------------------------------------------------
    def _animal_label(self, spec: BidsExportSpec) -> str:
        # BIDS labels must be alphanumeric; collapse the animal_id.
        safe = "".join(c if c.isalnum() else "" for c in spec.animal_id)
        return safe or "subject"

    def _subject_dir(self, spec: BidsExportSpec) -> Path:
        return self.out_dir / f"sub-{self._animal_label(spec)}"

    def _session_dir(self, spec: BidsExportSpec) -> Path:
        # BIDS sessions use YYYYMMDD; use the trial's experiment_date if present.
        try:
            dt = datetime.fromisoformat(spec.experiment_date)
            sess = dt.strftime("%Y%m%d")
        except Exception:
            sess = "ses-01"
        return self._subject_dir(spec) / f"ses-{sess}"

    # ---- writers ----------------------------------------------------------
    def _write_dataset_description(self, spec: BidsExportSpec) -> None:
        desc = {
            "Name": "deepgait v2 trial export",
            "BIDSVersion": "1.9.0 (deepgait-extended)",
            "DatasetType": "raw",
            "GeneratedBy": [{
                "Name": "deepgait",
                "Version": spec.app_version,
                "Description": "Mouse/rat gait analyzer",
            }],
            "SourceDataset": f"trial_id={spec.trial_id}",
            "Species": spec.species,
        }
        (self.out_dir / "dataset_description.json").write_text(
            json.dumps(desc, indent=2, ensure_ascii=False)
        )

    def _write_participants(self, spec: BidsExportSpec) -> None:
        tsv = self.out_dir / "participants.tsv"
        with open(tsv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["participant_id", "species", "trial_id", "operator"])
            w.writerow([
                f"sub-{self._animal_label(spec)}",
                spec.species, spec.trial_id, spec.operator,
            ])

    def _write_session(self, spec: BidsExportSpec) -> None:
        sess_dir = self._session_dir(spec)
        sess_dir.mkdir(parents=True, exist_ok=True)
        beh_dir = sess_dir / "behav"
        beh_dir.mkdir(exist_ok=True)
        # sessions.tsv: minimal
        (sess_dir / f"sub-{self._animal_label(spec)}_sessions.tsv").write_text(
            f"session_id\tacq_time\n"
            f"ses-{spec.experiment_date[:10]}\t{spec.experiment_date}\n"
        )

    def _write_videos(self, spec: BidsExportSpec) -> None:
        sess_dir = self._session_dir(spec)
        for role, payload in spec.cameras.items():
            video = payload.get("video")
            if video is None:
                continue
            video_path = Path(video)
            if not video_path.is_file():
                logger.warning("BIDS export: video missing %s", video_path)
                continue
            bids_role = CAMERA_ROLE_TO_BIDS.get(role, role)
            dest = sess_dir / f"sub-{self._animal_label(spec)}_cam-{bids_role}_video.mp4"
            shutil.copy2(video_path, dest)
            # sidecar JSON
            sidecar = sess_dir / f"sub-{self._animal_label(spec)}_cam-{bids_role}_video.json"
            sidecar.write_text(json.dumps({
                "SamplingFrequency": spec.fps,
                "FrameCount": spec.n_frames,
                "CameraRole": role,
                "Source": str(video_path),
            }, indent=2))

    def _write_pose(self, spec: BidsExportSpec) -> None:
        sess_dir = self._session_dir(spec)
        beh_dir = sess_dir / "behav"
        for role, payload in spec.cameras.items():
            pose = payload.get("pose_2d")
            if pose is None:
                continue
            bids_role = CAMERA_ROLE_TO_BIDS.get(role, role)
            # Save as gzipped TSV: columns x_0,y_0,lh_0,...,x_11,y_11,lh_11
            arr = np.asarray(pose)
            n_frames, n_bp, n_cols = arr.shape
            tsv_path = beh_dir / (
                f"sub-{self._animal_label(spec)}_cam-{bids_role}_pose.tsv.gz"
            )
            with gzip.open(tsv_path, "wt", newline="", encoding="utf-8") as f:
                w = csv.writer(f, delimiter="\t")
                header = [f"{c}{i}" for i in range(n_bp) for c in ("x", "y", "lh")]
                w.writerow(header)
                for frame in arr:
                    w.writerow([float(v) for v in frame.reshape(-1)])

    def _write_gait(self, spec: BidsExportSpec) -> None:
        sess_dir = self._session_dir(spec)
        beh_dir = sess_dir / "behav"
        # coordination + summary as JSON
        if spec.coordination:
            (beh_dir / f"sub-{self._animal_label(spec)}_coordination.json").write_text(
                json.dumps(spec.coordination, indent=2)
            )
        if spec.summary:
            (beh_dir / f"sub-{self._animal_label(spec)}_summary.json").write_text(
                json.dumps(spec.summary, indent=2)
            )
        # per-paw metrics as one gzipped TSV per paw
        if spec.gait_per_paw:
            for paw, metrics in spec.gait_per_paw.items():
                # All metrics for one paw stacked column-wise.
                names = list(metrics.keys())
                if not names:
                    continue
                cols = [np.asarray(metrics[n]).reshape(-1) for n in names]
                n_rows = max(c.shape[0] for c in cols)
                mat = np.full((n_rows, len(names)), np.nan, dtype=np.float64)
                for j, c in enumerate(cols):
                    mat[:c.shape[0], j] = c
                tsv_path = beh_dir / (
                    f"sub-{self._animal_label(spec)}_paw-{paw}_metrics.tsv.gz"
                )
                with gzip.open(tsv_path, "wt", newline="", encoding="utf-8") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(names)
                    for row in mat:
                        w.writerow(["" if np.isnan(v) else f"{v:.6f}" for v in row])

    def _write_readme(self, spec: BidsExportSpec) -> None:
        readme = self.out_dir / "README.md"
        readme.write_text(
            f"# deepgait v2 BIDS-like export\n\n"
            f"- Trial: `{spec.trial_id}`\n"
            f"- Animal: `{spec.animal_id}` ({spec.species})\n"
            f"- Operator: `{spec.operator}`\n"
            f"- Recorded: `{spec.experiment_date}`\n"
            f"- Frames: {spec.n_frames} @ {spec.fps} fps\n"
            f"- deepgait version: {spec.app_version}\n\n"
            f"This directory follows a BIDS-*inspired* layout. See\n"
            f"`dataset_description.json` for provenance.\n"
        )

    def __repr__(self) -> str:
        return f"BidsExporter(out_dir={self.out_dir!s})"
