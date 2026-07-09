"""Active-project management and data persistence for gait analysis trials.

Concept
-------
One "project" = one experiment session.  It has::

    projects/<project_name>/
    ├── project.json          ← metadata (experimenter, px_per_mm, …)
    ├── rawdata/
    │   └── videos/           ← raw input videos
    └── data/
        └── <trial_name>/     ← processed output for one trial
            ├── cycles_summary.csv
            ├── footprints_detail.csv
            ├── trial_data.json
            ├── footprints.db
            ├── cumulative_intensity.png
            └── cumulative_overlay.png

The "active" project is stored in ``projects/.active`` so downstream
tools (CLI / GUI) know where to write without user intervention.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from deepgait3.core.data.schema import TrialData
from deepgait3.core.data.exporter import export_trial

# Default location relative to the deepgait-v3 repo root
_DEFAULT_PROJECTS_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "projects"


class ProjectManager:
    """Manage project creation, activation, and trial data persistence."""

    def __init__(self, projects_root: Path | None = None):
        self.root = Path(projects_root) if projects_root else _DEFAULT_PROJECTS_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    # ── project CRUD ─────────────────────────────────────────────────────

    def create(self, name: str, experimenter: str = "", px_per_mm: float = 3.0) -> Path:
        """Create a new project directory with standard structure.

        Returns the project directory path.
        """
        proj_dir = self.root / name
        if proj_dir.exists():
            raise FileExistsError(f"Project '{name}' already exists at {proj_dir}")

        (proj_dir / "rawdata" / "videos").mkdir(parents=True)
        (proj_dir / "data").mkdir(parents=True)

        meta = {
            "project_name": name,
            "project_path": str(proj_dir.resolve()),
            "experimenter": experimenter,
            "px_per_mm": px_per_mm,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (proj_dir / "project.json").write_text(json.dumps(meta, indent=2))
        return proj_dir

    def activate(self, name: str) -> Path:
        """Set the active project and return its directory."""
        proj_dir = self.root / name
        if not proj_dir.is_dir():
            raise FileNotFoundError(f"Project '{name}' not found at {proj_dir}")
        (self.root / ".active").write_text(name + "\n")
        return proj_dir

    @property
    def active_name(self) -> Optional[str]:
        """Name of the currently active project, or None."""
        active_file = self.root / ".active"
        if not active_file.exists():
            return None
        return active_file.read_text().strip()

    @property
    def active_dir(self) -> Optional[Path]:
        """Directory of the currently active project, or None."""
        name = self.active_name
        return self.root / name if name else None

    def list_projects(self) -> list[str]:
        """Return sorted list of project names."""
        return sorted(
            d.name for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    # ── trial data persistence ────────────────────────────────────────────

    def save_trial(
        self,
        trial: TrialData,
        trial_name: str | None = None,
        project_name: str | None = None,
    ) -> Path:
        """Export trial data into ``<project>/data/<trial_name>/``.

        If *project_name* is None, the active project is used.
        If *trial_name* is None, defaults to ``{mouse_id}_{timestamp}``.
        Returns the trial output directory.
        """
        if project_name:
            proj_dir = self.root / project_name
        else:
            proj_dir = self.active_dir
            if proj_dir is None:
                raise RuntimeError(
                    "No active project. Call activate() first or pass project_name."
                )

        if trial_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            trial_name = f"{trial.mouse_id or 'trial'}_{ts}"

        trial_dir = proj_dir / "data" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Set trial metadata
        trial.trial_name = trial_name

        # Export structured data
        export_trial(trial, trial_dir)

        # Copy visualisation artifacts from the pipeline output
        # (if they exist alongside the trial)
        return trial_dir

    def get_trial_dir(
        self,
        trial_name: str,
        project_name: str | None = None,
    ) -> Path:
        """Return the path to a trial's data directory."""
        if project_name:
            proj_dir = self.root / project_name
        else:
            proj_dir = self.active_dir
            if proj_dir is None:
                raise RuntimeError("No active project.")
        return proj_dir / "data" / trial_name

    # ── raw data ──────────────────────────────────────────────────────────

    def add_video(self, video_path: Path, project_name: str | None = None) -> Path:
        """Copy a video into the project's rawdata/videos/ directory.

        Returns the destination path.
        """
        if project_name:
            proj_dir = self.root / project_name
        else:
            proj_dir = self.active_dir
            if proj_dir is None:
                raise RuntimeError("No active project.")

        dest_dir = proj_dir / "rawdata" / "videos"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / video_path.name
        if not dest.exists():
            shutil.copy2(video_path, dest)
        return dest
