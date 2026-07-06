"""Project folder management for deepgait gait analysis.

Provides project creation, configuration persistence, and a recent-projects
registry stored under ``$XDG_DATA_HOME/deepgait/projects.json``.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_CONFIG_FILENAME = "project.json"
VIDEOS_DIRNAME = "videos"
DATA_DIRNAME = "data"
RECENT_PROJECTS_MAX = 20


def _deepgait_data_dir() -> Path:
    """Return ``$XDG_DATA_HOME/deepgait``, creating it if necessary."""
    base = Path.home() / ".local" / "share" / "deepgait"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _recent_projects_path() -> Path:
    return _deepgait_data_dir() / "projects.json"


# ---------------------------------------------------------------------------
# ProjectConfig
# ---------------------------------------------------------------------------
class ProjectConfig:
    """In-memory representation of a project's ``project.json``."""

    def __init__(
        self,
        project_name: str,
        project_path: Path,
        experimenter: str = "",
        px_per_mm: float = 3.0,
    ) -> None:
        self.project_name = project_name
        self.project_path = Path(project_path)
        self.experimenter = experimenter
        self.px_per_mm = px_per_mm
        self.created_at: str = datetime.now(timezone.utc).isoformat()

    @property
    def videos_dir(self) -> Path:
        return self.project_path / VIDEOS_DIRNAME

    @property
    def data_dir(self) -> Path:
        return self.project_path / DATA_DIRNAME

    def to_dict(self) -> Dict:
        return {
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "experimenter": self.experimenter,
            "px_per_mm": self.px_per_mm,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ProjectConfig":
        cfg = cls(
            project_name=d["project_name"],
            project_path=Path(d["project_path"]),
            experimenter=d.get("experimenter", ""),
            px_per_mm=d.get("px_per_mm", 3.0),
        )
        cfg.created_at = d.get("created_at", cfg.created_at)
        return cfg

    def save(self) -> Path:
        """Write ``project.json`` into ``project_path``."""
        self.project_path.mkdir(parents=True, exist_ok=True)
        config_path = self.project_path / PROJECT_CONFIG_FILENAME
        config_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return config_path

    @classmethod
    def load(cls, project_path: Path) -> Optional["ProjectConfig"]:
        """Load ``project.json`` from a directory, or None."""
        config_path = Path(project_path) / PROJECT_CONFIG_FILENAME
        if not config_path.exists():
            return None
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None


# ---------------------------------------------------------------------------
# Project creation
# ---------------------------------------------------------------------------
def create_project(
    project_name: str,
    parent_dir: Path,
    experimenter: str = "",
    px_per_mm: float = 3.0,
) -> ProjectConfig:
    """Create a new project folder structure and return its config.

    Creates::

        {parent_dir}/{project_name}/
          ├── project.json
          ├── videos/
          └── data/

    Parameters
    ----------
    project_name : str
        Name of the project (used as folder name).
    parent_dir : Path
        Parent directory where the project folder will be created.
    experimenter : str
        Name of the experimenter.
    px_per_mm : float
        Pixel-to-mm calibration factor.

    Returns
    -------
    ProjectConfig

    Raises
    ------
    FileExistsError
        If the project directory already exists.
    """
    project_path = Path(parent_dir) / project_name
    if project_path.exists():
        raise FileExistsError(f"Project directory already exists: {project_path}")

    cfg = ProjectConfig(
        project_name=project_name,
        project_path=project_path,
        experimenter=experimenter,
        px_per_mm=px_per_mm,
    )
    cfg.videos_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.save()
    _add_to_recent(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Recent projects registry
# ---------------------------------------------------------------------------
def _load_recent_registry() -> List[Dict]:
    rp = _recent_projects_path()
    if not rp.exists():
        return []
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_recent_registry(entries: List[Dict]) -> None:
    rp = _recent_projects_path()
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _add_to_recent(cfg: ProjectConfig) -> None:
    entries = _load_recent_registry()
    # Remove duplicate by path.
    entries = [e for e in entries if e.get("project_path") != str(cfg.project_path)]
    entries.insert(0, cfg.to_dict())
    if len(entries) > RECENT_PROJECTS_MAX:
        entries = entries[:RECENT_PROJECTS_MAX]
    _save_recent_registry(entries)


def list_recent_projects() -> List[Dict]:
    """Return the recent-projects list sorted by recency (newest first)."""
    return _load_recent_registry()


def remove_from_recent(project_path: Path) -> None:
    """Remove a project entry from the recent-projects registry."""
    entries = _load_recent_registry()
    entries = [e for e in entries if e.get("project_path") != str(project_path)]
    _save_recent_registry(entries)


# ---------------------------------------------------------------------------
# Data output paths
# ---------------------------------------------------------------------------
def animal_data_dir(project_path: Path, animal_id: str) -> Path:
    """Return ``{project_path}/data/{animal_id}/``, creating it."""
    d = Path(project_path) / DATA_DIRNAME / animal_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_output_path(project_path: Path, animal_id: str) -> Path:
    """Return ``{project_path}/videos/{animal_id}.mp4``."""
    vdir = Path(project_path) / VIDEOS_DIRNAME
    vdir.mkdir(parents=True, exist_ok=True)
    return vdir / f"{animal_id}.mp4"
