"""DLC project config templates and generation.

Generates the ``config.yaml`` content that DLC expects, pre-populated with
deepgait's 12-point bodypart scheme.  This module has **no DLC dependency** —
it only produces YAML text and paths, so it is fully testable without DLC
installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from deepgait3.core._legacy import bodyparts


@dataclass(slots=True)
class ProjectSpec:
    """Inputs needed to create a DLC project config."""
    project: str               # e.g. "deepgait"
    experimenter: str          # e.g. "researcher"
    videos: list[str]          # video paths
    working_directory: str     # project root
    fps: int = 100
    video_width: int = 640
    video_height: int = 480
    numframes2pick: int = 20
    training_fraction: float = 0.8
    bodyparts: list[str] | None = None        # default: 12-point VGL scheme
    crop: tuple[int, int, int, int] | None = None  # x1, x2, y1, y2


def default_bodyparts() -> list[str]:
    """Return the VGL 12-point bodypart list."""
    return list(bodyparts.BODYPARTS_12)


def build_config_dict(spec: ProjectSpec) -> dict[str, Any]:
    """Build the DLC config.yaml dict from a ProjectSpec.

    This mirrors the structure DLC's own ``create_new_project`` writes, but
    we generate it ourselves so we can pre-set bodyparts, engine, and crop
    without needing DLC installed.
    """
    parts = spec.bodyparts or default_bodyparts()
    date = datetime.now().strftime("%Y-%m-%d")
    task = spec.project

    # video_sets: {path: {crop: "x1,x2,y1,y2"}}
    if spec.crop is not None:
        crop_str = f"{spec.crop[0]},{spec.crop[1]},{spec.crop[2]},{spec.crop[3]}"
    else:
        crop_str = f"0,{spec.video_width},0,{spec.video_height}"

    video_sets: dict[str, dict[str, str]] = {}
    for v in spec.videos:
        video_sets[str(Path(v).resolve())] = {"crop": crop_str}

    return {
        "Task": task,
        "scorer": spec.experimenter,
        "date": date,
        "project_path": str(Path(spec.working_directory).resolve()),
        "video_sets": video_sets,
        "bodyparts": parts,
        "start": 0,
        "stop": 1,
        "numframes2pick": spec.numframes2pick,
        "skeleton": [],
        "skeleton_color": "black",
        "pcutoff": 0.1,
        "TrainingFraction": [spec.training_fraction],
        "iteration": 0,
        "default_net_type": "resnet_50",
        "default_augmenter": "imgaug",
        "snapshotindex": -1,
        "batch_size": 8,
        "cropping": spec.crop is not None,
        # if cropping, X1 X2 Y1 Y2
        "x1": spec.crop[0] if spec.crop else 0,
        "x2": spec.crop[1] if spec.crop else spec.video_width,
        "y1": spec.crop[2] if spec.crop else 0,
        "y2": spec.crop[3] if spec.crop else spec.video_height,
        "engine": "pytorch",
        "multianimalproject": False,
        "identity": False,
        "individuals": ["animal1"],
    }


def write_config(spec: ProjectSpec, filename: str = "config.yaml") -> Path:
    """Generate config dict and write it to <working_directory>/<filename>.

    Returns the path to the written file.
    """
    cfg = build_config_dict(spec)
    out = Path(spec.working_directory) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out


def project_dir_name(spec: ProjectSpec) -> str:
    """Return the canonical DLC project directory name.

    DLC names projects: ``<Task><experimenter><date>`` (e.g., deepgait-res-2026-6-16).
    """
    date = datetime.now().strftime("%Y-%m-%d")
    return f"{spec.project}-{spec.experimenter}-{date}"
