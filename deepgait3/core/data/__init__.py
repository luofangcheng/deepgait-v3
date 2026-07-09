"""DeepGait v3 — data layer: schema, export, project management, pipeline.

Usage::

    from deepgait3.core.data import TrialData, ExtractedCycle, FootprintRecord
    from deepgait3.core.data import extract_trial, export_trial
    from deepgait3.core.data import ProjectManager

    pm = ProjectManager()
    pm.activate("V3-test1")
    trial = extract_trial(video_path, output_dir)
    pm.save_trial(trial)
"""
from deepgait3.core.data.schema import TrialData, ExtractedCycle, FootprintRecord
from deepgait3.core.data.exporter import export_trial
from deepgait3.core.data.project import ProjectManager
from deepgait3.core.data.pipeline import extract_trial

__all__ = [
    "TrialData", "ExtractedCycle", "FootprintRecord",
    "export_trial", "extract_trial",
    "ProjectManager",
]
