"""DeepLabCut workflow: thin wrappers around DLC's 10-step pipeline.

Each function lazily imports DLC, so this module imports cleanly even when
DLC is not installed (only the actual DLC calls fail then).  This lets us
unit-test the surrounding logic (config generation, path discovery, result
plumbing) without a working DLC environment.

DLC 3.x PyTorch workflow (10 steps):
    1. create_new_project      -> 6. create_training_dataset
    2. edit config (done by dlc_config.py)
    3. extract_frames          -> 7. train_network
    4. label_frames (napari)   -> 8. evaluate_network
    5. check_labels            -> 9. analyze_videos / 10. filterpredictions

deepgait exposes the steps you actually automate (1,3,6,7,8,9,10); steps 2
and 4 are config-edit / GUI, handled separately.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from deepgait3.core._legacy import dlc_config, dlc_results, gait_export
from deepgait3.core._legacy.pipeline import analyze as gait_analyze
from deepgait3.core._legacy.results import GaitResults

log = logging.getLogger(__name__)


class DLCNotInstalledError(ImportError):
    """Raised when a DLC function is called but DLC is not importable."""


def _require_dlc() -> Any:
    """Import and return the deeplabcut module, or raise a clear error."""
    try:
        import deeplabcut  # noqa: WPS433  (lazy import)
        return deeplabcut
    except ImportError as e:  # pragma: no cover  (environment-dependent)
        raise DLCNotInstalledError(
            "DeepLabCut is not installed. Install it with:\n"
            "  pip install 'deeplabcut[gui]'\n"
            "or see https://github.com/DeepLabCut/DeepLabCut"
        ) from e


# ---------------------------------------------------------------------------
# Step 1: create project
# ---------------------------------------------------------------------------

def create_project(
    spec: dlc_config.ProjectSpec,
) -> Path:
    """Create a DLC project and write its config.yaml.

    Uses DLC's ``create_new_project`` to set up the directory structure, then
    overwrites config.yaml with deepgait's 12-point bodypart template.

    Args:
        spec: ProjectSpec with project name, experimenter, videos, working dir.

    Returns:
        Path to the written config.yaml.
    """
    try:
        dlc = _require_dlc()
        # Let DLC create the project skeleton (directories, default config).
        cfg_from_dlc = dlc.create_new_project(
            project=spec.project,
            experimenter=spec.experimenter,
            videos=spec.videos,
            working_directory=spec.working_directory,
        )
        log.info("DLC project created: %s", cfg_from_dlc)
        # Overwrite config with deepgait's template (sets bodyparts, engine, etc.)
        spec.working_directory = str(Path(cfg_from_dlc).parent)
        cfg_path = dlc_config.write_config(spec, filename=Path(cfg_from_dlc).name)
        log.info("deepgait config written: %s", cfg_path)
        return cfg_path
    except DLCNotInstalledError:
        # DLC not installed — fall back to writing just the config so the user
        # can inspect the project structure and complete setup later.
        log.warning("DLC not installed; writing config only")
        spec.working_directory = str(
            Path(spec.working_directory).resolve()
            / dlc_config.project_dir_name(spec)
        )
        return dlc_config.write_config(spec)


# ---------------------------------------------------------------------------
# Step 3: extract frames
# ---------------------------------------------------------------------------

def extract_frames(config: str | Path, mode: str = "automatic", algo: str = "kmeans") -> None:
    """Extract frames for labeling (DLC step 3)."""
    dlc = _require_dlc()
    dlc.extract_frames(str(config), mode=mode, algo=algo)
    log.info("Frames extracted for %s", config)


# ---------------------------------------------------------------------------
# Step 6: create training dataset
# ---------------------------------------------------------------------------

def create_training_dataset(
    config: str | Path,
    net_type: str = "resnet_50",
    num_shuffles: int = 1,
) -> None:
    """Generate the training dataset (DLC step 6, PyTorch engine)."""
    dlc = _require_dlc()
    try:
        from deeplabcut.utils import Engine
        engine = Engine.PYTORCH
    except ImportError:  # pragma: no cover
        engine = None  # DLC < 3.0 fallback
    dlc.create_training_dataset(
        str(config),
        num_shuffles=num_shuffles,
        net_type=net_type,
        engine=engine,
    )
    log.info("Training dataset created (net=%s, engine=%s)", net_type, engine)


# ---------------------------------------------------------------------------
# Step 7: train
# ---------------------------------------------------------------------------

def train_network(
    config: str | Path,
    epochs: int = 200,
    batch_size: int = 8,
    device: str | None = None,
) -> None:
    """Train the network (DLC step 7).

    Uses PyTorch ``epochs`` (not TF ``maxiters``).
    """
    dlc = _require_dlc()
    dlc.train_network(
        str(config),
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )
    log.info("Training complete: %s", config)


# ---------------------------------------------------------------------------
# Step 8: evaluate
# ---------------------------------------------------------------------------

def evaluate_network(config: str | Path, plotting: bool = True) -> None:
    """Evaluate the trained network (DLC step 8)."""
    dlc = _require_dlc()
    dlc.evaluate_network(str(config), plotting=plotting)
    log.info("Evaluation complete: %s", config)


# ---------------------------------------------------------------------------
# Step 9 & 10: analyze + filter
# ---------------------------------------------------------------------------

def analyze_videos(
    config: str | Path,
    videos: list[str],
    save_as_csv: bool = True,
    device: str | None = None,
    batch_size: int = 8,
    cropping: tuple[int, int, int, int] | None = None,
    destfolder: str | None = None,
) -> None:
    """Run pose estimation on videos (DLC step 9)."""
    dlc = _require_dlc()
    kwargs: dict[str, Any] = dict(
        save_as_csv=save_as_csv,
        device=device,
        batch_size=batch_size,
    )
    if cropping is not None:
        kwargs["cropping"] = list(cropping)
    if destfolder:
        kwargs["destfolder"] = destfolder
    dlc.analyze_videos(str(config), videos, **kwargs)
    log.info("Videos analyzed: %s", videos)


def filter_predictions(
    config: str | Path,
    videos: list[str],
    filtertype: str = "median",
    windowlength: int = 5,
    save_as_csv: bool = True,
) -> None:
    """Filter predictions (DLC step 10)."""
    dlc = _require_dlc()
    dlc.filterpredictions(
        str(config), videos,
        filtertype=filtertype,
        windowlength=windowlength,
        save_as_csv=save_as_csv,
    )
    log.info("Predictions filtered: %s", videos)


# ---------------------------------------------------------------------------
# One-shot: analyze video(s) end-to-end and produce gait report
# ---------------------------------------------------------------------------

def analyze_video_gait(
    config: str | Path,
    videos: list[str],
    fps: int = 100,
    mode: str = "catwalk",
    device: str | None = None,
    batch_size: int = 8,
    crop_to_body: tuple[int, int, int, int] | None = None,
    do_filter: bool = True,
    real_world_multiplier: float = 1.0,
    video_height: float | None = None,
    export_excel: bool = True,
) -> dict[str, GaitResults]:
    """Run DLC pose estimation then deepgait gait analysis on each video.

    Args:
        config: DLC config.yaml path (must point to a trained project).
        videos: list of video paths.
        fps, mode, multiplier, video_height: gait analysis params.
        device, batch_size, crop_to_body: DLC inference params.
        do_filter: apply DLC filterpredictions before gait analysis.
        export_excel: write <video>.xlsx next to each video.

    Returns:
        Dict mapping video path -> GaitResults.
    """
    # DLC steps
    analyze_videos(config, videos, device=device, batch_size=batch_size, cropping=crop_to_body)
    if do_filter:
        filter_predictions(config, videos)

    # Gait analysis per video
    results: dict[str, GaitResults] = {}
    for video in videos:
        out = dlc_results.find_dlc_outputs(video)
        csv_path = out.best_csv
        if csv_path is None:
            log.warning("No DLC CSV found for %s, skipping gait analysis", video)
            continue
        res = gait_analyze(
            csv_path,
            fps=fps,
            mode=mode,
            video_height=video_height,
            real_world_multiplier=real_world_multiplier,
        )
        results[video] = res
        if export_excel:
            out_path = Path(video).with_suffix(".gait.xlsx")
            gait_export.to_excel(res, out_path)
            log.info("Gait report: %s", out_path)
    return results
