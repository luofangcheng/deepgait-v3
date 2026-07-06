"""Locate DLC analysis output files.

DLC writes outputs next to the analyzed video with a naming convention::

    <videoname><scorer><date>.h5
    <videoname><scorer><date>.csv
    <videoname><scorer><date>_meta.json
    <videoname><scorer><date>_filtered.h5
    <videoname><scorer><date>_filtered.csv

This module finds those files without needing DLC installed — it only does
glob-based path discovery, so it is fully testable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DLCOutput:
    """A set of DLC analysis outputs for one video."""
    video: Path
    h5: Path | None = None
    csv: Path | None = None
    meta: Path | None = None
    filtered_h5: Path | None = None
    filtered_csv: Path | None = None

    @property
    def best_csv(self) -> Path | None:
        """Prefer filtered CSV, fall back to raw CSV."""
        return self.filtered_csv or self.csv

    @property
    def has_results(self) -> bool:
        return self.h5 is not None or self.csv is not None


def find_dlc_outputs(
    video_path: str | Path,
    search_dir: str | Path | None = None,
) -> DLCOutput:
    """Find DLC outputs for a given video.

    Args:
        video_path: path to the source video (e.g., mouse.mp4).
        search_dir: directory to search in; defaults to the video's directory.

    Returns:
        DLCOutput with all discovered file paths (None where not found).
    """
    video = Path(video_path)
    search = Path(search_dir) if search_dir else video.parent
    stem = video.stem  # filename without extension

    out = DLCOutput(video=video)

    # DLC naming: <videoname><scorer><date>[_filtered][.ext]
    # We glob for any file starting with the video stem.
    pattern = f"{re.escape(stem)}*"
    for candidate in sorted(search.glob(pattern)):
        name = candidate.name
        if stem not in name:
            continue
        low = name.lower()
        if low.endswith("_filtered.csv"):
            out.filtered_csv = candidate
        elif low.endswith("_filtered.h5"):
            out.filtered_h5 = candidate
        elif low.endswith("_meta.json"):
            out.meta = candidate
        elif low.endswith(".csv") and out.csv is None:
            out.csv = candidate
        elif low.endswith(".h5") and out.h5 is None:
            out.h5 = candidate
    return out


def find_all_dlc_outputs(
    search_dir: str | Path,
    video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[DLCOutput]:
    """Find DLC outputs for all videos in a directory.

    Returns one DLCOutput per source video found (even if no DLC results exist).
    """
    search = Path(search_dir)
    videos = sorted(
        p for p in search.iterdir()
        if p.is_file() and p.suffix.lower() in video_extensions
    )
    return [find_dlc_outputs(v) for v in videos]
