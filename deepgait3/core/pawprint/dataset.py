"""Dataset abstraction for fTIR videos.

Wraps a list of video files so that algorithms and benchmarks can iterate
over them uniformly. New sources (frame directories, .h5 stacks, ROS bags)
can plug in here without changing the algorithm code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    n_frames: int
    width: int
    height: int
    fps: float

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0


def video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = VideoInfo(
        path=Path(path),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
    )
    cap.release()
    return info


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(frame_idx_1based, BGR ndarray)`` for every decoded frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        yield idx, frame
    cap.release()


def collect_videos(
    roots: list[str | Path],
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
) -> list[Path]:
    """Recursively find all videos under the given roots (case-insensitive)."""
    extensions_lower = tuple(e.lower() for e in extensions)
    out: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_file() and root.suffix.lower() in extensions_lower:
            out.append(root)
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions_lower:
                out.append(path)
    out.sort()
    # De-dup
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in out:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            deduped.append(p)
    return deduped


__all__ = ["VideoInfo", "video_info", "iter_frames", "collect_videos"]