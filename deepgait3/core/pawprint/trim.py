"""Video trimming by first/last footprint frame.

Reuses :class:`YoloPawDetector` and :class:`IoUFootprintTracker` to detect
footprints in a video, derives the temporal extent of paw contact events,
and re-encodes the source video cropped to that interval.

Pure-Python, zero Qt dependency.  Designed to be called from a background
thread by the GUI (``TrimWorker``) or from CLI scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .tracker import FootprintTrack, IoUFootprintTracker
from .yolo_detector import YoloPawDetector


class TrimError(RuntimeError):
    """Raised when a video cannot be trimmed (empty tracks, unreadable file, …)."""


def find_trim_range(tracks: list[FootprintTrack]) -> Tuple[int, int]:
    """Return ``(first_frame_idx, last_frame_idx)`` inclusive.

    Indices are 0-based frame numbers as passed to
    :meth:`IoUFootprintTracker.update` (typically ``idx + 1`` where ``idx``
    is 0-based).

    Empty / no-footprint tracks return ``(0, -1)`` to signal "no trim".
    """
    if not tracks:
        return 0, -1
    firsts = [t.foots[0][0] for t in tracks if t.foots]
    lasts = [t.foots[-1][0] for t in tracks if t.foots]
    if not firsts:
        return 0, -1
    return min(firsts), max(lasts)


def _read_all_frames(src_path: Path) -> Tuple[list[np.ndarray], float, tuple[int, int]]:
    """Load every BGR frame from a video into memory."""
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise TrimError(f"Cannot open video: {src_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise TrimError(f"Empty video: {src_path}")
    return frames, fps, (h, w)


def _write_trimmed(
    frames: list[np.ndarray],
    dst_path: Path,
    first_frame: int,
    last_frame: int,
    fps: float,
) -> int:
    """Encode ``frames[first_frame:last_frame+1]`` to ``dst_path`` with mp4v.

    Returns the number of frames written.
    """
    h, w = frames[0].shape[:2]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise TrimError(f"Cannot open VideoWriter: {dst_path}")
    n_out = 0
    try:
        for idx in range(first_frame, last_frame + 1):
            writer.write(frames[idx])
            n_out += 1
    finally:
        writer.release()
    return n_out


def trim_video(
    src_path: Path,
    dst_path: Path,
    *,
    detector: YoloPawDetector,
    iou_min: float = 0.3,
    max_gap_frames: int = 3,
    min_area_px: int = 5,
    warmup_frames: int = 30,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Detect first/last footprint frames and re-encode the source cropped to that range.

    Parameters
    ----------
    src_path
        Source video path.
    dst_path
        Destination trimmed video path.  Parent dir is created if missing.
    detector
        Initialised :class:`YoloPawDetector` instance.
    iou_min, max_gap_frames, min_area_px, warmup_frames
        Tracker and detector parameters.
    progress_cb
        Optional ``callable(phase, pct, msg)`` invoked during detection and
        re-encoding.  ``phase`` is ``"detect"`` or ``"encode"``.

    Returns
    -------
    info : dict
        ``{"src", "dst", "n_in", "n_out", "n_tracks", "first_frame", "last_frame"}``
    """
    frames, fps, (h, w) = _read_all_frames(src_path)
    n_in = len(frames)

    n_bg = min(warmup_frames, n_in)
    bg_green = np.stack([f[:, :, 1].astype(np.float32) for f in frames[:n_bg]])
    bg_G = np.median(bg_green, axis=0) if n_bg > 0 else np.zeros((h, w), dtype=np.float32)

    tracker = IoUFootprintTracker(
        frame_shape=(h, w), iou_min=iou_min, max_gap_frames=max_gap_frames,
    )

    batch_size = 16
    for batch_start in range(0, n_in, batch_size):
        batch_end = min(batch_start + batch_size, n_in)
        batch_frames = frames[batch_start:batch_end]
        all_fm = detector.detect_batch(batch_frames, bg_G, min_area_px=min_area_px)
        for i, fm_list in enumerate(all_fm):
            tracker.update(batch_start + i + 1, fm_list)
        if progress_cb is not None:
            progress_cb("detect", int(batch_end * 100 / n_in), f"detecting {batch_end}/{n_in}")

    tracks = tracker.finalize()
    first_frame, last_frame = find_trim_range(tracks)
    if last_frame < first_frame or first_frame < 0:
        raise TrimError(f"No footprints detected in {src_path}")

    # tracker uses 1-based frame idx; frames list is 0-based
    first_idx = max(0, first_frame - 1)
    last_idx = min(n_in - 1, last_frame - 1)

    if progress_cb is not None:
        progress_cb("encode", 0, f"encoding [{first_idx}..{last_idx}]")

    n_out = _write_trimmed(frames, dst_path, first_idx, last_idx, fps)

    if progress_cb is not None:
        progress_cb("encode", 100, f"wrote {n_out} frames")

    return {
        "src": src_path,
        "dst": dst_path,
        "n_in": n_in,
        "n_out": n_out,
        "n_tracks": len(tracks),
        "first_frame": first_idx,
        "last_frame": last_idx,
        "fps": fps,
    }


__all__ = ["TrimError", "find_trim_range", "trim_video"]