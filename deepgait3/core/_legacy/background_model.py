"""Rolling-median background model for FTIR walkway frames.

Used by the live gait analysis pipeline (``gait_ftir.py``) to
separate the animal's footprints from the static walkway surface.
Based on the MouseWalker / GAITOR approach: a per-pixel median
over a sliding window of recent frames serves as the background
model; pixels that deviate by more than `offset` are classified
as foreground (footprint).

Typical usage::

    bg = RollingMedianBackground(warmup_frames=30)
    for frame in video:
        bg.update(frame)
        fg = bg.foreground_mask(frame)
"""
from __future__ import annotations

import numpy as np
from typing import Optional


class RollingMedianBackground:
    """Per-pixel rolling-window median background model.

    Parameters
    ----------
    warmup_frames : int
        Number of initial frames to collect before the model becomes
        ready.  During warmup, ``foreground_mask`` returns all-False.
    update_every : int
        Interval (in frames) between median recomputations.
        Higher = less CPU, more lag.
    window_size : int
        Number of recent frames kept in the ring buffer.
    offset : int
        Per-channel pixel intensity offset that must be exceeded
        for a pixel to be considered foreground.
    """

    def __init__(
        self,
        warmup_frames: int = 30,
        update_every: int = 5,
        window_size: int = 50,
        offset: int = 25,
    ) -> None:
        self.warmup_frames = warmup_frames
        self.update_every = update_every
        self.window_size = window_size
        self.offset = offset
        self._buffer: list = []
        self._median: Optional[np.ndarray] = None
        self._frame_count = 0
        self._shape: Optional[tuple] = None

    @property
    def ready(self) -> bool:
        return self._median is not None

    @property
    def is_ready(self) -> bool:
        """Alias for footprint_v2 compatibility."""
        return self.ready

    def get_median(self) -> Optional[np.ndarray]:
        """Return the current background median image (BGR uint8).

        Returns ``None`` before the model is ready.
        Compatible with :class:`deepgait3.core._legacy.footprint_v2.BackgroundModel`.
        """
        if self._median is None:
            return None
        return np.clip(self._median, 0, 255).astype(np.uint8)

    def update(self, frame: np.ndarray) -> None:
        """Add a frame to the ring buffer and recompute median if due.

        Parameters
        ----------
        frame : np.ndarray
            BGR or grayscale frame (uint8).  Will be converted to
            float internally.
        """
        self._frame_count += 1
        f = frame.astype(np.float32)
        if self._shape is None:
            self._shape = f.shape
        # ring buffer
        self._buffer.append(f)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)
        # Recompute median
        if (
            self._frame_count <= self.warmup_frames
            or self._frame_count % self.update_every == 0
        ):
            if len(self._buffer) >= 2:
                stack = np.stack(self._buffer, axis=0)
                self._median = np.median(stack, axis=0)

    def foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a boolean mask of foreground (footprint) pixels.

        Returns all-False while the model is still warming up.
        """
        if not self.ready:
            return np.zeros(frame.shape[:2], dtype=bool)
        diff = np.abs(frame.astype(np.float32) - self._median)
        return (np.any(diff > self.offset, axis=-1)).astype(np.uint8) * 255
