"""Automatic run detection for the FTIR walkway.

Monitors a live (or offline) stream of ``FootprintSequence`` objects
and emits a run-completed event when the animal has finished one
traversal of the runway.  Supports multi-run automatic segmentation.

State machine
-------------

* **WAITING** — no footprints yet; waiting for animal to enter.
* **RUNNING** — at least one paw detected; collecting frames.
* **FINISHED** — animal has left; run can be extracted.

Usage (live)::

    detector = RunDetector(fps=100, walkway_length_mm=400)
    for seq in frame_stream:
        result = detector.process_frame(seq)
        if result is not None:
            print(f"Run completed: {result.n_frames} frames")
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


class RunDetectorState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class RunResult:
    """A completed run (one traversal)."""
    start_frame: int         # global frame index of first footprint
    end_frame: int           # global frame index of last footprint + 1
    sequences: List[object] = field(default_factory=list)
    animal_id: str = ""

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


class RunDetector:
    """Detect individual runs (single traversals) from a stream of
    ``FootprintSequence`` objects.

    Parameters
    ----------
    fps : float
    walkway_length_mm : float
    min_frames_for_run : int
        Minimum number of frames a valid run must span.
    empty_frames_to_finish : int
        Number of consecutive frames with zero paws before a run
        is considered finished.
    """

    def __init__(
        self,
        fps: float = 100.0,
        walkway_length_mm: float = 400.0,
        min_frames_for_run: int = 20,
        empty_frames_to_finish: int = 30,
    ) -> None:
        self.fps = fps
        self.walkway_length_mm = walkway_length_mm
        self.min_frames_for_run = min_frames_for_run
        self.empty_frames_to_finish = empty_frames_to_finish
        self._state = RunDetectorState.WAITING
        self._frame_idx = 0
        self._run_start_idx = -1
        self._empty_count = 0
        self._buffer: list = []
        self.animal_id = ""

    @property
    def state(self) -> RunDetectorState:
        return self._state

    def process_frame(self, sequence: object) -> Optional[RunResult]:
        """Ingest one ``FootprintSequence`` and return ``RunResult``
        if a run has just finished (else ``None``).

        Parameters
        ----------
        sequence : FootprintSequence
        """
        self._frame_idx += 1
        any_stance = _has_any_stance(sequence)

        if self._state == RunDetectorState.WAITING:
            if any_stance:
                self._state = RunDetectorState.RUNNING
                self._run_start_idx = self._frame_idx
                self._empty_count = 0
                self._buffer = [sequence]

        elif self._state == RunDetectorState.RUNNING:
            self._buffer.append(sequence)
            if any_stance:
                self._empty_count = 0
            else:
                self._empty_count += 1
            if self._empty_count >= self.empty_frames_to_finish:
                self._state = RunDetectorState.FINISHED

        if self._state == RunDetectorState.FINISHED:
            n_active = self._frame_idx - self._run_start_idx - self._empty_count
            if n_active >= self.min_frames_for_run:
                result = RunResult(
                    start_frame=self._run_start_idx,
                    end_frame=self._frame_idx - self._empty_count,
                    sequences=list(self._buffer[:-self._empty_count or len(self._buffer)]),
                    animal_id=self.animal_id,
                )
            else:
                result = None  # too short, ignore
            # Reset for next run
            self._state = RunDetectorState.WAITING
            self._buffer = []
            self._empty_count = 0
            self._run_start_idx = -1
            return result

        return None

    def finish(self) -> Optional[RunResult]:
        """Force-finish the current run (called at end-of-stream)."""
        if self._state != RunDetectorState.RUNNING:
            return None
        n_active = self._frame_idx - self._run_start_idx
        if n_active >= self.min_frames_for_run:
            result = RunResult(
                start_frame=self._run_start_idx,
                end_frame=self._frame_idx,
                sequences=list(self._buffer),
                animal_id=self.animal_id,
            )
        else:
            result = None
        self._state = RunDetectorState.FINISHED
        return result

    def reset(self) -> None:
        self._state = RunDetectorState.WAITING
        self._frame_idx = 0
        self._run_start_idx = -1
        self._empty_count = 0
        self._buffer.clear()


def _has_any_stance(sequence: object) -> bool:
    """Return True if any paw in the sequence is in stance."""
    feet = getattr(sequence, "feet", None)
    if feet is None:
        return False
    for paw in ("LF", "RF", "LH", "RH"):
        foot = feet.get(paw)
        if foot is not None and getattr(foot, "is_in_stance", False):
            return True
    return False
