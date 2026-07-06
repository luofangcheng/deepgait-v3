"""Structured result containers for gait analysis.

A GaitResults holds one animal's full analysis output:
  - per-paw results (stance/swing timing, stride, angle, width)
  - body-level summary (symmetry, frequency)

All durations in ms, all lengths in real-world units (after multiplier).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass(slots=True)
class PawResults:
    """Per-paw analysis output."""
    name: str                       # e.g. "RightFore"
    side: str                       # "left" | "right"
    limb: str                       # "fore" | "hind"

    # Timing
    stance_duration_ms: float = 0.0
    swing_duration_ms: float = 0.0
    n_strides: int = 0

    # Stride
    stride_length_mean: float = 0.0     # mean stride length (real units)
    stride_length_variability: float = 0.0
    stride_frequency_hz: float = 0.0

    # Angle
    paw_angle_mean_deg: float = 0.0     # mean during stance

    # Stance width
    stance_width_mean: float = 0.0      # distance to body axis (real units)

    # Per-frame arrays (for plotting)
    in_stance: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    stride_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    paw_angles: np.ndarray = field(default_factory=lambda: np.array([]))

    def as_summary_dict(self) -> dict[str, Any]:
        """Flat dict of scalar metrics (excludes per-frame arrays)."""
        d = asdict(self)
        # Drop large arrays from summary
        for k in ("in_stance", "stride_lengths", "paw_angles"):
            d.pop(k, None)
        return d


@dataclass(slots=True)
class GaitResults:
    """Full analysis output for one animal / one video."""
    # Per-paw results
    paws: dict[str, PawResults] = field(default_factory=dict)

    # Body-level metrics
    gait_symmetry_index: float = 0.0    # 0..1, 1 = perfect L/R symmetry

    # Metadata
    n_frames: int = 0
    fps: int = 0
    source_csv: str = ""

    def paw(self, name: str) -> PawResults:
        """Convenience accessor."""
        return self.paws[name]

    def summary_table(self) -> list[dict[str, Any]]:
        """One row per paw, ready for DataFrame/pandas."""
        rows = []
        for paw in self.paws.values():
            rows.append(paw.as_summary_dict())
        return rows

    def as_summary_dict(self) -> dict[str, Any]:
        """Flat dict of scalar metrics for the whole animal."""
        return {
            "gait_symmetry_index": self.gait_symmetry_index,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "source_csv": self.source_csv,
            "paws": {name: paw.as_summary_dict() for name, paw in self.paws.items()},
        }
