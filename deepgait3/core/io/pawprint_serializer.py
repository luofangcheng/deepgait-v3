"""v0.4.2 serializer (same as v0.4.1)."""
from __future__ import annotations
from pathlib import Path
from typing import List
import pickle
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..pawprint.models import PawPrint


def save_pawprints(prints, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(prints, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pawprints(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pawprint_database(prints, root_dir):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "pawprints_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "print_id", "touchdown_frame", "liftoff_frame", "true_liftoff_frame",
            "n_frames", "n_extension_frames", "duration_s",
            "max_area_mm2", "peak_centroid_x_mm", "peak_centroid_y_mm",
            "peak_pressure", "decay_tau_ms", "decay_R2", "is_clean_liftoff",
            "dir_path",
        ])
        for p in prints:
            cx, cy = p.peak_frame_centroid_xy_mm
            dir_path = "prints/print_%03d" % p.print_id
            n_ext = sum(1 for d in p.decay_phase_mask if d)
            tau_str = ("%.4f" % p.decay_tau_ms) if p.decay_tau_ms is not None else ""
            r2_str = ("%.4f" % p.decay_R2) if p.decay_R2 is not None else ""
            w.writerow([
                p.print_id, p.touchdown_frame, p.liftoff_frame,
                p.true_liftoff_frame, p.n_frames, n_ext,
                "%.4f" % p.duration_s, "%.3f" % p.max_area_mm2,
                "%.3f" % cx, "%.3f" % cy, "%.3f" % p.peak_pressure,
                tau_str, r2_str, int(p.is_clean_liftoff), dir_path,
            ])


__all__ = ["save_pawprints", "load_pawprints", "save_pawprint_database"]
