"""Ground-truth evaluation harness for footprint detectors.

If you provide a JSON file mapping print_id → {frame_idx, cx_px, cy_px},
this module computes:
- precision  = TP / (TP + FP)
- recall     = TP / (TP + FN)
- F1 score
- per-print error in (x, y, t)

Without GT, returns None and the caller should fall back to the
heuristic score in ``autoresearch``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PrintGT:
    """One manually-labelled footprint."""
    print_id: int
    frame_idx: int           # touchdown frame (1-based, matches iter_frames)
    cx_px: float
    cy_px: float
    paw_id: Optional[str] = None  # if labelled, else None


def load_gt_json(path: str | Path) -> list[PrintGT]:
    """Load ground truth from a JSON file.

    Expected schema::

        [
          {"print_id": 0, "frame_idx": 31, "cx_px": 200, "cy_px": 150},
          {"print_id": 1, "frame_idx": 35, "cx_px": 250, "cy_px": 150,
           "paw_id": "LF"},
          ...
        ]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PrintGT(**{k: r[k] for k in ("print_id", "frame_idx", "cx_px", "cy_px")
                        if k in r},
                    paw_id=r.get("paw_id")) for r in raw]


@dataclass
class EvalResult:
    n_gt: int
    n_pred: int
    n_tp: int
    n_fp: int
    n_fn: int
    precision: float
    recall: float
    f1: float
    mean_error_px: float   # avg centroid error on matched pairs
    median_error_px: float


def evaluate(
    gt: list[PrintGT],
    predictions: list[dict],
    match_distance_px: float = 25.0,
    match_frame_tolerance: int = 5,
    require_paw_match: bool = False,
    spatial_check: bool = True,
) -> EvalResult:
    """Greedy nearest-neighbor match between GT prints and predictions.

    Each prediction must have keys ``frame_idx``, ``cx``, ``cy``.  Predictions
    may also have ``paw_id``; if ``require_paw_match=True``, two prints
    only match if their ``paw_id`` is equal.

    Set ``spatial_check=False`` to ignore pixel coordinates entirely and
    only match on frame_idx (useful when GT coords come from a software
    that uses a different coordinate system than the video).
    """
    matched_gt = set()
    matched_pred = set()
    errors: list[float] = []
    for pi, pred in enumerate(predictions):
        best_gi = -1
        best_d = float("inf")
        for gi, g in enumerate(gt):
            if gi in matched_gt:
                continue
            dt = abs(g.frame_idx - pred["frame_idx"])
            if dt > match_frame_tolerance:
                continue
            if require_paw_match and pred.get("paw_id") != g.paw_id:
                continue
            if spatial_check:
                d = math.hypot(g.cx_px - pred["cx"], g.cy_px - pred["cy"])
                if d > match_distance_px:
                    continue
            else:
                # When spatial_check is off, score by temporal proximity
                d = float(dt)
            if d < best_d:
                best_d = d
                best_gi = gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            if spatial_check:
                errors.append(best_d)
    n_tp = len(matched_gt)
    n_fp = len(predictions) - n_tp
    n_fn = len(gt) - n_tp
    p = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    r = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    mean_err = sum(errors) / len(errors) if errors else 0.0
    sorted_err = sorted(errors)
    median_err = sorted_err[len(sorted_err) // 2] if sorted_err else 0.0
    return EvalResult(
        n_gt=len(gt), n_pred=len(predictions),
        n_tp=n_tp, n_fp=n_fp, n_fn=n_fn,
        precision=p, recall=r, f1=f1,
        mean_error_px=mean_err, median_error_px=median_err,
    )


__all__ = ["PrintGT", "load_gt_json", "EvalResult", "evaluate"]