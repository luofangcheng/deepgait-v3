"""Foot Pattern analysis — step sequence, regularity index, BOS, support patterns.

CatWalk XT "Coordination" + "Support" category parameters.

References
----------
* Timotius et al. (2023) — CatWalk XT parameter review (Table 1)
* Hamers et al. (2001) — Regularity Index, step sequence classification
* kb/08_catwalk_metrics_spec.md — deepgait parameter spec
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Step sequence classification
# ---------------------------------------------------------------------------
# Canonical step sequences in CatWalk (Hamers 2001).
# CA: cruciate A   — RF → LF → RH → LH
# CB: cruciate B   — LF → RF → LH → RH
# AA: alternate A  — RF → RH → LF → LH
# AB: alternate B  — LF → RH → RF → LH
# RA: rotary A     — RF → LF → LH → RH
# RB: rotary B     — LF → RF → RH → LH

_SEQUENCE_PATTERNS = {
    "CA": ["RF", "LF", "RH", "LH"],
    "CB": ["LF", "RF", "LH", "RH"],
    "AA": ["RF", "RH", "LF", "LH"],
    "AB": ["LF", "RH", "RF", "LH"],
    "RA": ["RF", "LF", "LH", "RH"],
    "RB": ["LF", "RF", "RH", "LH"],
}


def classify_step_sequence(
    stance_onsets: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Classify the step sequence pattern from 4-paw stance onset timing.

    For each step cycle (4 consecutive stance onsets, one per paw), the
    observed paw order is matched against the 6 canonical patterns.

    Parameters
    ----------
    stance_onsets : {paw: 1D array of stance onset frame indices}

    Returns
    -------
    Dict mapping pattern name → percentage of steps matching that pattern.
    Keys: CA, CB, AA, AB, RA, RB, regular (sum of all).
    """
    # Collate all onset frames across paws.
    all_onsets: List[Tuple[int, str]] = []
    for paw in ("LF", "RF", "LH", "RH"):
        arr = stance_onsets.get(paw)
        if arr is None or len(arr) == 0:
            continue
        for onset_frame in arr:
            all_onsets.append((int(onset_frame), paw))
    all_onsets.sort()

    if len(all_onsets) < 4:
        return {p: 0.0 for p in ["CA", "CB", "AA", "AB", "RA", "RB", "regular"]}

    # Sliding window of 4 onsets.
    counts: Dict[str, int] = {p: 0 for p in _SEQUENCE_PATTERNS}
    total_windows = 0
    for i in range(len(all_onsets) - 3):
        window = [all_onsets[j][1] for j in range(i, i + 4)]
        # Check each paw appears exactly once.
        if len(set(window)) < 4:
            continue
        total_windows += 1
        for pattern, seq in _SEQUENCE_PATTERNS.items():
            if window == seq:
                counts[pattern] += 1
                break

    if total_windows == 0:
        return {p: 0.0 for p in ["CA", "CB", "AA", "AB", "RA", "RB", "regular"]}

    result: Dict[str, float] = {}
    regular = 0
    for p in ["CA", "CB", "AA", "AB", "RA", "RB"]:
        pct = round(100.0 * counts[p] / total_windows, 1)
        result[p] = pct
        regular += counts[p]
    result["regular"] = round(100.0 * regular / total_windows, 1)
    return result


# ---------------------------------------------------------------------------
# Regularity Index (Hamers 2001)
# ---------------------------------------------------------------------------
def compute_regularity_index(
    step_sequence_result: Dict[str, float],
) -> float:
    """Regularity Index = % of normal step cycles (any of the 6 patterns).

    Hamers 2001: RI = (N_normal_step_cycles / N_total_strides) × 100.
    """
    return step_sequence_result.get("regular", 0.0)


# ---------------------------------------------------------------------------
# Base of Support (BOS)
# ---------------------------------------------------------------------------
def compute_bos(
    centroids_x: Dict[str, np.ndarray],
    centroids_y: Dict[str, np.ndarray],
    in_stance: Dict[str, np.ndarray] | None = None,
    px_per_mm: float = 1.0,
) -> Dict[str, float]:
    """Compute Base of Support (BOS) — medio-lateral distance between paw pairs.

    BOS front = mean |LF_x - RF_x| during stance.
    BOS hind  = mean |LH_x - RH_x| during stance.

    Returns cm.
    """
    result: Dict[str, float] = {
        "bos_front_paws_cm": 0.0,
        "bos_hind_paws_cm": 0.0,
    }
    for pair_label, paw_a, paw_b in [
        ("bos_front_paws_cm", "LF", "RF"),
        ("bos_hind_paws_cm", "LH", "RH"),
    ]:
        cx_a = centroids_x.get(paw_a)
        cx_b = centroids_x.get(paw_b)
        if cx_a is None or cx_b is None:
            continue
        n = min(len(cx_a), len(cx_b))
        if n == 0:
            continue
        # Only frames where BOTH paws are in stance.
        mask = np.ones(n, dtype=bool)
        if in_stance:
            sa = in_stance.get(paw_a)
            sb = in_stance.get(paw_b)
            if sa is not None and sb is not None:
                mask = (sa[:n] > 0) & (sb[:n] > 0)
        if not mask.any():
            continue
        dists = np.abs(cx_a[:n][mask] - cx_b[:n][mask])
        bos_mm = float(np.mean(dists)) / px_per_mm
        result[pair_label] = round(bos_mm / 10.0, 3)  # mm → cm
    return result


# ---------------------------------------------------------------------------
# Support patterns
# ---------------------------------------------------------------------------
def compute_support_patterns(
    in_stance: Dict[str, np.ndarray],
    paw_names: Tuple[str, ...] = ("LF", "RF", "LH", "RH"),
) -> Dict[str, float]:
    """Compute support category percentages.

    Returns:
        support_zero_pct, support_single_pct, support_diagonal_pct,
        support_girdle_pct, support_lateral_pct, support_three_pct,
        support_four_pct.
    """
    n = min(len(v) for v in in_stance.values()) if in_stance else 0
    if n == 0:
        return {k: 0.0 for k in [
            "support_zero_pct", "support_single_pct", "support_diagonal_pct",
            "support_girdle_pct", "support_lateral_pct",
            "support_three_pct", "support_four_pct",
        ]}

    # Frame-level paw count.
    frame_counts = np.zeros(n, dtype=int)
    for paw in paw_names:
        arr = in_stance.get(paw)
        if arr is not None:
            frame_counts += (arr[:n] > 0).astype(int)

    total = n
    zero = int((frame_counts == 0).sum())
    single = int((frame_counts == 1).sum())
    diag = int((frame_counts == 2).sum())  # simplified: any 2-paw
    triple = int((frame_counts == 3).sum())
    four = int((frame_counts == 4).sum())

    # More precise: diagonal = opposing paws (LF+RH or RF+LH)
    lf = in_stance.get("LF", np.zeros(n, dtype=np.int8))[:n] > 0
    rf = in_stance.get("RF", np.zeros(n, dtype=np.int8))[:n] > 0
    lh = in_stance.get("LH", np.zeros(n, dtype=np.int8))[:n] > 0
    rh = in_stance.get("RH", np.zeros(n, dtype=np.int8))[:n] > 0

    diagonal = int(((lf & rh) | (rf & lh)).sum())
    girdle = int(((lf & rf) | (lh & rh)).sum())  # front or hind pair
    lateral = int(((lf & lh) | (rf & rh)).sum())  # same side

    return {
        "support_zero_pct": round(100.0 * zero / total, 1),
        "support_single_pct": round(100.0 * single / total, 1),
        "support_diagonal_pct": round(100.0 * diagonal / total, 1),
        "support_girdle_pct": round(100.0 * girdle / total, 1),
        "support_lateral_pct": round(100.0 * lateral / total, 1),
        "support_three_pct": round(100.0 * triple / total, 1),
        "support_four_pct": round(100.0 * four / total, 1),
    }
