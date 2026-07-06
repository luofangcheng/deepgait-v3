"""FTIR intensity analysis: green-channel statistics and asymmetry index.

Physics:
    FTIR green light (525 nm) escapes at paw contact points.
    Contact pressure ∝ light intensity.  Intensity is relative (depends on LED
    brightness, camera exposure, animal weight), so cross-group comparison
    must use the **asymmetry index** (ipsi / contra) rather than absolute values.

Pipeline:
    1. Extract green channel from BGR frame
    2. Apply footprint mask (from footprint module)
    3. Compute mean / max / sum intensity per footprint
    4. Asymmetry index = ipsi / contra (or contra / ipsi, whichever > 1)

Output:
    - Per-paw: mean_intensity, max_intensity, sum_intensity
    - Body-level: asymmetry_index (≥1, 1=perfect symmetry)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from deepgait3.core._legacy import footprint


@dataclass(slots=True)
class IntensityResult:
    """Intensity metrics for a single paw/footprint."""
    paw_name: str
    mean_intensity: float
    max_intensity: float
    sum_intensity: float
    area_px: int


@dataclass(slots=True)
class AsymmetryResult:
    """Pair-wise asymmetry between two paws (e.g. left vs right)."""
    pair: tuple[str, str]          # (paw_a, paw_b)
    asymmetry_index: float         # ≥1, ratio of larger/smaller
    ratio: float                   # raw ratio (may be <1)


def extract_green_channel(frame: np.ndarray) -> np.ndarray:
    """Extract the green channel from a BGR frame.

    Args:
        frame: BGR image (H×W×3).

    Returns:
        Green channel (H×W, uint8).
    """
    return frame[:, :, 1]


def measure_intensity(
    green: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float, float]:
    """Compute mean, max, and sum intensity within a binary mask.

    Args:
        green: single-channel intensity image.
        mask: binary mask (same shape as green).

    Returns:
        (mean, max, sum) intensity values.
    """
    masked = green[mask > 0]
    if masked.size == 0:
        return 0.0, 0.0, 0.0
    return float(masked.mean()), float(masked.max()), float(masked.sum())


def analyze_intensities(
    frame: np.ndarray,
    footprints: list[footprint.Footprint],
) -> list[IntensityResult]:
    """Measure green-channel intensity for each footprint.

    Args:
        frame: BGR image.
        footprints: detected footprints (with centroids/bboxes).

    Returns:
        List of IntensityResult, one per matched footprint.
    """
    green = extract_green_channel(frame)
    results: list[IntensityResult] = []
    for fp in footprints:
        # Build per-footprint mask from bbox (fast) or full component mask (precise)
        # Use bbox for speed; full component mask would need labels array
        x, y, w, h = fp.bbox
        region_mask = np.zeros_like(green, dtype=np.uint8)
        region_mask[y:y+h, x:x+w] = 255
        mean_i, max_i, sum_i = measure_intensity(green, region_mask)
        results.append(IntensityResult(
            paw_name=fp.matched_paw or f"unmatched_{fp.label}",
            mean_intensity=mean_i,
            max_intensity=max_i,
            sum_intensity=sum_i,
            area_px=fp.area_px,
        ))
    return results


def compute_asymmetry(
    left: IntensityResult,
    right: IntensityResult,
    metric: str = "mean_intensity",
) -> AsymmetryResult:
    """Compute asymmetry index between two paws.

    Args:
        left, right: IntensityResult for the two paws.
        metric: which intensity metric to compare (mean_intensity, max_intensity, sum_intensity).

    Returns:
        AsymmetryResult with asymmetry_index ≥ 1 (1 = perfect symmetry).
    """
    val_l = getattr(left, metric)
    val_r = getattr(right, metric)
    if val_l == 0 and val_r == 0:
        ratio = 1.0
    elif val_l == 0 or val_r == 0:
        ratio = float("inf")
    else:
        ratio = val_l / val_r
    asym = max(ratio, 1.0 / ratio) if ratio != 0 and ratio != float("inf") else float("inf")
    return AsymmetryResult(
        pair=(left.paw_name, right.paw_name),
        asymmetry_index=asym,
        ratio=ratio,
    )


def analyze_asymmetries(
    intensities: list[IntensityResult],
    pairs: Sequence[tuple[str, str]] | None = None,
) -> list[AsymmetryResult]:
    """Compute asymmetry for all left/right pairs.

    Args:
        intensities: list of per-paw IntensityResult.
        pairs: list of (left_name, right_name) tuples.  Default: fore and hind pairs.

    Returns:
        List of AsymmetryResult.
    """
    if pairs is None:
        pairs = [
            ("LeftFore", "RightFore"),
            ("LeftHind", "RightHind"),
        ]
    by_name = {r.paw_name: r for r in intensities}
    results: list[AsymmetryResult] = []
    for l_name, r_name in pairs:
        if l_name not in by_name or r_name not in by_name:
            continue
        results.append(compute_asymmetry(by_name[l_name], by_name[r_name]))
    return results


# ---------------------------------------------------------------------------
# One-shot convenience
# ---------------------------------------------------------------------------

def analyze_frame(
    frame: np.ndarray,
    paw_positions: dict[str, tuple[float, float]] | None = None,
    px_per_mm: float | None = None,
    hsv_lower: tuple[int, int, int] = (35, 50, 30),
    hsv_upper: tuple[int, int, int] = (85, 255, 255),
    min_area_px: int = 50,
) -> tuple[list[IntensityResult], list[AsymmetryResult]]:
    """Full FTIR intensity pipeline for a single frame.

    Args:
        frame: BGR image.
        paw_positions: optional DLC paw positions for footprint matching.
        px_per_mm: optional calibration.
        hsv_lower/upper: green segmentation thresholds.
        min_area_px: minimum footprint area.

    Returns:
        (intensities, asymmetries)
    """
    from deepgait3.core._legacy import footprint as fp
    footprints = fp.analyze_frame(frame, paw_positions, px_per_mm, hsv_lower, hsv_upper, min_area_px)
    intensities = analyze_intensities(frame, footprints)
    asymmetries = analyze_asymmetries(intensities)
    return intensities, asymmetries
