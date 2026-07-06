"""Unit tests for color-score based paw detection.

Verifies each of the 4 ported algorithms (ExG, Lab a*, ExGR, ColorDist)
produces a higher score for a green-ish blob than for a red-ish or
background pixel — the minimal invariant for any "is this a paw?"
detector.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepgait3.core.pawprint.scoring import (
    SCORING_ALGORITHMS,
    compute_cive,
    compute_color_distance_green,
    compute_exg,
    compute_exgr,
    compute_gli,
    compute_lab_astar,
    compute_mexg,
    compute_ngrdi,
    compute_vdvi,
)
from deepgait3.core.pawprint.scoring_detection import detect_blobs_from_score


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def three_color_image() -> np.ndarray:
    """BGR image (40×60) with three regions: green / red / dark background."""
    img = np.full((40, 60, 3), 10, dtype=np.uint8)  # dark background
    img[5:25, 5:25] = (10, 200, 10)   # green square (BGR)
    img[5:25, 30:50] = (10, 10, 200)  # red-ish square (BGR)
    return img


# ---------------------------------------------------------------------------
# Each score function must rank green > red > background
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,fn", [
    ("exg", compute_exg),
    ("lab_astar", compute_lab_astar),
    ("exgr", compute_exgr),
    ("color_distance", compute_color_distance_green),
    ("cive", compute_cive),
    ("vdvi", compute_vdvi),
    ("ngrdi", compute_ngrdi),
    ("mexg", compute_mexg),
    ("gli", compute_gli),
])
def test_score_green_higher_than_red(name, fn, three_color_image):
    """All algorithms must score the green blob higher than the red blob."""
    score = fn(three_color_image)
    green_mean = score[5:25, 5:25].mean()
    red_mean = score[5:25, 30:50].mean()
    bg_mean = score[25:40, 5:25].mean()
    assert green_mean > red_mean, f"{name}: green {green_mean:.1f} should beat red {red_mean:.1f}"
    assert green_mean > bg_mean, f"{name}: green {green_mean:.1f} should beat bg {bg_mean:.1f}"


def test_scoring_registry_has_all_algorithms():
    """Registry must contain all 9 algorithms (4 original + 5 new vegetation indices)."""
    expected = {"exg", "lab_astar", "exgr", "color_distance",
                "cive", "vdvi", "ngrdi", "mexg", "gli"}
    assert set(SCORING_ALGORITHMS) == expected
    assert len(SCORING_ALGORITHMS) == 9


# ---------------------------------------------------------------------------
# Score map shape + dtype invariants
# ---------------------------------------------------------------------------

def test_exg_dtype_and_shape():
    img = np.full((100, 200, 3), 100, dtype=np.uint8)
    s = compute_exg(img)
    assert s.shape == (100, 200)
    assert s.dtype == np.int16


def test_color_distance_uses_target():
    """Picking a target color equal to the input must give the max score (300)."""
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    s = compute_color_distance_green(img, target_bgr=(100, 100, 100))
    assert s.min() == 300  # distance 0 → score 300
    assert s.max() == 300


# ---------------------------------------------------------------------------
# Detection: threshold + connected components
# ---------------------------------------------------------------------------

def test_detect_blobs_from_score_basic():
    """Single green blob on dark bg → exactly 1 detected blob with right area."""
    img = np.full((40, 60, 3), 10, dtype=np.uint8)
    img[5:25, 5:25] = (10, 200, 10)
    score = compute_exg(img)
    mask, blobs = detect_blobs_from_score(score, threshold=80, min_area=5)
    assert len(blobs) == 1
    assert 18 * 18 * 0.5 < blobs[0]["area"] < 18 * 18 * 1.5
    # Centroid should be near the middle of the green square
    assert 10 < blobs[0]["cx"] < 20
    assert 10 < blobs[0]["cy"] < 20


def test_detect_blobs_drops_submin_area():
    """Tiny noise components below min_area get filtered out."""
    img = np.full((40, 60, 3), 10, dtype=np.uint8)
    # Two tiny green pixels far apart
    img[5, 5] = (10, 200, 10)
    img[35, 55] = (10, 200, 10)
    score = compute_exg(img)
    mask, blobs = detect_blobs_from_score(score, threshold=80, min_area=5)
    assert len(blobs) == 0  # both pixels < min_area after morphology