"""Tests for FTIR footprint and intensity analysis.

Generates synthetic FTIR frames (black background + green paw prints) so the
full pipeline can be verified without real hardware images.
"""
from __future__ import annotations

import numpy as np
import pytest

from deepgait3.core._legacy import footprint, intensity


# ---------------------------------------------------------------------------
# Synthetic FTIR frame generator
# ---------------------------------------------------------------------------

def make_synthetic_ftir_frame(
    shape: tuple[int, int] = (480, 640),
    paw_centers: list[tuple[int, int]] | None = None,
    paw_radii: list[int] | None = None,
    noise: bool = True,
) -> np.ndarray:
    """Create a synthetic FTIR frame: black background + green elliptical paw prints.

    Args:
        shape: (H, W).
        paw_centers: list of (x, y) centers. Default: 4 paws at fixed positions.
        paw_radii: list of radii for each paw. Default: [30, 28, 32, 27].
        noise: add random green speckle noise around paw regions only.

    Returns:
        BGR frame (H×W×3, uint8).
    """
    h, w = shape
    if paw_centers is None:
        paw_centers = [(150, 200), (450, 200), (150, 350), (450, 350)]
    if paw_radii is None:
        paw_radii = [30, 28, 32, 27]

    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    for (cx, cy), r in zip(paw_centers, paw_radii):
        # Draw green ellipse (BGR: G=255)
        axes = (r, int(r * 0.7))  # slightly elongated
        angle = 30
        color = (0, 255, 0)  # pure green in BGR
        cv2.ellipse(frame, (cx, cy), axes, angle, 0, 360, color, -1)
        if noise:
            # Add small local noise inside the ellipse bounding box only
            rng = np.random.default_rng(42)
            x1, y1 = max(0, cx - r - 5), max(0, cy - r - 5)
            x2, y2 = min(w, cx + r + 5), min(h, cy + r + 5)
            local = rng.integers(0, 40, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
            local[:, :, 0] = local[:, :, 0] // 8   # B very low
            local[:, :, 2] = local[:, :, 2] // 8   # R very low
            roi = frame[y1:y2, x1:x2]
            roi = cv2.add(roi, local)
            frame[y1:y2, x1:x2] = roi

    return frame


# ---------------------------------------------------------------------------
# Footprint tests
# ---------------------------------------------------------------------------

def test_segment_green_finds_paws():
    """Segmentation should detect the green paw regions."""
    frame = make_synthetic_ftir_frame()
    mask = footprint.segment_green(frame)
    assert mask.dtype == np.uint8
    assert mask.shape == frame.shape[:2]
    # There should be significant white pixels (the 4 paws)
    assert mask.sum() > 1000


def test_clean_mask_removes_noise():
    """Morphological cleaning should remove small noise while keeping paws."""
    frame = make_synthetic_ftir_frame(noise=True)
    raw = footprint.segment_green(frame)
    cleaned = footprint.clean_mask(raw)
    # Cleaned mask should have fewer small components than raw
    n_raw = len(footprint.extract_footprints(raw, min_area_px=10))
    n_clean = len(footprint.extract_footprints(cleaned, min_area_px=10))
    assert n_clean <= n_raw


def test_extract_footprints_finds_four():
    """With 4 synthetic paws, should find 4 footprints."""
    frame = make_synthetic_ftir_frame()
    mask = footprint.segment_green(frame)
    mask = footprint.clean_mask(mask)
    fps = footprint.extract_footprints(mask, min_area_px=50)
    assert len(fps) == 4
    for fp in fps:
        assert fp.area_px > 500
        assert fp.major_axis > fp.minor_axis


def test_extract_footprints_px_per_mm():
    """px_per_mm should convert area to mm²."""
    frame = make_synthetic_ftir_frame()
    mask = footprint.segment_green(frame)
    mask = footprint.clean_mask(mask)
    px_per_mm = 10.0
    fps = footprint.extract_footprints(mask, px_per_mm=px_per_mm, min_area_px=50)
    for fp in fps:
        expected_mm2 = fp.area_px / (px_per_mm ** 2)
        assert fp.area_mm2 == pytest.approx(expected_mm2, abs=1e-6)


def test_match_footprints_to_paws():
    """Nearest-neighbor matching should assign footprints to correct paws."""
    frame = make_synthetic_ftir_frame(
        paw_centers=[(150, 200), (450, 200), (150, 350), (450, 350)],
    )
    mask = footprint.segment_green(frame)
    mask = footprint.clean_mask(mask)
    fps = footprint.extract_footprints(mask, min_area_px=50)
    # DLC positions roughly match the paw centers
    paw_positions = {
        "LeftFore": (150.0, 200.0),
        "RightFore": (450.0, 200.0),
        "LeftHind": (150.0, 350.0),
        "RightHind": (450.0, 350.0),
    }
    matched = footprint.match_footprints_to_paws(fps, paw_positions)
    matched_names = {f.matched_paw for f in matched if f.matched_paw is not None}
    assert len(matched_names) == 4
    assert "LeftFore" in matched_names
    assert "RightHind" in matched_names


def test_analyze_frame_one_shot():
    """analyze_frame convenience should run full pipeline."""
    frame = make_synthetic_ftir_frame()
    paw_positions = {
        "LeftFore": (150.0, 200.0),
        "RightFore": (450.0, 200.0),
        "LeftHind": (150.0, 350.0),
        "RightHind": (450.0, 350.0),
    }
    fps = footprint.analyze_frame(frame, paw_positions=paw_positions)
    assert len(fps) == 4
    matched = [f for f in fps if f.matched_paw is not None]
    assert len(matched) == 4


# ---------------------------------------------------------------------------
# Intensity tests
# ---------------------------------------------------------------------------

def test_extract_green_channel():
    """Green channel should be brightest where paws are."""
    frame = make_synthetic_ftir_frame()
    green = intensity.extract_green_channel(frame)
    assert green.shape == frame.shape[:2]
    assert green.dtype == np.uint8
    # Paw centers should have high green values
    assert green[200, 150] > 200
    assert green[350, 450] > 200
    # Background should be lower than paw regions (relative check)
    bg = green[50, 50]
    paw = green[200, 150]
    assert bg < paw


def test_measure_intensity():
    """Intensity within a mask should be high for green regions."""
    frame = make_synthetic_ftir_frame()
    green = intensity.extract_green_channel(frame)
    # Create a mask around the first paw
    mask = np.zeros_like(green, dtype=np.uint8)
    mask[170:230, 120:180] = 255
    mean_i, max_i, sum_i = intensity.measure_intensity(green, mask)
    assert mean_i > 100
    assert max_i == 255 or max_i > 200
    assert sum_i > 0


def test_analyze_intensities():
    """analyze_intensities should return one result per matched footprint."""
    frame = make_synthetic_ftir_frame()
    paw_positions = {
        "LeftFore": (150.0, 200.0),
        "RightFore": (450.0, 200.0),
        "LeftHind": (150.0, 350.0),
        "RightHind": (450.0, 350.0),
    }
    fps = footprint.analyze_frame(frame, paw_positions=paw_positions)
    intensities = intensity.analyze_intensities(frame, fps)
    assert len(intensities) == 4
    for ir in intensities:
        assert ir.mean_intensity > 0
        assert ir.max_intensity >= ir.mean_intensity
        assert ir.sum_intensity > 0
        assert ir.area_px > 0


def test_compute_asymmetry_perfect():
    """Identical left/right intensities → asymmetry = 1.0."""
    left = intensity.IntensityResult("LeftFore", 100.0, 200.0, 5000.0, 50)
    right = intensity.IntensityResult("RightFore", 100.0, 200.0, 5000.0, 50)
    asym = intensity.compute_asymmetry(left, right)
    assert asym.asymmetry_index == pytest.approx(1.0, abs=1e-6)
    assert asym.ratio == pytest.approx(1.0, abs=1e-6)


def test_compute_asymmetry_asymmetric():
    """Left 2× right → asymmetry = 2.0."""
    left = intensity.IntensityResult("LeftFore", 200.0, 400.0, 10000.0, 50)
    right = intensity.IntensityResult("RightFore", 100.0, 200.0, 5000.0, 50)
    asym = intensity.compute_asymmetry(left, right)
    assert asym.asymmetry_index == pytest.approx(2.0, abs=1e-6)
    assert asym.ratio == pytest.approx(2.0, abs=1e-6)


def test_analyze_asymmetries():
    """analyze_asymmetries should compute fore and hind pairs."""
    intensities = [
        intensity.IntensityResult("LeftFore", 100.0, 200.0, 5000.0, 50),
        intensity.IntensityResult("RightFore", 120.0, 220.0, 6000.0, 50),
        intensity.IntensityResult("LeftHind", 90.0, 180.0, 4500.0, 50),
        intensity.IntensityResult("RightHind", 95.0, 190.0, 4750.0, 50),
    ]
    asyms = intensity.analyze_asymmetries(intensities)
    assert len(asyms) == 2
    # Fore pair: 120/100 = 1.2
    fore = [a for a in asyms if "Fore" in a.pair[0]][0]
    assert fore.asymmetry_index == pytest.approx(1.2, abs=1e-6)


def test_analyze_frame_one_shot():
    """intensity.analyze_frame convenience should run full pipeline."""
    frame = make_synthetic_ftir_frame()
    paw_positions = {
        "LeftFore": (150.0, 200.0),
        "RightFore": (450.0, 200.0),
        "LeftHind": (150.0, 350.0),
        "RightHind": (450.0, 350.0),
    }
    intensities, asyms = intensity.analyze_frame(frame, paw_positions=paw_positions)
    assert len(intensities) == 4
    assert len(asyms) == 2
    for ir in intensities:
        assert ir.mean_intensity > 0
    for a in asyms:
        assert a.asymmetry_index >= 1.0
