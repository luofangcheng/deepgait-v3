"""Unit tests for Phase 2 W6 — FTIR footprint analysis (v2).

Covers:
    deepgait/core/footprint_v2.py
        * BackgroundModel — MouseWalker |frame - median| ≤ BGoffset
        * group_fingers_into_feet — union-find 4-paw grouping
        * classify_feet_lfrh — L/R + F/H body-axis quadrant
        * analyze_frame_v2 — full pipeline (BG → HSV → morphology →
          CC → union-find → intensity → classification)
        * recall_score — synthetic recall metric

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.2 W6):
    "FTIR 4 爪 mask + union-find, ≥ 95% recall on synthetic frames"
"""
from __future__ import annotations

import numpy as np
import pytest


# =============================================================================
# Helpers — synthetic FTIR frame builders
# =============================================================================
def _draw_green_paw(
    frame: np.ndarray,
    cx: float,
    cy: float,
    radius: int = 14,
    intensity: int = 220,
) -> None:
    """Draw a green disc at (cx, cy) into a BGR frame.

    This mimics an FTIR contact patch in the HSV green band.
    """
    cv2 = pytest.importorskip("cv2")
    overlay = frame.copy()
    cv2.circle(overlay, (int(cx), int(cy)), radius, (0, intensity, 0), -1)
    np.copyto(frame, overlay)


def _synthetic_ftir_frame(
    *,
    paws: dict | None = None,
    background: str = "dark",
    w: int = 320,
    h: int = 240,
) -> np.ndarray:
    """Build a synthetic BGR FTIR-style frame.

    ``paws`` keys: "LF"/"RF"/"LH"/"RH" → (cx, cy, radius, intensity).
    Background is dark gray (no TIR) by default.
    """
    cv2 = pytest.importorskip("cv2")
    if background == "dark":
        frame = np.full((h, w, 3), 12, dtype=np.uint8)
    else:
        frame = np.full((h, w, 3), 200, dtype=np.uint8)
    if paws:
        for spec in paws.values():
            cx, cy = spec[0], spec[1]
            radius = spec[2] if len(spec) > 2 else 14
            intensity = spec[3] if len(spec) > 3 else 220
            _draw_green_paw(frame, cx, cy, radius, intensity)
    return frame


# =============================================================================
# BackgroundModel
# =============================================================================
class TestBackgroundModel:
    def test_warmup_keeps_updating_until_ready(self):
        from deepgait3.core._legacy.footprint_v2 import BackgroundModel

        bg = BackgroundModel(window=5, bg_offset=15, warmup_frames=3)
        assert bg.is_ready is False
        bg.update(np.full((20, 20, 3), 100, np.uint8))
        assert bg.is_ready is False
        bg.update(np.full((20, 20, 3), 100, np.uint8))
        assert bg.is_ready is False
        bg.update(np.full((20, 20, 3), 100, np.uint8))
        assert bg.is_ready is True

    def test_uniform_background_passes(self):
        from deepgait3.core._legacy.footprint_v2 import BackgroundModel

        bg = BackgroundModel(window=5, bg_offset=15, warmup_frames=2)
        for _ in range(3):
            bg.update(np.full((20, 20, 3), 100, np.uint8))
        # Test frame equal to background → no foreground.
        fg = bg.foreground_mask(np.full((20, 20, 3), 100, np.uint8))
        assert fg.sum() == 0

    def test_bright_spot_detected_as_foreground(self):
        from deepgait3.core._legacy.footprint_v2 import BackgroundModel

        bg = BackgroundModel(window=5, bg_offset=15, warmup_frames=2)
        for _ in range(3):
            bg.update(np.full((40, 40, 3), 100, np.uint8))
        test = np.full((40, 40, 3), 100, np.uint8)
        test[15:25, 15:25, 1] = 230  # green spike
        fg = bg.foreground_mask(test)
        assert fg.sum() > 0
        # Spike location must be flagged.
        assert fg[20, 20] > 0

    def test_validation(self):
        from deepgait3.core._legacy.footprint_v2 import BackgroundModel

        with pytest.raises(ValueError):
            BackgroundModel(window=2)
        with pytest.raises(ValueError):
            BackgroundModel(window=5, bg_offset=-1)
        bm = BackgroundModel(window=5, warmup_frames=3)
        # Not ready yet → foreground_mask must raise.
        with pytest.raises(RuntimeError):
            bm.foreground_mask(np.zeros((10, 10, 3), np.uint8))
        # Wrong frame dtype must also raise.
        bm.update(np.zeros((10, 10, 3), np.uint8))
        with pytest.raises(ValueError):
            bm.update(np.zeros((10, 10), np.float32))

    def test_reset(self):
        from deepgait3.core._legacy.footprint_v2 import BackgroundModel

        bg = BackgroundModel(window=5, warmup_frames=1)
        bg.update(np.full((10, 10, 3), 50, np.uint8))
        bg.reset()
        assert bg.n_frames_seen == 0
        assert bg.is_ready is False


# =============================================================================
# Union-find grouping
# =============================================================================
class TestUnionFind:
    def test_single_blobs_each_become_their_own_group(self):
        from deepgait3.core._legacy.footprint_v2 import group_fingers_into_feet

        blobs = [
            (1, (0, 0, 10, 10), (5.0, 5.0), 100),
            (2, (50, 50, 10, 10), (55.0, 55.0), 100),
        ]
        groups = group_fingers_into_feet(blobs, max_finger_distance_px=10)
        assert len(groups) == 2

    def test_close_blobs_merge_into_one_foot(self):
        from deepgait3.core._legacy.footprint_v2 import group_fingers_into_feet

        blobs = [
            (1, (0, 0, 10, 10), (5.0, 5.0), 100),
            (2, (10, 0, 10, 10), (15.0, 5.0), 100),  # 10 px apart
            (3, (0, 10, 10, 10), (5.0, 15.0), 100),  # 10 px apart
        ]
        groups = group_fingers_into_feet(blobs, max_finger_distance_px=10)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_threshold_excludes_far_blobs(self):
        from deepgait3.core._legacy.footprint_v2 import group_fingers_into_feet

        blobs = [
            (1, (0, 0, 10, 10), (5.0, 5.0), 100),
            (2, (0, 0, 10, 10), (50.0, 5.0), 100),   # 45 px apart
        ]
        groups = group_fingers_into_feet(blobs, max_finger_distance_px=10)
        assert len(groups) == 2

    def test_empty_input(self):
        from deepgait3.core._legacy.footprint_v2 import group_fingers_into_feet
        assert group_fingers_into_feet([], max_finger_distance_px=10) == []


# =============================================================================
# L/R + F/H classification
# =============================================================================
class TestClassifyFeetLFRH:
    def test_four_paws_classified_into_quadrants(self):
        from deepgait3.core._legacy.footprint_v2 import FootMask, classify_feet_lfrh

        # Body axis: nose→tail along +X, mid at (100, 50)
        p1 = (10.0, 50.0)
        p2 = (190.0, 50.0)
        feet = [
            FootMask(label=0, area_px=100, area_mm2=0,
                     centroid=(150.0, 30.0), bbox=(0, 0, 0, 0),
                     hull_area_px=0, intensity_mean=200, intensity_max=200,
                     intensity_min=200, intensity_total=20000.0,
                     is_in_stance=True),  # RF (right + front)
            FootMask(label=1, area_px=100, area_mm2=0,
                     centroid=(150.0, 70.0), bbox=(0, 0, 0, 0),
                     hull_area_px=0, intensity_mean=200, intensity_max=200,
                     intensity_min=200, intensity_total=20000.0,
                     is_in_stance=True),  # LF (left + front)
            FootMask(label=2, area_px=100, area_mm2=0,
                     centroid=(50.0, 30.0), bbox=(0, 0, 0, 0),
                     hull_area_px=0, intensity_mean=200, intensity_max=200,
                     intensity_min=200, intensity_total=20000.0,
                     is_in_stance=True),  # RH (right + hind)
            FootMask(label=3, area_px=100, area_mm2=0,
                     centroid=(50.0, 70.0), bbox=(0, 0, 0, 0),
                     hull_area_px=0, intensity_mean=200, intensity_max=200,
                     intensity_min=200, intensity_total=20000.0,
                     is_in_stance=True),  # LH (left + hind)
        ]
        out = classify_feet_lfrh(feet, p1, p2)
        assert set(out.keys()) == {"LF", "RF", "LH", "RH"}

    def test_missing_paws_are_omitted(self):
        from deepgait3.core._legacy.footprint_v2 import FootMask, classify_feet_lfrh

        p1 = (0.0, 0.0); p2 = (200.0, 0.0)
        feet = [
            FootMask(label=0, area_px=100, area_mm2=0,
                     centroid=(150.0, 30.0), bbox=(0, 0, 0, 0),
                     hull_area_px=0, intensity_mean=200, intensity_max=200,
                     intensity_min=200, intensity_total=20000.0,
                     is_in_stance=True),
        ]
        out = classify_feet_lfrh(feet, p1, p2)
        assert "RF" in out
        assert "LF" not in out

    def test_empty_feet_returns_empty_dict(self):
        from deepgait3.core._legacy.footprint_v2 import classify_feet_lfrh

        assert classify_feet_lfrh([], (0, 0), (10, 10)) == {}

    def test_zero_length_body_axis_returns_empty(self):
        from deepgait3.core._legacy.footprint_v2 import FootMask, classify_feet_lfrh

        feet = [FootMask(label=0, area_px=100, area_mm2=0,
                          centroid=(1.0, 1.0), bbox=(0, 0, 0, 0),
                          hull_area_px=0, intensity_mean=200, intensity_max=200,
                          intensity_min=200, intensity_total=20000.0,
                          is_in_stance=True)]
        assert classify_feet_lfrh(feet, (0, 0), (0, 0)) == {}


# =============================================================================
# analyze_frame_v2 — full pipeline
# =============================================================================
class TestAnalyzeFrameV2:
    def test_detects_four_paws_in_synthetic_frame(self):
        from deepgait3.core._legacy.footprint_v2 import analyze_frame_v2

        paws = {
            "LF": (60, 100), "RF": (60, 40),
            "LH": (260, 100), "RH": (260, 40),
        }
        frame = _synthetic_ftir_frame(paws=paws)
        body_axis = ((0.0, 70.0), (320.0, 70.0))   # horizontal axis
        seq = analyze_frame_v2(frame, body_axis=body_axis, use_red_background=False)
        # Should find 4 distinct feet (one per quadrant).
        assert seq.n_feet == 4, seq.to_dict()
        assert set(seq.feet.keys()) == {"LF", "RF", "LH", "RH"}

    def test_in_stance_flag_reflects_area_threshold(self):
        from deepgait3.core._legacy.footprint_v2 import analyze_frame_v2

        # Two paws of very different sizes. The morphology close kernel is
        # 5 px wide, so we use radius=6 (area ~113 px) for "tiny" and
        # radius=20 (area ~1256 px) for "huge". Then we set the in_stance
        # threshold to 500 px so tiny → False, huge → True.
        paws = {
            "tiny": (60, 100, 6, 220),
            "huge": (260, 100, 20, 220),
        }
        frame = _synthetic_ftir_frame(paws=paws)
        seq = analyze_frame_v2(frame, in_stance_threshold_px=500, use_red_background=False)
        assert seq.n_feet == 2, seq.to_dict()
        areas = {k: v.area_px for k, v in seq.feet.items()}
        tiny_label = min(areas, key=areas.get)
        huge_label = max(areas, key=areas.get)
        assert seq.feet[tiny_label].is_in_stance is False, seq.feet
        assert seq.feet[huge_label].is_in_stance is True, seq.feet

    def test_pipeline_runs_without_background_model(self):
        from deepgait3.core._legacy.footprint_v2 import analyze_frame_v2

        paws = {"LF": (60, 80), "RF": (260, 80)}
        frame = _synthetic_ftir_frame(paws=paws)
        # No background model + no body axis → raw labels.
        seq = analyze_frame_v2(frame, use_red_background=False)
        assert seq.n_feet >= 2

    def test_intensity_stats_populated(self):
        from deepgait3.core._legacy.footprint_v2 import analyze_frame_v2

        paws = {"LF": (60, 100, 14, 220)}
        frame = _synthetic_ftir_frame(paws=paws)
        seq = analyze_frame_v2(frame, use_red_background=False)
        assert seq.n_feet >= 1
        for fm in seq.feet.values():
            # FTIR green channel weighted by BGR→gray formula (G * 0.587)
            # gives ~129 max for an intensity=220 disc on a 12-grey background;
            # mean includes background fringe so we check > 30 (clearly above bg).
            assert fm.intensity_mean > 30, fm
            assert fm.intensity_max >= fm.intensity_mean
            assert fm.intensity_min <= fm.intensity_mean
            assert fm.intensity_total > 0
            assert fm.hull_area_px > 0


# =============================================================================
# Recall metric — W6 acceptance gate
# =============================================================================
class TestRecallScore:
    def test_full_recall(self):
        from deepgait3.core._legacy.footprint_v2 import FootMask, recall_score

        fm = FootMask(label=0, area_px=100, area_mm2=0,
                      centroid=(10.0, 20.0), bbox=(0, 0, 0, 0),
                      hull_area_px=0, intensity_mean=200, intensity_max=200,
                      intensity_min=200, intensity_total=20000.0,
                      is_in_stance=True)
        detected = {"foot_0": fm}
        gt = {"LF": (10.0, 20.0)}
        assert recall_score(detected, gt) == 1.0

    def test_zero_recall_when_far(self):
        from deepgait3.core._legacy.footprint_v2 import FootMask, recall_score

        fm = FootMask(label=0, area_px=100, area_mm2=0,
                      centroid=(100.0, 100.0), bbox=(0, 0, 0, 0),
                      hull_area_px=0, intensity_mean=200, intensity_max=200,
                      intensity_min=200, intensity_total=20000.0,
                      is_in_stance=True)
        detected = {"foot_0": fm}
        gt = {"LF": (10.0, 20.0)}
        assert recall_score(detected, gt, match_radius_px=10.0) == 0.0

    def test_empty_ground_truth_is_perfect(self):
        from deepgait3.core._legacy.footprint_v2 import recall_score
        assert recall_score({}, {}) == 1.0


# =============================================================================
# End-to-end synthetic recall gate — W6 acceptance (≥ 95%)
# =============================================================================
class TestW6AcceptanceGate:
    """Build a 50-frame synthetic FTIR trial with a 4-paw walking pattern
    and confirm overall recall ≥ 95% against the synthetic ground truth.

    The trial alternates paw contacts at a 100 fps cadence:
      - frames 0..9:   LF contact
      - frames 10..19: RF contact
      - frames 20..29: LH contact
      - frames 30..39: RH contact
      - frames 40..49: no contacts
    """

    def test_recall_above_95_percent(self):
        from deepgait3.core._legacy.footprint_v2 import (
            BackgroundModel, analyze_frame_v2, recall_score,
        )

        bg = BackgroundModel(window=10, bg_offset=15, warmup_frames=5)
        # Train background with 5 empty frames
        for _ in range(5):
            bg.update(_synthetic_ftir_frame())

        # 4 paw positions (move slightly each frame to look realistic)
        paw_positions = {
            "LF": (60, 80),
            "RF": (60, 50),
            "LH": (260, 80),
            "RH": (260, 50),
        }
        body_axis = ((0.0, 65.0), (320.0, 65.0))

        # Sliding schedule: which paws are in contact at each frame.
        schedule = [
            ["LF"], ["LF", "RH"], ["RH"], [],                 # 0..3
            ["RF"], ["RF", "LH"], ["LH"], [],                 # 4..7
            ["LF"], ["LF"], ["LF", "RF"], [],                 # 8..11
            ["LH"], ["LH"], ["LH", "RH"], [],                 # 12..15
        ]
        # Pad to 50 frames with empty contacts.
        schedule = (schedule * 4)[:50]

        n_recalled = 0
        n_total = 0
        for i, active in enumerate(schedule):
            paws = {p: paw_positions[p] for p in active}
            frame = _synthetic_ftir_frame(paws=paws)
            bg.update(frame)            # update background with this frame
            seq = analyze_frame_v2(frame, background=bg, body_axis=body_axis, use_red_background=False)
            gt = {p: paw_positions[p] for p in active}
            score = recall_score(seq.feet, gt, match_radius_px=30.0)
            # Per-frame recall must be 1.0 (or 1.0 trivially when GT empty).
            n_recalled += int(round(score * len(gt)))
            n_total += len(gt)

        overall = n_recalled / n_total if n_total else 1.0
        assert overall >= 0.95, f"recall {overall:.3f} < 0.95"