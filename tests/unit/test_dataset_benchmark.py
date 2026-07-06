"""Unit tests for the dataset loader and benchmark evaluation harness.

``dataset.py`` wraps fTIR videos uniformly. ``benchmark.py`` computes
precision/recall/F1 against user-supplied ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deepgait3.core.pawprint.benchmark import (
    PrintGT,
    evaluate,
    load_gt_json,
)
from deepgait3.core.pawprint.dataset import (
    collect_videos,
    iter_frames,
    video_info,
)


# ---------------------------------------------------------------------------
# dataset.collect_videos
# ---------------------------------------------------------------------------

def test_collect_videos_finds_mp4_recursively(tmp_path: Path):
    """Recursively find .mp4 files in nested directories."""
    (tmp_path / "sub1").mkdir()
    (tmp_path / "sub1" / "a.mp4").write_bytes(b"")
    (tmp_path / "sub2").mkdir()
    (tmp_path / "sub2" / "b.mp4").write_bytes(b"")
    (tmp_path / "sub2" / "c.AVI").write_bytes(b"")  # case insensitive
    (tmp_path / "ignore.txt").write_bytes(b"")
    found = collect_videos([tmp_path])
    names = sorted(p.name for p in found)
    assert names == ["a.mp4", "b.mp4", "c.AVI"]


def test_collect_videos_dedups(tmp_path: Path):
    """Same path passed twice is deduped."""
    p = tmp_path / "a.mp4"
    p.write_bytes(b"")
    found = collect_videos([p, p, tmp_path])
    assert len(found) == 1
    assert found[0].name == "a.mp4"


def test_collect_videos_handles_missing_paths(tmp_path: Path):
    """Missing roots are skipped, not raised."""
    missing = tmp_path / "does-not-exist"
    found = collect_videos([missing])
    assert found == []


# ---------------------------------------------------------------------------
# dataset.video_info + iter_frames
# ---------------------------------------------------------------------------

def test_video_info_round_trip(tmp_path: Path):
    """Write a tiny mp4, verify video_info reads metadata correctly."""
    import cv2
    path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h, fps, n = 100, 50, 30.0, 10
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened()
    for _ in range(n):
        writer.write(np.full((h, w, 3), 50, dtype=np.uint8))
    writer.release()

    info = video_info(path)
    assert info.width == w
    assert info.height == h
    assert info.n_frames == n
    assert info.fps == pytest.approx(fps, abs=0.1)
    assert info.duration_s == pytest.approx(n / fps, abs=0.05)


def test_iter_frames_yields_one_based_index(tmp_path: Path):
    """Frame indices from iter_frames are 1-based (matches detector semantics)."""
    import cv2
    path = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (50, 50))
    for _ in range(5):
        writer.write(np.full((50, 50, 3), 0, dtype=np.uint8))
    writer.release()

    indices = [i for i, _ in iter_frames(path)]
    assert indices == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# benchmark.load_gt_json + evaluate
# ---------------------------------------------------------------------------

def test_load_gt_json(tmp_path: Path):
    path = tmp_path / "gt.json"
    path.write_text(json.dumps([
        {"print_id": 0, "frame_idx": 31, "cx_px": 200.5, "cy_px": 150.0},
        {"print_id": 1, "frame_idx": 40, "cx_px": 250.0, "cy_px": 150.0,
         "paw_id": "LF"},
    ]))
    gt = load_gt_json(path)
    assert len(gt) == 2
    assert gt[0].cx_px == 200.5
    assert gt[1].paw_id == "LF"


def test_evaluate_perfect_match():
    """Predictions exactly matching GT → precision=recall=F1=1, error=0."""
    gt = [PrintGT(0, 31, 100, 100), PrintGT(1, 40, 200, 200)]
    preds = [{"frame_idx": 31, "cx": 100, "cy": 100},
             {"frame_idx": 40, "cx": 200, "cy": 200}]
    r = evaluate(gt, preds, match_distance_px=10.0)
    assert r.n_tp == 2 and r.n_fp == 0 and r.n_fn == 0
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0
    assert r.mean_error_px == 0.0


def test_evaluate_all_misses():
    """No predictions match → precision=0, recall=0."""
    gt = [PrintGT(0, 31, 100, 100)]
    preds = [{"frame_idx": 99, "cx": 500, "cy": 500}]  # far away
    r = evaluate(gt, preds, match_distance_px=10.0, match_frame_tolerance=5)
    assert r.n_tp == 0
    assert r.n_fp == 1
    assert r.n_fn == 1
    assert r.precision == 0.0 and r.recall == 0.0 and r.f1 == 0.0


def test_evaluate_within_distance_tolerance():
    """A prediction 8 px off the GT centre still counts as a TP."""
    gt = [PrintGT(0, 31, 100, 100)]
    preds = [{"frame_idx": 31, "cx": 105, "cy": 105}]   # 7.07 px off
    r = evaluate(gt, preds, match_distance_px=10.0)
    assert r.n_tp == 1
    assert r.mean_error_px == pytest.approx(7.07, abs=0.1)


def test_evaluate_frame_tolerance_filter():
    """A prediction too far in time is excluded from matching."""
    gt = [PrintGT(0, 31, 100, 100)]
    preds = [{"frame_idx": 50, "cx": 100, "cy": 100}]   # frame delta 19
    r = evaluate(gt, preds, match_distance_px=10.0, match_frame_tolerance=5)
    assert r.n_tp == 0
    assert r.n_fp == 1
    assert r.n_fn == 1