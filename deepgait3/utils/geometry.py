"""Geometry utilities translated from VisualGaitLab MathUtils.cs.

All functions operate on NumPy arrays for vectorized computation.
Coordinates are in mathematical convention (origin at bottom-left, Y up).
"""
from __future__ import annotations

import numpy as np


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint of two points.  Shape: (..., 2) -> (..., 2)."""
    return (a + b) / 2.0


def distance_between_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance.  Shape: (..., 2), (..., 2) -> (...)."""
    return np.linalg.norm(a - b, axis=-1)


def slope(y2: np.ndarray, y1: np.ndarray, x2: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Slope (y2-y1)/(x2-x1) with safe division."""
    dx = x2 - x1
    # Avoid division by zero; vertical line -> inf, handled by arctan2 later
    with np.errstate(divide='ignore', invalid='ignore'):
        m = (y2 - y1) / dx
    m = np.where(dx == 0, np.inf, m)
    return m


def angle_from_slope(m: np.ndarray) -> np.ndarray:
    """Angle in degrees from slope, handling all quadrants like VGL MathUtils.

    VGL logic (C# Math.Atan):
        - slope >= 0  -> angle = atan(slope) * 180/π
        - slope < 0   -> angle = 180 + atan(slope) * 180/π
    This gives [0, 180) for forward direction.
    """
    rad = np.arctan(m)
    deg = np.degrees(rad)
    deg = np.where(m >= 0, deg, 180.0 + deg)
    return deg


def paw_angle(
    toe: np.ndarray,
    heel: np.ndarray,
    ref_p1: np.ndarray,
    ref_p2: np.ndarray,
) -> np.ndarray:
    """Angle between paw long-axis (heel->toe) and reference line (ref_p1->ref_p2).

    All inputs shape (N, 2).  Returns angle in degrees [0, 180].
    """
    paw_dx = toe[:, 0] - heel[:, 0]
    paw_dy = toe[:, 1] - heel[:, 1]
    ref_dx = ref_p2[:, 0] - ref_p1[:, 0]
    ref_dy = ref_p2[:, 1] - ref_p1[:, 1]

    # Use arctan2 for robust quadrant handling
    paw_rad = np.arctan2(paw_dy, paw_dx)
    ref_rad = np.arctan2(ref_dy, ref_dx)

    diff = np.abs(paw_rad - ref_rad)
    # Normalize to [0, π]
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    return np.degrees(diff)


def distance_point_to_line(
    point: np.ndarray,
    line_p1: np.ndarray,
    line_p2: np.ndarray,
) -> np.ndarray:
    """Perpendicular distance from point to infinite line through line_p1 and line_p2.

    All inputs shape (..., 2).  Returns (...).
    """
    # Line vector
    dx = line_p2[..., 0] - line_p1[..., 0]
    dy = line_p2[..., 1] - line_p1[..., 1]
    # Point relative to line_p1
    px = point[..., 0] - line_p1[..., 0]
    py = point[..., 1] - line_p1[..., 1]
    # Cross product magnitude / line length
    cross = np.abs(dx * py - dy * px)
    line_len = np.sqrt(dx * dx + dy * dy)
    # Avoid div0: if line is a point, distance is distance to that point
    line_len = np.where(line_len == 0, 1, line_len)
    return cross / line_len


def find_runs(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """Find contiguous runs in a 1-D integer array.

    Returns list of (start_index, end_index, value) for each run.
    end_index is exclusive (Python slice convention).
    """
    if arr.size == 0:
        return []
    # Find boundaries where value changes
    diff = np.diff(arr)
    boundaries = np.where(diff != 0)[0] + 1
    # Split indices
    splits = np.concatenate(([0], boundaries, [arr.size]))
    runs = []
    for i in range(len(splits) - 1):
        s, e = int(splits[i]), int(splits[i + 1])
        runs.append((s, e, int(arr[s])))
    return runs


def flip_y(y: np.ndarray, video_height: float) -> np.ndarray:
    """Flip Y coordinates from image (top-left origin) to mathematical (bottom-left origin).

    VGL does this on CSV read: y = VideoHeight - y_csv.
    """
    return video_height - y
