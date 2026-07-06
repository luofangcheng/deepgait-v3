"""DLC output I/O: read CSV, flip Y, filter low-likelihood frames.

Handles both DLC "old" multi-header CSV and "new" single-header CSV formats.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from deepgait3.core._legacy import bodyparts


def read_dlc_csv(
    path: str | Path,
    video_height: float | None = None,
    likelihood_threshold: float = 0.1,
    bodyparts_list: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read DLC CSV and return dict of bodypart -> (N, 3) array [x, y, likelihood].

    If video_height is given, Y is flipped: y = video_height - y_csv.
    Frames with likelihood < threshold get x,y set to NaN (but likelihood kept).

    Args:
        path: path to DLC CSV (e.g., *_DLC_resnet50_..._filtered.csv)
        video_height: pixel height of source video for Y-flip.
        likelihood_threshold: frames below this get NaN x,y.
        bodyparts_list: if None, use deepgait3.core._legacy.bodyparts.BODYPARTS_12.

    Returns:
        dict mapping bodypart name to ndarray of shape (N_frames, 3).
    """
    path = Path(path)
    # W16 fix (PKU-P0#1): validate the file before letting pandas
    # explode with an opaque EmptyDataError or KeyError. The PI cancelled
    # the file dialog, then typed an empty path into the line edit, and
    # the original code crashed deep in pandas with no friendly message.
    if not path.exists():
        raise FileNotFoundError(
            f"DLC CSV not found: {path}. Please pick a valid file."
        )
    if path.stat().st_size == 0:
        raise ValueError(
            f"DLC CSV at {path} is empty. Re-export from DeepLabCut."
        )
    # W16 fix (EMBL-P0#1): force UTF-8 so non-ASCII filenames / user
    # names ("Müller", accented characters) do not raise
    # UnicodeDecodeError on Windows default-cp1252 installs.
    try:
        df = pd.read_csv(path, header=[0, 1, 2], encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"DLC CSV at {path} is not UTF-8 decodeable. "
            f"Re-export from DeepLabCut as UTF-8. ({e})"
        ) from e
    except pd.errors.EmptyDataError as e:
        # Only reachable when the file has bytes but no usable rows.
        raise ValueError(
            f"DLC CSV at {path} has no data rows. Re-export from "
            f"DeepLabCut or check the file. ({e})"
        ) from e
    except pd.errors.ParserError as e:
        # Header=[0,1,2] demands 3 header lines; one-row files or
        # tab-separated files that aren't really DLC exports trip this.
        raise ValueError(
            f"DLC CSV at {path} does not look like a DeepLabCut export "
            f"(expected 3 header rows: scorer / bodypart / coords). "
            f"Re-export from DLC. ({e})"
        ) from e

    if len(df.columns) < 3:
        raise ValueError(
            f"DLC CSV at {path} does not look like a DeepLabCut export "
            f"(expected 3-row MultiIndex header scorer/bodypart/coords, "
            f"got {len(df.columns)} columns). Re-export from DLC."
        )

    # Flatten multi-index columns to simple names like "bodypart_x"
    # DLC columns: scorer, bodypart, coords -> (scorer, bodypart, x/y/likelihood)
    # We want to map bodypart -> x, y, likelihood columns
    scorer = df.columns[0][0]  # first-level header is usually uniform scorer name
    # Rebuild simple column index
    df.columns = ["_".join(col).strip() for col in df.columns.values]

    parts = bodyparts_list or bodyparts.BODYPARTS_12
    result: dict[str, np.ndarray] = {}
    for part in parts:
        # Try to find columns matching this bodypart
        x_col = _find_col(df.columns, part, "x")
        y_col = _find_col(df.columns, part, "y")
        l_col = _find_col(df.columns, part, "likelihood")
        if x_col is None or y_col is None or l_col is None:
            raise ValueError(f"Could not find columns for bodypart '{part}' in {path}")

        x = df[x_col].to_numpy(dtype=float, copy=True)
        y = df[y_col].to_numpy(dtype=float, copy=True)
        p = df[l_col].to_numpy(dtype=float, copy=True)

        if video_height is not None:
            y = video_height - y

        # Mask low-likelihood frames
        low = p < likelihood_threshold
        x[low] = np.nan
        y[low] = np.nan

        result[part] = np.column_stack((x, y, p))
    return result


def _find_col(cols: pd.Index, part: str, coord: str) -> str | None:
    """Find column name containing bodypart and coordinate suffix."""
    # Patterns: "scorer_bodypart_x", "bodypart_x", etc.
    candidates = [c for c in cols if part in c and coord in c.lower()]
    if not candidates:
        return None
    # Prefer exact match or shortest match
    return min(candidates, key=len)


def load_paw_keypoints(
    data: dict[str, np.ndarray],
    paw: bodyparts.Paw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract toe and heel (x, y) arrays from loaded DLC data for a paw.

    Returns: toe_x, toe_y, heel_x, heel_y  (each shape (N,))
    """
    toe = data[paw.toe]
    heel = data[paw.heel]
    return toe[:, 0], toe[:, 1], heel[:, 0], heel[:, 1]


def load_body_axis(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract Nose, Butt, MidPointRight, MidPointLeft -> CoM and axis endpoints.

    Returns:
        nose_x, nose_y, butt_x, butt_y, com_x, com_y
    """
    nose = data["Nose"]
    butt = data["Butt"]
    mpr = data["MidPointRight"]
    mpl = data["MidPointLeft"]
    com_x = (mpr[:, 0] + mpl[:, 0]) / 2.0
    com_y = (mpr[:, 1] + mpl[:, 1]) / 2.0
    return nose[:, 0], nose[:, 1], butt[:, 0], butt[:, 1], com_x, com_y
