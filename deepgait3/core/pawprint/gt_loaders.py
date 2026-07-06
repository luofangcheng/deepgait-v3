"""Convert real-world ground-truth sources into ``PrintGT`` lists for the
benchmark module.

Supported sources:
- Commercial gait-analysis Excel exports (8-sheet structure, Chinese
  headers).  Converts per-print touchdown time + spatial (x_cm, y_cm)
  into per-frame (frame_idx, cx_px, cy_px) given a video's ``px_per_mm``
  + fps + origin conventions.
- A flat JSON file with the schema already documented in benchmark.py.

The commercial-software Excel format detected here:
- Sheet "时间序列数据" (Time-Series Data) has columns:
    脚印 | 图像 | 首次着地时间(s) | 步幅(cm) | 步伐位置坐标 | ...
- 25 prints, each row = one footprint with touchdown time + spatial
  coordinate in centimetres.
- Sheet "逐帧压力值" (Per-Frame Pressure) has 134 rows × 17 cols:
    帧数 | LF总 | LF最大 | LF最小 | LF平均 | RF总 | ... | LH/RH ...
- Sheet "单爪步态数据" (Per-Paw Gait Data): per-paw aggregates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from deepgait3.core.pawprint.benchmark import PrintGT


# ---------------------------------------------------------------------------
# Coordinate convention helpers
# ---------------------------------------------------------------------------

@dataclass
class CoordTransform:
    """Map physical (cm, s) → pixel (x_px, y_px, frame_idx).

    fTIR convention:
      - origin at one end of the walkway (e.g. start)
      - X along walking direction (cm)
      - Y across the walkway (cm)
      - ``px_per_mm`` is set from calibration (default 1.92 → 19.2 px/cm)
    """
    px_per_mm: float = 1.92
    fps: int = 60
    origin_x_px: float = 0.0    # pixel coordinate corresponding to X=0 cm
    origin_y_px: float = 0.0

    @property
    def px_per_cm(self) -> float:
        return self.px_per_mm * 10.0

    def time_to_frame(self, t_s: float) -> int:
        return max(1, int(round(t_s * self.fps)) + 1)

    def xycm_to_xy_px(self, x_cm: float, y_cm: float) -> tuple[float, float]:
        return (self.origin_x_px + x_cm * self.px_per_cm,
                self.origin_y_px + y_cm * self.px_per_cm)


# ---------------------------------------------------------------------------
# Excel loader
# ---------------------------------------------------------------------------

_XY_RE = re.compile(r"[\(\s]*(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)[\s\)]*")


def _parse_xy_cm(text: str) -> tuple[float, float]:
    """Parse ``"(133, 436)"`` or ``"133, 436"`` → (133.0, 436.0)."""
    if text is None:
        return (0.0, 0.0)
    s = str(text).strip()
    m = _XY_RE.search(s)
    if not m:
        return (0.0, 0.0)
    return (float(m.group(1)), float(m.group(2)))


def load_gt_from_xlsx(
    xlsx_path: str | Path,
    fps: int = 60,
    coord_unit: str = "px",
    drop_invalid: bool = True,
) -> list[PrintGT]:
    """Convert the commercial Excel format to ``PrintGT`` list.

    Reads sheet "时间序列数据" and uses columns:
      脚印, 首次着地时间(s), 步伐位置坐标

    Args:
        xlsx_path: path to the .xlsx file
        fps: frames per second of the source video
        coord_unit: "px" (the 步伐位置坐标 column is already in pixels,
                   which matches the test1.mp4 1920×384 frame) or "cm"
                   (then the helper converts to px using default fTIR
                   scale 1.92 px/mm).  Tested: the commercial export
                   writes pixel coordinates directly, so default is "px".
        drop_invalid: skip rows where touchdown time or step distance is 0

    Returns:
        list of ``PrintGT`` (one per valid footprint)
    """
    df = pd.read_excel(xlsx_path, sheet_name="时间序列数据")
    df.columns = [str(c).strip() for c in df.columns]

    col_paw = next(c for c in df.columns if c == "脚印")
    col_td = next(c for c in df.columns if "首次着地时间" in c)
    col_step = next((c for c in df.columns if "步幅" in c), None)
    col_xy = next(c for c in df.columns if "步伐位置" in c)

    out: list[PrintGT] = []
    for i, row in df.iterrows():
        paw = str(row[col_paw]).strip()
        td = float(row[col_td])
        step = float(row[col_step]) if col_step and pd.notna(row[col_step]) else 0.0
        x_raw, y_raw = _parse_xy_cm(row[col_xy])
        if drop_invalid and (td <= 0 or step <= 0):
            continue
        if coord_unit == "px":
            cx_px, cy_px = x_raw, y_raw
        else:
            t = CoordTransform(px_per_mm=1.92, fps=fps)
            cx_px, cy_px = t.xycm_to_xy_px(x_raw, y_raw)
        frame_idx = max(1, int(round(td * fps)) + 1)
        out.append(PrintGT(
            print_id=len(out),
            frame_idx=frame_idx,
            cx_px=cx_px,
            cy_px=cy_px,
            paw_id=paw,
        ))
    return out


# ---------------------------------------------------------------------------
# Per-frame pressure curves (for tuning the BiScale pressure gate)
# ---------------------------------------------------------------------------

def load_per_frame_pressure(xlsx_path: str | Path) -> pd.DataFrame:
    """Load the 逐帧压力值 sheet as a DataFrame indexed by frame number.

    Returns columns like ``LF_total``, ``LF_max``, ``LF_min``, ``LF_mean``,
    ``RF_total``, ... ``RH_mean``.
    """
    df = pd.read_excel(xlsx_path, sheet_name="逐帧压力值")
    df.columns = [str(c).strip() for c in df.columns]
    # First column is the frame index (renamed '帧数')
    frame_col = df.columns[0]
    df = df.rename(columns={frame_col: "frame_idx"})
    df = df.set_index("frame_idx")
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# JSON convenience
# ---------------------------------------------------------------------------

def save_gt_json(gt: list[PrintGT], path: str | Path) -> None:
    """Serialize a list of PrintGT to the JSON schema expected by benchmark.load_gt_json."""
    raw = [
        {
            "print_id": g.print_id,
            "frame_idx": g.frame_idx,
            "cx_px": g.cx_px,
            "cy_px": g.cy_px,
            "paw_id": g.paw_id,
        }
        for g in gt
    ]
    Path(path).write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "CoordTransform",
    "load_gt_from_xlsx",
    "load_per_frame_pressure",
    "save_gt_json",
]