"""Stage 1 end-to-end smoke demo.

Synthesizes a tiny fTIR-like video (a single bright footprint sliding
across the walkway over ~30 frames) and runs ``PawPrintExtractor`` on
it. Verifies:

- The video opens and the warmup completes.
- At least one ``PawPrint`` is recovered.
- The recovered ``PawPrint`` has a non-zero peak area, valid time
  stamps, and matches the spatial layout of the synthetic input.

Run from ``deepgait-v3/``::

    python examples/stage1_smoke.py

Output: prints a one-line summary per PawPrint + a final pass/fail verdict.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Allow running the script from the repo root without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepgait3.core.pawprint import PawPrintExtractor


def synthesize_ftir_video(
    out_path: Path,
    n_warmup: int = 30,
    n_active: int = 30,
    height: int = 384,
    width: int = 1920,
    bg_value: int = 10,
    paw_intensity: int = 200,
    paw_radius_px: int = 12,
    start_x_px: int = 200,
    end_x_px: int = 600,
    cy_px: int = 150,
) -> Path:
    """Write a synthetic H.264 MP4 to ``out_path``."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 60.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter could not open {out_path}")
    # Warmup frames: empty walkway
    for _ in range(n_warmup):
        frame = np.full((height, width, 3), bg_value, dtype=np.uint8)
        writer.write(frame)
    # Active frames: a bright paw blob sliding from start_x to end_x
    for i in range(n_active):
        t = i / max(n_active - 1, 1)
        cx = int(start_x_px + (end_x_px - start_x_px) * t)
        frame = np.full((height, width, 3), bg_value, dtype=np.uint8)
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy_px) ** 2 <= paw_radius_px ** 2
        frame[mask] = (0, paw_intensity, 0)  # BGR: pure green channel
        writer.write(frame)
    writer.release()
    return out_path


def main() -> int:
    print("=" * 60)
    print("DeepGait v3 — Stage 1 smoke demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "synthetic_ftir.mp4"
        print(f"→ synthesizing {video_path}")
        synthesize_ftir_video(video_path)
        info = cv2.VideoCapture(str(video_path))
        n_frames = int(info.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(info.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info.release()
        print(f"  {n_frames} frames @ {w}×{h}")

        print("→ running PawPrintExtractor(balanced)")
        ext = PawPrintExtractor(
            px_per_mm=1.92, fps=60, sensitivity_mode="balanced",
            warmup_frames=30,
        )
        pawprints = ext(str(video_path))

    print(f"→ recovered {len(pawprints)} PawPrint(s)")
    for pp in pawprints:
        print(
            f"  print_id={pp.print_id} "
            f"td={pp.touchdown_frame} lo={pp.liftoff_frame} "
            f"peak={pp.peak_area_frame} area={pp.max_area_mm2:.2f}mm² "
            f"centroid(mm)={pp.peak_frame_centroid_xy_mm} "
            f"n_frames={pp.n_frames} duration={pp.duration_s:.3f}s"
        )

    # Verdict
    if not pawprints:
        print("FAIL: no pawprints detected from synthetic input")
        return 1
    pp = pawprints[0]
    assert pp.n_frames >= 2, f"too few frames: {pp.n_frames}"
    assert pp.duration_s > 0, f"non-positive duration: {pp.duration_s}"
    assert pp.max_area_mm2 > 0, f"zero peak area: {pp.max_area_mm2}"
    # Synthetic blob had radius 12 px @ 1.92 px/mm → area ≈ π·r²/mm²
    expected_area_mm2 = np.pi * (12 / 1.92) ** 2
    assert 0.5 * expected_area_mm2 < pp.max_area_mm2 < 2.0 * expected_area_mm2, (
        f"peak area {pp.max_area_mm2:.2f}mm² out of band "
        f"[{0.5*expected_area_mm2:.2f}, {2*expected_area_mm2:.2f}]"
    )
    print("PASS: Stage 1 end-to-end OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())