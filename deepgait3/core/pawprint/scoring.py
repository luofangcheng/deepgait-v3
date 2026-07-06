"""Color-based paw candidate scoring functions.

Four algorithms ported from ``deepgait-v2/test_paw_detection.py`` for use
in V3's A/B comparison framework against the v0.4.2 ``detect_blobs``
(which uses a relative ``G - bg_G`` background-difference score).

All four functions take a BGR frame and return a per-pixel score map
as ``np.int16`` of the same H×W shape.  Higher score = more "green-like".

Conventions:
- Higher = more green-paw-like (consistent with v2 ``delta = G - bg_G``).
- ``compute_*`` are pure functions (no I/O, no matplotlib) so they can be
  unit-tested and reused inside the per-frame loop of the A/B driver.
"""
from __future__ import annotations

import numpy as np
import cv2


def compute_exg(img_bgr: np.ndarray) -> np.ndarray:
    """Excess Green Index: ``ExG = 2G - R - B``.

    Range roughly [-510, +510] for uint8 inputs.  High positive = green dominant.
    Source: Woebbecke et al. 1995 (weed detection).
    """
    B = img_bgr[:, :, 0].astype(np.int16)
    G = img_bgr[:, :, 1].astype(np.int16)
    R = img_bgr[:, :, 2].astype(np.int16)
    return 2 * G - R - B


def compute_lab_astar(img_bgr: np.ndarray) -> np.ndarray:
    """CIELAB a* channel, **negated** so green scores positive (range ~[-127, +128]).

    CIELAB convention: a*<0 = green, a*>0 = red.  OpenCV stores it as uint8
    centered at 128, so subtract 128 to recover signed, then negate to match
    the V3 convention of "higher score = more green-like".
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return -(lab[:, :, 1].astype(np.int16) - 128)


def compute_exgr(img_bgr: np.ndarray) -> np.ndarray:
    """``ExGR = ExG - ExR`` (Meyer's formula ``ExR = 1.4 R - G``).

    More robust to varying lighting than plain ExG.
    Source: Meyer et al. 1998.
    """
    B = img_bgr[:, :, 0].astype(np.int16)
    G = img_bgr[:, :, 1].astype(np.int16)
    R = img_bgr[:, :, 2].astype(np.int16)
    exg = 2 * G - R - B
    exr = 1.4 * R - G
    return exg - exr.astype(np.int16)


def compute_color_distance_green(
    img_bgr: np.ndarray,
    target_bgr: tuple[int, int, int] = (30, 200, 30),
) -> np.ndarray:
    """Euclidean distance in RGB to a target green, inverted so higher = greener.

    ``score = 300 - |pixel - target|`` keeps the high-is-green convention.
    """
    target = np.array(target_bgr, dtype=np.float32)
    diff = img_bgr.astype(np.float32) - target.reshape(1, 1, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return (300.0 - dist).astype(np.int16)


def compute_cive(img_bgr: np.ndarray) -> np.ndarray:
    """Color Index of Vegetation Extraction.

    CIVE = 0.441*R - 0.811*G + 0.385*B + 18.78745

    Source: Kataoka et al. 2003 (sugar beet weed detection).
    Range roughly [-30, +30]; lower = more vegetation-like.  We NEGATE so
    the V3 convention (higher = greener) holds.
    """
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)
    cive = 0.441 * R - 0.811 * G + 0.385 * B + 18.78745
    # Negate + scale to int16 for consistency with other algorithms.
    return ((-cive) * 100).astype(np.int16)


def compute_vdvi(img_bgr: np.ndarray) -> np.ndarray:
    """Visible-band Difference Vegetation Index.

    VDVI = (2*G - R - B) / (2*G + R + B)

    Source: Xiaoqin et al. 2014 (drone-based crop monitoring).
    Range [-1, +1]; higher = more vegetation.  Scaled to int16 [-1000, +1000].
    """
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)
    num = 2 * G - R - B
    den = 2 * G + R + B + 1e-6
    return ((num / den) * 1000).astype(np.int16)


def compute_ngrdi(img_bgr: np.ndarray) -> np.ndarray:
    """Normalized Green-Red Difference Index.

    NGRDI = (G - R) / (G + R + 1e-6)

    Source: Tucker 1979; widely used in remote sensing.
    Range [-1, +1]; higher = more green-dominant.
    """
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)
    return ((G - R) / (G + R + 1e-6) * 1000).astype(np.int16)


def compute_mexg(img_bgr: np.ndarray) -> np.ndarray:
    """Modified Excess Green Index.

    MExG = 1.262*G - 0.884*R - 0.311*B

    Source: Woebbecke 1995 modification (more weighted to green, less to red).
    Range [-766, +766]; higher = greener.
    """
    B = img_bgr[:, :, 0].astype(np.int16)
    G = img_bgr[:, :, 1].astype(np.int16)
    R = img_bgr[:, :, 2].astype(np.int16)
    return 1.262 * G - 0.884 * R - 0.311 * B


def compute_gli(img_bgr: np.ndarray) -> np.ndarray:
    """Green Leaf Index (Louhaichi et al. 2001).

    GLI = (2*G - R - B) / (2*G + R + B + 1e-6)

    Similar to VDVI but published earlier for rangeland monitoring.
    Higher = greener.
    """
    B = img_bgr[:, :, 0].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    R = img_bgr[:, :, 2].astype(np.float32)
    num = 2 * G - R - B
    den = 2 * G + R + B + 1e-6
    return ((num / den) * 1000).astype(np.int16)


# ---------------------------------------------------------------------------
# Registry — keeps A/B driver code data-driven
# ---------------------------------------------------------------------------

SCORING_ALGORITHMS: dict[str, callable] = {
    "exg": compute_exg,
    "lab_astar": compute_lab_astar,
    "exgr": compute_exgr,
    "color_distance": compute_color_distance_green,
    "cive": compute_cive,
    "vdvi": compute_vdvi,
    "ngrdi": compute_ngrdi,
    "mexg": compute_mexg,
    "gli": compute_gli,
}


__all__ = [
    "compute_exg",
    "compute_lab_astar",
    "compute_exgr",
    "compute_color_distance_green",
    "compute_cive",
    "compute_vdvi",
    "compute_ngrdi",
    "compute_mexg",
    "compute_gli",
    "SCORING_ALGORITHMS",
]