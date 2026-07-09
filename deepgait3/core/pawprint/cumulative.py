"""GPU-accelerated cumulative footprint rendering.

This module implements the track-level union mask algorithm that was
validated in ``experiment/cumulative_union_experiment.py``.  It is intended
to replace the pixel-wise max-merge logic currently scattered across
``demo_video_gpu.py``, ``pipeline.py:_save_cumulative`` and
``gui/gait_experiment_tab.py``.

Design goals
------------
- Keep masks and intensities on GPU for the heavy per-track union step.
- Fall back to CPU automatically when CUDA is unavailable.
- Provide both ``build_cumulative_union`` (numpy API) and
  ``CumulativeBuilder`` (stateful object) so the GUI can accumulate tracks
  incrementally if desired in the future.

Current behaviour
-----------------
For each ``FootprintTrack``:
  1. OR all per-frame binary masks to obtain the spatial footprint.
  2. Within that union, take the maximum green-delta intensity observed at
     each pixel across all frames of the track.
  3. Optionally apply morphological closing to bridge small palm-toe gaps.
  4. Merge the per-track result into a global canvas with ``torch.maximum``.

The output is a floating-point intensity image that can be rendered with
``render_overlay()`` (black background, green channel = intensity).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .tracker import FootprintTrack


def _device():
    """Return cuda if available, else cpu."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _track_to_global_tensors(
    track: FootprintTrack,
    H: int,
    W: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Convert a track's FootMask sequence to full-frame uint8 masks and float intensities.

    Returns ``(masks, intensities)`` arrays with shape ``(N, H, W)`` ready for
    GPU upload.  If the track has no valid footprints, returns ``(None, None)``.
    """
    if not track.foots:
        return None, None

    n = len(track.foots)
    masks = np.zeros((n, H, W), dtype=np.uint8)
    intensities = np.zeros((n, H, W), dtype=np.float32)

    for idx, (_, fm) in enumerate(track.foots):
        px1, py1, px2, py2 = fm.bbox_xyxy_padded
        if px2 <= px1 or py2 <= py1:
            continue
        ch, cw = py2 - py1, px2 - px1

        mask_crop = fm.mask_padded.astype(np.uint8)
        if mask_crop.shape[:2] != (ch, cw):
            mask_crop = cv2.resize(mask_crop, (cw, ch))
        intensity_crop = fm.raw_intensity_crop
        if intensity_crop.shape[:2] != (ch, cw):
            intensity_crop = cv2.resize(intensity_crop, (cw, ch))

        masks[idx, py1:py2, px1:px2] = mask_crop
        intensities[idx, py1:py2, px1:px2] = np.maximum(intensity_crop, 0.0) * mask_crop

    return masks, intensities


def build_cumulative_union(
    tracks: List[FootprintTrack],
    shape: Tuple[int, int],
    *,
    closing_kernel: Optional[int] = None,
    use_gpu: bool = True,
) -> np.ndarray:
    """Build a cumulative footprint image using track-level union masks.

    Parameters
    ----------
    tracks
        Finalized footprint tracks from ``IoUFootprintTracker.finalize()``.
    shape
        ``(H, W)`` frame shape.
    closing_kernel
        If > 1, apply elliptical morphological closing to each track's union
        mask before merging.  This can bridge small gaps between palm and toe
        regions.  ``None`` or ``<= 1`` disables closing.
    use_gpu
        Use CUDA if available.  Falls back to CPU automatically.

    Returns
    -------
    cum_intensity
        Floating-point image ``(H, W)`` with the maximum observed intensity at
        each pixel inside each track's union mask.
    """
    import cv2
    import torch

    H, W = shape
    device = _device() if use_gpu else torch.device("cpu")

    cum = torch.zeros((H, W), dtype=torch.float32, device=device)

    for track in tracks:
        masks_np, intensities_np = _track_to_global_tensors(track, H, W)
        if masks_np is None or masks_np.shape[0] == 0:
            continue

        masks_t = torch.from_numpy(masks_np).to(device)
        intensities_t = torch.from_numpy(intensities_np).to(device)

        # Union mask: OR across all frames of this track
        union_mask = masks_t.sum(dim=0) > 0  # (H, W) bool

        # Optional morphological closing on CPU (small masks, cheap)
        if closing_kernel and closing_kernel > 1:
            union_u8 = union_mask.cpu().numpy().astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (closing_kernel, closing_kernel),
            )
            closed_u8 = cv2.morphologyEx(union_u8, cv2.MORPH_CLOSE, kernel)
            union_mask = torch.from_numpy(closed_u8 > 0).to(device)

        # Max intensity inside union mask
        masked_intensity = torch.where(
            union_mask,
            intensities_t.max(dim=0).values,
            torch.zeros(1, device=device, dtype=torch.float32),
        )

        # Merge into global cumulative image
        cum = torch.maximum(cum, masked_intensity)

    return cum.cpu().numpy()


def build_cumulative_union_cpu(
    tracks: List[FootprintTrack],
    shape: Tuple[int, int],
    *,
    closing_kernel: Optional[int] = None,
) -> np.ndarray:
    """CPU-only version of ``build_cumulative_union`` for tests/edge cases."""
    return build_cumulative_union(tracks, shape, closing_kernel=closing_kernel, use_gpu=False)


def render_overlay(
    cum_intensity: np.ndarray,
    colormap: Optional[str] = "green",
) -> np.ndarray:
    """Render a cumulative intensity image as a coloured BGR overlay.

    Parameters
    ----------
    cum_intensity
        Floating-point image from ``build_cumulative_union``.
    colormap
        ``"green"`` for the traditional black-background green-on-white
        CatWalk-style output.  Currently only green is implemented.

    Returns
    -------
    overlay
        ``(H, W, 3)`` uint8 BGR image.
    """
    overlay = np.zeros((*cum_intensity.shape, 3), dtype=np.uint8)
    vmax = cum_intensity.max()
    if vmax > 0:
        norm = (cum_intensity / vmax * 255).astype(np.uint8)
        if colormap == "green":
            overlay[:, :, 1] = norm  # BGR green channel
        else:
            overlay[:, :, 1] = norm
    return overlay


__all__ = [
    "build_cumulative_union",
    "build_cumulative_union_cpu",
    "render_overlay",
]
