"""IoU tracker on FootMask."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from .models import FootMask


@dataclass
class FootprintTrack:
    track_id: int
    foots: List[Tuple[int, FootMask]] = field(default_factory=list)
    ref_mask_global: Optional[np.ndarray] = None
    ref_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_frame: int = -1

    @property
    def n_frames(self): return len(self.foots)


class IoUFootprintTracker:
    def __init__(self, frame_shape, iou_min=0.3, max_gap_frames=3,
                 ref_window_frames=5):
        self.H, self.W = frame_shape
        self.iou_min = float(iou_min)
        self.max_gap_frames = int(max_gap_frames)
        self.ref_window = int(ref_window_frames)
        self.active: Dict[int, FootprintTrack] = {}
        self.closed: List[FootprintTrack] = []
        self.next_id = 0

    @staticmethod
    def _mask_iou(m1, m2, b1, b2):
        ux1, uy1, ux2, uy2 = (min(b1[0], b2[0]), min(b1[1], b2[1]),
                              max(b1[2], b2[2]), max(b1[3], b2[3]))
        if ux2 <= ux1 or uy2 <= uy1: return 0.0
        s1 = m1[uy1:uy2, ux1:ux2]
        s2 = m2[uy1:uy2, ux1:ux2]
        inter = int(np.logical_and(s1, s2).sum())
        if inter == 0: return 0.0
        union = int(np.logical_or(s1, s2).sum())
        return inter / union if union > 0 else 0.0

    def _foot_global_mask(self, fm):
        px1, py1, px2, py2 = fm.bbox_xyxy_padded
        full = np.zeros((self.H, self.W), dtype=bool)
        full[py1:py2, px1:px2] = fm.mask_padded
        return full, (px1, py1, px2, py2)

    def _refresh_ref_mask(self, track):
        recent = track.foots[-self.ref_window:]
        if not recent: return
        x1 = min(fm.bbox_xyxy_padded[0] for _, fm in recent)
        y1 = min(fm.bbox_xyxy_padded[1] for _, fm in recent)
        x2 = max(fm.bbox_xyxy_padded[2] for _, fm in recent)
        y2 = max(fm.bbox_xyxy_padded[3] for _, fm in recent)
        full = np.zeros((self.H, self.W), dtype=bool)
        for _, fm in recent:
            bx1, by1, bx2, by2 = fm.bbox_xyxy_padded
            full[by1:by2, bx1:bx2] |= fm.mask_padded
        track.ref_mask_global = full
        track.ref_bbox = (x1, y1, x2, y2)

    def update(self, frame_idx, footmasks):
        stale = [tid for tid, t in self.active.items()
                 if frame_idx - t.last_frame > self.max_gap_frames]
        for tid in stale:
            self.closed.append(self.active.pop(tid))
        if not footmasks: return
        foot_masks = [self._foot_global_mask(fm) for fm in footmasks]
        if self.active:
            tids = list(self.active.keys())
            tracks = [self.active[tid] for tid in tids]
            iou_mat = np.zeros((len(tids), len(footmasks)), dtype=np.float32)
            for ti, tr in enumerate(tracks):
                for bi, (m, bb) in enumerate(foot_masks):
                    iou_mat[ti, bi] = self._mask_iou(tr.ref_mask_global, m, tr.ref_bbox, bb)
            used_t, used_b = set(), set()
            flat = [(float(iou_mat[ti, bi]), ti, bi)
                    for ti in range(len(tids)) for bi in range(len(footmasks))]
            flat.sort(reverse=True)
            for iou, ti, bi in flat:
                if iou < self.iou_min: break
                if ti in used_t or bi in used_b: continue
                tid = tids[ti]
                tr = self.active[tid]
                tr.foots.append((frame_idx, footmasks[bi]))
                tr.last_frame = frame_idx
                self._refresh_ref_mask(tr)
                used_t.add(ti); used_b.add(bi)
            for bi in range(len(footmasks)):
                if bi in used_b: continue
                self._new_track(frame_idx, footmasks[bi], foot_masks[bi])
        else:
            for bi, fm in enumerate(footmasks):
                self._new_track(frame_idx, fm, foot_masks[bi])

    def _new_track(self, frame_idx, fm, foot_mask_pair):
        m, bb = foot_mask_pair
        tr = FootprintTrack(track_id=self.next_id, foots=[(frame_idx, fm)],
                            ref_mask_global=m, ref_bbox=bb,
                            last_frame=frame_idx)
        self.active[self.next_id] = tr
        self.next_id += 1

    def finalize(self):
        self.closed.extend(self.active.values())
        self.active.clear()
        return self.closed


__all__ = ["IoUFootprintTracker", "FootprintTrack"]
