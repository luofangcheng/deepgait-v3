"""Union-find distance clustering."""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from .models import FootMask


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1


def cluster_blobs_into_feet(blobs, D_merge_px=23.0, frame_shape=(384, 1920),
                              pressure_k=18.0, pressure_alpha=0.75,
                              pressure_b=8.0):
    if not blobs: return []
    n = len(blobs)
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i+1, n):
            dx = blobs[i]["cx_px"] - blobs[j]["cx_px"]
            dy = blobs[i]["cy_px"] - blobs[j]["cy_px"]
            if dx*dx + dy*dy <= D_merge_px*D_merge_px:
                uf.union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)
    H, W = frame_shape
    foots: List[FootMask] = []
    for indices in groups.values():
        ux1 = min(blobs[i]["bbox_xyxy_padded"][0] for i in indices)
        uy1 = min(blobs[i]["bbox_xyxy_padded"][1] for i in indices)
        ux2 = max(blobs[i]["bbox_xyxy_padded"][2] for i in indices)
        uy2 = max(blobs[i]["bbox_xyxy_padded"][3] for i in indices)
        h_u, w_u = uy2-uy1, ux2-ux1
        umask = np.zeros((h_u, w_u), dtype=bool)
        uraw = np.zeros((h_u, w_u), dtype=np.float32)
        ubg = np.zeros((h_u, w_u), dtype=np.float32)
        weight_cnt = np.zeros((h_u, w_u), dtype=np.float32)
        for i in indices:
            bx1, by1, bx2, by2 = blobs[i]["bbox_xyxy_padded"]
            sl_y = slice(by1-uy1, by2-uy1)
            sl_x = slice(bx1-ux1, bx2-ux1)
            umask[sl_y, sl_x] |= blobs[i]["mask_padded"]
            uraw[sl_y, sl_x] = np.maximum(uraw[sl_y, sl_x], blobs[i]["delta_padded"])
            ubg[sl_y, sl_x] += blobs[i]["bg_padded"]
            weight_cnt[sl_y, sl_x] += 1
        ubg = np.where(weight_cnt > 0, ubg / np.maximum(weight_cnt, 1), 0)
        total_area = sum(blobs[i]["area_px"] for i in indices)
        if total_area > 0:
            cx = sum(blobs[i]["cx_px"] * blobs[i]["area_px"] for i in indices) / total_area
            cy = sum(blobs[i]["cy_px"] * blobs[i]["area_px"] for i in indices) / total_area
        else:
            cx = cy = 0.0
        tx1 = min(blobs[i]["bbox_xyxy"][0] for i in indices)
        ty1 = min(blobs[i]["bbox_xyxy"][1] for i in indices)
        tx2 = max(blobs[i]["bbox_xyxy"][2] for i in indices)
        ty2 = max(blobs[i]["bbox_xyxy"][3] for i in indices)
        uraw_clean = np.maximum(uraw, 0.0)
        pmap = pressure_k * np.power(np.maximum(uraw_clean - pressure_b, 0.0), pressure_alpha)
        if umask.any():
            mean_int = float(uraw[umask].mean())
            peak_int = float(uraw.max())
        else:
            mean_int = peak_int = 0.0
        touches_edge = any(blobs[i]["touches_edge"] for i in indices)
        foots.append(FootMask(
            blob_indices=list(indices),
            centroid_px=(cx, cy),
            bbox_xyxy=(tx1, ty1, tx2, ty2),
            bbox_xyxy_padded=(ux1, uy1, ux2, uy2),
            mask_padded=umask, raw_intensity_crop=uraw,
            bg_intensity_crop=ubg, pressure_map=pmap.astype(np.float32),
            total_area_px=total_area,
            mean_intensity=mean_int, peak_intensity=peak_int,
            touches_edge=touches_edge,
        ))
    return foots


__all__ = ["cluster_blobs_into_feet"]
