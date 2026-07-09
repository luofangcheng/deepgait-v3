# IoU 爪印跟踪

文件：`deepgait3/core/pawprint/tracker.py`

**核心 API**：
- `FootprintTrack(track_id, foots=[(frame_idx, FootMask)], ref_mask_global, ref_bbox, last_frame)`
- `IoUFootprintTracker(frame_shape, iou_min=0.3, max_gap_frames=3, ref_window_frames=5)`
- `update(frame_idx, footmasks)` — 每帧调用一次
- `finalize()` — 处理完最后帧调用，返回所有 track 列表

```mermaid
flowchart TD
    Start(["update frame_idx, footmasks"]) --> Stale["stale = tid for tid, t in active\nif frame_idx - t.last_frame 大于 max_gap_frames"]
    Stale --> CloseStale["for tid in stale\nclosed.append active.pop tid"]
    CloseStale --> Empty{"footmasks 是否空?"}
    Empty -->|"是"| Return(["return"])
    Empty -->|"否"| GlobalMasks["foot_masks = _foot_global_mask fm for each detection"]
    GlobalMasks --> HasActive{"active 是否非空?"}
    HasActive -->|"否"| NewAll["for bi, fm in enumerate footmasks\n_new_track frame_idx, fm, foot_masks bi"]
    HasActive -->|"是"| IoUMat["iou_mat = zeros len tids, len footmasks\nfor ti, tr in tracks\n  for bi, m, bb in foot_masks\n    iou_mat ti, bi = _mask_iou tr.ref_mask_global, m, tr.ref_bbox, bb"]
    IoUMat --> Flat["flat = iou_mat ti, bi, ti, bi for all i, j\nflat.sort reverse=True\n贪心按 IoU 从大到小匹配"]
    Flat --> Loop["for iou, ti, bi in flat"]
    Loop --> Thr{"iou 大于等于 iou_min?"}
    Thr -->|"否"| Unmatched["for bi not in used_b\n_new_track frame_idx, footmasks bi, foot_masks bi"]
    Thr -->|"是"| Used{"ti in used_t or bi in used_b?"}
    Used -->|"是"| Loop
    Used -->|"否"| Append["tr.foots.append frame_idx, footmasks bi\ntr.last_frame = frame_idx\n_refresh_ref_mask tr\nused_t.add ti, used_b.add bi"]
    Append --> Loop
    Unmatched --> Return
    NewAll --> Return

    subgraph Refresh["_refresh_ref_mask track"]
        RS["recent = track.foots 负 ref_window 后 5 帧"]
        RS --> Union["遍历 recent\n构造 full = zeros H, W bool\nfull 累加 OR mask_padded\n保存为 track.ref_mask_global\nref_bbox = minmax bbox_xyxy_padded"]
    end

    style Refresh fill:#bbf,stroke:#333
```

## 关键点

- **贪心匹配**：按 IoU 从大到小，先解决高重叠对，再处理剩余 detection。
- **track 关闭条件**：当 `frame_idx - last_frame > max_gap_frames` 时被移入 `closed`。
- **ref_mask 滑窗**：只保留最近 5 帧的 union mask，**容忍慢形变**（如脚掌抬起/落下的轮廓变化）。
- **`_foot_global_mask`**：把 `mask_padded` 投影回全帧 `bool[H, W]`，便于跨帧 IoU 计算。
- **track 互斥**：一个 detection 只属于一个 track；已用过的 track/detection 跳过。
- **`finalize()`** 把 active 全部移入 closed，返回完整 track 列表。
- **零 GPU 依赖**，全部 CPU numpy，<1ms/帧。
