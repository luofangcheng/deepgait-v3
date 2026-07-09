# 视频去头掐尾（Trim by Footprint）

文件：`deepgait3/core/pawprint/trim.py`

**核心 API**：
- `find_trim_range(tracks)` — 返回 `(first_frame, last_frame)` 1-based 帧号；空 tracks 返回 `(0, -1)`
- `trim_video(src, dst, *, detector, iou_min=0.3, max_gap_frames=3, min_area_px=5, warmup_frames=30, progress_cb=None)` — 重编码到 dst
- `_read_all_frames(src_path)` — 读入全部帧
- `_write_trimmed(frames, dst, first_idx, last_idx, fps)` — 写 mp4v

```mermaid
flowchart TD
    Start(["trim_video src, dst, detector, ..."]) --> Read["_read_all_frames src\nfps, H, W, frames list"]
    Read --> Bg["bg_G = median green\n前 warmup_frames = 30 帧\n不够则用全帧"]
    Bg --> TrkInit["IoUFootprintTracker frame_shape"]
    TrkInit --> BatchLoop["for batch_start in 0..n_in step 16"]

    subgraph DetectLoop["YOLO 批推理 + tracker"]
        BatchLoop --> Det["detector.detect_batch batch_frames, bg_G, min_area_px"]
        Det --> Update["for i, fm_list in enumerate all_fm\ntracker.update batch_start + i + 1, fm_list"]
        Update --> Cb["progress_cb detect, pct, msg"]
        Cb --> NextBatch{"还有 batch?"}
        NextBatch -->|"是"| BatchLoop
        NextBatch -->|"否"| Fin["tracks = tracker.finalize"]
    end

    Fin --> Range["first_frame, last_frame = find_trim_range tracks"]
    Range --> Valid{"last_frame 小于 first_frame\n或 first_frame 小于 0?"}
    Valid -->|"是"| Raise["raise TrimError\n无脚印检测到"]
    Valid -->|"否"| Map["tracker 用 1-based\nframes list 是 0-based\nfirst_idx = max 0, first_frame - 1\nlast_idx = min n_in - 1, last_frame - 1"]
    Map --> Cb2["progress_cb encode, 0, 编码 区间"]
    Cb2 --> Write["_write_trimmed frames, dst, first_idx, last_idx, fps\ncv2.VideoWriter mp4v"]
    Write --> Ret(["return info dict\nsrc, dst, n_in, n_out, n_tracks, first_frame, last_frame, fps"])

    style DetectLoop fill:#f9f,stroke:#333,stroke-width:2px
```

### `find_trim_range` 细节

```mermaid
flowchart LR
    In["tracks list FootprintTrack"] --> Emp{"tracks 是否空?"}
    Emp -->|"是"| Z["return 0, -1"]
    Emp -->|"否"| Firsts["firsts = t.foots 0 0 for t in tracks if t.foots"]
    Firsts --> Lasts["lasts = t.foots -1 0 for t in tracks if t.foots"]
    Lasts --> Out["return min firsts, max lasts"]
```

## 关键点

- **复用现有 YOLO + IoU tracker**，不重新发明 detection 链路。
- **tracker 帧号 1-based**：`tracker.update(batch_start + i + 1, ...)`，`i=0` 表示 `batch_start` 对应的 0-based 帧。
- **写帧时必须 -1 转 0-based**：`first_idx = first_frame - 1`。
- **空 tracks 触发 TrimError**，让调用方决定是否忽略（不静默吃错）。
- **进度回调两阶段**：`phase="detect"` 和 `phase="encode"`，GUI 可以分别显示。
- **暖身帧数** `warmup_frames=30`：30 帧用于估计中值背景，不够则退化为全帧。
- **输出位置**：由调用方决定，默认 `projects/<name>/rawdata/videos/trimmed/<animal_id>.mp4`。
- **零 Qt 依赖**，可在 `TrimWorker` (QThread) 中调用。
