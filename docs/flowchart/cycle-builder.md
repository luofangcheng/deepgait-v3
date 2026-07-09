# Track → FootprintCycle 装配

文件：`deepgait3/core/pawprint/cycle_builder.py`

**核心 API**：
- `build_cycles(tracks, fps, px_per_mm, min_frames=2) -> List[FootprintCycle]`
- 输出按 touchdown_frame 升序，cycle_id 从 1 开始递增

```mermaid
flowchart TD
    Start(["build_cycles tracks, fps, px_per_mm, min_frames"]) --> Loop["for each FootprintTrack"]
    Loop --> Short{"len track.foots 小于 min_frames?"}
    Short -->|"是"| Skip["continue 下一 track"]
    Short -->|"否"| PerFrame["for each frame_idx, fm in track.foots"]
    PerFrame --> Area["area_mm2 = total_area_px / px_per_mm pow 2\nintensity_stats = mean_intensity, peak_intensity, mean_pressure, peak_pressure"]
    Area --> Centroid["centroid_x_mm = centroid_px 0 / px_per_mm\ncentroid_y_mm = centroid_px 1 / px_per_mm"]
    Centroid --> Bbox["bbox_x1, y1, x2, y2"]
    Bbox --> Fr["build FrameRecord frame, area_mm2, intensity_stats, centroid_mm, bbox"]
    Fr --> NextFr{"还有 frame?"}
    NextFr -->|"是"| PerFrame
    NextFr -->|"否"| Peak["peak_area_idx = argmax areas\npeak_intensity_idx = argmax intensities\ntrue_liftoff = 最后一帧 mean_intensity 大于等于 decay_thr"]
    Peak --> Dur["duration_s = n_frames / fps\nloading, unloading, weight_bearing 段时长"]
    Dur --> Build["FootprintCycle track_id, frames, peak_area_idx, peak_intensity_idx, true_liftoff, durations"]
    Build --> Skip

    Skip --> NextTrack{"还有 track?"}
    NextTrack -->|"是"| Loop
    NextTrack -->|"否"| Sort["按 touchdown_frame 升序排序\ncycle_id = 1..N 分配"]
    Sort --> Done(["return List FootprintCycle"])

    style PerFrame fill:#bbf,stroke:#333
```

## 关键点

- **每个 track → 一个 cycle 候选**：track 内部所有帧聚合成一个脚印事件。
- **最短帧数过滤** `min_frames=2`：单帧的 track 被丢弃，避免噪声。
- **三段时间分解**：duration 拆成 loading / weight_bearing / unloading 三段，便于后续 CatWalk 步态分析。
- **true_liftoff**：用 mean_intensity 阈值判定"真正离地"瞬间，区别于"已离开但背景未稳定"。
- **物理单位换算**：`px / px_per_mm` 得到 mm 物理距离；`px² / px_per_mm²` 得到 mm² 物理面积。
- **cycle_id 不包含 paw 身份**：本步骤只做时间排序，paw_id 由 Stage 2 (`pose3d`) 注入。
- **数据契约**：`FootprintCycle` 字段定义在 `core/pawprint/models.py`，与 Stage 1 → Stage 2/3/4 的数据流一致。
- **零 Qt 依赖**，纯算法层。
