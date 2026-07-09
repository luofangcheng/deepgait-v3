# 数据采集 Tab（原"新建实验"）

文件：`deepgait3/gui/gait_experiment_tab.py`

> **本 tab 重命名时间**：2026-07-08 计划将"新建实验" → "数据采集"。本 tab 职责是"把 raw video 拿进项目"，**不再做累积图渲染**（累积图已迁到[数据分析 Tab](data-analysis-tab.md)）。

**核心组件**：
- `ExperimentEntryTable(4列：组别 / 动物编号 / 状态 / 备注)`
- `GaitExperimentTab` — 录制 + trim 调度
- 内部状态：`_camera`, `_video_writer`, `_trim_worker`, `_trim_jobs`

## 状态机总览

```mermaid
flowchart TD
    Start(["GaitExperimentTab __init__"]) --> Source{"数据源选择"}
    Source -->|"C1 相机"| Connect["_on_connect_camera\n从初始化 tab 取 bottom 相机"]
    Source -->|"视频文件夹"| LoadFolder["_on_load_folder\n弹文件夹对话框 + 遍历 mp4/avi/mov/mkv"]

    Connect --> StartRecord["_on_start → _start_camera_record"]
    LoadFolder --> Populate["entry_table.setRowCount 0\nfor v in videos: add_entry group=folder.name, animal_id=v.stem, note=str v\nstatus = trim 中 0%"]
    Populate --> ScheduleFolder["_schedule_folder_trim videos, group_name\njobs = row, src, dst, animal_id\n_launch_trim_worker jobs"]

    StartRecord --> Validate["validate row, animal_id, camera"]
    Validate --> InitW["_init_video_writer\npath = project_path / rawdata / videos / animal_id_timestamp.mp4\nwriter = cv2.VideoWriter mp4v"]
    InitW --> Live["_live_timer.start 33ms\n_fps_timer.start 1000ms"]
    Live --> Tick["_live_tick grab_one frame"]
    Tick --> Write["video_writer.write frame\n每 3 帧调 _show_preview"]
    Write --> StopCheck{"_on_stop?"}
    StopCheck -->|"否"| Tick
    StopCheck -->|"是"| Release["_video_writer.release"]
    Release --> ScheduleSingle["_schedule_single_trim row, src, dst, animal_id\n_launch_trim_worker jobs"]

    ScheduleFolder --> Worker
    ScheduleSingle --> Worker
    Worker(["TrimWorker.run 串行处理每个 job"])
```

## TrimWorker 串行调度细节

```mermaid
flowchart TD
    Start(["_launch_trim_worker jobs"]) --> Running{"_trim_worker 是否还在运行?"}
    Running -->|"是"| Skip["return 跳过本次"]
    Running -->|"否"| Create["worker = TrimWorker jobs\nsignal 连接: video_started → _on_trim_started\nvideo_progress → _on_trim_progress\nvideo_done → _on_trim_done\nvideo_failed → _on_trim_failed\nall_done → _on_all_trim_done\nworker.start"]
    Create --> Status["status_label = trim 中 N 个视频"]

    subgraph TrimWorker["TrimWorker.run 详细"]
        Loop["for idx, job in enumerate jobs"]
        Started["video_started.emit idx, total, src"]
        Trim["trim_video src, dst, detector, progress_cb"]
        Pct["video_progress.emit idx, total, pct, msg"]
        Result{"成功?"}
        Done["video_done.emit idx, src, dst"]
        Fail["video_failed.emit idx, src, error"]
        NextJob{"还有 job?"}
        AllDone["all_done.emit success, failed"]
    end
```

## UI 状态映射

| 阶段 | 状态列文字 |
|---|---|
| 刚录入 | `待录制` |
| 录制中 | `采集中...` |
| 录制完等待 trim | `trim 中...` |
| Trim 中带进度 | `trim 中 (1/3, 45%)` |
| Trim 完成 | `完成 /abs/path/.../trimmed/C57-001.mp4` |
| Trim 失败 | `trim 失败: <error>` |

## 关键点

- **路径修复**：从 `videos/` 改为 `project_path / "rawdata" / "videos"`，与 v3 项目布局一致。
- **串行 Trim**：单 GPU 一次只跑一个 trim，避免显存竞争。
- **C1 相机模式**：仅录制原始视频 + 实时预览原视频，**不做累积图**（GUI 性能 + 算法独立原则）。
- **文件夹模式**：自动用文件夹名当组别、视频文件名当动物编号。
- **TrimWorker 在 QThread 跑**，[详细流程见 background-workers.md](background-workers.md)。
- **提示**："注：累积脚印图渲染请到「数据分析」tab"
- **trim 算法**：[video-trim.md](video-trim.md)
