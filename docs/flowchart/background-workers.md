# QThread 后台工人

文件：`deepgait3/gui/workers.py`

> 所有后台线程统一用 `QThread` + `Signal` 模式。worker 与 GUI 通过信号槽解耦，UI 永远只发信号、不阻塞等结果。

## Worker 列表

| 类 | 用途 | 主信号 |
|---|---|---|
| `GaitWorker` | legacy 步态分析 | `result_ready(object)` / `progress(str)` / `error(str)` |
| `FTIRWorker` | 单帧 pawprint 预览 | `result_ready(list)` / `progress(str)` / `error(str)` |
| `DLCWorker` | DeepLabCut 子流程 | `step_done(str)` / `progress(str)` / `error(str)` |
| `CameraWorker` | 单相机/视频源抓帧 | `frame_ready(ndarray)` / `fps_updated(float)` / `error(str)` |
| `TrimWorker` | 多视频 trim 调度 | `video_started` / `video_progress` / `video_done` / `video_failed` / `all_done` |
| `BatchAnalysisWorker` | 批分析 | `video_started` / `video_progress` / `video_completed` / `all_completed` / `error_occurred` |

## TrimWorker 流程（数据采集 tab 主用）

```mermaid
flowchart TD
    Start(["TrimWorker.run jobs"]) --> InitDet["懒加载 YoloPawDetector\nproperty: detector 第一次访问时建"]
    InitDet --> Loop["for idx, job in enumerate jobs"]
    Loop --> Started["video_started.emit idx, total, str src"]
    Started --> Trim["trim_video src, dst, detector\nprogress_cb = lambda phase, pct, msg\n  video_progress.emit idx, total, pct, msg"]
    Trim --> Result{"trim_video 成功?"}
    Result -->|"是"| Done["video_done.emit idx, str src, str dst\nsuccess += 1"]
    Result -->|"否, TrimError"| FailE["video_failed.emit idx, str src, str e\nfailed += 1"]
    Result -->|"否, 其它异常"| FailU["video_failed.emit idx, str src, f 意外: e\nfailed += 1"]
    Done --> Next
    FailE --> Next
    FailU --> Next
    Next{"还有 job?"}
    Next -->|"是"| Loop
    Next -->|"否"| All["all_done.emit success, failed"]

    style Loop fill:#f9f,stroke:#333,stroke-width:2px
```

## BatchAnalysisWorker 流程（数据分析 tab 主用）

```mermaid
flowchart TD
    Start(["BatchAnalysisWorker.run entries"]) --> Loop["for idx, entry in enumerate entries"]
    Loop --> Abort{"_abort 已被设置?"}
    Abort -->|"是"| Break["break 出循环"]
    Abort -->|"否"| Get["animal_id = entry.get animal_id\nvideo_path = entry.get video_path"]
    Get --> Empty{"video_path 为空?"}
    Empty -->|"是"| Skip["continue 下一个"]
    Empty -->|"否"| Started["video_started.emit idx, animal_id"]
    Started --> Process["_process_one idx, animal_id, video_path\n返回 result dict"]
    Process --> Result{"_process_one 成功?"}
    Result -->|"是"| Done["video_completed.emit idx, str out_dir, metrics"]
    Result -->|"否, 异常"| Err["error_occurred.emit f animal_id: e"]
    Done --> Next
    Err --> Next
    Skip --> Next
    Next{"还有 entry?"}
    Next -->|"是"| Loop
    Next -->|"否"| All["all_completed.emit results"]

    style Process fill:#bbf,stroke:#333
```

## CameraWorker 流程（数据采集 tab 备选）

```mermaid
flowchart TD
    Start(["CameraWorker.run"]) --> Open["cap = cv2.VideoCapture source\ncap.set W, H"]
    Open --> InitW["writer = cv2.VideoWriter mp4v if save_path"]
    InitW --> Loop["while not _stop"]
    Loop --> Read["ret, frame = cap.read"]
    Read --> Check{"ret?"}
    Check -->|"否, 文件到末尾"| Break["break out of loop"]
    Check -->|"否, 相机丢帧"| Loop
    Check -->|"是"| Write["writer.write frame if writer"]
    Write --> Emit["frame_ready.emit frame\n每 30 帧 fps_updated.emit fps\nframe_count += 1"]
    Emit --> Loop
    Break --> Fin["writer.release, cap.release"]
    Fin --> End(["end"])
```

## 关键点

- **统一 `QThread + Signal` 模式**：与 Qt 主线程天然兼容，自动 join 不会泄露。
- **`try/except` 包裹整个 `run`**：单个 job 失败不会拖死整个 worker。
- **懒加载 detector**：`TrimWorker.detector` 是 property，第一次访问时建一次，后续复用。
- **跨平台摄像头**：[multi-camera-sync.md](multi-camera-sync.md) 是更复杂的多相机硬件同步方案。
- **GUI 端消费**：[data-acquisition-tab.md](data-acquisition-tab.md) 连接 `TrimWorker` 的 5 个信号；[data-analysis-tab.md](data-analysis-tab.md) 连接 `BatchAnalysisWorker` 的 5 个信号。
