# 多相机硬件同步

文件：
- `deepgait3/hardware/camera/multi_cam.py` — `MultiCameraManager`, `FrameBus`, `SyncReport`, `MockCamera`
- `deepgait3/hardware/camera/trigger.py` — `TriggerController`, `PulseEvent`（RP2040 硬件触发，软件 fallback）
- `deepgait3/hardware/camera/base.py` — `ICamera`, `FrameInfo`, `CameraFactory`

> 当前默认 roster 为 4 路（`bottom / left / right / top`），3D 姿态走 C2/C3/C4。**5 路扩展**（C1 + 4 个 1024×1024 顶部相机）见"扩展 5 路"小节。

## 关键常量

| 名称 | 值 | 含义 |
|---|---|---|
| `SYNC_TOLERANCE_MS` | 1.0 | 帧间最大时间戳差，越界即"失同步" |
| `DEFAULT_ROSTER` | `("bottom", "left", "right", "top")` | 默认 4 路相机角色 |
| `trigger_line` | 0 (Hikvision) / 1 (Basler) | 硬件触发 GPIO 线号 |
| `queue_capacity` | 16 | `FrameBus` 每路最大缓存帧 |

## 启动 / 抓帧 / 同步检测

```mermaid
flowchart TD
    Start(["MultiCameraManager.enumerate_default n=4"]) --> Roles["roles = DEFAULT_ROSTER 0..n"]
    Roles --> Loop["for idx, role in enumerate roles"]
    Loop --> Factory["try cam = CameraFactory.create idx"]
    Factory -->|"失败"| Mock["fallback MockCamera serial=MOCK-role, fps=100"]
    Factory -->|"成功"| Real["真相机 Hikvision/Basler"]
    Mock --> Append
    Real --> Append["cams.append role, cam"]
    Append --> NextRole{"还有 role?"}
    NextRole -->|"是"| Loop
    NextRole -->|"否"| Build["mgr = MultiCameraManager cameras, trigger_line\n若未指定, Linux=1, 其它=0"]
    Build --> Ready(["manager 就绪, 尚未 open"])

    Start2(["mgr.start_all"]) --> OpenAll["open_all\n遍历调用 cam.open"]
    OpenAll --> ConfigTrig["configure_trigger\ncam.configure_hardware_trigger line, edge=rising"]
    ConfigTrig --> StartCont["for role, cam in roster\ncam.start_continuous lambda f, role: _on_frame role, f\n_on_frame → bus.push role, f"]
    StartCont --> SetRun["_running = True\n_sync_samples.clear"]

    Grab(["mgr.grab_quartet timeout_ms=1000"]) --> Bus["snap = bus.wait_quartet timeout_ms\n阻塞直到每路至少一帧"]
    Bus --> Record["_record_sync snap\ntimestamps = snap.frame.timestamp_ns\nmedian = statistics.median\ndeltas_ms = role: abs ts - median / 1e6 for role, frame in snap.items\n_sync_samples.append deltas_ms"]
    Record --> Return(["return snap Dict role, FrameInfo"])

    Report(["mgr.get_sync_report"]) --> Samples["samples = list _sync_samples"]
    Samples --> Worst["worst = max max s.values for s in samples\nworst_idx = argmax\nper_role = samples worst_idx"]
    Worst --> Build["SyncReport in_sync = worst 小于等于 sync_tolerance_ms\nmax_delta_ms = float worst\nper_role_delta_ms = per_role\nsample_size = len samples"]
    Build --> Ret(["return SyncReport"])
```

## 同步判定细节

```mermaid
flowchart LR
    In["snap 4 个 FrameInfo"] --> Med["median = statistics.median timestamps_ns"]
    Med --> Delta["for role, frame in snap.items\n  deltas role = abs frame.timestamp_ns - median / 1e6"]
    Delta --> Sample["_sync_samples.append deltas"]
    Sample --> Calc["max_per_sample = max s.values\nworst = max max_per_sample\nper_role_worst = samples argmax worst"]
    Calc --> InSync{"worst 小于等于 sync_tolerance_ms?"}
    InSync -->|"是"| Good["in_sync = True"]
    InSync -->|"否"| Bad["in_sync = False 告警"]
```

## 扩展 5 路

5 路扩展主要是**增加 roster 大小**和**LED flash 同步基准**：

```mermaid
flowchart LR
    R["5 路相机"] --> L["flash 检测\n找每路视频中第一次白光亮起帧\nt0 = max flash_frames"]
    L --> M["运动区间\nC1 走 footprint tracks\n4 路顶部相机走帧间差分"]
    M --> Inter["common_start = max starts\ncommon_end = min ends"]
    Inter --> Cut["for each video:\n  offset = t0 - flash\n  trim = common + offset"]
```

## 关键点

- **硬件触发是同步的根本**：`TriggerController` 驱动 RP2040 PIO 输出 100Hz 触发方波，所有相机在同一个上升沿开始曝光。
- **FrameBus 屏蔽线程细节**：消费者只需要 `wait_quartet` 拿完整 quartet，不需要自己管 slot。
- **MockCamera 提供可重复测试**：`clock_offset_ms` 注入可控制 jitter，便于单元测试 sync monitor。
- **Hikvision/Basler trigger 线号不同**：Windows Hikvision 走 Line 0，Linux Basler 走 Line 1，平台相关。
- **`SyncReport.in_sync` 判定**：worst delta ≤ `SYNC_TOLERANCE_MS` (1ms) 即算同步；超出会写日志告警。
- **扩展 5 路**：增加一个 `top_center` 或 `top_4` slot，配合 LED flash 检测做帧级对齐；5 路在 `enumerate_default(n=5)` 时生效。
- **零 Qt 依赖**，可独立运行测试。
