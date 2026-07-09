# GPU 累积图 — Track-level Union Mask

文件：`deepgait3/core/pawprint/cumulative.py`

**核心 API**：
- `build_cumulative_union(tracks, (H, W), *, closing_kernel=None, use_gpu=True)` — 返回 `(H, W) float32` 累积图
- `build_cumulative_union_cpu(tracks, (H, W), *, closing_kernel=None)` — CPU 回退
- `render_overlay(cum_intensity)` — 返回黑底绿光 `uint8` BGR 图
- `_track_to_global_tensors(track, H, W)` — 把单个 track 的所有帧 mask 投影到全帧并上采样

```mermaid
flowchart TD
    Start(["build_cumulative_union tracks, (H, W), use_gpu=True"]) --> Device["device = cuda if use_gpu and cuda else cpu"]
    Device --> InitCum["cum = torch.zeros (H, W, float32, device)"]
    InitCum --> LoopTrack["for each FootprintTrack"]

    subgraph PerTrackLoop["单 track 处理"]
        LoopTrack --> Tensors["_track_to_global_tensors track, H, W\n生成 (N, H, W) uint8 masks + float32 intensities\n上采样到 full-frame 尺寸"]
        Tensors --> NoneChk{"masks is None?"}
        NoneChk -->|"是"| Skip["continue 下一 track"]
        NoneChk -->|"否"| Upload["masks_t, intensities_t = to device"]
        Upload --> ToFloat["if masks_t.dtype 不是 float32\nmasks_t = masks_t.float 避免 interpolate 报错"]
        ToFloat --> Union["union_mask = masks_t.sum dim=0 大于 0\nOR 并集所有帧 mask"]
        Union --> CloseOpt{"closing_kernel 大于 1?"}
        CloseOpt -->|"是"| Morph["mask 移到 cpu\ncv2.morphologyEx MORPH_CLOSE 椭圆 kernel\n再上传回 device"]
        CloseOpt -->|"否"| MaxInt
        Morph --> MaxInt["max_intensity = intensities_t.max dim=0.values\nunion 内每个像素取所有帧最大强度"]
        MaxInt --> Masked["masked = torch.where union_mask, max_intensity, 0"]
        Masked --> Merge["cum = torch.maximum cum, masked"]
    end

    Merge --> NextTrack{"还有 track?"}
    NextTrack -->|"是"| LoopTrack
    NextTrack -->|"否"| ToCPU["return cum.cpu.numpy float32"]
    ToCPU --> End(["返回 (H, W) float32 cum_intensity"])

    style PerTrackLoop fill:#f9f,stroke:#333,stroke-width:2px
```

### 渲染（`render_overlay`）

```mermaid
flowchart LR
    In["cum_intensity float32 H, W"] --> Norm["norm = cum 减 0 除 255 clip 0, 1\n绿通道 = norm * 255 转 uint8\n红蓝通道 = 0"]
    Norm --> Out["BGR uint8 H, W 3\n黑底绿光叠加图"]
```

## 关键点

- **空间维度**：每个 track 的 union mask = 所有帧 mask 的 OR，**先聚合再取强度**，比像素级 max-merge 干净。
- **强度维度**：union mask 内每个像素取该 track 全部帧中的最大强度。
- **可选闭运算**：`closing_kernel > 1` 时用椭圆 kernel 做 MORPH_CLOSE，连上"脚掌-脚趾"之间的间隙。
- **GPU 加速**：所有 per-track tensor 操作都在 GPU 上；mask→numpy 上传和 (可选) 闭运算是 CPU。
- **dtype 修复**：`if masks_t.dtype != float32: masks_t = masks_t.float()` — 防止 `torch.nn.functional.interpolate` 报 "NotImplementedError for Byte"。
- **CPU 回退**：`use_gpu=False` 时自动用 numpy 路径（`build_cumulative_union_cpu`）。
- **零 Qt 依赖**，纯算法层，可 CLI 直接调用。
- **三个调用点已统一**：`core/pawprint/pipeline.py:_save_cumulative`、`experiment/demo_video_gpu.py`、`gui/gait_analysis_tab.py`（预览合并）。
