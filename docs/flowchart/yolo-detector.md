# YOLOv8n-seg 爪印检测

文件：`deepgait3/core/pawprint/yolo_detector.py`

**核心 API**：
- `YoloPawDetector(model_path, conf=0.25)` — 加载权重，conf 阈值
- `detect_single(frame, bg_G, *, min_area_px=5)` — 单帧，返回 `List[FootMask]`
- `detect_batch(frames, bg_G, *, min_area_px=5)` — 批量 16 帧，返回 `List[List[FootMask]]`
- `_results_to_footmasks(frame, results, bg_G, h, w, min_area_px)` — YOLO 输出 → FootMask 转换

```mermaid
flowchart TD
    Start(["detect_single frame, bg_G"]) --> Wrap["detect_batch frames = frame 单帧"]
    Wrap --> BatchLoop["for batch_start in 0..N step BATCH_SIZE=16"]
    BatchLoop --> Yolo["model batch, verbose=False, stream=False, conf"]
    Yolo --> PerFrame["for frame, results in zip batch, results_batch"]
    PerFrame --> Convert["_results_to_footmasks frame, results, bg_G, h, w, min_area_px"]
    Convert --> Append["all_footmasks.append footmasks"]
    Append --> NextBatch{"还有 batch?"}
    NextBatch -->|"是"| BatchLoop
    NextBatch -->|"否"| Return(["返回 footmasks 0 单帧结果"])

    subgraph ConvertFlow["_results_to_footmasks 详细流程"]
        CStart["results.masks is None?"] -->|"是"| CEmpty["return 空列表"]
        CStart -->|"否"| Resize["masks_np = results.masks.data.cpu.numpy\ncv2.resize 到 h, w"]
        Resize --> Delta["G = frame 绿色通道\ndelta = G - bg_G"]
        Delta --> ForMask["for i in range N"]
        ForMask --> Binarize["mask_bin = mask > 0.5"]
        Binarize --> AreaFilter{"area_px 大于等于 min_area_px?"}
        AreaFilter -->|"否"| SkipMask["continue"]
        AreaFilter -->|"是"| Moments["cv2.moments 求质心 cx, cy"]
        Moments --> Bbox["bbox = minmax + pad 2\nmask_crop, delta_crop"]
        Bbox --> Stats["in_mask = delta_crop mask_crop\nmean_intensity, peak_intensity\npressure_map = 18 * max delta - 8, 0 pow 0.75"]
        Stats --> BuildFM["FootMask centroid_px, bbox_xyxy, bbox_xyxy_padded\nmask_padded, raw_intensity_crop, pressure_map\ntotal_area_px, mean_intensity, peak_intensity\ntouches_edge"]
        BuildFM --> NextMask{"还有 mask?"}
        NextMask -->|"是"| ForMask
        NextMask -->|"否"| End["return footmasks"]
    end

    style ConvertFlow fill:#f9f,stroke:#333,stroke-width:2px
```

## 关键点

- `detect_single` 只是 `detect_batch` 的薄包装，重活都在 `_results_to_footmasks`。
- **mask resize 走 cv2**（CPU 路径，保留与 `experiment/demo_video_gpu.py` 一致），不依赖 torch 插值。
- **背景差 `delta = G - bg_G`** 决定 `mean_intensity` 和 `pressure_map`，与 `experiment` 保持一致。
- **面积阈值** `min_area_px` 决定是否丢弃过小 mask；过小通常对应噪声。
- **bbox padding 2 像素**，给 mask 边缘留余量，避免后续 tracker 的 IoU 计算偏差。
- **BATCH_SIZE=16** 是经验值，受 RTX 3060 显存约束；如换更小的卡需下调。
- **零 Qt 依赖**，可直接 CLI 调用。
