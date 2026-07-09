# CLAUDE.md — DeepGait v3

## 核心原则

### 算法优先原则 (Algorithm First)

**新算法为最高优先级。GUI 必须适配算法，禁止算法妥协兼容 GUI。**

- 算法模块独立开发、独立测试，零 Qt 依赖，可 CLI 直接调用
- GUI 仅作为调用方引用算法，不在 GUI 代码中内嵌算法逻辑
- 如需修改 GUI 以适应新算法，直接推倒重构 GUI，不做兼容层
- 禁止为兼容旧 GUI 而保留旧数据格式的桥接代码（如 legacy FootprintSequence）

### 实验先行原则

新算法在 `experiment/` 目录独立开发调试，通过验证后集成到 `deepgait3/core/`。

集成条件：
- 精度达标（如 mask IoU > 0.85）
- 速度达标（如 < 8ms/frame）
- 稳定性验证（100帧连续无崩溃）
- 输出格式与现有数据契约兼容

## 项目架构

```
deepgait-v3/
├── deepgait3/                # Python 包
│   ├── gui/                  # 应用层 — PySide6 GUI，零算法逻辑
│   ├── core/                 # 算法层 — 纯 Python，零 Qt 依赖
│   │   ├── pawprint/         # Stage 1: fTIR 足迹提取 (YOLOv8-seg)
│   │   ├── data/             # 数据层: schema, exporter, project, pipeline
│   │   ├── _legacy/          # 未迁移的 v2.0 算法 (过渡期，逐步删除)
│   │   ├── calibration/      # Stage 2: 相机标定 (planned)
│   │   ├── triangulation/    # Stage 2: 3D 三角化 (planned)
│   │   ├── fusion/           # Stage 2: 脚踝-脚印匹配 (planned)
│   │   ├── metrics/          # Stage 3: 步态参数 (planned)
│   │   └── report/           # Stage 4: 报告导出 (planned)
│   ├── hardware/             # 硬件抽象层 — 相机驱动、DLC 子进程
│   ├── io/                   # 数据 I/O — HDF5, NWB, BIDS
│   ├── license/              # 许可证模块 (Cython .so)
│   ├── security/             # 安全模块 (Cython .so)
│   └── utils/                # 共享工具
├── projects/                 # 项目数据 (不入 Python 包)
│   └── <name>/
│       ├── project.json      # 项目元数据
│       ├── .active            # 激活标记
│       ├── rawdata/videos/   # 原始视频
│       └── data/<trial>/     # 处理输出
├── experiment/               # 算法实验 (独立于主工程)
├── tests/                    # 测试
├── docs/                     # 文档
└── examples/                 # 示例脚本
```

## 当前算法: YOLOv8-seg + IoU Tracker

### 技术栈

- **检测**: YOLOv8n-seg (3.26M 参数), RTX 3060 GPU 批量推理
- **追踪**: IoUFootprintTracker (CPU, <1ms)
- **模型**: `experiment/outputs/pawprint_yolo/weights/best.pt`
- **训练数据**: 106 张 fTIR 标注帧, Box mAP50=0.87

### 数据流

```
Video → [YOLO batch GPU] → List[FootMask] → IoU Tracker → build_cycles
  → ExtractedCycle (21字段) → CSV/JSON 导出至 projects/<active>/data/
```

### 已删除的旧算法

- `detection.py` (blob detection)
- `grouping.py` (spatial clustering)
- `scoring.py`, `scoring_detection.py` (color scoring)
- `extractor.py` (PawPrintExtractor)
- `biscale.py`, `track_merger.py`

## 关键 API

### YOLO 检测

```python
from deepgait3.core.pawprint import YoloPawDetector
det = YoloPawDetector()  # 自动加载 best.pt, 自动选 GPU
footmasks = det.detect_single(frame_bgr, bg_G)  # 单帧
footmasks_list = det.detect_batch(frames, bg_G)  # 批量 16 帧
```

### 数据导出

```python
from deepgait3.core.data import ProjectManager, extract_trial

pm = ProjectManager()
pm.activate("V3-test1")

trial = extract_trial(video_path, output_dir, mouse_id="C57_001")
pm.save_trial(trial)
```

### CLI

```bash
python -m deepgait3 extract                     # 处理激活项目中的视频
python -m deepgait3 extract --video /path/to/video.mp4  # 指定视频
python -m deepgait3 project list                # 列出项目
python -m deepgait3 project activate NAME       # 激活项目
python -m deepgait3 gui                         # 启动 GUI
```

## 开发命令

```bash
# 安装
pip install -e ".[dev]"

# 测试
pytest tests/unit/ -v

# 实验训练
python experiment/train_yolo.py

# 实验演示
python experiment/demo_video.py
python experiment/demo_video_gpu.py
```

## 数据契约

### FootMask (pawprint) — 检测输出

位于 `deepgait3/core/pawprint/models.py`。关键字段: `mask_padded`, `raw_intensity_crop`, `centroid_px`, `bbox_xyxy`, `bbox_xyxy_padded`, `total_area_px`, `mean_intensity`, `peak_intensity`, `pressure_map`.

### ExtractedCycle (data) — 步态分析输入

位于 `deepgait3/core/data/schema.py`。21 个字段覆盖 `gait_metrics_module` 全部输入需求: temporal (touchdown/liftoff/duration/3-phase), spatial (centroid/length/width/area), pressure (mean/peak/stand/ratio), CoP (path/displacement).

## 技术约束

- Python 3.10–3.13
- PySide6 (LGPL)，禁止引入 PyQt6
- DeepLabCut: 独立 conda 环境，subprocess 隔离
- Cython .so: `_legacy_shim.py` 提供 `deepgait.*` 兼容
- GPU: NVIDIA RTX 3060 (开发), 生产需支持纯 CPU 回退
