# Algorithm Experiments

算法升级实验目录。此处的代码独立于主工程，用于快速试错和原型验证。

## 目录结构

```
experiment/
├── README.md           # 本文件
├── data/               # 训练/测试数据
│   ├── raw/            # 原始 fTIR 图像
│   └── labels/         # 标注文件 (YOLO format)
├── train/              # 训练脚本和配置
├── eval/               # 评估脚本 (精度 + 速度 benchmark)
└── outputs/            # 训练产物 (模型权重、日志)
```

## 实验目标

1. YOLOv8-seg 替换 blob detection + clustering
2. ByteTrack 替换 IoU tracker
3. 在 RTX 3060 上达到实时（<10ms/frame）

## 集成条件

实验满足以下条件后，才能合并到 `deepgait3/core/`：
- [ ] 精度：mask IoU > 0.85 vs 当前方案
- [ ] 速度：< 8ms/frame（含 GPU 推理 + 后处理）
- [ ] 稳定性：100 帧连续测试无崩溃
- [ ] 输出兼容：生成 FootMask 格式与当前 pipeline 一致
