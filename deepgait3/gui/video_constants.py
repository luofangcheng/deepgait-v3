"""Video size constants shared by the data acquisition and analysis tabs.

Centralising the dimensions here makes them easy to grep when the
hardware mounting changes (e.g. when the Stage 2 calibration step
confirms the actual resolution of the top cameras).
"""
from __future__ import annotations


# C1 底部 FTIR 走道 — wide 1920×384 横向带状
BOTTOM_CAMERA_SIZE: tuple[int, int] = (1920, 384)

# 顶部 4 相机 (left / right / top) — 3D 姿态采集
# TODO(2026-07-08): 顶部 4 相机的实际硬件尺寸可能与 (1024, 1024) 不同,
# 后续 Stage 2 标定后需要根据实际安装调整。修改此处即可，所有调用方
# (data_acquisition_tab, data_analysis_tab) 自动生效。
TOP_CAMERA_SIZE: tuple[int, int] = (1024, 1024)

# 同步健康度阈值 (ms) — 与 multi_cam.SYNC_TOLERANCE_MS 保持一致,
# GUI 层用此值决定同步 label 的颜色 (绿/红)。
SYNC_TOLERANCE_MS: float = 1.0

# 缩略图显示尺寸 (用于 2×2 网格, 非录制尺寸)
THUMBNAIL_DISPLAY_SIZE: tuple[int, int] = (320, 320)

# 缩略图跳帧: 每 3 帧更新一次, 降低 CPU/GUI 负载
THUMBNAIL_SKIP_FRAMES: int = 3


__all__ = [
    "BOTTOM_CAMERA_SIZE",
    "TOP_CAMERA_SIZE",
    "SYNC_TOLERANCE_MS",
    "THUMBNAIL_DISPLAY_SIZE",
    "THUMBNAIL_SKIP_FRAMES",
]
