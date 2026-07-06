"""跨厂商相机抽象接口。"""
# deepgait/hardware/camera/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Dict, Any
import numpy as np


@dataclass
class FrameInfo:
    """单帧图像及其元数据。"""
    image: np.ndarray          # (H, W, 3) BGR uint8
    frame_number: int          # 相机内部帧号
    timestamp_ns: int          # 硬件时间戳（纳秒）
    camera_serial: str         # 相机序列号
    exposure_us: float         # 实际曝光时间（μs）
    gain_db: float             # 实际增益（dB）


class ICamera(ABC):
    """所有相机的抽象基类。"""

    @abstractmethod
    def open(self) -> None:
        """打开相机，配置默认参数。"""

    @abstractmethod
    def close(self) -> None:
        """关闭相机，释放资源。"""

    @abstractmethod
    def grab_one(self, timeout_ms: int = 5000) -> FrameInfo:
        """采集单帧。"""

    @abstractmethod
    def start_continuous(self, callback) -> None:
        """启动连续采集。callback: Callable[[FrameInfo], None]"""

    @abstractmethod
    def stop_continuous(self) -> None:
        """停止连续采集。"""

    @abstractmethod
    def configure_hardware_trigger(self, line: int = 0, edge: str = 'rising') -> None:
        """配置硬件触发。

        Args:
            line: Line 0/1/2/3（海康）或 Line1（Basler）
            edge: 'rising' 或 'falling'
        """

    @abstractmethod
    def set_exposure_us(self, exposure_us: float) -> None:
        """设置曝光时间（μs）。"""

    @abstractmethod
    def set_gain_db(self, gain_db: float) -> None:
        """设置增益（dB）。"""

    @abstractmethod
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """设置 ROI。"""

    @abstractmethod
    def get_serial(self) -> str:
        """获取相机序列号。"""

    @abstractmethod
    def get_model(self) -> str:
        """获取相机型号。"""

    # ------------------------------------------------------------------
    # W17 (refactor 初始化 tab): full parameter set
    # ------------------------------------------------------------------
    @abstractmethod
    def set_brightness(self, value: int) -> None:
        """设置亮度（设备相对值，范围见 :meth:`get_supported_features`）。"""

    @abstractmethod
    def set_contrast(self, value: int) -> None:
        """设置对比度（设备相对值，范围见 :meth:`get_supported_features`）。"""

    @abstractmethod
    def set_pixel_format(self, fmt: str) -> None:
        """设置像素格式（如 ``"BGR8"`` / ``"Mono8"`` / ``"BayerRG8"``）。

        合法格式列表见 :meth:`get_supported_features`。
        """

    @abstractmethod
    def set_fps(self, fps: int) -> None:
        """设置目标采集帧率（fps）。

        实现应尽量让实际帧率逼近该值；如硬件不支持应抛
        :class:`ValueError`。
        """

    @abstractmethod
    def get_supported_features(self) -> Dict[str, Any]:
        """返回当前设备支持的参数范围 / 离散选项。

        返回 dict，至少包含以下 key：

        * ``"brightness"``: ``(min, max, default, step)``
        * ``"contrast"``: ``(min, max, default, step)``
        * ``"exposure_us"``: ``(min, max, default)``
        * ``"gain_db"``: ``(min, max, default, step)``
        * ``"fps"``: ``(min, max, default)``
        * ``"pixel_format"``: list of supported format strings
        * ``"roi"``: ``{"min_w": int, "min_h": int, "max_w": int, "max_h": int}``
        """

    @abstractmethod
    def snapshot_config(self) -> Dict[str, Any]:
        """返回当前所有可调参数的快照（用于「保存预设」）。

        返回 dict，可直接 :func:`json.dump`。"""

    @abstractmethod
    def restore_config(self, cfg: Dict[str, Any]) -> None:
        """从 snapshot 恢复所有可调参数。"""

    # ------------------------------------------------------------------
    # 默认 feature 表（供子类的 get_supported_features 复用）
    # ------------------------------------------------------------------
    @staticmethod
    def _default_features() -> Dict[str, Any]:
        """通用默认 feature 表（保守值）。

        真实相机会在自己的 ``get_supported_features()`` 中覆盖。
        """
        return {
            "brightness": (-100, 100, 0, 1),
            "contrast": (-100, 100, 0, 1),
            "exposure_us": (50.0, 30_000.0, 5_000.0),
            "gain_db": (0.0, 20.0, 0.0, 0.1),
            "fps": (1, 500, 100),
            "pixel_format": ["BGR8", "Mono8", "BayerRG8", "BayerGB8"],
            "roi": {"min_w": 64, "min_h": 64, "max_w": 4096, "max_h": 4096},
        }


class CameraFactory:
    """跨厂商相机工厂（按平台自动选择）。"""

    @staticmethod
    def create(camera_id: int = 0, serial: str = None) -> ICamera:
        import platform
        system = platform.system()

        if system == 'Windows':
            # Windows 首选海康（高性价比 + 国内供应链）
            from .hikvision import HikvisionCamera
            return HikvisionCamera(camera_id=camera_id, serial=serial)
        elif system == 'Linux':
            # Linux 用 Basler（海康 Linux 驱动不成熟）
            from .basler import BaslerCamera
            return BaslerCamera(camera_id=camera_id)
        else:
            raise NotImplementedError(
                f"Platform '{system}' not supported. "
                "deepgait v2 supports Windows (Hikvision) + Linux (Basler)."
            )

