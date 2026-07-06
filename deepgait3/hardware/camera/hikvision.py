"""海康威视 MV-CL 相机 Python 实现。
完整文档见 kb/21_camera_sdk.md。
"""
# deepgait/hardware/camera/hikvision.py
import ctypes
import threading
import time
import numpy as np
from typing import Callable, Optional, Dict, Any

from .base import ICamera, FrameInfo
from .hikvision_sdk import (
    _MVS_SDK, _MV_CC_DEVICE_INFO,
    MVS_OK, MVS_ERROR_CODES,
    TRIGGER_SOURCE_LINE0, TRIGGER_SOURCE_LINE1,
    TRIGGER_SOURCE_LINE2, TRIGGER_SOURCE_LINE3,
    PIXEL_FORMAT_BGR8, mvs_check,
)


class HikvisionCamera(ICamera):
    """海康威视 MV-CL 工业相机。

    通过 ctypes 直接调用 MvCameraControl.dll / libMvCameraControl.so。
    """

    def __init__(self, camera_id: int = 0, serial: str = None):
        """
        Args:
            camera_id: 设备索引 (0/1/2/3)
            serial: 相机序列号（如 'FTIR001'）。如果指定则精确匹配。
        """
        self.camera_id = camera_id
        self.serial_filter = serial
        self._handle: Optional[ctypes.c_void_p] = None
        self._device_info: Optional[_MV_CC_DEVICE_INFO] = None
        self._is_open = False
        self._width = 0
        self._height = 0
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_callback: Optional[Callable] = None

    def _find_device(self) -> None:
        """枚举海康设备并选择。"""
        n_devices = ctypes.c_uint()
        # TLayerType: 0x00000002 = USB3 (我们用 USB3)
        ret = _MVS_SDK.MV_CC_EnumDevices(0x00000002, None, ctypes.byref(n_devices))
        mvs_check(ret, "EnumDevices (count)")

        if n_devices.value == 0:
            raise RuntimeError(
                "No Hikvision USB3 camera found. "
                "Check USB3 cable and MVS driver installation."
            )

        devices = (_MV_CC_DEVICE_INFO * n_devices.value)()
        ret = _MVS_SDK.MV_CC_EnumDevices(0x00000002, devices, ctypes.byref(n_devices))
        mvs_check(ret, "EnumDevices (list)")

        if self.serial_filter:
            for i in range(n_devices.value):
                if devices[i].Serial.decode(errors='ignore') == self.serial_filter:
                    self._device_info = devices[i]
                    return
            raise RuntimeError(f"Hikvision camera serial '{self.serial_filter}' not found")
        else:
            idx = min(self.camera_id, n_devices.value - 1)
            self._device_info = devices[idx]

    def open(self) -> None:
        """打开 + 配置 + 启动预览。"""
        if self._is_open:
            return
        try:
            self._find_device()
            serial = self._device_info.Serial.decode(errors='ignore')

            # 1. 创建句柄
            self._handle = ctypes.c_void_p()
            ret = _MVS_SDK.MV_CC_CreateHandle(None, serial.encode(), self._handle)
            mvs_check(ret, "CreateHandle")

            # 2. 打开设备
            ret = _MVS_SDK.MV_CC_OpenDevice(self._handle, 0, 0, 0)
            mvs_check(ret, "OpenDevice")

            # 3. 读取图像尺寸
            width = ctypes.c_uint()
            height = ctypes.c_uint()
            _MVS_SDK.MV_CC_GetIntValue(self._handle, b'Width', width)
            _MVS_SDK.MV_CC_GetIntValue(self._handle, b'Height', height)
            self._width = width.value
            self._height = height.value

            # 4. 默认配置
            self._apply_default_config()
            self._is_open = True
        except Exception as e:
            self._cleanup()
            raise

    def _apply_default_config(self) -> None:
        """应用默认配置。

        目标硬件：海康威视 MV-CL042-10GM
        规格：4MP IMX264/265 CMOS GigE Vision
        默认帧率：100 fps @ 4MP 全分辨率（物理可达 200 fps，留有 2× 余量）
        """
        # 像素格式：BGR8（OpenCV 兼容）
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'PixelFormat', PIXEL_FORMAT_BGR8)

        # 关闭自动模式
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'ExposureAuto', 0)  # Off
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'GainAuto', 0)
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'BalanceWhiteAuto', 0)

        # 曝光：5 ms（适配 100 fps @ 4MP 全分辨率触发，物理可达 200 fps）
        _MVS_SDK.MV_CC_SetFloatValue(self._handle, b'ExposureTime', 5000.0)  # 5ms
        _MVS_SDK.MV_CC_SetFloatValue(self._handle, b'Gain', 0.0)

    def configure_hardware_trigger(self, line: int = 0, edge: str = 'rising') -> None:
        """配置硬件触发（Line 0/1/2/3, GPIN 输入）。

        Args:
            line: 0=Line0, 1=Line1, 2=Line2, 3=Line3
            edge: 'rising' (上升沿) 或 'falling' (下降沿)
        """
        if not self._is_open:
            raise RuntimeError("Camera not opened")

        # 1. 触发模式：On
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'TriggerMode', 1)

        # 2. 触发源
        sources = {0: TRIGGER_SOURCE_LINE0, 1: TRIGGER_SOURCE_LINE1,
                   2: TRIGGER_SOURCE_LINE2, 3: TRIGGER_SOURCE_LINE3}
        source = sources.get(line, TRIGGER_SOURCE_LINE0)
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'TriggerSource', source)

        # 3. 触发沿
        activation = 0 if edge == 'rising' else 1  # RisingEdge / FallingEdge
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'TriggerActivation', activation)

        # 4. 触发延迟
        _MVS_SDK.MV_CC_SetFloatValue(self._handle, b'TriggerDelay', 0.0)

        # 5. 关闭帧率控制（由触发决定）
        _MVS_SDK.MV_CC_SetEnumValue(self._handle, b'AcquisitionFrameRateEnable', 0)

    def set_exposure_us(self, exposure_us: float) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        _MVS_SDK.MV_CC_SetFloatValue(self._handle, b'ExposureTime', exposure_us)

    def set_gain_db(self, gain_db: float) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        _MVS_SDK.MV_CC_SetFloatValue(self._handle, b'Gain', gain_db)

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """设置 ROI。"""
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        _MVS_SDK.MV_CC_SetIntValue(self._handle, b'OffsetX', x)
        _MVS_SDK.MV_CC_SetIntValue(self._handle, b'OffsetY', y)
        _MVS_SDK.MV_CC_SetIntValue(self._handle, b'Width', width)
        _MVS_SDK.MV_CC_SetIntValue(self._handle, b'Height', height)
        self._width = width
        self._height = height

    # ------------------------------------------------------------------
    # W17: extended parameter set (brightness / contrast / fps / format)
    # ------------------------------------------------------------------
    def set_brightness(self, value: int) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        # 海康 SDK 节点名: "Brightness"。范围通常 -100 ~ 100。
        if not -100 <= value <= 100:
            raise ValueError(f"brightness {value} out of range [-100, 100]")
        _MVS_SDK.MV_CC_SetIntValue(self._handle, b'Brightness', value)

    def set_contrast(self, value: int) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        if not -100 <= value <= 100:
            raise ValueError(f"contrast {value} out of range [-100, 100]")
        _MVS_SDK.MV_CC_SetIntValue(self._handle, b'Contrast', value)

    def set_pixel_format(self, fmt: str) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        # 海康枚举名: "PixelFormat", 接受 'BayerRG8', 'Mono8', 'RGB8Packed' 等
        if not fmt:
            raise ValueError("pixel_format must be a non-empty string")
        _MVS_SDK.MV_CC_SetEnumValue(
            self._handle, b'PixelFormat', fmt.encode('ascii')
        )

    def set_fps(self, fps: int) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        if not 1 <= fps <= 500:
            raise ValueError(f"fps {fps} out of range [1, 500]")
        # 海康: 启用 AcquisitionFrameRate 控制（TriggerMode=Off 时生效）
        _MVS_SDK.MV_CC_SetEnumValue(
            self._handle, b'AcquisitionFrameRateEnable', 1
        )
        _MVS_SDK.MV_CC_SetFloatValue(
            self._handle, b'AcquisitionFrameRate', float(fps)
        )

    def get_supported_features(self) -> Dict[str, Any]:
        # 海康具体范围取自官方 MVS SDK 文档 (MV-CL042-10GM 默认值)
        return {
            "brightness": (-100, 100, 0, 1),
            "contrast": (-100, 100, 0, 1),
            "exposure_us": (50.0, 30_000.0, 5_000.0),
            "gain_db": (0.0, 20.0, 0.0, 0.1),
            "fps": (1, 200, 100),  # 海康相机典型上限 200 fps
            "pixel_format": ["Mono8", "BayerRG8", "BayerGB8", "BGR8"],
            "roi": {"min_w": 64, "min_h": 64, "max_w": 2048, "max_h": 2048},
        }

    def snapshot_config(self) -> Dict[str, Any]:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        # 从 SDK 读取当前实际值；若失败回退到内部 state
        def _read_int(name: bytes, default: int) -> int:
            v = ctypes.c_int()
            ret = _MVS_SDK.MV_CC_GetIntValue(self._handle, name, v)
            if ret == 0:
                return int(v.value)
            return default

        def _read_float(name: bytes, default: float) -> float:
            v = ctypes.c_float()
            ret = _MVS_SDK.MV_CC_GetFloatValue(self._handle, name, v)
            if ret == 0:
                return float(v.value)
            return default

        return {
            "width": self._width,
            "height": self._height,
            "fps": int(_read_float(b'AcquisitionFrameRate', 100.0)),
            "exposure_us": _read_float(b'ExposureTime', 5000.0),
            "gain_db": _read_float(b'Gain', 0.0),
            "brightness": _read_int(b'Brightness', 0),
            "contrast": _read_int(b'Contrast', 0),
            "pixel_format": "BGR8",  # SDK 枚举读取较复杂，简化处理
        }

    def restore_config(self, cfg: Dict[str, Any]) -> None:
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        # 按顺序恢复，依赖顺序：pixel_format → roi → brightness/contrast
        # → exposure → gain → fps
        if "pixel_format" in cfg:
            try:
                self.set_pixel_format(cfg["pixel_format"])
            except Exception:
                pass
        if all(k in cfg for k in ("x", "y", "width", "height")):
            self.set_roi(int(cfg["x"]), int(cfg["y"]),
                          int(cfg["width"]), int(cfg["height"]))
        if "brightness" in cfg:
            self.set_brightness(int(cfg["brightness"]))
        if "contrast" in cfg:
            self.set_contrast(int(cfg["contrast"]))
        if "exposure_us" in cfg:
            self.set_exposure_us(float(cfg["exposure_us"]))
        if "gain_db" in cfg:
            self.set_gain_db(float(cfg["gain_db"]))
        if "fps" in cfg:
            try:
                self.set_fps(int(cfg["fps"]))
            except ValueError:
                pass

    def grab_one(self, timeout_ms: int = 5000) -> FrameInfo:
        """采集单帧。"""
        if not self._is_open:
            raise RuntimeError("Camera not opened")

        # 分配 BGR buffer
        buf_size = self._width * self._height * 3
        buf = (ctypes.c_ubyte * buf_size)()
        buf_len = ctypes.c_uint(buf_size)

        # 采集
        ret = _MVS_SDK.MV_CC_GetImageForBGR(
            self._handle, ctypes.byref(buf), buf_size, ctypes.byref(buf_len)
        )
        mvs_check(ret, "GetImageForBGR")

        # 转 numpy
        img = np.frombuffer(buf, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        ).copy()

        # 元数据
        exposure = ctypes.c_float()
        gain = ctypes.c_float()
        _MVS_SDK.MV_CC_GetFloatValue(self._handle, b'ExposureTime', exposure)
        _MVS_SDK.MV_CC_GetFloatValue(self._handle, b'Gain', gain)

        return FrameInfo(
            image=img,
            frame_number=0,  # TODO: 实际帧号需用 MV_CC_GetFrameCount
            timestamp_ns=0,  # TODO: 实际时间戳需用 MV_CC_GetTimestamp
            camera_serial=self._device_info.Serial.decode(errors='ignore'),
            exposure_us=exposure.value,
            gain_db=gain.value,
        )

    def start_continuous(self, callback) -> None:
        """启动连续采集。"""
        if not self._is_open:
            raise RuntimeError("Camera not opened")
        self._frame_callback = callback
        self._stop_event.clear()
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True,
            name=f"hikvision-cam-{self.camera_id}"
        )
        self._grab_thread.start()

    def _grab_loop(self) -> None:
        """连续采集循环。"""
        while not self._stop_event.is_set():
            try:
                frame = self.grab_one(timeout_ms=200)
                if self._frame_callback:
                    self._frame_callback(frame)
            except Exception as e:
                # 错误时短暂休息（避免忙等）
                if 'TIMEOUT' in str(e) or 'NOT_FIND' in str(e):
                    time.sleep(0.5)  # 相机断开时多休息
                else:
                    time.sleep(0.01)
                if 'NOT_FIND' in str(e) or 'USB' in str(e):
                    # 尝试重连
                    try:
                        self.close()
                        time.sleep(1)
                        self.open()
                    except Exception:
                        pass

    def stop_continuous(self) -> None:
        """停止连续采集。"""
        self._stop_event.set()
        if self._grab_thread:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None

    def close(self) -> None:
        """关闭相机。"""
        self.stop_continuous()
        self._cleanup()

    def _cleanup(self) -> None:
        if self._handle and self._handle.value:
            try:
                _MVS_SDK.MV_CC_CloseDevice(self._handle)
            except Exception:
                pass
            try:
                _MVS_SDK.MV_CC_DestroyHandle(self._handle)
            except Exception:
                pass
            self._handle = None
            self._is_open = False

    def get_serial(self) -> str:
        if self._device_info:
            return self._device_info.Serial.decode(errors='ignore')
        return ''

    def get_model(self) -> str:
        return 'Hikvision MV-CL042-10GM'


# 错误类（与之前 deepgait/core/errors.py 集成）
class CameraError(Exception):
    """相机通用错误。"""
    pass


class CameraDisconnectedError(CameraError):
    """相机断开。"""
    pass


class CameraTimeoutError(CameraError):
    """相机帧超时。"""
    pass
