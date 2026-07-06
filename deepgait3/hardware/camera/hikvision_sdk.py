"""海康威视 MVS SDK ctypes 绑定。
完整定义见 kb/21_camera_sdk.md。
"""
# deepgait/hardware/camera/hikvision_sdk.py
import ctypes
import os
import sys


def _load_mvs_library():
    """加载海康 MVS 动态库。"""
    if sys.platform == 'win32':
        paths = [
            r'C:\Program Files (x86)\MVS\Development\Bin\MvCameraControl.dll',
            r'C:\Program Files\MVS\Development\Bin\MvCameraControl.dll',
        ]
        for path in paths:
            if os.path.exists(path):
                return ctypes.cdll.LoadLibrary(path)
        raise FileNotFoundError(
            "MvCameraControl.dll not found. "
            "Install MVS SDK from https://www.hikrobotics.com/cn/mvsdownload/"
        )
    return ctypes.cdll.LoadLibrary("libMvCameraControl.so")


# 全局 SDK 句柄
_MVS_SDK = _load_mvs_library()


# 设备信息结构
class _MV_CC_DEVICE_INFO(ctypes.Structure):
    _fields_ = [
        ('nMajorVer', ctypes.c_ubyte),
        ('nMinorVer', ctypes.c_ubyte),
        ('nSDKBuildVersion', ctypes.c_uint),
        ('nSpecialVersion', ctypes.c_ubyte),
        ('MacAddr', ctypes.c_ubyte * 6),
        ('Serial', ctypes.c_char * 16),
        ('UserDefinedName', ctypes.c_char * 16),
        ('reserved', ctypes.c_ubyte * 232),
        ('DevTypeInfo', ctypes.c_uint),
        ('Reserved1', ctypes.c_uint),
        ('Reserved2', ctypes.c_uint),
        ('Reserved3', ctypes.c_uint),
    ]


# 函数签名设置
_MVS_SDK.MV_CC_EnumDevices.argtypes = [ctypes.c_uint, ctypes.POINTER(_MV_CC_DEVICE_INFO), ctypes.POINTER(ctypes.c_uint)]
_MVS_SDK.MV_CC_EnumDevices.restype = ctypes.c_int

_MVS_SDK.MV_CC_CreateHandle.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_MVS_SDK.MV_CC_CreateHandle.restype = ctypes.c_int

_MVS_SDK.MV_CC_OpenDevice.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_int]
_MVS_SDK.MV_CC_OpenDevice.restype = ctypes.c_int

_MVS_SDK.MV_CC_CloseDevice.argtypes = [ctypes.c_void_p]
_MVS_SDK.MV_CC_CloseDevice.restype = ctypes.c_int

_MVS_SDK.MV_CC_DestroyHandle.argtypes = [ctypes.c_void_p]
_MVS_SDK.MV_CC_DestroyHandle.restype = ctypes.c_int

_MVS_SDK.MV_CC_GetImageForBGR.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
_MVS_SDK.MV_CC_GetImageForBGR.restype = ctypes.c_int

_MVS_SDK.MV_CC_StartGrabbing.argtypes = [ctypes.c_void_p, ctypes.c_uint]
_MVS_SDK.MV_CC_StartGrabbing.restype = ctypes.c_int

_MVS_SDK.MV_CC_StopGrabbing.argtypes = [ctypes.c_void_p]
_MVS_SDK.MV_CC_StopGrabbing.restype = ctypes.c_int

_MVS_SDK.MV_CC_SetEnumValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
_MVS_SDK.MV_CC_SetEnumValue.restype = ctypes.c_int

_MVS_SDK.MV_CC_SetFloatValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
_MVS_SDK.MV_CC_SetFloatValue.restype = ctypes.c_int

_MVS_SDK.MV_CC_SetIntValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
_MVS_SDK.MV_CC_SetIntValue.restype = ctypes.c_int

_MVS_SDK.MV_CC_GetFloatValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float)]
_MVS_SDK.MV_CC_GetFloatValue.restype = ctypes.c_int

_MVS_SDK.MV_CC_GetIntValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint)]
_MVS_SDK.MV_CC_GetIntValue.restype = ctypes.c_int


# 错误码
MVS_OK = 0x00000000
MVS_ERROR_CODES = {
    0x80000000: 'E_HANDLE',
    0x80000001: 'E_GENERIC',
    0x80000008: 'E_TIMEOUT',
    0x80000009: 'E_ACCESS_DENIED',
    0x8000000A: 'E_ABNORMAL_IMAGE',
    0x80000011: 'E_NOT_FIND_DEVICE',
    0x80000020: 'E_USB_DEVICE_DISCONNECT',
}


# 触发源
TRIGGER_SOURCE_LINE0 = 1
TRIGGER_SOURCE_LINE1 = 2
TRIGGER_SOURCE_LINE2 = 3
TRIGGER_SOURCE_LINE3 = 4
TRIGGER_SOURCE_SOFTWARE = 6


# 像素格式
PIXEL_FORMAT_BGR8 = 0x02180014
PIXEL_FORMAT_BAYER_RG8 = 0x01080009


def mvs_check(ret: int, op: str) -> None:
    """检查海康 MVS 返回码，失败则抛异常。"""
    if ret != MVS_OK:
        code_name = MVS_ERROR_CODES.get(ret, f'UNKNOWN(0x{ret:08X})')
        raise RuntimeError(f"Hikvision MVS {op} failed: {code_name} (0x{ret:08X})")
