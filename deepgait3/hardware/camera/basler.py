"""Basler Camera driver (Linux fallback for Hikvision).

Implements :class:`deepgait3.hardware.camera.base.ICamera` against the pypylon
SDK. If pypylon / libpylon is not installed (e.g. on a developer workstation
without Basler hardware, or on CI), :class:`BaslerCamera` falls back to a
deterministic synthetic frame source so unit tests can still exercise the
public API end-to-end.

Selection strategy (Windows / Linux) is centralised in
:func:`deepgait3.hardware.camera.base.CameraFactory.create`. The fallback
behaviour here mirrors the same spirit: prefer real hardware, degrade
gracefully, and log loudly.

References
----------
* kb/14_hardware_sync.md   — hardware trigger + multi-cam architecture
* kb/21_camera_sdk.md      — pypylon API + Line1 trigger source
* MODULES.md §1.2.b        — ICamera contract
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional, Dict, Any

import numpy as np

from .base import FrameInfo, ICamera


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware availability probe
# ---------------------------------------------------------------------------
def _pypylon_available() -> bool:
    """Return True if pypylon is importable on this machine."""
    try:
        import pypylon  # noqa: F401
        return True
    except Exception:
        return False


_PYPYLON = _pypylon_available()


# ---------------------------------------------------------------------------
# BaslerCamera
# ---------------------------------------------------------------------------
class BaslerCamera(ICamera):
    """Basler GigE / USB3 camera (Linux preferred backend).

    Args:
        camera_id: device index among detected Basler cameras (0..N-1).
        serial: optional serial number for explicit device selection.
        width / height: ROI (defaults to Basler acA1920-150uc: 1920×1200).
        fps: target frame rate (default 100).
        use_mock: if True (or pypylon unavailable), emit synthetic frames
            instead of talking to hardware. Tests rely on this.
    """

    def __init__(
        self,
        camera_id: int = 0,
        serial: Optional[str] = None,
        width: int = 1920,
        height: int = 1200,
        fps: int = 100,
        use_mock: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.serial_filter = serial
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._is_open = False
        self._serial = serial or f"BASLER-MOCK-{camera_id:04d}"
        self._model = "Basler acA1920-150uc" if not use_mock else "Basler-Mock"

        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable] = None
        self._frame_counter = 0

        # Use mock if requested or if pypylon is unavailable on this host.
        self._use_mock = use_mock or not _PYPYLON
        if self._use_mock and not use_mock:
            logger.warning(
                "BaslerCamera: pypylon unavailable, falling back to synthetic "
                "frames (camera_id=%d). Real Basler acquisition disabled.",
                camera_id,
            )

    # ---- ICamera API ------------------------------------------------------
    def open(self) -> None:
        if self._is_open:
            return
        if not self._use_mock:
            self._open_real()
        else:
            # Mock path: no IO, just mark open.
            time.sleep(0.0)
        self._is_open = True
        logger.info("BaslerCamera opened: serial=%s model=%s",
                    self._serial, self._model)

    def close(self) -> None:
        self.stop_continuous()
        self._is_open = False
        logger.info("BaslerCamera closed: serial=%s", self._serial)

    def grab_one(self, timeout_ms: int = 5000) -> FrameInfo:
        if not self._is_open:
            raise RuntimeError("BaslerCamera.grab_one(): camera not open")
        if self._use_mock:
            return self._make_mock_frame()
        return self._grab_real(timeout_ms)

    def start_continuous(self, callback: Callable[[FrameInfo], None]) -> None:
        if not self._is_open:
            raise RuntimeError("BaslerCamera.start_continuous(): camera not open")
        if self._grab_thread and self._grab_thread.is_alive():
            return
        self._stop_event.clear()
        self._callback = callback
        self._grab_thread = threading.Thread(
            target=self._grab_loop, name=f"basler-grab-{self.camera_id}",
            daemon=True,
        )
        self._grab_thread.start()

    def stop_continuous(self) -> None:
        self._stop_event.set()
        if self._grab_thread:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None

    def configure_hardware_trigger(self, line: int = 1, edge: str = "rising") -> None:
        # Basler convention: Line1 / Line2 / Line3 / Line4 (no Line0).
        if line not in (1, 2, 3, 4):
            raise ValueError(f"Basler trigger line must be 1..4, got {line}")
        if edge not in ("rising", "falling"):
            raise ValueError(f"edge must be 'rising' or 'falling', got {edge!r}")
        if not self._use_mock:
            self._configure_trigger_real(line, edge)
        logger.info("BaslerCamera trigger configured: Line%d, %s edge",
                    line, edge)

    def set_exposure_us(self, exposure_us: float) -> None:
        if exposure_us <= 0:
            raise ValueError("exposure_us must be positive")
        if not self._use_mock:
            self._set_exposure_real(exposure_us)

    def set_gain_db(self, gain_db: float) -> None:
        if not self._use_mock:
            self._set_gain_real(gain_db)

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("ROI width/height must be positive")
        self._width, self._height = int(width), int(height)
        if not self._use_mock:
            self._set_roi_real(x, y, width, height)

    def get_serial(self) -> str:
        return self._serial

    def get_model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # W17: extended parameter set (brightness / contrast / fps / format)
    # ------------------------------------------------------------------
    def set_brightness(self, value: int) -> None:
        if not -100 <= value <= 100:
            raise ValueError(f"brightness {value} out of range [-100, 100]")
        if not self._use_mock:
            try:
                self._camera.Brightness.SetValue(int(value))
                return
            except Exception as e:
                logger.warning("Basler set_brightness failed: %s; using mock", e)
        self._brightness = int(value)

    def set_contrast(self, value: int) -> None:
        if not -100 <= value <= 100:
            raise ValueError(f"contrast {value} out of range [-100, 100]")
        if not self._use_mock:
            try:
                self._camera.Contrast.SetValue(int(value))
                return
            except Exception as e:
                logger.warning("Basler set_contrast failed: %s; using mock", e)
        self._contrast = int(value)

    def set_pixel_format(self, fmt: str) -> None:
        if not fmt:
            raise ValueError("pixel_format must be a non-empty string")
        if not self._use_mock:
            try:
                self._camera.PixelFormat.SetValue(str(fmt))
                return
            except Exception as e:
                logger.warning("Basler set_pixel_format failed: %s; using mock", e)
        self._pixel_format = str(fmt)

    def set_fps(self, fps: int) -> None:
        if not 1 <= fps <= 500:
            raise ValueError(f"fps {fps} out of range [1, 500]")
        self._fps = int(fps)
        if not self._use_mock:
            try:
                # Basler: 需先关闭 AcquisitionFrameRateEnable 才能手动设
                self._camera.AcquisitionFrameRateEnable.SetValue(True)
                self._camera.AcquisitionFrameRate.SetValue(float(fps))
            except Exception as e:
                logger.warning("Basler set_fps failed: %s; using mock", e)

    def get_supported_features(self) -> Dict[str, Any]:
        return {
            "brightness": (-100, 100, 0, 1),
            "contrast": (-100, 100, 0, 1),
            "exposure_us": (50.0, 1_000_000.0, 5_000.0),  # Basler 上限更大
            "gain_db": (0.0, 36.0, 0.0, 0.1),
            "fps": (1, 500, 100),
            "pixel_format": ["BayerRG8", "BayerGB8", "BayerGR8", "BayerBG8",
                              "Mono8", "RGB8", "BGR8"],
            "roi": {"min_w": 64, "min_h": 64, "max_w": 4096, "max_h": 4096},
        }

    def snapshot_config(self) -> Dict[str, Any]:
        return {
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "exposure_us": 5000.0,
            "gain_db": 0.0,
            "brightness": getattr(self, "_brightness", 0),
            "contrast": getattr(self, "_contrast", 0),
            "pixel_format": getattr(self, "_pixel_format", "BGR8"),
        }

    def restore_config(self, cfg: Dict[str, Any]) -> None:
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

    # ---- helpers ----------------------------------------------------------
    def _make_mock_frame(self) -> FrameInfo:
        """Emit a deterministic synthetic frame for tests / no-HW dev."""
        self._frame_counter += 1
        rng = np.random.default_rng(self._frame_counter ^ self.camera_id)
        image = rng.integers(0, 256, size=(self._height, self._width, 3),
                             dtype=np.uint8)
        return FrameInfo(
            image=image,
            frame_number=self._frame_counter,
            timestamp_ns=time.monotonic_ns(),
            camera_serial=self._serial,
            exposure_us=5000.0,
            gain_db=0.0,
        )

    def _grab_loop(self) -> None:
        period_s = 1.0 / max(self._fps, 1)
        while not self._stop_event.is_set():
            frame = self.grab_one()
            if self._callback is not None:
                try:
                    self._callback(frame)
                except Exception:
                    logger.exception("BaslerCamera callback raised")
            # Best-effort pacing (mock path only; real path is hardware-paced).
            if self._use_mock:
                self._stop_event.wait(period_s)

    # ---- real-hardware paths (lazy-imported) ------------------------------
    def _open_real(self) -> None:
        from pypylon import pylon  # type: ignore[import-not-found]

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found on the bus")
        dev = devices[self.camera_id]
        if self.serial_filter and dev.GetSerialNumber() != self.serial_filter:
            raise RuntimeError(
                f"Basler serial mismatch: requested {self.serial_filter}, "
                f"got {dev.GetSerialNumber()}"
            )
        self._serial = dev.GetSerialNumber()
        self._camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateDevice(dev)
        )
        self._camera.Open()

    def _grab_real(self, timeout_ms: int) -> FrameInfo:
        grab = self._camera.GrabOne(timeout_ms)
        if not grab.GrabSucceeded():
            raise RuntimeError(f"Basler grab failed: {grab.ErrorDescription}")
        converter = self._converter  # created lazily
        img = converter.Convert(grab)
        arr = img.GetArray()
        return FrameInfo(
            image=arr,
            frame_number=int(grab.ImageNumber),
            timestamp_ns=int(grab.ImageTimestamp),
            camera_serial=self._serial,
            exposure_us=float(grab.GetExposureTime()),
            gain_db=float(grab.GetGain()),
        )

    def _configure_trigger_real(self, line: int, edge: str) -> None:
        cam = self._camera
        cam.TriggerSelector.SetValue("FrameStart")
        cam.TriggerMode.SetValue("On")
        cam.TriggerSource.SetValue(f"Line{line}")
        cam.TriggerActivation.SetValue(
            "RisingEdge" if edge == "rising" else "FallingEdge"
        )
        cam.AcquisitionMode.SetValue("Continuous")

    def _set_exposure_real(self, exposure_us: float) -> None:
        self._camera.ExposureTime.SetValue(float(exposure_us))

    def _set_gain_real(self, gain_db: float) -> None:
        self._camera.Gain.SetValue(float(gain_db))

    def _set_roi_real(self, x: int, y: int, width: int, height: int) -> None:
        cam = self._camera
        cam.OffsetX.SetValue(int(x))
        cam.OffsetY.SetValue(int(y))
        cam.Width.SetValue(int(width))
        cam.Height.SetValue(int(height))

    @property
    def _converter(self):
        """Lazy pypylon image-format converter (BGR8 ndarray)."""
        if not hasattr(self, "_converter_inst"):
            from pypylon import pylon  # type: ignore[import-not-found]

            self._converter_inst = pylon.ImageFormatConverter()
            self._converter_inst.OutputPixelFormat = pylon.PixelType_BGR8packed
        return self._converter_inst

    # ---- introspection ----------------------------------------------------
    @property
    def is_mock(self) -> bool:
        return self._use_mock

    def __repr__(self) -> str:
        return (f"BaslerCamera(id={self.camera_id}, serial={self._serial}, "
                f"model={self._model}, mock={self._use_mock})")