"""Multi-camera synchronised acquisition manager.

Owns a fixed-size roster of four :class:`ICamera` instances (C1 bottom
FTIR + C2/C3/C4 DLC pose) and coordinates their start, configuration
and synchronised frame delivery.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.1 W2):
    "4 cameras enumerate + frame grab pass"

Acceptance gate (docs/DEVELOPMENT_PLAN.md §9.1):
    "4 cameras synchronised (frame-to-frame error < 1 ms)"

Design
------
* **Roster**: a labelled list of (:class:`ICamera`, role, expected_fps).
  The label is what the GUI / recorder uses as a stable handle
  (``"bottom"``, ``"left"``, ``"right"``, ``"top"``).
* **Synchronous start**: when the user calls :meth:`start_all`, every
  camera is switched into hardware-trigger mode first
  (line 0 on Hikvision, line 1 on Basler) and then ``start_continuous``
  is dispatched. The hardware trigger — driven by the RP2040 PIO in
  :mod:`trigger` — is what actually makes the frames arrive in lockstep,
  not the host loop.
* **Frame bus**: :class:`FrameBus` accumulates the most recent
  :class:`FrameInfo` per camera (one slot per role). The recorder
  writes a complete quartet per trigger pulse.
* **Sync monitor**: :meth:`get_sync_report` returns the maximum
  observed frame-to-frame timestamp delta across the roster, which
  must stay below ``SYNC_TOLERANCE_MS`` (1 ms per acceptance gate).

The class is hardware-agnostic — it accepts any :class:`ICamera`. The
unit tests pass :class:`MockCamera` (defined in this module) so the
"4 cameras enumerate + grab pass" gate can be verified on a developer
laptop without Hikvision / Basler hardware.
"""
from __future__ import annotations

import logging
import platform
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import CameraFactory, FrameInfo, ICamera


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYNC_TOLERANCE_MS: float = 1.0
"""Maximum permitted inter-camera frame timestamp delta, in milliseconds."""

DEFAULT_ROSTER: Tuple[str, ...] = ("bottom", "left", "right", "top")
"""Standard 4-camera roster for deepgait v2 (C1..C4)."""


# ---------------------------------------------------------------------------
# SyncReport
# ---------------------------------------------------------------------------
@dataclass
class SyncReport:
    """Snapshot of multi-camera synchronisation health."""
    in_sync: bool
    """Whether the latest quartet is within ``SYNC_TOLERANCE_MS``."""

    max_delta_ms: float
    """Worst observed frame timestamp delta across the latest quartet, ms."""

    per_role_delta_ms: Dict[str, float] = field(default_factory=dict)
    """Per-pair delta against the reference (median) frame, ms."""

    sample_size: int = 0
    """Number of complete quartets observed."""

    def as_dict(self) -> dict:
        return {
            "in_sync": self.in_sync,
            "max_delta_ms": self.max_delta_ms,
            "per_role_delta_ms": dict(self.per_role_delta_ms),
            "sample_size": self.sample_size,
        }


# ---------------------------------------------------------------------------
# MockCamera — deterministic synthetic source for tests
# ---------------------------------------------------------------------------
class MockCamera(ICamera):
    """Deterministic :class:`ICamera` used in unit tests.

    The frame timestamps advance by exactly ``1e9 / fps`` nanoseconds per
    ``grab_one`` call, so a quartet of :class:`MockCamera` instances
    driven by the same logical clock stays well inside the ±1 ms sync
    gate. A configurable ``clock_offset_ms`` lets tests inject per-camera
    jitter to exercise the sync monitor's failure detection.
    """

    def __init__(
        self,
        serial: str,
        fps: int = 100,
        width: int = 2448,
        height: int = 2048,
        clock_offset_ms: float = 0.0,
    ) -> None:
        self._serial = serial
        self._fps = int(fps)
        self._width = int(width)
        self._height = int(height)
        self._clock_offset_ms = float(clock_offset_ms)
        self._is_open = False
        self._frame_number = 0
        self._start_ns: Optional[int] = None
        self._model = "MockCamera"
        self._grab_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callback: Optional[Callable] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        self._is_open = True
        self._start_ns = time.perf_counter_ns()

    def close(self) -> None:
        self.stop_continuous()
        self._is_open = False

    def grab_one(self, timeout_ms: int = 5000) -> FrameInfo:
        if not self._is_open:
            raise RuntimeError("MockCamera.grab_one(): camera not open")
        with self._lock:
            self._frame_number += 1
            fn = self._frame_number
            base = self._start_ns or time.perf_counter_ns()
            ts = base + int(self._frame_number * 1_000_000_000 / self._fps)
            ts += int(self._clock_offset_ms * 1_000_000)
        image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        # Embed the frame number in the top-left 4×4 patch so tests can
        # verify they got distinct frames from each camera.
        image[0, 0, 0] = fn & 0xFF
        return FrameInfo(
            image=image,
            frame_number=fn,
            timestamp_ns=ts,
            camera_serial=self._serial,
            exposure_us=5000.0,
            gain_db=0.0,
        )

    def start_continuous(self, callback: Callable[[FrameInfo], None]) -> None:
        if not self._is_open:
            raise RuntimeError("MockCamera.start_continuous(): camera not open")
        if self._grab_thread and self._grab_thread.is_alive():
            return
        self._stop_event.clear()
        self._callback = callback
        period_s = 1.0 / max(self._fps, 1)
        self._grab_thread = threading.Thread(
            target=self._loop, args=(period_s,),
            name=f"mock-cam-{self._serial}", daemon=True,
        )
        self._grab_thread.start()

    def stop_continuous(self) -> None:
        self._stop_event.set()
        if self._grab_thread:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None

    def configure_hardware_trigger(self, line: int = 0, edge: str = "rising") -> None:
        # Mock accepts any line; not connected to real wiring.
        if line not in (0, 1, 2, 3):
            raise ValueError(f"line must be 0..3, got {line}")

    def set_exposure_us(self, exposure_us: float) -> None:
        if exposure_us <= 0:
            raise ValueError("exposure_us must be positive")

    def set_gain_db(self, gain_db: float) -> None:
        pass

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("ROI width/height must be positive")
        self._width, self._height = int(width), int(height)

    def get_serial(self) -> str:
        return self._serial

    def get_model(self) -> str:
        return self._model

    # ------------------------------------------------------------------
    # W17: extended parameter set (mock — state only, no hardware)
    # ------------------------------------------------------------------
    def set_brightness(self, value: int) -> None:
        if not -100 <= value <= 100:
            raise ValueError(f"brightness {value} out of range [-100, 100]")
        self._brightness = int(value)

    def set_contrast(self, value: int) -> None:
        if not -100 <= value <= 100:
            raise ValueError(f"contrast {value} out of range [-100, 100]")
        self._contrast = int(value)

    def set_pixel_format(self, fmt: str) -> None:
        if not fmt:
            raise ValueError("pixel_format must be a non-empty string")
        self._pixel_format = str(fmt)

    def set_fps(self, fps: int) -> None:
        if not 1 <= fps <= 500:
            raise ValueError(f"fps {fps} out of range [1, 500]")
        self._fps = int(fps)

    def get_supported_features(self) -> Dict[str, Any]:
        return ICamera._default_features()

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
            self.set_pixel_format(cfg["pixel_format"])
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

    def _loop(self, period_s: float) -> None:
        while not self._stop_event.is_set():
            frame = self.grab_one()
            if self._callback is not None:
                try:
                    self._callback(frame)
                except Exception:
                    logger.exception("MockCamera callback raised")
            self._stop_event.wait(period_s)


# ---------------------------------------------------------------------------
# FrameBus — per-role latest-frame buffer
# ---------------------------------------------------------------------------
class FrameBus:
    """Thread-safe latest-frame buffer keyed by camera role.

    Each :meth:`push` stores the frame under its role. :meth:`wait_quartet`
    blocks (up to ``timeout_ms``) until a fresh quartet is available —
    i.e. every role has produced at least one frame since the last
    consume. This is the synchronisation primitive used by the recorder.
    """

    def __init__(self, roles: Sequence[str], queue_capacity: int = 16) -> None:
        self._roles = tuple(roles)
        self._lock = threading.Condition(threading.Lock())
        self._slots: Dict[str, deque] = {r: deque(maxlen=queue_capacity)
                                          for r in self._roles}
        self._last_consume_ns: Dict[str, int] = {r: 0 for r in self._roles}
        self._quartet_event = threading.Event()

    def push(self, role: str, frame: FrameInfo) -> None:
        if role not in self._slots:
            raise KeyError(f"unknown role: {role!r}; known={self._roles}")
        with self._lock:
            self._slots[role].append(frame)
            self._lock.notify_all()

    def wait_quartet(self, timeout_ms: int = 1000) -> Optional[Dict[str, FrameInfo]]:
        """Return the latest frame per role, or ``None`` on timeout."""
        deadline = time.monotonic() + timeout_ms / 1000.0
        with self._lock:
            while True:
                snap = {}
                for role, slot in self._slots.items():
                    if not slot:
                        snap = None
                        break
                    snap[role] = slot[-1]
                if snap is not None:
                    return snap
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(timeout=remaining)

    def latest(self) -> Dict[str, Optional[FrameInfo]]:
        with self._lock:
            return {r: (s[-1] if s else None) for r, s in self._slots.items()}

    def reset(self) -> None:
        with self._lock:
            for s in self._slots.values():
                s.clear()


# ---------------------------------------------------------------------------
# MultiCameraManager
# ---------------------------------------------------------------------------
class MultiCameraManager:
    """Owns a roster of 4 cameras and provides synchronised access.

    Parameters
    ----------
    cameras : sequence of (role, ICamera)
        One entry per physical camera. ``role`` is a short label
        (e.g. ``"bottom"``, ``"left"``, ``"right"``, ``"top"``).
    trigger_line : int
        Hardware trigger line to configure on every camera (0 for
        Hikvision, 1 for Basler). The caller picks the right value
        for the installed camera family.
    sync_tolerance_ms : float
        Maximum permitted inter-camera timestamp delta, default 1.0 ms
        (per DEVELOPMENT_PLAN §9.1 acceptance gate).

    Use :meth:`enumerate_default` for a typical build where the four
    cameras are detected automatically.
    """

    def __init__(
        self,
        cameras: Sequence[Tuple[str, ICamera]],
        trigger_line: int = 0,
        trigger_edge: str = "rising",
        sync_tolerance_ms: float = SYNC_TOLERANCE_MS,
    ) -> None:
        if not cameras:
            raise ValueError("MultiCameraManager requires at least one camera")
        self._roster = list(cameras)
        self._roles = [r for r, _ in cameras]
        self._trigger_line = int(trigger_line)
        self._trigger_edge = str(trigger_edge)
        self._sync_tolerance_ms = float(sync_tolerance_ms)
        self._bus = FrameBus(self._roles)
        self._sync_samples: deque = deque(maxlen=1024)
        # Per-camera FPS estimation (W22): sliding window of hardware
        # timestamp_ns samples. Locked by self._lock below.
        self._fps_window_size: int = 30
        self._fps_timestamps: Dict[str, "deque[int]"] = {
            r: deque(maxlen=self._fps_window_size) for r in self._roles
        }
        self._lock = threading.Lock()
        self._running = False

    # ---- construction helpers --------------------------------------------
    @classmethod
    def enumerate_default(
        cls,
        n_cameras: int = 4,
        trigger_line: Optional[int] = None,
        sync_tolerance_ms: float = SYNC_TOLERANCE_MS,
    ) -> "MultiCameraManager":
        """Enumerate cameras via :class:`CameraFactory` and build a roster.

        On platforms without a working SDK (CI, dev laptops without
        hardware), every slot becomes a :class:`MockCamera` so the
        resulting manager is still useful for testing and demos.
        """
        roles = list(DEFAULT_ROSTER[:n_cameras])
        cams: List[Tuple[str, ICamera]] = []
        used_mock = False
        for idx, role in enumerate(roles):
            try:
                cam = CameraFactory.create(camera_id=idx)
            except Exception as e:
                logger.warning(
                    "CameraFactory.create(%d) failed (%s); using MockCamera",
                    idx, e,
                )
                cam = MockCamera(serial=f"MOCK-{role}", fps=100)
                used_mock = True
            cams.append((role, cam))
        # Hikvision uses Line 0; Basler uses Line 1.
        if trigger_line is None:
            sysname = platform.system()
            trigger_line = 1 if sysname == "Linux" else 0
        mgr = cls(
            cameras=cams,
            trigger_line=trigger_line,
            sync_tolerance_ms=sync_tolerance_ms,
        )
        mgr._built_with_mock = used_mock  # type: ignore[attr-defined]
        return mgr

    # ---- introspection ----------------------------------------------------
    @property
    def roles(self) -> List[str]:
        return list(self._roles)

    @property
    def is_running(self) -> bool:
        return self._running

    def get_camera(self, role: str) -> ICamera:
        for r, cam in self._roster:
            if r == role:
                return cam
        raise KeyError(f"unknown role: {role!r}")

    def __len__(self) -> int:
        return len(self._roster)

    # ---- lifecycle --------------------------------------------------------
    def open_all(self) -> None:
        for _, cam in self._roster:
            cam.open()
        logger.info("MultiCameraManager: opened %d cameras", len(self._roster))

    def close_all(self) -> None:
        self.stop_all()
        for _, cam in self._roster:
            cam.close()
        logger.info("MultiCameraManager: closed %d cameras", len(self._roster))

    def configure_trigger(self) -> None:
        for role, cam in self._roster:
            cam.configure_hardware_trigger(
                line=self._trigger_line, edge=self._trigger_edge,
            )
        logger.info(
            "MultiCameraManager: hardware trigger Line %d, %s edge on %d cameras",
            self._trigger_line, self._trigger_edge, len(self._roster),
        )

    def start_all(self) -> None:
        """Open every camera, configure hardware trigger, start grabbing."""
        self.open_all()
        self.configure_trigger()
        for role, cam in self._roster:
            cam.start_continuous(lambda f, role=role: self._on_frame(role, f))
        with self._lock:
            self._sync_samples.clear()
            self._running = True

    def stop_all(self) -> None:
        for _, cam in self._roster:
            cam.stop_continuous()
        with self._lock:
            self._running = False

    def grab_quartet(self, timeout_ms: int = 1000) -> Optional[Dict[str, FrameInfo]]:
        """Block until one frame from every camera is available."""
        snap = self._bus.wait_quartet(timeout_ms=timeout_ms)
        if snap is None:
            return None
        self._record_sync(snap)
        return snap

    # ---- sync monitoring --------------------------------------------------
    def _on_frame(self, role: str, frame: FrameInfo) -> None:
        self._bus.push(role, frame)
        # W22: also append the hardware timestamp to the per-role
        # sliding window so get_fps_per_role() can compute a real FPS.
        with self._lock:
            dq = self._fps_timestamps.get(role)
            if dq is not None:
                dq.append(frame.timestamp_ns)

    def _record_sync(self, snap: Dict[str, FrameInfo]) -> None:
        timestamps = [f.timestamp_ns for f in snap.values()]
        if not timestamps:
            return
        median = statistics.median(timestamps)
        deltas_ms = {
            role: abs(frame.timestamp_ns - median) / 1_000_000.0
            for role, frame in snap.items()
        }
        with self._lock:
            self._sync_samples.append(deltas_ms)

    def get_sync_report(self) -> SyncReport:
        """Compute the current sync health snapshot."""
        with self._lock:
            samples = list(self._sync_samples)
        if not samples:
            return SyncReport(in_sync=False, max_delta_ms=float("inf"),
                              sample_size=0)
        max_per_sample = [max(s.values()) for s in samples]
        worst = max(max_per_sample)
        # Use the worst-sample's per-role breakdown as the report detail.
        worst_idx = max_per_sample.index(worst)
        per_role = samples[worst_idx]
        return SyncReport(
            in_sync=worst <= self._sync_tolerance_ms,
            max_delta_ms=float(worst),
            per_role_delta_ms=dict(per_role),
            sample_size=len(samples),
        )

    # ---- per-camera FPS ---------------------------------------------------
    def get_fps_per_role(self) -> Dict[str, float]:
        """Return a sliding-window FPS estimate per camera role.

        The estimate is derived from the per-role deque of hardware
        ``timestamp_ns`` samples appended in :meth:`_on_frame`. The
        window size is ``_fps_window_size`` (default 30 frames).

        Returns
        -------
        dict
            ``{role: fps}``. When the window for a role has fewer than
            two samples (e.g. just after start), or when the delta is
            non-positive (clock anomaly), the value is ``float('nan')``.
            Callers should treat NaN as "no data yet" and render "—".
        """
        out: Dict[str, float] = {}
        with self._lock:
            for role, dq in self._fps_timestamps.items():
                if len(dq) < 2:
                    out[role] = float("nan")
                    continue
                dt_s = (dq[-1] - dq[0]) / 1e9
                if dt_s <= 0:
                    out[role] = float("nan")
                    continue
                out[role] = (len(dq) - 1) / dt_s
        return out

    # ---- context manager --------------------------------------------------
    def __enter__(self) -> "MultiCameraManager":
        self.start_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_all()

    def __repr__(self) -> str:
        cams = ", ".join(f"{r}={cam.get_model()}/{cam.get_serial()}"
                         for r, cam in self._roster)
        return f"MultiCameraManager([{cams}], trigger_line={self._trigger_line})"