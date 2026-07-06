"""RP2040 hardware trigger controller (GPIN Line 0 → camera Line 0/1).

deepgait v2 uses an external hardware trigger (Option A in
kb/14_hardware_sync.md §2) so that all four cameras start their exposure
on the same edge — this guarantees frame-level synchronisation with
±100 µs accuracy, well inside the ±1 ms acceptance gate from
``DEVELOPMENT_PLAN.md §9.1``.

Architecture
------------
* **RP2040 PIO** generates a precise periodic pulse train on a GPIO line.
  MicroPython reference firmware is sketched in kb/14_hardware_sync.md §3.
* **SN74LV1T34** level shifter converts the 3.3 V RP2040 output to the
  5 V TTL level required by the Hikvision/Basler opto-isolated trigger
  inputs (the trigger is wired to GPIN Line 0 on every camera).
* **TriggerController** in this module is the *host-side* controller.
  It opens the serial port to the RP2040, sends ``START <hz>`` and
  ``STOP`` commands, and exposes a heartbeat (alive / pulse count) for
  the recording pipeline to monitor.

If the RP2040 / serial port is unavailable (developer laptop, CI,
shipping builds without dongle), ``TriggerController`` falls back to a
software-pulse source that ticks on ``time.perf_counter`` and emits
``PulseEvent`` records with the same wall-clock semantics. This keeps
``MultiCameraManager`` hardware-agnostic.

References
----------
* kb/14_hardware_sync.md §2 (Option A), §3 (RP2040 firmware), §6 (wiring)
* docs/DEVELOPMENT_PLAN.md §5.2 (``hardware/camera/hikvision.py``)
* docs/REQUIREMENTS.md FR-HW-005 (hardware trigger mandatory)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pulse event
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PulseEvent:
    """Single hardware trigger pulse as observed by the host."""
    sequence: int           # monotonically increasing pulse index (1-based)
    timestamp_ns: int       # host perf_counter_ns at pulse receive time
    source: str             # "rp2040" | "mock"

    @property
    def timestamp_us(self) -> float:
        return self.timestamp_ns / 1_000.0


# ---------------------------------------------------------------------------
# Serial-port probe
# ---------------------------------------------------------------------------
def _serial_available() -> bool:
    try:
        import serial  # type: ignore[import-not-found]  # pyserial
        return True
    except Exception:
        return False


_SERIAL = _serial_available()


# ---------------------------------------------------------------------------
# Trigger controller
# ---------------------------------------------------------------------------
class TriggerController:
    """RP2040 trigger controller with software-fallback.

    Args:
        port: serial port (e.g. ``/dev/ttyACM0`` on Linux, ``COM7`` on Windows).
            If ``None``, the controller uses the software fallback.
        baud: serial baud rate (default 115200 — matches the reference
            MicroPython firmware in kb/14_hardware_sync.md §3).
        target_hz: nominal trigger frequency (default 100 Hz).
        pulse_width_us: width of the active-high pulse (default 10 µs,
            easily detected by Hikvision/Basler trigger inputs).
        on_pulse: optional callback fired on every received pulse.
            Signature: ``Callable[[PulseEvent], None]``.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = 115200,
        target_hz: int = 100,
        pulse_width_us: int = 10,
        on_pulse: Optional[Callable[[PulseEvent], None]] = None,
    ) -> None:
        if target_hz <= 0:
            raise ValueError("target_hz must be positive")
        if pulse_width_us <= 0:
            raise ValueError("pulse_width_us must be positive")

        self.port = port
        self.baud = baud
        self.target_hz = int(target_hz)
        self.pulse_width_us = int(pulse_width_us)
        self._on_pulse = on_pulse

        self._use_mock = (port is None) or (not _SERIAL)
        self._ser = None
        self._reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sequence = 0
        self._started_at_ns: Optional[int] = None
        self._last_pulse_ns: Optional[int] = None
        self._pulse_count = 0
        self._lock = threading.Lock()

    # ---- public API -------------------------------------------------------
    def start(self) -> None:
        """Begin emitting / receiving trigger pulses."""
        if self._reader_thread and self._reader_thread.is_alive():
            return
        self._stop_event.clear()
        with self._lock:
            self._sequence = 0
            self._pulse_count = 0
            self._started_at_ns = None
            self._last_pulse_ns = None
        if self._use_mock:
            self._reader_thread = threading.Thread(
                target=self._mock_loop, name="trigger-mock",
                daemon=True,
            )
        else:
            self._open_serial()
            self._send_cmd(f"START {self.target_hz} {self.pulse_width_us}")
            self._reader_thread = threading.Thread(
                target=self._serial_loop, name="trigger-rp2040",
                daemon=True,
            )
        self._reader_thread.start()
        logger.info(
            "TriggerController started: source=%s target_hz=%d",
            "mock" if self._use_mock else f"rp2040:{self.port}",
            self.target_hz,
        )

    def stop(self) -> None:
        """Stop emitting / receiving pulses and release resources."""
        self._stop_event.set()
        if self._reader_thread:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        if self._ser is not None:
            try:
                self._send_cmd("STOP")
            except Exception:
                logger.exception("TriggerController: STOP command failed")
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        logger.info("TriggerController stopped")

    def is_alive(self) -> bool:
        """Return True if pulses have been received within the last 2× period."""
        # NB: do NOT take self._lock here — callers (get_heartbeat) already
        # hold it and threading.Lock is non-reentrant, so a second acquire
        # would deadlock the calling thread. Instead, peek the timestamp
        # under the same lock by going through _alive_locked.
        return self._alive_locked()

    def _alive_locked(self) -> bool:
        """Caller must already hold ``self._lock``."""
        if self._pulse_count == 0 or self._last_pulse_ns is None:
            return False
        elapsed_ns = time.perf_counter_ns() - self._last_pulse_ns
        max_idle_ns = int(2.0 * 1_000_000_000 / self.target_hz)
        return elapsed_ns <= max_idle_ns

    def get_heartbeat(self) -> dict:
        """Return a snapshot of trigger health for the recording monitor."""
        with self._lock:
            return {
                "running": self._reader_thread is not None
                           and self._reader_thread.is_alive(),
                "source": "mock" if self._use_mock else "rp2040",
                "target_hz": self.target_hz,
                "pulse_count": self._pulse_count,
                "alive": self._alive_locked(),
            }

    # ---- internals --------------------------------------------------------
    def _emit_pulse(self, source: str) -> PulseEvent:
        with self._lock:
            self._sequence += 1
            seq = self._sequence
            ts = time.perf_counter_ns()
            self._last_pulse_ns = ts
            self._pulse_count += 1
            if self._started_at_ns is None:
                self._started_at_ns = ts
        ev = PulseEvent(sequence=seq, timestamp_ns=ts, source=source)
        if self._on_pulse is not None:
            try:
                self._on_pulse(ev)
            except Exception:
                logger.exception("TriggerController.on_pulse callback raised")
        return ev

    def _mock_loop(self) -> None:
        """Software-emulated pulse train at ``target_hz``.

        Uses small wait windows so the loop wakes up promptly when
        :meth:`stop` sets the event (test runtimes and pytest capture
        threads can otherwise interact badly with long single waits).
        """
        period_s = 1.0 / self.target_hz
        chunk_s = min(period_s, 0.005)
        while not self._stop_event.is_set():
            self._emit_pulse("mock")
            # Sleep, but wake up promptly on stop.
            self._stop_event.wait(chunk_s)

    def _open_serial(self) -> None:
        import serial  # type: ignore[import-not-found]

        self._ser = serial.Serial(self.port, self.baud, timeout=0.1)

    def _send_cmd(self, cmd: str) -> None:
        if self._ser is None:
            return
        self._ser.write((cmd + "\n").encode("ascii"))

    def _serial_loop(self) -> None:
        """Read ``PULSE <seq>`` lines from the RP2040 firmware."""
        assert self._ser is not None
        while not self._stop_event.is_set():
            try:
                raw = self._ser.readline()
            except Exception:
                logger.exception("TriggerController: serial read failed")
                break
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line.startswith("PULSE"):
                continue
            # We trust the firmware's monotonic sequence; just emit a host
            # pulse with our own timestamp.
            self._emit_pulse("rp2040")

    # ---- introspection ----------------------------------------------------
    @property
    def is_mock(self) -> bool:
        return self._use_mock

    def __enter__(self) -> "TriggerController":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        return (f"TriggerController(port={self.port!r}, "
                f"hz={self.target_hz}, mock={self._use_mock})")