"""License heartbeat — Layer 3 of deepgait v2.

A background thread that periodically re-verifies the license and
fires a "license lost" callback if the verification fails or the
dongle stops responding. Designed to prevent the "unplug the dongle
mid-session to keep using the software" attack.

References
----------
* kb/19_license_management.md §5
* MODULES.md §3 "Heartbeat"
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

from .verifier import LicenseVerifier


logger = logging.getLogger(__name__)


class LicenseHeartbeat:
    """Background license watchdog.

    Parameters
    ----------
    verifier : LicenseVerifier
        The verifier to monitor. Must already have been verified once.
    interval_s : float
        Re-verification period (default 60 s — matches MODULES.md §3).
    on_lost : callable, optional
        Optional fallback ``on_lost`` callback in addition to the ones
        registered via ``verifier.on_lost``.
    """

    def __init__(
        self,
        verifier: LicenseVerifier,
        interval_s: float = 60.0,
        on_lost: Optional[Callable[[], None]] = None,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self._verifier = verifier
        self._interval_s = float(interval_s)
        self._on_lost = on_lost
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_check_ts: float = 0.0
        self._misses = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._misses = 0
        self._thread = threading.Thread(
            target=self._loop, name="license-heartbeat", daemon=True,
        )
        self._thread.start()
        logger.info("LicenseHeartbeat started (interval=%.1fs)", self._interval_s)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("LicenseHeartbeat stopped")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running": self.is_alive(),
                "interval_s": self._interval_s,
                "last_check_ts": self._last_check_ts,
                "misses": self._misses,
            }

    # ---- internals --------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._verifier.verify()
                with self._lock:
                    self._last_check_ts = time.time()
                    self._misses = 0
            except Exception as e:
                logger.warning("LicenseHeartbeat: verification failed: %s", e)
                with self._lock:
                    self._misses += 1
                    misses = self._misses
                # Two consecutive misses = assume dongle gone.
                if misses >= 2:
                    logger.error("LicenseHeartbeat: license LOST — notifying")
                    self._fire_lost()
                    return
            # Wait, but wake up promptly on stop.
            self._stop_event.wait(self._interval_s)

    def _fire_lost(self) -> None:
        try:
            self._verifier._notify_lost()  # type: ignore[attr-defined]
        except Exception:
            logger.exception("LicenseHeartbeat: verifier notify failed")
        if self._on_lost is not None:
            try:
                self._on_lost()
            except Exception:
                logger.exception("LicenseHeartbeat: on_lost callback raised")

    def __enter__(self) -> "LicenseHeartbeat":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"LicenseHeartbeat(interval={self._interval_s}s, alive={self.is_alive()})"