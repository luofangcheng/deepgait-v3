"""Debug monitor for the PySide6 GUI.

Injects a Qt event filter + AppState signal logger that streams
structured JSON events to stdout and/or a log file.  Use with::

    deepgait3 gui --debug              # log to stdout
    deepgait3 gui --debug --log /tmp/dg.log  # log to file
    deepgait3 gui --debug --quiet      # file only (via --log)

The JSON-line format is::

    {"ts": 1234567890.123, "type": "signal|action|tab|menu|event",
     "detail": "...", "payload": {...}}

``type`` values
  ``signal`` — AppState signal emitted
  ``tab``    — QTabWidget tab switched
  ``menu``   — menu action triggered
  ``action`` — button / combobox / checkbox user action (filtered subset)
  ``worker`` — worker started / finished
  ``state``  — AppState setter called (backtrace included)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from typing import Any, Optional, TextIO

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QMenu,
    QTabWidget,
    QWidget,
)


# ---------------------------------------------------------------------------
# QEvent type → readable name
# ---------------------------------------------------------------------------
# PySide6: ``dir(QEvent)`` does NOT list enum members (they live on
# ``QEvent.Type``).  Iterate the enum directly instead.
_EVENT_NAMES: dict[QEvent.Type, str] = {}
for _ev in QEvent.Type:
    _EVENT_NAMES[_ev] = _ev.name


# Events worth logging (user-intent actions).
_LOG_EVENT_TYPES = frozenset({
    QEvent.Type.MouseButtonPress,
    QEvent.Type.KeyPress,
    QEvent.Type.Close,
})

# Widget classes whose MouseButtonPress / KeyPress events are interesting.
_INTERESTING_CLASSES = frozenset({
    "QPushButton", "QToolButton", "QCheckBox", "QRadioButton",
    "QComboBox", "QSpinBox", "QDoubleSpinBox", "QSlider",
    "QLineEdit", "QTextEdit", "QPlainTextEdit",
    "QTabBar", "QListView", "QTreeView", "QTableView",
    "QListWidget", "QTreeWidget", "QTableWidget",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obj_path(obj: QObject) -> str:
    """Best-effort widget path: ``ClassName(objectName)``."""
    cls = obj.__class__.__name__
    name = obj.objectName()
    if name:
        return f"{cls}({name})"
    return cls


def _payload_summary(obj: Any, max_len: int = 200) -> Any:
    """Return a brief, JSON-serializable summary of *obj*."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= max_len else obj[:max_len] + "…"
    if isinstance(obj, dict):
        return {k: _payload_summary(v, max_len // max(1, len(obj)))
                for k, v in list(obj.items())[:20]}
    if isinstance(obj, (list, tuple)):
        return [_payload_summary(v, max_len // max(1, len(obj)))
                for v in list(obj)[:10]]
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        return {k: _payload_summary(v, max_len // 2) for k, v in d.items()
                if not k.startswith("_")}
    return str(obj)[:max_len]


# ---------------------------------------------------------------------------
# DebugMonitor
# ---------------------------------------------------------------------------

class DebugMonitor(QObject):
    """Global debug monitor that attaches to a running QApplication.

    Parameters
    ----------
    app_state:
        The shared ``AppState`` whose signals should be logged.
    output:
        File-like object for JSON lines.  Defaults to ``sys.stderr`` so it
        does not interfere with stdout pipelines.
    verbose:
        When ``True`` (default), also log filtered QEvents.  Set to
        ``False`` to only see AppState signal + tab / menu actions.
    """

    # Mirror of AppState signals for logging — declared again here so we
    # can connect without leaking the signal object into JSON.
    _STATE_SIGNAL_NAMES = (
        "gait_results_changed",
        "footprint_changed",
        "pose_3d_changed",
        "calibration_changed",
        "project_changed",
        "status_message_changed",
        "camera_config_changed",
    )

    def __init__(
        self,
        app_state: QObject,
        output: TextIO | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self._state = app_state
        self._out: TextIO = output or sys.stderr
        self._verbose = verbose
        self._start_time = time.time()
        self._events_seen = 0     # suppress spam from identical rapid events

        # Register on the QApplication once it exists.
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # Wire state signals — connect to all named signals on app_state.
        for sig_name in self._STATE_SIGNAL_NAMES:
            sig: Signal = getattr(self._state, sig_name, None)
            if sig is not None:
                try:
                    sig.connect(self._make_slot(sig_name))
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Signal slot factory
    # ------------------------------------------------------------------
    def _make_slot(self, name: str):
        """Return a callable that logs the signal emission."""
        def _slot(payload: Any) -> None:
            self._emit("signal", name, payload=payload)
        return _slot

    # ------------------------------------------------------------------
    # Qt event filter
    # ------------------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if not self._verbose:
            return False

        etype: QEvent.Type = event.type()
        if etype not in _LOG_EVENT_TYPES:
            return False

        # Only log events on interesting widgets.
        target_class = watched.__class__.__name__
        if target_class not in _INTERESTING_CLASSES:
            return False

        # Rate-limit: max ~20 events/sec to avoid JSON flood.
        now = time.time()
        self._events_seen += 1
        if self._events_seen % 5 != 0:
            return False

        target = _obj_path(watched)
        ename = _EVENT_NAMES.get(etype, etype.name)

        extra: dict[str, Any] = {}
        if etype == QEvent.Type.MouseButtonPress:
            from PySide6.QtGui import QMouseEvent
            if isinstance(event, QMouseEvent):
                extra["button"] = str(event.button().name)
                extra["pos"] = (event.pos().x(), event.pos().y())
        elif etype == QEvent.Type.KeyPress:
            from PySide6.QtGui import QKeyEvent
            if isinstance(event, QKeyEvent):
                extra["key"] = event.key()
                extra["text"] = event.text()

        self._emit("event", ename, target=target, extra=extra)
        return False  # never filter — always let event propagate

    # ------------------------------------------------------------------
    # Public hooks (called by main_window at the relevant call sites)
    # ------------------------------------------------------------------
    def log_tab_switch(self, tab_widget: QTabWidget, index: int) -> None:
        name = tab_widget.tabText(index) if 0 <= index < tab_widget.count() else "?"
        self._emit("tab", name, index=index)

    def log_menu_action(self, text: str) -> None:
        self._emit("menu", text)

    def log_worker_start(self, worker_name: str, args: Any = None) -> None:
        self._emit("worker", "start", detail=worker_name, payload=args)

    def log_worker_done(self, worker_name: str, ok: bool = True) -> None:
        self._emit("worker", "done", detail=worker_name, payload={"ok": ok})

    def log_state_setter(self, prop: str, value_summary: Any) -> None:
        """Called by monkey-patched AppState setters to log write access."""
        self._emit("state", prop, payload=value_summary)

    # ------------------------------------------------------------------
    # Low-level emit
    # ------------------------------------------------------------------
    def _emit(
        self,
        ev_type: str,
        detail: str,
        *,
        index: int = -1,
        target: str = "",
        payload: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        ts = round(time.time() - self._start_time, 4)
        rec: dict[str, Any] = {"ts": ts, "type": ev_type, "detail": detail}
        if index >= 0:
            rec["index"] = index
        if target:
            rec["target"] = target
        if payload is not None:
            rec["payload"] = _payload_summary(payload)
        if extra:
            rec.update(extra)
        try:
            self._out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._out.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Convenience: monkey-patch AppState setters for automatic logging
# ---------------------------------------------------------------------------

def patch_appstate_setters(state: QObject, monitor: DebugMonitor) -> None:
    """Wrap each ``set_*`` method on *state* so it calls
    ``monitor.log_state_setter`` before the real setter."""
    for attr in dir(state):
        if not attr.startswith("set_"):
            continue
        orig = getattr(state, attr)
        if not callable(orig):
            continue

        def _wrap(_orig=orig, _name=attr):
            def _wrapped(*args, **kwargs):
                summary = [_payload_summary(a) for a in args]
                if kwargs:
                    summary.append({k: _payload_summary(v) for k, v in kwargs.items()})
                monitor.log_state_setter(_name, summary)
                return _orig(*args, **kwargs)
            return _wrapped

        try:
            setattr(state, attr, _wrap(orig, attr))
        except Exception:
            pass
