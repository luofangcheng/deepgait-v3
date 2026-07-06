"""DLC subpackage — Layer 1 (hardware adapter).

deepgait v2 delegates all DeepLabCut work to a separate conda
environment (kb/12 §11) so the GPU + PyTorch + DLC stack can evolve
independently from the closed-source GUI / business logic. This package
provides the in-process side of that boundary:

* :class:`DLCSubprocessRunner` — spawns `conda run -n dlc …` subprocesses
  and parses structured progress events from stdout.
* :class:`MockDLCSubprocessRunner` — same API, no subprocess, used by
  the unit tests so the "DLC subprocess skeleton" acceptance gate can
  be verified on CI without a real conda env or GPU.
* :class:`AsyncDLCSubprocessRunner` — wraps any runner in a worker
  thread so the PySide6 GUI never blocks on DLC.

IPC contract
------------
The DLC subprocess emits one JSON object per line on stdout, e.g.::

    {"event": "progress", "stage": "training", "epoch": 5, "total_epochs": 200, "loss": 0.012}
    {"event": "progress", "stage": "analyze", "frame": 1234, "total_frames": 5000}
    {"event": "result", "config_path": "...", "metrics": {"train_rmse": 2.4}}
    {"event": "error", "message": "..."}

Non-JSON lines (DLC's normal logging) are forwarded to the optional
``log_callback`` so the GUI / loguru handlers can show them.
"""
from .subprocess_runner import (
    DLCProgress,
    DLCResult,
    DLCError,
    ProgressCallback,
    ResultCallback,
    ErrorCallback,
    LogCallback,
    _BaseRunner,
    DLCSubprocessRunner,
    MockDLCSubprocessRunner,
    AsyncDLCSubprocessRunner,
)

__all__ = [
    "DLCProgress",
    "DLCResult",
    "DLCError",
    "ProgressCallback",
    "ResultCallback",
    "ErrorCallback",
    "LogCallback",
    "_BaseRunner",
    "DLCSubprocessRunner",
    "MockDLCSubprocessRunner",
    "AsyncDLCSubprocessRunner",
]