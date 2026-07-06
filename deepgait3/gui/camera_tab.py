"""Backward-compat shim: the old ``CameraTab`` has been renamed to
:class:`InitializationTab` and moved to ``initialization_tab.py``.

This shim keeps ``from deepgait3.gui.camera_tab import CameraTab``
working for external code and the existing test suite, while the
actual implementation lives in :mod:`deepgait3.gui.initialization_tab`.

W17: tab renamed "相机采集" → "初始化" and reordered to position 1
(moved from position 8). See ``docs/CHANGELOG.md`` 2.0.1 entry.
"""
from __future__ import annotations

from .initialization_tab import (
    CameraConfigGroup,
    InitializationTab,
    LivePreviewPanel,
    MultiCameraConfigPanel,
    CameraTab,                       # explicit alias for clarity
    DEFAULT_ROLES,
    ROLE_LABELS,
    _preset_dir,
    _save_preset,
    _load_preset,
)

__all__ = [
    "CameraTab",
    "InitializationTab",
    "MultiCameraConfigPanel",
    "LivePreviewPanel",
    "CameraConfigGroup",
    "DEFAULT_ROLES",
    "ROLE_LABELS",
    "_preset_dir",
    "_save_preset",
    "_load_preset",
]
