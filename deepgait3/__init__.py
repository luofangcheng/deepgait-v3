"""DeepGait v3 — fTIR + 3D pose + CatWalk-compatible gait analysis platform.

This package merges the v2.0 GUI/I-O/hardware framework (from ``deep-gailt/``)
with the v4.x algorithm modules (from ``deepgait-v2/``) into a single,
modern codebase.

Importing ``deepgait3`` also installs a compatibility shim so legacy
``deepgait.*`` import statements (used inside Cython-compiled
``.so`` modules from ``deep-gailt/``) resolve through the merged tree.
"""
from __future__ import annotations

from ._legacy_shim import install_legacy_aliases as _install_aliases

__version__ = "0.1.0.dev0"

try:
    _install_aliases()
except Exception:
    pass
