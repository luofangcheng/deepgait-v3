"""Compatibility shim for Cython-compiled modules with hard-coded ``deepgait.*`` paths.

The deep-gailt Layer 3-4 modules (gait_algorithms, footprint_v2,
triangulation_3d, license.*, security.*) were compiled against the
top-level ``deepgait`` package. After restructuring the compiled ``.so``
objects still ``import deepgait.core.*`` strings at load time.

This module creates synthetic namespace packages so those imports resolve
to the new flat layout::

    deepgait.core           -> deepgait3.core._legacy
    deepgait.core.license   -> deepgait3.license
    deepgait.core.security  -> deepgait3.security
    deepgait.gui            -> deepgait3.gui
    deepgait.hardware       -> deepgait3.hardware
    deepgait.io             -> deepgait3.io
    deepgait.utils          -> deepgait3.utils
"""
from __future__ import annotations

import importlib
import sys
import types


# ---------------------------------------------------------------------------
# Intermediate namespace nodes that the .so loader traverses.
# The compiled modules do ``import deepgait.core.footprint``, which
# traverses ``deepgait`` -> ``deepgait.core`` before hitting the leaf.
# We must create synthetic modules for each intermediate node.
# ---------------------------------------------------------------------------
_INTERMEDIATE_NAMESPACES = [
    "deepgait",
]

# ---------------------------------------------------------------------------
# Leaf mappings: legacy top-level name -> new deepgait3 location.
#
# ``deepgait.core.license`` and ``deepgait.core.security`` are NOT listed
# because they resolve as real subdirectories under ``core/_legacy/`` via
# symlinks pointing to ``deepgait3/license/`` and ``deepgait3/security/``.
# ---------------------------------------------------------------------------
_LEGACY_MAP = {
    "deepgait.core": "deepgait3.core._legacy",
    "deepgait.gui": "deepgait3.gui",
    "deepgait.hardware": "deepgait3.hardware",
    "deepgait.io": "deepgait3.io",
    "deepgait.utils": "deepgait3.utils",
}


def _ensure_namespace(name: str) -> None:
    """Create an empty namespace package if it does not exist."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []       # namespace packages need __path__
        mod.__package__ = name
        sys.modules[name] = mod


def install_legacy_aliases() -> None:
    """Register ``deepgait.*`` in sys.modules pointing at the migrated tree.

    Idempotent: if ``deepgait`` is already in :data:`sys.modules` we do
    nothing — calling this twice is harmless.
    """
    if "deepgait" in sys.modules:
        return

    # 1. Create synthetic intermediate namespace packages.
    for ns in _INTERMEDIATE_NAMESPACES:
        _ensure_namespace(ns)

    # 2. Import each new-location package and ALIAS the legacy name to it.
    #    Use direct assignment (not setdefault) because some names may have
    #    been pre-created as empty intermediate namespaces in step 1.
    for legacy_name, new_name in _LEGACY_MAP.items():
        try:
            importlib.import_module(new_name)
            sys.modules[legacy_name] = sys.modules[new_name]
        except Exception:  # pragma: no cover - best-effort
            continue

    # 3. Wire up parent module attributes so ``import deepgait.core`` works.
    #    ``sys.modules`` lookups are enough for ``__import__``, but direct
    #    attribute access (``deepgait.core``) requires the parent module to
    #    have a matching attribute.
    for legacy_name, new_name in _LEGACY_MAP.items():
        if "." not in legacy_name:
            continue
        parent_name, _, child_name = legacy_name.rpartition(".")
        parent_mod = sys.modules.get(parent_name)
        child_mod = sys.modules.get(legacy_name)
        if parent_mod is not None and child_mod is not None:
            setattr(parent_mod, child_name, child_mod)
