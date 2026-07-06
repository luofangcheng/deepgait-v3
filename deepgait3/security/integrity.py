"""Module integrity verification (Layer 3, security).

Compares the SHA-256 of every critical module against a known-good
hash list supplied by the dongle (or a pinned build manifest). Any
mismatch means the bytecode has been patched by an attacker.

References
----------
* kb/20_security_anti_tamper.md §3
* MODULES.md §3 "IntegrityVerifier"
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from ..license.backend import LicenseBackend


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class IntegrityError(RuntimeError):
    """Generic integrity check failure."""


class ModuleHashMismatch(IntegrityError):
    """One or more modules failed the hash check."""

    def __init__(self, mismatches: Dict[str, str]):
        self.mismatches = dict(mismatches)
        super().__init__(f"{len(mismatches)} module(s) failed integrity check")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_module_hashes(modules: Iterable[Union[str, Path]]) -> Dict[str, str]:
    """Compute SHA-256 (hex) for each module path. Missing files map to ""."""
    out: Dict[str, str] = {}
    for m in modules:
        p = Path(m)
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        else:
            out[str(p)] = ""
    return out


# ---------------------------------------------------------------------------
# IntegrityVerifier
# ---------------------------------------------------------------------------
class IntegrityVerifier:
    """Critical-module hash verifier.

    Parameters
    ----------
    backend : LicenseBackend
        Backend used to fetch the pinned hash list from inside the
        dongle (``execute_in_dongle("module_hashes", *paths)``).
    modules : list of paths
        Modules to verify (paths relative to CWD or absolute).
    baseline : dict, optional
        Pre-pinned baseline. If provided, overrides the dongle query —
        useful for offline tests and for the `manifest_hash.json`
        generated at build time (see kb/20 §3.4).
    """

    def __init__(
        self,
        backend: Optional[LicenseBackend] = None,
        modules: Optional[List[Union[str, Path]]] = None,
        baseline: Optional[Dict[str, str]] = None,
    ) -> None:
        self._backend = backend
        self._modules = [str(m) for m in (modules or [])]
        self._baseline = dict(baseline) if baseline else None

    # ---- public API -------------------------------------------------------
    def verify(self) -> bool:
        """Verify every module against the baseline.

        Returns ``True`` if all hashes match. Raises
        :class:`ModuleHashMismatch` otherwise. Returns ``True`` if no
        baseline is available (skipped).
        """
        baseline = self._baseline
        if baseline is None and self._backend is not None:
            try:
                baseline = dict(self._backend.execute_in_dongle(
                    "module_hashes", *self._modules,
                ) or {})
            except Exception as e:
                logger.warning(
                    "IntegrityVerifier: dongle hash fetch failed (%s); skipping",
                    e,
                )
                return True
        if not baseline:
            return True

        current = compute_module_hashes(self._baseline_basemodules(baseline))
        mismatches: Dict[str, Dict[str, str]] = {}
        for path, expected in baseline.items():
            actual = current.get(path, "")
            if actual != expected:
                mismatches[path] = {"expected": expected, "actual": actual}
        if mismatches:
            raise ModuleHashMismatch(mismatches)
        return True

    def refresh_baseline(self, modules: Optional[List[Union[str, Path]]] = None) -> Dict[str, str]:
        """Compute a fresh baseline from the on-disk modules."""
        mods = [str(m) for m in (modules or self._modules)]
        self._baseline = compute_module_hashes(mods)
        return dict(self._baseline)

    def save_baseline(self, path: Union[str, Path]) -> None:
        if self._baseline is None:
            raise IntegrityError("no baseline loaded")
        Path(path).write_text(
            "\n".join(f"{k}\t{v}" for k, v in sorted(self._baseline.items())),
        )

    def load_baseline(self, path: Union[str, Path]) -> None:
        out: Dict[str, str] = {}
        for line in Path(path).read_text().splitlines():
            if "\t" in line:
                k, v = line.split("\t", 1)
                out[k.strip()] = v.strip()
        self._baseline = out

    # ---- internal ---------------------------------------------------------
    def _baseline_basemodules(self, baseline: Dict[str, str]) -> List[str]:
        # Hash the same set of paths the baseline was built from.
        return list(baseline.keys())

    def __repr__(self) -> str:
        n = len(self._baseline) if self._baseline else 0
        return f"IntegrityVerifier(modules={len(self._modules)}, baseline={n})"