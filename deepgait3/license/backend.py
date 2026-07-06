"""Abstract LicenseBackend + MockBackend (Phase 1 default).

In Phase 4 (W11) the real dongle backends will be added:

* Sentinel HL       → ``sentinel.py``  (kb/17 §3)
* Wibu CodeMeter    → ``codemeter.py`` (kb/17 §4)
* SenseLock (深思)   → ``senselock.py`` (kb/17 §5)

All real backends share the same interface — :class:`LicenseBackend` —
so the rest of the codebase never depends on a specific vendor.

CLOSED-SOURCE NOTE: when the real backends land, this module's
default real-hardware hooks will be hidden behind a Cython-compiled
``_vendor_native.py`` so the dongle API calls cannot be replayed.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LicenseBackendError(RuntimeError):
    """Generic backend failure (USB missing, RPC timeout, bad DLL)."""


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------
class LicenseBackend(ABC):
    """Vendor-neutral abstract interface for hardware dongles.

    A backend exposes just three primitives the rest of the system needs:

    * ``get_dongle_id()`` — stable identifier of the physical dongle.
    * ``get_features_bitmap()`` — 64-bit bitmask of unlocked features.
    * ``execute_in_dongle(name, *args)`` — run a named function *inside*
      the dongle's secure element and return its result. This is how
      we enforce that license verification happens in tamper-resistant
      hardware: the signing key never leaves the dongle.

    Implementations (Phase 4): SentinelBackend, CodeMeterBackend,
    SenseLockBackend. Phase 1 ships only MockBackend.
    """

    @abstractmethod
    def get_dongle_id(self) -> str:
        ...

    @abstractmethod
    def get_features_bitmap(self) -> int:
        ...

    @abstractmethod
    def execute_in_dongle(self, name: str, *args, **kwargs):
        """Run a named protected function inside the dongle.

        The dongle's response must be verifiable by the caller (e.g.
        an HMAC over a known payload) — otherwise this abstraction is
        useless.
        """
        ...


# ---------------------------------------------------------------------------
# Mock backend (Phase 1 default — used in dev/CI/unit tests)
# ---------------------------------------------------------------------------
class MockBackend(LicenseBackend):
    """In-memory license backend used when no dongle is present.

    Configuration is supplied through:

    * ``dongle_id`` / ``features_bitmap`` / ``expiry_ts`` constructor args, or
    * a JSON license file (``license_path``).

    The "execute_in_dongle" hook is implemented as a Python HMAC over
    the input payload — sufficient for unit tests, NOT for production.
    Production systems must use a real backend.
    """

    DEFAULT_LICENSE_FILENAME = "license.json"
    DEFAULT_HMAC_KEY = b"deepgait-mock-hmac-key-DO-NOT-SHIP"

    # Pre-canned protected functions callable from execute_in_dongle.
    SUPPORTED_FUNCS = {
        "verify_signature",  # name, payload_b64, signature_b64
        "sign_payload",      # payload -> signature
        "module_hashes",     # none -> {module: hex_sha256}
    }

    def __init__(
        self,
        dongle_id: str = "MOCK-DONGLE-0000",
        features_bitmap: int = 0xFFFFFFFFFFFFFFFF,
        expiry_ts: int = 0,  # 0 = never expire
        license_path: Optional[Union[str, Path]] = None,
        hmac_key: Optional[bytes] = None,
    ) -> None:
        self._dongle_id = dongle_id
        self._features_bitmap = int(features_bitmap)
        self._expiry_ts = int(expiry_ts)
        self._hmac_key = hmac_key or self.DEFAULT_HMAC_KEY
        self._path = Path(license_path) if license_path else None
        if self._path is not None:
            self._load_from_file(self._path)

    # ---- public API -------------------------------------------------------
    def get_dongle_id(self) -> str:
        return self._dongle_id

    def get_features_bitmap(self) -> int:
        return self._features_bitmap

    def get_expiry_ts(self) -> int:
        return self._expiry_ts

    def execute_in_dongle(self, name: str, *args, **kwargs):
        """Run a named protected function. Raises :class:`LicenseBackendError`
        on unknown name or failed verification."""
        if name not in self.SUPPORTED_FUNCS:
            raise LicenseBackendError(f"unknown dongle function: {name!r}")
        fn = getattr(self, f"_fn_{name}", None)
        if fn is None:
            raise LicenseBackendError(f"dongle function not implemented: {name}")
        return fn(*args, **kwargs)

    # ---- file I/O ---------------------------------------------------------
    def _load_from_file(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            logger.warning("MockBackend: license file missing %s", path)
            return
        except Exception as e:
            raise LicenseBackendError(f"failed to read license file: {e}") from e

        header = raw.get("header", {})
        self._dongle_id = header.get("dongle_id", self._dongle_id)
        self._features_bitmap = int(header.get("features_bitmap",
                                               self._features_bitmap))
        self._expiry_ts = int(header.get("expiry_ts", self._expiry_ts))

    # ---- HMAC-backed "in-dongle" functions -------------------------------
    def _hmac(self, payload: bytes) -> bytes:
        return hmac.new(self._hmac_key, payload, hashlib.sha256).digest()

    def _fn_sign_payload(self, payload: Union[bytes, str]) -> bytes:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return self._hmac(payload)

    def _fn_verify_signature(
        self,
        payload_b64: str,
        signature_b64: str,
    ) -> bool:
        try:
            payload = base64.b64decode(payload_b64.encode("ascii"))
            signature = base64.b64decode(signature_b64.encode("ascii"))
        except Exception:
            return False
        return hmac.compare_digest(self._hmac(payload), signature)

    def _fn_module_hashes(self, *module_paths: str) -> Dict[str, str]:
        """Return SHA-256 hashes for the given module paths (relative to CWD)."""
        out: Dict[str, str] = {}
        for p in module_paths:
            fp = Path(p)
            if fp.is_file():
                out[p] = hashlib.sha256(fp.read_bytes()).hexdigest()
            else:
                out[p] = ""  # missing — IntegrityVerifier flags this
        return out

    # ---- helpers (for license issuance in scripts/) ----------------------
    @staticmethod
    def make_license_payload(
        customer_id: str,
        features_bitmap: int,
        expiry_ts: int = 0,
    ) -> dict:
        """Return the *unsigned* body of a license file."""
        return {
            "header": {
                "version": "2.0",
                "customer_id": customer_id,
                "dongle_id": f"DG-{uuid.uuid4().hex[:8].upper()}",
                "features_bitmap": int(features_bitmap),
                "expiry_ts": int(expiry_ts),
                "issued_at": int(time.time()),
            },
            "body": {"schema_version": "2.0"},
        }

    @classmethod
    def issue_license(
        cls,
        customer_id: str,
        features_bitmap: int = 0xFFFFFFFFFFFFFFFF,
        expiry_ts: int = 0,
        out_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate a signed license file (mock signing)."""
        payload = cls.make_license_payload(customer_id, features_bitmap, expiry_ts)
        body_bytes = json.dumps(payload["body"], separators=(",", ":")).encode()
        sig = cls("")._fn_sign_payload(body_bytes)
        out = {
            **payload,
            "signature_b64": base64.b64encode(sig).decode("ascii"),
            "payload_b64": base64.b64encode(body_bytes).decode("ascii"),
        }
        path = Path(out_path or cls.DEFAULT_LICENSE_FILENAME)
        path.write_text(json.dumps(out, indent=2))
        return path

    def __repr__(self) -> str:
        return (f"MockBackend(dongle_id={self._dongle_id!r}, "
                f"features=0x{self._features_bitmap:016x}, "
                f"expiry={self._expiry_ts or 'never'})")