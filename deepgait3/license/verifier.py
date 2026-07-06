"""License verification — Layer 3 of the deepgait v2 architecture.

:class:`LicenseVerifier` is the *only* object the rest of the codebase
is allowed to import for license decisions. It:

* Loads ``license.json`` (or a custom path) and verifies the dongle id.
* Asks the dongle to verify the body signature (in real hardware this
  happens inside the tamper-resistant element).
* Caches the verification result and exposes a feature bitmask query.
* Starts a heartbeat (anti-offline) and invokes a user-supplied
  callback when the license becomes invalid.

Closed-source: this module ships as Cython-compiled bytes in production
(kb/18 §2.3). The Python source here is the **debug build** used by
Phase 1 unit tests and the dev/CI environment.
"""
from __future__ import annotations

import base64
import enum
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .backend import LicenseBackend, LicenseBackendError


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class LicenseError(RuntimeError):
    """Generic license failure."""


class LicenseSignatureError(LicenseError):
    """The license body failed HMAC verification (tampered?)."""


class LicenseExpiredError(LicenseError):
    """License's expiry timestamp is in the past."""


class LicenseFeatureLockedError(LicenseError):
    """A requested feature is not enabled in the license bitmap."""

    def __init__(self, feature: str):
        super().__init__(f"feature not enabled: {feature!r}")
        self.feature = feature


class LicenseDongleMismatchError(LicenseError):
    """The license file is bound to a different dongle than the one present."""


# ---------------------------------------------------------------------------
# LicenseInfo
# ---------------------------------------------------------------------------
@dataclass
class LicenseInfo:
    """Decoded license metadata exposed via :meth:`LicenseVerifier.get_info`."""
    dongle_id: str = ""
    customer_id: str = ""
    features_bitmap: int = 0
    expiry_ts: int = 0
    issued_at: int = 0
    features: List[str] = field(default_factory=list)

    def is_expired(self, now_ts: Optional[int] = None) -> bool:
        if self.expiry_ts <= 0:
            return False
        return (now_ts or int(time.time())) >= self.expiry_ts

    def has_feature(self, name: str) -> bool:
        return name in self.features

    def to_dict(self) -> dict:
        return {
            "dongle_id": self.dongle_id,
            "customer_id": self.customer_id,
            "features_bitmap": self.features_bitmap,
            "expiry_ts": self.expiry_ts,
            "issued_at": self.issued_at,
            "features": list(self.features),
            "expired": self.is_expired(),
        }


class LicenseStatus(str, enum.Enum):
    """Coarse license lifecycle state."""
    INVALID = "invalid"
    VALID = "valid"
    EXPIRED = "expired"
    TRIAL = "trial"
    TAMPERED = "tampered"


# ---------------------------------------------------------------------------
# Feature name <-> bitmap map
# ---------------------------------------------------------------------------
# Bit assignments are stable: changing them breaks issued licenses.
FEATURE_BITS: Dict[str, int] = {
    "BASE":         1 << 0,
    "GAIT_2D":      1 << 1,
    "GAIT_3D":      1 << 2,
    "FTIR":         1 << 3,
    "DLC_TRAIN":    1 << 4,
    "DLC_INFER":    1 << 5,
    "BATCH_EXPORT": 1 << 6,
    "CUSTOM_PIPELINE": 1 << 7,
    "API":          1 << 8,
    # Bits 9..63 reserved for future expansion.
}


def bitmap_for_features(features: List[str]) -> int:
    """Compute the bitmap for a list of feature names."""
    bm = 0
    for f in features:
        if f not in FEATURE_BITS:
            raise LicenseError(f"unknown feature: {f!r}")
        bm |= FEATURE_BITS[f]
    return bm


def features_for_bitmap(bitmap: int) -> List[str]:
    """Return the feature names enabled by a bitmap (canonical order)."""
    return [name for name, bit in FEATURE_BITS.items() if bitmap & bit]


# ---------------------------------------------------------------------------
# LicenseVerifier
# ---------------------------------------------------------------------------
class LicenseVerifier:
    """One-stop license verification, feature gating, and dongle glue."""

    DEFAULT_LICENSE_FILENAME = "license.json"

    def __init__(
        self,
        backend: LicenseBackend,
        license_path: Optional[Union[str, Path]] = None,
        *,
        trial_mode: bool = False,
    ) -> None:
        self._backend = backend
        self._path = Path(license_path) if license_path else None
        self._trial_mode = bool(trial_mode)
        self._info = LicenseInfo()
        self._status = LicenseStatus.INVALID
        self._locked_callbacks: List[Callable[[], None]] = []

    # ---- public API -------------------------------------------------------
    def verify(self) -> LicenseStatus:
        """Validate the license (file + dongle signature + expiry)."""
        # Trial mode short-circuits: only the BASE feature is enabled and
        # the dongle is not consulted.
        if self._trial_mode:
            self._info = LicenseInfo(
                dongle_id="TRIAL-MODE",
                customer_id="trial-user",
                features_bitmap=FEATURE_BITS["BASE"],
                expiry_ts=0,
                issued_at=int(time.time()),
                features=["BASE"],
            )
            self._status = LicenseStatus.TRIAL
            return self._status

        # 1. Read license file (if path provided).
        try:
            data = self._load_license_file()
        except FileNotFoundError:
            logger.warning(
                "LicenseVerifier: no license file at %s",
                self._path or self.DEFAULT_LICENSE_FILENAME,
            )
            data = None
        except json.JSONDecodeError as e:
            logger.error("LicenseVerifier: license JSON parse error: %s", e)
            data = None

        if data is None:
            self._status = LicenseStatus.INVALID
            return self._status

        # 2. Match dongle id.
        header = data.get("header", {})
        expected_dongle_id = header.get("dongle_id", "")
        actual_dongle_id = self._backend.get_dongle_id()
        if expected_dongle_id and expected_dongle_id != actual_dongle_id:
            raise LicenseDongleMismatchError(
                f"license bound to dongle {expected_dongle_id!r}, "
                f"but present dongle is {actual_dongle_id!r}"
            )

        # 3. Verify signature INSIDE the dongle (HMAC in mock).
        payload_b64 = data.get("payload_b64", "")
        signature_b64 = data.get("signature_b64", "")
        if not (payload_b64 and signature_b64):
            raise LicenseSignatureError("license missing payload/signature")
        try:
            ok = bool(self._backend.execute_in_dongle(
                "verify_signature", payload_b64, signature_b64,
            ))
        except LicenseBackendError as e:
            raise LicenseSignatureError(f"dongle verify failed: {e}") from e
        if not ok:
            self._status = LicenseStatus.TAMPERED
            raise LicenseSignatureError("license signature invalid (tampered?)")

        # 4. Decode + cache info.
        try:
            body_bytes = base64.b64decode(payload_b64.encode("ascii"))
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception as e:
            raise LicenseError(f"license body decode failed: {e}") from e

        bitmap = int(header.get("features_bitmap", 0))
        self._info = LicenseInfo(
            dongle_id=expected_dongle_id or actual_dongle_id,
            customer_id=header.get("customer_id", ""),
            features_bitmap=bitmap,
            expiry_ts=int(header.get("expiry_ts", 0)),
            issued_at=int(header.get("issued_at", 0)),
            features=features_for_bitmap(bitmap),
        )
        # 5. Check expiry.
        if self._info.is_expired():
            self._status = LicenseStatus.EXPIRED
            raise LicenseExpiredError(
                f"license expired at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._info.expiry_ts))}"
            )

        self._status = LicenseStatus.VALID
        return self._status

    def is_valid(self) -> bool:
        """Cheap check: do NOT re-verify the signature. Cached status only."""
        return self._status in (LicenseStatus.VALID, LicenseStatus.TRIAL)

    def check_feature(self, feature: str) -> bool:
        if not self.is_valid():
            return False
        if feature not in FEATURE_BITS:
            raise LicenseError(f"unknown feature: {feature!r}")
        return bool(self._info.features_bitmap & FEATURE_BITS[feature])

    def require_feature(self, feature: str) -> None:
        """Raise :class:`LicenseFeatureLockedError` if the feature is locked."""
        if not self.check_feature(feature):
            raise LicenseFeatureLockedError(feature)

    def get_info(self) -> LicenseInfo:
        return self._info

    def get_status(self) -> LicenseStatus:
        return self._status

    # ---- loss callbacks ---------------------------------------------------
    def on_lost(self, callback: Callable[[], None]) -> None:
        """Register a callback fired when the license becomes invalid."""
        self._locked_callbacks.append(callback)

    def _notify_lost(self) -> None:
        for cb in self._locked_callbacks:
            try:
                cb()
            except Exception:
                logger.exception("LicenseVerifier: on_lost callback raised")

    # ---- internal ---------------------------------------------------------
    def _load_license_file(self) -> Optional[dict]:
        path = self._path or (Path.cwd() / self.DEFAULT_LICENSE_FILENAME)
        if not path.is_file():
            raise FileNotFoundError(path)
        return json.loads(path.read_text())

    def __repr__(self) -> str:
        return (f"LicenseVerifier(status={self._status.value}, "
                f"dongle={self._backend.get_dongle_id()!r})")