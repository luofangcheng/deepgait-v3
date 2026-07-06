"""License subpackage — Layer 3 of the deepgait v2 architecture.

Modules
-------
* ``backend``     — abstract ``LicenseBackend`` + ``MockBackend`` (CI/dev)
* ``verifier``    — ``LicenseVerifier`` (file + signature + feature check)
* ``heartbeat``   — ``LicenseHeartbeat`` (anti-offline timer + lost callback)

This package follows MODULES.md §3 + kb/19_license_management.md.
The dongle-specific backends (Sentinel HL, CodeMeter, SenseLock) are
*not* implemented in Phase 1 — they will be added in Phase 4 once the
final vendor is selected. Phase 1 ships the ``MockBackend`` so the
application can run end-to-end without a dongle.

CLOSED-SOURCE NOTE: in production these modules are compiled by
Cython (see kb/18_cython_nuitka.md); the public API intentionally hides
all internal signing keys and backends.
"""
from .backend import LicenseBackend, MockBackend, LicenseBackendError
from .verifier import (
    LicenseVerifier,
    LicenseInfo,
    LicenseStatus,
    LicenseError,
    LicenseFeatureLockedError,
    LicenseExpiredError,
    LicenseSignatureError,
    LicenseDongleMismatchError,
    FEATURE_BITS,
    bitmap_for_features,
    features_for_bitmap,
)
from .heartbeat import LicenseHeartbeat

__all__ = [
    "LicenseBackend",
    "MockBackend",
    "LicenseBackendError",
    "LicenseVerifier",
    "LicenseInfo",
    "LicenseStatus",
    "LicenseError",
    "LicenseFeatureLockedError",
    "LicenseExpiredError",
    "LicenseSignatureError",
    "LicenseDongleMismatchError",
    "FEATURE_BITS",
    "bitmap_for_features",
    "features_for_bitmap",
    "LicenseHeartbeat",
]