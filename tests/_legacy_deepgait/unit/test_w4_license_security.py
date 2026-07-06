"""Unit tests for Phase 1 W4 — Layer 3 License / Security.

Covers:
    deepgait/core/license/backend.py     — LicenseBackend + MockBackend
    deepgait/core/license/verifier.py    — LicenseVerifier (verify/feature/expiry)
    deepgait/core/license/heartbeat.py   — LicenseHeartbeat (background thread)
    deepgait/core/security/anti_debug.py — AntiDebugProbe + helpers
    deepgait/core/security/integrity.py  — IntegrityVerifier
    deepgait/core/security/tamper.py     — TamperPolicy + respond_to_detection

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.1 W4):
    "Encryption dongle skeleton runnable"
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# =============================================================================
# MockBackend
# =============================================================================
class TestMockBackend:
    def test_default_construction(self):
        from deepgait3.license import MockBackend

        b = MockBackend()
        assert b.get_dongle_id() == "MOCK-DONGLE-0000"
        assert b.get_features_bitmap() == 0xFFFFFFFFFFFFFFFF
        assert b.get_expiry_ts() == 0

    def test_load_license_from_file(self, tmp_path):
        from deepgait3.license import MockBackend

        # Use the bundled issue_license helper to make a known-good file.
        license_path = MockBackend.issue_license(
            customer_id="lab42",
            features_bitmap=0xFF,
            expiry_ts=0,
            out_path=tmp_path / "license.json",
        )
        assert license_path.is_file()
        b = MockBackend(license_path=license_path)
        # The dongle_id should now match the one written by issue_license.
        assert b.get_dongle_id().startswith("DG-")
        assert b.get_features_bitmap() == 0xFF

    def test_execute_in_dongle_sign_and_verify_round_trip(self):
        from deepgait3.license import MockBackend

        b = MockBackend()
        payload = b"hello-deepgait"
        sig = b.execute_in_dongle("sign_payload", payload)
        ok = b.execute_in_dongle(
            "verify_signature",
            __import__("base64").b64encode(payload).decode(),
            __import__("base64").b64encode(sig).decode(),
        )
        assert ok is True

    def test_execute_unknown_function_raises(self):
        from deepgait3.license import MockBackend, LicenseBackendError

        b = MockBackend()
        with pytest.raises(LicenseBackendError):
            b.execute_in_dongle("obviously_not_a_real_op")

    def test_execute_module_hashes(self, tmp_path):
        from deepgait3.license import MockBackend

        a = tmp_path / "a.py"; a.write_text("print('a')\n")
        b = tmp_path / "b.py"; b.write_text("print('b')\n")
        out = MockBackend().execute_in_dongle("module_hashes", str(a), str(b))
        assert len(out) == 2
        assert out[str(a)] and out[str(b)]


# =============================================================================
# LicenseVerifier
# =============================================================================
class TestLicenseVerifier:
    def _issued(self, tmp_path, **kwargs):
        from deepgait3.license import MockBackend

        return MockBackend.issue_license(out_path=tmp_path / "license.json",
                                          **kwargs)

    def test_verify_valid_license_returns_valid_status(self, tmp_path):
        from deepgait3.license import (
            LicenseStatus, LicenseVerifier, MockBackend,
        )

        path = self._issued(tmp_path, customer_id="lab42",
                            features_bitmap=0x0F, expiry_ts=0)
        # The backend reads the license_path on construction and takes its
        # dongle_id from there. So we just point the backend at the file.
        backend = MockBackend(license_path=path)
        verifier = LicenseVerifier(backend, license_path=path)
        status = verifier.verify()
        assert status == LicenseStatus.VALID
        assert verifier.is_valid() is True

    def test_verify_expired_license_raises(self, tmp_path):
        from deepgait3.license import (
            LicenseExpiredError, LicenseVerifier, LicenseStatus,
            MockBackend,
        )

        # Issue an already-expired license.
        path = MockBackend.issue_license(
            customer_id="lab42", features_bitmap=0x0F,
            expiry_ts=int(time.time()) - 10,  # 10 s in the past
            out_path=tmp_path / "license.json",
        )
        backend = MockBackend(license_path=path)
        verifier = LicenseVerifier(backend, license_path=path)
        with pytest.raises(LicenseExpiredError):
            verifier.verify()
        assert verifier.get_status() == LicenseStatus.EXPIRED

    def test_verify_dongle_mismatch_raises(self, tmp_path):
        from deepgait3.license import LicenseDongleMismatchError
        from deepgait3.license.verifier import LicenseVerifier
        from deepgait3.license import MockBackend

        path = self._issued(tmp_path, customer_id="lab42",
                            features_bitmap=0x0F, expiry_ts=0)
        # Use a backend with a DIFFERENT dongle id and NO license_path,
        # so its dongle_id stays at the constructor-supplied value.
        backend = MockBackend(dongle_id="WRONG-DONGLE-9999")
        verifier = LicenseVerifier(backend, license_path=path)
        with pytest.raises(LicenseDongleMismatchError):
            verifier.verify()

    def test_verify_tampered_signature_raises(self, tmp_path):
        from deepgait3.license import LicenseSignatureError
        from deepgait3.license.verifier import LicenseVerifier
        from deepgait3.license import MockBackend

        path = self._issued(tmp_path, customer_id="lab42",
                            features_bitmap=0x0F, expiry_ts=0)
        # Tamper with the signature field.
        with open(path) as f:
            data = json.load(f)
        data["signature_b64"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(data))
        # Backend reads the tampered file → matches dongle id → but
        # signature fails → tamper detected.
        backend = MockBackend(license_path=tampered)
        verifier = LicenseVerifier(backend, license_path=tampered)
        with pytest.raises(LicenseSignatureError):
            verifier.verify()

    def test_check_feature_and_require_feature(self, tmp_path):
        from deepgait3.license import (
            LicenseFeatureLockedError, LicenseVerifier, MockBackend,
            bitmap_for_features,
        )

        path = self._issued(tmp_path, customer_id="lab42",
                            features_bitmap=bitmap_for_features(["BASE", "GAIT_2D"]),
                            expiry_ts=0)
        backend = MockBackend(license_path=path)
        verifier = LicenseVerifier(backend, license_path=path)
        verifier.verify()
        assert verifier.check_feature("BASE") is True
        assert verifier.check_feature("GAIT_2D") is True
        assert verifier.check_feature("GAIT_3D") is False
        with pytest.raises(LicenseFeatureLockedError):
            verifier.require_feature("GAIT_3D")

    def test_trial_mode_unlocks_base_only(self):
        from deepgait3.license import LicenseStatus, LicenseVerifier, MockBackend

        verifier = LicenseVerifier(MockBackend(), trial_mode=True)
        status = verifier.verify()
        assert status == LicenseStatus.TRIAL
        assert verifier.is_valid() is True
        assert verifier.check_feature("BASE") is True
        assert verifier.check_feature("GAIT_3D") is False

    def test_missing_license_file_marks_invalid(self, tmp_path):
        from deepgait3.license import LicenseStatus, LicenseVerifier, MockBackend

        verifier = LicenseVerifier(
            MockBackend(), license_path=tmp_path / "nope.json",
        )
        status = verifier.verify()
        assert status == LicenseStatus.INVALID
        assert verifier.is_valid() is False

    def test_get_info_returns_decoded_payload(self, tmp_path):
        from deepgait3.license import LicenseVerifier, MockBackend

        path = self._issued(tmp_path, customer_id="lab42",
                            features_bitmap=0x0F, expiry_ts=0)
        backend = MockBackend(license_path=path)
        verifier = LicenseVerifier(backend, license_path=path)
        verifier.verify()
        info = verifier.get_info()
        assert info.customer_id == "lab42"
        assert set(info.features) >= {"BASE", "GAIT_2D"}
        assert info.to_dict()["customer_id"] == "lab42"

    def test_on_lost_callbacks_fire(self, tmp_path):
        from deepgait3.license import LicenseVerifier, MockBackend

        seen = []
        verifier = LicenseVerifier(MockBackend())
        verifier.on_lost(lambda: seen.append(1))
        verifier.on_lost(lambda: seen.append(2))
        verifier._notify_lost()  # type: ignore[attr-defined]
        assert seen == [1, 2]


# =============================================================================
# LicenseHeartbeat
# =============================================================================
class TestLicenseHeartbeat:
    def test_valid_license_stays_alive(self, tmp_path):
        from deepgait3.license import (
            LicenseHeartbeat, LicenseVerifier, MockBackend,
        )

        path = MockBackend.issue_license(
            customer_id="lab42", features_bitmap=0xFF,
            expiry_ts=0, out_path=tmp_path / "license.json",
        )
        backend = MockBackend(license_path=path)
        verifier = LicenseVerifier(backend, license_path=path)
        hb = LicenseHeartbeat(verifier, interval_s=0.1)
        with hb:
            time.sleep(0.3)
            assert hb.is_alive() is True
            assert hb.get_status()["misses"] == 0

    def test_lost_callback_fires_after_misses(self, tmp_path):
        from deepgait3.license import (
            LicenseHeartbeat, LicenseVerifier, MockBackend,
        )

        # A license file that fails HMAC verification (signature corrupted)
        # → verify() raises every time → 2 misses → on_lost fires.
        path = MockBackend.issue_license(
            customer_id="lab42", features_bitmap=0xFF,
            expiry_ts=0, out_path=tmp_path / "license.json",
        )
        with open(path) as f:
            data = json.load(f)
        data["signature_b64"] = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
        tampered = tmp_path / "tampered.json"
        tampered.write_text(json.dumps(data))
        backend = MockBackend(license_path=tampered)
        verifier = LicenseVerifier(backend, license_path=tampered)
        seen = []
        hb = LicenseHeartbeat(
            verifier, interval_s=0.05, on_lost=lambda: seen.append(1),
        )
        with hb:
            time.sleep(0.5)
        assert seen, "expected on_lost callback to fire after misses"

    def test_invalid_interval_raises(self):
        from deepgait3.license import LicenseHeartbeat, LicenseVerifier, MockBackend

        with pytest.raises(ValueError):
            LicenseHeartbeat(LicenseVerifier(MockBackend()), interval_s=0)
        with pytest.raises(ValueError):
            LicenseHeartbeat(LicenseVerifier(MockBackend()), interval_s=-1)


# =============================================================================
# AntiDebugProbe (lightweight tests — no actual debugger on CI)
# =============================================================================
class TestAntiDebug:
    def test_probe_runs_returns_report(self):
        from deepgait3.security import AntiDebugProbe, AntiDebugReport

        rep = AntiDebugProbe(deep=False).run()
        assert isinstance(rep, AntiDebugReport)
        assert "platform" in rep.raw_signals
        assert "pid" in rep.raw_signals

    def test_probe_on_clean_machine_is_clean(self):
        from deepgait3.security import AntiDebugProbe

        rep = AntiDebugProbe(deep=True).run()
        # CI / dev workstation must report no debugger. (If this fails, the
        # CI environment is contaminated — that's the point of the probe.)
        assert rep.debugger_attached is False, rep.to_dict()

    def test_helpers_dont_raise(self):
        from deepgait3.security import (
            is_debugger_attached, is_frida_present,
            is_virtual_machine, is_being_analyzed,
        )
        # Each must return a bool without raising.
        assert isinstance(is_debugger_attached(), bool)
        assert isinstance(is_frida_present(), bool)
        assert isinstance(is_virtual_machine(), bool)
        assert isinstance(is_being_analyzed(), bool)

    def test_legacy_respond_to_debug_detection_runs(self, caplog):
        from deepgait3.security import respond_to_debug_detection

        with caplog.at_level(logging.WARNING):
            # In CI / test env, severity "warning" → log-only action.
            respond_to_debug_detection(severity="warning")


# =============================================================================
# IntegrityVerifier
# =============================================================================
class TestIntegrityVerifier:
    def test_compute_module_hashes_round_trip(self, tmp_path):
        from deepgait3.security import compute_module_hashes

        a = tmp_path / "a.py"; a.write_text("x = 1\n")
        b = tmp_path / "b.py"; b.write_text("y = 2\n")
        h = compute_module_hashes([a, b])
        assert len(h) == 2
        # Recomputing must give the same values.
        h2 = compute_module_hashes([a, b])
        assert h == h2

    def test_refresh_baseline_captures_current_hashes(self, tmp_path):
        from deepgait3.security import IntegrityVerifier

        a = tmp_path / "a.py"; a.write_text("x = 1\n")
        b = tmp_path / "b.py"; b.write_text("y = 2\n")
        v = IntegrityVerifier(modules=[a, b])
        baseline = v.refresh_baseline()
        assert set(baseline.keys()) == {str(a), str(b)}

    def test_verify_passes_when_baseline_matches(self, tmp_path):
        from deepgait3.security import IntegrityVerifier

        a = tmp_path / "a.py"; a.write_text("x = 1\n")
        b = tmp_path / "b.py"; b.write_text("y = 2\n")
        v = IntegrityVerifier(modules=[a, b])
        v.refresh_baseline()
        assert v.verify() is True

    def test_verify_detects_tampered_module(self, tmp_path):
        from deepgait3.security import IntegrityVerifier, ModuleHashMismatch

        a = tmp_path / "a.py"; a.write_text("x = 1\n")
        b = tmp_path / "b.py"; b.write_text("y = 2\n")
        v = IntegrityVerifier(modules=[a, b])
        v.refresh_baseline()
        # Tamper with one module.
        b.write_text("# attacker payload\ny = 999\n")
        with pytest.raises(ModuleHashMismatch) as exc_info:
            v.verify()
        assert str(b) in exc_info.value.mismatches

    def test_verify_skipped_when_no_baseline_no_backend(self):
        from deepgait3.security import IntegrityVerifier

        v = IntegrityVerifier()  # no backend, no baseline → skip
        assert v.verify() is True

    def test_baseline_save_and_load(self, tmp_path):
        from deepgait3.security import IntegrityVerifier

        a = tmp_path / "a.py"; a.write_text("x = 1\n")
        v = IntegrityVerifier(modules=[a])
        v.refresh_baseline()
        manifest = tmp_path / "manifest.txt"
        v.save_baseline(manifest)
        # New verifier reads the manifest and matches.
        v2 = IntegrityVerifier(modules=[a])
        v2.load_baseline(manifest)
        assert v2.verify() is True


# =============================================================================
# TamperPolicy
# =============================================================================
class TestTamperPolicy:
    def test_default_action_mapping(self):
        from deepgait3.security import (
            TamperAction, TamperLevel, TamperPolicy,
        )

        p = TamperPolicy()
        assert p.choose_action("low") == TamperAction.LOG
        assert p.choose_action("medium") == TamperAction.DEGRADE
        assert p.choose_action("high") == TamperAction.REFUSE
        assert p.choose_action("hard") == TamperAction.TERMINATE

    def test_unknown_severity_defaults_to_low(self):
        from deepgait3.security import TamperAction, TamperPolicy

        assert TamperPolicy().choose_action("nonsense") == TamperAction.LOG

    def test_overrides_apply(self):
        from deepgait3.security import (
            TamperAction, TamperLevel, TamperPolicy,
        )

        p = TamperPolicy(level_overrides={
            TamperLevel.LOW: TamperAction.DEGRADE,
        })
        assert p.choose_action("low") == TamperAction.DEGRADE

    def test_degrade_fires_callback(self):
        from deepgait3.security import (
            TamperAction, TamperLevel, TamperPolicy,
        )

        seen = []
        p = TamperPolicy(on_degrade=lambda: seen.append(1))
        p.execute(TamperAction.DEGRADE)
        assert seen == [1]

    def test_refuse_fires_callback(self):
        from deepgait3.security import TamperAction, TamperPolicy

        seen = []
        p = TamperPolicy(on_refuse=lambda: seen.append(1))
        p.execute(TamperAction.REFUSE)
        assert seen == [1]