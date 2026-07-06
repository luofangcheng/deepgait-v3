"""Unit tests for Phase 1 W2 — Layer 1 camera drivers.

Covers:
    deepgait/hardware/camera/basler.py     — BaslerCamera (with pypylon fallback)
    deepgait/hardware/camera/trigger.py   — RP2040 TriggerController
    deepgait/hardware/camera/multi_cam.py — MultiCameraManager + MockCamera

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.1 W2):
    "4 cameras enumerate + frame grab pass"

Acceptance gate (docs/DEVELOPMENT_PLAN.md §9.1):
    "4 cameras synchronised (frame-to-frame error < 1 ms)"

These tests do NOT require physical hardware: BaslerCamera and the
multi-camera roster use the in-process MockCamera fallback. They will
therefore run on CI and on developer laptops without Hikvision/Basler
kit. Real-hardware coverage is gated by the ``hardware`` marker (see
conftest.py) and only enabled when ``DEEPGAIT_HAS_HW=1``.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# Shared fixtures
# =============================================================================
@pytest.fixture
def mock_roster():
    """Return 4 MockCameras aligned to the standard deepgait roster."""
    from deepgait3.hardware.camera.multi_cam import MockCamera, DEFAULT_ROSTER

    return [(role, MockCamera(serial=f"MOCK-{role}", fps=100))
            for role in DEFAULT_ROSTER]


# =============================================================================
# MockCamera
# =============================================================================
class TestMockCamera:
    def test_open_close(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cam = MockCamera(serial="X", fps=100)
        cam.open()
        assert cam.get_serial() == "X"
        assert cam.get_model() == "MockCamera"
        cam.close()

    def test_grab_one_increments_frame_number(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cam = MockCamera(serial="X", fps=100, width=64, height=48)
        cam.open()
        f1 = cam.grab_one()
        f2 = cam.grab_one()
        cam.close()
        assert f1.frame_number == 1
        assert f2.frame_number == 2
        assert f1.image.shape == (48, 64, 3)
        assert f1.image.dtype == np.uint8

    def test_grab_one_requires_open(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cam = MockCamera(serial="X", fps=100)
        with pytest.raises(RuntimeError):
            cam.grab_one()

    def test_set_roi_validates(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cam = MockCamera(serial="X", fps=100)
        with pytest.raises(ValueError):
            cam.set_roi(0, 0, 0, 100)
        with pytest.raises(ValueError):
            cam.set_roi(0, 0, 100, 0)
        cam.set_roi(10, 20, 100, 80)  # ok

    def test_set_exposure_validates(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cam = MockCamera(serial="X", fps=100)
        with pytest.raises(ValueError):
            cam.set_exposure_us(0.0)
        cam.set_exposure_us(100.0)

    def test_configure_trigger_accepts_known_lines(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cam = MockCamera(serial="X", fps=100)
        for line in (0, 1, 2, 3):
            cam.configure_hardware_trigger(line=line, edge="rising")
        with pytest.raises(ValueError):
            cam.configure_hardware_trigger(line=5, edge="rising")

    def test_continuous_callback_fires(self):
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cam = MockCamera(serial="X", fps=200, width=8, height=8)
        cam.open()
        seen = []

        def cb(frame):
            seen.append(frame.frame_number)

        cam.start_continuous(cb)
        time.sleep(0.10)  # ~20 frames at 200 fps
        cam.stop_continuous()
        assert len(seen) >= 3, f"expected >=3 callbacks, got {len(seen)}"


# =============================================================================
# FrameBus
# =============================================================================
class TestFrameBus:
    def test_quartet_blocking_until_complete(self):
        from deepgait3.hardware.camera.multi_cam import FrameBus

        bus = FrameBus(("a", "b", "c", "d"))
        # Push only 3 roles → wait_quartet should time out.
        for role, fn in (("a", 1), ("b", 1), ("c", 1)):
            from deepgait3.hardware.camera.base import FrameInfo

            bus.push(role, FrameInfo(image=np.zeros((2, 2, 3), np.uint8),
                                     frame_number=fn, timestamp_ns=fn * 1_000,
                                     camera_serial=role, exposure_us=0.0,
                                     gain_db=0.0))
        t0 = time.monotonic()
        snap = bus.wait_quartet(timeout_ms=100)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert snap is None
        assert 90 <= elapsed_ms <= 500, elapsed_ms  # respected timeout

    def test_quartet_returns_when_all_roles_present(self):
        from deepgait3.hardware.camera.base import FrameInfo
        from deepgait3.hardware.camera.multi_cam import FrameBus

        bus = FrameBus(("a", "b"))
        for role in ("a", "b"):
            bus.push(role, FrameInfo(image=np.zeros((2, 2, 3), np.uint8),
                                     frame_number=1, timestamp_ns=1,
                                     camera_serial=role, exposure_us=0.0,
                                     gain_db=0.0))
        snap = bus.wait_quartet(timeout_ms=200)
        assert snap is not None
        assert set(snap.keys()) == {"a", "b"}

    def test_push_rejects_unknown_role(self):
        from deepgait3.hardware.camera.multi_cam import FrameBus

        bus = FrameBus(("a", "b"))
        from deepgait3.hardware.camera.base import FrameInfo

        with pytest.raises(KeyError):
            bus.push("ghost", FrameInfo(image=np.zeros((2, 2, 3), np.uint8),
                                        frame_number=1, timestamp_ns=1,
                                        camera_serial="g", exposure_us=0.0,
                                        gain_db=0.0))


# =============================================================================
# MultiCameraManager — W2 acceptance: "4 cameras enumerate + grab pass"
# =============================================================================
class TestMultiCameraManager:
    def test_roster_size_must_be_positive(self):
        from deepgait3.hardware.camera.multi_cam import MultiCameraManager

        with pytest.raises(ValueError):
            MultiCameraManager(cameras=[])

    def test_roster_is_listed(self, mock_roster):
        from deepgait3.hardware.camera.multi_cam import MultiCameraManager

        mgr = MultiCameraManager(cameras=mock_roster, trigger_line=0)
        assert len(mgr) == 4
        assert set(mgr.roles) == {"bottom", "left", "right", "top"}
        assert mgr.is_running is False

    def test_get_camera_round_trip(self, mock_roster):
        from deepgait3.hardware.camera.multi_cam import MultiCameraManager

        mgr = MultiCameraManager(cameras=mock_roster)
        cam = mgr.get_camera("left")
        assert cam.get_serial() == "MOCK-left"

    def test_unknown_role_raises_keyerror(self, mock_roster):
        from deepgait3.hardware.camera.multi_cam import MultiCameraManager

        mgr = MultiCameraManager(cameras=mock_roster)
        with pytest.raises(KeyError):
            mgr.get_camera("diagonal")

    def test_open_close_all(self, mock_roster):
        from deepgait3.hardware.camera.multi_cam import MultiCameraManager

        mgr = MultiCameraManager(cameras=mock_roster)
        mgr.open_all()
        mgr.configure_trigger()
        mgr.close_all()
        assert mgr.is_running is False

    def test_start_all_grabs_quartet_under_sync_tolerance(self, mock_roster):
        """Acceptance gate: 4 cameras synchronised, error < 1 ms."""
        from deepgait3.hardware.camera.multi_cam import (
            MultiCameraManager, SYNC_TOLERANCE_MS,
        )

        mgr = MultiCameraManager(cameras=mock_roster, trigger_line=0)
        with mgr:
            snap = mgr.grab_quartet(timeout_ms=2000)
            assert snap is not None, "no quartet produced within timeout"
            assert set(snap.keys()) == set(mgr.roles)
            for frame in snap.values():
                assert frame.image.shape[2] == 3
                assert frame.image.dtype == np.uint8
            report = mgr.get_sync_report()
            assert report.in_sync, report.as_dict()
            assert report.max_delta_ms <= SYNC_TOLERANCE_MS, report.as_dict()
            assert report.sample_size >= 1

    def test_sync_report_detects_jitter(self, mock_roster):
        """Inject a 5 ms clock offset on one camera → report must flag OOR."""
        from deepgait3.hardware.camera.base import FrameInfo
        from deepgait3.hardware.camera.multi_cam import (
            FrameBus, MultiCameraManager,
        )

        # Build a roster where 'top' is 5 ms ahead of the others.
        mgr = MultiCameraManager(cameras=mock_roster, sync_tolerance_ms=1.0)
        # Build a snapshot and feed it through _record_sync directly.
        from deepgait3.hardware.camera.multi_cam import MockCamera

        cams = dict(mock_roster)
        # Replace 'top' with one offset by +5 ms.
        cams["top"] = MockCamera(serial="MOCK-top", fps=100, clock_offset_ms=5.0)

        # Manually exercise _record_sync via grab_quartet: open all,
        # grab one frame per camera, push to the bus.
        bus = mgr._bus  # type: ignore[attr-defined]
        for role, cam in cams.items():
            cam.open()
            cam.configure_hardware_trigger(line=0)
            cam.start_continuous(lambda f, role=role: bus.push(role, f))
        time.sleep(0.15)  # let some frames flow
        snap = bus.wait_quartet(timeout_ms=2000)
        assert snap is not None
        mgr._record_sync(snap)  # type: ignore[attr-defined]
        report = mgr.get_sync_report()
        assert not report.in_sync
        assert report.max_delta_ms > 1.0

    def test_enumerate_default_falls_back_to_mock(self, monkeypatch):
        """On dev laptops without hardware, enumerate_default returns a
        manager whose cameras are MockCamera instances (via CameraFactory
        fallback)."""
        from deepgait3.hardware.camera import multi_cam

        # Force the factory's Windows path to fail (no Hikvision on Linux).
        def _raise(*a, **kw):
            raise RuntimeError("simulated: no Hikvision SDK")
        monkeypatch.setattr(multi_cam.CameraFactory, "create",
                            staticmethod(_raise))
        mgr = multi_cam.MultiCameraManager.enumerate_default(n_cameras=4)
        assert len(mgr) == 4
        # All four cameras must be MockCamera.
        from deepgait3.hardware.camera.multi_cam import MockCamera

        for _, cam in mgr._roster:  # type: ignore[attr-defined]
            assert isinstance(cam, MockCamera), type(cam)


# =============================================================================
# BaslerCamera (mock path)
# =============================================================================
class TestBaslerCameraMock:
    def test_basler_falls_back_to_mock_when_pypylon_missing(self, monkeypatch):
        """Without pypylon, BaslerCamera must auto-degrade and emit frames."""
        from deepgait3.hardware.camera import basler

        monkeypatch.setattr(basler, "_PYPYLON", False)
        cam = basler.BaslerCamera(camera_id=0, width=32, height=24)
        assert cam.is_mock is True
        cam.open()
        f = cam.grab_one()
        assert f.image.shape == (24, 32, 3)
        cam.close()

    def test_basler_explicit_use_mock(self):
        from deepgait3.hardware.camera.basler import BaslerCamera

        cam = BaslerCamera(camera_id=2, width=16, height=16,
                           use_mock=True, serial="ABC123")
        assert cam.is_mock is True
        assert cam.get_serial() == "ABC123"
        cam.open()
        cam.grab_one()
        cam.close()

    def test_basler_serial_serial_filter_round_trip(self):
        from deepgait3.hardware.camera.basler import BaslerCamera

        cam = BaslerCamera(camera_id=0, serial="SN-007", use_mock=True)
        assert cam.get_serial() == "SN-007"

    def test_basler_trigger_rejects_invalid_line(self):
        from deepgait3.hardware.camera.basler import BaslerCamera

        cam = BaslerCamera(camera_id=0, use_mock=True)
        cam.open()
        with pytest.raises(ValueError):
            cam.configure_hardware_trigger(line=0)  # Basler: Line 0 invalid
        with pytest.raises(ValueError):
            cam.configure_hardware_trigger(line=1, edge="invalid")

    def test_basler_exposure_validation(self):
        from deepgait3.hardware.camera.basler import BaslerCamera

        cam = BaslerCamera(camera_id=0, use_mock=True)
        cam.open()
        with pytest.raises(ValueError):
            cam.set_exposure_us(-1.0)
        cam.set_exposure_us(5000.0)
        cam.close()


# =============================================================================
# TriggerController (mock path — no pyserial installed)
# =============================================================================
class TestTriggerControllerMock:
    def test_start_emits_pulses(self):
        from deepgait3.hardware.camera.trigger import TriggerController

        tc = TriggerController(target_hz=200, pulse_width_us=10)
        try:
            tc.start()
            assert tc.is_mock is True
            time.sleep(0.10)  # ~20 pulses at 200 Hz
            hb = tc.get_heartbeat()
            assert hb["running"] is True
            assert hb["source"] == "mock"
            assert hb["pulse_count"] >= 5
        finally:
            tc.stop()

    def test_pulse_callback_fires(self):
        from deepgait3.hardware.camera.trigger import (
            PulseEvent, TriggerController,
        )

        seen = []

        def cb(ev: PulseEvent):
            seen.append(ev)

        tc = TriggerController(target_hz=200, on_pulse=cb)
        try:
            tc.start()
            time.sleep(0.08)
        finally:
            tc.stop()
        assert len(seen) >= 3
        # Sequences are 1-based and monotonic.
        assert [e.sequence for e in seen] == sorted(e.sequence for e in seen)
        assert seen[0].sequence >= 1
        assert seen[0].source == "mock"

    def test_is_alive_after_start(self):
        from deepgait3.hardware.camera.trigger import TriggerController

        tc = TriggerController(target_hz=100)
        try:
            tc.start()
            time.sleep(0.05)
            assert tc.is_alive() is True
        finally:
            tc.stop()
        # After stop, no more pulses — is_alive requires fresh activity.
        # Give a fresh start to confirm the gate works.
        tc.start()
        try:
            time.sleep(0.05)
            assert tc.is_alive() is True
        finally:
            tc.stop()

    def test_invalid_target_hz_rejected(self):
        from deepgait3.hardware.camera.trigger import TriggerController

        with pytest.raises(ValueError):
            TriggerController(target_hz=0)
        with pytest.raises(ValueError):
            TriggerController(target_hz=-10)
        with pytest.raises(ValueError):
            TriggerController(pulse_width_us=0)

    def test_context_manager(self):
        from deepgait3.hardware.camera.trigger import TriggerController

        with TriggerController(target_hz=200) as tc:
            assert tc.is_alive() or tc.get_heartbeat()["pulse_count"] >= 1
        # After __exit__, controller is stopped.
        hb = tc.get_heartbeat()
        assert hb["running"] is False

    def test_double_start_is_idempotent(self):
        from deepgait3.hardware.camera.trigger import TriggerController

        tc = TriggerController(target_hz=200)
        try:
            tc.start()
            tc.start()  # should not raise or duplicate threads
            time.sleep(0.05)
            assert tc.get_heartbeat()["pulse_count"] >= 1
        finally:
            tc.stop()