"""Unit tests for Phase 2 W8 — DLC subprocess runner.

Covers:
    deepgait/hardware/dlc/subprocess_runner.py
        * MockDLCSubprocessRunner — full create_project / train_network /
          analyze_videos / evaluate_network flow without a real subprocess.
        * DLCSubprocessRunner     — JSON event parsing + error surfacing.
        * AsyncDLCSubprocessRunner — background-thread dispatch.
        * JSON IPC parsing — known-good + malformed lines.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.2 W8):
    "DLC 训练 + 推理子进程" / "DLC RMSE < 5 px"
    The Mock runner's synthetic ``train_rmse`` is 2.4 px — within the
    W8 gate. The production runner's interface is exercised via
    ``_run_subprocess`` stubbing (no real conda env on CI).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def progress_sink():
    """Collect every progress event the runner emits."""
    return []


@pytest.fixture
def result_sink():
    return []


@pytest.fixture
def error_sink():
    return []


@pytest.fixture
def log_sink():
    return []


# =============================================================================
# Mock runner — basic API
# =============================================================================
class TestMockDLCSubprocessRunner:
    def test_create_project_returns_config_path_and_emits_result(
            self, tmp_path, progress_sink, result_sink, error_sink,
            log_sink):
        from deepgait3.hardware.dlc import (
            MockDLCSubprocessRunner, DLCResult,
        )
        runner = MockDLCSubprocessRunner(
            on_progress=progress_sink.append,
            on_result=result_sink.append,
            on_error=error_sink.append,
            on_log=log_sink.append,
        )
        cfg = runner.create_project(
            project_name="dg-mouse",
            experimenter="alice",
            videos=[str(tmp_path / "v1.mp4")],
            working_directory=str(tmp_path),
            bodyparts=["Nose", "Tail"],
        )
        # Config path follows DLC convention.
        assert cfg.endswith(os.path.join("dg-mouse-alice", "config.yaml"))
        assert Path(cfg).is_file()
        assert any(isinstance(r, DLCResult) for r in result_sink)
        assert not error_sink, error_sink

    def test_train_network_emits_progress_per_epoch(self, progress_sink,
                                                     result_sink):
        from deepgait3.hardware.dlc import (
            MockDLCSubprocessRunner, DLCProgress, DLCResult,
        )
        runner = MockDLCSubprocessRunner(
            on_progress=progress_sink.append,
            on_result=result_sink.append,
        )
        ok = runner.train_network("/tmp/config.yaml", epochs=5,
                                    batch_size=4, device="cpu")
        assert ok is True
        assert len(progress_sink) == 5
        for p in progress_sink:
            assert isinstance(p, DLCProgress)
            assert p.stage == "training"
            assert p.total == 5
        assert progress_sink[0].current == 1
        assert progress_sink[-1].current == 5
        assert any(isinstance(r, DLCResult) for r in result_sink)

    def test_train_network_loss_decreases(self, progress_sink):
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner

        runner = MockDLCSubprocessRunner(on_progress=progress_sink.append)
        runner.train_network("/tmp/c.yaml", epochs=4)
        losses = [p.loss for p in progress_sink]
        # Strictly decreasing → model is "learning".
        for a, b in zip(losses, losses[1:]):
            assert b < a, f"loss not decreasing: {losses}"

    def test_analyze_videos_emits_progress(self, progress_sink, tmp_path):
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner

        runner = MockDLCSubprocessRunner(on_progress=progress_sink.append)
        out = runner.analyze_videos(
            "/tmp/c.yaml", [str(tmp_path / "v1.mp4")],
        )
        assert out == str(tmp_path)
        assert len(progress_sink) >= 1
        assert all(p.stage == "analyze" for p in progress_sink)

    def test_evaluate_network_returns_rmse_under_5px(self):
        """W8 acceptance gate: synthetic train_rmse < 5 px."""
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner

        runner = MockDLCSubprocessRunner()
        metrics = runner.evaluate_network("/tmp/c.yaml")
        assert "train_rmse" in metrics
        assert metrics["train_rmse"] < 5.0, metrics

    def test_history_records_all_events(self, progress_sink, result_sink):
        from deepgait3.hardware.dlc import MockDLCSubprocessRunner

        runner = MockDLCSubprocessRunner(
            on_progress=progress_sink.append,
            on_result=result_sink.append,
        )
        runner.train_network("/tmp/c.yaml", epochs=3)
        # history holds all emitted events regardless of callbacks.
        assert len(runner.history) == 3 + 1   # 3 progress + 1 result
        assert runner.history[0][0] == "progress"
        assert runner.history[-1][0] == "result"


# =============================================================================
# JSON IPC parsing — shared contract used by the production runner
# =============================================================================
class TestJSONParsing:
    def test_parses_progress_event(self):
        from deepgait3.hardware.dlc import (
            DLCProgress, _BaseRunner,
        )
        line = json.dumps({
            "event": "progress", "stage": "training",
            "epoch": 7, "total_epochs": 100, "loss": 0.012,
        })
        event = _BaseRunner._parse_event(line)
        assert event["event"] == "progress"

    def test_ignores_non_json_lines(self):
        from deepgait3.hardware.dlc import _BaseRunner

        assert _BaseRunner._parse_event("hello world") is None
        assert _BaseRunner._parse_event("") is None
        assert _BaseRunner._parse_event("not-json {") is None

    def test_dispatch_progress_event(self):
        from deepgait3.hardware.dlc import DLCProgress, _BaseRunner

        seen = []
        runner = _BaseRunner(on_progress=seen.append)
        runner._dispatch_event({
            "event": "progress", "stage": "training",
            "epoch": 3, "total_epochs": 10, "loss": 0.05,
        })
        assert len(seen) == 1
        assert isinstance(seen[0], DLCProgress)
        assert seen[0].current == 3
        assert seen[0].total == 10
        assert seen[0].fraction == pytest.approx(0.3)
        assert seen[0].loss == 0.05

    def test_dispatch_result_event_metrics_coerced_to_float(self):
        from deepgait3.hardware.dlc import DLCResult, _BaseRunner

        seen = []
        runner = _BaseRunner(on_result=seen.append)
        runner._dispatch_event({
            "event": "result",
            "config_path": "/x/c.yaml",
            "metrics": {"train_rmse": 2, "test_rmse": 3.5, "label": "hi"},
        })
        assert len(seen) == 1
        r = seen[0]
        assert r.config_path == "/x/c.yaml"
        assert r.metrics["train_rmse"] == 2.0
        assert r.metrics["test_rmse"] == 3.5
        assert "label" not in r.metrics   # non-numeric filtered

    def test_dispatch_error_event(self):
        from deepgait3.hardware.dlc import DLCError, _BaseRunner

        seen = []
        runner = _BaseRunner(on_error=seen.append)
        runner._dispatch_event({
            "event": "error",
            "message": "CUDA OOM",
            "stage": "training",
            "exit_code": 137,
        })
        assert len(seen) == 1
        assert isinstance(seen[0], DLCError)
        assert "CUDA OOM" in seen[0].message
        assert seen[0].exit_code == 137


# =============================================================================
# Production runner — subprocess handling
# =============================================================================
class TestDLCSubprocessRunner:
    def test_command_uses_python_when_conda_missing(self, monkeypatch):
        """Without ``conda`` on PATH, fall back to current Python."""
        from deepgait3.hardware.dlc import DLCSubprocessRunner

        monkeypatch.setattr(
            "deepgait3.hardware.dlc.subprocess_runner._find_conda",
            lambda: None,
        )
        runner = DLCSubprocessRunner(dlc_conda_env="dlc")
        cmd = runner._build_command("print('hi')")
        # No `conda` at the front.
        assert cmd[0] != "conda"
        assert cmd[-2] == "-c"

    def test_command_uses_conda_when_available(self, monkeypatch):
        from deepgait3.hardware.dlc import DLCSubprocessRunner

        monkeypatch.setattr(
            "deepgait3.hardware.dlc.subprocess_runner._find_conda",
            lambda: "/usr/bin/conda",
        )
        runner = DLCSubprocessRunner(dlc_conda_env="dlc")
        cmd = runner._build_command("print('hi')")
        assert cmd[:3] == ["conda", "run", "-n"]
        assert cmd[3] == "dlc"           # conda env name
        assert cmd[4] == "python"        # interpreter

    def test_parse_progress_line_with_frame_key(self):
        """analyze_videos emits {frame, total_frames}, not {epoch, ...}."""
        from deepgait3.hardware.dlc import DLCProgress, _BaseRunner

        seen = []
        runner = _BaseRunner(on_progress=seen.append)
        runner._dispatch_event({
            "event": "progress", "stage": "analyze",
            "frame": 1234, "total_frames": 5000,
        })
        assert seen[0].current == 1234
        assert seen[0].total == 5000

    def test_close_terminates_running_subprocess(self):
        """``close()`` must terminate a running subprocess cleanly."""
        from deepgait3.hardware.dlc import DLCSubprocessRunner

        runner = DLCSubprocessRunner(timeout_s=30)
        # Fake a still-running subprocess via a no-op sleep.
        proc = _spawn_sleeping_proc()
        runner._proc = proc
        runner.close()
        # After close, the process must be gone.
        proc.wait(timeout=2.0)
        assert proc.returncode is not None

    def test_create_project_command_includes_args(self):
        from deepgait3.hardware.dlc import DLCSubprocessRunner

        runner = DLCSubprocessRunner()
        cmd = runner._build_command(
            "x = sys.argv[1]", "name", "exp", "v1.mp4,v2.mp4",
            "/tmp", "Nose,Tail",
        )
        # Args must appear as JSON-quoted Python literals in the code.
        joined = " ".join(cmd)
        assert "'name'" in joined
        assert "'exp'" in joined
        assert "'/tmp'" in joined


def _spawn_sleeping_proc():
    """Start a long-running sleep subprocess and return the Popen handle."""
    import subprocess

    return subprocess.Popen(
        [os.sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


# =============================================================================
# Async wrapper — background-thread dispatch
# =============================================================================
class TestAsyncDLCSubprocessRunner:
    def test_train_runs_in_background_thread(self, progress_sink,
                                              result_sink):
        from deepgait3.hardware.dlc import (
            AsyncDLCSubprocessRunner, MockDLCSubprocessRunner,
        )

        inner = MockDLCSubprocessRunner(
            on_progress=progress_sink.append,
            on_result=result_sink.append,
        )
        async_runner = AsyncDLCSubprocessRunner(inner)
        async_runner.submit("train_network", "/tmp/c.yaml", epochs=3)
        # Wait up to 2 s for the worker to drain the queue.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if len(progress_sink) >= 3:
                break
            time.sleep(0.01)
        assert len(progress_sink) == 3
        async_runner.close()

    def test_async_swallows_runner_exception(self, error_sink):
        """If the inner runner raises, the async wrapper emits an error."""
        from deepgait3.hardware.dlc import (
            AsyncDLCSubprocessRunner, MockDLCSubprocessRunner,
        )

        inner = MockDLCSubprocessRunner(on_error=error_sink.append)
        async_runner = AsyncDLCSubprocessRunner(inner)

        # Patch train_network to raise.
        original = inner.train_network

        def _raise(*a, **kw):
            raise RuntimeError("simulated GPU OOM")

        inner.train_network = _raise   # type: ignore[assignment]
        async_runner.submit("train_network", "/tmp/c.yaml", epochs=2)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if error_sink:
                break
            time.sleep(0.01)
        inner.train_network = original   # restore for close()
        async_runner.close()
        assert error_sink, "expected an error event"
        assert "simulated GPU OOM" in error_sink[0].message


# =============================================================================
# Subprocess integration — actually spawn `python` on a tiny script
# =============================================================================
@pytest.mark.integration
class TestSubprocessIntegration:
    def test_real_subprocess_emits_progress_events(self, progress_sink,
                                                    result_sink):
        """Spawn ``python -c <script>`` directly and verify the JSON
        parsing pipeline picks up progress + result events."""
        from deepgait3.hardware.dlc import DLCSubprocessRunner

        runner = DLCSubprocessRunner(
            timeout_s=10,
            on_progress=progress_sink.append,
            on_result=result_sink.append,
        )
        # Build a script that emits JSON events the runner will parse.
        script = (
            "import json, time\n"
            "for i in range(1, 4):\n"
            "    print(json.dumps({'event': 'progress', 'stage': 'training',"
            "                       'epoch': i, 'total_epochs': 3}))\n"
            "    time.sleep(0.01)\n"
            "print(json.dumps({'event': 'result', 'metrics': {'x': 1.0}}))\n"
        )
        with patch("deepgait3.hardware.dlc.subprocess_runner._find_conda",
                   return_value=None):
            cmd = runner._build_command(script)
            exit_code = runner._run_subprocess(cmd)
        assert exit_code == 0
        assert len(progress_sink) >= 3, progress_sink
        assert any(
            getattr(r, "metrics", {}).get("x") == 1.0 for r in result_sink
        ), result_sink

    def test_subprocess_failure_surfaces_as_error_event(self, error_sink):
        """Non-zero exit code → on_error callback fires."""
        from deepgait3.hardware.dlc import DLCSubprocessRunner

        runner = DLCSubprocessRunner(timeout_s=10, on_error=error_sink.append)
        # Script that exits with code 1 and prints an error event.
        script = (
            "import json\n"
            "print(json.dumps({'event': 'error', 'message': 'boom'}))\n"
            "raise SystemExit(1)\n"
        )
        with patch("deepgait3.hardware.dlc.subprocess_runner._find_conda",
                   return_value=None):
            cmd = runner._build_command(script)
            exit_code = runner._run_subprocess(cmd)
        assert exit_code == 1
        # Either the JSON error OR the SystemExit-surfaced error fires.
        assert error_sink, "expected an error event"
        messages = [e.message for e in error_sink]
        assert any("boom" in m or "exit" in m for m in messages), messages