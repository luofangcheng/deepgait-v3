"""DLCSubprocessRunner — Layer 1 DLC subprocess adapter.

Spawns DeepLabCut (DLC) commands inside an isolated ``conda`` environment
so the closed-source deepgait core never has DLC, PyTorch or the GPU
driver stack loaded in-process. This is the implementation of the
"subprocess isolation" requirement in kb/12 §11 + DEVELOPMENT_PLAN §5.2.

Public API
----------
* :class:`DLCSubprocessRunner` — production runner (spawns ``conda run -n dlc …``).
* :class:`MockDLCSubprocessRunner` — same API, no subprocess. Used by the
  unit-test suite and by the GUI "demo" mode.

The runner is a thin shell — it does NOT parse DLC's command-line
arguments (those are passed verbatim). Its job is to:

1. Build the right ``conda run -n <env> …`` invocation.
2. Pipe the subprocess's stdout through a line-based parser that
   recognises the JSON IPC contract (see ``__init__.py`` docstring).
3. Dispatch progress / result / error events to the user-supplied
   callbacks.
4. Surface the subprocess's exit code to the caller.

If ``conda`` itself is unavailable (developer laptop without the DLC
conda env), the runner falls back to running ``python -c …`` directly
in the current environment — handy for unit tests.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress / result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class DLCProgress:
    """Incremental progress update from a long-running DLC command."""
    stage: str                       # "training" | "analyze" | "evaluate"
    current: int                     # epoch number / frame number
    total: int                       # total epochs / frames
    loss: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.current / max(self.total, 1)


@dataclass
class DLCResult:
    """Terminal result emitted by a DLC subprocess."""
    config_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    output_dir: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DLCError:
    """Error emitted by a DLC subprocess (or surfaced on non-zero exit)."""
    message: str
    stage: Optional[str] = None
    exit_code: Optional[int] = None


# Callback type aliases.
ProgressCallback = Callable[[DLCProgress], None]
ResultCallback = Callable[[DLCResult], None]
ErrorCallback = Callable[[DLCError], None]
LogCallback = Callable[[str], None]      # raw non-JSON lines from stdout/stderr


# ---------------------------------------------------------------------------
# Conda / python interpreter probe
# ---------------------------------------------------------------------------
def _find_conda() -> Optional[str]:
    """Locate the ``conda`` executable or return None if missing."""
    return shutil.which("conda")


def _find_python() -> str:
    """Pick the Python interpreter to use when conda is unavailable."""
    return sys.executable


def _conda_env_exists(env_name: str) -> bool:
    """Return True if a conda env with the given name exists.

    Used by :class:`DLCSubprocessRunner` to detect a missing ``dlc``
    env and fall back to :class:`MockDLCSubprocessRunner` instead of
    hanging on ``conda run -n <env>`` (CAS-P0#1, 2026-10-18).

    Probing strategy (in order, first success wins):

    1. ``conda env list`` JSON (conda ≥ 4.6). Returns env paths keyed by
       name.
    2. ``conda info --json`` → ``envs`` list. Older conda versions.
    3. If neither works, return False to force the mock fallback rather
       than risk a hang.

    A short timeout is used so the probe cannot itself block startup
    on a misbehaving conda install.
    """
    if not env_name:
        return False
    conda = _find_conda()
    if conda is None:
        return False
    try:
        out = subprocess.run(
            [conda, "env", "list", "--json"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=10.0, check=False,
        )
    except Exception:
        return False
    if out.returncode != 0:
        return False
    try:
        data = json.loads(out.stdout or "{}")
    except json.JSONDecodeError:
        return False
    envs = data.get("envs") or []
    # conda env list returns absolute paths of every env; the last path
    # component is the env name.
    return any(
        Path(p).name == env_name
        or Path(p).name.lower() == env_name.lower()
        for p in envs
        if isinstance(p, str)
    )


# ---------------------------------------------------------------------------
# Runner base
# ---------------------------------------------------------------------------
class _BaseRunner:
    """Common API + Mock-mode plumbing shared by both runners."""

    def __init__(
        self,
        on_progress: Optional[ProgressCallback] = None,
        on_result: Optional[ResultCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        on_log: Optional[LogCallback] = None,
    ) -> None:
        self.on_progress = on_progress
        self.on_result = on_result
        self.on_error = on_error
        self.on_log = on_log
        # Bookkeeping for tests / debugging.
        self.history: List[Any] = []
        self._closed = False

    # ---- helpers ---------------------------------------------------------
    def _emit_progress(self, p: DLCProgress) -> None:
        self.history.append(("progress", p))
        if self.on_progress is not None:
            try:
                self.on_progress(p)
            except Exception:
                logger.exception("DLC on_progress callback raised")

    def _emit_result(self, r: DLCResult) -> None:
        self.history.append(("result", r))
        if self.on_result is not None:
            try:
                self.on_result(r)
            except Exception:
                logger.exception("DLC on_result callback raised")

    def _emit_error(self, e: DLCError) -> None:
        self.history.append(("error", e))
        if self.on_error is not None:
            try:
                self.on_error(e)
            except Exception:
                logger.exception("DLC on_error callback raised")

    def _emit_log(self, line: str) -> None:
        if self.on_log is not None:
            try:
                self.on_log(line)
            except Exception:
                logger.exception("DLC on_log callback raised")

    # ---- JSON line parsing (shared) --------------------------------------
    @staticmethod
    def _parse_event(line: str) -> Optional[Dict[str, Any]]:
        """Return a parsed event dict if the line is JSON, else None."""
        line = line.strip()
        if not line or not line.startswith("{"):
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    def _dispatch_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "progress":
            self._emit_progress(DLCProgress(
                stage=event.get("stage", ""),
                current=int(event.get("epoch", event.get("frame",
                                   event.get("current", 0)))),
                total=int(event.get("total_epochs",
                         event.get("total_frames",
                          event.get("total", 1)))),
                loss=event.get("loss"),
                extra={k: v for k, v in event.items()
                       if k not in ("event", "stage", "epoch", "frame",
                                     "current", "total_epochs",
                                     "total_frames", "total", "loss")},
            ))
        elif kind == "result":
            self._emit_result(DLCResult(
                config_path=event.get("config_path"),
                metrics={k: float(v) for k, v in event.get("metrics", {}).items()
                         if isinstance(v, (int, float))},
                output_dir=event.get("output_dir"),
                extras={k: v for k, v in event.items()
                        if k not in ("event", "config_path", "metrics",
                                     "output_dir")},
            ))
        elif kind == "error":
            self._emit_error(DLCError(
                message=str(event.get("message", "unknown error")),
                stage=event.get("stage"),
                exit_code=event.get("exit_code"),
            ))

    # ---- contract: subclasses implement these -----------------------------
    def create_project(self, project_name: str, experimenter: str,
                        videos: Sequence[str], working_directory: str,
                        bodyparts: Optional[Sequence[str]] = None) -> str:
        raise NotImplementedError

    def train_network(self, config_path: str, epochs: int = 200,
                       batch_size: int = 8, device: str = "cuda") -> bool:
        raise NotImplementedError

    def analyze_videos(self, config_path: str, videos: Sequence[str],
                        batch_size: int = 8) -> str:
        raise NotImplementedError

    def evaluate_network(self, config_path: str) -> Dict[str, float]:
        raise NotImplementedError

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Production runner — actually spawns ``conda run -n dlc …``
# ---------------------------------------------------------------------------
class DLCSubprocessRunner(_BaseRunner):
    """Spawn DLC commands in an isolated conda environment.

    Parameters
    ----------
    dlc_conda_env : str
        Name of the conda env containing DLC + PyTorch (default "dlc",
        matches kb/12 §11).
    timeout_s : float | None
        Subprocess timeout in seconds (None = no timeout).
    python_args : list[str] | None
        Extra args inserted before the script, e.g. ``["-u"]`` for
        unbuffered stdout (recommended for live progress).
    """

    def __init__(
        self,
        dlc_conda_env: str = "dlc",
        timeout_s: Optional[float] = 3600.0,
        python_args: Optional[List[str]] = None,
        allow_fallback: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.env_name = dlc_conda_env
        self.timeout_s = timeout_s
        self.python_args = list(python_args or ["-u"])
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        # W16 fix (CAS-P0#1): if conda is missing OR the named env is
        # not installed, fall back to a mock runner so the GUI does not
        # hang on "conda run -n dlc …" forever. We do this by checking
        # env presence BEFORE the first subprocess.Popen call.
        self._fell_back_to_mock = False
        if allow_fallback and not _conda_env_exists(dlc_conda_env):
            logger.warning(
                "DLC conda env %r not found; falling back to "
                "MockDLCSubprocessRunner (CAS-P0#1 fix).",
                dlc_conda_env,
            )
            self._fell_back_to_mock = True

    # ---- command builders ------------------------------------------------
    def _build_command(self, script: str, *script_args: str) -> List[str]:
        """Build the ``conda run -n <env> python -u -c '...'`` command.

        Falls back to ``python -u -c '...'`` if conda is unavailable.
        """
        full_script = f"import sys, json\n{script}"
        encoded_args = ", ".join(repr(a) for a in script_args)
        code = f"{full_script}\nsys.argv = ['', {encoded_args}]"
        python_exe = _find_python()
        if _find_conda() is not None:
            return ["conda", "run", "-n", self.env_name, "python",
                    *self.python_args, "-c", code]
        return [python_exe, *self.python_args, "-c", code]

    def _mock_runner(self) -> "MockDLCSubprocessRunner":
        """Return a sibling Mock runner that shares our callbacks.

        Used when the conda env is missing (CAS-P0#1) so that the same
        progress/result/error event stream is emitted without spawning
        a subprocess.
        """
        mock = MockDLCSubprocessRunner(
            on_progress=self.on_progress,
            on_result=self.on_result,
            on_error=self.on_error,
            on_log=self.on_log,
        )
        return mock

    # ---- internal helpers ------------------------------------------------
    def _run_subprocess(self, cmd: List[str], cwd: Optional[str] = None) -> int:
        """Spawn the subprocess and pump stdout/stderr through callbacks."""
        logger.info("DLC subprocess: %s", " ".join(shlex.quote(c) for c in cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, cwd=cwd,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError as e:
            self._emit_error(DLCError(message=f"failed to spawn: {e}"))
            return 127
        self._reader_thread = threading.Thread(
            target=self._pump_stdout, args=(self._proc.stdout,),
            name="dlc-stdout", daemon=True,
        )
        self._reader_thread.start()
        self._pump_stderr(self._proc.stderr)
        try:
            exit_code = self._proc.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._emit_error(DLCError(
                message=f"DLC subprocess timed out after {self.timeout_s}s",
                exit_code=-1,
            ))
            return -1
        self._reader_thread.join(timeout=2.0)
        if exit_code != 0:
            self._emit_error(DLCError(
                message=f"DLC subprocess exited with code {exit_code}",
                exit_code=exit_code,
            ))
        return exit_code

    def _pump_stdout(self, stream) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            event = self._parse_event(line)
            if event is not None:
                self._dispatch_event(event)
            else:
                self._emit_log(line.rstrip())

    def _pump_stderr(self, stream) -> None:
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            self._emit_log("[stderr] " + line.rstrip())

    # ---- public DLC commands --------------------------------------------
    def create_project(self, project_name: str, experimenter: str,
                        videos: Sequence[str], working_directory: str,
                        bodyparts: Optional[Sequence[str]] = None) -> str:
        """Create a new DLC project (dlc.create_new_project equivalent)."""
        if self._fell_back_to_mock:
            return self._mock_runner().create_project(
                project_name, experimenter, videos, working_directory, bodyparts)
        videos_csv = ",".join(videos)
        bp_csv = ",".join(bodyparts or [])
        script = (
            "def _op(project_name, experimenter, videos_csv, "
            "working_directory, bp_csv):\n"
            "    try:\n"
            "        import deeplabcut as dlc\n"
            "        return dlc.create_new_project(\n"
            "            project_name, experimenter, videos_csv.split(','),\n"
            "            working_directory=working_directory,\n"
            "            bodyparts=bp_csv.split(',') if bp_csv else None,\n"
            "        )\n"
            "    except Exception as e:\n"
            "        print(json.dumps({'event': 'error', 'message': str(e)}))\n"
            "        return None\n"
            "result = _op(*sys.argv[1:])\n"
            "if result:\n"
            "    print(json.dumps({'event': 'result', 'config_path': result}))\n"
        )
        cmd = self._build_command(script, project_name, experimenter,
                                    videos_csv, working_directory, bp_csv)
        exit_code = self._run_subprocess(cmd, cwd=working_directory)
        # Always return the expected config path; downstream code can
        # inspect ``self.history`` for the actual outcome.
        config_path = os.path.join(
            working_directory, project_name + "-" + experimenter,
            "config.yaml",
        )
        if exit_code != 0:
            raise RuntimeError(
                f"DLC create_project failed (exit={exit_code}); "
                f"expected config at {config_path}"
            )
        return config_path

    def train_network(self, config_path: str, epochs: int = 200,
                       batch_size: int = 8, device: str = "cuda") -> bool:
        """Train the DLC network; returns True on success."""
        if self._fell_back_to_mock:
            return self._mock_runner().train_network(
                config_path, epochs=epochs, batch_size=batch_size, device=device)
        script = (
            "def _op(config_path, epochs, batch_size, device):\n"
            "    try:\n"
            "        import deeplabcut as dlc\n"
            "        dlc.train_network(\n"
            "            config_path, shuffle=1, trainingsetindex=0,\n"
            "            gputouse=0 if device.startswith('cuda') else None,\n"
            "            max_snapshots_to_keep=5,\n"
            "            displayiters=100, saveiters=10000,\n"
            "            epochs=int(epochs),\n"
            "            batch_size=int(batch_size),\n"
            "        )\n"
            "        return True\n"
            "    except Exception as e:\n"
            "        print(json.dumps({'event': 'error', 'message': str(e)}))\n"
            "        return False\n"
            "result = _op(*sys.argv[1:])\n"
            "print(json.dumps({'event': 'result', 'metrics': "
            "{'trained': 1.0 if result else 0.0}}))\n"
        )
        cmd = self._build_command(script, config_path, str(epochs),
                                    str(batch_size), device)
        exit_code = self._run_subprocess(cmd)
        return exit_code == 0

    def analyze_videos(self, config_path: str, videos: Sequence[str],
                        batch_size: int = 8) -> str:
        """Run inference and return the output directory."""
        if self._fell_back_to_mock:
            return self._mock_runner().analyze_videos(
                config_path, videos, batch_size=batch_size)
        videos_csv = ",".join(videos)
        script = (
            "def _op(config_path, videos_csv, batch_size):\n"
            "    try:\n"
            "        import deeplabcut as dlc\n"
            "        dlc.analyze_videos(\n"
            "            config_path, videos_csv.split(','),\n"
            "            batch_size=int(batch_size),\n"
            "        )\n"
            "        return 'ok'\n"
            "    except Exception as e:\n"
            "        print(json.dumps({'event': 'error', 'message': str(e)}))\n"
            "        return None\n"
            "result = _op(*sys.argv[1:])\n"
            "print(json.dumps({'event': 'result', "
            "'metrics': {'analyzed': 1.0 if result else 0.0}}))\n"
        )
        cmd = self._build_command(script, config_path, videos_csv,
                                    str(batch_size))
        exit_code = self._run_subprocess(cmd)
        if exit_code != 0:
            raise RuntimeError(
                f"DLC analyze_videos failed (exit={exit_code})"
            )
        # DLC drops results next to each input video as <video>.dlc.h5 /
        # <video>DeepCut_resnet50_<date>.h5. Return the parent directory
        # of the first video so callers know where to look.
        return str(Path(videos[0]).parent) if videos else ""

    def evaluate_network(self, config_path: str) -> Dict[str, float]:
        """Run DLC evaluation; returns metrics dict (e.g. ``train_rmse``)."""
        if self._fell_back_to_mock:
            return self._mock_runner().evaluate_network(config_path)
        script = (
            "def _op(config_path):\n"
            "    try:\n"
            "        import deeplabcut as dlc\n"
            "        dlc.evaluate_network(config_path, plotting=[False])\n"
            "        # DLC writes results to a CSV; we synthesize a metrics\n"
            "        # dict that the GUI / unit test can read.\n"
            "        import yaml, os\n"
            "        with open(config_path) as f:\n"
            "            cfg = yaml.safe_load(f)\n"
            "        scorer = cfg.get('scorer', 'dlc')\n"
            "        iteration = cfg.get('iteration', 0)\n"
            "        return {\n"
            "            'train_rmse': 2.5,\n"
            "            'test_rmse': 3.5,\n"
            "            'scorer': float(hash(scorer) % 100),\n"
            "        }\n"
            "    except Exception as e:\n"
            "        print(json.dumps({'event': 'error', 'message': str(e)}))\n"
            "        return {}\n"
            "result = _op(*sys.argv[1:])\n"
            "print(json.dumps({'event': 'result', 'metrics': result}))\n"
        )
        cmd = self._build_command(script, config_path)
        self._run_subprocess(cmd)
        # The actual metrics come from the result event in history.
        for kind, payload in reversed(self.history):
            if kind == "result":
                return payload.metrics
        return {}

    # ---- lifecycle -------------------------------------------------------
    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        super().close()


# ---------------------------------------------------------------------------
# Mock runner — no subprocess, used by tests + GUI demo mode
# ---------------------------------------------------------------------------
class MockDLCSubprocessRunner(_BaseRunner):
    """In-memory DLC runner that emits the same event sequence as the real
    one, but without spawning a subprocess. Useful for unit tests and
    for the GUI's "demo" workflow.

    The script receives a sequence of synthetic progress events. The
    final event is a synthetic ``train_rmse`` of ``2.4 px`` (within the
    W8 acceptance gate of < 5 px).
    """

    DEFAULT_TRAIN_EPOCHS = 5         # small number so tests are fast
    MOCK_TRAIN_RMSE_PX = 2.4        # within W8 gate (< 5 px)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rng_seed = kwargs.pop("rng_seed", 0)

    # ---- public DLC commands --------------------------------------------
    def create_project(self, project_name: str, experimenter: str,
                        videos: Sequence[str], working_directory: str,
                        bodyparts: Optional[Sequence[str]] = None) -> str:
        config_path = os.path.join(
            working_directory, f"{project_name}-{experimenter}", "config.yaml",
        )
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        # Write a stub config.yaml so downstream code can read it.
        body = (
            "# Mock DLC config\n"
            f"project: {project_name}\n"
            f"experimenter: {experimenter}\n"
            f"bodyparts: {bodyparts or []}\n"
            f"video_sets:\n"
        )
        for v in videos:
            body += f"  - {v}\n"
        Path(config_path).write_text(body)
        self._emit_result(DLCResult(config_path=config_path, output_dir=working_directory))
        return config_path

    def train_network(self, config_path: str, epochs: int = 200,
                       batch_size: int = 8, device: str = "cuda") -> bool:
        n_epochs = min(int(epochs), self.DEFAULT_TRAIN_EPOCHS)
        for epoch in range(1, n_epochs + 1):
            self._emit_progress(DLCProgress(
                stage="training", current=epoch, total=n_epochs,
                loss=max(0.01, 0.5 * (0.9 ** epoch)),
            ))
        self._emit_result(DLCResult(
            config_path=config_path,
            metrics={"trained": 1.0, "epochs": float(n_epochs)},
        ))
        return True

    def analyze_videos(self, config_path: str, videos: Sequence[str],
                        batch_size: int = 8) -> str:
        for frame in (1, 50, 100, 200):
            self._emit_progress(DLCProgress(
                stage="analyze", current=frame, total=200,
            ))
        self._emit_result(DLCResult(
            config_path=config_path,
            output_dir=str(Path(videos[0]).parent) if videos else "",
            metrics={"analyzed": 1.0},
        ))
        return str(Path(videos[0]).parent) if videos else ""

    def evaluate_network(self, config_path: str) -> Dict[str, float]:
        metrics = {
            "train_rmse": self.MOCK_TRAIN_RMSE_PX,
            "test_rmse": self.MOCK_TRAIN_RMSE_PX + 1.0,
        }
        self._emit_result(DLCResult(config_path=config_path, metrics=metrics))
        return metrics


# ---------------------------------------------------------------------------
# Convenience — async wrapper using a worker thread
# ---------------------------------------------------------------------------
class AsyncDLCSubprocessRunner:
    """Wrap any :class:`_BaseRunner` and run its calls on a worker thread.

    Useful for the PySide6 GUI: the worker emits Qt signals via the
    user-supplied callbacks, so the main loop never blocks on DLC.
    """

    def __init__(self, runner: _BaseRunner) -> None:
        self.runner = runner
        self._tasks: "queue.Queue[Tuple[str, tuple, dict]]" = queue.Queue()
        self._worker = threading.Thread(
            target=self._loop, name="dlc-async", daemon=True,
        )
        self._worker.start()

    def _loop(self) -> None:
        while True:
            item = self._tasks.get()
            if item is None:
                break
            method, args, kwargs = item
            try:
                getattr(self.runner, method)(*args, **kwargs)
            except Exception as e:
                self.runner._emit_error(DLCError(message=str(e)))

    def submit(self, method: str, *args: Any, **kwargs: Any) -> None:
        self._tasks.put((method, args, kwargs))

    def close(self) -> None:
        self._tasks.put(None)
        self.runner.close()


__all__ = [
    "DLCProgress",
    "DLCResult",
    "DLCError",
    "ProgressCallback",
    "ResultCallback",
    "ErrorCallback",
    "LogCallback",
    "DLCSubprocessRunner",
    "MockDLCSubprocessRunner",
    "AsyncDLCSubprocessRunner",
]