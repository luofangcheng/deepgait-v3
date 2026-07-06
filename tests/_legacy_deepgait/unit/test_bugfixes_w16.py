"""Regression tests for the W16 P0 bugs collected during 3-PI beta.

These tests correspond one-to-one to the bug IDs in
``docs/beta_feedback/{cas,embl,pku}-v2.0-*.md`` and the resolution notes
in ``docs/BETA_TEST_REPORT.md``.  Run with::

    pytest tests/unit/test_bugfixes_w16.py -v

Test inventory
--------------
* test_cas_p0_1_dlc_conda_env_missing_falls_back_to_mock
* test_cas_p0_2_triangulation_with_one_camera_raises_clean_error
* test_embl_p0_1_dlc_subprocess_runner_uses_utf8_encoding
* test_pku_p0_1_read_dlc_csv_empty_file_raises_friendly_error
* test_pku_p0_1_read_dlc_csv_missing_file_raises_friendly_error
* test_pku_p0_1_read_dlc_csv_wrong_header_raises_friendly_error
* test_p1_pyuqtgraph_time_axis_formatter
* test_p2_compute_angles_returns_nan_for_missing_joints
* test_p2_triangulation_rms_display_two_decimals
"""
from __future__ import annotations

import io
import sys
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# CAS-P0#1 — DLC conda env missing -> mock fallback
# ---------------------------------------------------------------------------
class TestCasP0_1DlcCondaEnvMissing(unittest.TestCase):
    """CAS reported the GUI hangs forever on ``conda run -n dlc`` when the
    env is missing.  Fix: ``DLCSubprocessRunner.__init__`` probes the env
    and falls back to a mock runner when the env is absent.
    """

    def test_conda_env_missing_triggers_mock_fallback(self) -> None:
        from deepgait3.hardware.dlc import subprocess_runner as sr
        # Pretend conda is on PATH but the env 'dlc' is not installed.
        with mock.patch.object(sr, "_find_conda", return_value="/usr/bin/conda"), \
             mock.patch.object(sr, "_conda_env_exists", return_value=False):
            events: list[tuple[str, object]] = []
            runner = sr.DLCSubprocessRunner(
                dlc_conda_env="dlc",
                on_progress=lambda p: events.append(("progress", p)),
                on_result=lambda r: events.append(("result", r)),
                on_error=lambda e: events.append(("error", e)),
            )
            self.assertTrue(runner._fell_back_to_mock)
            # The four public methods must all short-circuit through the
            # mock and emit at least one result event.
            config = runner.create_project("p", "alice", ["v.mp4"], "/tmp", ["Nose"])
            self.assertTrue(config.endswith("config.yaml"))
            self.assertTrue(runner.train_network(config, epochs=2))
            self.assertTrue(any(k == "result" for k, _ in events))
            out = runner.analyze_videos(config, ["v.mp4"])
            self.assertTrue(out)
            metrics = runner.evaluate_network(config)
            self.assertIn("train_rmse", metrics)

    def test_conda_env_present_does_not_fall_back(self) -> None:
        from deepgait3.hardware.dlc import subprocess_runner as sr
        with mock.patch.object(sr, "_find_conda", return_value="/usr/bin/conda"), \
             mock.patch.object(sr, "_conda_env_exists", return_value=True):
            runner = sr.DLCSubprocessRunner(dlc_conda_env="dlc")
            self.assertFalse(runner._fell_back_to_mock)

    def test_no_conda_at_all_triggers_mock_fallback(self) -> None:
        from deepgait3.hardware.dlc import subprocess_runner as sr
        with mock.patch.object(sr, "_find_conda", return_value=None):
            runner = sr.DLCSubprocessRunner(dlc_conda_env="dlc")
            self.assertTrue(runner._fell_back_to_mock)


# ---------------------------------------------------------------------------
# CAS-P0#2 — 3D Triangulation crash on 1-camera calibration
# ---------------------------------------------------------------------------
class TestCasP0_2TriangulationOneCamera(unittest.TestCase):
    """CAS reported an IndexError deep in numpy.linalg.lstsq when only
    one camera is calibrated.  Fix: ``_check_two_cameras`` raises a
    clear ``ValueError`` at every public entry point.
    """

    def test_dlt_triangulate_one_camera_raises(self) -> None:
        from deepgait3.core._legacy import triangulation_3d as t3
        P = np.eye(3, 4)
        with self.assertRaises(ValueError) as cm:
            t3.dlt_triangulate([P], [np.zeros(2)])
        self.assertIn(">= 2 cameras", str(cm.exception))

    def test_triangulate_ransac_one_camera_raises(self) -> None:
        from deepgait3.core._legacy import triangulation_3d as t3
        P = np.eye(3, 4)
        with self.assertRaises(ValueError) as cm:
            t3.triangulate_ransac([P], [np.zeros(2)])
        self.assertIn(">= 2 cameras", str(cm.exception))

    def test_optim_points_one_camera_raises(self) -> None:
        from deepgait3.core._legacy import triangulation_3d as t3
        P = np.eye(3, 4)
        init_3d = np.zeros((3, 3))
        bones = [(0, 1, 1.0), (1, 2, 1.0)]
        with self.assertRaises(ValueError) as cm:
            t3.optim_points(
                [P],
                [np.zeros((3, 2))],
                init_3d,
                bones,
            )
        self.assertIn(">= 2 cameras", str(cm.exception))

    def test_check_two_cameras_mismatched_lengths_raises(self) -> None:
        from deepgait3.core._legacy import triangulation_3d as t3
        P = np.eye(3, 4)
        with self.assertRaises(ValueError) as cm:
            t3._check_two_cameras([P, P, P], [np.zeros(2), np.zeros(2)])
        self.assertIn("must match", str(cm.exception))


# ---------------------------------------------------------------------------
# EMBL-P0#1 — UnicodeDecodeError on non-ASCII paths
# ---------------------------------------------------------------------------
class TestEmblP0_1Utf8Encoding(unittest.TestCase):
    """EMBL reported a ``UnicodeDecodeError: 'charmap' codec`` on
    Windows usernames with umlauts.  Fix: ``subprocess.Popen`` is now
    invoked with ``encoding='utf-8', errors='replace'``.
    """

    def test_popen_uses_utf8(self) -> None:
        from deepgait3.hardware.dlc import subprocess_runner as sr
        captured: dict[str, object] = {}
        real_popen = sr.subprocess.Popen

        def fake_popen(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            captured["encoding"] = kwargs.get("encoding")
            captured["errors"] = kwargs.get("errors")
            # Return a stub object that mimics Popen for the rest of
            # the call chain (we never actually .wait() it because
            # _fell_back_to_mock is True).
            m = mock.MagicMock()
            m.stdout = None
            m.stderr = None
            m.returncode = 0
            return m

        with mock.patch.object(sr, "_find_conda", return_value=None), \
             mock.patch.object(sr.subprocess, "Popen", side_effect=fake_popen):
            runner = sr.DLCSubprocessRunner(dlc_conda_env="dlc")
            # When conda is missing we go to mock fallback BEFORE
            # Popen; verify the runner code path that does call Popen
            # would pass encoding='utf-8' by inspecting _build_command
            # and calling _run_subprocess directly.
            runner._fell_back_to_mock = False
            runner._run_subprocess([sys.executable, "-c", "pass"])
        self.assertEqual(captured.get("encoding"), "utf-8")
        self.assertEqual(captured.get("errors"), "replace")

    def test_read_dlc_csv_forces_utf8(self) -> None:
        """Even with a cp1252 filesystem locale the CSV reader must
        accept a UTF-8 DeepLabCut export."""
        from deepgait3.core._legacy.gait_io import read_dlc_csv
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "test_embl_p0_1_dlc.csv"
            self._write_minimal_dlc_csv(str(tmp))
            with mock.patch("locale.getpreferredencoding", return_value="cp1252"):
                data = read_dlc_csv(tmp, bodyparts_list=["Nose"])
            # Should successfully read at least one bodypart.
            self.assertIn("Nose", data)
            self.assertEqual(data["Nose"].shape, (5, 3))

    def _write_minimal_dlc_csv(self, path: str) -> str:
        # Build a minimal 3-row-header DLC CSV in-memory.
        bodyparts = ["Nose"]
        scorer = "alice"
        # 1 bodypart × 3 coords (x, y, likelihood) = 3 columns
        header_top = [scorer, scorer, scorer]
        header_mid = ["Nose", "Nose", "Nose"]
        header_bot = ["x", "y", "likelihood"]
        rows = []
        for frame in range(5):
            rows.append([100.0 + frame, 200.0, 0.99])
        df = pd.DataFrame(rows)
        # Replace columns with a MultiIndex
        df.columns = pd.MultiIndex.from_arrays(
            [header_top, header_mid, header_bot],
            names=["scorer", "bodypart", "coords"],
        )
        df.to_csv(path)
        return path


# ---------------------------------------------------------------------------
# PKU-P0#1 — Empty / missing / malformed DLC CSV
# ---------------------------------------------------------------------------
class TestPkuP0_1EmptyDlcCsv(unittest.TestCase):
    """PKU reported ``KeyError: 'scorer'`` in ``gait_io.read_dlc_csv``
    when the user picked an empty / cancelled file.  Fix: the function
    now validates the file and raises a friendly ``ValueError``.
    """

    def test_missing_file_raises_friendly_error(self) -> None:
        from deepgait3.core._legacy.gait_io import read_dlc_csv
        with self.assertRaises(FileNotFoundError) as cm:
            read_dlc_csv("/tmp/this_file_definitely_does_not_exist_xyz_123.csv")
        self.assertIn("not found", str(cm.exception).lower())

    def test_empty_file_raises_friendly_error(self) -> None:
        from deepgait3.core._legacy.gait_io import read_dlc_csv
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "empty_dlc.csv"
            tmp.write_text("")
            with self.assertRaises(ValueError) as cm:
                read_dlc_csv(tmp)
            self.assertIn("empty", str(cm.exception).lower())

    def test_malformed_header_raises_friendly_error(self) -> None:
        from deepgait3.core._legacy.gait_io import read_dlc_csv
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "malformed_dlc.csv"
            tmp.write_text("col1,col2\n1,2\n")  # not a 3-row header
            with self.assertRaises(ValueError) as cm:
                read_dlc_csv(tmp)
            self.assertIn("DeepLabCut", str(cm.exception))


# ---------------------------------------------------------------------------
# P1#1 — pyqtgraph X-axis formatter
# ---------------------------------------------------------------------------
class TestPkulP1AxisFormatter(unittest.TestCase):
    """PKU reported X-axis labels like 0.05000001.  Fix: a
    ``tickStrings`` formatter rounds to 2 decimals.
    """

    def test_format_time_axis_rounds(self) -> None:
        # Import the helper from gait_tab; this guards against
        # accidental removal of the formatter.
        from deepgait3.gui.gait_tab import _format_time_axis
        out = _format_time_axis([0.05, 0.05000001, 1.234567, 60.0])
        self.assertEqual(out, ["0.05", "0.05", "1.23", "60.00"])


# ---------------------------------------------------------------------------
# P2 — compute_angles returns NaN for missing joints; RMS 2 decimals
# ---------------------------------------------------------------------------
class TestP2ComputeAngles(unittest.TestCase):
    def test_compute_angles_returns_nan_for_missing_joints(self) -> None:
        from deepgait3.core._legacy.anipose_wrapper import AniposeWrapper
        wrapper = AniposeWrapper()
        # 5 frames, 3 joints: child, parent, grandparent.
        keypoints = np.zeros((5, 3, 3))
        # Set the child and parent for frames 0..2, leave frames 3..4 NaN.
        keypoints[0:3, 0] = [0.0, 0.0, 0.0]
        keypoints[0:3, 1] = [1.0, 0.0, 0.0]
        keypoints[0:3, 2] = [1.0, 1.0, 0.0]
        keypoints[3:5] = np.nan
        skeleton = [("parent", "child"), ("gp", "parent")]
        angles = wrapper.compute_angles(
            keypoints,
            skeleton,
            bodyparts=["child", "parent", "gp"],
        )
        arr = angles["child"]
        # Frames 0..2 should be 90 deg, frames 3..4 should be NaN (not 0).
        self.assertTrue(np.isnan(arr[3]))
        self.assertTrue(np.isnan(arr[4]))
        self.assertAlmostEqual(arr[0], 90.0, places=3)


class TestP2TriangulationRmsDisplay(unittest.TestCase):
    def test_reproj_rms_px_display_two_decimals_in_status_text(self) -> None:
        """The status label must show the RMS with 2 decimal places
        (the bug was 0.12345678901234568 px).  The display formatting
        lives in triangulation_3d_tab; this test ensures the format
        spec is ``:.2f`` (CAS-P2#4).
        """
        # We don't import the GUI module to avoid PySide6 import cost
        # in a headless unit test; instead check the regex of the
        # source file.  This catches the regression directly.
        src_path = Path("deepgait/gui/triangulation_3d_tab.py")
        text = src_path.read_text(encoding="utf-8")
        # The status label must use {:.2f} (2 decimals) not {:.3f} or
        # default repr.
        self.assertIn("reproj_rms_px:.2f", text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
