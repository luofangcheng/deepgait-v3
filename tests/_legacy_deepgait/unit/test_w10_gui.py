"""Unit tests for Phase 3 W10 — GaitTab pyqtgraph + FTIRTab footprint_v2.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.3 W10):
    "GaitTab + FTIRTab" / "步态表格 + FTIR 实时可视化" / "UI ≥ 30 fps"

Covers:
  * :class:`StanceSwingSequenceChart` (pyqtgraph) — stance/swing + stride histogram.
  * :class:`FTIRTab` (footprint_v2 + pyqtgraph ImageView) — full pipeline.
  * :class:`FTIRWorker` migrated to ``footprint_v2.analyze_frame_v2``.
  * 30 fps interactive update gate (synthetic).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytest


# =============================================================================
# Fixtures
# =============================================================================
@pytest.fixture
def main_window(qtbot):
    from deepgait3.gui.main_window import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def _synthetic_ftir_frame(n_paws: int = 4, w: int = 320, h: int = 240) -> np.ndarray:
    """Build a synthetic FTIR frame with N green paw discs on dark background."""
    frame = np.full((h, w, 3), 12, dtype=np.uint8)
    positions = [(60, 100, 14), (60, 50, 14),
                 (260, 100, 14), (260, 50, 14)][:n_paws]
    for cx, cy, r in positions:
        cv2.circle(frame, (cx, cy), r, (0, 220, 0), -1)
    return frame


# =============================================================================
# StanceSwingSequenceChart — pyqtgraph interactive chart
# =============================================================================
class TestStanceSwingSequenceChart:
    def test_constructs_without_crash(self, qtbot):
        from deepgait3.gui.gait_tab import StanceSwingSequenceChart
        chart = StanceSwingSequenceChart()
        qtbot.addWidget(chart)

    def test_update_stance_renders(self, qtbot):
        from deepgait3.gui.gait_tab import StanceSwingSequenceChart
        chart = StanceSwingSequenceChart()
        qtbot.addWidget(chart)
        in_stance = {"LF": np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], dtype=np.int8),
                     "RF": np.zeros(10, dtype=np.int8),
                     "LH": np.zeros(10, dtype=np.int8),
                     "RH": np.zeros(10, dtype=np.int8)}
        chart.update_stance(in_stance)
        assert chart.getPlotItem() is not None

    def test_reset_clears(self, qtbot):
        from deepgait3.gui.gait_tab import StanceSwingSequenceChart
        chart = StanceSwingSequenceChart()
        qtbot.addWidget(chart)
        in_stance = {"LF": np.array([1, 0, 1], dtype=np.int8),
                     "RF": np.zeros(3, dtype=np.int8),
                     "LH": np.zeros(3, dtype=np.int8),
                     "RH": np.zeros(3, dtype=np.int8)}
        chart.update_stance(in_stance)
        chart.reset()  # should not crash

    def test_empty_arrays_dont_crash(self, qtbot):
        from deepgait3.gui.gait_tab import StanceSwingSequenceChart
        chart = StanceSwingSequenceChart()
        qtbot.addWidget(chart)
        chart.update_stance({"LF": np.array([], dtype=np.int8),
                            "RF": np.array([], dtype=np.int8),
                            "LH": np.array([], dtype=np.int8),
                            "RH": np.array([], dtype=np.int8)})


# =============================================================================
# GaitTab — pyqtgraph integration
# =============================================================================
class TestGaitTabPyqtgraph:
    def test_seq_chart_attribute_exists(self, qtbot, main_window):
        gait_tab = main_window.tab_by_name("gait")
        qtbot.addWidget(gait_tab)
        from deepgait3.gui.gait_tab import StanceSwingSequenceChart
        assert isinstance(gait_tab.seq_chart, StanceSwingSequenceChart)

    def test_seq_chart_handles_stance(self, qtbot, main_window):
        """Sequence chart should handle stance update without crash."""
        gait_tab = main_window.tab_by_name("gait")
        qtbot.addWidget(gait_tab)
        in_stance = {"LF": np.array([1, 0, 1], dtype=np.int8),
                     "RF": np.array([0, 1, 0], dtype=np.int8),
                     "LH": np.array([0, 0, 0], dtype=np.int8),
                     "RH": np.array([0, 0, 0], dtype=np.int8)}
        gait_tab.seq_chart.update_stance(in_stance)
        assert gait_tab.seq_chart.getPlotItem() is not None


# =============================================================================
# FTIRTab — footprint_v2 + pyqtgraph ImageView
# =============================================================================
class TestFTIRTabV2:
    def test_constructs_without_crash(self, qtbot):
        from deepgait3.gui.ftir_tab import FTIRTab
        tab = FTIRTab()
        qtbot.addWidget(tab)
        assert tab._footprints == []
        assert tab._sequence is None

    def test_has_pyqtgraph_image_views(self, qtbot):
        from deepgait3.gui.ftir_tab import FTIRTab
        import pyqtgraph as pg
        tab = FTIRTab()
        qtbot.addWidget(tab)
        assert isinstance(tab.image_view, pg.ImageView)
        assert isinstance(tab.mask_view, pg.ImageView)

    def test_hsv_sliders_retain_legacy_api(self, qtbot):
        """Legacy tests poke ``tab.h_low`` etc. — must still work."""
        from deepgait3.gui.ftir_tab import FTIRTab
        tab = FTIRTab()
        qtbot.addWidget(tab)
        tab.h_low.setValue(50)
        assert tab.h_low.value() == 50

    def test_update_previews_renders_to_imageview(self, qtbot):
        from deepgait3.gui.ftir_tab import FTIRTab
        tab = FTIRTab()
        qtbot.addWidget(tab)
        tab._frame = _synthetic_ftir_frame()
        tab._update_previews()
        # ImageView must now have a non-trivial image.
        assert tab.image_view.image is not None
        assert tab.mask_view.image is not None

    def test_analyze_runs_footprint_v2_pipeline(self, qtbot, tmp_path):
        """End-to-end: load image → analyze → table populated with FootMask."""
        from deepgait3.gui.ftir_tab import FTIRTab
        frame = _synthetic_ftir_frame(n_paws=4)
        img_path = tmp_path / "ftir.png"
        cv2.imwrite(str(img_path), frame)

        tab = FTIRTab()
        qtbot.addWidget(tab)
        tab._frame = frame
        tab.analyze_btn.setEnabled(True)
        qtbot.mouseClick(tab.analyze_btn, __import__("PySide6").QtCore.Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: len(tab._footprints) > 0, timeout=5000)

        assert tab._sequence is not None
        assert len(tab._footprints) >= 1
        # Table must have the v2 columns populated.
        assert tab.table.rowCount() == len(tab._footprints)
        assert tab.export_btn.isEnabled()
        # Each FootMask must have v2 fields.
        for fm in tab._footprints:
            assert hasattr(fm, "hull_area_px")
            assert hasattr(fm, "is_in_stance")
            assert hasattr(fm, "intensity_mean")

    def test_export_writes_csv_with_v2_fields(self, qtbot, tmp_path, monkeypatch):
        from deepgait3.gui.ftir_tab import FTIRTab
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        frame = _synthetic_ftir_frame()
        img_path = tmp_path / "ftir.png"
        cv2.imwrite(str(img_path), frame)

        tab = FTIRTab()
        qtbot.addWidget(tab)
        tab._frame = frame
        tab.analyze_btn.setEnabled(True)
        qtbot.mouseClick(tab.analyze_btn, __import__("PySide6").QtCore.Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: len(tab._footprints) > 0, timeout=5000)

        out_csv = tmp_path / "out.csv"
        # Stub both the file dialog AND the success message box — the
        # latter would otherwise block the event loop in headless mode.
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(out_csv), "")),
        )
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
        tab._on_export()
        assert out_csv.is_file()
        text = out_csv.read_text()
        assert "hull_area_px" in text
        assert "is_in_stance" in text

    def test_result_updates_main_window_shared_state(self, qtbot, main_window, tmp_path):
        """FTIR analysis must update ``main_window.current_ftir_footprints``."""
        from deepgait3.gui.ftir_tab import FTIRTab
        frame = _synthetic_ftir_frame()
        img_path = tmp_path / "ftir.png"
        cv2.imwrite(str(img_path), frame)

        tab = FTIRTab()
        # W17: FTIR tab moved from position 1 to position 2 (initialization
        # tab is now first).
        main_window.tabs.removeTab(2)
        main_window.tabs.insertTab(2, tab, "FTIR 分析")
        qtbot.addWidget(tab)

        tab._frame = frame
        tab.analyze_btn.setEnabled(True)
        qtbot.mouseClick(tab.analyze_btn, __import__("PySide6").QtCore.Qt.MouseButton.LeftButton)
        qtbot.waitUntil(lambda: main_window.current_ftir_footprints is not None,
                        timeout=5000)
        assert len(main_window.current_ftir_footprints) >= 1


# =============================================================================
# FTIRWorker — v2 migration
# =============================================================================
class TestFTIRWorkerV2:
    def test_signal_emits_footprint_sequence(self, qtbot):
        """FTIRWorker.result_ready must emit a FootprintSequence, not a tuple."""
        from deepgait3.core._legacy.footprint_v2 import FootprintSequence
        from deepgait3.gui.workers import FTIRWorker
        frame = _synthetic_ftir_frame(n_paws=2)

        received: list = []
        worker = FTIRWorker(frame=frame)
        worker.result_ready.connect(received.append)
        worker.start()
        qtbot.waitUntil(lambda: len(received) > 0, timeout=5000)
        assert isinstance(received[0], FootprintSequence)
        assert received[0].n_feet >= 1

    def test_error_signal_on_bad_frame(self, qtbot):
        from deepgait3.gui.workers import FTIRWorker
        worker = FTIRWorker(frame=np.zeros((10, 10), dtype=np.uint8))
        errors: list = []
        worker.error.connect(errors.append)
        worker.start()
        qtbot.waitUntil(lambda: len(errors) > 0 or not worker.isRunning(),
                        timeout=5000)
        # Either error fired or worker exited (graceful).
        assert worker.isRunning() is False


# =============================================================================
# W10 acceptance gate: UI ≥ 30 fps (interactive update)
# =============================================================================
@pytest.mark.performance
class TestW10FpsGate:
    def test_imageview_refresh_under_33ms(self, qtbot):
        """30 fps = 33.3 ms per frame. Refreshing the ImageView must take
        less than that on a synthetic frame."""
        from deepgait3.gui.ftir_tab import FTIRTab
        tab = FTIRTab()
        qtbot.addWidget(tab)
        tab._frame = _synthetic_ftir_frame()
        # Warm-up.
        tab._update_previews()
        # Timed loop.
        durations = []
        for _ in range(10):
            t0 = time.perf_counter()
            tab._update_previews()
            durations.append(time.perf_counter() - t0)
        mean_ms = 1000.0 * sum(durations) / len(durations)
        assert mean_ms < 33.3, f"refresh {mean_ms:.2f} ms > 33.3 ms (30 fps)"

    def test_gait_chart_refresh_under_33ms(self, qtbot):
        """StanceSwingSequenceChart re-render must also stay under the 30 fps gate."""
        from deepgait3.gui.gait_tab import StanceSwingSequenceChart
        chart = StanceSwingSequenceChart()
        qtbot.addWidget(chart)
        in_stance = np.random.default_rng(0).integers(0, 2, size=500)
        # Warm-up.
        chart.update_stance({"LF": in_stance,
                             "RF": np.zeros_like(in_stance),
                             "LH": np.zeros_like(in_stance),
                             "RH": np.zeros_like(in_stance)})
        durations = []
        for _ in range(10):
            t0 = time.perf_counter()
            chart.update_stance({"LF": in_stance,
                                 "RF": np.zeros_like(in_stance),
                                 "LH": np.zeros_like(in_stance),
                                 "RH": np.zeros_like(in_stance)})
            durations.append(time.perf_counter() - t0)
        mean_ms = 1000.0 * sum(durations) / len(durations)
        assert mean_ms < 33.3, f"chart refresh {mean_ms:.2f} ms > 33.3 ms"