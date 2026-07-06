"""Main window for the deepgait v2 GUI (PySide6).

``QMainWindow`` with a ``QTabWidget`` containing 8 tabs:

    1. 初始化     (InitializationTab)  — multi-camera config + live preview
    2. 步态分析   (GaitTab)            — DLC CSV → gait_algorithms → table
    3. FTIR 分析  (FTIRTab)            — footprint_v2 + 4-paw diagram
    4. DLC 工作流 (DLCTab)             — 5-step DLC wizard
    5. 3D 标定    (Calibration3DTab)   — ChArUco intrinsic + extrinsic
    6. 3D 三角化  (Triangulation3DTab) — DLT+RANSAC 3D pose
    7. 步态编辑   (EditorTab)          — in_stance manual editing
    8. 文献图表   (ChartsTab)          — publication-quality plots

The previously-named "相机采集" tab (CameraTab) was renamed to
"初始化" (InitializationTab) in W17 and moved from position 8 to
position 1 so users see the hardware setup first. The old ``CameraTab``
class is still importable as a thin alias (see ``camera_tab.py``).

All tabs share a single :class:`AppState` (Qt signals). The legacy
attributes (``current_gait_results`` etc.) are kept in sync so existing
Phase 0 tests continue to pass.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.3 W9):
    "8 tab 切换无崩溃"
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from deepgait3.core._legacy import results as core_results
from deepgait3.gui.shared_state import (
    AppState,
    CalibrationView,
    GaitResultsView,
    Pose3DResultsView,
)
from deepgait3.gui.style import apply_style


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Primary application window with 8 tabs + shared AppState."""

    # Tab order — must match docs/DEVELOPMENT_PLAN §6.3.
    # W17: "initialization" (formerly "camera") is now position 0.
    TAB_ORDER = (
        "initialization",
        "project", "experiment", "analysis",
        "gait", "ftir", "dlc",
        "calibration_3d", "triangulation_3d",
        "editor", "charts",
    )

    # Backward-compat alias: code that used the old key still resolves.
    TAB_KEY_ALIASES = {
        "camera": "initialization",
    }

    def __init__(self) -> None:
        super().__init__()
        # Force matplotlib to bind to PySide6 (NOT PyQt6) before any
        # chart widget is constructed. Set here as well as in conftest.py
        # so the GUI works when launched standalone (without pytest).
        os.environ.setdefault("QT_API", "pyside6")

        self.setWindowTitle("deepgait — 小鼠步态分析仪 v2.0")
        self.setMinimumSize(1280, 820)

        # Shared application state — single source of truth.
        self.app_state = AppState()

        # Legacy attributes (kept for backward compatibility with
        # Phase 0 tests; synced from AppState signals).
        self.current_gait_results: Optional[core_results.GaitResults] = None
        self.current_ftir_footprints: Optional[list] = None
        self.current_ftir_intensities: Optional[list] = None

        # Tab widget (created lazily by _build_ui).
        self.tabs: Optional[QTabWidget] = None
        self._tab_widgets: dict[str, QWidget] = {}

        self._build_ui()
        self._build_menu()
        self._build_status()
        self._wire_state()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.tabs = QTabWidget()
        # Tabs are created lazily so a single import failure (e.g.
        # opencv missing) doesn't crash the whole window.
        # W17: initialization tab (formerly "camera" at position 7)
        # is now first so the user lands on hardware setup.
        self._build_initialization_tab()
        self._build_project_tab()
        self._build_experiment_tab()
        self._build_analysis_tab()
        self._build_gait_tab()
        self._build_ftir_tab()
        self._build_dlc_tab()
        self._build_calibration_3d_tab()
        self._build_triangulation_3d_tab()
        self._build_editor_tab()
        self._build_charts_tab()

        layout.addWidget(self.tabs)

    def _build_gait_tab(self) -> None:
        from deepgait3.gui.gait_tab import GaitTab
        try:
            tab = GaitTab()
        except TypeError:
            # Newer signature expects AppState; fall back to legacy.
            tab = GaitTab(parent=None)
        self._tab_widgets["gait"] = tab
        self.tabs.addTab(tab, "步态分析")

    def _build_ftir_tab(self) -> None:
        from deepgait3.gui.ftir_tab import FTIRTab
        tab = FTIRTab()
        self._tab_widgets["ftir"] = tab
        self.tabs.addTab(tab, "FTIR 分析")

    def _build_dlc_tab(self) -> None:
        from deepgait3.gui.dlc_tab import DLCTab
        tab = DLCTab()
        self._tab_widgets["dlc"] = tab
        self.tabs.addTab(tab, "DLC 工作流")

    def _build_calibration_3d_tab(self) -> None:
        from deepgait3.gui.calibration_3d_tab import Calibration3DTab
        tab = Calibration3DTab(self.app_state)
        self._tab_widgets["calibration_3d"] = tab
        self.tabs.addTab(tab, "3D 标定")

    def _build_triangulation_3d_tab(self) -> None:
        from deepgait3.gui.triangulation_3d_tab import Triangulation3DTab
        tab = Triangulation3DTab(self.app_state)
        self._tab_widgets["triangulation_3d"] = tab
        self.tabs.addTab(tab, "3D 三角化")

    def _build_editor_tab(self) -> None:
        from deepgait3.gui.editor_tab import EditorTab
        tab = EditorTab()
        self._tab_widgets["editor"] = tab
        self.tabs.addTab(tab, "步态编辑")

    def _build_charts_tab(self) -> None:
        from deepgait3.gui.charts_tab import ChartsTab
        tab = ChartsTab()
        self._tab_widgets["charts"] = tab
        self.tabs.addTab(tab, "文献图表")

    def _build_initialization_tab(self) -> None:
        """Tab 1: multi-camera configuration + live preview (W17)."""
        from deepgait3.gui.initialization_tab import InitializationTab
        tab = InitializationTab(app_state=self.app_state)
        self._tab_widgets["initialization"] = tab
        # Backward-compat alias: "camera" still resolves to this tab.
        self._tab_widgets["camera"] = tab
        self.tabs.addTab(tab, "初始化")

    def _build_project_tab(self) -> None:
        """Tab 2: 新建项目 — project creation + recent-projects browser."""
        from deepgait3.gui.gait_project_tab import GaitProjectTab
        tab = GaitProjectTab()
        tab.project_opened.connect(self._on_project_opened)
        self._tab_widgets["project"] = tab
        self.tabs.addTab(tab, "新建项目")

    def _build_experiment_tab(self) -> None:
        """Tab 3: 新建实验 — C1 live acquisition + footprint preview."""
        from deepgait3.gui.gait_experiment_tab import GaitExperimentTab
        tab = GaitExperimentTab()
        tab.set_app_state(self.app_state)
        self._tab_widgets["experiment"] = tab
        self.tabs.addTab(tab, "新建实验")

    def _build_analysis_tab(self) -> None:
        """Tab 4: 数据分析 — batch processing + pressure heatmap."""
        from deepgait3.gui.gait_analysis_tab import GaitAnalysisTab
        tab = GaitAnalysisTab()
        tab.set_app_state(self.app_state)
        self._tab_widgets["analysis"] = tab
        self.tabs.addTab(tab, "数据分析")

    # ------------------------------------------------------------------
    # Menu / status
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件")
        file_menu.addAction("打开步态 CSV…", self._on_open_csv)
        file_menu.addAction("打开 FTIR 图像…", self._on_open_image)
        file_menu.addAction("打开标定目录…", self._on_open_calibration_dir)
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close)

        help_menu = menu.addMenu("帮助")
        help_menu.addAction("关于 deepgait", self._on_about)

    def _build_status(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")
        self.status.addPermanentWidget(QLabel("deepgait v2.0"))

    # ------------------------------------------------------------------
    # AppState wiring — sync signals to legacy attrs + status bar
    # ------------------------------------------------------------------
    def _on_project_opened(self, cfg) -> None:
        """Route project-open events from GaitProjectTab to AppState."""
        from deepgait3.gui.shared_state import GaitProjectView
        view = GaitProjectView(
            project_name=cfg.project_name,
            project_path=str(cfg.project_path),
            experimenter=cfg.experimenter,
            px_per_mm=cfg.px_per_mm,
        )
        self.app_state.set_project(view)
        self.status.showMessage(
            f"已打开项目: {cfg.project_name}", 5000,
        )

    def _wire_state(self) -> None:
        self.app_state.status_message_changed.connect(self.status.showMessage)

    # ------------------------------------------------------------------
    # Menu handlers
    # ------------------------------------------------------------------
    def _on_open_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择步态 CSV", "", "CSV 文件 (*.csv);;所有文件 (*.*)",
        )
        if not path:
            return
        self.tabs.setCurrentWidget(self._tab_widgets["gait"])
        gait_tab = self._tab_widgets["gait"]
        # New-style tabs expose load_csv; legacy tabs don't (just update
        # their path field).
        if hasattr(gait_tab, "load_csv"):
            try:
                gait_tab.load_csv(path)
            except Exception as e:
                QMessageBox.critical(self, "加载失败", f"{e}")
        elif hasattr(gait_tab, "path_edit"):
            gait_tab.path_edit.setText(path)
            if hasattr(gait_tab, "analyze_btn"):
                gait_tab.analyze_btn.setEnabled(True)
        self.app_state.set_status_message(f"已加载 CSV: {path}")

    def _on_open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 FTIR 图像", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif);;所有文件 (*.*)",
        )
        if not path:
            return
        self.tabs.setCurrentWidget(self._tab_widgets["ftir"])
        ftir_tab = self._tab_widgets["ftir"]
        if hasattr(ftir_tab, "path_edit"):
            ftir_tab.path_edit.setText(path)
            if hasattr(ftir_tab, "analyze_btn"):
                ftir_tab.analyze_btn.setEnabled(True)
        self.app_state.set_status_message(f"已加载图像: {path}")

    def _on_open_calibration_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择标定根目录", "",
        )
        if not path:
            return
        self.tabs.setCurrentWidget(self._tab_widgets["calibration_3d"])
        calib_tab = self._tab_widgets["calibration_3d"]
        if hasattr(calib_tab, "path_edit"):
            calib_tab.path_edit.setText(path)
            calib_tab._images_root = path  # type: ignore[attr-defined]
            if hasattr(calib_tab, "calibrate_btn"):
                calib_tab.calibrate_btn.setEnabled(True)
        self.app_state.set_status_message(f"已选择标定目录: {path}")

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "关于 deepgait",
            "<h2>deepgait 小鼠步态分析仪 v2.0</h2>"
            "<p>4 路海康 GigE 相机 + DeepLabCut + 3D 三角化 + FTIR 脚印</p>"
            "<p>PySide6 (LGPL) · 闭源商业版</p>"
            "<p>v2.0 — 2026</p>",
        )

    # ------------------------------------------------------------------
    # Convenience accessors for tests
    # ------------------------------------------------------------------
    def tab_by_name(self, name: str) -> Optional[QWidget]:
        """Return the QWidget for the named tab, or None if not built.

        Accepts both the new key ("initialization") and the legacy
        alias ("camera") so the test suite and external code that used
        the old name continue to work after the W17 rename + reorder.
        """
        if name in self._tab_widgets:
            return self._tab_widgets[name]
        # Honor legacy alias ("camera" → "initialization").
        canonical = self.TAB_KEY_ALIASES.get(name, name)
        return self._tab_widgets.get(canonical)

    @property
    def n_tabs(self) -> int:
        return self.tabs.count() if self.tabs else 0


def launch_gui(argv: list[str] | None = None) -> int:
    """Entry point: create QApplication and show MainWindow.

    Platform selection:
      * CI / headless runs set ``QT_QPA_PLATFORM=offscreen`` (or call
        with ``--offscreen``) — this is the default for unit tests.
      * Normal desktop runs honour ``$DISPLAY`` / ``$WAYLAND_DISPLAY``
        and pick ``xcb`` (X11) or ``wayland`` automatically.
      * Users can force ``--platform offscreen`` for a hidden window
        (useful for screenshots / smoke tests on a real display).

    Debug mode (``--debug``):
        Injects a ``DebugMonitor`` that streams structured JSON events
        to stderr (default) or a log file (``--log PATH``).  Every
        AppState signal emission, tab switch, menu action, and filtered
        user event is logged so the backend data flow is observable.
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--offscreen", action="store_true",
                        help="force offscreen QPA platform")
    parser.add_argument("--platform", default=None,
                        help="explicit QPA platform (e.g. xcb, wayland)")
    parser.add_argument("--debug", action="store_true",
                        help="enable real-time debug event logging")
    parser.add_argument("--log", default=None, metavar="PATH",
                        help="write debug log to file (requires --debug)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress stderr debug output (use with --log)")
    args, _ = parser.parse_known_args(argv if argv else sys.argv[1:])

    os.environ.setdefault("QT_API", "pyside6")
    if args.offscreen or os.environ.get("DEEPGAIT_OFFSCREEN") == "1":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    elif args.platform:
        os.environ["QT_QPA_PLATFORM"] = args.platform
    # else: leave QT_QPA_PLATFORM unset so Qt auto-detects.

    app = QApplication(sys.argv)
    apply_style(app)
    win = MainWindow()

    # ------------------------------------------------------------------
    # Debug monitor setup
    # ------------------------------------------------------------------
    monitor: object = None
    if args.debug:
        from deepgait3.gui.debug_monitor import (
            DebugMonitor,
            patch_appstate_setters,
        )

        # Choose output: file (--log) and/or stderr (unless --quiet).
        import sys as _sys
        if args.log:
            log_fh = open(args.log, "w", encoding="utf-8")
            if args.quiet:
                monitor_out = log_fh
            else:
                # Tee to both stderr and file.
                class _Tee:
                    def __init__(self, *files):
                        self._files = files
                    def write(self, text):
                        for f in self._files:
                            f.write(text)
                    def flush(self):
                        for f in self._files:
                            f.flush()
                monitor_out = _Tee(_sys.stderr, log_fh)
        else:
            monitor_out = _sys.stderr

        monitor = DebugMonitor(
            win.app_state,
            output=monitor_out,
            verbose=True,  # --debug = full events
        )
        patch_appstate_setters(win.app_state, monitor)

        # Hook tab switches.
        if win.tabs is not None:
            win.tabs.currentChanged.connect(
                lambda idx: monitor.log_tab_switch(win.tabs, idx)
            )

        # Hook menu actions — iterate the menu bar.
        menu_bar = win.menuBar()
        if menu_bar is not None:
            for menu in menu_bar.findChildren(QMenu):
                menu.aboutToShow.connect(
                    lambda m=menu: monitor.log_menu_action(
                        f"menu-open: {m.title()}"
                    )
                )
                for action in menu.actions():
                    if action.text():
                        action.triggered.connect(
                            lambda checked=False, a=action:
                            monitor.log_menu_action(a.text())
                        )

        _sys.stderr.write(
            f"[DEBUG] monitor active (pid={os.getpid()})\n"
        )
        _sys.stderr.flush()

        # Store on the window so tests can introspect.
        win._debug_monitor = monitor  # type: ignore[attr-defined]

    win.show()
    return app.exec()


__all__ = ["MainWindow", "launch_gui"]