"""Tab 3: 数据采集 (Data Acquisition) — deepgait v3.

Responsibilities
----------------
- Drive the 4 cameras (C1 bottom FTIR 1920×384 + 3 top DLC 1024×1024)
  via :class:`MultiCameraManager`.
- Show 1 wide preview (C1 bottom, 5:1 aspect) + 4×1 preview strip
  for the 4 cameras using :class:`FourCamStrip` (pyqtgraph
  ``ImageView`` GPU texture rendering, per-camera FPS label).
- Record all 4 streams under
  ``<project>/rawdata/videos/<animal_id>/<session_ts>/<role>.mp4``
  (per-session subdirectory enforces the 4-camera integrity rule).
- Display per-camera hardware FPS (sliding-window estimate) and a
  global elapsed acquisition time ``HH:MM:SS``.
- Browse already-recorded sessions: pick a directory, populate the
  table with one row per session, mark incomplete rows in red and
  disable their selection.

Notes
-----
- This tab does NOT trim — the 数据分析 tab owns trim.
- Cumulative footprint rendering lives in the 数据分析 tab.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepgait3.core.data.session import SessionInfo, SessionStore
from deepgait3.gui.video_constants import BOTTOM_CAMERA_SIZE
from deepgait3.gui.widgets.camera_preview_strip import FourCamStrip
from deepgait3.hardware.camera.base import FrameInfo
from deepgait3.hardware.camera.multi_cam import MultiCameraManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExperimentEntryTable — 5 columns (组别 / 动物编号 / 状态 / 完整性 / 备注)
# ---------------------------------------------------------------------------
class ExperimentEntryTable(QTableWidget):
    """Per-session or per-entry table.

    Columns:
        0 组别, 1 动物编号, 2 状态, 3 完整性, 4 备注
    """

    COL_GROUP = 0
    COL_ANIMAL_ID = 1
    COL_STATUS = 2
    COL_INTEGRITY = 3
    COL_NOTE = 4

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(
            ["组别", "动物编号", "状态", "完整性", "备注"]
        )
        self.horizontalHeader().setSectionResizeMode(
            self.COL_GROUP, QHeaderView.ResizeMode.Interactive,
        )
        self.horizontalHeader().setSectionResizeMode(
            self.COL_ANIMAL_ID, QHeaderView.ResizeMode.Interactive,
        )
        self.horizontalHeader().setSectionResizeMode(
            self.COL_STATUS, QHeaderView.ResizeMode.Stretch,
        )
        self.horizontalHeader().setSectionResizeMode(
            self.COL_INTEGRITY, QHeaderView.ResizeMode.Stretch,
        )
        self.horizontalHeader().setSectionResizeMode(
            self.COL_NOTE, QHeaderView.ResizeMode.Stretch,
        )
        self.setColumnWidth(self.COL_GROUP, 80)
        self.setColumnWidth(self.COL_ANIMAL_ID, 140)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)

    def add_entry(
        self, group: str = "", animal_id: str = "", note: str = "",
    ) -> int:
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, self.COL_GROUP, QTableWidgetItem(group))
        self.setItem(row, self.COL_ANIMAL_ID, QTableWidgetItem(animal_id))
        self.setItem(row, self.COL_STATUS, QTableWidgetItem("待录制"))
        self.setItem(row, self.COL_INTEGRITY, QTableWidgetItem("—"))
        self.setItem(row, self.COL_NOTE, QTableWidgetItem(note))
        return row

    def remove_selected(self) -> None:
        for row in sorted(
            {i.row() for i in self.selectedIndexes()}, reverse=True,
        ):
            self.removeRow(row)

    def entries(self) -> List[Dict[str, str]]:
        out = []
        for row in range(self.rowCount()):
            def text(col: int) -> str:
                item = self.item(row, col)
                return item.text().strip() if item else ""

            out.append({
                "group": text(self.COL_GROUP),
                "animal_id": text(self.COL_ANIMAL_ID),
                "status": text(self.COL_STATUS),
                "integrity": text(self.COL_INTEGRITY),
                "note": text(self.COL_NOTE),
            })
        return out

    def set_status(self, row: int, status: str) -> None:
        if 0 <= row < self.rowCount():
            self.setItem(row, self.COL_STATUS, QTableWidgetItem(status))

    def set_integrity(
        self, row: int, text: str, complete: bool,
        missing: Optional[List[str]] = None,
    ) -> None:
        """Set the integrity column.

        ``complete=True``  → green foreground on a normal cell.
        ``complete=False`` → red foreground + light red background, with
        a tooltip listing the missing roles, and the row made
        unselectable.
        """
        if not (0 <= row < self.rowCount()):
            return
        item = QTableWidgetItem(text)
        if complete:
            item.setForeground(QBrush(QColor("#1B5E20")))
        else:
            item.setForeground(QBrush(QColor("#B71C1C")))
            item.setBackground(QBrush(QColor("#FFEBEE")))
            if missing:
                item.setToolTip(f"缺失: {', '.join(missing)}")
            # Disable selection for the whole row.
            for col in range(self.columnCount()):
                cell = self.item(row, col)
                if cell is None:
                    cell = QTableWidgetItem("")
                    self.setItem(row, col, cell)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEnabled)
        self.setItem(row, self.COL_INTEGRITY, item)


# ---------------------------------------------------------------------------
# GaitExperimentTab — 4-camera live preview + record + session browse
# ---------------------------------------------------------------------------
class GaitExperimentTab(QWidget):
    """数据采集 tab.  4-camera live preview + recording + session browse."""

    ALL_ROLES: tuple = ("bottom", "left", "right", "top")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._app_state = None
        self._cam_mgr: Optional[MultiCameraManager] = None

        # Recording state
        self._is_running = False
        self._is_paused = False
        self._frame_index = 0
        self._current_row = -1
        self._fps = 100.0
        self._video_writers: Dict[str, cv2.VideoWriter] = {}
        self._writers_initialized: bool = False
        self._entry: Optional[Dict[str, str]] = None
        self._ts: str = ""

        # Elapsed time accounting
        self._elapsed_start: Optional[float] = None
        self._elapsed_paused_total: float = 0.0
        self._pause_started_at: Optional[float] = None

        # Preview widgets
        self._cam_strip: Optional[FourCamStrip] = None

        # Timers
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._live_tick)
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._on_sync_tick)
        self._fps_timer = QTimer(self)
        self._fps_timer.timeout.connect(self._update_fps_label)
        self._fps_counter = 0
        self._last_fps_time = time.perf_counter()

        self._build_ui()

    def set_app_state(self, app_state) -> None:
        self._app_state = app_state
        self._update_project_display()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setSpacing(4)
        main.setContentsMargins(6, 6, 6, 6)

        # -- Project info --------------------------------------------------
        self.project_label = QLabel("当前项目: 未打开")
        main.addWidget(self.project_label)

        # -- Data source ---------------------------------------------------
        src_gb = QGroupBox("数据源")
        src_l = QHBoxLayout(src_gb)
        self.connect_btn = QPushButton("连接全部相机")
        self.connect_btn.clicked.connect(self._on_connect_all_cameras)
        src_l.addWidget(self.connect_btn)
        self.load_recorded_btn = QPushButton("加载已录制视频")
        self.load_recorded_btn.clicked.connect(self._on_load_recorded)
        src_l.addWidget(self.load_recorded_btn)
        self.disconnect_btn = QPushButton("断开")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.clicked.connect(self._on_disconnect_all_cameras)
        src_l.addWidget(self.disconnect_btn)
        self.source_status = QLabel("未连接")
        self.source_status.setStyleSheet("color: #888;")
        src_l.addWidget(self.source_status)
        src_l.addStretch()
        main.addWidget(src_gb)

        # -- Entry table ---------------------------------------------------
        table_gb = QGroupBox("实验列表 / 已录制 session")
        tl = QVBoxLayout(table_gb)
        tl.setSpacing(2)
        tl.setContentsMargins(4, 8, 4, 4)
        self.entry_table = ExperimentEntryTable()
        self.entry_table.setMinimumHeight(60)
        self.entry_table.setMaximumHeight(220)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ 添加行")
        add_btn.clicked.connect(lambda: self.entry_table.add_entry())
        btn_row.addWidget(add_btn)
        del_btn = QPushButton("删除选中")
        del_btn.clicked.connect(self.entry_table.remove_selected)
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        tl.addLayout(btn_row)
        tl.addWidget(self.entry_table)
        main.addWidget(table_gb)

        # -- Control buttons ----------------------------------------------
        ctrl_row = QHBoxLayout()
        self.start_btn = QPushButton("▶ 开始采集")
        self.start_btn.clicked.connect(self._on_start)
        ctrl_row.addWidget(self.start_btn)
        self.pause_btn = QPushButton("⏸ 暂停")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self._on_pause)
        ctrl_row.addWidget(self.pause_btn)
        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self.stop_btn)
        self.record_check = QCheckBox("录制视频")
        self.record_check.setChecked(True)
        ctrl_row.addWidget(self.record_check)
        ctrl_row.addStretch()
        self.status_label = QLabel("就绪")
        ctrl_row.addWidget(self.status_label)
        # Elapsed time (HH:MM:SS)
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setStyleSheet(
            "font-family: monospace; font-weight: bold; color: #1565C0;"
        )
        self.elapsed_label.setMinimumWidth(80)
        ctrl_row.addWidget(QLabel("采集时间:"))
        ctrl_row.addWidget(self.elapsed_label)
        main.addLayout(ctrl_row)

        # -- Live preview --------------------------------------------------
        preview_gb = QGroupBox("实时预览")
        preview_l = QVBoxLayout(preview_gb)
        preview_l.setSpacing(4)

        # Top: C1 bottom wide preview (5:1)
        preview_l.addWidget(QLabel("C1 底部 (FTIR 1920×384) 宽幅"))
        self.video_preview = QLabel()
        self.video_preview.setStyleSheet("background: black;")
        self.video_preview.setMinimumSize(960, 192)
        self.video_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )
        self.video_preview.setScaledContents(True)
        preview_l.addWidget(self.video_preview, stretch=1)

        # 4×1 strip for the 4 cameras (pyqtgraph ImageView)
        preview_l.addWidget(QLabel("4 路相机 4×1 (每路带 FPS)"))
        self._cam_strip = FourCamStrip(self)
        preview_l.addWidget(self._cam_strip, stretch=2)
        main.addWidget(preview_gb)

        # -- Footer hint + frame + sync info -----------------------------
        hint = QLabel("注：本 tab 仅采集录制，trim 留到「数据分析」tab")
        hint.setStyleSheet("color: #888; font-style: italic;")
        main.addWidget(hint)
        self.frame_label = QLabel("Frame: 0 | FPS: --")
        self.frame_label.setStyleSheet("font-family: monospace; color: #888;")
        main.addWidget(self.frame_label)
        self.sync_label = QLabel("同步: --")
        self.sync_label.setStyleSheet("font-family: monospace; color: #888;")
        main.addWidget(self.sync_label)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _on_connect_all_cameras(self) -> None:
        if self._cam_mgr is not None:
            return
        try:
            main_win = self.window()
            init_tab = main_win.tab_by_name("initialization")
            cam_dict = init_tab.config_panel.cameras()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法访问初始化 Tab: {e}")
            return
        if not cam_dict:
            QMessageBox.warning(
                self, "错误",
                "未找到任何相机，请先在「初始化」Tab 中点击「检测相机」。",
            )
            return
        roster = [(r, cam_dict[r]) for r in self.ALL_ROLES if r in cam_dict]
        if len(roster) < len(self.ALL_ROLES):
            missing = [r for r in self.ALL_ROLES if r not in cam_dict]
            QMessageBox.warning(
                self, "错误",
                f"缺少相机角色: {', '.join(missing)}。\n"
                f"请到「初始化」Tab 检测全部 4 台相机。",
            )
            return
        try:
            groups = init_tab.config_panel.groups()
            bot = groups.get("bottom")
            if bot is not None and hasattr(bot, "fps_spin"):
                self._fps = float(bot.fps_spin.value() or 100)
        except Exception:
            pass
        try:
            mgr = MultiCameraManager(cameras=roster, trigger_line=0)
            mgr.start_all()
        except Exception as e:
            logger.exception("MultiCameraManager.start_all failed")
            QMessageBox.critical(self, "连接失败", f"无法启动相机: {e}")
            return
        self._cam_mgr = mgr
        self.source_status.setText(
            f"● {len(roster)} 路相机在线 ({self._fps:.0f} fps)"
        )
        self.source_status.setStyleSheet("color: #2E7D32; font-weight: bold;")
        self.connect_btn.setEnabled(False)
        self.disconnect_btn.setEnabled(True)
        self._live_timer.start(33)
        self._sync_timer.start(500)
        self._fps_timer.start(1000)
        logger.info("Connected %d cameras via MultiCameraManager", len(roster))

    def _on_disconnect_all_cameras(self) -> None:
        self._live_timer.stop()
        self._sync_timer.stop()
        if self._is_running:
            self._on_stop()
        if self._cam_mgr is not None:
            try:
                self._cam_mgr.close_all()
            except Exception:
                logger.exception("MultiCameraManager.close_all failed")
            self._cam_mgr = None
        self.source_status.setText("未连接")
        self.source_status.setStyleSheet("color: #888;")
        self.connect_btn.setEnabled(True)
        self.disconnect_btn.setEnabled(False)
        self.sync_label.setText("同步: --")
        self.sync_label.setStyleSheet("font-family: monospace; color: #888;")
        if self._cam_strip is not None:
            self._cam_strip.reset()
        logger.info("Disconnected cameras")

    # ------------------------------------------------------------------
    # Session browse (load recorded videos)
    # ------------------------------------------------------------------
    def _on_load_recorded(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "选择 rawdata/videos 目录",
            str(self._project_raw_videos_dir() or Path.cwd()),
        )
        if not d:
            return
        store = SessionStore(Path(d))
        sessions = store.list_sessions()
        if not sessions:
            QMessageBox.information(
                self, "提示",
                f"未在 {d} 发现任何 session 子目录。\n"
                f"期望布局: <animal_id>/<session_ts>/<role>.mp4",
            )
            return
        self._populate_session_table(sessions)

    def _populate_session_table(self, sessions: List[SessionInfo]) -> None:
        self.entry_table.setRowCount(0)
        for s in sessions:
            row = self.entry_table.add_entry(
                group="",
                animal_id=s.animal_id,
                note=s.session_ts,
            )
            self.entry_table.set_status(row, "已加载")
            if s.is_complete:
                self.entry_table.set_integrity(
                    row, "完整 (4/4)", complete=True,
                )
            else:
                missing = sorted(s.roles_missing)
                self.entry_table.set_integrity(
                    row,
                    f"缺失: {', '.join(missing)}",
                    complete=False,
                    missing=missing,
                )
        self.status_label.setText(
            f"已加载 {len(sessions)} 个 session"
        )

    # ------------------------------------------------------------------
    # Acquisition control
    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self._cam_mgr is None:
            QMessageBox.warning(self, "提示", "请先连接全部相机。")
            return
        entries = self.entry_table.entries()
        row = self.entry_table.currentRow()
        if row < 0:
            row = 0
        if row >= len(entries):
            QMessageBox.warning(self, "提示", "请先添加实验行。")
            return
        entry = entries[row]
        if not entry["animal_id"]:
            QMessageBox.warning(self, "提示", "请填写动物编号。")
            return
        self._current_row = row
        self._entry = entry
        self._is_running = True
        self._is_paused = False
        self._frame_index = 0
        self._writers_initialized = False
        self._ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._video_writers.clear()
        # Reset elapsed-time accounting
        self._elapsed_start = time.perf_counter()
        self._elapsed_paused_total = 0.0
        self._pause_started_at = None
        self.elapsed_label.setText("00:00:00")

        self.entry_table.set_status(row, "采集中...")
        self.status_label.setText(f"采集中: {entry['animal_id']}")
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self._fps_counter = 0
        self._last_fps_time = time.perf_counter()

    def _on_pause(self) -> None:
        self._is_paused = not self._is_paused
        if self._is_paused:
            self._pause_started_at = time.perf_counter()
        else:
            if self._pause_started_at is not None:
                self._elapsed_paused_total += (
                    time.perf_counter() - self._pause_started_at
                )
                self._pause_started_at = None
        self.pause_btn.setText("▶ 继续" if self._is_paused else "⏸ 暂停")
        # Refresh the label so the frozen value shows up immediately
        # (the live tick is paused and won't update it for us).
        self._update_elapsed_label()

    def _on_stop(self) -> None:
        self._is_running = False
        self._live_timer.stop()
        for role, w in list(self._video_writers.items()):
            try:
                w.release()
                logger.info("Released writer for role %s", role)
            except Exception:
                logger.exception("Failed to release writer for %s", role)
        self._video_writers.clear()
        self._writers_initialized = False
        # Reset elapsed-time accounting
        self._elapsed_start = None
        self._elapsed_paused_total = 0.0
        self._pause_started_at = None
        self.elapsed_label.setText("00:00:00")
        if self._current_row >= 0 and self._entry is not None:
            self.entry_table.set_status(
                self._current_row,
                f"已停止 ({self._entry['animal_id']})",
            )
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ 暂停")
        self.status_label.setText("已停止")

    # ------------------------------------------------------------------
    # Per-frame capture (pull mode via MultiCameraManager)
    # ------------------------------------------------------------------
    def _live_tick(self) -> None:
        if not self._is_running or self._is_paused:
            return
        if self._cam_mgr is None:
            return
        snap = self._cam_mgr.grab_quartet(timeout_ms=50)
        if snap is None:
            return
        # Lazy init writers on first frame using actual frame shapes
        if not self._writers_initialized and self._entry is not None:
            try:
                self._init_video_writers(self._entry, snap)
                self._writers_initialized = True
            except Exception as e:
                logger.exception("Writer init failed")
                QMessageBox.critical(self, "录制失败", f"无法初始化视频写入: {e}")
                self._is_running = False
                return
        self._frame_index += 1
        self._fps_counter += 1
        # Update elapsed time label every frame (cheap; 33ms cadence).
        self._update_elapsed_label()
        recording = self.record_check.isChecked()
        for role, info in snap.items():
            frame = info.image
            if frame is None:
                continue
            if recording and role in self._video_writers:
                try:
                    self._video_writers[role].write(frame)
                except Exception:
                    logger.exception("Writer write failed for %s", role)
            if role == "bottom":
                self._show_wide_preview(frame)
            if self._cam_strip is not None:
                self._cam_strip.set_image(role, frame)

    def _update_elapsed_label(self) -> None:
        if self._elapsed_start is None:
            return
        elapsed = time.perf_counter() - self._elapsed_start - self._elapsed_paused_total
        if self._is_paused and self._pause_started_at is not None:
            # Freeze the label at the moment of pause.
            elapsed = self._pause_started_at - self._elapsed_start - self._elapsed_paused_total
        elapsed = max(0, int(elapsed))
        hh, rem = divmod(elapsed, 3600)
        mm, ss = divmod(rem, 60)
        self.elapsed_label.setText(f"{hh:02d}:{mm:02d}:{ss:02d}")

    def _show_wide_preview(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        pw = max(self.video_preview.width(), 480)
        ph = max(self.video_preview.height(), 96)
        target_aspect = BOTTOM_CAMERA_SIZE[0] / BOTTOM_CAMERA_SIZE[1]  # 5.0
        widget_aspect = pw / max(ph, 1)
        if widget_aspect > target_aspect:
            nh = ph
            nw = int(nh * target_aspect)
        else:
            nw = pw
            nh = int(nw / target_aspect)
        nw, nh = max(1, nw), max(1, nh)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (nw, nh))
        qimg = QImage(img.data, nw, nh, img.strides[0], QImage.Format.Format_RGB888)
        self.video_preview.setPixmap(QPixmap.fromImage(qimg))

    def _update_fps_label(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_fps_time
        if dt > 0:
            fps = self._fps_counter / dt
            self.frame_label.setText(
                f"Frame: {self._frame_index} | FPS: {fps:.1f}",
            )
        self._fps_counter = 0
        self._last_fps_time = now

    def _on_sync_tick(self) -> None:
        if self._cam_mgr is None:
            self.sync_label.setText("同步: --")
            self.sync_label.setStyleSheet("font-family: monospace; color: #888;")
            return
        rpt = self._cam_mgr.get_sync_report()
        txt = (
            f"同步: max Δ={rpt.max_delta_ms:.2f} ms "
            f"(sample={rpt.sample_size})"
        )
        if rpt.in_sync and rpt.sample_size > 0:
            color = "#2E7D32"
        elif rpt.sample_size == 0:
            color = "#888"
        else:
            color = "#C62828"
        self.sync_label.setText(txt)
        self.sync_label.setStyleSheet(
            f"font-family: monospace; color: {color};"
        )
        # Push per-camera hardware FPS to the 4×1 strip
        if self._cam_strip is not None:
            fps_map = self._cam_mgr.get_fps_per_role()
            for role, fps in fps_map.items():
                self._cam_strip.set_fps(role, fps)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _project_raw_videos_dir(self) -> Optional[Path]:
        if not (self._app_state and self._app_state.project.is_valid):
            return None
        d = Path(self._app_state.project.project_path) / "rawdata" / "videos"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _recorded_path_for(
        self, animal_id: str, role: str, ts: str,
    ) -> Optional[Path]:
        d = self._project_raw_videos_dir()
        if d is None:
            return None
        store = SessionStore(d)
        return store.recorded_path_for(animal_id, role, ts)

    # ------------------------------------------------------------------
    # Video writers
    # ------------------------------------------------------------------
    def _init_video_writers(
        self,
        entry: Dict[str, str],
        snap: Dict[str, FrameInfo],
    ) -> None:
        """Create one cv2.VideoWriter per role using actual frame shapes.

        Writers are opened with the actual ``FrameInfo.image.shape`` of
        the first frame, so we never hard-code a resolution that may
        differ from real hardware.  Output paths are derived from
        :class:`SessionStore.recorded_path_for`, which produces the
        per-session subdirectory layout.
        """
        base_dir = self._project_raw_videos_dir()
        if base_dir is None:
            raise RuntimeError("未打开有效项目，无法写入视频")
        animal_id = entry.get("animal_id") or "session"
        fps = float(self._fps or 100.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for role in self.ALL_ROLES:
            info = snap.get(role)
            if info is None or info.image is None:
                logger.warning("No frame for role %s, skip writer", role)
                continue
            h, w = info.image.shape[:2]
            path = self._recorded_path_for(animal_id, role, self._ts)
            if path is None:
                continue
            writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"无法打开 VideoWriter: {path}")
            self._video_writers[role] = writer
            logger.info(
                "Recording role=%s to %s (%dx%d @ %.0ffps)",
                role, path, w, h, fps,
            )

    # ------------------------------------------------------------------
    # Project display
    # ------------------------------------------------------------------
    def _update_project_display(self) -> None:
        if self._app_state and self._app_state.project.is_valid:
            p = self._app_state.project
            self.project_label.setText(
                f"当前项目: {p.project_name} ({p.project_path})",
            )
        else:
            self.project_label.setText("当前项目: 未打开")
