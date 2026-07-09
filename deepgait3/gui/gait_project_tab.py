"""Tab 1: 新建项目 (New Project) — deepgait v2.0.

Creates project folder structure, manages the recent-projects list,
and publishes the active project to ``AppState`` so other tabs
can discover the current project path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepgait3.core._legacy import project_manager as pm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_PARENT_DIR = str(Path.home() / "deepgait_projects")


class GaitProjectTab(QWidget):
    """Project creation and recent-projects browser.

    Emits ``project_opened`` when the user opens or creates a project.
    Other tabs (数据采集, 数据分析) listen for this to update their
    current-project display.
    """

    project_opened = Signal(object)  # pm.ProjectConfig

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_project: Optional[pm.ProjectConfig] = None
        self._build_ui()
        self._refresh_recent()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # -- Create section --------------------------------------------------
        create_gb = QGroupBox("创建新项目")
        create_l = QVBoxLayout(create_gb)
        create_l.setSpacing(8)

        # Name row
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("项目名称:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如: SCI实验001")
        self.name_edit.setMinimumWidth(250)
        name_row.addWidget(self.name_edit)
        name_row.addStretch()
        create_l.addLayout(name_row)

        # Path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("保存路径:"))
        self.path_edit = QLineEdit(_DEFAULT_PARENT_DIR)
        self.path_edit.setMinimumWidth(300)
        path_row.addWidget(self.path_edit)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._on_browse_path)
        path_row.addWidget(browse_btn)
        path_row.addStretch()
        create_l.addLayout(path_row)

        # Experimenter row
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("实验员:"))
        self.exp_edit = QLineEdit()
        self.exp_edit.setPlaceholderText("实验员姓名")
        self.exp_edit.setMinimumWidth(200)
        exp_row.addWidget(self.exp_edit)
        exp_row.addStretch()
        create_l.addLayout(exp_row)

        # Calibration row
        cal_row = QHBoxLayout()
        cal_row.addWidget(QLabel("px_per_mm:"))
        from PySide6.QtWidgets import QDoubleSpinBox
        self.px_spin = QDoubleSpinBox()
        self.px_spin.setRange(1.0, 20.0)
        self.px_spin.setValue(3.0)
        self.px_spin.setDecimals(1)
        self.px_spin.setToolTip("像素/毫米标定系数")
        cal_row.addWidget(self.px_spin)
        cal_row.addStretch()
        create_l.addLayout(cal_row)

        create_btn = QPushButton(" 创建项目 ")
        create_btn.setStyleSheet("font-weight: bold; padding: 6px 24px;")
        create_btn.clicked.connect(self._on_create_project)
        create_l.addWidget(create_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(create_gb)

        # -- Recent projects -------------------------------------------------
        recent_gb = QGroupBox("最近项目")
        recent_l = QVBoxLayout(recent_gb)

        self.recent_table = QTableWidget(0, 3)
        self.recent_table.setHorizontalHeaderLabels(
            ["项目名称", "路径", "创建时间"]
        )
        self.recent_table.horizontalHeader().setStretchLastSection(True)
        self.recent_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents,
        )
        self.recent_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows,
        )
        self.recent_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers,
        )
        self.recent_table.doubleClicked.connect(self._on_open_recent)
        recent_l.addWidget(self.recent_table)

        btn_row = QHBoxLayout()
        open_btn = QPushButton("打开选中项目")
        open_btn.clicked.connect(self._on_open_recent)
        btn_row.addWidget(open_btn)
        del_btn = QPushButton("删除")
        del_btn.clicked.connect(self._on_delete_recent)
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        recent_l.addLayout(btn_row)

        layout.addWidget(recent_gb)

        # -- Status ----------------------------------------------------------
        self.status_label = QLabel("未打开项目")
        layout.addWidget(self.status_label)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_browse_path(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "选择项目保存目录", self.path_edit.text(),
        )
        if d:
            self.path_edit.setText(d)

    def _on_create_project(self) -> None:
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "错误", "请输入项目名称")
            return
        parent_dir = Path(self.path_edit.text().strip() or _DEFAULT_PARENT_DIR)
        try:
            cfg = pm.create_project(
                name, parent_dir,
                experimenter=self.exp_edit.text().strip(),
                px_per_mm=self.px_spin.value(),
            )
        except FileExistsError:
            QMessageBox.warning(
                self, "错误",
                f"项目目录已存在:\n{parent_dir / name}",
            )
            return
        self._current_project = cfg
        self.status_label.setText(
            f"当前项目: {cfg.project_name} ({cfg.project_path})",
        )
        self._refresh_recent()
        self.project_opened.emit(cfg)

    def _refresh_recent(self) -> None:
        entries = pm.list_recent_projects()
        self.recent_table.setRowCount(len(entries))
        for i, e in enumerate(entries):
            self.recent_table.setItem(i, 0, QTableWidgetItem(
                e.get("project_name", ""),
            ))
            self.recent_table.setItem(i, 1, QTableWidgetItem(
                e.get("project_path", ""),
            ))
            self.recent_table.setItem(i, 2, QTableWidgetItem(
                e.get("created_at", "")[:10],
            ))

    def _on_open_recent(self) -> None:
        row = self.recent_table.currentRow()
        if row < 0:
            return
        path_str = self.recent_table.item(row, 1)
        if not path_str:
            return
        cfg = pm.ProjectConfig.load(Path(path_str.text()))
        if cfg is None:
            QMessageBox.warning(self, "错误", "项目配置文件无效")
            pm.remove_from_recent(Path(path_str.text()))
            self._refresh_recent()
            return
        self._current_project = cfg
        self.status_label.setText(
            f"当前项目: {cfg.project_name} ({cfg.project_path})",
        )
        self.project_opened.emit(cfg)

    def _on_delete_recent(self) -> None:
        row = self.recent_table.currentRow()
        if row < 0:
            return
        path_str = self.recent_table.item(row, 1)
        if path_str:
            pm.remove_from_recent(Path(path_str.text()))
        self._refresh_recent()
