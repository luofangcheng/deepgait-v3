"""Tab 3: DLC Quick Entry.

Project config + 5 action buttons (create/extract/label/train/analyze) + log.
All heavy operations run in DLCWorker thread.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog,
    QGroupBox, QSpinBox, QListWidget, QListWidgetItem, QTextEdit, QMessageBox,
    QLabel, QDialog, QDialogButtonBox, QComboBox, QFormLayout,
)

from deepgait3.core._legacy import bodyparts
from deepgait3.core._legacy.dlc_config import ProjectSpec
from deepgait3.gui.workers import DLCWorker


class TrainDialog(QDialog):
    """Dialog for training parameters."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("训练参数 (GTX 3060 预设)")
        layout = QFormLayout(self)

        self.backbone_combo = QComboBox()
        self.backbone_combo.addItems(["resnet_50", "hrnet_w32", "hrnet_w48", "cspnext_m"])
        layout.addRow("Backbone:", self.backbone_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(200)
        layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(8)
        layout.addRow("Batch size:", self.batch_spin)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu", "mps"])
        layout.addRow("Device:", self.device_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_params(self) -> dict:
        device = self.device_combo.currentText()
        return {
            "net_type": self.backbone_combo.currentText(),
            "epochs": self.epochs_spin.value(),
            "batch_size": self.batch_spin.value(),
            "device": None if device == "auto" else device,
        }


class AnalyzeDialog(QDialog):
    """Dialog for selecting videos to analyze."""

    def __init__(self, videos: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择要分析的视频")
        layout = QVBoxLayout(self)

        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for v in videos:
            item = QListWidgetItem(v)
            item.setSelected(True)
            self.video_list.addItem(item)
        layout.addWidget(self.video_list)

        from PySide6.QtWidgets import QCheckBox
        self.filter_check = QCheckBox("分析后滤波 (filterpredictions)")
        self.filter_check.setChecked(True)
        layout.addWidget(self.filter_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_selection(self) -> tuple[list[str], bool]:
        selected = [self.video_list.item(i).text() for i in range(self.video_list.count())
                    if self.video_list.item(i).isSelected()]
        return selected, self.filter_check.isChecked()


class DLCTab(QWidget):
    """Tab 3: DLC quick entry with 5-step workflow.

    Phase 3 W11: now integrates with :class:`DLCSubprocessRunner`
    (W8 deliverable) for isolated conda-env DLC execution. Falls back
    to the legacy in-process :class:`DLCWorker` when no runner is
    injected via :meth:`set_runner`.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._videos: list[str] = []
        self._config_path: str = ""
        self._worker: DLCWorker | None = None
        # W11: optional DLCSubprocessRunner (injected for tests / GUI demo).
        self._runner = None
        self._build_ui()

    def set_runner(self, runner) -> None:
        """Inject a DLCSubprocessRunner (or MockDLCSubprocessRunner).

        When set, all DLC operations (create_project / train_network /
        analyze_videos) go through the subprocess runner instead of
        the legacy in-process DLCWorker.
        """
        self._runner = runner

    @property
    def has_runner(self) -> bool:
        return self._runner is not None

    def _build_ui(self) -> None:
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(10)

        # Project config
        cfg_group = QGroupBox("项目配置")
        cfg_layout = QVBoxLayout(cfg_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("项目名:"))
        self.project_edit = QLineEdit("deepgait")
        row1.addWidget(self.project_edit)
        row1.addWidget(QLabel("标注者:"))
        self.experimenter_edit = QLineEdit("researcher")
        row1.addWidget(self.experimenter_edit)
        cfg_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 1000)
        self.fps_spin.setValue(100)
        row2.addWidget(self.fps_spin)
        row2.addWidget(QLabel("宽度:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10000)
        self.width_spin.setValue(640)
        row2.addWidget(self.width_spin)
        row2.addWidget(QLabel("高度:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 10000)
        self.height_spin.setValue(480)
        row2.addWidget(self.height_spin)
        row2.addWidget(QLabel("抽帧数:"))
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(1, 1000)
        self.num_frames_spin.setValue(20)
        row2.addWidget(self.num_frames_spin)
        cfg_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.add_video_btn = QPushButton("添加视频")
        self.add_video_btn.clicked.connect(self._on_add_videos)
        row3.addWidget(self.add_video_btn)
        self.work_dir_edit = QLineEdit()
        self.work_dir_edit.setPlaceholderText("工作目录...")
        row3.addWidget(self.work_dir_edit, stretch=1)
        self.work_dir_btn = QPushButton("浏览")
        self.work_dir_btn.clicked.connect(self._on_browse_work_dir)
        row3.addWidget(self.work_dir_btn)
        cfg_layout.addLayout(row3)

        self.video_list = QListWidget()
        self.video_list.setMaximumHeight(80)
        cfg_layout.addWidget(self.video_list)

        main.addWidget(cfg_group)

        # Bodyparts
        bp_group = QGroupBox("12 点标注部位 (来自 VisualGaitLab)")
        bp_layout = QVBoxLayout(bp_group)
        self.bp_list = QListWidget()
        for bp in bodyparts.BODYPARTS_12:
            self.bp_list.addItem(QListWidgetItem(bp))
        bp_layout.addWidget(self.bp_list)
        main.addWidget(bp_group)

        # Action buttons
        action_layout = QHBoxLayout()
        self.create_btn = QPushButton("1. 新建项目")
        self.create_btn.clicked.connect(self._on_create_project)
        self.extract_btn = QPushButton("2. 抽帧")
        self.extract_btn.clicked.connect(self._on_extract)
        self.label_btn = QPushButton("3. 启动标注 (napari)")
        self.label_btn.clicked.connect(self._on_label)
        self.train_btn = QPushButton("4. 训练")
        self.train_btn.clicked.connect(self._on_train)
        self.analyze_btn = QPushButton("5. 分析视频")
        self.analyze_btn.clicked.connect(self._on_analyze)
        for btn in [self.create_btn, self.extract_btn, self.label_btn, self.train_btn, self.analyze_btn]:
            action_layout.addWidget(btn)
        main.addLayout(action_layout)

        # Log
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        main.addWidget(log_group, stretch=1)

    def _log(self, msg: str) -> None:
        self.log_edit.append(msg)
        if self.parent() and hasattr(self.parent(), "status"):
            self.parent().status.showMessage(msg)

    def _on_add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频", "",
            "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*.*)"
        )
        for p in paths:
            if p not in self._videos:
                self._videos.append(p)
                self.video_list.addItem(QListWidgetItem(p))

    def _on_browse_work_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if path:
            self.work_dir_edit.setText(path)

    def _get_spec(self) -> ProjectSpec:
        return ProjectSpec(
            project=self.project_edit.text(),
            experimenter=self.experimenter_edit.text(),
            videos=self._videos,
            working_directory=self.work_dir_edit.text() or ".",
            fps=self.fps_spin.value(),
            video_width=self.width_spin.value(),
            video_height=self.height_spin.value(),
            numframes2pick=self.num_frames_spin.value(),
        )

    def _on_create_project(self) -> None:
        if not self._videos:
            QMessageBox.warning(self, "提示", "请先添加视频")
            return
        if self._runner is not None:
            return self._on_create_project_v2()
        spec = self._get_spec()
        self._log(f"创建项目: {spec.project} (实验者: {spec.experimenter})")
        self._worker = DLCWorker(step="create_project", spec=spec)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    # ------------------------------------------------------------------
    # W11: DLCSubprocessRunner path (isolated conda env)
    # ------------------------------------------------------------------
    def _on_create_project_v2(self) -> None:
        """Create project via DLCSubprocessRunner."""
        project = self.project_edit.text() or "deepgait"
        experimenter = self.experimenter_edit.text() or "researcher"
        working_dir = self._working_dir or "."
        self._log(f"创建项目 (subprocess): {project}")
        try:
            cfg = self._runner.create_project(
                project_name=project,
                experimenter=experimenter,
                videos=self._videos,
                working_directory=working_dir,
            )
            self._config_path = cfg
            self._log(f"项目配置: {cfg}")
        except Exception as e:
            self._on_error(str(e))

    def _on_extract(self) -> None:
        config = self._prompt_config()
        if not config:
            return
        self._log(f"抽帧: {config}")
        self._worker = DLCWorker(step="extract_frames", config=config)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_label(self) -> None:
        config = self._prompt_config()
        if not config:
            return
        self._log(f"启动标注: {config}")
        # Labeling opens napari GUI — run in foreground (not in worker)
        from deepgait3.core._legacy import dlc_workflow
        try:
            dlc = dlc_workflow._require_dlc()
            dlc.label_frames(config)
            self._log("标注已启动 (关闭 napari 后继续)")
        except dlc_workflow.DLCNotInstalledError as e:
            QMessageBox.critical(self, "错误", str(e))
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def _on_train(self) -> None:
        config = self._prompt_config()
        if not config:
            return
        if self._runner is not None:
            return self._on_train_v2(config)
        dlg = TrainDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        params = dlg.get_params()
        params["config"] = config
        self._log(f"训练: backbone={params['net_type']}, epochs={params['epochs']}")
        self._worker = DLCWorker(step="train", **params)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_train_v2(self, config: str) -> None:
        """Train via DLCSubprocessRunner."""
        self._log(f"训练 (subprocess): {config}")
        try:
            ok = self._runner.train_network(config, epochs=5, batch_size=1)
            self._log("训练完成" if ok else "训练失败")
        except Exception as e:
            self._on_error(str(e))

    def _on_analyze(self) -> None:
        config = self._prompt_config()
        if not config:
            return
        if self._runner is not None:
            return self._on_analyze_v2(config)
        dlg = AnalyzeDialog(self._videos, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        selected, do_filter = dlg.get_selection()
        if not selected:
            QMessageBox.warning(self, "提示", "请选择要分析的视频")
            return
        self._log(f"分析视频: {selected}")
        self._worker = DLCWorker(step="analyze_videos", config=config, videos=selected)
        self._worker.step_done.connect(self._on_step_done)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_analyze_v2(self, config: str) -> None:
        """Analyze via DLCSubprocessRunner."""
        if not self._videos:
            QMessageBox.warning(self, "提示", "请先添加视频")
            return
        self._log(f"分析视频 (subprocess): {self._videos}")
        try:
            out = self._runner.analyze_videos(config, self._videos)
            self._log(f"分析完成，输出: {out}")
        except Exception as e:
            self._on_error(str(e))

    def _prompt_config(self) -> str | None:
        if self._config_path:
            return self._config_path
        # Try to find config in work_dir
        work_dir = self.work_dir_edit.text()
        if work_dir:
            for cfg in Path(work_dir).rglob("config.yaml"):
                self._config_path = str(cfg)
                return self._config_path
        # Ask user
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 config.yaml", "", "YAML (*.yaml *.yml);;所有文件 (*.*)"
        )
        if path:
            self._config_path = path
        return path or None

    def _on_step_done(self, msg: str) -> None:
        self._log(f"✓ {msg}")

    def _on_error(self, msg: str) -> None:
        self._log(f"✗ 错误: {msg}")
        QMessageBox.critical(self, "错误", msg)

    def _on_worker_finished(self) -> None:
        self._worker = None
