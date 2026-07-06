"""3D triangulation tab — multi-camera pose → 3D keypoints + viewer.

Loads the 2D pose CSVs from each DLC camera, requires a finished
calibration (from :class:`Calibration3DTab` via AppState), runs
:meth:`AniposeWrapper.triangulate`, and reports per-frame reprojection
error + publishes a :class:`Pose3DResultsView` to AppState.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.3 W11):
    "DLT+RANSAC, reprojection < 3 px"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepgait3.core._legacy.anipose_wrapper import (
    AniposeWrapper, DEFAULT_BODYPARTS_12,
)
from deepgait3.core._legacy.triangulation_3d import reprojection_error
from deepgait3.gui.shared_state import AppState, Pose3DResultsView


logger = logging.getLogger(__name__)


class Triangulation3DTab(QWidget):
    """Multi-camera 2D → 3D triangulation tab.

    Workflow
    --------
    1. Wait for a calibration to be published on AppState (from
       Calibration3DTab). Until then, the triangulate button stays
       disabled.
    2. User picks one or more 2D pose CSVs (one per camera).
    3. ``triangulate`` calls :meth:`AniposeWrapper.triangulate`.
    4. Per-camera reprojection error is shown in a table.
    5. :class:`Pose3DResultsView` is published to AppState so the 3D
       viewer (W12) can render it.

    Signals
    -------
    triangulation_finished : Signal(object)
        Emitted with the :class:`Pose3DResultsView`.
    """

    triangulation_finished = Signal(object)

    def __init__(
        self,
        app_state: AppState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = app_state
        self._pose_2d_paths: Dict[str, str] = {}
        self._state.calibration_changed.connect(self._on_calibration_changed)
        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Calibration status ---
        self.calib_status = QLabel("等待 Calibration3DTab 完成标定…")
        root.addWidget(self.calib_status)

        # --- 2D pose CSV list (one per camera) ---
        bar = QHBoxLayout()
        bar.addWidget(QLabel("2D pose CSV（每相机一个）:"))
        self.add_btn = QPushButton("添加 CSV…")
        self.add_btn.clicked.connect(self._on_add_csv)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self._on_clear_csvs)
        bar.addWidget(self.add_btn)
        bar.addWidget(self.clear_btn)
        bar.addStretch()
        self.triangulate_btn = QPushButton("三角化")
        self.triangulate_btn.clicked.connect(self._on_triangulate)
        self.triangulate_btn.setEnabled(False)
        bar.addWidget(self.triangulate_btn)
        root.addLayout(bar)

        self.csv_list = QListWidget()
        self.csv_list.setAlternatingRowColors(True)
        root.addWidget(self.csv_list, 1)

        # --- Reprojection error table ---
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(
            ["相机 / Camera", "Reprojection RMS (px)"],
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table)

        self.status_label = QLabel("就绪")
        root.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    @Slot(object)
    def _on_calibration_changed(self, calib) -> None:
        if calib is None or not getattr(calib, "cameras", None):
            self.calib_status.setText("等待 Calibration3DTab 完成标定…")
            self.triangulate_btn.setEnabled(False)
            return
        rms_vals = [v for v in calib.reproj_rms_per_cam.values()
                    if v != float("inf")]
        mean_rms = float(np.mean(rms_vals)) if rms_vals else float("inf")
        self.calib_status.setText(
            f"已加载标定 — {len(calib.cameras)} 相机 / 平均 RMS {mean_rms:.3f} px"
        )
        # Enable triangulation only when at least 2 cameras are calibrated
        # AND at least 2 CSVs are loaded.
        self._refresh_enable_state()

    @Slot()
    def _on_add_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 2D pose CSV", "",
            "CSV 文件 (*.csv);;所有文件 (*.*)",
        )
        if not path:
            return
        cam_name = Path(path).stem
        # If a calibration is loaded, prefer matching the cam_name.
        calib = self._state.calibration
        if calib and calib.cameras:
            # Match by prefix; fall back to the file stem.
            cam_name = next(
                (c for c in calib.cameras if c in cam_name), cam_name,
            )
        self._pose_2d_paths[cam_name] = path
        self.csv_list.addItem(f"{cam_name}: {path}")
        self._refresh_enable_state()
        self.status_label.setText(f"已添加 {len(self._pose_2d_paths)} CSV")

    @Slot()
    def _on_clear_csvs(self) -> None:
        self._pose_2d_paths.clear()
        self.csv_list.clear()
        self._refresh_enable_state()
        self.status_label.setText("已清空 CSV 列表")

    @Slot()
    def _on_triangulate(self) -> None:
        calib = self._state.calibration
        if calib is None:
            QMessageBox.warning(self, "缺少标定", "请先在 3D 标定 tab 完成标定")
            return
        if len(self._pose_2d_paths) < 2:
            QMessageBox.warning(self, "CSV 不足", "至少需要 2 个相机的 CSV")
            return
        try:
            view = self._run_triangulation(calib)
        except Exception as e:
            QMessageBox.critical(self, "三角化失败", f"{e}")
            logger.exception("triangulation_3d_tab: triangulation failed")
            return
        self._populate_table(view)
        self._state.set_pose_3d(view)
        self.triangulation_finished.emit(view)
        self.status_label.setText(
            f"三角化完成 — {view.n_frames} 帧 / RMS {view.reproj_rms_px:.2f} px"
        )

    # ------------------------------------------------------------------
    # Pure logic — unit-testable
    # ------------------------------------------------------------------
    def _load_pose_2d(self, csv_path: str) -> np.ndarray:
        """Load a DLC 2D pose CSV into a (T, J, 2) array.

        The CSV uses the DLC multi-index header (scorer, bodypart, coord)
        where ``coord ∈ {x, y, likelihood}``.
        """
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        # Reshape (T, J, 3) → (T, J, 2). Bodyparts are assumed to be the
        # canonical 12-point VGL scheme; we take the first 12 found.
        bodyparts = list(dict.fromkeys(df.columns.get_level_values(1)))
        n_bp = min(len(bodyparts), len(DEFAULT_BODYPARTS_12))
        n_frames = len(df)
        out = np.zeros((n_frames, n_bp, 2), dtype=np.float64)
        for j, bp in enumerate(bodyparts[:n_bp]):
            try:
                out[:, j, 0] = df.xs(bp, level=1, axis=1)["x"].values
                out[:, j, 1] = df.xs(bp, level=1, axis=1)["y"].values
            except KeyError:
                pass
        return out

    def _run_triangulation(self, calib) -> Pose3DResultsView:
        wrapper = AniposeWrapper(prefer="auto")
        pose_2d: Dict[str, np.ndarray] = {}
        for cam_name, path in self._pose_2d_paths.items():
            if cam_name not in calib.cameras:
                logger.warning(
                    "triangulate: 跳过未标定的相机 %s", cam_name,
                )
                continue
            pose_2d[cam_name] = self._load_pose_2d(path)
        if len(pose_2d) < 2:
            raise ValueError(
                "匹配到标定的相机 < 2；无法三角化"
            )
        # Use only the first 50 frames for speed in the stub. W12 will
        # add a real-time / batch toggle.
        max_frames = 50
        for k in pose_2d:
            pose_2d[k] = pose_2d[k][:max_frames]
        keypoints_3d = wrapper.triangulate(pose_2d, calib)
        # Recompute mean reprojection error.
        cam_names = list(pose_2d.keys())
        Ps = [calib.cameras[n].P for n in cam_names]
        errs: List[float] = []
        n_frames = keypoints_3d.shape[0]
        n_bp = keypoints_3d.shape[1]
        for t in range(n_frames):
            for j in range(n_bp):
                X = keypoints_3d[t, j]
                if np.any(np.isnan(X)):
                    continue
                pts = [pose_2d[c][t, j] for c in cam_names]
                errs.append(reprojection_error(Ps, pts, X))
        mean_rms = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("inf")
        return Pose3DResultsView(
            n_frames=int(n_frames),
            n_bodyparts=int(n_bp),
            reproj_rms_px=mean_rms,
            source=wrapper.active_method,
        )

    def _populate_table(self, view: Pose3DResultsView) -> None:
        self.table.setRowCount(1)
        self.table.setItem(0, 0, QTableWidgetItem("(all cameras)"))
        self.table.setItem(0, 1, QTableWidgetItem(f"{view.reproj_rms_px:.2f}"))

    def _refresh_enable_state(self) -> None:
        calib = self._state.calibration
        calib_ok = calib is not None and len(calib.cameras) >= 2
        csv_ok = len(self._pose_2d_paths) >= 2
        self.triangulate_btn.setEnabled(calib_ok and csv_ok)


__all__ = ["Triangulation3DTab"]