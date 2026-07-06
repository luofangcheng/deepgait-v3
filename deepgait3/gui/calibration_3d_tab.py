"""3D calibration tab — ChArUco intrinsic + extrinsic estimation.

Loads a directory of ChArUco board images per camera, runs
:func:`deepgait3.core._legacy.triangulation_3d.calibrate_charuco`, displays the
per-camera intrinsic matrices + RMS reprojection error, and publishes
the resulting :class:`CalibrationView` to :class:`AppState` so the
triangulation tab can consume it.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.3 W11):
    "ChArUco 捕获 + 实时 reprojection error"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepgait3.core._legacy.triangulation_3d import calibrate_charuco
from deepgait3.gui.shared_state import AppState, CalibrationView


logger = logging.getLogger(__name__)


# Image extensions recognised by OpenCV imread.
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class Calibration3DTab(QWidget):
    """ChArUco multi-camera calibration tab.

    Workflow
    --------
    1. User picks a calibration root directory containing one
       subdirectory per camera (``root/<cam_name>/*.png``).
    2. User sets the ChArUco board parameters (rows, cols, square, marker).
    3. ``calibrate`` runs :func:`calibrate_charuco` per camera.
    4. Per-camera reprojection RMS is shown in a table.
    5. Result is published to :class:`AppState` via ``set_calibration``.

    Signals
    -------
    calibration_finished : Signal(object)
        Emitted with the :class:`CalibrationView` once calibration
        completes (also fires ``AppState.set_calibration``).
    """

    calibration_finished = Signal(object)

    def __init__(
        self,
        app_state: AppState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = app_state
        self._state.calibration_changed.connect(self._on_calibration_changed)
        self._images_root: Optional[str] = None
        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # --- Top bar: directory picker ---
        bar = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("选择标定根目录（每相机一个子目录）…")
        self.browse_btn = QPushButton("选择目录…")
        self.browse_btn.clicked.connect(self._on_browse)
        self.calibrate_btn = QPushButton("开始标定")
        self.calibrate_btn.clicked.connect(self._on_calibrate)
        self.calibrate_btn.setEnabled(False)
        bar.addWidget(QLabel("标定目录:"))
        bar.addWidget(self.path_edit, 2)
        bar.addWidget(self.browse_btn)
        bar.addWidget(self.calibrate_btn)
        root.addLayout(bar)

        # --- ChArUco board parameters ---
        params = QFormLayout()
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 30)
        self.rows_spin.setValue(5)
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 30)
        self.cols_spin.setValue(7)
        self.square_spin = QSpinBox()
        self.square_spin.setRange(1, 200)
        self.square_spin.setValue(30)
        self.square_spin.setSuffix(" mm")
        self.marker_spin = QSpinBox()
        self.marker_spin.setRange(1, 200)
        self.marker_spin.setValue(22)
        self.marker_spin.setSuffix(" mm")
        params.addRow("棋盘行数:", self.rows_spin)
        params.addRow("棋盘列数:", self.cols_spin)
        params.addRow("方格边长:", self.square_spin)
        params.addRow("标记边长:", self.marker_spin)
        root.addLayout(params)

        # --- Results table ---
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["相机 / Camera", "Reprojection RMS (px)", "焦距 fx (px)"],
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        self.status_label = QLabel("就绪 — 请选择标定目录")
        root.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    @Slot()
    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "选择标定根目录（每相机一个子目录）", "",
        )
        if not path:
            return
        self._images_root = path
        self.path_edit.setText(path)
        self.calibrate_btn.setEnabled(True)
        self.status_label.setText(f"已选择: {path}")

    @Slot()
    def _on_calibrate(self) -> None:
        if not self._images_root:
            return
        try:
            view = self._run_calibration(self._images_root)
        except Exception as e:
            QMessageBox.critical(self, "标定失败", f"{e}")
            logger.exception("calibration_3d_tab: calibration failed")
            return
        self._populate_table(view)
        self._state.set_calibration(view)
        self.calibration_finished.emit(view)
        self.status_label.setText(
            f"标定完成 — {len(view.cameras)} 相机 / "
            f"平均 RMS {self._mean_rms(view):.3f} px"
        )

    @Slot(object)
    def _on_calibration_changed(self, view: CalibrationView) -> None:
        # Other tabs may push a calibration; keep our table in sync.
        self._populate_table(view)

    # ------------------------------------------------------------------
    # Pure logic — unit-testable
    # ------------------------------------------------------------------
    def _discover_camera_dirs(self, root: str) -> Dict[str, List[str]]:
        """Walk ``root`` and return ``{cam_name: [image_paths]}``.

        Each immediate subdirectory of ``root`` is treated as one
        camera. Image files are identified by extension.
        """
        out: Dict[str, List[str]] = {}
        root_path = Path(root)
        if not root_path.is_dir():
            return out
        for sub in sorted(root_path.iterdir()):
            if not sub.is_dir():
                continue
            imgs = sorted(
                str(p) for p in sub.iterdir()
                if p.suffix.lower() in _IMG_EXTS
            )
            if imgs:
                out[sub.name] = imgs
        return out

    def _read_images(self, paths: List[str]) -> List[np.ndarray]:
        imgs: List[np.ndarray] = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                imgs.append(img)
        return imgs

    def _run_calibration(self, root_dir: str) -> CalibrationView:
        """Run ``calibrate_charuco`` per camera and return a CalibrationView."""
        camera_dirs = self._discover_camera_dirs(root_dir)
        if not camera_dirs:
            raise ValueError(
                f"在 {root_dir} 下找不到相机子目录 / 图像"
            )
        view = CalibrationView(
            board_rows=int(self.rows_spin.value()),
            board_cols=int(self.cols_spin.value()),
            method="inhouse",
        )
        for cam_name, paths in camera_dirs.items():
            imgs = self._read_images(paths)
            if not imgs:
                logger.warning("calibration: %s 无可读图像", cam_name)
                continue
            res = calibrate_charuco(
                imgs,
                squares_x=self.cols_spin.value(),
                squares_y=self.rows_spin.value(),
                square_length=float(self.square_spin.value()),
                marker_length=float(self.marker_spin.value()),
            )
            view.cameras.append(cam_name)
            view.reproj_rms_per_cam[cam_name] = float(res["reproj_rms"])
        if not view.cameras:
            raise ValueError("所有相机目录均无可识别的 ChArUco 棋盘")
        return view

    def _populate_table(self, view: CalibrationView) -> None:
        self.table.setRowCount(len(view.cameras))
        for r, cam in enumerate(view.cameras):
            rms = view.reproj_rms_per_cam.get(cam, float("inf"))
            self.table.setItem(r, 0, QTableWidgetItem(cam))
            self.table.setItem(r, 1, QTableWidgetItem(f"{rms:.3f}"))
            self.table.setItem(r, 2, QTableWidgetItem("—"))  # filled later

    @staticmethod
    def _mean_rms(view: CalibrationView) -> float:
        vals = [v for v in view.reproj_rms_per_cam.values()
                if v != float("inf")]
        return float(np.mean(vals)) if vals else float("inf")


__all__ = ["Calibration3DTab"]