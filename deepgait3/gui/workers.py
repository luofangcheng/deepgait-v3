"""QThread workers for long-running operations.

All workers emit signals for progress, results, and errors.
The GUI connects these signals to slots that update the UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from deepgait3.core._legacy import footprint, intensity, pipeline, results
from deepgait3.core._legacy.dlc_config import ProjectSpec
from deepgait3.core._legacy.dlc_workflow import DLCNotInstalledError


class GaitWorker(QThread):
    """Run gait analysis in a background thread."""

    result_ready = Signal(object)   # GaitResults
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        csv_path: str,
        fps: int = 100,
        mode: str = "catwalk",
        video_height: float | None = None,
        likelihood_threshold: float = 0.1,
        treadmill_speed: float | None = None,
        real_world_multiplier: float = 1.0,
        autocorrect: bool = True,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.fps = fps
        self.mode = mode
        self.video_height = video_height
        self.likelihood_threshold = likelihood_threshold
        self.treadmill_speed = treadmill_speed
        self.real_world_multiplier = real_world_multiplier
        self.autocorrect = autocorrect

    def run(self) -> None:
        try:
            self.progress.emit("正在读取 CSV...")
            res = pipeline.analyze(
                self.csv_path,
                fps=self.fps,
                mode=self.mode,
                video_height=self.video_height,
                likelihood_threshold=self.likelihood_threshold,
                treadmill_speed=self.treadmill_speed,
                real_world_multiplier=self.real_world_multiplier,
                autocorrect=self.autocorrect,
            )
            self.progress.emit("分析完成")
            self.result_ready.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class FTIRWorker(QThread):
    """Run FTIR footprint + intensity analysis in a background thread.

    Phase 3 W10: migrated from the v1 ``footprint.analyze_frame`` to the
    v2 :func:`deepgait3.core._legacy.footprint_v2.analyze_frame_v2`, which adds
    MouseWalker-style background modelling, union-find 4-paw grouping,
    and L/R + F/H quadrant classification (W6 deliverable).

    W18 (2026-07-05): replaced ``footprint_v2.analyze_frame_v2`` with
    :func:`deepgait3.core.pawprint.single_frame.detect_single_frame`, which uses
    the pawprint module's ``detect_blobs`` + ``cluster_blobs_into_feet``
    primitives (tau_paw thresholding, DBSCAN merging) backed by the
    deepgait-v2 dynamics_v0.4.2 algorithm.
    """

    result_ready = Signal(object)   # FootprintSequence (footprint_v2)
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        frame: Any,  # numpy array (BGR)
        body_axis: tuple[tuple[float, float], tuple[float, float]] | None = None,
        px_per_mm: float | None = None,
        hsv_lower: tuple[int, int, int] = (35, 50, 30),
        hsv_upper: tuple[int, int, int] = (85, 255, 255),
        min_area_px: int = 50,
        background: Any | None = None,   # BackgroundModel or None
        in_stance_threshold_px: int = 30,
        # pawprint-specific (W18)
        tau_paw: float = 10.0,
        D_merge_px: float = 23.0,
        walkway_roi: tuple[int, int, int, int] = (0, 15, 1920, 360),
        bbox_pad_px: int = 8,
    ) -> None:
        super().__init__()
        self.frame = frame
        self.body_axis = body_axis
        self.px_per_mm = px_per_mm
        self.hsv_lower = hsv_lower
        self.hsv_upper = hsv_upper
        self.min_area_px = min_area_px
        self.background = background
        self.in_stance_threshold_px = in_stance_threshold_px
        self.tau_paw = tau_paw
        self.D_merge_px = D_merge_px
        self.walkway_roi = walkway_roi
        self.bbox_pad_px = bbox_pad_px

    def run(self) -> None:
        try:
            import numpy as np
            from deepgait3.core.pawprint.single_frame import detect_single_frame

            self.progress.emit("正在分割足印 (pawprint tau=%.1f)..." % self.tau_paw)

            # Compute green-channel background.
            if self.background is not None:

                bg_median = self.background.get_median()
                if bg_median is not None and bg_median.size > 0:
                    bg_G = bg_median[:, :, 1].astype(np.float32)
                else:
                    bg_G = self.frame[:, :, 1].astype(np.float32)
            else:
                bg_G = self.frame[:, :, 1].astype(np.float32)

            seq = detect_single_frame(
                self.frame,
                bg_G,
                tau_paw=self.tau_paw,
                min_area_px=self.min_area_px,
                D_merge_px=self.D_merge_px,
                walkway_roi=self.walkway_roi,
                bbox_pad_px=self.bbox_pad_px,
                mouse_det=None,
                px_per_mm=self.px_per_mm,
                body_axis=self.body_axis,
                in_stance_threshold_px=self.in_stance_threshold_px,
            )
            self.progress.emit(f"分析完成 — {seq.n_feet} 足检测")
            self.result_ready.emit(seq)
        except Exception as e:
            self.error.emit(str(e))


class DLCWorker(QThread):
    """Run a DLC workflow step in a background thread."""

    step_done = Signal(str)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, step: str, **kwargs: Any) -> None:
        super().__init__()
        self.step = step
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            from deepgait3.core._legacy import dlc_workflow
            if self.step == "create_project":
                self.progress.emit("正在创建项目...")
                spec = self.kwargs["spec"]
                dlc_workflow.create_project(spec)
                self.step_done.emit("项目创建完成")
            elif self.step == "extract_frames":
                self.progress.emit("正在抽帧...")
                dlc_workflow.extract_frames(self.kwargs["config"])
                self.step_done.emit("抽帧完成")
            elif self.step == "train":
                self.progress.emit("正在生成训练集...")
                dlc_workflow.create_training_dataset(
                    self.kwargs["config"],
                    net_type=self.kwargs.get("net_type", "resnet_50"),
                )
                self.progress.emit("正在训练...")
                dlc_workflow.train_network(
                    self.kwargs["config"],
                    epochs=self.kwargs.get("epochs", 200),
                    batch_size=self.kwargs.get("batch_size", 8),
                    device=self.kwargs.get("device", None),
                )
                self.step_done.emit("训练完成")
            elif self.step == "analyze_videos":
                self.progress.emit("正在分析视频...")
                dlc_workflow.analyze_videos(
                    self.kwargs["config"],
                    self.kwargs["videos"],
                    device=self.kwargs.get("device", None),
                    batch_size=self.kwargs.get("batch_size", 8),
                )
                self.step_done.emit("视频分析完成")
            else:
                self.error.emit(f"未知步骤: {self.step}")
        except DLCNotInstalledError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))


class CameraWorker(QThread):
    """Capture frames from a camera or video file in a background thread.

    Emits ``frame_ready(np.ndarray)`` for every frame captured.
    Set ``stop()`` or ``running = False`` to terminate the loop.
    """

    frame_ready = Signal(object)   # np.ndarray (BGR)
    fps_updated = Signal(float)
    error = Signal(str)

    def __init__(
        self,
        source: int | str = 0,           # camera index or video path
        target_fps: int = 30,
        width: int = 640,
        height: int = 480,
        save_path: str | None = None,     # optional video writer
    ) -> None:
        super().__init__()
        self.source = source
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.save_path = save_path
        self._stop = False
        self._cap: cv2.VideoCapture | None = None

    def stop(self) -> None:
        self._stop = True

    def _open_capture(self) -> "cv2.VideoCapture | None":
        import cv2
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def run(self) -> None:
        import cv2
        self._cap = self._open_capture()
        if self._cap is None:
            self.error.emit(f"无法打开视频源: {self.source}")
            return
        writer: cv2.VideoWriter | None = None
        if self.save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                self.save_path, fourcc, self.target_fps,
                (self.width, self.height),
            )

        frame_count = 0
        start = cv2.getTickCount()
        try:
            while not self._stop:
                ok, frame = self._cap.read()
                if not ok:
                    if isinstance(self.source, str):  # file ended
                        break
                    continue
                if writer:
                    writer.write(frame)
                self.frame_ready.emit(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = (cv2.getTickCount() - start) / cv2.getTickFrequency()
                    fps = frame_count / elapsed if elapsed > 0 else 0.0
                    self.fps_updated.emit(fps)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if writer:
                writer.release()
            if self._cap:
                self._cap.release()
            self._cap = None

