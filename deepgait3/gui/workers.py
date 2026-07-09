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
from deepgait3.core.pawprint.trim import TrimError, trim_video
from deepgait3.core.pawprint.yolo_detector import YoloPawDetector


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
    """Run YOLO footprint detection in a background thread.

    Uses :func:`deepgait3.core.pawprint.single_frame.detect_single_frame`
    which runs YOLOv8n-seg on GPU, returning native pawprint FootMask objects
    with precise mask data.
    """

    result_ready = Signal(object)   # list[FootMask] (pawprint FootMask)
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        frame: Any,
        min_area_px: int = 5,
        conf: float = 0.25,
        background: Any | None = None,
        # accepted for caller compat, ignored
        tau_paw: float = 0,
        D_merge_px: float = 0,
        walkway_roi: tuple = (),
        bbox_pad_px: int = 0,
        body_axis: Any = None,
        px_per_mm: float | None = None,
        hsv_lower: tuple = (),
        hsv_upper: tuple = (),
        in_stance_threshold_px: int = 0,
    ) -> None:
        super().__init__()
        self.frame = frame
        self.min_area_px = min_area_px
        self.conf = conf
        self.background = background

    def run(self) -> None:
        try:
            import numpy as np
            from deepgait3.core.pawprint.single_frame import detect_single_frame

            self.progress.emit("YOLO pawprint detection...")

            # Compute green-channel background
            if self.background is not None:
                bg_median = self.background.get_median()
                if bg_median is not None and bg_median.size > 0:
                    bg_G = bg_median[:, :, 1].astype(np.float32)
                else:
                    bg_G = self.frame[:, :, 1].astype(np.float32)
            else:
                bg_G = self.frame[:, :, 1].astype(np.float32)

            footmasks = detect_single_frame(
                self.frame, bg_G,
                min_area_px=self.min_area_px,
                conf=self.conf,
            )
            self.progress.emit(f"分析完成 — {len(footmasks)} 足检测")
            self.result_ready.emit(footmasks)
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



class TrimWorker(QThread):
    """Background trim worker.  Processes a list of videos sequentially.

    For each video:
      1. ``video_started(idx, total, src_path)``
      2. ``video_progress(idx, total, pct, msg)`` during YOLO + tracker + encode
      3. ``video_done(idx, src_path, dst_path)`` on success
         OR ``video_failed(idx, src_path, error_msg)`` on failure

    After the batch:
      ``all_done(success_count, failed_count)`` is emitted exactly once.
    """

    video_started = Signal(int, int, str)
    video_progress = Signal(int, int, int, str)
    video_done = Signal(int, str, str)
    video_failed = Signal(int, str, str)
    all_done = Signal(int, int)

    def __init__(
        self,
        jobs: list[dict],
        *,
        model_path: str | Path | None = None,
        iou_min: float = 0.3,
        max_gap_frames: int = 3,
        min_area_px: int = 5,
    ) -> None:
        super().__init__()
        self.jobs = jobs  # each: {"src": Path, "dst": Path}
        self.model_path = model_path
        self.iou_min = iou_min
        self.max_gap_frames = max_gap_frames
        self.min_area_px = min_area_px
        self._detector: YoloPawDetector | None = None

    @property
    def detector(self) -> YoloPawDetector:
        if self._detector is None:
            self._detector = YoloPawDetector(model_path=self.model_path)
        return self._detector

    def run(self) -> None:
        success = 0
        failed = 0
        total = len(self.jobs)
        for idx, job in enumerate(self.jobs):
            src = Path(job["src"])
            dst = Path(job["dst"])
            self.video_started.emit(idx, total, str(src))
            try:
                def cb(phase: str, pct: int, msg: str) -> None:
                    self.video_progress.emit(idx, total, pct, msg)

                result = trim_video(
                    src, dst,
                    detector=self.detector,
                    iou_min=self.iou_min,
                    max_gap_frames=self.max_gap_frames,
                    min_area_px=self.min_area_px,
                    progress_cb=cb,
                )
                self.video_done.emit(idx, str(src), str(result["dst"]))
                success += 1
            except TrimError as e:
                self.video_failed.emit(idx, str(src), str(e))
                failed += 1
            except Exception as e:
                self.video_failed.emit(idx, str(src), f"unexpected: {e}")
                failed += 1
        self.all_done.emit(success, failed)
