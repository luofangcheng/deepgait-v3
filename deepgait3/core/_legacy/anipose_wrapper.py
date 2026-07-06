"""Anipose wrapper (Layer 4 of deepgait v2).

Provides a thin, stable facade over the 3D pipeline described in
kb/10_anipose_3d_pipeline.md:

* ``AniposeWrapper.calibrate(images_per_cam, board)``
* ``AniposeWrapper.triangulate(pose_2d, calibration)``
* ``AniposeWrapper.compute_angles(keypoints_3d, skeleton)``

If the third-party ``aniposelib`` package is available (the canonical
implementation), it is used as-is. Otherwise, the wrapper falls back to
deepgait's in-house implementation in
:mod:`deepgait3.core._legacy.triangulation_3d`. This keeps unit tests runnable
without aniposelib installed.

CLOSED-SOURCE NOTE: production builds compile this module with Cython
(kb/18_cython_nuitka.md §2.3) so the bone graph and triangulation
algorithm cannot be trivially reverse-engineered.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .triangulation_3d import (
    Camera,
    calibrate_charuco,
    camera_from_intrinsics,
    dlt_triangulate,
    filter_3d,
    optim_points,
    reprojection_error,
    triangulate_ransac,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# aniposelib availability probe
# ---------------------------------------------------------------------------
def _aniposelib_available() -> bool:
    try:
        import aniposelib  # noqa: F401
        return True
    except Exception:
        return False


ANIPOSELIB_AVAILABLE = _aniposelib_available()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class CharucoBoard:
    """ChArUco board definition (matches both OpenCV and aniposelib)."""
    squares_x: int
    squares_y: int
    square_length: float       # mm
    marker_length: float       # mm
    dictionary_id: int = None  # default: cv2.aruco.DICT_6X6_250

    def to_dict(self) -> dict:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length": self.square_length,
            "marker_length": self.marker_length,
            "dictionary_id": self.dictionary_id,
        }


@dataclass
class CalibrationResult:
    """Per-camera intrinsics + extrinsics from ChArUco calibration."""
    cameras: Dict[str, Camera]
    reproj_rms_per_cam: Dict[str, float]
    board: CharucoBoard
    method: str                              # "aniposelib" | "inhouse"
    extras: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default skeleton — 12-keypoint VGL scheme (kb/10 §7 option b)
# ---------------------------------------------------------------------------
DEFAULT_SKELETON_12 = [
    # (parent_name, child_name) — connectivity for bone-length targets.
    ("Nose", "MidPointFront"),
    ("MidPointFront", "Tail"),
    ("MidPointFront", "MidPointLeft"),
    ("MidPointFront", "MidPointRight"),
    ("MidPointLeft", "FrontLeft1"),
    ("MidPointLeft", "FrontLeft2"),
    ("MidPointRight", "FrontRight1"),
    ("MidPointRight", "FrontRight2"),
    ("MidPointLeft", "HindLeft1"),
    ("MidPointLeft", "HindLeft2"),
    ("MidPointRight", "HindRight1"),
    ("MidPointRight", "HindRight2"),
]


DEFAULT_BODYPARTS_12 = [
    "Nose", "Tail",
    "MidPointFront", "MidPointLeft", "MidPointRight",
    "FrontLeft1", "FrontLeft2",
    "FrontRight1", "FrontRight2",
    "HindLeft1", "HindLeft2",
    "HindRight1", "HindRight2",
]


# ---------------------------------------------------------------------------
# AniposeWrapper
# ---------------------------------------------------------------------------
class AniposeWrapper:
    """Stable facade over aniposelib OR the in-house fallback.

    Usage::

        board = CharucoBoard(squares_x=7, squares_y=5,
                              square_length=30, marker_length=22)
        wrapper = AniposeWrapper()
        calib = wrapper.calibrate(images_per_cam, board)
        pose_3d = wrapper.triangulate(pose_2d_dict, calib)
        angles = wrapper.compute_angles(pose_3d, DEFAULT_SKELETON_12)
    """

    def __init__(self, prefer: str = "auto") -> None:
        """``prefer`` ∈ {"auto", "aniposelib", "inhouse"}.

        * "auto"      — use aniposelib if installed, else in-house.
        * "aniposelib" — fail if aniposelib is missing.
        * "inhouse"    — always use the in-house implementation.
        """
        if prefer not in ("auto", "aniposelib", "inhouse"):
            raise ValueError("prefer must be auto | aniposelib | inhouse")
        if prefer == "aniposelib" and not ANIPOSELIB_AVAILABLE:
            raise RuntimeError("aniposelib requested but not installed")
        self.prefer = prefer

    # ---- selection helpers ------------------------------------------------
    @property
    def active_method(self) -> str:
        if self.prefer == "inhouse":
            return "inhouse"
        if self.prefer == "aniposelib":
            return "aniposelib"
        return "aniposelib" if ANIPOSELIB_AVAILABLE else "inhouse"

    # ---- 1. calibrate -----------------------------------------------------
    def calibrate(
        self,
        images_per_cam: Dict[str, List[np.ndarray]],
        board: CharucoBoard,
    ) -> CalibrationResult:
        """Calibrate one or more cameras from ChArUco detection.

        ``images_per_cam`` maps camera name → list of BGR images showing
        the board. The intrinsic K and distortion are computed per
        camera; extrinsics (rvec/tvec of each board pose) are stored on
        the :class:`Camera` for downstream triangulation seeding.
        """
        if self.active_method == "aniposelib":
            return self._calibrate_aniposelib(images_per_cam, board)
        return self._calibrate_inhouse(images_per_cam, board)

    def _calibrate_inhouse(
        self,
        images_per_cam: Dict[str, List[np.ndarray]],
        board: CharucoBoard,
    ) -> CalibrationResult:
        cameras: Dict[str, Camera] = {}
        rms: Dict[str, float] = {}
        # 1) Intrinsics per camera.
        for cam_name, imgs in images_per_cam.items():
            res = calibrate_charuco(
                imgs,
                squares_x=board.squares_x, squares_y=board.squares_y,
                square_length=board.square_length,
                marker_length=board.marker_length,
                dictionary_id=board.dictionary_id,
            )
            # Use the first rvec/tvec as the camera's "world" pose (relative
            # to the board). In a real multi-cam setup the extrinsics
            # between cameras come from triangulating a known target;
            # for the single-camera-calibrated stub we just attach the
            # first board pose so Camera.project() works.
            if res["rvecs"]:
                cameras[cam_name] = camera_from_intrinsics(
                    name=cam_name, K=res["K"], dist=res["dist"],
                    rvec=res["rvecs"][0], tvec=res["tvecs"][0],
                )
            else:
                cameras[cam_name] = Camera(name=cam_name, P=np.eye(3, 4))
            rms[cam_name] = res["reproj_rms"]
        return CalibrationResult(
            cameras=cameras, reproj_rms_per_cam=rms,
            board=board, method="inhouse",
        )

    def _calibrate_aniposelib(
        self,
        images_per_cam: Dict[str, List[np.ndarray]],
        board: CharucoBoard,
    ) -> CalibrationResult:  # pragma: no cover - exercised when installed
        from aniposelib.cameras import CameraGroup
        cg = CameraGroup(list(images_per_cam.keys()))
        for name, imgs in images_per_cam.items():
            cg.calibrate_cameras(
                {name: imgs},
                board_square_length_mm=board.square_length,
                board_marker_length_mm=board.marker_length,
            )
        cameras: Dict[str, Camera] = {}
        rms: Dict[str, float] = {}
        for name in images_per_cam:
            sh = cg.cameras[name]
            P = sh.get_P() if hasattr(sh, "get_P") else np.eye(3, 4)
            cameras[name] = Camera(name=name, P=np.asarray(P, dtype=np.float64))
            rms[name] = float(getattr(sh, "reproj_error", float("inf")))
        return CalibrationResult(
            cameras=cameras, reproj_rms_per_cam=rms,
            board=board, method="aniposelib",
        )

    # ---- 2. triangulate ---------------------------------------------------
    def triangulate(
        self,
        pose_2d: Dict[str, np.ndarray],
        calibration: CalibrationResult,
        *,
        use_ransac: bool = True,
        ransac_iters: int = 100,
        ransac_threshold_px: float = 3.0,
        optim: bool = True,
        bones: Optional[Sequence[Tuple[int, int, float]]] = None,
        smoothness_weight: float = 1.0,
    ) -> np.ndarray:
        """Triangulate per-frame 3D pose from multi-camera 2D observations.

        Parameters
        ----------
        pose_2d : ``{camera_name: array of shape (T, J, 2)}``
        calibration : result of :meth:`calibrate`.
        use_ransac : robust RANSAC triangulation (default True).
        optim : run :func:`optim_points` joint refinement (default True).
        bones : bone-length targets (parent_idx, child_idx, length_mm).
        """
        cam_names = list(pose_2d.keys())
        cams = [calibration.cameras[n] for n in cam_names]
        P_list = [c.P for c in cams]

        T, J, _ = pose_2d[cam_names[0]].shape
        keypoints_3d = np.full((T, J, 3), np.nan, dtype=np.float64)
        for t in range(T):
            for j in range(J):
                pts = [pose_2d[c][t, j] for c in cam_names]
                if any(np.any(np.isnan(p)) or np.any(np.isinf(p)) for p in pts):
                    continue
                if use_ransac:
                    keypoints_3d[t, j] = triangulate_ransac(
                        P_list, pts, n_iters=ransac_iters,
                        threshold_px=ransac_threshold_px,
                    )
                else:
                    keypoints_3d[t, j] = dlt_triangulate(P_list, pts)

        # Confidence = inverse median reprojection error (clipped to [0, 1]).
        confidence = np.zeros((T, J), dtype=np.float64)
        for t in range(T):
            for j in range(J):
                X = keypoints_3d[t, j]
                if np.any(np.isnan(X)):
                    continue
                err = reprojection_error(P_list,
                    [pose_2d[c][t, j] for c in cam_names], X)
                confidence[t, j] = 1.0 / (1.0 + err)

        if optim and bones:
            keypoints_3d = optim_points(
                P_list,
                [pose_2d[c] for c in cam_names],
                keypoints_3d, bones,
                smoothness_weight=smoothness_weight,
                bone_weight=1.0, reproj_weight=1.0,
            )
        return keypoints_3d

    # ---- 3. compute_angles ------------------------------------------------
    def compute_angles(
        self,
        keypoints_3d: np.ndarray,
        skeleton: Sequence[Tuple[str, str]],
        bodyparts: Optional[Sequence[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute per-frame joint angles for a named skeleton.

        Each joint is the angle at ``child`` in the chain
        ``grandparent → parent → child`` (i.e. the angle between the
        two incident bone vectors). When the parent has no parent in
        the graph (root), the angle is reported as 0.
        """
        if bodyparts is None:
            bodyparts = DEFAULT_BODYPARTS_12
        # Build parent map.
        children = [c for _, c in skeleton]
        parents = {c: p for p, c in skeleton}
        angles: Dict[str, np.ndarray] = {}
        X = np.asarray(keypoints_3d, dtype=np.float64)
        T, J, _ = X.shape
        for j, name in enumerate(bodyparts):
            if name not in parents or j >= J:
                continue
            parent = parents[name]
            try:
                p_idx = bodyparts.index(parent)
            except ValueError:
                continue
            # Find grandparent.
            gp_idx = None
            if parent in parents:
                gp = parents[parent]
                try:
                    gp_idx = bodyparts.index(gp)
                except ValueError:
                    gp_idx = None
            # W16 fix (EMBL-P2#4): emit NaN for frames where the joint /
            # parent / grandparent is missing or coincident, so the
            # caller can decide what to do, instead of the misleading
            # 0.0 from a silent default.
            arr = np.full(T, np.nan, dtype=np.float64)
            for t in range(T):
                if np.any(np.isnan(X[t, j])) or np.any(np.isnan(X[t, p_idx])):
                    continue
                v1 = X[t, p_idx] - X[t, j]
                v2 = (X[t, gp_idx] - X[t, p_idx]) if gp_idx is not None else v1
                # Angle between v1 and v2 (degrees).
                n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                if n1 < 1e-9 or n2 < 1e-9:
                    continue
                cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                arr[t] = float(np.degrees(np.arccos(cos)))
            angles[name] = arr
        return angles

    # ---- 4. filter_3d passthrough ----------------------------------------
    def filter_3d(
        self,
        keypoints_3d: np.ndarray,
        window: int = 5,
        max_jump_mm: Optional[float] = None,
    ) -> np.ndarray:
        return filter_3d(keypoints_3d, window=window, max_jump_mm=max_jump_mm)


__all__ = [
    "CharucoBoard",
    "CalibrationResult",
    "DEFAULT_BODYPARTS_12",
    "DEFAULT_SKELETON_12",
    "AniposeWrapper",
    "ANIPOSELIB_AVAILABLE",
]