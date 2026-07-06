"""3D triangulation + calibration (Layer 4 of deepgait v2).

Pure-Python implementation of the algorithm family used by
``anipose`` (kb/10_anipose_3d_pipeline.md §5):

1. ``dlt_triangulate`` — Hartley-Sturm direct linear transform (DLT)
   from an arbitrary number of camera views. SVD-based, with optional
   normalisation (Hartley preconditioning) for numerical stability.

2. ``reprojection_error`` — per-camera Euclidean distance between the
   observed 2D points and the projection of the triangulated 3D point.

3. ``triangulate_ransac`` — robust N-camera triangulation. Samples
   2-camera pairs, computes a candidate 3D point per pair, picks the
   consensus point with the lowest median reprojection error.

4. ``optim_points`` — joint refinement minimising reprojection error
   + bone-length consistency + temporal smoothness (the Anipose
   "optim_points" innovation, kb/10 §5.3).

5. ``filter_3d`` — sliding-window median + linear interpolation to
   remove outlier frames (Anipose ``filter_3d``, kb/10 §6).

6. ``calibrate_charuco`` — ChArUco intrinsic calibration for one camera,
   returning ``(K, dist, rvecs, tvecs, reproj_rms)``. Multi-camera
   extrinsics are computed by triangulating a ChArUco origin from ≥ 2
   intrinsics-calibrated cameras (see ``calibrate_multi_camera``).

Acceptance gate (DEVELOPMENT_PLAN §6.2 W7):
    "reprojection error < 3 px"

References
----------
* kb/10_anipose_3d_pipeline.md (sections 4-6)
* MODULES.md §2.5 "triangulation_3d.py / .pyx"
* docs/REQUIREMENTS.md FR-3D-001..009
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera model — projection matrix P (3×4)
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """One calibrated camera with a 3×4 projection matrix.

    ``P = K [R | t]`` where K is the intrinsic matrix, R the world-to-
    camera rotation and t the world-to-camera translation. The model
    assumes zero lens distortion — distortion correction is the
    caller's responsibility (see ``undistort_points``).
    """
    name: str
    P: np.ndarray                 # shape (3, 4)
    K: Optional[np.ndarray] = None
    dist: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project 3D world points → 2D image points.

        Accepts ``(3,)``, ``(N, 3)`` or any shape that flattens to a
        ``(N, 3)`` matrix; returns the matching ``(2,)`` or ``(N, 2)``
        shape.
        """
        arr = np.asarray(points_3d, dtype=np.float64)
        single = arr.ndim == 1
        pts = arr.reshape(-1, 3)
        homog = np.column_stack((pts, np.ones(len(pts))))
        proj = homog @ self.P.T            # (N, 3)
        z = proj[:, 2:3]
        z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        out = proj[:, :2] / z
        if single:
            return out.reshape(2)
        return out


def camera_from_intrinsics(
    name: str, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
    dist: Optional[np.ndarray] = None,
) -> Camera:
    """Build a :class:`Camera` from intrinsic + extrinsic OpenCV-style params.

    P = K [R | t] where R is built from rvec via Rodrigues.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    R, _ = _rodrigues(rvec)
    P = np.hstack([R, tvec.reshape(3, 1)])  # world -> camera: [R|t]
    P = K @ P                               # world -> pixel
    return Camera(
        name=name, P=P, K=K,
        dist=np.asarray(dist, dtype=np.float64).reshape(-1) if dist is not None else None,
        rvec=rvec, tvec=tvec,
    )


def _rodrigues(rvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OpenCV-compatible Rodrigues rotation-vector → matrix converter.

    Returns (R, dR/drvec) where dR is unused but matches OpenCV's
    signature for cross-compatibility.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3), np.zeros((3, 3, 3))
    k = rvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0],
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R, np.zeros((3, 3, 3))


# ---------------------------------------------------------------------------
# 0. Public guard helpers (W16 P0#2 fix)
# ---------------------------------------------------------------------------
def _check_two_cameras(P_list, points_2d_list=None) -> None:
    """Raise a clear ``ValueError`` when fewer than 2 cameras are given.

    Replaces the opaque ``IndexError`` deep in ``numpy.linalg.lstsq``
    that bit CAS in beta (CAS-P0#2, 2026-10-18) when they had only
    calibrated the ``left`` camera before clicking 三角化.

    The error message is intentionally short so it can be surfaced
    verbatim in a ``QMessageBox``.
    """
    n = len(P_list)
    if n < 2:
        raise ValueError(
            f"3D triangulation needs >= 2 cameras (got {n}). "
            f"Run 3D Calibration first to calibrate the missing cameras."
        )
    if points_2d_list is not None and n != len(points_2d_list):
        raise ValueError(
            f"P_list has {n} cameras but points_2d_list has "
            f"{len(points_2d_list)} observations — they must match."
        )


# ---------------------------------------------------------------------------
# 1. DLT triangulation (Hartley-Sturm, multi-camera)
# ---------------------------------------------------------------------------
def dlt_triangulate(
    P_list: Sequence[np.ndarray],
    points_2d_list: Sequence[np.ndarray],
) -> np.ndarray:
    """Triangulate a single 3D point from N ≥ 2 cameras via DLT.

    Parameters
    ----------
    P_list : sequence of (3, 4) projection matrices
    points_2d_list : sequence of (2,) image points (one per camera)

    Returns
    -------
    (3,) 3D point in world coordinates.

    References
    ----------
    Hartley & Sturm, "Triangulation", CVIU 1997.
    """
    if len(P_list) < 2:
        raise ValueError("dlt_triangulate requires >= 2 cameras")
    if len(P_list) != len(points_2d_list):
        raise ValueError("P_list and points_2d_list must have the same length")

    # Hartley preconditioning: translate each camera so its principal
    # point is at the origin and rescale so its average image radius is
    # 1. This dramatically improves numerical conditioning.
    normed_Ps = []
    normed_pts = []
    for P, pt in zip(P_list, points_2d_list):
        pt_arr = np.asarray(pt, dtype=np.float64).reshape(-1)[:2]
        T, _ = _hartley_normalisation(pt_arr)
        normed_Ps.append(T @ P)
        normed_pts.append(T @ np.array([pt_arr[0], pt_arr[1], 1.0]))
    normed_pts = np.asarray(normed_pts)            # (N, 3)

    # Build the 2N × 4 design matrix A: each camera contributes two rows.
    A_rows = []
    for P, x in zip(normed_Ps, normed_pts):
        A_rows.append(x[0] * P[2] - P[0])
        A_rows.append(x[1] * P[2] - P[1])
    A = np.asarray(A_rows)
    # SVD: X is the right-singular vector with smallest singular value.
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return X[:3] / X[3]


def _hartley_normalisation(pt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """3×3 normalisation matrix that centres & scales a single image point.

    Returns (T, T_inv) such that T @ [x, y, 1]^T ≈ [0, 0, 1]^T and
    T_inv undoes the transform.
    """
    T = np.array([
        [1.0,  0.0, -pt[0]],
        [0.0,  1.0, -pt[1]],
        [0.0,  0.0,  1.0],
    ])
    s = max(np.linalg.norm(pt[:2]), 1.0)
    T = T @ np.diag([1.0 / s, 1.0 / s, 1.0])
    T_inv = np.diag([s, s, 1.0]) @ np.array([
        [1.0, 0.0, pt[0]],
        [0.0, 1.0, pt[1]],
        [0.0, 0.0, 1.0],
    ])
    return T, T_inv


# ---------------------------------------------------------------------------
# 2. Reprojection error
# ---------------------------------------------------------------------------
def reprojection_error(
    P_list: Sequence[np.ndarray],
    points_2d_list: Sequence[np.ndarray],
    point_3d: np.ndarray,
) -> float:
    """Mean Euclidean reprojection error (pixels).

    Returns ``inf`` if any projection is degenerate (Z near zero).
    """
    point_3d = np.asarray(point_3d, dtype=np.float64).reshape(3)
    homog = np.array([*point_3d, 1.0])
    errs = []
    for P, pt in zip(P_list, points_2d_list):
        proj = P @ homog
        if abs(proj[2]) < 1e-9:
            return float("inf")
        x, y = proj[0] / proj[2], proj[1] / proj[2]
        errs.append(np.hypot(x - pt[0], y - pt[1]))
    return float(np.mean(errs))


def reprojection_error_per_camera(
    P_list: Sequence[np.ndarray],
    points_2d_list: Sequence[np.ndarray],
    point_3d: np.ndarray,
) -> np.ndarray:
    """Per-camera reprojection errors (length N)."""
    point_3d = np.asarray(point_3d, dtype=np.float64).reshape(3)
    homog = np.array([*point_3d, 1.0])
    errs = []
    for P, pt in zip(P_list, points_2d_list):
        proj = P @ homog
        if abs(proj[2]) < 1e-9:
            errs.append(float("inf"))
        else:
            x, y = proj[0] / proj[2], proj[1] / proj[2]
            errs.append(float(np.hypot(x - pt[0], y - pt[1])))
    return np.asarray(errs)


# ---------------------------------------------------------------------------
# 3. RANSAC multi-camera triangulation
# ---------------------------------------------------------------------------
def triangulate_ransac(
    P_list: Sequence[np.ndarray],
    points_2d_list: Sequence[np.ndarray],
    n_iters: int = 100,
    threshold_px: float = 3.0,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """Robust N-camera triangulation via RANSAC over 2-camera pairs.

    For each iteration, sample 2 cameras, DLT-triangulate, then score
    the candidate 3D point by median reprojection error across ALL
    cameras. The candidate with the lowest median error wins.

    Parameters
    ----------
    P_list, points_2d_list : N ≥ 2 cameras' projections + observations.
    n_iters : RANSAC iterations (default 100 — matches Anipose default).
    threshold_px : reprojection threshold for inlier counting (logged).
    rng_seed : optional RNG seed for reproducibility.

    Returns
    -------
    (3,) 3D point estimate.
    """
    n_cams = len(P_list)
    if n_cams < 2:
        raise ValueError("triangulate_ransac requires >= 2 cameras")
    rng = np.random.default_rng(rng_seed)

    best_X: Optional[np.ndarray] = None
    best_score = float("inf")
    best_inliers = 0

    for _ in range(n_iters):
        idx = rng.choice(n_cams, size=2, replace=False)
        P_a, P_b = P_list[idx[0]], P_list[idx[1]]
        pt_a, pt_b = points_2d_list[idx[0]], points_2d_list[idx[1]]
        try:
            X = dlt_triangulate([P_a, P_b], [pt_a, pt_b])
        except Exception:
            continue
        if np.any(np.isnan(X)):
            continue
        errs = reprojection_error_per_camera(P_list, points_2d_list, X)
        if np.any(np.isinf(errs)):
            continue
        score = float(np.median(errs))
        inliers = int(np.sum(errs <= threshold_px))
        if inliers > best_inliers or (inliers == best_inliers and score < best_score):
            best_X = X
            best_score = score
            best_inliers = inliers

    if best_X is None:
        # Fallback: plain DLT on all cameras.
        return dlt_triangulate(P_list, points_2d_list)
    return best_X


# ---------------------------------------------------------------------------
# 4. optim_points — joint refinement (reprojection + bone-length + smooth)
# ---------------------------------------------------------------------------
# Bone graph: each entry is (parent_idx, child_idx, target_length).
# Caller supplies this so the algorithm is bodypart-scheme agnostic.
BoneEdge = Tuple[int, int, float]


def optim_points(
    P_list: Sequence[np.ndarray],
    points_2d_per_cam: Sequence[np.ndarray],
    initial_3d: np.ndarray,
    bones: Sequence[BoneEdge],
    *,
    smoothness_weight: float = 1.0,
    bone_weight: float = 1.0,
    reproj_weight: float = 1.0,
    n_iters: int = 1,
) -> np.ndarray:
    """Joint refinement of per-frame 3D points.

    Optimises one frame at a time (or a single static 3D point) by
    minimising:
        * reprojection error (all cameras)
        * bone-length consistency (each bone length ≈ target)
        * smoothness (||d²X/dt²||²)  — when ``initial_3d`` has shape
          ``(T, n_bodyparts, 3)``

    Parameters
    ----------
    P_list : list of (3,4) projection matrices
    points_2d_per_cam : list of arrays. Each array is either:
        * (n_bodyparts, 2) for a single frame, or
        * (T, n_bodyparts, 2) for a sequence of frames.
    initial_3d : (n_bodyparts, 3) or (T, n_bodyparts, 3)
    bones : list of (parent_idx, child_idx, target_length)
    smoothness_weight, bone_weight, reproj_weight : scalar weights.
    n_iters : number of refinement passes.

    Returns
    -------
    Refined 3D points with the same shape as ``initial_3d``.
    """
    try:
        from scipy.optimize import least_squares
    except ImportError:    # pragma: no cover - scipy is a hard dep
        logger.warning("scipy.optimize.least_squares unavailable; "
                       "optim_points is a no-op")
        return np.asarray(initial_3d, dtype=np.float64)

    # W16 fix (CAS-P0#2): 1-camera calibration used to crash deep in
    # numpy.linalg.lstsq with an opaque IndexError. Now we fail fast
    # with a clear message at every public entry point.
    _check_two_cameras(P_list)

    P_list = [np.asarray(P, dtype=np.float64) for P in P_list]
    init = np.asarray(initial_3d, dtype=np.float64)

    if init.ndim == 2:
        return _optim_single_frame(
            least_squares, P_list, points_2d_per_cam, init, bones,
            reproj_weight, bone_weight,
        )
    # (T, J, 3) sequence.
    refined = init.copy()
    for t in range(init.shape[0]):
        cams_obs = [pts[t] for pts in points_2d_per_cam]
        refined[t] = _optim_single_frame(
            least_squares, P_list, cams_obs, refined[t], bones,
            reproj_weight, bone_weight,
        )
    if smoothness_weight > 0 and refined.shape[0] >= 3:
        for _ in range(n_iters):
            refined = _smooth_one_pass(refined, smoothness_weight)
    return refined


def _optim_single_frame(
    least_squares, P_list, points_2d, init_3d, bones,
    reproj_weight, bone_weight,
) -> np.ndarray:
    """Optimise one frame's 3D points against reprojection + bone costs."""
    n_bp = init_3d.shape[0]
    n_cams = len(P_list)
    obs = [np.asarray(p, dtype=np.float64) for p in points_2d]

    def _cost(x: np.ndarray) -> np.ndarray:
        X = x.reshape(n_bp, 3)
        residuals: List[float] = []
        # Reprojection error.
        for P, pt in zip(P_list, obs):
            for j in range(n_bp):
                proj = P @ np.array([X[j, 0], X[j, 1], X[j, 2], 1.0])
                if abs(proj[2]) < 1e-9:
                    residuals.append(1e3); residuals.append(1e3)
                    continue
                rx = proj[0] / proj[2] - pt[j, 0]
                ry = proj[1] / proj[2] - pt[j, 1]
                residuals.append(reproj_weight * rx)
                residuals.append(reproj_weight * ry)
        # Bone-length consistency.
        for parent, child, target in bones:
            d = np.linalg.norm(X[parent] - X[child]) - target
            residuals.append(bone_weight * 100.0 * d)
        return np.asarray(residuals)

    # Choose solver: 'lm' needs m >= n; with too few cameras the
    # reprojection residuals may under-determine the system, so fall
    # back to 'trf' (trust-region reflective, supports m < n).
    n_vars = n_bp * 3
    n_resid = 2 * n_bp * n_cams + len(bones)
    method = "lm" if n_resid >= n_vars else "trf"
    res = least_squares(_cost, init_3d.flatten(), method=method)
    return res.x.reshape(n_bp, 3)


def _smooth_one_pass(X: np.ndarray, weight: float) -> np.ndarray:
    """One Laplacian smoothing pass: X[t] += weight * (X[t-1] + X[t+1] - 2 X[t])."""
    out = X.copy()
    out[1:-1] += weight * (X[:-2] - 2 * X[1:-1] + X[2:])
    return out


# ---------------------------------------------------------------------------
# 5. filter_3d — median + linear interpolation
# ---------------------------------------------------------------------------
def filter_3d(
    keypoints_3d: np.ndarray,
    window: int = 5,
    max_jump_mm: Optional[float] = None,
) -> np.ndarray:
    """Sliding-window median filter + optional jump clipping.

    Parameters
    ----------
    keypoints_3d : (T, J, 3) 3D positions (any unit; the jump threshold
        is in the same units).
    window : median filter window (odd integer; default 5).
    max_jump_mm : if set, frames whose per-joint jump from the previous
        frame exceeds this threshold are replaced with NaN and linearly
        interpolated.

    Returns
    -------
    (T, J, 3) filtered keypoints.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    X = np.asarray(keypoints_3d, dtype=np.float64).copy()
    T, J, _ = X.shape
    half = window // 2
    # Median filter.
    med = np.zeros_like(X)
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        med[t] = np.median(X[lo:hi], axis=0)
    X = med

    # Optional jump-based outlier rejection.
    if max_jump_mm is not None and T > 1:
        jumps = np.linalg.norm(np.diff(X, axis=0), axis=2)  # (T-1, J)
        bad = jumps > max_jump_mm
        # Mark the bad destination frames as NaN.
        for j in range(J):
            for t in range(1, T):
                if bad[t - 1, j]:
                    X[t, j] = np.nan
        # Linear interpolate NaNs.
        for j in range(J):
            for c in range(3):
                col = X[:, j, c]
                mask = np.isnan(col)
                if mask.any() and not mask.all():
                    valid = np.where(~mask)[0]
                    col[mask] = np.interp(np.where(mask)[0], valid, col[valid])
                    X[:, j, c] = col
    return X


# ---------------------------------------------------------------------------
# 6. ChArUco intrinsic calibration (one camera)
# ---------------------------------------------------------------------------
def calibrate_charuco(
    images: Sequence[np.ndarray],
    *,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dictionary_id: int = None,
) -> Dict[str, np.ndarray]:
    """ChArUco intrinsic calibration for one camera.

    Returns a dict with keys:
        * ``K``            — (3, 3) intrinsic matrix
        * ``dist``         — (5,) or more distortion coefficients
        * ``rvecs``        — list of (3,) per-image rotation vectors
        * ``tvecs``        — list of (3,) per-image translation vectors
        * ``reproj_rms``   — RMS reprojection error in pixels

    Falls back gracefully when ``cv2.aruco`` is unavailable (returns
    NaN-filled arrays and ``reproj_rms = inf`` so callers can detect it).
    """
    try:
        import cv2
        aruco = cv2.aruco
        dictionary = (
            aruco.getPredefinedDictionary(dictionary_id)
            if dictionary_id is not None
            else aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        )
        board = aruco.CharucoBoard_create(
            squares_x, squares_y, square_length, marker_length, dictionary,
        )
    except Exception as e:
        logger.warning("calibrate_charuco: OpenCV aruco unavailable (%s)", e)
        return {
            "K": np.full((3, 3), np.nan), "dist": np.full(5, np.nan),
            "rvecs": [], "tvecs": [],
            "reproj_rms": float("inf"),
        }

    all_corners, all_ids = [], []
    for img in images:
        if img.ndim == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)
        if ids is None or len(ids) == 0:
            continue
        ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            corners, ids, gray, board,
        )
        if ret and charuco_ids is not None and len(charuco_ids) >= 4:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)

    if not all_corners:
        return {
            "K": np.eye(3), "dist": np.zeros(5),
            "rvecs": [], "tvecs": [],
            "reproj_rms": float("inf"),
        }

    flags = cv2.CALIB_RATIONAL_MODEL
    try:
        ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, images[0].shape[1::-1],
            None, None, flags=flags,
        )
    except Exception:
        ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
            all_corners, all_ids, board, images[0].shape[1::-1],
            None, None,
        )
    return {
        "K": np.asarray(K, dtype=np.float64),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1),
        "rvecs": [np.asarray(r, dtype=np.float64).reshape(3) for r in rvecs],
        "tvecs": [np.asarray(t, dtype=np.float64).reshape(3) for t in tvecs],
        "reproj_rms": float(ret),
    }


# ---------------------------------------------------------------------------
# 7. Public exports
# ---------------------------------------------------------------------------
__all__ = [
    "Camera",
    "camera_from_intrinsics",
    "dlt_triangulate",
    "reprojection_error",
    "reprojection_error_per_camera",
    "triangulate_ransac",
    "optim_points",
    "filter_3d",
    "calibrate_charuco",
    "_check_two_cameras",
]