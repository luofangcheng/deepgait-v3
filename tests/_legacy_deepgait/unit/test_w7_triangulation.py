"""Unit tests for Phase 2 W7 — 3D triangulation + Anipose wrapper.

Covers:
    deepgait/core/triangulation_3d.py
        * Camera / camera_from_intrinsics
        * dlt_triangulate (Hartley-Sturm, multi-camera)
        * reprojection_error + reprojection_error_per_camera
        * triangulate_ransac (robust N-camera)
        * optim_points (joint reprojection + bone-length refinement)
        * filter_3d (median + jump-clipping interpolation)
        * calibrate_charuco (single-camera intrinsic)

    deepgait/core/anipose_wrapper.py
        * AniposeWrapper.calibrate / .triangulate / .compute_angles
        * CharucoBoard / DEFAULT_BODYPARTS_12 / DEFAULT_SKELETON_12
        * Auto fallback to in-house when aniposelib is missing

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.2 W7):
    "reprojection error < 3 px"
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# Fixtures — synthetic multi-camera rig
# =============================================================================
@pytest.fixture
def synthetic_rig():
    """Build a 3-camera rig + a single 3D point with KNOWN projection.

    The cameras look at the origin from different angles; the 3D point
    is at world (10, 20, 30) mm. We project it through each camera to
    obtain the 2D observations, then triangulate back and verify the
    reconstruction error is < 3 px (the W7 acceptance gate).
    """
    from deepgait3.core._legacy.triangulation_3d import camera_from_intrinsics

    # Three cameras with realistic intrinsics for a 2448×2048 sensor.
    K_common = np.array([
        [2500.0, 0.0, 1224.0],
        [0.0, 2500.0, 1024.0],
        [0.0, 0.0, 1.0],
    ])
    # rvec, tvec — three different viewpoints.
    cam_specs = [
        ("front",  np.array([0.0, 0.0, 0.0]),            np.array([0.0, 0.0, 500.0])),
        ("left",   np.array([0.0, np.pi / 2.0, 0.0]),    np.array([-300.0, 0.0, 0.0])),
        ("top",    np.array([np.pi / 2.0, 0.0, 0.0]),    np.array([0.0, 200.0, -300.0])),
    ]
    cameras = [
        camera_from_intrinsics(name, K_common, rvec, tvec)
        for name, rvec, tvec in cam_specs
    ]
    P_list = [c.P for c in cameras]

    # Ground-truth 3D point.
    X_world = np.array([10.0, 20.0, 30.0])
    points_2d = [c.project(X_world) for c in cameras]

    return {
        "cameras": cameras,
        "P_list": P_list,
        "X_world": X_world,
        "points_2d": points_2d,
    }


# =============================================================================
# Camera model + projection
# =============================================================================
class TestCamera:
    def test_project_round_trip(self):
        from deepgait3.core._legacy.triangulation_3d import camera_from_intrinsics

        cam = camera_from_intrinsics(
            name="test",
            K=np.array([[1000.0, 0.0, 500.0],
                         [0.0, 1000.0, 500.0],
                         [0.0, 0.0, 1.0]]),
            rvec=np.zeros(3),
            tvec=np.array([0.0, 0.0, 100.0]),  # 100 mm along +Z
        )
        # World point on +Z axis → projects to principal point.
        X = np.array([0.0, 0.0, 50.0])
        x = cam.project(X)
        np.testing.assert_allclose(x, [500.0, 500.0], atol=1e-6)

    def test_camera_from_intrinsics_validates_shapes(self):
        from deepgait3.core._legacy.triangulation_3d import camera_from_intrinsics

        with pytest.raises(ValueError):
            camera_from_intrinsics("x", np.eye(3), np.zeros(2), np.zeros(3))
        with pytest.raises(ValueError):
            camera_from_intrinsics("x", np.eye(3), np.zeros(3), np.zeros(2))


# =============================================================================
# DLT triangulation — the core algorithm
# =============================================================================
class TestDLTTriangulate:
    def test_recovers_ground_truth_within_3px(self, synthetic_rig):
        """W7 acceptance gate: triangulation recovers GT within 3 px."""
        from deepgait3.core._legacy.triangulation_3d import (
            dlt_triangulate, reprojection_error,
        )

        X = dlt_triangulate(synthetic_rig["P_list"],
                             synthetic_rig["points_2d"])
        err = reprojection_error(synthetic_rig["P_list"],
                                  synthetic_rig["points_2d"], X)
        assert err < 3.0, f"reprojection error {err:.3f} px >= 3 px gate"
        # World-space recovery must also be tight (the cameras were
        # built around the GT point).
        assert np.linalg.norm(X - synthetic_rig["X_world"]) < 1.0

    def test_requires_two_cameras(self):
        from deepgait3.core._legacy.triangulation_3d import dlt_triangulate

        with pytest.raises(ValueError):
            dlt_triangulate([np.eye(3, 4)], [np.zeros(2)])

    def test_rejects_length_mismatch(self, synthetic_rig):
        from deepgait3.core._legacy.triangulation_3d import dlt_triangulate

        with pytest.raises(ValueError):
            dlt_triangulate(synthetic_rig["P_list"],
                             synthetic_rig["points_2d"][:1])


# =============================================================================
# Reprojection error
# =============================================================================
class TestReprojectionError:
    def test_zero_error_at_ground_truth(self, synthetic_rig):
        from deepgait3.core._legacy.triangulation_3d import reprojection_error

        err = reprojection_error(synthetic_rig["P_list"],
                                  synthetic_rig["points_2d"],
                                  synthetic_rig["X_world"])
        assert err < 1e-6, f"expected ~0, got {err}"

    def test_per_camera_errors(self, synthetic_rig):
        from deepgait3.core._legacy.triangulation_3d import (
            reprojection_error, reprojection_error_per_camera,
        )

        per = reprojection_error_per_camera(synthetic_rig["P_list"],
                                            synthetic_rig["points_2d"],
                                            synthetic_rig["X_world"])
        assert per.shape == (3,)
        # All errors ~0 because point IS the GT.
        np.testing.assert_allclose(per, 0.0, atol=1e-6)
        # Mean should match reprojection_error().
        mean = reprojection_error(synthetic_rig["P_list"],
                                   synthetic_rig["points_2d"],
                                   synthetic_rig["X_world"])
        np.testing.assert_allclose(per.mean(), mean, atol=1e-9)


# =============================================================================
# RANSAC triangulation
# =============================================================================
class TestRANSACTriangulation:
    def test_recovers_ground_truth(self, synthetic_rig):
        from deepgait3.core._legacy.triangulation_3d import (
            triangulate_ransac, reprojection_error,
        )

        X = triangulate_ransac(synthetic_rig["P_list"],
                                synthetic_rig["points_2d"],
                                n_iters=50, threshold_px=3.0,
                                rng_seed=0)
        err = reprojection_error(synthetic_rig["P_list"],
                                  synthetic_rig["points_2d"], X)
        assert err < 3.0

    def test_robust_to_outlier_camera(self, synthetic_rig):
        """One camera with massive noise — RANSAC must still find GT."""
        from deepgait3.core._legacy.triangulation_3d import (
            triangulate_ransac, reprojection_error,
        )

        # Corrupt the third camera's observation by 50 px.
        points_2d = list(synthetic_rig["points_2d"])
        points_2d[2] = points_2d[2] + np.array([50.0, -50.0])
        X = triangulate_ransac(synthetic_rig["P_list"], points_2d,
                                n_iters=200, threshold_px=3.0, rng_seed=0)
        err = reprojection_error(synthetic_rig["P_list"], points_2d, X)
        # The mean error includes the outlier camera, so allow more slack.
        assert err < 30.0, f"unexpected RANSAC error {err:.2f}"
        # But the recovered 3D point should still be close to GT.
        assert np.linalg.norm(X - synthetic_rig["X_world"]) < 5.0

    def test_two_cameras_still_works(self, synthetic_rig):
        from deepgait3.core._legacy.triangulation_3d import triangulate_ransac

        X = triangulate_ransac(synthetic_rig["P_list"][:2],
                                synthetic_rig["points_2d"][:2])
        assert X.shape == (3,)
        assert not np.any(np.isnan(X))

    def test_invalid_inputs_raise(self):
        from deepgait3.core._legacy.triangulation_3d import triangulate_ransac

        with pytest.raises(ValueError):
            triangulate_ransac([np.eye(3, 4)], [np.zeros(2)])


# =============================================================================
# optim_points — joint refinement
# =============================================================================
class TestOptimPoints:
    def test_refines_noisy_observations(self):
        from deepgait3.core._legacy.triangulation_3d import (
            dlt_triangulate, optim_points, reprojection_error,
            camera_from_intrinsics,
        )

        # Two-camera setup with one known bone (length ≈ 50 mm).
        K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0],
                       [0.0, 0.0, 1.0]])
        cam_a = camera_from_intrinsics(
            "a", K, np.zeros(3), np.array([0.0, 0.0, 200.0]),
        )
        cam_b = camera_from_intrinsics(
            "b", K, np.array([0.0, np.pi / 2.0, 0.0]),
            np.array([-200.0, 0.0, 0.0]),
        )
        # Two bodyparts with known relative geometry.
        X1 = np.array([0.0, 0.0, 0.0])
        X2 = np.array([40.0, 30.0, 0.0])      # bone length = 50 mm
        noisy1 = cam_a.project(X1) + np.array([2.0, 1.0])
        noisy2 = cam_a.project(X2) + np.array([-1.0, 2.0])
        noisy1b = cam_b.project(X1) + np.array([1.0, -1.0])
        noisy2b = cam_b.project(X2) + np.array([2.0, 1.0])

        # Initial estimate from raw DLT.
        X_init = np.array([
            dlt_triangulate([cam_a.P, cam_b.P], [noisy1, noisy1b]),
            dlt_triangulate([cam_a.P, cam_b.P], [noisy2, noisy2b]),
        ])
        bones = [(0, 1, 50.0)]
        X_refined = optim_points(
            [cam_a.P, cam_b.P],
            [np.array([noisy1, noisy2]), np.array([noisy1b, noisy2b])],
            X_init, bones,
            bone_weight=10.0, reproj_weight=1.0,
        )
        # The refined estimate should have a smaller reprojection error
        # than the initial one.
        err_init = (
            reprojection_error([cam_a.P, cam_b.P], [noisy1, noisy1b], X_init[0]) +
            reprojection_error([cam_a.P, cam_b.P], [noisy2, noisy2b], X_init[1])
        ) / 2
        err_refined = (
            reprojection_error([cam_a.P, cam_b.P], [noisy1, noisy1b], X_refined[0]) +
            reprojection_error([cam_a.P, cam_b.P], [noisy2, noisy2b], X_refined[1])
        ) / 2
        assert err_refined <= err_init + 1e-3, (
            f"optim_points did not improve: init={err_init:.3f} "
            f"refined={err_refined:.3f}"
        )

    def test_bone_length_constraint(self):
        """After optim_points with a strong bone weight, the bone length
        should be very close to the target.

        W16 fix: switched from 1-camera to 2-camera observation to
        align with the new ``_check_two_cameras`` guard (CAS-P0#2).
        With 2 cameras, the under-determined bone length is properly
        constrained by the reprojection residuals.
        """
        from deepgait3.core._legacy.triangulation_3d import (
            optim_points, camera_from_intrinsics,
        )

        K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0],
                       [0.0, 0.0, 1.0]])
        # Two cameras offset along X (W16: 2 cameras needed since CAS-P0#2).
        cam_a = camera_from_intrinsics(
            "a", K, np.zeros(3), np.array([-50.0, 0.0, 200.0]),
        )
        cam_b = camera_from_intrinsics(
            "b", K, np.zeros(3), np.array([ 50.0, 0.0, 200.0]),
        )
        # Two points with intentionally WRONG initial bone length.
        X_init = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # length=10
        bones = [(0, 1, 50.0)]                                    # target=50
        # Observing a known geometry: both pts at z=0.
        obs_a = np.array([cam_a.project(X_init[0]), cam_a.project(X_init[1])])
        obs_b = np.array([cam_b.project(X_init[0]), cam_b.project(X_init[1])])
        X_refined = optim_points(
            [cam_a.P, cam_b.P], [obs_a, obs_b], X_init, bones,
            bone_weight=100.0, reproj_weight=1.0,
        )
        actual_length = np.linalg.norm(X_refined[0] - X_refined[1])
        # Strong bone constraint should pull the length toward 50 mm.
        assert 45.0 < actual_length < 55.0, actual_length


# =============================================================================
# filter_3d — smoothing
# =============================================================================
class TestFilter3D:
    def test_removes_single_outlier(self):
        from deepgait3.core._legacy.triangulation_3d import filter_3d

        # 5 frames of a moving point + one spike in frame 2.
        X = np.zeros((5, 1, 3), dtype=np.float64)
        for t in range(5):
            X[t, 0, 0] = float(t)
        X[2, 0, 0] = 100.0  # outlier
        out = filter_3d(X, window=3)
        # The median filter (window=3) should erase the spike.
        assert out[2, 0, 0] < 5.0, out[2, 0, 0]

    def test_validates_window(self):
        from deepgait3.core._legacy.triangulation_3d import filter_3d

        with pytest.raises(ValueError):
            filter_3d(np.zeros((3, 1, 3)), window=0)
        with pytest.raises(ValueError):
            filter_3d(np.zeros((3, 1, 3)), window=2)   # even

    def test_jump_clipping_interpolates(self):
        from deepgait3.core._legacy.triangulation_3d import filter_3d

        X = np.zeros((4, 1, 3), dtype=np.float64)
        for t in range(4):
            X[t, 0, 0] = float(t)
        X[2, 0, 0] = 1000.0  # big spike
        out = filter_3d(X, window=3, max_jump_mm=50.0)
        # The clipped+interpolated value must be a reasonable neighbour,
        # not 1000.
        assert abs(out[2, 0, 0] - 2.0) < 5.0, out[2, 0, 0]


# =============================================================================
# AniposeWrapper
# =============================================================================
class TestAniposeWrapper:
    def test_inhouse_method_falls_back_when_aniposelib_missing(self):
        from deepgait3.core._legacy.anipose_wrapper import (
            ANIPOSELIB_AVAILABLE, AniposeWrapper,
        )

        # We may or may not have aniposelib installed; this test only
        # exercises the auto-prefer fallback path.
        wrapper = AniposeWrapper(prefer="auto")
        assert wrapper.active_method in ("aniposelib", "inhouse")
        if not ANIPOSELIB_AVAILABLE:
            assert wrapper.active_method == "inhouse"

    def test_invalid_prefer_raises(self):
        from deepgait3.core._legacy.anipose_wrapper import AniposeWrapper

        with pytest.raises(ValueError):
            AniposeWrapper(prefer="nonsense")

    def test_compute_angles_handles_3bp_skeleton(self):
        from deepgait3.core._legacy.anipose_wrapper import AniposeWrapper

        # Build a synthetic (T=10, J=3, 3) trajectory where:
        #   grandparent at origin
        #   parent at (1, 0, 0)
        #   child at (1, 1, 0)         → v1 = (0,1,0), v2 = (1,0,0)
        # The angle between v1 and v2 is 90° for every frame.
        T = 10
        X = np.zeros((T, 3, 3), dtype=np.float64)
        for t in range(T):
            X[t, 0] = [0.0, 0.0, 0.0]   # grandparent
            X[t, 1] = [1.0, 0.0, 0.0]   # parent
            X[t, 2] = [1.0, 1.0, 0.0]   # child  (90° bend)
        skel = [("GP", "P"), ("P", "C")]
        bps = ["GP", "P", "C"]
        angles = AniposeWrapper(prefer="inhouse").compute_angles(
            X, skel, bodyparts=bps,
        )
        assert "C" in angles
        np.testing.assert_allclose(angles["C"], 90.0, atol=1.0)

    def test_calibrate_and_triangulate_round_trip(self, synthetic_rig):
        """End-to-end: feed the rig's cameras to AniposeWrapper and
        triangulate the GT point. Reprojection error must stay below
        the W7 acceptance gate (< 3 px)."""
        from deepgait3.core._legacy.anipose_wrapper import (
            AniposeWrapper, CharucoBoard, DEFAULT_BODYPARTS_12,
        )

        # We don't have actual ChArUco images — feed the rig's
        # projection matrices directly via a synthetic calibration.
        wrapper = AniposeWrapper(prefer="inhouse")
        calib = wrapper.calibrate(
            # Empty image lists → K=identity fallback; we patch in our
            # known cameras below.
            images_per_cam={c.name: [] for c in synthetic_rig["cameras"]},
            board=CharucoBoard(squares_x=7, squares_y=5,
                                square_length=30, marker_length=22),
        )
        # Inject the synthetic projection matrices.
        for c in synthetic_rig["cameras"]:
            calib.cameras[c.name].P = c.P

        # Build pose_2d dict: T=1, J=12, 2 (use only the first 3 J for
        # the test point — pad the rest with zeros).
        pose_2d = {}
        for c in synthetic_rig["cameras"]:
            arr = np.zeros((1, len(DEFAULT_BODYPARTS_12), 2), dtype=np.float64)
            arr[0, 0] = c.project(synthetic_rig["X_world"])
            pose_2d[c.name] = arr

        pose_3d = wrapper.triangulate(pose_2d, calib, use_ransac=True)
        assert pose_3d.shape == (1, len(DEFAULT_BODYPARTS_12), 3)
        recovered = pose_3d[0, 0]
        # The first bodypart should be near the GT.
        assert np.linalg.norm(recovered - synthetic_rig["X_world"]) < 1.0, (
            f"recovered {recovered} vs GT {synthetic_rig['X_world']}"
        )

    def test_filter_3d_passthrough(self):
        from deepgait3.core._legacy.anipose_wrapper import AniposeWrapper

        wrapper = AniposeWrapper(prefer="inhouse")
        X = np.zeros((5, 1, 3), dtype=np.float64)
        X[:, 0, 0] = [0, 1, 100, 3, 4]   # spike
        out = wrapper.filter_3d(X, window=3)
        assert out[2, 0, 0] < 5.0


# =============================================================================
# Performance gate — 1000-frame triangulation under 5 seconds
# =============================================================================
@pytest.mark.performance
class TestW7Performance:
    def test_triangulate_1000_frames_under_30s(self, synthetic_rig):
        """W7 perf gate: 1000 frames × 12 bodyparts × 3 cameras under 30 s.

        NOTE: the original DEVELOPMENT_PLAN §5.2 target was 1000 frames
        < 5 ms, which is physically impossible for a per-bodypart Python
        loop (audit C4). The realistic Python baseline is ~6-12 s; the
        bound is set loosely (30 s) to absorb CI variability. Phase 4
        W13 plans a Cython rewrite that brings this down to < 100 ms.
        """
        from deepgait3.core._legacy.anipose_wrapper import (
            AniposeWrapper, CharucoBoard, DEFAULT_BODYPARTS_12,
        )

        wrapper = AniposeWrapper(prefer="inhouse")
        calib = wrapper.calibrate(
            {c.name: [] for c in synthetic_rig["cameras"]},
            CharucoBoard(squares_x=7, squares_y=5,
                          square_length=30, marker_length=22),
        )
        for c in synthetic_rig["cameras"]:
            calib.cameras[c.name].P = c.P

        rng = np.random.default_rng(0)
        T = 1000
        J = len(DEFAULT_BODYPARTS_12)
        pose_2d = {}
        for c in synthetic_rig["cameras"]:
            arr = np.zeros((T, J, 2), dtype=np.float64)
            for t in range(T):
                for j in range(J):
                    X = np.array([rng.normal(0, 30),
                                   rng.normal(0, 30),
                                   rng.normal(0, 30)])
                    arr[t, j] = c.project(X)
            pose_2d[c.name] = arr

        # Warm-up (numba JIT, numpy caches, etc.).
        wrapper.triangulate({k: v[:10] for k, v in pose_2d.items()}, calib)
        t0 = time.perf_counter()
        wrapper.triangulate(pose_2d, calib, use_ransac=False)
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"triangulate took {elapsed:.2f}s > 30s"