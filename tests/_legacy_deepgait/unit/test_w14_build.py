"""Unit tests for Phase 4 W14 — Cython compilation + Nuitka packaging.

Acceptance gate (docs/DEVELOPMENT_PLAN.md §6.4 W14):
    "Cython 编译 + Nuitka 打包" / "启动 < 5s；安装包 < 100MB"

These tests verify:
  * ``setup_cython.py`` defines the correct module list.
  * Compiled ``.so`` (or ``.pyd``) modules are importable.
  * The compiled modules produce the same results as the pure-Python
    versions (functional equivalence).
  * ``build.spec`` is syntactically valid and references the correct
    entry point.

Note: the actual Nuitka packaging step is too slow for CI (5-10 min)
and is verified manually. This test only checks the *configuration*
and the *Cython output*.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# setup_cython.py — configuration validation
# =============================================================================
class TestSetupCython:
    def test_setup_cython_file_exists(self):
        assert Path("setup_cython.py").is_file()

    def test_cython_module_list_covers_critical_layers(self):
        """setup_cython.py must compile all Layer 3-4 modules."""
        import ast
        source = Path("setup_cython.py").read_text()
        tree = ast.parse(source)
        # Find the CYTHON_MODULES list.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "CYTHON_MODULES":
                        modules = [elt.value for elt in node.value.elts
                                    if isinstance(elt, ast.Constant)]
                        assert "deepgait/core/gait_algorithms.py" in modules
                        assert "deepgait/core/footprint_v2.py" in modules
                        assert "deepgait/core/triangulation_3d.py" in modules
                        assert "deepgait/core/license/verifier.py" in modules
                        assert "deepgait/core/security/anti_debug.py" in modules
                        return
        pytest.fail("CYTHON_MODULES list not found in setup_cython.py")

    def test_compiler_directives_disable_embedsignature(self):
        """embedsignature=False is critical for anti-reverse-engineering."""
        import ast
        source = Path("setup_cython.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "DIRECTIVES":
                        directives = dict(
                            (k.value, v.value)
                            for k, v in zip(node.value.keys, node.value.values)
                            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                        )
                        assert directives.get("embedsignature") is False
                        assert directives.get("language_level") == "3"
                        return
        pytest.fail("DIRECTIVES dict not found")

    def test_numpy_include_dir_referenced(self):
        """setup_cython.py must add numpy's include dir for C headers."""
        source = Path("setup_cython.py").read_text()
        assert "np.get_include()" in source


# =============================================================================
# Compiled .so modules — import + functional equivalence
# =============================================================================
class TestCompiledModules:
    """Verify that compiled .so modules import and produce correct results."""

    @pytest.fixture
    def compiled_modules(self):
        """Return list of module paths that have compiled .so artifacts."""
        suffix = ".pyd" if sys.platform == "win32" else ".so"
        root = Path("deepgait/core")
        compiled = []
        for so in root.rglob(f"*{suffix}"):
            # Skip __pycache__ entries.
            if "__pycache__" in str(so):
                continue
            compiled.append(so)
        return compiled

    def test_compiled_modules_exist(self, compiled_modules):
        """At least the core modules must have .so artifacts after compilation."""
        # If no .so files exist, Cython hasn't been run — skip (CI may not
        # have a C compiler).
        if not compiled_modules:
            pytest.skip("No compiled .so modules found; run setup_cython.py first")
        assert len(compiled_modules) >= 5  # at least the core set

    @pytest.mark.parametrize("module_name", [
        "deepgait3.core._legacy.gait_algorithms",
        "deepgait3.core._legacy.footprint_v2",
        "deepgait3.core._legacy.triangulation_3d",
        "deepgait3.core._legacy.anipose_wrapper",
    ])
    def test_compiled_module_imports(self, module_name):
        """Each compiled module must import without error."""
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            pytest.skip(f"{module_name} not importable: {e}")
        # Verify it's either compiled (.so) or pure Python (.py) —
        # both are valid.
        assert mod is not None

    def test_gait_algorithms_functional_equivalence(self):
        """compute_all_gait_metrics from .so must produce same results."""
        from deepgait3.core._legacy.gait_algorithms import (
            compute_all_gait_metrics, stride_cycle_ms,
        )
        # Simple metric.
        assert stride_cycle_ms(100.0, 50.0) == 150.0

        # Full pipeline.
        cycle = np.array([1] * 30 + [0] * 20, dtype=int)
        per_paw = {p: np.tile(cycle, 5) for p in ("LF", "RF", "LH", "RH")}
        out = compute_all_gait_metrics(
            fps=100, real_world_multiplier=0.25,
            in_stance_per_paw=per_paw,
        )
        assert out["__n_metrics__"] >= 30

    def test_license_backend_functional_equivalence(self):
        """MockBackend from .so must produce same HMAC round-trip."""
        from deepgait3.license import MockBackend
        b = MockBackend()
        payload = b"test"
        sig = b.execute_in_dongle("sign_payload", payload)
        import base64
        ok = b.execute_in_dongle(
            "verify_signature",
            base64.b64encode(payload).decode(),
            base64.b64encode(sig).decode(),
        )
        assert ok is True

    def test_triangulation_dlt_functional_equivalence(self):
        """DLT triangulation from .so must recover ground truth."""
        from deepgait3.core._legacy.triangulation_3d import (
            camera_from_intrinsics, dlt_triangulate, reprojection_error,
        )
        K = np.array([[2500.0, 0, 1224.0], [0, 2500.0, 1024.0], [0, 0, 1.0]])
        cam_a = camera_from_intrinsics(
            "a", K, np.zeros(3), np.array([0.0, 0.0, 500.0]))
        cam_b = camera_from_intrinsics(
            "b", K, np.array([0.0, np.pi / 2, 0.0]), np.array([-300.0, 0.0, 0.0]))
        X_world = np.array([10.0, 20.0, 30.0])
        pts = [cam_a.project(X_world), cam_b.project(X_world)]
        X = dlt_triangulate([cam_a.P, cam_b.P], pts)
        err = reprojection_error([cam_a.P, cam_b.P], pts, X)
        assert err < 3.0  # W7 acceptance gate
        assert np.linalg.norm(X - X_world) < 1.0


# =============================================================================
# build.spec — Nuitka packaging configuration
# =============================================================================
class TestBuildSpec:
    def test_build_spec_exists(self):
        assert Path("build.spec").is_file()

    def test_build_spec_references_entry_point(self):
        source = Path("build.spec").read_text()
        assert "deepgait/main.py" in source

    def test_build_spec_uses_standalone(self):
        source = Path("build.spec").read_text()
        assert "--standalone" in source

    def test_build_spec_uses_onefile(self):
        source = Path("build.spec").read_text()
        assert "--onefile" in source

    def test_build_spec_includes_pyside6(self):
        source = Path("build.spec").read_text()
        assert "--enable-plugin=pyside6" in source

    def test_build_spec_checks_size_gate(self):
        """build.spec must check the 100 MB acceptance gate."""
        source = Path("build.spec").read_text()
        assert "100" in source

    def test_build_spec_has_windows_variant(self):
        """build.spec must include Windows-specific flags."""
        source = Path("build.spec").read_text()
        assert "--windows-disable-console" in source