"""Skeleton smoke test for the new tests/ directory layout.

Verifies the package metadata declared in pyproject.toml is importable and
matches the v2.0 contract. This guards against accidental metadata drift
(e.g. version, Python constraint) breaking CI or packaging.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict:
    if sys.version_info >= (3, 11):
        import tomllib

        with open(ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)
    import tomli  # type: ignore[import-not-found]

    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomli.load(f)


def test_package_importable() -> None:
    """The `deepgait` package must import without error."""
    importlib.import_module("deepgait")


def test_pyproject_version_is_v2() -> None:
    """pyproject declares deepgait v2.0.0 (DEVELOPMENT_PLAN header)."""
    cfg = _load_pyproject()
    assert cfg["project"]["version"] == "2.0.0"


def test_pyproject_python_constraint() -> None:
    """Python constraint is 3.10–3.12 (DEVELOPMENT_PLAN §1.1, §5.1)."""
    cfg = _load_pyproject()
    requires_python = cfg["project"]["requires-python"]
    assert ">=3.10" in requires_python
    assert "<3.13" in requires_python


def test_pyproject_pyside6_dependency() -> None:
    """PySide6 (LGPL) is the mandated GUI framework, not PyQt6."""
    cfg = _load_pyproject()
    deps = cfg["project"]["dependencies"]
    assert any(d.lower().startswith("pyside6") for d in deps), deps
    assert not any("pyqt6" in d.lower() for d in deps), "PyQt6 (GPL) must not be used"


def test_pyproject_entry_point_exists() -> None:
    """The `deepgait` console script maps to the CLI main()."""
    cfg = _load_pyproject()
    scripts = cfg["project"].get("scripts", {})
    assert scripts.get("deepgait") == "deepgait.__main__:main"


def test_test_directories_exist() -> None:
    """Phase 1 W1 test layout directories exist (DEVELOPMENT_PLAN §4.1)."""
    for sub in ("unit", "integration", "performance", "data"):
        assert (ROOT / "tests" / sub).is_dir(), f"tests/{sub} missing"


def test_docs_exist() -> None:
    """The three core development docs must be present."""
    docs = ROOT / "docs"
    for name in ("REQUIREMENTS.md", "DEVELOPMENT_PLAN.md", "MODULES.md"):
        assert (docs / name).is_file(), f"docs/{name} missing"


@pytest.mark.integration
def test_smoke_pipeline_runs_on_synthetic_csv(tmp_path: Path) -> None:
    """A minimal end-to-end: synthesize a tiny DLC CSV and run analyze().

    This is a thin integration smoke test confirming core/pipeline still
    works from the new tests/ layout.
    """
    import numpy as np
    import pandas as pd

    from deepgait3.core._legacy import bodyparts
    from deepgait3.core._legacy.pipeline import analyze

    # Build a minimal 2-frame, 12-bodypart DLC-format CSV.
    n = 2
    scorer = "scorer"
    cols = pd.MultiIndex.from_product(
        [[scorer], bodyparts.BODYPARTS_12, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    rng = np.random.default_rng(0)
    data = rng.random((n, len(bodyparts.BODYPARTS_12) * 3))
    df = pd.DataFrame(data, columns=cols)
    csv_path = tmp_path / "mini.csv"
    df.to_csv(csv_path, index=False)

    # analyze() takes a CSV path, not a DataFrame.
    res = analyze(csv_path, fps=100, mode="free")
    assert res is not None
