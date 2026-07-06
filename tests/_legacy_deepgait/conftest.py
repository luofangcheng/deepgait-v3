"""Local pytest hooks for the legacy deep-gailt test suite.

The root :file:`pyproject.toml` declares the canonical markers
(``gpu``, ``hardware``, ``slow``, ``integration``).  A handful of
extra markers appear in the legacy files that we transplant from
``deep-gailt/tests/``:

* ``performance`` — wall-clock vs throughput micro-benchmarks

Register them here so ``--strict-markers`` keeps working without
touching the root config.
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "performance: legacy wall-clock benchmarks")
