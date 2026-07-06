"""Unit tests (deep, fast, no external resources).

Layout per docs/DEVELOPMENT_PLAN.md §4.1:
    tests/unit/         — pure-function unit tests, < 1s each
    tests/integration/  — multi-module end-to-end tests
    tests/performance/  — benchmark/regression tests (marked `slow`)
    tests/data/         — shared binary fixtures

Existing baseline tests remain under deepgait/tests/ for backward compat.
"""
