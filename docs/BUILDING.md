# Building DeepGait v3

## Prerequisites

- Python 3.10–3.13
- Qt 6 libraries (for GUI)
- Optional: Basler pypylon, Hikvision MVS SDK, CUDA (for DeepLabCut)

## Quick Start

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit/

# Launch GUI
python -m deepgait3 gui

# Run Stage 1 (footprint extraction) from CLI
python -m deepgait3 stage1 /path/to/mouse_dir
```

## Development Setup

```bash
# Full dev install with all optional dependencies
pip install -e ".[dev,basler,anipose,nwb,trigger,onnx]"

# Lint and format
ruff check deepgait3/
ruff format --check deepgait3/

# Type check (optional)
mypy deepgait3/
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit/ -v

# Legacy tests
pytest tests/_legacy_deepgait/ -v

# Exclude slow tests
pytest -m "not slow"

# Parallel execution
pytest -n auto
```

## Build

```bash
# Compile core modules to .so (Cython)
python setup_cython.py build_ext --inplace

# Build standalone executable (Nuitka, requires Cython build first)
./build.spec
```

## Project Structure

```
deepgait-v3/
├── deepgait3/          # Main Python package
│   ├── gui/            # Application layer (PySide6)
│   ├── core/           # Algorithm layer (zero Qt dependency)
│   │   ├── pawprint/   # Stage 1: footprint extraction
│   │   └── _legacy/    # Unmigrated v2.0 algorithms
│   ├── hardware/       # Camera drivers, DLC subprocess
│   ├── io/             # HDF5, NWB, BIDS I/O
│   └── utils/          # Shared utilities
├── projects/           # Per-project data
├── tests/              # Test suite
└── docs/               # Documentation
```
