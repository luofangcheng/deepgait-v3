# DeepGait v3

Next-generation gait analysis platform combining:

- **GUI / framework** from [`deep-gailt/`](../deep-gailt/) (v2.0 commercial, 5-layer L1–L5 architecture)
- **Algorithms** from [`deepgait-v2/`](../deepgait-v2/) (v4.x 4-stage pipeline: dynamics → pose3d → metrics → report)

## Status

🚧 **In development (2026-06-25)**. Module-by-module integration. See [`docs/DESIGN.md`](docs/DESIGN.md) for the integration plan and [`docs/reports/`](docs/reports/) for progress notes.

## Layout

```
deepgait-v3/
├── deepgait3/           # Main package (mirror of v1 L1–L5 layers)
│   ├── core/
│   │   ├── pawprint/    # Stage 1: fTIR footprint extraction (from dynamics_v0.4.2)
│   │   ├── calibration/ # Stage 2: ChArUco multi-camera (from pose3d)
│   │   ├── triangulation/ # Stage 2: DLT triangulation (from pose3d)
│   │   ├── fusion/      # Stage 2: ankle-to-pawprint matching (from pose3d)
│   │   ├── metrics/     # Stage 3: 32 CatWalk params (from gait_metrics)
│   │   └── report/      # Stage 4: Excel/CSV/HTML/PNG (from gait_report)
│   ├── gui/             # L5 — reuses v1 PySide6 framework with adapter layer
│   ├── io/              # L2 — HDF5 / NWB / BIDS (v1, extended for PawPrint)
│   ├── hardware/        # L1 — Hikvision MVS / Basler pylon (v1)
│   ├── license/         # L3 — dongle, heartbeat (v1)
│   └── security/        # L3 — anti-debug, integrity (v1)
├── docs/                # Project documentation (DESIGN, implementation notes)
├── tests/               # unit / integration / performance / fixtures
├── examples/            # End-to-end demos
├── configs/             # YAML config templates
└── scripts/             # Maintenance scripts (graphify ingest, etc.)
```

## Quick start (after first module lands)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest                              # run all tests
ruff check deepgait3                # lint
python -m deepgait3 info <dlc_csv>  # CLI entry point
```

## Documentation

| Doc | Purpose |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture, integration rules, migration plan |
| [`docs/MIGRATION.md`](docs/MIGRATION.md) | Per-module migration checklist (v2 algo → v3 core/) |
| [`docs/reports/`](docs/reports/) | Status updates per integration milestone |
| [`docs/best-practice/`](docs/best-practice/) | Coding conventions, patterns |
| [`docs/implementation/`](docs/implementation/) | Implementation details per stage |
| [`docs/tips/`](docs/tips/) | Tips & gotchas |

## Key design constraints

1. **PawPrint schema is the cross-module contract** (36 fields, see `docs/MIGRATION.md`).
2. **Label-agnostic persistence**: Stage 1 outputs `PawPrint` with `paw_id=None`; Stage 2 fills `paw_id`.
3. **Fail-graceful**: missing optional fields fall back to defaults; never crash a stage.
4. **Cython compilation deferred**: v3 keeps core modules as pure Python initially. Promote hot paths to Cython only after profiling.