# DeepGait v3 Architecture

## Layered Design

```
┌─────────────────────────────────────────────┐
│                 deepgait3/gui/               │  ← Application Layer
│        PySide6, zero algorithm logic         │
├─────────────────────────────────────────────┤
│                 deepgait3/core/              │  ← Algorithm Layer
│   pawprint / calibration / triangulation     │
│   fusion / metrics / report                  │
│   Pure Python, zero Qt dependency            │
├─────────────────────────────────────────────┤
│   deepgait3/hardware/  │  deepgait3/io/      │  ← I/O & Hardware
│   Camera drivers, DLC  │  HDF5, NWB, BIDS   │
└─────────────────────────────────────────────┘
```

## Package Layout

```
deepgait3/
├── gui/                     # Application layer (PySide6)
│   ├── main_window.py       # MainWindow with tab manager
│   ├── shared_state.py      # AppState — single source of truth
│   ├── workers.py           # QThread workers for long-running tasks
│   ├── style.py             # Qt stylesheet
│   └── *_tab.py             # 10+ tab modules
│
├── core/                    # Algorithm layer
│   ├── pawprint/            # Stage 1: fTIR footprint extraction ✓
│   │   ├── models.py        # PawPrint, FootprintCycle, TrialResult
│   │   ├── detection.py     # Blob detection primitives
│   │   ├── grouping.py      # Spatial clustering of blobs
│   │   ├── tracker.py       # IoU-based frame-to-frame tracking
│   │   ├── cycle_builder.py # Track → FootprintCycle assembly
│   │   ├── pipeline.py      # Stage1Pipeline orchestrator
│   │   └── single_frame.py  # GUI adapter (FootprintSequence bridge)
│   ├── _legacy/             # Unmigrated v2.0 algorithms
│   │   ├── footprint_v2.py  # Old FootprintSequence (GUI data contract)
│   │   ├── gait_algorithms.py  # CatWalk metric computation
│   │   └── ...              # 20+ more modules
│   ├── calibration/         # Stage 2: ChArUco calibration (planned)
│   ├── triangulation/       # Stage 2: DLT triangulation (planned)
│   ├── fusion/              # Stage 2: ankle-to-paw matching (planned)
│   ├── metrics/             # Stage 3: 32 gait parameters (planned)
│   └── report/              # Stage 4: Excel/CSV/HTML export (planned)
│
├── hardware/                # Hardware abstraction
│   ├── camera/              # Hikvision, Basler, multi-camera
│   └── dlc/                 # DLC subprocess runner
│
├── io/                      # Data I/O
│   ├── h5_reader.py         # HDF5 SWMR reader
│   ├── h5_writer.py         # HDF5 SWMR writer
│   ├── bids_exporter.py     # BIDS format export
│   └── nwb_exporter.py      # Neurodata Without Borders export
│
├── license/                 # License enforcement (Cython)
├── security/                # Anti-tamper protections (Cython)
└── utils/                   # Shared utilities (geometry)
```

## Data Flow (4-Stage Pipeline)

```
Stage 1: pawprint/        Stage 2: calibration/      Stage 3: metrics/       Stage 4: report/
                         + triangulation/
                         + fusion/
┌──────────────┐         ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│ fTIR video   │────────>│ Camera calib │────────>│ Per-paw      │────────>│ Excel export │
│   ↓          │         │   ↓          │         │ Per-pair     │         │ CSV export   │
│ Blob detect  │         │ DLT 3D recon │         │ Coordination │         │ HTML report  │
│   ↓          │         │   ↓          │         │ Trial stats  │         │ PNG plots    │
│ Clustering   │         │ Ankle→Paw    │         └──────────────┘         └──────────────┘
│   ↓          │         └──────────────┘
│ IoU tracking │
│   ↓          │
│ PawPrint[]   │
└──────────────┘
```

## Key Design Principles

1. **Algorithm-GUI Separation**: Core algorithms are pure Python with zero Qt dependency. GUI calls algorithms through workers (QThread), never embedding algorithm logic directly.
2. **Label-Agnostic Persistence**: Stage 1 outputs PawPrints without limb identity (`paw_id`). Identity is assigned in Stage 2 to avoid early classification errors contaminating downstream analysis.
3. **Stable Data Contract**: The PawPrint schema (36 fields) is the fixed boundary between stages. Fields may be added but never removed.
4. **Fail-Graceful**: Missing optional fields fall back to default values rather than raising exceptions.

## Project Data Layout

```
projects/<name>/
├── project.json    # Project metadata (experimenter, px_per_mm, etc.)
├── rawdata/        # Raw input: videos, DLC CSV files
└── data/           # Processed output: HDF5, reports, visualizations
```

## GUI Architecture

- **8+ Tabs**: Gait, fTIR, DLC, Calibration 3D, Triangulation 3D, Editor, Charts, Camera
- **Shared State**: `AppState` dataclass as single source of truth, connected via Qt signals
- **Workers**: Long-running tasks run in `QThread` workers, communicating via signal/slot
- **Bilingual**: Chinese/English via Qt `tr()` and JSON locale files

## Migration Status

| Stage | Module | Status |
|-------|--------|--------|
| 1 | `core/pawprint/` | **Done** — 17 source files |
| 2 | `core/calibration/` | Planned — from `pose3d_v0.2.0` |
| 2 | `core/triangulation/` | Planned |
| 2 | `core/fusion/` | Planned |
| 3 | `core/metrics/` | Planned — from `gait_metrics_module` |
| 4 | `core/report/` | Planned — from `gait_report_module` |

Until migration is complete, the GUI continues to use `core/_legacy/` modules.
