# DeepGait v3 — Design Specification

**Status**: Draft (2026-06-25)
**Owner**: DeepGait team
**Supersedes**: none
**Related**: [`MIGRATION.md`](MIGRATION.md), [`../deepgait-v2/report.md`](../../deepgait-v2/report.md), [`../deep-gailt/CLAUDE.md`](../../deep-gailt/CLAUDE.md)

## Table of Contents

1. [Goals & non-goals](#1-goals--non-goals)
2. [Architecture overview](#2-architecture-overview)
3. [The PawPrint contract](#3-the-pawprint-contract)
4. [Module mapping (v2 algo → v3 core/)](#4-module-mapping-v2-algo--v3-core)
5. [Integration rules (the "do not violate" list)](#5-integration-rules-the-do-not-violate-list)
6. [Migration sequence (work plan)](#6-migration-sequence-work-plan)
7. [Risk register](#7-risk-register)
8. [Out of scope (v3.0)](#8-out-of-scope-v30)

## 1. Goals & non-goals

### 1.1 Goals

- Replace v1 (deep-gailt) L4 algorithm layer with v2 (deepgait-v2) v4.x algorithms.
- Keep v1 L1 (hardware) / L2 (I/O) / L3 (license+security) / L5 (GUI) intact.
- Produce a single PyPI-installable package `deepgait3` with one CLI, one GUI.
- CatWalk XT compatibility for the 32 gait parameters (per `gait_metrics v0.1.0`).
- Backward-compatible v1 GUI workflows via an **adapter layer** — no v1 user should need to relearn the UI.

### 1.2 Non-goals (v3.0)

- Multi-animal tracking. v3.0 assumes one animal per trial (same as v2).
- Toe segmentation (`toe_positions_xy_mm` remains `None` until v0.5 of dynamics).
- SFI (Sciatic Functional Index) computation.
- Group statistics (control vs treated cross-trial analysis) — deferred to v3.1.
- Real-time analysis (≥ 60 fps live). v3.0 is post-hoc only.

## 2. Architecture overview

### 2.1 Layer reuse from v1

| Layer | Source | Reuse policy |
|---|---|---|
| L5 GUI (PySide6 8-tab) | v1 `deepgait/gui/` | **Reuse 100%**, add adapter imports |
| L3 license | v1 `deepgait/core/license/` | Reuse 100% |
| L3 security | v1 `deepgait/core/security/` | Reuse 100% |
| L2 I/O (HDF5/NWB/BIDS) | v1 `deepgait/io/` | Extend schema for new PawPrint (36 fields) |
| L1 hardware (Hikvision/Basler) | v1 `deepgait/hardware/` | Reuse 100% |

### 2.2 Layer replacement (L4 algorithm)

| Stage | v1 location (to remove) | v3 location | Source from v2 |
|---|---|---|---|
| Stage 1: fTIR → PawPrint | `deepgait/core/footprint.py` | `deepgait3/core/pawprint/` | `dynamics_v0.4.2/dynamics_v04_module/dynamics_v04/` |
| Stage 2a: ChArUco calibration | `deepgait/core/triangulation_3d.py` | `deepgait3/core/calibration/` | `pose3d_v0.2.0/pose3d_module/pose3d/calibration.py` |
| Stage 2b: DLT triangulation | `deepgait/core/triangulation_3d.py` | `deepgait3/core/triangulation/` | `pose3d_v0.2.0/pose3d_module/pose3d/triangulator.py` |
| Stage 2c: ankle-to-pawprint | new in v3 | `deepgait3/core/fusion/` | `pose3d_v0.2.0/pose3d_module/pose3d/fusion.py` + `apply_to_pawprints.py` |
| Stage 3: CatWalk metrics | `deepgait/core/gait_algorithms.py` + `gait_ftir.py` | `deepgait3/core/metrics/` | `gait_metrics_module/gait_metrics/` |
| Stage 4: report export | `deepgait/core/gait_export.py` | `deepgait3/core/report/` | `gait_report v0.1.0/gait_report_module/gait_report/` |

### 2.3 Data flow (v3.0 trial)

```
┌──────────────┐    ┌────────────────────┐    ┌───────────────┐
│ L1 hardware  │───▶│ L2 io/ (HDF5 SWMR) │───▶│ L4 core/      │
│ Hikvision×4  │    │                    │    │               │
│ + DLC subprocess    │                    │    │ 1. pawprint/  │
│ + RP2040 trigger   │                    │    │ 2. calib+tri  │
└──────────────┘    └────────────────────┘    │    +fusion    │
                                              │ 3. metrics/   │
                                              │ 4. report/    │
                                              └───────┬───────┘
                                                      │
                                                      ▼
                                              ┌───────────────┐
                                              │ L5 gui (reuse)│
                                              │  + L3 license │
                                              └───────────────┘
```

The same QThread worker pattern from v1 (`deepgait/gui/workers.py`) drives each stage. The worker queues into the shared `AppState`.

## 3. The PawPrint contract

### 3.1 Why this matters

`PawPrint` is the **single data object** that crosses every stage boundary. If its schema drifts, integration silently breaks (e.g., pose3d finds `peak_frame_centroid_xy_mm = None` and stages 3-4 compute 0.0s).

### 3.2 Schema (36 fields, frozen in v3.0)

See [`MIGRATION.md` § PawPrint schema](MIGRATION.md#pawprint-schema-36-fields) for the full field table. Source of truth: `deepgait-v2/dynamics_v0.4.2/dynamics_v04_module/dynamics_v04/models.py`.

### 3.3 Stability rules

1. **No field can be removed** without a major version bump. v3.0 = 36 fields, locked.
2. **New fields require**: (a) default value, (b) fail-graceful behavior in downstream stages.
3. **`paw_id` is `None` from Stage 1**, populated by Stage 2. Do not pre-fill in Stage 1.

## 4. Module mapping (v2 algo → v3 core/)

### 4.1 Stage 1: pawprint/ ↔ dynamics_v0.4.2

| v2 source file | v3 target | Notes |
|---|---|---|
| `models.py` (FrameData, FootMask, PawPrint, QualityFlags, Linkage3D) | `core/pawprint/models.py` | Copy verbatim. Update import paths. |
| `detection.py` | `core/pawprint/detection.py` | Copy verbatim. |
| `grouping.py` | `core/pawprint/grouping.py` | Copy verbatim. |
| `tracker.py` | `core/pawprint/tracker.py` | Copy verbatim. |
| `extractor.py` (PawPrintExtractor) | `core/pawprint/extractor.py` | Copy verbatim. |
| `serializer.py` | `core/pawprint/serializer.py` | Move to `core/io/` instead — serializer is I/O concern. |

### 4.2 Stage 2: calibration/ + triangulation/ + fusion/ ↔ pose3d_v0.2.0

| v2 source file | v3 target | Notes |
|---|---|---|
| `keypoint_schema.py` | `core/calibration/keypoint_schema.py` | Copy verbatim. |
| `calibration.py` | `core/calibration/calibration.py` | Copy verbatim. |
| `models.py` (CameraIntrinsic/Extrinsic, CalibrationPack, Keypoint2D/3D, Skeleton3D, Linkage3DResult) | `core/calibration/models.py` + `core/triangulation/models.py` | Split by responsibility. |
| `dlc_loader.py` | `core/triangulation/dlc_loader.py` | Copy verbatim. |
| `triangulator.py` | `core/triangulation/triangulator.py` | Copy verbatim. |
| `fusion.py` | `core/fusion/fusion.py` | Copy verbatim. |
| `apply_to_pawprints.py` | `core/fusion/apply_to_pawprints.py` | Copy verbatim. |
| `mock_data.py`, `mock_dlc.py` | `tests/fixtures/` | Move to test fixtures, not shipped. |

### 4.3 Stage 3: metrics/ ↔ gait_metrics v0.1.0

| v2 source file | v3 target | Notes |
|---|---|---|
| `models.py` (PerPawMetrics, PerPairMetrics, TrialMetrics, GaitMetrics) | `core/metrics/models.py` | Copy verbatim. |
| `per_paw.py` | `core/metrics/per_paw.py` | Copy verbatim. |
| `inter_paw.py` | `core/metrics/inter_paw.py` | Copy verbatim. |
| `coordination.py` | `core/metrics/coordination.py` | Copy verbatim. |
| `compute.py` | `core/metrics/compute.py` | Copy verbatim. |

### 4.4 Stage 4: report/ ↔ gait_report v0.1.0

| v2 source file | v3 target | Notes |
|---|---|---|
| `csv_export.py` | `core/report/csv_export.py` | Copy verbatim. |
| `excel_export.py` | `core/report/excel_export.py` | Copy verbatim. |
| `plots.py` | `core/report/plots.py` | Copy verbatim. |
| `html_report.py` | `core/report/html_report.py` | Copy verbatim. |
| `pipeline.py` (run_full_pipeline) | `core/report/pipeline.py` | Copy verbatim. |
| `_gait_metrics_lite.py` | **DELETE** | Duplicate of `gait_metrics/`; not needed once Stage 3 is a hard dep. |

### 4.5 Adapter layer (v1 ↔ v3 bridge)

`deepgait3/core/adapters/` exposes **v1-shaped functions** so v1 GUI code keeps working without modification.

```python
# deepgait3/core/adapters/__init__.py
from deepgait3.core.pawprint import PawPrintExtractor
from deepgait3.core.metrics import compute_gait_metrics

def extract_footprints(video_path, **kwargs):  # v1 signature
    """Adapter: v1 extract_footprints → v3 PawPrintExtractor."""
    ext = PawPrintExtractor(**kwargs)
    return ext(video_path)

def compute_all_gait_metrics(in_stance, fps, ...):  # v1 signature
    """Adapter: v1 in_stance ndarray → v3 pawprint list → metrics."""
    # Conversion logic: in_stance → pseudo pawprints → metrics
    ...
```

This is the only place v1's frame-based logic (`in_stance: ndarray`) is converted into v3's print-based logic. Everything else uses native v3 types.

## 5. Integration rules (the "do not violate" list)

| # | Rule | Why |
|---|---|---|
| 1 | Never modify a Stage N's interface to accommodate Stage N+1. | Coupling backwards. |
| 2 | Always pass the full `PawPrint` object between stages; never extract a subset. | Field drift prevention. |
| 3 | `linkage_to_3d` is filled in-place by Stage 2; do not copy to a separate structure. | Single source of truth. |
| 4 | Cross-stage units: distance in **mm**, time in **s**, fps as `int`. | Cross-stage consistency. |
| 5 | Default `px_per_mm=1.92` and `fps=60` MUST be honored across all stages. | Matches v2 hardware contract. |
| 6 | All public functions return **typed containers** (dataclass or `List[Dataclass]`), never bare tuples. | GUI serialization needs named access. |
| 7 | Logging: `loguru.logger` everywhere (v1 convention), not `print()` or stdlib `logging`. | Single log channel. |
| 8 | No `print()` in production code paths. | The 8 GUI tabs and CLI must share one log channel. |
| 9 | Tests live in `tests/unit/`, `tests/integration/`, `tests/performance/`. | Matches v1 layout. |
| 10 | Cython compilation **deferred** to v3.1+ — keep all v3.0 modules pure Python. | Easier debugging during integration. |

## 6. Migration sequence (work plan)

The user has chosen **copy-in** mode and **design-first** workflow. Recommended order:

| # | Stage | Estimated LOC | Why this order |
|---|---|---|---|
| 1 | Stage 1: pawprint/ | ~800 | Most isolated, no upstream deps, fails loudly if wrong |
| 2 | Stage 3: metrics/ | ~400 | Pure function on PawPrint[], easy to unit test |
| 3 | Stage 4: report/ | ~600 | Depends on Stage 3 only |
| 4 | Stage 2a: calibration/ | ~200 | ChArUco is independent of stages 1/3/4 |
| 5 | Stage 2b: triangulation/ | ~300 | Depends on calibration only |
| 6 | Stage 2c: fusion/ | ~200 | Depends on triangulation + pawprint |
| 7 | Adapter layer | ~300 | Bridges v1 GUI |
| 8 | GUI integration | varies | Wire new core/ into v1 GUI tabs |
| 9 | I/O schema extension | ~150 | Add 36-field PawPrint to HDF5 writer |
| 10 | End-to-end smoke test | ~200 | One real trial, all stages |

Stages 1, 3, 4 form a **vertical slice** that can be tested without Stage 2 — Stage 3 accepts synthetic pawprints. Stage 2 is built later as a horizontal addition.

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | v1 GUI imports Cython-compiled v1 functions that no longer exist | High | Build broken | Adapter layer + keep old API in v1 during transition |
| R2 | PawPrint 36-field schema drifts during copy-in (typo in field name) | Medium | Silent zero metrics | Schema unit test that constructs every field |
| R3 | `px_per_mm` mismatch between Stage 1 default (1.92) and v1 GUI override | Medium | Wrong stride length in mm | Centralize in `config.yaml`, validate at worker start |
| R4 | dynamics v0.4.2 has **zero unit tests** upstream | High | Refactor regression | Add minimum 8 tests in v3 (see `deepgait-v2/report.md` §12) |
| R5 | pose3d calibration uses 1-frame `solvePnP` (placeholder per `report.md` §11.2) | Medium | Camera extrinsics drift over time | Document in v3 as known limitation; v3.1 swap for bundle adjustment |
| R6 | Cython build pipeline conflicts between v1 and v3 (if v1 still installed) | Low | Dev confusion | Separate venvs; v3 ships its own `setup_cython.py` only when needed |
| R7 | Stage 4 `gait_report` brings a duplicate `gait_metrics_lite.py` | Low | Codebase bloat | Delete during copy-in (see §4.4) |
| R8 | Anipose vs pose3d DLT triangulation: two competing 3D paths | Medium | GUI confusion | Keep both as opt-in (pose3d default, anipose for power users) |

## 8. Out of scope (v3.0)

- Cloud sync, multi-user collaboration.
- DL model training (DLC). Reuse v1 subprocess invocation.
- Real-time live analysis during recording.
- Multi-animal / multi-walkway setups.
- Mobile / tablet UI.

These are v3.1+ candidates and will get their own DESIGN addenda.