# DeepGait v3 — Migration Checklist

**Source**: [`../deepgait-v2/`](../deepgait-v2/)
**Target**: [`../deepgait-v3/deepgait3/`](../deepgait-v3/)
**Strategy**: copy-in + deep refactor (see [`DESIGN.md` §4](DESIGN.md#4-module-mapping-v2-algo--v3-core))

## Table of Contents

1. [PawPrint schema (36 fields)](#pawprint-schema-36-fields)
2. [Stage 1: dynamics_v0.4.2 → core/pawprint/](#stage-1-dynamics_v042--corepawprint)
3. [Stage 2: pose3d_v0.2.0 → core/calibration + triangulation + fusion/](#stage-2-pose3d_v020--corecalibration--triangulation--fusion)
4. [Stage 3: gait_metrics v0.1.0 → core/metrics/](#stage-3-gait_metrics-v010--coremetrics)
5. [Stage 4: gait_report v0.1.0 → core/report/](#stage-4-gait_report-v010--corereport)
6. [Adapter layer](#adapter-layer)
7. [Per-module status](#per-module-status)

## PawPrint schema (36 fields)

Source of truth: `deepgait-v2/dynamics_v0.4.2/dynamics_v04_module/dynamics_v04/models.py`

| Group | Field | Type | Stage that fills it |
|---|---|---|---|
| A: ID+time | `print_id` | int | 1 |
| A | `touchdown_frame` | int | 1 |
| A | `liftoff_frame` | int | 1 |
| A | `true_liftoff_frame` | int | 1 |
| A | `peak_area_frame` | int | 1 |
| A | `peak_intensity_frame` | int | 1 |
| A | `duration_s` | float | 1 |
| A | `time_to_peak_area_s` | float | 1 |
| A | `time_to_peak_intensity_s` | float | 1 |
| B: CatWalk phases | `loading_duration_s` | float | 1 |
| B | `weight_bearing_duration_s` | float | 1 |
| B | `unloading_duration_s` | float | 1 |
| C: Per-frame | `frames` | List[FrameData] | 1 |
| D: Geometry | `max_area_mm2` | float | 1 |
| D | `peak_frame_centroid_xy_mm` | (float, float) | 1 |
| D | `peak_frame_bbox_xyxy` | tuple | 1 |
| D | `print_length_mm` | float | 1 |
| D | `print_width_mm` | float | 1 |
| D | `print_orientation_deg` | float | 1 |
| E: Shape | `compactness` | float | 1 |
| E | `toe_positions_xy_mm` | Optional[List] | 1 (always None in v0.4.2) |
| E | `centroid_drift_mm` | float | 1 |
| F: Pressure dyn | `stand_index` | float | 1 |
| F | `rising_slope` | float | 1 |
| F | `peak_pressure` | float | 1 |
| F | `mean_pressure_at_peak` | float | 1 |
| F | `pressure_area_ratio` | float | 1 |
| F | `raw_intensity_curve` | List[float] | 1 |
| F | `touchdown_intensity` | float | 1 |
| F | `liftoff_intensity` | float | 1 |
| G: CoP | `cop_trajectory_mm` | List[Tuple[float, float]] | 1 |
| G | `cop_path_length_mm` | float | 1 |
| G | `cop_displacement_mm` | float | 1 |
| H: Decay (v0.4.1+) | `max_area_curve` | List[float] | 1 |
| H | `decay_phase_mask` | List[bool] | 1 |
| H | `decay_tau_ms` | Optional[float] | 1 |
| H | `decay_R2` | Optional[float] | 1 |
| H | `is_clean_liftoff` | bool | 1 |
| I: Quality | `quality.touches_edge` | bool | 1 |
| I | `quality.merged_with_neighbor` | bool | 1 |
| I | `quality.n_frames` | int | 1 |
| I | `quality.min_area_below_thresh` | bool | 1 |
| I | `quality.saturated_pixels_pct` | float | 1 |
| I | `quality.snr` | float | 1 |
| J: 3D linkage | `linkage_to_3d.paw_id` | Optional[str] | 2 |
| J | `linkage_to_3d.ankle_3d_at_peak` | Optional[(x,y,z)] | 2 |
| J | `linkage_to_3d.ankle_2d_projection` | Optional[(x,y)] | 2 |
| J | `linkage_to_3d.match_distance_mm` | Optional[float] | 2 |

## Stage 1: dynamics_v0.4.2 → core/pawprint/

**Source**: `deepgait-v2/dynamics_v0.4.2/dynamics_v04_module/dynamics_v04/`

| v2 file | LOC | v3 path | Migration action |
|---|---|---|---|
| `models.py` | ~200 | `core/pawprint/models.py` | Copy. Add `__all__`. Fix import to `numpy` etc. |
| `detection.py` | ~150 | `core/pawprint/detection.py` | Copy. |
| `grouping.py` | ~100 | `core/pawprint/grouping.py` | Copy. |
| `tracker.py` | ~200 | `core/pawprint/tracker.py` | Copy. |
| `extractor.py` | ~150 | `core/pawprint/extractor.py` | Copy. |
| `serializer.py` | ~150 | `core/io/pawprint_serializer.py` | Move (I/O concern, not algorithm). |
| `tests/` | 0 ❌ | `tests/unit/test_pawprint_*.py` | **Write 8 tests** (see `deepgait-v2/report.md` §12) |

**Public API after migration** (`deepgait3.core.pawprint.__init__`):
```python
from .models import FrameData, FootMask, PawPrint, QualityFlags, Linkage3D
from .extractor import PawPrintExtractor
```

**Verification step**: Run `pytest tests/unit/test_pawprint_*.py -v` after copy. Must pass all 8 tests before proceeding.

## Stage 2: pose3d_v0.2.0 → core/calibration + triangulation + fusion/

**Source**: `deepgait-v2/pose3d_v0.2.0/pose3d_module/pose3d/`

| v2 file | LOC | v3 path | Migration action |
|---|---|---|---|
| `keypoint_schema.py` | ~80 | `core/calibration/keypoint_schema.py` | Copy. |
| `calibration.py` | ~150 | `core/calibration/calibration.py` | Copy. |
| `models.py` | ~120 | `core/calibration/models.py` + `core/triangulation/models.py` | Split (Cam* vs Skel*). |
| `dlc_loader.py` | ~80 | `core/triangulation/dlc_loader.py` | Copy. |
| `triangulator.py` | ~120 | `core/triangulation/triangulator.py` | Copy. |
| `fusion.py` | ~150 | `core/fusion/fusion.py` | Copy. |
| `apply_to_pawprints.py` | ~50 | `core/fusion/apply_to_pawprints.py` | Copy. |
| `mock_data.py` | ~100 | `tests/fixtures/mock_pose3d.py` | Move (test fixture). |
| `mock_dlc.py` | ~80 | `tests/fixtures/mock_dlc.py` | Move (test fixture). |

**Public API after migration** (`deepgait3.core.calibration.__init__`):
```python
from .keypoint_schema import DEFAULT_SCHEMA, build_default_schema, build_custom_schema, PAW_IDS
from .calibration import calibrate_cameras_charuco, save_calibration, load_calibration
from .models import CameraIntrinsic, CameraExtrinsic, CalibrationPack
```

`deepgait3.core.triangulation.__init__`:
```python
from .models import Keypoint2D, Keypoint3D, Skeleton3D
from .triangulator import triangulate_keypoint, triangulate_skeleton
from .dlc_loader import load_dlc_h5, load_dlc_csv
```

`deepgait3.core.fusion.__init__`:
```python
from .fusion import project_3d_to_camera, match_ankle_to_pawprint
from .apply_to_pawprints import apply_linkage_to_pawprints
from .models import Linkage3DResult
```

## Stage 3: gait_metrics v0.1.0 → core/metrics/

**Source**: `deepgait-v2/gait_metrics_module/gait_metrics/`

| v2 file | LOC | v3 path | Migration action |
|---|---|---|---|
| `models.py` | ~120 | `core/metrics/models.py` | Copy. |
| `per_paw.py` | ~150 | `core/metrics/per_paw.py` | Copy. |
| `inter_paw.py` | ~80 | `core/metrics/inter_paw.py` | Copy. |
| `coordination.py` | ~100 | `core/metrics/coordination.py` | Copy. |
| `compute.py` | ~80 | `core/metrics/compute.py` | Copy. |

**Public API**: `from deepgait3.core.metrics import compute_gait_metrics, PerPawMetrics, PerPairMetrics, TrialMetrics, GaitMetrics`

## Stage 4: gait_report v0.1.0 → core/report/

**Source**: `deepgait-v2/gait_report v0.1.0/gait_report_module/gait_report/`

| v2 file | LOC | v3 path | Migration action |
|---|---|---|---|
| `csv_export.py` | ~120 | `core/report/csv_export.py` | Copy. |
| `excel_export.py` | ~80 | `core/report/excel_export.py` | Copy. |
| `plots.py` | ~250 | `core/report/plots.py` | Copy. |
| `html_report.py` | ~150 | `core/report/html_report.py` | Copy. |
| `pipeline.py` | ~80 | `core/report/pipeline.py` | Copy. |
| `_gait_metrics_lite.py` | ~150 | **DELETE** | Duplicate of `gait_metrics/`; not needed. |

**Public API**: `from deepgait3.core.report import run_full_pipeline`

## Adapter layer

**Location**: `deepgait3/core/adapters/__init__.py`

Purpose: let v1 GUI code call v3 internals without changes.

| v1 function (must keep callable) | v3 backing |
|---|---|
| `extract_footprints(video_path, **kw)` | `PawPrintExtractor(video_path, **kw)` |
| `compute_all_gait_metrics(in_stance, fps, ...)` | Convert `in_stance` ndarray → synthetic `PawPrint[]` → `compute_gait_metrics()` |
| `calibrate_charuco(videos, ...)` | `calibrate_cameras_charuco(videos, ...)` |
| `compute_ftir_gait_metrics(...)` | `run_full_pipeline(...)` |
| `RunDetector`, `RunResult` | Re-export from `core.pawprint` with v1 aliases |

## Per-module status

| Stage | Source LOC | Status | Tests | Notes |
|---|---|---|---|---|
| 1: pawprint | ~800 | 🚧 planned | 0/8 | Awaiting user review of DESIGN.md |
| 2a: calibration | ~200 | 🚧 planned | 0 | |
| 2b: triangulation | ~300 | 🚧 planned | 0 | |
| 2c: fusion | ~200 | 🚧 planned | 0 | |
| 3: metrics | ~400 | 🚧 planned | 0 | |
| 4: report | ~600 | 🚧 planned | 0 | |
| Adapter | ~300 | 🚧 planned | 0 | |
| **Total** | **~2800** | | **0/8+** | |

Update this table after each stage lands.