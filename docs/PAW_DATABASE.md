# DeepGait v3 — Multi-dimensional PawPrint Database

**Decision**: HDF5 (1 trial = 1 `.h5` file, layered groups)
**Date**: 2026-06-25
**Why**: reuse v1's existing HDF5 SWMR infrastructure, zero new deps, fast random access via memory-mapping.

## Requirements

1. Per-print record with text (trial_id, notes), numeric (36 PawPrint fields + per-frame arrays), and image (mask, peak frame PNG, pressure map).
2. Fast reads: open a single trial in < 100 ms, retrieve any one print in < 10 ms.
3. Cross-trial aggregation is a future concern (v3.1) — not in this schema.

## Schema

```
trial_2026-06-25_mouse03_baseline.h5
│
├── /                        # root attributes (text)
│   ├── trial_id             string scalar: "2026-06-25_mouse03_baseline"
│   ├── animal_id            string scalar: "mouse03"
│   ├── treatment            string scalar: "baseline"
│   ├── fps                  int   scalar: 60
│   ├── px_per_mm            float scalar: 1.92
│   ├── walkway_length_mm    float scalar: 1000.0
│   ├── fTIR_video_path      string scalar (relative)
│   ├── timestamp            string scalar (ISO 8601)
│   └── notes                string scalar (free-form)
│
├── /detector/               # how was this trial processed?
│   ├── algorithm            string: "exg" | "lab_astar" | "exgr" | "color_distance" | "v3_bg_sub"
│   ├── threshold            float
│   ├── min_area_px          int
│   └── sensitivity_mode     string: "strict" | "balanced" | "loose"
│
├── /pawprints/              # one group per PawPrint
│   ├── print_000/
│   │   ├── meta/            # 36 scalar fields
│   │   │   ├── print_id                int
│   │   │   ├── touchdown_frame         int
│   │   │   ├── liftoff_frame           int
│   │   │   ├── true_liftoff_frame      int
│   │   │   ├── peak_area_frame         int
│   │   │   ├── peak_intensity_frame    int
│   │   │   ├── duration_s              float
│   │   │   ├── time_to_peak_area_s     float
│   │   │   ├── time_to_peak_intensity_s float
│   │   │   ├── loading_duration_s      float
│   │   │   ├── weight_bearing_duration_s float
│   │   │   ├── unloading_duration_s    float
│   │   │   ├── max_area_mm2            float
│   │   │   ├── peak_frame_centroid_x_mm float
│   │   │   ├── peak_frame_centroid_y_mm float
│   │   │   ├── peak_frame_bbox_xyxy    int[4]
│   │   │   ├── print_length_mm         float
│   │   │   ├── print_width_mm          float
│   │   │   ├── print_orientation_deg   float
│   │   │   ├── compactness             float
│   │   │   ├── centroid_drift_mm       float
│   │   │   ├── stand_index             float
│   │   │   ├── rising_slope            float
│   │   │   ├── peak_pressure           float
│   │   │   ├── mean_pressure_at_peak   float
│   │   │   ├── pressure_area_ratio     float
│   │   │   ├── touchdown_intensity     float
│   │   │   ├── liftoff_intensity       float
│   │   │   ├── cop_path_length_mm      float
│   │   │   ├── cop_displacement_mm     float
│   │   │   ├── decay_tau_ms            float (or nan)
│   │   │   ├── decay_R2                float (or nan)
│   │   │   ├── is_clean_liftoff        bool
│   │   │   ├── paw_id                  string (None until Stage 2)
│   │   │   └── match_distance_mm       float (None until Stage 2)
│   │   │
│   │   ├── timeseries/        # per-frame arrays (numeric)
│   │   │   ├── frame_idx            int[N]
│   │   │   ├── time_s               float[N]
│   │   │   ├── area_mm2             float[N]
│   │   │   ├── mean_pressure        float[N]
│   │   │   ├── mean_intensity       float[N]
│   │   │   ├── peak_intensity       float[N]
│   │   │   ├── cop_x_mm             float[N]
│   │   │   ├── cop_y_mm             float[N]
│   │   │   ├── decay_phase          bool[N]
│   │   │   └── max_area_curve       float[N]   # cumulative
│   │   │
│   │   ├── images/             # image arrays (variable shape per print)
│   │   │   ├── peak_frame_bgr       uint8[H, W, 3]   # cropped to padded bbox
│   │   │   ├── peak_mask            bool[H, W]
│   │   │   ├── peak_pressure_map    float32[H, W]
│   │   │   └── peak_overlay_png     uint8[H, W, 3]   # visualization PNG
│   │   │
│   │   └── quality/
│   │       ├── touches_edge               bool
│   │       ├── merged_with_neighbor       bool
│   │       ├── n_frames                   int
│   │       ├── min_area_below_thresh      bool
│   │       ├── saturated_pixels_pct       float
│   │       └── snr                        float
│   │
│   ├── print_001/  ... (same layout)
│   └── print_NNN/
│
└── /index/                  # flat numeric arrays for fast cross-print queries
    ├── print_ids            int[N]
    ├── paw_ids              string[N]  (empty until Stage 2)
    ├── touchdown_frames     int[N]
    ├── liftoff_frames       int[N]
    ├── peak_frames          int[N]
    ├── max_areas_mm2        float[N]
    ├── durations_s          float[N]
    └── print_centroids_x_mm float[N]   # for fast spatial filtering
```

## Why HDF5

| Property | HDF5 | CSV+NPZ | MongoDB | Parquet |
|---|---|---|---|---|
| Random read per print | <10 ms (memory-map) | 100+ ms (npz open) | <50 ms (BSON parse) | column-only |
| Image inside same file | yes (dataset) | no (separate file) | yes (GridFS) | no |
| Schema enforcement | typed datasets | none | BSON schema | typed columns |
| Cross-print query | need index group | need to load all | native | native |
| Single-file portability | yes | no (folder) | no (DB) | yes |
| Already in v1 stack | yes | no | no | no |

## Reading examples

```python
import h5py

with h5py.File("trial_2026-06-25_mouse03_baseline.h5", "r") as f:
    # Text
    print(f.attrs["trial_id"], f.attrs["treatment"])

    # All prints' centroids (fast — flat array in /index/)
    cx = f["/index/print_centroids_x_mm"][:]

    # One print's metadata
    pp = f["/pawprints/print_007"]
    duration = pp["meta/duration_s"][()]
    peak_frame = pp["meta/peak_area_frame"][()]

    # Timeseries
    timeseries = pp["timeseries"]
    times = timeseries["time_s"][:]
    areas = timeseries["area_mm2"][:]

    # Image
    peak_bgr = pp["images/peak_frame_bgr"][:]
    mask = pp["images/peak_mask"][:]
```

## Out of scope for v3.0

- Cross-trial aggregation (Parquet index of all trials — v3.1)
- Write concurrency / SWMR streaming (reuse v1 `h5_writer.py`)
- Compression tuning (default gzip level 4 is fine)
- Encryption (v1 license layer handles this at the file level)

## Migration path from v2

`dynamics_v04/serializer.py::save_pawprint_database` writes a folder-based
DB (one folder per print with `metadata.json`, `peak_frame.png`,
`curves.png`, `frames/frame_NNN.npz`).  V3's `core/io/pawprint_serializer.py`
must keep that folder layout as an **export format** for back-compat, but
the **canonical DB** is now the single HDF5 file described above.