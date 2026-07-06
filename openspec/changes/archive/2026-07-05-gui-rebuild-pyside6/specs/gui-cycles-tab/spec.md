## ADDED Requirements

### Requirement: Cycle table view
The Cycles tab SHALL display a `QTableView` over the `footprint_cycle`
table of the chosen `footprints.db`. Columns: `cycle_id`,
`touchdown_frame`, `liftoff_frame`, `peak_area_frame`, `duration_s`,
`max_area_mm2`, `centroid_at_peak_x_mm`, `centroid_at_peak_y_mm`,
`is_clean_liftoff`, `n_frames`. Rows MUST be sorted by `cycle_id`.

#### Scenario: Cycle table reflects the SQLite footprint_cycle table
- **WHEN** the user opens a trial with 42 cycles
- **THEN** the table shows 42 rows in ascending cycle_id order

### Requirement: Frame sub-table on row selection
When the user selects a cycle row, the tab SHALL display a child
`QTableView` over `footprint_frame` rows whose `cycle_id` matches.
Columns: `frame`, `time_s`, `area_mm2`, `area_px`, `centroid_x_mm`,
`centroid_y_mm`, `mean_intensity`, `peak_intensity`, `mean_pressure`,
`peak_pressure`, `is_peak_area`, `is_peak_intensity`.

#### Scenario: Selecting cycle 7 shows its frames
- **WHEN** the user clicks cycle_id=7
- **THEN** the sub-table lists exactly the frames belonging to cycle 7
- **AND** rows are sorted by `frame` ascending

### Requirement: Per-print thumbnail strip
The Cycles tab SHALL show a horizontal strip of thumbnails, one per
`png_path` in the selected cycle's frames, loaded from `output_dir/<png_path>`.
A missing PNG SHALL render a grey placeholder, not raise an error.

#### Scenario: Missing thumbnail is gracefully rendered
- **WHEN** the cycle references `per_print/cycle_0007_frame_0042.png` but the file is absent
- **THEN** the strip shows a grey placeholder for that slot

### Requirement: Empty-state banner
If no cycle rows exist, the tab SHALL display `No cycles detected for this trial`.

#### Scenario: Empty cycle table
- **WHEN** the user opens a trial whose pipeline produced zero cycles
- **THEN** the banner is shown
- **AND** no exception is raised