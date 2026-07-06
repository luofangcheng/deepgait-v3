## ADDED Requirements

### Requirement: Mouse directory picker
The Mouse tab SHALL provide a directory picker (`QFileDialog.getExistingDirectory`)
labeled `Mouse Directory`. The selected path SHALL be remembered via
`QSettings`.

#### Scenario: User picks a mouse directory
- **WHEN** the user clicks `Browse...` and selects `/data/mouse_001`
- **THEN** the path field shows `/data/mouse_001`
- **AND** the path is persisted to `QSettings`

### Requirement: Run parameters form
The Mouse tab SHALL expose a form with the following fields, each with the
documented Stage-1 default: `mouse_id` (string, derived from directory name),
`tau_paw` (float, default `10.0`), `roi_pad` (int, default `50`),
`fps` (float, default `60.0`), `px_per_mm` (float, default `1.92`).

#### Scenario: Form pre-fills with documented defaults
- **WHEN** the tab is opened for the first time after install
- **THEN** every field shows its documented default value

### Requirement: Run and Cancel buttons
The Mouse tab SHALL provide `Run` (primary) and `Cancel` (secondary)
buttons. `Run` is enabled when a mouse directory is selected and the
worker is idle; `Cancel` is enabled only while the worker is running.

#### Scenario: Buttons toggle with worker state
- **WHEN** the worker is idle
- **THEN** `Run` is enabled and `Cancel` is disabled
- **AND** while the worker runs, `Run` is disabled and `Cancel` is enabled

### Requirement: Live progress and log pane
The Mouse tab SHALL display a `QProgressBar` (0-100) driven by the
worker's `progress(int)` signal and a `LogPane` widget driven by the
`log_line(str)` signal.

#### Scenario: Progress and log stream during a run
- **WHEN** the pipeline emits `progress(42)` and `log_line("loaded 600 frames")`
- **THEN** the progress bar shows 42
- **AND** the log pane appends the line at the bottom

### Requirement: Trial list table mirrors the SQLite trial table
The Mouse tab SHALL display a `QTableView` whose model is the `trial`
table of the chosen `output_dir/footprints.db`. Columns: `trial_id`,
`mouse_id`, `created_at`, `num_frames`, `fps`, `px_per_mm`. The model
SHALL open the SQLite file in read-only mode.

#### Scenario: Trial row appears after a successful run
- **WHEN** the worker emits `finished(TrialResult)`
- **THEN** the trial list gains one row matching `TrialResult.mouse_id`