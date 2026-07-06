## ADDED Requirements

### Requirement: PySide6 main window with 8 tabs
The system SHALL provide a `QMainWindow` (subclass `DeepGait3Window`) that
contains exactly eight tabs in this fixed order: `Mouse`, `Footprints`,
`Cycles`, `Stage 2`, `Stage 3`, `Stage 4`, `Editor`, `Charts`, `Camera`,
`Settings`. The window title SHALL be `DeepGait v3`.

#### Scenario: Application launches with the 8 tabs
- **WHEN** the user runs `deepgait3 gui`
- **THEN** the main window appears with the eight tabs in the documented order
- **AND** the window title is `DeepGait v3`

### Requirement: Stage 1 pipeline runs on a background thread
The system SHALL run `Stage1Pipeline.run(frame_dir, output_dir)` on a
`QThread`-hosted worker object so the UI thread stays responsive. The worker
MUST emit `progress(int)`, `log_line(str)`, `finished(TrialResult)`, and
`failed(str)` Qt signals.

#### Scenario: Long pipeline does not freeze the UI
- **WHEN** the user clicks `Run` on the Mouse tab
- **THEN** the progress bar updates incrementally
- **AND** the window can be moved, resized, and tab-switched without freezing

### Requirement: Cancellation stops the worker cleanly
The worker MUST expose a `cancel()` slot that flips an internal
`QAtomicInt` flag. The worker SHALL poll the flag once per processed frame
and SHALL stop raising new frames; any partial SQLite output SHALL be
written to `footprints.db.partial` and removed on the next successful run.

#### Scenario: User cancels a running pipeline
- **WHEN** the user clicks `Cancel` while the worker is processing
- **THEN** the worker stops within one frame
- **AND** no `footprints.db` is committed for that trial

### Requirement: Settings persist via QSettings
The system SHALL persist the last-selected mouse directory, last output
directory, current tab index, and Settings-tab parameter values via
`QSettings` under organization `DeepGait3` and application `DeepGait3`.

#### Scenario: Settings survive an application restart
- **WHEN** the user changes the Mouse Directory field and restarts the GUI
- **THEN** the field is pre-populated with the previously chosen directory

### Requirement: Logging goes through loguru
The system SHALL install a `loguru` `InterceptHandler` on the root Python
logger so all worker and pipeline logs flow into the GUI's log pane. The
codebase SHALL NOT call `print()` from any production path under
`deepgait3/gui/`.

#### Scenario: Worker log lines appear in the log pane
- **WHEN** the pipeline writes a log line via `loguru.logger.info(...)`
- **THEN** the line appears in the GUI log pane within 100 ms

### Requirement: GUI subcommand is exposed
`deepgait3.cli.main` SHALL accept a `gui` subcommand that constructs a
`QApplication` and shows `DeepGait3Window`. `python -m deepgait3 gui` SHALL
behave identically.

#### Scenario: Headless invocation shows help without error
- **WHEN** the user runs `deepgait3 gui --help` with `QT_QPA_PLATFORM=offscreen`
- **THEN** argparse prints the `gui` subcommand usage and exits 0