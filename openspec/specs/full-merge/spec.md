# full-merge Specification

## Purpose
TBD - created by archiving change full-merge-deep-gailt. Update Purpose after archive.
## Requirements
### Requirement: Single deepgait3 namespace
The system SHALL expose deep-gailt's whole package under
`deepgait3.deepgait.*` so that a single import path works for GUI,
hardware, and core.

#### Scenario: Legacy GUI import resolves
- GIVEN `deepgait3` is installed
- WHEN `from deepgait3.deepgait.gui.main_window import MainWindow`
- THEN the import resolves without legacy-shim indirection.

### Requirement: CLI gui subcommand
`deepgait3 gui` SHALL launch the merged PySide6 main window.

#### Scenario: GUI launches from CLI
- GIVEN `deepgait3` is installed and `QT_QPA_PLATFORM=offscreen`
- WHEN `deepgait3 gui` runs
- THEN the command returns 0 after a 11-tab window is shown.

