# gui-settings-tab Specification

## Purpose
TBD - created by archiving change gui-rebuild-pyside6. Update Purpose after archive.
## Requirements
### Requirement: Stage-1 defaults form
The Settings tab SHALL expose a form mirroring the Stage-1 `DEFAULTS`
dict in `deepgait3.core.pawprint.pipeline`. Fields: `tau_paw` (10.0),
`roi_pad` (50), `fps` (60.0), `px_per_mm` (1.92), `walkway_roi` (4-tuple
default `(0, 15, 1920, 360)`), `iou_min` (0.3), `max_gap_frames` (3),
`min_print_frames` (2). Each field SHALL have a `Reset to default` button
alongside it.

#### Scenario: Defaults match the Stage-1 pipeline
- **WHEN** the user clicks `Reset to default` next to `tau_paw`
- **THEN** `tau_paw` becomes `10.0`

### Requirement: Persistence of Settings values
All Settings-tab values SHALL be persisted via `QSettings` under
`DeepGait3/DeepGait3/stage1_defaults/<key>` and SHALL be reloaded on
startup. On startup, the Mouse tab's form SHALL be pre-populated from the
same `QSettings` keys.

#### Scenario: User edits tau_paw, restarts app, sees the new value
- **WHEN** the user sets `tau_paw=12.5` and closes the app
- **AND** the user reopens the app
- **THEN** the Settings tab shows `tau_paw=12.5`
- **AND** the Mouse tab's `tau_paw` form field is pre-filled with `12.5`

### Requirement: Settings flow into Stage1Pipeline at Run time
The Stage-1 pipeline constructor MUST receive the current Settings-tab
values as keyword arguments when the user clicks `Run` on the Mouse
tab. The GUI SHALL NOT modify the `DEFAULTS` dict in `pipeline.py`.

#### Scenario: Stage1Pipeline receives the edited tau_paw
- **WHEN** the user edits `tau_paw` to `12.5` and clicks `Run`
- **THEN** the worker calls `Stage1Pipeline(mouse_id=..., tau_paw=12.5, ...)`

### Requirement: Validation rejects invalid values
Negative numeric values SHALL be rejected on focus-out with an inline
error message; the offending field's value is restored to its previous
value. The Mouse tab's Run button SHALL be disabled while any Settings
field is invalid.

#### Scenario: User enters a negative px_per_mm
- **WHEN** the user types `-1.0` into `px_per_mm` and tabs away
- **THEN** the field shows `1.92` again
- **AND** a red error label appears next to the field
- **AND** the Run button on the Mouse tab is disabled until the error is cleared

