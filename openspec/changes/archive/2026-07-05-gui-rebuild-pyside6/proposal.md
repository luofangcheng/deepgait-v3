## Why

The Stage-1 footprint pipeline (`deepgait3.core.pawprint.pipeline.Stage1Pipeline`)
is complete and exposed via the `deepgait3 stage1` CLI, but end users have no
graphical way to pick a mouse directory, watch the pipeline run, or inspect the
resulting `TrialResult`. A CLI alone does not substitute for an interactive
workbench in a research-lab setting, and DESIGN.md §2 promises an L5 GUI that
wraps the new core. We need a PySide6 shell now so Stage 1 is usable by
non-CLI users; Stages 2-4 are still scaffolds and will appear as placeholders.

## What Changes

- Add a fresh `deepgait3.gui` package (PySide6, LGPL-compliant, matches the v1
  framework choice) that wraps the existing `Stage1Pipeline.run()` and the
  SQLite writer at `output_dir/footprints.db`. No port of v1's GUI code — the
  v3 GUI is built directly on top of the v3 core data contracts.
- Introduce an 8-tab `QMainWindow` (`Mouse`, `Footprints`, `Cycles`,
  `Stage 2`, `Stage 3`, `Stage 4`, `Editor`, `Charts`, `Camera`, `Settings`).
  Mouse / Footprints / Cycles / Settings tabs are wired to real Stage-1 code;
  Stage 2-4 tabs are functional shells that display a "Coming soon" pane
  guarded by a runtime check on `deepgait3.core.calibration` /
  `triangulation` / `fusion` / `metrics` / `report` availability.
- Run the pipeline on a `QThread` worker (`Stage1Worker`) so the UI stays
  responsive while frames are processed; the worker emits per-stage progress
  signals that feed a progress bar and a live log pane.
- Persist the latest trial list and tab settings via `QSettings`
  (`org.deepgait.deepgait3` scope) so a user can resume after restart.
- Expose the GUI as `deepgait3 gui` (add a subcommand to `cli.py`) and via the
  installed entry point so `python -m deepgait3 gui` also works.

## Capabilities

### New Capabilities

- `gui-shell`: PySide6 `QMainWindow` with the 8-tab layout, shared `AppState`,
  the `Stage1Worker` QThread, and `QSettings` persistence.
- `gui-mouse-tab`: Mouse-directory picker, parameter form (mouse_id, tau_paw,
  roi_pad, fps, px_per_mm), Run / Cancel buttons, live progress + log pane,
  and a trial-list table that reflects the SQLite `trial` table.
- `gui-footprints-tab`: Cumulative-mask / cumulative-intensity /
  cumulative-overlay viewer backed by `Stage1Pipeline`'s visualization outputs
  (`cumulative_mask.png`, `cumulative_intensity.png`,
  `cumulative_overlay.png`) plus a frame slider that scrubs `per_frame/`.
- `gui-cycles-tab`: Table view over the `footprint_cycle` table; selecting a
  row reveals its `footprint_frame` rows plus thumbnails from `per_print/`.
- `gui-settings-tab`: Form for the v3 Stage-1 defaults (`tau_paw`, `roi_pad`,
  `fps`, `px_per_mm`, `walkway_roi`, `iou_min`, `max_gap_frames`,
  `min_print_frames`), persisted to `QSettings`.

### Modified Capabilities

None — `deepgait3` has no existing specs. The Stage-1 pipeline and SQLite
schema are unchanged; the GUI is a read/write client on top of them.

## Impact

- New package: `deepgait3/gui/` (10-12 modules under 400 lines each, per the
  "many small files" rule in the global coding-style guidelines).
- `deepgait3/cli.py`: add a `gui` subcommand (≈10 LOC) — no other CLI changes.
- `pyproject.toml`: add `PySide6>=6.6,<7` (already present) to `[project]
  dependencies` (confirmed already listed) and a `[project.gui]` extra that
  pulls in nothing extra. No new hard deps.
- Tests: `tests/gui/` using `pytest-qt` with `QT_QPA_PLATFORM=offscreen`.
  Existing `tests/unit/` for `pawprint/` are unchanged.
- Backward compatibility: the `stage1` CLI subcommand continues to work
  untouched. GUI failure must not block CLI users.
- No Cython changes in v3.0 (per DESIGN.md rule 10).