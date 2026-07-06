## 1. Package scaffold

- [ ] 1.1 Create `deepgait3/gui/` package with empty `__init__.py`
- [ ] 1.2 Create `deepgait3/gui/tabs/`, `deepgait3/gui/widgets/`, `deepgait3/gui/workers/` subpackages
- [ ] 1.3 Add `deepgait3/gui/db_reader.py` (stdlib sqlite3 thin wrapper, WAL mode, read-only)
- [ ] 1.4 Add `deepgait3/gui/app_state.py` (QSettings-backed dataclass for last_dir, last_output_dir, tab_index)

## 2. CLI subcommand

- [ ] 2.1 Add `gui` subparser to `deepgait3/cli.py`
- [ ] 2.2 Add `_cmd_gui(args)` that constructs `QApplication` and shows `DeepGait3Window`
- [ ] 2.3 Wire `python -m deepgait3 gui` to the same entry point
- [ ] 2.4 Smoke-test `deepgait3 gui --help` under `QT_QPA_PLATFORM=offscreen`

## 3. Stage 1 worker

- [ ] 3.1 Implement `Stage1Worker(QObject)` with signals `progress(int)`, `log_line(str)`, `finished(TrialResult)`, `failed(str)`
- [ ] 3.2 Add `cancel()` slot flipping a `QAtomicInt` flag
- [ ] 3.3 On entry, call `Stage1Pipeline(...).run(frame_dir, output_dir)` and forward per-frame progress as percent
- [ ] 3.4 On cancellation, write to `footprints.db.partial` and remove on next success
- [ ] 3.5 Install `loguru.InterceptHandler` on the worker thread's root logger

## 4. Main window shell

- [ ] 4.1 Implement `DeepGait3Window(QMainWindow)` with a `QTabWidget` containing 8 tabs in fixed order
- [ ] 4.2 Add `tabs/base_tab.py` exposing `BaseTab(QWidget)` with shared layout helpers
- [ ] 4.3 Implement `tabs/stage_stub_tab.py` for Stages 2-4 (centered "Coming soon" pane with banner)
- [ ] 4.4 Implement stub tabs: `editor_tab.py`, `charts_tab.py`, `camera_tab.py` reusing `stage_stub_tab.py`
- [ ] 4.5 Wire tab order persistence via `QSettings` on `currentChanged` and on close

## 5. Mouse tab

- [ ] 5.1 Implement `tabs/mouse_tab.py` with directory picker, Run/Cancel buttons, parameter form
- [ ] 5.2 Wire form fields to `AppState` defaults from `QSettings`
- [ ] 5.3 Connect Run to `Stage1Worker` on a fresh `QThread`; connect signals to progress bar and log pane
- [ ] 5.4 Add trial list `QTableView` driven by `db_reader.list_trials(output_dir)`
- [ ] 5.5 Disable Run when no directory selected or while worker is busy; enable Cancel only while busy

## 6. Footprints tab

- [ ] 6.1 Implement `tabs/footprints_tab.py` with output-directory picker and viewer
- [ ] 6.2 Validate that the three cumulative PNGs exist; show warning and disable viewer if missing
- [ ] 6.3 Render the three PNGs in a stacked viewer that preserves aspect ratio
- [ ] 6.4 Add per-frame slider that lists `per_frame/frame_NNNN_det.png` and previews the selected file
- [ ] 6.5 Show "Run in progress — viewer disabled" banner while the worker is running

## 7. Cycles tab

- [ ] 7.1 Implement `tabs/cycles_tab.py` with `QTableView` over `footprint_cycle`
- [ ] 7.2 Implement child `QTableView` over `footprint_frame` filtered by selected `cycle_id`
- [ ] 7.3 Add horizontal thumbnail strip using `QListView` in `IconMode`, loading each `png_path`
- [ ] 7.4 Render a grey `QPixmap` placeholder for missing thumbnails
- [ ] 7.5 Show `No cycles detected for this trial` banner when the cycle table is empty

## 8. Settings tab

- [ ] 8.1 Implement `tabs/settings_tab.py` with a form for the Stage-1 `DEFAULTS` keys
- [ ] 8.2 Add per-field `Reset to default` buttons
- [ ] 8.3 Persist every value via `QSettings` under `DeepGait3/DeepGait3/stage1_defaults/<key>`
- [ ] 8.4 On focus-out, validate numeric fields (positive); restore prior value and show inline error on failure
- [ ] 8.5 Wire Settings values into the Mouse tab form on startup
- [ ] 8.6 Pipe Settings values into `Stage1Pipeline(...)` kwargs at Run time (do not mutate `DEFAULTS`)

## 9. Shared widgets

- [ ] 9.1 Implement `widgets/log_pane.py` (`QTextEdit` with auto-scroll, loguru-backed)
- [ ] 9.2 Implement `widgets/progress_panel.py` (`QProgressBar` + label)
- [ ] 9.3 Implement `widgets/image_view.py` (aspect-preserving `QLabel` viewer)

## 10. Tests

- [ ] 10.1 Add `tests/gui/conftest.py` setting `QT_QPA_PLATFORM=offscreen`
- [ ] 10.2 Add `tests/gui/test_shell.py` asserting 8 tabs in fixed order
- [ ] 10.3 Add `tests/gui/test_mouse_tab.py` with a fake pipeline emitting synthetic progress and logs
- [ ] 10.4 Add `tests/gui/test_cycles_tab.py` against a fixtures DB (5 cycles, 3 frames each)
- [ ] 10.5 Add `tests/gui/test_settings_tab.py` for persistence and validation
- [ ] 10.6 Run `pytest tests/gui -v` and confirm all green

## 11. Lint and CI

- [ ] 11.1 `ruff check deepgait3/gui tests/gui`
- [ ] 11.2 `ruff format --check deepgait3/gui tests/gui`
- [ ] 11.3 Confirm `pytest -n auto tests/gui tests/unit` stays green
- [ ] 11.4 Confirm `deepgait3 stage1` CLI is unchanged (smoke run on a tiny fixtures tree)