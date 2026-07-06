## Context

`deepgait3.core.pawprint.pipeline.Stage1Pipeline.run(frame_dir, output_dir)` is
the only Stage-1 entry point and is callable from the `deepgait3 stage1` CLI.
Its outputs are deterministic files on disk:

- `output_dir/footprints.db` — SQLite (`trial`, `mouse_roi`,
  `footprint_cycle`, `footprint_frame` tables; see `core/pawprint/db.py`).
- `output_dir/cumulative_mask.png`,
  `output_dir/cumulative_intensity.png`,
  `output_dir/cumulative_overlay.png` — overview visualisations.
- `output_dir/per_frame/frame_NNNN_det.png` — per-frame debug frames.
- `output_dir/per_print/cycle_NNNN_frame_NNNN.png` — footprint crops.
- `output_dir/mouse_roi.txt` — text dump.

DESIGN.md §2 says "L5 GUI (PySide6 8-tab) reuse 100% from v1 + add adapter
imports", but the v1 GUI is not yet wired into the deepgait-v3 package and
v1's data contract is `in_stance: ndarray`, not `TrialResult`. Forcing v1
GUI on top of v3 requires the entire adapter layer (DESIGN.md §4.5). We
choose to build a small, Stage-1-only GUI now and leave the v1-reuse path
to a later change once Stages 2-4 land.

The runtime constraint set comes from the global coding-style rules: many
small files (<800 lines), no mutation, loguru-only, `print()` banned, and
the `pyproject.toml` already pins `PySide6>=6.6,<7` and `pytest-qt`.

## Goals / Non-Goals

**Goals**

- 8-tab PySide6 `QMainWindow` with mouse-friendly defaults.
- Stage-1 pipeline runs in a `QThread` worker; UI stays responsive.
- Tabs read from the SQLite file the pipeline writes — never re-derive data.
- Persist user settings via `QSettings`.
- Headless-safe tests via `pytest-qt` (`QT_QPA_PLATFORM=offscreen`).
- Single dependency delta: zero new runtime deps (PySide6 already declared).

**Non-Goals**

- Stages 2-4 functionality (placeholder panes only).
- Reuse of v1 `deepgait/gui/` code or its 8-tab assets.
- Cython compilation of GUI modules (DESIGN.md rule 10).
- Real-time live capture / live analysis (DESIGN.md §1.2 non-goal).
- Multi-animal, SFI, group stats (DESIGN.md §1.2 non-goals).
- Internationalization (the v1 bilingual system stays in v1 for now).

## Decisions

### D1 — Fresh GUI, no v1 reuse

**Choice.** Build `deepgait3.gui` from scratch on top of `Stage1Pipeline`
and the SQLite schema.
**Why.** v1's GUI expects v1's `in_stance` ndarray contract. Wiring v1 GUI
to v3 needs the adapter layer from DESIGN.md §4.5, which is staged later.
**Alternatives.** (a) Port v1 GUI verbatim — blocked by missing adapter;
(b) Build only a minimal Stage-1 launcher, no 8 tabs — rejected because
DESIGN.md promises 8 tabs and the team already has tab icons.

### D2 — 8 tabs, only 4 wired to Stage 1

**Choice.** Implement `MouseTab`, `FootprintsTab`, `CyclesTab`,
`SettingsTab` against real code; `Stage2Tab`, `Stage3Tab`, `Stage4Tab`,
`EditorTab`, `ChartsTab`, `CameraTab` render a `QWidget` with a centered
"Coming soon — Stage N not yet implemented" label and are otherwise no-op.
**Why.** Keeps the visual layout aligned with v1 so the eventual v1-reuse
migration is mechanical; honours the rule "no features beyond what was
asked" while not regressing the layout contract.
**Alternatives.** Hide the stub tabs — rejected; v1 users will look for
them at known positions.

### D3 — `Stage1Worker` (QThread + worker-object pattern)

**Choice.** Use the QObject worker moved onto a `QThread` (not a subclass
of `QThread`). Worker emits `progress(int)`, `log_line(str)`, `finished(TrialResult)`,
`failed(str)`. The main window connects these to the progress bar, log
pane, and tab state.
**Why.** Same pattern DESIGN.md §2.3 calls out for v1; cleanly emits Qt
signals; cancellable via a `QAtomicInt` flag the worker polls each frame.
**Alternatives.** `QProcess` wrapping `deepgait3 stage1` — rejected
because it loses typed `TrialResult` access and forces JSON re-parsing.

### D4 — Read SQLite for tab display

**Choice.** Tabs query `footprints.db` via `sqlite3` (stdlib). The
`db.py` schema is the contract.
**Why.** Decouples tabs from the pipeline in-process state. Re-opening
a previous trial (different `output_dir`) is then free.
**Alternatives.** Re-use `TrialResult` object — rejected; only the running
worker has it.

### D5 — `QSettings` (org.deepgait.deepgait3 / DeepGait3)

**Choice.** Tab indices, last mouse directory, last output directory, and
the Settings-tab parameter values persist via `QSettings`.
**Why.** Cross-platform, zero-dependency, matches v1 behaviour.
**Alternatives.** JSON in `~/.config/deepgait3/` — would work but adds a
file-format responsibility the global rules don't ask for.

### D6 — File layout

```
deepgait3/gui/
    __init__.py
    app.py              # QApplication setup, DeepGait3Window
    main_window.py      # DeepGait3Window (QMainWindow, 8 tabs)
    app_state.py        # AppState (last_dir, last_output_dir, settings)
    workers/
        __init__.py
        stage1_worker.py
    tabs/
        __init__.py
        base_tab.py
        mouse_tab.py
        footprints_tab.py
        cycles_tab.py
        stage_stub_tab.py   # used by Stage2/3/4
        editor_tab.py       # stub
        charts_tab.py       # stub
        camera_tab.py       # stub
        settings_tab.py
    widgets/
        __init__.py
        log_pane.py
        progress_panel.py
        image_view.py
    db_reader.py        # sqlite3 thin wrapper used by tabs
```

Every module ≤400 lines; `base_tab.py` ≤200. All follow global rules.

### D7 — Logging channel

`loguru.logger` everywhere (DESIGN.md rule 7). `QTextEdit`-backed log pane
subscribes via a `loguru.InterceptHandler` so `print()` never leaks.

### D8 — Headless testing

Tests in `tests/gui/` use `pytest-qt` (`qtbot`). A `conftest.py` at the
package root already sets `QT_QPA_PLATFORM=offscreen`; the global
`pyproject.toml` already pins `pytest-qt>=4.4` in the `dev` extra.

## Risks / Trade-offs

- [Stub tabs feel half-finished] → Mark them visually with a yellow
  banner and version-targeted tooltip ("Stage 2 ships in v3.1").
- [Worker cancellation races with SQLite writes] → Worker commits
  transactions incrementally (one cycle at a time) and checks the cancel
  flag between cycles; an unfinished trial leaves a `_partial.db` that
  the GUI renames to `footprints.db` only on successful completion.
- [Settings persistence conflicts with multi-instance use] → `QSettings`
  is process-shared by design; we accept last-writer-wins.
- [SQLite locked when worker writes and tab reads] → Use WAL mode
  (`PRAGMA journal_mode=WAL`) on connection; tabs open read-only.
- [PySide6 wheels on Windows require MSVC redistributable] → already
  accepted in v1; no new constraint.
- [GUI startup time with cold PySide6 import ≈ 0.8 s] → acceptable for
  desktop workflow; document in DESIGN.md follow-up.

## Migration Plan

- Phase 1 (this change): ship `deepgait3.gui` with 4 wired tabs +
  6 stub tabs; `deepgait3 gui` CLI subcommand; `tests/gui/` smoke tests.
- Phase 2 (later change): adapter layer per DESIGN.md §4.5; v1 GUI tabs
  ported over.
- Phase 3 (Stages 2-4 land): stub tabs replace their placeholders with
  real functionality, one change per stage.

Rollback: deleting `deepgait3/gui/` and the `gui` subcommand from
`cli.py` restores the prior CLI-only state. No database schema changes,
so no data migration is needed.

## Open Questions

- Should the Camera tab reuse v1's Hikvision/Basler driver wrappers, or
  stub entirely in v3.0? (TBD — v3 has no L1 hardware yet.)
- Should `SettingsTab` write back into the Stage-1 `DEFAULTS` dict in
  `pipeline.py`, or stay GUI-local and only flow into `Stage1Pipeline`
  constructor kwargs? (TBD — current design: GUI-local; `Stage1Pipeline`
  kwargs are the integration point.)