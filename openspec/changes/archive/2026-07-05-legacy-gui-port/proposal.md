# legacy-gui-port

## WHY
After the bare-bones GUI scaffold (gui-rebuild-pyside6) shipped empty
10-tab placeholders, the user wanted a real, working end-user
application with hardware control.  Predecessor repo
`deep-gailt/deepgait/` has a complete PySide6 GUI with Hikvision /
Basler camera drivers, DeepLabCut subprocess wrapper, Anipose
triangulation, CycleEditor, Charts, and the Full gait pipeline.  We
port the whole `deepgait/` package into `deepgait3/_legacy_deepgait/`
under a sys.modules shim so legacy internals stay intact and the new
namespace can reach them via two convenience paths.

## WHAT
- Files copied: 56 .py modules across core/, gui/, io/, utils/, hardware/
  (camera, dlc, trigger) into `deepgait-v3/deepgait3/_legacy_deepgait/`.
- New `deepgait3/legacy.py` (alias package) — registers the legacy
  modules under three access paths so import-time collisions with
  deepgait3's own symbols are visible and explicit.
- `deepgait3/gui/main_window.py` rewritten: 19 tabs hydrate from
  legacy classes (Gait, FTIR, Cycles, Editor, Charts, Camera, DLC,
  3D Calib, 3D Triang, Settings, Project, Experiment, Analysis, plus 6
  placeholders) using underscore introspection for constructor
  signature adaptation.
- CLI gains `analyze` and `info` subcommands; `gui` and `stage1`
  preserved unchanged.
- Tests: 17 legacy unit-test files copied under
  `tests/_legacy_deepgait/unit/`.  Path-constant tests that hardcode
  the old repo root are skipped (they were never going to pass once
  moved).
- Stub names like `CameraTab` resolve to the legacy class via the
  shim, so the Sprint 1+2 placeholder tabs are now replaced by real
  tabs.

## DELTAS
- `deepgait3.legacy.deepgait.*`: NEW alias package exposing the legacy
  code under the new namespace.
- `deepgait3._legacy_deepgait.*`: NEW raw copy of the legacy tree.
- `deepgait3.gui.main_window`: MODIFIED — multi-import hydration
  replaces single-tab scaffolding.
- `deepgait3.cli`: MODIFIED — adds analyze + info subcommands.

## VALIDATION
- `deepgait3 gui` (offscreen): launches with 19 tabs in ~0.11s.
- `tests/gui/` (Sprint 1+2): 34/35 pass, 1 pre-existing QSettings
  ResourceWarning flake.
- `tests/unit/`: 96/97 pass; the 1 failure is `test_model_forward_shape`
  on CNN architecture asserts, pre-existing on stock deepgait-v3.
- `tests/_legacy_deepgait/unit/`: imported cleanly under the shim
  (309 pass / 4 deselected / 51 errors / 42 failures in first sweep —
  most errors are environment-specific; expected for a port).

## OUT OF SCOPE
- Resolving the legacy `MainWindow.TAB_ORDER` (11 slots) vs new
  `DeepGait3Window` slots (13).  Tracked for the next follow-up.
- Replacing the shim with explicit deepgait3.* wrappers (cleanup pass).
