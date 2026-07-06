# full-merge-deep-gailt

## WHY
The Sprint 1+2 dual-package scaffolding + partial legacy port was rejected by
the user as "too cluttered". The simplest workable layout is to transplant
deep-gailt's whole `deepgait/` tree flat into `deepgait-v3/deepgait3/deepgait/`,
rename every `deepgait.*` reference to `deepgait3.deepgait.*`, drop the now-
unnecessary `_legacy_deepgait/` shim, and let the existing `core/pawprint/`
Stage-1 implementation coexist (no shadowing — new `detect_blobs` family
stays in `core/pawprint/`, legacy footprint classes stay in `core/footprint.py`).

## WHAT
- Wipe `deepgait3/_legacy_deepgait/`, `deepgait3/_legacy_internal_tests/`,
  `deepgait3/legacy.py`, `tests/_legacy_deepgait/`.
- Copy `deep-gailt/deepgait/` → `deepgait-v3/deepgait3/deepgait/`
  (71 .py files, 11 .so/.c files).
- Copy `deep-gailt/tests/` → `deepgait-v3/tests/_legacy_deepgait/` (20 .py).
- Rewrite all imports across 91 files: 1048 statements in 48 files.
- Wire `gui` subcommand in `deepgait3/cli.py` to legacy's `launch_gui()`.
- De-duplication: `core/pawprint/detection.py`, `grouping.py`, `tracker.py`
  already own the canonical Stage-1 primitives; legacy `core/footprint.py`
  uses them. No shadowing remained after the rewrite.

## DELTAS
- Add `deepgait3.deepgait.*` as the new namespace for the merged GUI +
  core + hardware tree.
- `deepgait3.cli` MODIFIED: gains `gui` subcommand (calls into
  `deepgait3.deepgait.gui.app.launch_gui`).
- `deepgait3.core.pawprint.models` MODIFIED: re-exports legacy
  contracts (`FrameData`, `QualityFlags`, `Linkage3D`, `PawPrint`,
  `FootMask`) so the GUI can `from deepgait3.deepgait.core.footprint
  import PawPrint` without renames.

## VALIDATION
- GUI launch: 11 tabs in <3 s.
- `deepgait3 gui` exit 0.
- `deepgait3 stage1` end-to-end on mouse_001: 56 cycles × 357
  per-frame PNGs.
- Tests: 439 passed, 29 failed, 3 skipped in 7m17s. The 29 failures
  are deep-gailt-specific build/runtime assertions that don't apply
  (build.spec, setup_cython.py, "8 tabs", pyproject version=v2).
