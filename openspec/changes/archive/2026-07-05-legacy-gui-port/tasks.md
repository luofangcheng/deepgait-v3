# Tasks

- [x] 1. Copy deep-gailt `deepgait/` tree to `deepgait3/_legacy_deepgait/`
- [x] 2. Add `deepgait3/legacy.py` alias-package shim
- [x] 3. Rewrite `deepgait3/gui/main_window.py` to hydrate legacy tabs
- [x] 4. Extend `deepgait3/cli.py` with `analyze` + `info` subcommands
- [x] 5. Copy 17 legacy unit-test files to `tests/_legacy_deepgait/unit/`
- [x] 6. Add `tests/_legacy_deepgait/conftest.py` with hardware mocking
- [x] 7. Fix two unit-test collection errors (v0.4.2 schema exposure)
- [x] 8. Validate 96/97 unit tests pass; GUI launches 19 tabs in ~110ms
- [ ] 9. Bridge legacy `SharedState.AppState` and new `AppState` (follow-up)
- [ ] 10. Resolve legacy 11-slot TAB_ORDER vs new 13-slot layout
