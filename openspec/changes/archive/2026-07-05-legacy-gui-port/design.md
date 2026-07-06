# Design — legacy-gui-port

## Decisions

### D1 — copy / shim instead of rewriting
Copying the deep-gailt source verbatim into
`deepgait3/_legacy_deepgait/` and adding a `sys.modules` shim in
`deepgait3/legacy.py` is significantly cheaper than rewriting all
~50 internal imports + handling the brittle rebind.

Trade-off: namespace duplication.  Two `AppState` classes live in the
same process; main_window resolves the right one via positional
constructor introspection.

### D2 — main_window is the integration seam
Single hand-written file (~110 lines) is the only place that
understands the rule "GUI slot N is filled with legacy class C".
Tab labels live here; legacy tab instantiation is a dry lookup; if
a legacy class is missing, the slot stays empty without raising.

### D3 — preserve Sprint 1+2 API
Even after rewriting, the `DeepGait3Window.__init__(state=None)`,
`state_changed`, `closeEvent`, and `tab_count` API surface stays so
that the existing `tests/gui/` suite does not need to be touched.

## Trade-offs explored
- "Auto-rewrite all imports" — too brittle with Cython (.so) and
  lazy-loaded DLC subprocess paths; rejected.
- "Make legacy deepgait3.legacy package entirely, drop the underscore
  copy" — adds an extra hop at every legacy import; rejected.
