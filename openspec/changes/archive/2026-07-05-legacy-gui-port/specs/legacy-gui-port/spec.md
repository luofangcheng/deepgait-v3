# legacy-gui-port Capability

## Purpose
Make the existing deep-gailt GUI runnable inside deepgait-v3 without a
full rewrite.

## ADDED Requirements

### Requirement: Legacy module exposure
The system SHALL expose the deep-gailt `deepgait` package under two
paths inside `deepgait3`:
- `deepgait3._legacy_deepgait.*` — direct copy of the tree.
- `deepgait3.legacy.deepgait.*` — alias package that resolves to the
  same module objects.

#### Scenario: existing imports keep working
- GIVEN source code copied from deep-gailt that does
  `from deepgait.core import footprint`
- WHEN the package is imported after the alias is installed
- THEN the import resolves to a module object identical to
  `deepgait3._legacy_deepgait.core.footprint`.

### Requirement: Main window hydrates legacy tabs
The system SHALL instantiate the 19 tabs declared in
`DeepGait3Window` from the legacy `_legacy_deepgait.gui` package
without source rewrites.

#### Scenario: GUI launches with 19 tabs in under 3 seconds
- GIVEN offscreen QPA platform
- WHEN `DeepGait3Window()` is shown and `app.exec()` returns
- THEN the tab count is at least 13 and cold-start time is below 3 s.
