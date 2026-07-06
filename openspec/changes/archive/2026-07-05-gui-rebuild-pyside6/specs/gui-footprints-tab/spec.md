## ADDED Requirements

### Requirement: Output directory picker
The Footprints tab SHALL provide an `Output Directory` picker. On
selection, the tab SHALL validate that `cumulative_mask.png`,
`cumulative_intensity.png`, and `cumulative_overlay.png` exist. If any
is missing, the tab SHALL show a non-blocking warning and disable the
viewer.

#### Scenario: User opens an output directory with all three PNGs
- **WHEN** the user selects an output directory produced by a successful run
- **THEN** all three cumulative images are visible in the viewer

### Requirement: Cumulative image viewer
The Footprints tab SHALL display the three cumulative PNGs in a stacked
viewer, each labeled. Aspect ratio MUST be preserved; the viewer SHALL
fit images to the available width.

#### Scenario: Images render at the correct aspect ratio
- **WHEN** the viewer shows `cumulative_overlay.png`
- **THEN** the rendered image is not stretched

### Requirement: Per-frame scrubber
The Footprints tab SHALL list every `per_frame/frame_NNNN_det.png` in
ascending numeric order and SHALL provide a slider or dropdown that
loads the chosen frame into a preview pane.

#### Scenario: User scrubs to frame 0420
- **WHEN** the user moves the slider to position 0420
- **THEN** the preview pane shows `per_frame/frame_0420_det.png`

### Requirement: Stale-output warning
The viewer SHALL freeze and show `Run in progress — viewer disabled` if
the user changes the output directory while a run is in progress.

#### Scenario: Mid-run directory change is rejected
- **WHEN** the worker is running
- **AND** the user picks a new output directory
- **THEN** the viewer shows the frozen banner