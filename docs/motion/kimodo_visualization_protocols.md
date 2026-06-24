# KIMODO Visualization Protocols

This document defines reusable visualization contracts for KIMODO-style motion
conditions. Exporters should write protocol data into the manifest; viewers and
offline renderers should consume these fields instead of hard-coding task names.

## Condition Protocols

| Condition family | KIMODO tasks | Manifest fields | Visual encoding | Correctness check | KIMODO status |
|---|---|---|---|---|---|
| Text prompt | E1 text-to-motion | `frame_semantics.mode=text_only` | No spatial overlay; whole motion is generated output | Generated motion is finite and uses the requested prompt | Supported |
| Sparse full-body frame | E2 inbetween, E3 full-body keyframe, E7 first-frame, E8 loop endpoint | `frame_semantics.condition_frames`; optional same-timeline `gt` panel | Condition frames are highlighted; GT panel is shown only if the source timeline exactly matches output length | At condition frames, constrained global positions should match the SOMA targets after model-space canonicalization and inverse transform | Supported with repaint/imputation enabled |
| Continuous condition range | E14 transition stitching, E15 prepend start pose | `frame_semantics.condition_ranges`; `layout_json`; optional `condition_motion_135`; `panel_visible_ranges.condition_smpl` | Show a `Condition SMPL` panel only on condition ranges, and hide it on generated spans; generated SOMA/SMPL remain visible for the whole output | Boundaries must use the exported layout, not frontend heuristics; source clips with mismatched timelines must not be shown as fake GT | Supported with repaint/imputation and task-specific layout export |
| End-effector position target | E4 end-effector | `condition_overlays.joint_targets` with `frame`, `joint_index`, `joint_name`, `position` | On the primary generated SMPL panel only, show the current frame's target points as compact colored anchors with a vertical locator line and floor ring; hide targets from other frames | Target error should be reported against SOMA constraint joints, not raw SMPL joints with different proportions | Supported, but sparse targets can still leave neighboring generated frames unconstrained |
| Root trajectory | E5 root2d | `condition_overlays.root_trajectory.points`; `frame_semantics.condition_ranges` | On the primary generated SMPL panel only, show one clean XZ path rail with start/end dots, a current-frame cursor, and one top-down XZ inset; keep the generated body colored as generated output | Generated root XZ should track the target polyline at constrained frames | Supported for XZ only; XYZ trajectory is not a KIMODO Root2D task |
| Foot contact range | E6 foot contact | `frame_semantics.condition_ranges` derived from source/GT foot contact frames | Highlight contact ranges; optional foot target markers when exported | Contact frames should be detected in SOMA space, because the constraint is applied in SOMA space | Supported |
| Body-part rotation mask | E10 part-level control | Unsupported marker, no generated panel by default | Do not show forced subset constraints as a valid KIMODO result | KIMODO has no native arbitrary body-part rotation mask; forced subset constraints produce invalid meshes | Unsupported |
| Body-part position mask | None in KIMODO | Future `condition_overlays.bodypart_targets` or per-joint target sets | Draw selected joint targets and mask labels | Requires a native model/task definition before evaluation | Unsupported |

## Export Rules

- A `gt` panel must only be exported when the reference motion is valid and has
  the same frame count as the generated output.
- Transition/prepend tasks may export `condition_motion_135` as a generated-
  timeline condition source. Viewers must pair it with
  `panel_visible_ranges.condition_smpl` so it is visible only where the model was
  constrained.
- Transition and prepend tasks should export their explicit layout metadata
  (`N_cond_a`, `N_transition`, `N_cond_b`, etc.) so viewers can derive ranges.
- Spatial overlays should live under `condition_overlays` and use world-space
  coordinates matching the displayed motion panels.
- Spatial overlays should appear only on the primary generated panel. Reference,
  SOMA, and condition-source panels should stay visually quiet so reviewers can
  compare motion quality without duplicate condition graphics.
- KIMODO constrained inference should use in-loop repaint/imputation for
  constrained channels and should not rely on final hard compositing.
