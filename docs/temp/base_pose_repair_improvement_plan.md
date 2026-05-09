# Base Pose Repair Demo Import And Improvement Plan

## Current State

- Demo viewer: `motion_annot_web/keypose_eval/app.py`
- Demo data: `output/eval_keyframe_pose_v3/local_rot/`
- Imported score site: `motion_annot_web/score_m2m/score_m2m.db`
- Visible task after whitelist update: `base_pose_edit / keypose_demo_v3`
- Pair cases: 155
- Pair A: `PureBlend`
- Pair B: `HyMotion-M2M-HybridPolish`
- Running site: `score_m2m_web.py --port 8083`

The existing best method is `Hybrid Blend + Boundary Polish`:

1. Propagate the keypose correction through pure blending.
2. Run M2M only on narrow boundary bands where the blend weight drops to zero.
3. Preserve translation and force exact keypose rotations.

This works when the target base pose is close to the original key pose, but its main failure mode is structural: the global edit is still produced by a fixed correction field. If the base pose differs strongly from the original key pose, the blend can create non-human intermediate poses before the model sees the result. Boundary polishing cannot fix errors that are already spread through the whole edited region.

## Proposed Direction

### 1. Confidence-Gated Hybrid

Keep pure blend only for small or locally compatible edits. For each selected keypose, compute:

- `delta_norm`: target pose change magnitude.
- `peak_ratio`: max correction / mean correction over frames.
- `pose_neighbor_consistency`: whether nearby frames have similar body pose.
- optional FK-space end-effector displacement for hands/feet.

If the edit is small, use current `HybridPolish`. If the edit is large, switch to model-centric local regeneration.

Expected benefit: keeps the current strong baseline for easy cases while avoiding blend artifacts for hard cases.

### 2. Anchor-Window Regeneration

For hard cases, mask a wider window around the keypose and regenerate the whole local motion segment with MAN `flow_interp` replacement guidance:

- observed: window boundaries + exact base pose frame.
- generated: frames inside the window except the base pose frame.
- postprocess: only preserve global translation if needed; avoid pose-level hard blending inside the window.

This reuses the train-consistent imputation path in `_man` models instead of relying on a hand-designed correction propagation field.

Initial sweep:

- window radius: `24, 40, 60`
- replacement guidance: `flow_interp`
- model: `uncond_fm_man`, optionally `uncond_fm_man_globalrot`

Primary metrics:

- `kf_mpjpe`
- `global_mpjpe`
- `boundary_smoothness`
- `overall_smoothness`
- `foot_skating`

Hard-case slice:

- cases with top 25% `correction_diffs`
- cases where PureBlend improves keypose but hurts smoothness or support stability

### 3. Two-Stage Model Polish Without Full Blend

For very large edits, use a coarse-to-fine path:

1. Coarse anchor-window regeneration with wide mask.
2. Boundary-only polish on the regenerated segment boundaries.
3. Optional small residual blend only on the keypose frame neighborhood, not across the full motion.

This avoids using blend as the main edit mechanism while retaining the useful boundary smoother.

## Taiji Validation Plan

Use the two `lzy_debug_machine_x` machines as parallel experiment runners:

### Machine 1: Quick Ablation

Run 20-40 cases across parameter sweeps:

```bash
python3 scripts/run_keypose_imputation.py \
  --model uncond_fm_man \
  --num-cases 40 \
  --num-steps 50
```

Add the new confidence-gated / anchor-window variants before running the sweep. Compare against:

- `pure_blend`
- `hybrid_blend_boundary_polish`
- existing `local_edit_flow_interp`
- existing `anchor_inbetween_flow_interp`

### Machine 2: Full Inference

After selecting the best quick setting, run all 155 cases and write outputs under:

```text
output/eval_keyframe_pose_v3/local_rot/<new_variant_name>/
```

Then extend the score import script to include the new variant as a third model or pair it against the current `HybridPolish`.

## Import Workflow For 8083

Current import script:

```bash
python3 motion_annot_web/score_m2m/scripts_oneoff_import_base_pose_keypose_demo.py
```

For a new variant, create a sibling import script or generalize the current one so that:

- `score_tasks.task_id = base_pose_edit`
- `score_tasks.setting = keypose_demo_v3`
- `score_tasks.model_name = <new_variant_name>`
- `score_tasks.gen_motion_path` points to the original NPZ containing `output_motion`
- `pair_cases` pairs the new variant against the selected baseline

After changing visible tasks/models, restart:

```bash
cd motion_annot_web/score_m2m
SCORE_M2M_PREWARM_IMPORTS=0 python3 -u score_m2m_web.py --port 8083
```

## Notes

- Do not judge only by averaged metrics. Large base-pose edits can improve `global_mpjpe` while visibly changing unintended limbs, so keep the 8083 human pair review in the loop.
- Keep `flow_interp` as the first MAN guidance choice because it matches the mask-aware flow training path better than plain `skip_last`.
- Avoid adding more post-hoc full-motion blend as the core fix; use blend as a fallback or local residual only.
