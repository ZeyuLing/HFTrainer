# KIMODO - Kinematic Motion Diffusion

KIMODO is integrated as a native hftrainer runtime under
`hftrainer.models.motion.kimodo.network`. Unlike the T2M-only baselines, the
KIMODO pipeline is multi-task: the same bundle covers plain text generation and
KIMODO's kinematic-control interfaces.

Implementation status: the artifact inference path is now **ref_repo-free** and
does not use `_vendor`, `sys.path` injection, or `os.chdir`. The
SOMA-RP artifact packages the checkpoint snapshot and local text encoders; the
G1 artifacts package their checkpoint snapshots and resolve the shared KIMODO
text encoders from the SOMA-RP artifact on first model load.


| Item                       | Value                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Bundle / Pipeline**      | `KIMODOBundle` / `KIMODOPipeline`                                                                                           |
| **Processed HF artifacts** | `[SOMA-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp)`, `[G1-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp)`, `[G1-SEED](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed)`, `[SMPLX-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-smplx-rp)` (private / license review) |
| **Default model**          | `Kimodo-SOMA-RP-v1`                                                                                                         |
| **Runtime implementation** | `hftrainer.models.motion.kimodo.network` native model / motion representation / skeleton modules                            |
| **Supported skeletons**    | SOMA, Unitree G1, SMPL-X                                                                                                    |
| **Text encoder**           | LLM2Vec / Meta-Llama-3 local text encoder; stored in SOMA-RP and shared by G1 artifacts                                    |
| **SMPL mesh bridge**       | `hftrainer.motion.retarget.KIMODOSOMAToSMPLRetargeter`                                                                      |


## Weights

The SOMA-RP model-zoo artifact is self-contained, including the KIMODO
checkpoint snapshot and text-encoder files needed by KIMODO's local encoder.
The G1 artifacts store their own KIMODO checkpoint snapshots and point
`text_encoders_repo` at the SOMA-RP artifact to avoid duplicating the same
16GB LLM2Vec / Meta-Llama tree in every KIMODO repository.


| Variant        | Processed Hugging Face artifact                                                                 | Native skeleton | Status                                                                          |
| -------------- | ----------------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------- |
| SOMA-RP        | `[ZeyuLing/hftrainer-kimodo-soma-rp](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp)` | SOMA            | packaged for hftrainer, T2M visual path checked; private pending license review |
| G1-RP          | `[ZeyuLing/hftrainer-kimodo-g1-rp](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp)`     | Unitree G1      | packaged for hftrainer; reuses shared text encoders from SOMA-RP                |
| G1-SEED        | `[ZeyuLing/hftrainer-kimodo-g1-seed](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed)` | Unitree G1      | packaged for hftrainer; reuses shared text encoders from SOMA-RP                |
| SMPLX-RP       | `[ZeyuLing/hftrainer-kimodo-smplx-rp](https://huggingface.co/ZeyuLing/hftrainer-kimodo-smplx-rp)` | SMPL-X          | packaged for hftrainer; reuses shared text encoders from SOMA-RP                |


Because the KIMODO artifacts depend on third-party text-encoder weights, their
redistribution status follows the upstream KIMODO, LLM2Vec, and Meta Llama
licenses. The uploaded hftrainer repos are kept private until those
redistribution terms are reviewed. The Meta-Llama text encoder is stored once in
standard Transformers `safetensors` shards inside the SOMA-RP artifact;
duplicate upstream `original/*.pth` exports are intentionally omitted.

## Loading

Load from Hugging Face with the same `from_pretrained` style used by the other
hftrainer model-zoo entries:

```python
from hftrainer.pipelines.motion.kimodo_pipeline import KIMODOPipeline

pipe = KIMODOPipeline.from_pretrained(
    "ZeyuLing/hftrainer-kimodo-soma-rp",
    device="cuda",
)
out = pipe.text_to_motion("a person walks forward.", num_frames=150)
```

Use the matching repo id for Unitree G1 or SMPL-X variants:

```python
pipe = KIMODOPipeline.from_pretrained(
    "ZeyuLing/hftrainer-kimodo-g1-rp",  # or hftrainer-kimodo-g1-seed
    device="cuda",
)
out = pipe.text_to_motion("a humanoid robot walks forward.", num_frames=150)
```

```python
pipe = KIMODOPipeline.from_pretrained(
    "ZeyuLing/hftrainer-kimodo-smplx-rp",
    device="cuda",
)
out = pipe.text_to_motion("a person walks forward.", num_frames=150)
```

For G1 and SMPLX artifacts, `KIMODOBundle.from_pretrained` resolves the shared
`text_encoders/` tree from `ZeyuLing/hftrainer-kimodo-soma-rp` when the model is
loaded. Since these repos are private during license review, run with an
authenticated Hugging Face token that can read the KIMODO hftrainer repos.

The lower-level bundle is still available when you need direct access to
KIMODO runtime outputs:

```python
from hftrainer.models.motion.kimodo import KIMODOBundle
from hftrainer.pipelines.motion.kimodo_pipeline import KIMODOPipeline

bundle = KIMODOBundle.from_pretrained(
    "ZeyuLing/hftrainer-kimodo-soma-rp",
    device="cuda",
)
pipe = KIMODOPipeline(bundle)
```

For environments that need an explicit local snapshot:

```python
from huggingface_hub import snapshot_download
from hftrainer.models.motion.kimodo import KIMODOBundle

path = snapshot_download("ZeyuLing/hftrainer-kimodo-soma-rp")
bundle = KIMODOBundle.from_pretrained(path, device="cuda")
```

Local construction from official NVIDIA checkpoint folders is still available:

```python
from hftrainer.models.motion.kimodo import KIMODOBundle

bundle = KIMODOBundle.from_pretrained(
    "Kimodo-SOMA-RP-v1",
    device="cuda",
    text_encoder_mode="local",
    checkpoint_dir="checkpoints/kimodo/local_models",
    text_encoders_dir="checkpoints/kimodo/text_encoders",
)
```

## Supported Tasks

The wrapper exposes the task surface KIMODO itself supports:


| Task                          | Pipeline API                                              | KIMODO control                           |
| ----------------------------- | --------------------------------------------------------- | ---------------------------------------- |
| Text-to-motion                | `text_to_motion()` / `__call__`                           | prompt only                              |
| Multi-prompt motion           | `multi_prompt()`                                          | segment stitching with transition frames |
| Full-body keyframes           | `fullbody_keyframe_constraint()` + `constrained_motion()` | `FullBodyConstraintSet`                  |
| End-effector control          | `end_effector_constraint()` / hand-foot helpers           | `EndEffectorConstraintSet`               |
| 2D root path / waypoints      | `root2d_constraint()` + `constrained_motion()`            | `Root2DConstraintSet`                    |
| Saved KIMODO demo constraints | `constraints_from_json()` + `constrained_motion()`        | official constraint JSON                 |


Examples:

```python
# Multi-prompt stitching.
out = pipe.multi_prompt(
    ["a person walks forward.", "the person turns left."],
    [90, 90],
)

# Constraint JSON produced by the official KIMODO demos.
constraints = pipe.constraints_from_json("constraints.json", device="cuda")
out = pipe.constrained_motion(
    "a person follows the specified path.",
    num_frames=180,
    constraints=constraints,
)

# Programmatic 2D root waypoints.
root2d = pipe.root2d_constraint(
    frame_indices=[0, 60, 120],
    smooth_root_2d=[[0.0, 0.0], [0.8, 0.1], [1.5, 0.6]],
    device="cuda",
)
out = pipe.constrained_motion(
    "a person walks along a curved path.",
    num_frames=150,
    constraints=[root2d],
)
```

## Retargeting To SMPL

For mesh inspection and HumanML3D-style evaluation, retarget KIMODO SOMA output
through the rotation-aware SOMA-to-SMPL operator:

```python
from hftrainer.motion.retarget import KIMODOSOMAToSMPLRetargeter

retargeter = KIMODOSOMAToSMPLRetargeter(device="cuda")
smpl = retargeter.retarget_file("kimodo_debug_npz/000000.npz")
motion_135 = smpl["motion_135"]
```

The debug NPZ must contain KIMODO `global_rot_mats`. Position-only IK is a
fallback for diagnostics and should not be used as the mesh-quality path.

## Artifact Format

`KIMODOBundle.save_pretrained()` stores a complete runtime artifact:

```text
kimodo_config.json
model_index.json
README.md
kimodo_checkpoint/Kimodo-SOMA-RP-v1/
text_encoders/
```

G1 and SMPLX artifacts use the same format but omit `text_encoders/` locally:

```text
kimodo_config.json
model_index.json
README.md
kimodo_checkpoint/Kimodo-G1-RP-v1/
```

`KIMODOBundle.from_pretrained(path_or_repo)` resolves the artifact-local
`kimodo_checkpoint/` and `text_encoders/` directories into KIMODO's
`CHECKPOINT_DIR` and `TEXT_ENCODERS_DIR` environment hooks before calling the
native hftrainer loader. Artifact checkpoint configs that still contain
`kimodo.*` Hydra targets are rewritten in memory to
`hftrainer.models.motion.kimodo.network.*`, so existing artifacts remain
loadable without repacking.

Package the default SOMA-RP model:

```bash
python3 scripts/eval/convert_kimodo_checkpoint.py \
  --model_name Kimodo-SOMA-RP-v1 \
  --checkpoint_dir checkpoints/kimodo/local_models \
  --text_encoders_dir checkpoints/kimodo/text_encoders \
  --out_dir checkpoints/kimodo/hftrainer_soma_rp \
  --verify
```

Package a G1 checkpoint from a local Hugging Face snapshot:

```bash
python3 scripts/eval/convert_kimodo_checkpoint.py \
  --model_name Kimodo-G1-RP-v1 \
  --checkpoint_source checkpoints/kimodo/hub/models--nvidia--Kimodo-G1-RP-v1/snapshots/<sha> \
  --out_dir checkpoints/kimodo/hftrainer_g1_rp \
  --no_text_encoder \
  --text_encoders_repo ZeyuLing/hftrainer-kimodo-soma-rp \
  --copy_mode hardlink \
  --verify
```

## Evaluation Status

## Rewrite Verification

The runtime rewrite was checked against the previous KIMODO package path with
the same checkpoint, prompt, seed, device, and generated length. To keep the
oracle deterministic and lightweight, the checks use KIMODO's built-in
`text_encoder="dummy"` mode, `diffusion_steps=1`, `num_frames=16`, and
`seed=1234`; this still exercises the checkpoint, denoiser, diffusion sampler,
motion representation, and skeleton export path for each body family.

| Variant | Core output keys | max abs diff |
|---|---|---:|
| SOMA-RP | `local_rot_mats`, `global_rot_mats`, `posed_joints`, `root_positions`, `smooth_root_pos`, `foot_contacts`, `global_root_heading` | 0.0 |
| G1-RP | same | 0.0 |
| G1-SEED | same | 0.0 |
| SMPLX-RP | same | 0.0 |

Current status is intentionally split by task. The trusted HumanML3D T2M numbers
below are from the SMPL-X-RP hftrainer artifact, exported through the
SMPL-X/SOMA-to-SMPL bridge and then scored with the persisted hftrainer
evaluators.

### HumanML3D T2M Metrics (SMPL-X-RP, 2026-06-16)

Prediction/evaluation root:
`outputs/evaluation/kimodo_smplx_hml3d_smpl_ms272_v100x1_20260616`.

| Evaluator | Samples | FID ↓ | R-Precision Top-1 / 2 / 3 ↑ | Diversity → | MM-Dist ↓ | Metric JSON |
|---|---:|---:|---:|---:|---:|---|
| HumanML3D-263 | 2,478 | 1.8425 | 0.3135 / 0.4818 / 0.5925 | 9.1488 | 4.2810 | `metrics/verify_hml263.json` |
| MotionStreamer-272 | 7,392 | 143.9169 | 0.3225 / 0.4601 / 0.5413 | 25.3156 | 21.7065 | `metrics/verify_ms272.json` |

The MotionStreamer-272 score uses the explicit
`KIMODO SMPL-X output -> SMPL motion_135 -> MotionStreamer-272` conversion
chain. The GT(real) sanity row in the same run matches the persisted
MotionStreamer evaluator contract (`R@1=0.706`, `Diversity≈27.36`,
`MM-Dist≈15.01`), so these numbers are suitable for the current model card.

Reproduce the run with the Taiji submit helper:

```bash
python3 scripts/submit/submit_kimodo_hml3d_smpl_ms272_taiji.py \
  --out-root outputs/evaluation/kimodo_smplx_hml3d_smpl_ms272_v100x1_20260616 \
  --feature-namespace kimodo_smplx_t2m_hml3d_smpl_ms272_20260616 \
  --num-jobs 24 \
  --gpus-per-job 1 \
  --no-submit-cache
```

## TP2M HumanML3D Metrics

These rows score KIMODO on the canonical HumanML3D TP2M protocols
`humanml3d_official_test_c1/c5/c9` with the MotionStreamer-272 evaluator
and the selected-caption text directory used by the T2M leaderboard. Each
condition has `ids_with_required_files=4042`; `nb` is the evaluator-consumed
count after the standard min/max motion-length filter.

| Cond frames | nb | FID native | FID refk | R@1 | R@2 | R@3 | MM-Dist | Diversity | Metric JSON |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 3968 | 82.5598 | 85.6194 | 0.5249 | 0.6895 | 0.7692 | 19.3009 | 26.1583 | `outputs/evaluation/tp2m/humanml3d_official_test_c1/ms272/kimodo/metrics/motionstreamer.json` |
| 5 | 3968 | 80.3807 | 83.5092 | 0.5378 | 0.6991 | 0.7749 | 19.1992 | 26.1540 | `outputs/evaluation/tp2m/humanml3d_official_test_c5/ms272/kimodo/metrics/motionstreamer.json` |
| 9 | 3968 | 79.1218 | 82.1577 | 0.5305 | 0.7041 | 0.7717 | 19.1658 | 26.2024 | `outputs/evaluation/tp2m/humanml3d_official_test_c9/ms272/kimodo/metrics/motionstreamer.json` |

Recompute command:

```bash
env RUN_ROOT=outputs/evaluation/tp2m/_runs/ms272_metrics_20260629 \
    GPU_LIST=0,1,2,3,4,5,6,7 SKIP_CACHE=1 \
    TEXT_DIR=outputs/evaluation/t2m/humanml3d_official_test/captions/gt_motionclip_selected_20260622/texts \
    bash scripts/eval/run_tp2m_ms272_metrics_remote.sh
```

Latest Taiji recompute: `tp2m_ms272_metrics_eval272_v100_0629_2124` plus
`tp2m_ms272_metrics_fill3-V100-1x8-2139` for the no-cache fill run.

| Area                                                     | Status                                                                                                      |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| T2M SMPLX-RP                                             | generated with the hftrainer artifact, retargeted to SMPL, and scored with persisted evaluators             |
| HumanML3D / MS272 metrics                                | trusted for SMPLX-RP run above; always copy future values from generated JSON files, not by hand            |
| Multi-prompt / root path / keyframe / inbetween controls | API-supported in `KIMODOPipeline`; visualization must follow the task protocol manifest                     |
| End-effector controls                                    | API-supported; visualization must render exported end-effector target points                                |
| Body-part control                                        | unsupported for KIMODO; do not report forced subset constraints as a valid KIMODO task                      |
| G1-RP / G1-SEED                                          | packaged as hftrainer artifacts; runtime wrapper supports the native Unitree G1 skeleton                    |
| SMPLX-RP                                                 | packaged as an hftrainer artifact; runtime wrapper supports the native SMPL-X skeleton                      |


## Visualization Protocol

KIMODO visualization is defined by reusable hftrainer motion protocols, not by a
one-off web page. The task/panel/frame contracts live in
`hftrainer.motion.visualization.kimodo` and
`hftrainer.motion.visualization.protocol`; the detailed condition families are
documented in `docs/motion/kimodo_visualization_protocols.md`.

KIMODO is not a single T2M viewer case. Each supported task must expose the
generated motion plus the task condition that the model was asked to satisfy.
The current manifest layout is:

```text
outputs/evaluation/<kimodo-viewer-root>/
  _manifest.json              # protocol rows from hftrainer.motion.visualization
  _captions.json
  gt/<case>.npz               # same-timeline reference / condition-source SMPL motion
  condition_smpl/<case>.npz   # generated-timeline condition source, visible only on constrained frames
  kimodo_smpl/<case>.npz      # generated KIMODO retargeted to SMPL
  kimodo_soma/<case>.npz      # generated native SOMA mesh
```

`_manifest.json` includes `frame_semantics`, `condition_overlays`,
`panel_visible_ranges`, `missing_panels`, and `diagnostics`. Missing panels are
source/export limitations, not UI bugs. A `gt` panel is only valid when the
reference motion is finite, nonzero, and on the same frame timeline as the
generated motion. For transition/prepend tasks, use `condition_smpl` plus
`panel_visible_ranges.condition_smpl` instead of fabricating full-timeline GT.

Run the viewer:

```bash
PYTHONPATH=$PWD HFTRAINER_SKIP_AUTOREGISTER=1 \
python3 motion_annot_web/m2m_eval_viewer/retarget_smpl_app.py \
  --host 0.0.0.0 --port 8216 \
  --root outputs/evaluation/kimodo_all_tasks_mesh_viewer_20260615_refactor \
  --model gt=gt=GT=#2ca02c \
  --model condition=condition_smpl=Condition-SMPL=#f59e0b \
  --model soma=kimodo_soma=KIMODO-SOMA=#f97316 \
  --model smpl=kimodo_smpl=KIMODO-SMPL=#9467bd \
  --case-mode union --color-mode condition --list-captions
```

Current live debug URL on the workstation uses port 80:

```text
http://21.6.58.73/
```

Task protocols:


| Task                      | Condition shown in viewer                                                                                                                                                                                                          | Generated output shown                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Text-to-motion            | text prompt                                                                                                                                                                                                                        | full generated motion                                           |
| Full-body keyframes       | exact keyframe indices from `keyframe_indices`                                                                                                                                                                                     | generated inter-keyframe motion                                 |
| Inbetween endpoints       | first and last frames                                                                                                                                                                                                              | generated middle frames                                         |
| Transition stitching      | `condition_smpl` on A-tail/B-head ranges from `layout_json`; panel hidden during the generated transition                                                                                                                          | generated transition in `kimodo_soma` and `kimodo_smpl`         |
| Prepend start pose        | `condition_smpl` on target start pose and conditioned motion-A prefix; panel hidden during the generated prepend transition                                                                                                        | generated prepend transition in `kimodo_soma` and `kimodo_smpl` |
| 2D root path              | `condition_overlays.root_trajectory` rendered only on the primary generated SMPL panel as one clean XZ path rail, start/end dots, a current-frame cursor, and one top-down XZ inset; generated body color remains generated output | generated body motion following the path                        |
| Constraint JSON           | sparse saved KIMODO constraints, shown as every-30-frame markers when metadata is sparse                                                                                                                                           | generated constrained motion                                    |
| Body-part control         | not shown; KIMODO has no native arbitrary body-part mask task                                                                                                                                                                      | unsupported                                                     |
| Multi-prompt / local edit | segment prompt or edit mask                                                                                                                                                                                                        | stitched / edited motion                                        |
| Style edit                | style instruction                                                                                                                                                                                                                  | style-transferred motion                                        |
| End-effector control      | `condition_overlays.joint_targets` rendered only on the primary generated SMPL panel; each frame shows only its active target points as compact colored anchors with a vertical locator line and floor ring                        | generated motion satisfying sparse targets                      |


Condition frames are markers or overlays on the generated sequence, not a
separate stream of motion. If the mesh visibly jumps around a keyframe marker,
that is not an expected viewer-side transition. Check
`diagnostics.continuity` in the manifest and then debug the KIMODO constraint /
retarget export for that sample.

The viewer consumes `_manifest.json` instead of guessing from directory names.
Each panel has an explicit role (`reference`, `generated`, `generated_native`),
and each case has `frame_semantics` so discrete keyframes/endpoints and
continuous-control tasks are displayed differently.

Relevant scripts:

- `scripts/submit/submit_kimodo_hml3d_smpl_ms272_taiji.py` submits the current
KIMODO T2M -> SOMA -> SMPL -> HumanML3D/MS272 path.
- `scripts/analysis/build_kimodo_task_mesh_viewer.py` assembles a compact
task-protocol-aware GT / KIMODO-SMPL / optional KIMODO-SOMA visual sanity
fixture for the supported KIMODO task surface.
- `scripts/submit/submit_kimodo_t2m_eval.py` keeps the older T2M submission
entry point.
- `scripts/kimodo/run_kimodo_all_tasks.py` and `tools/run_kimodo_all_tasks.py`
cover the broader KIMODO task family.
- `scripts/eval/run_kimodo_tp2m_table2.sh` covers prefix-pose + text generation.
- `scripts/eval/run_e10_kimodo_batch.sh` and
`scripts/eval/run_e10_kimodo_h3d500_metrics.sh` cover the E10 bridge path.
