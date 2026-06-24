# ProtoMotions G1 Tracker

Unitree G1 physics-tracking baseline integrated for PhysFlow generator rewards,
tracker benchmarks, visual comparisons, and tracker fine-tuning experiments.
The canonical reward path is ref-repo independent: the evaluation adapter,
checkpoint, ONNX export, config, and robot assets are packaged inside this
repository.

| | |
|---|---|
| **Task** | Unitree G1 motion tracking |
| **Runtime / reward** | `PhysicsJudgeReward` and ProtoMotions benchmark scripts |
| **Native representation** | ProtoMotions motion library, G1 qpos, MuJoCo/IsaacGym rollout states, `robot_frames` JSON |
| **Checkpoint** | `hftrainer/models/motion/physflow/trackers/protomotions/vendor/data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt` |
| **Deployment export** | `.../compiled_models/unified_pipeline.onnx`, `.../compiled_models/unified_pipeline.yaml` |
| **Robot assets** | `.../protomotions/data/assets/mjcf/g1_holo_compat.xml`, `.../urdf/for_retargeting/g1.urdf`, `.../mesh/G1/` |
| **Model Zoo status** | Local research baseline bundle; no Hugging Face artifact |
| **Ref-repo runtime dependency** | No |

## Weights

The bundled release checkpoint is the G1 `g1-bones-deploy` tracker from the
ProtoMotions release. PhysFlow uses it in two modes:

| Mode | Artifact | Use |
|---|---|---|
| Full tracker checkpoint | `last.ckpt` | IsaacGym/Lightning evaluation and tracker fine-tuning |
| ONNX deployment | `compiled_models/unified_pipeline.onnx` | quick rollout export and frozen reward path |
| Experiment config | `experiment_config.py` | resume/fine-tune settings for G1 tracking |

Canonical path constants live in
`hftrainer.models.motion.physflow.trackers.paths`.

## Training

The release checkpoint is used as the starting point for all local ProtoMotions
tracker experiments. Fine-tuning resumes from `last.ckpt` with G1 motion
libraries built from AMASS, HYMotion, KIMODO, or adversarially selected PhysFlow
outputs.

```bash
RUN_TAG=amass_sanity \
NGPU=1 \
RUN_EVAL=1 \
bash scripts/embodied/run_gt_replay_tracker_train_eval.sh
```

Large benchmark/eval launches use the IsaacGym py3.8 environment and shard over
GPUs:

```bash
RUN_NODE_SETUP=1 NUM_SHARDS=8 NUM_ENVS=256 \
bash scripts/embodied/run_amass_g1_proto_baseline_eval.sh

RUN_NODE_SETUP=1 NUM_SHARDS=8 NUM_ENVS=64 \
bash scripts/embodied/run_lafan1_g1_proto_baseline_eval.sh
```

On current nodes the known working Python is
`/root/physflow_isaacgym_py38_cu118/bin/python`.

## Inference

Use the ONNX deployment when a lightweight MuJoCo visual export is enough:

```bash
/root/physflow_isaacgym_py38_cu118/bin/python \
  scripts/embodied/run_g1_rl_tracker_export.py \
  --input data/LAFAN1_Retargeted_for_G1/UnitreeG1/dance1_subject1.npz \
  --output-dir output/physflow_tracker_migration_viz/robot_frames/protomotions \
  --onnx hftrainer/models/motion/physflow/trackers/protomotions/vendor/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
  --mjcf hftrainer/models/motion/physflow/trackers/protomotions/vendor/protomotions/data/assets/mjcf/g1_holo_compat.xml \
  --skip-retarget \
  --motion-dir output/physflow_tracker_migration_viz/protomotions_motion_clip \
  --subsample 2
```

Use the IsaacGym evaluation path when comparing training checkpoints or
reporting benchmark metrics.

## Evaluation

Current shared benchmark roots:

```text
data/AMASS_Retarged_for_G1/g1/
data/LAFAN1_Retargeted_for_G1/UnitreeG1/
```

Reported PhysFlow tables should include success/completion, global and
root-frame MPJPE, MPJVE, root-height error, and failure rate using the same
sample set as Any2Track and Humanoid-GPT whenever possible.

## Verification

Run the structural verifier:

```bash
python3 scripts/embodied/verify_physflow_tracker_baselines.py
```

The migration smoke visualization from 2026-06-17 includes ProtoMotions in:

```text
output/physflow_tracker_migration_viz/manifest.json
output/physflow_tracker_migration_viz/robot_frames/protomotions/dance1_subject1.json
```

Viewer:

```text
http://21.6.58.73:8110/physflow_triplet?manifest=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/physflow_tracker_migration_viz/manifest.json
```
