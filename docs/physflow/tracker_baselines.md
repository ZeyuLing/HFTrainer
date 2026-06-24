# PhysFlow Tracker Baseline Bundle

This document is the canonical entrypoint for the three PhysFlow tracking
baselines used in generator reward, tracker benchmark, and co-evolution
experiments:

- ProtoMotions G1 released tracker
- Any2Track G1 general tracker
- Humanoid-GPT G1 zero-shot tracker

The runtime code, checkpoints, robot assets, and baseline configs are now
vendored inside this repository. New experiments should not depend on
`ref_repo/ProtoMotions`, `ref_repo/OpenTrack`, or `ref_repo/Humanoid-GPT`.

Each method has its own Model Zoo card with artifact layout, training/inference
recipes, evaluation protocol, and migration verification records:

- [`../model_zoo/protomotions_g1_tracker.md`](../model_zoo/protomotions_g1_tracker.md)
- [`../model_zoo/any2track_g1_tracker.md`](../model_zoo/any2track_g1_tracker.md)
- [`../model_zoo/humanoid_gpt_g1_tracker.md`](../model_zoo/humanoid_gpt_g1_tracker.md)

The aggregate Model Zoo index is
[`../model_zoo/physflow_tracker_baselines.md`](../model_zoo/physflow_tracker_baselines.md).

## Layout

Canonical paths are defined in:

```text
hftrainer/models/motion/physflow/trackers/paths.py
```

Bundle locations:

```text
hftrainer/models/motion/physflow/trackers/protomotions/vendor/
hftrainer/models/motion/physflow/trackers/any2track/
hftrainer/models/motion/physflow/trackers/humanoid_gpt/
```

Dataset locations used by the bundled benchmark scripts:

```text
data/AMASS_Retarged_for_G1/g1/
data/LAFAN1_Retargeted_for_G1/UnitreeG1/
```

## Verify The Bundle

Run the path and import checks:

```bash
python3 scripts/embodied/verify_physflow_tracker_baselines.py
```

Run a tiny Any2Track MuJoCo+ONNX rollout smoke test:

```bash
python3 scripts/embodied/verify_physflow_tracker_baselines.py \
  --any2track-smoke \
  --any2track-max-steps 8
```

If running the smoke test in the IsaacGym py3.8 tracker environment, skip the
hftrainer reward-adapter imports because that environment does not include
`mmengine`:

```bash
/root/physflow_isaacgym_py38_cu118/bin/python \
  scripts/embodied/verify_physflow_tracker_baselines.py \
  --skip-hftrainer-imports \
  --any2track-smoke \
  --any2track-max-steps 8
```

Check the Humanoid-GPT py3.11 worker environment after building it:

```bash
python3 scripts/embodied/verify_physflow_tracker_baselines.py \
  --check-hgpt-venv
```

The verifier fails if a required default path points outside this repository or
contains `ref_repo`.

## ProtoMotions

Bundled assets include:

```text
.../protomotions/vendor/protomotions/
.../protomotions/vendor/deployment/
.../protomotions/vendor/examples/
.../protomotions/vendor/data/scripts/
.../protomotions/vendor/data/pretrained_models/motion_tracker/g1-bones-deploy/
.../protomotions/vendor/protomotions/data/assets/{mjcf,urdf,mesh}/
```

Important files:

```text
.../g1-bones-deploy/last.ckpt
.../g1-bones-deploy/compiled_models/unified_pipeline.onnx
.../g1-bones-deploy/compiled_models/unified_pipeline.yaml
.../g1-bones-deploy/experiment_config.py
```

AMASS-G1 benchmark:

```bash
RUN_NODE_SETUP=1 \
NUM_SHARDS=8 \
NUM_ENVS=256 \
bash scripts/embodied/run_amass_g1_proto_baseline_eval.sh
```

LAFAN1-G1 benchmark:

```bash
RUN_NODE_SETUP=1 \
NUM_SHARDS=8 \
NUM_ENVS=64 \
bash scripts/embodied/run_lafan1_g1_proto_baseline_eval.sh
```

GT replay tracker training and before/after evaluation:

```bash
RUN_TAG=amass_sanity \
NGPU=1 \
RUN_EVAL=1 \
bash scripts/embodied/run_gt_replay_tracker_train_eval.sh
```

The ProtoMotions training/eval path still needs an IsaacGym-capable Python
environment. The standard Taiji entrypoints restore it from
`/apdcephfs_cq11/share_1467498/home/zeyuling/physflow_env` through
`scripts/embodied/cursor_physflow_taiji_node_setup.sh`. That environment is
driver and node dependent; the source code, configs, checkpoints, and robot
assets are in-repo.

## Any2Track

Bundled assets include:

```text
.../any2track/storage/logs/dagger/general_tracker_lafan1_v2/config.json
.../any2track/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx
.../any2track/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml
.../any2track/storage/assets/unitree_g1/assets/
```

LAFAN1-G1 benchmark:

```bash
NUM_SHARDS=8 \
bash scripts/embodied/run_lafan1_g1_opentrack_baseline_eval.sh
```

PhysFlow reward configs now use bundled defaults, so these fields are no longer
needed in config files:

```text
judge_onnx
judge_mjcf
any2track_config
```

The reward adapter is:

```text
hftrainer/models/motion/physflow/any2track_reward.py
```

## Humanoid-GPT

Bundled assets include:

```text
.../humanoid_gpt/physflow_hgpt_judge_server.py
.../humanoid_gpt/tracking/
.../humanoid_gpt/utils/
.../humanoid_gpt/deploy/
.../humanoid_gpt/projects/
.../humanoid_gpt/storage/ckpts/pns_wo_priv216.onnx
.../humanoid_gpt/storage/assets/unitree_g1_5010/
```

Build a fast ephemeral py3.11 worker venv on a Taiji node:

```bash
bash scripts/embodied/physflow_hgpt_node_setup.sh
export PHYSFLOW_HGPT_PYTHON=/dev/shm/hgpt_venv311/bin/python
```

Build a persistent in-repo venv for local validation:

```bash
PHYSFLOW_HGPT_VENV=hftrainer/models/motion/physflow/trackers/humanoid_gpt/.venv311 \
  bash scripts/embodied/physflow_hgpt_node_setup.sh
```

The reward adapter is:

```text
hftrainer/models/motion/physflow/hgpt_reward.py
```

Configs can omit these fields and use bundled defaults:

```text
judge_onnx
hgpt_python
```

Set `PHYSFLOW_HGPT_PYTHON` only when the venv is built somewhere other than the
bundle default.

## Co-Evolution Entrypoints

The current co-evolution launchers use the bundled ProtoMotions paths:

```bash
bash scripts/embodied/physflow_coevo_formal_node.sh
bash scripts/embodied/physflow_coevo_frontier_node.sh
```

The Python orchestrator also resolves the released frozen judge from the bundle:

```bash
python3 scripts/embodied/physflow_coevolve_orchestrator.py --help
```

## Notes

- Historical debug scripts under `scripts/embodied/` may still contain old
  `ref_repo` paths. They are not canonical experiment entrypoints.
- New model/reward/pipeline code should import paths from
  `hftrainer.models.motion.physflow.trackers.paths` instead of spelling out
  filesystem layouts.
- If a reproduction script needs data not already in `data/`, pass it explicitly
  through `AMASS_ROOT` or `LAFAN_ROOT` rather than using `ref_repo`.
