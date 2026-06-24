# Humanoid-GPT G1 Tracker

Humanoid-GPT is integrated as a Unitree G1 tracking baseline for PhysFlow. The
released ONNX policy and runtime worker are packaged under hftrainer so reward
and evaluation code can run without importing an upstream Humanoid-GPT checkout.

| | |
|---|---|
| **Task** | Unitree G1 motion tracking |
| **Runtime / reward** | `HgptJudgeReward`, `physflow_hgpt_judge_server.py` |
| **Native representation** | G1 qpos converted to Humanoid-GPT keypoint reference |
| **Checkpoint** | `hftrainer/models/motion/physflow/trackers/humanoid_gpt/storage/ckpts/pns_wo_priv216.onnx` |
| **Worker root** | `hftrainer/models/motion/physflow/trackers/humanoid_gpt/` |
| **Robot assets** | `.../storage/assets/unitree_g1_5010/scene_mjx_track.xml`, `g1_mjx.xml` |
| **Model Zoo status** | Local release-checkpoint baseline; no Hugging Face artifact |
| **Ref-repo runtime dependency** | No |

## Weights

The bundled policy is the released `pns_wo_priv216.onnx` checkpoint. PhysFlow
uses it through a persistent worker process because the Humanoid-GPT stack has
its own py3.11/JAX/MuJoCo dependencies.

Important files:

```text
hftrainer/models/motion/physflow/trackers/humanoid_gpt/physflow_hgpt_judge_server.py
hftrainer/models/motion/physflow/trackers/humanoid_gpt/tracking/
hftrainer/models/motion/physflow/trackers/humanoid_gpt/utils/
hftrainer/models/motion/physflow/trackers/humanoid_gpt/storage/ckpts/pns_wo_priv216.onnx
hftrainer/models/motion/physflow/trackers/humanoid_gpt/storage/assets/unitree_g1_5010/
```

## Training

The current in-repo status is release-checkpoint inference plus
reward/evaluation integration through a packaged worker. PhysFlow does not yet
provide a native hftrainer trainer for Humanoid-GPT from scratch.

For paper baselines, use the released `pns_wo_priv216.onnx` checkpoint unless a
row explicitly names a PhysFlow fine-tuned checkpoint.

The bundled `projects/LIMMT.md` note is relevant for future experiments: it
reports that curated motion subsets can improve multiple tracking systems,
including Any2Track, over full-corpus training.

## Inference

Build a persistent worker environment:

```bash
PHYSFLOW_HGPT_VENV=hftrainer/models/motion/physflow/trackers/humanoid_gpt/.venv311 \
bash scripts/embodied/physflow_hgpt_node_setup.sh
```

Run the worker:

```bash
cd hftrainer/models/motion/physflow/trackers/humanoid_gpt
.venv311/bin/python physflow_hgpt_judge_server.py \
  --load_path storage/ckpts/pns_wo_priv216.onnx \
  --freq 50
```

The worker reads one JSON request per line:

```json
{"job_dir": "/abs/dir/with/qpos_npz", "out": "/abs/metrics.json"}
```

Each input NPZ must contain G1 `qpos` and `frequency`. The worker converts qpos
to the keypoint reference expected by Humanoid-GPT and writes per-clip tracking
metrics.

## Evaluation

The PhysFlow reward adapter is:

```text
hftrainer/models/motion/physflow/hgpt_reward.py
```

Configs can omit these fields when using bundled defaults:

```text
judge_onnx
hgpt_python
```

Set `PHYSFLOW_HGPT_PYTHON` only when the worker venv is built outside the
default bundle path.

Current comparison tables should report success/completion and native
Humanoid-GPT tracking errors, and include the same visual cases used for
ProtoMotions and Any2Track whenever possible.

## Verification

Check files and worker imports:

```bash
python3 scripts/embodied/verify_physflow_tracker_baselines.py \
  --check-hgpt-venv
```

The migration smoke visualization from 2026-06-17 includes Humanoid-GPT in:

```text
output/physflow_tracker_migration_viz/manifest.json
output/physflow_tracker_migration_viz/robot_frames/humanoid_gpt/dance1_subject1.json
```

Viewer:

```text
http://21.6.58.73:8110/physflow_triplet?manifest=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/physflow_tracker_migration_viz/manifest.json
```
