# PhysFlow Tracker Baseline Index

This page is an index for the Unitree G1 tracker baselines used by PhysFlow.
Each method has its own model card; do not treat this page as a model card.

| Method | Card | Canonical role |
|---|---|---|
| ProtoMotions G1 Tracker | [protomotions_g1_tracker.md](protomotions_g1_tracker.md) | primary released tracker, frozen judge, fine-tuning base |
| Any2Track G1 Tracker | [any2track_g1_tracker.md](any2track_g1_tracker.md) | released ONNX baseline and MuJoCo reward wrapper |
| Humanoid-GPT G1 Tracker | [humanoid_gpt_g1_tracker.md](humanoid_gpt_g1_tracker.md) | released ONNX baseline through py3.11 worker |

Shared engineering notes, bundle layout, and launch commands live in
[`../physflow/tracker_baselines.md`](../physflow/tracker_baselines.md).

Canonical path constants:

```text
hftrainer/models/motion/physflow/trackers/paths.py
```

Structural verification:

```bash
python3 scripts/embodied/verify_physflow_tracker_baselines.py
```

Tracker-only Any2Track smoke test:

```bash
/root/physflow_isaacgym_py38_cu118/bin/python \
  scripts/embodied/verify_physflow_tracker_baselines.py \
  --skip-hftrainer-imports \
  --any2track-smoke \
  --any2track-max-steps 8
```

Migration smoke viewer:

```text
http://21.6.58.73:8110/physflow_triplet?manifest=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/physflow_tracker_migration_viz/manifest.json
```
