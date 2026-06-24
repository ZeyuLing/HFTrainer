# Any2Track G1 Tracker

Any2Track is the canonical paper/table name for the OpenTrack release used by
PhysFlow. The local bundle packages the released Unitree G1 DAgger tracker ONNX,
its config, and the MuJoCo robot assets so evaluation and reward code no longer
depend on `ref_repo/OpenTrack`.

| | |
|---|---|
| **Task** | Unitree G1 motion tracking |
| **Runtime / reward** | `Any2TrackJudgeReward`, `scripts/embodied/eval_opentrack_onnx_mujoco.py` |
| **Native representation** | G1 `qpos`/`qvel` NPZ, MuJoCo body rollout, `robot_frames` JSON |
| **Checkpoint** | `hftrainer/models/motion/physflow/trackers/any2track/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx` |
| **Config** | `.../general_tracker_lafan1_v2/config.json` |
| **Robot assets** | `.../storage/assets/unitree_g1/scene_mjx_flat_terrain.xml`, `.../assets/` |
| **Model Zoo status** | Local release-checkpoint baseline; no Hugging Face artifact |
| **Ref-repo runtime dependency** | No |

## Weights

The packaged `config.json` and `model.onnx` are byte-identical to the local
OpenTrack release copy verified on 2026-06-17:

```text
config.json sha256: 3dac8c821f0b2326fc1644a8d4626e84a27b23cf74762f7d04881bf8b7082c86
model.onnx  sha256: 1320bddd4b981876e84243c0d336c8eaef57e202499b694cfbdc626fab3972c1
```

## Training

The current in-repo Any2Track status is release-checkpoint inference plus
PhysFlow reward/evaluation integration. Full Any2Track retraining has not been
promoted to a native hftrainer trainer. When a table says "Any2Track", it should
use this release checkpoint unless the row explicitly names a PhysFlow
fine-tuned checkpoint.

The upstream release trains a general G1 tracker from retargeted motion
references with DAgger-style data aggregation. PhysFlow currently uses the
checkpoint as:

- a frozen physics judge through
  `hftrainer/models/motion/physflow/any2track_reward.py`;
- a benchmark baseline on AMASS-G1 and LAFAN1-G1;
- a candidate tracker family for future adversarial/co-evolution experiments.

## Inference

Run the dependency-light MuJoCo+ONNX evaluator:

```bash
/root/physflow_isaacgym_py38_cu118/bin/python \
  scripts/embodied/eval_opentrack_onnx_mujoco.py \
  --motion-dir data/LAFAN1_Retargeted_for_G1/UnitreeG1 \
  --xml hftrainer/models/motion/physflow/trackers/any2track/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml \
  --config hftrainer/models/motion/physflow/trackers/any2track/storage/logs/dagger/general_tracker_lafan1_v2/config.json \
  --onnx hftrainer/models/motion/physflow/trackers/any2track/storage/logs/dagger/general_tracker_lafan1_v2/checkpoints/model.onnx \
  --manifest output/physflow_tracker_migration_viz/lafan1_one.json \
  --output-json output/physflow_tracker_migration_viz/any2track_metrics.json \
  --output-csv output/physflow_tracker_migration_viz/any2track_metrics.csv \
  --max-steps 160 \
  --frames-dir output/physflow_tracker_migration_viz/robot_frames \
  --frames-manifest output/physflow_tracker_migration_viz/any2track_manifest.json \
  --frame-stride 2
```

The wrapper matches the OpenTrack play env control loop: residual action is
added to the reference joint target, PD torques are clipped by the released
`TORQUE_LIMIT`, and reference velocities are recalculated from qpos when the
config requests it.

## Evaluation

Quick LAFAN1-G1 benchmark:

```bash
NUM_SHARDS=8 \
bash scripts/embodied/run_lafan1_g1_opentrack_baseline_eval.sh
```

AMASS-G1 benchmark:

```bash
NUM_SHARDS=8 \
bash scripts/embodied/run_amass_g1_opentrack_baseline_eval.sh
```

Primary metrics are success rate, global MPJPE, root-frame MPJPE, MPJVE,
root-height error, joint error, and paper-style success.

## Implementation Checks

Static sanity cases were generated on 2026-06-17 and evaluated with the bundled
checkpoint:

| Motion | Success | MPJPE | local MPJPE | MPJVE | Root-H |
|---|---:|---:|---:|---:|---:|
| `static_default` | 1 | 34.70 mm | 10.43 mm | 0.0161 m/s | 4.29 mm |
| `static_tpose` | 1 | 15.38 mm | 7.42 mm | 0.0099 m/s | 5.00 mm |

This confirms the migrated wrapper can track a static T-pose, but the released
policy still has centimeter-level body motion in static standing. That residual
wobble is separate from migration correctness and should be considered when
using Any2Track as a visual baseline.

The bug fixed during migration review: partial-source G1 trajectories now match
OpenTrack's `TrajectoryHandler.filter_and_extend` behavior by filling missing
source joints with `0.0` rather than the policy `DEFAULT_QPOS`.

Static debug viewer:

```text
http://21.6.58.73:8110/physflow_triplet?manifest=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/any2track_static_debug_zero_fill/view_manifest.json
```

Migration smoke viewer:

```text
http://21.6.58.73:8110/physflow_triplet?manifest=/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/physflow_tracker_migration_viz/manifest.json
```
