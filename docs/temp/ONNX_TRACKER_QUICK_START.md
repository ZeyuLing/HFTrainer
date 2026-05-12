# ONNX Tracker / MuJoCo Simulation — Quick Start

## 3-Minute Summary

**What is it?** A neural network policy (ONNX) that drives a robot to track motion in physics simulation.

**Input:** `.pt` motion cache file (33 bodies, 29 DOFs, 50 Hz)
**Output:** Simulated motion tracking (rendered frames or extractable trajectory)

---

## Commands (Copy-Paste Ready)

### Render Reference Motion (ideal/direct from cache)
```bash
python scripts/embodied/render_tracker_headless.py \
    --motion data/embodied_debug/robot_cache.pt \
    --output-dir /tmp/render_ref \
    --mode reference \
    --video
```
→ Output: `/tmp/render_ref/output.mp4`

### Render Tracked Motion (ONNX policy + physics sim)
```bash
python scripts/embodied/render_tracker_headless.py \
    --motion data/embodied_debug/robot_cache.pt \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    --output-dir /tmp/render_tracked \
    --mode tracked \
    --video
```
→ Output: `/tmp/render_tracked/output.mp4`

### Interactive Viewer Test
```bash
python ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    --motion data/embodied_debug/robot_cache.pt \
    --render \
    --loops 1 \
    --no-realtime
```
→ Opens MuJoCo viewer window; shows tracking in real-time

### Full Pipeline with Validation
```bash
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/experiments/hymotion_v3/eval_output/00000.npz \
    --output /tmp/g1_motion.pt \
    --validate
```
→ Converts HyMotion → SMPL-X → GMR → Robot cache, then validates with ONNX

---

## Key Paths

| What | Path |
|------|------|
| ONNX Model | `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx` |
| YAML Metadata | `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.yaml` |
| G1 MJCF | `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml` |
| Render Script | `scripts/embodied/render_tracker_headless.py` |
| Test Script | `ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py` |
| Pipeline | `scripts/embodied/pipeline_motion_to_robot.py` |

---

## How It Works (30 seconds)

1. **Load motion cache** → MotionPlayer (resampled @ 50 Hz)
2. **Load robot model** → MuJoCo + configure physics
3. **Loop over frames:**
   - Read robot state (joint pos/vel, body rotations)
   - Query 4 future motion reference frames
   - Run ONNX → get PD joint targets
   - Apply targets to MuJoCo
   - Step physics 20 substeps
   - Render frame (optional)
4. **Output** → PNG frames → MP4 (ffmpeg)

**Note:** Simulated state is **not automatically saved**. See *Extracting Output* below to export.

---

## Input/Output Formats

### Input: Motion Cache (.pt)
```python
{
    "dof_pos":      ndarray [T, 29]        # Joint angles
    "dof_vel":      ndarray [T, 29]        # Joint velocities
    "body_rot":     ndarray [T, 33, 4]     # Body quaternions (xyzw)
    "body_pos":     ndarray [T, 33, 3]     # Body positions (FK)
    "body_vel":     ndarray [T, 33, 3]     # Body velocities
    "body_ang_vel": ndarray [T, 33, 3]     # Body angular velocities
    "control_dt":   float                  # 0.02 (50 Hz)
    "num_frames":   int                    # Total frames
}
```

### ONNX Inputs (unified_pipeline.onnx)
- `current_dof_pos` [1, 29] — robot joints now
- `current_dof_vel` [1, 29] — robot velocities now
- `current_anchor_rot` [1, 4] — torso orientation (xyzw)
- `current_root_local_ang_vel` [1, 3] — pelvis angular velocity (body frame)
- `historical_processed_actions` [1, 1, 29] — last actions (smoothness)
- `mimic_future_dof_pos` [1, 4, 29] — next 4 frames' joint targets
- `mimic_future_dof_vel` [1, 4, 29] — next 4 frames' joint velocities
- `mimic_future_anchor_rot` [1, 4, 4] — next 4 frames' torso orientation

### ONNX Outputs
- `joint_pos_targets` [1, 29] — **PD targets** (main output)
- `actions` [1, 29] — raw network output (before PD transform)
- `stiffness_targets`, `damping_targets` — per-joint gains (pre-configured)

---

## Robot Spec (G1)

| Property | Value |
|----------|-------|
| Bodies | 33 (pelvis, head, limbs, hand, etc.) |
| DOFs (joints) | 29 |
| Control rate | 50 Hz (dt=0.02s) |
| Physics rate | 1000 Hz (dt=0.001s, 20 substeps/control) |
| Anchor body | torso_link (body index 16) for IMU obs |
| Root body | pelvis (body index 0) for base state |

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| `FileNotFoundError: ONNX not found` | Check path: `.../g1-bones-deploy/compiled_models/unified_pipeline.onnx` |
| `ONNX YAML not found` | `.yaml` must be alongside `.onnx` in same directory |
| High tracking error (>0.1 rad) | Motion quality issue or physics mismatch |
| Slow rendering | Use `--skip-frames 5` to render every 5th frame |
| ImportError with onnxruntime | `pip install onnxruntime` |

---

## Extracting Simulated Output

The scripts render but don't auto-save trajectory. To export simulated state to a new `.pt`:

```python
# Pseudo-code (add to render_tracker_headless.py in the loop)
trajectory = {"dof_pos": [], "dof_vel": [], "body_rot": [], ...}

for frame_idx in range(num_frames):
    # ... ONNX + MuJoCo step ...
    
    # Collect state
    trajectory["dof_pos"].append(data.qpos[7:].copy())
    trajectory["dof_vel"].append(data.qvel[6:].copy())
    body_rot_wxyz = data.xquat[1:].copy()
    body_rot_xyzw = mujoco_wxyz_to_xyzw(body_rot_wxyz)
    trajectory["body_rot"].append(body_rot_xyzw)
    trajectory["body_pos"].append(data.xpos[1:].copy())
    trajectory["body_vel"].append(data.cvel[1:, 3:].copy())
    trajectory["body_ang_vel"].append(data.cvel[1:, :3].copy())

# Save
import torch
import numpy as np
for key in trajectory:
    trajectory[key] = np.stack(trajectory[key])
trajectory["control_dt"] = 0.02
trajectory["num_frames"] = len(trajectory["dof_pos"])
torch.save(trajectory, "simulated_output.pt")
```

---

## Documentation Files

- **Full Guide**: `ONNX_TRACKER_GUIDE.md` (this repo)
- **ProtoMotions Docs**: `ref_repo/ProtoMotions/CLAUDE.md`
- **Deployment Contract**: `ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py` (lines 16–130)

---

## Key Hyperparameters

From `unified_pipeline.yaml`:

```yaml
timing:
  control_dt: 0.02          # Control updates @ 50 Hz
  physics_dt: 0.001         # Physics @ 1 kHz
  decimation: 20            # 20 substeps per control

control:
  stiffness: [40.18, 99.10, ...]  # Per-joint kp
  damping: [2.56, 6.31, ...]      # Per-joint kd
  action_ema_alpha: 1.0           # No filtering
  pd_target_max_accel: null       # No accel clamp

motion:
  future_step_indices: [1, 2, 4, 8]  # Look ahead 20, 40, 80, 160 ms
```

---

## Reference

- **Render script**: `scripts/embodied/render_tracker_headless.py` (~50 lines per mode)
- **Test script**: `ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py` (main loop ~260 lines)
- **State utils**: `ref_repo/ProtoMotions/deployment/state_utils.py` (quaternion conversions)
- **Motion player**: `ref_repo/ProtoMotions/deployment/motion_utils.py` (cache format spec)

