# Running the ONNX Tracker / MuJoCo Physics Simulation on ProtoMotions Cache Files

## Quick Summary

The ONNX tracker is a **motion tracking/imitation policy** that runs in MuJoCo simulation. It reads a motion reference (`.pt` cache file) and uses an ONNX neural network policy to generate PD joint position targets that make the robot track the motion. The simulation produces **dynamic (physics-based) tracked motion** output.

---

## Key Concepts

### Motion Cache Format (.pt)
The input motion cache is a **PyTorch-saved dict** with these keys:
```python
{
    "dof_pos":      np.ndarray [T, 29]           # Joint positions
    "dof_vel":      np.ndarray [T, 29]           # Joint velocities
    "body_rot":     np.ndarray [T, 33, 4]        # Body orientations (xyzw quaternion)
    "body_pos":     np.ndarray [T, 33, 3]        # Body positions (FK forward kinematics)
    "body_vel":     np.ndarray [T, 33, 3]        # Body linear velocities
    "body_ang_vel": np.ndarray [T, 33, 3]        # Body angular velocities
    "control_dt":   float                        # Timestep (typically 0.02s = 50Hz)
    "num_frames":   int                          # Total frames
}
```

### ONNX Model
- **Location**: `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx`
- **Metadata**: `unified_pipeline.yaml` (alongside the .onnx file)
- **Inputs**: Current robot state (DOF pos/vel, body rotations, angular velocity) + future motion references (4 steps ahead)
- **Outputs**: PD joint position targets, stiffness, damping (pre-configured)
- **Robot**: G1 humanoid (33 bodies, 29 DOFs)
- **Timestep**: physics_dt=0.001s, decimation=20, control_dt=0.02s

---

## How It Works (High Level)

### Phase 1: Initialization
1. Load motion cache → MotionPlayer object
2. Load G1 MJCF XML model → MuJoCo model
3. Set robot to first frame of motion (qpos, qvel = 0)

### Phase 2: Control Loop (per 50Hz control step)
```
For each frame t:
  1. Read current robot state from MuJoCo data buffers:
     - Joint positions: data.qpos[7:]      (skip 3 pos + 4 quat free-joint)
     - Joint velocities: data.qvel[6:]     (skip 6-DOF free-joint)
     - Body orientations: data.xquat[1:]   (quaternions in wxyz format)
  
  2. Derive needed quantities:
     - anchor_rot: torso quaternion (for IMU-like observation)
     - root_local_ang_vel: pelvis angular velocity in body-local frame
  
  3. Query motion references 25 steps (0.5s) into future
  
  4. Run ONNX forward pass:
     ONNX inputs:  current state + future motion
     ONNX output:  pd_targets (29 DOF position targets)
  
  5. Apply post-processing:
     - Acceleration clamp (limits second derivative)
     - EMA filter (exponential moving average smoothing)
  
  6. Write PD targets to MuJoCo control:
     data.ctrl[:] = pd_targets
  
  7. Step physics 20 substeps (decimation=20):
     for _ in range(20):
       mujoco.mj_step(model, data)
```

### Phase 3: Output
The simulation state can be **rendered** frame-by-frame, or the **full simulated trajectory** can be extracted and saved.

---

## Command-Line Scripts

### 1. **Headless Rendering** (Recommended for Visualization)
```bash
# Render reference motion (direct from cache, no ONNX)
python scripts/embodied/render_tracker_headless.py \
    --motion /path/to/motion_cache.pt \
    --output-dir /tmp/render_ref \
    --mode reference \
    --video

# Render tracked motion (ONNX policy simulation)
python scripts/embodied/render_tracker_headless.py \
    --motion /path/to/motion_cache.pt \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    --output-dir /tmp/render_tracked \
    --mode tracked \
    --video
```

**Modes:**
- `reference`: Renders the cache motion directly (no simulation) — shows ideal/reference motion
- `tracked`: Runs ONNX policy in MuJoCo simulation — shows what the robot actually does

**Output:** PNG frames in output directory; optional MP4 video (requires ffmpeg)

### 2. **Interactive Test/Validation** (ProtoMotions test script)
```bash
# From the ProtoMotions root (or accessible via sys.path)
python ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    --motion /path/to/motion_cache.pt \
    --loops 1 \
    --no-realtime
```

**Options:**
- `--loops N`: How many times to replay the motion (default: infinite with `--render`, 1 otherwise)
- `--render`: Open a real-time MuJoCo viewer window
- `--no-realtime`: Run as fast as possible (no wall-clock pacing)
- `--cache-motion`: Pre-resample and cache motion for faster future runs

**Output:** Console logs with tracking error (max joint ref error in radians)

### 3. **End-to-End Pipeline** (HyMotion → Robot Cache)
```bash
# Full conversion: HyMotion eval → SMPL-X → GMR → ProtoMotions cache → ONNX validation
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/.../npz/00000.npz \
    --output data/embodied_debug/robot_cache.pt \
    --validate  # This runs the ONNX tracker automatically
```

---

## Understanding the Output

### What Does the ONNX Policy Output?

The ONNX model outputs **PD position targets** (29 values for 29 DOFs):
```python
pd_targets = onnx_output[1]  # shape (29,)
```

These are then applied as implicit PD controllers in MuJoCo:
```
force = kp * (pd_target - current_position) + kd * (0 - current_velocity)
```

Where `kp` (stiffness) and `kd` (damping) are pre-configured per-joint (from the YAML metadata).

### Tracking Error
The metric is **max joint reference error** — the maximum absolute deviation (in radians) of any joint from the reference motion:
```python
ref_dof_pos = player.get_state_at_frame(frame_idx)["dof_pos"]
error = np.abs(data.qpos[7:] - ref_dof_pos).max()
```

### Can I Extract the Simulated Trajectory as a New Cache?

**Short answer: Not yet (not in the stock scripts).**

The `test_tracker_mujoco.py` and `render_tracker_headless.py` scripts run the simulation but don't save the output trajectory back to a `.pt` file. However, you can **modify** one of these scripts to collect and save the simulated state at each frame:

```python
# Pseudo-code for extracting simulated output
simulated_trajectory = {
    "dof_pos": [],
    "dof_vel": [],
    "body_rot": [],
    "body_pos": [],
    "body_vel": [],
    "body_ang_vel": [],
}

for frame_idx in range(num_frames):
    # ... run ONNX + MuJoCo step ...
    
    # Collect state (from render_tracker_headless.py lines ~490-503)
    body_rot_wxyz = data.xquat[1:].copy()
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz).astype(np.float32)
    root_rot_wxyz_mj = data.qpos[3:7].copy()
    body_rot[root_body_index] = mujoco_wxyz_to_xyzw(root_rot_wxyz_mj).astype(np.float32)
    
    # Collect all state
    simulated_trajectory["dof_pos"].append(data.qpos[7:].copy())
    simulated_trajectory["dof_vel"].append(data.qvel[6:].copy())
    simulated_trajectory["body_rot"].append(body_rot)
    simulated_trajectory["body_pos"].append(data.xpos[1:].copy())  # body positions
    simulated_trajectory["body_vel"].append(data.cvel[1:, 3:].copy())  # linear velocity
    simulated_trajectory["body_ang_vel"].append(data.cvel[1:, :3].copy())  # angular velocity

# Save
simulated_trajectory["control_dt"] = cache["control_dt"]
simulated_trajectory["num_frames"] = len(simulated_trajectory["dof_pos"])
for key in simulated_trajectory:
    if key not in ["control_dt", "num_frames"]:
        simulated_trajectory[key] = np.stack(simulated_trajectory[key])
torch.save(simulated_trajectory, "simulated_output.pt")
```

---

## ONNX Model Internals

### Inputs (from `unified_pipeline.yaml`)
| Name | Shape | Source | Meaning |
|------|-------|--------|---------|
| `current_dof_pos` | `[1, 29]` | Robot joints | Current joint angles |
| `current_dof_vel` | `[1, 29]` | Robot joints | Current joint velocities |
| `current_anchor_rot` | `[1, 4]` | Torso (body 16) | IMU-like observation (xyzw) |
| `current_root_local_ang_vel` | `[1, 3]` | Pelvis (body 0) | Angular velocity in body-local frame |
| `historical_processed_actions` | `[1, 1, 29]` | Previous step | Last applied actions (for smoothness) |
| `mimic_future_anchor_rot` | `[1, 4, 4]` | Motion player | 4 future steps of torso rotation |
| `mimic_future_dof_pos` | `[1, 4, 29]` | Motion player | 4 future steps of joint positions |
| `mimic_future_dof_vel` | `[1, 4, 29]` | Motion player | 4 future steps of joint velocities |

### Outputs
| Name | Shape | Meaning |
|------|-------|---------|
| `actions` | `[1, 29]` | Raw policy output (tanh-bounded) |
| `joint_pos_targets` | `[1, 29]` | **PD targets** (offset + scale applied) |
| `stiffness_targets` | `[1, 29]` | Per-joint stiffness (pre-configured) |
| `damping_targets` | `[1, 29]` | Per-joint damping (pre-configured) |

### Important Conventions

**Quaternions**: All quaternions are **xyzw** (ProtoMotions convention).
- MuJoCo uses wxyz internally → convert with `mujoco_wxyz_to_xyzw()`

**Body indexing**: MuJoCo stores world body at index 0.
- All body quaternions: `data.xquat[body_id + 1]` (offset by 1)
- Root body (pelvis): index 0 in the output arrays

**Angular velocity frame**:
- Root angular velocity must be in **body-local frame**
- Use `data.qvel[3:6]` directly (already local for free-joint)
- NOT `data.cvel[1, 0:3]` (that's world-frame, needs rotation)

---

## Physics Configuration (from YAML)

```yaml
timing:
  control_dt: 0.02       # 50 Hz control rate
  physics_dt: 0.001      # 1 ms physics substeps
  decimation: 20         # 20 substeps per control step
  
control:
  stiffness: [40.179, 99.098, ...]  # Per-joint kp (29 values)
  damping: [2.558, 6.309, ...]      # Per-joint kd (29 values)
  pd_target_max_accel: null         # No acceleration clamping
  action_ema_alpha: 1.0             # No EMA filtering (alpha=1.0 disables)

motion:
  future_step_indices: [1, 2, 4, 8]  # Look ahead 0.02s, 0.04s, 0.08s, 0.16s
```

---

## Code References

### Key Files

1. **Main tracker test**:
   - `ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py` (lines 456–795: `run()` function contains the main loop)

2. **Headless rendering**:
   - `scripts/embodied/render_tracker_headless.py` (lines 331–602: `render_tracked_mode()`)

3. **State utilities**:
   - `ref_repo/ProtoMotions/deployment/state_utils.py` (quaternion conversion, anchor_rot extraction)

4. **Motion playback**:
   - `ref_repo/ProtoMotions/deployment/motion_utils.py` (MotionPlayer class for motion queries)

5. **Pipeline**:
   - `scripts/embodied/pipeline_motion_to_robot.py` (lines 155–167: validation step)

### Key Functions

```python
# From render_tracker_headless.py
def render_tracked_mode(cache, model, data, onnx_path, output_dir, ...):
    """Run ONNX policy in MuJoCo, render headlessly."""
    # Loads YAML metadata, sets up physics, loops through frames

# From test_tracker_mujoco.py
def run(onnx_path, motion_file, num_loops=1, render=False, ...):
    """Interactive test with optional viewer."""

# From motion_utils.py
class MotionPlayer:
    def get_state_at_frame(frame_idx) -> dict
    def get_future_references(frame_idx, step_indices) -> dict
```

---

## Workflow Examples

### Example 1: Visualize Motion → Render Reference → Render Tracked

```bash
# Step 1: See what the reference motion looks like (ideal)
python scripts/embodied/render_tracker_headless.py \
    --motion data/embodied_debug/robot_cache.pt \
    --output-dir /tmp/ref \
    --mode reference \
    --video

# Step 2: See what the robot actually tracks (physics simulation)
python scripts/embodied/render_tracker_headless.py \
    --motion data/embodied_debug/robot_cache.pt \
    --output-dir /tmp/tracked \
    --mode tracked \
    --video

# Compare the two videos to assess tracking quality
ffplay /tmp/ref/output.mp4
ffplay /tmp/tracked/output.mp4
```

### Example 2: End-to-End Conversion + Validation

```bash
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/experiments/hymotion_v3/eval_output/00000.npz \
    --output /tmp/g1_motion.pt \
    --validate \
    --keep-intermediates
# Automatically runs validation with ONNX tracker
# Outputs: /tmp/g1_motion.pt (final cache), /tmp/g1_motion_smplx.npz, /tmp/g1_motion_gmr.pkl
```

### Example 3: Interactive Testing with Viewer

```bash
python ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx \
    --motion data/embodied_debug/robot_cache.pt \
    --render \
    --loops 3
# Opens a real-time MuJoCo viewer; you can rotate/zoom with mouse
# Loops the motion 3 times; spacebar to pause/resume
```

---

## Troubleshooting

### "FileNotFoundError: ONNX model not found"
→ Check the path: `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx`

### "ONNX metadata YAML not found"
→ The `.yaml` file must be **alongside** the `.onnx` file (same directory)

### High tracking error (max ref error > 0.1 rad)
→ Possible causes:
- Motion cache has discontinuities or jitter
- Physics parameters don't match motion
- Robot model differs from training model

### Slow rendering
→ Try `--skip-frames 5` to render every 5th frame instead of every 2nd

### No GPU/ONNX inference too slow
→ The ONNX model runs on CPU by default. On GPU, update the onnxruntime provider:
```python
session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

---

## Summary Table

| Task | Script | Input | Output | Notes |
|------|--------|-------|--------|-------|
| Visualize reference | `render_tracker_headless.py --mode reference` | `.pt` cache | PNG frames, MP4 video | No ONNX needed, fast |
| Visualize tracking | `render_tracker_headless.py --mode tracked` | `.pt` cache | PNG frames, MP4 video | Requires ONNX model |
| Test tracking | `test_tracker_mujoco.py` | `.pt` cache | Console logs + optional viewer | Interactive, can render live |
| Full pipeline | `pipeline_motion_to_robot.py --validate` | HyMotion NPZ | `.pt` cache, auto-validation | 3-step conversion + ONNX test |
| Extract sim trajectory | (custom script) | `.pt` cache | New `.pt` with simulated state | Must add to render script |

---

## Appendix: Motion Cache Structure

The output of simulated tracking is **not automatically saved** to a new cache file in the stock scripts. To export the simulated trajectory, you need to:

1. **Modify `render_tracker_headless.py`** or **`test_tracker_mujoco.py`** to collect state arrays during the loop
2. **Stack the arrays** into the cache format
3. **Save with `torch.save()`**

Expected structure of saved simulated output:
```python
{
    "dof_pos":      np.ndarray [T, 29],          # Simulated joint positions
    "dof_vel":      np.ndarray [T, 29],          # Simulated joint velocities
    "body_rot":     np.ndarray [T, 33, 4],       # Simulated body rotations
    "body_pos":     np.ndarray [T, 33, 3],       # Simulated body positions (FK-computed)
    "body_vel":     np.ndarray [T, 33, 3],       # Simulated body velocities
    "body_ang_vel": np.ndarray [T, 33, 3],       # Simulated angular velocities
    "control_dt":   float,                       # Same as input
    "num_frames":   int,                         # T
}
```

This can then be used for downstream tasks (rendering, evaluation, etc.) using `MotionPlayer`.
