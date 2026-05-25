# ProtoMotions Tracker MuJoCo Test — Code Structure Summary

**File**: `ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py`

**Purpose**: Standalone deployment contract demonstrating how to drive a ProtoMotions whole-body tracker ONNX policy using only raw MuJoCo state and ONNX runtime, with minimal dependency on the ProtoMotions training framework.

---

## 1. Imports & Constants (Lines 1–150)

### Core Dependencies
```python
import mujoco                    # Physics simulation
import onnxruntime as ort        # ONNX model inference
import yaml                      # Load YAML metadata
import numpy as np               # Numerical operations
import torch                     # For loading motion .pt files (first run only)
```

### Key Helper Imports
- `deployment.state_utils`: Conversion utilities for quaternion conventions and angular velocity derivations
  - `mujoco_wxyz_to_xyzw()`: Convert MuJoCo wxyz → ProtoMotions xyzw
  - `compute_anchor_rot_np()`: Derive IMU-body rotation (torso_link, body 16)
  - `compute_root_local_ang_vel_np()`: Derive pelvis angular velocity in local frame
  - `compute_yaw_offset_np()`: Calculate heading alignment between robot and reference

- `deployment.motion_player`: `MotionPlayer` class for querying future motion references
  - `get_future_references(frame_idx, nsteps)`: Returns future body rotations, DOF positions/velocities
  - `get_state_at_frame(frame_idx)`: Returns reference state at a specific frame

### Important Constants
- **Quaternion convention**: wxyz (MuJoCo) ↔ xyzw (ProtoMotions) at the read boundary
- **Body indexing**: `data.xquat[body_id + 1]` (world body at index 0)
- **Angular velocity storage**: `data.cvel[body_id + 1, 0:3]` (world frame)
- **Root DOF angular velocity**: `data.qvel[3:6]` (local frame, no conversion needed)

---

## 2. Main Pipeline Flow (Control Loop)

### Step 1: Read Robot State (Lines 334–370)

```python
def read_robot_state(data, anchor_body_index: int, root_body_index: int = 0):
    """Extract current state from MuJoCo data.
    
    Returns:
        dict with keys:
        - 'dof_pos': [29]           data.qpos[7:] (skip free joint)
        - 'dof_vel': [29]           data.qvel[6:] (skip free joint)
        - 'body_rot': [33, 4]       all body quaternions (xyzw)
        - 'root_local_ang_vel': [3] pelvis angular velocity (local frame)
    """
    # All bodies' FK-computed orientations
    body_rot_wxyz = data.xquat[1:].copy()  # [33, 4] wxyz format
    body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz)  # convert to xyzw
    
    # Direct root DOF ang vel (already in local frame)
    root_local_ang_vel = data.qvel[3:6].copy().astype(np.float32)
```

**Key observations**:
- **Body rotations** come from `data.xquat` (forward-kinematics precomputed in MuJoCo)
- **Root angular velocity** uses `data.qvel[3:6]` (DOF velocity, local frame)
- NOT using `data.cvel` for root ang vel (would need world→local conversion)

### Step 2: Derive Anchor & Root Angular Velocity (Lines 410–440)

```python
# Inside main loop (line 410+)
robot_state = read_robot_state(data, anchor_body_index)

# Current state for ONNX input
anchor_rot = compute_anchor_rot_np(robot_state["body_rot"], anchor_body_index)  # [4] xyzw
root_local_ang_vel = robot_state["root_local_ang_vel"]  # [3]

# Future references from motion file
future_refs = player.get_future_references(frame_idx, nsteps=25)
future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]  # [25, 4]
future_dof_pos = future_refs["dof_pos"]  # [25, 29]
future_dof_vel = future_refs["dof_vel"]  # [25, 29]
```

### Step 3: Query Future Motion (Motion Player)

```python
future_refs = player.get_future_references(frame_idx, nsteps=25)
# Returns dict with:
# - 'body_rot': [25, 33, 4]       future body rotations (xyzw)
# - 'dof_pos': [25, 29]           future DOF positions
# - 'dof_vel': [25, 29]           future DOF velocities
```

Motion Player interpolates from the motion file (50 fps default) to match control frequency.

### Step 4: Prepare ONNX Inputs (Lines 440–480)

```python
onnx_inputs = {
    "current_state_dof_pos":           dof_pos[None],          # [1, 29]
    "current_state_dof_vel":           dof_vel[None],          # [1, 29]
    "current_state_anchor_rot":        anchor_rot[None],       # [1, 4]
    "current_state_root_local_ang_vel": root_local_ang_vel[None], # [1, 3]
    "mimic_future_rot":                future_anchor_rot[None],   # [1, 25, 33, 4]
    "mimic_future_dof_pos":            future_dof_pos[None],      # [1, 25, 29]
    "mimic_future_dof_vel":            future_dof_vel[None],      # [1, 25, 29]
}
```

**Note**: `mimic_future_rot` contains **all 33 bodies**, but only the anchor body (16) drives tracking loss.

### Step 5: Run ONNX Inference (Lines 706–709)

```python
ort_out = session.run(actual_out_names, onnx_inputs)
pd_targets = ort_out[1].squeeze().copy()  # [29] raw PD position targets
```

**Output order** (from ONNX export):
1. `actions` — raw model output (tanh-bounded)
2. `joint_pos_targets` — **PD position targets** (offset + scale already applied)
3. `stiffness_targets` — per-DOF stiffness (constant)
4. `damping_targets` — per-DOF damping (constant)

### Step 6: Action Post-Processing (Lines 714–734)

#### 6a. PD Acceleration Clamp (Lines 715–720)

```python
if pd_target_max_accel is not None and prev_pd is not None and prev_prev_pd is not None:
    delta = pd_targets - prev_pd                      # 1st derivative
    prev_delta = prev_pd - prev_prev_pd               # previous 1st derivative
    accel = delta - prev_delta                        # 2nd derivative
    clamped_accel = np.clip(accel, -pd_target_max_accel, pd_target_max_accel)
    pd_targets = prev_pd + prev_delta + clamped_accel  # integrate back
```

**Purpose**: Prevents large accelerations in PD targets (matches simulator behavior).

#### 6b. EMA Action Filter (Lines 727–731)

```python
if use_ema:
    if ema_prev_targets is None:
        ema_prev_targets = pd_targets.copy()
    pd_targets = action_ema_alpha * pd_targets + (1.0 - action_ema_alpha) * ema_prev_targets
    ema_prev_targets = pd_targets.copy()
```

**Purpose**: Exponential moving average smoothing (α=0.9 means 90% new, 10% old).

### Step 7: Apply Action to MuJoCo (Lines 737–743)

```python
data.ctrl[:] = pd_targets          # Write PD targets to joint controls
for _ in range(decimation):        # Physics substeps (typically 5)
    mujoco.mj_step(model, data)
```

**Decimation**: Each control step = 5 physics steps (0.2 ms × 5 = 1 ms = 50 Hz control)

---

## 3. Body Pose Capture — Key Details

### Body Rotation Access Pattern

**Source**: `data.xquat[body_id + 1]`
- `data.xquat[0]` = world frame quaternion (unused)
- `data.xquat[1]` = pelvis (body 0)
- `data.xquat[16]` = torso_link (body 15) on G1
- `data.xquat[33]` = last body

**Format**: MuJoCo wxyz → convert to ProtoMotions xyzw via `mujoco_wxyz_to_xyzw()`

### Root Angular Velocity Access Pattern

**Two sources** (used differently):
1. **`data.qvel[3:6]`**: Free-joint angular velocity (local frame, **use as-is**)
   - This is what goes into ONNX as `current_state_root_local_ang_vel`
2. **`data.cvel[body_id + 1, 0:3]`**: Body angular velocity (world frame, requires conversion)
   - Used when needing world-frame ang vel (e.g., collision response)

**Critical**: MuJoCo documentation says `data.cvel[i, 0:3]` is world-frame, but the code uses `data.qvel[3:6]` for the model because it's already local-frame.

### Anchor Body Handling

```python
anchor_rot = compute_anchor_rot_np(body_rot, anchor_body_index)
# body_rot: [33, 4] all body quaternions (xyzw)
# anchor_body_index: 16 (torso_link on G1)
# Returns: [4] quaternion of anchor body
```

**Why separate?**: 
- Model observation uses **torso_link** (body 16) as "IMU body" for stability
- Pelvis (body 0) used for local ang vel because it's the kinematic root
- These are **different bodies** — confusion leads to silent bugs

---

## 4. Main Loop Structure (Lines 620–777)

```python
def run(...):
    # Setup phase
    model, data = load_mujoco_model(...)
    session = ort.InferenceSession(onnx_path)
    player = MotionPlayer(motion_file, target_fps=50)
    viewer = None if not render else mujoco.Viewer(model)
    
    # State tracking for post-processing
    prev_pd, prev_prev_pd = None, None
    ema_prev_targets = None
    
    # Main loop
    loop_idx = 0
    while loop_idx < num_loops:
        frame_idx = 0
        
        # Motion loop
        while frame_idx < player.motion_length:
            step_wall_start = time.perf_counter()
            
            # 1. Read robot state
            robot_state = read_robot_state(data, anchor_body_index)
            anchor_rot = compute_anchor_rot_np(...)
            
            # 2. Query future motion
            future_refs = player.get_future_references(frame_idx, nsteps=25)
            
            # 3. Build ONNX inputs
            onnx_inputs = {
                "current_state_dof_pos": ...,
                "current_state_anchor_rot": ...,
                "mimic_future_rot": ...,
                ...
            }
            
            # 4. Run ONNX inference
            ort_out = session.run(out_names, onnx_inputs)
            pd_targets = ort_out[1].squeeze().copy()
            
            # 5. Post-process (accel clamp + EMA)
            if pd_target_max_accel is not None:
                # ... accel clamping logic
            if use_ema:
                # ... EMA filtering logic
            
            # 6. Apply action + step physics
            data.ctrl[:] = pd_targets
            for _ in range(decimation):
                mujoco.mj_step(model, data)
            
            # 7. Optional: sync viewer
            if viewer is not None and viewer.is_running():
                viewer.sync()
            
            # 8. Real-time pacing
            if realtime:
                elapsed = time.perf_counter() - step_wall_start
                sleep_time = control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # 9. Logging (every 100 steps)
            frame_idx += 1
            total_steps += 1
            if frame_idx % 100 == 0:
                root_height = float(data.qpos[2])
                speed_ratio = sim_elapsed / wall_elapsed
                log.info(f"step={total_steps} frame={frame_idx} root_h={root_height:.3f}")
        
        loop_idx += 1
    
    # Summary
    avg_ort_ms = total_ort_ms / max(total_steps, 1)
    avg_sim_ms = total_sim_ms / max(total_steps, 1)
    log.info(f"avg ONNX inference: {avg_ort_ms:.2f} ms/step")
    log.info(f"avg physics: {avg_sim_ms:.2f} ms/step")
```

---

## 5. Critical Conventions & Gotchas

### ✅ Quaternion Convention
- **MuJoCo**: wxyz (w is scalar, xyz is vector part)
- **ProtoMotions**: xyzw (xyz is vector part, w is scalar)
- **Action**: Convert at read boundary with `mujoco_wxyz_to_xyzw()`

### ✅ Body Indexing
- `data.xquat[body_id + 1]` NOT `data.xquat[body_id]`
- `data.cvel[body_id + 1]` NOT `data.cvel[body_id]`
- World body occupies index 0

### ✅ Root Angular Velocity Frame
- `data.qvel[3:6]` = **local frame** (use as-is)
- `data.cvel[body_id + 1, 0:3]` = **world frame** (needs rotation conversion)
- Model expects **local frame** for `current_state_root_local_ang_vel`

### ✅ Anchor Body vs Root Body
- **Anchor** (torso_link, body 16): Used for `anchor_rot` observation
- **Root** (pelvis, body 0): Used for `root_local_ang_vel` observation
- **Different bodies** — mixing causes silent failures

### ⚠️ Motion Realignment
- Training uses `realign_motion_with_humanoid_on_each_step` (XY snapping)
- **Not needed for this config** because observations use only rotations + DOF pos/vel
- If future configs use position-dependent obs, realignment must be added

---

## 6. Performance Metrics

The code tracks:
- **ONNX inference time**: `total_ort_ms` accumulates per-step inference duration
- **Physics time**: `total_sim_ms` accumulates per-step physics substeps
- **Max joint error**: `max_pd_diff` = max absolute error between actual DOF pos and reference

Final output:
```
avg ONNX inference : X.XX ms/step
avg physics        : Y.YY ms/step
max joint ref error: Z.ZZZZ rad
```

---

## 7. CLI Interface

```bash
python deployment/test_tracker_mujoco.py \
    --onnx path/to/unified_pipeline.onnx \
    --motion path/to/motion.motion \
    --cache-motion \
    --render \
    --loops 3 \
    --no-realtime \
    --action-ema-alpha 0.95
```

**Arguments**:
- `--onnx` (required): Path to ONNX model
- `--motion` (required): Path to motion .pt or .motion file
- `--cache-motion`: Save 50fps cached .pt version
- `--loops`: Number of motion loops (default: ∞ with --render, 1 otherwise)
- `--render`: Open MuJoCo viewer
- `--no-realtime`: Disable real-time pacing (run max speed)
- `--action-ema-alpha`: Override EMA alpha from YAML (default: from metadata)

---

## 8. Motion File Handling

### Raw Motion File (first run)
```python
player = MotionPlayer(motion_file="data/walk.motion")
# Interpolates to 50fps, requires protomotions library
```

### Cached Motion File (subsequent runs)
```python
player = MotionPlayer(motion_file="data/walk.50fps.pt")
# Pre-sampled at 50fps, **no protomotions imports needed**
```

**Cache creation**:
```bash
python deployment/test_tracker_mujoco.py \
    --onnx model.onnx \
    --motion data/walk.motion \
    --cache-motion
# Creates: data/walk.50fps.pt
```

---

## Summary Table

| Component | Lines | Key Function |
|-----------|-------|--------------|
| **Imports** | 132–150 | Core deps: mujoco, ort, numpy, yaml |
| **Robot state** | 334–370 | `read_robot_state()` extracts qpos, qvel, body rotations |
| **Anchor rotation** | ~419 | `compute_anchor_rot_np()` from torso body (16) |
| **Root ang vel** | ~416 | Uses `data.qvel[3:6]` (local frame, no conversion) |
| **Motion player** | ~410–428 | Queries future refs via `player.get_future_references()` |
| **ONNX prep** | 440–480 | Formats 7 inputs: current DOF + anchor + future motion |
| **ONNX inference** | 706–709 | Runs `session.run()`, extracts PD targets |
| **Accel clamp** | 715–720 | Limits 2nd derivative of PD targets |
| **EMA filter** | 727–731 | Smooths targets with exponential moving average |
| **Physics step** | 737–743 | Writes to `data.ctrl`, runs decimation substeps |
| **Main loop** | 620–777 | Full integration with timing, logging, viewer sync |

