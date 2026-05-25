# ProtoMotions Body Pose Capture — Code Flow Reference

## Complete Call Stack for One Control Step

```
test_tracker_mujoco.py::run() [line 615]
    │
    ├─→ load_mujoco_model() [line 541]
    │   └─→ Returns: model (MjModel), yaml_meta
    │
    ├─→ MotionPlayer(...) [line 575]
    │   └─→ Loads .pt motion file, interpolates to 50 Hz
    │
    ├─→ set_initial_pose(model, data, player) [line 645]  ◄─── Initialization
    │   ├─→ frame0 = player.get_state_at_frame(0) [line 375]
    │   ├─→ data.qpos[0:3] = root_pos [line 382]
    │   ├─→ data.qpos[3:7] = xyzw_to_wxyz(root_quat) [line 384]
    │   ├─→ data.qpos[7:] = dof_pos [line 385]
    │   └─→ mujoco.mj_forward(model, data) [line 388]
    │
    └─→ Main loop [line 660]
        │
        └─→ for frame_idx in range(num_frames): [line 661]
            │
            ├─→ robot_state = get_robot_state_from_mujoco(...) [line 680]  ◄─── STATE EXTRACTION
            │   │
            │   ├─→ body_rot_wxyz = data.xquat[1:].copy() [line 353]
            │   │   └─→ Gets all body quaternions, skips world at [0]
            │   │
            │   ├─→ body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz) [line 354]
            │   │   └─→ Converts wxyz → xyzw (CRITICAL!)
            │   │
            │   ├─→ body_rot[0] = mujoco_wxyz_to_xyzw(data.qpos[3:7]) [line 359]
            │   │   └─→ Override root with canonical qpos
            │   │
            │   ├─→ root_local_ang_vel = data.qvel[3:6] [line 363]
            │   │   └─→ Body-local angular velocity (already correct frame)
            │   │
            │   └─→ return {
            │       "dof_pos":            data.qpos[7:],
            │       "dof_vel":            data.qvel[6:],
            │       "body_rot":           body_rot,        # [num_bodies, 4]
            │       "root_local_ang_vel": root_local_ang_vel
            │   } [line 365-370]
            │
            ├─→ future_refs = player.get_future_references(...) [line 681]
            │   └─→ Returns: body_rot, dof_pos, dof_vel (25 steps ahead)
            │
            ├─→ onnx_inputs = build_onnx_inputs(...) [line 683]  ◄─── PREPARE ONNX
            │   │
            │   ├─→ dof_pos = robot_state["dof_pos"] [line 413]
            │   ├─→ body_rot = robot_state["body_rot"] [line 415]
            │   ├─→ root_local_ang_vel = robot_state["root_local_ang_vel"] [line 416]
            │   │
            │   ├─→ anchor_rot = compute_anchor_rot_np(body_rot, 16) [line 419]
            │   │   └─→ Extract torso rotation from body_rot[16]
            │   │
            │   └─→ Build key_to_array dict [line 429]
            │       ├─ "current.dof_pos": dof_pos[None]                    [1, 29]
            │       ├─ "current.dof_vel": dof_vel[None]                    [1, 29]
            │       ├─ "current.anchor_rot": anchor_rot[None]              [1, 4]
            │       ├─ "current.root_local_ang_vel": root_local_ang_vel[None] [1, 3]
            │       ├─ "mimic.future_rot": future_refs["body_rot"][None]   [1, 25, 33, 4]
            │       ├─ "mimic.future_dof_pos": future_refs["dof_pos"][None] [1, 25, 29]
            │       └─ "mimic.future_dof_vel": future_refs["dof_vel"][None] [1, 25, 29]
            │
            ├─→ ort_out = session.run(actual_out_names, onnx_inputs) [line 708]  ◄─── INFERENCE
            │   └─→ Output includes: pd_targets [1, 29]
            │
            ├─→ pd_targets = ort_out[1].squeeze() [line 712]
            │   └─→ Shape: [29]
            │
            ├─→ if pd_target_max_accel is not None: [line 715]  ◄─── POST-PROCESS
            │   │   Apply acceleration clamp [line 715-720]
            │   └─→ Limits 2nd derivative of targets
            │
            ├─→ if use_ema: [line 727]
            │   └─→ Apply EMA filter [line 730]
            │
            ├─→ data.ctrl[:] = pd_targets [line 737]  ◄─── APPLY CONTROL
            │
            ├─→ for _ in range(decimation): [line 741]  ◄─── PHYSICS STEP
            │       mujoco.mj_step(model, data)  [line 742]
            │   └─→ 4 substeps (decimation=4)
            │
            ├─→ (Optional) Track PD error [line 746-749]
            │   └─→ ref_dof_pos = player.get_state_at_frame(frame_idx)["dof_pos"]
            │
            └─→ (Optional) Sync viewer [line 752-755]
                └─→ Display in MuJoCo viewer if rendering
```

---

## Key Functions & Their Signatures

### 1. `get_robot_state_from_mujoco()` [Line 327]

```python
def get_robot_state_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_index: int = 0,
) -> dict:
    """
    Reads raw MuJoCo state and converts to ProtoMotions representation.
    
    Returns:
    -------
    {
        "dof_pos":            np.ndarray [num_dofs],      # float32
        "dof_vel":            np.ndarray [num_dofs],      # float32
        "body_rot":           np.ndarray [num_bodies, 4], # float32, xyzw
        "root_local_ang_vel": np.ndarray [3],             # float32, body-local
    }
    """
```

**Inputs:**
- `model`: MuJoCo model (read-only)
- `data`: MuJoCo data struct (latest physics state)
- `root_body_index`: Index of root body (default: pelvis at 0)

**Key operations:**
1. Line 353: Extract all body quats: `data.xquat[1:]` (wxyz)
2. Line 354: Convert to xyzw: `mujoco_wxyz_to_xyzw(body_rot_wxyz)`
3. Line 359: Override root with qpos (canonical)
4. Line 363: Get root ang. vel: `data.qvel[3:6]` (body-local)

---

### 2. `set_initial_pose()` [Line 373]

```python
def set_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion_player: MotionPlayer
) -> None:
    """Initialize robot at first frame of motion."""
```

**Operations:**
1. Line 375: Get frame 0 from player
2. Line 382: Set root XYZ: `data.qpos[0:3] = root_pos`
3. Line 384: Set root quat: `data.qpos[3:7] = xyzw_to_wxyz(quat)`
4. Line 385: Set joint angles: `data.qpos[7:] = dof_pos`
5. Line 388: Recompute FK: `mujoco.mj_forward(model, data)`

**Critical:** Quaternion conversion is **reversed** here (xyzw → wxyz).

---

### 3. `build_onnx_inputs()` [Line 400]

```python
def build_onnx_inputs(
    robot_state: dict,                    # From get_robot_state_from_mujoco()
    future_refs: dict,                    # From player.get_future_references()
    onnx_name_to_key: dict,               # Name mapping
    anchor_body_index: int,               # e.g., 16 for torso
    num_dofs: int,                        # e.g., 29
    prev_actions: np.ndarray | None = None,
) -> dict:
    """Assemble ONNX input dict with batch dimension."""
```

**Extracts from robot_state:**
- Line 413: `dof_pos = robot_state["dof_pos"]`
- Line 414: `dof_vel = robot_state["dof_vel"]`
- Line 415: `body_rot = robot_state["body_rot"]`
- Line 416: `root_local_ang_vel = robot_state["root_local_ang_vel"]`

**Computes:**
- Line 419: `anchor_rot = compute_anchor_rot_np(body_rot, anchor_body_index)`
- Line 426: `future_anchor_rot = future_refs["body_rot"][:, anchor_body_index, :]`

**Returns:** Dict with keys like:
```python
{
    "current.dof_pos": dof_pos[None],                    # [1, ndofs]
    "current.anchor_rot": anchor_rot[None],              # [1, 4]
    "mimic.future_rot": future_refs["body_rot"][None],   # [1, nsteps, nb, 4]
    ...
}
```

---

### 4. `compute_anchor_rot_np()` [Helper]

```python
def compute_anchor_rot_np(body_rot, anchor_body_index):
    """Extract anchor body rotation from full body_rot array."""
    # Pseudocode:
    anchor_rot = body_rot[anchor_body_index]  # [4] xyzw
    return anchor_rot[None]  # Add batch dim → [1, 4]
```

---

## MuJoCo Data Structure Quick Reference

```
mujoco.MjModel:
    ├─ nq: Number of generalized coordinates (DOFs)
    │  └─ For G1: 7 (free joint) + 29 (actuated) = 36
    │
    ├─ nv: Number of generalized velocities
    │  └─ For G1: 6 (free) + 29 (actuated) = 35
    │
    ├─ nbody: Number of bodies (including world)
    │  └─ For G1: 34 (world + 33 actual bodies)
    │
    └─ nuserdata, nmotor, etc.

mujoco.MjData:
    ├─ qpos [nq]: Generalized coordinates
    │  │  Format: [free_xyz, free_quat_wxyz, joint_angles...]
    │  │  Indices: [0:3, 3:7, 7:36]
    │  │  For G1: 36 values
    │  │
    │  └─ qpos[0:3]   = root position (world frame)
    │     qpos[3:7]   = root quaternion (wxyz, world frame)
    │     qpos[7:]    = joint angles (29 values)
    │
    ├─ qvel [nv]: Generalized velocities
    │  │  Format: [free_ang_vel, free_lin_vel, joint_vels...]
    │  │  For G1: 35 values
    │  │
    │  └─ qvel[0:3]   = root angular velocity (body-local frame!)
    │     qvel[3:6]   = root linear velocity (world frame)
    │     qvel[6:]    = joint angular velocities (29 values)
    │
    ├─ xquat [nbody, 4]: Body quaternions (wxyz format)
    │  │  Computed by FK during mj_step()
    │  │
    │  └─ xquat[0]    = world body (skip)
    │     xquat[1]    = body 0 (pelvis)
    │     xquat[2]    = body 1
    │     ...
    │     xquat[33]   = body 32
    │
    ├─ cvel [nbody, 6]: Body 6D velocities (angular, linear)
    │  │  Frame: body-local or world depending on context
    │  │
    │  └─ cvel[i, 0:3] = angular velocity of body i
    │     cvel[i, 3:6] = linear velocity of body i
    │
    └─ ctrl [nmotor]: Control inputs (PD targets or torques)
       └─ For G1: 29 values (one per DOF)
```

---

## Quaternion Conversion Details

### Convention:
```
wxyz (MuJoCo):  [w, x, y, z]  ← scalar first, vector second
xyzw (ProtoMotions): [x, y, z, w]  ← vector first, scalar second
```

### Conversion:
```python
# wxyz → xyzw
def mujoco_wxyz_to_xyzw(quat_wxyz):
    return quat_wxyz[..., [1, 2, 3, 0]]

# xyzw → wxyz (reverse)
def xyzw_to_wxyz(quat_xyzw):
    return quat_xyzw[..., [3, 0, 1, 2]]
```

### In context:
```python
# Reading from MuJoCo (wxyz input)
root_quat_wxyz = data.qpos[3:7]           # [w, x, y, z]
root_quat_xyzw = root_quat_wxyz[[1,2,3,0]]  # [x, y, z, w]

# Writing to MuJoCo (wxyz output)
root_quat_xyzw = np.array([x, y, z, w])  # [x, y, z, w]
data.qpos[3:7] = root_quat_xyzw[[3,0,1,2]]  # [w, x, y, z]
```

---

## Main Loop Structure

```python
# Lines 660-776
for loop_idx in range(num_loops):
    # Reset or continue from last frame
    
    for frame_idx in range(num_frames_per_loop):
        # ─── STATE EXTRACTION (line 680)
        robot_state = get_robot_state_from_mujoco(model, data, root_body_index=0)
        
        # ─── FUTURE REFERENCES (line 681)
        future_refs = player.get_future_references(...)
        
        # ─── PREPARE ONNX (line 683)
        onnx_inputs = build_onnx_inputs(robot_state, future_refs, ...)
        
        # ─── ONNX INFERENCE (lines 707-712)
        t0 = time.perf_counter()
        ort_out = session.run(actual_out_names, onnx_inputs)
        pd_targets = ort_out[1].squeeze().copy()
        total_ort_ms += (time.perf_counter() - t0) * 1000.0
        
        # ─── POST-PROCESSING (lines 714-734)
        # Apply accel clamp (if enabled)
        if pd_target_max_accel is not None and prev_pd is not None:
            accel = (pd_targets - prev_pd) - (prev_pd - prev_prev_pd)
            accel = np.clip(accel, -pd_target_max_accel, +pd_target_max_accel)
            pd_targets = prev_pd + (prev_pd - prev_prev_pd) + accel
        
        prev_prev_pd = prev_pd
        prev_pd = pd_targets.copy()
        
        # Apply EMA filter (if enabled)
        if use_ema:
            if ema_prev_targets is None:
                ema_prev_targets = pd_targets.copy()
            pd_targets = action_ema_alpha * pd_targets + (1-action_ema_alpha) * ema_prev_targets
            ema_prev_targets = pd_targets.copy()
        
        prev_actions = pd_targets.copy()
        
        # ─── APPLY CONTROL (line 737)
        data.ctrl[:] = pd_targets
        
        # ─── PHYSICS SIMULATION (lines 741-743)
        t0 = time.perf_counter()
        for _ in range(decimation):  # decimation=4
            mujoco.mj_step(model, data)
        total_sim_ms += (time.perf_counter() - t0) * 1000.0
        
        # ─── DIAGNOSTICS (lines 746-775)
        # Track error, display status, sync viewer
```

---

## Data Dimensions Summary (G1 Example)

```
Input dimensions:
  num_dofs = 29
  num_bodies = 33

Robot state (single step, no batch):
  dof_pos [29]
  dof_vel [29]
  body_rot [33, 4] xyzw
  root_local_ang_vel [3]

Future references (25 steps, no batch):
  dof_pos [25, 29]
  dof_vel [25, 29]
  body_rot [25, 33, 4] xyzw

ONNX inputs (with batch dim):
  current.dof_pos [1, 29]
  current.dof_vel [1, 29]
  current.anchor_rot [1, 4]
  current.root_local_ang_vel [1, 3]
  mimic.future_rot [1, 25, 33, 4]
  mimic.future_dof_pos [1, 25, 29]
  mimic.future_dof_vel [1, 25, 29]

ONNX outputs:
  actions [1, 29]
  joint_pos_targets [1, 29]
  stiffness_targets [1, 29]
  damping_targets [1, 29]

After post-processing (squeeze batch):
  pd_targets [29]
```

---

## Critical Implementation Details

### 1. Body Index Offset (Line 353)
```python
body_rot_wxyz = data.xquat[1:]  # Skip world at index 0
# Now body_rot_wxyz[0] corresponds to body 0 (pelvis)
# And body_rot_wxyz[i] corresponds to body i
```

### 2. Root Quaternion Precedence (Line 359)
```python
# Override root rotation with qpos version (more canonical)
root_rot_wxyz = data.qpos[3:7]
body_rot[0] = mujoco_wxyz_to_xyzw(root_rot_wxyz)
# Avoids FK rounding errors on root
```

### 3. Angular Velocity Frame (Line 363)
```python
root_local_ang_vel = data.qvel[3:6]
# This is ALREADY body-local, no rotation transformation needed
# Unlike data.cvel[0, 0:3] which is world-frame
```

### 4. Anchor Body Extraction (Line 419)
```python
anchor_rot = compute_anchor_rot_np(body_rot, 16)
# For G1, anchor_body_index=16 (torso_link)
# Extracts just that one body's rotation for policy obs
```

### 5. Batch Dimension Addition (Lines 430-438)
```python
key_to_array["current.dof_pos"] = dof_pos[None]  # [29] → [1, 29]
# Add batch dim with [None] indexing
# All ONNX inputs must have batch dimension
```

---

## File Cross-References

| Function | File | Lines | Purpose |
|----------|------|-------|---------|
| `get_robot_state_from_mujoco()` | test_tracker_mujoco.py | 327-370 | Extract state from MuJoCo |
| `set_initial_pose()` | test_tracker_mujoco.py | 373-392 | Initialize robot pose |
| `build_onnx_inputs()` | test_tracker_mujoco.py | 400-439 | Assemble ONNX inputs |
| `compute_anchor_rot_np()` | state_utils.py | (imported) | Extract anchor rotation |
| `mujoco_wxyz_to_xyzw()` | state_utils.py | (imported) | Quat conversion |
| `MotionPlayer` | motion_utils.py | (imported) | Motion clip playback |
| `run()` | test_tracker_mujoco.py | 615-800 | Main entry point |
| Main loop | test_tracker_mujoco.py | 660-776 | Control loop |

