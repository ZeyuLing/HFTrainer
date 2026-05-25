# ProtoMotions Body Pose Capture Analysis
## Reference: `test_tracker_mujoco.py` (MuJoCo Inference Test)

---

## Executive Summary

The ProtoMotions tracker deployment captures body poses at the **MuJoCo read boundary** (where robot state is extracted from the physics engine). The key function is **`get_robot_state_from_mujoco()`** (lines 327-370), which converts MuJoCo's internal quaternion representation (wxyz) to the ProtoMotions standard (xyzw) and extracts both body rotations and DOF positions/velocities.

### Critical Convention Conversions:
| Aspect | MuJoCo | ProtoMotions | Function |
|--------|--------|--------------|----------|
| **Quaternion format** | wxyz | xyzw | `mujoco_wxyz_to_xyzw()` |
| **Body indexing** | `data.xquat[i+1]` (world at 0) | 0-indexed | Skip index 0 |
| **Root quaternion source** | `data.qpos[3:7]` (canonical) | `data.xquat` (FK) | Prefer qpos for root |
| **Root ang. vel. frame** | Already body-local | Body-local | Use `data.qvel[3:6]` directly |

---

## Main Data Flow (Main Loop: lines 660-776)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Control Step i (50 Hz)                       │
├─────────────────────────────────────────────────────────────────┤
│ 1. get_robot_state_from_mujoco(model, data, root_body_index)   │
│    └─> Extracts: dof_pos, dof_vel, body_rot, root_local_ang_vel
│                                                                 │
│ 2. compute_anchor_rot_np(body_rot, anchor_body_index)           │
│    └─> Anchor rotation for torso/pelvis (from body_rot)         │
│                                                                 │
│ 3. future_refs = player.get_future_references(...)             │
│    └─> Future motion: body_rot, dof_pos, dof_vel (25 steps)    │
│                                                                 │
│ 4. build_onnx_inputs(...) → dict with [1, ...] batch dims      │
│    └─> ONNX inputs: current state + future refs                │
│                                                                 │
│ 5. ort.run(session, onnx_inputs) → pd_targets [num_dofs]       │
│                                                                 │
│ 6. PD acceleration clamp (if enabled)                          │
│ 7. EMA filter on targets (if enabled)                          │
│ 8. data.ctrl[:] = pd_targets                                   │
│                                                                 │
│ 9. mujoco.mj_step() × decimation (4 substeps)                  │
│                                                                 │
│ 10. (Optional) Measure tracking error vs reference             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Function: `get_robot_state_from_mujoco()` (lines 327-370)

### Signature
```python
def get_robot_state_from_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_index: int = 0,
) -> dict:
    """Extract robot state from MuJoCo as ProtoMotions representation."""
```

### Implementation Details

#### 1. **Body Rotations** (lines 351-359)
```python
# Step A: Extract all body quaternions in wxyz from MuJoCo
body_rot_wxyz = data.xquat[1:].copy()          # [num_bodies, 4]
#                           ↑ skip world body at index 0

# Step B: Convert wxyz → xyzw
body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz)  # [num_bodies, 4]

# Step C: For root body, prefer canonical qpos[3:7] (FK-computed xquat
#         can have rounding errors)
root_rot_wxyz = data.qpos[3:7].copy()
body_rot[root_body_index] = mujoco_wxyz_to_xyzw(root_rot_wxyz)
```

**Key points:**
- `data.xquat[i]` stores the world-frame quaternion for body i (FK-computed)
- For the root body (pelvis), `data.qpos[3:7]` is the **canonical** quaternion
  - This matches robojudo's `base_quat` path
  - Avoids FK rounding errors
- All output in **xyzw** format (ProtoMotions standard)

#### 2. **DOF Positions & Velocities** (lines 366-367)
```python
"dof_pos":  data.qpos[7:].copy().astype(np.float32),  # [num_dofs]
#                    ↑ skip 7-DOF free joint (3 pos + 4 quat)

"dof_vel":  data.qvel[6:].copy().astype(np.float32),  # [num_dofs]
#                    ↑ skip 6-DOF free joint (3 pos + 3 ang_vel)
```

#### 3. **Root Angular Velocity** (line 363)
```python
root_local_ang_vel = data.qvel[3:6].copy().astype(np.float32)  # [3]
#                                 ↑ Free joint ang_vel indices
```

**Why `data.qvel[3:6]` directly?**
- MuJoCo's free-joint quaternion derivative is **already in body-local frame**
  (unlike `data.cvel[0, 0:3]` which requires inverse rotation)
- No further transformation needed
- Avoids `quat_rot_inv(pelvis_rot, pelvis_ω)` computation on this one

#### 4. **Return Structure**
```python
return {
    "dof_pos":            [...],     # [num_dofs] (e.g., 29 for G1)
    "dof_vel":            [...],     # [num_dofs]
    "body_rot":           [...],     # [num_bodies, 4] xyzw
    "root_local_ang_vel": [...],     # [3]
}
```

---

## Quaternion Conversion: `mujoco_wxyz_to_xyzw()`

**Source:** `state_utils.py` (imported)

**Conversion:**
```
Input:  wxyz = [w, x, y, z]  (MuJoCo format)
Output: xyzw = [x, y, z, w]  (ProtoMotions format)
```

**Implementation (typical):**
```python
def mujoco_wxyz_to_xyzw(quat_wxyz):
    """Convert wxyz to xyzw via array reindexing."""
    return quat_wxyz[..., [1, 2, 3, 0]]  # [w,x,y,z] -> [x,y,z,w]
```

---

## Anchoring Body Concept

### What is `anchor_body_index`?

The "anchor body" is the **primary reference frame** for the policy observation. For G1:
- **anchor_body_index = 16** (torso_link)
- Used to compute `anchor_rot = body_rot[anchor_body_index]`
- This is sent to the ONNX model as `current_state_anchor_rot`

### Why separate from root?

- **Root (pelvis):** For free-joint kinematics, foot contact stability, trajectory
- **Anchor (torso):** For IMU-like observations, upper-body orientation, visual feedback

### In code (lines 418-419):
```python
# Anchor rotation: works for any anchor body (pelvis, torso, etc.)
anchor_rot = compute_anchor_rot_np(body_rot, anchor_body_index)  # [4]
```

---

## Pose Capture During Initialization

### `set_initial_pose()` (lines 373-392)

Initializes the robot at frame 0 of the motion file:

```python
def set_initial_pose(model, data, motion_player: MotionPlayer) -> None:
    frame0 = motion_player.get_state_at_frame(0)
    
    # Extract root pose from motion player
    root_pos  = frame0["body_pos"][0]       # [3]
    root_quat = frame0["body_rot"][0]       # [4] xyzw
    
    # Set MuJoCo state
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat[[3, 0, 1, 2]]  # xyzw -> wxyz (reverse!)
    data.qpos[7:]  = frame0["dof_pos"]
    
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)  # Recompute FK
```

**Key:**
- Motion file stores xyzw; MuJoCo needs wxyz
- Reverse index: `[[3, 0, 1, 2]]` converts xyzw → wxyz
- `mj_forward()` recomputes all body rotations (xquat)

---

## ONNX Input Assembly (lines 400-439)

### Function: `build_onnx_inputs()`

Maps semantic context keys to arrays with batch dimension:

```python
key_to_array = {
    "current.dof_pos":             dof_pos[None],                # [1, ndofs]
    "current.dof_vel":             dof_vel[None],                # [1, ndofs]
    "current.anchor_rot":          anchor_rot[None],             # [1, 4]
    "current.root_local_ang_vel":  root_local_ang_vel[None],     # [1, 3]
    "historical.processed_actions": prev_actions[None, None],    # [1, 1, ndofs]
    
    # Future references (25 steps ahead)
    "mimic.future_anchor_rot": future_anchor_rot[None],          # [1, nsteps, 4]
    "mimic.future_rot":     future_refs["body_rot"][None],       # [1, nsteps, nb, 4]
    "mimic.future_dof_pos": future_refs["dof_pos"][None],        # [1, nsteps, ndofs]
    "mimic.future_dof_vel": future_refs["dof_vel"][None],        # [1, nsteps, ndofs]
    ...
}
```

---

## Key Body Indexing Conventions

| Item | Convention | Example (G1) |
|------|-----------|----------|
| `data.xquat` | `[0]` = world, `[1+]` = bodies | pelvis at `[1]`, torso at `[17]` |
| `data.qpos[0:7]` | Free joint (world frame): `[x, y, z, w, x, y, z]` | Pelvis position + orientation |
| `data.qpos[7:]` | Actuated DOFs (joint angles) | Joint angles for 22 joints + 1 finger |
| `data.qvel[0:6]` | Free joint velocity (6 DOF) | 3 linear + 3 angular (body-local) |
| `data.qvel[6:]` | Actuated DOF velocities | Joint angular velocities |
| `data.cvel[i]` | Body i's 6D velocity | `[0:3]` = angular (body-local), `[3:6]` = linear |

---

## Main Loop Phases (lines 660-776)

### Phase 1: State Extraction (lines 680-704)
```python
robot_state = get_robot_state_from_mujoco(model, data, root_body_index=0)
future_refs = player.get_future_references(...)
onnx_inputs = build_onnx_inputs(robot_state, future_refs, ...)
```

### Phase 2: ONNX Inference (lines 706-712)
```python
t0 = time.perf_counter()
ort_out = session.run(actual_out_names, onnx_inputs)
pd_targets = ort_out[1].squeeze().copy()  # [num_dofs]
```

### Phase 3: Action Post-Processing (lines 714-734)
1. **PD acceleration clamp** (lines 715-720)
   - Limits 2nd derivative of targets
   - Matches `base_simulator._apply_accel_clamp()`
   
2. **EMA filter** (lines 727-731)
   - Exponential moving average on targets
   - `new_target = alpha * current + (1-alpha) * prev`
   - Matches `MujocoSimulator._action_filter_alpha`

### Phase 4: Physics Simulation (lines 737-743)
```python
data.ctrl[:] = pd_targets  # Set control inputs
for _ in range(decimation):  # Default: 4 substeps
    mujoco.mj_step(model, data)
```

### Phase 5: Diagnostics & Visualization (lines 745-775)
- Track PD error vs reference
- Display root height, speed ratio
- Sync viewer if rendering

---

## Deployment Contract (Key Guarantees)

1. **State extraction is position-invariant**
   - No absolute position in observations
   - Only rotations, DOF angles, velocities, angular velocities
   - Motion realignment not needed (unlike training)

2. **Quaternion convention strictly enforced**
   - Input: MuJoCo wxyz
   - Output: ProtoMotions xyzw
   - Conversion at read boundary only

3. **Body indexing is stable**
   - Same MJCF → same body ID mapping
   - Anchor body from YAML metadata
   - Root always body 0 (pelvis)

4. **Control pipeline matches training**
   - PD targets before accel clamp
   - Then EMA filter
   - Then physics substeps
   - Same decimation factor

---

## Critical Gotchas

1. **Don't mix root and anchor bodies**
   - Root (pelvis): free-joint kinematics, foot contact
   - Anchor (torso): IMU-like obs
   - Using wrong one silently breaks training-deployment alignment

2. **Quaternion mismatch is silent**
   - wxyz ↔ xyzw off-by-one errors don't crash, just produce garbage poses
   - Always verify with ground truth reference

3. **Body index offset**
   - `data.xquat[body_id + 1]`, not `data.xquat[body_id]`
   - World body at index 0 is a trap

4. **Angular velocity frame**
   - `data.qvel[3:6]` already body-local (no rotation needed)
   - `data.cvel[0, 0:3]` is world-frame (rotation needed)
   - Wrong choice produces asymmetric physics

5. **Motion player clock vs physics clock**
   - Motion player runs at 25 Hz (by default), policy at 50 Hz
   - Proper interpolation is crucial
   - See `MotionPlayer.get_future_references()` for details

---

## Reference: Input Dimensions for G1

```
num_dofs = 29  (22 joints + 1 finger, per YAML)
num_bodies = 33
root_body_index = 0 (pelvis)
anchor_body_index = 16 (torso_link)

ONNX Inputs:
  current_state_dof_pos         [1, 29]
  current_state_dof_vel         [1, 29]
  current_state_anchor_rot      [1, 4]
  current_state_root_local_ang_vel [1, 3]
  historical.processed_actions  [1, 1, 29]
  mimic.future_rot              [1, 25, 33, 4]  (25 future steps, all bodies)
  mimic.future_dof_pos          [1, 25, 29]
  mimic.future_dof_vel          [1, 25, 29]
  mimic.future_anchor_rot       [1, 25, 4]     (extracted from mimic.future_rot)
```

---

## Summary Table: Where Each Piece Comes From

| Observation | Source | Conversion | Frame |
|-------------|--------|-----------|-------|
| DOF position | `data.qpos[7:]` | None | Body-local (joint angles) |
| DOF velocity | `data.qvel[6:]` | None | Body-local |
| Body rotation (all) | `data.xquat[1+body_id]` | wxyz→xyzw | World-frame |
| Body rotation (root) | `data.qpos[3:7]` | wxyz→xyzw | World-frame |
| Root ang. velocity | `data.qvel[3:6]` | None | Body-local |
| Anchor rotation | `body_rot[anchor_id]` | Already xyzw | World-frame |
| Future motion | `MotionPlayer.get_future_references()` | Pre-interpolated | — |

