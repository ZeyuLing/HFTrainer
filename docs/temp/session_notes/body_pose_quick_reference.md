# ProtoMotions Body Pose Capture — Quick Reference

## TL;DR: The One Function That Matters

**`get_robot_state_from_mujoco(model, data, root_body_index=0)`** extracts all robot state from MuJoCo.

```python
def get_robot_state_from_mujoco(model, data, root_body_index=0):
    """
    Output:
    -------
    {
        "dof_pos":             [num_dofs],           # Joint angles
        "dof_vel":             [num_dofs],           # Joint velocities  
        "body_rot":            [num_bodies, 4],      # ALL body rotations (xyzw)
        "root_local_ang_vel":  [3],                  # Pelvis angular velocity
    }
    """
```

---

## Quaternion Conversion Chain (The One Thing That Gets People)

```
MuJoCo Physics Engine
    ↓
data.xquat[i]  →  [w, x, y, z]  (wxyz format, world-frame)
    ↓
mujoco_wxyz_to_xyzw()  →  [x, y, z, w]  (xyzw format, ProtoMotions standard)
    ↓
ONNX Model Input
```

**For root body, prefer qpos over xquat:**
```
data.qpos[3:7]  (canonical)  ✓ PREFERRED
  ↓
wxyz → xyzw
  ↓
body_rot[0]

data.xquat[1]   (FK-computed)  ✗ Has rounding error
```

---

## Body Indexing (Another Thing That Gets People)

```
MuJoCo:           Python/ProtoMotions:
data.xquat[0]  =  World (skip this)
data.xquat[1]  =  Body 0 (pelvis, root)
data.xquat[2]  =  Body 1
...
data.xquat[33] =  Body 32

↓ In code ↓
body_rot_wxyz = data.xquat[1:]       # [num_bodies, 4]
body_rot = mujoco_wxyz_to_xyzw(...)  # [num_bodies, 4] in xyzw
```

**The +1 offset is built in:** `data.xquat[i + 1]` for body i.

---

## Main Loop at 50 Hz

```
Per control step:

1. robot_state = get_robot_state_from_mujoco(model, data)
   ├─ dof_pos, dof_vel
   ├─ body_rot (all 33 bodies, xyzw)
   └─ root_local_ang_vel

2. future_refs = player.get_future_references(...)
   ├─ 25 steps ahead (0.5 seconds)
   ├─ body_rot, dof_pos, dof_vel for each future step
   └─ Pre-interpolated to control frequency

3. onnx_inputs = build_onnx_inputs(robot_state, future_refs, ...)
   ├─ Add batch dim [1, ...] to everything
   ├─ Extract anchor_rot = body_rot[16]
   └─ Assemble semantic keys for ONNX

4. pd_targets = ort.run(session, onnx_inputs)
   └─ Output is [num_dofs] PD position targets

5. pd_targets = apply_accel_clamp(pd_targets, prev_pd, prev_prev_pd)
6. pd_targets = apply_ema_filter(pd_targets, prev_pd, alpha=0.2)
7. data.ctrl[:] = pd_targets

8. mujoco.mj_step(model, data)  ×4  (substeps, decimation=4)

9. → Back to step 1
```

---

## Coordinate Frame Summary

| Item | Frame | Notes |
|------|-------|-------|
| DOF positions | Joint-space | Not world XYZ |
| DOF velocities | Joint-space | Angular velocities |
| Body rotations | World-frame | 6D via xyzw quaternion |
| Root pos (XYZ) | World-frame | In `data.qpos[0:3]`, not exported |
| Root ang. vel | Body-local | Already body-local, no rotation needed |
| Anchor rot | World-frame | Used for policy obs |

**Key insight:** Observations are **position-invariant** — no XYZ coordinates!
This is why motion realignment isn't needed in deployment.

---

## Anchor Body Concept

**Anchor body = IMU frame for policy observation**

- For G1: `anchor_body_index = 16` (torso_link)
- Extracted: `anchor_rot = body_rot[16]` (world-frame xyzw)
- Sent to ONNX: `current_state_anchor_rot` [1, 4]
- Purpose: Primary reference frame for upper-body tracking

**Different from root:**
- Root (pelvis): Kinematic chain root, free-joint, foot contact
- Anchor (torso): IMU-like observation, human-interpretable

---

## Input Dimensions (G1 Example)

```
ONNX Batch Inputs:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current State (batch=1):
  current_state_dof_pos              [1, 29]
  current_state_dof_vel              [1, 29]
  current_state_anchor_rot           [1, 4]
  current_state_root_local_ang_vel   [1, 3]
  historical.processed_actions       [1, 1, 29]

Future References (batch=1, horizon=25):
  mimic.future_rot                   [1, 25, 33, 4]  ← ALL bodies
  mimic.future_anchor_rot            [1, 25, 4]      ← Extracted from above
  mimic.future_dof_pos               [1, 25, 29]
  mimic.future_dof_vel               [1, 25, 29]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output:
  joint_pos_targets                  [1, 29]  ← After PD & scaling
  actions                            [1, 29]  ← Raw (for debugging)
  stiffness_targets                  [1, 29]  ← Per-DOF (const)
  damping_targets                    [1, 29]  ← Per-DOF (const)
```

---

## Post-Processing Pipeline

```
ONNX Output: pd_targets [29]
    ↓
if use_accel_clamp:
    delta = pd_targets - prev_pd
    prev_delta = prev_pd - prev_prev_pd
    accel = delta - prev_delta
    accel = clip(accel, -max_accel, +max_accel)
    pd_targets = prev_pd + prev_delta + accel
    ↓
if use_ema:
    pd_targets = alpha × pd_targets + (1-alpha) × prev_pd
    ↓
data.ctrl[:] = pd_targets
    ↓
Physics Engine (4 substeps)
```

**Why both?**
- Accel clamp: Limits jerky movements (matches sim)
- EMA filter: Low-pass smoothing (matches sim)
- Both are **not** baked into ONNX—applied at deployment

---

## Initialization: `set_initial_pose(model, data, motion_player)`

```python
frame0 = motion_player.get_state_at_frame(0)  # Get frame 0 data

# Root position (absolute, world-frame)
data.qpos[0:3] = frame0["body_pos"][0]        # [x, y, z]

# Root orientation (must convert xyzw → wxyz for MuJoCo)
root_quat_xyzw = frame0["body_rot"][0]        # [x, y, z, w]
root_quat_wxyz = root_quat_xyzw[[3, 0, 1, 2]] # [w, x, y, z]
data.qpos[3:7] = root_quat_wxyz

# Joint angles
data.qpos[7:] = frame0["dof_pos"]             # [theta_1, ..., theta_29]

# Zero initial velocity
data.qvel[:] = 0.0

# Recompute all FK (updates data.xquat for all bodies)
mujoco.mj_forward(model, data)
```

**Key:** Motion file uses xyzw; MuJoCo expects wxyz. Index swap: `[[3, 0, 1, 2]]`.

---

## Common Mistakes → Silent Failures

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Use `data.cvel[0, 0:3]` for root ang. vel | Asymmetric tracking (looks fine until it doesn't) | Use `data.qvel[3:6]` |
| Forget +1 offset on xquat | Rotations offset by one body | Use `data.xquat[1 + body_id]` |
| Quaternion wxyz ↔ xyzw mismatch | Weird pose flipping | Always convert at boundary |
| Mix root & anchor bodies | Works but breaks at boundaries | Root=pelvis (0), Anchor=torso (16) |
| Don't call `mj_forward()` after init | xquat isn't updated | Call immediately after setting qpos |
| Use xquat for root instead of qpos | FK rounding → slight tracking error | Prefer `data.qpos[3:7]` for root |

---

## At a Glance: State Extraction

```python
# Step 1: Get all wxyz quaternions from MuJoCo
body_rot_wxyz = data.xquat[1:].copy()      # Skip world body at [0]

# Step 2: Convert to xyzw (ProtoMotions standard)
body_rot = mujoco_wxyz_to_xyzw(body_rot_wxyz)

# Step 3: Override root with qpos (more canonical)
body_rot[0] = mujoco_wxyz_to_xyzw(data.qpos[3:7])

# Step 4: Collect everything
state = {
    "dof_pos":            data.qpos[7:],     # Joint angles
    "dof_vel":            data.qvel[6:],     # Joint velocities
    "body_rot":           body_rot,           # All bodies, xyzw
    "root_local_ang_vel": data.qvel[3:6],    # Pelvis ang. vel, body-local
}

# Step 5: Extract anchor for policy
anchor_rot = state["body_rot"][16]  # Torso rotation, xyzw
```

**That's it.** Everything else is packing these into ONNX batches.

---

## Files to Reference

- **`test_tracker_mujoco.py`** (lines 327-370): `get_robot_state_from_mujoco()` — core extraction
- **`test_tracker_mujoco.py`** (lines 373-392): `set_initial_pose()` — initialization
- **`test_tracker_mujoco.py`** (lines 400-439): `build_onnx_inputs()` — packing for ONNX
- **`test_tracker_mujoco.py`** (lines 660-776): Main loop — full pipeline
- **`state_utils.py`**: `mujoco_wxyz_to_xyzw()` — quaternion conversion
- **`deployment/export_bm_tracker_onnx.py`**: ONNX export reference

