# Technical Analysis: ONNX Tracker + MuJoCo Physics Simulation Pipeline

## Executive Summary

The tracker export system (`run_tracker_export.py`) is a **closed-loop physics simulation** that:
1. Loads a **reference motion cache** (kinematic FK poses from SMPL/motion retargeting)
2. Runs an **ONNX policy** to generate PD target joint angles in real-time
3. Simulates the **G1 humanoid robot** in MuJoCo with full physics (gravity, contact, inertia)
4. Exports the **physically realistic motion** to a new cache file

The key insight: **Reference motion** = clean kinematic data. **Tracked motion** = physics-respecting simulation. If the motion is dynamically implausible, the robot falls; if it's sound, it tracks closely.

---

## 1. MuJoCo Model Setup

### 1.1 XML Model Files
- **Default**: `g1_holo_compat.xml` (33 bodies, 29 DOF free-joint humanoid)
- **Structure**:
  - **Free-joint** at root (6 DOF: 3 position + 3 rotation in quat)
  - **29 actuated joints** (legs, torso, arms, wrists)
  - **Root body**: `pelvis` (index 0) — position + orientation in `qpos[0:7]`
  - **Anchor body**: `torso_link` (index 16) — IMU reference for orientation
  
### 1.2 Physics Parameters
From `unified_pipeline.yaml`:

| Parameter | Value | Role |
|-----------|-------|------|
| **control_dt** | 0.02s (50 Hz) | Policy update rate, also export frame rate |
| **physics_dt** | 0.001s (1000 Hz) | MuJoCo substep rate |
| **decimation** | 20 | Physics substeps per policy step: 20 × 0.001s = 0.02s |
| **gravity** | [0, 0, -9.81] m/s² | Applied by MuJoCo default |

### 1.3 Friction & Contact
- **MuJoCo default friction** (from MJCF): Each geom has friction, density, contact parameters
- **No explicit contact processing** in the export code — pure MuJoCo physics
- **Feet** have collision geoms (capsules) to prevent sliding through ground

### 1.4 XML Patching
`_patch_mjcf_xml()` strips sensors and adds:
- **Ground plane** if missing: `type="plane"` at z=0
- **Lighting** for rendering consistency

---

## 2. ONNX Tracker Policy

### 2.1 Model Inputs (8 tensors)
All inputs reshaped to batch size 1:

| Input | Shape | Type | Description |
|-------|-------|------|-------------|
| **current_dof_pos** | [1, 29] | float32 | Joint angles (radians) from `data.qpos[7:]` |
| **current_dof_vel** | [1, 29] | float32 | Joint velocities from `data.qvel[6:]` |
| **current_anchor_rot** | [1, 4] | xyzw quat | Torso rotation (anchor body, index 16) |
| **current_root_local_ang_vel** | [1, 3] | float32 | Pelvis angular velocity in **local frame** |
| **historical_processed_actions** | [1, 1, 29] | float32 | Previous PD targets (used for smoothing) |
| **mimic_future_anchor_rot** | [1, 4, 4] | xyzw quat | Future torso rotations at steps [1,2,4,8] |
| **mimic_future_dof_pos** | [1, 4, 29] | float32 | Future joint angles at steps [1,2,4,8] |
| **mimic_future_dof_vel** | [1, 4, 29] | float32 | Future joint velocities at steps [1,2,4,8] |

### 2.2 Future Step Indices
```
[1, 2, 4, 8] steps ahead of current frame
× 0.02s control_dt = [0.02, 0.04, 0.08, 0.16] seconds ahead
```

### 2.3 Model Outputs (4 tensors)
| Output | Shape | Purpose |
|--------|-------|---------|
| **actions** | [1, 29] | Raw actions (auxiliary output) |
| **joint_pos_targets** | [1, 29] | **Primary**: PD target joint angles |
| **stiffness_targets** | [1, 29] | Adaptive stiffness per joint |
| **damping_targets** | [1, 29] | Adaptive damping per joint |

### 2.4 Policy Input Construction (Lines 346-392)
```python
# 1. Extract current state from MuJoCo
robot_state = {
    "dof_pos": data.qpos[7:],          # Current joint angles
    "dof_vel": data.qvel[6:],          # Current joint velocities
    "body_rot": body_rot_array,        # All 33 body quaternions
    "root_local_ang_vel": data.qvel[3:6]  # Local frame (from free joint)
}

# 2. Compute anchor rotation (torso IMU)
anchor_rot = compute_anchor_rot_np(robot_state["body_rot"], 16)

# 3. Heading alignment (first step only)
heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)

# 4. Get future references from motion cache
future_refs = player.get_future_references(frame_idx, [1, 2, 4, 8])
future_refs["body_rot"] = apply_heading_offset_np(heading_offset, ...)

# 5. Build ONNX input dict
onnx_inputs = {
    "current_dof_pos": robot_state["dof_pos"][None],
    "current_dof_vel": robot_state["dof_vel"][None],
    "current_anchor_rot": anchor_rot[None],
    "current_root_local_ang_vel": robot_state["root_local_ang_vel"][None],
    "historical_processed_actions": prev_actions[None, None],  # [1, 1, 29]
    "mimic_future_anchor_rot": future_anchor_rot[None],
    "mimic_future_rot": future_refs["body_rot"][None],
    "mimic_future_dof_pos": future_refs["dof_pos"][None],
    "mimic_future_dof_vel": future_refs["dof_vel"][None],
}
```

**Key point**: The policy is **tracking-aware** — it gets future references and must generate actions to follow them.

---

## 3. Physics Simulation Loop

### 3.1 Per-Frame Sequence (Lines 319-436)
```
Frame i:
  ├─ Record state BEFORE physics step
  │  ├─ Body positions from data.xpos[1:num_bodies+1]
  │  ├─ Body rotations from data.xquat[1:] or qpos[3:7]
  │  ├─ DOF positions from data.qpos[7:]
  │  ├─ DOF velocities from data.qvel[6:]
  │  ├─ Body velocities from data.cvel[body, 3:6]
  │  └─ Body angular velocities from data.cvel[body, 0:3]
  │
  ├─ Detect fall: root_h < 0.3m → status="fell"
  │
  ├─ Get current robot state (dof_pos, dof_vel, body_rot, root_local_ang_vel)
  │
  ├─ Compute anchor rotation (torso orientation)
  │
  ├─ Fetch future references from motion cache
  │  └─ Apply heading offset to align with current robot orientation
  │
  ├─ Run ONNX inference
  │  ├─ Input: current state + future references
  │  └─ Output: joint_pos_targets (29-dim)
  │
  ├─ Optional: Clamp PD target acceleration if pd_target_max_accel set
  │
  ├─ Optional: EMA smooth targets if action_ema_alpha < 1.0
  │
  ├─ Apply control and step physics
  │  ├─ data.ctrl[:] = pd_targets  (command PD targets)
  │  └─ FOR substep in range(decimation=20):
  │     └─ mujoco.mj_step(model, data)  (0.001s physics update)
  │
  └─ END
```

### 3.2 PD Control Implementation
From `load_mujoco_model_for_sim()` (Lines 161-174):

```python
# For each DOF i:
kp = stiffness[i]  # From YAML (e.g., 40.18 for hip_pitch)
kd = damping[i]    # From YAML (e.g., 2.56 for hip_pitch)

model.actuator_gainprm[i, 0] = kp          # Proportional gain
model.actuator_biastype[i] = 1             # Implicit PD
model.actuator_biasprm[i, 0] = 0.0         # Bias offset
model.actuator_biasprm[i, 1] = -kp         # P term coefficient
model.actuator_biasprm[i, 2] = -kd         # D term coefficient
model.actuator_ctrllimited[i] = 0          # No control limits (commanded value is reference)
```

**MuJoCo implicit PD formula**:
```
τ = kp × (ctrl - q) - kd × qvel
```

Where:
- `ctrl` = PD target from ONNX (data.ctrl[i])
- `q` = current joint position (data.qpos[7+i])
- `qvel` = current joint velocity (data.qvel[6+i])

### 3.3 Passive Forces (Zeroed)
```python
model.jnt_stiffness[:] = 0.0     # No joint-level springs
model.dof_damping[:] = 0.0       # No passive damping
model.dof_frictionloss[:] = 0.0  # No friction loss
```

All damping comes from **explicit kd control**, not physics parameters.

---

## 4. Control Parameters by Joint (G1)

From `g1.py` + `unified_pipeline.yaml`:

| Joint Group | Stiffness (kp) | Damping (kd) | Role |
|-------------|---|---|---|
| **Hip pitch/yaw** (×2) | 40.18 | 2.56 | Upper leg rotation |
| **Hip roll** (×2) | 99.10 | 6.31 | Side leg abduction |
| **Knee** (×2) | 99.10 | 6.31 | Leg extension |
| **Ankle pitch/roll** (×2) | 28.50 | 1.81 | Foot orientation |
| **Waist yaw** | 40.18 | 2.56 | Torso twist |
| **Waist roll/pitch** | 28.50 | 1.81 | Torso bend/lean |
| **Shoulders** (×4) | 14.25 | 0.91 | Arm abduction/pitch |
| **Elbows** (×2) | 14.25 | 0.91 | Elbow flex |
| **Wrists** (×6) | 14.25 / 16.78 | 0.91 / 1.07 | Wrist rotation |

**Stiffness derivation** (BeyondMimic formulas):
```
k = f²m  where f = 10 Hz (natural frequency), m = motor armature
d = 2ζfm  where ζ = 2 (damping ratio, heavily damped)
```

Example hip_pitch:
- armature = 0.01018 kg
- f = 10 Hz, ζ = 2
- k = (2π×10)² × 0.01018 ≈ 40.18
- d = 2 × 2 × (2π×10) × 0.01018 ≈ 2.56

---

## 5. Motion Cache Format & Export

### 5.1 Input Reference Motion (from gmr_to_protomotions)
```python
reference_cache = {
    "dof_pos": (T, 29),           # FK reference joint angles
    "dof_vel": (T, 29),           # Time derivatives
    "body_rot": (T, 33, 4),       # xyzw quaternions, 33 bodies
    "body_pos": (T, 33, 3),       # Forward kinematics positions
    "body_vel": (T, 33, 3),       # Linear velocities
    "body_ang_vel": (T, 33, 3),   # Angular velocities
    "control_dt": 0.02,
    "num_frames": T,
}
```

### 5.2 Output Tracked Motion (from physics simulation)
Same structure, but:
- **body_pos** / **body_rot** come from `mj_step()` output (physically constrained)
- **body_vel** / **body_ang_vel** come from MuJoCo's cvel (consistent with physics)
- Feet don't slide through ground
- If motion is invalid → robot falls → low root height in output

### 5.3 Recording Logic (Lines 320-337)
```python
out_body_pos[frame_idx] = data.xpos[1:num_bodies+1]      # All 33 bodies
out_body_rot[frame_idx] = mujoco_wxyz_to_xyzw(data.xquat[1:])

# For root body, use canonical free-joint quaternion:
out_body_rot[frame_idx, 0] = mujoco_wxyz_to_xyzw(data.qpos[3:7])

out_dof_pos[frame_idx] = data.qpos[7:]
out_dof_vel[frame_idx] = data.qvel[6:]

# Body velocities from cvel: [ang_vel(3), lin_vel(3)]
cvel = data.cvel[1:num_bodies+1]
out_body_ang_vel[frame_idx] = cvel[:, 0:3]  # First 3 elements
out_body_vel[frame_idx] = cvel[:, 3:6]      # Last 3 elements
```

### 5.4 Status Determination (Lines 474-479)
```python
if fall_frame is not None:
    status = "fell"              # Root dropped below 0.3m
elif root_height_min < 0.4:
    status = "unstable"          # Marginal, never fell but very low
else:
    status = "success"           # Root stayed > 0.4m throughout
```

---

## 6. Reward/Tracking Signals (Not Exported)

The export code **does NOT compute tracking rewards** — it only runs inference and records state.

Tracking quality is implicit in the **physics simulation**:
- If ONNX generates good PD targets → robot tracks motion → small state error
- If ONNX generates bad targets → robot drifts or falls

The summary JSON records success/failure metrics:
- `status`: fell / unstable / success
- `fall_frame`: Frame number when root_h < 0.3m (if applicable)
- `root_height_min`: Minimum root height across simulation

---

## 7. YAML Metadata Structure

File: `unified_pipeline.yaml`

```yaml
type: unified_pipeline
dt: 0.02

# Actuator configuration
joint_names: [left_hip_pitch_joint, ..., right_wrist_yaw_joint]  # 29 joints
body_names: [pelvis, head, left_hip_pitch_link, ...]              # 33 bodies
default_joint_stiffness: [40.18, 99.10, ...]                      # Per joint
default_joint_damping: [2.56, 6.31, ...]                          # Per joint

# ONNX I/O specification
policy_inputs:
  - name: current_anchor_rot
    key: current.anchor_rot
    shape: [1, 4]
    kind: anchor_rot
  - ...

policy_outputs:
  - name: joint_pos_targets
    kind: joint_pos_targets
    shape: [1, 29]
    joint_names: [...]
  - ...

# Metadata
robot:
  num_bodies: 33
  num_dofs: 29
  anchor_body_name: torso_link
  anchor_body_index: 16
  root_body_name: pelvis
  root_body_index: 0

control:
  stiffness: [40.18, 99.10, ...]    # kp per DOF
  damping: [2.56, 6.31, ...]        # kd per DOF
  pd_target_max_accel: null         # Optional acceleration clamp
  action_ema_alpha: 1.0             # No EMA filtering (1.0 = no smoothing)

timing:
  control_dt: 0.02                  # 50 Hz policy
  physics_dt: 0.001                 # 1000 Hz MuJoCo
  decimation: 20                    # 20 substeps per policy step

motion:
  future_step_indices: [1, 2, 4, 8]
  future_dt_seconds: [0.02, 0.04, 0.08, 0.16]
```

---

## 8. G1 Robot Structure

From `g1.py`:

| Property | Value |
|----------|-------|
| **Total DOF** | 29 |
| **Legs** | 2 × 6 = 12 DOF (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll) |
| **Torso** | 3 DOF (waist_yaw, waist_roll, waist_pitch) |
| **Arms** | 2 × 7 = 14 DOF (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw) |
| **Anchor body** | torso_link (index 16) — IMU reference |
| **Root body** | pelvis (index 0) — base of free joint |
| **Total bodies** | 33 (includes fixed visual geometries) |
| **Default root height** | 0.8m (standing) |

---

## 9. What Changes for SMPL Humanoid

### 9.1 SMPL Skeletal Structure (from `smpl.py`)

| Property | G1 | SMPL Humanoid | Change |
|----------|----|----|--------|
| **Total DOF** | 29 | ~23 | Fewer arm DOF (3 per shoulder vs 3.5) |
| **Legs** | 2×6=12 DOF | 2×3=6 DOF | Each leg: Hip(3D) + Knee(1D) + Ankle(3D) = 7 → simpler to 3 per leg × 2 |
| **Torso** | 3 DOF | ~6-9 DOF | Spine + Chest + Torso joints more articulated |
| **Arms** | 14 DOF | 12-14 DOF | Fewer wrist DOF (3D wrist not 6D) |
| **MJCF file** | g1_holo_compat.xml | smpl_humanoid.xml | Different mesh, joint topology |
| **Joint names** | left_hip_pitch_joint | L_Hip_x, L_Hip_y, L_Hip_z | Different naming scheme |
| **Body names** | pelvis, torso_link, head | Pelvis, Torso, Head | Different case/naming |

### 9.2 SMPL XML Structure (from smpl_humanoid.xml)
```xml
<body name="Pelvis">
  <joint name="L_Hip_x" axis="1 0 0" stiffness="800" damping="80" />
  <joint name="L_Hip_y" axis="0 1 0" stiffness="800" damping="80" />
  <joint name="L_Hip_z" axis="0 0 1" stiffness="800" damping="80" />
  <body name="L_Knee">
    <joint name="L_Knee_x" axis="1 0 0" stiffness="800" damping="80" />
    <joint name="L_Knee_y" axis="0 1 0" stiffness="800" damping="80" />
    <joint name="L_Knee_z" axis="0 0 1" stiffness="800" damping="80" />
    ...
```

**3D ball joints** (roll/pitch/yaw per joint) instead of G1's sequential axis joints.

### 9.3 SMPL Control Parameters (from `smpl.py`)

| Joint Pattern | Stiffness | Damping | Effort Limit |
|---|---|---|---|
| `.*_(Hip\|Knee\|Ankle)_.*` | 800 | 80 | 500 |
| `.*_Toe_.*` | 500 | 50 | 500 |
| `(Torso\|Spine\|Chest)_.*` | 1000 | 100 | 500 |
| `(Neck\|Head\|.*_Shoulder\|.*_Elbow)_.*` | 500 | 50 | 500 |
| `.*_(Wrist\|Hand)_.*` | 300 | 30 | 500 |

### 9.4 Changes Required

#### **4.4.1 ONNX Model Retraining**
- Current G1 ONNX: 29 DOF input/output
- SMPL ONNX needed: ~23 DOF input/output
- **Retraining required**: Policy was trained on G1 data; SMPL needs new training data + network

#### **4.4.2 Motion Retargeting**
- Reference motion from SMPL model (e.g., from MoCap retargeted to SMPL)
- Must match SMPL DOF count (23) and joint names
- `MotionPlayer` loads via `control_dt` — compatible
- **Change**: Use SMPL cache files with 23 DOF, not G1's 29

#### **4.4.3 YAML Metadata Update**
```yaml
# G1 (current)
robot:
  num_dofs: 29
  num_bodies: 33
  joint_names: [left_hip_pitch_joint, ..., right_wrist_yaw_joint]

# SMPL (new)
robot:
  num_dofs: 23  # or whatever SMPL config uses
  num_bodies: 24  # Pelvis, L/R Hip/Knee/Ankle/Toe, Torso, Spine, Chest, Neck, Head, L/R Shoulder/Elbow/Wrist/Hand
  joint_names: [L_Hip_x, L_Hip_y, L_Hip_z, L_Knee_x, ...]
```

#### **4.4.4 Updated `load_mujoco_model_for_sim()`
```python
# Still works if:
# 1. MJCF path points to smpl_humanoid.xml
# 2. stiffness/damping lists match num_actuators
# 3. Joint names are corrected in iteration

# Example call:
model, data = load_mujoco_model_for_sim(
    mjcf_path="ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml",
    stiffness=[800, 800, 800, 800, ...],  # 23 values
    damping=[80, 80, 80, 80, ...],        # 23 values
    physics_dt=0.001
)
```

#### **4.4.5 Updated `run_tracker_and_export()`
```python
run_tracker_and_export(
    motion_cache_path="path/to/smpl_motion.pt",  # 23 DOF
    output_path="path/to/smpl_tracked.pt",
    onnx_path="path/to/smpl_onnx/unified_pipeline.onnx",  # SMPL-trained ONNX
    mjcf_path="ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml",
)
```

#### **4.4.6 Body Indexing
- **G1**: anchor_body_index=16 (torso_link)
- **SMPL**: anchor_body_index=? (need to count body order in smpl_humanoid.xml)
- From XML snippet, **Torso** likely index ~10-12 depending on tree traversal

#### **4.4.7 Default Root Height
- **G1**: 0.8m
- **SMPL**: 0.95m (from `smpl.py`)
- Used only for visualization/validation, not simulation

---

## 10. Implementation Checklist for SMPL Adaptation

| Task | Current (G1) | SMPL Requirement | Effort |
|------|---|---|---|
| **ONNX policy** | Trained on G1 | Need SMPL-trained ONNX | High (retraining) |
| **Motion cache** | 29 DOF | 23 DOF (or actual count) | Low (retargeting) |
| **MJCF XML** | g1_holo_compat.xml | smpl_humanoid.xml | None (exists) |
| **YAML metadata** | 29 joint_names, 16 anchor_idx | 23 joint_names, ~10 anchor_idx | Low (config) |
| **Stiffness/damping** | G1 values (40, 99, etc.) | SMPL values (800, 500, 1000) | Low (copy from smpl.py) |
| **Body indexing** | 33 bodies, index 16 anchor | ~24 bodies, index ? anchor | Low (count from XML) |
| **Export code** | No changes needed | No changes needed | None |
| **MotionPlayer** | No changes | No changes | None |
| **fall_detection** | root_h < 0.3m | root_h < 0.3m (same) | None |

---

## 11. Key Insights

### 11.1 Why Physics Simulation?
- **Reference motion**: Pure kinematics, may violate physical constraints (sliding feet, impossible accelerations)
- **Tracked motion**: Respects gravity, contact friction, inertia — more realistic
- **Use case**: Evaluate if motion is **dynamically feasible** for real robot execution

### 11.2 ONNX Policy Role
- **Not learned mapping** (reference → tracked): The ONNX is a **tracking controller**
- **Tracks future motion** from the reference cache
- **Looks ahead** 4 steps (0.02, 0.04, 0.08, 0.16s) to preview upcoming targets
- **Closed-loop**: Uses current state + future reference to decide current action

### 11.3 Initialization Sensitivity
- **First frame** is set exactly from reference motion (no dynamics yet)
- **Heading offset** computed on frame 0 to align robot IMU with motion IMU
- **EMA filtering** can smooth actions (if alpha < 1.0) to reduce jitter

### 11.4 Failure Modes
1. **Falls early** (frame < 100): Motion is too dynamic or requires higher stiffness
2. **Unstable** (root_h ∈ [0.3, 0.4]): Marginal — may teeter but complete
3. **Success** (root_h > 0.4): Good tracking, physically plausible

### 11.5 Alignment with Current Codebase
- `run_tracker_export.py` is **independent** of the main HyMotion M2M pipeline
- Uses **ProtoMotions deployment utilities** (state_utils, motion_utils)
- Can be run in **CPU-only mode** (MuJoCo + ONNX Runtime on CPU)
- Complements `convert_cache_to_json.py` → `batch_pipeline_to_web.py` for visualization

---

## 12. Detailed Code Flow

### 12.1 Main Entry Point (`main()`, Lines 593-700)
```
Parse CLI args
  ├─ Single motion mode: --motion, --output
  └─ Batch mode: --motion-dir, --output-dir, --pattern

Resolve ONNX path (relative to repo root)

IF batch mode:
  batch_run(motion_dir, output_dir, onnx_path, mjcf_path, ...)
ELSE:
  run_tracker_and_export(motion_cache_path, output_path, onnx_path, mjcf_path)
```

### 12.2 `run_tracker_and_export()` (Lines 189-491)
```
1. Load YAML metadata (robot config, timing, control)
2. Load ONNX session (InferenceSession on CPU)
3. Load motion cache via MotionPlayer
4. Load MuJoCo model + initialize physics
5. Set initial pose from frame 0 of reference motion
6. Allocate output arrays (num_frames × num_bodies/dofs)

7. FOR each frame in motion:
   a. Record state BEFORE physics step
   b. Fall detection
   c. Get current robot state
   d. Compute heading offset (frame 0 only)
   e. Fetch future motion references
   f. Build ONNX inputs
   g. Run ONNX inference → PD targets
   h. Optional: Clamp acceleration
   i. Optional: EMA smooth
   j. Apply control + physics step (decimation substeps)
   k. Log progress every 100 frames

8. Save tracked cache .pt
9. Compute status (fell/unstable/success)
10. Return summary dict
```

### 12.3 Batch Mode (`batch_run()`, Lines 499-585)
```
List all .pt files matching pattern
Filter to max_motions if specified
FOR each file:
  IF skip_existing and output exists:
    Skip
  ELSE:
    run_tracker_and_export() → summary dict
    Append to results list
Write summary JSON
Print summary table
```

---

## Summary Table: Tracker Pipeline vs SMPL Adaptation

| Component | G1 Current | SMPL Required | Notes |
|-----------|---|---|---|
| **Physics Engine** | MuJoCo 3.0+ | MuJoCo 3.0+ | No change |
| **Control Loop** | 50 Hz (0.02s dt) | Same | No change |
| **MJCF Model** | g1_holo_compat.xml | smpl_humanoid.xml | Use existing SMPL MJCF |
| **DOF Count** | 29 | ~23 | Fewer DOF, simpler joint structure |
| **ONNX Policy** | G1-trained (29 in/out) | SMPL-trained (23 in/out) | **Must retrain** on SMPL motion data |
| **Motion Cache** | 29 DOF motion files | 23 DOF motion files | Retarget ref motion to SMPL |
| **Stiffness/Damping** | From BeyondMimic | From smpl.py config | Update in load_mujoco_model_for_sim() |
| **Anchor Body** | torso_link (idx 16) | Torso or Chest (idx ~10-12) | Recompute from SMPL MJCF |
| **Export Format** | (T, 33, 4) rotations | (T, 24, 4) rotations | Adapt num_bodies in YAML |
| **Fall Threshold** | 0.3m | 0.3m | Probably same |
| **Export Code** | run_tracker_export.py | Same code works | Just update config paths |

---

## Conclusion

The `run_tracker_export.py` system is a **self-contained physics sim evaluator** that:
1. Takes reference (kinematic) motion
2. Runs learned tracking policy via ONNX
3. Simulates with full physics
4. Exports physically plausible trajectory

**To adapt for SMPL humanoid**:
1. ✅ Use smpl_humanoid.xml (exists)
2. ⚠️ **Retrain ONNX on SMPL data** (biggest work)
3. ✅ Retarget reference motions to SMPL (uses existing tools)
4. ✅ Update YAML config (stiffness, damping, body indices)
5. ✅ No code changes to export pipeline

**Timeline**: Export code = ready. ONNX policy = requires new training data + training run. Motion retargeting = can use existing scripts.
