# Technical Analysis: Tracker Export System & Physics Simulation

## Executive Summary

This document provides a complete technical breakdown of `scripts/embodied/run_tracker_export.py` — a **closed-loop physics simulation system** that runs an ONNX tracker policy in MuJoCo and exports physically-realistic body states. The system is **G1-robot-specific** but the architecture can be adapted for SMPL humanoids with key modifications to joint configuration, control parameters, and coordinate representations.

---

## 1. MuJoCo Simulation Setup

### 1.1 Model Loading & XML Patching

**XML Source:**
- Default: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml`
- Alternative models available: `g1_bm.xml`, `g1_holo.xml`, `smpl_humanoid.xml`, `smplx_humanoid.xml`

**Patching Process** (`_patch_mjcf_xml`):
1. Removes all `<sensor>` elements (MuJoCo simulation doesn't need them; reduces computation)
2. Adds ground plane if missing:
   - Type: `plane` geometry
   - Size: `0 0 0.05` (thin plane, 5cm z-height)
   - RGBA: Gray `0.7 0.7 0.7 1`
3. Adds directional light if missing:
   - Position: `2 0 5.0`
   - Direction: `0 0 -1` (pointing down)

**Why patch?** MuJoCo deployment doesn't need the full sensor suite; removing sensors reduces memory/compute overhead. The ground plane ensures stable foot contact.

### 1.2 Physics Parameters

**Timestep Configuration:**
```yaml
physics_dt: 0.001          # 1 ms physics step
control_dt: 0.02           # 20 ms control/observation frame (50 Hz)
decimation: 20             # mujoco.mj_step() called 20× per control step
effective_Hz: 50           # control_dt period
```

**Passive Forces (Zeroed for Clean Simulation):**
```python
model.jnt_stiffness[:] = 0.0      # No passive joint stiffness
model.dof_damping[:] = 0.0        # No passive viscous damping
model.dof_frictionloss[:] = 0.0   # No dry friction
```
All motion control comes from **PD actuators only**, matching training conditions.

**PD Actuators (Per-Joint):**
Each joint i has:
- `gainprm[i, 0]` = K_p (stiffness)
- `biasprm[i, 0]` = 0.0 (no constant bias)
- `biasprm[i, 1]` = -K_p (implicit spring zero point)
- `biasprm[i, 2]` = -K_d (implicit damping coefficient)
- `ctrllimited[i]` = 0 (no torque saturation in simulation)

The MuJoCo PD controller implements:
```
τ = K_p · (q_target - q_current) - K_d · q̇
```

### 1.3 G1 Robot Structure

**From YAML metadata (unified_pipeline.yaml):**

**DOF Count:** 29 joints
```
Legs (12):
  - left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint
  - left_knee_joint
  - left_ankle_pitch_joint, left_ankle_roll_joint
  [× right side]

Waist (3):
  - waist_yaw_joint, waist_roll_joint, waist_pitch_joint

Arms (14):
  - left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw
  - left_elbow_joint
  - left_wrist_roll, left_wrist_pitch, left_wrist_yaw
  [× right side]
```

**Body Count:** 33 bodies
- Root: `pelvis` (index 0, free joint → 7 DOFs in qpos: [x,y,z, w,x,y,z])
- Anchor: `torso_link` (index 16, for IMU observations)
- End effectors: `left/right_ankle_roll_link`, `left/right_rubber_hand`

**Key Body Hierarchy:**
```
pelvis (pos=0, 0, 0.793 m)
├── head (pos=0, 0, 0.4 rel)
├── left_hip_pitch_link
│   └── left_hip_roll_link
│       └── left_hip_yaw_link
│           └── left_knee_link
│               └── left_ankle_pitch_link
│                   └── left_ankle_roll_link ← foot contact
└── ... (right leg mirror, waist, arms)
```

**Stiffness & Damping by Joint Type:**
```
Hip (pitch/yaw):      K_p=40.2,   K_d=2.56
Hip (roll):           K_p=99.1,   K_d=6.31
Knee:                 K_p=99.1,   K_d=6.31
Ankle:                K_p=28.5,   K_d=1.81
Shoulder/elbow:       K_p=14.3,   K_d=0.91
Wrist (roll/yaw):     K_p=16.8,   K_d=1.07
Wrist (pitch):        K_p=16.8,   K_d=1.07
```
Derived from: `K_p = m·ω_n²`, `K_d = 2·ζ·m·ω_n` with `ω_n=10 Hz`, `ζ=2.0` (overdamped).

---

## 2. ONNX Tracker Policy

### 2.1 Model Path & Format

**Default model:**
```
ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/
├── compiled_models/
│   ├── unified_pipeline.onnx    (22.6 MB)
│   └── unified_pipeline.yaml    (metadata)
├── last.ckpt                    (238 MB, full checkpoint)
└── resolved_configs_inference.pt
```

**Model type:** "unified_pipeline" — a feed-forward Transformer-based motion tracker trained via reinforcement learning (PPO + motion imitation loss).

### 2.2 Input Specification

**8 inputs, all batch-dim=1:**

```yaml
1. current_anchor_rot        [1, 4]           xyzw quaternion of torso_link
2. current_dof_pos           [1, 29]          joint angles (radians)
3. current_dof_vel           [1, 29]          joint angular velocities
4. current_root_local_ang_vel[1, 3]           angular velocity in pelvis frame
5. historical_processed_actions[1, 1, 29]     previous action (for recurrence)
6. mimic_future_anchor_rot   [1, 4, 4]        future torso rotations @ steps [1,2,4,8]
7. mimic_future_dof_pos      [1, 4, 29]       future joint targets @ steps [1,2,4,8]
8. mimic_future_dof_vel      [1, 4, 29]       future joint velocities
```

**Keyframe times:** Steps [1, 2, 4, 8] correspond to:
- 1 step → 0.02 s
- 2 steps → 0.04 s
- 4 steps → 0.08 s
- 8 steps → 0.16 s

The model **looks 160ms ahead** to plan smoother tracking.

### 2.3 Output Specification

**4 outputs:**

```yaml
1. actions              [1, 29]    raw policy outputs (unused in tracker export)
2. joint_pos_targets    [1, 29]    **PRIMARY OUTPUT** — PD target positions
3. stiffness_targets    [1, 29]    per-joint K_p values (optional override)
4. damping_targets      [1, 29]    per-joint K_d values (optional override)
```

**In the tracker export:**
- Uses `joint_pos_targets` (ort_out[1]) for MuJoCo control
- Ignores `stiffness_targets` and `damping_targets` (uses config-file values)

### 2.4 Input Construction

**From robot state + motion references:**

```python
# Step 1: Extract current robot state from MuJoCo
robot_state = {
    "dof_pos": data.qpos[7:],              # 29 joint angles
    "dof_vel": data.qvel[6:],              # 29 joint velocities
    "body_rot": body_rotations_xyzw,       # 33 bodies, xyzw format
    "root_local_ang_vel": data.qvel[3:6],  # FREE JOINT local ang vel (already local!)
}

# Step 2: Compute anchor rotation (torso IMU frame)
anchor_rot = robot_state["body_rot"][16]  # torso_link quaternion

# Step 3: Load future motion references from MotionPlayer
future_refs = player.get_future_references(
    frame_idx, 
    future_step_indices=[1, 2, 4, 8]  # 4 keyframes
)
# Returns: {body_rot: [4, 33, 4], dof_pos: [4, 29], dof_vel: [4, 29]}

# Step 4: Heading alignment
# Compute yaw offset between robot and motion (first frame only)
heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)
# Apply to all future references (rotate to robot's heading)
future_refs["body_rot"] = apply_heading_offset_np(heading_offset, future_refs["body_rot"])

# Step 5: Build ONNX inputs
onnx_inputs = {
    "current_anchor_rot": anchor_rot[None],                        # [1, 4]
    "current_dof_pos": robot_state["dof_pos"][None],              # [1, 29]
    "current_dof_vel": robot_state["dof_vel"][None],              # [1, 29]
    "current_root_local_ang_vel": robot_state["root_local_ang_vel"][None],  # [1, 3]
    "historical_processed_actions": prev_actions[None, None],     # [1, 1, 29]
    "mimic_future_anchor_rot": future_refs["body_rot"][:, 16, :][None],    # [1, 4, 4]
    "mimic_future_dof_pos": future_refs["dof_pos"][None],         # [1, 4, 29]
    "mimic_future_dof_vel": future_refs["dof_vel"][None],         # [1, 4, 29]
}
```

**Heading alignment rationale:** The motion (reference) may be facing a different yaw direction than the current robot. Heading offset corrects this by rotating all future body references into the robot's current heading frame. This allows the policy to track rotations relative to the robot's body frame.

### 2.5 Inference Loop

```python
# Each control frame (every 20ms):
for frame_idx in range(num_frames):
    # 1. Record current state
    out_body_pos[frame_idx] = data.xpos[1:num_bodies+1]
    out_body_rot[frame_idx] = mujoco_wxyz_to_xyzw(data.xquat[1:num_bodies+1])
    out_dof_pos[frame_idx] = data.qpos[7:]
    out_dof_vel[frame_idx] = data.qvel[6:]
    out_body_vel[frame_idx] = data.cvel[1:num_bodies+1, 3:6]
    out_body_ang_vel[frame_idx] = data.cvel[1:num_bodies+1, 0:3]
    
    # 2. Construct ONNX inputs (as above)
    onnx_inputs = {...}
    
    # 3. Run inference
    ort_out = session.run(onnx_output_names, onnx_inputs)
    pd_targets = ort_out[1].squeeze()  # [29] joint position targets
    
    # 4. Optional: clamp acceleration (smooth targets)
    if pd_target_max_accel is not None:
        accel = (pd_targets - prev_pd) - (prev_pd - prev_prev_pd)
        accel_clamped = np.clip(accel, -max_accel, max_accel)
        pd_targets = prev_pd + (prev_pd - prev_prev_pd) + accel_clamped
    
    # 5. Optional: EMA filtering
    if action_ema_alpha < 1.0:
        pd_targets = alpha * pd_targets + (1 - alpha) * ema_prev
    
    # 6. Set MuJoCo control and step
    data.ctrl[:] = pd_targets  # MuJoCo applies PD control
    for _ in range(decimation):  # decimation=20
        mujoco.mj_step(model, data)  # 1 ms physics step
```

---

## 3. Physics Simulation: Frame-by-Frame Execution

### 3.1 State Recording

**At EVERY control frame (50 Hz):**

```python
# MuJoCo model.xpos: body COM positions (world frame)
out_body_pos[frame_idx] = data.xpos[1:34]  # shape: [33, 3]

# MuJoCo model.xquat: body orientations (wxyz format)
body_rot_wxyz = data.xquat[1:34]
out_body_rot[frame_idx] = mujoco_wxyz_to_xyzw(body_rot_wxyz)  # convert to xyzw

# DOF state
out_dof_pos[frame_idx] = data.qpos[7:]     # shape: [29]
out_dof_vel[frame_idx] = data.qvel[6:]     # shape: [29]

# Body velocities from cvel: [3 ang_vel, 3 lin_vel]
cvel = data.cvel[1:34]  # world-frame angular velocity
out_body_ang_vel[frame_idx] = cvel[:, 0:3]
out_body_vel[frame_idx] = cvel[:, 3:6]
```

**Important:** These represent **the state BEFORE the control step** — the current frame where the policy computes the next action.

### 3.2 Physics Integration

**Per control frame (20 ms):**

```
MuJoCo control input: data.ctrl[29] = joint_pos_targets

For decimation=20:
  For i in 0..19:
    mujoco.mj_step(model, data)  # Integrate 1 ms
    
    Internally:
      1. Compute control forces from PD:
         τ = K_p*(q_target - q_current) - K_d*q̇
      2. Add gravity
      3. Compute contacts with ground
      4. Solve constraint dynamics
      5. Integrate: q_new = q + dt*v; v_new = v + dt*a
      6. Update xpos, xquat, xvel, xangvel
```

**Result:** After 20 ms real-time, robot has moved through 20 physics steps, settling into a new equilibrium under the PD control targets.

### 3.3 Contact Dynamics

The G1 XML includes foot collision capsules (7 per foot) for foot-ground interaction:
```xml
<geom name="left_foot1_collision" type="capsule" size="0.01" 
      fromto="0.1 -0.026 -0.025 0.05 -0.027 -0.025" contype="1" conaffinity="1" />
<!-- ... 6 more foot capsules ... -->
```

**Contact parameters (implicit):**
- Friction: `tangential_friction=1.0` (default)
- Restitution: 0 (inelastic)
- Margin: `0.001` (penetration tolerance)

When feet contact ground:
1. Normal impulses prevent interpenetration
2. Friction forces prevent foot sliding
3. Ground reaction forces appear in constraint forces

This is why the tracked motion appears **physically plausible** — feet don't slide through the ground as they would in pure FK.

---

## 4. Reward/Tracking Signals

### 4.1 What's Being Optimized?

The ONNX model was **trained via RL (PPO) with motion imitation rewards**:

```
Total_reward = w_pose * L_pose + w_shape * L_shape + w_vel * L_vel + ...
```

Where:
- **L_pose**: Difference between robot joint angles and reference motion → minimizes pose error
- **L_shape**: (if applicable) Body/end-effector tracking
- **L_vel**: Joint velocity error → ensures dynamics match reference
- **L_gait**: (if applicable) Foot contact timing
- **Regularization**: Action smoothness, torque minimization

### 4.2 Tracking Quality Metrics

The system monitors **fall detection** during simulation:

```python
FALL_HEIGHT_THRESHOLD = 0.3  # meters

# During simulation:
root_h = data.qpos[2]  # pelvis height
if root_h < 0.3:
    fall_frame = frame_idx
    status = "fell"

# Post-simulation:
root_height_min = min(root_h for all frames)
if root_height_min < 0.4:  # 40 cm
    status = "unstable"
else:
    status = "success"
```

Higher `root_height_min` → more stable motion → motion is more physically plausible.

### 4.3 What's NOT Directly Tracked

- **Ground reaction forces (GRF):** The system doesn't explicitly optimize for realistic foot forces. These emerge implicitly from contact dynamics.
- **Muscle energy:** No model of metabolic cost or muscle activation.
- **Balance:** Explicitly, only via pose tracking. A motion that's off-balance won't fall if the PD control is strong enough, but it will diverge from the reference.

---

## 5. Results Export Format

### 5.1 Output Cache File Structure

```python
# File: output_path (e.g., "tracked_00000.pt")
torch.save({
    "dof_pos": np.ndarray([num_frames, 29], dtype=float32),
    "dof_vel": np.ndarray([num_frames, 29], dtype=float32),
    "body_rot": np.ndarray([num_frames, 33, 4], dtype=float32),  # xyzw
    "body_pos": np.ndarray([num_frames, 33, 3], dtype=float32),
    "body_vel": np.ndarray([num_frames, 33, 3], dtype=float32),
    "body_ang_vel": np.ndarray([num_frames, 33, 3], dtype=float32),
    "control_dt": 0.02,  # control period in seconds
    "num_frames": num_frames,  # total frames
}, output_path)
```

**Comparison with Reference Cache:**

| Field | Reference (FK only) | Tracked (Physics) |
|-------|---------------------|-------------------|
| `body_pos` | From FK, exact match to reference | Differs due to contact constraints |
| `body_rot` | From FK, exact match to reference | May differ if motion is off-balance |
| `body_vel` | Computed from motion interpolation | From MuJoCo integration |
| `dof_pos` | From interpolation | From PD control targeting |

### 5.2 File Size

Typical: 0.1–10 MB per motion depending on length.
```
Size = num_frames * (29*4 + 29*4 + 33*4*4 + 33*3*4 + 33*3*4 + 33*3*4) bytes
     = num_frames * (116 + 4224 + 396 + 396 + 396) bytes
     ≈ num_frames * 5.5 KB
```

### 5.3 Status Summary

Each export produces a `tracker_summary.json`:
```json
[
  {
    "id": "pipeline_00000",
    "status": "success|fell|unstable|error",
    "num_frames": 1000,
    "fall_frame": null,
    "root_height_min": 0.85,
    "duration_s": 20.0,
    "sim_time_s": 3.2,
    "output_path": ".../tracked_00000.pt"
  },
  ...
]
```

---

## 6. MuJoCo Environment Physics Configuration

### 6.1 Physics Solver Parameters

(From MJCF or model defaults, not explicitly shown in script):

```
Solver:
  - type: PGS (projected Gauss-Seidel)
  - iterations: depends on MJCF <option>
  
Contact:
  - margin: 0.001 m (1 mm)
  - friction combine: average
  
Gravity:
  - [0, 0, -9.81] m/s² (standard Earth gravity)
```

### 6.2 Timestep Relationship

```
Real time = 0.02 s (control frame)
Physics dt = 0.001 s (integration step)
Decimation = 20
Total physics steps = 20

Total simulated time = 20 * 0.001 = 0.02 s ✓
```

### 6.3 Initial Conditions

```python
# Frame 0 pose from reference motion
frame0 = player.get_state_at_frame(0)
root_pos = frame0["body_pos"][0]         # [3] meters
root_quat_xyzw = frame0["body_rot"][0]   # [4] quaternion
dof_pos_0 = frame0["dof_pos"]            # [29] radians

# Set MuJoCo state
data.qpos[0:3] = root_pos
data.qpos[3:7] = root_quat_xyzw[[3, 0, 1, 2]]  # xyzw → wxyz (MuJoCo format!)
data.qpos[7:] = dof_pos_0
data.qvel[:] = 0.0  # Start from rest

# Forward kinematics
mujoco.mj_forward(model, data)  # Compute xpos, xquat, etc.
```

**Note:** Velocities are NOT initialized from the motion reference. The robot starts from rest and the policy quickly accelerates to match the motion. This is more realistic for physical deployment.

---

## 7. G1 Robot Specifics

### 7.1 Joint Configuration Details

**Full DOF List (29 total):**

```
0-1:   left_hip_pitch, left_hip_roll
2:     left_hip_yaw
3:     left_knee
4-5:   left_ankle_pitch, left_ankle_roll
6-7:   right_hip_pitch, right_hip_roll
8:     right_hip_yaw
9:     right_knee
10-11: right_ankle_pitch, right_ankle_roll
12:    waist_yaw
13-14: waist_roll, waist_pitch
15-17: left_shoulder_pitch/roll/yaw
18:    left_elbow
19-21: left_wrist_roll/pitch/yaw
22-24: right_shoulder_pitch/roll/yaw
25:    right_elbow
26-28: right_wrist_roll/pitch/yaw
```

### 7.2 Action Space

**Dimensionality:** 29D (one per DOF)
**Range:** Typically [-1, 1] in normalized policy output, then scaled to joint ranges (from MJCF `limited="true" range="..."`)

### 7.3 Key Body Indices

```
Index 0:  pelvis (root, free joint)
Index 1:  head
Index 16: torso_link (anchor for observations)
Index 7:  left_ankle_roll_link (foot)
Index 13: right_ankle_roll_link (foot)
Index 24: left_rubber_hand
Index 32: right_rubber_hand
```

---

## 8. Adaptation for SMPL Humanoid

### 8.1 Key Differences: G1 vs SMPL

| Aspect | G1 | SMPL |
|--------|----|----|
| **DOF Count** | 29 | 63 (22 body dofs × 3 euler angles) |
| **Body Count** | 33 | ~23 bodies (SMPL skeleton) |
| **Feet** | 2 roll joints + contact capsules | 1 joint per toe, simpler geometry |
| **Hands** | 7 dofs total (3D wrist) | Often 15+ dofs (finger articulation) |
| **Torso** | Waist (3) + chest articulation | Spine (3 segments) + chest |
| **Head** | Fixed (IMU body) | 3 dofs (neck) + head |
| **Default height** | 0.8 m | 0.95 m (taller humanoid) |

### 8.2 SMPL XML Structure (Already Exists!)

File: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml`

**Key differences in SMPL XML:**

```xml
<!-- Each joint in SMPL has 3 dofs (x, y, z rotations) -->
<body name="L_Hip" pos="-0.0068 0.0695 -0.0914">
  <joint name="L_Hip_x" axis="1 0 0" stiffness="800" damping="80" />
  <joint name="L_Hip_y" axis="0 1 0" stiffness="800" damping="80" />
  <joint name="L_Hip_z" axis="0 0 1" stiffness="800" damping="80" />
  <body name="L_Knee" ...>
    <joint name="L_Knee_x" ... />
    <joint name="L_Knee_y" ... />
    <joint name="L_Knee_z" ... />
```

Control parameters already defined in `smpl.py`:
```python
".*_(Hip|Knee|Ankle)_.*": ControlInfo(
    stiffness=800,
    damping=80,
    effort_limit=500,
    velocity_limit=100,
)
```

### 8.3 Required Changes for SMPL Tracker Export

1. **YAML Metadata:** Create `unified_pipeline_smpl.yaml` with:
   - `num_dofs`: 63 (instead of 29)
   - `num_bodies`: 23 (instead of 33)
   - `body_names`: SMPL body list (Pelvis, L_Hip, ... Head)
   - `joint_names`: SMPL joint list (L_Hip_x, L_Hip_y, ...)
   - Updated `default_joint_stiffness` and `default_joint_damping`

2. **MJCF Path:** Use `smpl_humanoid.xml` instead of `g1_holo_compat.xml`

3. **Control Configuration:** Use `SmplRobotConfig` from `robot_configs/smpl.py`

4. **Anchor Body:** Change from `torso_link` (index 16) to `Torso` or `Pelvis` (index depends on SMPL structure)

5. **ONNX Model:** Train a new ONNX tracker for SMPL:
   - Input features will have 63 dims (not 29)
   - Output will have 63 action dims
   - Re-train via `protomotions/train_agent.py` with `--robot-name smpl`

6. **Motion Cache Format:** Reference motions must already be in SMPL format (63 dims per frame)

7. **Initial Pose:** `default_dof_pos` from SMPL config is all-zeros or pre-defined standing pose

### 8.4 Step-by-Step Adaptation

```bash
# 1. Create SMPL ONNX tracker (one-time training)
python protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/smpl_bones_mini.pt \
    --num-envs 4096
# Export to ONNX via deployment/export_bm_tracker_onnx.py

# 2. Adapt tracker export script
cp scripts/embodied/run_tracker_export.py scripts/embodied/run_tracker_export_smpl.py
# Changes:
#   - _DEFAULT_MJCF = "...smpl_humanoid.xml"
#   - _DEFAULT_ONNX = "...motion_tracker/smpl-bones/compiled_models/unified_pipeline.onnx"
#   - load_mujoco_model_for_sim: read stiffness/damping from SMPL config

# 3. Run tracker export on SMPL motions
python scripts/embodied/run_tracker_export_smpl.py \
    --motion output/embodied_comparison/data/caches/smpl_pipeline_00000.pt \
    --output output/embodied_comparison/data/tracked_caches/smpl_tracked_00000.pt
```

### 8.5 Known Challenges for SMPL

1. **Higher DOF count:** 63 dims × 4 (future steps) × inference per frame = more compute
2. **Foot contact:** SMPL has simpler foot geometry (single box) vs G1's multi-capsule feet
   - May need explicit contact modeling or post-processing foot-lock
3. **Hand articulation:** If SMPL includes finger DOFs, PD gains need tuning
4. **Coordinate representation:** G1 uses local joint angles; SMPL uses Euler angles (order matters!)
5. **No real robot deployment:** Unlike G1, SMPL humanoid won't run on physical hardware

---

## Summary: What Changes for SMPL

**Minimal changes required:**
- MJCF path: g1_holo_compat.xml → smpl_humanoid.xml
- ONNX model: Retrain for 63 DOFs
- Stiffness/damping: Use SMPL config values
- Robot config: Use SmplRobotConfig

**No changes needed:**
- Physics simulation loop (generic)
- State recording format (works with any DOF count)
- Output cache format (works with any num_bodies)
- Fall detection logic (works with any root height threshold)

The architecture is **robot-agnostic** once the ONNX model and MJCF are prepared.

