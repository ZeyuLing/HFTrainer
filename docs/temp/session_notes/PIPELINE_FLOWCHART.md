# SMPL-to-Robot Retargeting Pipeline: Visual Flowchart & Data Transformations

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HyMotion T2M Inference Output: motion_201 (T, 201)                          │
│ - Includes 22 body joints in a compact representation                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Extract motion_135                                                 │
│ - File: (implicit in HyMotion pipeline)                                     │
│ - Takes first 135 dims from motion_201                                      │
│ - Format: [transl(3) + 22×rot6d(132)]                                       │
│ - Output NPZ: motion_135.npz                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: motion_135 → SMPL-X Conversion                                     │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Script: motion135_to_smplx.py                                          │ │
│ │                                                                          │ │
│ │ Input:  motion_135(T, 135)                                             │ │
│ │         - transl(3)        → kept as-is                                │ │
│ │         - rot6d(T, 22, 6)  → reorder [0,2,4,1,3,5] (row→col major)   │ │
│ │                                                                          │ │
│ │ Process: Gram-Schmidt orthogonalization                                 │ │
│ │         - a1, a2 = rot6d[:,:,:3], rot6d[:,:,3:6]                       │ │
│ │         - b1 = normalize(a1)                                            │ │
│ │         - b2 = normalize(a2 - (a2·b1)b1)                               │ │
│ │         - b3 = b1 × b2                                                  │ │
│ │         - → rotmat (T, 22, 3, 3)                                       │ │
│ │                                                                          │ │
│ │ Output: SMPL-X NPZ                                                      │ │
│ │         - pose_body(T, 63)  : 21 joints × 3 axis-angle                 │ │
│ │         - root_orient(T, 3) : pelvis rotation                          │ │
│ │         - trans(T, 3)       : translation                              │ │
│ │         - betas(10)         : shape params (zeros)                     │ │
│ │         - gender            : "neutral"                                │ │
│ │         - mocap_frame_rate  : 30 fps                                   │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ ⚠️  Potential Issues:                                                        │
│   - Issue #1.2: Small epsilon (1e-8) in Gram-Schmidt normalization         │
│   - Issue #1.3: No validation of output quaternions                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SMPL-X → GMR Retargeting                                           │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Script: gmr_retarget_headless.py                                        │ │
│ │                                                                          │ │
│ │ Input:  SMPL-X NPZ (Y-up frame, 22 body joints)                        │ │
│ │         actual_human_height (auto-detect or override)                  │ │
│ │                                                                          │ │
│ │ Process:                                                                │ │
│ │  1. Load SMPL-X in PyTorch, run through body_model                     │ │
│ │  2. Align FPS (resample if needed) → get_smplx_data_offline_fast()    │ │
│ │  3. Pre-compute ground offset (lowest Z across all frames)             │ │
│ │  4. For each frame:                                                    │ │
│ │     - Scale SMPL-X based on actual_human_height                        │ │
│ │     - Apply pos/rot offsets (body alignment)                           │ │
│ │     - Run GMR IK to target robot (G1)                                  │ │
│ │     - Optionally apply per-frame foot grounding                        │ │
│ │  5. Clamp joint positions to mechanical limits (hard clip)             │ │
│ │                                                                          │ │
│ │ Output: GMR PKL                                                         │ │
│ │         - fps: 30                                                       │ │
│ │         - root_pos(T, 3)    : root position (Z-up frame)               │ │
│ │         - root_rot(T, 4)    : root rotation (xyzw)                     │ │
│ │         - dof_pos(T, 29)    : 29 joint angles                          │ │
│ │         Note: rot_offset is BAKED INTO root_rot                        │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ ⚠️  Critical Issues:                                                         │
│   - Issue #2.1: Ground offset is global min (doesn't account for gait)     │
│   - Issue #2.2: Per-frame foot grounding disabled (comment in code)        │
│   - Issue #2.3: Hard joint limit clipping (no smoothing)                   │
│   - Issue #2.4: No velocity information preserved                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: GMR PKL → ProtoMotions Cache (THE MAIN PROBLEM AREA)              │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Script: gmr_to_protomotions.py                                          │ │
│ │                                                                          │ │
│ │ Input:  GMR PKL (30Hz, Y-up frame with rot_offset baked in)            │ │
│ │                                                                          │ │
│ │ Step 4.1: Coordinate Frame Conversion                                  │ │
│ │  ┌────────────────────────────────────────────────────────────────────┐ │
│ │  │ Convert root_pos from SMPL-X Y-up to MuJoCo Z-up                  │ │
│ │  │ Transformation matrix from rot_offset = [0.5, -0.5, -0.5, -0.5]  │ │
│ │  │ Mapping: [x, y, z]_smplx → [z, x, y]_mujoco                      │ │
│ │  │                                                                    │ │
│ │  │ Remove GMR's rot_offset from root_rot                             │ │
│ │  │ root_rot_corrected = root_rot * rot_offset.inv()                  │ │
│ │  │                                                                    │ │
│ │  │ ⚠️  Issue #3.1, #3.2: Frame conversion correctness                 │ │
│ │  └────────────────────────────────────────────────────────────────────┘ │
│ │                                                                          │ │
│ │ Step 4.2: FK Ground Correction (Optional but Default)                  │ │
│ │  ┌────────────────────────────────────────────────────────────────────┐ │
│ │  │ FOR EACH FRAME t:                                                 │ │
│ │  │  1. Set MuJoCo qpos = [root_pos[t], root_rot[t], dof_pos[t]]      │ │
│ │  │  2. Run mujoco.mj_forward() to compute body positions (FK)        │ │
│ │  │  3. Find minimum foot Z from foot bodies                          │ │
│ │  │  4. Compute z_offset = ground_clearance - min_foot_z             │ │
│ │  │  5. Adjust root_pos[t, 2] += z_offset                            │ │
│ │  │                                                                    │ │
│ │  │ 🔴 CRITICAL ISSUE #3.3: Frame-by-frame independent adjustment    │ │
│ │  │    Each frame adjusted separately WITHOUT continuity               │ │
│ │  │    Result: Jagged root Z trajectory → pelvis jitter               │ │
│ │  │                                                                    │ │
│ │  │    Example: If z_offset = [−0.02, −0.005, −0.03, ...]            │ │
│ │  │    Root Z changes by [−0.02, +0.015, −0.025, ...] between frames │ │
│ │  │    → Velocity spikes from frame-to-frame jumps                    │ │
│ │  └────────────────────────────────────────────────────────────────────┘ │
│ │                                                                          │ │
│ │ Step 4.3: MuJoCo FK for All Bodies                                      │ │
│ │  ┌────────────────────────────────────────────────────────────────────┐ │
│ │  │ FOR EACH FRAME:                                                   │ │
│ │  │  - Set qpos with corrected root_pos                               │ │
│ │  │  - Run mujoco.mj_forward()                                        │ │
│ │  │  - Extract body_pos[t, b] = data.xpos[b+1]  (33 bodies)           │ │
│ │  │  - Extract body_rot[t, b] = data.xquat[b+1] → xyzw               │ │
│ │  │                                                                    │ │
│ │  │ ⚠️  Issue #3.4: If FK correction changed root_pos too much,       │ │
│ │  │    body FK will be different from original GMR output             │ │
│ │  └────────────────────────────────────────────────────────────────────┘ │
│ │                                                                          │ │
│ │ Step 4.4: Resampling 30Hz → 50Hz                                        │ │
│ │  ┌────────────────────────────────────────────────────────────────────┐ │
│ │  │ times_src = np.arange(T) / 30.0  (30Hz source)                    │ │
│ │  │ times_tgt = np.arange(T') * 0.02 (50Hz target)                    │ │
│ │  │                                                                    │ │
│ │  │ dof_pos_resampled:   linear interpolation per joint                │ │
│ │  │ body_pos_resampled:  linear interpolation per body                 │ │
│ │  │ body_rot_resampled:  SLERP (quaternion spherical linear interp)   │ │
│ │  │                                                                    │ │
│ │  │ ⚠️  Issue #3.4: Linear interp in joint space can produce invalid  │ │
│ │  │    poses (e.g., crossing singularities). SLERP produces smooth    │ │
│ │  │    quaternions but may not match linear-interp DOF positions.     │ │
│ │  └────────────────────────────────────────────────────────────────────┘ │
│ │                                                                          │ │
│ │ Step 4.5: Velocity Computation (THE SECOND CRITICAL ISSUE)             │ │
│ │  ┌────────────────────────────────────────────────────────────────────┐ │
│ │  │ 🔴 CRITICAL ISSUE #3.5: Simple Finite Differences (NO SMOOTHING)  │ │
│ │  │                                                                    │ │
│ │  │ dof_vel[t] = (dof_pos[t+1] - dof_pos[t]) / dt                     │ │
│ │  │ dof_vel[0] = dof_vel[1]  # Copy first velocity (discontinuity!)  │ │
│ │  │                                                                    │ │
│ │  │ body_vel[t] = (body_pos[t+1] - body_pos[t]) / dt                 │ │
│ │  │ body_vel[0] = body_vel[1]  # Copy first velocity                  │ │
│ │  │                                                                    │ │
│ │  │ body_ang_vel[t] = (drot[t] * drot[t-1].inv()).as_rotvec() / dt   │ │
│ │  │ body_ang_vel[0] = body_ang_vel[1]  # Copy first velocity          │ │
│ │  │                                                                    │ │
│ │  │ Problems:                                                         │ │
│ │  │  1. Simple finite diff amplifies noise in positions               │ │
│ │  │     If dof_pos has even small jitter, dof_vel becomes noisy       │ │
│ │  │                                                                    │ │
│ │  │  2. First frame discontinuity (v[0] = v[1], not computed)        │ │
│ │  │     If motion starts from rest, v[1] may be nonzero               │ │
│ │  │                                                                    │ │
│ │  │  3. No smoothing or filtering                                     │ │
│ │  │     Raw finite differences are inherently noisy                   │ │
│ │  │                                                                    │ │
│ │  │  4. Mismatch between DOF and body velocities                      │ │
│ │  │     If FK is nonlinear, these won't be consistent                 │ │
│ │  │                                                                    │ │
│ │  │ Result: Velocity field is highly noisy and discontinuous          │ │
│ │  │         → ONNX tracker receives bad velocity references           │ │
│ │  │         → Controller can't follow smooth motion                   │ │
│ │  └────────────────────────────────────────────────────────────────────┘ │
│ │                                                                          │ │
│ │ Output: ProtoMotions cache .pt                                          │ │
│ │         - dof_pos(T', 29)      : joint angles (50Hz)                    │ │
│ │         - dof_vel(T', 29)      : joint velocities (NOISY!)              │ │
│ │         - body_rot(T', 33, 4)  : body rotations (xyzw)                  │ │
│ │         - body_pos(T', 33, 3)  : body positions (with Z jitter!)        │ │
│ │         - body_vel(T', 33, 3)  : body velocities (NOISY!)               │ │
│ │         - body_ang_vel(T', 33, 3) : angular velocities (NOISY!)         │ │
│ │         - control_dt: 0.02                                              │ │
│ │         - num_frames: T'                                                │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ 🔴 MAIN PROBLEM AREA: Issues #3.3 and #3.5 directly cause trembling        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Rendering & Visualization                                         │
│                                                                              │
│ OPTION A: Reference Mode (render_tracker_headless.py --mode reference)     │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ FOR EACH FRAME:                                                         │ │
│ │  - Extract qpos from cache:                                             │ │
│ │    qpos = [root_pos, root_rot_wxyz, dof_pos]                            │ │
│ │  - Set MuJoCo state: data.qpos[:] = qpos                                │ │
│ │  - Run FK only: mujoco.mj_forward()                                     │ │
│ │  - Render frame                                                         │ │
│ │                                                                          │ │
│ │ ✓ Pure kinematic rendering (no physics)                                 │ │
│ │ ✓ Shows exact cache content                                             │ │
│ │ ✓ If trembling visible, it's IN THE CACHE (not ONNX/physics)           │ │
│ │                                                                          │ │
│ │ 🔴 This is the DIAGNOSTIC: trembling in reference = cache problem       │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ OPTION B: Tracked Mode (run_tracker_export.py or render_tracker_headless)  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ FOR EACH FRAME:                                                         │ │
│ │  1. Read robot state from simulation                                    │ │
│ │  2. Run ONNX tracker policy with motion reference (from cache)          │ │
│ │  3. Apply PD control: torques = stiffness * (target - current)          │ │
│ │  4. Step MuJoCo physics N times                                         │ │
│ │  5. Record body state from physics                                      │ │
│ │  6. Render or export state                                              │ │
│ │                                                                          │ │
│ │ ⚠️  If reference motion is trembling:                                    │ │
│ │    - Tracker receives noisy position targets                            │ │
│ │    - Tracker receives noisy velocity references                         │ │
│ │    - PD controller oscillates trying to follow                          │ │
│ │    - Result: Tracked motion is even MORE trembling                      │ │
│ │    - Physics can't stabilize noisy input                                │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ Output: Rendered video (.mp4 or frame sequence .png)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Web Visualization                                                  │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │ Script: convert_cache_to_json.py                                        │ │
│ │                                                                          │ │
│ │ Input:  ProtoMotions cache .pt                                          │ │
│ │ Output: JSON for Three.js browser visualization                         │ │
│ │                                                                          │ │
│ │ Format:                                                                 │ │
│ │ {                                                                       │ │
│ │   "fps": 50,                                                            │ │
│ │   "num_frames": T',                                                     │ │
│ │   "joint_names": ["left_hip_pitch_joint", ...],                        │ │
│ │   "frames": [                                                           │ │
│ │     {"root_pos": [x,y,z], "root_quat": [x,y,z,w], "dof_pos": [...]} │ │
│ │   ]                                                                     │ │
│ │ }                                                                       │ │
│ │                                                                          │ │
│ │ This directly encodes the trembling in the cache!                       │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Dimensions Reference

| Stage | Data | Format | Frames | Dims | FPS |
|-------|------|--------|--------|------|-----|
| 1 | motion_135 | NPZ | T | 135 | 30 |
| 2 | SMPL-X | NPZ | T | pose(63)+root(3)+trans(3)=69 | 30 |
| 3 | GMR output | PKL | T | [root(3), quat(4), dof(29)]=36 | 30 |
| 4 | ProtoMotions cache | .pt | T' | dof(29), body_pos(33,3), body_rot(33,4) | 50 |
| 5 | Video/renders | MP4/PNG | T'/skip | RGB | variable |
| 6 | JSON | JSON | T'/subsample | root+dof | variable |

## Key Transformations

### Rot6D → Rotation Matrix
```
Input (row-major):   [R00, R01, R10, R11, R20, R21]
Reorder [0,2,4,1,3,5]: [R00, R10, R20, R01, R11, R21]  (column-major)
Gram-Schmidt:
  b1 = a1 / ||a1||
  b2 = (a2 - (a2·b1)b1) / ||(a2 - (a2·b1)b1)||
  b3 = b1 × b2
Output (3×3): [b1 | b2 | b3]
```

### SMPL-X Y-up → MuJoCo Z-up
```
rot_offset quaternion: [0.5, -0.5, -0.5, -0.5]  (wxyz)
Meaning: 120° rotation around axis [1,1,1] (normalized)
Maps: X_smplx → Z_mujoco, Y_smplx → X_mujoco, Z_smplx → Y_mujoco

For positions:
  [x, y, z]_smplx → apply rotation → [z, x, y]_mujoco

For rotations:
  q_gmr_output = rot_offset applied during IK
  q_corrected = q_gmr_output * rot_offset.inv()
```

### 30Hz → 50Hz Resampling
```
Source: T frames at 1/30 = 0.0333s intervals
Target: T' frames at 1/50 = 0.02s intervals

For positions (DOF, body_pos):
  Linear interpolation: pos_interp = pos_src[t] + α * (pos_src[t+1] - pos_src[t])
  
For rotations (body_rot):
  SLERP: rot_interp = Slerp(rot_src[t], rot_src[t+1], α)
```

### Velocity from Finite Differences
```
Forward difference:
  vel[1:] = (pos[1:] - pos[:-1]) / dt
  vel[0] = vel[1]  ← DISCONTINUITY!

Problems:
  - No smoothing → amplifies noise
  - First frame hardcoded → breaks continuity
  - When dt = 0.02, even small position errors → large velocity errors
```

## Problem Chain Visualization

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          TREMBLING PROBLEM CHAIN                             │
└──────────────────────────────────────────────────────────────────────────────┘

Source SMPL Motion (smooth) ─────────────────────────────────────────┐
                                                                     │
                                                                     ▼
          ┌─────────────────────────────────────────────────────────────┐
          │ Motion135 → SMPL-X conversion (accurate)                    │
          └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
          ┌─────────────────────────────────────────────────────────────┐
          │ GMR retargeting (mostly accurate, slight jitter possible)   │
          └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
          ┌─────────────────────────────────────────────────────────────┐
          │ gmr_to_protomotions.py - PROBLEM AREA                       │
          │                                                              │
          │ Issue #3.3: FK Ground Correction (frame-by-frame)           │
          │  → Independent per-frame Z adjustments                      │
          │  → Root height becomes discontinuous                        │
          │  → Results in pelvis angular velocity spikes                │
          │                                                              │
          │ Issue #3.5: Finite Difference Velocity                      │
          │  → Simple finite diff on noisy positions                    │
          │  → Amplifies jitter into velocity spikes                    │
          │  → First frame discontinuity                                │
          │                                                              │
          │ Resampling 30→50Hz                                          │
          │  → Linear interp produces interpolation artifacts           │
          │                                                              │
          │ Result: Cache contains trembling!                           │
          └─────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │Reference Render  │ │ONNX Policy       │ │JSON Visualization│
        │(pure FK)         │ │Simulation        │ │(web)             │
        │                  │ │                  │ │                  │
        │Trembling visible │ │Tracker oscillates│ │Trembling encoded │
        │↓                 │ │trying to follow  │ │in positions      │
        │Confirms: problem │ │noisy targets     │ │↓                 │
        │in cache          │ │↓                 │ │Visible to user   │
        │                  │ │Trembling worse!  │ │                  │
        └──────────────────┘ └──────────────────┘ └──────────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                                    ▼
                        ┌──────────────────────┐
                        │ USER SEES TREMBLING! │
                        └──────────────────────┘
```

## Dependency Graph

```
motion_135.npz
       ├─→ motion135_to_smplx.py
       │   ├─→ SMPL-X.npz (Issues: #1.2, #1.3)
       │   └─→ gmr_retarget_headless.py
       │       ├─→ GMR PKL (Issues: #2.1, #2.2, #2.3)
       │       └─→ gmr_to_protomotions.py ◄─── **MAIN PROBLEM**
       │           ├─→ ProtoMotions cache.pt (Issues: #3.1-#3.6)
       │           │   ├─→ render_tracker_headless.py (reference)
       │           │   │   └─→ Frames/video
       │           │   │
       │           │   ├─→ render_tracker_headless.py (tracked)
       │           │   │   ├─→ ONNX inference
       │           │   │   ├─→ Physics simulation
       │           │   │   └─→ Frames/video
       │           │   │
       │           │   └─→ run_tracker_export.py
       │           │       └─→ Tracked cache.pt
       │           │
       │           └─→ convert_cache_to_json.py
       │               └─→ motion.json (for Three.js)
       │
       └─→ Reference motion data (from source SMPL)
```

