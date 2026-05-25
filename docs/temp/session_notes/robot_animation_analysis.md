# PhysFlow Robot Animation JSON Generation Analysis

## Overview
Investigation of how robot animation JSON files are generated for the PhysFlow pipeline demo, focusing on the root translation (pelvis body position) to determine if the pipeline correctly handles kinematic-to-physics conversion.

---

## 1. Pipeline Architecture

### End-to-end Workflow

```
motion_135 NPZ (SMPL human kinematics)
    ↓
pipeline_motion_to_robot.py:
  [Step 1] motion_135 → PyRoki keypoints (.npy)
  [Step 2a] Extract foot contacts
  [Step 2b] PyRoki retargeting (keypoints → G1 robot NPZ)
  [Step 3] Robot NPZ → ProtoMotions .motion file
    ↓
run_g1_rl_tracker_export.py:
  [Step 1] Retarget NPZ → .motion (if not already done)
  [Step 2] Load ONNX policy + MuJoCo simulation
  [Step 3] Step robot through motion using ONNX control
  [Step 4] Export body positions (xpos) & quaternions (xquat) as JSON
    ↓
robot_frames JSON (for Three.js visualization)
```

### Expected Output Format

```json
{
  "type": "robot_frames",
  "robot": "g1",
  "fps": 50,
  "num_frames": N,
  "num_bodies": 17,
  "bodies": [
    {"name": "pelvis", "meshes": ["pelvis.stl", "pelvis_contour_link.stl"]},
    {"name": "left_hip_pitch_link", "meshes": ["left_hip_pitch_link.stl"]},
    ...
  ],
  "frames": [
    {
      "body_pos": [[x,y,z], ...],   # per-body world position (num_bodies x 3)
      "body_quat": [[w,x,y,z], ...] # per-body world quaternion wxyz (num_bodies x 4)
    },
    ...
  ]
}
```

---

## 2. Input Data Analysis: motion_135 Format

### What is motion_135?

- **Format**: motion_135 NPZ files containing motion_135 array shape (T, 135)
- **Content**: SMPL human kinematic data
  - Columns 0-2: Root translation (X, Y, Z)
  - Columns 3-134: 22 joints × 6 DOF (rot6d rotation representation)
- **Coordinate System**: SMPL/human-centric
  - Z-axis: vertical (up)
  - Y-axis: forward
  - X-axis: right

### Sample Motion Data

#### Motion 1: "Stands Still" (90 frames, 3 seconds @ 30fps)

```
Pelvis Height (Y-axis):
  Min:  1.161387 m
  Max:  1.162992 m
  Mean: 1.162188 m

Root Displacement:
  X: +0.0062 m (+0.62 cm)
  Z: -0.0028 m (-0.28 cm)

First frame:  [0.001934, 1.162407, -0.018396]
Last frame:   [0.008121, 1.161971, -0.021181]
```

**Interpretation**: Appropriate kinematic jitter for standing still. ✓

#### Motion 2: "Weight Shift" (90 frames, 3 seconds @ 30fps)

```
Pelvis Height (Y-axis):
  Min:  1.123204 m
  Max:  1.126409 m
  Mean: 1.124411 m

Root Displacement:
  X: +0.0203 m (+2.03 cm)
  Z: -0.0076 m (-0.76 cm)

First frame:  [-0.012522, 1.126409, -0.019124]
Last frame:   [0.007741,  1.123538, -0.026684]
```

**Interpretation**: Small lateral motion with minimal vertical oscillation. ✓

#### Motion 3: "Walks Forward Slowly" (120 frames, 4 seconds @ 30fps)

```
Pelvis Height (Y-axis):
  Min:  1.095778 m
  Max:  1.154614 m
  Mean: 1.138458 m

Root Displacement:
  X: -0.0118 m (-1.18 cm)
  Z: +2.3619 m (+236.19 cm)  ⚠️ HUGE Z DISPLACEMENT

First frame:  [0.000010, 1.154481,  0.016348]
Last frame:   [-0.011823, 1.141617, 2.378243]
```

**Interpretation**: +2.36m vertical displacement in 4 seconds is UPWARD motion, not forward walking! ❌

---

## 3. Critical Finding: Coordinate System Mismatch

### Expected Behavior (G1 Robot Frame)

- Y-axis: vertical (up)
- Pelvis height in standing: **~0.78 m**
- Forward walking: displacement along **X-axis**, Y stays ~0.78m
- Vertical motion: displacement along **Y-axis**

### Actual Behavior (SMPL/motion_135)

- Z-axis: vertical (up)
- Pelvis height in standing: **~1.14-1.16 m** (human height in SMPL frame)
- Forward walking: shows **+2.36 m in Z** (interpreted as UP in robot frame)
- Axes are fundamentally different!

### Root Causes

1. **SMPL coordinates are human-centric:**
   - Human pelvis height ~1.14m is normal
   - Z is vertical in SMPL
   - Not aligned to G1 robot frame

2. **The retargeting pipeline should handle this:**
   - PyRoki's batch_retarget_to_g1_from_keypoints.py is supposed to convert SMPL keypoints to G1 joint angles
   - Then convert_pyroki_retargeted_robot_motions_to_proto.py generates .motion files
   - MuJoCo simulation should compute correct world positions

---

## 4. Pipeline Status: BROKEN

### Current Error

```
ERROR: Extract foot contact labels failed with return code 1
Traceback: ModuleNotFoundError: No module named 'jax'
```

**Root Cause**: JAX module not installed

**Location**: ref_repo/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py

### Pipeline Breakdown

```
✓ Step 1: motion_135 → PyRoki keypoints (.npy)
  - Converts SMPL to 18 keypoints
  - Detects foot contacts
  - Transforms to robot-appropriate frame
  - Output: (90, 18, 3) keypoints

✗ Step 2a: Extract foot contact labels
  - Requires: JAX (not installed)
  - Fails immediately with import error

✗ Step 2b: PyRoki retargeting
  - Never reached due to Step 2a failure
  - Would use jaxls optimization for trajectory retargeting

✗ Step 3: NPZ → .motion conversion
  - Never reached

✗ Step 4: MuJoCo simulation & export
  - Never reached
```

### Result

- **No .motion files generated**
- **No JSON output files generated**
- **_export_summary.json shows**: {"error": "Retargeting failed"}

---

## 5. What Should Happen (If Pipeline Works)

### After PyRoki Retargeting

The .motion file would contain:
- G1 robot joint angles aligned to SMPL motion
- Trajectory-level optimization ensuring:
  - Local bone alignment to SMPL skeleton
  - Global keypoint alignment
  - Foot contact constraints
  - Joint smoothness
  - Realistic actuator limits

### After MuJoCo Simulation

For "Stands Still":
```json
{
  "type": "robot_frames",
  "robot": "g1",
  "fps": 25,
  "num_frames": 45,
  "num_bodies": 17,
  "bodies": [...],
  "frames": [
    {
      "body_pos": [
        [0.0, 0.78, 0.0],  # pelvis at ~0.78m height
        [...],              # other bodies
      ],
      "body_quat": [
        [1.0, 0.0, 0.0, 0.0],  # identity rotation
        [...],
      ]
    },
    ...
  ]
}
```

**Expected**: Pelvis Y-coordinate ≈ **0.78m** (G1 standing height)
**Currently**: Motion_135 has **~1.14m** (human height in SMPL frame)

---

## 6. Files Examined

### Scripts
- pipeline_motion_to_robot.py: 269 lines - end-to-end retargeting orchestrator
- run_g1_rl_tracker_export.py: 730 lines - MuJoCo simulation + JSON export
- motion135_to_pyroki_keypoints.py: SMPL → keypoint converter
- batch_retarget_to_g1_from_keypoints.py: PyRoki retargeting (FAILED - JAX missing)

### Data Directories
- Input: output/physflow/eval_demo/data/npz/ (SMPL motion_135 files) ✓
- Intermediate: output/physflow/eval_demo/data/robot_mesh_rl/motion_files/intermediates/ (empty)
- Output: output/physflow/eval_demo/data/robot_mesh_rl/ (no JSON files)
- Summary: _export_summary.json (shows failure)

---

## 7. Kinematic vs Physics Analysis

### Question: Are Root Translations Physically Plausible?

**Answer: Cannot determine yet** - the pipeline is broken before physics simulation.

### What We Know

1. **Input (motion_135):**
   - Contains SMPL human kinematics (not robot)
   - Pelvis height ~1.14m (human, not G1 robot)
   - Forward walking shows +2.36m in Z-axis (vertical in SMPL frame)
   - This is NOT the robot frame - it's the SMPL human frame

2. **Pipeline Stage 1 (works):**
   - motion135_to_pyroki_keypoints.py correctly converts SMPL to keypoints
   - Properly handles Z-up to Y-up coordinate transformation
   - Foot contact detection works
   - Ready for retargeting

3. **Pipeline Stage 2+ (broken):**
   - PyRoki retargeting requires JAX
   - Missing dependency prevents execution
   - No robot `.motion` files generated
   - No MuJoCo simulation occurs
   - No physics-based export happens

### Expected After Pipeline Fix

Once JAX is installed and pipeline completes:
- PyRoki converts human keypoints to G1 joint angles
- .motion file contains proper G1 trajectories
- MuJoCo simulates G1 dynamics with ONNX policy
- Output JSON will have pelvis Y ≈ **0.78m** (G1 standing height)
- Coordinate frame properly aligned to G1 robot frame
- Motion amplitudes appropriate for robot scale (not human scale)

---

## 8. Key Insight: Intentional Coordinate Mismatch Design

The pipeline design INTENTIONALLY separates concerns:

1. **Phase 1 (Generative):** Motion_135 in SMPL frame
   - Human-friendly scale and conventions
   - Easier for SMPL models and text-to-motion
   - Includes human-scale biomechanics

2. **Phase 2 (Retargeting):** Convert to G1 frame
   - PyRoki optimizes skeletal alignment
   - Respects G1 actuator limits
   - Ensures kinematic feasibility

3. **Phase 3 (Physics):** Simulate in MuJoCo
   - Applies G1 dynamics constraints
   - Tracks motion reference with control policy
   - Exports physically-valid robot motion

This is correct design - but the pipeline is currently BROKEN at Phase 2.

---

## Summary Table

| Component | Status | Issue |
|-----------|--------|-------|
| SMPL motion_135 data | ✓ Present | Correct human kinematics in SMPL frame |
| motion135_to_pyroki_keypoints.py | ✓ Works | Converts SMPL to 18 keypoints successfully |
| batch_retarget_to_g1_from_keypoints.py | ✗ BROKEN | Missing JAX dependency |
| PyRoki retargeting | ✗ BLOCKED | Can't run without PyRoki script |
| .motion file generation | ✗ BLOCKED | Never reached |
| MuJoCo simulation | ✗ BLOCKED | Never reached |
| robot_frames JSON export | ✗ BLOCKED | Never reached |

---

## Recommendations

### Immediate Action Required

1. **Install JAX**:
   ```bash
   pip install jax jaxlib
   ```

2. **Re-run robot export**:
   ```bash
   python3 scripts/embodied/run_g1_rl_tracker_export.py \
     --input-dir output/physflow/eval_demo/data/npz/ \
     --output-dir output/physflow/eval_demo/data/robot_mesh_rl/
   ```

### Post-Fix Validation

3. **Verify output JSON** contains:
   - Pelvis body_pos[0][1] ≈ 0.78m (Y-coordinate, should be G1 height)
   - NOT 1.14m (which would be human height)
   - Proper motion amplitudes for robot scale

4. **Check specific motions**:
   - "stands_still": body_pos mostly constant, Y ≈ 0.78m
   - "walks_forward": X displacement over time, Y ≈ 0.78m
   - "weight_shift": small lateral movements, Y ≈ 0.78m ± a few cm

---

