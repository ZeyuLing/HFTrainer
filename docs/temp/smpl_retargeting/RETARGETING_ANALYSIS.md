# SMPL-to-Robot Retargeting Pipeline: Full Analysis & Trembling Issue Diagnosis

## Pipeline Overview

The end-to-end retargeting pipeline converts HyMotion's `motion_135` format → Unitree G1 robot motions through 7 major steps:

```
motion_135 NPZ (T, 135)
    ↓ [motion135_to_smplx.py]
SMPL-X NPZ (pose_body, root_orient, trans)
    ↓ [gmr_retarget_headless.py]
GMR PKL (root_pos, root_rot, dof_pos at 30Hz)
    ↓ [gmr_to_protomotions.py]
ProtoMotions cache .pt (FK + resampling + velocities at 50Hz)
    ↓ [render_tracker_headless.py] OR [run_tracker_export.py]
Reference/Tracked robot motion for rendering or ONNX simulation
    ↓ [convert_cache_to_json.py]
Three.js JSON for web visualization
```

---

## Component-by-Component Analysis

### 1. **Motion135 → SMPL-X Conversion** (`motion135_to_smplx.py`)

**Data Format:**
- Input: `motion_135` shape (T, 135) = [transl(3) + 22×rot6d(132)]
- Output: SMPL-X NPZ with pose_body(T,63), root_orient(T,3), trans(T,3) in axis-angle

**Potential Issues:**

**Issue #1.1: Rot6D Reordering Bug (CRITICAL)**
```python
# Line 39: Row-major → column-major reorder
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
```
- HyMotion outputs: `[R00, R01, R10, R11, R20, R21]` (row-major)
- Gram-Schmidt expects: `[R00, R10, R20, R01, R11, R21]` (column-major)
- **Reorder is correct**, but worth verifying HyMotion's actual format matches documentation

**Issue #1.2: Low Epsilon in Gram-Schmidt**
```python
# Line 44: Normalization with small epsilon
b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
```
- `1e-8` epsilon is small; if norms approach `1e-8`, normalization can amplify noise
- **Could contribute to numerical instability in subsequent steps**

**Issue #1.3: No Validation of Output**
- No verification that output quaternions are valid/normalized
- Invalid quaternions → downstream instability

---

### 2. **GMR Retargeting** (`gmr_retarget_headless.py`)

**Data Flow:**
- Input: SMPL-X NPZ (22 body joints in Y-up frame)
- Output: GMR PKL with root_pos(T,3), root_rot(T,4), dof_pos(T,29) at 30Hz

**Potential Issues:**

**Issue #2.1: Ground Offset Pre-computation (IMPORTANT)**
```python
# Lines 112-140: compute_ground_offset()
ground_offset = compute_ground_offset(retarget, smplx_data_frames)
retarget.set_ground_offset(ground_offset)
```
- Pre-scans all frames to find **lowest body Z position globally**
- Sets this as reference ground
- **Problem**: Doesn't account for foot/ankle separation
- **Result**: If one frame has feet touching ground while another doesn't, ground offset may be wrong

**Issue #2.2: Per-Frame Foot Grounding**
```python
# Line 196: Retargeting with offset_to_ground=True (from pipeline line 125)
qpos = retarget.retarget(frame_data, offset_to_ground=args.offset_to_ground)
```
- GMR's `offset_to_ground()` is called **per-frame**
- Purpose: Adjust foot position after IK
- **Risk**: Independent per-frame adjustments can cause **frame-to-frame discontinuities**
- **Effect**: Foot position/orientation can jump between frames

**Issue #2.3: Joint Limit Clamping Without Smoothing**
```python
# Lines 85-109: clamp_joint_limits()
clamped[:, i] = np.clip(clamped[:, i], lo, hi)
```
- Hard clipping without smoothing
- **Issue**: Sharp state transitions at joint limits
- **Example**: If knee hits limit, clamped value changes abruptly → velocity spike → trembling

**Issue #2.4: No Velocity Information from GMR**
- GMR outputs only **positions**, not velocities
- Velocities computed downstream via finite differences
- **Consequence**: No information about motion smoothness is preserved

---

### 3. **ProtoMotions Conversion** (`gmr_to_protomotions.py`)

**Key Steps:**
1. Load GMR PKL (30Hz)
2. Convert coordinate frames (SMPL-X Y-up → MuJoCo Z-up)
3. **FK ground correction** (optional but important)
4. Run MuJoCo FK for all 33 bodies
5. Resample 30Hz → 50Hz
6. **Compute velocities via finite differences**

**Potential Issues:**

**Issue #3.1: Root Position Frame Conversion**
```python
# Lines 92-111: convert_root_pos_to_zup()
def convert_root_pos_to_zup(root_pos):
    rot_offset = _get_gmr_rot_offset()  # [0.5, -0.5, -0.5, -0.5] wxyz
    return rot_offset.inv().apply(root_pos)
```
- Applies frame rotation to positions: `[x, y, z]_smplx → [z, x, y]_mujoco`
- **Correct approach** but coordinate systems must match exactly

**Issue #3.2: Root Rotation Offset Removal**
```python
# Lines 69-89: remove_gmr_root_offset()
corrected = root_rots * rot_offset.inv()
```
- GMR applies `rot_offset` during retargeting
- Code removes it, expecting near-identity result
- **Issue**: If GMR's offset application is different from expected, this can leave residual rotation

**Issue #3.3: FK Ground Correction (CRITICAL POTENTIAL SOURCE OF TREMBLING)**
```python
# Lines 155-229: fk_ground_correction()
for t in range(T):
    # Set MuJoCo state
    data.qpos[:3] = root_pos[t]
    data.qpos[3:7] = root_rot_wxyz
    data.qpos[7:] = dof_pos[t]
    mujoco.mj_forward(model, data)
    
    # Find min foot Z
    min_foot_z = min(data.xpos[bi][2] for bi in foot_body_indices)
    
    # Adjust root Z
    z_offset = ground_clearance - min_foot_z
    corrected_root_pos[t, 2] = root_pos[t, 2] + z_offset
```
- **Frame-by-frame independent adjustment** of root Z based on FK
- **Major Problem**: Each frame adjusted separately → no continuity guarantee
- **Effect**: Root height can jump frame-to-frame, causing trembling in:
  - Pelvis angular velocity (from discrete Z changes)
  - Foot positions (when FK is recomputed with new root Z)

**Example Scenario:**
```
Frame t-1: min_foot_z = 0.02m  → z_offset = -0.02  → root_z adjusted down
Frame t:   min_foot_z = 0.005m → z_offset = -0.005 → root_z adjusted less
Frame t+1: min_foot_z = 0.03m  → z_offset = -0.03  → root_z adjusted more
Result: Z trajectory = [z0-0.02, z1-0.005, z2-0.03] → jagged, discontinuous
```

**Issue #3.4: Resampling Creates Interpolation Artifacts**
```python
# Lines 296-342: resample_motion()
dof_interp = interp1d(times_src, dof_pos, axis=0, kind='linear')
dof_pos_resampled = dof_interp(times_tgt)
```
- Linear interpolation used for joint angles
- **Issue**: No validation that interpolated poses are valid (could cross singularities)
- **Issue**: Linear interp in quaternion space (SLERP for body_rot) → mismatch
- **Consequence**: Resampled DOF positions may not match resampled body rotations

**Issue #3.5: Velocity Computation via Finite Differences (CRITICAL FOR TREMBLING)**
```python
# Lines 345-384: compute_velocities()

# DOF velocity
dof_vel = np.zeros_like(dof_pos)
dof_vel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt  # Simple finite diff
dof_vel[0] = dof_vel[1]  # Copy second frame's velocity to first

# Body angular velocity
for b in range(num_bodies):
    for t in range(1, T):
        drot = rots[t] * rots[t-1].inv()
        body_ang_vel[t, b] = drot.as_rotvec() / dt
    body_ang_vel[0, b] = body_ang_vel[1, b]  # Copy second frame
```

**Problems:**
1. **Simple finite differences amplify high-frequency noise**
   - Any jitter in positions → large velocity spikes
   - Especially bad near singularities

2. **First frame velocity copied from second frame**
   - Discontinuity at t=0
   - If motion starts from rest, first frame will show non-zero velocity

3. **No smoothing/filtering applied**
   - Raw finite differences = highly noisy
   - Trackers expect smooth velocity fields

4. **Mismatch between dof_vel and FK-computed body velocities**
   - dof_vel computed from dof_pos differences
   - body_vel computed from body_pos differences
   - If FK is non-linear, these won't be consistent

**Issue #3.6: Control DT Mismatch in Velocities**
```python
# Line 471-472
dof_vel, body_vel, body_ang_vel = compute_velocities(
    dof_pos_r, body_pos_r, body_rot_r, args.control_dt
)
```
- Velocities computed with `control_dt = 0.02` (50Hz)
- But if resampling produced different timestamps, dt might be off
- **Result**: Velocity magnitudes wrong, tracking system can't follow

---

### 4. **Reference Rendering** (`render_tracker_headless.py`)

**Mode: Reference** (lines 220-323)
```python
# Extract state from cache
root_pos = body_pos[frame_idx, 0, :]        # From FK
root_rot_xyzw = body_rot[frame_idx, 0, :]
dof = dof_pos[frame_idx, :]

# Set MuJoCo qpos directly (NO simulation)
qpos = np.concatenate([root_pos, root_rot_wxyz, dof])
data.qpos[: len(qpos)] = qpos
data.qvel[:] = 0.0
mujoco.mj_forward(model, data)  # FK only
```

**Issue #4.1: Rendering Shows Pure FK Output**
- No dynamics applied
- Trembling visible in reference mode means trembling is in the **cache itself**
- Points to Issues #3.3, #3.5

---

### 5. **Tracked Mode & ONNX Policy** (`run_tracker_export.py`)

**Core Loop** (lines 319-435):
1. Record body state from MuJoCo simulation
2. Run ONNX policy inference
3. Apply PD control
4. Step physics
5. Repeat

**Issue #5.1: Tracking Follows Reference Motion**
- If reference motion is trembling, tracker tries to follow
- PD control with limited bandwidth → oscillation
- **Result**: Tracked mode amplifies trembling

**Issue #5.2: EMA Filter Can Mask Issues**
```python
# Lines 420-428: EMA action filter
if use_ema:
    pd_targets = action_ema_alpha * pd_targets + (1 - action_ema_alpha) * ema_prev_targets
```
- If `action_ema_alpha` is close to 1.0, filter has little effect
- Trembling still visible

---

### 6. **Known GMR Issues from Code**

**In `gmr_retarget_headless.py` line 125:**
```python
"--no-offset-to-ground",  # FK correction handles grounding
```
- Comment suggests per-frame foot grounding is **disabled** in favor of FK correction
- But FK correction itself can cause trembling (Issue #3.3)

**In pipeline line 125-126:**
```python
gmr_cmd += ["--no-offset-to-ground",  # FK correction handles grounding]
```
- Confirms GMR's per-frame grounding is disabled
- All grounding done in `gmr_to_protomotions.py` via FK

---

## Summary of Trembling Sources

### **PRIMARY (Most Likely):**

1. **Frame-by-Frame FK Ground Correction** ⚠️ CRITICAL
   - Issue #3.3: Independent per-frame root Z adjustment
   - Each frame adjusted separately without continuity
   - **Direct cause of pelvis height jitter**

2. **Finite Differences Velocity Computation** ⚠️ CRITICAL  
   - Issue #3.5: Simple finite diff amplifies noise
   - First frame discontinuity
   - **Direct cause of velocity/acceleration spikes**

3. **Joint Limit Clamping Without Smoothing** ⚠️ IMPORTANT
   - Issue #2.3: Hard clipping causes state jumps
   - **Causes trembling when joints near limits**

### **SECONDARY (Contributing):**

4. Per-frame foot grounding in GMR (Issue #2.2)
5. Resampling interpolation artifacts (Issue #3.4)
6. Ground offset pre-computation (Issue #2.1)
7. Quaternion numerical precision (Issue #1.2)

---

## Diagnostic Approach

### **Step 1: Check Reference Motion Trembling**
```bash
# Render reference (pure FK, no physics)
python scripts/embodied/render_tracker_headless.py \
    --motion output/embodied_t2m_v4/data/caches/pipeline_XXX.pt \
    --mode reference \
    --output-dir /tmp/ref_render
```
- If trembling visible → problem is in cache (issues #3.3, #3.5)
- If smooth → problem is in ONNX tracking

### **Step 2: Inspect Cache Data**
```python
import torch
cache = torch.load('pipeline_XXX.pt', weights_only=False)
print(cache['dof_pos'].shape)  # (T, 29)
print(cache['dof_vel'].shape)  # (T, 29)

# Check velocity continuity
dof_vel = cache['dof_vel'].numpy()
vel_jumps = np.max(np.abs(dof_vel[1:] - dof_vel[:-1]), axis=1)
print(f"Max velocity jump per frame: {np.max(vel_jumps)}")
print(f"Mean velocity jump: {np.mean(vel_jumps)}")

# Check position continuity  
dof_pos = cache['dof_pos'].numpy()
pos_jumps = np.max(np.abs(dof_pos[1:] - dof_pos[:-1]), axis=1)
print(f"Max position jump per frame: {np.max(pos_jumps)}")

# Check root height consistency
body_pos = cache['body_pos'].numpy()  # (T, 33, 3)
root_z = body_pos[:, 0, 2]  # Pelvis Z
root_z_diffs = np.diff(root_z)
print(f"Root Z range: [{root_z.min():.4f}, {root_z.max():.4f}]")
print(f"Root Z changes per frame: min={root_z_diffs.min():.6f}, max={root_z_diffs.max():.6f}, std={root_z_diffs.std():.6f}")
```

### **Step 3: Check FK Ground Correction Impact**
```bash
# Retarget with FK correction DISABLED
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/.../npz/00000.npz \
    --output /tmp/no_fk_correction.pt \
    --no-fk-ground-correction

# Compare cache with/without FK correction
```

### **Step 4: Check Velocity Computation**
```python
# Manually compute velocities with smoothing
import scipy.signal

dof_pos_original = cache['dof_pos'].numpy()
dt = cache['control_dt']

# Method 1: Savitzky-Golay filter (smooth derivative)
dof_vel_smooth = np.zeros_like(dof_pos_original)
for j in range(29):
    dof_vel_smooth[:, j] = scipy.signal.savgol_filter(
        dof_pos_original[:, j],
        window_length=5,  # Use 5-frame window
        polyorder=2,      # Fit quadratic
        deriv=1,          # First derivative
        delta=dt
    )

# Check how different it is from the original finite diff velocities
orig_vel = cache['dof_vel'].numpy()
print(f"Velocity difference (smooth vs raw): {np.mean(np.abs(dof_vel_smooth - orig_vel))}")
```

---

## Recommended Fixes (Priority Order)

### **Fix #1: Smooth FK Ground Correction** (HIGH PRIORITY)
Replace frame-by-frame adjustment with a **global smooth adjustment**:

```python
def fk_ground_correction_smooth(mjcf_path, root_pos, root_rot_xyzw, dof_pos, ...):
    # Find ground offset as before
    offset = compute_ground_offset(...)
    
    # Compute foot height for each frame
    foot_z_all = []
    for t in range(T):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = quat_xyzw_to_wxyz(root_rot_xyzw[t])
        data.qpos[7:] = dof_pos[t]
        mujoco.mj_forward(model, data)
        
        foot_z = min(data.xpos[bi][2] for bi in foot_body_indices)
        foot_z_all.append(foot_z)
    
    foot_z_all = np.array(foot_z_all)
    
    # Smooth the foot height trajectory (e.g., with Savitzky-Golay)
    foot_z_smooth = scipy.signal.savgol_filter(foot_z_all, window_length=7, polyorder=2)
    
    # Compute smooth root Z adjustments
    z_offsets = ground_clearance - foot_z_smooth
    z_offsets = scipy.signal.savgol_filter(z_offsets, window_length=7, polyorder=2)
    
    # Apply smoothed adjustments
    corrected_root_pos = root_pos.copy()
    corrected_root_pos[:, 2] += z_offsets
    
    return corrected_root_pos, foot_z_all
```

### **Fix #2: Smooth Velocity Computation** (HIGH PRIORITY)
Replace simple finite differences with Savitzky-Golay filtering:

```python
def compute_velocities_smooth(dof_pos, body_pos, body_rot_xyzw, dt, window_length=5):
    """Compute velocities using Savitzky-Golay filter for smoothing."""
    import scipy.signal
    
    T = dof_pos.shape[0]
    num_bodies = body_pos.shape[1]
    
    # Ensure window_length is odd and at most T
    window_length = min(window_length, T if T % 2 == 1 else T - 1)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(3, window_length)
    
    # DOF velocity: Savitzky-Golay derivative
    dof_vel = np.zeros_like(dof_pos)
    for j in range(dof_pos.shape[1]):
        dof_vel[:, j] = scipy.signal.savgol_filter(
            dof_pos[:, j],
            window_length=window_length,
            polyorder=2,
            deriv=1,
            delta=dt
        )
    
    # Body linear velocity
    body_vel = np.zeros_like(body_pos)
    for b in range(num_bodies):
        for d in range(3):
            body_vel[:, b, d] = scipy.signal.savgol_filter(
                body_pos[:, b, d],
                window_length=window_length,
                polyorder=2,
                deriv=1,
                delta=dt
            )
    
    # Body angular velocity: From quaternion time derivatives
    body_ang_vel = np.zeros((T, num_bodies, 3), dtype=np.float32)
    for b in range(num_bodies):
        # Quaternion angular velocity from time-varying rotation
        for t in range(T):
            # Use numerical quaternion differentiation
            if t > 0 and t < T - 1:
                q_prev = R.from_quat(body_rot_xyzw[t-1])
                q_curr = R.from_quat(body_rot_xyzw[t])
                q_next = R.from_quat(body_rot_xyzw[t+1])
                
                # Central difference for angular velocity
                drot_dt = q_next * q_curr.inv()
                rotvec = drot_dt.as_rotvec()
                body_ang_vel[t, b] = rotvec / (2 * dt)
            elif t == 0:
                q0 = R.from_quat(body_rot_xyzw[0])
                q1 = R.from_quat(body_rot_xyzw[1])
                drot_dt = q1 * q0.inv()
                body_ang_vel[0, b] = drot_dt.as_rotvec() / dt
            else:  # t == T-1
                q_prev = R.from_quat(body_rot_xyzw[-2])
                q_curr = R.from_quat(body_rot_xyzw[-1])
                drot_dt = q_curr * q_prev.inv()
                body_ang_vel[-1, b] = drot_dt.as_rotvec() / dt
    
    return dof_vel.astype(np.float32), body_vel.astype(np.float32), body_ang_vel.astype(np.float32)
```

### **Fix #3: Smooth Joint Clamping** (MEDIUM PRIORITY)
Apply smoothed blending near joint limits:

```python
def clamp_joint_limits_smooth(dof_pos, joint_limits, blend_margin=0.1):
    """Clamp joint positions with smooth blending near limits."""
    clamped = dof_pos.copy()
    
    for i, joint_name in enumerate(G1_JOINT_ORDER):
        if joint_name not in joint_limits:
            continue
        
        lo, hi = joint_limits[joint_name]
        range_width = hi - lo
        blend_lo = lo + blend_margin * range_width
        blend_hi = hi - blend_margin * range_width
        
        # Smooth blend beyond limits
        below_blend = clamped[:, i] < blend_lo
        above_blend = clamped[:, i] > blend_hi
        
        # Hard clamp first
        clamped[:, i] = np.clip(clamped[:, i], lo, hi)
        
        # Apply smoothing for values in blend region
        # Linear blend from value to limit
        blend_vals_lo = clamped[below_blend, i]
        t_lo = (blend_vals_lo - lo) / (blend_margin * range_width)
        clamped[below_blend, i] = blend_lo + (blend_vals_lo - blend_lo) * (1 - t_lo)
        
        blend_vals_hi = clamped[above_blend, i]
        t_hi = (blend_vals_hi - blend_hi) / (blend_margin * range_width)
        clamped[above_blend, i] = blend_hi + (blend_vals_hi - blend_hi) * (1 - t_hi)
    
    return clamped
```

### **Fix #4: Disable FK Ground Correction (or Make it Optional)** (MEDIUM PRIORITY)
If feet already positioned correctly by GMR, FK correction may be unnecessary:

```bash
python scripts/embodied/pipeline_motion_to_robot.py \
    --input work_dirs/.../npz/00000.npz \
    --output data/embodied_debug/robot_cache.pt \
    --no-fk-ground-correction
```

---

## Implementation Path

1. **Immediate**: Test without FK ground correction
2. **Short-term**: Add smooth FK correction
3. **Short-term**: Replace finite diff velocities with Savitzky-Golay
4. **Medium-term**: Add smooth joint clamping
5. **Long-term**: Consider alternative grounding strategies (center-of-mass based, etc.)

