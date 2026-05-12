# KIMODO Root/Translation Representation - Detailed Analysis

## 1. KIMODO Motion Representation (333 dims for 27-joint skeleton)

### Feature Layout (Per-Frame)

| Component | Dims | Total | Description |
|-----------|------|-------|-------------|
| `smooth_root_pos` | [0:3] | 3 | Smoothed pelvis position (x, z smoothed; y raw) |
| `global_root_heading` | [3:5] | 2 | [cos(ψ), sin(ψ)] where ψ is yaw angle around Y-axis |
| `local_joints_positions` | [5:86] | 81 | 27 joints × 3: positions relative to smooth_root (xz relative, y absolute) |
| `global_rot_data` | [86:248] | 162 | 27 joints × 6: 6D continuous rotation (world-frame) |
| `velocities` | [248:329] | 81 | 27 joints × 3: global joint velocities |
| `foot_contacts` | [329:333] | 4 | {L_heel, L_toe, R_heel, R_toe} binary flags |

**Total: 3 + 2 + 81 + 162 + 81 + 4 = 333 dims**

### Key Properties

1. **Smooth Root Position (dims [0:3])**
   - NOT raw pelvis; ADMM-smoothed horizontal (x,z) components
   - Y remains absolute (raw) height
   - Smoothing reduces high-frequency jitter while preserving motion intent
   - Designed for animator workflow (straight line/curve constraints)

2. **Global Root Heading (dims [3:5])**
   - 2D angle representation: [cos(ψ), sin(ψ)]
   - ψ = yaw rotation around Y-axis (gravity-up)
   - Normalized representation (always unit length if properly encoded)

3. **Local Joint Positions (dims [5:86])**
   - Relative to smooth_root in XZ plane
   - Y component is absolute height (NOT relative)
   - Formula: `local_pos = global_pos - smooth_root_pos` with Y as absolute
   - Conversion: `global_pos[..., [0,2]] = local_pos[..., [0,2]] + smooth_root_pos[..., [0,2]]`

4. **Global Joint Rotations (dims [86:248])**
   - 6D continuous rotation (first 2 columns of 3×3 rotation matrix)
   - World-frame (not parent-relative like SMPL)
   - Enables direct imputation of end-effector constraints without IK

## 2. Smooth Root Smoothing Details (ADMM Algorithm)

### Algorithm: `get_smooth_root_pos()` + `smooth_signal()` + `TrajectorySmoother`

**Input:** Raw pelvis positions [B, T, 3]
**Output:** Smoothed positions [B, T, 3]

**Process:**
1. Extract XZ plane: `root_translations_xz = hip_translations[..., [0, 2]]`
2. Keep Y separate: `root_translations_y = hip_translations[..., [1]]`
3. Apply ADMM smoothing to XZ only (multigrid strategy):
   - Margins: 0.06 m per frame (soft constraint radius)
   - Objective: minimize acceleration while staying close to original
   - Multigrid: start coarse (stepsize = 2^levels), double resolution iteratively
   - ADMM iterations: 500 (default), over-relaxation α = 1.8

**Mathematical Formulation (TrajectorySmoother):**
- **Acceleration matrix A:** Second-order differences: `A[i,i-1:i+1] = [-1, 2, -1]`
- **System matrix:** `M = pos_weight·I + A^T A + stepsize·I` (LU-factored for efficiency)
- **ADMM minimization:** 
  ```
  minimize: ||A·x||² + pos_weight·||x - x_target||²
  s.t.     ||z_i - z_t[i]|| ≤ margin[i]  (soft constraint per frame)
  ```

**ADMM Iterations:** x-update, z-update, u-update (100-500 iterations per level)

**Output:** Smooth trajectory preserving Y, smoothing only XZ plane motion

### Why This Design?
- Reduces foot skating (from 7.59 cm/s → 3.87 cm/s in tests)
- Matches animator workflow (direct lines/curves vs. noisy mocap)
- Decouples trajectory control from body pose generation
- Maintains natural ground contact by controlling acceleration

---

## 3. Current HyMotion M2M Root Representation

### Tensor Layout (SMPL-22, 138 dims total)

**After `process_transl()` with `transl_type="abs_rel"`:**

| Section | Dims | Description |
|---------|------|-------------|
| **Absolute Translation** | [0:3] | World-space root position (xyz) |
| **Relative Translation** | [3:6] | Frame-to-frame delta: `Δ = pos[t] - pos[t-1]`, first frame is [0,0,0] |
| **Local Joint Rotations** | [6:138] | SMPL-22 local rotations (row-major 6D): 22 joints × 6 dims |

**Total: 3 + 3 + 132 = 138 dims**

### Properties
- **Coordinate Frame:** Local/parent-relative rotations (SMPL convention)
- **Root Handling:** Decoupled as "abs_rel translation" (absolute + relative split)
- **No Separate Y:** Height embedded in absolute translation dims [1]
- **No Heading Angle:** Heading derived from root rotation (local [0, :6])
- **No Foot Contact:** Not explicitly modeled
- **No Velocity:** Not directly in feature vector

### Conversion to HyMotion Features
```python
# From load_smplx.py: process_transl() + process_smplx_pose()
abs_trans = [T, 3]  # world coordinates
rel_trans = np.concatenate([np.zeros((1, 3)), abs_trans[1:] - abs_trans[:-1]])  # frame deltas
abs_rel_transl = np.concatenate([abs_trans, rel_trans], axis=-1)  # [T, 6]

# Poses: axis-angle → 6D rotation
pose_6d = axis_angle_to_rotation_6d(poses_55_axis_angle, out_type="smpl_22")  # [T, 22, 6]

# Final: concat
motion = np.concatenate([abs_rel_transl, pose_6d.reshape(T, -1)], axis=-1)  # [T, 138]
```

---

## 4. KIMODO ↔ SMPL Conversion

### A. SMPL Pelvis → KIMODO Smooth Root

**Forward (SMPL → KIMODO):**

1. **Get pelvis (root) position:** `pelvis_pos = global_positions[:, root_idx, :]  # [T, 3]`

2. **Smooth XZ plane (ADMM):**
   ```python
   smooth_root = get_smooth_root_pos(pelvis_pos)  # [T, 3]
   # smooth_root[..., [0,2]] = smoothed XZ
   # smooth_root[..., 1] = raw Y (unchanged)
   ```

3. **Compute heading from first 2 joints:**
   ```python
   root_heading_angle = compute_heading_angle(global_positions, skeleton)  # [T]
   global_root_heading = torch.stack([
       torch.cos(root_heading_angle),
       torch.sin(root_heading_angle)
   ], dim=-1)  # [T, 2]
   ```

4. **Compute local joint positions (xz relative, y absolute):**
   ```python
   local_pos = global_positions - smooth_root[:, None, :]  # [T, J, 3]
   local_pos[..., 1] = global_positions[..., 1]  # Y stays absolute
   ```

### B. KIMODO Smooth Root → SMPL Pelvis

**Inverse (KIMODO → SMPL):**

1. **Extract smooth root and heading:**
   ```python
   smooth_root_pos = features[:, slice_dict["smooth_root_pos"]]  # [T, 3]
   global_root_heading = features[:, slice_dict["global_root_heading"]]  # [T, 2]
   local_joints_positions = features[:, slice_dict["local_joints_positions"]]  # [T, J, 3]
   ```

2. **Recover global joint positions:**
   ```python
   posed_joints_from_pos = local_joints_positions.clone()
   posed_joints_from_pos[..., 0] += smooth_root_pos[..., None, 0]  # add X
   posed_joints_from_pos[..., 2] += smooth_root_pos[..., None, 2]  # add Z
   # Y already absolute
   
   # Extract root (pelvis) position
   root_positions = posed_joints_from_pos[..., skeleton.root_idx, :]  # [T, 3]
   ```

3. **Recover root rotation from heading + global rot:**
   ```python
   # Extract pelvis global rotation from global_rot_data
   pelvis_rot_matrix = cont6d_to_matrix(global_rot_data[:, skeleton.root_idx, :])  # [T, 3, 3]
   ```

4. **Convert global rotations → local rotations (FK inverse):**
   ```python
   global_rot_mats = cont6d_to_matrix(global_rot_data)  # [T, J, 6] → [T, J, 3, 3]
   local_rot_mats = global_rots_to_local_rots(global_rot_mats, skeleton)  # [T, J, 3, 3]
   ```

### C. Key Differences in Reconstruction

| Aspect | KIMODO → SMPL | What's Lost/Gained |
|--------|---------------|-------------------|
| **Root Position** | Recovered exactly from smooth_root + local_pos | Exact (smooth_root used as reference) |
| **Root Heading** | [cos(ψ), sin(ψ)] → ψ via atan2 | Exact |
| **Root Rotation** | Global 6D rotation matrix for pelvis | Can reconstruct full 3×3 matrix |
| **Local Rotations** | Global → Local via inverse FK chain | Exact using kinematics |
| **Smooth Info** | Lost during inverse (smooth_root used as reference) | One-way: can't recover original noisy pelvis |

**Critical:** Inverse loses smoothness information. `root_positions` from inverse are relative to smooth_root, not original noisy pelvis.

---

## 5. Heading Angle Computation

### `compute_heading_angle()` Implementation

```python
def compute_heading_angle(global_joints_positions, skeleton):
    """
    Compute heading angle from forward direction vector.
    Uses skeleton-specific "forward direction" logic.
    """
    # Typical: average pelvis and spine1 forward vector
    # Or: use root forward from rotation matrix
    # Result: angle ψ where [cos(ψ), sin(ψ)] is heading in XZ plane
    # Convention: +Z forward, +X right (Y-up)
```

**Convention:** 
- ψ = 0 → forward (+Z)
- ψ = π/2 → left (-X)
- ψ = π → backward (-Z)
- ψ = -π/2 → right (+X)

---

## 6. Global vs. Local Joint Rotations

### KIMODO: Global Rotations (6D continuous)

```python
# Storage: 6D continuous (first 2 columns of 3×3 matrix)
global_rot_6d = matrix_to_cont6d(rot_matrix)  # [J, 3, 3] → [J, 6]

# Decoding:
rot_matrix = cont6d_to_matrix(global_rot_6d)  # [J, 6] → [J, 3, 3]

# Advantage: can directly impute end-effector orientation without IK
```

### HyMotion M2M: Local Rotations (6D continuous, row-major)

```python
# Storage: local rotations (parent-relative)
local_rot_6d_rowmajor = axis_angle_to_rotation_6d(poses_aa)  # [T, J, 3] → [T, J, 6]

# Conversion: row-major ↔ column-major
# Training: row-major [R00, R01, R10, R11, R20, R21]
# rotation_convert.py: column-major [R00, R10, R20, R01, R11, R21]
# fk_utils.py handles conversion via _ROW_TO_COL, _COL_TO_ROW indices

# Advantage: separates body pose (joint angles) from trajectory control
```

---

## 7. Conversion Implementation Strategy

### SMPL-22 (135-dim: 3 trans + 22×6 rot6d) ↔ KIMODO SOMA-30 (333-dim)

**Path (from CLAUDE.md):**

1. **SMPL-22 → Global Rotations & Positions**
   ```python
   local_rot_6d = smpl_poses[:, 6:]  # skip first 6D root rotation
   local_rot_mats = rot6d_to_matrix(local_rot_6d)  # [T, 22, 3, 3]
   
   # Forward FK on SMPL-22 skeleton
   global_rot_mats, global_positions = fk(
       local_rot_mats,
       smpl_trans,  # [T, 3]
       smpl_skeleton
   )  # [T, 22, 3, 3], [T, 22, 3]
   ```

2. **SMPL-22 → SOMA-30 Retarget**
   ```python
   # Copy matching joints (22 → 30):
   soma_global_rots[..., matching_indices, :, :] = smpl_global_rots
   # Fill SOMA-specific joints (neck2, eyes, etc.) via interpolation/rules
   ```

3. **SOMA-30 Global Rotations → Local Rotations**
   ```python
   soma_local_rots = global_rots_to_local_rots(soma_global_rots, soma_skeleton)
   ```

4. **SOMA-30 FK (using SOMA rig)**
   ```python
   _, soma_global_positions = fk(soma_local_rots, soma_root_pos, soma_skeleton)
   # Positions now follow SOMA bone lengths, not SMPL
   ```

5. **SOMA-30 → KIMODO Representation**
   ```python
   smooth_root = get_smooth_root_pos(soma_root_pos)  # [T, 3]
   global_root_heading = compute_heading_angle(soma_global_positions, soma_skeleton)
   local_positions = soma_global_positions - smooth_root[:, None, :]
   local_positions[..., 1] = soma_global_positions[..., 1]  # Y absolute
   global_rot_6d = matrix_to_cont6d(soma_global_rots)  # [T, 30, 6]
   velocities = compute_vel_xyz(soma_global_positions, fps)
   foot_contacts = foot_detect_from_pos_and_vel(...)
   
   # Pack all 333 dims
   features = [smooth_root, global_root_heading, local_positions, 
               global_rot_6d, velocities, foot_contacts]
   ```

---

## 8. Key Implementation Details

### A. Dimensions for Constraint Application (create_conditions)

| Constraint Type | Dims Affected | Semantics |
|-----------------|---------------|-----------|
| `smooth_root_2d` | [0, 2] | XZ plane (2D waypoint) |
| `root_y_pos` | [1] | Height only |
| `global_root_heading` | [3:5] | [cos ψ, sin ψ] |
| `global_joints_rots` | [86:248] | 27 joints × 6 dims |
| `global_joints_positions` | [5:86] | 27 joints × 3 dims (requires smooth_root_2d constrained first) |

### B. Imputation Mechanism (TwostageDenoiser)

```python
# Before each denoising step:
if motion_mask is not None:
    x = x * (1 - motion_mask) + observed_motion * motion_mask
    # Element-wise selection: use observed where mask=1, else use noisy x
    
    x_extended = torch.cat([x, motion_mask], dim=-1)  # 333 + 333 = 666
    # Transformer sees both motion and mask as extra input channel
```

### C. 6D Rotation Representation

```python
# Continuous 6D: first 2 columns of orthonormal 3×3 matrix
# R = [r0, r1, r2] where r0, r1 are the 6D input
# r2 = r0 × r1 (cross product) to enforce orthogonality

# Conversion:
def matrix_to_cont6d(mat: [T, 3, 3]) -> [T, 6]:
    return mat[..., :2, :].reshape(T, 6)  # flatten first 2 columns

def cont6d_to_matrix(rot6d: [T, 6]) -> [T, 3, 3]:
    x = rot6d[..., :3]  # first column
    y = rot6d[..., 3:]  # second column
    z = cross(x, y)  # third column
    return stack([x, y, z], dim=-1)  # [T, 3, 3]
```

---

## 9. Summary: KIMODO vs HyMotion M2M

| Feature | KIMODO | HyMotion M2M |
|---------|--------|--------------|
| **Root Position** | smooth_root (3D, ADMM-smoothed XZ, raw Y) | abs + rel (3D absolute, 3D delta) |
| **Root Heading** | [cos ψ, sin ψ] (2D angle) | Embedded in local root rotation (first 6D) |
| **Joint Rotations** | Global 6D continuous (world-frame) | Local 6D continuous (parent-relative) |
| **Joint Positions** | Local relative (xz rel, y abs) + velocities | Not directly present |
| **Foot Contacts** | Explicit 4D binary flags | Not modeled |
| **Total Dims** | 333 (27 joints) | 138 (22 joints) |
| **Smooth Trajectory** | Yes (ADMM algorithm) | No (abs_rel only) |
| **Constraint Model** | Imputation at every diffusion step | VACE conditioning (reactive/inactive) |
| **IK Required** | No (global positions imputed directly) | Yes (local constraints need FK inversion) |

---

## 10. References in Code

**KIMODO Files:**
- `/ref_repo/KIMODO/kimodo/kimodo/motion_rep/smooth_root.py` - ADMM smoothing
- `/ref_repo/KIMODO/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py` - KimodoMotionRep class
- `/ref_repo/KIMODO/kimodo/kimodo/constraints.py` - constraint types
- `/ref_repo/KIMODO/CLAUDE.md` - tech details

**HyMotion M2M Files:**
- `/hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` - process_transl(), process_smplx_pose()
- `/hftrainer/datasets/motion/motionhub/transforms/fk_utils.py` - FK/IFK, rot6d conversions
