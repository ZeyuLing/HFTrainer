# KIMODO Root/Translation Representation - Detailed Analysis

## Executive Summary

This document provides a complete technical breakdown of:
1. **KIMODO's smooth root + heading representation** (smooth traj + heading approach)
2. **HyMotion M2M's current root representation** (abs_rel transl)
3. **How to convert between KIMODO root and SMPL pelvis**
4. **Key implementation details for bidirectional conversion**

---

## 1. KIMODO Root Representation: "Smooth Traj + Heading"

### 1.1 Overall Motion Feature Layout (Per Frame)

**Total Dimension: 333 dims** (for 27-joint skeleton)

```
[dim:0-332] = concat(
    [0:3]     smooth_root_pos,          # 3 dims
    [3:5]     global_root_heading,      # 2 dims
    [5:86]    local_joints_positions,   # 27 × 3 = 81 dims
    [86:248]  global_rot_data,          # 27 × 6 = 162 dims (6D continuous)
    [248:329] velocities,               # 27 × 3 = 81 dims
    [329:333] foot_contacts             # 4 dims
)
```

### 1.2 Root Component: Smooth Root Position

**Dims: [0:3]** — `smooth_root_pos`

```python
smooth_root_pos = [x_smooth, y_absolute, z_smooth]
  where:
    x_smooth, z_smooth = heavily smoothed horizontal components (via ADMM)
    y_absolute = raw pelvis height (NOT smoothed)
```

**Key characteristics:**
- **Smoothing method**: ADMM-based trajectory smoother (`smooth_root.py:smooth_signal`)
  - Uses multigrid approach: coarse-to-fine resolution
  - Margin constraints: ±0.06 m allowed deviation per frame
  - Minimizes acceleration while staying close to original trajectory
  - More stable than raw pelvis for trajectory following tasks

- **Relationship to SMPL pelvis**:
  - Horizontal (XZ) plane: heavily smoothed version of pelvis position
  - Vertical (Y): exactly the raw pelvis Y coordinate
  - Difference stored separately: `hips_offset = root_positions - smooth_root_pos`

**Creation code** (`kimodo_motionrep.py:83-84`):
```python
smooth_root_pos = get_smooth_root_pos(root_positions)  # SMPL pelvis → smooth root
hips_offset = root_positions - smooth_root_pos         # deviation tracking
```

---

## 1.3 Heading Angle Representation

**Dims: [3:5]** — `global_root_heading`

```python
global_root_heading = [cos(ψ), sin(ψ)]
  where:
    ψ = global heading angle around Y-axis (in radians)
    Computed from hip vector direction
```

**Heading computation** (`feature_utils.py:112-126`):
```python
def compute_heading_angle(posed_joints, skeleton):
    r_hip, l_hip = skeleton.hip_joint_idx  # right/left hip indices
    diff = posed_joints[..., r_hip] - posed_joints[..., l_hip]  # hip vector
    heading_angle = torch.atan2(diff[..., 2], -diff[..., 0])    # atan2(z, -x)
    return heading_angle
```

**Why 2D representation [cos ψ, sin ψ]?**
- Avoids angle discontinuity at ±π
- Natively encodes rotation angle in continuous form
- Invertible: `ψ = atan2(sin_ψ, cos_ψ)`

**Inverse (get angle from heading)**:
```python
psi = torch.atan2(global_root_heading[:, 1], global_root_heading[:, 0])
```

---

## 1.4 Local Joint Positions (Relative to Smooth Root)

**Dims: [5:86]** — `local_joints_positions` (27 joints × 3 dims)

```python
local_joints_positions[..., j] = [x_rel, y_abs, z_rel]
  where:
    x_rel = joint_x - smooth_root_x        # relative to smooth root XZ
    y_abs = joint_y                         # absolute height
    z_rel = joint_z - smooth_root_z
```

**Key insight**: 
- XZ dimensions are **relative** to smooth root (allows translation via smooth root)
- Y dimension is **absolute** (preserves individual joint heights)

**Creation** (`kimodo_motionrep.py:84-86`):
```python
hips_offset = root_positions - smooth_root_pos      # [B,T,3]
hips_offset[..., 1] = root_positions[..., 1]        # force Y = raw pelvis Y
local_joints_positions = (
    fk_positions_relative_to_pelvis + hips_offset[:, :, None]
)
```

---

## 1.5 Global Rotation Representation

**Dims: [86:248]** — `global_rot_data` (27 joints × 6D continuous)

```python
global_rot_data[..., j] = 6D continuous rotation
  = first two rows of the 3×3 global rotation matrix (column-major)
  = [R00, R10, R20, R01, R11, R21]
```

**Critical difference from SMPL**:
- KIMODO: **World-frame global rotations** (no parent dependency)
- SMPL: Local relative rotations (parent-relative)

**Why global rotations?**
- Allows direct imputation of world-space constraints
- Example: "right hand at world position (x,y,z)" can be directly constrained
- No IK chain needed for end-effector control

**Conversion utilities**:
```python
global_rot_matrices = cont6d_to_matrix(global_rot_data)      # [B,T,J,3,3]
global_rot_data = matrix_to_cont6d(global_rot_matrices)      # reverse
```

---

## 1.6 Velocities and Foot Contacts

**Dims: [248:329]** — `velocities` (27 joints × 3 dims)
```python
velocities[..., j] = [vx, vy, vz] = fps * (position[t] - position[t-1])
```
- **Not constrained** during diffusion (derived from positions)
- Used in loss computation for smoothness

**Dims: [329:333]** — `foot_contacts` (4 dims binary)
```python
foot_contacts = [L_heel, L_toe, R_heel, R_toe]  ∈ {0, 1}
```
- **Not constrained** (derived from positions + velocities)
- Computed via contact detection: `foot_detect_from_pos_and_vel`
- **threshold_vel**: 0.10 m/s (velocity threshold)
- **threshold_pos**: 0.15 m (height threshold above ground)

---

## 2. HyMotion M2M Current Root Representation

### 2.1 Motion Feature Layout (Per Frame)

**Total Dimension: 138 dims** (for SMPL-22 skeleton)

```
[dim:0-137] = concat(
    [0:6]     abs_rel_translation,    # 6 dims (3 absolute + 3 relative)
    [6:138]   local_joint_rots_6d     # 22 joints × 6 = 132 dims (6D continuous)
)
```

### 2.2 Root Translation: Absolute + Relative (abs_rel)

**Dims: [0:6]**

```python
abs_rel_transl = concat(
    abs_trans[t],      # dims [0:3] — absolute world position of pelvis
    rel_trans[t]       # dims [3:6] — relative velocity (position[t] - position[t-1])
)
```

**Where:**
- `abs_trans[0] = [0, 0, 0]` (arbitrary global reference)
- `abs_trans[t] = abs_trans[t-1] + rel_trans[t]` (cumulative)
- `rel_trans[t] = abs_trans[t] - abs_trans[t-1]` for t > 0

**Advantages:**
- Decouples trajectory and velocity
- Relative component is more stable for differencing
- Flexible encoding of motion trajectory

**Disadvantages:**
- Not smoothed (raw pelvis)
- Direct trajectory constraints require solving inverse problem
- Heading implicitly in relative motion direction (not explicit)

### 2.3 Joint Rotations: Local 6D (SMPL-style)

**Dims: [6:138]** (22 joints × 6D continuous)

```python
local_joints_rots_6d[..., j] = 6D continuous representation
  of local (parent-relative) rotation for joint j
```

**Coordinate frame**: **Local/relative** (parent-relative, SMPL convention)

**Key difference from KIMODO**:
- Requires FK to compute global positions for world-space constraints
- Hierarchical dependency: child rotation depends on parent

---

## 3. Conversion: KIMODO Root → SMPL Pelvis

### 3.1 Forward Direction: SMPL Pelvis → KIMODO Root

**Input:**
- SMPL: `abs_trans[T, 3]` (world pelvis position)
- SMPL: `local_joint_rots[T, J, 3, 3]` (local rotation matrices)
- SMPL: `root_positions[T, 3]` (same as abs_trans)

**Process** (`kimodo_motionrep.py:__call__`):

```python
# Step 1: Compute FK to get global positions & rotations
global_joints_rots, global_joints_positions, _ = fk(
    local_joint_rots, root_positions, skeleton
)

# Step 2: Extract heading from global joint positions
root_heading_angle = compute_heading_angle(global_joints_positions, skeleton)
global_root_heading = torch.stack([
    torch.cos(root_heading_angle),
    torch.sin(root_heading_angle)
], dim=-1)

# Step 3: Smooth the root trajectory
smooth_root_pos = get_smooth_root_pos(root_positions)

# Step 4: Make joint positions relative to smooth root
hips_offset = root_positions - smooth_root_pos
hips_offset[..., 1] = root_positions[..., 1]  # Y stays absolute
local_joints_positions = fk_positions_relative_to_pelvis + hips_offset[..., None]

# Step 5: Convert global rotations to 6D continuous
global_rot_data = matrix_to_cont6d(global_joints_rots)

# Step 6: Compute velocities and foot contacts
velocities = compute_vel_xyz(global_joints_positions, fps)
foot_contacts = foot_detect_from_pos_and_vel(
    global_joints_positions, velocities, skeleton, 0.15, 0.10
)

# Step 7: Pack into motion feature vector
motion_features = concat([
    smooth_root_pos,           # [T, 3]
    global_root_heading,       # [T, 2]
    local_joints_positions,    # [T, J, 3]
    global_rot_data,           # [T, J, 6]
    velocities,                # [T, J, 3]
    foot_contacts,             # [T, 4]
])
```

---

### 3.2 Inverse Direction: KIMODO Root → SMPL Pelvis

**Input:**
- KIMODO: `smooth_root_pos[T, 3]`
- KIMODO: `global_root_heading[T, 2]`
- KIMODO: `local_joints_positions[T, J, 3]`
- KIMODO: `global_rot_data[T, J, 6]`

**Process** (`kimodo_motionrep.py:inverse`):

```python
# Step 1: Decode 6D rotations to matrices
global_rot_mats = cont6d_to_matrix(global_rot_data)  # [B,T,J,3,3]

# Step 2: Convert global rotations to local rotations
local_rot_mats = global_rots_to_local_rots(global_rot_mats, skeleton)
# This uses inverse FK: local[j] = parent_global^T @ global[j]

# Step 3: Reconstruct global joint positions from local joint positions
posed_joints_from_pos = local_joints_positions.clone()
posed_joints_from_pos[..., 0] += smooth_root_pos[..., None, 0]  # add smooth_root_x
posed_joints_from_pos[..., 2] += smooth_root_pos[..., None, 2]  # add smooth_root_z

# Step 4: Extract pelvis position
root_positions = posed_joints_from_pos[..., skeleton.root_idx, :]
# root_positions ≈ smooth_root_pos (with small deviation)

# Step 5: Optional - run FK to recompute positions from rotations
local_rot_mats, posed_joints, _ = skeleton.fk(local_rot_mats, root_positions)

# Output structure
output = {
    'local_rot_mats': local_rot_mats,       # [B,T,J,3,3] — SMPL-compatible
    'global_rot_mats': global_rot_mats,     # [B,T,J,3,3]
    'posed_joints': posed_joints,            # [B,T,J,3]
    'root_positions': root_positions,        # [B,T,3] — SMPL pelvis
    'smooth_root_pos': smooth_root_pos,      # [B,T,3]
    'global_root_heading': global_root_heading,
}
```

**Key insight**: 
- Conversion is **lossy** if smooth_root_pos was smoothed
- The smoothing removes high-frequency noise, so recovery is approximate
- But `root_positions` from step 4 contains the "true" pelvis trajectory with smoothing applied

---

## 3.3 Global to Local Rotation Conversion

### FK (Forward Kinematics): Local → Global

```python
# SMPL22_PARENTS: parent index for each joint
for j, parent in enumerate(SMPL22_PARENTS):
    if parent < 0:
        global_rot[..., j] = local_rot[..., j]
    else:
        global_rot[..., j] = global_rot[..., parent] @ local_rot[..., j]
```

### IFK (Inverse FK): Global → Local

```python
for j, parent in enumerate(SMPL22_PARENTS):
    if parent < 0:
        local_rot[..., j] = global_rot[..., j]
    else:
        # For pure rotations: inv(R) = R^T
        parent_inv = global_rot[..., parent].transpose(-2, -1)
        local_rot[..., j] = parent_inv @ global_rot[..., j]
```

**Implementation files**:
- **Numpy (dataset transforms)**: `fk_utils.py:local_to_global_rot6d()`, `fk_utils.py:global_to_local_rot6d()`
- **Torch (inference)**: `fk_utils.py:local_to_global_rot6d_torch()`, `fk_utils.py:global_to_local_rot6d_torch()`

---

## 4. Constraint Application: Imputation Mechanism

### 4.1 Constraint Types in KIMODO

KIMODO's `create_conditions()` supports 5 constraint types:

| Constraint | Dims | Purpose |
|---|---|---|
| `smooth_root_2d` | [0,2] | Constrain 2D trajectory (X,Z plane) |
| `root_y_pos` | [1] | Constrain pelvis height |
| `global_root_heading` | [3:5] | Constrain heading angle [cos ψ, sin ψ] |
| `global_joints_rots` | [86:248] | Constrain per-joint global rotations (6D) |
| `global_joints_positions` | [5:86] | Constrain per-joint global positions (3D) |

### 4.2 Imputation During Diffusion

**Before each denoising step** (`twostage_denoiser.py:98-103`):

```python
# Create mask and observed motion tensors
observed_motion = torch.zeros(T, motion_rep_dim)
motion_mask = torch.zeros(T, motion_rep_dim, dtype=bool)

# Fill in constrained dimensions
# For each constraint type, set observed_motion[frame_idx, dims] = constraint_value
# and motion_mask[frame_idx, dims] = True

# During diffusion forward pass:
x_t = x_t * (1 - motion_mask) + observed_motion * motion_mask  # Direct imputation
x_extended = torch.cat([x_t, motion_mask], dim=-1)  # [T, 2×motion_rep_dim]
x_pred = transformer(x_extended, text_embedding, timestep)  # Model predicts denoising
```

**Key mechanism**:
- Constrained dimensions forcibly replaced at **every denoising step**
- Mask appended as extra input channel tells model which dims are constrained
- Model learns to denoise unconstrained dimensions while preserving constrained ones

---

## 5. Key Implementation Details for Conversion

### 5.1 Row-Major vs Column-Major 6D Rotation

**Training data convention**: Row-major 6D
```
rot6d_row_major = [R00, R01, R10, R11, R20, R21]
  where [R00, R01, ...] are first two rows of 3×3 matrix, in row-major order
```

**Rotation convert library**: Column-major 6D
```
rot6d_col_major = [R00, R10, R20, R01, R11, R21]
  where [R00, R10, R20, R01, R11, R21] is column-major ordering
```

**Conversion reorder indices**:
```python
_ROW_TO_COL = [0, 2, 4, 1, 3, 5]    # row-major → column-major
_COL_TO_ROW = [0, 3, 1, 4, 2, 5]    # column-major → row-major
```

**Implementation** (`fk_utils.py`):
```python
def _rot6d_row_to_matrix_np(rot6d_row):
    rot6d_col = rot6d_row[..., _ROW_TO_COL]  # reorder
    return rotation_6d_to_matrix(rot6d_col)  # use lib function

def _matrix_to_rot6d_row_np(mat):
    rot6d_col = matrix_to_rotation_6d(mat)    # lib function
    return rot6d_col[..., _COL_TO_ROW]  # reorder back
```

---

### 5.2 SMPL Kinematic Tree

SMPL-22 parent indices (`fk_utils.py:29-52`):

```python
SMPL22_PARENTS = [
    -1,    # 0: Pelvis (root)
    0, 0, 0,     # 1-3: L_Hip, R_Hip, Spine1
    1, 2,        # 4-5: L_Knee, R_Knee
    3,           # 6: Spine2
    4, 5,        # 7-8: L_Ankle, R_Ankle
    6, 7, 8,     # 9-11: Spine3, L_Foot, R_Foot
    9, 9, 9,     # 12-14: Neck, L_Collar, R_Collar
    12, 13, 14,  # 15-17: Head, L_Shoulder, R_Shoulder
    16, 17,      # 18-19: L_Elbow, R_Elbow
    18, 19,      # 20-21: L_Wrist, R_Wrist
]
```

---

### 5.3 Smooth Root ADMM Algorithm

**Algorithm** (`smooth_root.py:smooth_signal`):

```python
def smooth_signal(x, margins, pos_weight=0, admm_iters=500, alpha_overrelax=1.8):
    """
    Multigrid trajectory smoothing with margin constraints.
    
    Args:
        x: Input trajectory [T, D]
        margins: Allowed radius per frame [T], default 0.06 m
        pos_weight: How much to preserve original signal (0 = full smoothing)
        admm_iters: ADMM iterations per multigrid level
        alpha_overrelax: Over-relaxation coefficient (1.8 default)
    
    Returns:
        x_smoothed: Smoothed trajectory [T, D]
    """
    # Build acceleration matrix A (second derivative)
    A[i, i-1:i+2] = [-1, 2, -1]  # Laplacian for smoothness
    
    # System matrix: M = pos_weight*I + A^T @ A
    # Solves: minimize ||x - x_original||^2 + ||A @ x||^2
    #         subject to: ||x[t] - target[t]|| <= margin[t]
    
    # Multigrid approach: start coarse, interpolate, refine
    for level in [coarse, medium, fine]:
        for admm_iteration in range(admm_iters):
            x_update(x, z, u, constraints)
            z_update(z, x, u, margins)        # Project to margins
            u_update(u, x, z, alpha_overrelax)
    
    return x_smoothed
```

**Default margin**: 0.06 m (6 cm) per frame

---

## 6. Practical Conversion Example

### 6.1 Complete Pipeline: SMPL → KIMODO → SMPL

```python
# =============================================================================
# INPUT: SMPL Motion (standardized for HyMotion M2M)
# =============================================================================
import torch
import numpy as np
from kimodo import KimodoMotionRep
from fk_utils import local_to_global_rot6d, global_to_local_rot6d

# Assume: SMPL data in HyMotion format
smpl_motion = torch.load("motion.pth")  # [T, 138] = [T, 6 + 22*6]
abs_rel_transl = smpl_motion[:, :6]     # [T, 6]
local_rot6d = smpl_motion[:, 6:]        # [T, 132]

# Extract absolute translation
abs_trans = abs_rel_transl[:, :3]       # Cumulative position

# Convert 6D local rotations to matrices
local_rot_matrices = rot6d_to_matrix(local_rot6d)  # [T, 22, 3, 3]

# =============================================================================
# STEP 1: SMPL → KIMODO
# =============================================================================
kimodo_rep = KimodoMotionRep(skeleton=soma30_skeleton, fps=30)

# Convert to KIMODO representation
kimodo_features = kimodo_rep(
    local_joint_rots=local_rot_matrices.unsqueeze(0),  # [1, T, 22, 3, 3]
    root_positions=abs_trans.unsqueeze(0),              # [1, T, 3]
    to_normalize=False,
)
# Output shape: [1, T, 333]

# =============================================================================
# STEP 2: Access KIMODO components
# =============================================================================
kimodo_features = kimodo_features.squeeze(0)  # [T, 333]

smooth_root_pos = kimodo_features[:, 0:3]
global_root_heading = kimodo_features[:, 3:5]
local_joints_positions = kimodo_features[:, 5:86].reshape(T, 27, 3)
global_rot_data = kimodo_features[:, 86:248].reshape(T, 27, 6)
velocities = kimodo_features[:, 248:329].reshape(T, 27, 3)
foot_contacts = kimodo_features[:, 329:333]

print(f"Smooth root (first frame): {smooth_root_pos[0]}")
print(f"Heading angle: {torch.atan2(global_root_heading[0, 1], global_root_heading[0, 0])} rad")

# =============================================================================
# STEP 3: KIMODO → SMPL (Inverse)
# =============================================================================
output = kimodo_rep.inverse(
    features=kimodo_features.unsqueeze(0),  # [1, T, 333]
    is_normalized=False,
    posed_joints_from="rotations",
    return_numpy=False,
)

recovered_local_rot_mats = output['local_rot_mats'].squeeze(0)  # [T, 22, 3, 3]
recovered_root_positions = output['root_positions'].squeeze(0)   # [T, 3]

# Convert back to 6D
recovered_local_rot6d = matrix_to_rot6d(recovered_local_rot_mats)  # [T, 22, 6]

# Recompute abs_rel translation
recovered_rel_transl = torch.diff(
    recovered_root_positions, 
    dim=0, 
    prepend=recovered_root_positions[[0]] * 0
)
recovered_abs_rel_transl = torch.cat([
    recovered_root_positions,
    recovered_rel_transl
], dim=-1)  # [T, 6]

recovered_motion = torch.cat([
    recovered_abs_rel_transl,
    recovered_local_rot6d.reshape(T, -1)
], dim=-1)  # [T, 138]

# Verify shapes
assert recovered_motion.shape == smpl_motion.shape
print(f"Original motion shape: {smpl_motion.shape}")
print(f"Recovered motion shape: {recovered_motion.shape}")

# Compare root position error
root_pos_error = torch.norm(recovered_root_positions - abs_trans)
print(f"Root position reconstruction error: {root_pos_error:.6f} m")
```

---

## 7. Summary Table: KIMODO vs HyMotion

| Aspect | KIMODO | HyMotion M2M |
|---|---|---|
| **Total dims** | 333 (27 joints) | 138 (22 joints) |
| **Root translation dims** | 3 (smooth root) | 6 (abs + rel) |
| **Root heading dims** | 2 ([cos ψ, sin ψ]) | 0 (implicit in relative motion) |
| **Joint rotation type** | Global (world-frame) | Local (parent-relative) |
| **Joint rotation dims** | 27×6 = 162 | 22×6 = 132 |
| **Smooth trajectory** | Yes (ADMM smoother) | No (raw pelvis) |
| **Heading explicit** | Yes [cos ψ, sin ψ] | No |
| **Local joint pos** | Yes (relative to root) | No |
| **Velocities** | Yes (27 joints × 3) | No |
| **Foot contacts** | Yes (4 dims) | No |
| **Constraint types** | 5 specific types | 1 generic (motion_mask) |
| **IK required** | No (global pos direct) | Yes (if constraining world positions) |

---

## 8. Key Files Reference

### KIMODO Motion Representation
- `kimodo_motionrep.py`: Main KimodoMotionRep class (forward/inverse)
- `smooth_root.py`: ADMM smoothing algorithm
- `feature_utils.py`: Heading angle, velocity, rotation helpers
- `conditioning.py`: Constraint application logic

### HyMotion FK Utilities
- `fk_utils.py`: Local ↔ global rotation conversion (row/col major handling)
- `load_smplx.py`: SMPL-X loading and motion processing

---

## Appendix: Quick Reference Formulas

### Heading angle extraction
```
ψ = atan2(heading[1], heading[0])
heading = [cos(ψ), sin(ψ)]
```

### Reconstruct pelvis from smooth root
```
root_pos_reconstructed = smooth_root_pos  (approximately, if smooth was heavy)
root_pos_exact = [smooth_root_x + local_joint_positions[0,0], 
                  local_joint_positions[0,1],
                  smooth_root_z + local_joint_positions[0,2]]
```

### Local to global rotation
```
global[0] = local[0]
global[j] = global[parent[j]] @ local[j]  for j > 0
```

### Global to local rotation
```
local[0] = global[0]
local[j] = global[parent[j]]^T @ global[j]  for j > 0
```

### 6D rotation conversion
```
rot6d_col_major = [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]
rot6d_row_major = [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]
```
