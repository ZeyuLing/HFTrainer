# KIMODO → HyMotion M2M: Concrete Mapping Examples

## Example 1: Single Frame Representation

### Scenario
- SMPL skeleton with pelvis at world position (1.0, 0.9, 2.0)
- Body in neutral pose (mostly identity rotations)
- Head turning 30° left (yaw = +π/6)

### KIMODO 333D Representation

```
Frame 0 of sequence:

[0:3]     smooth_root_pos = [1.0, 0.9, 2.0]
          # Smoothed XZ: [1.0, 2.0], raw Y: 0.9

[3:5]     global_root_heading = [cos(π/6), sin(π/6)] = [0.866, 0.500]
          # 30° left turn (ψ = π/6)

[5:86]    local_joints_positions (27 joints × 3)
          # e.g., joint 1: [-0.1, 0.2, 0.0]  (relative XZ, absolute Y)
          # e.g., joint 3 (head): [0.0, 1.5, -0.05]

[86:248]  global_rot_data (27 joints × 6)
          # e.g., joint 0 (pelvis): [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  (identity)
          # e.g., joint 15 (head): [cos(30°), sin(30°), ...]

[248:329] velocities (27 joints × 3)
          # First frame: mostly zeros (or small from smoothing)

[329:333] foot_contacts = [1.0, 0.0, 1.0, 0.0]
          # Left foot & right foot in contact
```

### HyMotion M2M 138D Representation

```
Frame 0 of sequence:

[0:3]     absolute_translation = [1.0, 0.9, 2.0]
          # Exact world position

[3:6]     relative_translation = [0.0, 0.0, 0.0]
          # First frame has zero delta

[6:138]   local_joint_rotations (22 joints × 6, row-major 6D)
          # e.g., joint 0 (pelvis): [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  (identity matrix)
          # e.g., joint 15 (head): [cos6d_x, ..., cos6d_y, ...]
          #                        (parent-relative rotation)
```

### Key Differences
1. **KIMODO has 333 dims, HyMotion has 138 dims** (27 vs 22 joints)
2. **KIMODO smooth_root [1.0, 0.9, 2.0] = pelvis (no smoothing on first frame typically)**
3. **HyMotion's abs_translation [1.0, 0.9, 2.0] is identical to pelvis**
4. **KIMODO heading [0.866, 0.500] separate; HyMotion encodes in root rotation**
5. **KIMODO has foot_contacts [1,0,1,0]; HyMotion infers from positions**

---

## Example 2: Multi-Frame Trajectory (walking forward)

### Scenario
- 10 frames at 30 fps (0.33 seconds)
- Character walks forward (+Z direction) at 1.5 m/s
- Slight sinusoidal sway in X

### Pelvis Trajectory (Noisy mocap)
```
Frame  X       Y       Z       Notes
0      0.0     0.9     0.0     Start
1      0.01    0.91    0.05    Small jitter
2      0.02    0.89    0.10    Noisy Y
3      0.03    0.92    0.15
4      0.02    0.90    0.20    Some backtrack in X
5      0.04    0.91    0.25
6      0.05    0.89    0.30    Continued jitter
7      0.06    0.92    0.35
8      0.07    0.90    0.40
9      0.08    0.91    0.45
```

### After ADMM Smoothing (0.06m margin, 500 iters)
```
Frame  smooth_X  smooth_Y  smooth_Z  (KIMODO output)
0      0.0       0.9       0.0       # Smoothed to remove jitter
1      0.01      0.91      0.05      # Follows original trajectory
2      0.02      0.89      0.10      # But smoother acceleration
3      0.03      0.92      0.15      # Better for control
4      0.03      0.90      0.20      # X backtrack reduced
5      0.04      0.91      0.25
6      0.05      0.89      0.30
7      0.06      0.92      0.35
8      0.07      0.90      0.40
9      0.08      0.91      0.45
```

### KIMODO Feature Encoding
```
Frame  smooth_root_pos        global_root_heading  local_joint_pos...
0      [0.0, 0.9, 0.0]       [cos(0), sin(0)]    [joints relative]
1      [0.01, 0.91, 0.05]    [0.999, 0.035]     
2      [0.02, 0.89, 0.10]    [0.998, 0.061]     
...
9      [0.08, 0.91, 0.45]    [0.999, 0.045]
```

### HyMotion M2M Feature Encoding
```
Frame  abs_trans          rel_trans         local_rot_6d...
0      [0.0, 0.9, 0.0]   [0.0, 0.0, 0.0]  [joints local]
1      [0.01, 0.91, 0.05] [0.01, 0.01, 0.05]
2      [0.02, 0.89, 0.10] [0.01, -0.02, 0.05]
3      [0.03, 0.92, 0.15] [0.01, 0.03, 0.05]
4      [0.02, 0.90, 0.20] [-0.01, -0.02, 0.05]  ← noisy!
5      [0.04, 0.91, 0.25] [0.02, 0.01, 0.05]
...
9      [0.08, 0.91, 0.45] [0.00, 0.00, 0.05]   ← close to zero
```

### Observation
- **KIMODO smooth_root** removes X-jitter, shows clean trajectory
- **HyMotion rel_trans[4] = [-0.01, -0.02, 0.05]** shows backward step (noisy)
- **KIMODO for trajectory following:** animator draws smooth line, KIMODO matches it
- **HyMotion for trajectory following:** must filter noisy deltas or retarget

---

## Example 3: Constraint Application in KIMODO (vs VACE in HyMotion)

### Scenario: "End-effector control - right hand to (2.0, 1.5, 1.0) at frame 50"

### KIMODO Imputation (create_conditions)

```python
# Input constraint:
constraint = EndEffectorConstraintSet(
    frame_indices=[50],
    joint_names=["RightHand"],
    global_joints_positions=torch.tensor([[[2.0, 1.5, 1.0]]])
)

# Step 1: Convert to observation + mask
# Assume smooth_root at frame 50 is [0.5, 0.9, 1.5]
# Hand is joint index 21 in SOMA-30

# Step 2: Compute local position
local_hand_pos_xz = [2.0 - 0.5, 0.0, 1.0 - 1.5] = [1.5, 0.0, -0.5]
local_hand_pos_full = [1.5, 1.5, -0.5]  # Y=1.5 absolute

# Step 3: Fill observed_motion and motion_mask
observed_motion[50, 5:86][21*3:(21+1)*3] = [1.5, 1.5, -0.5]  # dims [68:71]
motion_mask[50, 5:86][21*3:(21+1)*3] = [True, True, True]

# Also fill root if not already constrained:
observed_motion[50, 0:3][[0,2]] = [0.5, 1.5]  # smooth_root_2d
motion_mask[50, 0:3][[0,2]] = [True, True]

# Step 4: During diffusion
x_t = diffusion_noisy_motion  # shape [batch, 333]
x_t = x_t * (1 - motion_mask) + observed_motion * motion_mask
# Hand dims forcibly set to [1.5, 1.5, -0.5]
# Other dims allowed to denoise

x_extended = torch.cat([x_t, motion_mask], dim=-1)  # 666 dims
output = transformer(x_extended, text_embed, time_embed)
```

**Result:** Hand position locked at (2.0, 1.5, 1.0), body denoises freely around it.

### HyMotion M2M VACE Conditioning (Equivalent Task)

```python
# Input constraint: same end-effector to (2.0, 1.5, 1.0) at frame 50

# Step 1: Create universal mask [T, 138]
mask = torch.zeros(T, 138)
# Right wrist is joint 21 in SMPL-22
# Dims [6 + 21*6 : 6 + 21*6 + 6] = [132:138]
mask[50, 132:138] = 1.0

# Step 2: Prepare VACE inputs
# (Assume motion_clean is GT, motion_noisy is diffusion input)
motion_observed = motion_clean.clone()
motion_unobserved = motion_noisy.clone()

# Step 3: Condition vectors
reactive = motion_observed * mask
inactive = motion_unobserved * (1 - mask)

# Step 4: Concat with noise
x_extended = torch.cat([
    motion_noisy,
    inactive,
    reactive,
    mask
], dim=-1)  # [T, 138*4] = [T, 552]
output = model(x_extended, text_embed, time_embed)
```

**Difference:**
- **KIMODO:** Direct imputation at every step (hard constraint)
- **HyMotion:** VACE channels tell model what's observed vs. unobserved (soft guidance)
- **KIMODO:** Can exactly match constraint; HyMotion learns to respect mask probabilistically

---

## Example 4: 6D Rotation Encoding

### Single Joint Rotation: 30° around Y-axis

### KIMODO: Global 6D
```python
# Rotation matrix (yaw 30° around Y):
R = [[cos(30°), 0, sin(30°)],
     [0,         1, 0      ],
     [-sin(30°), 0, cos(30°)]]

  = [[0.866,  0, 0.500],
     [0,      1, 0    ],
     [-0.500, 0, 0.866]]

# Extract first 2 columns and flatten (6D continuous):
global_rot_6d = [0.866, 0, -0.500, 0, 1, 0]
```

### HyMotion M2M: Local 6D (row-major)
```python
# For SMPL root (joint 0), this is relative to world frame (parent=-1)
# So local = global in this case

# But stored as row-major per load_smplx.py:
# axis_angle_to_rotation_6d with conversion:
#   col-major [R00, R10, R20, R01, R11, R21] 
#   → row-major [R00, R01, R10, R11, R20, R21]

# From our matrix:
col_major = [0.866, 0, -0.500, 0, 1, 0]  # [R00, R10, R20, R01, R11, R21]
row_major = [0.866, 0, 0, 1, -0.500, 0]  # [R00, R01, R10, R11, R20, R21]

local_rot_6d = row_major  # HyMotion storage
```

### Conversion Path
```
KIMODO global_rot_6d [J, 6]
  → cont6d_to_matrix [J, 3, 3]
  → matrix_to_rot6d_row [J, 6] (HyMotion format)
  → HyMotion local_rot_6d [J, 6]
```

---

## Example 5: Constraint Dimension Mapping

### Scenario: Full-body keyframe at frame 30 (KIMODO style)

```python
# Input: 27-joint skeleton, all joints + root constrained

# KIMODO dimensions (333D):
dims_constrained = {
    "smooth_root_2d":        [0, 2],         # 2 dims
    "root_y_pos":            [1],            # 1 dim
    "global_root_heading":   [3, 4],         # 2 dims
    "local_joints_pos":      [5:86],         # 27*3 = 81 dims (all 27 joints)
    "global_rot_data":       [86:248],       # 27*6 = 162 dims (all 27 joints)
    # velocities NOT constrained (computed from positions)
    # foot_contacts NOT constrained (computed from positions/velocities)
}
# Total constrained: 2 + 1 + 2 + 81 + 162 = 248 dims out of 333

# HyMotion M2M equivalent (138D):
# Need to:
# 1. Map 27-joint root to 22-joint root
# 2. Map 27 local positions to 22 local rotations (via FK)
# 3. Map heading + smooth_root_2d + root_y to abs/rel translation

dims_constrained_hymotion = {
    "abs_translation":       [0:3],          # 3 dims (root position)
    "rel_translation":       [3:6],          # 3 dims (delta)
    "local_rot_6d":          [6:138],        # 22*6 = 132 dims (all 22 joints)
}
# Total: 3 + 3 + 132 = 138 dims (full sequence constrained)
```

---

## Practical Conversion Pseudocode

```python
def kimodo_to_hymotion_feature(kimodo_features):
    """
    Convert 333D KIMODO features to 138D HyMotion features.
    Assumes SOMA-30 input, outputs SMPL-22.
    """
    # Unpack KIMODO
    smooth_root_pos = kimodo_features[0:3]                    # [T, 3]
    global_root_heading = kimodo_features[3:5]                # [T, 2]
    local_joints_positions = kimodo_features[5:86]            # [T, 27*3]
    global_rot_data = kimodo_features[86:248]                 # [T, 27*6]
    velocities = kimodo_features[248:329]                     # [T, 27*3]
    foot_contacts = kimodo_features[329:333]                  # [T, 4]
    
    # Step 1: Reconstruct global positions (27 joints)
    local_joints_pos_3d = local_joints_positions.reshape(T, 27, 3)
    global_joints_positions = local_joints_pos_3d.clone()
    global_joints_positions[..., 0] += smooth_root_pos[..., 0, None]
    global_joints_positions[..., 2] += smooth_root_pos[..., 2, None]
    # Y already absolute
    
    # Step 2: Convert heading to rotation matrix (30D)
    heading_angle = atan2(global_root_heading[1], global_root_heading[0])
    # Store in root rotation somehow
    
    # Step 3: Convert 6D global rotations to 3x3 matrices (27 joints)
    global_rot_mats = cont6d_to_matrix(global_rot_data)  # [T, 27, 3, 3]
    
    # Step 4: Convert global → local rotations (SOMA-30)
    local_rot_mats = global_rots_to_local_rots(global_rot_mats)  # [T, 27, 3, 3]
    
    # Step 5: Retarget SOMA-30 → SMPL-22
    smpl_global_positions = global_joints_positions[..., soma_to_smpl_indices, :]  # [T, 22, 3]
    smpl_local_rot_mats = local_rot_mats[..., soma_to_smpl_indices, :, :]  # [T, 22, 3, 3]
    
    # Step 6: Extract SMPL root position and heading
    smpl_root_pos = smpl_global_positions[:, 0, :]  # [T, 3]
    smpl_heading_from_rot = extract_heading_from_matrix(smpl_local_rot_mats[:, 0])
    
    # Step 7: Compute relative translation
    rel_trans = smpl_root_pos[1:] - smpl_root_pos[:-1]
    rel_trans = np.concatenate([[0, 0, 0]], rel_trans, axis=0)
    abs_rel_trans = np.concatenate([smpl_root_pos, rel_trans], axis=-1)  # [T, 6]
    
    # Step 8: Convert local rot matrices to 6D row-major
    local_rot_6d = matrix_to_rot6d_row(smpl_local_rot_mats)  # [T, 22, 6]
    
    # Step 9: Concat
    hymotion_features = np.concatenate([
        abs_rel_trans,                               # [T, 6]
        local_rot_6d.reshape(T, 22*6)               # [T, 132]
    ], axis=-1)  # [T, 138]
    
    return hymotion_features
```

---

## Summary: Key Mapping Points

| KIMODO | HyMotion M2M | Notes |
|--------|--------------|-------|
| **smooth_root_pos [0:3]** | abs_trans [0:3] | Root position (direct map) |
| **global_root_heading [3:5]** | Embedded in local_rot_6d[0, :6] | Heading as root yaw |
| **local_joints_pos [5:86]** | Derived from FK of local_rot_6d | Positions not stored |
| **global_rot_data [86:248]** | local_rot_6d [6:138] | Global → Local via IFK |
| **velocities [248:329]** | Not in HyMotion | Computed on-the-fly |
| **foot_contacts [329:333]** | Not in HyMotion | Inferred from sliding |

