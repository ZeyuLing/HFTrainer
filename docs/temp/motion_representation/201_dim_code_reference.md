# 201-Dim Format - Code Reference & Examples

## Quick Answer

The **66 dimensions** in dims [135:201] of the 201-dim format are:

```
22 joints × 3D positions (Scheme D encoding)
= 22 × 3 = 66 dimensions (INCLUDING pelvis)
```

**NOT:** 21×3 + something else, and **NOT** RIC in narrow sense. It's world positions with a specific encoding scheme.

---

## Code Evidence

### 1. From `compute_201dim_stats.py` (lines 106-128)

This is the definitive source for how 201-dim positions are computed:

```python
def _compute_position_channels(
    motion_135_local: np.ndarray, 
    bone_offsets: torch.Tensor
) -> np.ndarray:
    """Compute 66-dim position channels from LOCAL rotation motion via FK.
    
    Returns:
        (T, 66) position channels in Scheme D 
        (XZ rel pelvis, Y absolute world).
    """
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

    motion_t = torch.from_numpy(motion_135_local).float()
    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(motion_t, bone_offsets)
        #                                       ↑ returns (T, 22, 3)

    # Scheme D: XZ relative to pelvis, Y absolute world
    pelvis_world = world_pos[:, 0:1, :]  # (T, 1, 3)
    joint_pos_D = world_pos.clone()
    joint_pos_D[..., 0] -= pelvis_world[..., 0]  # X: relative to pelvis
    joint_pos_D[..., 2] -= pelvis_world[..., 2]  # Z: relative to pelvis
    # Y: keep absolute world height

    return joint_pos_D.reshape(-1, 66).numpy()  # ← 22 * 3 = 66
```

**Key points:**
- Uses FK to get world positions: (T, 22, 3)
- Applies Scheme D encoding
- **ALL 22 joints included** (no filtering)
- Flattens to 66 dims

### 2. From `compute_198dim.py` (lines 41-71)

Shows the difference - 198-dim EXCLUDES pelvis:

```python
def compute_position_channels(
    motion_135: Tensor,
    bone_offsets: Tensor,
) -> Tensor:
    """Compute 63-dim position channels from 135-dim.
    
    Returns:
        (*, 63) position channels (21 joints, Scheme D).
        ↑ NOTE: 21 joints = 63 dims (pelvis excluded!)
    """
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

    with torch.no_grad():
        world_pos, _, _, _ = motion135_to_fk(
            motion_135, bone_offsets, rotation_space='local'
        )
        #         returns (*, 22, 3)

    pelvis_world = world_pos[..., 0:1, :]  # (*, 1, 3)

    # Scheme D: XZ relative to pelvis, Y absolute world
    joint_pos = world_pos[..., 1:, :].clone()  # (*, 21, 3) ← SKIP PELVIS!
    joint_pos[..., 0] -= pelvis_world[..., 0]  # X: relative to pelvis
    joint_pos[..., 2] -= pelvis_world[..., 2]  # Z: relative to pelvis
    # Y: keep absolute world height

    return joint_pos.reshape(*leading, 63)  # ← 21 * 3 = 63
```

**Comparison:**
- 201-dim: `world_pos[..., 0:, :]` → 22 joints → 66 dims
- 198-dim: `world_pos[..., 1:, :]` → 21 joints (skip index 0) → 63 dims

### 3. From `load_o6dp.py` (lines 1-20)

The o6dp_1103 format comment confirms the layout:

```python
"""Load pre-processed o6dp_1103 motion representation.

For 22 joints (joints_num=22), the layout is 201 dims:
  - [0:3]        abs translation (3)
  - [3:9]        root global rot6d (6)
  - [9:135]      body local rot6d ((22-1)*6=126)
  - [135:201]    RIC joints 3D (22*3=66)  ← THIS IS IT!
                                    ↑
                                All 22 joints
"""
```

### 4. From `load_o6dp.py` (lines 31-54)

Extraction function showing the structure:

```python
def _extract_22j_from_52j(motion_52j: np.ndarray) -> np.ndarray:
    """Extract 22-joint 201-dim representation from 52-joint 471-dim."""
    T = motion_52j.shape[0]

    # Parse 52-joint layout
    transl = motion_52j[:, 0:3]             # (T, 3)
    root_rot6d = motion_52j[:, 3:9]         # (T, 6)
    body_rot6d_52 = motion_52j[:, 9:315]    # (T, 306) = 51 joints * 6
    ric_52 = motion_52j[:, 315:471]         # (T, 156) = 52 joints * 3

    # Extract first 21 body joints
    body_rot6d_22 = body_rot6d_52[:, :21 * 6]  # (T, 126)
    # Extract first 22 RIC joints (including root!)
    ric_22 = ric_52[:, :22 * 3]                 # (T, 66)
                                    ↑ 22 * 3 = 66

    # Concatenate: [transl(3), root_rot6d(6), body_rot6d(126), ric(66)] = 201
    return np.concatenate([transl, root_rot6d, body_rot6d_22, ric_22], axis=-1)
```

---

## Forward Kinematics: How World Positions Are Computed

### From `differentiable_fk.py` (lines 97-136)

```python
def motion135_to_fk(
    motion_denorm: Tensor,
    bone_offsets: Tensor,
    rotation_space: str = 'local',
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Parse 135-dim and run FK.
    
    Returns:
        world_positions: (*, 22, 3) world-space joint positions
        world_rotations: (*, 22, 3, 3) world-space rotation matrices
        translation: (*, 3) root translation
        local_rotmat: (*, 22, 3, 3) local rotation matrices
    """
    leading = motion_denorm.shape[:-1]

    # Parse 135-dim: [trans(3), rot6d(22*6=132)]
    translation = motion_denorm[..., :3]        # (*, 3)
    rot6d_flat = motion_denorm[..., 3:135]      # (*, 132)
    rot6d = rot6d_flat.reshape(*leading, 22, 6) # (*, 22, 6)

    if rotation_space == 'global':
        # Convert global to local if needed
        rot6d = global_to_local_rot6d_torch(rot6d)

    # Convert row-major rot6d to rotation matrices
    local_rotmat = rot6d_to_rotmat_row_major(rot6d)  # (*, 22, 3, 3)

    # Run FK
    world_pos, world_rot = differentiable_fk(
        local_rotmat, translation, bone_offsets
    )

    return world_pos, world_rot, translation, local_rotmat
```

### FK Implementation (lines 29-68)

```python
def differentiable_fk(
    local_rotmat: Tensor,
    translation: Tensor,
    bone_offsets: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Differentiable FK for SMPL-22 skeleton."""
    leading_shape = local_rotmat.shape[:-3]
    
    world_rot_list = [None] * 22
    world_pos_list = [None] * 22

    for j in range(22):
        parent = SMPL22_PARENTS[j]
        if parent < 0:  # Root (pelvis)
            world_rot_list[j] = local_rotmat[..., j, :, :]
            world_pos_list[j] = translation + bone_offsets[j]
        else:  # Non-root joints
            world_rot_list[j] = world_rot_list[parent] @ local_rotmat[..., j, :, :]
            offset_rotated = (
                world_rot_list[parent] @ bone_offsets[j].unsqueeze(-1)
            ).squeeze(-1)
            world_pos_list[j] = world_pos_list[parent] + offset_rotated

    world_pos = torch.stack(world_pos_list, dim=-2)  # (*, 22, 3)
    world_rot = torch.stack(world_rot_list, dim=-3)  # (*, 22, 3, 3)

    return world_pos, world_rot  # BOTH include all 22 joints!
```

---

## Scheme D Encoding Explained

### The Encoding

For each joint i (0 to 21):

```python
# Given: world_pos[i] in (T, 3) format [x, y, z]
# Also: pelvis_world = world_pos[0]

pos_encoded[i, 0] = world_pos[i, 0] - pelvis_world[0]  # X relative
pos_encoded[i, 1] = world_pos[i, 1]                    # Y absolute
pos_encoded[i, 2] = world_pos[i, 2] - pelvis_world[2]  # Z relative
```

### Why This Scheme?

From `compute_198dim.py` docstring:

```python
"""Scheme D: XZ relative to pelvis, Y absolute world.

Rationale:
- Horizontal (XZ) relative: More invariant to global locomotion,
  more compressible, more stable across sequences
- Vertical (Y) absolute: Preserves ground contact info
  (ankles have Y ≈ 0, important for footfall detection)
"""
```

---

## Dimension Breakdown Example

For a single frame:

```
motion_201: (201,) = concatenated vector

[0:3]       Translation (3 dims)
[3:135]     Rotation channels (132 dims)
            ├─ [3:9]      Root (joint 0) rot6d
            ├─ [9:15]     Joint 1 (L_Hip) rot6d
            ├─ [15:21]    Joint 2 (R_Hip) rot6d
            ├─ ...
            └─ [129:135]  Joint 21 (R_Wrist) rot6d
[135:201]   Position channels (66 dims)
            ├─ [135:138]  Joint 0 (Pelvis) pos [x_rel, y_abs, z_rel]
            ├─ [138:141]  Joint 1 (L_Hip) pos [x_rel, y_abs, z_rel]
            ├─ [141:144]  Joint 2 (R_Hip) pos [x_rel, y_abs, z_rel]
            ├─ ...
            └─ [198:201]  Joint 21 (R_Wrist) pos [x_rel, y_abs, z_rel]
```

**Explicit position channel mapping:**
```python
for joint_idx in range(22):
    pos_start = 135 + joint_idx * 3
    x_pos = motion_201[pos_start + 0]      # relative to pelvis
    y_pos = motion_201[pos_start + 1]      # absolute height
    z_pos = motion_201[pos_start + 2]      # relative to pelvis
```

---

## Joint Ordering (SMPL-22)

```python
# From fk_utils.py SMPL22_PARENTS:
JOINTS = {
    0:  "Pelvis",        1:  "L_Hip",       2:  "R_Hip",
    3:  "Spine1",        4:  "L_Knee",      5:  "R_Knee",
    6:  "Spine2",        7:  "L_Ankle",     8:  "R_Ankle",
    9:  "Spine3",        10: "L_Foot",      11: "R_Foot",
    12: "Neck",          13: "L_Collar",    14: "R_Collar",
    15: "Head",          16: "L_Shoulder",  17: "R_Shoulder",
    18: "L_Elbow",       19: "R_Elbow",     20: "L_Wrist",
    21: "R_Wrist",
}

# Position dims 135-201 follows this order:
# 135-137: Pelvis
# 138-140: L_Hip
# 141-143: R_Hip
# ...
# 198-200: R_Wrist
```

---

## Statistics Computation Example

From `compute_201dim_stats.py`:

```python
# Per sample:
motion_201 = np.concatenate([motion_135, pos_66], axis=-1)  # (T, 201)

# Across all samples (multiprocessing):
all_frames = np.concatenate(frames_list, axis=0).astype(np.float64)  # (N, 201)
n = all_frames.shape[0]
s = all_frames.sum(axis=0)         # (201,) sum
sq = (all_frames ** 2).sum(axis=0) # (201,) sum of squares

# Final statistics:
mean = s / n                           # (201,)
variance = (sq / n) - (s / n) ** 2     # (201,)
std = np.sqrt(variance)                # (201,)

# Saved as:
# data/hymotion_m2m_data/_stats_201dim/Mean.npy  shape (201,)
# data/hymotion_m2m_data/_stats_201dim/Std.npy   shape (201,)
```

---

## File Loading Pipeline

```python
# 1. User loads motion data
results = {'motion_path': 'path/to/motion.npz', 'fps': 30}

# 2. LoadSmplx55 transform
loader = LoadSmplx55(
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs',
    smpl_type='smpl_22',
)
results = loader.transform(results)
# results['motion'] is now (T, 135)

# 3. (Optional) Compute198DimPosition transform
pos_computer = Compute198DimPosition(key='motion')
results = pos_computer.transform(results)
# results['motion'] is now (T, 198) if enabled

# 4. Or load pre-computed
loader_o6dp = LoadO6dp(key='motion', joints_num=22)
results = {'motion_path': 'path/to/motion_201dim.npy'}
results = loader_o6dp.transform(results)
# results['motion'] is now (T, 201)
```

---

## Row-Major vs Column-Major Rotation

### The Conversion

From `load_smplx.py` lines 88-93:

```python
# rotation_convert.py outputs column-major
out = axis_angle_to_rotation_6d(aa_flat)  # [R00,R10,R20, R01,R11,R21]

# Convert to row-major for HyMotion
out = out[:, :, [0, 3, 1, 4, 2, 5]]  # [R00,R01, R10,R11, R20,R21]
```

### Layout

```
Row-major (HyMotion):    [R₀₀, R₀₁, R₁₀, R₁₁, R₂₀, R₂₁]
Column-major (standard): [R₀₀, R₁₀, R₂₀, R₀₁, R₁₁, R₂₁]
                          ↓    ↓    ↓    ↓    ↓    ↓
Reindex [0,3,1,4,2,5]: = [0    3    1    4    2    5]
```

This is handled automatically in the loading pipeline, so users don't need to worry about it.

---

## Summary Table

| Property | 201-Dim | 198-Dim | 135-Dim |
|----------|---------|---------|---------|
| Translation | 3 | 3 | 3 |
| Rotation | 132 | 132 | 132 |
| Positions | 66 | 63 | 0 |
| **Total** | **201** | **198** | **135** |
| Pelvis in pos | ✓ YES | ✗ NO | N/A |
| Pos encoding | Scheme D | Scheme D | N/A |
| Use case | o6dp_1103 | FK consistency | Rotation-only |

---

## Key Takeaways

1. **66 dims = 22 joints × 3D** (not 21×3 + something)
2. **Includes pelvis** (unlike 198-dim)
3. **Scheme D encoding:** XZ relative to pelvis, Y absolute
4. **Computed via FK** from local rotations
5. **Row-major 6D rotation** (non-standard convention)
6. **All 22 SMPL joints** in both rotation and position channels
