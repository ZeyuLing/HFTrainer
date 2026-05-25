# 201-Dim Motion Format Analysis

## Overview

The **201-dim motion format** in this codebase is a specific representation used throughout the HyMotion project. It combines rotation information with forward-kinematics-derived joint positions.

### Layout (201 dimensions)

```
[0:3]      (3)   Translation (SMPL absolute world translation)
[3:135]    (132) Rotation: 22 joints × 6D rot6d
           └─ [3:9]       (6)   Root rotation (rot6d)
           └─ [9:135]     (126) Body 21 non-root joints × 6D rot6d
[135:201]  (66)  Joint Positions: 22 joints × 3D positions (RIC-like)
```

---

## Component 1: Translation (Dims 0-3)

**What:** SMPL absolute world translation of the root/pelvis.

**Type:** Absolute world coordinates (X, Y, Z)

**Source:** Loaded directly from SMPL `trans` field in the NPZ file.

**Usage:** Represents the global locomotion/position of the character in world space.

---

## Component 2: Rotation (Dims 3-135)

### Encoding Details

**Type:** **Row-major 6D rotation vectors** (6 values per joint)

**Format Convention:**
- Row-major: `[R₀₀, R₀₁, R₁₀, R₁₁, R₂₀, R₂₁]` (first two rows of rotation matrix)
- This is **different** from the column-major convention used in some other libraries
- Conversion happens via `load_smplx.py` lines 90-93:
  ```python
  # Column-major from rotation_convert -> row-major for HyMotion
  out = out[:, :, [0, 3, 1, 4, 2, 5]]
  ```

### Joint Breakdown

- **Joint 0 (Pelvis/Root):** Dims 3-9
- **Joints 1-21 (Body joints):** Dims 9-135
  - L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2, L_Ankle, R_Ankle, Spine3, L_Foot, R_Foot, Neck, L_Collar, R_Collar, Head, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow, L_Wrist, R_Wrist

### Rotation Space Options

The rotation can be in two spaces (selectable in LoadSmplx55):

1. **Local rotation** (parent-relative) - default in training
   - Each joint's rotation relative to its parent in the kinematic tree
   - Used for Forward Kinematics (FK) to compute world positions
   
2. **Global rotation** (world-relative) - optional
   - Each joint's rotation in world frame
   - Can be converted from local via kinematic chain accumulation
   - Conversion happens via `local_to_global_rot6d_torch()` in `fk_utils.py`

---

## Component 3: Joint Positions (Dims 135-201)

### The Key Question: What Are These 66 Dimensions?

**Answer:** **22 joints × 3D positions in "Scheme D" encoding**

### Scheme D Encoding

This is **NOT** simple Cartesian coordinates. It's a mixed representation:

```python
# From compute_198dim.py, lines 65-69:
pelvis_world = world_pos[..., 0:1, :]  # (T, 1, 3)
joint_pos_D = world_pos.clone()
joint_pos_D[..., 0] -= pelvis_world[..., 0]  # X: relative to pelvis
joint_pos_D[..., 2] -= pelvis_world[..., 2]  # Z: relative to pelvis
# Y: keep absolute world height
```

**Per Joint (3 dims: X, Y, Z):**
- **X:** Relative to pelvis in the XZ plane (pelvis_X subtracted)
- **Y:** Absolute world height (Y coordinate kept as-is)
- **Z:** Relative to pelvis in the XZ plane (pelvis_Z subtracted)

### Why This Encoding?

1. **Horizontal stability:** XZ coordinates relative to pelvis are more invariant to global locomotion
2. **Vertical grounding:** Absolute Y preserves ground contact information (ankles have small Y values at ground level)
3. **Space efficiency:** More compressible than raw world positions since XZ vary less across the sequence

### Dimensions Breakdown

- **22 joints total** (all joints, including pelvis)
- **66 = 22 × 3** dimensions
- **NOT excluded:** Pelvis IS included (see line 128 of compute_201dim_stats.py: `.reshape(-1, 66)`)

This differs from the 198-dim format which excludes pelvis (21 × 3 = 63 dims).

---

## Comparison: 201-Dim vs 198-Dim vs 135-Dim

### 135-Dim Format (Base)
```
[0:3]      (3)   Translation
[3:135]    (132) 22 joints × 6D rot6d
```
**Pure rotation + translation. NO position info.**

### 198-Dim Format
```
[0:3]      (3)   Translation
[3:135]    (132) 22 joints × 6D rot6d
[135:198]  (63)  21 joints × 3D (pelvis excluded) in Scheme D
```
From `compute_198dim.py`: Line 66 skips pelvis: `joint_pos = world_pos[..., 1:, :].clone()`

### 201-Dim Format (o6dp_1103)
```
[0:3]      (3)   Translation
[3:135]    (132) 22 joints × 6D rot6d  (technically [3:9] root + [9:135] body)
[135:201]  (66)  22 joints × 3D (ALL joints, including pelvis) in Scheme D
```
From `compute_201dim_stats.py` line 128: ALL 22 joints included

**Key difference:** 201-dim includes pelvis position; 198-dim excludes it.

---

## Data Flow: How 201-Dim Is Constructed

### Path 1: From Raw SMPL (Most Common)

```
NPZ File (SMPL parameters)
  ├─ trans: (T, 3) - absolute world translation
  ├─ poses: (T, 165) - 55 joints × 3 axis-angle
  └─ mocap_framerate: int
         ↓
LoadSmplx55 (load_smplx.py)
  ├─ Extract 22 joints from 55 (body-only)
  ├─ Convert axis-angle → row-major 6D rot6d
  └─ Output: motion_135 = [trans(3) + rot6d_22j(132)] = (T, 135)
         ↓
Compute201DimPosition (or Compute198DimPosition)
  ├─ Run Forward Kinematics on motion_135
  │  (uses LOCAL rotation to compute world positions)
  ├─ Apply Scheme D encoding
  ├─ Include all 22 joints (for 201-dim)
  └─ Output: motion_201 = [motion_135 + pos_66] = (T, 201)
```

### Path 2: From Pre-computed o6dp_1103 (Alternative)

```
Pre-computed NPZ (o6dp_1103 format)
  └─ motion_135: (T, 135) - already in target format
         ↓
LoadSmplx55._load_precomputed_135()
  └─ Fast path: use motion_135 directly (no augmentation)
```

---

## Forward Kinematics (FK) Details

### How Positions Are Computed

**Function:** `motion135_to_fk()` in `differentiable_fk.py`

1. Parse 135-dim:
   ```
   translation = motion[0:3]
   rot6d = motion[3:135].reshape(22, 6)
   ```

2. Convert rot6d → rotation matrices (row-major convention)

3. Run FK through SMPL-22 kinematic tree:
   ```python
   for j in range(22):
       parent = SMPL22_PARENTS[j]
       if parent < 0:  # Root
           world_rot[j] = local_rot[j]
           world_pos[j] = translation + bone_offsets[j]
       else:
           world_rot[j] = world_rot[parent] @ local_rot[j]
           world_pos[j] = world_pos[parent] + world_rot[parent] @ bone_offsets[j]
   ```

4. Result: World positions (T, 22, 3)

### Kinematic Tree (SMPL-22)

```
Pelvis (0)
├─ L_Hip (1) → L_Knee (4) → L_Ankle (7) → L_Foot (10)
├─ R_Hip (2) → R_Knee (5) → R_Ankle (8) → R_Foot (11)
├─ Spine1 (3) → Spine2 (6) → Spine3 (9)
│  ├─ Neck (12) → Head (15)
│  ├─ L_Collar (13) → L_Shoulder (16) → L_Elbow (18) → L_Wrist (20)
│  └─ R_Collar (14) → R_Shoulder (17) → R_Elbow (19) → R_Wrist (21)
```

---

## Key Files and Functions

### Dataset Loading & Construction

| File | Key Function | Purpose |
|------|--------------|---------|
| `load_smplx.py` | `LoadSmplx55.transform()` | Load SMPL NPZ, convert to rotation, compute motion_135 |
| `compute_198dim.py` | `Compute198DimPosition.transform()` | 135 → 198 (add positions, exclude pelvis) |
| `load_o6dp.py` | `LoadO6dp.transform()` | Load pre-computed o6dp_1103, optionally extract 22j |
| `fk_utils.py` | `local_to_global_rot6d_torch()` | Local ↔ Global rotation conversion |

### Statistics & Validation

| File | Purpose |
|------|---------|
| `compute_201dim_stats.py` | Compute Mean/Std for 201-dim format |
| `compute_198dim_stats.py` | Compute Mean/Std for 198-dim format |

### Core Kinematics

| File | Key Function | Purpose |
|------|--------------|---------|
| `differentiable_fk.py` | `motion135_to_fk()` | Parse 135-dim and run FK → world positions |
| `differentiable_fk.py` | `fk_to_motion135()` | Inverse: rotation matrices + translation → 135-dim |
| `fk_utils.py` | `SMPL22_PARENTS` | Kinematic tree structure |

---

## Rotation Conventions (Important!)

### Row-Major vs Column-Major

This codebase uses **row-major 6D rotation**, which is different from many libraries.

```
Row-major (HyMotion):     [R₀₀, R₀₁, R₁₀, R₁₁, R₂₀, R₂₁]
Column-major (standard):  [R₀₀, R₁₀, R₂₀, R₀₁, R₁₁, R₂₁]
```

**Conversion (from `load_smplx.py` lines 90-93):**
```python
# Column-major from rotation_convert → row-major
out = out[:, :, [0, 3, 1, 4, 2, 5]]
```

**Why?** Geometry.py (the network's motion encoder/decoder) expects row-major natively.

---

## Summary: The 66 Dimensions

### Direct Answer

The **66 dimensions** in the 201-dim format are:

✅ **22 joints × 3D = 66 dimensions (including pelvis)**

### NOT:
- ❌ 21 joints × 3D + something else
- ❌ RIC (root-invariant coordinates) in the narrow sense
- ❌ Excluding pelvis

### Encoding Details

Each 3D position per joint is encoded as:
```
pos[j] = [
    world_pos[j].x - pelvis_world.x,    # X: relative to pelvis
    world_pos[j].y,                      # Y: absolute world height
    world_pos[j].z - pelvis_world.z      # Z: relative to pelvis
]
```

### Derivation

Computed via **Forward Kinematics from local rotations**, then applying Scheme D encoding:
1. Parse 135-dim (trans + rot6d)
2. Run FK on LOCAL rotations → world positions
3. Subtract pelvis position from X and Z
4. Keep Y absolute
5. Flatten to 66 dims

### Purpose

This mixed representation balances:
- **Locality:** XZ coordinates relative to pelvis are invariant to global locomotion
- **Vertical grounding:** Absolute Y preserves contact point information
- **Differentiability:** Used in FK consistency loss during training

---

## Related Statistics Files

When computing stats (Mean.npy, Std.npy):

```python
# compute_201dim_stats.py workflow:

# 1. Load motion via LoadSmplx55 → motion_135
motion_135_local = results['motion']

# 2. Run FK to get world positions
with torch.no_grad():
    world_pos, _, _, _ = motion135_to_fk(motion_135_local, bone_offsets)

# 3. Apply Scheme D encoding
pelvis_world = world_pos[:, 0:1, :]
joint_pos_D = world_pos.clone()
joint_pos_D[..., 0] -= pelvis_world[..., 0]  # X relative
joint_pos_D[..., 2] -= pelvis_world[..., 2]  # Z relative
# Y: absolute

# 4. Flatten to 66 dims
pos_66 = joint_pos_D.reshape(-1, 66)

# 5. Concatenate: [motion_135, pos_66] = 201 dims
motion_201 = np.concatenate([motion_135, pos_66], axis=-1)

# 6. Compute mean/std across all data
```

The resulting Mean.npy and Std.npy files have shape **(201,)** each.

