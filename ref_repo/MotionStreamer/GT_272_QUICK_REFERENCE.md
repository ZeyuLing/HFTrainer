# Quick Reference: GT 272-Dim Rotation Pipeline

## TL;DR - Direct Answers to Your Questions

### Q1: Does it use SMPL rotations directly (from AMASS) or IK-derived rotations?
**A: SMPL rotations DIRECTLY from AMASS/HumanML3D**
- No IK involved
- Evidence: `representation_272.py` line 64-65 loads `smpl_85_face_z_transform` which contains raw SMPL axis-angles
- These are converted axis-angle → quaternion → rotation matrix, then extracted as 6D

### Q2: Does it use `matrix_to_rotation_6d` (row-major) or `quaternion_to_cont6d_np` (column-major)?
**A: `matrix_to_rotation_6d` (ROW-MAJOR)**
- Code: `representation_272.py` line 116 uses `rotations_matrix[..., :2, :]`
- This is ROW-MAJOR: takes first 2 ROWS → `[R00 R01 R02 R10 R11 R12]`
- `quaternion_to_cont6d_np()` exists in codebase but is NOT used for GT 272-dim

---

## File Locations

| File | Location | Purpose |
|------|----------|---------|
| **representation_272.py** | `272-dim-Motion-Representation/representation_272.py` | Main GT 272-dim generator |
| **convert_prism_to_272.py** | `./convert_prism_to_272.py` | PRISM → 272-dim converter |
| **face_z_align_util.py** | `272-dim-Motion-Representation/utils/face_z_align_util.py` | Contains `matrix_to_rotation_6d()` |
| **Output** | `humanml3d_272/motion_data/` | Generated .npy files |

---

## Rotation Extraction: The Critical Line

### representation_272.py, Line 116:
```python
final_x[:,8+6*njoint:8+12*njoint] = np.reshape(rotations_matrix[..., :, :2, :], (nfrm,-1))
```

**Breaking it down:**
```
rotations_matrix.shape = (T, 22, 3, 3)  # T frames, 22 joints, 3×3 rotation matrices
rotations_matrix[..., :2, :]            # Take first 2 rows → (T, 22, 2, 3)
reshape to (T, 22*6)                    # Flatten → (T, 132)
```

**These 132 dims (indices 132-263) are the joint rotations in 6D ROW-MAJOR format:**
```
Joint 0: [R00 R01 R02 | R10 R11 R12] (dims 132-137)
Joint 1: [R00 R01 R02 | R10 R11 R12] (dims 138-143)
...
Joint 21: [R00 R01 R02 | R10 R11 R12] (dims 258-263)
```

---

## Process Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│              AMASS/HumanML3D Motion Data                        │
│         (SMPL axis-angle rotations + translations)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  representation_272.py Processing:   │
        │                                      │
        │  1. Load SMPL axis-angles (66 dims) │
        │  2. axis_angle → quaternion [w,x,y,z]
        │  3. quaternion → rotation matrix    │
        │                                      │
        │     rotation_matrix.shape = (T,22,3,3)
        │                                      │
        │  4. Extract first 2 ROWS (ROW-MAJOR)
        │     rotations_matrix[..., :2, :]   │
        │     → (T, 22, 2, 3)                │
        │                                      │
        │  5. Reshape to (T, 22*6) = (T, 132)
        │                                      │
        │  6. Concatenate with:               │
        │     - Root XZ velocity (2 dims)     │
        │     - Heading angular vel (6 dims)  │
        │     - Joint positions (66 dims)     │
        │     - Joint velocities (66 dims)    │
        │                                      │
        └──────────────────┬───────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Output: 272-Dim Representation      │
        │                                      │
        │  [2 + 6 + 66 + 66 + 132 = 272]     │
        │                                      │
        │  Dims 132-263: 6D Rotations (ROW)   │
        └──────────────────────────────────────┘
```

---

## Comparing Two Rotation Extraction Methods

### Method 1: matrix_to_rotation_6d() [USED FOR GT]
**Location:** `face_z_align_util.py` line 961-974
```python
def matrix_to_rotation_6d(matrix):  # (*, 3, 3)
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
```

**Convention:** ROW-MAJOR (first 2 rows)
```
Matrix:              6D output:
[R00 R01 R02]        [R00, R01, R02, R10, R11, R12]
[R10 R11 R12]
[R20 R21 R22]
```

**Used in:**
- `representation_272.py` line 109 (heading angular velocity)
- `convert_prism_to_272.py` (entire rotation pipeline)

---

### Method 2: quaternion_to_cont6d_np() [NOT USED FOR GT]
**Location:** `face_z_align_util.py` line 279-282
```python
def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d
```

**Convention:** COLUMN-MAJOR (first 2 columns)
```
Matrix:              6D output:
[R00 R01 R02]        [R00, R10, R20, R01, R11, R21]
[R10 R11 R12]        (col 0)        (col 1)
[R20 R21 R22]
```

**Status:** Available but NOT used for GT 272-dim format

---

## Important: SMPL vs SMPLX

The GT 272-dim format uses **SMPL** (22 joints), NOT SMPLX.

From `convert_prism_to_272.py` lines 9-14:
```
IMPORTANT: The GT 272-dim data was computed with SMPL body model.
Using SMPLX produces physically impossible skeletons due to different body templates.
PRISM outputs 63-dim body_pose (21 SMPLX body joints), so we pad to 69-dim
(23 SMPL body joints) with zeros for the last 2 hand joints.
```

### SMPL Structure:
```
22 joints = [
    0: Pelvis (root),
    1-2: Hip (L/R),
    3-4: Knee (L/R),
    5-6: Ankle (L/R),
    7: Neck,
    8-9: Shoulder (L/R),
    10-11: Elbow (L/R),
    12-13: Wrist (L/R),
    14-20: Toes/extras
]
```

Each joint has 3 axis-angle components → 22 × 3 = 66 dims of rotation parameters

---

## Face-Z Transform: Why "face Z+"?

From `face_z_align_util.py::face_z_transform()`:

1. Computes forward direction from hips and shoulders
2. Rotates to align with Z+ axis (character faces +Z)
3. Applied to first frame reference
4. This rotation is applied to all root orientations and translations

**Result:** All GT 272-dim sequences are normalized so that the character's initial facing direction is along Z+.

---

## Coordinate System: Local Frame with Global Heading Removed

The 272-dim representation uses:
- **XZ plane:** Locked to character's initial Z+ facing (heading removed)
- **Y axis:** Global (vertical)
- **Root position:** XZ-centered at origin for first frame
- **Floor height:** Normalized (y_min = 0)

This means:
- Position dims (8-65) are in a heading-removed frame
- Velocity dims (66-131) are in a heading-removed frame
- Rotation dims (132-263) have heading removed from root (joint 0) only
- Root XZ velocity (0-1) is in heading-removed frame
- Heading angular velocity (2-7) captures frame's rotation speed

---

## Summary: What Goes Into Each 6D Rotation

For each joint j at each frame t:

**Input:** SMPL axis-angle `aa[t,j]` ∈ ℝ³ (from AMASS)

**Processing:**
```
1. axis_angle → quaternion:     q = exp_map(aa)
2. quaternion → 3×3 matrix:     R = quat_to_mat(q)
3. Extract first 2 rows:        [r0, r1] = R[:2, :]
4. Flatten:                      6D = [r0[0], r0[1], r0[2], r1[0], r1[1], r1[2]]
```

**Output:** 6D ∈ ℝ⁶ in ROW-MAJOR format

This 6D representation can be reconstructed to a full 3×3 rotation matrix using Gram-Schmidt orthogonalization (Zhou et al. 2019).

