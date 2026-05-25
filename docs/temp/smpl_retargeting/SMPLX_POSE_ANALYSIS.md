# SMPL-X Pose Processing Analysis

## Overview

The `process_smplx_pose` function in `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` converts full SMPL-X 55-joint axis-angle rotations to a target joint subset and rotation representation format.

---

## 1. Joint Selection: "SMPL_22" from Full SMPL-X

### Joint Indices

The function uses simple linear indexing to select joints:

```python
IDX_SMPL22 = np.arange(22, dtype=np.int64)  # [0, 1, 2, ..., 21]
IDX_SMPLH = np.arange(52, dtype=np.int64)   # [0, 1, 2, ..., 51]
IDX_SMPLX55 = np.arange(55, dtype=np.int64) # [0, 1, 2, ..., 54]
```

### SMPL-22 Includes:
- **Joints 0-21**: The core SMPL body joints (22 joints total)
  - Joint 0: Pelvis (root)
  - Joints 1-4: Left leg (hip, knee, ankle, foot)
  - Joints 5-8: Right leg (hip, knee, ankle, foot)
  - Joints 9-14: Spine chain (belly, chest, neck, head)
  - Joints 15-20: Left arm (shoulder, elbow, wrist, hand, thumb, pinky)
  - Joints 21: Right shoulder

**Excludes**: Joints 22-54 (jaw, eyes, and fine hand articulation from SMPL-X)

### Selection Code (Line 69-80):

```python
if out_type == "smpl_22":
    sel = IDX_SMPL22          # Select first 22 joints
elif out_type == "smplh":
    sel = IDX_SMPLH           # Select first 52 joints
else:
    sel = IDX_SMPLX55         # Select all 55 joints

aa = aa[:, sel, :]  # [T, J, 3] - applies the mask
```

The selection is done via simple NumPy indexing on the reshaped axis-angle array.

---

## 2. Rotation 6D Conversion Pipeline

### 2.1 Overview Path

```
axis_angle (3D) 
    ↓
axis_angle_to_matrix() → rotation matrix (3×3)
    ↓
matrix_to_rotation_6d() → 6D representation (6D)
    ↓
Rearrange columns (normalize for HyMotion convention)
    ↓
rotation_6d (6D)
```

### 2.2 Conversion Functions

#### **Step 1: Axis-Angle → Rotation Matrix**

**Function**: `axis_angle_to_matrix()` (line 127-155 in rotation_convert.py)

For **NumPy** (used in this pipeline):
```python
def axis_angle_to_matrix(axis_angle):
    # Uses scipy.spatial.transform.Rotation
    return R.from_rotvec(axis_angle).as_matrix()
```

Input: `axis_angle` shape `(..., 3)` - rotation vector = axis × angle (radians)
Output: Rotation matrix shape `(..., 3, 3)`

#### **Step 2: Rotation Matrix → 6D Representation**

**Function**: `matrix_to_rotation_6d()` (line 455-460 in rotation_convert.py)

```python
def matrix_to_rotation_6d(matrix):
    M = _reshape_matrix9(matrix)
    if _is_numpy(M):
        return _stack_cols01_np(M)  # Stack first two columns
    else:
        return _stack_cols01_torch(M)

def _stack_cols01_np(Rm):
    # Concatenate first two columns of rotation matrix
    return np.concatenate([Rm[..., 0:3, 0], Rm[..., 0:3, 1]], axis=-1)
```

Input: Rotation matrix shape `(..., 3, 3)` or `(..., 9)`
Output: 6D vector shape `(..., 6)` 

**What it extracts:**
```
Rotation Matrix:      6D Representation:
[R00 R01 R02]    →    [R00, R10, R20,  R01, R11, R21]
[R10 R11 R12]         ↑ Column 0         ↑ Column 1
[R20 R21 R22]
```

This is **column-major ordering**: `[R00, R10, R20, R01, R11, R21]`

---

## 3. Exact Joint Ordering & Output Shape

### 3.1 Joint Ordering in Output

**For SMPL-22:**
- Output shape: `[T, 22 × D]` where D depends on rotation format
- Joints ordered as: `[0, 1, 2, ..., 21]` (simple linear order)
- Flattened format: `[J0_rot, J1_rot, J2_rot, ..., J21_rot]`

For **rotation_6d**: 
- Shape: `[T, 22 × 6] = [T, 132]`
- Layout: Each joint gets 6 dimensions in sequence

### 3.2 Dimension Ordering Per Joint

**Before HyMotion fix (Column-major from `matrix_to_rotation_6d`):**
```
6D = [R00, R10, R20, R01, R11, R21]
```

**After HyMotion rearrangement (Line 93, Row-major):**
```python
out = out[:, :, [0, 3, 1, 4, 2, 5]]
```

This applies the permutation: `[0, 3, 1, 4, 2, 5]`

```
Column-major:  [R00, R10, R20, R01, R11, R21]
                 0    1    2    3    4    5
                            ↓ permute [0,3,1,4,2,5]
Row-major:     [R00, R01, R10, R11, R20, R21]
                 0    1    2    3    4    5
```

**Result: Grouped as matrix rows**
```
[R00, R01,  R10, R11,  R20, R21]
```

This matches HyMotion's convention for storing rotation matrix columns in row-major order.

---

## 4. Output Shape and Dimension Ordering

### Complete Flow

```python
def process_smplx_pose(
    pose_55_axis_angle: np.ndarray,  # [T, 165] or [T, 55, 3]
    rot_type: str,                   # "axis_angle" | "rotation_6d" | "quaternion" | "euler"
    out_type: str,                   # "smpl_22" | "smplh" | "smplx_55"
) -> np.ndarray:
```

### Dimension Mapping

| `rot_type` | D per joint | Formula | Example (SMPL-22) |
|---|---|---|---|
| `"axis_angle"` | 3 | `[T, J*3]` | `[T, 66]` |
| `"rotation_6d"` | 6 | `[T, J*6]` | `[T, 132]` |
| `"quaternion"` | 4 | `[T, J*4]` | `[T, 88]` |
| `"euler"` | 3 | `[T, J*3]` | `[T, 66]` |

### Joint Count (`J`)

| `out_type` | J | Notes |
|---|---|---|
| `"smpl_22"` | 22 | Core SMPL body joints |
| `"smplh"` | 52 | SMPL + hand articulation |
| `"smplx_55"` | 55 | Full SMPL-X with jaw/eyes |

### Final Output

```python
return out.reshape(T, J * D).astype(np.float32)  # [T, J*D]
```

**Example output for SMPL-22 + rotation_6d:**
- Shape: `[T, 132]`
- Layout: `[J0_rot6d[0-5], J1_rot6d[0-5], ..., J21_rot6d[0-5]]`
- Each joint occupies 6 consecutive dimensions
- Within each joint, 6D is in **row-major order** after rearrangement

---

## 5. Complete Processing Pipeline Code

```python
def process_smplx_pose(
    pose_55_axis_angle: np.ndarray,  # [T, 165] or [T, 55, 3]
    rot_type: str,
    out_type: str,
) -> np.ndarray:
    """
    Convert SMPL-X 55-joint axis-angle pose to target joint set & rotation representation.
    """
    assert out_type in ["smpl_22", "smplh", "smplx_55"]
    assert rot_type in ["axis_angle", "rotation_6d", "quaternion", "euler"]

    # Step 1: Normalize to [T, 55, 3]
    if pose_55_axis_angle.ndim == 2 and pose_55_axis_angle.shape[1] == 55 * 3:
        T = pose_55_axis_angle.shape[0]
        aa = pose_55_axis_angle.reshape(T, 55, 3)
    # ... (handle SMPL-H padding, etc.)

    # Step 2: Select joint subset
    IDX_SMPL22 = np.arange(22, dtype=np.int64)   # [0..21]
    IDX_SMPLH = np.arange(52, dtype=np.int64)    # [0..51]
    IDX_SMPLX55 = np.arange(55, dtype=np.int64)  # [0..54]

    if out_type == "smpl_22":
        sel = IDX_SMPL22
    elif out_type == "smplh":
        sel = IDX_SMPLH
    else:
        sel = IDX_SMPLX55

    aa = aa[:, sel, :]  # [T, J, 3]
    T, J, _ = aa.shape
    aa_flat = aa.reshape(T * J, 3)  # [T*J, 3]

    # Step 3: Convert rotation representation
    if rot_type == "axis_angle":
        out = aa  # [T,J,3]
        D = 3
    elif rot_type == "rotation_6d":
        # axis_angle_to_rotation_6d outputs column-major: [R00,R10,R20, R01,R11,R21]
        out = axis_angle_to_rotation_6d(aa_flat).reshape(T, J, 6)
        # HyMotion convention is row-major: [R00,R01, R10,R11, R20,R21]
        # Rearrange: col_major[0,3,1,4,2,5] -> row_major
        out = out[:, :, [0, 3, 1, 4, 2, 5]]
        D = 6
    elif rot_type == "quaternion":
        out = axis_angle_to_quaternion(aa_flat).reshape(T, J, 4)
        D = 4
    elif rot_type == "euler":
        out = axis_angle_to_euler(aa_flat).reshape(T, J, 3)
        D = 3

    # Step 4: Flatten and return
    return out.reshape(T, J * D).astype(np.float32)  # [T, J*D]
```

---

## 6. Rotation 6D Reconstruction

To convert 6D back to rotation matrix:

```python
def rotation_6d_to_matrix(d6):
    """
    d6: [..., 6] - first two columns of rotation matrix
    Returns: [..., 3, 3] orthonormal rotation matrix
    """
    x_raw = d6[..., 0:3]    # First column
    y_raw = d6[..., 3:6]    # Second column
    
    # Orthonormalize via Gram-Schmidt
    x = normalize(x_raw)                 # Normalize column 0
    z = cross(x, y_raw)                  # Get orthogonal column (cross product)
    z = normalize(z)                     # Normalize
    y = cross(z, x)                      # Recover column 1
    
    return stack([x, y, z], axis=-1)     # [3, 3]
```

**Key insight**: Only 2 columns + normalization + cross products = full matrix recovery

---

## 7. Key Implementation Details

### Input Shape Handling
The function is flexible with input shapes:
- `[T, 165]` → reshaped to `[T, 55, 3]`
- `[T, 55, 3]` → used directly
- `[T, 52, 3]` → padded to `[T, 55, 3]` (SMPL-H → SMPL-X)

### SMPL-H Compatibility
If input is SMPL-H (52 joints), the code pads with zeros:
```python
# Insert 3 zero joints (jaw + 2 eyes) after neck (joint 22)
aa = np.concatenate([
    pose_55_axis_angle[:, :22],      # First 22 joints
    np.zeros((T, 3, 3)),              # 3 padding joints
    pose_55_axis_angle[:, 22:],       # Remaining 30 joints
], axis=1)
```

### Output Data Type
Always converts to `float32` for consistency:
```python
return out.reshape(T, J * D).astype(np.float32)
```

---

## 8. Example Usage

```python
# Load SMPL-X pose from file
poses = np.load("motion.npz")["poses"]  # [100, 165] - 100 frames, 55 joints × 3

# Convert to SMPL-22 with 6D rotation
result = process_smplx_pose(poses, rot_type="rotation_6d", out_type="smpl_22")
# Shape: [100, 132] - 100 frames, 22 joints × 6 dimensions

# Convert to SMPL-H with quaternions
result = process_smplx_pose(poses, rot_type="quaternion", out_type="smplh")
# Shape: [100, 208] - 100 frames, 52 joints × 4 dimensions
```

---

## 9. Rotation Conversion Functions Reference

All rotation conversion functions are in:
`hftrainer/models/motion/components/utils/geometry/rotation_convert.py`

### Core Functions Used

| Function | Input | Output | Purpose |
|---|---|---|---|
| `axis_angle_to_matrix()` | `(..., 3)` | `(..., 3, 3)` | Rodrigues formula |
| `axis_angle_to_quaternion()` | `(..., 3)` | `(..., 4)` | w, x, y, z format |
| `axis_angle_to_euler()` | `(..., 3)` | `(..., 3)` | XYZ default order |
| `axis_angle_to_rotation_6d()` | `(..., 3)` | `(..., 6)` | Zhou et al. 2019 |
| `matrix_to_rotation_6d()` | `(..., 3, 3)` | `(..., 6)` | Extract first 2 columns |
| `rotation_6d_to_matrix()` | `(..., 6)` | `(..., 3, 3)` | Gram-Schmidt + cross |

### Supported Input Types
- NumPy arrays (uses SciPy for fast path)
- PyTorch tensors (native implementation)

---

## Summary

1. **Joint Selection**: SMPL-22 simply takes the first 22 joints from SMPL-X 55 via linear indexing
2. **6D Conversion**: axis_angle → matrix → extract first 2 columns → rearrange to row-major
3. **Joint Ordering**: Linear order `[0, 1, ..., J-1]`, flattened with rotation dims
4. **Output Shape**: `[T, J × D]` where J ∈ {22, 52, 55} and D ∈ {3, 4, 6}
5. **Key Detail**: HyMotion rearranges 6D from column-major to row-major via `[0, 3, 1, 4, 2, 5]` permutation
