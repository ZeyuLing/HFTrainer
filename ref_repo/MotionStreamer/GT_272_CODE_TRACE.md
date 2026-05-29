# Code Trace: How GT 272-Dim Rotations Are Generated

## Complete Code Path from Input to 272-Dim Output

### Step 1: Input Files
```
Location: MotionStreamer/humanml3d_272/smpl_85_face_z_transform/
Files:
  - smpl_85_face_z_transform/*.npy       (T, 85) SMPL parameters
  - smpl_85_face_z_transform_joints/*.npy (T, 22, 3) joint positions from FK
```

### Step 2: Parse SMPL Parameters
**File:** `representation_272.py` lines 64-67
```python
rotation_smpl_axis_angle = np.load(file.replace('smpl_85_face_z_transform_joints', 
                                                  'smpl_85_face_z_transform'))
# Shape: (T, 85)
# Dims [0:3] = root orientation
# Dims [3:66] = body pose (21 joints × 3)
# Dims [66:72] = hand joints (ignored for SMPL22)
# Dims [72:75] = translation
# Dims [75:85] = shape parameters (betas)

# Extract rotation parameters for 22 SMPL joints
rotation_smpl_axis_angle_22j = rotation_smpl_axis_angle[:, :66]  # (T, 66)
```

### Step 3: Axis-Angle to Quaternion
**File:** `representation_272.py` line 65 + `face_z_align_util.py` lines 191-206
```python
from utils.face_z_align_util import expmap_to_quaternion

# Convert axis-angle to quaternion [w, x, y, z]
rotations_wxyz = expmap_to_quaternion(
    rotation_smpl_axis_angle[:, :66].reshape(nfrm, njoint, 3)
)
# Input shape: (T, 22, 3)     # T frames, 22 joints, 3-dim axis-angle
# Output shape: (T, 22, 4)    # T frames, 22 joints, 4-dim quaternion [w,x,y,z]
```

**Implementation in face_z_align_util.py (lines 191-206):**
```python
def expmap_to_quaternion(e):
    '''
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using 
    the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    '''
    assert e.shape[-1] == 3
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)
    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)           # magnitude
    w = np.cos(0.5 * theta).reshape(-1, 1)                      # scalar part
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e               # vector part
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)
```

### Step 4: Quaternion to Rotation Matrix
**File:** `representation_272.py` line 67 + `face_z_align_util.py` lines 274-276
```python
from utils.face_z_align_util import quaternion_to_matrix_np

rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)
# Input shape: (T, 22, 4)     # quaternions [w,x,y,z]
# Output shape: (T, 22, 3, 3) # rotation matrices
```

**Implementation in face_z_align_util.py (lines 274-276):**
```python
def quaternion_to_matrix_np(quaternions):
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()

def quaternion_to_matrix(quaternions):  # (lines 245-271)
    '''
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    '''
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
```

**Result:** rotation matrix in standard form:
```
[R00 R01 R02]
[R10 R11 R12]
[R20 R21 R22]
```

### Step 5: Extract First 2 ROWS (THE CRITICAL STEP)
**File:** `representation_272.py` line 116
```python
final_x[:,8+6*njoint:8+12*njoint] = np.reshape(
    rotations_matrix[..., :, :2, :],    # ← TAKE FIRST 2 ROWS
    (nfrm,-1)
)

# rotations_matrix[..., :2, :]
# └─ rotations_matrix has shape (T, 22, 3, 3)
#    └─ [..., :2, :] means "all frames, all joints, rows 0-1, all columns"
#       └─ Result shape: (T, 22, 2, 3)
#
# np.reshape(..., (nfrm,-1)) converts (T, 22, 2, 3) → (T, 132)
# because T * 22 * 2 * 3 = T * 132
```

**What this extracts:**
```
From each 3×3 rotation matrix:

[R00 R01 R02]      Extract       [R00 R01 R02]
[R10 R11 R12]  ───────────────>  [R10 R11 R12]
[R20 R21 R22]      first 2 rows

Then flatten as ROW-MAJOR:
6D = [R00, R01, R02, R10, R11, R12]
```

### Step 6: Assemble Full 272-Dim Vector
**File:** `representation_272.py` lines 102-116

```python
size_frame = 8 + njoint*3 + njoint*3 + njoint*6  # = 272
final_x = np.zeros((nfrm, size_frame))

# Dims 0-1: Root XZ velocity (heading removed)
final_x[1:,:2] = velocities_root_xy_no_heading   # (T-1, 2)

# Dims 2-7: Heading angular velocity (as 6D rotation)
final_x[1:,2:8] = matrix_to_rotation_6d(
    torch.from_numpy(global_heading_diff_rot)
).numpy()                                          # (T-1, 6)

# Dims 8-73: Joint positions (heading removed)
final_x[:,8:8+3*njoint] = np.reshape(
    positions_no_heading, 
    (nfrm,-1)
)                                                 # (T, 66)

# Dims 74-139: Joint velocities (heading removed)
final_x[1:,8+3*njoint:8+6*njoint] = np.reshape(
    velocities_no_heading, 
    (nfrm-1,-1)
)                                                 # (T-1, 66)

# Dims 140-271: Joint rotations (6D, ROW-MAJOR)   ← THE ANSWER
final_x[:,8+6*njoint:8+12*njoint] = np.reshape(
    rotations_matrix[..., :2, :],                 # First 2 rows
    (nfrm,-1)
)                                                 # (T, 132)

# Final shape: (T, 272)
np.save(output_file, final_x)
```

---

## The matrix_to_rotation_6d Function

This function is called for heading rotation (dims 2-7) and is the reference implementation for the convention:

**File:** `face_z_align_util.py` lines 961-974

```python
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    '''
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)
    Returns:
        6D rotation representation, of size (*, 6)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    '''
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
```

**What it does:**
1. Takes first 2 rows: `matrix[..., :2, :]` → `(..., 2, 3)`
2. Reshapes to 6D: `.reshape(..., 6)` → `(..., 6)`
3. Result: `[R00, R01, R02, R10, R11, R12]` (ROW-MAJOR)

**Why "dropping the last row"?** Because mathematically, in an orthogonal matrix, the third row can be recovered from the first two via cross product (Gram-Schmidt).

---

## Alternative: quaternion_to_cont6d_np (NOT USED)

This is available but NOT used for GT 272-dim:

**File:** `face_z_align_util.py` lines 279-282

```python
def quaternion_to_cont6d_np(quaternions):
    rotation_mat = quaternion_to_matrix_np(quaternions)
    cont_6d = np.concatenate([rotation_mat[..., 0], rotation_mat[..., 1]], axis=-1)
    return cont_6d
```

**What it does (COLUMN-MAJOR):**
1. Takes first 2 COLUMNS: `rotation_mat[..., 0]` and `rotation_mat[..., 1]`
2. Concatenates: `[col0, col1]` → `[R00, R10, R20, R01, R11, R21]`
3. Result: `[R00, R10, R20, R01, R11, R21]` (COLUMN-MAJOR)

**Key difference:**
```
matrix_to_rotation_6d():    [R00 R01 R02 R10 R11 R12]  ← ROW-MAJOR (USED)
quaternion_to_cont6d_np():  [R00 R10 R20 R01 R11 R21]  ← COLUMN-MAJOR (NOT used)
```

---

## How PRISM Predictions Are Converted

**File:** `convert_prism_to_272.py`

### PRISM Output Format:
```python
pred = np.load(pred_file)  # NPZ format
global_orient = pred['global_orient']  # (T, 3) axis-angle
body_pose = pred['body_pose']          # (T, 63) axis-angle (21 joints)
transl = pred['transl']                # (T, 3)
betas = pred['betas']                  # (10,)
```

### Conversion Steps:
```python
1. Apply face_z_transform()
   └─ Rotates first frame to face Z+

2. Run FK using SMPL (NOT SMPLX!)
   smpl_model = smplx.create(model_type='smpl', ...)
   output = smpl_model(global_orient=..., body_pose=..., 
                       transl=..., betas=...)
   joints = output.joints[:, :22, :]  # First 22 joints

3. Call compute_representation_272()
   └─ Uses exact same logic as representation_272.py
   └─ Calls matrix_to_rotation_6d() for rotations
```

**The key function in convert_prism_to_272.py (lines 449-451):**
```python
# Line 131: matrix_to_rotation_6d is called inline
rot6d = rotations_matrix[..., :2, :]  # (T, 22, 2, 3)
final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(
    rot6d, (nfrm, -1)
)  # (T, 132)
```

---

## Summary: The Complete Rotation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: SMPL Axis-Angle Rotations (from AMASS/HumanML3D)       │
│        Shape: (T, 66) — 22 joints × 3-dim axis-angle           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │ expmap_to_quaternion()                  │
        │ (face_z_align_util.py:191-206)         │
        │ axis_angle → quaternion [w,x,y,z]     │
        │ Output: (T, 22, 4)                     │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ quaternion_to_matrix_np()               │
        │ (face_z_align_util.py:274-276)         │
        │ quaternion → 3×3 rotation matrix       │
        │ Output: (T, 22, 3, 3)                  │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ matrix[..., :2, :]  [ROW-MAJOR]        │
        │ (representation_272.py:116)            │
        │ Extract first 2 rows                   │
        │ Output: (T, 22, 2, 3)                  │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ reshape(..., (T, 132))                 │
        │ Flatten to 6D per joint                │
        │ Output: (T, 132)                       │
        │                                        │
        │ 6D Format (per joint):                 │
        │ [R00 R01 R02 R10 R11 R12]  (ROW-MAJ)  │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │ Pack into 272-dim vector               │
        │ Dims 132-263: Joint rotations (6D)    │
        │ + Dims 0-131: Other data               │
        │                                        │
        │ OUTPUT: (T, 272) ✓                     │
        └────────────────────────────────────────┘
```

