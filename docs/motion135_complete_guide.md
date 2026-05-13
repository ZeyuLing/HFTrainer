# Complete Guide: Converting motion_135 to SMPL Joint Positions

This document provides a comprehensive guide to computing SMPL joint positions from the motion_135 representation in the HyMotion M2M motion completion model.

## Quick Reference

**Input Format**: `motion_135` (T, 135) = 3D translation + 22 SMPL joints in 6D rotation  
**Output Format**: Joint positions (T, 22, 3) in world coordinates  
**Two Implementation Paths**:
1. **Standalone script**: `scripts/embodied/motion135_to_smplx.py` → converts to SMPL-X NPZ for GMR retargeting
2. **Programmatic API**: `SmplxLite.fk()` in `hftrainer/models/motion/components/body_models/smplx_lite.py`

---

## Pipeline Overview

### Step 1: Extract Components from motion_135
```python
motion_135: (T, 135)
transl = motion_135[:, :3]              # (T, 3) - absolute translation
rot6d = motion_135[:, 3:].reshape(T, 22, 6)  # (T, 22, 6) - 22 joints, row-major
```

### Step 2: Convert rot6d to Rotation Matrices (Critical: Row-Major to Column-Major)
```
The rot6d uses row-major layout [R00,R01, R10,R11, R20,R21] in motion_135
BUT Gram-Schmidt expects column-major [R00,R10,R20, R01,R11,R21]
→ REORDER: [0,2,4,1,3,5] BEFORE Gram-Schmidt
```

**Gram-Schmidt Orthogonalization**:
- Normalize first column: `b1 = a1 / ||a1||`
- Orthogonalize second: `b2 = (a2 - <b1,a2>*b1) / ||...||`
- Cross product: `b3 = b1 × b2`
- Result: `rotmat = [b1, b2, b3]` (orthonormal 3×3 matrix)

### Step 3: Convert Rotation Matrix to Axis-Angle
```python
# Using scipy for robustness (handles near-zero and near-π cases)
from scipy.spatial.transform import Rotation as R
rot = R.from_matrix(rotmat_flat)
axis_angle = rot.as_rotvec()  # (T, 22, 3)
```

### Step 4: Split Root and Body Rotations
```python
root_orient = axis_angle[:, 0, :]          # (T, 3) - pelvis (joint 0)
pose_body = axis_angle[:, 1:22, :].reshape(T, 63)  # (T, 63) - 21 body joints
```

### Step 5: Run Forward Kinematics to Get Joint Positions
```python
# Using SMPL-X body model with skeleton + kinematics
smplx_model = SmplxLite("checkpoints/smpl_models/smplx/")
joints = smplx_model.fk(
    transl=transl,          # (T, 3)
    global_orient=root_orient,  # (T, 3)
    body_pose=pose_body,    # (T, 63)
    betas=None              # Use default zeros
)  # Returns (T, 22, 3)
```

---

## Key Technical Details

### ⚠️ CRITICAL: Rot6d Convention Mismatch

This is the #1 cause of errors. The motion_135 representation uses a **row-major** convention:
```
Layout: [R00, R01, R10, R11, R20, R21]
```

But the Gram-Schmidt orthogonalization process expects **column-major**:
```
Layout: [R00, R10, R20, R01, R11, R21]
```

**The Fix**: Before Gram-Schmidt, reorder by indices `[0, 2, 4, 1, 3, 5]`:
```python
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]  # ← Apply THIS before Gram-Schmidt
a1 = rot6d[..., :3]
a2 = rot6d[..., 3:6]
# ... continue with Gram-Schmidt
```

**Verification**: After Gram-Schmidt, rotation matrix should satisfy:
- `||col_i|| ≈ 1.0` (normalized columns)
- Orthogonal: `col_i · col_j ≈ 0` for i ≠ j
- Determinant ≈ 1.0

**Incorrect result** if reordering is skipped:
```
Matrix norm: > 2.0 (should be ~1.73 for orthonormal)
Output range: [-10, 13] (should be [-1, 1] after normalization)
```

### SMPL-22 Joint Structure

```
Joint Index | Name          | Parent
------------|---------------|--------
0           | Pelvis        | -1 (root)
1-2         | Hip (L/R)     | 0
3           | Spine1        | 0
4-5         | Knee (L/R)    | 1-2
6           | Spine2        | 3
7-8         | Ankle (L/R)   | 4-5
9           | Spine3        | 6
10-11       | Foot (L/R)    | 7-8
12          | Neck          | 9
13-14       | Collar (L/R)  | 9
15          | Head          | 12
16-17       | Shoulder (L/R)| 13-14
18-19       | Elbow (L/R)   | 16-17
20-21       | Wrist (L/R)   | 18-19
```

---

## Standalone Conversion Script

**File**: `scripts/embodied/motion135_to_smplx.py`

**Usage**:
```bash
python scripts/embodied/motion135_to_smplx.py \
    work_dirs/eval/motion_135.npz \
    output/motion_smplx.npz \
    --fps 30
```

**Input NPZ format**:
```python
{
    'motion_135': (T, 135),      # Required
    'positions': (T, 22, 3),     # Optional (used for validation)
    'translation': (T, 3)        # Optional
}
```

**Output NPZ format** (SMPL-X compatible):
```python
{
    'pose_body': (T, 63),        # 21 body joints in axis-angle
    'root_orient': (T, 3),       # Pelvis rotation in axis-angle
    'trans': (T, 3),             # Translation
    'betas': (10,),              # Shape parameters (zeros)
    'gender': 'neutral',         # Gender string
    'mocap_frame_rate': 30       # FPS
}
```

---

## Complete Example: End-to-End Conversion

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major 6D rot to rotation matrix."""
    # Row-major → column-major reorder
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    rotmat = np.stack([b1, b2, b3], axis=-1)
    return rotmat

def rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to axis-angle."""
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    rot = R.from_matrix(rotmat_flat)
    aa_flat = rot.as_rotvec()
    return aa_flat.reshape(*orig_shape, 3)

# Main conversion
motion_135 = np.load('motion_135.npz', allow_pickle=True)['motion_135']
T = motion_135.shape[0]

# Step 1: Extract components
transl = motion_135[:, :3]                  # (T, 3)
rot6d = motion_135[:, 3:].reshape(T, 22, 6)  # (T, 22, 6)

# Step 2: Convert to rotation matrices
rotmat = rot6d_to_rotmat(rot6d)             # (T, 22, 3, 3)

# Step 3: Convert to axis-angle
axis_angle = rotmat_to_axis_angle(rotmat)   # (T, 22, 3)

# Step 4: Split root and body
root_orient = axis_angle[:, 0, :]           # (T, 3)
body_pose = axis_angle[:, 1:, :].reshape(T, 63)  # (T, 63)

print(f"Conversion complete!")
print(f"  transl: {transl.shape}")
print(f"  root_orient: {root_orient.shape}")
print(f"  body_pose: {body_pose.shape}")
```

---

## References

- **SMPL Model**: https://smpl.is.tue.mpg.de/
- **SMPL-X Paper**: "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image"
- **SmplxLite Source**: `hftrainer/models/motion/components/body_models/smplx_lite.py`
- **Motion Processing**: `hftrainer/models/motion/CLAUDE.md` (§Motion Representation)

