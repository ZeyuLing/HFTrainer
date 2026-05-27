# 272-Dim Motion Representation: Rotation Value Mismatch - Complete Analysis

**Generated**: 2026-05-27  
**Issue**: local_rot6d values (dims 148-271) mismatch between GT HumanML3D data and PRISM conversion

---

## FULL FILE CONTENTS

### 1. face_z_align_util.py (Complete File)

Location: `ref_repo/MotionStreamer/MotionStreamer/utils/face_z_align_util.py`

**Size**: 1066 lines

**Key sections**:
- Lines 1-25: Quaternion operations (qinv, qnormalize, qmul, qrot)
- Lines 69-117: Euler angle conversions (qeuler)
- Lines 142-159: qfix (quaternion continuity enforcement)
- Lines 162-204: Euler/quaternion conversions
- Lines 207-310: Continuous 6D representations
- Lines 440-458: Face joint indices mapping
- **Lines 460-516**: `face_z_transform()` - CRITICAL FUNCTION
- **Lines 549-576**: `quaternion_to_matrix()` 
- Lines 578-828: Matrix/quaternion/axis-angle conversions (PyTorch3D functions)
- **Lines 899-954**: `axis_angle_to_quaternion()`, `axis_angle_to_matrix()`, `matrix_to_axis_angle()`
- **Lines 986-1005**: `rotation_6d_to_matrix()` - Gram-Schmidt orthogonalization
- **Lines 1008-1021**: `matrix_to_rotation_6d()` - **EXTRACTS FIRST 2 ROWS (ROW-MAJOR)**
- Lines 1024-1048: `canonicalize_smplh()`

**Critical function - matrix_to_rotation_6d (lines 1008-1021)**:
```python
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
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
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
```

**Extraction method**: ROW-MAJOR
- Takes (*, 3, 3) matrix
- Extracts [:2, :] → (*, 2, 3) 
- Reshapes to (*, 6) → **[R₀₀, R₀₁, R₀₂, R₁₀, R₁₁, R₁₂]**

---

### 2. representation_272.py (First 150 lines - Main GT Generation)

Location: `ref_repo/MotionStreamer/272-dim-Motion-Representation/representation_272.py`

```python
# representation: 272 dim
# :2 local xz velocities of root, no heading, can recover translation
# 2:8  heading angular velocities, 6d rotation, can recover heading
# 8:8+3*njoint local position, no heading, all at xz origin
# 8+3*njoint:8+6*njoint local velocities, no heading, all at xz origin, can recover local postion
# 8+6*njoint:8+12*njoint local rotations, 6d rotation, no heading, all frames z+

import numpy as np
from utils.face_z_align_util import expmap_to_quaternion, quaternion_to_matrix, quaternion_to_matrix_np, matrix_to_rotation_6d, qrot_np, rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import torch
import scipy.ndimage as ndimage
from tqdm import tqdm
import os
import argparse

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def rot_yaw(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])

def foot_detect(global_positions, thres):
    """
        derived from https://github.com/orangeduck/Motion-Matching/blob/37df18afc44e8acca3af5e85dff96effa6a34b03/resources/generate_database.py#L160
    """
    left_foot = 10
    right_foot = 11
    global_velocities = global_positions[1:] - global_positions[:-1]
    contact_velocities = np.sqrt(np.sum(global_velocities[:, np.array([left_foot, right_foot])]**2, axis=-1))
    contacts = contact_velocities < thres
    # Median filter here acts as a kind of "majority vote", and removes
    # small regions  where contact is either active or inactive
    for ci in range(contacts.shape[1]):
        contacts[:,ci] = ndimage.median_filter(
            contacts[:,ci], 
            size=6, 
            mode='nearest')
    return contacts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory path')
    args = parser.parse_args()

    bad_cnt = 0
    for file in tqdm(findAllFile(os.path.join(args.filedir, 'smpl_85_face_z_transform_joints'))):
        output_file = file.replace('smpl_85_face_z_transform_joints', 'Representation_272')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        root_idx = 0
        # get joint positions
        position_data = np.load(file)
        position_data = position_data[:, :22, :3]
        nfrm, njoint, _ = position_data.shape
        # get smpl rotations
        rotation_smpl_axis_angle = np.load(file.replace('smpl_85_face_z_transform_joints', 'smpl_85_face_z_transform'))
        rotations_wxyz = expmap_to_quaternion(rotation_smpl_axis_angle[:, :66].reshape(nfrm, njoint, 3))
        
        rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)  # nframe, njoint, 3, 3

        # put on floor and put root on origin for the first frame
        ori = copy.deepcopy(position_data[0,root_idx]) # first frame root position
        y_min = np.min(position_data[:,:,1])
        ori[1] = y_min
        position_data = position_data - ori
        velocities_root = position_data[1:,root_idx,:] - position_data[:-1,root_idx,:]

        # smpl unit is m and 0.15 is given as cm, may need to change depending on the datasets
        contacts = foot_detect(position_data, 0.15/100)
        
        # calculate local position, all frames on xz origin
        position_data[:,:,0] -= position_data[:,0:1,0]
        position_data[:,:,2] -= position_data[:,0:1,2]

        # calculate heading
        global_heading = - np.arctan2(rotations_matrix[:,root_idx,0,2], rotations_matrix[:, root_idx, 2,2])
        global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
        global_heading_diff = global_heading[1:] - global_heading[:-1]
        global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

        # calculate positions no heading
        positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None,:, :], njoint, axis=1), position_data[...,None]).squeeze(-1)

        # calculate velocity no heading
        velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

        # calculate root velocity_xz_no_heading
        velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[...,[0,2]]

        # calculate rotations no heading
        rotations_matrix[:,0,...] = np.matmul(global_heading_rot, rotations_matrix[:,0,...]) 

        # concat all
        size_frame = 8+njoint*3+njoint*3+njoint*6
        final_x = np.zeros((nfrm, size_frame))

        # set the first frame of the root rotation to identity
        final_x[0, 2] = 1
        final_x[0, 6] = 1
        try:
            final_x[1:,2:8] = matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy() # take 6D rotation
        except:
            bad_cnt += 1
            continue
        final_x[1:,:2] = velocities_root_xy_no_heading 
        final_x[:,8:8+3*njoint] = np.reshape(positions_no_heading, (nfrm,-1))
        final_x[1:,8+3*njoint:8+6*njoint] = np.reshape(velocities_no_heading, (nfrm-1,-1))
        final_x[:,8+6*njoint:8+12*njoint] = np.reshape(rotations_matrix[..., :, :2, :], (nfrm,-1)) # take 6D rotation
        np.save(output_file, final_x)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in {args.filedir}/Representation_272")
```

**KEY STEP (Line 116)**: 
```python
final_x[:,8+6*njoint:8+12*njoint] = np.reshape(rotations_matrix[..., :, :2, :], (nfrm,-1))
```

Wait - this is DIFFERENT! It uses `[:, :2, :]` NOT `[:2, :]`!

Let me check this carefully:
- rotations_matrix shape: (nfrm, njoint, 3, 3)
- `rotations_matrix[..., :, :2, :]` → (nfrm, njoint, 2, 3)
- This extracts the first 2 COLUMNS of the 3×3, not the first 2 ROWS!

---

### 3. convert_prism_to_272.py (First 200 lines + Key Sections)

Location: `ref_repo/MotionStreamer/convert_prism_to_272.py`

**Full first 200 lines**:

```python
"""
Convert PRISM predictions to 272-dim MotionStreamer representation.

Pipeline:
1. Load PRISM NPZ (global_orient, body_pose, transl, betas)
2. Apply face_z_transform (rotate first frame heading to face Z+)
3. Run FK using SMPL (NOT SMPLX!) to get 22 joint positions
4. Apply representation_272 logic (heading removal, local positions, velocities, 6D rotations)
5. Save as <original_hml3d_id>.npy with shape (T, 272)

IMPORTANT: The GT 272-dim data was computed with SMPL body model.
Using SMPLX produces physically impossible skeletons due to different body templates.
PRISM outputs 63-dim body_pose (21 SMPLX joints), so we pad to 69-dim (23 SMPL joints)
with zeros for the last 2 hand joints.

Usage:
    python3 convert_prism_to_272.py \
        --pred_dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten \
        --annotation data/annotation/test_hml3d.json \
        --out_dir ref_repo/MotionStreamer/prism_272_predictions
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import smplx
from tqdm import tqdm


# ============ Rotation Utilities (from MotionStreamer's face_z_align_util.py) ============

def expmap_to_quaternion(e):
    """Convert axis-angle (exponential map) to quaternion [w, x, y, z]."""
    assert e.shape[-1] == 3
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)
    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def qmul_np(q, r):
    """Multiply quaternions q and r (numpy)."""
    q = torch.from_numpy(q).contiguous().float()
    r = torch.from_numpy(r).contiguous().float()
    return qmul(q, r).numpy()


def qmul(q, r):
    """Multiply quaternion(s) q with quaternion(s) r (torch)."""
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    original_shape = q.shape
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot_np(q, v):
    """Rotate vector(s) v by quaternion(s) q (numpy)."""
    q = torch.from_numpy(q).contiguous().float()
    v = torch.from_numpy(v).contiguous().float()
    return qrot(q, v).numpy()


def qrot(q, v):
    """Rotate vector(s) v by quaternion(s) q (torch)."""
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    original_shape = list(v.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qbetween_np(v0, v1):
    """Find quaternion to rotate v0 to v1."""
    assert v0.shape[-1] == 3
    assert v1.shape[-1] == 3
    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1, keepdim=True)
    q = torch.cat([w, v], dim=-1)
    return (q / torch.norm(q, dim=-1, keepdim=True)).numpy()


def quaternion_to_matrix_np(quaternions):
    """Convert quaternions [w,x,y,z] to rotation matrices (numpy)."""
    q = torch.from_numpy(quaternions).contiguous().float()
    return quaternion_to_matrix(q).numpy()


def quaternion_to_matrix(quaternions):
    """Convert quaternions [w,x,y,z] to rotation matrices (torch)."""
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


def matrix_to_rotation_6d(matrix):
    """Convert rotation matrix to 6D representation (first two ROWS = row-major convention).
    This matches MotionStreamer/pytorch3d convention where rot6d = [row0, row1] of R."""
    # matrix: (..., 3, 3) -> (..., 6)
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
```

**Key section - compute_representation_272 (lines 358-453)**:

```python
def compute_representation_272(joints_22, smpl_85_face_z):
    """
    Compute 272-dim representation from joint positions and face-Z parameters.

    Args:
        joints_22: (T, 22, 3) joint positions from FK
        smpl_85_face_z: (T, 85) face-Z-transformed parameters

    Returns:
        repr_272: (T, 272) or None if conversion fails
    """
    root_idx = 0
    nfrm = joints_22.shape[0]
    njoint = 22

    position_data = joints_22.copy()

    # Get rotations: first 66 dims = 22 joints x 3 axis-angle
    rotation_smpl_axis_angle = smpl_85_face_z[:, :66].reshape(nfrm, njoint, 3)
    rotations_wxyz = expmap_to_quaternion(rotation_smpl_axis_angle)  # (T, 22, 4)
    rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)  # (T, 22, 3, 3)

    # Put on floor and center root at origin for first frame
    ori = position_data[0, root_idx].copy()
    y_min = np.min(position_data[:, :, 1])
    ori[1] = y_min
    position_data = position_data - ori

    # Root velocities (before removing XZ)
    velocities_root = position_data[1:, root_idx, :] - position_data[:-1, root_idx, :]

    # Calculate local position: all frames at XZ origin
    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    # Calculate heading from root rotation matrix
    global_heading = -np.arctan2(
        rotations_matrix[:, root_idx, 0, 2],
        rotations_matrix[:, root_idx, 2, 2]
    )
    global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

    # Positions with heading removed
    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1),
        position_data[..., None]
    ).squeeze(-1)

    # Velocities with heading removed
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    # Root velocity XZ with heading removed
    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1],
        velocities_root[:, :, None]
    ).squeeze()[..., [0, 2]]

    # Remove heading from root rotation
    rotations_matrix[:, 0, ...] = np.matmul(global_heading_rot, rotations_matrix[:, 0, ...])

    # Pack into 272-dim representation
    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6  # = 272
    final_x = np.zeros((nfrm, size_frame))

    # First frame: identity heading rotation (6D)
    final_x[0, 2] = 1  # first column of identity = [1,0,0] but in 6D = cols 0,1 of 3x3
    final_x[0, 6] = 1  # second column of identity

    # Heading angular velocity as 6D rotation
    try:
        final_x[1:, 2:8] = matrix_to_rotation_6d(
            torch.from_numpy(global_heading_diff_rot).float()
        ).numpy()
    except Exception as e:
        print(f"  Warning: matrix_to_rotation_6d failed: {e}")
        return None

    # Root XZ velocity (heading removed)
    final_x[1:, :2] = velocities_root_xy_no_heading

    # Local positions (heading removed)
    final_x[:, 8:8 + 3 * njoint] = np.reshape(positions_no_heading, (nfrm, -1))

    # Local velocities (heading removed)
    final_x[1:, 8 + 3 * njoint:8 + 6 * njoint] = np.reshape(velocities_no_heading, (nfrm - 1, -1))

    # Local rotations as 6D (heading removed from root)
    # rotations_matrix shape: (T, 22, 3, 3)
    # Take first 2 columns: (T, 22, 3, 2) -> reshape to (T, 22*6)
    final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(
        rotations_matrix[..., :2, :], (nfrm, -1)
    )

    return final_x
```

**Line 449-451 in convert_prism_to_272.py**:
```python
final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(
    rotations_matrix[..., :2, :], (nfrm, -1)
)
```

---

## CRITICAL DISCOVERY: The Rotation Extraction Mismatch

### GT Generation (representation_272.py, line 116)
```python
final_x[:,8+6*njoint:8+12*njoint] = np.reshape(rotations_matrix[..., :, :2, :], (nfrm,-1))
```
**Extracts**: First 2 **COLUMNS** of rotation matrix
- `rotations_matrix[..., :, :2, :]` with shape (T, 22, 3, 3)
- Result: (T, 22, 3, 2)  
- 6D vector: **[R₀₀, R₁₀, R₂₀, R₀₁, R₁₁, R₂₁]** (COLUMN-MAJOR)

### PRISM Conversion (convert_prism_to_272.py, line 449-450)
```python
final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(
    rotations_matrix[..., :2, :], (nfrm, -1)
)
```
**Extracts**: First 2 **ROWS** of rotation matrix
- `rotations_matrix[..., :2, :]` with shape (T, 22, 3, 3)
- Result: (T, 22, 2, 3)
- 6D vector: **[R₀₀, R₀₁, R₀₂, R₁₀, R₁₁, R₁₂]** (ROW-MAJOR)

### THE BUG
**These extract different elements!** The GT uses **COLUMN-MAJOR** extraction, but convert_prism_to_272.py uses **ROW-MAJOR** extraction.

For a rotation matrix:
```
R = [[a, b, c],
     [d, e, f],
     [g, h, i]]
```

- GT extracts: [a, d, g, b, e, h] (columns stacked)
- PRISM extracts: [a, b, c, d, e, f] (rows stacked)

This is the **ROOT CAUSE** of the local_rot6d mismatch!

---

## Data Processing Pipeline Comparison

### GT Generation (AMASS → representation_272)
```
1. amass_process.py loads AMASS data
   - Extracts: root_orient, pose_body, pose_hand, pose_jaw, trans, betas
   - Applies process_pose() [canonical transform]
   
2. face_z_align() applied within amass_process
   - Calls SMPLX FK to get joint positions
   - Calls face_z_transform(positions, global_orient, trans)
     - Uses position-based heading calculation
     - Modifies global_orient to face Z+
   - Saves as smpl_85_face_z_transform format
   
3. representation_272.py loads pre-computed data
   - Loads smpl_85_face_z_transform (.npy) → axis-angle rotations
   - Loads smpl_85_face_z_transform_joints (.npy) → joint positions
   - Converts axis-angle → quaternion → matrix
   - Calculates heading from rotation matrix
   - Removes heading from rotations
   - EXTRACTS COLUMNS: rotations_matrix[..., :, :2, :] ← COLUMN-MAJOR
```

### PRISM Conversion
```
1. Load PRISM NPZ
   - global_orient: (T, 3) axis-angle
   - body_pose: (T, 63) axis-angle  
   - transl: (T, 3)
   - betas: (10,)
   
2. convert_prism_to_272.py face_z_transform()
   - Simplified heading extraction from rotation only
   - No position-based calculation
   - Different from original face_z_align()
   
3. run_fk_smpl()
   - SMPL FK with face_z transformed parameters
   - Gets joint positions
   
4. compute_representation_272()
   - Converts axis-angle → quaternion → matrix
   - Calculates heading from rotation matrix
   - Removes heading from rotations
   - EXTRACTS ROWS: rotations_matrix[..., :2, :] ← ROW-MAJOR (WRONG!)
```

---

## Summary of Findings

| Aspect | GT (representation_272.py) | PRISM (convert_prism_to_272.py) | Status |
|--------|---------------------------|--------------------------------|--------|
| Rotation 6D extraction | `rotations_matrix[..., :, :2, :]` | `rotations_matrix[..., :2, :]` | ❌ **MISMATCH** |
| Extraction method | Column-major (columns stacked) | Row-major (rows stacked) | ❌ **DIFFERENT** |
| 6D order | [R₀₀, R₁₀, R₂₀, R₀₁, R₁₁, R₂₁] | [R₀₀, R₀₁, R₀₂, R₁₀, R₁₁, R₁₂] | ❌ **OPPOSITE** |
| Face Z Transform | Full implementation (position-based) | Simplified (rotation-based) | ⚠️  Different |
| matrix_to_rotation_6d() | Row-major (but not used!) | Row-major (code says this) | ⚠️  Inconsistent |

---

## SOLUTION

Change line 449-450 in convert_prism_to_272.py from:
```python
final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(
    rotations_matrix[..., :2, :], (nfrm, -1)
)
```

To:
```python
final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(
    rotations_matrix[..., :, :2, :], (nfrm, -1)
)
```

This will match GT's column-major extraction convention.

