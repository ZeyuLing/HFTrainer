# KIMODO Root Rotation: Code Reference Guide

## File Paths (from ref_repo root)

```
ref_repo/KIMODO/kimodo/kimodo/
├── motion_rep/
│   ├── reps/
│   │   ├── kimodo_motionrep.py       ← MAIN ANALYSIS FILE
│   │   ├── base.py                   ← Base class methods
│   │   └── __init__.py
│   ├── feature_utils.py              ← Heading computation
│   ├── conditioning.py
│   ├── feet.py
│   ├── smooth_root.py
│   └── stats.py
├── skeleton/
│   ├── transforms.py                 ← Rotation conversions
│   ├── kinematics.py                 ← FK functions
│   └── definitions.py
├── exports/
│   └── smplx.py                      ← SMPL conversion
├── geometry.py                        ← 6D ↔ matrix conversion
└── tools.py
```

---

## Critical Code Sections

### 1. Feature Layout Definition
**File**: `motion_rep/reps/kimodo_motionrep.py`  
**Lines**: 26-41

```python
def __init__(self, skeleton, fps, stats_path: Optional[str] = None):
    nbjoints = skeleton.nbjoints
    
    self.size_dict = {
        "smooth_root_pos": torch.Size([3]),           # [0:3]
        "global_root_heading": torch.Size([2]),       # [3:5] ← YAW ONLY
        "local_joints_positions": torch.Size([nbjoints, 3]),  # [5:86]
        "global_rot_data": torch.Size([nbjoints, 6]), # [86:248] ← FULL 3D
        "velocities": torch.Size([nbjoints, 3]),      # [248:329]
        "foot_contacts": torch.Size([4]),             # [329:333]
    }
    self.last_root_feature = "global_root_heading"  # Note: misleading!
```

### 2. Forward Pass (Feature Extraction)
**File**: `motion_rep/reps/kimodo_motionrep.py`  
**Lines**: 50-106  
**Key lines**: 74-90

```python
@ensure_batched(local_joint_rots=5, root_positions=3, lengths=1)
def __call__(self, local_joint_rots, root_positions, to_normalize, lengths=None):
    """Convert local rotations and root trajectory into smooth-root features."""
    
    # LINE 74-78: Compute global rotations and positions
    global_joints_rots,                    # [B,T,J,3,3] ← Full 3D!
    global_joints_positions,
    local_joints_positions_origin_is_pelvis = fk(
        local_joint_rots, root_positions, self.skeleton
    )
    
    # LINE 80-81: Extract YAW ONLY
    root_heading_angle = compute_heading_angle(
        global_joints_positions, self.skeleton
    )  # Returns [B,T] angles
    global_root_heading = torch.stack(
        [torch.cos(root_heading_angle), torch.sin(root_heading_angle)], 
        dim=-1
    )  # [B,T,2] - YAW ONLY
    
    # LINE 83-86: Position calculation
    smooth_root_pos = get_smooth_root_pos(root_positions)
    hips_offset = root_positions - smooth_root_pos
    hips_offset[..., 1] = root_positions[..., 1]
    local_joints_positions = local_joints_positions_origin_is_pelvis + hips_offset[:, :, None]
    
    # LINE 88-89: Velocities and foot contacts
    velocities = compute_vel_xyz(global_joints_positions, self.fps, lengths=lengths)
    foot_contacts = foot_detect_from_pos_and_vel(...)
    
    # LINE 90: CONVERT ALL ROTATIONS TO 6D ← KEY STEP!
    global_rot_data = matrix_to_cont6d(global_joints_rots)
    # Result: [B,T,27,6]
    # INCLUDES full root 3D rotation at global_rot_data[..., 0, :]
    
    # LINE 92-102: Pack all features
    features, _ = einops.pack([
        smooth_root_pos,              # 3 dims
        global_root_heading,          # 2 dims (yaw only)
        local_joints_positions,       # 81 dims
        global_rot_data,              # 162 dims (FULL 3D ROOT HERE!)
        velocities,                   # 81 dims
        foot_contacts,                # 4 dims
    ], "batch time *")
    
    if to_normalize:
        features = self.normalize(features)
    return features  # [B,T,333]
```

### 3. Inverse Reconstruction
**File**: `motion_rep/reps/kimodo_motionrep.py`  
**Lines**: 162-215  
**Key lines**: 175-194

```python
@ensure_batched(features=3)
def inverse(self, features, is_normalized, posed_joints_from="rotations",
           return_numpy: bool = False):
    """Decode smooth-root features into motion tensors."""
    
    if is_normalized:
        features = self.unnormalize(features)
    
    # LINE 178-185: Unpack features
    [
        smooth_root_pos,
        global_root_heading,              # [B,T,2] - UNPACKED
        local_joints_positions,
        global_rot_data,                  # [B,T,27,6] - KEY INPUT
        velocities,
        foot_contacts,
    ] = einops.unpack(features, self.ps, "batch time *")
    
    # LINE 187: Convert 6D → 3x3 matrices for ALL joints
    global_rot_mats = cont6d_to_matrix(global_rot_data)
    # Result: [B,T,27,3,3]
    # global_rot_mats[..., 0, :, :] is the FULL 3D root rotation matrix!
    
    # LINE 188: Convert global → local rotations (applies to root too!)
    local_rot_mats = global_rots_to_local_rots(global_rot_mats, self.skeleton)
    # Result: [B,T,27,3,3]
    # local_rot_mats[..., 0, :, :] is root in SMPL local frame
    
    # LINE 190-193: Reconstruct positions
    posed_joints_from_pos = local_joints_positions.clone()
    posed_joints_from_pos[..., 0] += smooth_root_pos[..., None, 0]
    posed_joints_from_pos[..., 2] += smooth_root_pos[..., None, 2]
    root_positions = posed_joints_from_pos[..., self.skeleton.root_idx, :]
    
    foot_contacts = foot_contacts > 0.5
    
    # LINE 196-202: FK if needed
    if posed_joints_from == "rotations":
        _, posed_joints, _ = self.skeleton.fk(local_rot_mats, root_positions)
    else:
        posed_joints = posed_joints_from_pos
    
    # LINE 204-215: Return dict
    output_tensor_dict = {
        "local_rot_mats": local_rot_mats,           # [B,T,27,3,3]
        "global_rot_mats": global_rot_mats,         # [B,T,27,3,3]
        "posed_joints": posed_joints,
        "root_positions": root_positions,
        "smooth_root_pos": smooth_root_pos,
        "foot_contacts": foot_contacts,
        "global_root_heading": global_root_heading, # [B,T,2] - NOT USED
    }
    if return_numpy:
        return to_numpy(output_tensor_dict)
    return output_tensor_dict

# CRITICAL: global_root_heading is NEVER USED in reconstruction!
```

### 4. Heading Angle Computation
**File**: `motion_rep/feature_utils.py`  
**Lines**: 111-126

```python
@ensure_batched(posed_joints=4)
def compute_heading_angle(posed_joints: torch.Tensor, skeleton: SkeletonBase):
    """Compute the heading direction from joint positions using the hip vector."""
    
    # Get left and right hip indices
    r_hip, l_hip = skeleton.hip_joint_idx
    
    # Compute hip vector (right to left)
    diff = posed_joints[:, :, r_hip] - posed_joints[:, :, l_hip]
    
    # Heading angle from hip vector (X-Z plane only!)
    heading_angle = torch.atan2(diff[..., 2], -diff[..., 0])
    return heading_angle  # [B,T]
    
    # NOTE: This is YAW ONLY - does NOT capture pitch or roll
```

### 5. Global → Local Rotation Conversion
**File**: `skeleton/transforms.py`  
**Lines**: 12-39

```python
def global_rots_to_local_rots(global_joint_rots: torch.Tensor, skeleton):
    """Convert global rotations to local rotations using skeleton hierarchy.
    
    This applies to ALL joints, including the root!
    """
    
    # Pack all rotations into flat array
    global_joint_mats, ps = einops.pack(
        [global_joint_rots],
        "* nbjoints dim1 dim2",
    )
    
    # Get parent rotation matrices
    parent_rot_mats = global_joint_mats[:, skeleton.joint_parents]
    
    # Set root parent to identity (root has no parent)
    parent_rot_mats[:, skeleton.root_idx] = torch.eye(3)
    
    # Compute local rotations: R_local = R_parent^T @ R_global
    parent_rot_mats_inv = parent_rot_mats.transpose(2, 3)
    local_rot_mats = torch.einsum(
        "T N m n, T N n o -> T N m o",
        parent_rot_mats_inv,
        global_joint_mats,
    )
    
    [local_rot_mats] = einops.unpack(local_rot_mats, ps, "* nbjoints dim1 dim2")
    return local_rot_mats  # Same shape as input
```

### 6. SMPL Conversion (AMASS Format)
**File**: `exports/smplx.py`  
**Lines**: 15-63  
**Key lines**: 34, 62

```python
@ensure_batched(local_rot_mats=5, root_positions=3, lengths=1)
def get_amass_parameters(local_rot_mats, root_positions, skeleton, z_up=True):
    """Convert KIMODO output to AMASS/SMPL-compatible format."""
    
    # Pelvis offset handling
    pelvis_offset = skeleton.neutral_joints[skeleton.root_idx].cpu().numpy()
    trans = root_positions - pelvis_offset
    
    # LINE 34: Extract ROOT rotation matrix (FULL 3x3)
    root_rot_mats = to_numpy(local_rot_mats[:, :, 0])  # [B,T,3,3]
    
    # Convert all rotations to axis-angle
    local_rot_axis_angle = to_numpy(
        matrix_to_axis_angle(to_torch(local_rot_mats))
    )  # [B,T,27,3]
    
    # Extract body (non-root) joints
    pose_body = einops.rearrange(
        local_rot_axis_angle[:, :, 1:], 
        "b t j d -> b t (j d)"
    )  # [B,T,78]
    
    # Handle coordinate transform if needed
    if z_up:
        y_up_to_z_up = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        rot_z_180 = np.array([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        y_up_to_z_up = np.matmul(rot_z_180, y_up_to_z_up)
        root_rot_mats = np.matmul(y_up_to_z_up, root_rot_mats)
        trans = np.matmul(trans + pelvis_offset, y_up_to_z_up.T) - pelvis_offset
    
    # LINE 62: Convert root matrix to axis-angle (3D)
    root_orient = to_numpy(matrix_to_axis_angle(to_torch(root_rot_mats)))
    # Result: [B,T,3] - FULL pitch, roll, yaw!
    
    return trans, root_orient, pose_body
```

---

## Data Flow Diagram

```
INPUT: local_joint_rots [B,T,27,3,3], root_positions [B,T,3]
  ↓
  fk() [kinematics.py]
  ↓
OUTPUT: global_joint_rots [B,T,27,3,3], global_joint_positions [B,T,27,3]
  ↓
compute_heading_angle() [feature_utils.py:125]
  ↓
heading_angle [B,T]  ← YAW ONLY
  ↓
global_root_heading = [cos(ψ), sin(ψ)] [B,T,2]
  ↓
matrix_to_cont6d(global_joint_rots) [geometry.py]
  ↓
global_rot_data [B,T,27,6]
  ↓  ← Includes FULL 3D root at index [0]!
  
PACK INTO FEATURES [B,T,333]
├─ dims [0:3]:    smooth_root_pos
├─ dims [3:5]:    global_root_heading (YAW ONLY)
├─ dims [5:86]:   local_joints_positions
├─ dims [86:248]: global_rot_data (FULL 3D ROOT HERE!)
├─ dims [248:329]: velocities
└─ dims [329:333]: foot_contacts

RECONSTRUCTION:
  cont6d_to_matrix(global_rot_data[..., 0, :]) [geometry.py]
  ↓
  global_rot_mats[..., 0, :, :] [3,3]  ← Full 3D root!
  ↓
  global_rots_to_local_rots() [transforms.py]
  ↓
  local_rot_mats[..., 0, :, :] [3,3]   ← In SMPL local frame
  ↓
  matrix_to_axis_angle() [geometry.py]
  ↓
  root_orient [3]  ← SMPL format (pitch, roll, yaw)
```

---

## Dimension Layout Reference

### Input Space (before motion_rep)
```
local_joint_rots: [B, T, 27, 3, 3]  ← Root at index 0
root_positions: [B, T, 3]
```

### KIMODO Feature Space [B, T, 333]
```
[0:3]      smooth_root_pos [3]
[3:5]      global_root_heading [2] ← YAW ONLY, not used in recon
[5:86]     local_joints_positions [27×3]
[86:91]    global_rot_data[0] [6] ← ROOT FULL 3D ROTATION!
[91:248]   global_rot_data[1:27] [26×6]
[248:329]  velocities [27×3]
[329:333]  foot_contacts [4]
```

### Output Space (after inverse)
```
local_rot_mats: [B, T, 27, 3, 3]     ← Root at index 0
global_rot_mats: [B, T, 27, 3, 3]    ← Root at index 0
root_positions: [B, T, 3]
```

### SMPL Format (AMASS)
```
trans: [B, T, 3]        ← Root translation
root_orient: [B, T, 3]  ← Root rotation (axis-angle, pitch/roll/yaw)
pose_body: [B, T, 78]   ← 26 joints × 3 (axis-angle)
```

---

## Debugging Checklist

To verify full 3D rotation preservation:

1. **Check feature extraction**:
   - Confirm `global_rot_data[..., 0, :]` has 6 dimensions
   - Compare `matrix_to_cont6d(global_joints_rots[..., 0, :, :])` output

2. **Check reconstruction**:
   - Confirm `cont6d_to_matrix()` recovers 3×3 for root
   - Compare reconstructed matrix with original (should be ~identical)

3. **Check SMPL conversion**:
   - Confirm `matrix_to_axis_angle()` produces [3] for root
   - Compare output with expected SMPL format

4. **Verify non-usage of heading**:
   - Check that reconstruction never reads `global_root_heading`
   - Confirm reconstruction uses `global_rot_data[0]` instead

---

## Key Takeaways

| Item | Location | Key Insight |
|------|----------|------------|
| Feature layout | `kimodo_motionrep.py:34-41` | `global_rot_data[0]` = full 3D rotation |
| Forward pass | `kimodo_motionrep.py:74-90` | Heading is YAW ONLY summary |
| Inverse reconstruction | `kimodo_motionrep.py:187-188` | Uses `global_rot_data`, not heading |
| Heading computation | `feature_utils.py:125` | Extracts ψ from hip vector only |
| Global→Local conversion | `transforms.py:33` | Applies to all joints including root |
| SMPL export | `smplx.py:62` | Converts root matrix to 3D axis-angle |

**Bottom line**: Full 3D root rotation is in `global_rot_data[0]`, not `global_root_heading`. The 2D heading is auxiliary. ✅
