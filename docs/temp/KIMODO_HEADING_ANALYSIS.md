# KIMODO Heading Representation Analysis: Can Full SMPL Root Rotation Be Recovered?

## Executive Summary

**Question**: Does KIMODO's 2D heading representation [cos(ψ), sin(ψ)] lose pitch/roll information? Can we fully convert KIMODO output back to SMPL's global orient (root rotation)?

**Answer**: 
- ✅ **NO, KIMODO does NOT lose pitch/roll**
- ✅ **YES, full SMPL root rotation can be recovered**
- The 2D heading is a **summary feature** for canonicalization and constraints, NOT the actual storage
- The **full 3D root rotation is preserved** in the `global_rot_data[0]` (6D continuous representation)

---

## Part 1: KIMODO Motion Representation

### Feature Layout

From `kimodo/motion_rep/reps/kimodo_motionrep.py` lines 34-41:

```python
self.size_dict = {
    "smooth_root_pos":        torch.Size([3]),        # dims [0:3]
    "global_root_heading":    torch.Size([2]),        # dims [3:5]      ← 2D heading [cos(ψ), sin(ψ)]
    "local_joints_positions": torch.Size([27, 3]),   # dims [5:86]
    "global_rot_data":        torch.Size([27, 6]),   # dims [86:248]    ← 27 joints × 6D rotation
    "velocities":             torch.Size([27, 3]),   # dims [248:329]
    "foot_contacts":          torch.Size([4]),       # dims [329:333]
}
```

**Total: 333 dims for 27-joint skeleton**

### Key Components

1. **`smooth_root_pos` [3]**: XYZ position of smoothed pelvis
   - X, Z are heavily smoothed (animator-friendly)
   - Y is absolute height
   
2. **`global_root_heading` [2]**: Yaw angle as [cos(ψ), sin(ψ)]
   - Extracted from hip vector
   - Line 80-81: `root_heading_angle = compute_heading_angle(...)` → then `[cos(ψ), sin(ψ)]`
   - **Used for**: canonicalization, constraint hints, feature rotation
   - **NOT used for**: root orientation reconstruction
   
3. **`global_rot_data` [27×6]**: **FULL 3D rotation for all joints**
   - Includes ROOT's full 3D rotation at index 0: `global_rot_data[..., 0, :]`
   - 6D continuous representation (can reconstruct 3×3 rotation matrix)
   - Lines 74-75, 90: Global rotations from FK are converted to 6D
   - **This preserves pitch, roll, AND yaw of the root**

---

## Part 2: Forward Pass (Feature Extraction)

### Code Path: `KimodoMotionRep.__call__()` (lines 50-106)

```python
def __call__(self, local_joint_rots, root_positions, to_normalize, lengths=None):
    # Step 1: Compute global rotations and positions via FK
    global_joints_rots,                    # [B,T,J,3,3] - FULL 3D rotations
    global_joints_positions,               # [B,T,J,3]
    local_joints_positions_origin_is_pelvis = fk(local_joint_rots, root_positions, self.skeleton)
    
    # Step 2: Extract heading angle (YAW ONLY)
    root_heading_angle = compute_heading_angle(global_joints_positions, self.skeleton)
                        # Line 125: heading_angle = atan2(diff[..., 2], -diff[..., 0])
                        # where diff = r_hip - l_hip
    global_root_heading = torch.stack([torch.cos(root_heading_angle), 
                                       torch.sin(root_heading_angle)], dim=-1)
    
    # Step 3: Convert ALL joint rotations to 6D (INCLUDING ROOT at index 0)
    global_rot_data = matrix_to_cont6d(global_joints_rots)
    # Result: [B,T,27,6]
    # global_rot_data[..., 0, :] contains the FULL 3D root rotation in 6D form
    
    # Step 4: Pack features
    features = einops.pack([
        smooth_root_pos,              # 3 dims
        global_root_heading,          # 2 dims (YAW ONLY - summary!)
        local_joints_positions,       # 81 dims
        global_rot_data,              # 162 dims (INCLUDES full root 3D rotation!)
        velocities,                   # 81 dims
        foot_contacts,                # 4 dims
    ], "batch time *")
    return features  # [B,T,333]
```

### Critical Insight

The `global_root_heading [2D]` appears to be the root rotation, but it's **NOT**:
- It's only a **summary** (yaw-only projection)
- The **actual full 3D root rotation** is in `global_rot_data[..., 0, :]` (6 dims)
- This 6D form fully encodes pitch, roll, AND yaw

---

## Part 3: Inverse Reconstruction (Feature Decoding)

### Code Path: `KimodoMotionRep.inverse()` (lines 162-215)

```python
def inverse(self, features, is_normalized, posed_joints_from="rotations", return_numpy=False):
    # Step 1: Unpack features
    [
        smooth_root_pos,
        global_root_heading,           # [B,T,2] - UNPACKED but NOT USED!
        local_joints_positions,
        global_rot_data,               # [B,T,27,6] - THIS is reconstructed!
        velocities,
        foot_contacts,
    ] = einops.unpack(features, self.ps, "batch time *")
    
    # Step 2: Convert 6D rotations back to 3×3 matrices for ALL joints
    global_rot_mats = cont6d_to_matrix(global_rot_data)
    # Result: [B,T,27,3,3]
    # global_rot_mats[..., 0, :, :] contains the FULL 3D root rotation matrix
    
    # Step 3: Convert global → local rotations (converts root too!)
    local_rot_mats = global_rots_to_local_rots(global_rot_mats, self.skeleton)
    # Line 188: Calls global_rots_to_local_rots which converts ALL joints including root
    # Result: [B,T,27,3,3] in local (parent-relative) coordinates
    
    # Step 4: Reconstruct root positions
    posed_joints_from_pos = local_joints_positions.clone()
    posed_joints_from_pos[..., 0] += smooth_root_pos[..., None, 0]
    posed_joints_from_pos[..., 2] += smooth_root_pos[..., None, 2]
    root_positions = posed_joints_from_pos[..., self.skeleton.root_idx, :]
    
    # Step 5: Perform FK to get posed joint positions
    if posed_joints_from == "rotations":
        _, posed_joints, _ = self.skeleton.fk(local_rot_mats, root_positions)
    
    # Step 6: Return reconstruction
    output_tensor_dict = {
        "local_rot_mats": local_rot_mats,          # [B,T,27,3,3] - LOCAL rotations
        "global_rot_mats": global_rot_mats,        # [B,T,27,3,3] - GLOBAL rotations
        "posed_joints": posed_joints,
        "root_positions": root_positions,
        "smooth_root_pos": smooth_root_pos,
        "foot_contacts": foot_contacts > 0.5,
        "global_root_heading": global_root_heading, # [B,T,2] - NOT USED, for reference only
    }
    return output_tensor_dict
```

### Key Observations

1. **`global_root_heading` is NOT used for reconstruction** (line 181 unpacks it, but no usage)
2. **Full 3D rotation is reconstructed** from `global_rot_data[..., 0, :]`
3. **Root rotation includes pitch and roll**, not just yaw

---

## Part 4: SMPL Conversion (AMASS Format)

### Code Path: `exports/smplx.py:get_amass_parameters()` (lines 15-63)

```python
def get_amass_parameters(local_rot_mats, root_positions, skeleton, z_up=True):
    """Convert KIMODO output to AMASS/SMPL-compatible format."""
    
    # Extract root rotation matrix (FULL 3×3)
    root_rot_mats = to_numpy(local_rot_mats[:, :, 0])  # [B,T,3,3] - ROOT joint
    
    # Convert root matrix to axis-angle (3D representation)
    root_orient = to_numpy(matrix_to_axis_angle(to_torch(root_rot_mats)))
    # Result: [B,T,3] - axis-angle representation
    # This fully captures pitch, roll, and yaw!
    
    # Handle body joints (remaining 26)
    local_rot_axis_angle = to_numpy(matrix_to_axis_angle(to_torch(local_rot_mats)))
    pose_body = einops.rearrange(local_rot_axis_angle[:, :, 1:], "b t j d -> b t (j d)")
    # Result: [B,T,78] - 26 joints × 3 dims
    
    # Handle coordinate transform (Y-up to Z-up if needed)
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
        combined_rotation = rot_z_180 @ y_up_to_z_up
        root_rot_mats = np.matmul(combined_rotation, root_rot_mats)  # Apply transform
    
    return trans, root_orient, pose_body
```

### SMPL Output Format

```
trans: [B,T,3]           - pelvis translation
root_orient: [B,T,3]     - root rotation (axis-angle)
pose_body: [B,T,78]      - 26 body joints (axis-angle)
```

**The `root_orient [3D]` fully contains pitch, roll, and yaw** ✅

---

## Part 5: Where Is Information Stored?

### Feature Storage Hierarchy

```
KIMODO Features [333 dims]
├── smooth_root_pos [3]
│   └── Position only - NO orientation info here
├── global_root_heading [2]    ← MISLEADING NAME!
│   └── YAW ONLY [cos(ψ), sin(ψ)]
│   └── For canonicalization/hints ONLY
│   └── NOT used for reconstruction
├── local_joints_positions [81]
│   └── Position of body joints (no orientation)
├── global_rot_data [162]      ← ACTUAL FULL 3D ROTATIONS!
│   ├── Root rotation [6]: FULL 3D (pitch, roll, yaw)
│   └── Body rotations [156]: 26 joints × 6D each
├── velocities [81]
└── foot_contacts [4]

Reconstruction flow:
global_rot_data[0, :] [6D] 
    ↓ cont6d_to_matrix()
    [3×3 rotation matrix] 
    ↓ global_rots_to_local_rots()
    [3×3 local rotation matrix]
    ↓ matrix_to_axis_angle()
    [3D axis-angle] ← SMPL root_orient
```

---

## Part 6: Answer to Your Questions

### Q1: Does KIMODO truly lose pitch/roll of the root?

**Answer: NO**

- The 2D `global_root_heading [cos(ψ), sin(ψ)]` stores YAW ONLY
- **BUT** the full 3D root rotation is in `global_rot_data[0]` as 6D continuous rotation
- This 6D form is losslessly reversible to 3×3 matrix (includes pitch and roll)
- The confusion arises from the *misleading naming* of `global_root_heading`

### Q2: How does KIMODO reconstruct SMPL output for evaluation?

**Answer: Via direct matrix conversion**

Path: `global_rot_data[..., 0, :]` [6D] → `cont6d_to_matrix()` → `global_rots_to_local_rots()` → `matrix_to_axis_angle()` → SMPL `root_orient` [3D]

All three DOF (pitch, roll, yaw) are preserved at each step.

### Q3: Is there a way to preserve full root rotation in a KIMODO-style representation?

**Answer: ALREADY PRESERVED!**

KIMODO already does this correctly:
- Store root rotation as 6D continuous (or full 3×3 matrix)
- Add 2D yaw-only summary for canonicalization/guidance (optional)
- During reconstruction, use only the 6D form

**Best practice for custom representation**:
```
Root representation options:
1. [6D continuous]: Recommended - same as KIMODO
2. [3×3 matrix]: Lossless but redundant (9 dims vs 6)
3. [3D axis-angle]: Efficient (3 dims) but harder to interpolate
4. [3D Euler]: NOT recommended - gimbal lock issues

Keep 2D heading [cos(ψ), sin(ψ)] ONLY for:
- Canonicalization hints
- Text prompt encoding
- Constraint specification
NOT for reconstruction!
```

---

## Part 7: Key File Locations and Line Numbers

### Critical Code References

| File Path | Lines | Content |
|-----------|-------|---------|
| `kimodo/motion_rep/reps/kimodo_motionrep.py` | 26-41 | Feature layout (`size_dict`) |
| `kimodo/motion_rep/reps/kimodo_motionrep.py` | 50-106 | Forward pass (`__call__`) |
| `kimodo/motion_rep/reps/kimodo_motionrep.py` | 74-90 | FK + global rotation extraction |
| `kimodo/motion_rep/reps/kimodo_motionrep.py` | 80-81 | Heading computation |
| `kimodo/motion_rep/reps/kimodo_motionrep.py` | 162-215 | Inverse reconstruction |
| `kimodo/motion_rep/reps/kimodo_motionrep.py` | 187-188 | 6D→matrix→local rotations |
| `kimodo/motion_rep/feature_utils.py` | 112-126 | Heading angle computation |
| `kimodo/skeleton/transforms.py` | 12-39 | Global→local rotation conversion |
| `kimodo/exports/smplx.py` | 15-63 | AMASS/SMPL conversion |
| `kimodo/exports/smplx.py` | 62 | Root orientation extraction |

### Geometry Functions

| Function | File | Purpose |
|----------|------|---------|
| `cont6d_to_matrix()` | `geometry.py` | Convert 6D continuous → 3×3 rotation |
| `matrix_to_cont6d()` | `geometry.py` | Convert 3×3 rotation → 6D continuous |
| `matrix_to_axis_angle()` | `geometry.py` | Convert 3×3 rotation → 3D axis-angle |
| `global_rots_to_local_rots()` | `skeleton/transforms.py` | Convert global → local rotations |
| `compute_heading_angle()` | `motion_rep/feature_utils.py` | Extract yaw from joint positions |

---

## Part 8: Implications for Your Pipeline

### For Motion Representation

✅ **KIMODO's approach is sound**:
- Stores full 3D rotation in `global_rot_data`
- Uses 2D heading as auxiliary feature (not primary)
- Reconstruction perfectly recovers all 3 DOF

### For SMPL Compatibility

✅ **Full conversion to SMPL is possible**:
```
KIMODO output → AMASS format
├── root_orient [3D axis-angle]: ✅ Full pitch/roll/yaw
├── pose_body [21×3]: ✅ All joint rotations
├── trans [3D]: ✅ Root translation
└── All parameters needed for SMPL rendering
```

### For Your Use Case

If you're adapting KIMODO for your needs:

1. **Preserve the structure**: Keep 6D rotation storage
2. **Add pitch/roll guidance**: If needed, add auxiliary 2D features for pitch/roll separately
3. **Test reconstruction**: Verify round-trip conversion preserves accuracy
4. **Use axis-angle for SMPL**: Convert to 3D axis-angle at output time

---

## Conclusion

**KIMODO does NOT lose pitch/roll information.** The 2D heading is a *summary feature* for canonicalization and constraints, not the actual root rotation storage. The full 3D root rotation is preserved in the 6D continuous representation within `global_rot_data[0]`, and it's perfectly recoverable to SMPL format.

The confusion arises from:
1. The name `global_root_heading` suggesting it's the primary root representation (it's not)
2. The existence of 2D yaw-only information alongside full 3D storage (redundant by design)
3. The reconstruction code not explicitly using `global_root_heading` (by design - it's auxiliary)

For inference output and SMPL evaluation, KIMODO's approach is correct and complete.
