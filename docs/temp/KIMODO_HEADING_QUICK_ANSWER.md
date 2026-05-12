# KIMODO Heading Representation: Quick Answer

## Your Core Question
> Does KIMODO's 2D heading [cos(ψ), sin(ψ)] representation lose pitch/roll information? Can we recover full SMPL root rotation?

## Answer
✅ **NO - KIMODO does NOT lose pitch/roll**
✅ **YES - full SMPL root rotation can be recovered perfectly**

---

## Why the Confusion?

KIMODO stores TWO representations of root orientation:

### 1. **`global_root_heading` [2D] - YAW ONLY**
- What: `[cos(ψ), sin(ψ)]` 
- Where: dims [3:5] of 333-dim feature vector
- Used for: Canonicalization, constraints, text hints
- **NOT used for reconstruction**

### 2. **`global_rot_data[0]` [6D] - FULL 3D ROTATION** ⭐
- What: 6D continuous rotation (can reconstruct 3×3 matrix)
- Where: dims [86:91] of 333-dim feature vector  
- Contains: pitch, roll, AND yaw
- **Used for reconstruction** ✅

---

## Information Flow

```
Forward (Feature Extraction):
  local_joint_rots [27, 3×3]
  ↓ FK
  global_joint_rots [27, 3×3]  ← Full 3D root rotation at index 0
  ↓ matrix_to_cont6d
  global_rot_data [27, 6]      ← 6D form preserves all info!
  
  ALSO computed (for guidance):
  global_root_heading [2]      ← YAW ONLY summary

Inverse (Reconstruction):
  global_rot_data [27, 6]      ← Use this!
  ↓ cont6d_to_matrix
  global_rot_mats [27, 3×3]    ← Recover full rotation
  ↓ global_rots_to_local_rots
  local_rot_mats [27, 3×3]     ← Local coordinates
  ↓ matrix_to_axis_angle
  SMPL root_orient [3]         ← Axis-angle (pitch, roll, yaw)
  
  Note: global_root_heading [2] is NEVER used! ↓ Auxiliary only
```

---

## Key Evidence from Code

### File: `kimodo_motionrep.py`

**Line 80-81 (Forward):**
```python
root_heading_angle = compute_heading_angle(global_joints_positions, self.skeleton)
global_root_heading = torch.stack([torch.cos(root_heading_angle), 
                                   torch.sin(root_heading_angle)], dim=-1)
# Result: [B,T,2] - YAW ONLY
```

**Line 90 (Still Forward):**
```python
global_rot_data = matrix_to_cont6d(global_joints_rots)
# Result: [B,T,27,6] - Includes full 3D root rotation at [0]
```

**Line 187-188 (Inverse):**
```python
global_rot_mats = cont6d_to_matrix(global_rot_data)  # Recover 3×3 matrices
local_rot_mats = global_rots_to_local_rots(global_rot_mats, self.skeleton)
# No use of global_root_heading!
```

### File: `exports/smplx.py`

**Line 34, 62:**
```python
root_rot_mats = to_numpy(local_rot_mats[:, :, 0])          # Get root matrix
root_orient = matrix_to_axis_angle(root_rot_mats)          # Convert to 3D
# Result: [B,T,3] - FULL pitch, roll, yaw in axis-angle form
```

---

## Information Storage Breakdown

| Component | Size | Contains | Used for |
|-----------|------|----------|----------|
| `smooth_root_pos` | 3D | XYZ position only | Root trajectory |
| `global_root_heading` | 2D | ψ (yaw) as [cos, sin] | Hints, canonicalization |
| `global_rot_data[0]` | 6D | **Full 3D rotation** | **Root orientation** ✅ |
| `global_rot_data[1:27]` | 156D | Body rotations | Body joints |

---

## Why This Design?

KIMODO intentionally stores:
1. **Full 3D rotation** [6D] in primary features (for accuracy)
2. **Yaw summary** [2D] as auxiliary (for convenience in canonicalization)

This is **redundant by design** - the 2D heading is just a projection of the 6D root rotation.

---

## For Your Pipeline

### If using KIMODO output directly:
✅ Root rotation is preserved perfectly
✅ Can convert to SMPL format without loss
✅ All 3 DOF (pitch, roll, yaw) available

### If adapting KIMODO's approach:
✅ Use 6D continuous rotation (not 3×3, not axis-angle)
✅ Add 2D heading as **optional** auxiliary feature
⚠️ Don't rely on heading for reconstruction
✅ Convert to axis-angle only at final output

---

## Bottom Line

The 2D heading is a **"nice-to-have" summary**, not the "real" root representation. The **"real" root rotation lives in `global_rot_data[0]`** as a 6D continuous form, which fully preserves pitch, roll, and yaw. KIMODO's reconstruction path perfectly recovers the full 3D rotation for SMPL output.

**Conclusion: You CAN fully convert KIMODO to SMPL with no information loss.** ✅
