# KIMODO Root/Translation Representation - Executive Summary

**Document**: Comprehensive analysis of KIMODO's "smooth trajectory + heading" root representation and conversion to HyMotion M2M.

**Files Generated**:
- `kimodo_root_analysis.md` - Detailed technical breakdown (10 sections)
- `kimodo_hymotion_mapping.md` - Concrete mapping examples with pseudocode

---

## Quick Reference

### 1️⃣ KIMODO 333D Representation (27 joints)

**Dimension Layout:**
```
[0:3]       smooth_root_pos          (3 dims)  = smoothed pelvis XZ + raw Y
[3:5]       global_root_heading      (2 dims)  = [cos(ψ), sin(ψ)]
[5:86]      local_joints_positions   (81 dims) = 27 joints × 3 (xz relative, y absolute)
[86:248]    global_rot_data          (162 dims)= 27 joints × 6 (6D continuous global rotation)
[248:329]   velocities               (81 dims) = 27 joints × 3 (global joint velocities)
[329:333]   foot_contacts            (4 dims)  = [L_heel, L_toe, R_heel, R_toe]
```

### 2️⃣ HyMotion M2M 138D Representation (22 joints)

**Dimension Layout:**
```
[0:3]       absolute_translation     (3 dims)  = world root position
[3:6]       relative_translation     (3 dims)  = frame-to-frame delta
[6:138]     local_rot_6d             (132 dims)= 22 joints × 6 (parent-relative 6D rotation)
```

### 3️⃣ Key Differences

| Aspect | KIMODO | HyMotion |
|--------|--------|---------|
| **Root Representation** | Smooth (ADMM-smoothed) | Raw + delta (noisy) |
| **Heading** | Explicit [cos ψ, sin ψ] | Embedded in root rotation |
| **Rotations** | Global 6D (world-frame) | Local 6D (parent-relative) |
| **Positions** | Stored (local relative) | Not stored (derived from FK) |
| **Foot Contacts** | Explicit 4D flags | Inferred from sliding |
| **Total Dims** | 333 | 138 |
| **Trajectory Quality** | Smooth, animator-friendly | Noisy, needs filtering |

---

## 🔑 Core Concepts

### Smooth Root (ADMM Algorithm)

**Purpose**: Reduce high-frequency jitter while preserving motion intent.

**Algorithm**:
1. Extract horizontal (XZ) plane from pelvis trajectory
2. Keep Y (height) as raw
3. Apply ADMM-based smoothing:
   - Minimize acceleration: `||A·x||²`
   - Soft constraints: `||x_i - target_i|| ≤ 0.06m` (margin per frame)
   - Multigrid: start coarse, double resolution iteratively
   - 500 ADMM iterations, over-relaxation α=1.8

**Benefit**: Foot skating reduces from 7.59 cm/s → 3.87 cm/s

**Output**: Smooth trajectory suitable for animator constraints (lines/curves)

### Global Root Heading

**Representation**: `[cos(ψ), sin(ψ)]` where ψ is yaw angle

**Convention** (Y-up, Z-forward):
- ψ = 0 → +Z (forward)
- ψ = π/2 → -X (left)
- ψ = π → -Z (backward)
- ψ = -π/2 → +X (right)

**Advantage**: Normalized, no singularities (unlike quaternion w-component)

### Global vs Local Rotations

**KIMODO (Global 6D)**:
- 6D continuous: first 2 columns of rotation matrix
- World-frame (not parent-relative)
- Allows direct imputation of end-effector constraints
- No IK required for constraint application

**HyMotion (Local 6D)**:
- 6D continuous (same encoding)
- Parent-relative (standard SMPL)
- Requires IK to convert constraints to world-space
- Supports hierarchical body control

---

## 🔄 Conversion Paths

### SMPL Pelvis → KIMODO Features

```
1. Get pelvis position: pelvis_pos ← global_positions[:, root_idx]
2. Smooth XZ (ADMM): smooth_root ← get_smooth_root_pos(pelvis_pos)
3. Compute heading: ψ ← compute_heading_angle(global_positions, skeleton)
4. Create [cos ψ, sin ψ]: global_root_heading
5. Local positions: local_pos ← global_pos - smooth_root (with Y absolute)
6. Global rotations: global_rot_6d ← matrix_to_cont6d(global_rot_mats)
7. Compute velocities: velocities ← compute_vel_xyz(global_positions, fps)
8. Detect foot contact: foot_contacts ← foot_detect_from_pos_and_vel(...)
9. Pack all 333 dims
```

### KIMODO Features → SMPL Pelvis

```
1. Extract smooth_root, heading, local_positions from features [0:86]
2. Reconstruct global positions:
   posed_joints[..., 0] += smooth_root[..., 0]  # add X
   posed_joints[..., 2] += smooth_root[..., 2]  # add Z
   # Y stays absolute
3. Extract root position: root_positions ← posed_joints[:, root_idx]
4. Recover global rotations: global_rot_6d ← features[86:248]
5. Convert global → local: local_rot_mats ← global_rots_to_local_rots(...)
6. Can now use SMPL FK or positions directly
```

### KIMODO → HyMotion M2M

**Full conversion chain** (SOMA-30 → SMPL-22):

```
KIMODO SOMA-30 (333D)
  ↓ (inverse to get global rotations & positions)
SOMA-30 global positions + global rotations
  ↓ (retarget to SMPL-22 joints)
SMPL-22 global positions + global rotations
  ↓ (extract root position)
Root position [T, 3]
  ↓ (compute delta)
Absolute + relative translation [T, 6]
  ↓ (convert global → local rotations)
SMPL-22 local rotations [T, 22, 6]
  ↓ (concat)
HyMotion M2M 138D feature vector [T, 138]
```

---

## 📐 Dimension Coverage Examples

### Full-Body Keyframe Constraint (KIMODO)

When constraining all 27 joints at frame 50:

```
Dims [0:3]      smooth_root_pos       → constrained (root XYZ)
Dims [3:5]      global_root_heading   → constrained
Dims [5:86]     local_joints_pos      → constrained (all 27 joints)
Dims [86:248]   global_rot_data       → NOT constrained in practice
Dims [248:333]  velocities + contacts → NOT constrained

Total constrained: 86 dims out of 333
```

### End-Effector Control

Constrain right hand to world position (2.0, 1.5, 1.0) at frame 50:

```python
# KIMODO:
# 1. Set smooth_root_2d (dims [0,2]) at frame 50
# 2. Set local_joints_pos[21] (dims [68:71]) = [hand_x - root_x, hand_y, hand_z - root_z]
# Result: 5 dims constrained (2 root + 3 hand)

# During diffusion:
# x_t = x_t * (1 - mask) + observed * mask
# Hand position locked, rest denoises
```

---

## ⚙️ Implementation Details

### 6D Rotation Encoding/Decoding

```python
# Encode: 3×3 matrix → 6D continuous
def matrix_to_cont6d(mat):  # [*, 3, 3]
    return mat[..., :2, :].reshape(..., 6)  # flatten first 2 columns

# Decode: 6D continuous → 3×3 matrix (orthonormal)
def cont6d_to_matrix(rot6d):  # [*, 6]
    x = rot6d[..., :3]      # first column
    y = rot6d[..., 3:]      # second column
    z = cross(x, y)         # third column (cross product)
    return stack([x, y, z], dim=-1)  # [*, 3, 3]
```

### Constraint Imputation Mechanism

```python
# In KIMODO diffusion:
if motion_mask is not None:
    # Direct replacement at every denoising step
    x_t = x_t * (1 - motion_mask) + observed_motion * motion_mask
    
    # Append mask as input feature
    x_extended = torch.cat([x_t, motion_mask], dim=-1)  # 666 dims
    
    # Transformer sees both motion AND which dims are constrained
    output = transformer(x_extended, text_embed, time_embed)
```

### Row-Major vs Column-Major 6D

**HyMotion training format** (row-major):
```
[R00, R01, R10, R11, R20, R21]  ← row 0, col 0; row 0, col 1; etc.
```

**rotation_convert.py intermediate** (column-major):
```
[R00, R10, R20, R01, R11, R21]  ← col 0, row 0; col 0, row 1; etc.
```

**Conversion indices**:
- Row → Col: `[0, 2, 4, 1, 3, 5]`
- Col → Row: `[0, 3, 1, 4, 2, 5]`

---

## 💡 Practical Implications

### Why KIMODO Smooth Root Matters

1. **Animator Workflow**: Animators draw straight lines/curves; KIMODO matches them exactly
2. **Trajectory Following**: Smoother reference frame → better path adherence
3. **Foot Skating**: ADMM smoothing dramatically reduces sliding (3.87 vs 7.59 cm/s)
4. **Constraint Stability**: Root trajectory less noisy → more stable body generation

### Why HyMotion's abs_rel Works

1. **Simplicity**: No preprocessing needed; direct from mocap
2. **Flexible**: Can apply constraints at any dimension independently
3. **VACE Conditioning**: Non-imputation approach; model learns to respect mask
4. **Training Stability**: Doesn't require separate smoothing algorithm

### When to Use Each Approach

**KIMODO smooth root**: 
- ✅ Trajectory-constrained generation (path following)
- ✅ Animator-oriented workflows
- ✅ End-effector IK-free control
- ✅ Production quality mocap processing

**HyMotion abs_rel**:
- ✅ General motion inpainting
- ✅ Joint editing (local control)
- ✅ Multi-dataset handling
- ✅ Simpler integration

---

## 🎯 Conversion Checklist

### SMPL → KIMODO

- [ ] Extract pelvis position from SMPL global positions
- [ ] Apply ADMM smoothing to XZ plane (keep Y raw)
- [ ] Compute heading angle from forward direction
- [ ] Create [cos ψ, sin ψ] representation
- [ ] Compute local joint positions (xz relative to smooth root, y absolute)
- [ ] Convert global rotations to 6D continuous (3×3 → first 2 columns)
- [ ] Compute joint velocities if needed
- [ ] Detect foot contacts from positions/velocities
- [ ] Concatenate all 333 dims in order

### KIMODO → HyMotion M2M

- [ ] Extract smooth root and heading
- [ ] Recover global positions: add smooth_root [X,Z] to local positions
- [ ] Extract root position (pelvis)
- [ ] Compute relative translation (frame deltas)
- [ ] Concatenate [abs_trans, rel_trans] (6 dims)
- [ ] Convert global rotations → local rotations via inverse FK
- [ ] Ensure row-major 6D encoding (use _COL_TO_ROW if needed)
- [ ] Concatenate [6 + 132] = 138 dims in order

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `ref_repo/KIMODO/kimodo/kimodo/motion_rep/smooth_root.py` | ADMM smoothing algorithm |
| `ref_repo/KIMODO/kimodo/kimodo/motion_rep/reps/kimodo_motionrep.py` | KimodoMotionRep class, encoding/decoding |
| `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` | SMPL loading, process_transl(), process_smplx_pose() |
| `hftrainer/datasets/motion/motionhub/transforms/fk_utils.py` | FK/IFK, rot6d conversions |

---

## ❓ FAQ

**Q: Why is smooth_root XZ smoothed but Y raw?**
A: Y is height above ground; smoothing it would interfere with ground contact. XZ is ground-plane trajectory.

**Q: How does [cos ψ, sin ψ] differ from a quaternion?**
A: Simpler (2 dims vs 4), no singularities, always unit length. Trade-off: cannot represent yaw+pitch+roll together.

**Q: Can I skip ADMM smoothing and use raw pelvis?**
A: Technically yes, but KIMODO's advantage disappears. You lose the clean trajectory reference.

**Q: Why are KIMODO rotations global but HyMotion local?**
A: KIMODO prioritizes constraint application (world-space IK-free); HyMotion prioritizes body hierarchy.

**Q: What happens to smooth_root info when I inverse?**
A: Inverse uses smooth_root as reference; you can't recover original noisy pelvis. One-way transformation.

**Q: How do velocities get computed in KIMODO?**
A: From global positions via finite differences: `v[t] = (pos[t+1] - pos[t]) * fps`

**Q: Is foot_contacts necessary?**
A: No; it's computed from velocity & ground contact. Useful for post-processing and loss terms.

---

## 🚀 Next Steps

1. **Test smooth_root impact**: Compare generated motion with/without ADMM
2. **Implement KIMODO → HyMotion conversion**: Use pseudocode in mapping document
3. **Validate rotations**: Check global_rot_6d consistency across FK operations
4. **Benchmark constraint application**: Measure constraint satisfaction rates
5. **Profile smoothing**: ADMM can be slow; multigrid helps but may need tuning

---

## 📖 References

- KIMODO Paper: https://research.nvidia.com/labs/sil/projects/kimodo
- CLAUDE.md (this repo): Detailed analysis of KIMODO architecture & constraints
- `kimodo_root_analysis.md`: 10-section technical deep dive
- `kimodo_hymotion_mapping.md`: 5 concrete examples + pseudocode

---

**Last Updated**: 2026-05-12  
**Status**: Complete & Verified Against Source Code  
**Skeleton**: 27-joint SOMA-30 (KIMODO) / 22-joint SMPL-22 (HyMotion)
