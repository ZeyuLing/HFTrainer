# SMPL-X Pose Processing Documentation

Complete analysis and reference guides for understanding the `process_smplx_pose` function and related rotation conversions in the HyMotion dataset pipeline.

## 📁 Documentation Files

This package contains 3 comprehensive guides:

### 1. **QUICK_REFERENCE.md** ⚡ (Start Here!)
- **Best for**: Quick lookup, TL;DR answers
- **Length**: ~180 lines
- **Contains**:
  - 4 key questions answered directly
  - Quick lookup tables
  - Formula cheat sheets
  - Common issues & fixes
  
**Read this if you need answers fast.**

### 2. **SMPLX_POSE_ANALYSIS.md** 📚 (Deep Dive)
- **Best for**: Complete understanding
- **Length**: ~360 lines
- **Contains**:
  - Detailed explanation of joint selection
  - Step-by-step rotation conversion pipeline
  - Complete code listings with annotations
  - Reconstruction formulas
  - Implementation details & edge cases
  - Full reference tables

**Read this for comprehensive understanding.**

### 3. **VISUAL_GUIDE.md** 🎨 (Visual Learning)
- **Best for**: Visual understanding, data flow
- **Length**: ~370 lines
- **Contains**:
  - ASCII diagrams and flowcharts
  - Joint hierarchy visualization
  - Rotation representation space
  - 6D rearrangement explanation (visual)
  - Data shape evolution
  - Decision trees & function traces

**Read this if you learn better visually.**

---

## 🎯 Quick Navigation

### I want to know...

**"How does SMPL-22 selection work?"**
→ QUICK_REFERENCE.md § 1 OR SMPLX_POSE_ANALYSIS.md § 1

**"How does 6D rotation work?"**
→ QUICK_REFERENCE.md § 2 OR VISUAL_GUIDE.md § 3

**"What's the output shape for X config?"**
→ QUICK_REFERENCE.md § 4 OR SMPLX_POSE_ANALYSIS.md § 4

**"Can I see the full code?"**
→ SMPLX_POSE_ANALYSIS.md § 5

**"What's the [0,3,1,4,2,5] permutation?"**
→ VISUAL_GUIDE.md § 3 (best visual explanation)

**"Show me data flowing through the system"**
→ VISUAL_GUIDE.md § 4 (shape evolution) OR § 7 (function trace)

**"I'm getting wrong results, what could be wrong?"**
→ VISUAL_GUIDE.md § 9 (edge cases) OR QUICK_REFERENCE.md (common issues)

---

## 🔑 Key Findings at a Glance

### 1️⃣ Joint Selection
- **SMPL-22**: Simply takes first 22 joints from SMPL-X via `np.arange(22)`
- **No remapping**: Just sequential indexing `aa[:, 0:22, :]`
- **Result**: 22 core body joints (pelvis, legs, spine, arms, head)

### 2️⃣ 6D Conversion
```
axis_angle (3D)
    ↓ [via SciPy Rodrigues]
rotation_matrix (3×3)
    ↓ [extract first 2 columns]
6D column-major: [R00,R10,R20, R01,R11,R21]
    ↓ [rearrange indices [0,3,1,4,2,5]]
6D row-major: [R00,R01, R10,R11, R20,R21] ← HyMotion convention
```

### 3️⃣ Output Shape
- **Formula**: `[T, num_joints × dims_per_rotation]`
- **Example**: SMPL-22 + 6D → `[T, 132]` (22 × 6)
- **Dimension order**: Sequential by joint, then by dimension within joint

### 4️⃣ The Critical Rearrangement
- **Source**: `matrix_to_rotation_6d()` outputs column-major
- **Target**: HyMotion wants row-major ordering
- **Solution**: Permutation `[0, 3, 1, 4, 2, 5]` at line 93
- **Only for 6D**: Other formats (quaternion, euler) don't get rearranged

---

## 📍 Source Code Locations

| Component | File | Lines |
|-----------|------|-------|
| **Main function** | `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` | 16-104 |
| **Axis-angle ↔ Matrix** | `rotation_convert.py` | 127-156, 158-195 |
| **Matrix → 6D** | `rotation_convert.py` | 455-460 |
| **6D → Matrix** | `rotation_convert.py` | 434-452 |
| **Axis-angle → 6D** | `rotation_convert.py` | 476-477 |
| **Full rotation convert** | `rotation_convert.py` | 524-601 |

---

## 💡 Usage Examples

### Example 1: Load and convert SMPL-X to SMPL-22 with 6D

```python
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose
import numpy as np

# Load SMPL-X poses: [100 frames, 165 dims = 55 joints × 3]
poses_55 = np.load("motion.npz")["poses"]  # Shape: [100, 165]

# Convert to SMPL-22 with 6D rotation
result = process_smplx_pose(poses_55, rot_type="rotation_6d", out_type="smpl_22")
# Output shape: [100, 132] (22 joints × 6 dims)
```

### Example 2: Convert to different formats

```python
# Same input, different outputs:

# Axis-angle: [100, 66] (22 × 3)
aa = process_smplx_pose(poses_55, rot_type="axis_angle", out_type="smpl_22")

# Quaternion: [100, 88] (22 × 4)
quat = process_smplx_pose(poses_55, rot_type="quaternion", out_type="smpl_22")

# Euler: [100, 66] (22 × 3)
euler = process_smplx_pose(poses_55, rot_type="euler", out_type="smpl_22")
```

### Example 3: Reconstruct rotation matrix from 6D

```python
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_matrix
)

# Take output from Example 1
poses_6d = result  # [100, 132]

# Reshape to [100, 22, 6] for easier handling
poses_6d_reshaped = poses_6d.reshape(100, 22, 6)

# Convert first joint's 6D back to rotation matrix
joint_0_6d = poses_6d_reshaped[0, 0, :]  # [6]
joint_0_mat = rotation_6d_to_matrix(joint_0_6d)  # [3, 3]
```

---

## 🧪 Testing Your Understanding

### Question 1: Output shape
**Q**: If I have `[50, 165]` SMPL-X poses and want SMPL-H with quaternion rotation, what's the output shape?

**A**: `[50, 208]` 
- SMPL-H = 52 joints
- Quaternion = 4 dims per joint
- 52 × 4 = 208 ✓

### Question 2: Dimension mapping
**Q**: In the output `[100, 132]` (SMPL-22 + 6D), dimensions [6:12] represent what?

**A**: Joint 1's 6D representation in row-major order:
- `[6]` = R00, `[7]` = R01, `[8]` = R10, `[9]` = R11, `[10]` = R20, `[11]` = R21 ✓

### Question 3: Rearrangement
**Q**: Why only 6D gets rearranged but not quaternion or euler?

**A**: Because:
- 6D is derived from first 2 matrix columns (column-major)
- Other formats don't depend on matrix column ordering
- HyMotion needs row-major for their specific use case ✓

---

## 🐛 Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Output has NaN | Check input axis-angles for validity (too large angles?) |
| Wrong shape | Verify input shape and make sure you're not mixing [T,165] vs [T,55,3] |
| 6D doesn't reconstruct properly | Ensure you're using the row-major version (after rearrangement) |
| 0s in pose data | Normal if input is SMPL-H (gets padded for jaw/eyes) |
| Dimension mismatch | Use formula: output_dim = num_joints × dims_per_rotation |

---

## 📊 File Statistics

- **QUICK_REFERENCE.md**: ~4.8 KB, 183 lines (quick answers)
- **SMPLX_POSE_ANALYSIS.md**: ~11 KB, 358 lines (detailed analysis)
- **VISUAL_GUIDE.md**: ~22 KB, 374 lines (visual explanations)
- **Total**: ~915 lines of documentation

---

## 🚀 Next Steps

1. **Start here**: Read QUICK_REFERENCE.md for a 5-minute overview
2. **Deep dive**: Read SMPLX_POSE_ANALYSIS.md for complete understanding
3. **Visualize**: Read VISUAL_GUIDE.md for data flow and diagrams
4. **Implement**: Use the examples and reference tables to build your solution
5. **Validate**: Test against the "Testing Your Understanding" questions

---

## 📝 Notes

- All code examples assume working directory: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`
- Documentation accurate as of: **May 20, 2026**
- Source files:
  - `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py`
  - `hftrainer/models/motion/components/utils/geometry/rotation_convert.py`

---

## 🔗 Related Functions

- `LoadSmplx55` - Main class that uses `process_smplx_pose`
- `axis_angle_to_rotation_6d()` - Direct axis-angle to 6D conversion
- `rotation_6d_to_matrix()` - Reverse conversion (6D → rotation matrix)
- `axis_angle_to_quaternion()` - Axis-angle to quaternion
- `axis_angle_to_euler()` - Axis-angle to Euler angles
- `process_transl()` - Translation processing (sibling to pose processing)

---

## 📞 Quick Troubleshooting

**"My output doesn't match expected shape"**
- Check your input dimensionality (is it [T,165] or [T,55,3]?)
- Verify joint count: SMPL-22=22, SMPL-H=52, SMPL-X=55
- Calculate: output_dim = joints × dims_per_rot

**"6D values look wrong"**
- Verify you're using **row-major**, not column-major
- Check the permutation was applied: `[0, 3, 1, 4, 2, 5]`
- If coming from file, check it was processed with the right rot_type

**"Reconstruction fails"**
- Use `rotation_6d_to_matrix()` to convert 6D back to matrix
- Ensure 6D is in row-major format (post-rearrangement)
- Check for numerical precision issues

---

Generated: May 20, 2026
Source: `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` + `rotation_convert.py`
