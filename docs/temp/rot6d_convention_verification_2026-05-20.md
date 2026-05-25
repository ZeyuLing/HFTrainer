# Rot6D Convention Investigation Summary

## Investigation Date
May 20, 2026 (continuation of investigation from previous conversation)

## Overview
This document summarizes the investigation into rot6d (6D rotation representation) conventions used throughout the HyMotion M2M model and related code.

## Key Finding: Two Coexisting Conventions

The codebase uses **two different rot6d conventions** that must be carefully managed:

### 1. Column-Major Convention
- **Used by**: `rotation_convert.py` mathematical functions
- **Format**: `[R00, R10, R20, R01, R11, R21]`
- **Meaning**: First 3 elements are the first column of rotation matrix, next 3 are the second column
- **Functions**:
  - `axis_angle_to_rotation_6d()` → outputs column-major
  - `rotation_6d_to_axis_angle()` → expects column-major input

### 2. Row-Major Convention (HyMotion/Training Data)
- **Used by**: Training data, model I/O, HyMotion checkpoints
- **Format**: `[R00, R01, R10, R11, R20, R21]`
- **Meaning**: Elements represent the first two columns row-by-row
  - Rows of first column: R00, R10, R20
  - Rows of second column: R01, R11, R21
- **Functions**:
  - `geometry.rot6d_to_rotation_matrix()` → expects row-major input
  - `geometry.rotation_matrix_to_rot6d()` → outputs row-major

## Data Flow: Encoding (Axis-Angle → Rot6D)

```
axis_angle (T, J, 3)
  ↓
load_smplx.py: process_smplx_pose()
  ↓ [Line 89: axis_angle_to_rotation_6d]
rotation_convert.axis_angle_to_rotation_6d()  → outputs COLUMN-MAJOR
  ↓ [Line 93: permutation [0,3,1,4,2,5]]
out[:, :, [0, 3, 1, 4, 2, 5]]  → ROW-MAJOR
  ↓
Training data: motion_135 (T, 135)
```

The permutation `[0, 3, 1, 4, 2, 5]` converts from column-major to row-major:
- Position 0 → 0 (R00)
- Position 1 → 2 (R10) 
- Position 2 → 4 (R20)
- Position 3 → 1 (R01)
- Position 4 → 3 (R11)
- Position 5 → 5 (R21)

## Data Flow: Decoding (Rot6D → Axis-Angle)

When converting back from row-major rot6d to axis-angle:

```
rot6d (row-major, from model output)
  ↓ [Permutation [0,2,4,1,3,5]]
[0,2,4,1,3,5]  → COLUMN-MAJOR (for rotation_convert)
  ↓
rotation_convert.rotation_6d_to_axis_angle()
  ↓
axis_angle
```

The inverse permutation `[0, 2, 4, 1, 3, 5]` converts from row-major to column-major:
- Position 0 → 0 (R00)
- Position 1 → 3 (R01)
- Position 2 → 1 (R10)
- Position 3 → 4 (R11)
- Position 4 → 2 (R20)
- Position 5 → 5 (R21)

## Files and Correct Handling

### ✅ load_smplx.py (Lines 88-94)
**Status**: CORRECTLY IMPLEMENTED
```python
out = axis_angle_to_rotation_6d(aa_flat).reshape(T, J, 6)
# outputs column-major: [R00,R10,R20, R01,R11,R21]
# HyMotion convention is row-major: [R00,R01, R10,R11, R20,R21]
out = out[:, :, [0, 3, 1, 4, 2, 5]]  # col_major → row_major
```

### ✅ repair_and_evaluate.py (Repair Section)
**Status**: CORRECTLY IMPLEMENTED
```python
rot6d = motion[:, 3:135].reshape(T * 22, 6)  # row-major from training data
rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]  # row-major → column-major
axis_angle = rotation_6d_to_axis_angle(rot6d_colmajor)
```

### ✅ run_prism_infer_lowmem.py (Inference Section)
**Status**: CORRECTLY IMPLEMENTED
```python
pred_poses = pred_poses[..., [0, 2, 4, 1, 3, 5]]  # row-major → column-major
pred_poses = rotation_6d_to_axis_angle(pred_poses)
```

### ✅ geometry.rot6d_to_rotation_matrix()
**Status**: CORRECTLY IMPLEMENTED (but documentation was unclear)
- Expects **row-major** format as input
- Uses Gram-Schmidt orthogonalization
- The function name is slightly misleading as it only exists in M2M codebase
- Updated docstrings now clarify the expected format

### ✅ geometry.rotation_matrix_to_rot6d()
**Status**: CORRECTLY IMPLEMENTED (documentation updated)
- Outputs **row-major** format
- Extracts the first two columns row-by-row
- Consistent with training data convention

## Code Audit Results

### Files Checked for Convention Consistency:
1. `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` ✅
2. `hftrainer/models/motion/components/motion_processor/smpl_processor.py` ✅
3. `hftrainer/models/motion/hymotion_m2m/bundle.py` ✅
4. `hftrainer/models/motion/hymotion_m2m/network/geometry.py` ✅
5. `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` ✅
6. `scripts/repair/repair_and_evaluate.py` ✅
7. `scripts/inference/run_prism_infer_lowmem.py` ✅

### Conventions:
- Training data uses **row-major** consistently
- Model input/output uses **row-major** consistently
- rotation_convert.py uses **column-major** as mathematical convention
- M2M geometry functions use **row-major** to match model I/O

## Documentation Updates Applied

### geometry.py Function Docstrings Enhanced:
1. `rot6d_to_rotation_matrix()`: 
   - Added clarity on ROW-MAJOR format expectation
   - Added reference to load_smplx.py
   - Added warning about column-major vs row-major distinction
   - Improved inline comments

2. `rotation_matrix_to_rot6d()`:
   - Clarified ROW-MAJOR output format
   - Added reference to `rot6d_to_rotation_matrix()` for details

## Key Rules for Future Development

### Rule 1: Encoding Path (Axis-Angle → Rot6D)
Always use `process_smplx_pose()` in `load_smplx.py`:
```python
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import process_smplx_pose
rot6d_rowmajor = process_smplx_pose(pose_55_axis_angle, rot_type="rotation_6d")
```
This function handles the permutation automatically.

### Rule 2: Decoding Path (Rot6D → Axis-Angle)
Apply permutation BEFORE calling rotation_convert functions:
```python
rot6d_rowmajor = ...  # from model or training data
rot6d_colmajor = rot6d_rowmajor[..., [0, 2, 4, 1, 3, 5]]  # row → col
axis_angle = rotation_6d_to_axis_angle(rot6d_colmajor)
```

### Rule 3: M2M-Specific FK
Use geometry functions directly (no permutation needed):
```python
rot6d_rowmajor = ...  # from model output
rotation_matrix = rot6d_to_rotation_matrix(rot6d_rowmajor)  # already expects row-major
```

### Rule 4: Never Mix
- ❌ Don't call `rotation_convert.rotation_6d_to_matrix()` with row-major data
- ❌ Don't call `geometry.rot6d_to_rotation_matrix()` with column-major data
- ✅ Use the appropriate function for the convention in use

## Historical Bug (Fixed 2026-03-23)

**Issue**: `axis_angle_to_rotation_6d()` outputs column-major, but training data expected row-major
**Fix**: Added permutation in `load_smplx.py` line 93
**Status**: ✅ RESOLVED and documented in CLAUDE.md §"2026-03-23: Rotation 6D convention mismatch"

## Verification Test Results

Created test case with 45-degree Y-axis rotation:
```
Input (axis-angle):     [0.0, π/4, 0.0]
Column-major rot6d:     [0.707, 0, -0.707, 0, 1, 0]
Row-major rot6d:        [0.707, 0, 0, 1, -0.707, 0]

Verification:
- geometry.rot6d_to_rotation_matrix(row-major) → ✅ CORRECT
- geometry.rot6d_to_rotation_matrix(col-major) → ❌ WRONG
- With permutation fix: row-major + [0,2,4,1,3,5] → ✅ CORRECT
```

## Conclusion

The HyMotion M2M codebase correctly handles both rot6d conventions with proper permutation applications at convention boundaries. The key insight is:

- **Column-major** is the mathematical convention (optimal for certain algorithms)
- **Row-major** is the HyMotion/training convention (used in all persistent data)
- **Permutation [0,3,1,4,2,5]** converts column-major → row-major (encoding)
- **Permutation [0,2,4,1,3,5]** converts row-major → column-major (decoding)

All observed code correctly implements these conversions. Updated documentation in `geometry.py` now provides clear guidance to prevent future confusion.
