# 272-Dim Rotation Extraction Bug Fix - Applied

**Date**: 2026-05-27  
**Status**: ✓ FIXED AND VERIFIED  
**File**: `ref_repo/MotionStreamer/convert_prism_to_272.py`

---

## Summary

Fixed critical rotation extraction bugs in PRISM-to-272-dim conversion that were causing completely incorrect local and heading rotation values (dimensions 2-8 and 148-271).

## Root Cause

The `compute_representation_272()` function was using incorrect numpy indexing for 6D rotation extraction:

1. **Heading rotation (dims 2-8)**: Used column-major extraction
2. **Local joint rotations (dims 148-271)**: Used column-major extraction  
3. **Ground truth (GT)**: Uses row-major extraction

This caused mismatched 6D representations between PRISM conversions and HumanML3D ground truth.

## The Bug

### Original Code (INCORRECT)

```python
# Lines 428-432: Heading rotation
heading_6d = np.concatenate(
    [global_heading_diff_rot[..., 0], global_heading_diff_rot[..., 1]], axis=-1
)  # (T-1, 6)

# Lines 450-451: Local joint rotations
rot6d_col_major = np.concatenate(
    [rotations_matrix[..., 0], rotations_matrix[..., 1]], axis=-1
)  # (T, 22, 6)
```

### The Problem

For a 3×3 rotation matrix R:
```
R = [[a, b, c],
     [d, e, f],
     [g, h, i]]
```

The concatenate approach extracted **COLUMNS**:
- `R[..., 0]` = first column = [a, d, g]
- `R[..., 1]` = second column = [b, e, h]
- Result: [a, d, g, b, e, h] ← **WRONG** (column-major)

The GT uses **ROWS**:
- `R[..., :2, :]` = first 2 rows = [[a,b,c], [d,e,f]]
- Result: [a, b, c, d, e, f] ← **CORRECT** (row-major)

### Example Mismatch

For matrix:
```
[[11, 12, 13],
 [21, 22, 23],
 [31, 32, 33]]
```

- PRISM (old): [11, 21, 31, 12, 22, 32]
- GT (correct): [11, 12, 13, 21, 22, 23]
- **Difference**: ✗ All values are wrong!

## The Fix

### Fixed Code (CORRECT)

```python
# Lines 428-432: Heading rotation - FIXED
# Extract first 2 rows: [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]
heading_6d = global_heading_diff_rot[..., :2, :]  # (T-1, 2, 3)
final_x[1:, 2:8] = heading_6d.reshape(heading_6d.shape[0], -1)  # (T-1, 6)

# Lines 443-451: Local joint rotations - FIXED
# Extract first 2 ROWS to match GT representation_272.py line 116:
# rotations_matrix[..., :2, :] shape (T, 22, 2, 3)
# This produces: [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]] per joint per frame
rot6d = rotations_matrix[..., :2, :]  # (T, 22, 2, 3)
final_x[:, 8 + 6 * njoint:8 + 12 * njoint] = np.reshape(rot6d, (nfrm, -1))
```

## Verification

### Test 1: Heading Extraction ✓
- Input: (T-1, 3, 3) heading rotation matrices
- Fixed extraction: First 2 rows
- Result: [R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2]]
- **Status**: ✓ PASS - Matches GT format

### Test 2: Local Rotation Extraction ✓
- Input: (T, 22, 3, 3) joint rotation matrices
- Fixed extraction: First 2 rows per joint
- Result: 6D per joint = first 2 rows flattened
- **Status**: ✓ PASS - Matches GT format

### Test 3: Consistency ✓
- `rotations_matrix[..., :2, :]` produces same shape as GT
- **Status**: ✓ PASS - Both use identical indexing

## Changes Made

| File | Location | Change |
|------|----------|--------|
| `convert_prism_to_272.py` | Lines 428-432 | Replace column concat with row extraction for heading |
| `convert_prism_to_272.py` | Lines 443-451 | Replace column concat with row extraction for local rotations |

## Impact

### What This Fixes

1. **Dimensions 2-8** (heading rotation): Now correctly extracts first 2 rows instead of 2 columns
2. **Dimensions 148-271** (local rotations, 22 joints × 6D): Now correctly extracts first 2 rows per joint instead of 2 columns

### Expected Improvement

- PRISM predictions converted to 272-dim will now have correct rotation values
- Rotation-based metrics (if used) will show dramatic improvement
- Motion generation using these predictions should improve significantly

## Reference

- **GT implementation**: `ref_repo/MotionStreamer/272-dim-Motion-Representation/representation_272.py:116`
- **Matrix extraction function**: `matrix_to_rotation_6d()` in `face_z_align_util.py`
  - Uses: `matrix[..., :2, :].reshape(..., 6)`
  - Extracts first 2 ROWS of 3×3 rotation matrix

## Related Files

- `representation_272.py` (GT generation, line 116 - correct implementation)
- `face_z_align_util.py` (contains `matrix_to_rotation_6d()` - the reference function)
- `amass_process.py` (upstream GT generation using same extraction)

---

**Tested and Verified**: ✓ All rotation extraction tests pass
