# Rot6D Alignment Validation Tools

This directory contains tools to validate and debug the rot6d (6D rotation representation) alignment throughout the PRISM pipeline.

## Quick Start

```bash
# Run the basic test suite
python test_alignment.py --verbose

# Validate real motion data
python rot6d_validator.py --motion_npz /path/to/motion.npz --config /path/to/prism/config.py
```

## Files

- **`rot6d_validator.py`** — Comprehensive validation framework with:
  - `Rot6DValidator` class for orthonormality checking
  - `PrismPipelineValidator` class for end-to-end pipeline validation
  - Methods to verify normalization roundtrips, VAE input shapes, and rot6d convention preservation

- **`test_alignment.py`** — Executable test suite with:
  - Reordering indices correctness tests
  - Row-major rot6d orthonormality tests
  - Normalization/denormalization roundtrip verification
  - Motion shape after rearrange validation
  - Rot6D norm per-joint checks
  - Reordering consistency verification

## Key Concepts

### Rot6D Conventions

PRISM uses **row-major rot6d** format: `[R00, R01, R10, R11, R20, R21]`

This differs from column-major used by low-level rotation math: `[R00, R10, R20, R01, R11, R21]`

**Conversion:**
- Column-major → Row-major: `[0, 3, 1, 4, 2, 5]`
- Row-major → Column-major: `[0, 2, 4, 1, 3, 5]`

### Data Flow

```
SMPL axis-angle (T, 66)
    ↓
rot_convert() → column-major rot6d (T, 132)
    ↓
dataset loading: reorder [0,3,1,4,2,5] → row-major rot6d (T, 132)
    ↓
combine with translation (3) → motion_vec (T, 135)
    ↓
normalize per-dimension → normalized motion (T, 135)
    ↓
rearrange to (B, T, 22, 6) → VAE input
```

## Validation Checklist

Before training or inference, verify:

1. ✅ **Reordering applied per-joint**: Each of 22 joints reordered independently, not just first 6 dims
2. ✅ **Correct direction**: Using `[0,3,1,4,2,5]` for col→row, not `[0,2,4,1,3,5]`
3. ✅ **Normalization roundtrip**: `(motion - denormalize(normalize(motion))).max() < 1e-5`
4. ✅ **VAE input shape**: `(B, T, 22, 6)` with correct per-joint rot6d
5. ✅ **Orthonormality**: Each joint's 6D should form R@R^T≈I when reconstructed
6. ✅ **Per-dimension stats**: Mean/std correctly applied across all 135 dims

## Usage Examples

### Example 1: Quick Validation Test

```python
import torch
import numpy as np
from test_alignment import Rot6DAlignmentTests

tester = Rot6DAlignmentTests(verbose=True)

# Run core tests
tester.test_reordering_indices()
tester.test_row_major_rot6d_orthonormality()

# Load real data and test
motion_data = np.load('motion.npz')
motion = torch.from_numpy(motion_data['motion']).float()
tester.test_normalize_denormalize_roundtrip(motion.numpy())

print(f"\nPassed: {tester.tests_passed}, Failed: {tester.tests_failed}")
```

### Example 2: Pipeline Validation

```python
import torch
from rot6d_validator import PrismPipelineValidator

# Initialize validator
validator = PrismPipelineValidator(
    smpl_processor=your_smpl_processor,
    vae_config={"in_channels": 6, "out_channels": 6}
)

# Load motion data
motion = torch.randn(32, 100, 135)  # Batch of 32, 100 frames, 135 dims

# Run checks
is_valid, diag = validator.check_normalization_roundtrip(motion)
print(f"Normalization valid: {is_valid}, {diag}")

is_valid, diag = validator.check_vae_input_shape(motion)
print(f"VAE shape valid: {is_valid}, {diag}")

is_valid, diag = validator.check_rot6d_convention_preservation(motion)
print(f"Rot6D convention preserved: {is_valid}, {diag}")
```

### Example 3: Debugging Rot6D Issues

If you see:
- **Rot6D norms >> 1.0**: Likely wrong reordering direction
- **VAE output out of range [-10, 13]**: Check if reordering is being applied per-joint, not globally
- **NaN/Inf in training loss**: Verify orthonormality is maintained after reordering

Use the validator to check:
```python
# In bundle.encode_motion(), add:
is_valid, diag = validator.check_rot6d_convention_preservation(motion_normalized)
if not is_valid:
    print(f"WARNING: Rot6D convention mismatch detected!")
    print(f"Diagnostics: {diag}")
```

## Integration into CI/Testing

Add to your test pipeline:
```bash
# In tests/smoke/ or similar
python -m pytest scripts/debug/rot6d_validation/test_alignment.py -v
```

## Reference Documentation

See `/hftrainer/models/motion/CLAUDE.md` for:
- Complete rot6d convention history
- Bug debugging reference
- Per-dimension normalization details
- Related issues in M2M/T2M pipelines

## Common Mistakes

❌ **Mistake 1**: Only reordering first 6 dimensions
```python
motion_wrong = motion[:, [0,3,1,4,2,5]]  # Only affects dims 0-5!
```

✅ **Correct**: Reorder per-joint
```python
for j in range(22):
    start = 3 + j * 6
    motion[:, start:start+6] = motion[:, start:start+6][:, [0,3,1,4,2,5]]
```

❌ **Mistake 2**: Using column-major directly
```python
rot6d = rotation_convert.axis_angle_to_rotation_6d(pose)  # Column-major!
vae_input = rearrange(rot6d, ...)  # Wrong convention
```

✅ **Correct**: Apply reordering
```python
rot6d = rotation_convert.axis_angle_to_rotation_6d(pose)  # Column-major
rot6d = rot6d[:, [0,3,1,4,2,5]]  # Convert to row-major
vae_input = rearrange(rot6d, ...)  # Correct convention
```

## Debugging Workflow

1. **Suspect rot6d issue?** → Run `test_alignment.py --verbose`
2. **Specific motion file?** → Use `rot6d_validator.py` with `--motion_npz`
3. **Integration test?** → Add validation calls to bundle.encode_motion()
4. **CI integration?** → Add test_alignment.py to smoke test suite

## Authors & References

- Diagnostic work completed May 21, 2026
- Based on comprehensive analysis of PRISM/VERMO/Prism MCM pipelines
- See docs/temp/rot6d_convention_verification_2026-05-20.md for detailed investigation

