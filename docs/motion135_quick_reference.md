# motion_135 → SMPL Joints: Quick Reference

## The Problem
Convert `motion_135 (T, 135)` → SMPL joint positions `(T, 22, 3)`

## The 5-Step Solution

| Step | Input | Operation | Output |
|------|-------|-----------|--------|
| 1 | `(T, 135)` | Extract: first 3 dims = transl, rest = rot6d | `transl: (T,3)`, `rot6d: (T,22,6)` |
| 2 | `rot6d: (T,22,6)` | **Reorder [0,2,4,1,3,5]** then Gram-Schmidt | `rotmat: (T,22,3,3)` |
| 3 | `rotmat: (T,22,3,3)` | scipy `Rotation.from_matrix()` | `axis_angle: (T,22,3)` |
| 4 | `axis_angle: (T,22,3)` | Split: joint[0] = root, joints[1:] = body | `root_orient: (T,3)`, `pose_body: (T,63)` |
| 5 | All above | SMPL-X FK | **`joints: (T,22,3)`** ✓ |

## Critical: The Reordering Trick

**motion_135 uses ROW-MAJOR**: `[R00, R01, R10, R11, R20, R21]`
**Gram-Schmidt needs COLUMN-MAJOR**: `[R00, R10, R20, R01, R11, R21]`

```python
# DO THIS:
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]  # Reorder indices
# THEN Gram-Schmidt
```

**If you skip this: matrices have norm 2.5+ and axis-angle > 10 rad** ❌

## Code Templates

### Template 1: Standalone Script (No GPU Needed)
```bash
python scripts/embodied/motion135_to_smplx.py \
    input_motion_135.npz \
    output_smplx.npz \
    --fps 30
```

### Template 2: Convert + FK in Python
```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from hftrainer.models.motion.components.body_models.smplx_lite import SmplxLite

# Load motion_135
motion_135 = np.load('motion_135.npz')['motion_135']  # (T, 135)

# Extract and convert
transl = motion_135[:, :3]
rot6d = motion_135[:, 3:].reshape(-1, 22, 6)

# Gram-Schmidt (with reordering!)
rot6d_cm = rot6d[..., [0, 2, 4, 1, 3, 5]]  # ← KEY STEP
a1, a2 = rot6d_cm[..., :3], rot6d_cm[..., 3:]
b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
b2 = (a2 - np.sum(b1*a2, -1, keepdims=True)*b1) / (np.linalg.norm(...) + 1e-8)
b3 = np.cross(b1, b2)
rotmat = np.stack([b1, b2, b3], -1)  # (T, 22, 3, 3)

# Convert to axis-angle
rot = R.from_matrix(rotmat.reshape(-1, 3, 3))
axis_angle = rot.as_rotvec().reshape(-1, 22, 3)

# Split root/body
root_orient = axis_angle[:, 0]
body_pose = axis_angle[:, 1:].reshape(-1, 63)

# FK
smplx = SmplxLite("checkpoints/smpl_models/smplx/")
joints, _, _ = smplx.fk(
    torch.tensor(transl[None]).float(),
    torch.tensor(root_orient[None]).float(),
    torch.tensor(body_pose[None]).float()
)
joint_positions = joints[0].numpy()  # (T, 22, 3) ✓
```

## What Can Go Wrong

| Symptom | Cause | Fix |
|---------|-------|-----|
| Rotation matrix norm > 2.0 | Missing reorder `[0,2,4,1,3,5]` | Add it before Gram-Schmidt |
| Axis-angle values > 10 rad | Non-orthonormal matrix from Gram-Schmidt | Check your reorder |
| Output shape wrong | Reshaping error on rot6d | Should be `(T, 22, 6)` not `(T, 132)` |
| Joint positions y > 100m | Translation not extracted | Make sure `transl = motion[:, :3]` |
| NaN in output | Division by ~zero in normalization | Add epsilon: `1e-8` |

## File Locations

| What | Path |
|------|------|
| Standalone script | `scripts/embodied/motion135_to_smplx.py` |
| SmplxLite class | `hftrainer/models/motion/components/body_models/smplx_lite.py` |
| Rotation utils | `hftrainer/models/motion/components/utils/geometry/rotation_convert.py` |
| FK utils | `hftrainer/models/motion/components/utils/geometry/matrix.py` |
| SMPL models | `checkpoints/smpl_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.npz` |

## Expected Output Ranges

After FK, joint positions should satisfy:
- **Pelvis Y**: 0.9 to 1.1 m (height above ground)
- **Pelvis XZ**: -5 to +5 m (depends on motion)
- **Limb joint distances**: 0.2 to 0.5 m between parent-child (anatomically consistent)
- **No NaN or Inf values**

## Performance

- Conversion: ~10-50ms per second of motion (CPU, 30fps)
- Memory: ~1MB per 100 frames (float32)
- Can batch-process multiple motions on GPU

