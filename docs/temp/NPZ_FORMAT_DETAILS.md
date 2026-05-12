# NPZ Output Format: Complete Details

## Overview

T2M 1.0 evaluation generates `.npz` files containing motion data in multiple representations. Currently outputs **3 keys** with motion information split between different representations.

---

## Current NPZ Keys (3 keys)

### Key 1: `motion_135` ✓ (PRESENT)

- **Shape**: `(T, 135)` where T = frame count
- **Dtype**: `float32`
- **Range**: Denormalized (from checkpoint's mean/std)
- **Content**:
  ```
  [0:3]       → transl (3 dims) — root translation
  [3:135]     → rot6d (132 dims) — 22 joints × 6-dim rotations
  ```

**Example (T=60 frames)**:
```python
>>> data['motion_135'].shape
(60, 135)
>>> data['motion_135'][0, :3]  # First frame root translation
array([ 0.00936735,  1.1493286 ,  0.00434166], dtype=float32)
>>> data['motion_135'][0, 3:9]  # First 2 rotation dims (joint 0)
array([ 0.9851961, -0.00817597,  0.01952835,  0.9948842,  0.14690958, -0.08811654], dtype=float32)
```

**Note**: This is the **main motion representation** used for all evaluation metrics.

---

### Key 2: `positions` ✓ (PRESENT)

- **Shape**: `(T, 22, 3)` where T = frame count, 22 = SMPL joints
- **Dtype**: `float32`
- **Range**: World-space coordinates (meters)
- **Content**: 3D position of each joint in world coordinates

**Joint Order** (SMPL skeleton, 22 joints):
```
0:  pelvis (root)
1-3:  spine (lower, mid, upper)
4-5:  shoulders (left, right)
6-7:  elbows (left, right)
8-9:  wrists (left, right)
10-11: index fingers (left, right)
12:   neck
13-14: hips (left, right)
15-16: knees (left, right)
17-18: ankles (left, right)
19-20: big toes (left, right)
21:   left toes
```

**Example (T=60 frames)**:
```python
>>> data['positions'].shape
(60, 22, 3)
>>> data['positions'][0, 0, :]  # Pelvis position, frame 0
array([0.00757229, 0.9259952 , 0.03256079], dtype=float32)
>>> data['positions'][0, 1:3, :]  # Spine positions, frame 0
array([[ 0.07793683,  0.83572185,  0.04414062],
       [-0.05810372,  0.8341285 ,  0.02629144]], dtype=float32)
```

**Derivation**: Computed via Forward Kinematics from `motion_135`:
```python
# In eval script (line 357-359):
motion135_to_fk(output_135_t, bone_offsets, rotation_space=rotation_space)
# Output: (T, 22, 3) world positions
```

**Bone Offsets File**: `data/hymotion_m2m_data/bone_offsets_22.pt`

---

### Key 3: `translation` ✓ (PRESENT)

- **Shape**: `(T, 3)` 
- **Dtype**: `float32`
- **Range**: World-space, denormalized
- **Content**: Root translation (redundant with `motion_135[:, :3]`)

**Example**:
```python
>>> data['translation'].shape
(60, 3)
>>> data['translation'][0, :]  # Frame 0
array([0.00936735, 1.1493286 , 0.00434166], dtype=float32)

# Verify it matches motion_135
>>> np.allclose(data['translation'], data['motion_135'][:, :3])
True
```

**Purpose**: Convenience key for scripts that specifically need root translation without loading the full 135-dim array.

---

## Expected NPZ Keys for 201-dim Support

### Planned Addition: `motion_201` ⚠️ (MISSING)

To fully support 201-dim motion, the NPZ should include:

- **Shape**: `(T, 201)`
- **Content**:
  ```
  [0:3]        → transl (3 dims)
  [3:135]      → rot6d (132 dims)
  [135:201]    → positions_flat (66 dims) — 22 joints × 3 dims flattened
  ```

**Relationship**:
```python
motion_201 = np.concatenate([
    motion_135,
    positions.reshape(T, -1)  # (T, 22, 3) → (T, 66)
], axis=-1)

# Verify shape
assert motion_201.shape == (T, 201)
```

**Why it's missing**: The checkpoint is 201-dim, but the evaluation NPZ construction only saves:
1. `motion_135` (from denormalized network output)
2. `positions` (from FK inverse computation)
3. `translation` (redundant)

To get true 201-dim representation, need either:
- **Option A**: Extend the network to predict the position channel directly
- **Option B**: Save concatenated `motion_201` in the eval script

---

## Metric Derivations from NPZ

### From `motion_135`:

**Jitter (line 348)**:
```python
jitter_135 = compute_jitter_135(output_135)
# Compute: velocity[t] = motion[t+1] - motion[t]
#          jitter = mean(|velocity[t+1] - velocity[t]|)
```

**Rot6d Norms (line 332-340)**:
```python
rot6d_part = output_135[:, 3:135].reshape(T, 22, 6)
rot6d_norms = np.linalg.norm(rot6d_part, axis=-1)
# Per-frame norm of each 6D rotation vector
# Good value: ~1.414 (length of unit quaternion in 6D)
```

**Translation Ranges (line 334-343)**:
```python
transl = output_135[:, :3]
transl_range_x = transl[:, 0].max() - transl[:, 0].min()
transl_range_y = transl[:, 1].max() - transl[:, 1].min()
transl_range_z = transl[:, 2].max() - transl[:, 2].min()
```

### From `positions`:

**Velocity Metrics (line 379-386)**:
```python
joint_vel = np.diff(pos_np, axis=0) * 30.0  # 30 FPS
joint_acc = np.diff(joint_vel, axis=0) * 30.0
vel_mag = np.linalg.norm(joint_vel, axis=-1)
avg_velocity = vel_mag.mean()
max_velocity = vel_mag.max()
```

**Foot Ground Contact (line 364)**:
```python
foot_metrics = compute_foot_ground_metrics(pos_np, fps=30.0)
# Checks: contact penetration, sliding, etc.
```

**Bone Length Consistency (line 362-363)**:
```python
bl = compute_bone_length_cv(pos_np)
# CV = std / mean of bone lengths
# Should be ~0 for fixed skeleton
```

**Self-Penetration (line 390-400)**:
```python
spine_center = (pos_np[:, 3, :] + pos_np[:, 6, :] + pos_np[:, 9, :]) / 3
l_wrist_dist = np.linalg.norm(pos_np[:, 20, :] - spine_center, axis=-1)
r_wrist_dist = np.linalg.norm(pos_np[:, 21, :] - spine_center, axis=-1)
arm_penetration_ratio = ((l_wrist_dist < torso_radius) | (r_wrist_dist < torso_radius)).mean()
```

---

## File Size Expectations

### Typical Motion Sequence (360 frames = 12 seconds @ 30 FPS)

| Component | Shape | Bytes | Notes |
|-----------|-------|-------|-------|
| `motion_135` | (360, 135) | 194.4K | 360 × 135 × 4 bytes |
| `positions` | (360, 22, 3) | 95K | 360 × 22 × 3 × 4 bytes |
| `translation` | (360, 3) | 4.3K | 360 × 3 × 4 bytes |
| **Total (uncompressed)** | — | 293.7K | — |
| **NPZ (compressed)** | — | 50-100K | Depends on compressibility |

**Actual Example** (60 frames):
```bash
ls -lh work_dirs/m2m_v2_t2m_eval/caption_global/npz/00000001.npz
-rw-r--r-- 1 root root 45K Apr 15 15:49 00000001.npz  # 60 frames
```

---

## Loading & Processing NPZ

### Basic Loading

```python
import numpy as np

data = np.load('motion.npz')
motion_135 = data['motion_135']    # (T, 135)
positions = data['positions']      # (T, 22, 3)
transl = data['translation']       # (T, 3)
```

### Reconstruct Full Motion Representation

```python
T = motion_135.shape[0]

# Current 135-dim
transl = motion_135[:, :3]         # (T, 3)
rot6d = motion_135[:, 3:135]       # (T, 132)

# From FK computation
pos_flat = positions.reshape(T, -1)  # (T, 66)

# Combine to 201-dim (if implemented)
motion_201 = np.concatenate([motion_135, pos_flat], axis=-1)
assert motion_201.shape == (T, 201)
```

### Convert to SMPL Format

```python
from models.motion.components.utils.geometry.rotation_convert import rotation_6d_to_axis_angle
import torch

# 6D rotation → axis-angle (SMPL standard)
rot6d_reshaped = rot6d.reshape(-1, 22, 6)  # (T, 22, 6)
aa = rotation_6d_to_axis_angle(torch.from_numpy(rot6d_reshaped).float())
poses_aa = aa.numpy().reshape(T, 66)  # (T, 66) axis-angle format

# SMPL-compatible format
smpl_poses = poses_aa  # (T, 66) — SMPL joint angles
smpl_trans = transl    # (T, 3) — root translation
```

---

## Summary Table

| Aspect | Current | Planned (201-dim) |
|--------|---------|-------------------|
| **NPZ Keys** | 3 (motion_135, positions, translation) | 4 (add motion_201) |
| **motion_135** | ✓ Present | ✓ Keep |
| **positions** | ✓ Present (FK-derived) | ✓ Keep |
| **translation** | ✓ Present (redundant) | ✓ Keep |
| **motion_201** | ✗ Missing | ⚠️ Needed |
| **Position Channel** | Derived via FK | Train network to predict |
| **Data Pipeline** | 135-dim (LoadSmplx55) | Needs extension to 201-dim |

---

## Code References

### Where NPZ is Generated
- **File**: `scripts/eval/eval_m2m_v2_t2m.py`
- **Lines**: 371-377
- **Function**: `_run_one_cfg()` nested in `run_model_on_gpu()`

```python
npz_path = os.path.join(npz_dir, f'{prompt["id"]}.npz')
np.savez_compressed(
    npz_path,
    motion_135=output_135,
    positions=pos_np,
    translation=transl,
)
```

### Where FK Computes Positions
- **File**: `hftrainer/pipelines/motion/differentiable_fk.py`
- **Function**: `motion135_to_fk()`
- **Inputs**: motion_135 + bone_offsets
- **Outputs**: world_pos (T, 22, 3)

---

## Verification Checklist

✅ **motion_135 loaded & parsed**: (T, 135) with consistent rot6d norms
✅ **positions loaded & parsed**: (T, 22, 3) with sensible joint coordinates
✅ **translation matches**: motion_135[:, :3] == translation
✅ **FK consistency**: Compare positions vs. FK-computed from rot6d
✅ **Metrics computed**: All 20+ metrics in result.json

⚠️ **motion_201 not present**: Expected in future when data pipeline extended

