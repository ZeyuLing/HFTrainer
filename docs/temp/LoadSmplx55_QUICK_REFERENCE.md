# LoadSmplx55 - Quick Reference Cheat Sheet

## 📍 Location & Import

```python
# File path
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/datasets/motion/motionhub/transforms/load_smplx.py

# Imports
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
```

---

## 🚀 Quick Start

```python
loader = LoadSmplx55(
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs',
    smpl_type='smpl_22'
)

results = {'motion_path': 'motion.npz'}
results = loader(results)

# Output: torch.Tensor [T, 135] (T frames, 135 dims)
print(results['motion'].shape)
```

---

## 📊 Output Dimensions

| Configuration | Dims | Formula |
|---------------|------|---------|
| rotation_6d + 22j + abs | 135 | 3 + 6×22 |
| quaternion + 22j + abs | 91 | 3 + 4×22 |
| axis_angle + 22j + abs | 69 | 3 + 3×22 |
| rotation_6d + 22j + abs_rel | 138 | 6 + 6×22 |

---

## 🎯 Common Configurations

### Training (with augmentation)
```python
dict(type='LoadSmplx55',
     key='motion',
     rot_type='rotation_6d',
     transl_type='abs',
     smpl_type='smpl_22',
     transl_aug_prob=0.5,
     transl_aug_yaw_deg=90.0)
```

### Validation (no augmentation)
```python
dict(type='LoadSmplx55',
     key='motion',
     rot_type='rotation_6d',
     transl_type='abs',
     smpl_type='smpl_22',
     transl_aug_prob=0.0)
```

### Multi-person
```python
results = {
    'motion_path': ['person1.npz', 'person2.npz']
}
# Output: [P, T, D] = [2, T, 135]
```

---

## 📁 NPZ Format

### Required Keys (Raw SMPL)
```python
{
    'poses': [T, 165],           # axis-angle
    'trans': [T, 3],             # translation
    'mocap_framerate': int,      # optional, default 30
}
```

### Optional Keys (Pre-computed)
```python
{
    'motion_135': [T, 135],      # [trans(3) + rot6d_22(132)]
    'motion_198': [T, 198],      # [motion_135 + positions_flat(63)]
}
```

---

## ⚡ Simple Alternatives (No Class Overhead)

### Raw SMPL Loading (3 lines)
```python
import numpy as np
data = np.load('motion.npz', allow_pickle=True)
poses, trans = data['poses'], data['trans']  # [T, 165], [T, 3]
```

### Pre-computed 135 Loading (2 lines)
```python
data = np.load('motion.npz', allow_pickle=True)
motion_135 = data['motion_135']  # [T, 135]
```

### Generic Format Handler
```python
from hftrainer.datasets.motion.motionhub.transforms.load_editing_source import _load_motion_198_from_npz
motion_198 = _load_motion_198_from_npz('motion.npz')  # [T, 198]
```

---

## 🔧 Parameters

| Param | Values | Default | Use Case |
|-------|--------|---------|----------|
| `rot_type` | axis_angle, rotation_6d, quaternion, euler | rotation_6d | Training: 6d |
| `transl_type` | abs, rel, abs_rel | abs | Motion capture data |
| `smpl_type` | smpl_22, smplh, smplx_55 | smpl_22 | Standard choice |
| `transl_aug_prob` | [0, 1] | 0.0 | Training: 0.5 |
| `transl_aug_yaw_deg` | degrees | 180 | Max rotation range |
| `transl_aug_offset_std` | (sx, sy, sz) | (1, 0, 1) | XZ displacement |

---

## 📤 Output Fields

```python
results['motion']           # [T, D] tensor
results['num_frames']       # T (int)
results['num_joints']       # 22, 52, or 55
results['fps']              # int or None
results['duration']         # T / fps
results['rot_type']         # rotation representation
results['smpl_type']        # joint subset
results['transl_type']      # translation representation
results['aug_yaw_deg']      # applied rotation (0 if no aug)
results['aug_offset']       # applied [x, y, z] offset
results['num_person']       # 1 or P
```

---

## ❌ Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "NaN values found" | Invalid augmentation | Set `transl_aug_prob=0.0` |
| "Inconsistent T" | Multi-person frame mismatch | Set `require_same_T=False` |
| "Same fps required" | Multi-person FPS mismatch | Set `require_same_fps=False` |
| "No 'motion_135'" | Pre-computed format missing | Use raw SMPL (poses + trans) |

---

## 🏃 Performance Comparison

```
Direct NumPy:        ⭐⭐⭐ Fastest   (2 lines)
Pre-computed 135:    ⭐⭐⭐ Fast      (2 lines)
_load_motion_198:    ⭐⭐  Medium     (1 line, handles all formats)
LoadSmplx55:         ⭐   Slowest    (Full pipeline, augmentation)
```

**Recommendation**:
- For **inference/evaluation**: Use direct NumPy or pre-computed 135
- For **training**: Use LoadSmplx55 (augmentation support)
- For **preprocessing**: Use _load_motion_198_from_npz (flexibility)

---

## 🔗 Related Utilities

| Function | File | Purpose |
|----------|------|---------|
| `process_smplx_pose()` | load_smplx.py | Convert SMPL rotations |
| `process_transl()` | load_smplx.py | Convert translations |
| `apply_root_yaw_to_axis_angle()` | load_smplx.py | Apply Y-axis rotation |
| `_load_motion_198_from_npz()` | load_editing_source.py | Universal NPZ loader |
| `motion135_to_198()` | compute_198dim.py | Forward kinematics |

