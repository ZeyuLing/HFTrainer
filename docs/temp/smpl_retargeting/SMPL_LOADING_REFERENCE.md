# SMPL Motion Data Loading Reference Guide

## 1. LoadSmplx55 Class Definition

### File Path
```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/datasets/motion/motionhub/transforms/load_smplx.py
```

### Import Statement
```python
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
```

### Class Definition
**Location**: Lines 217-495 in `load_smplx.py`

```python
@TRANSFORMS.register_module(force=True)
class LoadSmplx55(BaseTransform):
    """
    Supports both single-person and multi-person SMPL-X loading with Y-up rigid body augmentation.
    - Single person: results[f"{key}_path"] is str, returns [T, D] torch.FloatTensor
    - Multi person: results[f"{key}_path"] is List[str], returns [P, T, D] torch.FloatTensor
    """
```

### Key Parameters
- `key` (str): Default "motion", results dict key
- `rot_type` (str): "rotation_6d" | "axis_angle" | "quaternion" | "euler"
- `transl_type` (str): "abs" | "rel" | "abs_rel"
- `smpl_type` (str): "smpl_22" | "smplh" | "smplx_55"
- Augmentation params: `transl_aug_prob`, `transl_aug_yaw_deg`, `transl_aug_offset_std`
- Multi-person constraints: `require_same_T`, `require_same_fps`

### Input NPZ Format
The loader expects NPZ files with one of these formats:

#### Format 1: Raw SMPL Parameters (Standard)
```python
data = np.load("motion.npz", allow_pickle=True)
# Required keys:
data["trans"]              # [T, 3] - world root translation
data["poses"]              # [T, 165] or [T, 55, 3] - axis-angle (55 joints × 3)
data["mocap_framerate"]    # int - optional, FPS
```

#### Format 2: Pre-computed motion_135 (PerMo Dataset)
```python
# Fast path - no augmentation applied
data["motion_135"]         # [T, 135] - pre-computed translation(3) + rot6d(132)
data["mocap_framerate"]    # int - optional, FPS
# Note: When both motion_135 and poses exist, raw SMPL conversion is used
```

#### Format 3: Pre-computed motion_198 (via LoadEditingSourceMotion)
```python
data["motion_198"]         # [T, 198] - full motion with positions
# Can also have motion_135 or raw SMPL params as fallback
```

### Output Format
- **Single person**: `torch.FloatTensor` of shape `[T, D]`
  - D = 3 (translation) + J×R (rotation representation)
  - J = 22 (smpl_22), 52 (smplh), 55 (smplx_55)
  - R = 3 (axis_angle/euler), 4 (quaternion), 6 (rotation_6d)

- **Multi person**: `torch.FloatTensor` of shape `[P, T, D]`
  - P = number of persons
  - Same D as single person

### Metadata Added to Results Dict
```python
results[key]              # Motion tensor
results["num_person"]     # 1 for single, P for multi
results["rot_type"]       # Rotation representation used
results["smpl_type"]      # Joint subset used
results["transl_type"]    # Translation type
results["num_frames"]     # T
results["duration"]       # T/fps or None
results["fps"]            # Framerate
results["num_joints"]     # 22, 52, or 55
results["aug_yaw_deg"]    # Applied rotation angle (0 if no aug)
results["aug_offset"]     # Applied XZ offset [x, 0, z] or [0, 0, 0]
results[f"{key}_paths"]   # For multi-person: list of paths
```

---

## 2. Simpler Alternatives for Loading SMPL NPZ Files

### Option A: Direct numpy.load (Minimal)
**Simplest approach** for just loading raw SMPL data:

```python
import numpy as np

def load_smpl_simple(path: str):
    """Load raw SMPL parameters from NPZ without any processing."""
    data = np.load(path, allow_pickle=True)
    trans = np.asarray(data["trans"], dtype=np.float32)      # [T, 3]
    poses = np.asarray(data["poses"], dtype=np.float32)      # [T, 165]
    fps = int(data["mocap_framerate"]) if "mocap_framerate" in data else None
    return trans, poses, fps

# Usage
trans, poses, fps = load_smpl_simple("motion.npz")
print(f"Loaded motion: {poses.shape[0]} frames, {fps} FPS")
```

**Pros**: No dependencies, < 10 lines
**Cons**: No representation conversion, no augmentation, no validation

---

### Option B: Helper Function (Recommended)
**Moderate complexity** for basic motion loading:

```python
import numpy as np
import torch
from typing import Tuple, Optional

def load_smpl_motion(
    path: str,
    rot_type: str = "rotation_6d",
    smpl_type: str = "smpl_22",
    transl_type: str = "abs",
) -> Tuple[torch.Tensor, dict]:
    """
    Load and convert SMPL motion from NPZ.
    
    Args:
        path: Path to NPZ file
        rot_type: "axis_angle", "rotation_6d", "quaternion", or "euler"
        smpl_type: "smpl_22", "smplh", or "smplx_55"
        transl_type: "abs", "rel", or "abs_rel"
    
    Returns:
        motion tensor [T, D], metadata dict
    """
    from hftrainer.datasets.motion.motionhub.transforms.load_smplx import (
        process_smplx_pose, process_transl
    )
    
    data = np.load(path, allow_pickle=True)
    trans = np.asarray(data["trans"], dtype=np.float32)
    poses = np.asarray(data["poses"], dtype=np.float32)
    fps = int(data["mocap_framerate"]) if "mocap_framerate" in data else None
    
    # Convert representations
    transl = process_transl(trans, transl_type)  # [T, 3] or [T, 6]
    pose = process_smplx_pose(poses, rot_type, smpl_type)  # [T, J*D]
    
    # Concatenate
    motion = np.concatenate([transl, pose], axis=-1)
    
    metadata = {
        "num_frames": motion.shape[0],
        "num_joints": int(pose.shape[1] // (3 if rot_type != "quaternion" else 4 if rot_type == "quaternion" else 6)),
        "fps": fps,
        "rot_type": rot_type,
        "smpl_type": smpl_type,
        "transl_type": transl_type,
    }
    
    return torch.from_numpy(motion), metadata

# Usage
motion, meta = load_smpl_motion("motion.npz", rot_type="rotation_6d", smpl_type="smpl_22")
print(f"Loaded {meta['num_frames']} frames, {meta['num_joints']} joints")
print(f"Motion shape: {motion.shape}")
```

**Pros**: Reuses LoadSmplx55 utilities, clean interface, supports all representations
**Cons**: Requires hftrainer imports, no multi-person support

---

### Option C: Pre-computed motion_198 (Fastest)
**When pre-computed data is available**:

```python
import numpy as np
import torch

def load_motion_198(path: str) -> torch.Tensor:
    """Load pre-computed 198-dim motion (e.g., PerMo dataset)."""
    data = np.load(path, allow_pickle=True)
    
    # Priority: motion_198 > motion_135 > raw SMPL
    if "motion_198" in data:
        motion = np.asarray(data["motion_198"], dtype=np.float32)
    elif "motion_135" in data:
        motion = np.asarray(data["motion_135"], dtype=np.float32)
        # Optional: expand from 135 to 198 if positions needed
        motion = np.pad(motion, ((0, 0), (0, 63)), mode='constant')
    else:
        raise ValueError("motion_198 or motion_135 not found in NPZ")
    
    return torch.from_numpy(motion)

# Usage
motion = load_motion_198("permo_motion.npz")
print(f"Loaded pre-computed motion: {motion.shape}")  # [T, 198]
```

**Pros**: Fastest, < 15 lines, no external processing
**Cons**: Requires data to be pre-computed, no augmentation

---

### Option D: LoadEditingSourceMotion (For Mixed Formats)
**When dealing with multiple NPZ formats** (raw SMPL, motion_135, motion_198):

```python
from hftrainer.datasets.motion.motionhub.transforms.load_editing_source import _load_motion_198_from_npz

# Automatically handles all formats and converts to 198-dim
motion = _load_motion_198_from_npz("motion.npz")  # torch.Tensor [T, 198]
```

**Pros**: Auto-detects format, robust, no manual branching
**Cons**: Only outputs 198-dim, requires hftrainer import

---

### Option E: LoadO6dp (For Pre-processed o6dp Format)
**If using o6dp_1103 representation** (another pre-computed format):

```python
import numpy as np

def load_o6dp(path: str, joints_num: int = 22) -> np.ndarray:
    """Load o6dp_1103 pre-processed motion."""
    motion = np.load(path).astype(np.float32)  # [T, 201] for 22 joints
    
    # Layout: [transl(3), root_rot6d(6), body_rot6d(126), ric(66)]
    return motion
```

**Pros**: Simple format, direct numpy load
**Cons**: Specific to o6dp representation, not raw SMPL

---

## 3. Comparison Table

| Method | Simplicity | Features | Multi-person | Augmentation | Recommended For |
|--------|-----------|----------|--------------|--------------|-----------------|
| **Direct numpy** | ⭐⭐⭐⭐⭐ | None | ❌ | ❌ | Quick prototyping |
| **Helper function** | ⭐⭐⭐⭐ | Rotation conversion | ❌ | ❌ | One-off processing |
| **Pre-computed 198** | ⭐⭐⭐⭐⭐ | None | ❌ | ❌ | Production (pre-computed) |
| **LoadEditingSource** | ⭐⭐⭐ | Auto-detect format | ❌ | ❌ | Multiple formats |
| **LoadSmplx55 (full)** | ⭐⭐ | Full pipeline | ✅ | ✅ | Training pipelines |
| **LoadO6dp** | ⭐⭐⭐⭐ | Pre-processed | ❌ | Limited | o6dp format |

---

## 4. Key Helper Functions in load_smplx.py

### `_read_one_person_npz(path: str)`
**Lines 208-214** - Basic NPZ reader
```python
def _read_one_person_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Union[int, None]]:
    """Read single-person NPZ: returns (trans[T,3], poses[T,165], fps or None)"""
    data = np.load(path, allow_pickle=True)
    abs_trans = np.asarray(data["trans"], dtype=np.float32)
    poses = np.asarray(data["poses"], dtype=np.float32)
    fps = int(data["mocap_framerate"]) if "mocap_framerate" in data else None
    return abs_trans, poses, fps
```

### `process_smplx_pose(pose_55_axis_angle, rot_type, out_type)`
**Lines 16-104** - Convert SMPL-X joint set and rotation representation
- Handles 55 joints, SMPL-H (52 joints), SMPL (22 joints)
- Converts between axis-angle, rotation_6d, quaternion, euler
- Returns: `[T, J*D]` where D depends on rot_type

### `process_transl(abs_trans, transl_type)`
**Lines 107-133** - Convert translation representation
- "abs": absolute translation
- "rel": relative translation (frame-to-frame difference)
- "abs_rel": concatenate absolute + relative

### `apply_root_yaw_to_axis_angle(pose_55_axis_angle, R_y)`
**Lines 151-205** - Apply Y-axis rotation augmentation
- Modifies root joint (index 0) rotation
- Other joints unchanged (parent-relative)

---

## 5. Complete Example: Full Pipeline Usage

```python
# In your config file:
dict(
    type='LoadSmplx55',
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs',
    smpl_type='smpl_22',
    transl_aug_prob=0.5,        # 50% chance of augmentation
    transl_aug_yaw_deg=180.0,   # ±180° random rotation
    transl_aug_offset_std=(1.0, 0.0, 1.0),  # XZ plane offset
    require_same_T=True,
    require_same_fps=True,
)

# In your dataset:
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55

loader = LoadSmplx55(
    key='motion',
    rot_type='rotation_6d',
    smpl_type='smpl_22',
)

results = {'motion_path': 'path/to/motion.npz'}
results = loader(results)

# Access results:
print(results['motion'].shape)      # [T, 135] for smpl_22 + rot6d + abs_transl
print(results['num_frames'])        # T
print(results['num_joints'])        # 22
print(results['fps'])               # Framerate
print(results['aug_yaw_deg'])       # Applied rotation (0 if no aug)
```

