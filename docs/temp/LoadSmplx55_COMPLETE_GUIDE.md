# LoadSmplx55 Class - Complete Guide

## File Location & Import

### Exact File Path
```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/datasets/motion/motionhub/transforms/load_smplx.py
```

### Correct Import Statements

**Option 1 - Direct class import:**
```python
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
```

**Option 2 - Via transforms module:**
```python
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55
```

**Option 3 - Via registry (if using in a pipeline config):**
```python
# In config file:
dict(type='LoadSmplx55', key='motion', rot_type='rotation_6d', transl_type='abs')
```

---

## Class Overview

`LoadSmplx55` is a PyTorch-style transform registered with MMCVs `BaseTransform` and the `TRANSFORMS` registry. It handles loading SMPL-X 55-joint motion data from NPZ files with support for:

- **Single-person** and **multi-person** motion loading
- **Y-up global rotation augmentation** (rotate around Y-axis)
- **XZ-plane translation augmentation** (horizontal displacement)
- **Joint subset conversion** (SMPL-X 55 → SMPL-H 52 → SMPL 22)
- **Rotation representation conversion** (axis-angle → rotation 6D, quaternion, Euler)
- **Translation representation** (absolute, relative, or combined)

---

## Constructor Parameters

```python
class LoadSmplx55(BaseTransform):
    def __init__(
        self,
        key: str = "motion",
        rot_type: str = "rotation_6d",
        transl_type: str = "abs",
        smpl_type: str = "smpl_22",
        transl_aug_prob: float = 0.0,
        transl_aug_yaw_deg: float = 180.0,
        transl_aug_offset_std: Tuple[float, float, float] = (1.0, 0.0, 1.0),
        require_same_T: bool = True,
        require_same_fps: bool = True,
    ):
```

### Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | str | `"motion"` | Results dict key to store/load motion data |
| `rot_type` | str | `"rotation_6d"` | Rotation representation: `axis_angle`, `rotation_6d`, `quaternion`, `euler` |
| `transl_type` | str | `"abs"` | Translation representation: `abs` (absolute), `rel` (relative), `abs_rel` (both) |
| `smpl_type` | str | `"smpl_22"` | Joint subset: `smpl_22` (22 joints), `smplh` (52 joints), `smplx_55` (55 joints) |
| `transl_aug_prob` | float | `0.0` | Probability [0,1] to apply Y-axis rotation + XZ translation augmentation |
| `transl_aug_yaw_deg` | float | `180.0` | Y-axis rotation range in degrees: `[-yaw_deg, +yaw_deg]` |
| `transl_aug_offset_std` | Tuple[float, float, float] | `(1.0, 0.0, 1.0)` | Standard deviation for XZ offsets (Y always 0) |
| `require_same_T` | bool | `True` | Multi-person: enforce all sequences have same frame count |
| `require_same_fps` | bool | `True` | Multi-person: enforce all sequences have same FPS |

---

## Expected NPZ Format

The class expects NPZ files with the following keys:

### Raw SMPL Format (Standard)
```python
{
    'poses': np.ndarray,              # [T, 165] axis-angle or [T, 55, 3]
    'trans': np.ndarray,              # [T, 3] absolute translation (world coords)
    'mocap_framerate': int,           # (optional) FPS, e.g. 30, 60
}
```

### Pre-computed 135-dim Format (PerMo & similar)
```python
{
    'motion_135': np.ndarray,         # [T, 135] pre-computed [trans(3) + rot6d_22(132)]
    'mocap_framerate': int,           # (optional) FPS
    # Note: 'poses' and 'trans' NOT present
}
```

### Data Types
- **poses**: float32, axis-angle representation in radians
  - Shape `[T, 165]` → reshaped to `[T, 55, 3]` internally
  - Handles both SMPL-X 55 joints, SMPL-H 52 joints (auto-padded to 55)
- **trans**: float32, world-space translation
- **mocap_framerate**: int, typically 30 or 60 Hz

---

## Output Format

### Single-person Motion (when input path is str)
```python
results['motion'] = torch.FloatTensor  # [T, D]
results['num_person'] = 1
results['num_frames'] = T
results['num_joints'] = 22 (or 52, 55 depending on smpl_type)
results['fps'] = mocap_framerate or None
results['duration'] = T / fps
results['rot_type'] = rot_type (passed through)
results['smpl_type'] = smpl_type (passed through)
results['transl_type'] = transl_type (passed through)
results['aug_yaw_deg'] = yaw_deg (0 if no augmentation)
results['aug_offset'] = [x, y, z] offset applied
```

### Multi-person Motion (when input path is List[str])
```python
results['motion'] = torch.FloatTensor  # [P, T, D] (P = num persons)
results['motion_paths'] = list of paths
results['num_person'] = len(paths)
results['num_frames'] = T
results['num_joints'] = 22 (or 52, 55)
results['fps'] = fps or list of fps
results['duration'] = duration or list of durations
```

### Output Dimension D
- **Translation**: 3 dims (if transl_type="abs" or "rel") or 6 dims (if "abs_rel")
- **Pose**:
  - `rot_type="axis_angle"`: 3 × num_joints
  - `rot_type="rotation_6d"`: 6 × num_joints
  - `rot_type="quaternion"`: 4 × num_joints
  - `rot_type="euler"`: 3 × num_joints

**Example**: SMPL 22 joints with rotation_6d + absolute translation
```
D = 3 (transl) + 6 × 22 (rotation) = 3 + 132 = 135 dims
```

---

## Usage Examples

### Example 1: Basic Single-person Loading
```python
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55

loader = LoadSmplx55(
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs',
    smpl_type='smpl_22',
)

results = {
    'motion_path': 'path/to/motion.npz'
}

results = loader(results)
print(results['motion'].shape)  # [T, 135]
print(results['num_frames'])    # T
```

### Example 2: With Y-axis Augmentation
```python
loader = LoadSmplx55(
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs',
    smpl_type='smpl_22',
    transl_aug_prob=0.5,           # 50% chance to augment
    transl_aug_yaw_deg=180.0,      # Rotate ±180° around Y-axis
    transl_aug_offset_std=(0.5, 0.0, 0.5),  # XZ offset std dev
)

results = loader(results)
print(results['aug_yaw_deg'])      # Applied yaw in degrees
print(results['aug_offset'])       # Applied [x, y, z] offset
```

### Example 3: Multi-person Motion
```python
results = {
    'motion_path': [
        'path/to/person1.npz',
        'path/to/person2.npz',
        'path/to/person3.npz',
    ]
}

results = loader(results)
print(results['motion'].shape)     # [3, T, 135] (3 persons)
print(results['num_person'])       # 3
```

### Example 4: In Training Config
```python
# config.py
pipeline = [
    dict(type='LoadSmplx55',
         key='motion',
         rot_type='rotation_6d',
         transl_type='abs',
         smpl_type='smpl_22',
         transl_aug_prob=0.5,
         transl_aug_yaw_deg=90.0),
    dict(type='Compute198DimPosition', key='motion'),
    dict(type='RandomCropPadding', clip_len=360),
    dict(type='PackInputs'),
]

data_cfg = dict(
    train=dict(pipeline=pipeline, ...),
    val=dict(pipeline=pipeline, ...),
)
```

---

## Alternative: Simpler NPZ Loading Methods

### Method 1: Direct NumPy (Simplest for Raw SMPL)
```python
import numpy as np

def load_raw_smpl(npz_path):
    """Simplest: direct NPZ loading without augmentation."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'poses': np.asarray(data['poses'], dtype=np.float32),  # [T, 165]
        'trans': np.asarray(data['trans'], dtype=np.float32),  # [T, 3]
        'fps': int(data['mocap_framerate']) if 'mocap_framerate' in data else 30,
    }

# Usage
motion = load_raw_smpl('motion.npz')
poses = motion['poses']          # [T, 165] axis-angle
trans = motion['trans']          # [T, 3] translation
fps = motion['fps']
```

### Method 2: Pre-computed 135-dim (Fastest)
```python
def load_motion_135(npz_path):
    """Load pre-computed 135-dim motion (no SMPL conversion needed)."""
    data = np.load(npz_path, allow_pickle=True)
    if 'motion_135' in data:
        return np.asarray(data['motion_135'], dtype=np.float32)  # [T, 135]
    else:
        raise ValueError(f"No 'motion_135' in {npz_path}")

# Usage (PerMo dataset, etc.)
motion_135 = load_motion_135('motion.npz')  # [T, 135]
# Format: [trans(3), rot6d_22joints(132)]
```

### Method 3: Generic Motion Loader (Used internally)
```python
from hftrainer.datasets.motion.motionhub.transforms.load_editing_source import _load_motion_198_from_npz

def load_motion_any_format(npz_path):
    """Load motion from any format (raw SMPL, 135-dim, or 198-dim)."""
    # Supports:
    # 1. Pre-computed 198-dim (if available)
    # 2. Pre-computed 135-dim (with FK to 198)
    # 3. Raw SMPL (convert to 135 then FK to 198)
    motion_198 = _load_motion_198_from_npz(npz_path)
    return motion_198  # [T, 198]

# Usage
motion = load_motion_any_format('motion.npz')
```

### Method 4: Direct NPZ for Pre-computed Motion (No SMPL Conversion)
```python
def load_precomputed_motion(npz_path):
    """For datasets already in target format (motion_135 or motion_198)."""
    data = np.load(npz_path, allow_pickle=True)
    
    # Try formats in order of preference
    if 'motion_198' in data:
        return np.asarray(data['motion_198'], dtype=np.float32)  # [T, 198]
    elif 'motion_135' in data:
        return np.asarray(data['motion_135'], dtype=np.float32)  # [T, 135]
    else:
        raise ValueError("No pre-computed motion keys in NPZ")

# Usage
motion = load_precomputed_motion('motion.npz')
```

### Comparison Table

| Method | Speed | Flexibility | Best For | Requires |
|--------|-------|-------------|----------|----------|
| Direct NumPy | ⭐⭐⭐ Fast | ⭐ Low | Raw SMPL inspection | Only numpy |
| Pre-computed 135 | ⭐⭐⭐ Fast | ⭐ Low | PerMo, eval output | Only numpy |
| _load_motion_198_from_npz | ⭐⭐ Medium | ⭐⭐⭐ High | Any format handling | FK utils + numpy |
| LoadSmplx55 | ⭐ Slow | ⭐⭐⭐ High | Training pipeline | Full class overhead + augmentation |

---

## Key Implementation Details

### Internal Functions (Useful Utilities)

**1. `process_smplx_pose()`** - Core rotation conversion
```python
def process_smplx_pose(
    pose_55_axis_angle: np.ndarray,  # [T, 165] or [T, 55, 3]
    rot_type: str,                   # "axis_angle", "rotation_6d", "quaternion", "euler"
    out_type: str,                   # "smpl_22", "smplh", "smplx_55"
) -> np.ndarray:  # [T, J * D]
```

**2. `process_transl()`** - Translation representation conversion
```python
def process_transl(
    abs_trans: np.ndarray,  # [T, 3]
    transl_type: str,       # "abs", "rel", "abs_rel"
) -> np.ndarray:  # [T, 3] or [T, 6]
```

**3. `apply_root_yaw_to_axis_angle()`** - Apply Y-axis rotation
```python
def apply_root_yaw_to_axis_angle(
    pose_55_axis_angle: np.ndarray,
    R_y: np.ndarray,  # [3, 3] rotation matrix
) -> np.ndarray:
```

**4. `_read_one_person_npz()`** - Single file loader
```python
def _read_one_person_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Union[int, None]]:
    """Returns (trans[T,3], poses[T,165], fps or None)"""
```

---

## NPZ Format Reference

### Common Keys in HyMotion NPZ Files

| Key | Shape | Type | Notes |
|-----|-------|------|-------|
| `poses` | [T, 165] | float32 | SMPL-X axis-angle (55 joints × 3) |
| `trans` | [T, 3] | float32 | World translation |
| `mocap_framerate` | scalar | int | FPS (usually 30 or 60) |
| `motion_135` | [T, 135] | float32 | Pre-computed [trans(3) + rot6d_22(132)] |
| `motion_198` | [T, 198] | float32 | Extended [motion_135 + positions_flat(63)] |
| `positions` | [T, 22, 3] | float32 | FK-computed joint positions |
| `translation` | [T, 3] | float32 | Redundant copy of trans |

---

## Common Issues & Solutions

### Issue: "NaN values found after augmentation"
**Cause**: Invalid augmentation parameters or corrupted input data
**Solution**:
- Verify NPZ file contains valid float32 data
- Check `transl_aug_offset_std` values are reasonable
- Try `transl_aug_prob=0.0` to disable augmentation temporarily

### Issue: "Inconsistent T after processing" (Multi-person)
**Cause**: Different sequences have different frame counts
**Solution**:
- Set `require_same_T=False` to allow auto-truncation to minimum T
- Or pre-process NPZ files to have same frame count

### Issue: "all persons must have the same fps"
**Cause**: Sequences loaded at different FPS
**Solution**:
- Set `require_same_fps=False`
- Or resample NPZ files to common FPS before loading

---

## Performance Tips

1. **Disable augmentation in validation**: Set `transl_aug_prob=0.0`
2. **Use pre-computed formats**: Prefer 135/198-dim NPZ over raw SMPL
3. **Cache loaded motion**: Store results if loading same file multiple times
4. **Batch multi-person**: Loading multiple persons in one call is more efficient

---

## Related Files

- **Rotation utilities**: `hftrainer/models/motion/components/utils/geometry/rotation_convert.py`
- **FK computation**: `hftrainer/datasets/motion/motionhub/transforms/compute_198dim.py`
- **Alternative loader**: `hftrainer/datasets/motion/motionhub/transforms/load_editing_source.py`
- **Registry**: `hftrainer/registry.py`

