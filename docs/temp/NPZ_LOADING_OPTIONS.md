# SMPL Motion Data Loading: All Options Compared

## Summary: 4 Ways to Load SMPL Motion NPZ Files

### 🎯 Quick Decision Tree

```
Your Use Case?
│
├─ "I need to quickly inspect/debug raw data"
│  └─ Use Method 1: Direct NumPy (⭐⭐⭐ Fastest, 3 lines)
│
├─ "My NPZ has pre-computed motion_135 or motion_198"
│  └─ Use Method 2: Direct Pre-computed (⭐⭐⭐ Fast, 2 lines)
│
├─ "I need to handle ANY format (raw/135/198) in preprocessing"
│  └─ Use Method 3: _load_motion_198_from_npz (⭐⭐ Medium, 1 line)
│
└─ "I'm building a training pipeline with augmentation"
   └─ Use Method 4: LoadSmplx55 Class (⭐ Slow, but feature-rich)
```

---

## Detailed Comparison

### Method 1: Direct NumPy (Raw SMPL Format)

**⭐⭐⭐ Fastest | ⭐ Most Flexible | Best for: Quick inspection**

```python
import numpy as np

def load_raw_smpl(npz_path):
    """Direct NumPy loading — minimal overhead."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'poses': np.asarray(data['poses'], dtype=np.float32),  # [T, 165]
        'trans': np.asarray(data['trans'], dtype=np.float32),  # [T, 3]
        'fps': int(data['mocap_framerate']) if 'mocap_framerate' in data else 30,
    }

# Usage
motion = load_raw_smpl('motion.npz')
poses = motion['poses']                    # [T, 165] axis-angle
trans = motion['trans']                    # [T, 3] translation
fps = motion['fps']

print(f"Loaded {poses.shape[0]} frames at {fps} FPS")
print(f"Pose shape: {poses.shape} (55 joints × 3 dims)")
print(f"Trans range: [{trans.min():.2f}, {trans.max():.2f}]")
```

**Pros:**
- ✅ Zero class overhead
- ✅ No dependencies beyond numpy
- ✅ Instant debugging
- ✅ Full control over what you load

**Cons:**
- ❌ No automatic format detection
- ❌ No augmentation support
- ❌ Manual multi-person handling
- ❌ No representation conversion

**When to use:**
- Inspecting raw data
- Quick scripts
- Debugging
- Prototyping

**Code size:** ~3 lines

---

### Method 2: Load Pre-computed Motion (135 or 198 dim)

**⭐⭐⭐ Fast | ⭐⭐ Flexible | Best for: Pre-processed datasets**

```python
import numpy as np

def load_motion_135(npz_path):
    """Load pre-computed 135-dim motion directly."""
    data = np.load(npz_path, allow_pickle=True)
    if 'motion_135' in data:
        return np.asarray(data['motion_135'], dtype=np.float32)  # [T, 135]
    else:
        raise ValueError(f"No 'motion_135' in {npz_path}")

def load_motion_198(npz_path):
    """Load pre-computed 198-dim motion directly."""
    data = np.load(npz_path, allow_pickle=True)
    if 'motion_198' in data:
        return np.asarray(data['motion_198'], dtype=np.float32)  # [T, 198]
    else:
        raise ValueError(f"No 'motion_198' in {npz_path}")

# Usage (PerMo, evaluation outputs, etc.)
motion_135 = load_motion_135('motion.npz')  # [T, 135]
# Format: [trans(3), rot6d_22joints(132)]

# Or for extended representation
motion_198 = load_motion_198('motion.npz')  # [T, 198]
# Format: [motion_135(135), positions_flat(63)]

print(f"Loaded motion: {motion_135.shape}")
trans = motion_135[:, :3]                   # [T, 3]
rot6d = motion_135[:, 3:135]                # [T, 132]
```

**Pros:**
- ✅ Very fast (pre-computed)
- ✅ No SMPL conversion needed
- ✅ Clean format (135 or 198 dims)
- ✅ Works with eval outputs, PerMo datasets

**Cons:**
- ❌ Requires pre-computed keys
- ❌ No fallback to raw SMPL
- ❌ No augmentation
- ❌ No format auto-detection

**When to use:**
- PerMo dataset
- Evaluation outputs
- Pre-processed motion files
- Datasets with motion_135 or motion_198 keys

**Code size:** ~2 lines

---

### Method 3: Universal Motion Loader (Handles All Formats)

**⭐⭐ Medium | ⭐⭐⭐ Most Flexible | Best for: Preprocessing pipelines**

```python
from hftrainer.datasets.motion.motionhub.transforms.load_editing_source import _load_motion_198_from_npz

def load_motion_any_format(npz_path, bone_offsets_path=None):
    """
    Universal loader — handles all SMPL motion formats:
    1. Pre-computed 198-dim (if available)
    2. Pre-computed 135-dim (runs FK to get 198)
    3. Raw SMPL params (converts to 135, then FK to 198)
    
    Returns: [T, 198] tensor
    """
    motion_198 = _load_motion_198_from_npz(npz_path, bone_offsets=None)
    return motion_198

# Usage — handles ANY format automatically
motion_198 = load_motion_any_format('motion.npz')  # [T, 198]

# Decompose
trans = motion_198[:, :3]                      # [T, 3]
rot6d = motion_198[:, 3:135]                   # [T, 132]
positions_flat = motion_198[:, 135:198]        # [T, 63] = [T, 21, 3]
```

**Supported Input Formats:**
```
✅ NPZ with 'motion_198'  → Return directly
✅ NPZ with 'motion_135'  → Run FK to compute positions
✅ NPZ with 'poses'+'trans'  → Convert SMPL to 135, then FK
```

**Pros:**
- ✅ Automatic format detection
- ✅ Fallback chain (198 → 135 → raw SMPL)
- ✅ One-liner usage
- ✅ Handles most real-world cases
- ✅ Integrates with FK pipeline

**Cons:**
- ❌ Slight overhead from format detection
- ❌ Requires FK bone offsets (lazy-loaded)
- ❌ No augmentation
- ❌ Returns 198 only (no choice)

**When to use:**
- Mixed datasets (multiple formats)
- Robust preprocessing pipelines
- When you don't know the format ahead of time
- Need positions + rotations

**Code size:** ~1 line

**Related file:** `/hftrainer/datasets/motion/motionhub/transforms/load_editing_source.py` (lines 58-110)

---

### Method 4: LoadSmplx55 Class (Full Training Pipeline)

**⭐ Slowest | ⭐⭐⭐ Most Features | Best for: Training pipelines**

```python
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55

# Initialize (usually in config)
loader = LoadSmplx55(
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs',
    smpl_type='smpl_22',
    transl_aug_prob=0.5,           # 50% chance to augment
    transl_aug_yaw_deg=90.0,       # Rotate ±90° around Y-axis
    transl_aug_offset_std=(0.5, 0.0, 0.5),
)

# Use in pipeline
results = {'motion_path': 'motion.npz'}
results = loader(results)

motion = results['motion']  # [T, 135] tensor (augmented)
aug_yaw = results['aug_yaw_deg']  # Applied rotation angle
aug_offset = results['aug_offset']  # Applied [x, y, z] offset
```

**Pros:**
- ✅ Full augmentation support (Y-axis rotation, XZ translation)
- ✅ Format auto-detection
- ✅ Multi-person support
- ✅ Flexible rotation representations
- ✅ Joint subset conversion
- ✅ Proper error handling
- ✅ MMCV registry integration

**Cons:**
- ❌ Class instantiation overhead
- ❌ Most complex
- ❌ Slowest (due to augmentation, representation conversion)
- ❌ Overkill for simple loading

**When to use:**
- Training pipelines
- Need augmentation
- Need to convert representations
- Need multi-person handling
- Using MMCV configs

**Features:**
- Rotation representations: axis_angle, rotation_6d, quaternion, euler
- Translation types: abs, rel, abs_rel
- Joint subsets: smpl_22, smplh (52), smplx_55 (55)
- Y-axis rotation augmentation
- XZ-plane translation augmentation
- Multi-person consistency checks

**Code size:** ~10 lines (initialization + usage)

**File location:** `/hftrainer/datasets/motion/motionhub/transforms/load_smplx.py`

---

## Comprehensive Comparison Table

| Aspect | Method 1: NumPy | Method 2: Pre-computed | Method 3: Universal | Method 4: LoadSmplx55 |
|--------|---|---|---|---|
| **Speed** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Code Lines** | ~3 | ~2 | ~1 | ~10 |
| **Dependencies** | numpy | numpy | FK utils | Full pipeline |
| **Format Support** | Raw SMPL only | Pre-computed only | All formats | All formats |
| **Auto-detect** | ❌ | ❌ | ✅ | ✅ |
| **Augmentation** | ❌ | ❌ | ❌ | ✅ |
| **Representations** | Raw only | As-is | As-is | Convert |
| **Multi-person** | ❌ | ❌ | ❌ | ✅ |
| **Error Handling** | Manual | Manual | Automatic | Comprehensive |
| **Use Case** | Debugging | Eval outputs | Preprocessing | Training |
| **Learning Curve** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## Data Flow Diagrams

### Method 1: Direct NumPy
```
NPZ file
  ├─ poses [T, 165] ──→ NumPy ──→ Your code
  ├─ trans [T, 3]   ──→ NumPy ──→ Your code
  └─ fps              ──→ NumPy ──→ Your code
```

### Method 2: Pre-computed
```
NPZ file
  └─ motion_135 [T, 135] ──→ NumPy ──→ Your code
```

### Method 3: Universal Loader
```
NPZ file (ANY format)
  │
  ├─ If motion_198 exists  ──→ Return directly
  ├─ Else if motion_135    ──→ FK (compute positions)
  └─ Else if poses+trans   ──→ Convert SMPL ──→ FK
  │
  └─→ motion_198 [T, 198] ──→ Your code
```

### Method 4: LoadSmplx55
```
NPZ file (ANY format)
  │
  ├─ Format detection ──→ Load poses + trans
  ├─ Optional augmentation ──→ Y-rotation, XZ offset
  ├─ Representation conversion ──→ rot6d, quaternion, etc.
  ├─ Joint subset selection ──→ 22, 52, or 55 joints
  └─→ motion tensor ──→ results dict ──→ Your pipeline
```

---

## When Each Method is Best

### 🔍 Quick Data Inspection
```python
# Method 1: Direct NumPy
data = np.load('motion.npz', allow_pickle=True)
print(f"Keys: {data.files}")
print(f"Poses shape: {data['poses'].shape}")
```

### 📊 Work with PerMo or Pre-computed Data
```python
# Method 2: Pre-computed
motion_135 = data['motion_135']  # [T, 135]
# Already in correct format, no conversion needed
```

### 🔄 Mixed Dataset Formats in Preprocessing
```python
# Method 3: Universal Loader
motion = _load_motion_198_from_npz(file)  # Works for any format
# No need to check format beforehand
```

### 🚀 Building Training Pipelines with Config
```python
# Method 4: LoadSmplx55
# In config:
dict(type='LoadSmplx55', 
     rot_type='rotation_6d',
     transl_aug_prob=0.5)  # Augmentation during training
```

---

## Output Format Comparison

### Method 1 Output
```python
{
    'poses': np.ndarray [T, 165],  # axis-angle
    'trans': np.ndarray [T, 3],
    'fps': int
}
```

### Method 2 Output
```python
np.ndarray [T, 135]  # [trans(3) + rot6d_22(132)]
# or
np.ndarray [T, 198]  # [motion_135 + positions_flat(63)]
```

### Method 3 Output
```python
torch.Tensor [T, 198]  # [trans(3) + rot6d_22(132) + positions_flat(63)]
```

### Method 4 Output
```python
{
    'motion': torch.Tensor [T, D],  # D depends on configuration
    'num_frames': int,
    'fps': int,
    'rot_type': str,
    'aug_yaw_deg': float,
    'aug_offset': list,
    # ... + other fields
}
```

---

## Recommendation Summary

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| Debugging data | Method 1 | Fastest, most control |
| Working with PerMo | Method 2 | Pre-computed, no overhead |
| Mixed formats | Method 3 | Automatic detection |
| Production training | Method 4 | Full features + augmentation |
| Quick scripts | Method 1 or 2 | Minimal code |
| Research preprocessing | Method 3 | Flexible, robust |
| Deep learning pipelines | Method 4 | Integrates with frameworks |

---

## References

- **Method 1 Code**: Direct numpy loading
- **Method 2 Code**: Lines 225-232 in `load_editing_source.py`
- **Method 3 Code**: Function `_load_motion_198_from_npz()` (lines 58-110 in `load_editing_source.py`)
- **Method 4 Code**: `load_smplx.py` (lines 217-496)
- **NPZ Format Details**: `NPZ_FORMAT_DETAILS.md`
- **LoadSmplx55 Complete Guide**: `LoadSmplx55_COMPLETE_GUIDE.md`

