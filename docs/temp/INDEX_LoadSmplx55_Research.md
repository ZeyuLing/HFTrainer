# LoadSmplx55 Research Documentation - Complete Index

## 📚 All Generated Documentation

This folder contains comprehensive research on the `LoadSmplx55` class and SMPL motion data loading.

### 📄 Documents Created

1. **LoadSmplx55_COMPLETE_GUIDE.md** (13KB)
   - Full class documentation with complete API reference
   - Constructor parameters explained
   - NPZ input/output formats
   - Usage examples (single/multi-person)
   - Alternative loading methods comparison
   - Implementation details
   - Troubleshooting guide
   - **Read this for:** Deep understanding of LoadSmplx55

2. **LoadSmplx55_QUICK_REFERENCE.md** (5KB)
   - Quick cheat sheet and quick start guide
   - Common configurations for training/validation
   - Parameter quick table
   - Output fields reference
   - Performance comparison
   - Related utilities
   - **Read this for:** Quick lookup and examples

3. **NPZ_LOADING_OPTIONS.md** (12KB)
   - 4 different ways to load SMPL motion NPZ files
   - Detailed pros/cons of each method
   - Decision tree for choosing the right method
   - Performance comparison table
   - Data flow diagrams
   - Use case recommendations
   - **Read this for:** Choosing the best loading method for your use case

4. **NPZ_FORMAT_DETAILS.md** (Existing - Reference)
   - NPZ file structure and keys
   - Expected data formats
   - Metric derivations
   - File size expectations
   - Loading examples
   - **Read this for:** Understanding NPZ structure

---

## 🎯 Quick Navigation by Use Case

### "I need to quickly load SMPL motion data"
→ Read: **NPZ_LOADING_OPTIONS.md** → Choose Method 1 (Direct NumPy)
```python
import numpy as np
data = np.load('motion.npz', allow_pickle=True)
poses = data['poses']  # [T, 165]
```

### "I'm working with PerMo dataset"
→ Read: **NPZ_LOADING_OPTIONS.md** → Method 2 (Pre-computed)
```python
motion_135 = data['motion_135']  # [T, 135]
```

### "I have mixed format NPZ files"
→ Read: **NPZ_LOADING_OPTIONS.md** → Method 3 (Universal Loader)
```python
from hftrainer.datasets.motion.motionhub.transforms.load_editing_source import _load_motion_198_from_npz
motion_198 = _load_motion_198_from_npz('motion.npz')
```

### "I'm building a training pipeline"
→ Read: **LoadSmplx55_COMPLETE_GUIDE.md** → Method 4 (LoadSmplx55 Class)
```python
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55
loader = LoadSmplx55(rot_type='rotation_6d', transl_aug_prob=0.5)
```

### "I need a quick reference"
→ Read: **LoadSmplx55_QUICK_REFERENCE.md** (1-page cheat sheet)

---

## 📍 File Location

```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
  └─ hftrainer/datasets/motion/motionhub/transforms/load_smplx.py
```

## ✅ Import Statements

```python
# Option 1: Direct import
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55

# Option 2: Via transforms module
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55

# Option 3: In MMCV config
dict(type='LoadSmplx55', key='motion', rot_type='rotation_6d', transl_type='abs')
```

---

## 🗺️ Document Structure

```
LoadSmplx55_COMPLETE_GUIDE.md
├─ File Location & Import
├─ Class Overview
├─ Constructor Parameters (Full table)
├─ Expected NPZ Format
├─ Output Format
├─ Usage Examples (4 examples)
├─ Alternative Loading Methods (4 methods)
│  ├─ Method 1: Direct NumPy
│  ├─ Method 2: Pre-computed
│  ├─ Method 3: Universal Loader
│  └─ Method 4: LoadSmplx55 Class
├─ Comparison Table
├─ Key Implementation Details
├─ NPZ Format Reference
├─ Common Issues & Solutions
├─ Performance Tips
└─ Related Files

LoadSmplx55_QUICK_REFERENCE.md
├─ Location & Import (2 options)
├─ Quick Start (3 lines of code)
├─ Output Dimensions Table
├─ Common Configurations (3 examples)
├─ NPZ Format
├─ Simple Alternatives (4 methods)
├─ Parameters Table
├─ Output Fields
├─ Troubleshooting
├─ Performance Comparison
└─ Related Utilities

NPZ_LOADING_OPTIONS.md
├─ Quick Decision Tree
├─ Detailed Comparison (4 methods)
│  ├─ Method 1: Direct NumPy
│  ├─ Method 2: Pre-computed
│  ├─ Method 3: Universal Loader
│  └─ Method 4: LoadSmplx55
├─ Comprehensive Comparison Table
├─ Data Flow Diagrams
├─ When Each Method is Best
├─ Output Format Comparison
├─ Recommendation Summary
└─ References
```

---

## 📊 Key Information Summary

### File Path
```
/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
hftrainer/datasets/motion/motionhub/transforms/load_smplx.py (Lines 217-496)
```

### Correct Imports
```python
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55
```

### Output Dimensions (SMPL 22 joints)
- rotation_6d + abs transl: **135 dims** (3 + 6×22)
- quaternion + abs transl: **91 dims** (3 + 4×22)
- axis_angle + abs transl: **69 dims** (3 + 3×22)
- rotation_6d + abs_rel transl: **138 dims** (6 + 6×22)

### 4 Loading Methods Overview
| Method | Speed | Simplicity | Augmentation | Best For |
|--------|-------|-----------|--------------|----------|
| Direct NumPy | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | Debugging |
| Pre-computed | ⭐⭐⭐ | ⭐⭐⭐ | ❌ | PerMo data |
| Universal | ⭐⭐ | ⭐⭐ | ❌ | Mixed formats |
| LoadSmplx55 | ⭐ | ⭐ | ✅ | Training |

---

## 🔍 Document Usage Examples

### Example 1: Quick Lookup
```
Question: "What's the output dimension for rotation_6d?"
Answer: LoadSmplx55_QUICK_REFERENCE.md → "Output Dimensions" → 135 dims
```

### Example 2: Understanding a Config
```
Question: "What does transl_aug_yaw_deg=90 mean?"
Answer: LoadSmplx55_COMPLETE_GUIDE.md → "Parameters Explained" → 
        "Y-axis rotation range in degrees"
```

### Example 3: Choosing a Method
```
Question: "Which is fastest for loading 1000 files?"
Answer: NPZ_LOADING_OPTIONS.md → "Comprehensive Comparison Table" → 
        Method 2 (Pre-computed) ⭐⭐⭐ fastest
```

### Example 4: Troubleshooting
```
Question: "I got 'NaN values found' error"
Answer: LoadSmplx55_COMPLETE_GUIDE.md → "Common Issues & Solutions" → 
        Set transl_aug_prob=0.0
```

---

## 🎓 Learning Path

**Beginner** (Just want to load data):
1. Read: NPZ_LOADING_OPTIONS.md (Choose Method 1 or 2)
2. Code: Copy the 2-3 line example

**Intermediate** (Using in training pipeline):
1. Read: LoadSmplx55_QUICK_REFERENCE.md
2. Read: NPZ_LOADING_OPTIONS.md → Method 4
3. Code: Modify the provided example

**Advanced** (Customizing LoadSmplx55):
1. Read: LoadSmplx55_COMPLETE_GUIDE.md
2. Reference: Source code (load_smplx.py)
3. Reference: Related utilities and dependencies

---

## ⚡ Quick Copy-Paste Code Snippets

### Method 1: Direct NumPy (Fastest)
```python
import numpy as np
data = np.load('motion.npz', allow_pickle=True)
poses = data['poses']        # [T, 165]
trans = data['trans']        # [T, 3]
fps = data.get('mocap_framerate', 30)
```

### Method 2: Pre-computed Motion
```python
data = np.load('motion.npz', allow_pickle=True)
motion_135 = data['motion_135']  # [T, 135]
```

### Method 3: Universal Loader
```python
from hftrainer.datasets.motion.motionhub.transforms.load_editing_source import _load_motion_198_from_npz
motion_198 = _load_motion_198_from_npz('motion.npz')  # [T, 198]
```

### Method 4: LoadSmplx55 with Augmentation
```python
from hftrainer.datasets.motion.motionhub.transforms import LoadSmplx55

loader = LoadSmplx55(
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs',
    smpl_type='smpl_22',
    transl_aug_prob=0.5,
    transl_aug_yaw_deg=90.0,
)

results = loader({'motion_path': 'motion.npz'})
motion = results['motion']  # [T, 135]
```

---

## 📚 Related Resources

- **Original Source**: `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py`
- **Rotation Utils**: `hftrainer/models/motion/components/utils/geometry/rotation_convert.py`
- **FK Computation**: `hftrainer/datasets/motion/motionhub/transforms/compute_198dim.py`
- **Alternative Loader**: `hftrainer/datasets/motion/motionhub/transforms/load_editing_source.py`
- **Registry**: `hftrainer/registry.py` (TRANSFORMS registry)

---

## ❓ FAQ

**Q: Which method should I use?**
A: See NPZ_LOADING_OPTIONS.md → Decision Tree section

**Q: What's the default output dimension?**
A: 135 dims (3 translation + 6×22 joints rotation in 6D format)

**Q: Can I use LoadSmplx55 without augmentation?**
A: Yes, set `transl_aug_prob=0.0`

**Q: Does it support multi-person?**
A: Yes, pass a list of paths instead of a single path

**Q: What NPZ formats are supported?**
A: Raw SMPL (poses+trans), pre-computed 135-dim, and pre-computed 198-dim

**Q: Where is the class defined?**
A: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` (Lines 217-496)

---

## 📝 Document Metadata

- **Generated**: May 19, 2026
- **Total Size**: ~30KB
- **Files**: 3 new documents + 1 existing reference
- **Content**: Complete coverage of LoadSmplx55 class and NPZ loading
- **Code Examples**: 15+ working examples
- **Diagrams**: Data flow charts, comparison tables

---

*For the most recent information, always refer to the source code at:*
`/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/datasets/motion/motionhub/transforms/load_smplx.py`

