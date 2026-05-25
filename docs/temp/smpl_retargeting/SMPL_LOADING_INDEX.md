# SMPL Motion Loading - Complete Index

## 📍 Quick Access

### Primary Reference
- **Full Guide**: [SMPL_LOADING_REFERENCE.md](./SMPL_LOADING_REFERENCE.md) - Complete 329-line reference with all details

### Class Definition
```
Location: hftrainer/datasets/motion/motionhub/transforms/load_smplx.py
Lines: 217-495
Import: from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
```

---

## 🎯 Finding What You Need

### Q: Where is LoadSmplx55 defined?
**A:** See **SMPL_LOADING_REFERENCE.md** → Section 1

### Q: What's the import statement?
**A:** 
```python
from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55
```

### Q: I need to load NPZ files - what's the simplest way?
**A:** See **SMPL_LOADING_REFERENCE.md** → Section 2 → Option A (Direct numpy.load - 5 lines)

### Q: I need rotation conversion - what's recommended?
**A:** See **SMPL_LOADING_REFERENCE.md** → Section 2 → Option B (Helper Function)

### Q: I have pre-computed motion_135/198 data - how do I load it?
**A:** See **SMPL_LOADING_REFERENCE.md** → Section 2 → Option C (Pre-computed 198-dim - 3 lines)

### Q: My NPZ format is unknown - what should I use?
**A:** See **SMPL_LOADING_REFERENCE.md** → Section 2 → Option D (Auto-detect)

### Q: I'm building a full training pipeline - how do I use LoadSmplx55?
**A:** See **SMPL_LOADING_REFERENCE.md** → Section 5 (Complete Example)

### Q: What NPZ formats does LoadSmplx55 support?
**A:** See **SMPL_LOADING_REFERENCE.md** → Section 1 → Input NPZ Format

---

## 📚 Document Structure

### SMPL_LOADING_REFERENCE.md (Main Guide)

1. **LoadSmplx55 Class Definition** (Lines 10-80)
   - File path, import, class definition
   - Key parameters
   - Input/output formats

2. **Simpler Alternatives** (Lines 84-240)
   - Option A: Direct numpy (⭐⭐⭐⭐⭐)
   - Option B: Helper function (⭐⭐⭐⭐)
   - Option C: Pre-computed 198 (⭐⭐⭐⭐⭐)
   - Option D: Auto-detect (⭐⭐⭐)
   - Option E: LoadO6dp (⭐⭐⭐⭐)

3. **Comparison Table** (Lines 245-260)
   - Side-by-side comparison of all methods

4. **Helper Functions** (Lines 265-310)
   - `_read_one_person_npz()`
   - `process_smplx_pose()`
   - `process_transl()`
   - `apply_root_yaw_to_axis_angle()`

5. **Complete Example** (Lines 315-329)
   - Full pipeline usage in config

---

## 🔧 Available Helper Functions

Located in `load_smplx.py`:

### 1. _read_one_person_npz(path: str)
**Lines: 208-214**
```python
Returns: (trans[T,3], poses[T,165], fps)
```

### 2. process_smplx_pose(pose, rot_type, out_type)
**Lines: 16-104**
```python
Converts: rotation representations + joint subsets
Returns: [T, J*D]
```

### 3. process_transl(abs_trans, transl_type)
**Lines: 107-133**
```python
Modes: "abs", "rel", "abs_rel"
Returns: [T, 3] or [T, 6]
```

### 4. apply_root_yaw_to_axis_angle(pose, R_y)
**Lines: 151-205**
```python
Applies: Y-axis rotation to root joint
Returns: modified pose with same shape
```

---

## 💾 NPZ Format Support

### Format 1: Raw SMPL (Standard)
```python
data["trans"]              # [T, 3]
data["poses"]              # [T, 165]
data["mocap_framerate"]    # int (optional)
```

### Format 2: Pre-computed 135-dim (PerMo)
```python
data["motion_135"]         # [T, 135]
data["mocap_framerate"]    # int (optional)
```

### Format 3: Pre-computed 198-dim (Full)
```python
data["motion_198"]         # [T, 198]
data["motion_135"]         # [T, 135] (fallback)
```

---

## 📊 Method Comparison at a Glance

| Method | Simplicity | Speed | Features | Multi-person | Augmentation |
|--------|-----------|-------|----------|--------------|--------------|
| Direct numpy | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | - | ❌ | ❌ |
| Helper function | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Conversion | ❌ | ❌ |
| Pre-computed 198 | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡⚡ | - | ❌ | ❌ |
| Auto-detect | ⭐⭐⭐ | ⚡⚡⚡⚡ | Format detect | ❌ | ❌ |
| LoadSmplx55 | ⭐⭐ | ⚡⚡⚡ | Full | ✅ | ✅ |
| LoadO6dp | ⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Pre-processed | ❌ | Limited |

---

## 🚀 Quick Start Decision Tree

```
Need to load SMPL NPZ files?
├─ Just raw parameters?
│  └─ Use: Direct numpy.load (Option 1)
├─ Need rotation conversion?
│  └─ Use: Helper function (Option 2)
├─ Pre-computed motion_135/198?
│  └─ Use: Pre-computed loader (Option 3)
├─ Unsure of NPZ format?
│  └─ Use: _load_motion_198_from_npz (Option 4)
├─ Full training pipeline?
│  └─ Use: LoadSmplx55 (Option 5)
└─ Using o6dp format?
   └─ Use: LoadO6dp (Option 6)
```

---

## 📁 Related Files in Same Directory

- `load_smplx.py` - Main SMPL loader (this file)
- `load_o6dp.py` - O6DP format loader
- `load_editing_source.py` - Auto-detect format handler
- `load_audio.py` - Audio loader
- `load_text.py` - Text/caption loader
- `compute_198dim.py` - FK computation for 198-dim
- `__init__.py` - Exports all loaders

---

## 📝 Code Examples Ready to Use

All code examples are complete and copy-paste ready. See SMPL_LOADING_REFERENCE.md for:
- Example 1: Direct numpy (5 lines)
- Example 2: Helper function (25 lines)
- Example 3: Pre-computed load (3 lines)
- Example 4: Auto-detect (1 line)
- Example 5: Full pipeline (config format)
- Example 6: O6DP format (2 lines)

---

## ✅ Summary of Findings

✓ **Class location**: Found at load_smplx.py lines 217-495
✓ **Import statement**: `from hftrainer.datasets.motion.motionhub.transforms.load_smplx import LoadSmplx55`
✓ **Input formats**: Raw SMPL, pre-computed 135-dim, pre-computed 198-dim
✓ **Output formats**: Single [T, D] or multi [P, T, D]
✓ **Simpler alternatives**: 6 options provided, from 1-25 lines
✓ **Helper functions**: 4 key utility functions documented
✓ **Reference guide**: Complete 329-line guide with examples

---

## 📞 Support Files

- Full guide: [SMPL_LOADING_REFERENCE.md](./SMPL_LOADING_REFERENCE.md)
- Code snippets: Available in reference guide sections
- This index: Quick navigation guide

---

**Last Updated:** May 19, 2026
**Status:** Complete ✅
