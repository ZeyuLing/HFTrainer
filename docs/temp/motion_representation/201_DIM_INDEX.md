# 201-Dim Motion Format - Complete Analysis Index

## Overview

This analysis package contains comprehensive documentation on the **201-dimensional motion format** used throughout the HyMotion codebase.

**Direct Answer to Your Question:**
The 66 dimensions in [135:201] are **22 joints × 3D = 66 dimensions (including pelvis)** in **Scheme D encoding** (XZ relative to pelvis, Y absolute world height).

---

## Documents in This Package

### 1. **201_DIM_INDEX.md** (this file)
Quick navigation guide and overview.

### 2. **201_dim_quick_ref.txt** ⭐ START HERE
- Visual ASCII diagrams
- Quick reference boxes
- Layout overview
- Joint ordering
- File locations
- Best for: Quick lookups and understanding the structure at a glance

### 3. **201_dim_format_analysis.md** 📖 COMPREHENSIVE GUIDE
- Detailed explanation of all components
- Translation (dims 0-3)
- Rotation encoding (dims 3-135)
- Joint positions (dims 135-201) - **the 66 dims explained**
- Scheme D encoding rationale
- Comparison with 198-dim and 135-dim formats
- Data flow and pipeline
- Forward kinematics details
- Rotation conventions (row-major vs column-major)
- Best for: Deep understanding and troubleshooting

### 4. **201_dim_code_reference.md** 💻 CODE EXAMPLES
- Direct code excerpts from source files
- Function signatures with explanations
- Concrete examples of how 66 dims are computed
- FK implementation walkthrough
- Dimension breakdown with indices
- Statistics computation example
- File loading pipeline
- Best for: Implementation details and debugging

---

## Key Findings Summary

### The 66 Dimensions Breakdown

```
Dims [135:201]: 66 dimensions total

Structure: 22 joints × 3D coordinates

Per joint encoding (Scheme D):
  [0] = world_x - pelvis_x  (X relative to pelvis)
  [1] = world_y             (Y absolute world height)
  [2] = world_z - pelvis_z  (Z relative to pelvis)

All 22 joints INCLUDED (unlike 198-dim which excludes pelvis)
```

### How They're Computed

1. Parse 135-dim motion (trans + rot6d)
2. Run Forward Kinematics on LOCAL rotations
3. Get world positions: (T, 22, 3)
4. Apply Scheme D encoding
5. Flatten to 66 dims
6. Concatenate to rotation data → 201-dim

### Key Files

| File | Purpose |
|------|---------|
| `compute_201dim_stats.py` | Definitive source for 201-dim position computation |
| `compute_198dim.py` | Shows the 198-dim variant (excludes pelvis) |
| `load_o6dp.py` | Loads pre-computed 201-dim npy files |
| `differentiable_fk.py` | FK implementation that generates positions |
| `load_smplx.py` | Loads SMPL and creates motion_135 |
| `fk_utils.py` | Kinematics utilities and joint tree |

---

## Quick Navigation

### I want to understand...

**...the overall structure quickly**
→ Read `201_dim_quick_ref.txt` (ASCII diagrams)

**...how the 66 dimensions are calculated**
→ Read `201_dim_code_reference.md` (Code Examples section)

**...why this encoding scheme exists**
→ Read `201_dim_format_analysis.md` (Scheme D Encoding section)

**...the exact code that computes positions**
→ Read `201_dim_code_reference.md` (Code Evidence, Item 1)

**...how it differs from 198-dim**
→ Read `201_dim_format_analysis.md` (Comparison section)

**...the rotation space conventions**
→ Read `201_dim_format_analysis.md` (Rotation Conventions section)

**...how to extract individual joint positions**
→ Read `201_dim_code_reference.md` (Dimension Breakdown Example section)

---

## Core Concepts

### Three Motion Formats

| Format | Dims | Content | Use |
|--------|------|---------|-----|
| **135-dim** | 135 | trans(3) + rot6d(132) | Rotation-only models |
| **198-dim** | 198 | 135-dim + positions(63) | FK consistency loss |
| **201-dim** | 201 | 135-dim + positions(66) | o6dp_1103 pre-computed |

**Key difference:** 201-dim includes pelvis in positions (22×3), 198-dim excludes it (21×3).

### Scheme D Encoding

Mixed absolute/relative representation:
- **XZ (horizontal):** Relative to pelvis → invariant to locomotion, more compressible
- **Y (vertical):** Absolute world height → preserves ground contact info

### Row-Major Rotation Convention

This codebase uses **row-major 6D rotation** (non-standard):
```
Row-major:     [R₀₀, R₀₁, R₁₀, R₁₁, R₂₀, R₂₁]
Column-major:  [R₀₀, R₁₀, R₂₀, R₀₁, R₁₁, R₂₁]
```

Conversion: index reordering [0,3,1,4,2,5]

---

## The SMPL-22 Skeleton

```
Pelvis (0)
├─ L_Hip (1) → L_Knee (4) → L_Ankle (7) → L_Foot (10)
├─ R_Hip (2) → R_Knee (5) → R_Ankle (8) → R_Foot (11)
├─ Spine1 (3) → Spine2 (6) → Spine3 (9)
│  ├─ Neck (12) → Head (15)
│  ├─ L_Collar (13) → L_Shoulder (16) → L_Elbow (18) → L_Wrist (20)
│  └─ R_Collar (14) → R_Shoulder (17) → R_Elbow (19) → R_Wrist (21)
```

All 22 joints appear in both rotation and position channels of 201-dim format.

---

## Source Code Locations

### Core Implementation
- **`hftrainer/pipelines/motion/differentiable_fk.py`**
  - FK computation
  - motion135_to_fk function
  - rotation matrix conversions

### Dataset Transforms
- **`hftrainer/datasets/motion/motionhub/transforms/`**
  - `load_smplx.py`: SMPL loading and motion_135 creation
  - `compute_198dim.py`: Position computation (reference for understanding)
  - `load_o6dp.py`: Load pre-computed o6dp_1103 format
  - `fk_utils.py`: Kinematics utilities
  - `local_to_global.py`: Rotation space conversion

### Statistics & Analysis
- **`scripts/data/compute_201dim_stats.py`**
  - DEFINITIVE source for 201-dim position computation
  - Shows exact formulas and conventions
  - Multiprocessing-based statistics computation

- **`scripts/data/compute_198dim_stats.py`**
  - Similar structure but excludes pelvis from positions

---

## Common Questions & Answers

**Q: Are the 66 dims exactly 22×3?**
A: Yes. Exactly 22 joints × 3 coordinates per joint = 66 dimensions.

**Q: Does it include pelvis?**
A: Yes. Pelvis (joint 0) IS included in all 22 joints.

**Q: Is this RIC (root-invariant coordinates)?**
A: Not in the strict sense. It's world positions with Scheme D encoding (XZ relative, Y absolute).

**Q: How are positions computed?**
A: Forward Kinematics from local rotations, then apply Scheme D encoding.

**Q: Why XZ relative but Y absolute?**
A: XZ relative makes positions invariant to global locomotion. Y absolute preserves ground contact info (ankles near Y≈0).

**Q: What's the difference from 198-dim?**
A: 201-dim includes pelvis in positions (66 dims), 198-dim excludes it (63 dims).

**Q: Can I use just the rotation part (135-dim)?**
A: Yes, for rotation-only models or when FK consistency isn't needed.

**Q: How are statistics computed?**
A: Across all frames and samples using mean and variance (Welford's algorithm for numerics).

---

## Implementation Checklist

If you're implementing or using this format:

- [ ] Understand rotation is **row-major** (not column-major)
- [ ] Know that **all 22 joints** are included in both rotation and positions
- [ ] Remember **Scheme D encoding** for positions (XZ rel, Y abs)
- [ ] FK must use **LOCAL rotations** to compute correct positions
- [ ] **Pelvis IS included** (dims 135-137), unlike some other formats
- [ ] Statistics should be computed with rotation_space matching your model
- [ ] FK consistency loss can use position channels for training

---

## File Processing Order

```
Raw SMPL NPZ
    ↓
LoadSmplx55 (load_smplx.py)
    ├─ Extract 22 joints from 55
    ├─ Convert axis-angle → row-major 6D rot6d
    └─ Output: motion_135 (T, 135)
    ↓
(Optional) Compute201DimPosition or Compute198DimPosition
    ├─ Run FK on LOCAL rotations
    ├─ Apply Scheme D encoding
    ├─ Include all 22 joints (for 201-dim)
    └─ Output: motion_201 (T, 201) or motion_198 (T, 198)
    ↓
(Optional) LocalToGlobalRotation
    ├─ Convert only rotation channels (dims 3-135)
    ├─ Position channels unchanged
    └─ Output: motion_201/198 with global rotation
    ↓
Ready for training/inference
```

---

## References & Cross-Links

### Key Functions
- `motion135_to_fk()` - Parse 135-dim and compute world positions
- `differentiable_fk()` - Core FK algorithm
- `compute_position_channels()` - Compute 63 or 66 position dims
- `LoadSmplx55.transform()` - Load SMPL and create motion_135

### Key Constants
- `SMPL22_PARENTS` - Kinematic tree structure
- `_ROW_TO_COL`, `_COL_TO_ROW` - Rotation format conversions
- Dims 0-3, 3-135, 135-201 - Semantic boundaries

### Related Statistics Files
- `_stats/Mean.npy`, `_stats/Std.npy` - Normalization constants (shape 201,)

---

## Version Info

- **Format:** o6dp_1103 (201-dim variant for 22 joints)
- **SMPL version:** 22 joints (body-only)
- **Rotation convention:** Row-major 6D
- **Position encoding:** Scheme D (XZ relative, Y absolute)
- **Created:** For HyMotion motion generation project

---

## Document History

Created: 2026-05-18
Last Updated: 2026-05-18

Contains analysis of:
- `compute_201dim_stats.py`
- `compute_198dim.py`
- `load_o6dp.py`
- `load_smplx.py`
- `differentiable_fk.py`
- `fk_utils.py`

---

## How to Use This Documentation

1. **For quick answers:** Start with `201_dim_quick_ref.txt`
2. **For understanding:** Read `201_dim_format_analysis.md` top-to-bottom
3. **For implementation:** Reference `201_dim_code_reference.md`
4. **For specific issues:** Search across all documents or refer to source code

All three documents are complementary and reference each other.
