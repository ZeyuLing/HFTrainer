# SMPL ↔ MuJoCo Conversion Analysis — README

## 🎯 Summary

This analysis comprehensively documents the SMPL-to-MuJoCo conversion functions in:
```
/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py
```

**Key Finding**: 
- ❌ **NO coordinate transforms** applied to root or body joints
- ✅ **Only representation changes**: axis-angle ↔ quaternion/Euler
- ✅ **Joint reordering** via `smpl_2_mujoco` mapping (index-based, not transform)
- ❌ **Y-up ↔ Z-up conversion**: NOT in these functions

---

## 📚 Documents Structure

### Primary Analysis (New — This Session)

1. **SMPL_MUJOCO_QUICK_SUMMARY.md** ⭐ **START HERE**
   - **Length**: ~5.5 KB (5-10 min read)
   - **Format**: Q&A with TL;DR tables
   - **Content**:
     - Forward/reverse conversion pipelines (visual)
     - Root vs body joint treatment comparison
     - `smpl_2_mujoco` mapping explanation with examples
     - Key coordinate system findings
     - qpos structure breakdown
   - **Best for**: Quick reference, understanding the big picture

2. **SMPL_MUJOCO_COORDINATE_ANALYSIS.md**
   - **Length**: ~13 KB (15-20 min read)
   - **Format**: Detailed section-by-section code analysis
   - **Content**:
     - Complete `smpl_to_qpose()` code with inline annotations (lines 331–405)
     - Complete `qpos_to_smpl()` code with inline annotations (lines 552–571)
     - Root vs body joint treatment table
     - `smpl_2_mujoco` building and usage
     - Coordinate frame status analysis
     - Summary table with evidence
   - **Best for**: Understanding the conversion pipeline in detail

3. **SMPL_MUJOCO_EXACT_CODE_REPORT.md**
   - **Length**: ~15 KB (20-30 min read)
   - **Format**: Line-by-line code inspection with evidence tables
   - **Content**:
     - Full annotated `smpl_to_qpose()` with step-by-step breakdown
     - Full annotated `qpos_to_smpl()` with step-by-step breakdown
     - Coordinate transform evidence table
     - Root vs body joint treatment (detailed comparison)
     - `smpl_2_mujoco` building & usage with concrete examples
     - Where coordinate conversion might occur (alternative locations)
   - **Best for**: Deep technical understanding, debugging, verification

4. **SMPL_MUJOCO_ANALYSIS_INDEX.md**
   - **Format**: Navigation index
   - **Content**:
     - Document index with descriptions
     - Key findings summary
     - Critical code sections table
     - Data flow diagrams
     - Use case guidance
     - Verification checklist
     - Q&A section
   - **Best for**: Navigation, finding specific information

### Reference Analysis (Previous Sessions)

These documents provide additional context and may overlap with new analysis:

- `SMPL_MUJOCO_REFERENCE_REPORT.md` (26 KB)
- `SMPL_MUJOCO_REPORT.md` (17 KB)
- `SMPL_MUJOCO_ANALYSIS.md` (13 KB)
- `SMPL_MUJOCO_CODE_REFERENCE.md` (12 KB)
- `SMPL_MUJOCO_DETAILED_ANALYSIS.md` (10 KB)
- `SMPL_MUJOCO_QUICK_REFERENCE.md` (7.5 KB)

---

## 🚀 Quick Start

### For a 5-minute answer:
```
Read: SMPL_MUJOCO_QUICK_SUMMARY.md
→ See "TL;DR" table at top
→ Find your question in "The smpl_2_mujoco Mapping" section
```

### For a 20-minute deep dive:
```
1. Skim: SMPL_MUJOCO_ANALYSIS_INDEX.md (2 min)
2. Read: SMPL_MUJOCO_COORDINATE_ANALYSIS.md (15 min)
3. Check: SMPL_MUJOCO_QUICK_SUMMARY.md for quick ref (3 min)
```

### For complete verification:
```
1. Read: SMPL_MUJOCO_ANALYSIS_INDEX.md (5 min)
2. Study: SMPL_MUJOCO_EXACT_CODE_REPORT.md (25 min)
3. Reference: SMPL_MUJOCO_COORDINATE_ANALYSIS.md (as needed)
```

---

## ❓ Most Common Questions

**Q: Does the code convert from Y-up (SMPL) to Z-up (MuJoCo)?**
→ NO. No coordinate frame transformation in these functions. See Section 3 of SMPL_MUJOCO_COORDINATE_ANALYSIS.md

**Q: What exactly does `smpl_2_mujoco` do?**
→ Maps SMPL joint indices to MuJoCo body order (pure reordering, not transformation). See "The `smpl_2_mujoco` Mapping" in SMPL_MUJOCO_QUICK_SUMMARY.md

**Q: Is the root treated differently from body joints?**
→ YES. Root output is quaternion (not Euler). See "Root Special Treatment" in SMPL_MUJOCO_QUICK_SUMMARY.md

**Q: Is the conversion reversible?**
→ YES. `qpos_to_smpl()` correctly inverts `smpl_to_qpose()`. See SMPL_MUJOCO_COORDINATE_ANALYSIS.md line 2

**Q: Where does axis-angle get converted?**
→ Lines 378-379: axis-angle → rotation matrix using `angle_axis_to_rotation_matrix()`. See SMPL_MUJOCO_EXACT_CODE_REPORT.md

**Q: Why does root become quaternion?**
→ MuJoCo expects root (free joint) as quaternion. Body joints use Euler angles. See "Root Special Treatment" in SMPL_MUJOCO_QUICK_SUMMARY.md

---

## 📊 Data Structure Summary

### SMPL Format
```
pose: (batch, 72)  = (batch, 24 joints × 3 axis-angle)
trans: (batch, 3)  = (batch, x, y, z)

Bone order: [Pelvis(0), L_Hip(1), R_Hip(2), Torso(3), L_Knee(4), ..., R_Hand(23)]
```

### MuJoCo qpos Format
```
qpos: (batch, 70) = (batch, 3 trans + 4 root_quat + 63 body_euler)

Structure: [x(0), y(1), z(2), root_quat(3:7), body_euler(7:70)]

Root quat: [w, x, y, z] (needs reordering for SciPy)
Body euler: ZYX order (roll, pitch, yaw-like)
```

### The Conversion
```
Input (SMPL):              Output (MuJoCo):
[24 × 3D axis-angle]    →  [3 trans, 4 root_quat, 63 body_euler]
in SMPL order               in MuJoCo order
```

---

## 🔍 What Each Document Covers

| Document | Lengths | Depth | Code | Tables | Examples |
|----------|---------|-------|------|--------|----------|
| QUICK_SUMMARY | 5.5KB | Medium | ✓✓ | ✓✓✓ | ✓✓ |
| COORDINATE_ANALYSIS | 13KB | Deep | ✓✓✓ | ✓✓ | ✓ |
| EXACT_CODE_REPORT | 15KB | Very Deep | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| ANALYSIS_INDEX | 7.2KB | High-level | ✓ | ✓✓ | ✓ |

---

## ✅ Verification Checklist

This analysis verified:
- [x] Lines 331–405 (`smpl_to_qpose()`) extracted and analyzed
- [x] Lines 552–571 (`qpos_to_smpl()`) extracted and analyzed
- [x] Root vs body joint handling distinguished
- [x] Coordinate transform presence/absence confirmed
- [x] `smpl_2_mujoco` building logic explained
- [x] `smpl_2_mujoco` usage with examples shown
- [x] Reverse conversion documented
- [x] Data flow diagrams provided
- [x] SMPL_BONE_ORDER_NAMES verified (24 joints)
- [x] MuJoCo qpos structure analyzed

---

## 📍 File Locations

All analysis files are in:
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
```

Original source code:
```
/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py
/ref_repo/OmniH2O/phc/phc/smpllib/smpl_parser.py (SMPL_BONE_ORDER_NAMES)
```

---

## 🎓 Key Learning Points

1. **Representation vs. Transform**: The code does representation conversion (axis-angle ↔ quaternion/Euler), NOT coordinate frame transformation.

2. **Root Special Case**: Root is always output as quaternion (MuJoCo requirement), while body joints are Euler angles.

3. **Reordering via Index Mapping**: `smpl_2_mujoco` is purely index-based, not a mathematical transformation. It reorders which joint goes where in the output array.

4. **Reversibility**: The conversion is mathematically reversible because all steps are invertible (quaternion ↔ axis-angle, Euler ↔ axis-angle).

5. **Offset vs. Transform**: Adding `body_pos[1]` to translation is a position offset, not a coordinate frame transformation.

---

## 📞 Next Steps

1. **Quick Reference**: Read SMPL_MUJOCO_QUICK_SUMMARY.md
2. **Understanding**: Study SMPL_MUJOCO_COORDINATE_ANALYSIS.md
3. **Verification**: Review SMPL_MUJOCO_EXACT_CODE_REPORT.md if needed
4. **Navigation**: Use SMPL_MUJOCO_ANALYSIS_INDEX.md to find specific sections

---

## 📝 Notes

- All analysis is based on code inspection (not execution)
- Line numbers verified against source file
- Code snippets copied verbatim
- No assumptions made beyond what's in the code
- Alternative interpretations noted where applicable

---

**Analysis Date**: 2026-05-15  
**Source**: `/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py`  
**Status**: ✅ Complete and verified  

---

Last updated: 2026-05-15 16:43 UTC
