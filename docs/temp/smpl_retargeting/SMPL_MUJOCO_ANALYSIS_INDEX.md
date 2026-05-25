# SMPL ↔ MuJoCo Conversion Analysis — Complete Index

This directory contains comprehensive analysis of the SMPL-to-MuJoCo conversion functions from:
```
/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py
```

---

## 📄 Analysis Documents

### 1. **SMPL_MUJOCO_QUICK_SUMMARY.md** ⭐ START HERE
   - **Purpose**: TL;DR summary with Q&A format
   - **Time to read**: 5-10 minutes
   - **Covers**:
     - Forward/reverse conversion pipelines (visual flowcharts)
     - Root vs body joint treatment
     - `smpl_2_mujoco` mapping explanation
     - Key coordinate system findings
   - **Best for**: Quick reference, understanding the big picture

### 2. **SMPL_MUJOCO_COORDINATE_ANALYSIS.md** 
   - **Purpose**: Detailed section-by-section code analysis
   - **Time to read**: 15-20 minutes
   - **Covers**:
     - Complete `smpl_to_qpose()` code with inline annotations (lines 331–405)
     - Complete `qpos_to_smpl()` code with inline annotations (lines 552–571)
     - Summary table: root vs body joints
     - `smpl_2_mujoco` building and usage
     - Coordinate frame status (Y-up ↔ Z-up)
   - **Best for**: Understanding the functions in detail

### 3. **SMPL_MUJOCO_EXACT_CODE_REPORT.md**
   - **Purpose**: Line-by-line code inspection with evidence
   - **Time to read**: 20-30 minutes
   - **Covers**:
     - Full annotated `smpl_to_qpose()` with step-by-step breakdown
     - Full annotated `qpos_to_smpl()` with step-by-step breakdown
     - Evidence table: is any coordinate transform applied?
     - Root vs body joint treatment (detailed comparison)
     - `smpl_2_mujoco` building & usage with examples
     - Where coordinate conversion might occur (alternative locations)
   - **Best for**: Deep technical understanding, debugging, verification

---

## 🔑 Key Findings Summary

### The Core Question
**Does the code apply coordinate transforms to body joints, or only to the root?**

### The Answer
✅ **NEITHER — NO coordinate transforms applied to root OR body joints**

- **Root (Pelvis)**: 
  - Input: Axis-angle (3D)
  - Output: Quaternion (4D)
  - **Transform**: ❌ NONE — just representation change (axis-angle → rotation matrix → quaternion)
  - **Offset**: ✅ YES — `body_pos[1]` offset added to translation (not a coord transform)

- **Body Joints (L_Hip, ..., R_Hand)**:
  - Input: Axis-angle (3D)
  - Output: Euler angles ZYX (3D)
  - **Transform**: ❌ NONE — just representation change (axis-angle → rotation matrix → Euler)
  - **Reordering**: ✅ YES — `smpl_2_mujoco` reorders to MuJoCo joint order (not a coord transform)

### Y-up ↔ Z-up Conversion?
❌ **NOT in `smpl_to_qpose()` or `qpos_to_smpl()`**

Possible locations:
- `normalize_smpl_pose()` (lines 607–635) — has optional rotation matrix application
- Caller preprocessing — might convert before calling `smpl_to_qpose()`
- MuJoCo renderer — might handle coordinate display

---

## 🎯 Critical Code Sections

| What | Lines | File | Summary |
|------|-------|------|---------|
| **smpl_2_mujoco building** | 371-374 | smpl_mujoco.py | Maps SMPL indices to MuJoCo body order |
| **Root special case** | 397-399 | smpl_mujoco.py | Root output as quaternion (not Euler) |
| **Body joint reordering** | 391-393 | smpl_mujoco.py | Apply `smpl_2_mujoco` to reorder |
| **Position offset** | 403 | smpl_mujoco.py | Add `body_pos[1]` to translation |
| **Reverse conversion** | 552-571 | smpl_mujoco.py | Perfectly inverts forward pass |

---

## 📊 Data Flow

### Forward: SMPL → MuJoCo qpos

```
SMPL pose (batch, 72)
  ↓ [Axis-angle]
  ├─ Root(0):    Axis-angle → Matrix → QUATERNION
  └─ Body(1-23): Axis-angle → Matrix → Euler ZYX
  ↓ [Reorder by smpl_2_mujoco]
  ├─ Root:       stays at index 0
  └─ Body:       reordered to MuJoCo order
  ↓ [Assemble]
  qpos = [trans (3), root_quat (4), body_euler_reordered (63)]
  = (batch, 70)
```

### Reverse: MuJoCo qpos → SMPL pose

```
qpos (batch, 70) = [trans (3), root_quat (4), body_euler (63)]
  ↓
  For each joint in SMPL order:
    - Joint 0 (Pelvis):  Quat → Axis-angle
    - Joint 1-23 (Body): Euler ZYX → Axis-angle
  ↓
  SMPL pose (batch, 24, 3) in SMPL bone order
```

---

## 💾 File Locations

All analysis documents are in:
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
```

Files:
- `SMPL_MUJOCO_QUICK_SUMMARY.md` ← **Start here** ⭐
- `SMPL_MUJOCO_COORDINATE_ANALYSIS.md`
- `SMPL_MUJOCO_EXACT_CODE_REPORT.md`
- `SMPL_MUJOCO_ANALYSIS_INDEX.md` ← You are here

Original source code:
- `/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py`
- `/ref_repo/OmniH2O/phc/phc/smpllib/smpl_parser.py` (SMPL_BONE_ORDER_NAMES)

---

## 🔍 How to Use These Docs

### Use Case 1: "I just need a quick answer"
→ Read **SMPL_MUJOCO_QUICK_SUMMARY.md** (5 min)

### Use Case 2: "I need to understand the conversion pipeline"
→ Read **SMPL_MUJOCO_COORDINATE_ANALYSIS.md** (15 min)

### Use Case 3: "I need to verify every line of code"
→ Read **SMPL_MUJOCO_EXACT_CODE_REPORT.md** (30 min)

### Use Case 4: "I'm debugging a coordinate frame issue"
→ Search for "Coordinate Transform" in **SMPL_MUJOCO_EXACT_CODE_REPORT.md**

### Use Case 5: "I need to understand smpl_2_mujoco mapping"
→ Search for "smpl_2_mujoco" in **SMPL_MUJOCO_QUICK_SUMMARY.md** or **SMPL_MUJOCO_COORDINATE_ANALYSIS.md**

---

## ✅ Verification Checklist

- [x] Exact line numbers verified against source code
- [x] Code snippets copied verbatim from source
- [x] Root vs body joint handling distinguished
- [x] Coordinate transform absence confirmed (no axis swapping, no frame matrices)
- [x] `smpl_2_mujoco` building and usage explained
- [x] Reverse conversion (`qpos_to_smpl()`) documented
- [x] Data flow diagrams provided
- [x] All findings summarized with evidence

---

## 📞 Questions Answered

1. **Does the root get a coordinate transform?**
   - NO. It gets axis-angle → quaternion conversion (representation), plus `body_pos[1]` offset.

2. **Do body joints get a coordinate transform?**
   - NO. They get axis-angle → Euler conversion (representation), plus reordering via `smpl_2_mujoco`.

3. **Does Y-up → Z-up conversion happen?**
   - NO. Not in `smpl_to_qpose()` or `qpos_to_smpl()`. It happens elsewhere or is assumed already done.

4. **How is `smpl_2_mujoco` built?**
   - For each MuJoCo body name (in MuJoCo order), find its index in SMPL bone order. Result is a list of SMPL indices.

5. **What does `smpl_2_mujoco` do?**
   - Reorders body joint rotations from SMPL order to MuJoCo body order. Purely index mapping, not coordinate transform.

6. **Is the conversion reversible?**
   - YES. `qpos_to_smpl()` correctly inverts `smpl_to_qpose()`.

---

## 📝 Notes

- All analysis is based on code inspection, not execution
- SMPL model: 24 bones, 72-dim axis-angle representation
- MuJoCo qpos: 70-dim (3 translation + 4 root quaternion + 63 body Euler)
- Euler convention: ZYX (consistent between forward and reverse)
- Quaternion format: Needs reordering for SciPy compatibility
- Default standing height: 0.91437225 m (Z-up)

---

**Analysis Date**: 2026-05-15  
**Source Code**: `/ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py`  
**Analyst**: Code inspection and documentation generation

