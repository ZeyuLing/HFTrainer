# ProtoMotions Reference Documentation Index

**Created:** 2026-05-19  
**Reference:** `ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py`

This directory contains comprehensive analysis of how ProtoMotions captures body poses during MuJoCo inference deployment. Use this index to find the right document for your use case.

---

## 📚 Document Overview

### 1. **Quick Start** (Start Here!)
- **File:** `body_pose_quick_reference.md` (8 KB)
- **Best for:** Understanding the big picture in 5 minutes
- **Contents:**
  - TL;DR: The one function that matters
  - Quaternion conversion chain (wxyz ↔ xyzw)
  - Body indexing (the +1 offset trap)
  - Main loop at 50 Hz (annotated pseudocode)
  - Common mistakes → silent failures
- **When to use:** First time reading, need quick answers

### 2. **Comprehensive Analysis**
- **File:** `body_pose_capture_analysis.md` (13 KB)
- **Best for:** Deep understanding of architecture
- **Contents:**
  - Executive summary with convention table
  - Main data flow diagram
  - Core function: `get_robot_state_from_mujoco()` (lines 327-370)
  - Quaternion conversion details
  - Anchoring body concept
  - Pose capture during initialization
  - ONNX input assembly
  - Key body indexing conventions
  - Main loop phases (5 phases detailed)
  - Deployment contract (key guarantees)
  - Critical gotchas (5 silent failure modes)
  - Input dimensions for G1
  - Summary table: data sources
- **When to use:** Implementing deployment code, need theoretical foundation

### 3. **Code Flow Reference**
- **File:** `code_flow_reference.md` (16 KB)
- **Best for:** Tracing code execution, line-by-line understanding
- **Contents:**
  - Complete call stack for one control step (annotated with line numbers)
  - Key functions & their signatures (4 main functions)
  - MuJoCo data structure reference
  - Quaternion conversion implementation
  - Main loop structure (pseudocode)
  - Data dimensions summary (G1 example)
  - Critical implementation details (5 detailed points)
  - File cross-references table
- **When to use:** Debugging, understanding state flow, modifying code

---

## 🎯 Quick Navigation by Use Case

### "I need to understand where body rotations come from"
→ Start: `body_pose_quick_reference.md` (Quaternion Conversion Chain section)  
→ Then: `body_pose_capture_analysis.md` (Core Function section, lines 351-359)  
→ Deep dive: `code_flow_reference.md` (Critical Implementation Details, point 2)

### "I'm implementing a new policy that needs body poses"
→ Start: `body_pose_capture_analysis.md` (ONNX Input Assembly section)  
→ Then: `code_flow_reference.md` (Key Functions & Signatures, `build_onnx_inputs()`)  
→ Copy: Example dimensions from Reference section

### "Rotation values look wrong / tracking is bad"
→ Read: `body_pose_quick_reference.md` (Common Mistakes → Silent Failures table)  
→ Check: `body_pose_capture_analysis.md` (Critical Gotchas section)  
→ Debug: `code_flow_reference.md` (Critical Implementation Details)

### "What is the 'anchor body'?"
→ Read: `body_pose_capture_analysis.md` (Anchoring Body Concept section)  
→ Or: `body_pose_quick_reference.md` (Anchor Body Concept section)

### "I need exact line numbers to modify"
→ Use: `code_flow_reference.md` (Complete Call Stack, every line referenced)

### "What's the quaternion format MuJoCo uses?"
→ Read: `body_pose_quick_reference.md` (Quaternion Conversion Chain)  
→ Or: `code_flow_reference.md` (Quaternion Conversion Details section)

---

## 📊 Document Cheat Sheet

| Document | Size | Key Topics | Best For |
|----------|------|-----------|----------|
| `body_pose_quick_reference.md` | 8 KB | TL;DR, conventions, main loop, gotchas | First read, quick answers |
| `body_pose_capture_analysis.md` | 13 KB | Architecture, functions, guarantees, gotchas | Understanding design |
| `code_flow_reference.md` | 16 KB | Call stack, line numbers, MuJoCo internals | Debugging, implementation |

---

## 🔑 Key Takeaways

### The Core Function (All You Really Need)
```python
def get_robot_state_from_mujoco(model, data, root_body_index=0):
    """Extract robot state from MuJoCo as ProtoMotions representation."""
    return {
        "dof_pos":            data.qpos[7:],           # [num_dofs]
        "dof_vel":            data.qvel[6:],           # [num_dofs]
        "body_rot":           body_rot,                # [num_bodies, 4] xyzw
        "root_local_ang_vel": data.qvel[3:6],         # [3]
    }
```

### The One Thing That Gets People (Quaternion Convention)
```
MuJoCo:        [w, x, y, z]  wxyz format
ProtoMotions:  [x, y, z, w]  xyzw format

Conversion:    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
```

### The Other Thing That Gets People (Body Indexing)
```
data.xquat[0]    = world body (SKIP this)
data.xquat[1]    = body 0 (pelvis, root)
data.xquat[2]    = body 1
...

Code: body_rot_wxyz = data.xquat[1:]  # Skip world
```

### The Silent Failure Modes (Check These When Debugging)
1. Using `data.cvel[0, 0:3]` instead of `data.qvel[3:6]` for angular velocity
2. Forgetting the +1 offset on xquat indexing
3. Quaternion wxyz ↔ xyzw mismatch
4. Mixing root (pelvis) and anchor (torso) bodies
5. Not calling `mj_forward()` after setting qpos

---

## 📁 File Locations

All documents are in:
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
```

**Reference code:**
```
ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py
├─ Lines 1-100: Imports, docstring, constants
├─ Lines 327-370: get_robot_state_from_mujoco()
├─ Lines 373-392: set_initial_pose()
├─ Lines 400-439: build_onnx_inputs()
└─ Lines 660-776: Main loop
```

---

## 🔗 Cross-References Between Documents

### `body_pose_quick_reference.md` references:
- Line 353: `data.xquat[1:].copy()` (see code_flow_reference.md)
- Quaternion conversion (see body_pose_capture_analysis.md)

### `body_pose_capture_analysis.md` references:
- Core function (lines 327-370, see code_flow_reference.md for details)
- ONNX assembly (lines 400-439, see code_flow_reference.md)

### `code_flow_reference.md` references:
- Every line number annotated
- Call stack shows relationships
- Critical implementation details with references

---

## ✅ Verification Checklist

Use this before implementing:

- [ ] I understand wxyz vs xyzw convention
- [ ] I know the +1 offset for body_rot indexing
- [ ] I know `data.qvel[3:6]` is already body-local (no rotation needed)
- [ ] I understand anchor body (torso) vs root (pelvis)
- [ ] I know `mj_forward()` must be called after setting qpos
- [ ] I understand ONNX input shapes (with batch dim)
- [ ] I know post-processing applies accel clamp then EMA filter
- [ ] I understand decimation (4 physics substeps per control step)

---

## 🚀 Next Steps

1. **First time?** Read `body_pose_quick_reference.md` (5 min)
2. **Need implementation?** Read `body_pose_capture_analysis.md` (15 min)
3. **Debugging?** Reference `code_flow_reference.md` (on-demand)
4. **Ready to code?** Copy dimensions from Reference section, use code_flow_reference.md for line numbers

---

## 📝 Document Statistics

- **Total documentation:** ~37 KB
- **Code examples:** ~15
- **Tables:** ~8
- **Diagrams:** ~5
- **Line number references:** ~50+
- **Critical gotchas identified:** ~15

---

## Questions?

If something is unclear:
1. Check the relevant section in the appropriate document
2. Look at the cross-references
3. Reference the actual code in test_tracker_mujoco.py
4. All three documents together provide complete coverage

**Happy deploying! 🚀**

