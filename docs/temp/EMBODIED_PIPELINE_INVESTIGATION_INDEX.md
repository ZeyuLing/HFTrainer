# Embodied Pipeline Retargeting Investigation - Complete Documentation Index

**Status**: ✅ INVESTIGATION COMPLETE  
**Date**: May 12, 2026  
**Bugs Identified**: 10 (4 Critical, 2 High, 2 Medium, 2 Low)  
**Total Documentation**: 2,743 lines across 7 files  

---

## 📋 Quick Navigation

### For Executives/Managers
👉 **START HERE**: `EMBODIED_PIPELINE_INVESTIGATION_SUMMARY.md`
- High-level overview
- Impact assessment
- Timeline estimates
- Action items by priority

### For Engineers Implementing Fixes
👉 **START HERE**: `EMBODIED_PIPELINE_FIX_PROPOSALS.md`
- Code-level fix proposals
- Before/after code examples
- Complexity estimates
- Priority ordering

### For Debugging Issues
👉 **START HERE**: `EMBODIED_PIPELINE_DEBUGGING_GUIDE.md`
- Symptom diagnosis flowchart
- Step-by-step debugging commands
- Inspection scripts (Python)
- Common error solutions

### For Comprehensive Technical Analysis
👉 **START HERE**: `EMBODIED_PIPELINE_BUG_ANALYSIS_DETAILED.md`
- Full 10-bug breakdown
- Root cause analysis
- Why each bug causes the reported symptoms
- Recommended fix priority

---

## 📁 Document Breakdown

### 1. **EMBODIED_PIPELINE_INVESTIGATION_SUMMARY.md** (185 lines)
**Best for**: Quick understanding + action items

**Contents**:
- Executive summary of all 10 bugs
- Files analyzed
- Key findings
- Pipeline symptom → bug mapping
- Immediate action items (3 phases)
- Verification checklist

**Read if**: You need to know what's broken and what to do about it


### 2. **EMBODIED_PIPELINE_BUG_ANALYSIS_DETAILED.md** (378 lines)
**Best for**: Deep technical understanding

**Contents**:
- Detailed explanation of each of 10 bugs
- Code locations (file:line)
- Why each bug causes reported symptoms
- Impact assessment
- Fix requirements
- Summary table
- Testing strategy

**Read if**: You want to understand the technical details before fixing


### 3. **EMBODIED_PIPELINE_FIX_PROPOSALS.md** (595 lines)
**Best for**: Implementing the fixes

**Contents**:
- Before/after code for each bug
- Specific functions to add/modify
- Exact line numbers to change
- Alternative fix approaches
- Priority ordering
- Complexity estimates

**Read if**: You're implementing the fixes and want concrete code


### 4. **EMBODIED_PIPELINE_DEBUGGING_GUIDE.md** (363 lines)
**Best for**: Validating bugs and testing fixes

**Contents**:
- Symptom diagnosis flowchart
- Debugging commands for each stage
- Python inspection scripts
- MuJoCo body index utilities
- Joint limit extraction
- Common error messages + solutions
- Systematic debugging checklist
- Performance impact table

**Read if**: You need to verify a bug or test that a fix worked


### 5. **EMBODIED_PIPELINE_BUG_ANALYSIS.md** (566 lines)
**Best for**: Comprehensive reference

**Contents**:
- Full detailed analysis (similar to #2 but standalone format)
- Better for printing/archiving
- Complete reference material

**Read if**: You want one comprehensive standalone file


### 6. **EMBODIED_PIPELINE_FIXES.md** (490 lines)
**Status**: Alternative/supplementary format of fix proposals

**Read if**: You prefer a different organizational structure


### 7. **EMBODIED_PIPELINE_QUICKREF.md** (166 lines)
**Best for**: Quick reference cards

**Contents**:
- One-page summaries
- Quick lookup tables
- Shorthand descriptions

**Read if**: You need to remember something quickly during coding


---

## 🎯 How to Use This Investigation

### Scenario 1: "I need to fix the bugs ASAP"
1. Read `EMBODIED_PIPELINE_INVESTIGATION_SUMMARY.md` (5 min)
2. Open `EMBODIED_PIPELINE_FIX_PROPOSALS.md` (as reference while coding)
3. Use `EMBODIED_PIPELINE_DEBUGGING_GUIDE.md` to test fixes

**Total time**: ~1 hour to understand + implement Phase 1 fixes

### Scenario 2: "I need to understand what's wrong"
1. Read `EMBODIED_PIPELINE_INVESTIGATION_SUMMARY.md` (5 min)
2. Read `EMBODIED_PIPELINE_BUG_ANALYSIS_DETAILED.md` (20 min)
3. Skim `EMBODIED_PIPELINE_DEBUGGING_GUIDE.md` to understand testing

**Total time**: ~30 min for complete understanding

### Scenario 3: "I need to debug specific problems"
1. Reference `EMBODIED_PIPELINE_DEBUGGING_GUIDE.md` symptom flowchart
2. Run appropriate debugging commands from that file
3. Use Python inspection scripts to validate hypothesis
4. Cross-reference with `EMBODIED_PIPELINE_BUG_ANALYSIS_DETAILED.md` for confirmation

**Total time**: ~10 min per issue

### Scenario 4: "I need to verify fixes work"
1. Use checklist from `EMBODIED_PIPELINE_INVESTIGATION_SUMMARY.md`
2. Run validation steps from `EMBODIED_PIPELINE_DEBUGGING_GUIDE.md`
3. Check specific measurements against bug descriptions

**Total time**: ~30 min for comprehensive verification

---

## 🐛 Bug Summary Table

| # | Severity | Component | Issue | Fix Time |
|---|----------|-----------|-------|----------|
| 1 | 🔴 CRIT | Pipeline | Ground correction double-failure | 30min |
| 2 | 🔴 CRIT | FK correction | Hardcoded wrong foot indices | 20min |
| 3 | 🔴 CRIT | FK correction | Body index offset error | 15min |
| 4 | 🔴 CRIT | Frame conversion | Position/rotation inconsistency | 45min |
| 5 | 🟡 HIGH | motion135_to_smplx | Unverified rot6d layout | 30min |
| 6 | 🟡 HIGH | FK correction | Overcorrection (single-pass) | 40min |
| 7 | 🟡 MED | gmr_to_protomotions | No joint limit clamping | 20min |
| 8 | 🟡 MED | motion135_to_smplx | No input validation | 15min |
| 9 | 🟡 LOW | gmr_to_protomotions | First frame velocity | 10min |
| 10 | 🟡 LOW | gmr_to_protomotions | Quaternion normalization | 10min |

**Total estimated fix time**: 3.5 hours (if done sequentially)

---

## 📊 Files Analyzed

```
scripts/embodied/
├── motion135_to_smplx.py        (130 lines) - Bugs #5, #8
├── gmr_retarget_headless.py     (157 lines) - Related to #1
├── gmr_to_protomotions.py       (515 lines) - Bugs #2, #3, #4, #6, #7, #9, #10
└── pipeline_motion_to_robot.py  (177 lines) - Bug #1
```

---

## 🔑 Key Concepts

### Coordinate System Mismatch
- **SMPL-X**: Y-up (X-right, Y-up, Z-forward)
- **MuJoCo**: Z-up (X-forward, Y-left, Z-up)
- **GMR rot_offset**: [0.5, -0.5, -0.5, -0.5] (wxyz) - 120° rotation to convert

**Critical**: Position and rotation conversions must be consistent or poses misalign

### Quaternion Formats
- **GMR output**: wxyz format (w first)
- **ProtoMotions**: xyzw format (w last)
- **Conversion**: `[w, x, y, z] → [x, y, z, w]`

**Critical**: Wrong conversion causes inverted rotations

### Ground Correction Strategy
- **Option A (Recommended)**: GMR's `offset_to_ground=True` handles grounding in IK solver
- **Option B (Buggy)**: Disable in GMR, try to fix with FK afterward (current)
- **Why B fails**: FK only adjusts Z, not joint angles → foot sliding

### rot6d Layout
- **Row-major**: `[R00, R01, R10, R11, R20, R21]`
- **Column-major**: `[R00, R10, R20, R01, R11, R21]`
- **Pipeline assumption**: HyMotion uses row-major, applies reorder [0,2,4,1,3,5]
- **Risk**: If wrong → rotations inverted → limbs bent incorrectly

---

## ✅ Verification Checklist

After implementing all fixes:

- [ ] Motion loads without crashes
- [ ] GMR output: feet above ground (Z > 0) throughout
- [ ] Quaternions: all have norm ≈ 1.0
- [ ] Joint angles: all within valid ranges
- [ ] Final cache: no body below Z=-0.01m
- [ ] Velocities: smooth, no sudden jumps
- [ ] Visual inspection: limbs bend naturally
- [ ] ONNX tracker validation: passes with good confidence
- [ ] Performance: pipeline completes in <1s per frame

---

## 📞 Questions to Ask

### If stuck implementing Bug #1 fix:
- Does GMR have issues with `offset_to_ground=True`?
- Is FK correction essential for something?
- Check GMR source: `ref_repo/GMR/fbx_offline_to_robot.py`

### If stuck implementing Bug #2 fix:
- Are there other foot-like bodies in G1 MJCF?
- Check XML: `ref_repo/ProtoMotions/.../g1_holo_compat.xml`
- Use MuJoCo script to list all bodies and their names

### If stuck implementing Bug #4 fix:
- What exactly does GMR's IK config apply?
- Check: `ref_repo/GMR/.../configs/smplx_to_g1.json`
- Test with identity quaternion and known vectors

### If stuck implementing Bug #5 fix:
- Get reference pose from HyMotion
- Verify T-pose produces correct joint angles
- Check HyMotion source: motion generation code

---

## 📚 External References

**GMR Repository**:
- Main: `ref_repo/GMR/`
- IK config: `ref_repo/GMR/.../configs/smplx_to_g1.json`
- Offline conversion: `ref_repo/GMR/fbx_offline_to_robot.py`

**ProtoMotions Repository**:
- Main: `ref_repo/ProtoMotions/`
- MJCF: `ref_repo/ProtoMotions/.../g1_holo_compat.xml`
- Test utilities: `ref_repo/ProtoMotions/deployment/test_tracker_mujoco.py`

**MuJoCo Documentation**:
- Body indexing: `data.xpos[body_id]` includes world at 0
- Quaternion format: wxyz (w magnitude, xyz vector)
- FK: `mujoco.mj_forward(model, data)`

---

## 🚀 Next Steps

1. **Read** `EMBODIED_PIPELINE_INVESTIGATION_SUMMARY.md` (5 min)
2. **Decide** on fix priority (stakeholder discussion)
3. **Assign** bugs to team members
4. **Implement** fixes using `EMBODIED_PIPELINE_FIX_PROPOSALS.md`
5. **Test** using `EMBODIED_PIPELINE_DEBUGGING_GUIDE.md`
6. **Validate** with checklist
7. **Document** findings and lessons learned

---

## 📝 Investigation Metadata

- **Investigator**: Claude Code
- **Investigation Start**: May 12, 2026
- **Investigation End**: May 12, 2026
- **Files Reviewed**: 4 pipeline scripts, ~1000 lines of code
- **Bugs Found**: 10 (4 Critical)
- **Root Causes Identified**: Coordinate system mismatches, quaternion convention errors, hardcoded indices, unverified assumptions
- **Confidence Level**: High (code analysis + referenced documentation)
- **Ready for Implementation**: Yes

---

**Status**: ✅ COMPLETE - Ready for developer handoff

