# Embodied Pipeline Retargeting - Bug Investigation Summary

**Investigation Date**: May 12, 2026  
**Scope**: 4 pipeline scripts + 10 bug categories identified  
**Severity**: 4 critical, 2 high, 2 medium, 2 low issues

---

## Files Analyzed

1. ✅ `scripts/embodied/motion135_to_smplx.py` (130 lines)
   - HyMotion 135-dim motion → SMPL-X NPZ
   - Bugs found: #5, #8

2. ✅ `scripts/embodied/gmr_retarget_headless.py` (157 lines)
   - SMPL-X NPZ → Robot joint positions via GMR
   - Related to: #1 (offset_to_ground flag)

3. ✅ `scripts/embodied/gmr_to_protomotions.py` (515 lines)
   - GMR output → ProtoMotions cache with FK
   - Bugs found: #2, #3, #4, #6, #7, #9, #10

4. ✅ `scripts/embodied/pipeline_motion_to_robot.py` (177 lines)
   - Orchestrator chaining 3 conversion steps
   - Bugs found: #1

---

## Key Findings

### 🔴 CRITICAL ISSUES (4)

#### Bug #1: Double-Failure Ground Correction Logic
- **File**: `pipeline_motion_to_robot.py:125`
- **Root Cause**: Disables per-frame grounding in GMR (`--no-offset-to-ground`), then tries to fix with FK correction that only adjusts Z, not joint angles
- **Impact**: Foot sliding, unnatural poses, joint angles don't match corrected root position
- **Fix**: Either (A) remove `--no-offset-to-ground` and disable FK correction OR (B) restructure pipeline to correct before retargeting
- **Complexity**: Medium (logic restructure)

#### Bug #2: Hardcoded Wrong Foot Body Indices
- **File**: `gmr_to_protomotions.py:184`
- **Root Cause**: Uses `[7, 13]` indices without verifying G1 MJCF structure
- **Impact**: FK correction checks wrong body heights, feet end up below/above ground
- **Fix**: Implement `get_foot_body_indices_from_mjcf()` to dynamically find foot bodies
- **Complexity**: Low

#### Bug #3: Wrong Body Index Offset in MuJoCo Access
- **File**: `gmr_to_protomotions.py:219`
- **Root Cause**: Uses `data.xpos[bi + 1][2]` but `data.xpos` already includes world body
- **Impact**: Potential array overflow, reads wrong body positions or garbage data
- **Fix**: Remove `+ 1` offset and add bounds checking
- **Complexity**: Low

#### Bug #4: Inconsistent Position/Rotation Frame Conversions
- **File**: `gmr_to_protomotions.py:69-111`
- **Root Cause**: Root position and rotation use different transformation conventions (active vs passive)
- **Impact**: Position and rotation don't align, pose misalignment in robot frame
- **Fix**: Verify GMR's frame conversion math, make both consistent, add test cases
- **Complexity**: Medium

### 🟡 HIGH PRIORITY ISSUES (2)

#### Bug #5: Unverified rot6d Layout Assumption
- **File**: `motion135_to_smplx.py:39`
- **Root Cause**: Claims HyMotion uses row-major [0,2,4,1,3,5] reorder but no verification
- **Impact**: Wrong rotation interpretation → distorted joint angles/limbs
- **Fix**: Add `validate_rot6d_layout()` test against known poses
- **Complexity**: Medium

#### Bug #6: FK Ground Correction Overcorrection
- **File**: `gmr_to_protomotions.py:207-228`
- **Root Cause**: Single-pass correction doesn't iterate or validate
- **Impact**: Can overshoot when starting from deformed poses
- **Fix**: Replace with iterative correction (max 3 iterations, tolerance 0.001m)
- **Complexity**: Medium

### 🟡 MEDIUM PRIORITY ISSUES (2)

#### Bug #7: No Joint Limit Clamping
- **File**: Throughout `gmr_to_protomotions.py`
- **Root Cause**: No validation that DOF angles stay within mechanical limits
- **Impact**: Deformed poses, mechanical violations
- **Fix**: Add `clamp_to_joint_limits()` after resampling
- **Complexity**: Low

#### Bug #8: No NPZ Input Validation
- **File**: `motion135_to_smplx.py:77-79`
- **Root Cause**: Assumes 'motion_135' key exists and has correct shape
- **Impact**: Silent failures or cryptic errors on bad input
- **Fix**: Add NPZ key validation and shape checking
- **Complexity**: Low

### 🟡 LOW PRIORITY ISSUES (2)

#### Bug #9: First Frame Velocity Edge Case
- **File**: `gmr_to_protomotions.py:363-365`
- **Root Cause**: Repeats first computed velocity instead of using central diff
- **Impact**: Minor animation discontinuity at start
- **Fix**: Use central differences or mirror velocity
- **Complexity**: Low

#### Bug #10: No Quaternion Normalization Check
- **File**: After frame conversions in `gmr_to_protomotions.py`
- **Root Cause**: Numerical errors can denormalize quaternions
- **Impact**: Rotation errors accumulate (low severity)
- **Fix**: Add normalize + assert after conversions
- **Complexity**: Low

---

## Pipeline Symptom → Bug Mapping

```
Symptom: Foot sliding ←→ Bug #1 (primary) + Bug #2 (feet wrong height)
Symptom: Ground penetration ←→ Bug #1 + Bug #2 + Bug #6
Symptom: Deformed poses ←→ Bug #5 (joints bent wrong) + Bug #7 (at limits)
Symptom: Unnatural tall poses ←→ Bug #1 + Bug #6 (overcorrection)
Symptom: Joint angles at limits ←→ Bug #7
```

---

## Immediate Action Items

### Must-Fix (Phase 1)
1. Decide on ground correction strategy for Bug #1
2. Implement dynamic foot body index lookup (Bug #2)
3. Verify MuJoCo body index offset (Bug #3)
4. Test frame conversion consistency (Bug #4)

### Should-Fix (Phase 2)
5. Validate rot6d layout with test (Bug #5)
6. Switch to iterative FK correction (Bug #6)

### Nice-To-Fix (Phase 3)
7. Add joint limit clamping (Bug #7)
8. Add input validation (Bug #8)
9. Fix edge cases (Bugs #9, #10)

---

## Documentation Artifacts Created

1. **EMBODIED_PIPELINE_BUG_ANALYSIS_DETAILED.md** - Full technical analysis
2. **EMBODIED_PIPELINE_DEBUGGING_GUIDE.md** - Debugging tools and scripts
3. **EMBODIED_PIPELINE_FIX_PROPOSALS.md** - Code-level fix proposals
4. **This file** - Executive summary

---

## Verification Checklist

After implementing fixes:

- [ ] Run pipeline with `--keep-intermediates` to inspect each stage
- [ ] Check GMR PKL: are feet above ground (Z > 0)?
- [ ] Check quaternion norms: all close to 1.0?
- [ ] Check joint angles: within valid limits?
- [ ] Check final cache: no ground penetration, velocities smooth?
- [ ] Run ONNX tracker validation: poses plausible?
- [ ] Visual inspection: limbs bent naturally, not deformed?
- [ ] Test with known motion: compare to reference?

---

## References

**Pipeline Configuration**:
- Default MJCF: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml`
- Default control dt: 0.02s (50Hz)
- Default FPS: 30Hz
- Current flags: `--no-offset-to-ground`, `--fk-ground-correction` (default True)

**Key Math**:
- GMR rot_offset: [0.5, -0.5, -0.5, -0.5] (wxyz) - 120° Y-up→Z-up conversion
- Frame axes:
  - SMPL-X: X-right, Y-up, Z-forward
  - MuJoCo: X-forward, Y-left, Z-up
- Quaternion format: xyzw (after conversion from GMR's wxyz)

**Related Code**:
- GMR source: `ref_repo/GMR/`
- ProtoMotions source: `ref_repo/ProtoMotions/`
- HyMotion M2M: Check for rot6d layout in motion_135 generation

