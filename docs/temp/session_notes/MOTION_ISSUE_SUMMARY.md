# V3 Reference Motion Data - Root Cause Analysis

## Problem Statement
User reports: "reference motion完全是不对的" (reference motion is completely wrong)

## Investigation Results

### ✅ CONFIRMED: Reference motions in v3 are fundamentally different from v2

### Key Numerical Differences (t2m_00000001)

| Metric | V2 | V3 | Delta | Issue |
|--------|----|----|-------|-------|
| **Root Pos X** | 0.0632 | 0.2862 | +0.2230 | ❌ Wrong |
| **Root Pos Y** | -0.0112 | 0.0168 | +0.0280 | ❌ Wrong |
| **Root Pos Z (Height)** | 0.7503 | 0.6828 | -0.0675 | ❌ Lower |
| **Root Quat X** | 0.0248 | 0.0910 | +0.0662 | ❌ Wrong rotation |
| **Root Quat Y** | -0.0166 | 0.0343 | +0.0509 | ❌ Wrong rotation |
| **Mean Height** | 0.7357 | 0.5942 | **-19.2%** | ❌ **19% too short!** |
| **Height Range** | 0.0530 | 0.2025 | +282% | ❌ Oscillates 5.8x more |

### Critical Findings

1. **Height Problem**: V3 character is 19.2% shorter than v2
   - Mean height: 0.7357 → 0.5942 (too low)
   - Likely indicates wrong character model or scale applied

2. **Position Mismatch**: Root X,Y positions are completely different
   - Not a simple offset or frame swap
   - Suggests coordinate transformation error OR different source motion

3. **Rotation Wrong**: Quaternion values differ significantly
   - Not just axis reordering
   - Actual different orientation in space

4. **Height Oscillation**: 5.8x larger than v2
   - V3: min=0.4803, max=0.6828 (range=0.2025)
   - V2: min=0.7025, max=0.7555 (range=0.0530)
   - Character seems to crouch/jump more in v3

5. **NPZ Frame Mismatch**:
   - NPZ file has 60 frames
   - JSON file has 99 frames
   - Inconsistent!

### Manifest Comparison

**V2 Manifest** (Likely correct):
```
- Simple structure (id, num_frames, fps)
- No model info
- Implies raw reference motions
```

**V3 Manifest** (Suspicious):
```
- Rich with metrics: "model": "HyMotion T2M 1.0-Lite"
- Contains: root_height_mean, max_joint_velocity, "fell" flag
- Looks like GENERATED OUTPUT, not reference!
```

## Root Cause Hypothesis

The v3 `/data/motions/` directory contains **GENERATED or TRANSFORMED motions**, NOT the original reference motions.

This could be due to:

### Option A: Wrong Data Source
- V2 imported CMU mocap raw motions → processed to SMPL-X
- V3 regenerated using v3 model instead of using v2 reference
- Result: Different character, different poses

### Option B: Transform Error
- Coordinate frame conversion went wrong
- Scale factor applied incorrectly
- Character model swap (different SMPL-X variant?)

### Option C: Retargeting Bug
- CMU mocap → SMPL-X → KIMODO pipeline
- One step has incorrect parameters
- Resulted in scaled-down, rotated character

### Option D: Downsampling Error
- NPZ has 60 frames, JSON has 99 frames
- Interpolation or upsampling went wrong
- Character squashes/stretches incorrectly

## Evidence Supporting Option C (Most Likely)

1. **Height reduction**: -19.2% (classic retargeting scale issue)
2. **NPZ/JSON frame mismatch**: 60 vs 99 frames
3. **Height oscillation increase**: Indicates joint constraints not properly transferred
4. **Manifest model info**: v3 has explicit "HyMotion T2M 1.0-Lite" - generated, not reference

## Recommended Fix

### Priority 1: Restore v2 reference motions
```bash
# In v3 pipeline, replace /data/motions/ with v2 reference
cp -r output/embodied_comparison_v2/data/motions/ \
       output/embodied_t2m_v3/data/reference_motions_correct/
```

### Priority 2: Investigate v3 pipeline code
- Find where `/data/motions/` is being populated
- Check if it's reading v2 reference correctly
- Look for transformation/retargeting code
- Verify SMPL-X model initialization

### Priority 3: Fix the transformation pipeline
- Verify CMU mocap → SMPL-X conversion
- Check retargeting parameters
- Confirm frame interpolation logic
- Validate character model scale

## Files Analyzed

| Path | Status | Size | Notes |
|------|--------|------|-------|
| `output/embodied_t2m_v3/data/motions/t2m_00000001.json` | ❌ Wrong | 76KB | Generated output, not reference |
| `output/embodied_t2m_v3/data/tracked_motions/t2m_00000001.json` | ❌ Wrong | 75KB | Identical to motions (also wrong) |
| `output/embodied_t2m_v3/data/npz/t2m_00000001.npz` | ⚠️ Check | 2KB | 60 frames but JSON has 99 |
| `output/embodied_comparison_v2/data/motions/t2m_00000001.json` | ✅ Correct | 77KB | Proper reference baseline |
| `output/embodied_t2m_v3/batch_report.json` | ⚠️ Misleading | 27KB | Marks wrong data as "ref_json" |

## Key Insight

**The batch_report.json labels v3/data/motions as "ref_json" (reference), but the actual data appears to be GENERATED OUTPUT from the v3 pipeline, not the original reference!**

This is the core issue: v3 is comparing generated motions against generated motions (not real references).

