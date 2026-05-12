# Motion Retargeting Height Estimation Analysis

## Problem Summary

The motion retargeting pipeline always estimates human height as **1.66m** because:
1. In `scripts/embodied/motion135_to_smplx.py`, betas are set to zeros: `betas=np.zeros(10, dtype=np.float32)`
2. In `ref_repo/GMR/general_motion_retargeting/utils/smpl.py`, height is computed as:
   ```python
   human_height = 1.66 + 0.1 * betas[0]
   ```
3. Since `betas[0] = 0`, height defaults to 1.66m regardless of actual human size

This causes incorrect IK scaling in `ref_repo/GMR/general_motion_retargeting/motion_retarget.py`:
```python
if actual_human_height is not None:
    ratio = actual_human_height / ik_config["human_height_assumption"]
else:
    ratio = 1.0  # <-- No scaling applied!
    
for key in ik_config["human_scale_table"].keys():
    ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
```

## Data Format Analysis

### Motion_135 Format (HyMotion Output)
- **Total dims**: 135 = 3 (translation) + 22×6 (rot6d) = 3 + 132
- **No local joint positions** included in the standard 135-dim representation
- Translation (first 3 dims): Global position in world space
- Rot6d (dims 3-134): 22 joints × 6D rotation representations (row-major layout)

### Available Joint Information
22 SMPL-X body joints (indices 0-21):
```
 0: pelvis          (root)        7: left_ankle       14: right_collar
 1: left_hip                       8: right_ankle      15: head
 2: right_hip                      9: spine3           16: left_shoulder
 3: spine1          10: left_foot  17: right_shoulder
 4: left_knee       11: right_foot 18: left_elbow
 5: right_knee      12: neck       19: right_elbow
 6: spine2          13: left_collar 20: left_wrist
                                    21: right_wrist
```

### Skeleton Chain for Height Estimation
Optimal chain for height (vertical Y-axis in standard 3D):
- **Pelvis (0)** → Left_Hip (1) / Right_Hip (2)
- → Left_Knee (4) / Right_Knee (5)
- → Left_Ankle (7) / Right_Ankle (8)
- → Left_Foot (10) / Right_Foot (11)
- **Pelvis (0)** → Spine1 (3) → Spine2 (6) → Spine3 (9) → Neck (12) → Head (15)

**Key insight**: Head height - Foot height ≈ human height

## Current Height Estimation Approaches

### Existing Implementation: XRobotRecorder.get_human_height()
Located in `ref_repo/GMR/general_motion_retargeting/xrobot_utils.py` (lines 774-820)

**What it does**:
- Analyzes Y-coordinates across all joints in body_data
- Computes frame_height = max(y_positions) - min(y_positions)
- Returns maximum height across all frames with 10% buffer

**Why not directly applicable**:
- Designed for XRobot body tracking data (24 joints with actual 3D positions)
- Works on world-space joint positions extracted from XRobot
- Motion_135 doesn't include local joint positions, only rotation + global translation

**Code reference**:
```python
# From xrobot_utils.py lines 790-820
for body_data in self.processed_body_data:
    y_positions = []
    for joint_data in body_data.values():
        y_pos = joint_data[0][1]  # Y coordinate of position
        y_positions.append(y_pos)
    frame_height = max(y_positions) - min(y_positions)
```

## Solution Approaches

### Option A: SMPL-X Forward Kinematics (FK) - RECOMMENDED
**Proposed Flow**:
1. In `motion135_to_smplx.py`:
   - Convert motion_135 to SMPL-X NPZ without betas modification
   - Extract joint positions from `smplx_output.joints` (world-space via FK)
   - Estimate height from pelvis-to-head distance
   - Store as custom field: `height_estimate`

2. In `load_smplx_file()` in `utils/smpl.py`:
   - After FK computation, measure: `height = joints[head_idx].max() - joints[feet_idx].min()`
   - Return actual measured height instead of formula

**Advantages**:
- ✅ Leverages existing SMPL-X model
- ✅ Accounts for actual skeletal proportions
- ✅ Measurable from FK output
- ✅ Non-intrusive (doesn't modify betas)
- ✅ Consistent across all motion sources

**Implementation Example**:
```python
# In load_smplx_file() after FK
joints_np = smplx_output.joints.detach().numpy()  # (T, 22, 3)

# Head is joint 15, Feet are joints 10, 11
# Measure from pelvis (0) for consistency
head_z = joints_np[:, 15, 2].mean()  # Mean head z across frames
foot_z = min(joints_np[:, 10, 2].min(), joints_np[:, 11, 2].min())
human_height = head_z - foot_z

# Validation
if not (1.3 <= human_height <= 2.3):
    human_height = 1.7  # Fallback to average
    
return smplx_data, body_model, smplx_output, human_height
```

### Option B: Using Local Positions Field (If Available)
**Assumption**: If motion data includes local joint positions as extra field (66 dims = 22×3)

**Limb Length Formula**:
```python
if 'positions' in data:  # Local joint positions relative to root
    positions = data['positions']  # (T, 22, 3)
    
    # Measure leg length: pelvis → ankle
    leg_vector = positions[:, 7, :] - positions[:, 0, :]  # left ankle - pelvis
    leg_length = np.linalg.norm(leg_vector, axis=1).mean()
    
    # Measure torso: pelvis → head
    torso_vector = positions[:, 15, :] - positions[:, 0, :]  # head - pelvis
    torso_length = np.linalg.norm(torso_vector, axis=1).mean()
    
    # Estimate height
    human_height = 2 * leg_length + torso_length
```

**Advantages**:
- ✅ No FK needed (purely data-driven)
- ✅ Fast (no model inference)

**Disadvantages**:
- ❌ Requires additional position field in motion data
- ❌ Less accurate without full skeleton

### Option C: Hybrid Approach (Best Practice)

**Step 1**: Store height in SMPL-X NPZ
```python
# In motion135_to_smplx.py
data = np.load(input_npz)
motion = data['motion_135']

# Check for optional positions field
if 'positions' in data:
    positions = data['positions']
else:
    positions = None

np.savez(
    output_npz,
    pose_body=...,
    betas=np.zeros(10),
    height_estimate=height_estimate,  # <-- NEW FIELD
    positions=positions,               # <-- NEW FIELD (optional)
    ...
)
```

**Step 2**: Compute height in `load_smplx_file()`
```python
def load_smplx_file(smplx_file, smplx_body_model_path):
    smplx_data = np.load(smplx_file, allow_pickle=True)
    body_model = smplx.create(...)
    
    # Perform FK
    smplx_output = body_model(...)
    
    # Method 1: Use FK if available
    joints_np = smplx_output.joints.detach().numpy()  # (T, 22, 3)
    head_y = joints_np[:, 15, 1].mean()
    foot_y = min(joints_np[:, 10, 1].min(), joints_np[:, 11, 1].min())
    human_height_fk = head_y - foot_y
    
    # Method 2: Use stored estimate (fallback)
    if 'height_estimate' in smplx_data:
        human_height = float(smplx_data['height_estimate'])
    elif 'positions' in smplx_data:
        # Compute from local positions
        positions = smplx_data['positions']
        leg_length = np.linalg.norm(positions[:, 7] - positions[:, 0], axis=1).mean()
        torso_length = np.linalg.norm(positions[:, 15] - positions[:, 0], axis=1).mean()
        human_height = 2 * leg_length + torso_length
    else:
        # Fallback to FK measurement
        human_height = human_height_fk
    
    # Validation
    if not (1.3 <= human_height <= 2.3):
        human_height = 1.7
    
    return smplx_data, body_model, smplx_output, human_height
```

**Step 3**: Pass to GMR
```python
# In scripts/embodied/smplx_to_robot.py (or equivalent)
_, body_model, smplx_output, human_height = load_smplx_file(...)
gmr = GeneralMotionRetargeting(
    src_human="smplx",
    tgt_robot=robot_name,
    actual_human_height=human_height  # <-- NOW USED!
)
```

## Recommendation Summary

| Aspect | Option A (FK) | Option B (Local Pos) | Option C (Hybrid) |
|--------|---------------|---------------------|------------------|
| Accuracy | High (±2-3cm) | Medium | **High** |
| Speed | Medium (1-5s/clip) | Fast (instant) | **Medium** |
| Dependencies | SMPL-X model | Motion data | **SMPL-X model** |
| Robustness | High | Medium | **High** |
| Implementation | Simple | Simple | **Simple** |
| **Recommendation** | ⚠️ | ⚠️ | ✅ |

## Implementation Checklist

- [ ] **Phase 1**: Implement Option A (FK-based height)
  - [ ] Modify `load_smplx_file()` to measure height from joints
  - [ ] Add validation (1.3-2.3m range)
  - [ ] Add logging with estimated height value
  - [ ] Test on sample motion clips

- [ ] **Phase 2**: Add height field to SMPL-X NPZ (optional)
  - [ ] Modify `motion135_to_smplx.py` to compute and store height
  - [ ] Add fallback validation in load_smplx_file()

- [ ] **Phase 3**: Integration test
  - [ ] Run full pipeline: motion_135 → NPZ → GMR
  - [ ] Verify `actual_human_height` is passed to GMR
  - [ ] Check that human_scale_table is correctly scaled
  - [ ] Compare robot motion before/after fix

## Files to Modify

1. **`ref_repo/GMR/general_motion_retargeting/utils/smpl.py`**
   - `load_smplx_file()` function (lines 14-55)
   - `load_gvhmr_pred_file()` function (lines 58-110) - similar fix
   - Replace hardcoded height formula with FK measurement

2. **`scripts/embodied/motion135_to_smplx.py`** (Optional for Phase 2)
   - Add height computation before NPZ save
   - Store `height_estimate` field

3. **Entry point script** (e.g., any script calling GeneralMotionRetargeting)
   - Ensure `actual_human_height` is passed to GMR constructor
   - Add logging for verified height values

## Testing Strategy

```python
# Test script
import numpy as np
from ref_repo.GMR.general_motion_retargeting.utils.smpl import load_smplx_file

# Test 1: Height measurement accuracy
_, _, _, height = load_smplx_file("sample.npz", "path/to/models")
assert 1.3 <= height <= 2.3, f"Height {height} out of reasonable range"
print(f"✓ Height measurement: {height:.2f}m")

# Test 2: GMR scaling
from ref_repo.GMR.general_motion_retargeting import GeneralMotionRetargeting
gmr = GeneralMotionRetargeting(
    src_human="smplx",
    tgt_robot="unitree_g1",
    actual_human_height=height
)
# Verify human_scale_table was scaled correctly
original_scale = 1.7
ratio = height / original_scale
print(f"✓ GMR scaling applied: ratio={ratio:.3f}")

# Test 3: Retargeting quality
# Compare robot motions with different heights
```

## Edge Cases & Fallbacks

| Scenario | Handling |
|----------|----------|
| **FK fails/NaN** | Fallback to 1.7m default + warning |
| **Height < 1.3m** | Likely data corruption, use 1.7m default |
| **Height > 2.3m** | Likely measurement error, use 1.7m default |
| **Motion too short** | Still measurable (use mean across frames) |
| **Feet not visible** | Use min(10,11,7,8) joints for foot position |

## Related Code References

### SMPL-X Joint Names (22 body joints)
```
0: pelvis, 1: left_hip, 2: right_hip, 3: spine1,
4: left_knee, 5: right_knee, 6: spine2,
7: left_ankle, 8: right_ankle, 9: spine3,
10: left_foot, 11: right_foot, 12: neck,
13: left_collar, 14: right_collar, 15: head,
16: left_shoulder, 17: right_shoulder,
18: left_elbow, 19: right_elbow,
20: left_wrist, 21: right_wrist
```

### FK Output Structure
```python
smplx_output.joints  # (T, 22, 3) world-space positions
smplx_output.vertices  # (T, 10475, 3) mesh vertices
smplx_output.global_orient  # (T, 3) root rotation
smplx_output.full_pose  # (T, 156) all pose params
```

### GMR Scaling Logic
```python
# motion_retarget.py lines 62-70
if actual_human_height is not None:
    ratio = actual_human_height / ik_config["human_height_assumption"]
else:
    ratio = 1.0  # <-- Currently always 1.0 due to 1.66m hardcoding

for key in ik_config["human_scale_table"].keys():
    ik_config["human_scale_table"][key] *= ratio
```

## Key Insights

1. **Height is fundamental to IK accuracy**: Without proper scaling, the robot will have incorrect limb lengths and joint ranges

2. **SMPL betas ≠ height**: SMPL betas encode shape (weight, muscle), not skeleton scale

3. **FK-based measurement is reliable**: SMPL-X FK gives joint positions that accurately reflect the skeleton scale

4. **Default 1.66m is arbitrary**: It's hardcoded as a fallback, not based on any particular assumption

5. **Height should be per-motion**: Different motion sequences may have been captured from different subjects

## FAQ

**Q: Why can't we just use SMPL betas?**
A: SMPL betas encode shape (weight, muscle), not scale. Height comes from skeleton scale which SMPL doesn't learn.

**Q: What if motion data includes multiple people?**
A: Current setup assumes single person per clip. For multi-person, measure height per trajectory.

**Q: How accurate is FK-based height?**
A: ±2-3cm typical error due to:
- Ground plane estimation (foot contact vs. full foot-ground contact)
- Head pose variation (tilted head changes measurement)
- Mesh topology (joints at bone centers, not actual surface)

**Q: Can we use GVHMR predicted height?**
A: Possibly - check if GVHMR predictions include shape parameters. See `load_gvhmr_pred_file()` for integration point.

**Q: What coordinate system is used?**
A: Check SMPL-X convention: typically Z or Y for vertical. Measure max-min across all frames to be safe.
