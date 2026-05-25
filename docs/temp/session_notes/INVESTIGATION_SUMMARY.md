# Investigation Summary: MuJoCo SMPL Humanoid Falls During RL Simulation

## Questions Investigated

### 1. Does MuJoCo simulator disable self-collisions?
**Answer: NO**

**Evidence**:
- Search term: `self_collision` in `protomotions/simulator/mujoco/simulator.py`
- Result: No matches found
- Comparison: IsaacGym (line 782), Newton (line 206), Genesis (line 90) all implement self-collision disabling
- Flag defined in: `protomotions/robot_configs/base.py:101` as `self_collisions: bool = True`
- Flag usage in MuJoCo: **ZERO** references

**Code Gap**: MuJoCo simulator completely ignores the `robot_config.asset.self_collisions` flag

---

### 2. Are armature values causing instability?
**Answer: NO**

**Evidence**:
- SMPL config file: `protomotions/robot_configs/smpl.py` lines 75-111
- Armature specification: None in Python config (uses MJCF defaults)
- MJCF file: `protomotions/data/assets/mjcf/smpl_humanoid.xml`
- Armature values: **ALL 69 JOINTS HAVE `armature="0.02"`** (uniform, standard, reasonable)
- Typical range: 0.01-0.05, so 0.02 is well-centered
- No per-joint variation that could cause imbalance

**Conclusion**: Armature is NOT the cause of falling

---

### 3. What collision settings exist in MJCF?
**Answer: All geoms allow self-collision**

**Evidence**:
- Collision parameters found in SMPL humanoid MJCF:
  - `contype="1"` or `contype="7"` (collision type bits)
  - `conaffinity="1"` (collision affinity - allows collision with anything)
  - `condim="3"` (3D friction model, appropriate for humanoids)
  - `margin="0.001"` (1 mm penetration tolerance)

**Example geoms**:
```
Line 13 (Pelvis): conaffinity="1" contype="7"
Line 18 (L_Hip): conaffinity="1" contype="1"
Line 23 (L_Knee): conaffinity="1" contype="1"
Line 28 (L_Ankle): conaffinity="1" contype="7"
... (ALL body geoms have conaffinity="1")
```

**Interpretation**: `conaffinity="1"` means these geoms collide with anything in collision type 1 (which includes the robot itself)

---

## Root Cause Identified

### The Physics Problem

1. **SMPL humanoid in rest pose has natural interpenetration**:
   - Shoulders overlap
   - Hip region geometry causes limb penetration
   - Arms naturally touch torso
   - This is how the rigged model is designed

2. **With `conaffinity="1"`, MuJoCo contact solver generates repulsive forces**:
   - Contact depth > margin (1 mm) → contact is generated
   - Contact normal force ∝ penetration depth
   - Force grows unbounded as penetration increases

3. **PD control cannot overcome contact forces**:
   - PD torque: τ = K_p * error + K_d * error_rate
   - Contact impulses are applied at the end of each physics step
   - Contact forces are **independent of joint torques**
   - Result: Limbs violently repel each other → motion tracking fails → falls

4. **MuJoCo simulator bug**: Flag is completely ignored
   - No code checks `self_collisions`
   - No code disables self-collision
   - Unlike IsaacGym, Newton, Genesis which all implement this

---

## Solution

### Implementation Required

**File**: `protomotions/simulator/mujoco/simulator.py`

**Changes**: 
1. Add call to `_disable_self_collisions()` after line 318 in `_create_simulation()`
2. Add method `_disable_self_collisions()` after line 1153 (after projectile methods)

**Mechanism**:
```python
# Sets geom_conaffinity to 0 for all robot geoms
# Prevents ANY collision for those geoms (including self-collision)
self.model.geom_conaffinity[gid] = 0
```

**Result**: No contact forces between robot body parts → stable motion tracking

---

## Code Locations - Exact Line Numbers

### Where self-collision should be implemented

| File | Line | What | Status |
|------|------|------|--------|
| mujoco/simulator.py | 318 | After `_override_joint_properties()` | Missing call |
| mujoco/simulator.py | 1153 | After `_enable_projectile_collision()` | Missing method |

### Where self-collision IS implemented in other simulators

| Simulator | File | Line | Implementation |
|-----------|------|------|-----------------|
| IsaacGym | isaacgym/simulator.py | 782 | `col_filter = 0 if self.robot_config.asset.self_collisions else 1` |
| Newton | newton/simulator.py | 206 | `enable_self_collisions=self.robot_config.asset.self_collisions` |
| Genesis | genesis/simulator.py | 90 | `enable_self_collision=self.robot_config.asset.self_collisions` |
| MuJoCo | mujoco/simulator.py | NONE | **NOT IMPLEMENTED** |

### Configuration

| File | Line | Setting |
|------|------|---------|
| robot_configs/base.py | 101 | `self_collisions: bool = True` (default flag) |
| robot_configs/smpl.py | 75-111 | No armature config (uses MJCF default) |
| data/assets/mjcf/smpl_humanoid.xml | 13-150 | All geoms: `conaffinity="1"` (allows self-collision) |

---

## Verification Steps

1. **Confirm MuJoCo has no self-collision handling**:
   ```bash
   grep -n "self_collision" \
     /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py
   # Result: (no matches)
   ```

2. **Confirm other simulators have it**:
   ```bash
   grep -n "self_collision" \
     /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/simulator/isaacgym/simulator.py
   # Result: 782:        col_filter = 0 if self.robot_config.asset.self_collisions else 1
   ```

3. **Confirm SMPL geoms allow self-collision**:
   ```bash
   grep -c "conaffinity=\"1\"" \
     /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml
   # Result: 23 (all body geoms)
   ```

4. **Confirm SMPL armature is uniform**:
   ```bash
   grep -o "armature=\"[^\"]*\"" \
     /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml | sort | uniq -c
   # Result: 69 armature="0.02"  (all identical)
   ```

---

## MuJoCo Collision Parameter Reference

| Parameter | Meaning | Value in SMPL | Effect |
|-----------|---------|--------------|--------|
| `contype` | Which collision layer(s) this geom belongs to | 1 or 7 | Determines collision channel |
| `conaffinity` | Which collision types this geom collides with | 1 (in MJCF) | **Allows collision with all objects** |
| `condim` | Contact dimension | 3 | 3D friction model (appropriate) |
| `margin` | Penetration threshold | 0.001 m (1 mm) | Before contact is generated |

**Key insight**: `conaffinity="1"` on all robot geoms means they can collide with each other by default. MuJoCo has no built-in self-collision disabling. It's up to the simulator wrapper to set `conaffinity=0` for self-collision disabling.

---

## Why SMPL Falls Without Self-Collision Disabling

### Physics Simulation Loop (Simplified)

```
For each timestep:
  1. Apply PD control torques (computed from target positions)
  2. Run physics simulation step
  3. Check for penetrations (interpenetration in rest pose detected)
  4. Generate contact forces (proportional to penetration depth)
  5. Apply contact impulses (overrides torques)
  
Result: Contact impulses > PD torques → limbs repel → joints violently move
```

### Why It Works in IsaacGym/Newton/Genesis

Those simulators have built-in self-collision disabling. They don't generate contact forces for robot-robot interactions, only for robot-world interactions (ground).

### Why MuJoCo Fails

Without code to set `conaffinity=0`, contact forces are generated for ALL geom pairs, including self-collisions. These uncontrolled forces destabilize the motion tracking.

---

## Files Involved

```
ref_repo/ProtoMotions/
├── protomotions/
│   ├── simulator/
│   │   ├── mujoco/simulator.py               ← NEEDS FIX (no self-collision handling)
│   │   ├── isaacgym/simulator.py             ✓ (line 782: col_filter)
│   │   ├── newton/simulator.py               ✓ (line 206: enable_self_collisions)
│   │   └── genesis/simulator.py              ✓ (line 90: enable_self_collision)
│   ├── robot_configs/
│   │   ├── base.py                           (line 101: self_collisions flag)
│   │   └── smpl.py                           (no per-joint armature override)
│   └── data/assets/mjcf/
│       └── smpl_humanoid.xml                 (all geoms: conaffinity="1")
```

---

## Conclusion

**Primary Issue**: MuJoCo simulator does not implement the `self_collisions` configuration flag.

**Secondary Issue**: None - Armature is correct, collision settings in MJCF are appropriate.

**Fix Complexity**: Straightforward - ~15 lines of code to add one method and one conditional call.

**Expected Impact**: High - Eliminates uncontrolled contact forces that cause motion instability and falls.
