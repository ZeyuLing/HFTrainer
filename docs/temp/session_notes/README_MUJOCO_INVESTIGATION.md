# MuJoCo SMPL Humanoid Investigation - Quick Reference

## 📋 Executive Summary

**Root Cause**: MuJoCo simulator is missing self-collision disabling code that all other simulators have.

**Impact**: Uncontrolled contact forces cause SMPL humanoid to fall during RL training.

**Status**: ✅ Investigation Complete - Solution Ready

---

## 📁 Documentation Files (in this directory)

| File | Purpose | Size | Read This First? |
|------|---------|------|------------------|
| `MUJOCO_SELF_COLLISION_INVESTIGATION_COMPLETE.md` | **START HERE** - Complete overview | 9.0 KB | ✅ YES |
| `mujoco_self_collision_fix.md` | Detailed analysis + fix design | 9.7 KB | Next |
| `INVESTIGATION_SUMMARY.md` | Q&A format with code locations | 8.7 KB | Reference |
| `mujoco_self_collision_fix.patch` | Ready-to-apply patch | 1.7 KB | Implementation |

---

## 🔍 Quick Answers

### Q: Why does SMPL humanoid fall?
**A**: MuJoCo generates uncontrolled contact forces between body parts due to natural interpenetration in rest pose (shoulders, hips). These forces exceed PD control torques.

### Q: Is armature the problem?
**A**: **NO** - All 69 joints use uniform `armature="0.02"` (standard value)

### Q: Is the MJCF broken?
**A**: **NO** - MJCF correctly has `conaffinity="1"` for all geoms (allows collision)

### Q: Why do other simulators work?
**A**: IsaacGym, Newton, Genesis all have code to disable self-collisions. MuJoCo doesn't.

### Q: How to fix?
**A**: Add ~15 lines of code to MuJoCo simulator. See `mujoco_self_collision_fix.patch`

---

## 🔧 Implementation Checklist

- [ ] Read `MUJOCO_SELF_COLLISION_INVESTIGATION_COMPLETE.md`
- [ ] Review `mujoco_self_collision_fix.md` for detailed design
- [ ] Apply patch: `cd ref_repo/ProtoMotions && git apply mujoco_self_collision_fix.patch`
- [ ] Or manually add the code shown in `mujoco_self_collision_fix.patch`
- [ ] Test with SMPL humanoid RL training
- [ ] Verify no regression on other simulators

---

## 📊 Evidence Summary

| Finding | Status |
|---------|--------|
| MuJoCo has self-collision code | ❌ NO (0 matches) |
| IsaacGym has self-collision code | ✅ YES (line 782) |
| Newton has self-collision code | ✅ YES (line 206) |
| Genesis has self-collision code | ✅ YES (line 90) |
| SMPL armature is consistent | ✅ YES (all 0.02) |
| SMPL MJCF allows self-collision | ✅ YES (all conaffinity="1") |
| Configuration flag exists | ✅ YES (base.py:101) |
| Flag is used by MuJoCo | ❌ NO (0 matches) |

---

## 📍 Key File Locations

```
ref_repo/ProtoMotions/
├── protomotions/
│   ├── simulator/
│   │   ├── mujoco/simulator.py             ← FIX NEEDED (line 318, 1153)
│   │   ├── isaacgym/simulator.py           ✓ Reference (line 782)
│   │   ├── newton/simulator.py             ✓ Reference (line 206)
│   │   └── genesis/simulator.py            ✓ Reference (line 90)
│   ├── robot_configs/
│   │   ├── base.py                         Flag def (line 101)
│   │   └── smpl.py                         Config (lines 75-111)
│   └── data/assets/mjcf/
│       └── smpl_humanoid.xml               MJCF (all geoms)
```

---

## 🚀 To Implement the Fix

### Option 1: Apply Patch
```bash
cd /path/to/ref_repo/ProtoMotions
git apply mujoco_self_collision_fix.patch
```

### Option 2: Manual Implementation
1. Open `protomotions/simulator/mujoco/simulator.py`
2. Find line 318 (after `self._override_joint_properties()`)
3. Add 4 lines to call `_disable_self_collisions()` 
4. Find line 1153 (after `_enable_projectile_collision()`)
5. Add new method (see patch or design doc)

See `mujoco_self_collision_fix.patch` for exact code.

---

## ✅ Verification Steps

```bash
# 1. Verify MuJoCo has no self-collision code
grep "self_collision" ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py
# Should return: (no matches)

# 2. Verify IsaacGym has it (for comparison)
grep "self_collision" ref_repo/ProtoMotions/protomotions/simulator/isaacgym/simulator.py
# Should return: line 782 with col_filter reference

# 3. Verify SMPL armature is uniform
grep -o 'armature="[^"]*"' ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml | sort | uniq -c
# Should return: 69 armature="0.02"

# 4. Verify SMPL MJCF allows self-collision
grep 'conaffinity="1"' ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml | wc -l
# Should return: 23 (all body geoms)
```

---

## 🧠 Physics Explanation

### Current Behavior (Without Fix)
```
PD Control:     τ = K_p * error + K_d * error_rate   (bounded)
Contact Force:  F ∝ penetration_depth                 (unbounded)

Result: Contact forces >> PD torques → limbs repel → falls
```

### After Fix
```
PD Control:     τ = K_p * error + K_d * error_rate   (bounded)
Contact Force:  0 (disabled for robot-robot pairs)     (zero)

Result: Only PD torques act → stable tracking → success
```

---

## 📞 Questions?

Refer to the detailed documentation files:
- **Comprehensive design**: `mujoco_self_collision_fix.md`
- **Q&A format**: `INVESTIGATION_SUMMARY.md`
- **Code reference**: `mujoco_self_collision_fix.patch`

---

**Last Updated**: 2026-05-21  
**Investigation Status**: ✅ COMPLETE  
**Fix Status**: 🟢 Ready to Apply
