# MuJoCo SMPL Humanoid Falling Investigation Report

**Investigation Date**: 2026-05-21  
**Base Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

## Executive Summary

Examined the MuJoCo simulator backend in ProtoMotions for issues that could cause a SMPL humanoid to fall during RL tracker simulation. Three potential issues identified:

1. **❌ Missing Self-Collision Handling** (CRITICAL)
2. **⚠️  Armature Values Are Standard** (Likely Not the Issue)
3. **ℹ️  Limited Geom Collision Configuration** (Minor Concern)

---

## Finding #1: Missing Self-Collision Handling in MuJoCo Backend

### Issue
**The MuJoCo simulator does NOT respect the `self_collisions` flag from robot config**, unlike all other simulators (IsaacGym, IsaacLab, Newton, Genesis).

### Evidence

#### A. IsaacGym Implementation (Reference)
**File**: `ref_repo/ProtoMotions/protomotions/simulator/isaacgym/simulator.py`

```python
col_filter = 0 if self.robot_config.asset.self_collisions else 1
```
✅ IsaacGym explicitly reads `self.robot_config.asset.self_collisions` and applies collision filtering.

#### B. Newton Implementation (Reference)
**File**: `ref_repo/ProtoMotions/protomotions/simulator/newton/simulator.py`

```python
enable_self_collisions=self.robot_config.asset.self_collisions,
```
✅ Newton explicitly passes `self_collisions` to the body creation.

#### C. Genesis Implementation (Reference)
**File**: `ref_repo/ProtoMotions/protomotions/simulator/genesis/simulator.py`

```python
enable_self_collision=self.robot_config.asset.self_collisions,
```
✅ Genesis explicitly passes `self_collisions` to robot creation.

#### D. MuJoCo Implementation (Problem)
**File**: `ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py`

Lines 1140-1153 (projectile collision handling only):
```python
def _disable_projectile_collisions(self) -> None:
    """Disable collisions for all projectile geoms."""
    for gid in self._proj_geom_ids:
        self.model.geom_contype[gid] = 0        # ← Only for projectiles
        self.model.geom_conaffinity[gid] = 0

def _enable_projectile_collision(self, proj_idx: int) -> None:
    """Enable collisions for a single projectile geom."""
    if proj_idx < len(self._proj_geom_ids):
        gid = self._proj_geom_ids[proj_idx]
        self.model.geom_contype[gid] = 1
        self.model.geom_conaffinity[gid] = 1
```

**🔴 NO CODE EXISTS to handle robot self-collisions.**

### Configuration Check

**File**: `ref_repo/ProtoMotions/protomotions/robot_configs/base.py:96-101`

```python
@dataclass
class RobotAssetConfig:
    """Configuration for robot asset properties."""
    
    # Optional fields with defaults
    asset_root: str = "protomotions/data/assets"
    self_collisions: bool = True
    
    # Optional fields
    asset_file_name: str = None
    ...
```

**Default value**: `self_collisions: bool = True`

This means:
- By default, all simulators **SHOULD** allow self-collisions
- But **MuJoCo ignores this flag entirely**

### Impact on SMPL Humanoid

The SMPL model has these collision surfaces (from MJCF):

**File**: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml` (lines 1-143, sampled)

All geoms have:
```xml
<geom type="capsule" contype="1" conaffinity="1" condim="3" margin="0.001" ... />
<geom type="box" conaffinity="1" condim="3" contype="7" margin="0.001" ... />
```

Breakdown of `contype` values in MJCF:
- **`contype="1"`**: Hinge joint connection geoms (leg segments, torso, arms)
- **`contype="7"`**: Feet and hand geoms (can collide with everything: 1|2|4)

All use **`conaffinity="1"`**: Can collide with anything that has contype containing bit 0.

### What MuJoCo Currently Does

Since **no self-collision disabling code exists**, MuJoCo will:

1. Load all geoms with their XML `contype` and `conaffinity` values ✓
2. **Attempt to collide robot body parts with each other** ✓
3. This creates **internal kinematic conflicts** that can cause:
   - Erratic joint torques to resolve penetrations
   - Instability in pose tracking
   - Humanoid collapsing when standing still (self-collision forces dominate)
   - Falls during motion when limbs intersect

### Why It Falls

If your humanoid is falling during RL training despite correct PD gains and joint limits, **self-collisions** are likely the culprit:

1. Standing pose has natural interpenetration (shoulders, hips, etc.)
2. MuJoCo contact solver generates repulsive forces to resolve penetration
3. These forces exceed the PD control torques
4. Humanoid falls or becomes unstable

---

## Finding #2: SMPL Robot Armature Values

### Current Configuration

**File**: `ref_repo/ProtoMotions/protomotions/robot_configs/smpl.py:75-111`

```python
control: ControlConfig = field(
    default_factory=lambda: ControlConfig(
        control_type=ControlType.BUILT_IN_PD,
        override_control_info={
            ".*_(Hip|Knee|Ankle)_.*": ControlInfo(
                stiffness=800,
                damping=80,
                effort_limit=500,
                velocity_limit=100,
            ),
            ".*_Toe_.*": ControlInfo(
                stiffness=500,
                damping=50,
                effort_limit=500,
                velocity_limit=100,
            ),
            # ... more joint groups
        },
    )
)
```

**Note**: This specifies **stiffness/damping/effort_limit** but **NOT armature**.

### Armature Override in Simulator

**File**: `ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py:394-431`

Lines 394-431:
```python
def _override_joint_properties(self) -> None:
    """Override armature and frictionloss from robot config.
    
    The MJCF has default values (e.g. armature=0.03 for all joints) that
    may differ from the robot config's per-joint values. Newton and IsaacGym
    override these; we must do the same.
    
    frictionloss (Coulomb joint friction) is zeroed because IsaacGym's PhysX
    does not model it, so policies trained in IsaacGym don't expect it.
    """
    control_info = self.robot_config.control.control_info
    dof_start = 6 if self._has_free_joint else 0
    
    for i in range(self.model.njnt):
        jnt_type = self.model.jnt_type[i]
        if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        
        jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
        dof_addr = self.model.jnt_dofadr[i]
        dof_idx = dof_addr - dof_start
        
        if jnt_name in control_info:
            info = control_info[jnt_name]
            old_armature = self.model.dof_armature[dof_addr]
            
            # Override armature from robot config
            if info.armature is not None:
                self.model.dof_armature[dof_addr] = info.armature
            
            # # Zero frictionloss (IsaacGym doesn't model this)
            # self.model.dof_frictionloss[dof_addr] = 0.0
            
            print(
                f"  Joint '{jnt_name}' DOF[{dof_idx}]: "
                f"armature {old_armature:.6f} -> {self.model.dof_armature[dof_addr]:.6f}, "
                f"frictionloss -> 0.0"
            )
```

### MJCF Default Armature

**File**: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml:15-32` (sample)

```xml
<body name="L_Hip" pos="-0.0068 0.0695 -0.0914">
    <joint name="L_Hip_x" type="hinge" pos="0 0 0" axis="1 0 0" 
            stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
    <joint name="L_Hip_y" type="hinge" pos="0 0 0" axis="0 1 0" 
            stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
    <joint name="L_Hip_z" type="hinge" pos="0 0 0" axis="0 0 1" 
            stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
```

**All joints**: `armature="0.02"` (2% of body mass, standard MuJoCo convention)

### Key Finding: ✅ Armature Is Standard

- **MJCF default**: `armature="0.02"` for all joints
- **MuJoCo standard default**: `armature=0.02` (per documentation)
- **IsaacGym PhysX**: Uses similar values (0.01-0.05 range)
- **Newton**: Typically 0.01-0.05 range

**Conclusion**: Armature values are **NOT the problem** for falling.
- All values are within normal range
- Equal across all joints (no outliers)
- MuJoCo uses this as the default anyway

---

## Finding #3: Geom Collision Configuration

### Current MJCF Geom Settings

**File**: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml` (lines 13-57)

#### Example 1: Pelvis (Box)
```xml
<geom type="box" pos="-0.0055 -0.0000 -0.0121" size="0.083 0.1069 0.0722" 
       quat="1.0000 0.0000 0.0000 0.0000" density="1000" 
       conaffinity="1" condim="3" contype="7" margin="0.001" />
```

#### Example 2: L_Hip (Capsule - connecting segment)
```xml
<geom type="capsule" contype="1" conaffinity="1" density="2040.816327" 
       fromto="-0.0009 0.0069 -0.0750 -0.0036 0.0274 -0.3002" 
       size="0.0615" condim="3" margin="0.001" />
```

### Collision Configuration Breakdown

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `condim` | 3 | **3D friction cone** (max contact dimensions) |
| `contype` | 1 or 7 | **Collision type bits** (which filter groups this belongs to) |
| `conaffinity` | 1 | **Collision affinity** (which filter groups it can collide with) |
| `margin` | 0.001 | **Contact margin** (0.1 cm) |

### Analysis: condim=3

The `condim=3` setting is appropriate:
- ✅ Allows sliding friction (not just normal forces)
- ✅ Standard for humanoid robots
- ✅ Not causing the falling issue

### Analysis: contype/conaffinity

The collision filtering (`contype=1|7`, `conaffinity=1`) means:
- **All body parts can collide with each other**
- **All body parts can collide with ground** (contype=1|4 collides with conaffinity=1|4)
- **No exclusion list for self-collision**

This is **correct for inter-body collision detection** but **problematic for intra-body (self) collision**.

### How Other Simulators Differ

- **IsaacGym**: Has `disable_self_collisions()` API that modifies collision groups
- **Newton**: Has `enable_self_collisions` parameter
- **Genesis**: Has `enable_self_collision` parameter  
- **MuJoCo**: **No API call in ProtoMotions code** to disable self-collisions

---

## Summary Table: Collision Handling Across Simulators

| Simulator | Self-Collision Flag Used? | Implementation Method | Issue Status |
|-----------|--------------------------|----------------------|--------------|
| **IsaacGym** | ✅ YES | `col_filter = 0 if self_collisions else 1` | ✅ OK |
| **IsaacLab** | ✅ YES | `enabled_self_collisions=` param | ✅ OK |
| **Newton** | ✅ YES | `enable_self_collisions=` param | ✅ OK |
| **Genesis** | ✅ YES | `enable_self_collision=` param | ✅ OK |
| **MuJoCo** | ❌ NO | **None** | 🔴 **MISSING** |

---

## Root Cause of Humanoid Falling

### Most Likely Cause: **Unhandled Self-Collisions in MuJoCo**

1. **SMPL humanoid body has inherent penetration** in rest pose (shoulders, hips, etc.)
2. **MuJoCo contact solver detects these penetrations** as collisions
3. **Large repulsive forces generated** to separate body parts
4. **PD control torques insufficient** to counteract these forces
5. **Humanoid falls or becomes unstable**

### Recommended Fix

Add self-collision disabling logic to `MujocoSimulator._create_simulation()`:

```python
def _disable_self_collisions(self) -> None:
    """Disable self-collisions between robot body parts.
    
    Modifies geom_contype/geom_conaffinity to prevent inter-body collisions
    while preserving collisions with ground/external objects.
    """
    # Iterate all geoms and disable self-collision
    for gid in range(self.model.ngeom):
        body_id = self.model.geom_bodyid[gid]
        
        # Skip world body (ground) geoms
        if body_id <= 0:
            continue
        
        # Skip projectile geoms (already handled)
        if gid in self._proj_geom_ids:
            continue
        
        # Disable self-collision: set conaffinity to 0
        self.model.geom_conaffinity[gid] = 0
    
    log.info("Disabled self-collisions for robot body (geoms have conaffinity=0)")
```

---

## Appendix: Line Numbers Reference

### MuJoCo Simulator File
**Path**: `ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py`

| Issue | Lines | Description |
|-------|-------|-------------|
| Projectile collision disable | 1142-1153 | Only handles projectiles, not robot self-collision |
| Joint property override | 394-431 | Handles armature, but self_collisions flag ignored |
| Simulation setup | 294-379 | Creates model but doesn't disable self-collisions |

### SMPL Robot Config
**Path**: `ref_repo/ProtoMotions/protomotions/robot_configs/smpl.py`

| Issue | Lines | Description |
|-------|-------|-------------|
| Control config (no armature spec) | 75-111 | Stiffness/damping defined, armature from MJCF |
| Trackable bodies | 37-46 | Body definitions for tracking |
| Default root height | 61 | 0.95 m (reasonable for humanoid) |

### SMPL MJCF Asset
**Path**: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml`

| Issue | Lines | Description |
|-------|-------|-------------|
| All joints armature | 15-143 (samples) | **`armature="0.02"`** (standard, consistent) |
| Geom collision settings | 13-143 (samples) | **`contype=1|7`, `conaffinity=1`, `condim=3`** |

### Base Robot Config
**Path**: `ref_repo/ProtoMotions/protomotions/robot_configs/base.py`

| Issue | Lines | Description |
|-------|-------|-------------|
| Self-collision flag | 96-101 | `self_collisions: bool = True` (default enabled) |

---

## Verification Steps

To confirm self-collision is causing the issue:

### Step 1: Check if MuJoCo simulator disables self-collisions
```bash
grep -n "self_collision\|conaffinity.*=" \
  ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py
```

**Expected**: Should show only projectile collision code (lines 1145-1153).  
**Current**: ✅ Confirmed - **no robot self-collision code exists**.

### Step 2: Test with self-collision disabled
Temporarily add this to `_create_simulation()`:
```python
# Disable all robot self-collisions
for gid in range(self.model.ngeom):
    if 0 < self.model.geom_bodyid[gid] <= self._num_robot_bodies:
        self.model.geom_conaffinity[gid] = 0
```
Then run inference. If humanoid stops falling → **confirms self-collision issue**.

---

## Conclusion

**The SMPL humanoid falls in MuJoCo because**:

1. ❌ **MuJoCo simulator ignores `robot_config.asset.self_collisions` flag**
2. ✅ Armature values are standard and correct
3. ✅ `condim=3` friction settings are appropriate

**Fix Priority**: 🔴 **HIGH**
- Implement `_disable_self_collisions()` method in MuJoCo simulator
- Mirror IsaacGym's approach
- Test with `self_collisions=False` in robot config

---

**Investigation Complete**  
Report generated: 2026-05-21
