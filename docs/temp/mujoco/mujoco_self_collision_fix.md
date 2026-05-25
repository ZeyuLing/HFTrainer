# MuJoCo SMPL Humanoid Falls During RL Simulation: Root Cause Analysis & Fix

## Executive Summary

The MuJoCo simulator in ProtoMotions is **NOT implementing robot self-collision disabling**, even though:
1. The `self_collisions` flag is defined in `RobotAssetConfig` (default: `True`)
2. All other simulators (IsaacGym, Newton, Genesis) properly implement this feature
3. The SMPL humanoid MJCF allows unrestricted self-collision through `conaffinity="1"` on all geoms

**Result**: The SMPL humanoid experiences uncontrolled self-collision repulsive forces that exceed PD control torques, causing instability and falls during RL training.

---

## Investigation Details

### 1. Self-Collision Handling in MuJoCo Simulator

**File**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py`

**Finding**: NO robot self-collision disabling code exists.

Current code only handles **projectile collisions** (lines 1142-1153):

```python
# Lines 1142-1153: Only handles projectiles, NOT robot self-collisions
def _disable_projectile_collisions(self) -> None:
    """Disable collisions for all projectile geoms."""
    for gid in self._proj_geom_ids:
        self.model.geom_contype[gid] = 0
        self.model.geom_conaffinity[gid] = 0

def _enable_projectile_collision(self, proj_idx: int) -> None:
    """Enable collisions for a single projectile geom."""
    if proj_idx < len(self._proj_geom_ids):
        gid = self._proj_geom_ids[proj_idx]
        self.model.geom_contype[gid] = 1
        self.model.geom_conaffinity[gid] = 1
```

**Critical Issue**: No call to `self.robot_config.asset.self_collisions` anywhere in MuJoCo simulator initialization.

---

### 2. How Other Simulators Handle Self-Collisions

#### IsaacGym (WORKING)
**File**: `protomotions/simulator/isaacgym/simulator.py:782`

```python
col_filter = 0 if self.robot_config.asset.self_collisions else 1
# col_filter=0: self-collisions ENABLED (default)
# col_filter=1: self-collisions DISABLED
```

Passed to `self._gym.create_actor(..., col_filter, ...)` which sets IsaacGym's collision filtering.

---

#### Newton (WORKING)
**File**: `protomotions/simulator/newton/simulator.py:206`

```python
self.robot.add_mjcf(
    asset_path,
    ignore_names=["floor", "ground"],
    collapse_fixed_joints=False,
    floating=not self.robot_config.asset.fix_base_link,
    enable_self_collisions=self.robot_config.asset.self_collisions,  # ← Direct flag
)
```

---

#### Genesis (WORKING)
**File**: `protomotions/simulator/genesis/simulator.py:90`

```python
enable_self_collision=self.robot_config.asset.self_collisions,  # ← Direct flag
```

---

#### MuJoCo (BROKEN)
**File**: `protomotions/simulator/mujoco/simulator.py`

**Search result**: NO MATCHES for `self_collision` anywhere in the file.

---

### 3. SMPL Robot Configuration

**File**: `protomotions/robot_configs/smpl.py`

**Key finding**: Armature is NOT configured per-joint in the Python config.

```python
# Lines 75-111: ControlConfig specifies stiffness/damping but NO armature
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
            # ... more joint groups ...
        },
    )
)
```

**Conclusion**: All joints use MJCF default of `armature="0.02"` (see below), which is standard and consistent.

---

### 4. SMPL Humanoid MJCF Collision Configuration

**File**: `protomotions/data/assets/mjcf/smpl_humanoid.xml`

**All 69 joints have uniform armature**:
```xml
<joint name="L_Hip_x" ... armature="0.02" ... />
<joint name="L_Hip_y" ... armature="0.02" ... />
<joint name="L_Hip_z" ... armature="0.02" ... />
<!-- All 69 joints: armature="0.02" -->
```

**All geoms allow self-collision** (sample geoms):

```xml
<!-- Line 13: Pelvis geom -->
<geom type="box" ... conaffinity="1" condim="3" contype="7" margin="0.001" />

<!-- Line 18: L_Hip geom -->
<geom type="capsule" contype="1" conaffinity="1" ... condim="3" margin="0.001" />

<!-- Line 23: L_Knee geom -->
<geom type="capsule" contype="1" conaffinity="1" ... condim="3" margin="0.001" />

<!-- Line 28: L_Ankle geom -->
<geom type="box" ... conaffinity="1" condim="3" contype="7" margin="0.001" />

<!-- ... ALL body geoms have conaffinity="1" ... -->
```

**Key MuJoCo collision parameters**:
- `contype`: Collision type (which collision layer this geom belongs to)
- `conaffinity`: Collision affinity (which collision types this geom collides WITH)
  - `conaffinity="1"` = collide with anything (including self)
- `condim`: Contact dimension (3D friction model)
- `margin`: Penetration margin before contact is generated (0.001 m = 1 mm)

---

### 5. RobotAssetConfig Self-Collisions Flag

**File**: `protomotions/robot_configs/base.py:101`

```python
@dataclass
class RobotAssetConfig:
    """Configuration for robot asset properties."""
    
    asset_root: str = "protomotions/data/assets"
    self_collisions: bool = True  # ← Flag defined but IGNORED by MuJoCo simulator
```

---

## Root Cause Analysis

1. **SMPL humanoid in rest pose has natural limb interpenetration**:
   - Shoulders overlap
   - Hip region has complex geometry
   - Arms naturally touch torso

2. **With `conaffinity="1"`, MuJoCo contact solver generates repulsive forces** to resolve penetrations

3. **PD control cannot overcome these repulsive forces**:
   - Typical PD torques: τ = K_p * (x_target - x) + K_d * (-v)
   - Contact forces grow unbounded as penetration increases
   - Result: Instability, limbs violently repelling → falls

4. **MuJoCo simulator ignores `self_collisions=True` flag** → No code to disable self-collision

---

## Solution: Implement Self-Collision Disabling in MuJoCo

### Required Changes

**File**: `protomotions/simulator/mujoco/simulator.py`

**Method to add** (similar to projectile collision handling):

```python
def _disable_self_collisions(self) -> None:
    """Disable collisions between robot body parts.
    
    This prevents contact forces from arising due to natural interpenetration
    in rest pose (e.g., shoulders, hips). Only called if self_collisions=False
    in robot_config.asset.
    """
    # Iterate through all geom IDs and disable self-collision
    # Body 0 is 'world' (floor), bodies 1+ are robot bodies
    # Get all robot geoms: self.model.geom_bodyid contains body index for each geom
    for gid in range(self.model.ngeom):
        body_id = self.model.geom_bodyid[gid]
        # Skip floor/world body
        if body_id > 0 and body_id < self._num_bodies:
            # Set conaffinity to 0 = no collision with anything
            self.model.geom_conaffinity[gid] = 0
```

**Integration point** (after line 318 in `_create_simulation()`):

```python
def _create_simulation(self) -> None:
    """Create the MuJoCo simulation environment."""
    # ... existing code ...
    
    # Override armature and frictionloss from robot config
    self._override_joint_properties()
    
    # NEW: Disable robot self-collisions if configured
    if not self.robot_config.asset.self_collisions:
        self._disable_self_collisions()
    
    # Build actuator-to-DOF mapping
    self._build_actuator_mapping()
    # ... rest of method ...
```

---

## Alternative: Disable in MJCF (Not Recommended)

Could modify SMPL humanoid MJCF to set `conaffinity="0"` for all geoms, but:
- **Not recommended**: Config system should handle this
- **Inflexible**: Can't toggle self-collisions without re-exporting MJCF
- **Duplicates information**: Flag already defined in `RobotAssetConfig`

---

## Validation Checklist

After implementing the fix:

1. **Verify flag is checked**: 
   - `grep -n "self_collisions" protomotions/simulator/mujoco/simulator.py`
   - Should find: `if not self.robot_config.asset.self_collisions:`

2. **Verify geom_conaffinity is modified**:
   - `grep -n "geom_conaffinity" protomotions/simulator/mujoco/simulator.py`
   - Should find: `self.model.geom_conaffinity[gid] = 0`

3. **Test on SMPL humanoid**:
   - Train with MuJoCo backend using SMPL humanoid
   - Should no longer fall during standing motion tracking
   - Contact forces between body parts should be zero

4. **Verify IsaacGym/Newton/Genesis still work**:
   - Each simulator already has its own implementation
   - No changes needed to those files

---

## Related Issues in MuJoCo Implementation

### Issue 1: Robot Geom ID Tracking
Current code doesn't maintain `_robot_geom_ids` list. Solution uses `geom_bodyid` array to filter robot bodies dynamically.

### Issue 2: Body ID Bounds
Need to know total number of bodies (`self._num_bodies`). This should already be tracked during robot initialization.

### Issue 3: Projectile Collision Independence
Projectile collision handling (lines 1142-1153) is independent and should not be affected by robot self-collision changes.

---

## References

**MuJoCo XML Attributes**:
- [MuJoCo Documentation: Collision attributes](https://mujoco.readthedocs.io/)
- `contype`: Collision type bits (determines which objects this geom can collide with)
- `conaffinity`: Collision affinity bits (determines which collision types this geom responds to)

**ProtoMotions Files**:
- MuJoCo simulator: `protomotions/simulator/mujoco/simulator.py`
- Robot config base: `protomotions/robot_configs/base.py`
- SMPL config: `protomotions/robot_configs/smpl.py`
- SMPL MJCF: `protomotions/data/assets/mjcf/smpl_humanoid.xml`
- IsaacGym reference: `protomotions/simulator/isaacgym/simulator.py:782`
- Newton reference: `protomotions/simulator/newton/simulator.py:206`
- Genesis reference: `protomotions/simulator/genesis/simulator.py:90`

