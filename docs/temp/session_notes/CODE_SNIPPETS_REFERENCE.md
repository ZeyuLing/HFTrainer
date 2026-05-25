# Code Snippets Reference - MuJoCo SMPL Investigation

## Issue #1: Missing Self-Collision Handling

### Current Code (MuJoCo Only Handles Projectiles)

**File**: `ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py`  
**Lines**: 1142-1153

```python
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

**Problem**: No equivalent code for robot self-collisions.

---

### Where Self-Collision Should Be Handled

**File**: `ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py`  
**Method**: `_create_simulation()` (lines 294-379)

**Current structure**:
```python
def _create_simulation(self) -> None:
    """Create the MuJoCo simulation environment."""
    # Load MJCF model
    asset_root = self.robot_config.asset.asset_root
    asset_file = self.robot_config.asset.asset_file_name
    asset_path = os.path.join(asset_root, asset_file)

    # MJCF injection below needs _proj_config before _init_projectiles runs
    self._resolve_proj_config()

    log.info("Loading MuJoCo model from: %s", asset_path)
    self.model = self._load_mjcf_stripped(asset_path, self._proj_config)
    self.data = mujoco.MjData(self.model)

    # Set timestep
    self.model.opt.timestep = 1.0 / self.config.sim.fps
    print(
        f"MuJoCo timestep: {self.model.opt.timestep:.4f}s ({self.config.sim.fps}Hz)"
    )

    # Zero passive forces from MJCF (we handle PD control via actuators)
    self._zero_passive_forces()

    # Override armature and frictionloss from robot config
    self._override_joint_properties()
    
    # ⚠️ MISSING: self-collision handling should go here!
    
    # Build actuator-to-DOF mapping
    self._build_actuator_mapping()
    # ... rest of code ...
```

**Recommended fix location**: After line 318 (`self._override_joint_properties()`), add:

```python
    # Disable self-collisions if configured
    if not self.robot_config.asset.self_collisions:
        self._disable_self_collisions()
```

---

### Comparison: How IsaacGym Does It

**File**: `ref_repo/ProtoMotions/protomotions/simulator/isaacgym/simulator.py`

IsaacGym applies collision filtering directly:
```python
col_filter = 0 if self.robot_config.asset.self_collisions else 1
# Then uses col_filter when creating rigid bodies
```

---

### Comparison: How Newton Does It

**File**: `ref_repo/ProtoMotions/protomotions/simulator/newton/simulator.py`

Newton passes self_collisions as a parameter:
```python
enable_self_collisions=self.robot_config.asset.self_collisions,
```

---

### Comparison: How Genesis Does It

**File**: `ref_repo/ProtoMotions/protomotions/simulator/genesis/simulator.py`

Genesis passes self_collision as a parameter:
```python
enable_self_collision=self.robot_config.asset.self_collisions,
```

---

## Issue #2: Armature Values (✅ Not the Problem)

### Where Armature is Configured

**File**: `ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py`  
**Lines**: 394-431

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

**Finding**: This code is CORRECT - it reads armature from robot config if available.

### SMPL Robot Config (No Armature Specified)

**File**: `ref_repo/ProtoMotions/protomotions/robot_configs/smpl.py`  
**Lines**: 75-111

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
                # ← Note: No armature specified
            ),
            ".*_Toe_.*": ControlInfo(
                stiffness=500,
                damping=50,
                effort_limit=500,
                velocity_limit=100,
            ),
            "(Torso|Spine|Chest)_.*": ControlInfo(
                stiffness=1000,
                damping=100,
                effort_limit=500,
                velocity_limit=100,
            ),
            "(Neck|Head|.*_Thorax|.*_Shoulder|.*_Elbow)_.*": ControlInfo(
                stiffness=500,
                damping=50,
                effort_limit=500,
                velocity_limit=100,
            ),
            ".*_(Wrist|Hand)_.*": ControlInfo(
                stiffness=300,
                damping=30,
                effort_limit=500,
                velocity_limit=100,
            ),
        },
    )
)
```

**Result**: Since armature is not specified in robot config, all joints use MJCF default value.

### MJCF Default Armature (All Joints)

**File**: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml`  
**Sample lines**: 15-32 (L_Hip to L_Toe_z)

```xml
<body name="L_Hip" pos="-0.0068 0.0695 -0.0914">
    <joint name="L_Hip_x" type="hinge" pos="0 0 0" axis="1 0 0" 
            stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
    <joint name="L_Hip_y" type="hinge" pos="0 0 0" axis="0 1 0" 
            stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
    <joint name="L_Hip_z" type="hinge" pos="0 0 0" axis="0 0 1" 
            stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
    <geom type="capsule" contype="1" conaffinity="1" density="2040.816327" 
           fromto="-0.0009 0.0069 -0.0750 -0.0036 0.0274 -0.3002" size="0.0615" 
           condim="3" margin="0.001" rgba="0.8 0.6 .4 1" />
    <body name="L_Knee" pos="-0.0045 0.0343 -0.3752">
        <joint name="L_Knee_x" type="hinge" pos="0 0 0" axis="1 0 0" 
                stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
        <joint name="L_Knee_y" type="hinge" pos="0 0 0" axis="0 1 0" 
                stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
        <joint name="L_Knee_z" type="hinge" pos="0 0 0" axis="0 0 1" 
                stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
        <geom type="capsule" contype="1" conaffinity="1" density="1234.567901" 
               fromto="-0.0087 -0.0027 -0.0796 -0.0350 -0.0109 -0.3184" size="0.0541" 
               condim="3" margin="0.001" rgba="0.8 0.6 .4 1" />
        <body name="L_Ankle" pos="-0.0437 -0.0136 -0.398">
            <joint name="L_Ankle_x" type="hinge" pos="0 0 0" axis="1 0 0" 
                    stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
            <joint name="L_Ankle_y" type="hinge" pos="0 0 0" axis="0 1 0" 
                    stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
            <joint name="L_Ankle_z" type="hinge" pos="0 0 0" axis="0 0 1" 
                    stiffness="800" damping="80" armature="0.02" range="-180.0000 180.0000" limited="true" />
```

**Finding**: All joints have `armature="0.02"` - **Standard and consistent across all 69 joints**.

### Base Robot Config Definition

**File**: `ref_repo/ProtoMotions/protomotions/robot_configs/base.py`  
**Lines**: 96-101

```python
@dataclass
class RobotAssetConfig:
    """Configuration for robot asset properties."""

    # Optional fields with defaults
    asset_root: str = "protomotions/data/assets"
    self_collisions: bool = True

    # Optional fields
    asset_file_name: str = None
    usd_asset_file_name: str = None
    usd_bodies_root_prim_path: str = None
    max_linear_velocity: float = 1000.0
    max_angular_velocity: float = 1000.0
    angular_damping: float = 0.0
    linear_damping: float = 0.0
```

**Note**: `self_collisions: bool = True` - but **MuJoCo ignores this flag**.

---

## Issue #3: Geom Collision Settings (✅ Appropriate)

### MJCF Geom Configuration Examples

**File**: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml`  
**Lines**: 13-57 (samples)

#### Pelvis (Box):
```xml
<geom type="box" pos="-0.0055 -0.0000 -0.0121" size="0.083 0.1069 0.0722" 
       quat="1.0000 0.0000 0.0000 0.0000" density="1000" 
       conaffinity="1" condim="3" contype="7" margin="0.001" rgba="0.8 0.6 .4 1" />
```

#### L_Hip Capsule:
```xml
<geom type="capsule" contype="1" conaffinity="1" density="2040.816327" 
       fromto="-0.0009 0.0069 -0.0750 -0.0036 0.0274 -0.3002" size="0.0615" 
       condim="3" margin="0.001" rgba="0.8 0.6 .4 1" />
```

#### L_Ankle (Box):
```xml
<geom type="box" pos="0.0242 0.0233 -0.0239" size="0.085 0.0483 0.0464" 
       quat="1.0000 0.0000 0.0000 0.0000" density="1000" 
       conaffinity="1" condim="3" contype="7" margin="0.001" rgba="0.8 0.6 .4 1" />
```

### Collision Parameter Meanings

| Parameter | Value | Meaning | Assessment |
|-----------|-------|---------|------------|
| `condim` | 3 | Max contact dimensions (3D friction cone) | ✅ Appropriate |
| `contype` | 1 or 7 | This geom belongs to collision filter group 1 or 7 | ⚠️ Allows self-collision |
| `conaffinity` | 1 | This geom can collide with geoms having contype bit 0 set | ⚠️ Allows self-collision |
| `margin` | 0.001 | Contact margin (0.1 cm) | ✅ Standard |

**contype semantics**:
- `contype="1"`: Leg connection segments (L_Hip, R_Hip, L_Knee, etc.)
- `contype="7"`: Feet and hand endpoints (L_Ankle, L_Toe, R_Hand, etc.) - can collide with all (1|2|4)

**conaffinity semantics**:
- `conaffinity="1"`: Can collide with anything that has contype & 1 (i.e., everything with bit 0 set)

**Result**: All body parts can collide with each other (self-collision enabled).

---

## Summary

### Critical Finding
🔴 **MuJoCo simulator lacks self-collision disabling logic that all other simulators have.**

### Verification Commands

Check for self-collision code in MuJoCo:
```bash
grep -n "self_collision\|disable.*self" \
  ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py
```
**Result**: No matches (except projectile code)

Check armature values in SMPL MJCF:
```bash
grep -o 'armature="[^"]*"' \
  ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml | sort | uniq -c
```
**Result**: All are `armature="0.02"` (69 occurrences)

Check for geom_conaffinity modifications in MuJoCo:
```bash
grep -n "geom_conaffinity\|geom_contype" \
  ref_repo/ProtoMotions/protomotions/simulator/mujoco/simulator.py
```
**Result**: Only lines 1145-1153 (projectiles), and 1152-1153 (enable)

---

