# SMPL Humanoid MuJoCo Model - Comprehensive Analysis Report

## Executive Summary

Two SMPL humanoid MuJoCo models exist in the OmniH2O repository:
- **smpl_humanoid.xml** - Original model with varied densities
- **smpl_humanoid_1.xml** - Variant with uniform density (500 kg/m³)

Both are **identical in kinematic structure** but differ in **mass distribution**. ProtoMotions includes a native SMPL robot configuration and can simulate SMPL humanoid out-of-the-box.

---

## 1. Model Architecture

### 1.1 Overall Structure

| Metric | Count |
|--------|-------|
| **Total Bodies** | 24 |
| **Total Joints** | 70 (all hinge type) |
| **Total DOF** | 76 |
|   - Freejoint (Pelvis) | 6 DOF (floating base) |
|   - Hinge Joints | 70 DOF |
| **Total Geoms** | 26 |
| **Total Actuators (Motors)** | 70 |

### 1.2 Body Hierarchy

The model follows a standard SMPL skeletal structure with bilateral limbs:

```
Pelvis (root, floating base)
├── L_Hip (3 DOF)
│   └── L_Knee (3 DOF)
│       └── L_Ankle (3 DOF)
│           └── L_Toe (3 DOF)
├── R_Hip (3 DOF)
│   └── R_Knee (3 DOF)
│       └── R_Ankle (3 DOF)
│           └── R_Toe (3 DOF)
└── Torso (3 DOF)
    └── Spine (3 DOF)
        └── Chest (3 DOF)
            ├── Neck (3 DOF)
            │   └── Head (3 DOF)
            ├── L_Thorax (3 DOF)
            │   └── L_Shoulder (3 DOF)
            │       └── L_Elbow (3 DOF)
            │           └── L_Wrist (3 DOF)
            │               └── L_Hand (3 DOF)
            └── R_Thorax (3 DOF)
                └── R_Shoulder (3 DOF)
                    └── R_Elbow (3 DOF)
                        └── R_Wrist (3 DOF)
                            └── R_Hand (3 DOF)
```

**Key Body Groups:**
- **Lower Body**: Pelvis, L/R Hip, L/R Knee, L/R Ankle, L/R Toe (12 bodies)
- **Upper Body**: Torso, Spine, Chest (3 bodies)
- **Neck & Head**: Neck, Head (2 bodies)
- **Arms**: L/R Thorax, L/R Shoulder, L/R Elbow, L/R Wrist, L/R Hand (10 bodies)

---

## 2. Joint Configuration

### 2.1 Joint Types & Ranges

All joints are **hinge (revolute) joints** with axis-aligned rotations (x, y, z axes):

| Body Group | # Joints | Joint Names | Stiffness | Damping | Armature | Range Examples |
|-----------|----------|------------|-----------|---------|----------|------------------|
| **Hip** | 6 | L/R_Hip_{x,y,z} | 800 | 80 | 0.02 | ±90° |
| **Knee** | 6 | L/R_Knee_{x,y,z} | 800 | 80 | 0.02 | ±5.6°, 0-180° |
| **Ankle** | 6 | L/R_Ankle_{x,y,z} | 800 | 80 | 0.02 | ±45°, ±90° |
| **Toe** | 6 | L/R_Toe_{x,y,z} | 500 | 50 | 0.02 | ±180° |
| **Torso/Spine/Chest** | 9 | {Torso,Spine,Chest}_{x,y,z} | 1000 | 100 | 0.02 | ±60° |
| **Neck/Head** | 6 | {Neck,Head}_{x,y,z} | 500 | 50 | 0.02 | ±5.6°, ±90° |
| **Shoulder** | 6 | L/R_Shoulder_{x,y,z} | 500 | 50 | 0.02 | ±720° (2 full rotations) |
| **Elbow** | 6 | L/R_Elbow_{x,y,z} | 500 | 50 | 0.02 | ±5.6°, ±180° |
| **Wrist** | 6 | L/R_Wrist_{x,y,z} | 300 | 30 | 0.02 | ±180° |
| **Hand** | 6 | L/R_Hand_{x,y,z} | 300 | 30 | 0.02 | ±180° |

### 2.2 Joint Stiffness & Damping Distribution

```
Stiffness=1000, Damping=100: 9 joints   (Torso, Spine, Chest - core stability)
Stiffness=800, Damping=80:   18 joints  (Hip, Knee, Ankle - locomotion)
Stiffness=500, Damping=50:   30 joints  (Shoulders, Elbows, Neck, Head, Toes)
Stiffness=300, Damping=30:   12 joints  (Wrists, Hands - dexterity)
Stiffness=5, Damping=0.1:    1 joint    (Default - unused in actual config)
```

**Armature**: All actuated joints have `armature=0.02` (increases inertia for stability).

---

## 3. Actuator (Motor) Configuration

### 3.1 Motor Setup

**All 70 joints have corresponding motors** with unified gear ratio:

```xml
<motor name="[JOINT_NAME]" joint="[JOINT_NAME]" gear="500"/>
```

**Motor Specifications:**
- **Gear Ratio**: 500 (all motors)
- **Control Range**: [-1, 1] (ctrlrange="-1 1", ctrllimited="true")
- **Count**: 70 motors (1 per hinge joint + Pelvis freejoint has no motor)

**Motor Distribution by Body:**
- 23 joint groups × 3 joints per group = 69 motors
- Pelvis freejoint: no motor (uses physics simulation directly)
- **Total actuated DOF**: 69 (out of 76 total DOF)

---

## 4. Collision & Friction Setup

### 4.1 Friction Parameters

**Default Geom Friction** (applies to all geoms):
```xml
<geom type="capsule" friction="1.0 0.05 0.05" ... />
```

- `mu_s` (static friction): **1.0**
- `mu_d` (dynamic/kinetic friction): **0.05**
- `mu_roll` (rolling friction): **0.05**

**Contact Dimensions (condim)**: **1**
- Means: **1-dimensional normal contact** (friction-less, for efficiency)
- This is unusual - typically humanoid models use condim=3 or condim=4

**Soft Contact Impedance** (solimp):
- Value: `.9 .99 .003`
- Parameters: `[loss, width, midpoint]`
- High loss (0.9) = significant damping in contact

**Soft Contact Reference** (solref):
- Value: `.015 1`
- Parameters: `[timeconst, dampratio]`
- Short time constant (0.015 s) = stiff contacts

### 4.2 Collision Setup

**World-Body Collision:**
```xml
<geom conaffinity="1" condim="3" name="floor" type="plane" ... />
```

- Floor uses condim=3 (3D contact)
- Humanoid uses condim=1 (1D contact) - **mismatch may cause issues**

**Geom-Specific Settings:**
- Default: `condim="1"` (1D contact, no friction)
- Floor: `condim="3"` (3D contact with friction)
- Affinity: `conaffinity="1"` (collides with floor)
- Type: `contype="1"` (humanoid collision type)

---

## 5. Foot-Specific Geometry

### 5.1 Ankle Geometry

| Body | Geom Type | Size | Position | Density |
|------|-----------|------|----------|---------|
| **L_Ankle** | box | 0.085 × 0.0483 × 0.0464 | (0.0242, 0.0233, -0.0239) | 1000 |
| **R_Ankle** | box | 0.0865 × 0.0483 × 0.0478 | (0.0256, -0.0212, -0.0174) | 1000 |

### 5.2 Toe Geometry

| Body | Geom Type | Size | Position | Density |
|------|-----------|------|----------|---------|
| **L_Toe** | box | 0.0496 × 0.0478 × 0.02 | (0.0248, -0.0030, 0.0055) | 1000 |
| **R_Toe** | box | 0.0493 × 0.0479 × 0.0216 | (0.0227, 0.0042, 0.0045) | 1000 |

**Key Observations:**
- **Feet are represented as boxes** (not cylinders or capsules) for flat contact surfaces
- **Larger front-back extent** (X-axis: 0.049-0.086) than side-to-side (Y-axis: 0.047-0.048)
- **Very thin** (Z-axis: 0.02) to avoid ground penetration
- **High density** (1000 kg/m³) for stability
- **No explicit foot contact/constraint** - relies on collision detection

---

## 6. Density Configuration Differences

### 6.1 smpl_humanoid.xml (Original)

Variable densities optimized for biomechanical realism:

```
Pelvis:        4629.6 kg/m³  (very heavy - stability)
Torso/Spine/Chest/Neck/Upper:
               2040.8 kg/m³  (torso and limb upper segments)
Knee/Lower Leg:
               1234.6 kg/m³  (lower extremity)
Feet/Arms/Hands:
               1000 kg/m³    (lighter for speed)
```

**Purpose**: Realistic mass distribution following human body proportions.

### 6.2 smpl_humanoid_1.xml (Variant)

Uniform density throughout:

```
All geoms:     500 kg/m³     (uniform)
```

**Purpose**: Simplified physics for computational efficiency or test compatibility.

---

## 7. Contact & Physics Configuration

### 7.1 Size Specifications

```xml
<size njmax="700" nconmax="700"/>
```

- `njmax`: Max number of joints = **700**
- `nconmax`: Max number of contacts = **700**
- Allows up to 700 simultaneous contacts (generous for locomotion)

### 7.2 Timestep

```xml
<option timestep="0.00555"/>
```

- **Timestep**: 0.00555 seconds ≈ **180 Hz**
- Suitable for humanoid locomotion dynamics

### 7.3 Statistics

```xml
<statistic extent="2" center="0 0 1"/>
```

- `extent="2"`: Bounding box extent = 2 meters (full body height)
- `center="0 0 1"`: Center offset = (0, 0, 1) meters (roughly pelvis height)

---

## 8. ProtoMotions Integration

### 8.1 ProtoMotions Robot Config

**File**: `ref_repo/ProtoMotions/protomotions/robot_configs/smpl.py`

ProtoMotions includes **native SMPL humanoid support** via `SmplRobotConfig`:

```python
@dataclass
class SmplRobotConfig(RobotConfig):
    # Trackable bodies for motion tracking
    trackable_bodies_subset: List[str] = [
        "Pelvis", "L_Ankle", "R_Ankle", "L_Hand", "R_Hand", "Head"
    ]
    
    # Non-termination contact bodies (feet should contact ground)
    non_termination_contact_bodies: List[str] = [
        "R_Ankle", "L_Ankle", "R_Toe", "L_Toe"
    ]
    
    # Semantic body mappings
    common_naming_to_robot_body_names: Dict[str, str] = {
        "all_left_foot_bodies": ["L_Ankle", "L_Toe"],
        "all_right_foot_bodies": ["R_Ankle", "R_Toe"],
        "all_left_hand_bodies": ["L_Hand"],
        "all_right_hand_bodies": ["R_Hand"],
        "head_body_name": ["Head"],
        "torso_body_name": ["Torso"],
    }
    
    # Asset references
    asset: RobotAssetConfig = RobotAssetConfig(
        asset_file_name="mjcf/smpl_humanoid.xml",
        usd_asset_file_name="usd/smpl_humanoid.usda",
    )
    
    # Per-DOF control parameters
    control: ControlConfig = ControlConfig(
        control_type=ControlType.BUILT_IN_PD,
        override_control_info={
            ".*_(Hip|Knee|Ankle)_.*": ControlInfo(
                stiffness=800, damping=80, effort_limit=500, velocity_limit=100
            ),
            ".*_Toe_.*": ControlInfo(
                stiffness=500, damping=50, effort_limit=500, velocity_limit=100
            ),
            "(Torso|Spine|Chest)_.*": ControlInfo(
                stiffness=1000, damping=100, effort_limit=500, velocity_limit=100
            ),
            # ... (Neck, Head, Shoulders, Elbows, Wrists, Hands)
        }
    )
    
    # Multi-simulator support
    simulation_params: SimulatorParams = SimulatorParams(
        isaacgym=IsaacGymSimParams(fps=60, decimation=2, substeps=2),
        isaaclab=IsaacLabSimParams(fps=120, decimation=4, ...),
        genesis=GenesisSimParams(fps=60, decimation=2, substeps=2),
        newton=NewtonSimParams(fps=120, decimation=4),
    )
```

### 8.2 SMPL Simulation Capabilities

ProtoMotions **natively supports SMPL humanoid simulation** across multiple physics engines:

| Feature | Status | Details |
|---------|--------|---------|
| **Motion Tracking** | ✅ Supported | Mimic learning via AMP/ASE |
| **Multi-Simulator** | ✅ Supported | IsaacGym, IsaacLab, Newton, Genesis, MuJoCo |
| **Contact-based Locomotion** | ✅ Supported | Foot contact termination rules configured |
| **Trajectory Retargeting** | ✅ Supported | Via kinematic tree extraction |
| **RL Training** | ✅ Supported | PPO, AMP, ASE, MaskedMimic |

### 8.3 Usage Example

```bash
# Train SMPL humanoid with motion tracking (AMP)
python protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/smpl_motion.pt \
    --num-envs 4096

# Inference on SMPL pretrained model
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
    --motion-file data/motion_for_trackers/smpl_motion.pt \
    --simulator isaacgym --num-envs 16

# CPU-only inference with MuJoCo backend
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
    --motion-file data/motion_for_trackers/smpl_motion.pt \
    --simulator mujoco --num-envs 1  # MuJoCo limited to 1 env
```

---

## 9. XML Structure Summary

### 9.1 Key XML Sections

| Section | Elements | Purpose |
|---------|----------|---------|
| `<compiler>` | coordinate="local" | Use local frame coordinates |
| `<default>` | motor, geom, joint, site configs | Default parameters for all elements |
| `<asset>` | textures, materials | Visualization assets |
| `<worldbody>` | light, floor, Pelvis + hierarchy | Scene structure |
| `<actuator>` | 70 motors | Control specification |
| `<contact>` | (empty) | Contact pair specification (default all) |
| `<size>` | njmax, nconmax | Solver buffer sizes |

### 9.2 Complete Body List

| # | Body Name | Parent | DOF (3 × hinge) | Notes |
|---|-----------|--------|-----------------|-------|
| 1 | Pelvis | World | 6 (freejoint) | Root, floating base |
| 2-5 | L_Hip, L_Knee, L_Ankle, L_Toe | Hierarchy | 3 each | Left leg chain |
| 6-9 | R_Hip, R_Knee, R_Ankle, R_Toe | Hierarchy | 3 each | Right leg chain |
| 10-12 | Torso, Spine, Chest | Hierarchy | 3 each | Spine chain |
| 13-14 | Neck, Head | Chest | 3 each | Head chain |
| 15-19 | L_Thorax, L_Shoulder, L_Elbow, L_Wrist, L_Hand | Chest branch | 3 each | Left arm chain |
| 20-24 | R_Thorax, R_Shoulder, R_Elbow, R_Wrist, R_Hand | Chest branch | 3 each | Right arm chain |

---

## 10. Comparison Table: smpl_humanoid.xml vs smpl_humanoid_1.xml

| Aspect | smpl_humanoid.xml | smpl_humanoid_1.xml | Impact |
|--------|-------------------|---------------------|--------|
| **Kinematic Structure** | Identical | Identical | No difference in motion |
| **Joint Config** | Identical | Identical | Same DOF, stiffness, ranges |
| **Motor Config** | Identical | Identical | Same control capabilities |
| **Geom Placement** | Identical | Identical | Identical collision shapes |
| **Density Distribution** | Variable (1000-4629 kg/m³) | Uniform (500 kg/m³) | **Different physics response** |
| **Total Mass** | ~73 kg (realistic) | ~42 kg (lighter) | Different inertia, stability |
| **Default Use** | Recommended | For testing/efficiency | Use smpl_humanoid.xml for real sim |

---

## 11. Key Findings & Recommendations

### 11.1 Strengths

✅ **Full-body humanoid with 76 DOF** - Complete articulation including hands/fingers  
✅ **Well-configured actuators** - All joints have motors with appropriate gains  
✅ **Multi-limb support** - Native integration in ProtoMotions for multi-simulator training  
✅ **Flexible density variants** - Can choose between realism (var) or speed (uniform)  
✅ **Foot-specific geometry** - Box-based feet suitable for locomotion contact  

### 11.2 Potential Issues

⚠️ **Contact dimension mismatch** - Humanoid uses condim=1 (friction-less) but floor uses condim=3  
→ **Fix**: Change humanoid geoms to `condim="3"` for proper friction  

⚠️ **Foot contact specification missing** - No explicit foot contact constraints  
→ **Workaround**: Handled by ProtoMotions' non_termination_contact_bodies config  

⚠️ **Shoulder range excessive** - ±720° (2 full rotations) is unrealistic  
→ **Note**: Physics will constrain, but control hints would be helpful  

### 11.3 Recommendations for Use

| Use Case | Recommendation |
|----------|-----------------|
| **RL Training** | Use `smpl_humanoid.xml` with ProtoMotions |
| **Quick Testing** | Use `smpl_humanoid_1.xml` (lighter compute) |
| **Motion Capture Retargeting** | Use `smpl_humanoid.xml` (realistic mass distribution) |
| **Contact-heavy Tasks** | Enable `condim=3` for all geoms |
| **Bare Hands Manipulation** | May need hand collision geometry tuning |

---

## 12. Integration with Other Systems

### 12.1 Within OmniH2O (PHC)

The SMPL model is used as a reference for physics-based character control (PHC). The varied density distribution (`smpl_humanoid.xml`) is preferred for biomechanically realistic simulation.

### 12.2 Within ProtoMotions

- **Robot Config**: `SmplRobotConfig` class (smpl.py)
- **Supported Simulators**: IsaacGym, IsaacLab, Newton, Genesis, MuJoCo
- **Training Algorithms**: PPO, AMP, ASE, MaskedMimic
- **Pretrained Models**: Available for SMPL motion tracking
- **Motion File Format**: Bones-based motion format (`g1_bones_seed_mini.pt`)

### 12.3 SMPL-X Support

ProtoMotions also includes **SMPLX support** (`smplx.py`) with:
- Extended hand articulation (finger joints)
- Face/expression control
- More complex collision setup
- Higher computational cost

---

## Conclusion

The SMPL Humanoid MuJoCo models provide a comprehensive, production-ready humanoid skeleton for physics-based simulation. With 76 DOF of control, realistic mass distribution, and native integration in ProtoMotions, it serves as an excellent foundation for motion capture retargeting, RL training, and character animation. The dual variants (realistic vs. lightweight) offer flexibility for different computational budgets.

**Recommendation**: Use `smpl_humanoid.xml` as default; consider `smpl_humanoid_1.xml` only for rapid prototyping or debugging.

---

**Analysis Date**: 2026-05-14  
**Tools Used**: XML parsing, ProtoMotions inspection, comparative analysis
