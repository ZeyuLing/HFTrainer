# SMPL Humanoid MuJoCo Simulation Code Analysis Report

## Executive Summary

The codebase contains **comprehensive SMPL humanoid physics simulation infrastructure** specifically designed for MuJoCo, with implementations spanning multiple projects (PHC, OmniH2O, ProtoMotions). This is a **mature, reusable codebase** rather than something that needs to be built from scratch.

---

## 1. SMPL MuJoCo XML Models (Physics Assets)

### Location & Files
```
/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/
├── smpl_humanoid.xml      (21 KB) ✓ Primary model
├── smpl_humanoid_1.xml    (21 KB) ✓ Variant
├── mesh_humanoid.xml      (24 KB) ✓ With mesh-based colliders
├── humanoid_template_local.xml (1.6 KB)

/ref_repo/ProtoMotions/protomotions/data/assets/mjcf/
├── smpl_humanoid.xml      ✓ Reusable copy
├── smplx_humanoid.xml     ✓ Extended model with fingers
```

### Key Features

**smpl_humanoid.xml anatomy:**
- **Skeleton structure**: 23 joints (Pelvis + L/R limbs + torso + spine + chest + neck + head + arms)
- **Free-floating base**: `<freejoint name="Pelvis"/>` for unconstrained root motion
- **Control parameters**:
  - Stiffness: 500-1000 N⋅m/rad (higher for torso/spine, lower for extremities)
  - Damping: 50-100 damping coefficients
  - Armature: 0.02 kg⋅m² (reduces numerical instability)
  - **Joint limits**: Realistic DOF ranges (e.g., hip ±90°, knee 0-180°)

**Collision geometry**:
- Capsules for limbs (femur, tibia, humerus, forearm)
- Boxes for feet/hands
- Sphere for head
- Density: 1000 kg/m³ (realistic human ~70-80 kg)

**MuJoCo-specific settings**:
```xml
<option timestep="0.00555"/>  <!-- 180 Hz simulation -->
<geom type="capsule" condim="1" friction="1.0 0.05 0.05" 
      solimp=".9 .99 .003" solref=".015 1"/>
```

---

## 2. SMPL ↔ MuJoCo Conversion Code

### Location
```
/ref_repo/OmniH2O/phc/phc/smpllib/
├── smpl_mujoco.py          (635 lines) ✓ Core conversion
├── smpl_parser.py          (22 KB)     ✓ SMPL parsing
├── smpl_local_robot.py     (102 KB)    ✓ Robot interface
└── smpl_eval.py            (8 KB)      ✓ Evaluation metrics
```

### Key Classes & Functions

**1. SMPLConverter (smpl_mujoco.py)**
- Bridges SMPL pose representations ↔ MuJoCo joint angles
- **Methods**:
  - `qpos_smpl_2_new()`: SMPL body pose → MuJoCo qpos (position/quaternion)
  - `qvel_smpl_2_new()`: Velocity conversion
  - `jpos_new_2_smpl()`: Joint position extraction
  - `get_new_jkp()`: Stiffness extraction
  - `get_new_torque_limit()`: Torque limits per joint

**2. Pose Conversion Functions (smpl_mujoco.py)**
```python
smpl_to_qpose(pose, mj_model, trans=None, ...)
  # Converts 72-dim SMPL pose + 3D translation to MuJoCo qpos
  # Handles quaternion/Euler conversions, batch processing
  # Input: Batch × 72 (pose in angle-axis)
  # Output: Batch × (7+nq_dofs) MuJoCo state

qpos_to_smpl(qpos, mj_model, ...)
  # Inverse: MuJoCo state → SMPL
  # Used for extracting poses after physics simulation

smpl_to_qpose_torch()
  # GPU-accelerated version (PyTorch)
```

**3. SMPL_Robot Class (smpl_local_robot.py)**
- High-level interface for SMPL humanoid control
- ~2600 lines of sophisticated control & IK logic
- **Key methods**:
  - `set_pose()`: Apply target pose with PD control
  - `forward_kinematics()`: Compute joint positions
  - `inverse_kinematics()`: Solve target reaching
  - `get_body_jpos()`: Query current joint positions
  - Contact handling for feet

---

## 3. Physics Simulation Environment

### Location
```
/ref_repo/OmniH2O/phc/phc/env/tasks/
├── humanoid.py             (89 KB) ✓ Base physics environment
├── humanoid_amp.py         (55 KB)   Motion imitation training
├── humanoid_im.py          (91 KB)   Inverse model
└── ... (getup, demo variants)

/ref_repo/ProtoMotions/protomotions/robot_configs/
├── smpl.py                 ✓ Config dataclass for SMPL
└── smplx.py
```

### Physics Loop (humanoid.py example)

The environment runs **continuous physics simulation** with:

1. **Contact detection**: Foot-ground interactions tracked
2. **PD control**: Joint target tracking with damping
3. **Reference tracking**: Reward shapes for motion imitation
4. **State space**: 
   - 7 DOF root (pos + quat) + 63 DOF joints + velocities
   - ~900-1000 dim observation space

**Isaac Gym integration**:
```python
from isaacgym import gymtorch, gymapi
# Batch GPU physics in 1000s of parallel environments
```

---

## 4. Foot Contact & Stability Features

### Existing Infrastructure

**4.1 Contact Detection (humanoid.py, humanoid_amp.py)**
```python
# Examples from codebase:
- self.feet_air_time: Track airtime vs ground contact
- self.contact_forces: Query contact forces (MuJoCo gets contact data)
- self.feet_contact_binary: Binary contact state
- non_termination_contact_bodies: Defined in robot configs (feet/toes)
```

**4.2 Foot Geometry**
- `L_Ankle`, `L_Toe`, `R_Ankle`, `R_Toe` bodies with box geometry
- Foot sizes: ~0.085m × 0.048m × 0.046m (realistic)
- **ConType/ConAffinity**: Set to enable ground collisions

**4.3 Ground Friction**
```xml
<geom friction="1.0 0.05 0.05" solimp=".9 .99 .003" solref=".015 1"/>
<!-- friction = [1.0 (tangent), 0.05, 0.05] -->
<!-- solimp = [spring_damping_impedance: 0.9, compliance: 0.99, limit: 0.003] -->
<!-- solref = [reference: 0.015, stiffness: 1] -->
```

### Foot Sliding / Ground Penetration Issues

**Current Limitations**:
- Standard MuJoCo contact model can produce footskate under fast poses
- No explicit contact constraint solving (beyond MuJoCo's default)
- Feet represented as simple boxes (not deformable/compliant)

**Potential Fixes Available in Codebase**:
1. **UnderPressure reference** in ref_repo index → deep learning for contact cleanup
2. **LODGE foot_contact.py** → footskate detection/learning
3. **Tuning opportunities**:
   - Increase foot friction coefficients
   - Reduce solver gap (solref lower limit)
   - Add contact sensors for feedback control

---

## 5. Related Projects & Dependencies

### 5.1 SMPLSim (External Dependency)
```bash
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master
```
- Used by PHC for **automatic SMPL mesh → MuJoCo XML generation**
- Provides `SkeletonTree`, `SkeletonMotion` classes
- Located at: `from smpl_sim.poselib.skeleton.skeleton3d import ...`

**Usage in codebase**:
```python
# From humanoid.py imports:
from smpl_sim.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
```

### 5.2 PHC (Perpetual Humanoid Control)
- **ICCV 2023 paper**: Real-time physics-based humanoid control
- Located at: `/ref_repo/PHC/phc/`
- Implements imitation control + fail-state recovery
- Full training pipeline with AMASS datasets
- **Already solves similar problems**: robust foot contact, physics stability

### 5.3 ProtoMotions (High-Level Framework)
- Multi-simulator abstraction (Isaac Gym, Isaac Lab, Genesis, Newton)
- SmplRobotConfig provides unified interface for SMPL
- Locations: `/ref_repo/ProtoMotions/protomotions/`

### 5.4 UHC (Universal Humanoid Control)
- Referenced imports: `from uhc.smpllib.smpl_parser import SMPL_Parser`
- SMPL skeleton utilities and parsers

---

## 6. File Paths: Complete Reference

### Core SMPL Physics Code
```
/ref_repo/OmniH2O/phc/phc/smpllib/
  ├── smpl_mujoco.py              # ⭐ SMPL ↔ MuJoCo conversion
  ├── smpl_local_robot.py         # ⭐ Robot control interface
  ├── smpl_parser.py              # SMPL skeleton parsing
  └── smpl_eval.py                # Evaluation utilities

/ref_repo/OmniH2O/phc/phc/env/tasks/
  ├── humanoid.py                 # ⭐ Base physics sim (89 KB)
  ├── base_task.py                # Parent task class
  └── humanoid_*.py               # Task variants (amp, im, etc.)

/ref_repo/OmniH2O/phc/phc/utils/
  ├── motion_lib_smpl.py          # Motion library
  └── torch_h1_humanoid_batch.py  # Batch simulation utilities
```

### XML Model Assets
```
/ref_repo/OmniH2O/phc/phc/data/assets/mjcf/
  ├── smpl_humanoid.xml           # ⭐ Primary 23-DOF humanoid
  ├── smpl_humanoid_1.xml         # Variant
  └── mesh_humanoid.xml           # Mesh-based version

/ref_repo/ProtoMotions/protomotions/data/assets/mjcf/
  ├── smpl_humanoid.xml           # Copy/variant
  └── smplx_humanoid.xml          # With hand fingers
```

### Configuration & Robot Definitions
```
/ref_repo/ProtoMotions/protomotions/robot_configs/
  ├── smpl.py                     # ⭐ SMPL config dataclass
  └── smplx.py                    # Extended SMPL with hands

/ref_repo/OmniH2O/phc/phc/data/
  └── assets/mjcf/humanoid_template_local.xml
```

---

## 7. Dependencies & Pip Packages

### Currently Used
```python
# From imports in smpl_mujoco.py and humanoid.py:
import mujoco                           # MuJoCo 2.0+
import torch, numpy, scipy              # Numerics
from uhc.smpllib.smpl_parser import ... # SMPL skeleton
from smpl_sim.poselib.skeleton import ... # SMPLSim library
```

### Installation
```bash
pip install mujoco torch numpy scipy
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git
```

**Note**: No separate `smplsim` or `uhc` pip packages found in site-packages (likely git-installed or local imports).

---

## 8. Recommendation: Build Strategy

### ✅ Can Reuse Directly

1. **SMPL XML Models** (`smpl_humanoid.xml`)
   - Drop-in ready for MuJoCo
   - Proper collision/contact setup
   - Realistic joint limits

2. **Conversion Functions** (`smpl_mujoco.py`)
   - `smpl_to_qpose()` and `qpos_to_smpl()` handle SMPL ↔ MuJoCo mapping
   - Batch-processing and GPU support

3. **Physics Environment** (humanoid.py + humanoid_amp.py)
   - Full simulation loop with contact detection
   - PD controller tuning
   - Motion imitation rewards

### ⚠️ May Need Customization

1. **Foot Sliding / Ground Penetration Fixes**:
   - Tune MuJoCo contact parameters in XML (`friction`, `solref`, `solimp`)
   - Implement contact-based constraints in control law
   - Consider mesh-based feet instead of boxes

2. **Custom Physics Constraints**:
   - Add explicit velocity tracking to prevent sliding
   - Implement contact Jacobian-based IK
   - Reference: `/ref_repo/UnderPressure/` has deep learning approaches for footskate

3. **Simulation Speed/Stability**:
   - Tune `timestep` in XML (currently 0.00555s = 180 Hz)
   - Adjust solver parameters

---

## 9. Quick Start Path

```bash
# 1. Load SMPL model
model = mujoco.MjModel.from_xml_file('smpl_humanoid.xml')
data = mujoco.MjData(model)

# 2. Convert SMPL pose to MuJoCo qpos
from phc.smpllib.smpl_mujoco import smpl_to_qpose
smpl_pose = np.random.randn(1, 72)  # 72D SMPL pose
qpos = smpl_to_qpose(smpl_pose, model)

# 3. Step physics
mujoco.mj_step(model, data)

# 4. Extract back to SMPL
from phc.smpllib.smpl_mujoco import qpos_to_smpl
smpl_pose_sim, trans = qpos_to_smpl(data.qpos, model)
```

---

## 10. Summary Table

| Component | Location | Status | Purpose |
|-----------|----------|--------|---------|
| **SMPL XML Models** | `phc/data/assets/mjcf/smpl_humanoid.xml` | ✅ Ready | Physics asset |
| **Pose Conversion** | `phc/smpllib/smpl_mujoco.py` | ✅ Ready | SMPL ↔ MuJoCo mapping |
| **Robot Interface** | `phc/smpllib/smpl_local_robot.py` | ✅ Ready | High-level control |
| **Physics Sim** | `phc/env/tasks/humanoid.py` | ✅ Ready | Full simulation loop |
| **Contact Handling** | humanoid.py, XML geom setup | ⚠️ Basic | May need enhancement |
| **Foot Sliding Fix** | Not explicit | ❌ Custom | Needs implementation |
| **Config** | `ProtoMotions/robot_configs/smpl.py` | ✅ Ready | Multi-sim config |

---

## Conclusion

**You have substantial reusable infrastructure:**
- ✅ Professional SMPL MuJoCo models with realistic collision geometry
- ✅ Proven SMPL ↔ MuJoCo conversion pipeline
- ✅ Full physics simulation environment (Isaac Gym backed)
- ✅ Contact detection framework

**To fix foot sliding/ground penetration:**
1. Tune existing XML parameters (friction, contact impedance)
2. Leverage conversion code for post-processing
3. Add control-level constraints (reference implementations in UnderPressure/LODGE)
4. Consider mesh-based geometry for feet

**No need to build from scratch** — build *on top* of this proven stack.

