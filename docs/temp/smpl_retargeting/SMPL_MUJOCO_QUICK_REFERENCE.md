# SMPL-MuJoCo Quick Reference Guide

## TL;DR: What Exists

| What | Where | Lines | Status |
|------|-------|-------|--------|
| **SMPL XML Model** | `ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml` | - | ✅ Ready |
| **Pose Conversion** | `ref_repo/OmniH2O/phc/phc/smpllib/smpl_mujoco.py` | 635 | ✅ Ready |
| **Robot Interface** | `ref_repo/OmniH2O/phc/phc/smpllib/smpl_local_robot.py` | 2600 | ✅ Ready |
| **Physics Loop** | `ref_repo/OmniH2O/phc/phc/env/tasks/humanoid.py` | 89K | ✅ Ready |
| **Foot Contact** | XML + humanoid.py | - | ⚠️ Basic |
| **Footskate Fix** | - | - | ❌ Custom |

---

## Directory Tree (Essential)

```
ref_repo/
├── OmniH2O/phc/phc/
│   ├── smpllib/
│   │   ├── smpl_mujoco.py         ⭐ Main conversion
│   │   ├── smpl_local_robot.py    ⭐ Robot class
│   │   ├── smpl_parser.py
│   │   └── smpl_eval.py
│   ├── env/tasks/
│   │   ├── humanoid.py            ⭐ Physics sim
│   │   ├── humanoid_amp.py
│   │   └── ...
│   ├── utils/
│   │   ├── motion_lib_smpl.py
│   │   └── torch_h1_humanoid_batch.py
│   └── data/assets/mjcf/
│       ├── smpl_humanoid.xml      ⭐ Primary model
│       ├── smpl_humanoid_1.xml
│       └── mesh_humanoid.xml
│
├── ProtoMotions/protomotions/
│   ├── robot_configs/
│   │   ├── smpl.py                ⭐ Config
│   │   └── smplx.py
│   └── data/assets/mjcf/
│       ├── smpl_humanoid.xml
│       └── smplx_humanoid.xml
│
├── PHC/phc/                       (Reference: full implementation)
└── UnderPressure/                 (Reference: footskate solutions)
```

---

## Key Functions at a Glance

### Load & Convert
```python
import mujoco
import numpy as np
from phc.smpllib.smpl_mujoco import smpl_to_qpose, qpos_to_smpl

# 1. Load model
model = mujoco.MjModel.from_xml_file('smpl_humanoid.xml')
data = mujoco.MjData(model)

# 2. Convert 72D SMPL pose to MuJoCo
smpl_pose = np.zeros(72)  # Batch size 1
smpl_trans = np.array([[0, 0, 1.0]])
qpos = smpl_to_qpose(smpl_pose, model, trans=smpl_trans)
data.qpos[:len(qpos)] = qpos

# 3. Step physics
mujoco.mj_step(model, data)

# 4. Convert back
pose_out, trans_out = qpos_to_smpl(data.qpos, model)
```

### Robot Control
```python
from phc.smpllib.smpl_local_robot import SMPL_Robot

robot = SMPL_Robot(
    model=model,
    data=data,
    new_model_name='name'  # Optional
)

# Apply pose with PD control
robot.set_pose(target_qpos)

# Get joint positions
joint_pos = robot.get_body_jpos()

# Forward kinematics
fk_result = robot.forward_kinematics()

# Inverse kinematics
ik_result = robot.inverse_kinematics(target_pos)
```

### Batch Processing
```python
# For batch processing (GPU)
from phc.smpllib.smpl_mujoco import smpl_to_qpose_torch
import torch

batch_poses = torch.randn(1000, 72)  # GPU tensor
batch_trans = torch.zeros(1000, 3)
batch_qpos = smpl_to_qpose_torch(
    batch_poses, model, trans=batch_trans
)
```

---

## SMPL Model Structure

**Skeleton (23 DOF):**
```
Pelvis (free joint)
├── L_Hip (x, y, z hinge)
│   └── L_Knee (x, y, z hinge)
│       └── L_Ankle (x, y, z hinge)
│           └── L_Toe (x, y, z hinge)
├── R_Hip (x, y, z hinge)
│   └── R_Knee (x, y, z hinge)
│       └── R_Ankle (x, y, z hinge)
│           └── R_Toe (x, y, z hinge)
└── Torso (x, y, z hinge)
    └── Spine (x, y, z hinge)
        └── Chest (x, y, z hinge)
            ├── Neck (x, y, z hinge)
            │   └── Head (x, y, z hinge)
            ├── L_Thorax (x, y, z hinge)
            │   └── L_Shoulder (x, y, z hinge)
            │       └── L_Elbow (x, y, z hinge)
            │           └── L_Wrist (x, y, z hinge)
            │               └── L_Hand (x, y, z hinge)
            └── R_Thorax (x, y, z hinge)
                └── R_Shoulder (x, y, z hinge)
                    └── R_Elbow (x, y, z hinge)
                        └── R_Wrist (x, y, z hinge)
                            └── R_Hand (x, y, z hinge)
```

**Total DOFs: 7 (free root) + 3×21 (joints) = 70 DOF state**

---

## Contact Detection (Existing)

```python
# From humanoid.py - how it currently works
feet_contact_binary = np.zeros(num_envs)

# Track contact forces
contact_forces = data.cfrc_ext  # External contact forces [N, T]

# Feet air time
feet_air_time = np.zeros(num_envs)

# Query specific body contact
foot_body_id = model.body('L_Ankle').id
foot_contacts = [c for c in data.contact if c.geom1 == floor_geom or c.geom2 == floor_geom]
```

**Bodies tracking contacts:**
- `L_Ankle`, `L_Toe` (left foot)
- `R_Ankle`, `R_Toe` (right foot)

---

## XML Contact Tuning Parameters

Located in `smpl_humanoid.xml`:

```xml
<!-- Global defaults -->
<geom type="capsule" 
      friction="1.0 0.05 0.05"      <!-- [tangent, rolling, spinning] -->
      solimp=".9 .99 .003"           <!-- [impedance params] -->
      solref=".015 1"                <!-- [reference distance, stiffness] -->
      condim="1"/>                   <!-- Collision type (1=friction, 3=with rolling) -->

<!-- Simulation timestep -->
<option timestep="0.00555"/>          <!-- 5.55ms = 180 Hz -->

<!-- Joint stiffness/damping -->
<joint ... stiffness="800" damping="80" ... />
```

**To reduce footskate:**
1. Increase friction: `friction="2.0 0.1 0.1"`
2. Lower solver tolerance: `solref=".005 1"`  (was `.015`)
3. Increase condim to 3 for rolling friction
4. Increase foot geom margins

---

## Dependencies

```bash
# Core requirements
pip install mujoco>=2.0 torch numpy scipy

# Optional but recommended
pip install git+https://github.com/ZhengyiLuo/SMPLSim.git@master

# For Isaac Gym integration
pip install isaacgym  # Requires signup at developer.nvidia.com
```

---

## Common Issues & Fixes

### Issue: Feet penetrating ground
**Solution:** Adjust XML margin and solver:
```xml
<geom ... margin="0.002" solref=".005 1" />
```

### Issue: Unnatural foot sliding
**Solution:** Implement contact constraint in control:
```python
# Pseudo-code
if foot_contact[i]:
    # Prevent foot velocity in XY plane
    contact_jacobian = compute_contact_jacobian(model, data, foot_body)
    null_space_velocity = project_to_null_space(control, contact_jacobian)
```

### Issue: Model doesn't load
**Solution:** Check XML path and mujoco version:
```bash
pip install --upgrade mujoco
python -c "import mujoco; print(mujoco.__version__)"
```

---

## Performance Tips

1. **Vectorized operations:** Use `smpl_to_qpose_torch()` for GPU batch processing
2. **Contact updates:** Only query contact data when needed (expensive)
3. **Solver settings:** Tune nconmax, iterations in physics options
4. **Timestep:** 5.55ms good balance; smaller = slower, larger = unstable

---

## References in Codebase

- **PHC**: `/ref_repo/PHC/` — Full humanoid control system (ICCV 2023)
- **UnderPressure**: `/ref_repo/UnderPressure/` — Footskate detection/fixing
- **LODGE**: `/ref_repo/LODGE/dld/losses/foot_contact.py` — Contact loss functions
- **ProtoMotions**: Multi-sim abstraction, config system

---

## Next Steps

1. ✅ Load `smpl_humanoid.xml` with `mujoco.MjModel.from_xml_file()`
2. ✅ Test conversion: `smpl_to_qpose()` → step physics → `qpos_to_smpl()`
3. ⚠️  Tune XML contact parameters for your specific motion
4. ❌ Implement custom footskate prevention (reference UnderPressure/LODGE)

**Total setup time: ~30 minutes to get basic sim running**

