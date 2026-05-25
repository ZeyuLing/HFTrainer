# SMPL Humanoid MuJoCo Physics Simulation - Practical Approaches Report

## Executive Summary

Your codebase has **excellent resources for SMPL-MuJoCo simulation**. ProtoMotions includes pre-built SMPL humanoid MuJoCo XML models, complete robot configurations, and a full pipeline for motion processing. The key assets are:

✅ **Ready-to-use**: `smpl_humanoid.xml` and `smplx_humanoid.xml` (238-491 lines each)  
✅ **Configured**: `SmplRobotConfig` and `SMPLXRobotConfig` in ProtoMotions  
✅ **Data pipeline**: Scripts to convert motion_135 NPZ → SMPL parameters → physics simulation  
✅ **SMPL models**: Full SMPL/SMPL-H/SMPL-X model files available in checkpoints  
✅ **Simulation framework**: ProtoMotions supports multiple simulators (MuJoCo via Newton/Genesis)

---

## 1. SMPL-MuJoCo XML Files Found

### Location: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/`

**Available Models:**
```
smpl_humanoid.xml          (238 lines) - Basic SMPL 22-joint humanoid
smplx_humanoid.xml         (491 lines) - SMPL-X with hands + facial features
soma23_humanoid.xml        (pre-configured alternative)
rigv1_humanoid.xml         (pre-configured alternative)
```

### XML Structure - SMPL Model

**Key Features:**
- **Joints**: Full kinematic tree from Pelvis → Limbs
- **Body segments**: 23 bodies (Pelvis + 22 joints)
- **Each joint**: 3 DOF (X, Y, Z hinge) with:
  - Stiffness: 500-1000 (configurable per body)
  - Damping: 50-100 (friction/damping)
  - Range: ±180° (full rotation)
  - Armature: 0.02 (motor properties)
  
- **Collision geometry**:
  - Capsules for limbs (legs, arms)
  - Boxes for feet, hands, head
  - Density values calibrated for realistic mass distribution
  - Condim=3 (3-contact detection)

- **Actuators**: 69 motors (3 DOF × 23 bodies)
  - Gear ratio: 500 (torque multiplier)
  - Motor control for each joint

**Example structure:**
```xml
<body name="L_Hip" pos="...">
  <joint name="L_Hip_x" type="hinge" axis="1 0 0" ... />
  <joint name="L_Hip_y" type="hinge" axis="0 1 0" ... />
  <joint name="L_Hip_z" type="hinge" axis="0 0 1" ... />
  <geom type="capsule" fromto="..." size="0.0615" density="2040.8" ... />
  <body name="L_Knee"> ... </body>
</body>
```

---

## 2. ProtoMotions SMPL Support

### Robot Configurations

#### **SmplRobotConfig** (`protomotions/robot_configs/smpl.py`)
```python
@dataclass
class SmplRobotConfig(RobotConfig):
    asset = RobotAssetConfig(
        asset_file_name="mjcf/smpl_humanoid.xml",
        usd_asset_file_name="usd/smpl_humanoid.usda",
    )
    
    control = ControlConfig(
        control_type=ControlType.BUILT_IN_PD,
        override_control_info={
            ".*_(Hip|Knee|Ankle)_.*": ControlInfo(stiffness=800, damping=80),
            ".*_Toe_.*": ControlInfo(stiffness=500, damping=50),
            "(Torso|Spine|Chest)_.*": ControlInfo(stiffness=1000, damping=100),
            # ... more configs
        }
    )
    
    simulation_params = SimulatorParams(
        isaacgym=IsaacGymSimParams(fps=60, decimation=2, substeps=2),
        isaaclab=IsaacLabSimParams(fps=120, decimation=4),
        newton=NewtonSimParams(fps=120, decimation=4),
    )
```

**Key fields:**
- `trackable_bodies_subset`: [Pelvis, L_Ankle, R_Ankle, L_Hand, R_Hand, Head]
- `non_termination_contact_bodies`: [R_Ankle, L_Ankle, R_Toe, L_Toe]
- `common_naming_to_robot_body_names`: Maps semantic names to body names
- `default_root_height`: 0.95 m (initial spawn height)

#### **SMPLXRobotConfig** (`protomotions/robot_configs/smplx.py`)
- Extended with hand detail: 30 hand joints (15 per hand)
- Individual finger control (Index, Middle, Pinky, Ring, Thumb)
- Contact bodies include all finger segments
- Lower stiffness for fingers (10-300 N/m)

### Environment Integration

**ProtoMotions provides:**
1. **MimicMotionManager**: Tracks reference motion, computes pose targets
2. **KinematicReplayControl**: Direct kinematic playback (no physics)
3. **MotionLib**: Loads motion sequences, samples motions per environment
4. **BaseEnv**: Handles reset, step, reward, termination logic

**Simulators support:**
- IsaacGym (NVIDIA GPU)
- IsaacLab (newer NVIDIA)
- Newton (CPU/GPU, MuJoCo-based)
- Genesis (CPU/GPU)

---

## 3. Data Pipeline: motion_135 → Physics Simulation

### Format: motion_135 NPZ

**Structure:**
```python
data = np.load("motion_135.npz")
motion_135: (T, 135)  # T frames, 135 dimensions
  - [0:3]:    translation (3D position)
  - [3:135]:  22 joints × 6D rotation (rot6d)
fps: int               # Frame rate (typically 30)
```

### Conversion Pipeline

**Script 1: `motion135_to_smplx.py`**
- Input: motion_135 NPZ
- Process:
  1. Split transl (3) + rot6d (22×6)
  2. Convert rot6d → rotation matrix (Gram-Schmidt)
  3. Convert rotation matrix → axis-angle (3 per joint)
- Output: SMPL-X NPZ with:
  ```python
  pose_body:      (T, 63)   # 21 body joints × 3 (axis-angle)
  root_orient:    (T, 3)    # Pelvis orientation
  trans:          (T, 3)    # Translation
  betas:          (10,)     # Shape params (zeros)
  gender:         "neutral"
  mocap_frame_rate: 30
  ```

**Script 2: `batch_npz_to_smpl_mesh_json.py`**
- Input: motion_135 NPZ
- Output: SMPL mesh JSON (for web visualization)
- Format:
  ```json
  {
    "type": "frames",
    "fps": 30,
    "frames": [
      [{
        "id": 0,
        "gender": "neutral",
        "smpl_type": "smplx",
        "Rh": [[rx, ry, rz]],      // root orientation
        "Th": [[tx, ty, tz]],      // translation
        "poses": [[p0, p1, ...]],  // all joints axis-angle
        "shapes": [[0, ..., 0]],   // shape coefficients
        "mocap_framerate": 30
      }],
      ...
    ]
  }
  ```

**Script 3: `generate_smpl_mesh_vertices.py`**
- Input: SMPL mesh JSON
- Process: Forward kinematics using `smplx` library
  ```python
  import smplx
  model = smplx.create(model_path, model_type="smplh", gender="neutral")
  output = model(
      global_orient=root_orient,
      body_pose=body_pose,
      left_hand_pose=left_hand_pose,
      right_hand_pose=right_hand_pose,
      transl=transl,
      betas=betas,
  )
  vertices = output.vertices  # (T, 6890, 3)
  ```
- Output: Binary vertex file (float16 per frame)

### Data Flow Diagram

```
motion_135.npz (T, 135)
        ↓
motion135_to_smplx.py
        ↓
smpl_x.npz (T, 63+3+3)
        ↓
Directly feed to MuJoCo controller
or
batch_npz_to_smpl_mesh_json.py
        ↓
smpl_mesh.json (full SMPL pose params)
        ↓
generate_smpl_mesh_vertices.py (SMPL FK)
        ↓
vertices.bin (T, 6890, 3 positions)
        ↓
Web visualization / Physics simulation
```

---

## 4. Available SMPL Model Files

### Location: `checkpoints/smpl_models/`

**Files present:**
```
smpl/                           (595 MB) - SMPL body models (neutral, male, female)
smplh/                          (symbolic link) - SMPL+H with hands
smplx/                          (3.2 GB) - SMPL-X with hands + face
J_regressor_h36m.npy            - Joint regressor for H3.6M keypoints
J_regressor_extra.npy           - Extra joint regressors
smpl_coco17_J_regressor.pt      - COCO17 keypoint regression
smpl_neutral_J_regressor.pt     - Neutral model regression
```

### Model Contents

Each SMPL model directory contains:
```
SMPL_NEUTRAL.pkl       # Neutral gender model (vertex 6890, 23 joints)
SMPL_MALE.pkl
SMPL_FEMALE.pkl
```

**What's inside a SMPL model (.pkl):**
- `v_template`: (6890, 3) mean vertex positions
- `J_regressor`: (23, 6890) sparse joint regressor
- `weights`: (6890, 23) blend weights (skinning)
- `faces`: (13776, 3) triangle indices
- `shapedirs`: (6890, 3, 10) shape basis vectors (PCA)
- `posedirs`: (6890, 3, 207) pose basis vectors (blend shapes)

---

## 5. Existing Integration in ProtoMotions

### Example: Using SMPL in an Environment

```python
# env_kinematic_playback.py supports SMPL
python examples/env_kinematic_playback.py \
    --robot-name=smpl \
    --simulator=isaacgym \
    --num-envs=1 \
    --motion-file=motion.pt \
    --experiment-path=examples/experiments/mimic/mlp.py
```

### How ProtoMotions Handles SMPL Motion

1. **Load motion** → MotionLib loads pose sequences
2. **Sample motion per env** → Select random motion clip
3. **Get target pose** → Extract pose_body, root_orient at time t
4. **Send to controller** → MimicMotionManager computes PD targets
5. **Simulate** → Physics solver (MuJoCo/IsaacGym) applies forces
6. **Compute reward** → Pose tracking error, contact penalties, etc.
7. **Reset** → When motion ends or episode terminates

### Available Observation Components

From `protomotions/envs/obs/`:
- `state_history_buffer.py`: Recent frames of state
- `terrain_obs.py`: Terrain height queries
- `scene_obs.py`: Object positions/velocities
- Multiple pose/velocity observation options

---

## 6. What's Missing (Practical Gaps)

### **1. Direct MuJoCo Python Integration**
- ✅ XML models exist but are used via Newton/Genesis/IsaacGym
- ❌ No direct `mujoco.py` wrapper for SMPL models
- **Solution**: Use Newton simulator (MuJoCo backend) via ProtoMotions

### **2. SMPL Shape (Beta) Dynamics**
- ✅ SMPL models support shape parameters (betas)
- ❌ No motion-in-the-loop shape optimization
- **Note**: Shape typically fixed during motion (zero betas = neutral)

### **3. Mesh-Based Contact Simulation**
- ✅ XML models use primitive shapes (capsules, boxes)
- ❌ No full SMPL mesh collision geometry
- **Why**: Computational cost; primitives sufficient for locomotion
- **Alternative**: Use high-res mesh for visual rendering only

### **4. Hand Control Details**
- ✅ SMPLX has hand joints (30 total)
- ⚠️ No independent finger animation in motion_135
- **Reason**: motion_135 is 22-joint SMPL (no hands), not SMPLX

### **5. Motion Database Integration**
- ✅ Scripts exist to convert motion_135 → SMPL-X NPZ
- ❌ No built-in AMASS dataset loader for SMPL
- **Workaround**: Motion data exists in motion_135 format; convert as needed

---

## 7. Most Practical Approach for Your Use Case

### **Recommended Path: Use ProtoMotions Directly**

**Why**: ProtoMotions has everything integrated; you just need to instantiate environments with SMPL.

### **Step 1: Prepare Motion Data**
```bash
# Your motion_135.npz files are already in the right format
# Convert to SMPL-X NPZ for motion tracking
python scripts/embodied/motion135_to_smplx.py \
    input_motion_135.npz \
    output_smpl_x.npz \
    --fps 30
```

### **Step 2: Create Environment**
```python
from protomotions.envs.base_env.env import BaseEnv
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.simulator.newton.simulator import NewtonSimulator

# Load your motion data
motion_data = np.load("output_smpl_x.npz")

# Create robot config
robot_cfg = SmplRobotConfig()

# Create simulator (Newton has MuJoCo backend)
sim_cfg = NewtonSimulatorConfig(fps=120, num_envs=4)
simulator = NewtonSimulator(sim_cfg, robot_cfg)

# Create environment
env_cfg = EnvConfig(...)
env = BaseEnv(env_cfg, robot_cfg, simulator)

# Step through simulation
for step in range(1000):
    obs, rew, done, info = env.step(actions)
    if done.any():
        obs = env.reset()
```

### **Step 3: Motion Tracking**
```python
# Use MimicMotionManager to track motion
from protomotions.envs.motion_manager import MimicMotionManager

motion_manager = MimicMotionManager(
    config=config,
    num_envs=4,
    env_dt=1/120,
    device=device,
    motion_lib=motion_lib,
)

# Get target poses each step
target_poses = motion_manager.get_target_poses(time_indices)
```

### **Step 4: Visualization / Analysis**
```python
# Option A: Direct MuJoCo viewer (if using Newton)
# Option B: Export to SMPL mesh JSON
python scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-file output_smpl_x.npz \
    --output-dir smpl_meshes/ \
    --smpl-type smplx

# Option C: Generate mesh vertices for rendering
python scripts/embodied/generate_smpl_mesh_vertices.py \
    --input-dir smpl_meshes/ \
    --output-dir smpl_vertices/ \
    --smpl-model-path checkpoints/smpl_models/
```

---

## 8. Alternative: Pure MuJoCo Approach (No ProtoMotions)

If you want **standalone MuJoCo control**:

### **Step 1: Load SMPL XML**
```python
import mujoco as mj

model = mj.MjModel.from_xml_file(
    "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml"
)
data = mj.MjData(model)
```

### **Step 2: Set Pose from motion_135**
```python
# Convert motion_135 → axis-angle
from scipy.spatial.transform import Rotation as R

motion = data['motion_135']  # (T, 135)
transl = motion[:, :3]       # (T, 3)
rot6d = motion[:, 3:].reshape(T, 22, 6)

# Gram-Schmidt to rotation matrix (see script above)
# Then to axis-angle

# Set in MuJoCo
data.qpos[:3] = transl[t]      # Root position
data.qpos[3:] = axis_angles    # All joint angles (should match XML order)
```

### **Step 3: Simulate**
```python
for step in range(steps):
    mj.mj_step(model, data)  # Physics step
    # Access: data.qpos (positions), data.qvel (velocities)
```

### **Challenges with pure MuJoCo:**
- Need to manage joint ordering carefully
- Limited observation pipeline compared to ProtoMotions
- No built-in reward functions or motion sampling
- Manual controller implementation required

---

## 9. Key Technical Details

### **Joint Ordering in SMPL XML**
The `smpl_humanoid.xml` defines 23 bodies (root + 22 joints):
```
0:  Pelvis (root, free joint = 6 DOF)
1:  L_Hip (3 joints)
2:  L_Knee (3 joints)
3:  L_Ankle (3 joints)
4:  L_Toe (3 joints)
5:  R_Hip (3 joints)
... (symmetric for right side)
...  Torso, Spine, Chest, Neck, Head, Shoulders, Elbows, Wrists, Hands
```

MuJoCo `data.qpos` order:
```
[0:7]    Pelvis: free joint (quat 4-element + pos 3-element)
[7:10]   L_Hip_x, L_Hip_y, L_Hip_z (hinge angles)
[10:13]  L_Knee_x, L_Knee_y, L_Knee_z
... (60+ more dimensions)
Total: ~69 dimensions (free joint = 7, 22 hinge joints × 3 = 66)
```

### **Rot6D to Axis-Angle Conversion**
```python
# HyMotion outputs row-major: [R00,R01, R10,R11, R20,R21]
# Reorder to column-major for Gram-Schmidt
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
# Then Gram-Schmidt + cross product to get 3×3 matrix
# Then scipy.spatial.transform.Rotation.from_matrix().as_rotvec()
```

### **Physics Stability Tips**
- Stiffness values (800-1000) are conservative; adjust per use case
- Damping (80-100) prevents oscillation; increase if unstable
- Armature (0.02) adds motor inertia; helps with control
- Condim=3: friction model with rolling resistance
- Margin=0.001: contact margin for stability

---

## 10. Summary Table

| Resource | Location | Status | Use Case |
|----------|----------|--------|----------|
| **SMPL XML models** | `ProtoMotions/data/assets/mjcf/smpl_humanoid.xml` | ✅ Ready | Direct MuJoCo simulation |
| **SMPLX XML model** | `ProtoMotions/data/assets/mjcf/smplx_humanoid.xml` | ✅ Ready | Hand-detailed simulation |
| **Robot configs** | `ProtoMotions/robot_configs/smpl.py` | ✅ Ready | ProtoMotions integration |
| **Motion lib** | `ProtoMotions/components/motion_lib.py` | ✅ Ready | Motion sampling & tracking |
| **motion_135 converter** | `scripts/embodied/motion135_to_smplx.py` | ✅ Ready | Convert to SMPL NPZ |
| **Mesh JSON converter** | `scripts/embodied/batch_npz_to_smpl_mesh_json.py` | ✅ Ready | Web visualization |
| **Mesh vertex generator** | `scripts/embodied/generate_smpl_mesh_vertices.py` | ✅ Ready | Rendering pipeline |
| **SMPL model files** | `checkpoints/smpl_models/` | ✅ Complete | FK/IK & rendering |
| **Kinematic playback** | `ProtoMotions/examples/env_kinematic_playback.py` | ✅ Works | Motion playback (no physics) |
| **Physics environments** | `ProtoMotions/envs/base_env/` | ✅ Ready | Physics-based training |
| **Hand articulation** | `ProtoMotions/robot_configs/smplx.py` | ✅ Supported | Detailed hand control |

---

## 11. Next Steps Recommendation

### **Phase 1: Validate (1-2 hours)**
1. Load `smpl_humanoid.xml` directly in MuJoCo
2. Convert sample motion_135 NPZ to SMPL-X NPZ
3. Visualize in web viewer

### **Phase 2: Implement Physics (2-4 hours)**
1. Create MimicMotionManager to track motion
2. Set up PD controller for joint targets
3. Test locomotion tracking on flat ground

### **Phase 3: Integrate RL (1-2 days)**
1. Define reward function (pose error, contact constraints)
2. Set up PPO training with ProtoMotions
3. Evaluate on motion imitation task

### **Phase 4: Extend (optional)**
1. Add terrain variations
2. Implement body shape optimization (adapt betas)
3. Multi-task learning (locomotion + manipulation)

---

## Sources & References

- **ProtoMotions**: Your local repository (`ref_repo/ProtoMotions/`)
- **UHC (Universal Humanoid Controller)**: https://github.com/harshaguda/UHC
- **SMPLSim**: https://github.com/ZhengyiLuo/SMPLSim
- **SMPL / SMPL-X**: https://smpl.is.tue.mpg.de/
- **MuJoCo Documentation**: https://mujoco.readthedocs.io/

