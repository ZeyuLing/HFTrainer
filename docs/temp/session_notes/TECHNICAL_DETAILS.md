# Technical Details: MuJoCo SMPL Humanoid RL Setup

## 1. Model Specifications

### SMPL Humanoid Structure
```
                    Head (0 DOF per se, attached to Neck)
                    |
              Neck (3 DOF: xyz rotations)
                    |
    L_Collar -- Chest (3 DOF) -- R_Collar
       |         |         |        |
    L_Shoulder  Spine    R_Shoulder |
       |      (3 DOF)      |        |
    L_Elbow     |      R_Elbow    R_Collar
       |      Torso     |         (attached)
    L_Wrist  (3 DOF)   R_Wrist
       |        |        |
    L_Hand   Pelvis   R_Hand
             (FREE   
             JOINT
             6 DOF)
             
             L_Hip       R_Hip
          (3 DOF ea)  (3 DOF ea)
             |           |
          L_Knee     R_Knee
          (3 DOF)    (3 DOF)
             |           |
          L_Ankle    R_Ankle
          (3 DOF)    (3 DOF)
             |           |
          L_Toe      R_Toe
          (3 DOF)    (3 DOF)
```

### DOF Count
- **Free Joint (Pelvis)**: 7 DOF
  - Position: 3 (x, y, z)
  - Orientation: 4 (quaternion)
- **Hinge Joints**: 24 bodies × 3 (all x/y/z rotations)
- **Total MuJoCo QPos**: 76 (3 pos + 4 quat + 69 rotation)
- **Total MuJoCo QVel**: 75 (6 linear+angular + 69 rotation vels)

### Actuators
- **Actuated**: 75 motors (all except root position — free joint is unactuated)
- **Control Range**: [-1, 1] (normalized torques)
- **Per-Joint Configuration**:
  ```
  Hip/Knee/Ankle:        stiffness=800, damping=80
  Toe joints:            stiffness=500, damping=50
  Torso/Spine/Chest:     stiffness=1000, damping=100
  ```

---

## 2. Coordinate Systems

### MuJoCo Convention (Z-up, XYZ axes)
- X: forward
- Y: right
- Z: up
- Quaternion: [x, y, z, w] (wxyz in some MuJoCo docs, but Python API uses xyzw)

### HyMotion/SMPL Convention (Y-up)
- X: forward
- Y: up
- Z: right

### Transformation Required
```python
# Y-up → Z-up (rotate around X-axis by -90°)
# Quaternion transform: quat_zup = quat_yup @ [cos(π/4), -sin(π/4), 0, 0]

def yup_to_zup_quaternion(quat_yup):
    """Convert Y-up quaternion to Z-up (xyzw format)."""
    # Rotation matrix around X-axis by -90°
    rot_x_neg90 = np.array([
        [1,  0,   0],
        [0,  0,   1],
        [0, -1,   0]
    ])
    # Convert to rotation matrix, apply, convert back
    from scipy.spatial.transform import Rotation as sRot
    R_yup = sRot.from_quat(quat_yup)
    R_transformed = sRot.from_matrix(rot_x_neg90 @ R_yup.as_matrix())
    return R_transformed.as_quat()

def yup_to_zup_position(pos_yup):
    """Convert Y-up position to Z-up."""
    x, y, z = pos_yup
    return np.array([x, -z, y])  # [x, z, -y] after rotation
```

---

## 3. Observation Space Design

### Option A: Raw State (151 dims) ✅ Recommended for Phase 1
```python
obs = np.concatenate([
    data.qpos,      # 76 dims: [3 pos, 4 quat, 69 joints]
    data.qvel,      # 75 dims: [3 lin_vel, 3 ang_vel, 69 joint_vels]
])  # Total: 151 dims
```

### Option B: Computed Features (higher semantic content)
```python
obs = np.concatenate([
    data.qpos[3:],          # Exclude root position, keep quat + joints (72 dims)
    data.qvel,              # Keep all velocities (75 dims)
    body_positions,         # FK output: 24 bodies × 3 coords (72 dims)
    body_velocities,        # Spatial velocities (72 dims)
    foot_contact,           # 2 dims: left/right foot contact binary
])  # Total: ~293 dims
```

### Option C: Mixed (recommended long-term)
Start with Option A, add Option B features as needed based on learning curves.

---

## 4. Action Space Design

### Normalized Torques (75 dims)
```python
action_space = gymnasium.spaces.Box(
    low=-1.0, high=1.0, shape=(75,), dtype=np.float32
)

# In MuJoCo control vector:
data.ctrl = action * ctrl_limits  # Scale normalized actions to physics
```

### Mapping
```python
# MuJoCo actuators in order (from XML):
actuator_names = [
    # Free joint actuators (skipped, index 0-5)
    # Hinge actuators (index 0-74 in actuator vector)
    "L_Hip_x", "L_Hip_y", "L_Hip_z",
    "L_Knee_x", "L_Knee_y", "L_Knee_z",
    ...
]

# action vector (75 dims) maps directly to actuators[1:76]
```

---

## 5. Reward Function Components

### Primary Tracking Loss
```python
def compute_tracking_reward(sim_qpos, ref_qpos, target_joints=None):
    """
    Penalize deviation from reference pose.
    
    Args:
        sim_qpos: Current simulated pose (76 dims)
        ref_qpos: Reference/target pose (76 dims)
        target_joints: Indices of joints to track (e.g., exclude root)
    """
    if target_joints is None:
        target_joints = np.arange(3, 76)  # All except root position
    
    # Rotation distance using geodesic
    sim_rot = sRot.from_quat(sim_qpos[3:7])      # Pelvis quaternion
    ref_rot = sRot.from_quat(ref_qpos[3:7])
    rot_error = (ref_rot.inv() * sim_rot).magnitude()  # Angle magnitude
    
    # Joint angle error (assume linear rotations after root)
    joint_error = np.linalg.norm(sim_qpos[7:] - ref_qpos[7:])
    
    # Combined reward (higher is better)
    reward = -rot_error - 0.1 * joint_error
    return reward
```

### Energy Regularization
```python
def compute_energy_penalty(data):
    """Penalize excessive joint torques."""
    # Torque magnitude
    torque_norm = np.linalg.norm(data.ctrl)
    
    # Power = torque × velocity
    power = np.sum(np.abs(data.ctrl * data.qvel[-75:]))
    
    penalty = -0.001 * torque_norm - 0.0001 * power
    return penalty
```

### Stability Bonus
```python
def compute_stability_bonus(data, env_config):
    """Encourage ground contact and balance."""
    # Contact forces
    left_foot_contact = np.any(data.contact.geom[:, 0] == 'L_Toe')
    right_foot_contact = np.any(data.contact.geom[:, 1] == 'R_Toe')
    
    # Height penalty (if pelvis falls below threshold)
    pelvis_height = data.xpos[1, 2]  # Body 1 (Pelvis), Z coord
    height_penalty = 0 if pelvis_height > env_config['min_height'] else -1.0
    
    # Bonus for bilateral contact (walking)
    contact_bonus = 0.1 if (left_foot_contact and right_foot_contact) else 0
    
    return height_penalty + contact_bonus
```

### Composite Reward
```python
def compute_reward(sim_qpos, ref_qpos, data, env_config):
    """Combine all reward components."""
    w_track = 1.0
    w_energy = 0.01
    w_stability = 0.1
    
    reward = (
        w_track * compute_tracking_reward(sim_qpos, ref_qpos) +
        w_energy * compute_energy_penalty(data) +
        w_stability * compute_stability_bonus(data, env_config)
    )
    return reward
```

---

## 6. Termination Conditions

```python
def compute_done(data, env_config, timestep):
    """Check if episode should terminate."""
    
    # Timeout
    if timestep >= env_config['max_episode_length']:
        return True, "timeout"
    
    # Pelvis height (falling)
    pelvis_height = data.xpos[1, 2]  # Body 1, Z coord
    if pelvis_height < env_config['min_height']:
        return True, "fall"
    
    # NaN check (numerical instability)
    if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
        return True, "nan"
    
    # Large angle violation
    for i, (joint_idx, limit) in enumerate(env_config['angle_limits']):
        if np.abs(data.qpos[joint_idx]) > limit:
            return True, f"joint_limit_{i}"
    
    return False, None
```

---

## 7. Integration with Stable-Baselines3

### Gymnasium Environment Template
```python
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

class SMPLHumanoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, mjcf_path, config=None):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.config = config or self._default_config()
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(151,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(75,), dtype=np.float32
        )
        
        self.timestep = 0
        self.ref_qpos = None
        
    def _default_config(self):
        return {
            'max_episode_length': 1000,
            'dt': 0.00555,  # MuJoCo default timestep
            'min_height': 0.5,
            'w_track': 1.0,
            'w_energy': 0.01,
            'w_stability': 0.1,
        }
    
    def step(self, action):
        # Apply action
        self.data.ctrl[:] = action
        
        # Step physics
        mujoco.mj_step(self.model, self.data, self.config['dt'])
        
        # Compute obs, reward, done
        obs = self._get_obs()
        reward = self._compute_reward()
        done, info = self._check_done()
        
        self.timestep += 1
        return obs, reward, done, False, info
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Random initialization
        self.data.qpos[:] = 0
        self.data.qpos[2] = 1.0  # Pelvis height
        
        # Small random perturbation
        self.data.qpos[7:] += np.random.uniform(-0.1, 0.1, 69)
        
        mujoco.mj_forward(self.model, self.data)
        
        self.timestep = 0
        self.ref_qpos = self.data.qpos.copy()  # Use initial pose as reference
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
    
    def _compute_reward(self):
        # Simplified version; extend with components above
        tracking_reward = -np.linalg.norm(self.data.qpos[7:] - self.ref_qpos[7:])
        energy_penalty = -0.001 * np.linalg.norm(self.data.ctrl)
        return tracking_reward + energy_penalty
    
    def _check_done(self):
        pelvis_height = self.data.xpos[1, 2]
        done = (
            self.timestep >= self.config['max_episode_length'] or
            pelvis_height < self.config['min_height']
        )
        return done, {}
    
    def render(self):
        pass  # Optional: add mujoco.viewer


# Training
from stable_baselines3 import PPO

env = SMPLHumanoidEnv("path/to/smpl_humanoid.xml")
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=1_000_000)
model.save("smpl_humanoid_ppo")
```

---

## 8. File Paths & References

### Key Source Files
```
ref_repo/OmniH2O/phc/phc/data/assets/mjcf/
  └── smpl_humanoid.xml            ← Main model

scripts/embodied/
  ├── run_smpl_physics_sim.py       ← Reference implementation
  ├── run_smpl_rl_tracker.py        ← RL inference
  └── debug_sim_stability.py        ← Physics validation

ref_repo/ProtoMotions/
  ├── protomotions/simulator/mujoco_simulator.py  ← Sim abstraction
  ├── protomotions/agents/ppo.py                  ← PPO implementation
  └── protomotions/envs/mdp_component.py          ← Reward components
```

### Output Directories
```
output/embodied_t2m_v4/
  ├── data/npz/                    ← Motion clips (135-dim HyMotion format)
  ├── data/meta/                   ← Metadata
  └── data/sim_stats/              ← Physics statistics
```

---

## 9. Debugging Tips

### Physics Validation
```python
import mujoco

# Load model
model = mujoco.MjModel.from_xml_path("smpl_humanoid.xml")
data = mujoco.MjData(model)

# Check DOF counts
print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")  # Should be 76, 75, 75

# Check body names
for i, name in enumerate(model.body_names):
    if i < 25:  # SMPL bodies only
        print(f"Body {i}: {name}")

# Verify contact
print(f"Geom count: {model.ngeom}")
for i in range(model.ngeom):
    print(f"Geom {i}: {model.geom_names[i]}")
```

### Forward Kinematics
```python
# Compute forward kinematics
mujoco.mj_forward(model, data)

# Get body positions
for i, name in enumerate(model.body_names[:25]):
    pos = data.xpos[i]
    print(f"{name}: {pos}")
```

### Physics Stepping
```python
# Manual stepping with diagnostics
for step in range(100):
    data.ctrl[:] = np.random.uniform(-1, 1, 75)  # Random actions
    mujoco.mj_step(model, data)
    
    if step % 10 == 0:
        print(f"Step {step}: height={data.xpos[1, 2]:.3f}, "
              f"energy={np.sum(np.abs(data.ctrl * data.qvel[-75:])):.3f}")
```

---

## 10. Performance Benchmarks (Reference)

### Training Speed
- **MuJoCo (CPU backend)**: ~500 steps/sec (single environment)
- **Gymnasium PPO**: ~1000 samples/sec (vectorized, if multi-env)
- **Expected training time**: 10M steps = 2.7 hours (CPU)

### Convergence Criteria
- **Tracking reward**: -0.5 to -0.1 (lower is better)
- **Episode length**: 500-1000 steps (not terminating early)
- **Success**: Policy maintains standing + tracks reference joints

