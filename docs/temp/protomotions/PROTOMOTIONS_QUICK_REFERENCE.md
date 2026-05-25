# ProtoMotions RL Training: Quick Reference Guide

## 🎯 Motion File Format at a Glance

```python
# What your T2M model should output:
motion_dict = {
    "dof_pos":           torch.Tensor([num_frames, num_dofs]),
    "dof_vel":           torch.Tensor([num_frames, num_dofs]),
    "rigid_body_pos":    torch.Tensor([num_frames, num_bodies, 3]),
    "rigid_body_rot":    torch.Tensor([num_frames, num_bodies, 4]),  # ⚠️ XYZW format!
    "rigid_body_vel":    torch.Tensor([num_frames, num_bodies, 3]),
    "rigid_body_ang_vel": torch.Tensor([num_frames, num_bodies, 3]),
    "fps": 30,  # or 60
}

torch.save(motion_dict, "output.motion")
```

## 📊 Data Flow Architecture

```
T2M Model Output (Text → Motion)
         ↓
    RobotState Dict
         ↓
    Save as .motion file
         ↓
    Create YAML manifest (optional)
         ↓
    MotionLib loads all motions
         ↓
    During RL training:
    - Sample motion ID (weighted by motion_weights)
    - Sample time t uniformly in [0, motion_length]
    - Get interpolated state at time t
    - Use as target for reward computation
```

## 🔑 Key Classes and Files

| Class | File | Purpose |
|-------|------|---------|
| `MotionLib` | `components/motion_lib.py` | Load, sample, interpolate motions |
| `RobotState` | `simulator/base_simulator/simulator_state.py` | State representation (kinematics) |
| `PPO` | `agents/ppo/agent.py` | RL agent for policy learning |
| `BaseEnv` | `envs/base_env/env.py` | RL environment with observations/rewards |
| Config | `examples/experiments/mimic/mlp.py` | Training experiment configuration |

## 🏗️ MotionLib Internal Structure

**After loading:** All motions concatenated into single tensors

```
MotionLib.gts    [total_frames, num_bodies, 3]    # Concatenated rigid body positions
MotionLib.grs    [total_frames, num_bodies, 4]    # Concatenated rigid body rotations
MotionLib.gvs    [total_frames, num_bodies, 3]    # Concatenated rigid body velocities
MotionLib.gavs   [total_frames, num_bodies, 3]    # Concatenated angular velocities
MotionLib.dvs    [total_frames, num_dofs]         # Concatenated DOF velocities
MotionLib.dps    [total_frames, num_dofs]         # Concatenated DOF positions

MotionLib.motion_num_frames    [num_motions]      # Frames per motion
MotionLib.length_starts        [num_motions]      # Start index per motion
MotionLib.motion_weights       [num_motions]      # Sampling weights
MotionLib.motion_lengths       [num_motions]      # Duration per motion (seconds)
MotionLib.motion_dt            [num_motions]      # 1.0 / fps per motion
```

## 📋 Motion File Loading Modes

### Mode 1: Single Motion File
```python
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="walk.motion"),
    device="cpu"
)
```

### Mode 2: YAML Manifest (Multiple Motions)
```yaml
# motions.yaml
motions:
  - file: walk.motion
    weight: 1.0
  - file: run.motion
    weight: 0.5
```
```python
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="motions.yaml"),
    device="cpu"
)
```

### Mode 3: Packaged .pt File (Fastest)
```python
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="motions.pt"),
    device="cpu"
)
```

### Mode 4: Directory of Motion Files
```python
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="./motion_dir/"),
    device="cpu"
)
# Loads all .motion files with equal weights
```

## 🎮 Training Configuration (from mlp.py)

```python
# Episode
max_episode_length = 1000 steps

# Termination
tracking_error_threshold = 0.5 m  # Early termination

# Reward Weights
gt_weight=0.5   # Root position
gr_weight=0.3   # Root rotation
gv_weight=0.1   # Root velocity
gav_weight=0.2  # Angular velocity
rh_weight=0.2   # Whole body

# Network Architecture
actor_layers = 6 × 1024 (ReLU)
critic_layers = 4 × 1024 (ReLU)
actor_lr = 2e-5
critic_lr = 1e-4
```

## 🔄 RobotState Conversion

**Important:** ProtoMotions uses **XYZW** quaternion ordering in COMMON state

```python
from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion

# Load from dict
state = RobotState.from_dict(data_dict, state_conversion=StateConversion.COMMON)

# Access fields
print(state["rigid_body_pos"])        # [batch, num_bodies, 3]
print(state["rigid_body_rot"])        # [batch, num_bodies, 4] in xyzw

# Convert to dict
dict_data = state.to_dict()

# Batch indexing
single_state = state[0]               # First frame/env
subset_state = state[torch.tensor([0, 2, 5])]  # Select frames
```

## 💾 Saving Packaged Motion Library

```python
# After loading individual motions via YAML or directory
motion_lib.save_to_file("packaged_motions.pt")

# Next time: directly load the .pt file (much faster)
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="packaged_motions.pt"),
    device="cpu"
)
```

## ⚙️ Sampling During Training

```python
# Sample motion at training time
motion_ids = motion_lib.sample_motions(num_samples=1)  # [1]
motion_times = torch.rand(1) * motion_lib.motion_lengths[motion_ids]  # [1]

# Get interpolated state (uses SLERP for rotations)
target_state = motion_lib.get_motion_state(motion_ids, motion_times)
# Returns: RobotState with all fields interpolated

# Access target data
target_pos = target_state.rigid_body_pos           # [1, num_bodies, 3]
target_rot = target_state.rigid_body_rot           # [1, num_bodies, 4]
target_dof = target_state.dof_pos                  # [1, num_dofs]
```

## ⚠️ Critical Constraints for T2M Output

| Constraint | Value | Why |
|-----------|-------|-----|
| Quaternion format | **XYZW** | ProtoMotions standard |
| Quaternion norm | 1.0 (normalized) | Interpolation requires unit quaternions |
| Units (position) | meters | Physics simulation expects SI units |
| Units (velocity) | m/s | Time in seconds |
| Units (angles) | radians | Standard in robotics |
| Angular velocity | rad/s | Rotational motion in radians/second |
| DOF count | match robot config | Must match `robot_config.kinematic_info.num_dofs` |
| Body count | match robot config | Must match number of rigid bodies |

## 🚀 Integration Steps

```
1. Generate motion with T2M model → tensor outputs
2. Convert to RobotState dict
3. torch.save() as .motion file
4. Create motions.yaml (if multiple motions)
5. MotionLib(MotionLibConfig(motion_file="motions.yaml"))
6. Pass motion_lib to environment
7. Launch training with mlp.py config
```

## 📍 Environment Observation Components

```python
observation = {
    "max_coords_obs":      # Robot state (normalized)
    "mimic_target_poses":  # Target from motion library (with velocities)
    "previous_actions":    # Action history (1 frame)
}
# Concatenated and fed to actor/critic networks
```

## 🎯 Reward Components

```
total_reward = 
    -gt_coef * ||root_pos_pred - root_pos_ref||²
    -gr_coef * ||root_rot_pred - root_rot_ref||²
    -gv_coef * ||root_vel_pred - root_vel_ref||²
    -gav_coef * ||angular_vel_pred - angular_vel_ref||²
    -rh_coef * ||body_states_pred - body_states_ref||²
    -action_smoothness_weight * smoothness_penalty
    -power_weight * power_penalty
    -contact_match_weight * contact_mismatch
```

## 🔧 Minimal Working Example

```python
import torch
from protomotions.components.motion_lib import MotionLib, MotionLibConfig

# 1. Create minimal motion dict
num_frames, num_dofs, num_bodies = 100, 67, 24

motion_dict = {
    "dof_pos": torch.randn(num_frames, num_dofs),
    "dof_vel": torch.randn(num_frames, num_dofs),
    "rigid_body_pos": torch.randn(num_frames, num_bodies, 3),
    "rigid_body_rot": torch.nn.functional.normalize(
        torch.randn(num_frames, num_bodies, 4), dim=-1
    ),
    "rigid_body_vel": torch.randn(num_frames, num_bodies, 3),
    "rigid_body_ang_vel": torch.randn(num_frames, num_bodies, 3),
    "fps": 30,
}

# 2. Save
torch.save(motion_dict, "test.motion")

# 3. Load
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="test.motion"),
    device="cpu"
)

# 4. Sample
motion_ids = motion_lib.sample_motions(1)
motion_times = torch.tensor([0.5])
state = motion_lib.get_motion_state(motion_ids, motion_times)

print(f"Loaded {motion_lib.num_motions()} motion(s)")
print(f"State shape: {state.rigid_body_pos.shape}")
```

---

**For full details, see: `PROTOMOTIONS_RL_TRAINING_ANALYSIS.md`**
