# ProtoMotions RL Training with MuJoCo Backend: Complete Analysis

## Overview
This document provides a thorough analysis of ProtoMotions' RL training pipeline for motion mimicry tasks using MuJoCo as a CPU-only single-environment simulator. It focuses on understanding the motion data format, how T2M model outputs can be integrated, and the complete data flow.

---

## 1. Experiment Configuration (MLP Motion Tracker)

**File Path:** `/ref_repo/ProtoMotions/examples/experiments/mimic/mlp.py`

### 1.1 Configuration Builder Functions

The experiment config defines several modular builder functions:

```python
def motion_lib_config(args: argparse.Namespace):
    """Build motion library configuration."""
    return MotionLibConfig(motion_file=args.motion_file)

def env_config(robot_cfg: RobotConfig, args: argparse.Namespace) -> EnvConfig:
    """Build environment configuration (training defaults)."""
    # Returns EnvConfig with:
    # - control_components: MimicControlConfig
    # - observation_components: max_coords_obs, previous_actions, mimic_target_poses
    # - termination_components: tracking_error threshold=0.5
    # - reward_components: action_smoothness, mimic_tracking_rewards, power, contact_match
    # - motion_manager: MimicMotionManagerConfig

def agent_config(robot_config, env_config, args) -> PPOAgentConfig:
    """Build PPO agent configuration."""
    # Returns PPOAgentConfig with:
    # - Actor network: 6 layers of 1024 units (ReLU)
    # - Critic network: 4 layers of 1024 units (ReLU)
    # - Both use normalize_obs=True, norm_clamp_value=5
    # - Learning rates: actor=2e-5, critic=1e-4
```

### 1.2 Key Configuration Parameters

| Component | Setting | Purpose |
|-----------|---------|---------|
| **Episode Length** | 1000 steps | Max episode duration |
| **Tracking Error Threshold** | 0.5 m | Early termination if exceeded |
| **Motion Manager** | `init_start_prob=0.2` | 20% chance of random start in motion |
| **Reward Weights** | `gt=0.5, gr=0.3, gv=0.1, gav=0.2, rh=0.2` | Pose+rotation+velocity+angular_vel+hand tracking |
| **Network Normalization** | obs normalize + clamp to 5σ | Handles wide range of input scales |

### 1.3 Observation Components

The environment observes:
1. **max_coords_obs**: Normalized robot state (joint positions/velocities)
2. **mimic_target_poses**: Target motion state with velocities
3. **previous_actions**: Action history (1 frame)

→ These are concatenated as input to the actor/critic networks.

---

## 2. Motion Data Format and Loading

**File Path:** `/ref_repo/ProtoMotions/protomotions/components/motion_lib.py`

### 2.1 MotionLib Class Overview

```python
class MotionLib:
    """Motion library for managing and sampling reference motion data."""
    
    # Stored tensor fields (all shape: [total_frames, ...])
    gts: torch.Tensor          # Global rigid body positions [total_frames, num_bodies, 3]
    grs: torch.Tensor          # Global rigid body rotations [total_frames, num_bodies, 4] (xyzw)
    gvs: torch.Tensor          # Global rigid body velocities [total_frames, num_bodies, 3]
    gavs: torch.Tensor         # Global rigid body angular velocities [total_frames, num_bodies, 3]
    dvs: torch.Tensor          # DOF velocities [total_frames, num_dofs]
    dps: torch.Tensor          # DOF positions (joint angles) [total_frames, num_dofs]
    contacts: torch.Tensor     # Rigid body contact info [total_frames, num_bodies]
    lrs: torch.Tensor          # Local rigid body rotations [total_frames, num_bodies, 4] (optional)
    
    # Metadata fields (per motion)
    motion_lengths: torch.Tensor         # Time duration of each motion [num_motions]
    motion_dt: torch.Tensor             # Delta time per frame [num_motions]
    motion_num_frames: torch.Tensor     # Number of frames per motion [num_motions]
    motion_weights: torch.Tensor        # Sampling weights per motion [num_motions]
    length_starts: torch.Tensor         # Start index for each motion [num_motions]
```

### 2.2 Supported Motion File Formats

ProtoMotions supports three loading mechanisms:

#### A. **Individual `.motion` or `.npz` files**
```python
# Single motion file
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="path/to/motion.motion"),
    device="cpu"
)
```

#### B. **YAML configuration** (weighted motion list)
```yaml
# motions.yaml
motions:
  - file: walk.motion
    weight: 1.0
  - file: run.motion
    weight: 0.5
  - file: jump.motion
    weight: 0.3
```

```python
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="path/to/motions.yaml"),
    device="cpu"
)
```

#### C. **Packaged `.pt` file** (fastest loading)
```python
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="path/to/motions.pt"),
    device="cpu"
)
```

#### D. **Directory of motion files**
```python
# All .motion files in directory are loaded with equal weights
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="path/to/motion_dir/"),
    device="cpu"
)
```

### 2.3 Motion Data Loading Pipeline

```
Input: motion_file (string path)
  ↓
[Check file extension]
  ├─ if .pt  → load_from_file() [FAST]
  └─ if .yaml/.motion/dir → _load_motions() [SLOWER]
  ↓
[For YAML files]
  └─ Parse YAML, collect motion files and weights
  ↓
[For each individual motion file]
  └─ torch.load(motion_file, weights_only=False)
     → Returns dict with RobotState fields
     → Create RobotState.from_dict()
  ↓
[Concatenate all motions]
  └─ torch.cat([all_motions_field], dim=0)
  ↓
[Create metadata tensors]
  ├─ motion_num_frames: [num_motions]
  ├─ length_starts: cumulative frame offsets
  ├─ motion_weights: sampling probabilities
  ├─ motion_lengths: time duration (in seconds)
  └─ motion_dt: 1.0 / fps per motion
  ↓
Output: Loaded MotionLib
```

### 2.4 Motion File Format Details

**When saving a single motion:** The file should contain a dict that can be converted to RobotState:

```python
# What's saved in a .motion file:
motion_dict = {
    "dof_pos": torch.Tensor([num_frames, num_dofs]),
    "dof_vel": torch.Tensor([num_frames, num_dofs]),
    "rigid_body_pos": torch.Tensor([num_frames, num_bodies, 3]),
    "rigid_body_rot": torch.Tensor([num_frames, num_bodies, 4]),  # xyzw format
    "rigid_body_vel": torch.Tensor([num_frames, num_bodies, 3]),
    "rigid_body_ang_vel": torch.Tensor([num_frames, num_bodies, 3]),
    "rigid_body_contacts": torch.Tensor([num_frames, num_bodies]),  # bool or float
    "fps": float,  # Frames per second
    # Optional:
    "local_rigid_body_rot": torch.Tensor([num_frames, num_bodies, 4]),
}

torch.save(motion_dict, "output.motion")
```

### 2.5 Field Mapping (MotionLib → RobotState)

```python
_motion_field_mapping = {
    "gts": "rigid_body_pos",      # Global positions
    "grs": "rigid_body_rot",      # Global rotations (xyzw)
    "gavs": "rigid_body_ang_vel", # Angular velocities
    "gvs": "rigid_body_vel",      # Linear velocities
    "dvs": "dof_vel",             # DOF velocities
    "dps": "dof_pos",             # DOF positions
}
```

### 2.6 Sampling and Interpolation

**Sampling a motion at time t:**

```python
# 1. Sample motion IDs (batch_size,) using motion_weights
motion_ids = motion_lib.sample_motions(num_samples=num_envs)

# 2. Sample motion times (batch_size,) uniformly within [0, motion_length]
motion_times = torch.rand(num_envs) * motion_lib.motion_lengths[motion_ids]

# 3. Get interpolated motion state
motion_state = motion_lib.get_motion_state(motion_ids, motion_times)
# Returns: RobotState with all fields interpolated between frames
```

**Interpolation method:**

```python
# For position/velocity fields: Linear interpolation
# For rotation fields: Spherical linear interpolation (slerp)
# For DOF positions: Linear (or slerp if local_rigid_body_rot provided)
# For contacts: OR boolean (averaged for float contacts)
```

---

## 3. PPO Agent Implementation

**File Path:** `/ref_repo/ProtoMotions/protomotions/agents/ppo/agent.py` (first 100 lines)

### 3.1 PPO Agent Class

```python
class PPO(BaseAgent):
    """Proximal Policy Optimization agent for motion tracking."""
    
    config: PPOAgentConfig
    
    def __init__(
        self,
        fabric: Fabric,           # Lightning Fabric for distributed training
        env: BaseEnv,             # RL environment
        config: PPOAgentConfig,   # PPO configuration
        root_dir: Optional[Path] = None,
    ):
        super().__init__(fabric, env, config, root_dir)
        self.tau: float = self.config.tau  # GAE lambda parameter
        # ... initialization continues
```

### 3.2 Key Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `tau` | float | GAE lambda parameter for advantage estimation |
| `e_clip` | float | PPO clipping parameter |
| `actor` | Network | Policy network (outputs actions) |
| `critic` | Network | Value network (estimates state value) |

### 3.3 Training Flow (from docstring)

```
1. collect experience through environment interaction
2. compute advantages using GAE (Generalized Advantage Estimation)
3. perform multiple epochs of minibatch updates
   - clipped surrogate objective for stable policy updates
   - separate optimizers for actor and critic
```

---

## 4. Environment Base Class

**File Path:** `/ref_repo/ProtoMotions/protomotions/envs/base_env/env.py` (first 80 lines)

### 4.1 BaseEnv Class Overview

```python
class BaseEnv:
    """Base environment for RL tasks."""
    
    config: EnvConfig                    # Environment configuration
    robot_config: RobotConfig            # Robot structure/properties
    device: torch.device                 # PyTorch device
    terrain: Terrain                     # Terrain for collision
    scene_lib: SceneLib                  # Object scene library
    motion_lib: MotionLib                # Reference motions
    simulator: Simulator                 # Physics simulator (MuJoCo, etc.)
    num_envs: int                        # Number of parallel environments
    max_episode_length: int              # Max steps per episode (mutable)
    dt: float                            # Simulation timestep
```

### 4.2 Key Mutable State Buffers

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `rew_buf` | `[num_envs]` | Accumulated rewards (resets each episode) |
| `reset_buf` | `[num_envs]` | Boolean flags for environments needing reset |
| `progress_buf` | `[num_envs]` | Episode step counter |
| `terminate_buf` | `[num_envs]` | Early termination flags |
| `extras` | dict | Per-step logging dictionary |

### 4.3 Motion Manager Integration

```python
motion_manager: MotionManager           # Samples motions for episode
motion_manager_disable_resample: bool   # Flag (controlled by evaluator)
```

The motion_manager handles:
- Sampling motions for the current episode
- Providing target pose observations
- Respecting `init_start_prob` for randomized starts

### 4.4 Environment API Contract

**Typical per-step flow:**

```
step(action) → (observations, rewards, dones, infos)

reset(env_ids) → observations

get_motion_state() → RobotState (target reference)
```

---

## 5. Robot State Data Class

**File Path:** `/ref_repo/ProtoMotions/protomotions/simulator/base_simulator/simulator_state.py`

### 5.1 RobotState Dataclass

```python
@dataclass
class RobotState(BaseBatchedState):
    """Represents robot state at one or more timesteps."""
    
    # All fields are [batch_size, ...] where batch_size is either:
    # - num_frames (when used as motion data)
    # - num_envs (when used as batched sim state during GPU training)
    
    dof_pos: Optional[torch.Tensor]        # [batch, num_dof]
    dof_vel: Optional[torch.Tensor]        # [batch, num_dof]
    dof_forces: Optional[torch.Tensor]     # [batch, num_dof]
    
    rigid_body_pos: Optional[torch.Tensor]        # [batch, num_bodies, 3]
    rigid_body_rot: Optional[torch.Tensor]        # [batch, num_bodies, 4]
    rigid_body_vel: Optional[torch.Tensor]        # [batch, num_bodies, 3]
    rigid_body_ang_vel: Optional[torch.Tensor]    # [batch, num_bodies, 3]
    
    rigid_body_contacts: Optional[torch.Tensor]        # [batch, num_bodies] (bool or float)
    rigid_body_contact_forces: Optional[torch.Tensor]  # [batch, num_bodies, 3]
    
    # Cache field for interpolation
    local_rigid_body_rot: Optional[torch.Tensor]       # [batch, num_bodies, 4]
    
    # Metadata
    state_conversion: Optional[StateConversion]  # COMMON or SIMULATOR
    fps: Optional[float]                         # Frames per second
```

### 5.2 Key Methods

```python
# Conversion between formats
from_dict(data: Dict[str, Tensor], state_conversion) → RobotState
to_dict() → Dict[str, Tensor]

# Indexing
state[env_ids] → RobotState (subset)
state[0] → RobotState (single env)
state["field_name"] → Tensor (field access)

# Metadata properties (when fps is set)
@property motion_num_frames() → int
@property motion_dt() → float  (= 1.0 / fps)
@property motion_length() → float  (= (motion_num_frames - 1) * motion_dt)
```

### 5.3 Quaternion Convention

**IMPORTANT:** ProtoMotions uses **xyzw quaternion ordering** in the "COMMON" state format.

```python
# Quaternion format: [x, y, z, w]
# This is consistent across MotionLib and RobotState when state_conversion=COMMON

# When converting to/from different simulators (IsaacGym, Genesis, etc.),
# use: state.convert_to_sim() and convert_to_common()
```

---

## 6. Complete Data Flow: T2M → RL Training

### 6.1 Step-by-Step Integration

```
┌─────────────────────────────────────────────────────────────┐
│ 1. T2M Generation Model                                     │
│    (Text → Motion Sequence)                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓ Generate synthetic motion sequence
                   │
┌──────────────────────────────────────────────────────────────┐
│ 2. Format Output as RobotState Dict                         │
│                                                              │
│ output_dict = {                                             │
│     "dof_pos": tensor([num_frames, num_dofs]),            │
│     "dof_vel": tensor([num_frames, num_dofs]),            │
│     "rigid_body_pos": tensor([num_frames, num_bodies, 3]),│
│     "rigid_body_rot": tensor([num_frames, num_bodies, 4]),│
│     "rigid_body_vel": tensor([num_frames, num_bodies, 3]),│
│     "rigid_body_ang_vel": tensor([num_frames, num_bodies]),│
│     "fps": 30,                                             │
│ }                                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓ torch.save()
                   │
┌──────────────────────────────────────────────────────────────┐
│ 3. Save as .motion File                                     │
│    torch.save(output_dict, "generated_motion.motion")      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓ Create YAML manifest (optional)
                   │
┌──────────────────────────────────────────────────────────────┐
│ 4. Create YAML Config (for multiple motions)                │
│                                                              │
│ motions.yaml:                                              │
│   motions:                                                │
│     - file: generated_motion.motion                        │
│       weight: 1.0                                          │
│     - file: existing_walk.motion                           │
│       weight: 0.5                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓ Create MotionLibConfig
                   │
┌──────────────────────────────────────────────────────────────┐
│ 5. Load into MotionLib                                      │
│                                                              │
│ motion_lib = MotionLib(                                     │
│     config=MotionLibConfig(motion_file="motions.yaml"),   │
│     device="cpu"                                           │
│ )                                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ↓ Sample during training
                   │
┌──────────────────────────────────────────────────────────────┐
│ 6. RL Training Loop                                         │
│                                                              │
│ motion_ids = motion_lib.sample_motions(num_envs=1)         │
│ motion_times = torch.rand(1) * motion_lengths[motion_ids]  │
│ target_state = motion_lib.get_motion_state(                │
│     motion_ids, motion_times                               │
│ )                                                          │
│                                                             │
│ # Use target_state in observations for PPO training        │
│ obs = env.reset()                                          │
│ for step in range(max_steps):                              │
│     action = agent.policy(obs)                             │
│     obs, reward, done, info = env.step(action)             │
│     # reward computed as: ||current_state - target_state|| │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Minimum Required Fields for T2M Output

For a T2M model to output motion compatible with ProtoMotions RL:

| Field | Shape | Type | Notes |
|-------|-------|------|-------|
| **dof_pos** | `[num_frames, num_dofs]` | float32 | Joint angles (in exp_map or rad) |
| **dof_vel** | `[num_frames, num_dofs]` | float32 | Joint velocities |
| **rigid_body_pos** | `[num_frames, num_bodies, 3]` | float32 | Global body positions |
| **rigid_body_rot** | `[num_frames, num_bodies, 4]` | float32 | Quaternions in **xyzw** format |
| **rigid_body_vel** | `[num_frames, num_bodies, 3]` | float32 | Linear velocities |
| **rigid_body_ang_vel** | `[num_frames, num_bodies, 3]` | float32 | Angular velocities |
| **fps** | scalar | int/float | Frames per second (e.g., 30 or 60) |

**Optional but recommended:**
- **rigid_body_contacts**: `[num_frames, num_bodies]` (bool) - For contact-aware rewards
- **local_rigid_body_rot**: `[num_frames, num_bodies, 4]` - For exp_map DOF conversion

### 6.3 Critical Numerical Constraints

```
✓ Quaternions MUST be in xyzw format (not wxyz)
✓ Quaternions should be normalized (||q|| = 1)
✓ Angles should be in radians
✓ Positions in meters
✓ Velocities in m/s
✓ Angular velocities in rad/s
```

---

## 7. Packaged Motion Files (.pt)

### 7.1 Creating a Packaged Motion Library

After loading individual motions, you can save as a single `.pt` file for fast loading:

```python
motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="motions.yaml"),
    device="cpu"
)

# Save all loaded motions into one .pt file
motion_lib.save_to_file("packaged_motions.pt")
```

### 7.2 What's Saved in a .pt File

```python
saved_dict = {
    "gts": torch.Tensor([total_frames, num_bodies, 3]),
    "grs": torch.Tensor([total_frames, num_bodies, 4]),
    "gavs": torch.Tensor([total_frames, num_bodies, 3]),
    "gvs": torch.Tensor([total_frames, num_bodies, 3]),
    "dvs": torch.Tensor([total_frames, num_dofs]),
    "dps": torch.Tensor([total_frames, num_dofs]),
    "motion_num_frames": torch.Tensor([num_motions], dtype=long),
    "length_starts": torch.Tensor([num_motions], dtype=long),
    "motion_weights": torch.Tensor([num_motions]),
    "motion_lengths": torch.Tensor([num_motions]),
    "motion_dt": torch.Tensor([num_motions]),
    "contacts": torch.Tensor([total_frames, num_bodies]) or None,
    "lrs": torch.Tensor([total_frames, num_bodies, 4]) or None,
    "motion_files": tuple of str,
}

torch.save(saved_dict, "packaged_motions.pt")
```

### 7.3 Multi-GPU Support

For distributed training, ProtoMotions supports rank-specific motion files:

```python
# In filenames, use "slurmrank" as a wildcard
motion_file = "data/motions/chunk_slurmrank.pt"

# Discovered files: chunk_00.pt, chunk_01.pt, chunk_02.pt, ...
# Each rank loads chunk_{rank % num_chunks}.pt
```

---

## 8. Reward Components in RL Training

From `mlp.py` env_config:

```python
reward_components = {
    # Tracking rewards (ground truth matching)
    "gt": gt_coef=-25.0,      # Root position tracking
    "gr": gr_coef=-5.0,       # Root rotation tracking  
    "gv": gv_coef=-0.5,       # Root velocity tracking
    "gav": gav_coef=-0.1,     # Root angular velocity
    "rh": rh_coef=-100.0,     # Whole body rigid body tracking
    
    # Regularization
    "action_smoothness": weight=-0.02,  # Penalize jerky actions
    "pow_rew": weight=-1e-5,            # Penalize high torques
    "contact_match": weight=-0.1,       # Match reference contacts
}
```

These rewards are computed as:

```
R = Σ weight * error_metric
  = gt_w * (||root_pos_pred - root_pos_ref||²)
  + gr_w * (||root_rot_pred - root_rot_ref||²)
  + ... (velocity terms)
  + regularization terms
```

---

## 9. Key File Paths Reference

| Component | File Path |
|-----------|-----------|
| **Experiment Config** | `examples/experiments/mimic/mlp.py` |
| **PPO Agent** | `protomotions/agents/ppo/agent.py` |
| **Motion Library** | `protomotions/components/motion_lib.py` |
| **Environment Base** | `protomotions/envs/base_env/env.py` |
| **Robot State** | `protomotions/simulator/base_simulator/simulator_state.py` |
| **Recording/Serialization** | `protomotions/simulator/base_simulator/record.py` |

---

## 10. Quick Reference: Motion Format Example

### 10.1 Minimal T2M Output Example

```python
import torch

# Generate a simple 3-frame motion for a humanoid
num_frames = 3
num_dofs = 67  # Humanoid DOF count
num_bodies = 24  # Humanoid body count

motion_dict = {
    # Joint angles (in exp_map for exp_map format)
    "dof_pos": torch.randn(num_frames, num_dofs),
    "dof_vel": torch.randn(num_frames, num_dofs),
    
    # Rigid body states (computed from FK or provided)
    "rigid_body_pos": torch.randn(num_frames, num_bodies, 3),
    "rigid_body_rot": torch.nn.functional.normalize(
        torch.randn(num_frames, num_bodies, 4), 
        dim=-1
    ),  # Normalized quaternions in xyzw
    "rigid_body_vel": torch.randn(num_frames, num_bodies, 3),
    "rigid_body_ang_vel": torch.randn(num_frames, num_bodies, 3),
    
    # Metadata
    "fps": 30,
}

# Save as motion file
torch.save(motion_dict, "my_generated_motion.motion")

# Load into MotionLib
from protomotions.components.motion_lib import MotionLib, MotionLibConfig

motion_lib = MotionLib(
    config=MotionLibConfig(motion_file="my_generated_motion.motion"),
    device="cpu"
)

print(f"Loaded {motion_lib.num_motions()} motion(s)")
print(f"Total length: {motion_lib.get_total_length():.2f} seconds")
```

---

## 11. Integration Checklist for T2M Models

- [ ] Output motion as dict with all 7 required fields
- [ ] Ensure quaternions are normalized and in **xyzw** format
- [ ] Set `fps` to match your T2M model's training framerate (e.g., 30 or 60)
- [ ] Ensure DOF counts match target robot (check `robot_config.kinematic_info.num_dofs`)
- [ ] Ensure body counts match target robot (check `robot_config` body definitions)
- [ ] Optionally provide `rigid_body_contacts` for contact-aware reward training
- [ ] Save output as `.motion` file using `torch.save()`
- [ ] Create YAML manifest if combining multiple T2M outputs
- [ ] Validate motion by loading into MotionLib and querying samples
- [ ] Launch RL training with motion file path in config

