# OmniH2O / PHC Robot Motion Imitation Infrastructure - Comprehensive Analysis

**Repository**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/OmniH2O/`

**Publications**:
- H2O: Learning Human-to-Humanoid Real-Time Whole-Body Teleoperation (IROS 2024)
- OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning (CoRL 2024)

---

## 1. MOTION IMITATION POLICY & TRAINING

### 1.1 RL Algorithm: PPO (Proximal Policy Optimization)

**Training Code Location**:
- `legged_gym/legged_gym/scripts/train_hydra.py` - Main training entry point
- `rsl_rl/rsl_rl/modules/actor_critic.py` - Policy network (MLP-based actor-critic)
- `rsl_rl/rsl_rl/runners/on_policy_runner.py` - PPO runner/optimizer

**Policy Architecture**:
```python
# From actor_critic.py
ActorCritic(nn.Module):
  - Actor: MLP with configurable hidden dims [256, 256, 256] (default)
  - Critic: MLP with same architecture
  - Both use ELU activation
  - Action noise: Learned std parameter per action dim
  - Distribution: Normal (Gaussian)
```

**Available Policies**:
- `ActorCritic` (MLP, feedforward)
- `ActorCriticRecurrent` (LSTM/GRU support for temporal history)
- `ActorCriticCost` (Constrained RL variant)

### 1.2 Training Paradigm: Teacher-Student Distillation + DAgger

The training follows a **two-stage approach**:

#### **Stage 1: Privileged Teacher Policy**
```bash
python legged_gym/scripts/train_hydra.py \
  --config-name=config_teleop \
  task=h1:teleop \
  run_name=OmniH2O_TEACHER \
  env.num_observations=913 \
  env.num_privileged_obs=990 \
  motion.teleop_obs_version=v-teleop-extend-max-full \
  motion=motion_full \
  motion.extend_head=True \
  num_envs=4096 \
  rewards=rewards_teleop_omnih2o_teacher
```

**Teacher Observations (913 dims)**: Privileged access to:
- Full reference motion state
- Terrain information
- Full body state estimates

#### **Stage 2: Student Policy via Distillation + DAgger**
```bash
python legged_gym/scripts/train_hydra.py \
  --config-name=config_teleop \
  task=h1:teleop \
  run_name=OmniH2O_STUDENT \
  env.num_observations=1665 \
  env.num_privileged_obs=1742 \
  motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel \
  train.distill=True \
  train.policy.init_noise_std=0.001 \
  env.add_short_history=True \
  env.short_history_length=25 \
  train.dagger.load_run_dagger=TEACHER_RUN_NAME \
  train.dagger.checkpoint_dagger=XXX \
  train.dagger.dagger_only=True
```

**Student Observations (1665 dims)**: No privileged info
- Current proprioceptive state (joint pos/vel)
- Body linear/angular velocity
- Reference motion tracking targets (from VR/teleoperation)
- **25-step history** of previous actions/states (for temporal context)

### 1.3 Policy Variants Supported

#### **A. History-Based Policies** (temporal context)
| Config | History Steps | Obs Dim | Architecture | Notes |
|--------|---|---|---|---|
| 0-step MLP | None | 90 | Feedforward MLP | Memoryless |
| 5-step MLP | 5 | 405 | Feedforward MLP | Limited context |
| 50-step MLP | 50 | 3240 | Feedforward MLP | Extended context |
| LSTM | Variable | 90 | LSTM | Recurrent state |
| GRU | Variable | 90 | GRU | Recurrent state |

#### **B. Tracking Point Variants** (control granularity)
- **Full body (23 points)**: All joints (pelvis, hips, knees, ankles, torso, shoulders, elbows)
- **Reduced (8 points)**: ankles, shoulders, elbows only (H2O baseline)
- **Custom subsets**: Via `motion.teleop_selected_keypoints_names`

#### **C. State Space Variants**
- With/without linear velocity
- With/without heading estimate
- Different observation versions: v-teleop, v-teleop-extend-vr-max, v-teleop-extend-max-full

---

## 2. PRE-TRAINED CHECKPOINTS

### 2.1 Status: **NOT INCLUDED IN REPO**

**Pre-trained models**: ❌ Not provided in this repository
- The repo contains **training scripts and infrastructure** but no checkpoint files
- **Motion files only**: `resources/motions/h1/stable_punch.pkl` (2.6MB example motion)

**For production models, the authors provide**:
- **Teacher checkpoint download**: Must train or contact authors
- **Student checkpoint download**: Must train or contact authors

### 2.2 Checkpoint Loading for Inference

```python
# From play_hydra.py
def play(cfg_hydra: DictConfig) -> None:
    env_cfg, train_cfg = cfg_hydra, cfg_hydra.train
    env = task_registry.make_env_hydra(name=cfg_hydra.task, hydra_cfg=cfg_hydra)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, 
        name=cfg_hydra.task, 
        args=cfg_hydra, 
        train_cfg=cfg_hydra.train
    )
    
    # Load checkpoint (if specified)
    if cfg_hydra.checkpoint > 0:
        ppo_runner.load(cfg_hydra.checkpoint)
    
    # Run inference
    obs = env.reset()
    while True:
        actions = ppo_runner.get_inference_policy(obs)  # or actor.forward(obs)
        obs, _, _, _ = env.step(actions)
```

**Checkpoint Format**:
- **File type**: PyTorch `.pt` files (model state_dict)
- **Storage location**: `output/h1/{exp_name}/models/{checkpoint_number}.pt`
- **Loading via hydra**: `load_run=OmniH2O_STUDENT checkpoint=55500`

---

## 3. ENVIRONMENT SETUP

### 3.1 Simulator: **Isaac Gym (Primary) + Optional MuJoCo**

**Simulator Stack**:
```
Isaac Gym Preview Release 4
  ├─ PhysX Physics Engine
  ├─ CUDA-enabled (GPU simulation)
  └─ Used for: Main training & inference

MuJoCo
  ├─ Optional: Used in PHC module for SMPL retargeting
  ├─ Via: phc/phc/smpllib/smpl_mujoco.py
  └─ Purpose: Forward kinematics for shape fitting
```

### 3.2 Dependencies Installation

```bash
# 1. Environment setup
conda create -n omnih2o python=3.8
conda activate omnih2o
pip3 install torch torchvision torchaudio

# 2. Isaac Gym (REQUIRED)
cd isaacgym/python && pip install -e .

# 3. Internal modules
pip install -e rsl_rl              # PPO trainer
pip install -e legged_gym          # RL environment
pip install -e phc                 # Motion retargeting

# 4. Python dependencies
pip install -r requirements.txt
```

**Key Dependencies** (from requirements.txt):
```
Core:
  - PyTorch (3.8+)
  - numpy==1.23.0 (specific version for compatibility)
  - scipy
  - scikit-image

RL/Sim:
  - isaacgym (preview 4)
  - gym
  - pytorch_lightning

Motion Processing:
  - git+https://github.com/ZhengyiLuo/smplx.git          # SMPL-X model
  - git+https://github.com/ZhengyiLuo/SMPLSim.git        # SMPL simulation
  - mujoco
  - chumpy

Utils:
  - hydra-core (configuration management)
  - wandb (experiment tracking)
  - opencv-python==4.6.0.66
  - joblib>=1.2.0
  - open3d (visualization)
  - onnxruntime==1.13.1 (ONNX export)
```

### 3.3 Hardware Requirements

**Minimum for Training**:
- GPU: NVIDIA GPU with 24GB+ VRAM (tested on A100, RTX 4090)
- CPU: 8+ cores
- RAM: 64GB+
- Parallel envs: 4096 by default (adjustable)

**For Inference Only**:
- GPU: 16GB VRAM sufficient
- CPU: 4+ cores

---

## 4. MOTION RETARGETING: SMPL → H1 Robot

### 4.1 Retargeting Pipeline (3 Steps)

#### **Step 1: Define H1 Forward Kinematics**
```
File: phc/phc/utils/torch_h1_humanoid_batch.py
Purpose: Implement FK from joint angles → end-effector positions
Content: Humanoid_Batch class with:
  - Joint chain definitions (H1 skeleton)
  - Forward kinematics computation
  - Optimization-friendly differentiable FK
```

#### **Step 2: Fit SMPL Shape to H1**
```bash
cd human2humanoid
python scripts/data_process/grad_fit_h1_shape.py
```

**Input**: SMPL models (SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl)
**Output**: `data/h1/shape_optimized_v1.pkl` (optimized SMPL shape parameters)

**Process**:
- Minimize distance between SMPL joint positions and H1 chain end-effectors
- Solve for beta (shape parameters) using gradient descent
- Result: SMPL shape that matches H1's morphology

#### **Step 3: Retarget AMASS → H1**
```bash
cd human2humanoid
python scripts/data_process/grad_fit_h1.py
```

**Input**: 
- AMASS dataset (15,886 motions in SMPL format)
- Shape-fitted SMPL parameters

**Output**: 
- H1-compatible motion file: `resources/motions/h1/amass_phc_filtered.pkl`
- Contains: Root translation + Joint rotations for H1

**Motion Format in PKL**:
```python
{
  "motion_id_1": {
    "root_trans_offset": torch.Tensor(T, 3),  # Root XYZ position offsets
    "pose_aa": torch.Tensor(T, 19, 3),        # 19 joints in axis-angle format
    "fps": 30,                                 # Motion frame rate
    # ... additional fields
  },
  "motion_id_2": { ... },
  ...
}
```

### 4.2 AMASS Dataset Preparation

```
human2humanoid/data/AMASS/AMASS_Complete/
  ├── ACCAD.tar.bz2
  ├── BMLhandball.tar.bz2
  ├── CMU.tar.bz2
  ├── MPI_HDM05.tar.bz2
  ├── ... (15+ datasets)
  └── Transitions.tar.bz2

# Extract all:
for file in *.tar.bz2; do tar -xvjf "$file"; done

# Result:
human2humanoid/data/AMASS/AMASS_Complete/
  ├── ACCAD/
  ├── CMU/
  ├── ...
  └── Transitions/
```

**Note**: Must download SMPL models separately from https://smpl.is.tue.mpg.de/

---

## 5. OBSERVATION & ACTION SPACE

### 5.1 Observation Space Versions

The system supports multiple observation versions configurable via `motion.teleop_obs_version`:

#### **v-teleop-extend-max-full (Teacher, 913 dims)**
```
Privileged observations:
├─ Robot state (current):
│  ├─ Root position (3)
│  ├─ Root orientation (6D representation)
│  ├─ Joint positions (19 × 1 = 19)
│  ├─ Joint velocities (19 × 1 = 19)
│  ├─ Body linear velocity (3)
│  ├─ Body angular velocity (3)
│  └─ Gravity vector (3)
│
├─ Reference motion state:
│  ├─ Ref body positions (multiple bodies × 3)
│  ├─ Ref body rotations (multiple bodies × 6D)
│  ├─ Ref body velocities (multiple bodies × 3)
│  ├─ Ref angular velocities (multiple bodies × 3)
│  ├─ Ref joint positions (19)
│  └─ Ref joint velocities (19)
│
├─ History:
│  └─ Last action (19)
│
└─ Task-specific:
   └─ Future motion samples (trajectory)
```

**Total: 913 dimensions** (includes full privileged reference motion)

#### **v-teleop-extend-vr-max-nolinvel (Student, 1665 dims)**
```
Non-privileged observations:
├─ Robot state:
│  ├─ Root position (3)
│  ├─ Root orientation (6D)
│  ├─ Joint positions (19)
│  ├─ Joint velocities (19)
│  ├─ Body linear velocity (3)
│  ├─ Body angular velocity (3)
│  ├─ Gravity vector (3)
│  └─ Heading (1)
│
├─ Tracking targets (from VR/reference):
│  ├─ Ref body positions × 25 steps (N_bodies × 3 × 25)
│  ├─ Ref body rotations × 25 steps (N_bodies × 6D × 25)
│  ├─ Ref body velocities × 25 steps (N_bodies × 3 × 25)
│  └─ Ref body angular velocities × 25 steps (N_bodies × 3 × 25)
│
├─ History:
│  ├─ Last 25 actions (19 × 25)
│  └─ Last 25 states (various)
│
└─ Control info:
   └─ Contact state (boolean per body)
```

**Total: 1665 dimensions** (includes 25-step history)

### 5.2 Observation Computation

**Key Function**: `HumanoidIm._compute_observations()` (humanoid_im.py)

**Multiple versions supported**:
```python
def compute_imitation_observations_v2(...):
    # Body pos + rot + vel + ang_vel (reference)
    # Joint pos + vel differences
    
def compute_imitation_observations_v3(...):
    # Reduced: body pos + rot + vel (no angular vel)
    
def compute_imitation_observations_v6(...):
    # Full tracking: local position + DOF + no diff
    
def compute_imitation_observations_v7(...):
    # Linear position + velocity only
```

**Selection**: Via config parameter `motion.teleop_obs_version`

### 5.3 Action Space

**Action Output**: **19-dimensional continuous action**

```python
# Joint targets for H1 (19 joints)
action = π.actor(observations)  # → (19,) continuous values

# Applied as position targets:
joint_targets = action
# Robot tracks targets via PD controllers
```

**Mapping to H1 Joints** (from h1_teleop_config.py):
```python
default_joint_angles = {
    'left_hip_yaw_joint':         0.0,
    'left_hip_roll_joint':        0.0,
    'left_hip_pitch_joint':      -0.4,
    'left_knee_joint':            0.8,
    'left_ankle_joint':          -0.4,
    
    'right_hip_yaw_joint':        0.0,
    'right_hip_roll_joint':       0.0,
    'right_hip_pitch_joint':     -0.4,
    'right_knee_joint':           0.8,
    'right_ankle_joint':         -0.4,
    
    'torso_joint':                0.0,
    
    'left_shoulder_pitch_joint':  0.0,
    'left_shoulder_roll_joint':   0.0,
    'left_shoulder_yaw_joint':    0.0,
    'left_elbow_joint':           0.0,
    
    'right_shoulder_pitch_joint': 0.0,
    'right_shoulder_roll_joint':  0.0,
    'right_shoulder_yaw_joint':   0.0,
    'right_elbow_joint':          0.0,
}
# 19 total joint degrees of freedom
```

**Control Frequency**:
- Simulation: 200 Hz (dt=0.005s)
- Control: 50 Hz (5 sim steps per control step)
- Motion reference: 30 FPS (standard mocap)

---

## 6. REFERENCE MOTION INPUT FORMAT

### 6.1 Motion File Format (.pkl)

```python
# Loading reference motions
motion_file = "resources/motions/h1/stable_punch.pkl"
motion_data = joblib.load(motion_file)

# Format (MotionLibH1/MotionLibSMPL)
motion_data = {
    "punch_001": {
        "root_trans_offset": np.array(T×3),      # Root XYZ translation
        "pose_aa": np.array(T×19×3),             # Joint rotations (axis-angle)
        "fps": 30,                                # Frames per second
        "gender": "neutral",                      # SMPL gender
        "betas": np.array(10,),                  # SMPL shape parameters
    },
    "punch_002": { ... },
}

# Where:
#   T = number of frames in motion
#   19 = number of joints (H1)
#   3 = axis-angle representation (3D rotation vector)
```

### 6.2 Motion Loading in Environment

```python
# From HumanoidIm.__init__()

# Load motion library
self._motion_lib = MotionLibH1(
    motion_file="resources/motions/h1/amass_phc_filtered.pkl",
    device=self.device,
    masterfoot_config=self._masterfoot_config,
    min_length=self._min_motion_len,
    im_eval=False  # Use training motions
)

# Load motions into batch
self._motion_lib.load_motions(
    skeleton_trees=self.skeleton_trees,           # H1 skeleton
    gender_betas=self.humanoid_shapes.cpu(),      # SMPL shapes (batch)
    limb_weights=self.humanoid_limb_and_weights.cpu(),
    random_sample=True,                           # Random motion sampling
    max_len=-1 if flags.test else self.max_len   # Motion length limit
)
```

### 6.3 Reference Motion Querying During Training

```python
# At each env step, get reference state for tracking:
time = self.progress_buf * self.dt + self._motion_start_times

# Query motion library
ref_motion_state = self._motion_lib.get_motion_state(
    motion_ids=self._sampled_motion_ids,  # Which motions (batch)
    motion_times=time,                     # Time in each motion
    offset=self._global_offset             # Global position offset
)

# Returns reference:
ref_motion_state = {
    'root_pos': torch.Tensor(N, 3),                    # Ref root position
    'root_rot': torch.Tensor(N, 4),                    # Ref root rotation (quat)
    'body_pos': torch.Tensor(N, num_bodies, 3),        # Ref body positions (FK)
    'body_rot': torch.Tensor(N, num_bodies, 4),        # Ref body rotations (quat)
    'body_vel': torch.Tensor(N, num_bodies, 3),        # Ref body velocities
    'body_ang_vel': torch.Tensor(N, num_bodies, 3),    # Ref angular velocities
    'dof_pos': torch.Tensor(N, 19),                    # Ref joint positions
}
```

### 6.4 Future Motion Sampling (Trajectory Hints)

When `motion.future_tracks=True`:

```python
# Sample future reference states (e.g., next 5 frames)
num_traj_samples = 5
time_steps = [t + i * dt for i in range(num_traj_samples)]

future_states = self._motion_lib.get_motion_state(
    motion_ids=self._sampled_motion_ids,
    motion_times=time_steps,  # Multiple future timestamps
)
# → Shape: (N, num_traj_samples, ...)
```

**Used for**: Curriculum learning where policy sees future trajectory hints

---

## 7. ISAAC GYM vs MUJOCO

### 7.1 Isaac Gym (Primary)

**Usage**: 
- ✅ **Main training environment** (4096 parallel envs)
- ✅ **Fast GPU simulation** (PhysX)
- ✅ **Tensorized operations** (batch efficiency)

**Why Isaac Gym**:
```python
# Vectorized environment
env = Legged_env(sim_device="cuda:0", num_envs=4096)
# All 4096 environments run in parallel on GPU
```

**Simulator Code**:
- `legged_gym/legged_gym/envs/base/legged_robot.py` - Base robot environment
- `phc/phc/env/tasks/humanoid_im.py` - Motion imitation task

### 7.2 MuJoCo (Auxiliary)

**Usage**:
- ❌ **NOT used for main training**
- ✅ **Used for motion retargeting** (forward kinematics fitting)
- ✅ **Optional: Used in PHC for SMPL mesh handling**

**Where MuJoCo is Used**:
```python
# From phc/phc/smpllib/smpl_mujoco.py
# SMPL_MuJoCo parser for shape fitting and visualization

from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
# This does FK computations for retargeting
```

### 7.3 MuJoCo-Only Solution Feasibility

**Question**: Can we train without Isaac Gym?
**Answer**: ❌ **NOT RECOMMENDED**

Why:
1. **No parallel vectorization**: MuJoCo CPU = single environment only
2. **Speed**: 4096 envs on CPU would be 10,000× slower
3. **Code tightly coupled**: Isaac Gym APIs used throughout

**BUT**: Could be adapted if:
- Reduce `num_envs` to 1-4 (single GPU)
- Use slower CPU+GPU hybrid simulation
- Rewrite environment wrapper for MuJoCo (significant effort)

---

## 8. TRAINING PIPELINE SUMMARY

```
┌─────────────────────────────────────────────────────────────────┐
│ Training Pipeline Overview                                       │
└─────────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   ├─ AMASS dataset (15,886 human motions in SMPL)
   ├─ SMPL model (male, female, neutral)
   └─ → Retarget to H1 skeleton
       └─ → amass_phc_filtered.pkl

2. TEACHER TRAINING (Stage 1)
   ├─ Config: config_teleop
   ├─ Observations: 913 dims (privileged)
   ├─ PPO optimizer
   ├─ 4096 parallel Isaac Gym envs
   ├─ Reward: Ref motion tracking + regularization
   └─ → Teacher checkpoint (e.g., step 55500)

3. STUDENT TRAINING (Stage 2 - DAgger + Distillation)
   ├─ Load teacher checkpoint
   ├─ Observations: 1665 dims (non-privileged + history)
   ├─ Training modes:
   │  ├─ Behavioral cloning from teacher
   │  ├─ On-policy RL (PPO)
   │  └─ Mixed via DAgger
   ├─ History length: 0, 5, or 50 steps
   ├─ Can train RNN variant (LSTM/GRU)
   └─ → Student checkpoint (e.g., step 50000)

4. INFERENCE/PLAYING
   ├─ Load student or teacher checkpoint
   ├─ Set num_envs=1
   ├─ Reset with initial reference motion
   ├─ Loop:
   │  ├─ obs = env.reset() or env.step(prev_action)
   │  ├─ action = policy.forward(obs)
   │  └─ env.step(action) → returns obs, reward, done, info
   └─ Optionally record/visualize with Open3D

5. EXPORT FOR DEPLOYMENT
   ├─ TorchScript JIT export
   ├─ ONNX export (with onnxruntime)
   └─ Deploy on Unitree H1 hardware
```

---

## 9. KEY CONFIGURATION FILES

```
legged_gym/legged_gym/cfg/
├── config_teleop.yaml              # Main config
├── asset/
│  └── asset_teleop.yaml            # Robot/asset configs
├── commands/
│  └── commands_teleop.yaml
├── control/
│  └── control_teleop.yaml           # PD gains, etc
├── domain_rand/
│  └── domain_rand_teleop.yaml       # Domain randomization
├── env/
│  └── env_teleop.yaml               # Environment parameters
├── init_state/
│  └── init_state_teleop.yaml        # Initial pose config
├── motion/
│  ├── motion_teleop.yaml            # Motion loading config
│  └── motion_full.yaml              # Full-body tracking
├── noise/
│  └── noise_teleop.yaml             # Sensor noise
├── rewards/
│  ├── rewards_teleop_omnih2o_teacher.yaml
│  └── rewards_teleop_omnih2o_student.yaml
└── train/
   └── ppo_teleop.yaml               # PPO hyperparameters
```

---

## 10. RUNNING INFERENCE

### 10.1 Loading and Running a Pre-trained Policy

```python
from legged_gym import task_registry
import torch

# 1. Create environment
cfg_hydra = {...}  # Load from config
env = task_registry.make_env_hydra(
    name="h1:teleop",
    hydra_cfg=cfg_hydra,
    env_cfg=cfg_hydra
)

# 2. Create policy runner
ppo_runner, train_cfg = task_registry.make_alg_runner(
    env=env,
    name="h1:teleop",
    args=cfg_hydra,
    train_cfg=cfg_hydra.train
)

# 3. Load checkpoint
ppo_runner.load(checkpoint=55500)  # Load teacher/student model

# 4. Get policy
actor = ppo_runner.alg.actor_critic.actor

# 5. Run inference loop
obs = env.reset()
for step in range(num_steps):
    with torch.no_grad():
        actions = actor(obs)
    obs, rewards, dones, info = env.step(actions)
    # obs shape: (1, num_obs)  if num_envs=1
    # actions shape: (1, 19)   for H1 robot
```

### 10.2 CLI Command for Playing

```bash
python legged_gym/legged_gym/scripts/play_hydra.py \
  --config-name=config_teleop \
  task=h1:teleop \
  env.num_observations=1665 \
  env.num_privileged_obs=1742 \
  motion.teleop_obs_version=v-teleop-extend-vr-max-nolinvel \
  motion.teleop_selected_keypoints_names=[] \
  motion.extend_head=True \
  num_envs=1 \
  sim_device=cuda:0 \
  load_run=OmniH2O_STUDENT \
  checkpoint=50000 \
  headless=False
```

**Output**:
- Real-time simulation visualization
- Robot tracks reference motion
- Can be exported to ONNX for deployment

---

## 11. SUMMARY TABLE

| Aspect | Details |
|--------|---------|
| **RL Algorithm** | PPO (Proximal Policy Optimization) |
| **Policy Network** | MLP actor-critic (3×256 hidden, ELU) |
| **RNN Variants** | LSTM, GRU (optional) |
| **Training Paradigm** | Teacher→Student + DAgger distillation |
| **Simulator** | Isaac Gym Preview 4 (GPU-accelerated) |
| **Parallel Envs** | 4096 (training), 1 (inference) |
| **Robot** | Unitree H1 humanoid (19 DOF) |
| **Motion Format** | PKL file (19×3 axis-angle + root trans) |
| **Obs Size (Teacher)** | 913 dims (privileged + future traj) |
| **Obs Size (Student)** | 1665 dims (no privilege + 25-step history) |
| **Action Size** | 19 dims (joint targets) |
| **Motion Library** | AMASS (15,886 retargeted motions) |
| **Control Freq** | 50 Hz (5× sim steps) |
| **Retargeting** | 3-step process (FK → shape fit → retarget) |
| **Pre-trained Models** | ❌ Not in repo (must train) |
| **Export Options** | TorchScript JIT, ONNX |

---

## 12. WHAT YOU NEED TO RUN INFERENCE

```
Required:
✓ Isaac Gym Preview 4 (installed + NVIDIA GPU)
✓ PyTorch + CUDA
✓ Pre-trained checkpoint (.pt file)
✓ Reference motion file (.pkl)
✓ H1 URDF asset (in resources/)

Optional:
○ Open3D (for visualization)
○ ONNX runtime (for ONNX export)
```

---

## References

- **Paper**: https://omni.human2humanoid.com/
- **Code**: Original repo (this analysis based on provided codebase)
- **Motion Retargeting**: Based on PHC (Perpetual Humanoid Control)
- **RL Framework**: RSL (Robotics Systems Lab, ETH Zurich)

