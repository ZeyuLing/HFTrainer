# OmniH2O Quick Reference Guide

## TL;DR - Key Findings

| Question | Answer |
|----------|--------|
| **RL Policy?** | ✅ PPO with MLP actor-critic (256³ hidden dims) |
| **Pre-trained models?** | ❌ NOT in repo (must train or download from authors) |
| **Simulator** | ✅ Isaac Gym Preview 4 (GPU-parallel, PhysX) |
| **Retargeting** | ✅ 3-step: FK→shape-fit→SMPL-to-H1 |
| **Obs space** | 913 dims (teacher) / 1665 dims (student with 25-step history) |
| **Action space** | 19 dims (H1 joint targets) |
| **Motion format** | PKL: {motion_id: {pose_aa (T×19×3), root_trans (T×3)}} |
| **Training paradigm** | Teacher→Student distillation + DAgger |
| **MuJoCo-only?** | ❌ Not practical (no GPU parallelization) |

---

## 1️⃣ MOTION IMITATION POLICY

### Where is the code?
- **Training**: `legged_gym/legged_gym/scripts/train_hydra.py`
- **Policy network**: `rsl_rl/rsl_rl/modules/actor_critic.py`
- **PPO optimizer**: `rsl_rl/rsl_rl/runners/on_policy_runner.py`

### What RL algorithm?
**PPO (Proximal Policy Optimization)**
- Continuous control (Gaussian policy)
- Actor: 256→256→256→19 (MLP)
- Critic: 256→256→256→1 (MLP)
- Activation: ELU
- Learned noise std per action

### Architecture variants?
- **Feedforward MLP** (default)
- **LSTM** (with history)
- **GRU** (with history)

---

## 2️⃣ PRE-TRAINED CHECKPOINTS

### Available in repo?
❌ **NO**

### What's included?
- ✅ Training infrastructure
- ✅ Example motion: `resources/motions/h1/stable_punch.pkl` (2.6MB)
- ❌ Teacher checkpoint
- ❌ Student checkpoint

### How to use them?
```bash
# If you had a checkpoint:
python legged_gym/legged_gym/scripts/play_hydra.py \
  --config-name=config_teleop \
  task=h1:teleop \
  load_run=OmniH2O_STUDENT \
  checkpoint=50000
```

### Checkpoint location
`output/h1/{exp_name}/models/{checkpoint_number}.pt`

---

## 3️⃣ SIMULATOR

### Primary: Isaac Gym Preview 4
```
✅ GPU-accelerated (PhysX)
✅ Parallel vectorization (4096 envs)
✅ TensorFlow-friendly (batch operations)
✅ REQUIRED for training
```

### Optional: MuJoCo
```
❌ NOT used for main training
✅ Used for SMPL retargeting
✅ Used for FK fitting
```

### Can I use only MuJoCo?
**NO - not practical**
- No GPU parallelization
- 4096 → 1 env = 10,000× slower
- Tightly coupled to Isaac Gym API

---

## 4️⃣ MOTION RETARGETING

### 3-Step Pipeline

**Step 1: H1 Forward Kinematics**
```
File: phc/phc/utils/torch_h1_humanoid_batch.py
Defines: H1 skeleton chain, FK computation
```

**Step 2: Fit SMPL Shape**
```bash
python scripts/data_process/grad_fit_h1_shape.py
# Output: data/h1/shape_optimized_v1.pkl
# Optimizes SMPL beta to match H1 morphology
```

**Step 3: Retarget AMASS**
```bash
python scripts/data_process/grad_fit_h1.py
# Input: 15,886 AMASS motions
# Output: resources/motions/h1/amass_phc_filtered.pkl
```

### Input format (AMASS)
```
data/AMASS/AMASS_Complete/
├── ACCAD/
├── CMU/
├── MPI_HDM05/
└── ...
```

---

## 5️⃣ OBSERVATION SPACE

### Teacher (913 dims)
```
├─ Robot state (current):       ~50 dims
├─ Reference motion state:      ~800 dims  ← PRIVILEGED
├─ History (last action):       19 dims
└─ Future trajectory:           ~44 dims
```

### Student (1665 dims)
```
├─ Robot state (current):       ~50 dims
├─ Reference motion state:      ~1500 dims ← 25-step HISTORY
└─ Contact state:               ~15 dims
```

### Breakdown (example student)
```
Proprioceptive:
  ├─ Root pos (3) + rot (6) = 9
  ├─ Joint pos (19) + vel (19) = 38
  ├─ Body lin/ang vel (6) + gravity (3) = 9
  └─ Heading (1)
  → Subtotal: ~60 dims

Reference (25-step history):
  ├─ Ref body pos (N × 3 × 25)
  ├─ Ref body rot (N × 6 × 25)
  ├─ Ref body vel (N × 3 × 25)
  ├─ Ref ang vel (N × 3 × 25)
  └─ Previous actions (19 × 25)
  → Subtotal: ~1500 dims

Total: 1665
```

---

## 6️⃣ ACTION SPACE

### Output
**19-dimensional continuous vector**

### Mapping (H1 Joints)
```
Legs (10 joints):
  ├─ L: yaw, roll, pitch + knee + ankle = 5
  └─ R: yaw, roll, pitch + knee + ankle = 5

Torso (1 joint):
  └─ torso

Arms (8 joints):
  ├─ L: pitch, roll, yaw + elbow = 4
  └─ R: pitch, roll, yaw + elbow = 4

Total: 19 DOF
```

### Control frequency
- Sim: 200 Hz (dt=0.005)
- Control: 50 Hz (5 sim steps per action)
- Motion ref: 30 FPS

---

## 7️⃣ REFERENCE MOTION FORMAT

### PKL file structure
```python
{
  "motion_id_1": {
    "root_trans_offset": np.array(T, 3),     # XYZ translation
    "pose_aa": np.array(T, 19, 3),           # Joint axis-angle
    "fps": 30,                                # Frame rate
    "gender": "neutral",                      # SMPL gender
    "betas": np.array(10),                   # SMPL shape
  },
  "motion_id_2": { ... },
}

# Where:
#   T = number of frames
#   19 = H1 joints
#   3 = axis-angle representation
```

### Loading
```python
import joblib

motion_lib = MotionLibH1(
    motion_file="resources/motions/h1/amass_phc_filtered.pkl",
    device="cuda:0"
)

# Query at timestep t:
ref_state = motion_lib.get_motion_state(
    motion_ids=[0, 1, 2],              # Which motions
    motion_times=[0.5, 0.5, 0.5]       # Time in each
)
# Returns: {root_pos, root_rot, body_pos, body_rot, ...}
```

---

## 8️⃣ TRAINING PIPELINE

### Stage 1: Teacher (Privileged)
```bash
python legged_gym/legged_gym/scripts/train_hydra.py \
  --config-name=config_teleop \
  task=h1:teleop \
  run_name=OmniH2O_TEACHER \
  env.num_observations=913 \
  num_envs=4096 \
  motion.teleop_obs_version=v-teleop-extend-max-full
```

**Key params**:
- Observations: 913 dims (with privileged info)
- Reward: Tracking loss + regularization
- Training time: ~12-24 hours (A100)

### Stage 2: Student (DAgger + Distillation)
```bash
python legged_gym/legged_gym/scripts/train_hydra.py \
  --config-name=config_teleop \
  task=h1:teleop \
  run_name=OmniH2O_STUDENT \
  env.num_observations=1665 \
  train.distill=True \
  env.add_short_history=True \
  env.short_history_length=25 \
  train.dagger.load_run_dagger=OmniH2O_TEACHER \
  train.dagger.checkpoint_dagger=55500
```

**Key params**:
- Observations: 1665 dims (no privilege + history)
- Mode: Behavioral cloning + on-policy RL
- Training time: ~12-24 hours (A100)

---

## 9️⃣ POLICY VARIANTS

### By temporal context:
| Variant | History | Obs Dim | Notes |
|---------|---------|---------|-------|
| 0-step MLP | None | 90 | Memoryless |
| 5-step MLP | 5 | 405 | Limited context |
| 50-step MLP | 50 | 3240 | Extended context |
| LSTM | Recurrent | 90 | Learned state |
| GRU | Recurrent | 90 | Learned state |

### By tracking points:
- **Full (23 points)**: All joints
- **Reduced (8 points)**: Ankles + shoulders + elbows (H2O)
- **Custom**: Via config

### By state space:
- With/without linear velocity
- With/without heading
- v-teleop vs v-teleop-extend-max vs v-teleop-extend-vr-max

---

## 🔟 INSTALLATION

```bash
# 1. Environment
conda create -n omnih2o python=3.8
conda activate omnih2o

# 2. PyTorch
pip install torch torchvision torchaudio

# 3. Isaac Gym (CRITICAL)
cd isaacgym/python && pip install -e .

# 4. Internal packages
pip install -e rsl_rl
pip install -e legged_gym
pip install -e phc

# 5. Dependencies
pip install -r requirements.txt
```

### Hardware
- **Training**: GPU 24GB+ (A100, RTX 4090)
- **Inference**: GPU 16GB
- **CPU**: 8+ cores
- **RAM**: 64GB

---

## Summary

```
┌─────────────────────────────────────────┐
│ OmniH2O Motion Imitation Stack          │
├─────────────────────────────────────────┤
│ PPO (RL)                                │
│  ↓                                      │
│ MLP Actor-Critic (19 outputs)           │
│  ↓                                      │
│ Isaac Gym (4096 parallel envs)          │
│  ↓                                      │
│ H1 Robot (19 DOF)                       │
│  ↓                                      │
│ Reference: AMASS→SMPL→H1 (motions)     │
└─────────────────────────────────────────┘

Training: Teacher (privileged) → Student (no privilege)
Framework: RSL + Legged Gym + PHC (motion retargeting)
Export: TorchScript JIT + ONNX
```

