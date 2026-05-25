# PHC (Perpetual Humanoid Control) — Complete Architecture Analysis

## Executive Summary

PHC is a **motion imitation system** that learns a universal policy to track arbitrary reference motions (from SMPL or H1 humanoid skeletons) in MuJoCo physics simulation. The policy runs on PPO with task rewards + adversarial discriminator rewards, trained on 1536 parallel environments for ~10M+ iterations.

**Core loop**: Observation (self + task) → Policy (MLP 1024-512) → Action (normalized joint velocities) → Physics simulation (MuJoCo, 30Hz) → Reward (position/rotation/velocity matching + power penalty) → PPO gradient updates.

---

## 1. OBSERVATION PIPELINE

### 1.1 Self Observation (Proprioceptive State)

The humanoid's own state at each timestep, computed via `_compute_humanoid_obs()`:

```
Self Obs = [
  dof_pos (joint angles, 3D axis-angle × 22 joints = 66D),
  dof_vel (joint velocities, 3D × 22 = 66D),
  root_pos (pelvis position, 3D),
  root_rot (pelvis rotation, 3D euler or quat normalized),
  root_lin_vel (pelvis linear velocity, 3D),
  root_ang_vel (pelvis angular velocity, 3D),
  contact_forces (force magnitude on key bodies × 4 feet, optional),
  ...other proprioceptive features...
]
```

**Key parameters** (from `h1_im_2.yaml`):
- `localRootObs: True` → root position/rotation in LOCAL frame
- `rootHeightObs: True` → root z-position included
- `numAMPObsSteps: 10` → history length for adversarial discriminator
- Multiple observation versions (`obs_v=1..9`) with different feature combinations

### 1.2 Task Observation (Reference Motion Features)

The target motion state + relative error, computed via `_compute_task_obs()`:

**Conceptually**:
```python
# Sample next 1-3 future frames from reference motion library
ref_root_pos, ref_root_rot = motion_lib.get_state(motion_id, time)
ref_body_pos, ref_body_rot = FK(ref_root, ref_dof)  # forward kinematics

# Compute relative features (in agent's local frame)
body_pos_diff = ref_body_pos - agent_body_pos
body_rot_diff = quaternion_error(ref_body_rot, agent_body_rot)
body_vel_diff = ref_body_vel - agent_body_vel

task_obs = [
  body_pos_diff (tracked bodies only, e.g., 9 bodies × 3D = 27D),
  body_rot_diff (quaternion error, 3D angle-axis × 9 = 27D),
  body_vel_diff (9 × 3D = 27D),
  body_ang_vel_diff (9 × 3D = 27D),
  ... repeat for future samples (trajSampleTimestepInv=3) if fut_tracks=True
]
```

**Observation Version Examples** (obs_v parameter):
- **obs_v=3**: minimal (pos+rot+vel differences only)
- **obs_v=6**: full body features (pos+rot+vel+ang_vel)
- **obs_v=9**: simplified (local pos+vel only, no rot diff)

**Total observation size** (example with obs_v=6, 9 tracked bodies, 1 sample):
```
= 66 (self dof) + 66 (dof_vel) + 3 (root_pos) + 3 (root_rot) + 3 (root_lin_vel) 
  + 3 (root_ang_vel) + 4 (contact forces)
  + 9 × (3 + 3 + 3 + 3)  [body pos/rot/vel/ang_vel differences]
= ~145 dims
```

**Augmentations**:
- `zero_out_far=True`: if agent too far from reference (>0.25m), zero out target rewards (only location reward)
- `_occl_training=True`: randomly occlude body parts (set to agent state) for robustness
- `fut_tracks_dropout=0.1`: randomly zero out future trajectory predictions

---

## 2. POLICY NETWORK ARCHITECTURE

### 2.1 Network Structure (network_builder.py)

```
Input Observation (145D, example)
  ↓
Normalize Input (running mean/std)
  ↓
MLP 1024-512-512
  ├─ Layer 1: Linear(145 → 1024) + ReLU + LayerNorm (optional)
  ├─ Layer 2: Linear(1024 → 512) + ReLU
  ├─ Layer 3: Linear(512 → 512) + ReLU
  ↓
Separate Actor/Critic Heads:
  ├─ Actor Head:
  │   ├─ Linear(512 → 22 × 3 = 66)  [mean of action distribution]
  │   ├─ Learnable sigma (fixed_sigma=True, learn_sigma=False)
  │   └─ Output: μ(s) ∈ [-∞, +∞] (unbounded)
  │
  └─ Critic Head:
      ├─ Linear(512 → 1)  [value function V(s)]
      └─ Output: v(s) ∈ ℝ
```

**Configuration** (from `learning/im.yaml`):
```yaml
mlp:
  units: [1024, 512]  # layer sizes
  activation: relu
  d2rl: False  # no D2RL variant

network:
  separate: True  # separate actor/critic
  
space:
  continuous:
    fixed_sigma: True
    sigma_init: "const_initializer", val=-2.9  # σ = exp(-2.9) ≈ 0.055
    learn_sigma: False  # fixed std dev
```

### 2.2 Action Output

```
μ = policy(obs)  ∈ ℝ^66  [unbounded means]
σ = exp(-2.9) * ones(66) ≈ 0.055  [fixed std, small exploration]

action ~ N(μ, σ²)
action_bounded = tanh(action)  ∈ [-1, 1]

Then action is clipped to [-1, 1] and scaled by:
  final_action = action_bounded × max_action_scale
```

The policy outputs **normalized joint velocities** (degrees/second, normalized to [-1,1]).

### 2.3 Initialization

- Actor initialization: `default` (Xavier uniform)
- Critic initialization: `default`
- Bias: zeros
- Activation: ReLU

---

## 3. ACTION → PHYSICS SIMULATION

### 3.1 Action Execution

```
policy_action ∈ [-1, 1]^66 (normalized velocities)
  ↓
Denormalize & scale to physical units
  ↓
PD Control Target: q_target = current_dof_pos + policy_action × dt
  ↓
MuJoCo PD Controller:
  τ = kp * (q_target - q_current) + kd * (v_target - v_current)
  where kp, kd are tuned per joint
  ↓
Apply torques to simulation
  ↓
MuJoCo Physics Step:
  substeps: 2  (run physics 2× per control frame)
  solver: TGS (Temporal Gauss-Seidel)
  num_position_iterations: 4
  dt_internal ≈ 0.0015s (30 Hz total: 1/30 = 0.033s)
```

**Control frequency**: 30 Hz (`controlFrequencyInv=2`, meaning sim runs at 60 Hz but control is 30 Hz)

### 3.2 Physics Simulation Parameters (physx section)

```yaml
sim:
  physx:
    contact_offset: 0.02
    rest_offset: 0.0
    bounce_threshold_velocity: 0.2
    max_depenetration_velocity: 10.0
    solver_type: 1  # TGS (more stable)
    num_threads: 4
    
  plane:
    staticFriction: 1.0
    dynamicFriction: 1.0
    restitution: 0.0  # no bouncing
```

---

## 4. REWARD FUNCTION

### 4.1 Task Reward (Motion Imitation)

**Primary reward** = `compute_imitation_reward()`:

```python
@torch.jit.script
def compute_imitation_reward(
    root_pos, root_rot, body_pos, body_rot, body_vel, body_ang_vel,
    ref_body_pos, ref_body_rot, ref_body_vel, ref_body_ang_vel,
    rwd_specs  # {"k_pos": 100, "k_rot": 10, "k_vel": 0.1, "k_ang_vel": 0.1,
               #  "w_pos": 0.5, "w_rot": 0.3, "w_vel": 0.1, "w_ang_vel": 0.1}
):
    # Body position reward (exponential decay)
    pos_diff = ref_body_pos - body_pos
    pos_dist = (pos_diff^2).mean(dim=-1).mean(dim=-1)  # MSE over all bodies
    r_pos = exp(-k_pos * pos_dist)  # k_pos=100 → very sharp penalty
    
    # Body rotation reward (quaternion error angle)
    quat_diff = quat_mul(ref_rot, quat_conjugate(agent_rot))
    angle_diff = quat_to_angle_axis(quat_diff)  # magnitude in radians
    angle_dist = (angle_diff^2).mean(dim=-1)
    r_rot = exp(-k_rot * angle_dist)  # k_rot=10
    
    # Velocity reward
    vel_diff = ref_body_vel - body_vel
    vel_dist = (vel_diff^2).mean(dim=-1).mean(dim=-1)
    r_vel = exp(-k_vel * vel_dist)  # k_vel=0.1
    
    # Angular velocity reward
    ang_vel_diff = ref_body_ang_vel - body_ang_vel
    ang_vel_dist = (ang_vel_diff^2).mean(dim=-1)
    r_ang_vel = exp(-k_ang_vel * ang_vel_dist)  # k_ang_vel=0.1
    
    # Weighted sum
    reward = w_pos * r_pos + w_rot * r_rot + w_vel * r_vel + w_ang_vel * r_ang_vel
    return reward
```

**Default weights**:
```yaml
reward_specs:
  k_pos: 100      # position penalty scale (exponential)
  k_rot: 10       # rotation penalty scale
  k_vel: 0.1      # velocity penalty scale
  k_ang_vel: 0.1  # angular velocity penalty scale
  w_pos: 0.5      # position weight (50%)
  w_rot: 0.3      # rotation weight (30%)
  w_vel: 0.1      # velocity weight (10%)
  w_ang_vel: 0.1  # angular velocity weight (10%)
```

### 4.2 Location Reward (Optional)

When `zero_out_far=True`:
```python
distance = ||root_pos - ref_root_pos||
if distance < 0.25m:
    # Full imitation reward (above)
else:
    # Only location reward (guide back)
    r_location = exp(-1.0 * distance^2)
```

### 4.3 Power Penalty

```python
power = |force · velocity|  per joint, summed
power_reward = -power_coefficient × power
power_coefficient = 0.0005  # default

# Only applied after frame 3 (first 3 frames excluded to avoid startup spikes)
power_reward[progress_buf <= 3] = 0
```

### 4.4 Discriminator Reward (Adversarial)

Runs in parallel with task reward:
```
total_reward = task_reward_w × r_task + disc_reward_w × r_disc
where:
  task_reward_w = 0.5
  disc_reward_w = 0.5
  r_disc = discriminator(obs) ∈ [0, 1]  # is this observation real motion?
```

**Discriminator updates**: trained on replay buffer of real motions vs. learned trajectories (AMP framework)

---

## 5. REFERENCE MOTION FORMAT & LOADING

### 5.1 Motion Data Structure (MotionLibSMPL)

Each motion file contains:
```python
{
    "pose_aa": (T, 72) float32,  # SMPL pose angles (22 joints × 3 axis-angle)
    "pose_quat_global": (T, 22, 4) float32,  # global quaternions
    "root_trans_offset": (T, 3) float32,  # root translation (adjusted for height)
    "root_rot": (T, 4) float32,  # root rotation quat
    "root_vel": (T, 3),
    "root_ang_vel": (T, 3),
    "dof_pos": computed via FK,
    "fps": 30 or 60,
    "gender_betas": (1 + 10,)  # gender (0=neutral, 1=male, 2=female) + SMPL shape params
}
```

**Loaded as**: `SkeletonMotion` object with forward kinematics cached

### 5.2 Motion Sampling During Training

```python
# Load motions from disk
motion_lib.load_motions(
    skeleton_trees=humanoid_skeletons,  # one per humanoid
    gender_betas=humanoid_shapes,       # one per humanoid
    random_sample=True,                  # randomly select motions
    max_len=300                          # max sequence length in frames
)

# Each timestep, for each env:
motion_id = sample_motion()  # random motion ID
motion_time = sample_time(motion_id)  # random frame in motion

# Fetch reference state:
ref_state = motion_lib.get_motion_state(motion_id, motion_time)
# returns: root_pos, root_rot, dof_pos, root_vel, root_ang_vel, ...
#          + forward kinematics for all bodies
```

### 5.3 Motion Library Configuration

```yaml
env:
  # Tracking
  trackBodies: [21 joint names]  # which bodies to track (subset)
  resetBodies: [all joint names]  # which bodies to reset from motion
  
  # Motion parameters
  min_length: -1  # no minimum
  numEnvs: 1536   # parallel environments
  episodeLength: 300  # max 10 seconds per episode (300 frames @ 30Hz)
```

---

## 6. TRAINING LOOP (PPO + AMP)

### 6.1 High-Level Flow

```
Initialize:
  - Load motion library (all .pkl motion files)
  - Create 1536 parallel environments
  - Initialize policy network (random weights)
  - Initialize discriminator network
  
for epoch in range(10000000):
    # Collect experience (rollout 32 frames per env)
    for step in range(horizon_length=32):
        obs = env.get_obs()
        action, value = policy.forward(obs)
        env.step(action)
        
        reward = env.compute_reward()
        disc_reward = discriminator(obs)
        next_obs = env.get_obs()
        
        store (obs, action, reward, disc_reward, value, next_obs)
    
    # Compute advantages (GAE)
    advantages = compute_gae(rewards, values, gamma=0.99, tau=0.95)
    
    # Update policy (mini_epochs=6, minibatch_size=16384)
    for mini_epoch in range(6):
        for batch in data_loader(shuffle=True):
            loss = ppo_loss(batch, old_policy)
            loss.backward()
            grad_norm_clip(50.0)
            optimizer.step()
    
    # Collect AMP discriminator data
    collect discriminator samples (policy vs. real motion)
    
    # Update discriminator (amp_mini_epochs iterations)
    for iteration in range(...):
        disc_loss = ... (binary cross-entropy)
        disc_loss.backward()
        optimizer_disc.step()
    
    # Save checkpoint every 2500 epochs
```

### 6.2 PPO Configuration

```yaml
config:
  ppo: True
  learning_rate: 2e-5  # very small LR
  lr_schedule: constant
  
  gamma: 0.99          # discount factor
  tau: 0.95            # GAE λ
  
  horizon_length: 32   # rollout steps
  minibatch_size: 16384  # 1536 envs × 32 steps = 49152, split into 3 batches
  mini_epochs: 6       # gradient updates per batch
  
  e_clip: 0.2          # PPO clipping range (20%)
  grad_norm: 50.0      # gradient clipping
  
  critic_coef: 5       # weight of value loss
  entropy_coef: 0.0    # no entropy regularization
  
  normalize_input: True
  normalize_value: True
```

### 6.3 AMP (Adversarial Motion Priors)

```yaml
amp_obs_demo_buffer_size: 200000   # store real motion observations
amp_replay_buffer_size: 200000     # store generated observations
amp_replay_keep_prob: 0.01         # 1% of generated samples kept

amp_batch_size: 512
amp_minibatch_size: 4096
disc_coef: 5                       # weight of discriminator loss
disc_logit_reg: 0.01               # regularization
disc_grad_penalty: 5               # gradient penalty coef
disc_reward_scale: 2               # reward scaling

task_reward_w: 0.5                 # combine task + disc rewards
disc_reward_w: 0.5

discriminator:
  units: [1024, 512]
  activation: relu
```

### 6.4 Training Hyperparameters Summary

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `numEnvs` | 1536 | parallel environments |
| `horizon_length` | 32 | frames collected per env before gradient update |
| `minibatch_size` | 16384 | 1536×32 / 3 (3 gradient updates per epoch) |
| `mini_epochs` | 6 | 6 PPO updates on same batch |
| `learning_rate` | 2e-5 | very conservative |
| `max_epochs` | 10M | train until convergence |
| `save_frequency` | 2500 | save checkpoint every 2500 epochs |
| `episodeLength` | 300 | max 10s per episode |

---

## 7. REFERENCE MOTION QUERY INTERFACE

### 7.1 Getting Motion State at Time T

```python
# Discrete time query
motion_id = 5  # motion index in library
motion_time = 2.0  # seconds into motion

# Motion library caches and returns:
result = motion_lib.get_motion_state(
    motion_ids=torch.tensor([5, 3, 1, ...]),  # batch of motion IDs
    motion_times=torch.tensor([2.0, 1.5, 3.0, ...]),  # times in seconds
    offset=None  # optional global offset
)

result = {
    "root_pos": (B, 3),          # pelvis position
    "root_rot": (B, 4),          # pelvis quaternion
    "dof_pos": (B, 66),          # joint angles (axis-angle)
    "root_vel": (B, 3),
    "root_ang_vel": (B, 3),
    "dof_vel": (B, 66),
    "rg_pos": (B, num_bodies, 3),  # rigid body positions (FK)
    "rb_rot": (B, num_bodies, 4),  # rigid body quaternions (FK)
    "body_vel": (B, num_bodies, 3),
    "body_ang_vel": (B, num_bodies, 3),
    # ... other fields
}
```

### 7.2 Caching Strategy

To avoid redundant computation, observations cache the most recent motion query:

```python
ref_motion_cache = {
    'motion_ids': previous_ids,
    'motion_times': previous_times,
    'offset': previous_offset,
    # ... all returned fields
}

# If query params unchanged, return cache immediately
# Otherwise, compute and cache
```

### 7.3 Future Trajectory Sampling (fut_tracks)

When `fut_tracks=True` and `numTrajSamples=3`:

```python
# Sample 3 future frames at intervals
for i in range(3):
    lookahead_time = (progress_buf + 1 + i * trajSampleTimestep) * dt
    future_state = motion_lib.get_motion_state(..., lookahead_time)
    # Concatenate to task observation
```

---

## 8. END-TO-END DATA FLOW DIAGRAM

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING ITERATION                             │
└──────────────────────────────────────────────────────────────────┘

Step 1: RESET PHASE (episode start)
  ├─ Random motion ID from library
  ├─ Random time in motion
  └─ Sample initial humanoid state from reference motion
     → Agent reset to match reference pose

Step 2: ROLLOUT (32 timesteps)
  for t in 0..31:
    ├─ Environment Observation
    │   ├─ Self obs: [dof_pos, dof_vel, root_pos/rot/vel, contacts]
    │   ├─ Task obs: query motion_lib at time (t+1) → reference state
    │   │            compute relative features
    │   └─ Total obs ≈ 145D
    │
    ├─ Policy Forward Pass
    │   ├─ Normalize input
    │   ├─ MLP(1024, 512, 512)
    │   ├─ Actor head → μ(s), σ (fixed)
    │   ├─ Sample action ~ N(μ, σ²)
    │   └─ Tanh clipping → action ∈ [-1, 1]
    │
    ├─ Physics Simulation
    │   ├─ PD control: τ = kp*(q_target - q) + kd*(v_target - v)
    │   ├─ MuJoCo step (30 Hz, 2 substeps)
    │   └─ Update: root_pos, root_rot, dof_pos, velocities, contacts
    │
    ├─ Reward Computation
    │   ├─ Query motion_lib at time t (now current frame)
    │   ├─ r_task = weighted sum of:
    │   │   - r_pos = exp(-100 * ||ref_pos - agent_pos||²)
    │   │   - r_rot = exp(-10 * angle_error²)
    │   │   - r_vel = exp(-0.1 * ||ref_vel - agent_vel||²)
    │   │   - r_ang_vel = similar
    │   │   - r_power = -0.0005 * |force · velocity|
    │   ├─ r_disc = discriminator(obs)  [0 or 1]
    │   ├─ r_total = 0.5*r_task + 0.5*r_disc
    │   └─ Store (obs, action, reward, value)
    │
    ├─ Early Termination Check
    │   └─ If ||agent - reference|| > 0.5m → reset episode

  ← repeat ×1536 environments in parallel

Step 3: ADVANTAGE COMPUTATION (after rollout)
  ├─ Forward pass on all collected obs to get V(s)
  ├─ Compute TD residuals: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
  ├─ GAE: A_t = Σ (γλ)^l δ_{t+l}  (λ=0.95)
  └─ returns: R_t = A_t + V(s_t)

Step 4: PPO UPDATE (mini_epochs=6)
  for mini_epoch in 1..6:
    ├─ Shuffle data
    ├─ Split into minibatches (16384 samples each)
    └─ For each minibatch:
        ├─ Compute policy loss:
        │   ratio = π_new(a|s) / π_old(a|s)
        │   clipped_ratio = clamp(ratio, 1±0.2)
        │   loss = -min(ratio*A, clipped_ratio*A)
        │
        ├─ Compute value loss:
        │   value_clipped = V_old + clamp(V_new - V_old, ±δ)
        │   loss = (R - V_clipped)²
        │
        ├─ Total loss = policy_loss + 5*value_loss
        ├─ Backward pass
        ├─ Clip gradients (max norm 50)
        └─ Optimizer step

Step 5: DISCRIMINATOR UPDATE
  ├─ Collect real motion observations (from motion library)
  ├─ Collect generated observations (from policy rollout)
  ├─ Binary classification:
  │   loss = BCE(disc(real), 1) + BCE(disc(generated), 0)
  │   + gradient penalty + logit regularization
  └─ Update discriminator weights

Step 6: CHECKPOINT SAVE (every 2500 epochs)
  └─ Save: policy weights, discriminator, optimizer state
```

---

## 9. UNIVERSAL vs. PER-MOTION POLICY

PHC is fundamentally a **universal policy**:

- **Single policy network** trained on 1000s of motions
- **Not a classifier** (no discrete motion IDs in input)
- **One forward pass** to get action for any reference motion
- The policy learns to **generalize** motion-following

However, the training uses **multi-motion curriculum**:
- Each episode randomly samples a different motion
- Rewards shaped by per-motion error (no task discrimination)
- The policy learns implicit motion representation via task observation

**Key difference from per-motion specialist policies**:
- ✅ Single 66D action output for all motions
- ✅ Reference motion info only in task observation
- ✅ Can generalize to held-out motions (test time)
- ❌ Cannot explicitly switch behaviors (no discrete action)

---

## 10. KEY CONFIGURATION FILES

### 10.1 Environment Config: `h1_im_2.yaml`
- 1536 parallel environments
- 300 frame episodes (10s @ 30Hz)
- obs_v=6 (full body features)
- trackBodies: [9 key joints]
- zero_out_far: penalize if too far from reference

### 10.2 Learning Config: `im.yaml`
- Algorithm: `im_amp` (motion imitation + adversarial)
- MLP: 1024-512 units
- PPO: lr=2e-5, horizon=32, mini_epochs=6
- AMP: disc_reward_w=0.5, task_reward_w=0.5

### 10.3 Motion Data
- Directory: `phc/data/motions/` (not shown, external)
- Format: PKL with pose_aa, root_trans, gender_betas
- Loaded via MotionLibSMPL (handles SMPL FK)

---

## 11. COMPLETE ARCHITECTURE SUMMARY

```
OBSERVATION (145D)
├─ Self State (78D):
│  ├─ dof_pos (66D)
│  ├─ dof_vel (66D, optional)
│  ├─ root pos/rot/vel/ang_vel (12D)
│  └─ contact forces (4D)
│
└─ Task State (67D):
   ├─ body pos diff (27D)
   ├─ body rot diff (27D)
   ├─ body vel diff (9D)
   └─ body ang_vel diff (4D, for some versions)

         ↓ Normalize

POLICY (MLP)
├─ Layer 1: Linear(145 → 1024) + ReLU + LayerNorm
├─ Layer 2: Linear(1024 → 512) + ReLU
├─ Layer 3: Linear(512 → 512) + ReLU
├─ Actor:   Linear(512 → 66) [mean]
└─ Critic:  Linear(512 → 1) [value]

         ↓ Sample Action

ACTION (66D) = tanh(μ + σ*noise) ∈ [-1, 1]

         ↓ Denormalize to joint velocities

PD CONTROLLER (MuJoCo)
τ = kp*(q_target - q) + kd*(v_target - v)
         ↓

PHYSICS SIMULATION (30 Hz)
Forward kinematics → all body pos/rot/vel

         ↓ Query reference from motion library

REWARD COMPUTATION
r_task = 0.5*exp(-100*pos_err²) + 0.3*exp(-10*rot_err²) 
         + 0.1*exp(-0.1*vel_err²) + 0.1*exp(-0.1*ang_vel_err²)
         - 0.0005*power

r_disc = discriminator(obs)

r_total = 0.5*r_task + 0.5*r_disc

         ↓ PPO Gradient Update

POLICY WEIGHTS UPDATE
```

---

## 12. LIMITATIONS & DESIGN CHOICES

1. **Fixed std deviation** (`learn_sigma=False`): Conservative exploration, focuses on mean prediction
2. **No explicit motion ID**: Policy must infer from task observations (implicit representation)
3. **Humanoid-specific**: Requires SMPL model + MuJoCo; not generalizable to other skeletons
4. **Physics-dependent**: Performance heavily depends on simulator parameters (friction, solver, etc.)
5. **Supervised by demonstrations**: Needs high-quality motion capture; can't create novel motions
6. **Single policy scale**: Works only for this humanoid model; needs retraining for different body shape

---

## References

- **Base Framework**: IsaacGym (NVIDIA GPU-accelerated simulator)
- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **Adversarial Learning**: AMP (Adversarial Motion Priors, Zhang et al. 2021)
- **Motion Representation**: SMPL (Skinned Multi-Person Linear Model)
- **Physics Engine**: MuJoCo (accessed via IsaacGym)

