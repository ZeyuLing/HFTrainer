# Quick Start: MuJoCo RL for SMPL Humanoid (No IsaacGym)

## 30-Second Answer

❌ **IsaacGym is NOT installed** and the project doesn't need it.

✅ **You already have MuJoCo 3.1.6** with a trained SMPL humanoid policy.

🎯 **What you can do RIGHT NOW**:
1. Run pre-trained ONNX policy for physics-corrected motion
2. Use PD-tracking baseline (deterministic physics)
3. Train new policies with ProtoMotions or Gymnasium+SB3

---

## Fastest Path: Use Pre-Trained Policy (5 minutes)

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python3 scripts/embodied/run_smpl_rl_tracker.py \
    --npz-file output/embodied_t2m_v4/data/npz/walk_forward.npz \
    --output-dir output/smpl_mesh_physics \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx \
    --mjcf ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --yaml ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml
```

**Output**: Physics-corrected motion JSON (ready for web visualization)

---

## Option A: ProtoMotions (Professional RL Framework)

### Setup
```bash
# Already available in ref_repo/ProtoMotions/
cd ref_repo/ProtoMotions
pip install -e .  # Install as development package
```

### Train Motion Imitation Policy
```python
from protomotions.simulator.factory import SimulatorFactory
from protomotions.env.tasks.humanoid_im import HumanoidImitation
from stable_baselines3 import PPO
import hydra
from omegaconf import OmegaConf

# Load config with MuJoCo backend (NOT IsaacGym)
cfg = OmegaConf.create({
    "env": {
        "sim_backend": "mujoco",  # Key: use MuJoCo, not IsaacGym
        "num_envs": 1,
        "headless": True,
    }
})

# Create simulator
simulator = SimulatorFactory.create("mujoco", cfg.env)

# Create task
task = HumanoidImitation(cfg, simulator)

# Train with PPO
policy = PPO("MlpPolicy", task, learning_rate=3e-4, n_steps=2048)
policy.learn(total_timesteps=1_000_000)
policy.save("trained_policy")
```

**Advantages**:
- ✅ Complete RL framework (AMP, PPO, MCP supported)
- ✅ No IsaacGym dependency
- ✅ Production-tested on AMASS (11,313 sequences)
- ✅ Pre-trained policies available

**Disadvantages**:
- Large codebase (learning curve)
- Uses Hydra config system

---

## Option B: DIY Gymnasium + Stable-Baselines3 (Simpler)

### Step 1: Create Environment
```python
import gymnasium as gym
import mujoco
import numpy as np
from pathlib import Path

class SMPLHumanoidEnv(gym.Env):
    """SMPL humanoid motion imitation environment."""
    
    def __init__(self, mjcf_path, motion_lib=None, render=False):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.render = render
        
        # 76 DOF state + 75 DOF velocity observation
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(151,), dtype=np.float32
        )
        
        # 75 actuators (all joints except root translation)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(75,), dtype=np.float32
        )
        
        self.motion_lib = motion_lib or []
        self.current_motion = 0
        self.step_count = 0
        self.max_steps = 1000
    
    def step(self, action):
        # Apply action as PD target
        self.data.ctrl[:] = action
        
        # Step physics
        for _ in range(10):  # Substeps
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        # Observation: [qpos + qvel]
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel
        ]).astype(np.float32)
        
        # Reward: motion tracking + energy minimization
        if len(self.motion_lib) > 0:
            ref_qpos = self.motion_lib[self.current_motion, self.step_count % len(self.motion_lib)]
            tracking_error = np.mean((self.data.qpos - ref_qpos) ** 2)
        else:
            tracking_error = 0.0
        
        energy_cost = 0.001 * np.sum(action ** 2)
        reward = -(tracking_error + energy_cost)
        
        # Termination
        terminated = self.data.qpos[2] < 0.3  # Fell
        truncated = self.step_count >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Random motion
        if len(self.motion_lib) > 0:
            self.current_motion = self.np_random.integers(0, len(self.motion_lib))
            self.data.qpos[:] = self.motion_lib[self.current_motion, 0]
        else:
            # Default standing pose
            self.data.qpos[:] = 0.0
            self.data.qpos[2] = 0.9  # Height
        
        self.data.qvel[:] = 0.0
        self.step_count = 0
        
        mujoco.mj_forward(self.model, self.data)
        
        obs = np.concatenate([
            self.data.qpos,
            self.data.qvel
        ]).astype(np.float32)
        
        return obs, {}
    
    def render(self, mode="human"):
        if mode == "human" and self.render:
            # Optional: use mujoco.viewer
            pass
```

### Step 2: Load Motion Library
```python
import numpy as np

def load_motion_library(npz_dir):
    """Load all motion_135 NPZ files into memory."""
    motions = []
    for npz_file in Path(npz_dir).glob("*.npz"):
        data = np.load(npz_file)
        motion_135 = data["motion_135"]  # (T, 135)
        # Convert to MuJoCo qpos format here...
        motions.append(motion_135)
    return np.array(motions)

motion_lib = load_motion_library("output/embodied_t2m_v4/data/npz")
```

### Step 3: Train
```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create environment
env = SMPLHumanoidEnv(
    mjcf_path="ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml",
    motion_lib=motion_lib,
)

# Wrap for stable-baselines3
env = DummyVecEnv([lambda: env])

# Train PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
)

model.learn(total_timesteps=1_000_000)
model.save("smpl_humanoid_ppo")
```

**Advantages**:
- ✅ Simple, minimal implementation
- ✅ Standard gymnasium API
- ✅ Works with stable-baselines3
- ✅ Easy to debug

**Disadvantages**:
- Need to handle coordinate transforms yourself
- Slower than vectorized training
- No advanced features (AMP, MCP)

---

## Option C: PD-Tracking Baseline (No RL)

Deterministic physics without learning:

```bash
python3 scripts/embodied/run_smpl_physics_sim.py \
    --npz-file output/embodied_t2m_v4/data/npz/walk.npz \
    --output-dir output/pd_corrected \
    --xml-path ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml
```

**Good for**:
- Baseline comparisons
- Physics validation
- Debugging coordinate systems

---

## File Structure Reference

```
ref_repo/
├── ProtoMotions/                          ← Complete RL framework
│   └── data/pretrained_models/
│       └── motion_tracker/smpl/
│           ├── compiled_models/
│           │   ├── unified_pipeline.onnx  ← Pre-trained policy
│           │   └── unified_pipeline.yaml  ← Configuration
│           └── resolved_configs.yaml
│
├── OmniH2O/
│   └── phc/phc/data/assets/mjcf/
│       ├── smpl_humanoid.xml             ← 76-DOF model
│       ├── smpl_humanoid_1.xml
│       └── mesh_humanoid.xml
│
└── (Other RL frameworks)
    ├── HumanPlus/HST/rsl_rl/
    ├── VideoMimic/simulation/videomimic_rl/rsl_rl/
    └── UH-1/rsl_rl/

scripts/embodied/
├── run_smpl_rl_tracker.py                ← Use ONNX policy (↑)
├── run_smpl_physics_sim.py               ← PD-tracking baseline
├── physflow_trainer.py                   ← Physics-aware training
└── (20+ diagnostic scripts)
```

---

## Comparison: Which Option?

| Task | Solution | Time |
|------|----------|------|
| Get physics-corrected motion NOW | Option A (ONNX) | 5 min |
| Understand RL framework | ProtoMotions | 1-2 days |
| Quick proof-of-concept policy | Gymnasium+SB3 | 2-3 days |
| Production system | ProtoMotions | 3-7 days |
| Speed benchmark | PD-tracking | 30 min |

---

## Troubleshooting

### Q: "No module named 'isaacgym'"
**A**: Expected! You don't need IsaacGym. Use MuJoCo instead.

### Q: "How do I train new policies?"
**A**: Choose ProtoMotions (full framework) or Gymnasium+SB3 (simpler). Both don't need IsaacGym.

### Q: "How do I convert my motion to MuJoCo format?"
**A**: Use functions in `scripts/embodied/run_smpl_physics_sim.py`:
```python
from scripts.embodied.run_smpl_physics_sim import (
    decode_motion_135,
    yup_to_zup,
    smpl_to_qpos,
)

# motion_135 (T, 135) → MuJoCo qpos (T, 76)
smpl_pose, transl, fps = decode_motion_135("path/to/motion.npz")
smpl_pose_zup, transl_zup = yup_to_zup(smpl_pose, transl)
qpos = smpl_to_qpos(smpl_pose_zup, transl_zup, body_pos_1)
```

### Q: "Why use MuJoCo instead of IsaacGym?"
**A**: 
- ✅ MuJoCo is already installed
- ✅ No GPU required (IsaacGym is GPU-only)
- ✅ Simpler, smaller footprint
- ✅ Equal physics quality for this task
- ✅ Better for motion imitation (less parallelization needed)

---

## Key References

| Document | Purpose |
|----------|---------|
| `RL_MUJOCO_OPTIONS_ANALYSIS.md` | Comprehensive analysis of all options |
| `MuJoCo_Gymnasium_SB3_Feasibility_Assessment.md` | Technical feasibility report |
| `scripts/embodied/run_smpl_rl_tracker.py` | ONNX policy inference code |
| `ref_repo/ProtoMotions/README.md` | ProtoMotions documentation |
| `ref_repo/ProtoMotions/docs/retargeting.md` | Humanoid setup guide |

---

## Next Steps

1. **Pick your approach** (table above)
2. **Install dependencies**:
   ```bash
   pip install gymnasium stable-baselines3
   ```
3. **Run the example** for your approach
4. **Monitor training** (ProtoMotions or Gymnasium)
5. **Export trained policy** for inference

Good luck! 🚀
