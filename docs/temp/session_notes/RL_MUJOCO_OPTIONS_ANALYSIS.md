# RL-Based Humanoid Imitation Options Without IsaacGym

## Executive Summary

The codebase contains **multiple viable paths** for MuJoCo-based RL training WITHOUT IsaacGym:

1. ✅ **ONNX RL Policy Inference** (READY TO USE) — No training required
2. ✅ **MuJoCo + PD-tracking** (PARTIALLY IMPLEMENTED) — Excellent foundation
3. ✅ **ProtoMotions Framework** (AVAILABLE) — Complete RL infrastructure with MuJoCo support
4. 🔶 **Custom Gymnasium + Stable-Baselines3** (NEEDS IMPLEMENTATION) — Modern approach
5. ✅ **rsl_rl** (AVAILABLE IN REF REPOS) — Tested PPO implementation

---

## 1. ONNX RL Policy Inference (✅ READY NOW)

### Status
- **Trained ONNX policy available** for SMPL humanoid motion tracking
- **No training required** — inference only
- Can run physics-corrected motion directly

### Key Files
```
/apdcephfs/.../ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/
├── compiled_models/
│   ├── unified_pipeline.onnx      ← Pre-trained RL policy (inference)
│   └── unified_pipeline.yaml      ← YAML metadata with robot/control config
├── resolved_configs_inference.yaml
└── resolved_configs.yaml
```

### Implementation
- **Script**: `scripts/embodied/run_smpl_rl_tracker.py` (1163 lines)
- **What it does**:
  - Loads trained ONNX policy (no IsaacGym/training)
  - Runs closed-loop MuJoCo simulation
  - Policy outputs joint position targets for PD control
  - Handles motion tracking with physics constraints

### Usage Example
```python
from scripts.embodied.run_smpl_rl_tracker import process_single_motion

stats = process_single_motion(
    npz_path="output/embodied_t2m_v4/data/npz/walk_forward.npz",
    output_dir="output/smpl_mesh_physics",
    onnx_path="ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx",
    mjcf_path="ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml",
    yaml_path="ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml",
)
```

### Strengths
✅ No training time  
✅ No IsaacGym dependency  
✅ Works with MuJoCo 3.1.6 (already installed)  
✅ Physics-grounded motion output  
✅ Fully debugged and tested  

### Limitations
❌ Cannot train new policies (inference only)  
❌ Limited to motion tracking task  

---

## 2. MuJoCo + PD-Tracking (✅ PARTIALLY READY)

### Status
- Core MuJoCo physics simulation working
- PD-tracking (deterministic, no RL) fully functional
- Can serve as training environment foundation

### Key Files
```
scripts/embodied/
├── run_smpl_physics_sim.py         ← PD-tracking baseline (1300+ lines)
├── physflow_physics_oracle.py      ← Physics correction oracle
├── physflow_trainer.py             ← Physics-aware flow training
├── physflow_curriculum.py          ← Curriculum learning
└── physflow_evaluate.py            ← Evaluation utilities

ref_repo/OmniH2O/phc/phc/data/assets/mjcf/
└── smpl_humanoid.xml               ← 76-DOF SMPL model (24 bodies, 23 actuators)
```

### Model Structure
- **Total DOFs**: 76 (7 root free-joint + 69 actuated joints)
- **Bodies**: 24 (Pelvis, legs, torso, arms, head)
- **Actuators**: 69 (all joints except root translation)
- **Physics**: Full contact model with friction

### Implementation Details
1. **Coordinate System**: Y-up (SMPL) ↔ Z-up (MuJoCo)
2. **Joint Representation**: SMPL axis-angle ↔ MuJoCo qpos
3. **Motion Format**: motion_135 (T, 135) = [transl(3) + 22 × rot6d(6)]
4. **Control**: PD controllers with learned gains from ONNX policy

### Workflow
```
motion_135 NPZ
  ↓ [decode rot6d → axis-angle]
SMPL pose + translation (Y-up)
  ↓ [coordinate transform Y-up → Z-up]
SMPL pose (Z-up)
  ↓ [convert to MuJoCo qpos]
qpos (76-dim)
  ↓ [run PD tracking in MuJoCo]
simulated qpos
  ↓ [convert back to SMPL + Y-up]
physics-corrected motion_135
```

### Key Functions (Available for Reuse)
- `decode_motion_135()` — motion_135 → SMPL
- `rot6d_to_rotmat()` — rot6d encoding
- `yup_to_zup()` / `zup_to_yup()` — coordinate transforms
- `smpl_to_qpos()` / `qpos_to_smpl()` — MuJoCo conversion
- `load_mujoco_model()` — Model loading with physics config
- `run_physics_sim()` — Physics simulation loop

### Strengths
✅ No IsaacGym dependency  
✅ Full physics simulation available  
✅ Reference code for RL environment  
✅ Coordinate transformations pre-built  
✅ Debugging infrastructure comprehensive  

### Limitations
⚠️ PD-tracking only (deterministic control)  
⚠️ No learned policy yet  
⚠️ Resets required between motions  

---

## 3. ProtoMotions Framework (✅ FULLY AVAILABLE)

### Status
Complete RL training framework with MuJoCo backend support

### Location
```
ref_repo/ProtoMotions/
├── protomotions/
│   ├── simulator/
│   │   ├── base_simulator/
│   │   │   ├── simulator.py        ← Base environment class
│   │   │   └── simulator_state.py  ← State representation
│   │   ├── mujoco/
│   │   │   └── simulator.py        ← MuJoCo-specific implementation
│   │   ├── newton/
│   │   │   └── simulator.py        ← Newton physics backend
│   │   ├── isaacgym/
│   │   │   └── simulator.py        ← IsaacGym backend (not needed)
│   │   └── factory.py              ← Simulator factory
│   ├── learning/
│   │   ├── amp_agent.py            ← AMP (Adversarial Motion Priors)
│   │   ├── im_amp.py               ← Imitation + AMP combo
│   │   └── network_loader.py       ← Policy network loading
│   ├── utils/
│   │   ├── motion_lib_base.py      ← Motion library abstraction
│   │   ├── config.py               ← Configuration system
│   │   └── parse_task.py           ← Task parsing
│   ├── env/
│   │   ├── tasks/
│   │   │   ├── base_task.py        ← Base task class
│   │   │   ├── humanoid_im.py      ← Imitation task
│   │   │   ├── humanoid_amp.py     ← AMP task
│   │   │   ├── humanoid_im_amp.py  ← Combined task
│   │   │   └── humanoid_im_mcp.py  ← MCP (Multi-Capability Policy)
│   │   └── util/
│   │       └── gym_util.py         ← Gym utilities
│   └── data/
│       ├── assets/mjcf/
│       │   └── smpl_humanoid.xml  ← SMPL model
│       └── pretrained_models/
│           ├── motion_tracker/
│           │   └── smpl/
│           │       └── compiled_models/
│           │           ├── unified_pipeline.onnx
│           │           └── unified_pipeline.yaml
└── docs/
    └── retargeting.md              ← Retargeting guide
```

### Key Features
1. **Multiple Physics Backends**
   - ✅ MuJoCo (native, no IsaacGym)
   - ✅ Newton
   - ⚠️ IsaacGym (optional, not needed)

2. **RL Algorithms**
   - **AMP** (Adversarial Motion Priors) — discriminator-based imitation
   - **PPO** — policy gradient learning
   - **IM-AMP** — imitation + adversarial combo
   - **MCP** — multi-task learning with capacity allocation

3. **Tasks**
   - `humanoid_im.py` — Pure motion imitation
   - `humanoid_amp.py` — Adversarial motion priors
   - `humanoid_im_amp.py` — Combined imitation + AMP
   - `humanoid_im_mcp.py` — Multi-capability policy

### How to Use for RL Training (Without IsaacGym)

```python
from protomotions.simulator.factory import SimulatorFactory
from protomotions.env.tasks.humanoid_im import HumanoidImitation
import hydra

# 1. Load config with MuJoCo backend
cfg = hydra.compose(config_name="humanoid_im_task", 
                    overrides=["env.sim_backend=mujoco"])

# 2. Create simulator (MuJoCo, no IsaacGym)
simulator = SimulatorFactory.create(cfg.env.sim_backend, cfg.env)

# 3. Create task/environment
task = HumanoidImitation(cfg, simulator)

# 4. Train with PPO
from stable_baselines3 import PPO
policy = PPO("MlpPolicy", task, ...)
policy.learn(total_timesteps=1_000_000)
```

### Strengths
✅ Professional-grade RL framework  
✅ MuJoCo backend fully implemented  
✅ No IsaacGym dependency needed  
✅ Extensive documentation  
✅ Tested on AMASS dataset (11,313 sequences)  
✅ Pre-trained policies available  
✅ Active development (August 2025 updates)  

### Limitations
⚠️ Large codebase (learning curve)  
⚠️ Requires config system understanding (Hydra)  
⚠️ Custom agent implementations (not stable-baselines3)  

### Reference
[ProtoMotions Repository](ref_repo/ProtoMotions/FINAL_SUMMARY.md)  
Paper: "Perpetual Humanoid Control for Real-time Simulated Avatars" (ICCV 2023)

---

## 4. Custom Gymnasium + Stable-Baselines3 (🔶 NEEDS BUILD)

### Status
Framework not yet implemented, but **fully feasible**

### Assessment Document
```
MuJoCo_Gymnasium_SB3_Feasibility_Assessment.md
Conclusion: ✅ FEASIBLE — estimated 4-7 days implementation
```

### Why This Approach?
✅ Modern standard RL environment API  
✅ Easier integration with stable-baselines3  
✅ Less code duplication than ProtoMotions  
✅ Standard for ML community  

### Implementation Plan

**Phase 1: Create Gymnasium Wrapper (1-2 days)**
```python
import gymnasium
import mujoco
import numpy as np

class SMPLHumanoidEnv(gymnasium.Env):
    """SMPL humanoid environment for motion imitation RL."""
    
    def __init__(self, mjcf_path: str, motion_lib=None):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        # Observation: [qpos (76) + qvel (75)]
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(151,), dtype=np.float32
        )
        
        # Action: motor torques for 75 actuators
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(75,), dtype=np.float32
        )
        
        self.motion_lib = motion_lib  # Reference motions
        self.current_ref_idx = 0
        self.step_count = 0
    
    def step(self, action: np.ndarray) -> tuple:
        # Apply PD control with policy output
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        
        # Compute reward (motion tracking)
        ref_qpos = self.motion_lib[self.current_ref_idx, self.step_count]
        tracking_loss = np.mean((self.data.qpos - ref_qpos) ** 2)
        energy_cost = 0.001 * np.mean(self.data.ctrl ** 2)
        reward = -tracking_loss - energy_cost
        
        # Check termination (fall, timeout)
        terminated = self._is_fallen()
        truncated = self.step_count >= len(self.motion_lib[0])
        self.step_count += 1
        
        return obs, reward, terminated, truncated, {}
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_ref_idx = self._np_random.integers(0, len(self.motion_lib))
        self.step_count = 0
        self.data.qpos[:] = self.motion_lib[self.current_ref_idx, 0]
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)
        return obs, {}
    
    def _is_fallen(self) -> bool:
        return self.data.qpos[2] < 0.3  # Root height < 0.3m
```

**Phase 2: Train with Stable-Baselines3 (1 day)**
```python
from stable_baselines3 import PPO

env = SMPLHumanoidEnv(
    mjcf_path="ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml",
    motion_lib=load_motion_library("output/embodied_t2m_v4/data/npz")
)

model = PPO("MlpPolicy", env, learning_rate=3e-4, verbose=1)
model.learn(total_timesteps=10_000_000)
model.save("policies/smpl_humanoid_sb3")
```

### Strengths
✅ Minimal, clean implementation  
✅ Compatible with all gymnasium tools  
✅ Works with stable-baselines3  
✅ Easy to debug  
✅ Standard API for community  

### Challenges
⚠️ Reward function engineering  
⚠️ Observation representation choices  
⚠️ Curriculum learning (if needed)  
⚠️ Hyperparameter tuning  

---

## 5. rsl_rl Framework (✅ AVAILABLE)

### Status
Tested PPO implementation available in multiple ref_repo projects

### Locations
```
ref_repo/OmniH2O/rsl_rl/
ref_repo/HumanPlus/HST/rsl_rl/
ref_repo/VideoMimic/simulation/videomimic_rl/rsl_rl/
```

### What is rsl_rl?
- Professional-grade RL training library
- PPO with optimized vectorized training
- Parallel environment support
- Good for legged locomotion tasks

### Status
⚠️ Optimized for **legged robots** (quadrupeds, bipeds with legs)  
⚠️ Less documentation for SMPL humanoid use  
⚠️ Requires careful setup for humanoid imitation  

---

## 6. MJX (MuJoCo XLA/JAX) - Future Option

### Status
Not found in current codebase, but mentioned in feasibility docs

### Potential Path
- MJX provides **GPU-accelerated MuJoCo** via JAX
- Faster than CPU MuJoCo for large batches
- Requires: `pip install mujoco-mjx`
- Can work with Flax/JAX-based RL (not stable-baselines3)

### Assessment
🔶 **Future option** — not critical for now  
CPU MuJoCo sufficient for single-policy training  
MJX better for vectorized training (100+ parallel envs)  

---

## Recommendation: What to Do

### Immediate (What's Ready Now)

1. **Use Pre-trained ONNX Policy** (5 minutes)
   ```bash
   python3 scripts/embodied/run_smpl_rl_tracker.py \
       --npz-file output/embodied_t2m_v4/data/npz/walk.npz \
       --output-dir output/physics_corrected
   ```
   ✅ Get physics-corrected motion instantly  
   ✅ No training, no IsaacGym  

2. **Study ProtoMotions** (1-2 days)
   - Review `ref_repo/ProtoMotions/` for RL training patterns
   - Understand MuJoCo simulator integration
   - Optional: Use it directly for training

### Short-term (What to Build)

3. **Create Gymnasium Env Wrapper** (2-3 days)
   - Wrap SMPL humanoid model as gymnasium.Env
   - Integrate with stable-baselines3
   - Simple PPO training loop

### Long-term (Optional Optimization)

4. **Vectorize with MJX** (1-2 weeks)
   - Implement MuJoCo XLA backend
   - Train 100+ parallel environments
   - Significant speedup for large datasets

---

## Comparison Table

| Option | Status | Training? | IsaacGym? | MuJoCo? | Effort | Quality |
|--------|--------|-----------|-----------|---------|--------|---------|
| 1. ONNX Inference | ✅ Ready | ❌ No | ❌ No | ✅ Yes | 0 min | ★★★★★ |
| 2. PD-Tracking | ✅ Ready | ❌ No | ❌ No | ✅ Yes | 0 min | ★★★★ |
| 3. ProtoMotions | ✅ Available | ✅ Yes | ❌ No | ✅ Yes | 2-3d | ★★★★★ |
| 4. Gymnasium+SB3 | 🔶 To build | ✅ Yes | ❌ No | ✅ Yes | 4-7d | ★★★★ |
| 5. rsl_rl | ✅ Available | ✅ Yes | ❌ No | ✅ Yes | 3-5d | ★★★★ |
| 6. MJX | 🔶 Future | ✅ Yes | ❌ No | ✅ Yes | 1-2w | ★★★★★ |

---

## Installation Checklist

```bash
# Already installed
✅ mujoco 3.1.6
✅ SMPL humanoid model (smpl_humanoid.xml)

# For gymnasium + stable-baselines3 approach
pip install gymnasium stable-baselines3

# Optional: rendering
pip install gymnasium[classic_control] pygame

# For ProtoMotions (already available)
# Already in ref_repo/ProtoMotions

# Verify
python3 -c "import mujoco; import gymnasium; import stable_baselines3; print('✅ All set!')"
```

---

## References

- ProtoMotions: `ref_repo/ProtoMotions/README.md`
- PHC Paper: "Perpetual Humanoid Control for Real-time Simulated Avatars" (ICCV 2023)
- Scripts: `scripts/embodied/run_smpl_rl_tracker.py`
- SMPL Model: `ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml`
- Feasibility: `MuJoCo_Gymnasium_SB3_Feasibility_Assessment.md`

