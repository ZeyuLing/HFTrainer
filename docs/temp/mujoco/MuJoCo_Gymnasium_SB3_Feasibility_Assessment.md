# MuJoCo + Gymnasium + Stable-Baselines3 Feasibility Assessment
**Date**: 2026-05-18  
**Project**: HyMotion Humanoid RL Training

---

## Executive Summary

✅ **FEASIBLE** — MuJoCo is available and the SMPL humanoid model exists, but **gymnasium and stable-baselines3 must be installed**. The codebase already uses MuJoCo for physics simulation (PD-tracking and RL inference), so the foundation is solid.

---

## Detailed Findings

### 1. Package Installation Status

| Package | Status | Version | Notes |
|---------|--------|---------|-------|
| **mujoco** | ✅ Installed | 3.1.6 | Physics simulator core |
| **gymnasium** | ❌ NOT installed | — | Required for modern RL env API |
| **gym** (old) | ❌ NOT installed | — | Legacy, avoid this |
| **stable-baselines3** | ❌ NOT installed | — | Required for RL algorithms |

**Action Required**: Install these packages:
```bash
pip install gymnasium stable-baselines3
# Optional: for rendering/testing
pip install gymnasium[classic_control]
```

---

### 2. MuJoCo Model Inspection

✅ **File exists** and is fully functional:  
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
  ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml
```

**Model Structure** (first 80 lines):
- **Root joint**: Pelvis with free joint (6 DOF: 3D translation + 3D rotation)
- **Legs**: 3 DOF per hip (x/y/z rotations), 3 DOF per knee (x/y/z), 3 DOF per ankle (x/y/z)
  - Left leg: L_Hip (3×xyz) → L_Knee (3×xyz) → L_Ankle (3×xyz) → L_Toe (3×xyz)
  - Right leg: R_Hip (3×xyz) → R_Knee (3×xyz) → R_Ankle (3×xyz) → R_Toe (3×xyz)
- **Torso chain**: Torso (3×xyz) → Spine (3×xyz) → Chest (3×xyz) → Neck → Head
- **Shoulders**: L/R Thorax → L/R Shoulder → L/R Elbow → L/R Wrist → L/R Hand

**Total DOFs**:
- Pelvis free joint: 7 (3 xyz pos + 4 quat rot)
- Limbs: 24 bodies × 3 hinge DOFs = 72
- **Total: 76 DOF** (MuJoCo internal), but **SMPL only uses 72 DOF** (22 joints × 3 rotations, no root position)

**Actuators**: 
- All joints use motor elements with `ctrlrange="-1 1"` (normalized torques)
- Stiffness/damping configured per-body (e.g., hip/knee: 800 stiffness, 80 damping)

---

### 3. Gymnasium-Compatible MuJoCo Environment Search

✅ **Found existing implementations** in the codebase:
- **ProtoMotions** (`ref_repo/ProtoMotions/`) implements a full MuJoCo-based RL framework
  - Uses simulator abstraction with MuJoCo backend support
  - Has PPO/AMP/ASE agents implemented
  - Supports both SMPL humanoid and robot avatars
  - **Does NOT use gymnasium** — uses custom `Simulator` abstraction

- **Embodied scripts** in `scripts/embodied/` (26 files):
  - `run_smpl_physics_sim.py` — PD-tracking in MuJoCo (no RL)
  - `run_smpl_rl_tracker.py` — RL inference using trained ONNX policy
  - Various diagnostic/debug scripts
  - **None use gymnasium directly** — all use raw MuJoCo Python API

**Conclusion**: No existing gymnasium wrapper found. Will need to create one.

---

### 4. Directory Structure: `scripts/embodied/`

```
scripts/embodied/                                (666 files, 70+ MB)
├── Physics Simulation (PD-tracking)
│   ├── run_smpl_physics_sim.py                ← Main: PD-track motion in MuJoCo
│   ├── run_smpl_rl_tracker.py                 ← RL inference with ONNX policy
│   ├── run_tracker_export.py
│   ├── render_tracker_headless.py
│   └── run_v6_full_regen.sh
│
├── Debugging & Diagnostics (20+ files)
│   ├── debug_sim_stability.py
│   ├── debug_pose_diagnostic.py
│   ├── debug_root_transform.py
│   ├── diag_actuator.py
│   ├── diagnose_oscillation.py
│   ├── debug_transform_comparison.py
│   ├── test_*.py (5+ unit tests)
│   └── ...
│
├── Data Pipeline
│   ├── batch_npz_to_smpl_joints.py
│   ├── batch_npz_to_smpl_mesh_json.py
│   ├── batch_pipeline_to_web.py
│   ├── batch_retarget_parallel.py
│   ├── batch_t2m_to_embodied.py
│   ├── gmr_retarget_headless.py
│   ├── gmr_to_protomotions.py
│   └── ...
│
├── Conversion Utilities
│   ├── motion135_to_pyroki_keypoints.py
│   ├── motion135_to_smplx.py
│   ├── hymotion_to_smplx.py
│   └── ...
│
└── Documentation
    ├── FORMAT_SPECIFICATION.md
    ├── VERIFICATION_SUMMARY.txt
    ├── detailed_code_analysis.md
    ├── hymotion_verification_report.md
    └── ...
```

**Key scripts for RL**:
- `run_smpl_rl_tracker.py` (47 KB) — uses ONNX policy, closes-loop MuJoCo sim
- `run_smpl_physics_sim.py` (47 KB) — PD-tracking baseline

---

### 5. Stable-Baselines3 Availability

❌ **NOT installed** and no reference implementations found in codebase.

However:
- **ProtoMotions** has custom PPO/AMP/ASE agents (not stable-baselines3)
- Could integrate stable-baselines3 if needed, but would need to either:
  1. Write gymnasium wrapper for SMPL humanoid
  2. Or wrap stable-baselines3 to work with existing ProtoMotions infrastructure

---

## Technical Feasibility Analysis

### ✅ Strengths

1. **MuJoCo 3.1.6 is modern and stable**
   - Full SMPL humanoid model available with proper DOF structure
   - Python API well-documented, easy to wrap

2. **Existing SMPL humanoid physics code**
   - `run_smpl_physics_sim.py` shows full PD-tracking pipeline
   - Can reuse coordinate transformations (Y-up ↔ Z-up, rot6d ↔ axis-angle)
   - Existing body mapping: SMPL ↔ MuJoCo names

3. **ProtoMotions framework as reference**
   - Complete RL training pipeline already works with MuJoCo backend
   - Can study `protomotions/simulator/mujoco_simulator.py` for implementation patterns

4. **Rich debugging infrastructure**
   - 20+ diagnostic scripts validate physics correctness
   - Ensures simulation won't have subtle bugs

### ⚠️ Challenges

1. **gymnasium wrapper must be created**
   - Need to implement `gymnasium.Env` subclass
   - Must handle:
     - State observation space (76 DOF qpos + qvel)
     - Action space (75 DOF motor controls, excluding root)
     - Reward design (tracking, regularization, etc.)
     - Termination conditions (fall, timeout, etc.)
     - Reset (random initialization or motion sequence)

2. **Observation representation choice**
   - Option A: Raw qpos/qvel (76 dims) — simple but less informative
   - Option B: Computed obs from existing code (e.g., joint angles, velocities) — richer but requires engineering
   - Option C: Mix of both — recommended

3. **Action space design**
   - MuJoCo qpos has 76 DOF, but only 75 actuators (root not actuated)
   - Must exclude root position/rotation from action space
   - How to handle root motion? (RL-controlled or environment-provided?)

4. **Reward function engineering**
   - Motion tracking: minimize ||simulated_pose - reference_pose||
   - Energy regularization: minimize torque magnitudes
   - Stability: encourage ground contact for feet
   - Style matching: optional, complex

5. **Integration decision**
   - **Standalone gymnasium env**: Easier integration with stable-baselines3, but duplicates ProtoMotions logic
   - **Wrapper around ProtoMotions**: Reuses tested code, but adds abstraction layer
   - **Extend ProtoMotions to output gymnasium-compatible interface**: Best long-term, most work

---

## Recommended Implementation Path

### Phase 1: Create Simple Gymnasium Wrapper (1-2 days)

**Goal**: Get basic RL training working with stable-baselines3

```python
class SMPLHumanoidEnv(gymnasium.Env):
    def __init__(self, mjcf_path, motion_file=None, use_rl_reward=True):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        # Observation space: [qpos (76) + qvel (75)]
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(151,), dtype=np.float32
        )
        
        # Action space: [motor torques for 75 actuators]
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(75,), dtype=np.float32
        )
    
    def step(self, action: np.ndarray):
        # Set control, step physics, compute obs/reward/done
        pass
    
    def reset(self, seed=None):
        # Random or motion-conditioned initialization
        pass
    
    def render(self, mode="human"):
        # Optional: use mujoco.viewer
        pass
```

### Phase 2: Integrate with Stable-Baselines3 (1 day)

```python
from stable_baselines3 import PPO

env = SMPLHumanoidEnv(
    mjcf_path="ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml",
    motion_file="output/embodied_t2m_v4/data/npz/walk_forward.npz"
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
```

### Phase 3: Enhance Reward Design (2-3 days)

- Reuse reward components from `scripts/embodied/run_smpl_physics_sim.py`
- Add FK-based tracking loss
- Add joint angle regularization
- Add foot contact encouragement

### Phase 4: Validate Against ProtoMotions (1-2 days)

- Train both implementations on same motion
- Compare convergence speed, final policy performance
- Debug any physics discrepancies

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **Root DOF control ambiguity** | High | Decide early: RL-controlled root vs. environment-provided trajectory |
| **Observation space design** | Medium | Use existing code as reference; iterate based on policy learning speed |
| **Reward hacking** | Medium | Monitor auxiliary metrics (energy cost, ground contact stability) |
| **Physics instability** | Low | Existing `run_smpl_physics_sim.py` validates this; reuse similar timestep/control scheme |
| **Stable-baselines3 requires gymnasium** | Low | Both are well-maintained; simple integration |

---

## Installation Script

```bash
#!/bin/bash
set -e

# Install gymnasium + stable-baselines3
pip install gymnasium stable-baselines3

# Optional: rendering support
pip install gymnasium[classic_control]
pip install pygame

# Verify installation
python3 -c "import gymnasium; print(f'gymnasium {gymnasium.__version__}')"
python3 -c "import stable_baselines3; print(f'stable-baselines3 {stable_baselines3.__version__}')"
python3 -c "import mujoco; print(f'mujoco {mujoco.__version__}')"

echo "✅ All packages installed successfully!"
```

---

## Conclusion

✅ **Proceed with implementation**. The groundwork is solid:
- MuJoCo 3.1.6 available, SMPL humanoid model exists
- Existing code provides reference implementations
- gymnasium + stable-baselines3 are straightforward dependencies
- Estimated effort: **4-7 days** for basic training loop, plus optimization tuning

**Next steps**:
1. Install gymnasium + stable-baselines3
2. Create `SMPLHumanoidEnv` wrapper
3. Train simple PPO policy
4. Compare with ProtoMotions reference
