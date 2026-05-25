# Reference Code Locations for Physics-Feedback Motion Generation

## 1. SOAR Implementation References

### SOAR Algorithm Documentation
- **File**: `ref_repo/SOAR/CLAUDE.md`
- **Sections**:
  - Lines 31-77: SOAR Algorithm pseudocode (Algorithm 1)
  - Lines 116-203: Adaptation to M2M v2 (exact pseudocode for M2M integration)
  - Lines 231-246: VACE context handling (how to preserve known regions)
  - Lines 253-261: Compute overhead analysis (2-2.5x vs SFT)

### SOAR Implementation in HY-SOAR (Reference)
- **Main Trainer**: `ref_repo/HY-SOAR/sora/train_soar_sd3_5m.py`
- **Reward Functions**: `ref_repo/HY-SOAR/sora/flow_grpo/rewards.py`
  - Lines 34-48: aesthetic_score pattern
  - Lines 51-63: clip_score pattern
  - Lines 271-322: multi_score aggregation (can use for multi-physics)
- **Reward Tracking**: `ref_repo/HY-SOAR/sora/flow_grpo/stat_tracking.py`

**Why look here**: To understand how to modify trainer loop for SOAR, especially:
- Stop-gradient wrapper for rollout
- Loss combination (L_base + λ * L_corr)
- Logging and checkpointing

---

## 2. M2M Architecture & Conditioning

### M2M CLAUDE.md (Full Reference)
- **File**: `hftrainer/models/motion/CLAUDE.md`
- **Key Sections**:
  - Lines 1-80: Critical padding constraints (MUST READ)
  - Lines 196-246: VACE conditioning mechanism
  - Lines 327-354: Training data flow (crucial for where to inject SOAR)
  - Lines 358-446: Padding & sequence-length convention
  - Lines 463-543: Known-region conditioning comparison (KIMODO vs UMO vs M2M)

### Flow Matching Backbone
- **File**: `hftrainer/models/motion/hymotion_m2m/` (main model)
- **Key Classes**:
  - `HyMotionMMDiT`: Backbone transformer
  - `HyMotionM2MTrainer`: Current SFT trainer (where SOAR will integrate)
  - Input preparation: `prepare_vace_input()` (critical for SOAR re-noise)

**Why look here**: 
- Understand exact input/output shapes for SOAR
- See how VACE conditioning works (replicate for SOAR's re-noised samples)
- Identify where to add SOAR loss in trainer

---

## 3. Physics-Based Metrics & Constraints

### Foot Skating Detection (Already Implemented)
- **File**: `scripts/eval/eval_m2m_v2_all_tasks.py`
- **Search**: Look for `"foot_skating"` or `"composite_skating_score"`
- **Relevant Code**:
  - Composite score formula (lines ~150-200 estimated)
  - Thresholds for contact/sliding detection
  - Joint indices for foot [L_Ankle, R_Ankle, L_Foot, R_Foot]

### M2M Motion Constraints Doc
- **File**: `docs/temp/EMBODIED_PIPELINE_BUG_ANALYSIS.md`
- **Key Sections**:
  - FK-based ground correction logic
  - Coordinate frame transforms (Y-up SMPL vs Z-up MuJoCo)
  - IK solver constraints (joint limits, feasibility)
  - Body index offset calculations

### M2M Evaluation Metrics
- **File**: `hftrainer/models/motion/hymotion_m2m/CLAUDE.md` (if separate)
- **Or**: Look in evaluation module for:
  - `jitter_pos` metric
  - `boundary_accel_jump` metric
  - Motion quality scores

**Why look here**: 
- Extract physics constraints for reward function
- Understand joint ordering (critical for IK feasibility checks)
- See what metrics to use for Phase 1 evaluation

---

## 4. Robotics RL & Physics Integration

### RL Trainer Pattern
- **File**: `ref_repo/UH-1/rsl_rl/rsl_rl/algorithms/ppo.py`
- **Pattern**: How PPO trainer combines behavior loss + reward signal
- **Can adapt**: For SOAR if we add reward regularization term

### Physics Simulator Integration
- **Location**: `ref_repo/ProtoMotions/`
- **Key Files**:
  - `examples/experiments/mimic/mlp.py` — IK/FK-based imitation
  - `protomotions/simulator/` — simulator backends (Genesis, IsaacGym, MuJoCo)
- **For JAX IK**: Look for `jax-md` or `dm_control` imports

### Reward Shaping Examples
- **File**: `ref_repo/ASAP/humanoidverse/config/rewards/motion_tracking/`
- **Pattern**: YAML configs show multi-term rewards:
  ```yaml
  tracking_error: 1.0
  action_rate: -0.01
  feet_contact: 0.5
  torque: -0.0002
  ```
- **Can adapt**: For physics constraints in motion domain

**Why look here**: 
- Template for multi-objective physics reward
- Understand how to weight different constraints
- Reference implementation of differentiable physics

---

## 5. Motion Dataset & Training Pipeline

### Training Data Processing
- **Dataset**: `train_hymotion_400h.json` (549K samples)
- **Processing Pipeline**: `hftrainer/datasets/motion/`
  - `LoadSMPLX` — raw motion loading
  - `PrepareM2Mv2Condition` — mask sampling
  - `RandomCropPadding` — sequence length handling

### Training Configuration
- **Main Trainer**: `hftrainer/models/motion/hymotion_m2m/trainer.py`
- **Key Functions**:
  - `HyMotionM2MTrainer._prepare_and_forward()` — forward pass logic
  - `HyMotionM2MLoss.compute()` — loss computation
  - Where to add SOAR loss term

**Why look here**: 
- Understand batch processing for SOAR
- See where to insert rollout step
- Identify where mask handling happens (important for preserving known regions)

---

## 6. Checkpoints & Post-Training Targets

### Best M2M Checkpoint
- **Location**: Model zoo (specific path in your setup)
- **Recommended**: `uncond_fm_man_046b` epoch 1000
- **Why**: Best performance, well-converged (safe for SOAR post-training)

### SOAR Post-Training Config (Template)
Would go in `configs/` somewhere:
```yaml
# SOAR post-training on M2M
base_checkpoint: uncond_fm_man_046b_epoch_1000
soar:
  enabled: true
  num_steps: 5000  # 5K-10K range
  lambda: 0.1      # Weight of correction loss
  num_auxiliary_noise_levels: 1  # N in pseudocode
  learning_rate: 2e-5
  mask_aware: true  # Preserve known regions
```

---

## 7. Exact Code Snippets to Reference

### SOAR Re-Noise Loop (from SOAR/CLAUDE.md lines 182-200)
```python
L_corr = 0
for n in range(N):
    t_prime ~ U[t1, 1]                     # Sample auxiliary noise level
    alpha = (t_prime - t1) / (1 - t1)      # Interpolation weight
    z_re = (1-alpha) * x_hat + alpha * x0 # Re-noise with SAME x0
    
    # Mask-aware: keep known regions clean at new t
    if mask_aware_noise:
        z_re = torch.where(keep_mask, x1, z_re)
    
    # Correction target: steer towards x1 (clean)
    v_corr = (z_re - x1) / t_prime
    
    # Forward model on re-noised point
    x_re_input = cat([z_re, vace_ctx])
    v_off = model(x_re_input, text, t_prime)
    L_corr += SmoothL1(v_off, v_corr, generation_mask=src_mask)

L_total = L_base + lambda * L_corr
```

**Lines to copy directly**: 182-203 from SOAR/CLAUDE.md

### VACE Context Preparation (from HyMotion M2M CLAUDE.md lines 227-236)
```python
# Prepare VACE input for both base and SOAR correction
inactive = src_motion * (1 - mask)  # preserved regions
reactive = src_motion * mask        # Completion: 0; Editing: pre-edit values
vace_context = cat([inactive, reactive, mask], dim=-1)

# Final model input (for base and SOAR correction)
x_input = cat([x_t, vace_context], dim=-1)
```

**Lines to copy**: 228-236 from hftrainer/models/motion/CLAUDE.md

### Physics Constraint Score Template (adapt from HY-SOAR)
```python
# Based on ref_repo/HY-SOAR/sora/flow_grpo/rewards.py structure
def physics_constraint_score(device):
    def _fn(motions, prompts, metadata):
        # motions: (B, T, 135)
        scores = []
        for motion in motions:
            # Check constraints
            skating = compute_foot_skating(motion)  # 0-1
            ik_feasibility = check_ik_feasible(motion)  # 0-1
            contact = check_foot_contact(motion)  # 0-1
            
            # Combine (higher = better)
            score = 0.4*contact + 0.3*ik_feasibility + 0.3*(1-skating)
            scores.append(score)
        return np.array(scores), {}
    return _fn
```

---

## 8. Testing & Evaluation Setup

### Single-Sample SOAR Test
- **Test file to create**: `tests/unit/test_soar_single_sample.py`
- **Check**: 
  1. Can we run rollout without error?
  2. Does re-noise produce valid (non-NaN) tensors?
  3. Does correction loss decrease with training steps?

### Multi-Sample Evaluation
- **Reuse**: `scripts/eval/eval_m2m_v2_all_tasks.py`
- **Add metrics**:
  - Before/after SOAR: boundary_accel_jump, foot_skating, jitter_pos
  - Physics-specific: IK feasibility score, contact violation rate

---

## 9. Search Terms for Code Navigation

If you want to find things quickly:

| What | Search Term | File Pattern |
|------|-------------|-------------|
| VACE conditioning | `prepare_vace_input` or `VACE` | `*.py` in hftrainer/models/motion/ |
| Mask handling | `generation_mask` or `src_mask` | `trainer.py`, `condition_sampler*.py` |
| Foot skating | `foot_skating` or `skating_score` | `eval_*.py`, `metrics.py` |
| ODE solver | `odeint` or `flow_matching` | `pipeline.py`, `inference.py` |
| IK/FK | `inverse_kinematics` or `forward_kinematics` | `ref_repo/KIMODO`, `ref_repo/ProtoMotions` |
| Stop-gradient | `torch.no_grad()` or `detach()` | `train_soar*.py` |
| Velocity prediction | `velocity` or `v_pred` | `*_loss.py`, `SOAR` |

---

## 10. Dependencies & Imports You'll Need

```python
# For SOAR
import torch
import torch.nn as nn
from torch.optim import AdamW

# For physics
import numpy as np
# Optional: JAX-based IK (dm_control or jax-md)
# import jax
# import jax.numpy as jnp

# For evaluation
import json
from pathlib import Path
```

---

## Quick File Dependency Map

```
SOAR Implementation:
  ├─ ref_repo/SOAR/CLAUDE.md (theory + pseudocode)
  ├─ ref_repo/HY-SOAR/sora/train_soar_sd3_5m.py (trainer template)
  └─ ref_repo/HY-SOAR/sora/flow_grpo/rewards.py (multi-reward pattern)

M2M Integration:
  ├─ hftrainer/models/motion/hymotion_m2m/trainer.py (where SOAR goes)
  ├─ hftrainer/models/motion/hymotion_m2m/CLAUDE.md (conditioning details)
  ├─ hftrainer/models/motion/CLAUDE.md (full system overview)
  └─ hftrainer/datasets/motion/ (mask sampling + data pipeline)

Physics Constraints:
  ├─ scripts/eval/eval_m2m_v2_all_tasks.py (foot skating metric)
  ├─ docs/temp/EMBODIED_PIPELINE_BUG_ANALYSIS.md (IK/FK details)
  └─ ref_repo/ProtoMotions/ (simulator backends)

Evaluation:
  ├─ scripts/eval/eval_m2m_v2_all_tasks.py (15 tasks)
  └─ scripts/pick_best_height_change.py (custom metrics)
```

---

**Next Step**: Open `ref_repo/SOAR/CLAUDE.md` and read lines 31-77, then lines 116-203. That's your implementation guide.
