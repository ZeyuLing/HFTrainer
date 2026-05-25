# Physics-SOAR Quick Start Implementation Guide

**Status:** Ready to implement  
**Expected Timeline:** 2-3 weeks  
**Difficulty:** Medium (leverages existing SOAR framework)

---

## 🎯 The Big Picture

You now have **two critical findings**:

### Finding 1: SOAR Framework Exists and Works
- **SOAR** (published 2026-04) is a post-training method for flow matching models
- It corrects "exposure bias" — the mismatch between training (ground-truth states) and inference (model-predicted states)
- **HY-SOAR** is open-source implementation for SD3.5-Medium (your HYMotion uses the same rectified flow framework)
- **Result:** +11% GenEval improvement on SD3.5 without any reward model

### Finding 2: Physics Can Replace the Correction Target
- Standard SOAR: `v_corr = (x_prime - x0_clean) / t_prime`
- **Physics-SOAR:** `v_corr = (x_prime - x_phys_target) / t_prime`
- The physics target `x_phys_target` is computed from MuJoCo evaluation
- **Result:** Dense physics feedback directly in the training loop, no gradients through physics

### Combined Result
**Physics-SOAR** = SOAR framework + physics-guided correction targets
- Timeline: **2-3 weeks** (shorter than policy gradient or DPO approaches)
- Architecture: **Post-training only** (no changes to existing model)
- Compatibility: **100%** with HYMotion M2M's VACE conditioning and _man variant

---

## 📋 Week-by-Week Implementation Plan

### **Week 1: Physics Evaluator + Basic Integration**

#### Day 1-2: Set Up Physics Evaluator
```bash
# File: hftrainer/models/motion/physics_evaluator.py

class FastPhysicsEvaluator:
    """Evaluate motion quality in MuJoCo without gradients"""
    
    def __init__(self, smpl_model_path, mjcf_path):
        # Load SMPL model and MuJoCo environment
        # Prepare 32+ parallel MuJoCo instances for batch eval
        pass
    
    def evaluate_batch(self, motions: Tensor) -> Dict:
        """
        Input: (B, T, 135) - batch of motions
        Output: {
            'collision_penalty': (B,),  # 0-1, lower is better
            'com_stability': (B,),      # 0-1, higher is better  
            'energy_eff': (B,),         # 0-1, higher is better
            'smoothness': (B,),         # 0-1, higher is better
            'overall_score': (B,)       # 0-1, weighted average
        }
        """
        pass
    
    def suggest_correction(self, motion: Tensor) -> Tensor:
        """
        Input: (T, 135) motion with physics issues
        Output: (T, 135) corrected motion
        Options:
          - Smooth via low-pass filter
          - Re-simulate with foot contact constraints
          - IK-correct to remove collisions
        """
        pass
```

**Deliverable:** Fast evaluator that processes 32 motions in <1 second

#### Day 3-5: Integrate into Training Loop
```bash
# File: hftrainer/models/motion/trainer_physics_soar.py

# Fork from existing trainer
# Add these functions:

def compute_physics_soar_loss(self, x_t, x0_clean, x1_noise, caption, t):
    """Main SOAR correction loss with physics"""
    
    with torch.no_grad():
        # Standard SOAR rollout
        v_rollout = self.model(x_t, caption, t)
        dt = -1.0 / 50  # 50-step ODE
        x_hat = x_t + dt * v_rollout
        
        loss_corr = 0
        for n in range(6):  # 6 auxiliary points
            t_prime = random.uniform(t, 1.0)
            alpha = (t_prime - t) / (1 - t)
            x_prime = (1-alpha) * x_hat + alpha * x1_noise
            
            # NEW: Physics evaluation
            x0_candidate = self.model.full_denoise(
                x_prime, caption, num_steps=5
            )
            physics_metrics = self.evaluator.evaluate(x0_candidate)
            
            # NEW: Physics-guided correction target
            if physics_metrics['overall_score'] < 0.7:
                x_phys_corrected = self.evaluator.suggest_correction(x0_candidate)
                x_phys_target = 0.7 * x_phys_corrected + 0.3 * x0_clean
            else:
                x_phys_target = x0_clean
            
            # Correction velocity
            v_corr = (x_prime - x_phys_target) / t_prime
            
            # Model prediction on off-trajectory point
            v_off = self.model(x_prime, caption, t_prime)
            loss_corr += F.smooth_l1_loss(v_off, v_corr)
        
        return loss_corr / 6

def train_step(self, batch):
    """Modified training step with Physics-SOAR"""
    x0_clean = batch['motion']
    caption = batch['caption']
    
    # Base loss (unchanged)
    x1_noise = torch.randn_like(x0_clean)
    t = torch.rand(x0_clean.shape[0])
    x_t = (1-t) * x0_clean + t * x1_noise
    v_pred = self.model(x_t, caption, t)
    loss_base = F.smooth_l1_loss(v_pred, x1_noise - x0_clean)
    
    # NEW: Physics-SOAR correction loss
    loss_soar = self.compute_physics_soar_loss(
        x_t, x0_clean, x1_noise, caption, t
    )
    
    # Combined loss
    loss_total = loss_base + self.lambda_soar * loss_soar
    
    # Backprop and optimize
    self.optimizer.zero_grad()
    loss_total.backward()
    self.optimizer.step()
    
    return {
        'loss_base': loss_base.item(),
        'loss_soar': loss_soar.item(),
        'loss_total': loss_total.item(),
    }
```

**Deliverable:** Complete training loop with Physics-SOAR integrated

#### Day 6-7: Testing and Debugging
- Run on small batch (batch_size=2) to check for errors
- Monitor loss curves: L_base should be stable, L_soar should decrease
- Check physics evaluator speed: target < 50ms per motion

**Metrics to monitor:**
```python
# In your training logger:
- loss_base (should be similar to baseline)
- loss_soar (should decrease over time)
- physics_score_avg (overall score of generated motions)
- collision_penalty_avg (should decrease)
- com_stability_avg (should increase)
```

---

### **Week 2: Hyperparameter Tuning + Validation**

#### Day 1-2: Start with Conservative Hyperparameters
```python
# Recommended starting values:
physics_soar_config = {
    'lambda_soar': 0.1,        # Start small, increase if needed
    'n_auxiliary_points': 4,   # Balance speed vs density
    'physics_threshold': 0.7,  # Trigger correction when score < 70%
    'blend_ratio': 0.3,        # 0.3 * corrected + 0.7 * clean
    'batch_physics': 16,       # Physics eval batch size
    'eval_frequency': 0.5,     # Eval on 50% of auxiliary points for speed
}

# Monitor during training:
# 1. Check loss_soar is meaningful (not NaN)
# 2. Ensure physics_score improves over time
# 3. No VRAM spikes from physics evaluation
```

#### Day 3-5: Ablation Studies
Run these experiments to find optimal hyperparameters:

```bash
# Exp 1: lambda_soar impact
for lambda_soar in [0.05, 0.1, 0.2, 0.5]:
    train(lambda_soar=lambda_soar, n_steps=1000)
    → Best lambda_soar?

# Exp 2: n_auxiliary_points impact  
for n in [2, 4, 6, 8]:
    train(n_auxiliary=n, n_steps=1000)
    → Best speed/quality tradeoff?

# Exp 3: Physics threshold impact
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    train(physics_threshold=threshold, n_steps=1000)
    → Best for physics improvement?

# Exp 4: Blend ratio impact
for ratio in [0.1, 0.3, 0.5, 0.7]:
    train(blend_ratio=ratio, n_steps=1000)
    → Best balance?
```

#### Day 6-7: Full Validation Run
```bash
# Train on full training data for 5-10K steps
python train_physics_soar.py \
    --config config.yaml \
    --lambda_soar 0.1 \
    --n_auxiliary 4 \
    --max_steps 10000 \
    --save_interval 1000

# Generate validation outputs
python scripts/generate_and_evaluate.py \
    --checkpoint ./checkpoints/physics_soar_step_10000 \
    --output_dir ./results/physics_soar \
    --num_samples 100
```

**Deliverables:**
- Hyperparameter ranges documented
- Validation metrics showing improvement over baseline
- Training curves (loss_base, loss_soar, physics_score)

---

### **Week 3: Final Benchmarking + Documentation**

#### Day 1-3: Comprehensive Evaluation
```bash
# Compare: Baseline vs Physics-SOAR

metrics_to_track = {
    'motion_text_alignment': compute_clip_score,  # How well does motion match text?
    'physics_quality': evaluate_physics,           # MuJoCo metrics
    'temporal_smoothness': compute_smoothness,     # Jitter/stability
    'diversity': compute_fid,                      # Is there variety?
    'human_preference': run_user_study,            # Does it look better?
}

# Generate comparison tables:
# | Metric | Baseline | Physics-SOAR | Improvement |
```

#### Day 4-7: Ablation Studies + Documentation

```bash
# Ablation: With/Without Physics
- Pure SOAR (no physics in correction target)  
- Physics-SOAR (full implementation)
- → Shows physics contribution

# Ablation: Physics Metrics
- Only collision penalty
- Only com stability
- Only energy efficiency
- All metrics (balanced)
- → Shows which metrics matter most

# Generate final report with:
- Summary of findings
- Hyperparameter recommendations
- Performance improvements
- Computational overhead analysis
- Next steps (DPO, fine-tuning, etc.)
```

---

## 🔧 Code Structure

```
hftrainer/
├── models/
│   └── motion/
│       ├── physics_evaluator.py          # NEW: Physics evaluation
│       ├── trainer_physics_soar.py       # NEW: SOAR training loop
│       └── (existing model files)
├── scripts/
│   ├── train_physics_soar.py             # NEW: Training script
│   ├── generate_and_evaluate.py          # NEW: Evaluation script
│   └── ablation_study.py                 # NEW: Run ablations
└── docs/
    └── PHYSICS_SOAR_QUICK_START.md       # THIS FILE
```

---

## ⚡ Key Code Templates

### Template 1: Physics Evaluator Stub
```python
# hftrainer/models/motion/physics_evaluator.py

import torch
import numpy as np
from typing import Dict, Tuple

class PhysicsEvaluator:
    def __init__(self, smpl_model_path: str, mjcf_path: str):
        """
        Initialize physics evaluator.
        
        Args:
            smpl_model_path: Path to SMPL model (e.g., ./data/smpl_models/)
            mjcf_path: Path to MuJoCo XML config
        """
        self.smpl_model = self._load_smpl(smpl_model_path)
        self.mujoco_env = self._load_mujoco(mjcf_path)
    
    def evaluate(self, motion: torch.Tensor) -> Dict:
        """
        Evaluate motion quality.
        
        Args:
            motion: (T, 135) SMPL motion parameters
        
        Returns:
            Dict with keys: collision_penalty, com_stability, energy_eff, 
                           smoothness, overall_score
        """
        # TODO: Implement metrics
        raise NotImplementedError
    
    def suggest_correction(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Suggest physically plausible correction.
        
        Args:
            motion: (T, 135) motion with physics issues
        
        Returns:
            (T, 135) corrected motion
        """
        # TODO: Implement correction
        raise NotImplementedError
    
    def _load_smpl(self, path: str):
        # Load SMPL model
        pass
    
    def _load_mujoco(self, path: str):
        # Load MuJoCo environment
        pass
```

### Template 2: Training Loop Stub
```python
# hftrainer/models/motion/trainer_physics_soar.py

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

class TrainerPhysicsSOAR:
    def __init__(self, model, evaluator, config):
        self.model = model
        self.evaluator = evaluator
        self.lambda_soar = config.lambda_soar
        self.n_auxiliary = config.n_auxiliary_points
        self.physics_threshold = config.physics_threshold
        self.blend_ratio = config.blend_ratio
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    
    def compute_physics_soar_loss(
        self,
        x_t: torch.Tensor,
        x0_clean: torch.Tensor,
        x1_noise: torch.Tensor,
        caption: str,
        t: torch.Tensor
    ) -> torch.Tensor:
        """Compute Physics-SOAR correction loss."""
        
        with torch.no_grad():
            # SOAR rollout step
            v_rollout = self.model(x_t, caption, t)
            dt = -1.0 / 50  # 50-step ODE
            x_hat = x_t + dt * v_rollout
            
            loss_corr = 0
            for n in range(self.n_auxiliary):
                # Re-noise to auxiliary point
                t_prime = torch.rand_like(t) * (1 - t) + t
                alpha = (t_prime - t) / (1 - t + 1e-8)
                x_prime = (1-alpha).view(-1, 1, 1) * x_hat + \
                          alpha.view(-1, 1, 1) * x1_noise
                
                # Physics evaluation
                x0_candidate = self.model.full_denoise(
                    x_prime, caption, num_steps=5
                )
                physics_metrics = self.evaluator.evaluate(x0_candidate)
                
                # Physics-guided correction target
                is_poor_quality = physics_metrics['overall_score'] < self.physics_threshold
                if is_poor_quality:
                    x_phys_corrected = self.evaluator.suggest_correction(x0_candidate)
                    x_phys_target = (
                        self.blend_ratio * x_phys_corrected +
                        (1 - self.blend_ratio) * x0_clean
                    )
                else:
                    x_phys_target = x0_clean
                
                # Correction velocity target
                v_corr = (x_prime - x_phys_target) / (t_prime.view(-1, 1, 1) + 1e-8)
                
                # Model prediction
                v_off = self.model(x_prime, caption, t_prime)
                
                # Loss
                loss_corr += F.smooth_l1_loss(v_off, v_corr)
        
        return loss_corr / self.n_auxiliary
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training iteration."""
        
        x0_clean = batch['motion']
        caption = batch['caption']
        
        # Base loss
        x1_noise = torch.randn_like(x0_clean)
        t = torch.rand(x0_clean.shape[0], 1, 1)
        x_t = (1-t) * x0_clean + t * x1_noise
        
        v_pred = self.model(x_t, caption, t)
        loss_base = F.smooth_l1_loss(v_pred, x1_noise - x0_clean)
        
        # Physics-SOAR loss
        loss_soar = self.compute_physics_soar_loss(
            x_t, x0_clean, x1_noise, caption, t.squeeze()
        )
        
        # Combined loss
        loss_total = loss_base + self.lambda_soar * loss_soar
        
        # Optimization
        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss_base': loss_base.item(),
            'loss_soar': loss_soar.item(),
            'loss_total': loss_total.item(),
            'physics_score': physics_metrics['overall_score'].mean().item(),
        }
```

---

## 📊 Expected Results

### After Week 1 (Integration Complete):
- ✅ Physics evaluator loads without errors
- ✅ Training loop runs with Physics-SOAR loss
- ✅ loss_soar decreases over time
- ✅ No VRAM issues

### After Week 2 (Tuning Complete):
- ✅ Optimal hyperparameters identified
- ✅ Physics-SOAR baseline established
- ✅ Ablations show which metrics matter
- ✅ Comparison metrics ready

### After Week 3 (Benchmarking Complete):
- ✅ Quantitative improvement over baseline
- ✅ Physics quality metrics improving
- ✅ Qualitative results (video comparisons)
- ✅ Ready for deployment or further refinement

---

## 🚨 Common Issues & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| Physics eval too slow (>100ms) | Not batching MuJoCo calls | Parallelize via ProcessPoolExecutor |
| loss_soar is NaN | Physics correction target overflow | Add clipping: `torch.clamp(v_corr, -1, 1)` |
| Physics metrics plateau | Threshold too high | Lower `physics_threshold` from 0.7 to 0.5 |
| VRAM spike during eval | Loading all motions at once | Use `batch_physics=8` instead of 32 |
| Model doesn't improve | lambda_soar too high | Start with 0.05 instead of 0.1 |

---

## 📚 Reference Documents

| Document | Purpose |
|----------|---------|
| `SOAR_PHYSICS_INTEGRATION_ANALYSIS.md` | Full technical deep-dive |
| `physics_gradients_RESEARCH.md` | Original physics+gradients research |
| `IMPLEMENTATION_ROADMAP.md` | Phase 1-3 overall strategy |
| `ref_repo/SOAR/CLAUDE.md` | SOAR framework details |
| `ref_repo/HY-SOAR/README.md` | Open-source SOAR implementation |

---

## ✅ Checklist for Launch

- [ ] Physics evaluator skeleton created (`physics_evaluator.py`)
- [ ] Training loop skeleton created (`trainer_physics_soar.py`)
- [ ] Training script created (`train_physics_soar.py`)
- [ ] Evaluation script created (`generate_and_evaluate.py`)
- [ ] First training run completes without errors
- [ ] Metrics logged (loss_base, loss_soar, physics_score)
- [ ] Hyperparameter sweep (Week 2) starts
- [ ] Ablation study plan documented
- [ ] Final benchmarking scheduled

---

**Status:** ✅ Ready to implement  
**Next Step:** Create `physics_evaluator.py` stub and run first training iteration

