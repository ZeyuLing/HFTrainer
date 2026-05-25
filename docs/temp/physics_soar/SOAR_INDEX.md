# SOAR Training System — Complete Index & Navigation Guide

> **Last Updated**: 2026-05-18  
> **Scope**: HyMotion M2M v2 SOAR post-training analysis  
> **Status**: Complete implementation reference with all code sections identified

---

## Document Map

This repository contains **three complementary documents** for SOAR training:

### 1. **SOAR_TRAINING_ANALYSIS.md** (735 lines)
   - **Audience**: Developers implementing or extending SOAR
   - **Content**: 
     - Complete method-by-method breakdown (lines 143-251 of trainer)
     - Detailed gradient flow diagrams
     - Mathematical derivations of correction targets
     - Integration with base M2M trainer
   - **Use when**: You need to understand every step of the algorithm

### 2. **SOAR_QUICK_REFERENCE.txt** (366 lines)
   - **Audience**: Users launching SOAR training and monitoring
   - **Content**:
     - File locations (trainer, configs, docs)
     - Quick method signatures
     - Hyperparameter table (with rationales)
     - Launch commands (copy-paste ready)
     - Monitoring metrics and health checks
     - Troubleshooting guide
   - **Use when**: You're setting up a training run or debugging

### 3. **SOAR_INDEX.md** (this document)
   - **Audience**: Navigation and cross-reference
   - **Content**: Index of all code locations, files, and concepts
   - **Use when**: You need to find where something is implemented

---

## Core Trainer File

### **hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py**

**Total Lines**: 437

**Key Sections**:

| Section | Lines | Purpose |
|---------|-------|---------|
| **Module docstring** | 1-35 | Algorithm overview + design principles |
| **Imports** | 37-43 | Dependencies (torch, registry, parent trainer) |
| **Class definition** | 50-99 | HyMotionM2MSoarTrainer with __init__, docstring |
| **_smooth_l1_loss** | 107-110 | Utility: elementwise SmoothL1 |
| **_masked_velocity_loss** | 112-140 | Core: mask-aware velocity loss computation |
| **_soar_correction_loss** | 143-251 | **CORE**: Full SOAR correction algorithm |
| **train_step** | 254-275 | **ENTRY POINT**: Orchestrates training step |
| **Unit tests** | 282-436 | Smoke tests, mask verification, CFG validation |

**Most Important Methods**:

1. **`train_step(batch)` (lines 254-275)**
   - Entry point called by training loop
   - Orchestrates: base forward → base loss → correction loss → combined loss
   - Returns: `{'loss': total_loss, 'loss_velocity': ..., 'loss_soar_corr': ...}`

2. **`_soar_correction_loss(ctx)` (lines 143-251)**
   - Core algorithm implementation
   - Splits into: R1 (rollout) + R2 (re-noising + correction)
   - Calls: `_masked_velocity_loss()` internally

3. **`_masked_velocity_loss(pred_vel, gt_vel, generation_mask, data_mask_temporal)` (lines 112-140)**
   - Applies mask weighting to loss
   - Handles variable-length sequences

---

## Configuration Files

### Location: `configs/hymotion_m2m_v2/soar/`

**All configurations inherit from base SFT configs** and add SOAR hyperparameters.

#### Primary Configurations:

| File | Purpose | Base | Key Variant |
|------|---------|------|-------------|
| **hymotion_m2m_v2_uncond_local_046b_soar.py** | Full training (5K steps) | uncond_local | Recommended starting point |
| **hymotion_m2m_v2_uncond_local_046b_soar_quickcheck.py** | Quick verification (400 steps) | uncond_local | Single GPU testing |
| **hymotion_m2m_v2_uncond_global_046b_soar.py** | Full training, global rotation | uncond_global | Alternative rotation convention |
| **hymotion_m2m_v2_caption_local_046b_soar.py** | Text-conditioned (with CFG) | caption_local | Future: text-guided SOAR |
| **hymotion_m2m_v2_caption_global_046b_soar.py** | Text-conditioned, global | caption_global | Future: text-guided SOAR |

#### SOAR Configuration Parameters (in all files):

```python
trainer = dict(
    type='HyMotionM2MSoarTrainer',
    val_num_steps=10,
    mask_aware_noise=True,              # Line 24
    soar_lambda=0.1,                    # Line 25: correction weight
    soar_num_aux=1,                     # Line 26: auxiliary points
    soar_K=50,                          # Line 27: sampling steps
    soar_cfg_scale=1.0,                 # Line 28: CFG scale
    soar_sigma_clamp=0.05,              # Line 29: numerical guard
)
```

#### Optimization Configuration (in all files):

```python
optimizer = dict(
    type='AdamW',
    lr=2e-5,                            # 5x smaller than SFT
    betas=[0.9, 0.99],
    weight_decay=0.0,
)

train_dataloader = dict(
    batch_size=14,                      # Half of SFT (28)
)

train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=5000,                     # SOAR only 5K steps
    val_interval=100000,                # Skip validation
    max_grad_norm=1.0,
)
```

---

## Documentation Files

### **docs/temp/soar_m2m_v2_post_training_plan.md** (570 lines)

**Author**: Auto-generated analysis  
**Date**: 2026-04-17  
**Status**: Proposal (later implemented as SOAR trainer)

**Sections**:
1. Executive Summary (key conclusions table)
2. SOAR Method Overview (exposure bias problem, algorithm, principles)
3. Applicability Analysis (framework compatibility, M2M's exposure bias severity)
4. Complete Implementation Plan (architecture changes, adapted algorithm, mask handling)
5. Hyperparameter Recommendations (primary, training, ablation plan)
6. Compute Budget (per-step cost, total GPU-hours)
7. Implementation Details (code changes, config, notes, gradient flow)
8. Evaluation Plan (metrics, protocol)
9. Risk Analysis and Mitigation (7 identified risks + mitigations)
10. Implementation Timeline (6 phases, ~8 days total)
11. Summary (key conclusions)

**Key Tables**:
- **Table 1**: SOAR suitability for M2M (YES, framework identical)
- **Table 2**: M2M exposure bias factors (50-step ODE, temporal, VACE, blend)
- **Table 3**: Hyperparameter recommendations (lambda=0.1, N=1, LR=2e-5)
- **Table 4**: Compute budget (SOAR E1: 3.5h on 8xA100)
- **Table 5**: Risk matrix (7 risks, mitigations)

### **docs/temp/physics_feedback_soar_analysis.md**

**Status**: Companion analysis (physics oracle integration notes)

---

## Base Trainer (Parent Class)

### **hftrainer/trainers/motion/hymotion_m2m_trainer.py** (621 lines)

**SOAR inherits everything except**:
- `train_step()` — overridden to add correction loss
- `__init__()` — extended with SOAR hyperparameters

**Key inherited methods** (used by SOAR):

| Method | Lines | Purpose |
|--------|-------|---------|
| `_prepare_and_forward(batch)` | 48-312 | Batch preparation, base forward pass |
| `_compute_base_loss(ctx)` | 314-418 | Computes on-trajectory velocity/x1 loss |
| `_compute_fk_keypoints(pred_x1, gt_x1)` | 429-465 | FK loss computation |
| `_compute_fk_consistency_loss(pred_x1, ...)` | 466-500 | FK consistency (198-dim models) |
| `_compute_kimodo_aux_loss(pred_x1, ...)` | 502-577 | KIMODO auxiliary losses |

**Context dict** returned by `_prepare_and_forward()` (keys used by SOAR):

```python
ctx = {
    'device': device,
    'src_motion': src_motion,           # (B, L, D)
    'tgt_motion': tgt_motion,           # (B, L, D)
    'src_mask': src_mask,               # (B, L, D) 1=generate, 0=known
    'tgt_padding_mask': tgt_padding_mask,  # (B, L) bool
    'x0': x0,                           # (B, L, D) noise
    'x1': x1,                           # (B, L, D) clean
    'x_t': x_t,                         # (B, L, D) noisy sample
    't': t,                             # (B, 1, 1) timestep
    'timesteps': timesteps,             # (B,) for model
    'pred': pred,                       # (B, L, D) model output (v_pred)
    'vace_context': vace_context,       # (B, L, 3D) VACE conditioning
    'vtxt_input': vtxt_input,           # (B, 1, D) text embeddings
    'ctxt_input': ctxt_input,           # (B, T_text, D) text context
    'ctxt_mask_temporal': ctxt_mask_temporal,  # (B, T_text) bool
    'generation_mask': generation_mask, # (B, L, D) 1=generate
    'text_available': text_available,   # (B,) bool
}
```

---

## Code Locations: Key Concepts

### Exposure Bias Problem

**What**: Train uses on-trajectory states (via GT forward process); inference uses model's own accumulated predictions (off-trajectory).

**Impact on M2M**: 
- 50-step ODE integration (more steps than SD3.5)
- Temporal self-attention (early errors propagate to future frames)
- VACE conditioning (generated region errors affect context understanding)

**SOAR Solution**: Dense correction loss on 1-step-off-trajectory states

**In Code**:
- Lines 164-186: Stop-gradient rollout (creates off-trajectory state)
- Lines 196-250: N auxiliary re-noise points (multiple off-trajectory samples)

### Stop-Gradient Rollout

**What**: Take one Euler step from x_t0 towards clean using model's own velocity, then detach.

**Why**: 
- Generates realistic off-trajectory state
- Detach prevents circular gradient flow
- Matches SOAR paper's design

**In Code**:
```python
# Lines 174-186
K = float(self.soar_K)
t1 = (t0 + 1.0 / K).clamp(max=1.0)
dt = t1 - t0
with torch.no_grad():
    v_rollout = v_pred.detach()
    x_hat = x_t0.detach() + v_rollout * dt
    if self.mask_aware_noise and src_mask is not None:
        keep_mask = 1 - src_mask
        x_hat = x_hat * src_mask + x1 * keep_mask
```

### Re-Noising & Correction Targets

**What**: From off-trajectory x_hat, sample N auxiliary points at t' ∈ [0, t1], each with correction target pointing back to clean.

**Why**: 
- Explores diverse off-trajectory states
- Multiple correction targets (N points) = dense supervision
- Shared noise z1 keeps re-noised states on transport ray

**In Code**:
```python
# Lines 196-250
for _ in range(self.soar_num_aux):
    rand = torch.rand(B, 1, 1, device=device, dtype=x1.dtype)
    t_prime = t1 * (1.0 - rand)  # Sample t' ∈ [0, t1]
    alpha = 1.0 - rand
    z_re = alpha * x_hat + (1.0 - alpha) * x0
    # ... mask-aware application ...
    one_minus_tp = (1.0 - t_prime).clamp_min(sigma_clamp)
    v_corr = (x1 - z_re) / one_minus_tp
    # ... forward pass on z_re with gradient ...
    corr = self._masked_velocity_loss(v_off, v_corr.detach(), ...)
```

### Mask-Aware Noise Integration

**What**: Known regions always remain clean (= x1) throughout SOAR computation.

**Why**: Known regions are deterministically replaced during inference via imputation; training should match this distribution.

**Application Points**:
1. Initial x_t (already done by parent trainer)
2. After rollout x_hat (lines 183-185)
3. Re-noised z_re (lines 211-214)
4. Loss computation (weighted by generation_mask in _masked_velocity_loss)

**In Code**:
```python
# Line 184-185 (x_hat masking)
if self.mask_aware_noise and src_mask is not None:
    keep_mask = 1 - src_mask
    x_hat = x_hat * src_mask + x1 * keep_mask

# Lines 211-214 (z_re masking)
if self.mask_aware_noise and src_mask is not None:
    keep_mask = 1 - src_mask
    z_re = z_re * src_mask + x1 * keep_mask
```

---

## Data Flow Diagram

```
Training Batch
└─ _prepare_and_forward(batch)
   ├─ Normalize motions
   ├─ Prepare VACE context
   ├─ Sample timestep t0 ~ U[0,1]
   ├─ Sample noise x0 ~ N(0,I)
   ├─ Create x_t0 = (1-t0)*x0 + t0*x1
   ├─ Apply mask-aware: x_t0[known]=x1
   ├─ Forward: v_pred = model(x_t0, ...)
   └─ Return context dict
                │
                ├─ _compute_base_loss(ctx)
                │   ├─ v_gt = x1 - x0
                │   ├─ L_base = loss(v_pred, v_gt)
                │   └─ return L_base
                │
                ├─ _soar_correction_loss(ctx)
                │   ├─ STEP R1: Rollout
                │   │   ├─ t1 = t0 + 1/K
                │   │   ├─ v_rollout = v_pred.detach()
                │   │   ├─ x_hat = x_t0 + v_rollout * (t1-t0)
                │   │   └─ x_hat[known] = x1
                │   │
                │   ├─ STEP R2: For each n in 1..N:
                │   │   ├─ Sample t' ~ U[0, t1]
                │   │   ├─ z_re = interp(x_hat, x0, t')
                │   │   ├─ z_re[known] = x1
                │   │   ├─ v_corr = (x1 - z_re) / (1-t')
                │   │   ├─ v_off = model(z_re, ..., t')
                │   │   ├─ L_corr_n = loss(v_off, v_corr)
                │   │   └─ accumulate
                │   │
                │   └─ return mean(L_corr_n)
                │
                └─ Combine: L_total = L_base + λ * L_corr
                    └─ return {'loss': L_total, ...}
```

---

## Test Coverage

### File: **hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py** (lines 282-436)

**Test 1: _test_mask_aware_preserves_known_regions() (lines 383-408)**
- Synthetic data: B=2, L=6, D=5
- Verifies: After masking, `(x_hat_masked - x1) * keep_mask ≈ 0`
- Status: ✅ Known regions preserved exactly

**Test 2: _test_cfg_scale_validation() (lines 411-428)**
- Verifies: NotImplementedError raised if cfg_scale ≠ 1.0
- Status: ✅ Correctly rejects unsupported CFG

**Test 3: _test_soar_shapes_and_finiteness() (lines 282-381)**
- Synthetic data: B=2, L=8, D=9, K=4
- Mock bundle with "perfect" model (v_pred == v_gt)
- Verifies: 
  - Shapes correct
  - Loss finite
  - Loss small with perfect bundle
  - Mask-aware handling
- Status: ✅ All checks pass

**Run Command**:
```bash
python -m hftrainer.trainers.motion.hymotion_m2m_soar_trainer
```

---

## Registry & Imports

### **hftrainer/trainers/motion/__init__.py**

```python
from hftrainer.trainers.motion.hymotion_m2m_soar_trainer import HyMotionM2MSoarTrainer

__all__ = [
    'HyMotionM2MTrainer', 
    'HyMotionM2MSoarTrainer',  # ← Registered
    ...
]
```

**Registration**: `@TRAINERS.register_module()` (line 49 of trainer file)

**Usage in configs**:
```python
trainer = dict(type='HyMotionM2MSoarTrainer', ...)
```

---

## Hyperparameter Quick Reference

### Conservative Defaults (Recommended)

| Hyperparameter | Default | Range | Notes |
|----------------|---------|-------|-------|
| `soar_lambda` | 0.1 | 0.05-0.5 | Start low; higher = stronger correction |
| `soar_num_aux` | 1 | 1-4 | Higher = more auxiliary points, more compute |
| `soar_K` | 50 | 50 | Must match inference `num_sampling_steps` |
| `soar_cfg_scale` | 1.0 | 1.0 only | No CFG (v1 limitation) |
| `soar_sigma_clamp` | 0.05 | 0.01-0.2 | Higher = less aggressive clamping |

### Training Hyperparameters (in config file)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `lr` | 2e-5 | 5x smaller than SFT (1e-4) |
| `batch_size` | 14 | Half of SFT (28); accounts for ~2x forwards |
| `max_iters` | 5000 | Short post-training (can extend to 10K) |
| `max_grad_norm` | 1.0 | Standard clipping |
| `betas` | [0.9, 0.99] | AdamW standard |
| `weight_decay` | 0.0 | No L2 regularization |

---

## Ablation Plan

**From soar_m2m_v2_post_training_plan.md (§5.3)**

| Exp | Config | Purpose | Status |
|-----|--------|---------|--------|
| E0 | Baseline (uncond_fm_man_046b ep1000) | Reference | Reference |
| E1 | λ=0.1, N=1, 5K steps | Minimal SOAR | Implemented |
| E2 | λ=0.1, N=2, 5K steps | More auxiliary points | Design ready |
| E3 | λ∈{0.05,0.1,0.5,1.0}, N=1 | Lambda sweep | Design ready |
| E4 | Best(E1-E3), 10K steps | Longer training | Design ready |
| E5 | Best(E4) + SDE paths | Stochastic exploration | Future |
| E6 | Same on caption model + CFG | Generalization | Future (needs CFG impl) |

---

## Compute Specifications

### GPU Memory (8xA100)

| Config | Batch Size | Per-GPU Memory | Total |
|--------|-----------|-----------------|-------|
| SFT | 28 | ~2.5 GB | ~20 GB |
| SOAR (N=1) | 14 | ~4-5 GB | ~35-40 GB |
| SOAR (N=2) | 10 | ~4-5 GB | ~35-40 GB |

### Training Duration

| Experiment | Steps | Cost/Step | Duration (8xA100) |
|------------|-------|-----------|-------------------|
| E1 (SOAR N=1) | 5,000 | 2.0x SFT | ~3.5 hours |
| E4 (SOAR N=1) | 10,000 | 2.0x SFT | ~7 hours |
| Full ablation (E1-E5) | ~30,000 | 2.0x avg | ~21 hours |

---

## Common Workflows

### Workflow 1: Quick Smoke Test (1 GPU, 5 min)

```bash
# Run unit tests
python -m hftrainer.trainers.motion.hymotion_m2m_soar_trainer

# Quick training check (400 steps)
python tools/dist_train.sh \
  configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar_quickcheck.py 1
```

**Expected output**: Tests pass ✅, training loop runs without errors

### Workflow 2: First Full Training (8 GPUs, 3.5h)

```bash
# Main E1 experiment
bash tools/dist_train.sh \
  configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar.py 8

# Monitor
tensorboard --logdir work_dirs/hymotion_m2m_v2_uncond_local_046b_soar

# Check results
python eval_full_benchmark.py \
  work_dirs/hymotion_m2m_v2_uncond_local_046b_soar/checkpoint-epoch_5
```

### Workflow 3: Ablation Study (8 GPUs, 21h total)

```bash
# Run experiments E1-E5 in sequence
for lambda in 0.05 0.1 0.5 1.0; do
    # Modify config: soar_lambda=$lambda
    bash tools/dist_train.sh config_lambda_$lambda.py 8
done
```

---

## FAQ & Troubleshooting Index

**Q: Where do I start?**  
A: Read SOAR_QUICK_REFERENCE.txt, then launch E1 with the quickcheck config.

**Q: How do I know if it's working?**  
A: Run unit tests, check TensorBoard for decreasing `loss_soar_corr`.

**Q: Why is memory usage so high?**  
A: SOAR needs 2 forward passes + 2 backward passes per step.

**Q: Can I use this with text-conditioned models?**  
A: Yes, but CFG support (cfg_scale ≠ 1.0) is a TODO. Start with uncond models.

**Q: How long does SOAR training take?**  
A: 5K steps ≈ 3.5h on 8xA100. 10K steps ≈ 7h.

---

## Key Takeaways

1. **SOAR is a targeted fix**: Addresses exposure bias in generated regions via correction loss on 1-step-off-trajectory states
2. **Complementary to _man**: Mask-aware noise handles known regions; SOAR handles generated regions
3. **Minimal code footprint**: ~150 new lines in one trainer file
4. **Proven effectiveness**: +11% GenEval on SD3.5-Medium; expected 5-10% improvement on M2M
5. **Conservative by default**: λ=0.1, N=1, K=50 are proven starting points
6. **Short duration**: 5-10K post-training steps vs 70K SFT epochs
7. **No annotation needed**: Fully self-supervised correction targets

---

## Document Statistics

| Document | Lines | Purpose |
|----------|-------|---------|
| SOAR_TRAINING_ANALYSIS.md | 735 | Complete method-by-method reference |
| SOAR_QUICK_REFERENCE.txt | 366 | User-facing launch & monitoring guide |
| SOAR_INDEX.md (this) | ~400 | Navigation & cross-reference |
| **Total Documentation** | **1500+** | Comprehensive coverage |

---

**Generated**: 2026-05-18  
**Completeness**: All code sections identified, all configurations documented, all workflows covered.
