# SOAR Training System — README & Getting Started

> **Status**: Complete implementation analysis  
> **Date**: 2026-05-18  
> **Audience**: All users implementing, launching, or studying SOAR post-training

---

## What is SOAR?

**SOAR (Self-Correction for Optimal Alignment and Refinement)** is a reward-free post-training method that addresses **exposure bias** in diffusion models:

- **The Problem**: Training uses ground-truth (on-trajectory) states, but inference uses model predictions (off-trajectory). This mismatch accumulates errors across 50 denoising steps.
- **The Solution**: SOAR corrects the model on off-trajectory states using self-supervised targets derived from clean data.
- **Key Benefit**: +5-11% improvement in generation quality (proven on SD3.5-Medium; expected 5-10% on HyMotion M2M)

---

## Three-Document System

This analysis provides **2,182 lines** of documentation across **three complementary documents**:

### 1. **SOAR_QUICK_REFERENCE.txt** ← **START HERE** (366 lines)
   - **For**: Users launching training runs
   - **Contains**: File locations, method signatures, hyperparameter tables, launch commands, monitoring metrics, troubleshooting
   - **Read time**: 10-15 minutes
   - **Use when**: Setting up a training run or debugging

### 2. **SOAR_TRAINING_ANALYSIS.md** (735 lines)
   - **For**: Developers understanding or extending the implementation
   - **Contains**: Complete method-by-method breakdown, gradient flow diagrams, mathematical derivations, unit tests, integration points
   - **Read time**: 30-45 minutes
   - **Use when**: You need to understand every step of the algorithm

### 3. **SOAR_INDEX.md** (544 lines)
   - **For**: Navigation and cross-reference
   - **Contains**: Index of all code locations, files, hyperparameters, workflows, FAQ
   - **Read time**: 5-10 minutes
   - **Use when**: You need to find where something is implemented

---

## Implementation Summary

**File**: `hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py` (437 lines)

**Class**: `HyMotionM2MSoarTrainer(HyMotionM2MTrainer)`

**What changed**: 
- One trainer file (inherits everything else from parent)
- ~150 new lines of SOAR-specific logic
- 5 config variants in `configs/hymotion_m2m_v2/soar/`

**Training loop**:
```
train_step(batch)
├─ Base forward: v_pred = model(x_t0)
├─ Base loss: L_base = MSE(v_pred, v_gt)
├─ Stop-gradient rollout: x_hat = detach(x_t0 + detach(v_pred) * dt)
├─ N auxiliary re-noises from x_hat
├─ Correction loss: L_corr = MSE(model(z_re), v_corr)
└─ Combined: L_total = L_base + λ*L_corr
```

---

## Quick Start (5 minutes)

### Step 1: Verify Installation
```bash
python -m hftrainer.trainers.motion.hymotion_m2m_soar_trainer
```
Expected output: `All SOAR trainer tests passed ✅`

### Step 2: Quick Smoke Test (1 GPU, 5 min)
```bash
python tools/dist_train.sh \
  configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar_quickcheck.py 1
```
Runs 400 training steps to verify everything works.

### Step 3: Full Training (8 GPUs, 3.5 hours)
```bash
bash tools/dist_train.sh \
  configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar.py 8
```
Runs 5,000 post-training steps (recommended starting point).

### Step 4: Monitor Training
```bash
tensorboard --logdir work_dirs/hymotion_m2m_v2_uncond_local_046b_soar
```
Watch for:
- `loss_velocity` stays stable (should be similar to base)
- `loss_soar_corr` decreases over time
- `loss` trends downward

---

## Recommended First Experiment

| Aspect | Value | Rationale |
|--------|-------|-----------|
| **Config** | `hymotion_m2m_v2_uncond_local_046b_soar.py` | Unconditional model, local rotation |
| **Base checkpoint** | epoch_485 | Latest SFT model, well-trained |
| **soar_lambda** | 0.1 | Conservative; avoid degrading base loss |
| **soar_num_aux** | 1 | Low compute overhead |
| **max_iters** | 5000 | Short post-training phase |
| **batch_size** | 14 | Half of SFT to account for ~2x forwards |
| **learning_rate** | 2e-5 | 5x smaller than SFT |
| **Hardware** | 8xA100 | ~40GB total memory |
| **Duration** | ~3.5 hours | Fast post-training |
| **Expected gain** | 5-10% | Boundary smoothness + temporal coherence |

---

## Key Hyperparameters

### SOAR Hyperparameters

```python
soar_lambda = 0.1        # Correction loss weight (0.05-0.5 range)
soar_num_aux = 1         # Auxiliary points per sample (1-4 typical)
soar_K = 50              # Sampling steps (MUST match inference)
soar_cfg_scale = 1.0     # CFG scale (ONLY 1.0 supported in v1)
soar_sigma_clamp = 0.05  # Numerical stability lower bound
```

### Training Hyperparameters

```python
lr = 2e-5                # Learning rate (5x smaller than SFT)
batch_size = 14          # Half of SFT batch
max_iters = 5000         # Post-training steps (5-10K range)
max_grad_norm = 1.0      # Gradient clipping
optimizer = 'AdamW'      # Standard
```

---

## File Locations

### Core Implementation
```
hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py    (437 lines)
hftrainer/trainers/motion/hymotion_m2m_trainer.py         (parent, 621 lines)
hftrainer/trainers/motion/__init__.py                     (registry)
```

### Configurations (5 variants)
```
configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar.py
configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_local_046b_soar_quickcheck.py
configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_uncond_global_046b_soar.py
configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_caption_local_046b_soar.py
configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_caption_global_046b_soar.py
```

### Documentation
```
docs/temp/soar_m2m_v2_post_training_plan.md               (570 lines, detailed plan)
SOAR_TRAINING_ANALYSIS.md                                 (735 lines, method reference)
SOAR_QUICK_REFERENCE.txt                                  (366 lines, launch guide)
SOAR_INDEX.md                                             (544 lines, navigation)
```

---

## Core Methods

### Entry Point: `train_step(batch)`
**Location**: lines 254-275 of trainer

Returns dict with:
- `'loss'` → total loss (scalar)
- `'loss_velocity'` → base velocity loss
- `'loss_soar_corr'` → correction loss
- Other loss components (motion smoothness, FK, etc.)

### Core Algorithm: `_soar_correction_loss(ctx)`
**Location**: lines 143-251 of trainer

```
STEP R1: Stop-gradient rollout (lines 164-186)
  t1 = clamp(t0 + 1/K, max=1)
  x_hat = detach(x_t0 + detach(v_pred) * dt)
  
STEP R2: N auxiliary re-noises (lines 196-251)
  for n in 1..N:
    t' ~ U[0, t1]
    z_re = interpolate(x_hat, x0, t')
    v_corr = (x1 - z_re) / (1-t')
    v_off = model(z_re, ..., t')
    L_corr_n = loss(v_off, v_corr)
  return mean(L_corr_n)
```

---

## Training Monitoring

### Key Metrics in TensorBoard

| Metric | Good Range | Action |
|--------|-----------|--------|
| `loss_velocity` | 0.3-0.5, stable | Should stay similar to base model |
| `loss_soar_corr` | > 0, decreasing | Should trend downward |
| `loss` (total) | Decreasing | Main metric to watch |
| `loss_soar_corr` / `loss_velocity` | ≈ λ (≈0.1) | Ratio should be close to lambda |

### Health Checks

✅ **Healthy indicators**:
- No NaN/Inf in any loss component
- GPU memory stable (~35-40GB)
- `loss_velocity` within expected range
- `loss_soar_corr` decreasing

❌ **Red flags**:
- `loss_velocity` increasing → lambda too high
- `loss_soar_corr` flat/increasing → model not learning correction
- NaN in loss → numerical instability (increase sigma_clamp)
- OOM error → reduce batch_size or enable gradient checkpointing

---

## Common Issues & Solutions

### Issue: `NotImplementedError: SOAR with text CFG`
**Cause**: `soar_cfg_scale ≠ 1.0`  
**Solution**: Use `soar_cfg_scale=1.0` (v1 limitation) or use unconditional model

### Issue: NaN in loss_soar_corr
**Cause**: Division by very small `(1-t')`  
**Solution**: Increase `soar_sigma_clamp` (try 0.1-0.2)

### Issue: Out of Memory
**Cause**: ~2x memory overhead from dual forward passes  
**Solutions**:
- Reduce batch_size (14 → 10-12)
- Ensure gradient_checkpointing enabled
- Reduce max_grad_norm
- Reduce soar_num_aux (already minimal at 1)

### Issue: loss_velocity increases
**Cause**: Correction loss weight too high  
**Solution**: Reduce soar_lambda (0.1 → 0.05)

---

## What Gets Improved?

**SOAR specifically targets**:
1. **Boundary smoothness** — Transition quality between known and generated regions (PRIMARY BENEFIT)
2. **Temporal coherence** — Consistency across frames (especially long sequences)
3. **Off-trajectory error correction** — Prevents error accumulation through 50-step ODE

**What stays the same**:
- Basic motion quality (should not degrade)
- Unconditional generation capability
- Text conditioning (when enabled)
- Inference speed

---

## Next Steps After Training

### Evaluation
```bash
# Full 12-task benchmark
python eval_full_benchmark.py \
  work_dirs/hymotion_m2m_v2_uncond_local_046b_soar/checkpoint-epoch_5

# Compare with baseline
python compare_models.py \
  work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_485 \
  work_dirs/hymotion_m2m_v2_uncond_local_046b_soar/checkpoint-epoch_5
```

### Ablation Studies (Optional)
Try different hyperparameter combinations (see SOAR_QUICK_REFERENCE.txt "Ablation Plan"):
- E2: Increase soar_num_aux to 2
- E3: Lambda sweep (0.05, 0.1, 0.5, 1.0)
- E4: Longer training (10K steps)

### Production Deployment
Once satisfied with evaluation:
```bash
# Copy checkpoint to production location
cp -r work_dirs/hymotion_m2m_v2_uncond_local_046b_soar/checkpoint-epoch_5 \
      models/hymotion_m2m_v2_uncond_local_046b_soar
```

---

## Summary Table

| Aspect | Value |
|--------|-------|
| **Implementation** | 1 trainer file, ~150 SOAR-specific lines |
| **Inference impact** | None (SOAR is post-training only) |
| **Training time** | 3.5h (5K steps) to 7h (10K steps) on 8xA100 |
| **Expected gain** | 5-10% improvement in boundary smoothness |
| **Data requirements** | None (self-supervised) |
| **Annotation required** | None |
| **Compatibility** | Works with all M2M configs (uncond/caption, local/global) |
| **Complexity** | Low (inherits all parent functionality) |
| **Risk level** | Low (conservative defaults, proven algorithm) |

---

## Further Reading

- **SOAR_TRAINING_ANALYSIS.md** — Complete method-by-method breakdown
- **SOAR_QUICK_REFERENCE.txt** — Launch and monitoring guide  
- **SOAR_INDEX.md** — Navigation and cross-reference
- **docs/temp/soar_m2m_v2_post_training_plan.md** — Detailed planning document (570 lines)
- **SOAR paper**: arXiv 2604.12617 (Tencent Hunyuan)

---

## Questions?

See **SOAR_INDEX.md** for:
- Troubleshooting FAQ
- Common workflows
- Detailed hyperparameter reference
- Code location index

See **SOAR_TRAINING_ANALYSIS.md** for:
- Algorithm walkthroughs with code
- Gradient flow diagrams
- Mathematical derivations
- Integration details

---

**Generated**: 2026-05-18  
**Completeness**: Complete end-to-end reference for SOAR training implementation

Start with SOAR_QUICK_REFERENCE.txt for launching, SOAR_TRAINING_ANALYSIS.md for understanding.
