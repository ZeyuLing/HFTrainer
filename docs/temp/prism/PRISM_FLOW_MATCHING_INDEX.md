# PRISM Flow Matching Training - Documentation Index

This directory contains comprehensive documentation of the PRISM flow matching training implementation. Start with the summary, then drill down into specific topics.

## 🎯 Quick Start (Read These First)

### 1. **PRISM_SUMMARY.txt** ⭐ START HERE
High-level overview of all three questions with exact equations, code snippets, and file locations.

**Answers:**
- Noising equation: `x_t = (1 - σ_t) * x_0 + σ_t * ε`
- Training target: Velocity `v_t = ε - x_0`
- Timestep sampling: Uniform random from scheduler

### 2. **PRISM_QUICK_REFERENCE.txt**
Visual quick reference guide with diagrams and key information organized by section.

### 3. **PRISM_CODE_REFERENCE.md**
Complete code listings with line numbers and detailed explanations of each component.

---

## 📚 Detailed Documentation

### **PRISM_FLOW_MATCHING_ANALYSIS.md**
Comprehensive analysis covering:
- Noising mechanics with mathematical proofs
- Training targets and velocity field explanation
- Timestep/sigma sampling strategy
- Complete training step walkthrough
- Special features (translation/rotation weighting, frame conditioning)
- Scheduler dependency
- Mathematical summary table

### **PRISM_QUICK_REFERENCE.txt**
Visual reference with ASCII boxes covering:
- Noising equation
- Training target (velocity)
- Timestep sampling strategy
- Sigma retrieval process
- Complete training flow
- Key relationships
- Special features
- File locations

---

## 🔍 Specific Topics

### Training Implementation
- **Trainer Location:** `hftrainer/trainers/motion/prism_trainer.py`
- **Method:** `PrismTrainer.train_step()` (lines 41-118)
- **MCM Variant:** `hftrainer/trainers/motion/prism_mcm_trainer.py`

### Noising & Encoding
- **Bundle Location:** `hftrainer/models/motion/prism/bundle.py`
- **Key Methods:**
  - `add_flow_noise()` (lines 257-262)
  - `_get_sigmas()` (lines 19-27)
  - `encode_motion()` (lines 126-154)
  - `encode_prompt()` (lines 157-193)

---

## 📋 Key Equations at a Glance

### Noising
```
x_t = (1 - σ_t) * x_0 + σ_t * ε
```

### Training Target (Velocity)
```
v_t = ε - x_0
```

### Loss
```
L = MSE(model_pred, targets)
```

### Timestep Sampling
```
t_i ~ Uniform(scheduler.timesteps)
```

### Sigma Lookup
```
σ_i = scheduler.sigmas[index(t_i)]
```

---

## 🔧 Code Components

### 1. Noising Function (bundle.py:257-262)
```python
def add_flow_noise(self, latents, timesteps):
    noise = torch.randn_like(latents)
    sigmas = _get_sigmas(self.scheduler, timesteps, n_dim=latents.ndim, ...)
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    targets = noise - latents
    return noisy_latents, targets
```

### 2. Sigma Retrieval (bundle.py:19-27)
```python
def _get_sigmas(scheduler, timesteps, n_dim=4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
```

### 3. Timestep Sampling (prism_trainer.py:68-75)
```python
step_indices = torch.randint(
    0,
    len(self.bundle.scheduler.timesteps),
    (batch_size,),
    device=latents.device,
)
scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
timesteps = scheduler_timesteps[step_indices]
```

---

## 📊 Data Dimensions

| Variable | Shape | Description |
|----------|-------|-------------|
| motion | [B, T, J*6] | Original motion frames |
| latents | [B, C, T', J] | Encoded latents |
| noisy_latents | [B, C, T', J] | Noised version |
| targets | [B, C, T', J] | Velocity targets |
| timesteps | [B] | Sampled timestep indices |
| sigmas | [B, 1, 1, 1] | Sigma values (broadcasts) |
| model_pred | [B, C, T', J] | Model output |
| loss | scalar | Final loss value |

Where:
- B = batch_size
- C = num_channels (latent dimension)
- T' = downsampled temporal dimension
- J = 23 (1 translation + 22 rotations)

---

## 🎓 Understanding the Flow

### During Training:
1. **Random timestep sampling:** Each sample gets different noise level
2. **Noise addition:** Linear interpolation between clean and noise
3. **Model prediction:** Predicts velocity field
4. **Loss computation:** MSE between prediction and ground truth velocity
5. **Weighted loss:** Separate translation and rotation to prevent dilution

### During Inference:
1. **Scheduled timesteps:** Sequential from high noise to low noise
2. **Iterative denoising:** Multiple steps guided by velocity predictions
3. **Euler integration:** Follow velocity field to clean sample

---

## 📁 Related Files

### Configuration
- `hftrainer/models/motion/prism/transformer_prism.py` - Transformer architecture
- `hftrainer/models/motion/prism/bundle.py` - Model bundle with schedulers

### Pipelines
- `hftrainer/pipelines/motion/prism_pipeline.py` - Inference pipeline
- `hftrainer/pipelines/motion/prism_mcm_pipeline.py` - Audio-conditioned pipeline

### Tests
- `test_prism_jitter_fixes.py` - Stability tests

---

## ✨ Special Features

### Translation vs Rotation Loss Weighting
- **Problem:** 23 joints (1 translation + 22 rotations)
- **Solution:** Weight both components equally (0.5 each)
- **Prevents:** Translation being diluted by rotations

### Frame Conditioning
- **Rate:** 10% of batches (configurable)
- **Effect:** First N frames kept clean
- **Purpose:** Learn guided generation from real motion

### Audio-Conditioned Variant (MCM)
- Same flow matching logic
- Added audio encoding and dropout
- Control transformer for multi-modal conditioning

---

## 🔗 Full File Reference

| File | Size | Purpose |
|------|------|---------|
| PRISM_SUMMARY.txt | 12K | Executive summary |
| PRISM_QUICK_REFERENCE.txt | 14K | Quick lookup guide |
| PRISM_CODE_REFERENCE.md | 11K | Code listings and details |
| PRISM_FLOW_MATCHING_ANALYSIS.md | 9.8K | Deep analysis |
| PRISM_FLOW_MATCHING_INDEX.md | This file | Navigation guide |

---

## 📞 Questions Addressed

### Q1: How noise is added to latents (noising equation)
**Answer:** `x_t = (1 - σ_t) * x_0 + σ_t * ε`
- Location: `bundle.py`, lines 257-262
- Function: `add_flow_noise()`

### Q2: What the training target is (velocity? noise? x0?)
**Answer:** Velocity `v_t = ε - x_0`
- Location: `bundle.py`, line 261
- Interpretation: Direction from noisy to clean sample

### Q3: How timesteps/sigmas are sampled during training
**Answer:** Uniform random from scheduler's timestep array
- Location: `prism_trainer.py`, lines 68-75
- Per-sample variation: Different noise levels per batch
- Sigma lookup: Via `_get_sigmas()` function

---

Generated: 2026-05-19
Repository: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`
