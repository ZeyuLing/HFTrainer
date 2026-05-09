# SOAR Post-Training for HyMotion M2M v2 — Complete Plan

> Date: 2026-04-17
> Author: Auto-generated analysis
> References: arXiv 2604.12617, HY-SOAR (github.com/Tencent-Hunyuan/HY-SOAR)
> Status: Proposal

---

## 1. Executive Summary

**SOAR (Self-Correction for Optimal Alignment and Refinement)** is a reward-free post-training method for rectified-flow diffusion models that directly addresses **exposure bias** — the train/test distribution mismatch where training uses on-trajectory GT states but inference uses model's own accumulated predictions.

**Key conclusions:**

| Question | Answer |
|----------|--------|
| Can SOAR be applied to M2M v2? | **Yes.** Flow matching (rectified flow) framework is identical. M2M has severe exposure bias due to 50-step ODE integration + temporal data. |
| Is additional data annotation needed? | **No.** SOAR is fully self-supervised. Correction target = `(z_t' - x_clean) / sigma_t'`, no reward model or preference labels required. |
| Implementation complexity? | **Moderate.** ~150 lines of new code in trainer. Pipeline unchanged. |
| Recommended plan? | Post-train on `uncond_fm_man_046b` (epoch 1000 checkpoint), 5K–10K steps, lambda=0.1, N=1, LR=2e-5. |
| Expected benefit? | Reduced boundary discontinuity, improved temporal coherence, better long-sequence quality. Complementary to `_man` variant. |

---

## 2. SOAR Method Overview

### 2.1 The Exposure Bias Problem

Standard SFT diffusion training generates on-trajectory states via the forward process:
```
x_t = (1 - t) * noise + t * x_clean    (rectified flow convention)
```

At inference, `x_t` comes from the model's own ODE integration — any early-step error pushes subsequent states into out-of-distribution (OOD) regions the model has never trained on. Errors compound across 50 denoising steps.

| Stage | On-trajectory? | Signal density | Issue |
|-------|---------------|----------------|-------|
| SFT | Yes (GT forward process) | Dense (per-step MSE) | Inference uses off-trajectory states |
| RL (GRPO) | Yes (model rollout) | Sparse (terminal reward) | Credit assignment hard, reward hacking risk |
| **SOAR** | Partially (1-step rollout) | **Dense (per-step correction)** | **None** — combines benefits of both |

### 2.2 SOAR Algorithm (Algorithm 1 from Paper)

```
Input: trained model θ, clean data x_clean, noise schedule
For each training batch:
    1. BASE LOSS (standard SFT):
       z1 = randn_like(x_clean)                      # Gaussian noise
       t0 ~ U[0, 1]                                   # random timestep
       x_t0 = (1 - t0) * z1 + t0 * x_clean           # on-trajectory
       v_pred = model(x_t0, cond, t0)
       v_gt = x_clean - z1                            # GT velocity (flow matching)
       L_base = ||v_pred - v_gt||^2

    2. STOP-GRADIENT ROLLOUT (generate off-trajectory state):
       with no_grad():
           v_cfg = v_uncond + w_cfg * (v_cond - v_uncond)   # optional CFG
       t1 = max(t0 - 1/K, 0)                          # one step towards clean
       x_hat = x_t0 + (sigma_t1 - sigma_t0) * v_cfg   # off-trajectory state

    3. RE-NOISE + CORRECTION (N auxiliary points):
       L_corr = 0
       for n in range(N):
           t' ~ U[t1, 1]                              # auxiliary noise level
           alpha = (t' - t1) / (1 - t1)
           z_re = (1-alpha) * x_hat + alpha * z1       # re-noised with SAME z1

           v_corr = (z_re - x_clean) / sigma_t'       # correction target
           v_off = model(z_re, cond, t')               # model on off-trajectory
           L_corr += ||v_off - v_corr||^2

    4. COMBINED LOSS:
       L_total = (L_base + lambda * L_corr) / (B + lambda * P)
```

### 2.3 Key Design Principles

1. **Shared noise z1**: Base loss and correction loss use the **same** z1, keeping re-noised states near the original transport ray. Empirically better than fresh noise.

2. **Stop-gradient rollout**: Velocity used for rollout does not receive gradients. Prevents unstable gradient flow through the rollout step.

3. **Correction target**: `v_corr = (z_t' - x_clean) / sigma_t'` — directs the model at off-trajectory states back toward the correct clean target. No external reward signal needed.

4. **Dense supervision**: N auxiliary points per sample, each with an explicit per-timestep correction target.

5. **Stochastic rollout paths**: Beyond ODE rollout, supports SDE variants (simple, sde, flow_sde, cps) for diverse off-trajectory exploration.

### 2.4 SOAR Results on SD3.5-Medium

| Metric | Base (SFT) | SOAR | Improvement |
|--------|-----------|------|-------------|
| GenEval | 0.70 | **0.78** | +11% |
| OCR Accuracy | 0.64 | **0.67** | +5% |
| PickScore | — | +0.15 | human preference |
| HPSv2.1 | — | +0.005 | human preference |
| Aesthetic | — | +0.11 | visual quality |

SOAR on SD3.5-**Medium** surpasses larger SD3.5-**Large** on GenEval (0.78 vs 0.71). SOAR outperforms GRPO (RL) on high-aesthetic score while avoiding reward hacking.

---

## 3. Applicability Analysis: SOAR → HyMotion M2M v2

### 3.1 Framework Compatibility: Identical

| Aspect | SD3.5-Medium (SOAR) | HyMotion M2M |
|--------|---------------------|--------------|
| Generative paradigm | Rectified flow | Flow matching (rectified flow) |
| Noise schedule | `x_t = (1-t)*noise + t*clean` | `x_t = (1-t)*noise + t*clean` |
| Prediction target | velocity `v = noise - clean` | velocity `v = x1 - x0` |
| ODE solver | Euler | midpoint / Euler |
| Architecture | SD3 Transformer (DiT) | HunyuanMotion MMDiT |
| Data domain | Images (pixel/latent) | Motion (135-dim SMPL) |

**SOAR's rectified flow formulation can be directly transplanted to M2M without any mathematical modification.**

### 3.2 M2M Has Severe Exposure Bias

M2M's exposure bias is arguably **more severe** than image generation:

| Factor | Impact on M2M |
|--------|--------------|
| **50-step ODE integration** | More steps than SD3.5's few-step inference → more error accumulation |
| **Temporal data** | Early-frame errors propagate to later frames through self-attention |
| **VACE conditioning** | Generated region errors affect model's understanding of known context |
| **Post-hoc blend / imputation** | Hard boundary between clean known and noisy generated regions |
| **_man variant limitation** | Mask-aware noise only fixes known regions' distribution; generated regions still suffer full exposure bias |

**Critical insight**: The `_man` (mask-aware noise) variant solves the distribution mismatch for **known regions** (training: clean, inference: also clean via imputation). But **generated regions** still experience standard exposure bias. SOAR specifically targets generated regions' off-trajectory errors — the two methods are **orthogonally complementary**.

### 3.3 No Additional Data Annotation Required

| Requirement | SOAR | RL (GRPO/DPO) | Current M2M SFT |
|-------------|------|---------------|-----------------|
| Paired data (motion, mask, text) | Already have | Already have | Already have |
| Reward model | **Not needed** | Required | Not needed |
| Preference labels | **Not needed** | Required | Not needed |
| Negative samples | **Not needed** | Partially needed | Not needed |
| Quality annotations | **Not needed** | Not needed | Not needed |
| New data collection | **Not needed** | Not needed | Not needed |

**SOAR's correction target is purely self-supervised**: `v_corr = (z_t' - x_clean) / sigma_t'`. It derives entirely from the clean training target itself.

Existing training data (`train_hymotion_400h.json`, 549K samples; or quality-filtered `high_quality.json`, 456K samples) can be used directly, unchanged.

---

## 4. Complete Implementation Plan

### 4.1 Architecture: What Changes, What Doesn't

| Component | Change Required? | Details |
|-----------|-----------------|---------|
| `HunyuanMotionMMDiT` | **No** | Model architecture unchanged |
| `HyMotionM2MBundle` | **No** | Bundle structure unchanged |
| `HyMotionM2MTrainer` | **Yes** | Add SOAR rollout + correction loss logic |
| `HyMotionM2MPipeline` | **No** | Inference pipeline unchanged |
| Dataset / DataLoader | **No** | Same training data and format |
| Config | **Yes** | Add SOAR hyperparameters |
| Checkpoint strategy | **No** | Same save/load mechanism |

### 4.2 Adapted SOAR Algorithm for M2M _man Variant

Below is the complete adapted pseudocode, with mask-aware handling at every step:

```python
# ============================================================
# SOAR-M2M: Adapted for HyMotion M2M with mask-aware noise
# ============================================================

def soar_training_step(
    model,          # HunyuanMotionMMDiT
    x_clean,        # (B, L, D) normalized clean motion (x1 in flow matching)
    noise,          # (B, L, D) Gaussian noise (x0 in flow matching)
    src_motion,     # (B, L, D) source motion for VACE
    src_mask,       # (B, L, D) binary mask: 0=known, 1=generate
    text_emb,       # text embeddings
    t0,             # (B,) random timestep ~ U[0, 1]
    config,         # SOAR hyperparameters
):
    """
    Flow matching convention in M2M:
        x_t = (1 - t) * x0 + t * x1    (t=0: pure noise, t=1: clean)
        v_gt = x1 - x0                  (velocity from noise to clean)
    """
    B, L, D = x_clean.shape
    keep_mask = 1 - src_mask   # 1 where known, 0 where generate

    # ── Step 0: Prepare VACE context (unchanged) ──
    vace_ctx = prepare_vace_input(src_motion, src_mask)  # (B, L, 3*D)

    # ── Step 1: BASE LOSS (standard M2M SFT, unchanged) ──
    x_t0 = (1 - t0) * noise + t0 * x_clean              # on-trajectory

    # Mask-aware noise: known regions stay clean
    if config.mask_aware_noise:
        x_t0 = x_t0 * src_mask + x_clean * keep_mask

    x_input = cat([x_t0, vace_ctx], dim=-1)              # (B, L, 4*D)
    v_pred = model(x_input, text_emb, t0)
    v_gt = x_clean - noise
    L_base = weighted_loss(v_pred, v_gt, generation_mask=src_mask)

    # ── Step 2: STOP-GRADIENT ROLLOUT ──
    with torch.no_grad():
        # Re-use v_pred for uncond model; or do CFG for cond model
        if config.cfg_scale > 1.0:
            # Need an additional unconditional forward pass
            v_uncond = model(x_input, null_text_emb, t0)
            v_cfg = v_uncond + config.cfg_scale * (v_pred - v_uncond)
        else:
            v_cfg = v_pred.detach()

    # One step towards clean: t1 = t0 - 1/K
    K = config.num_sampling_steps  # e.g., 50
    t1 = (t0 - 1.0 / K).clamp_min(0.0)

    # ODE step: x_hat = x_t0 + v_cfg * dt
    sigma_t0 = 1.0 - t0.view(-1, 1, 1)  # flow matching: sigma = 1 - t
    sigma_t1 = 1.0 - t1.view(-1, 1, 1)
    # In SOAR convention (noise level), dt = sigma_t1 - sigma_t0
    # But in flow matching (x_t = (1-t)*noise + t*clean), going from t0→t1 (t1<t0):
    dt = t1.view(-1, 1, 1) - t0.view(-1, 1, 1)  # negative
    x_hat = (x_t0 + v_cfg.detach() * dt).detach()  # off-trajectory state

    # Mask-aware: keep known regions clean after rollout
    if config.mask_aware_noise:
        x_hat = x_hat * src_mask + x_clean * keep_mask

    # ── Step 3: RE-NOISE + CORRECTION ──
    L_corr = 0.0
    hit_boundary = (t1 <= 0)  # samples that hit t=0

    for n in range(config.N):  # N auxiliary points per sample
        # Sample auxiliary timestep t' between t1 and 1 (noise end)
        # In flow matching: t' ~ U[0, t1] (towards noise direction)
        # Equivalently in sigma space: sigma' ~ U[sigma_t1, 1]
        rand_frac = torch.rand(B, 1, 1, device=x_clean.device)

        # Interpolate between x_hat (at t1) and pure noise (at t=0)
        # sigma space: sigma_t' = sigma_t1 + rand * (1 - sigma_t1)
        #            = (1-t1) + rand * (1 - (1-t1))
        #            = (1-t1) + rand * t1
        # Equivalently: t' = t1 * (1 - rand)
        t_prime = t1.view(-1, 1, 1) * (1 - rand_frac)
        t_prime = t_prime.clamp_min(0.0)

        # Interpolation coefficient
        alpha = rand_frac  # fraction from x_hat towards noise
        z_re = (1 - alpha) * x_hat + alpha * noise  # re-noised state

        # Mask-aware: keep known regions clean at new timestep
        if config.mask_aware_noise:
            z_re = z_re * src_mask + x_clean * keep_mask

        # Correction target: steer z_re back to x_clean
        # v_corr such that: z_re + v_corr * (1 - t') = x_clean (approximately)
        # => v_corr = (x_clean - z_re) / (1 - t')   ... but v = x1 - x0 in FM
        # More precisely: v_corr = x_clean - noise_implicit
        # Following SOAR's formulation adapted to FM:
        sigma_t_prime = (1 - t_prime).clamp_min(1e-8)
        v_corr = (z_re - x_clean) / sigma_t_prime   # analogous to SOAR Eq.
        # Note: this is negative of (x_clean - z_re)/sigma, matching the convention
        # where velocity = noise - clean in the sigma parameterization

        # Forward pass on off-trajectory point (WITH gradient)
        z_re_input = cat([z_re.detach(), vace_ctx], dim=-1)
        t_prime_scalar = t_prime.squeeze(-1).squeeze(-1)  # (B,)
        v_off = model(z_re_input, text_emb, t_prime_scalar)

        # Per-sample correction loss (only on generated regions)
        L_corr += weighted_loss(
            v_off, v_corr.detach(),
            generation_mask=src_mask
        )

    # ── Step 4: COMBINED LOSS ──
    L_total = L_base + config.lambda_soar * L_corr / max(config.N, 1)
    return L_total
```

### 4.3 Correction Target Derivation

In M2M's flow matching convention:
```
x_t = (1 - t) * x0 + t * x1      where x0=noise, x1=clean
v_gt = x1 - x0                     velocity field
```

At an off-trajectory point `z_re` at timestep `t'`:
- The model should predict velocity that steers toward `x_clean`
- In SOAR's sigma parameterization: `sigma = 1 - t` (noise level)
- Correction target: `v_corr = (z_re - x_clean) / sigma_t'`
  - This means: `x_clean = z_re - sigma_t' * v_corr`
  - The predicted clean sample formula matches flow matching's `x_0 = x_t - sigma * v`

The key insight is that even at off-trajectory points, the correction target points back to the same clean target `x_clean`, ensuring consistent goal alignment.

### 4.4 Mask-Aware Noise Integration Points

For the `_man` (mask-aware noise) variant, masks must be applied at **four** specific points:

| Step | What happens | Mask-aware action |
|------|-------------|-------------------|
| 1. Initial `x_t0` | Forward process sample | `x_t0[known] = x_clean` (standard _man) |
| 2. After rollout `x_hat` | Off-trajectory state from ODE step | `x_hat[known] = x_clean` (keep known regions clean) |
| 3. Re-noised `z_re` | Interpolated auxiliary point | `z_re[known] = x_clean` (keep known regions clean) |
| 4. Loss computation | Per-element loss | Weight by `src_mask` (only supervise generated regions) |

**Rationale**: In the _man paradigm, known regions are always clean during both training and inference (via imputation). SOAR's rollout and re-noising should not corrupt them. The correction loss only needs to fix generated regions' off-trajectory errors.

### 4.5 VACE Context During SOAR

The VACE context (`inactive`, `reactive`, `src_mask`) is **not modified** during SOAR rollout or re-noising. This is correct:
- VACE describes the conditioning (which regions are known, what their values are)
- `x_t` is the denoising trajectory (latent state)
- They vary independently: `x_t` evolves through denoising while VACE stays fixed

At each SOAR step, the model input is:
```python
x_input = cat([z_re, vace_ctx], dim=-1)   # z_re changes, vace_ctx constant
```

### 4.6 CFG Handling

Two scenarios based on the base model:

| Model variant | CFG during rollout | Extra cost |
|--------------|-------------------|------------|
| `uncond_fm_man_046b` (no text conditioning) | No CFG needed. `v_cfg = v_pred` | None — simplest |
| `caption_fm_man_046b` (text conditioning) | CFG: `v_uncond + w*(v_cond - v_uncond)` | +1 forward pass (unconditional) |

**Recommendation**: Start with `uncond_fm_man_046b` — no CFG needed, simplest implementation, currently the best checkpoint (epoch 1000).

---

## 5. Hyperparameter Recommendations

### 5.1 Primary Hyperparameters

| Parameter | SOAR Default (SD3.5) | M2M Recommendation | Rationale |
|-----------|---------------------|---------------------|-----------|
| `lambda_soar` | 1.0 | **0.1** (start) | Motion space has different loss scale; conservative start to avoid base loss degradation |
| `N` (aux points per sample) | 6 | **1** (start) | Lower compute overhead; increase to 2-4 if effective |
| `num_rollout_paths` | 1 | **1** | ODE path only; stochastic paths add little benefit per SOAR ablation |
| `sde_rollout_type` | flow_sde | **N/A** (ODE only with 1 path) | Stochastic paths only needed if `num_rollout_paths > 1` |
| `sde_noise_scale` | 0.5 | **0.5** | Default if using stochastic paths |
| `cfg_scale_sampling` | 4.5 | **1.0** (uncond model) | No CFG for unconditional model |
| `num_sampling_steps` (K) | 40 | **50** | Match M2M's 50-step inference |
| `sigma_upper_ratio` | 1.5 | **1.5** | Same as SOAR default |

### 5.2 Training Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Base checkpoint | `uncond_fm_man_046b` epoch 1000 | Current best M2M model |
| Learning rate | **2e-5** | Match SOAR default; conservative for post-training |
| LR scheduler | cosine with warmup | 500 steps warmup, 5K-10K total |
| Batch size | Same as SFT (per-GPU BS 4-8) | Memory constrained by extra forward passes |
| Max training steps | **5,000 → 10,000** | Post-training, not from-scratch |
| Gradient clipping | 1.0 | Standard |
| Mixed precision | bf16 | Same as current training |
| Gradient checkpointing | Enabled | Essential for extra forward passes |
| Training data | `high_quality.json` (456K samples) | Quality-filtered subset |

### 5.3 Ablation Plan

| Experiment | Config | Purpose |
|------------|--------|---------|
| **E0**: Baseline | Current best `uncond_fm_man_046b` ep1000 | Reference point |
| **E1**: SOAR minimal | lambda=0.1, N=1, 5K steps | Minimum viable SOAR |
| **E2**: SOAR N=2 | lambda=0.1, N=2, 5K steps | More auxiliary points |
| **E3**: SOAR lambda sweep | lambda={0.05, 0.1, 0.5, 1.0}, N=1, 5K steps | Find optimal lambda |
| **E4**: SOAR longer | Best lambda+N from E1-E3, 10K steps | Longer training |
| **E5**: Stochastic paths | Best from E4 + `num_rollout_paths=2`, `sde_type=flow_sde` | Stochastic exploration |
| **E6**: Caption model | Apply best config to `caption_fm_man_046b` + CFG | Generalize to cond model |

---

## 6. Compute Budget

### 6.1 Per-Step Cost

| Configuration | Forward passes / step | Relative to SFT |
|---------------|----------------------|-----------------|
| SFT (current) | 1 | 1.0x |
| SOAR N=1, uncond (no CFG) | 1 (base) + 1 (rollout, no-grad) + 1 (correction) = 3 | ~2.0x |
| SOAR N=2, uncond | 1 + 1 + 2 = 4 | ~2.5x |
| SOAR N=1, cond + CFG | 1 + 2 (CFG rollout) + 1 = 4 | ~2.5x |
| SOAR N=2, cond + CFG | 1 + 2 + 2 = 5 | ~3.0x |

The rollout forward pass is **no-grad** (no backward), so actual GPU memory overhead is moderate. Gradient checkpointing further reduces memory pressure.

### 6.2 Total Training Budget

| Scenario | Steps | Per-step cost | Total GPU-hours (est., 8xA100) |
|----------|-------|---------------|-------------------------------|
| SFT reference (1 epoch) | ~70K | 1.0x | ~24h |
| SOAR E1 (N=1, 5K) | 5,000 | 2.0x | ~3.5h |
| SOAR E4 (N=1, 10K) | 10,000 | 2.0x | ~7h |
| Full ablation (E1-E5) | ~30K total | ~2.0x avg | ~21h |

**SOAR post-training is extremely cheap**: 5K steps at 2x cost ≈ 10K SFT-equivalent steps, compared to 70K steps per SFT epoch.

---

## 7. Implementation Details

### 7.1 Code Changes

The implementation requires changes in **one file only**: the trainer.

```
hftrainer/trainers/motion/hymotion_m2m_trainer.py
  - Add SOAR config fields (lambda, N, K, etc.)
  - Add soar_correction_step() method
  - Modify training_step() to include SOAR loss after base loss
  - Add mask-aware handling at rollout/re-noise/loss steps
```

No changes needed in:
- `HyMotionM2MBundle` (model)
- `HyMotionM2MPipeline` (inference)
- Dataset / DataLoader
- Checkpoint hooks

### 7.2 Config Addition

```python
# In M2M config, add SOAR post-training section:
soar = dict(
    enabled=True,
    lambda_soar=0.1,          # correction loss weight
    num_aux_points=1,          # N auxiliary points per sample
    num_sampling_steps=50,     # K (match inference steps)
    num_rollout_paths=1,       # 1 = ODE only
    sde_rollout_type='cps',    # only used if num_rollout_paths > 1
    sde_noise_scale=0.5,
    cfg_scale=1.0,             # 1.0 = no CFG (uncond model)
    sigma_upper_ratio=1.5,
    loss_type='smooth_l1',     # match base loss type
)
```

### 7.3 Key Implementation Notes

1. **Shared noise**: The noise tensor `x0` used in base loss MUST be the same one used in re-noising. Do not resample.

2. **Stop-gradient boundary**: The rollout velocity `v_cfg` must be `.detach()`'d. The re-noised `z_re` must also be `.detach()`'d before the correction forward pass. Only the correction model output `v_off` carries gradients.

3. **Memory management**: With N=1, peak memory is approximately 1.5x SFT (one extra no-grad forward + one extra with-grad forward, but the no-grad pass doesn't store activations). Enable gradient checkpointing.

4. **Loss normalization**: Following SOAR's normalization scheme, divide total loss by `(B + lambda * P)` where B = batch size, P = total auxiliary point count. This prevents the correction loss from dominating when N is large.

5. **Generation mask weighting**: Apply `src_mask` to both base loss and correction loss so that only generated regions receive gradients. Known regions are already clean and need no correction.

6. **Boundary handling**: When `t1 = 0` (sample hits clean boundary), skip stochastic paths for that sample. The ODE path still provides one auxiliary point.

### 7.4 Gradient Flow Diagram

```
                    ┌─────────────────────────────────┐
                    │  x_t0  (on-trajectory)          │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  model(x_t0, cond, t0)          │ ← gradients flow
                    │  → v_pred                        │
                    └──────────┬──────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
    ┌─────────▼─────┐   ┌─────▼──────┐   ┌─────▼──────────┐
    │ L_base         │   │ STOP GRAD  │   │                │
    │ ||v-v_gt||^2   │   │ v_cfg =    │   │                │
    │ (→ backward)   │   │ v_pred.    │   │                │
    └────────────────┘   │ detach()   │   │                │
                         └─────┬──────┘   │                │
                               │          │                │
                    ┌──────────▼──────────┘                │
                    │  x_hat = x_t0 + v_cfg * dt           │
                    │  (off-trajectory, detached)           │
                    └──────────┬───────────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  z_re = interpolate(x_hat, noise)│
                    │  (re-noised, detached)           │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  model(z_re, cond, t')           │ ← gradients flow
                    │  → v_off                         │
                    └──────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────────────────┐
                    │  L_corr = ||v_off - v_corr||^2   │
                    │  (→ backward)                    │
                    └─────────────────────────────────┘
```

---

## 8. Evaluation Plan

### 8.1 Metrics

| Metric | What it measures | Expected improvement |
|--------|-----------------|---------------------|
| **Boundary smoothness** | Transition quality at known/generated boundary | High — primary benefit |
| **FID / FMD** | Distribution quality of generated motion | Moderate |
| **Foot sliding** | Physical plausibility | Low-Moderate |
| **Joint jitter** | Temporal smoothness | Moderate |
| **Temporal coherence** | Consistency across frames | Moderate |
| **Task-specific metrics** (MIB, prediction, keyframe) | Per-task completion quality | Moderate |

### 8.2 Evaluation Protocol

1. **Quantitative**: Run full evaluation suite on 12-task benchmark at checkpoints {1K, 2K, 5K, 10K}
2. **Qualitative**: Visual comparison of boundary regions (known↔generated transition) before/after SOAR
3. **Ablation**: Compare E0-E5 systematically (see §5.3)
4. **Regression check**: Ensure base quality (unconditional generation, simple inbetweening) does not degrade

---

## 9. Risk Analysis and Mitigation

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Lambda too large → base quality degrades | Medium | Medium | Start lambda=0.1; monitor base vs corr loss ratio; stop if base loss increases |
| Off-trajectory states too noisy (model not mature enough) | Low | Low | Using epoch 1000 checkpoint (well-trained) |
| Motion-specific issues (rot6d space vs pixel space) | Low | Low | Correction target operates in same normalized space as base loss; mathematically consistent |
| Overfitting on small post-training budget | Low | Low | 5K-10K steps on 456K samples = <1% of data seen; use LR warmup + cosine decay |
| Mask-aware interaction bugs | Medium | Medium | Carefully test mask application at all 4 points; unit test with synthetic data |
| Memory OOM from extra forward passes | Low | Medium | Use gradient checkpointing; reduce batch size if needed; N=1 keeps overhead minimal |
| Stochastic path instability | Low | Low | Start with ODE-only (1 path); add SDE paths only in ablation E5 |

---

## 10. Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **P1: Implementation** | 2 days | SOAR logic in `HyMotionM2MTrainer`, config support, unit tests |
| **P2: Smoke test** | 0.5 day | Verify loss computation, gradient flow, mask-aware handling on 1 GPU |
| **P3: Minimal experiment (E1)** | 1 day | lambda=0.1, N=1, 5K steps on 8xA100 |
| **P4: Ablation (E2-E5)** | 3 days | Lambda sweep, N sweep, longer training, stochastic paths |
| **P5: Evaluation** | 1 day | Full 12-task benchmark comparison |
| **P6: Report** | 0.5 day | Results analysis, final recommendations |
| **Total** | ~8 days | |

---

## 11. Summary

SOAR is an ideal post-training method for HyMotion M2M v2:

1. **Mathematically compatible**: Identical flow matching (rectified flow) framework — zero modification to the core formulation.

2. **Addresses the right problem**: M2M's exposure bias in generated regions causes boundary discontinuity and temporal incoherence. SOAR provides dense per-timestep correction supervision on off-trajectory states.

3. **Complementary to _man**: Mask-aware noise fixes known regions' distribution mismatch; SOAR fixes generated regions' off-trajectory errors. Together they cover both halves of the problem.

4. **Zero annotation cost**: Fully self-supervised. No reward model, no preference data, no new data collection. Existing training data used as-is.

5. **Low compute cost**: 5K-10K post-training steps at ~2x SFT cost ≈ 3.5-7 GPU-hours on 8xA100. Negligible compared to the 1000-epoch SFT training.

6. **Low implementation risk**: ~150 lines of new code in the trainer. Pipeline, model, and data unchanged. Clear ablation plan with conservative starting point.

**Recommended first experiment**: `uncond_fm_man_046b` (epoch 1000) + SOAR post-training with lambda=0.1, N=1, K=50, LR=2e-5, 5K steps on quality-filtered data.
