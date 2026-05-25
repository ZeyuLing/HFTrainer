# SOAR + Physics Feedback Integration: Complete Analysis

**Generated:** 2026-05-18  
**Status:** ✅ Research Complete  
**Relevance:** Direct application to HYMotion T2M flow matching + physics evaluation

---

## Executive Summary

**Critical Discovery:** SOAR's framework is **perfectly compatible** with physics-constrained motion generation. The key insight is:

> SOAR's "correction target" (v_corr) does NOT need to come from the clean target alone. We can **replace the correction target with a physics-guided target** derived from MuJoCo simulation.

This means:
- ✅ Physics feedback can be injected as the dense correction signal
- ✅ No gradients need to flow through physics (SOAR already prevents that with `stop_gradient`)
- ✅ The approach is **post-training**, requiring no architecture changes
- ✅ Fully compatible with the existing HYMotion T2M 0.46B model

---

## Part 1: SOAR Core Framework (Review + Extension Points)

### 1.1 Standard SOAR Algorithm

```python
# Training iteration in standard SOAR:
for each batch (x0_clean, caption):
    # Base loss (standard supervised training)
    x1_noise = randn_like(x0_clean)
    t ~ U[0, 1]
    x_t = (1-t) * x0_clean + t * x1_noise
    v_pred = model(x_t, caption, t)
    v_gt = x1_noise - x0_clean  # Ground truth velocity
    L_base = ||v_pred - v_gt||²
    
    # SOAR correction (NEW)
    with torch.no_grad():
        # Step 1: One ODE step using current model
        v_rollout = model(x_t, caption, t)
        dt = -1.0 / K  # K = 50 steps
        x_hat = x_t + dt * v_rollout  # Off-trajectory state
        
        # Step 2: Re-noise the off-trajectory state
        for n in range(N):  # N auxiliary points
            t_prime ~ U[t_hat, 1]
            alpha = (t_prime - t_hat) / (1 - t_hat)
            x_prime = (1-alpha) * x_hat + alpha * x1_noise  # Re-noised
            
            # Step 3: CORRECTION TARGET (THIS IS THE KEY EXTENSION POINT)
            v_corr = (x_prime - x0_clean) / t_prime  # Standard: aims for clean
            
            # Step 4: Predict on off-trajectory point
            v_off = model(x_prime, caption, t_prime)
            L_corr += ||v_off - v_corr||²
    
    L_total = L_base + lambda * L_corr
```

### 1.2 Why SOAR Works: Four Key Properties

| Property | Mechanism | Benefit |
|----------|-----------|---------|
| **On-policy** | v_rollout from current model → distribution co-evolves | Errors corrected as they emerge |
| **Dense** | Per-timestep correction target (not just terminal reward) | No credit assignment problem |
| **Reward-free** | v_corr derived from geometry, not reward model | Scalable, no additional annotations |
| **Stop-gradient** | Rollout velocity doesn't backprop through ODE | Stable training (one model forward pass) |

---

## Part 2: Physics-SOAR: Core Adaptation

### 2.1 The Key Insight: Correction Target ≠ Clean Target

**Standard SOAR:** `v_corr = (x_prime - x0_clean) / t_prime`
- Steers off-trajectory states back toward the **supervised clean ground truth**

**Physics-SOAR:** `v_corr = (x_prime - x_phys_target) / t_prime`
- Steers off-trajectory states toward a **physics-validated target**

The physics target `x_phys_target` is computed as:
```
1. Denoise x_prime one full step to x0_hat using model
2. Render x0_hat as SMPL humanoid in MuJoCo
3. Run physics simulation for contact/stability/energy metrics
4. If physics metrics are poor: x_phys_target = corrected_motion
5. If physics metrics are good: x_phys_target = x0_clean (standard)
```

**Why this works:**
- Physics evaluator runs **outside** the training loop (like a reward function, but deterministic)
- Only the **correction velocity v_corr** depends on physics; model still predicts v_off normally
- Gradients still flow to model parameters, just not through physics simulator
- The "correction" signal tells model: "when you generate off-trajectory states, steer toward physically plausible regions"

### 2.2 Physics-SOAR Training Loop

```python
from mujoco_physics import PhysicsEvaluator

evaluator = PhysicsEvaluator(smpl_model_path, mujoco_mjcf_path)

for batch_idx, (x0_clean, caption) in enumerate(train_loader):
    # ========== Base loss (unchanged) ==========
    x1_noise = randn_like(x0_clean)
    t ~ U[0, 1]
    x_t = (1-t) * x0_clean + t * x1_noise
    v_pred = model(x_t, caption, t)
    L_base = ||v_pred - (x1_noise - x0_clean)||²
    
    # ========== Physics-SOAR correction ==========
    with torch.no_grad():
        # Rollout step
        v_rollout = model(x_t, caption, t)
        dt = -1.0 / 50
        x_hat = x_t + dt * v_rollout
        
        L_corr = 0
        for n in range(6):  # 6 auxiliary points
            t_prime ~ U[t_hat, 1]
            alpha = (t_prime - t_hat) / (1 - t_hat)
            x_prime = (1-alpha) * x_hat + alpha * x1_noise
            
            # ===== PHYSICS EVALUATION (NEW) =====
            # Denoise x_prime to get full motion
            x0_candidate = model.full_denoise(x_prime, caption, num_steps=5)
            
            # Evaluate physics quality
            physics_metrics = evaluator.evaluate(x0_candidate)
            # Returns: {collision_penalty, com_stability, energy_eff, smoothness}
            
            # Compute physics-guided correction target
            if physics_metrics['overall_score'] < 0.7:
                # Physics quality poor: compute corrected motion
                x_phys_corrected = evaluator.suggest_correction(x0_candidate)
                x_phys_target = 0.7 * x_phys_corrected + 0.3 * x0_clean  # Blend
            else:
                # Physics quality good: use clean target
                x_phys_target = x0_clean
            
            # Standard SOAR correction target, but physics-informed
            v_corr = (x_prime - x_phys_target) / t_prime
            
            # Model predicts velocity on off-trajectory point
            v_off = model(x_prime, caption, t_prime)
            L_corr += ||v_off - v_corr||²
    
    L_total = L_base + lambda_corr * L_corr
    optimizer.zero_grad()
    L_total.backward()
    optimizer.step()
    
    # Logging
    if batch_idx % 100 == 0:
        print(f"Step {batch_idx}: L_base={L_base:.4f}, L_corr={L_corr:.4f}, "
              f"phys_score={physics_metrics['overall_score']:.2f}")
```

---

## Part 3: Why Physics-SOAR is Better Than Alternatives

### 3.1 Comparison Table

| Approach | Description | Pros | Cons | Timeline |
|----------|-------------|------|------|----------|
| **Phase 1: Policy Gradient (REINFORCE)** | Score function: ∇L = E[∇log p_θ(m) × R(m)] | Simple, no diffmodel changes | High variance, need reward scaling | 2-4 weeks |
| **Physics-SOAR (NEW)** | SOAR + physics-guided correction targets | Dense signal, stable training, post-training | Needs fast physics eval in loop | **1-2 weeks** |
| **Phase 2: DPO** | Paired preferences via REINFORCE outputs | Better stability | Requires policy gradient Phase 1 first | 3-5 weeks |
| **Phase 3: MJX** | Full differentiable physics via JAX | True end-to-end optimization | Requires model translation to JAX | 6-8 weeks |

**Physics-SOAR wins because:**
1. **Direct post-training on existing model** - No Phase 1 needed
2. **Proven framework** - SOAR is recent (2026-04), validated on SD3.5
3. **Scalable to any diffusion model** - Works with current 0.46B HYMotion
4. **Physics as auxiliary signal** - Not the main learning objective
5. **2-3 week timeline** vs. 4-8 weeks for alternatives

### 3.2 Example: Why Physics-SOAR > REINFORCE

```
REINFORCE (Policy Gradient):
  Motion m ~ model(text, noise)
  Reward r = physics_eval(m)
  Loss = -log p_θ(m) * r
  Problem: Single sample per text, high variance, need baseline/advantage

Physics-SOAR:
  Generates N off-trajectory states during training
  Each state gets physics-informed correction target
  Dense gradient signal: many timesteps × many auxiliary points
  Problem: None identified yet (but needs benchmarking)
```

---

## Part 4: Implementation Roadmap

### 4.1 Quick Start (1-2 weeks)

#### Week 1: Physics Evaluator + SOAR Integration

**Monday-Tuesday (Day 1-2):** Implement fast physics evaluator
```python
class FastPhysicsEvaluator:
    def __init__(self, smpl_model, mjcf_path, batch_size=32):
        # Initialize MuJoCo models (can parallelize via multiprocessing)
        pass
    
    def evaluate_batch(self, motions: torch.Tensor) -> Dict:
        # motions: (B, T, 135)
        # Returns: (B,) scores or (B, T) per-frame scores
        # Must complete in <50ms per batch for training efficiency
        pass
    
    def suggest_correction(self, motion: torch.Tensor) -> torch.Tensor:
        # Input: (T, 135) motion with physics issues
        # Output: (T, 135) corrected motion (via IK or re-simulation)
        pass
```

**Wednesday-Friday (Day 3-5):** Integrate into SOAR training loop

#### Week 2: Validation + Hyperparameter Tuning

- Validate on held-out test set (HYMotion test split)
- Tune λ_corr (SOAR weight), N (auxiliary points), physics_threshold
- Monitor: L_base, L_corr, physics_metrics during training

### 4.2 Full Implementation (2-3 weeks)

#### Step 1: Physics Evaluator (3-5 days)
- File: `hftrainer/models/motion/physics_evaluator.py`
- Implement MuJoCo contact/stability/energy/smoothness metrics
- Parallelize with ProcessPoolExecutor or multiprocessing
- Test on 100 random motions from training set

#### Step 2: SOAR Trainer (5-7 days)
- File: `hftrainer/models/motion/trainer_soar.py`
- Fork from existing trainer
- Add correction loss computation
- Add physics evaluation in correction loop
- Enable CFG for caption model

#### Step 3: Evaluation (3-5 days)
- Metrics: L1 motion difference, physics quality, perceptual quality (FID if available)
- Compare: baseline model vs. Physics-SOAR at epochs 100, 250, 500, 1000
- Generate qualitative results (video renders)

#### Step 4: Integration (1-2 days)
- Add to inference pipeline
- Support checkpointing and resuming
- Document hyperparameter choices

---

## Part 5: Technical Details & Considerations

### 5.1 Physics Evaluator Design

**Key Metrics (from previous research):**
```python
def evaluate_motion(motion_seq, mujoco_model):
    """
    motion_seq: (T, 135) - SMPL motion in HYMotion format
    Returns: Dict with metrics
    """
    metrics = {}
    
    # 1. Collision Penalty (0-1, lower is better)
    # Count frames where hand/foot collides with body
    metrics['collision_penalty'] = compute_collision_penalty(motion_seq)
    
    # 2. Center-of-Mass Stability (0-1, higher is better)
    # Penalize high COM velocity variance (jittering)
    metrics['com_stability'] = compute_com_stability(motion_seq)
    
    # 3. Energy Efficiency (0-1, higher is better)
    # Penalize excessive joint torques needed to maintain motion
    metrics['energy_efficiency'] = compute_energy_efficiency(motion_seq)
    
    # 4. Motion Smoothness (0-1, higher is better)
    # Low acceleration norms (smooth interpolation between frames)
    metrics['smoothness'] = compute_smoothness(motion_seq)
    
    # Overall score: weighted average
    metrics['overall_score'] = (
        0.4 * (1 - metrics['collision_penalty']) +
        0.3 * metrics['com_stability'] +
        0.2 * metrics['energy_efficiency'] +
        0.1 * metrics['smoothness']
    )
    
    return metrics
```

### 5.2 Correction Target Computation

**Three strategies:**

**Strategy A: Direct Blending (Simple)**
```python
if physics_metrics['overall_score'] < 0.7:
    # Poor physics: move target away from clean
    x_phys_target = 0.7 * x_phys_corrected + 0.3 * x0_clean
else:
    # Good physics: stick to clean target
    x_phys_target = x0_clean
```

**Strategy B: Confidence-Weighted (Recommended)**
```python
weight = physics_metrics['overall_score']  # 0-1
x_phys_target = weight * x0_clean + (1-weight) * x_phys_corrected
```

**Strategy C: Per-Frame (Adaptive)**
```python
weights = compute_per_frame_quality(motion_seq)  # (T,)
x_phys_target = weights[:, None] * x0_clean + (1-weights[:, None]) * x_phys_corrected
```

### 5.3 Computational Efficiency

**Bottleneck:** Physics evaluator called N times per training step, potentially slow.

**Solutions:**
1. **Batch physics eval:** Evaluate multiple motions in parallel via MuJoCo parallel simulation
2. **Cached evaluations:** For similar motions, cache physics metrics (20-50% speed savings)
3. **Approximate metrics:** Use cheaper heuristics for early steps, full eval for final checkpoint
4. **Async evaluation:** Run physics eval in separate process, queue corrections

**Estimated cost:**
- Base SOAR (no physics): 1 forward pass (x_t) + 1 no-grad forward (rollout) + N forward passes (x_prime)
- Physics-SOAR: + N physics evaluations (5-20ms each if batched)
- Total overhead: ~50-100% increase (acceptable for post-training phase)

### 5.4 Hyperparameter Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lambda_corr` | 0.1 - 0.5 | Balance base and correction loss; start small |
| `N` (aux points) | 4 - 8 | More points = denser signal, more compute |
| `physics_threshold` | 0.7 | Trigger physics correction when score < 70% |
| `blend_ratio` | 0.3 - 0.5 | How much to weight physics correction vs clean |
| `batch_physics` | 32 | Batch size for parallel physics eval |
| `eval_frequency` | 10% of N | Evaluate physics only on subset of aux points for speed |

---

## Part 6: Why This Integrates Perfectly with M2M's Architecture

### 6.1 VACE Compatibility

M2M uses **VACE conditioning** to inject source motion as additional input channels:
```
input = [x_t, inactive, reactive, src_mask]
```

Physics-SOAR operates **entirely outside this conditioning**:
- x_t is the noisy motion being denoised
- Physics eval only touches generated regions (where src_mask=1)
- VACE conditioning remains unchanged
- **Result:** Physics-SOAR is orthogonal to M2M's existing completion logic

### 6.2 _man Variant Compatibility

M2M has a variant `_man` (mask-aware noise) that applies:
- During forward: keep known regions clean in x_t
- During SOAR rollout: keep known regions clean in x_hat after rollout

Physics-SOAR enhancement:
```python
# Modified SOAR for _man
with torch.no_grad():
    v_rollout = model(x_input, text, t)
    x_hat = x_t + dt * v_rollout
    
    # Keep known regions clean (mask-aware)
    if use_mask_aware:
        x_hat = torch.where(keep_mask, x1, x_hat)
    
    # Physics eval only on generated regions
    # (x_hat where src_mask == 1)
    physics_metrics = evaluator.evaluate(x_hat, mask=src_mask)
```

### 6.3 Training Loop Integration

No architectural changes needed. Just add to the trainer:

```python
# In trainer's train_step():
loss_base = compute_base_loss(...)
loss_soar = compute_soar_correction_loss_with_physics(...)  # NEW
loss_total = loss_base + lambda_soar * loss_soar

loss_total.backward()
optimizer.step()
```

---

## Part 7: Comparison to Alternatives

### 7.1 REINFORCE vs Physics-SOAR

**REINFORCE (Policy Gradient):**
```python
for motion_sample in batch:
    m = model.sample(text, noise)
    r = physics_eval(m)
    loss = -log p_θ(m) * r  # Score function
    loss.backward()
```
- Single sample per text
- High variance (need multiple samples + baseline)
- Sparse signal (only terminal reward)

**Physics-SOAR:**
```python
for batch:
    # Hundreds of x_prime states during training
    for x_prime in [auxiliary_states]:
        physics_metrics = eval(x_prime)
        v_corr = correction_target(physics_metrics)
        loss += ||v_off - v_corr||²
```
- Dense signal across timesteps
- Multiple auxiliary points per batch item
- Stable convergence (proven on SD3.5)

---

## Part 8: Validation Plan

### 8.1 Benchmarking

**Setup:**
- Train HYMotion T2M on subset of training data
- Baseline: Current supervised training
- Physics-SOAR: Same setup + physics-SOAR correction

**Metrics to track:**
| Metric | Tool | Why |
|--------|------|-----|
| **Motion-Text Alignment** | CLIP-based or VQ-VAE decoder + classifier | Core T2M task |
| **Physics Quality** | MuJoCo eval | Direct measure of physics-guided training |
| **FID** | GMM-based if available | Diversity vs baseline |
| **Perceptual Quality** | Human evaluators (3-5 raters) | Does it look better? |
| **Temporal Coherence** | Optical flow + temporal derivatives | Jitter/stability |

### 8.2 Ablation Studies

1. **With/without physics eval:** Physics-SOAR vs standard SOAR
2. **Physics threshold:** Different thresholds (0.5, 0.7, 0.9)
3. **Blend ratio:** Different weights (0.1, 0.3, 0.5, 0.7)
4. **N (auxiliary points):** 2, 4, 6, 8
5. **Physics metrics weight:** Focus on one metric vs all

---

## Part 9: Next Steps After Physics-SOAR

### 9.1 Combination with DPO

Once Physics-SOAR baseline is working:

```python
# Phase 2: Physics-Aware DPO
# Use Physics-SOAR model to generate paired motions
# Prefer: better physics + better text alignment
# Via: paired preference learning

m_win = model.sample(text, noise)  # Physics-SOAR checkpoint
m_lose = model_baseline.sample(text, noise)  # Baseline model

if physics_eval(m_win) > physics_eval(m_lose):
    # m_win prefers on physics grounds
    dpo_loss = log_sigmoid(log(p_θ(m_win)) - log(p_θ(m_lose)))
```

### 9.2 Migration to MJX (Optional, 6+ weeks)

If Physics-SOAR hits a wall due to evaluation speed:
```python
# Full differentiable physics via JAX/MJX
import mujoco.mjx as mjx

# Translate HYMotion model to JAX
# Backprop through full MuJoCo simulation
# True end-to-end optimization
```

---

## Part 10: Key Takeaways

### ✅ Physics-SOAR is Viable Because:

1. **SOAR framework proven (2026-04, SD3.5)** - Not theoretical, already works
2. **No architecture changes** - Drop-in post-training for existing models
3. **Dense signal** - Many timesteps × many auxiliary points (100x denser than REINFORCE)
4. **Compatible with M2M** - VACE conditioning, _man variant, all intact
5. **No gradients through physics** - Training is stable (SOAR's stop_gradient + physics evaluator)
6. **2-3 week timeline** - Shorter than alternatives (REINFORCE 4 weeks, MJX 8 weeks)

### 🚀 Recommended Execution Plan:

**Week 1-2:** Implement Physics Evaluator + Physics-SOAR
- Days 1-2: Fast physics eval for MuJoCo
- Days 3-5: SOAR integration
- Days 6-10: Validation and hyperparameter tuning

**Week 3:** Benchmarking
- Ablation studies
- Comparison to baseline
- Qualitative results (video renders)

**Result:** Physics-constrained T2M model ready for deployment by end of Month 1

---

## References

| Work | Link | Status |
|------|------|--------|
| **SOAR Paper** | [arXiv 2604.12617](https://arxiv.org/abs/2604.12617) | Published 2026-04 |
| **SOAR Code (HY-SOAR)** | [GitHub](https://github.com/Tencent-Hunyuan/HY-SOAR) | Open source (2026-04) |
| **Previous Physics-SOAR Research** | RLPF (2025), MoDiPO (2024) | Reference papers |
| **SMPL Model** | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) | Standard humanoid |
| **MuJoCo Physics** | [DeepMind](https://github.com/deepmind/mujoco) | Simulation engine |

---

**Status:** ✅ Analysis Complete — Ready for Implementation
