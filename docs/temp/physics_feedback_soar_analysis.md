# Using Physics Feedback to Improve Motion Generation: Comprehensive Reference Analysis

**Document Purpose**: Analyze two key reference works (SOAR and embodied intelligence integration) and their applicability to physics-feedback-based motion generation.

**Date**: 2026-05-18 | **Working Directory**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## Executive Summary

### Three Core Findings

1. **SOAR (Self-Correction for Optimal Alignment and Refinement)**
   - ✅ **Directly applicable to M2M v2** with zero architectural changes
   - **Framework match**: Identical rectified flow / flow matching formulation
   - **Problem solved**: Exposure bias in generated regions (off-trajectory correction)
   - **Data cost**: Zero — fully self-supervised, uses only existing training data
   - **Physics potential**: SOAR's "correction oracle" (clean target) could be **replaced with physics-validated trajectories**

2. **Physics Feedback Integration Pathways** (from survey research)
   - **Direction A (Motion → Robot)**: HyMotion → Retargeting → RL Tracking → Physics Feedback Loop
   - **Direction B (Physics → Motion)**: Physics Validator → Reward Signal → RLPF/DPO Fine-tuning
   - **Hybrid approach**: SOAR post-training with physics-validated correction targets

3. **No Existing Reward/DPO Infrastructure in M2M**
   - ❌ No current RL/DPO/RLHF components in `hftrainer/models/motion/`
   - ❌ Motion domain lacks reward model scaffolding (unlike `ref_repo/HY-SOAR/sora/flow_grpo/`)
   - ✅ **Opportunity**: Physics validator → Reward model → DPO as Phase 2 after SOAR Phase 1

---

## Part 1: SOAR Reference Work Analysis

### 1.1 What is SOAR?

**SOAR = Self-Correction for Optimal Alignment and Refinement in Diffusion Models**

- **Paper**: arXiv:2604.12617 (April 2026)
- **Authors**: Tencent Hunyuan team + collaborators
- **Task**: Text-to-Image post-training on SD3.5-Medium (rectified flow)
- **Results**: +11% GenEval, +5% OCR, outperforms RL (GRPO) on human preference metrics

### 1.2 The Core Problem: Exposure Bias

| Stage | On-trajectory? | Signal | Issue |
|-------|---|---|---|
| **SFT (Standard)** | ✅ Yes (GT forward process) | Dense | ❌ Inference uses model's own predictions (off-trajectory) |
| **RL (GRPO)** | ✅ Yes (model rollout) | Sparse (terminal reward) | ❌ Credit assignment hard, reward hacking |
| **SOAR** | ⚠️ Partially (1-step rollout) | **Dense per-step correction** | ✅ Combines SFT + RL benefits |

**Why exposure bias matters for M2M**:
- M2M uses **50-step ODE integration** → more error accumulation than image generation
- **Temporal data**: Early-frame errors propagate through self-attention to later frames
- **VACE conditioning**: Generation errors in one region affect model's understanding of known regions
- **Post-hoc boundaries**: Hard transitions between known and generated regions (currently masked by post-processing)

### 1.3 SOAR Algorithm: Core Idea

```
For each training batch (x_clean, noise):
  
  1. BASE LOSS (standard SFT):
     x_t0 = (1-t0)*noise + t0*x_clean        # on-trajectory
     v_pred = model(x_t0, cond, t0)
     L_base = ||v_pred - (x_clean - noise)||²
  
  2. STOP-GRADIENT ROLLOUT (simulate inference):
     with torch.no_grad():
        v_cfg = model(x_t0, cond, t0)        # or with CFG
     x_hat = x_t0 + v_cfg * dt               # off-trajectory state
  
  3. RE-NOISE + CORRECTION (fix the off-trajectory error):
     for n in range(N):
        z_re = interpolate(x_hat, noise)     # re-noised auxiliary point
        v_corr = (z_re - x_clean) / sigma    # correction target
        v_off = model(z_re, cond, t')
        L_corr += ||v_off - v_corr||²
  
  4. COMBINED LOSS:
     L_total = L_base + lambda * L_corr
```

**Key insight**: The correction target `v_corr = (z_re - x_clean) / sigma` is **self-supervised** — it comes entirely from the clean training target, not from any external signal.

### 1.4 Why SOAR is Directly Applicable to M2M v2

| Aspect | SD3.5-Medium | HyMotion M2M |
|--------|---|---|
| **Paradigm** | Rectified flow | Flow matching (rectified flow) |
| **Noise schedule** | `x_t = (1-t)*noise + t*clean` | `x_t = (1-t)*noise + t*clean` |
| **Velocity target** | `v = noise - clean` | `v = x0 - x1` |
| **ODE solver** | Euler | Midpoint / Euler |
| **Architecture** | DiT | MMDiT |

✅ **Math is identical**. Only domain changes (pixels → motion). Zero architectural changes needed.

### 1.5 SOAR + M2M: Adapted Algorithm for _man Variant

The key adaptation point is **mask-aware noise** (known regions stay clean):

```python
def soar_training_step_m2m(
    model,          # HunyuanMotionMMDiT
    x_clean,        # (B, L, D) normalized clean motion
    x0,             # (B, L, D) noise
    src_motion,     # source (for VACE)
    src_mask,       # 0=known, 1=generate
    text_emb,
    config
):
    B, L, D = x_clean.shape
    keep_mask = 1 - src_mask
    
    # ── Step 1: BASE LOSS ──
    t0 ~ U[0, 1]
    x_t0 = (1 - t0) * x0 + t0 * x_clean
    
    # Mask-aware: keep known regions clean
    x_t0 = x_t0 * src_mask + x_clean * keep_mask
    
    vace_ctx = prepare_vace_input(src_motion, src_mask)
    x_input = cat([x_t0, vace_ctx], dim=-1)
    v_pred = model(x_input, text_emb, t0)
    v_gt = x_clean - x0
    L_base = weighted_loss(v_pred, v_gt, generation_mask=src_mask)
    
    # ── Step 2: ROLLOUT ──
    with torch.no_grad():
        v_cfg = v_pred.detach()  # or CFG
    
    t1 = max(t0 - 1/K, 0)
    dt = t1 - t0  # negative (towards clean)
    x_hat = x_t0 + v_cfg * dt
    
    # Mask-aware: keep known regions clean after rollout
    x_hat = x_hat * src_mask + x_clean * keep_mask
    
    # ── Step 3: RE-NOISE + CORRECTION ──
    L_corr = 0
    for n in range(N):
        t_prime ~ U[0, t1]
        alpha = rand_frac
        z_re = (1 - alpha) * x_hat + alpha * x0
        
        # Mask-aware: keep known regions clean
        z_re = z_re * src_mask + x_clean * keep_mask
        
        sigma_t_prime = (1 - t_prime).clamp_min(1e-8)
        v_corr = (z_re - x_clean) / sigma_t_prime
        
        z_re_input = cat([z_re.detach(), vace_ctx], dim=-1)
        v_off = model(z_re_input, text_emb, t_prime)
        L_corr += weighted_loss(v_off, v_corr.detach(), generation_mask=src_mask)
    
    L_total = L_base + lambda_soar * L_corr / N
    return L_total
```

### 1.6 SOAR Hyperparameters for M2M

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lambda_soar` | **0.1** (start) | Conservative; motion loss scale differs from pixels |
| `N` (aux points) | **1** (start) | Low overhead; increase to 2-4 if beneficial |
| `K` (sampling steps) | **50** | Match M2M's inference steps |
| `LR` | **2e-5** | Match SOAR default |
| `Warmup steps` | **500** | Standard |
| `Total post-train steps` | **5K–10K** | Post-training, not from-scratch |
| `Base checkpoint` | `uncond_fm_man_046b` epoch 1000 | Current best M2M model |
| `Batch size` | Same as SFT | 4-8 per GPU |
| `Data` | `high_quality.json` (456K samples) | Quality-filtered |

### 1.7 Implementation Complexity

**Code changes**:
- ~150 lines in `HyMotionM2MTrainer`
- Add `soar_correction_step()` method
- Modify `training_step()` to include SOAR loss
- No changes to model, pipeline, or data

**Compute cost**:
- SFT: 1 forward pass
- SOAR N=1: 1 (base) + 1 (rollout, no-grad) + 1 (correction) = **3 passes ≈ 2x cost**
- GPU-hours for 5K steps on 8xA100: **~3.5 hours** (negligible vs. 1000-epoch SFT)

### 1.8 Expected Benefits for M2M

| Benefit | Mechanism | Priority |
|---------|-----------|----------|
| **Reduced boundary discontinuity** | Off-trajectory states learn to correct back to clean target | 🔴 HIGH — current main pain point |
| **Improved temporal coherence** | Dense per-timestep correction across all 50 ODE steps | 🟡 MEDIUM |
| **Better long-sequence quality** | Later ODE steps trained on on-policy distribution | 🟡 MEDIUM |
| **Safe improvement** | L_base preserved; L_corr is additive supervision | ✅ Safe |
| **Complementary to _man** | _man fixes known regions; SOAR fixes generated regions | ✅ Orthogonal |

---

## Part 2: Physics Feedback Integration Pathways

### 2.1 Current Research Landscape (2026)

The embodied intelligence community (PARC, ASAP, VideoMimic, UH-1) has developed mature pipelines for:
- **Motion Generation → Robot Retargeting → Physics Validation → Data Feedback Loop**

### 2.2 Direction B: Physics Feedback → Motion Improvement

From `docs/temp/survey_motion_gen_embodied_v2_20260508.md`:

```
Motion Gen Model (HyMotion)
         ↓
    Retargeting (GMR, BeyondMimic)
         ↓
    Robot Simulation (IsaacGym, MuJoCo)
         ↓
    RL Tracking (PHC, ASAP)
         ↓
    Physics Validator
         ├─→ Pass (physically feasible)
         └─→ Fail (foot skating, collision, etc.)
             ↓
    Corrected Motion → Feedback to Generator
```

**Key projects demonstrating this loop**:

1. **PARC** (SIGGRAPH 2025): "Physics-based Augmentation with RL for Character Controllers"
   - Generates motion → RL tracking controller corrects → adds back to training data
   - Core insight: **iterative refinement through physics-in-the-loop**

2. **PHC** (Physics-based Humanoid Controller): Robust RL tracking of reference motions
   - Can validate HyMotion outputs for physical feasibility
   - Returns tracking residuals → potential reward signal

3. **GMR** (General Motion Retargeting, ICRA 2026):
   - CPU real-time retargeting from SMPL (HyMotion's output format) to 17+ robots
   - Ready-to-use bridge for HyMotion → Physics validation pipeline

### 2.3 SOAR vs. Physics Feedback: Complementary Roles

| Method | Input | Correction Oracle | Strength | Weakness |
|--------|-------|---|---|---|
| **SOAR** | Noisy trajectory | Clean target from data | Dense per-step, self-supervised | No physics awareness |
| **Physics Feedback** | Noisy trajectory | Simulation/RL validator | Physics-aware, domain-specific | Sparse (terminal), needs reward model |
| **Hybrid** | Noisy trajectory | Weighted combination | Best of both worlds | Needs careful integration |

### 2.4 Proposed Hybrid Framework: SOAR-Physics

**Phase 1: SOAR Post-Training (Weeks 1-2)**
- Baseline: SOAR on `uncond_fm_man_046b` epoch 1000
- 5K-10K steps, lambda=0.1, N=1
- Evaluate on M2M v2 benchmark (E1-E15)
- Goal: Establish SOAR effectiveness without physics

**Phase 2: Physics Validator Integration (Weeks 3-4)**
- Add lightweight physics checker:
  ```
  Generated motion → Retarget (GMR) → IsaacGym simulate → Check metrics:
    - Foot contact validity (foot Y < 0.08m when contact should occur)
    - Foot sliding distance (XZ displacement given contact)
    - Joint limits / collision detection
    - Root acceleration smoothness
  ```
- Create binary reward: `r_physics = 1.0 if pass all checks else 0.5`

**Phase 3: SOAR-Physics Fusion (Weeks 5-6)**
- Modify SOAR correction target:
  ```
  v_corr_hybrid = alpha * v_corr_soar + (1 - alpha) * v_corr_physics
  
  where:
    v_corr_soar = (z_re - x_clean) / sigma     # existing SOAR
    v_corr_physics = r_physics * v_corr_soar   # reward-weighted
    alpha = learnable or fixed weight (~0.7)
  ```
- Alternative: Use physics validator as **data filter** instead of loss
  ```
  Keep only trajectories with r_physics > threshold in training set
  ```

---

## Part 3: Gap Analysis — What Doesn't Exist Yet

### 3.1 Reward Infrastructure in Motion Domain

Searched for: `reward`, `dpo`, `preference`, `rlhf` in `hftrainer/`

**Results**:
- ❌ No reward models in `hftrainer/models/motion/`
- ❌ No DPO training code
- ❌ No preference labels or RLHF setup
- ⚠️ `ref_repo/HY-SOAR/sora/flow_grpo/` has **reward infrastructure for images**, but NOT motion

**Contrast**: Image generation (SORA, SD3.5) already has:
```
ref_repo/HY-SOAR/sora/flow_grpo/
  ├── rewards.py              # Reward function definitions
  ├── reward_ckpt_path.py     # Reward model checkpoint management
  ├── aesthetic_scorer.py     # Aesthetic metric
  ├── hpsv2_scorer.py         # Human preference score v2
  ├── imagereward_scorer.py   # ImageReward model
  └── ocr.py                  # OCR-based evaluation
```

**For motion**, we have:
```
hftrainer/
  ├── evaluation/             # Metrics-only, no rewards
  ├── models/motion/          # Generator architecture only
  └── pipelines/motion/       # Inference only
```

### 3.2 Physics Simulation Integration

**Currently available in ref_repo** (but not integrated with M2M):
- `ref_repo/ProtoMotions/` — Full physics simulator
- `ref_repo/UH-1/` — IsaacGym-based humanoid training
- `ref_repo/PARC/` — Physics-based tracking controller
- `ref_repo/KIMODO/` — Motion correction with constraints

**Missing**:
- ❌ Lightweight physics validator for SMPL motion
- ❌ Bridging code: HyMotion output → Physics simulation → Reward
- ❌ DPO training loop using physics rewards

### 3.3 Existing Evaluation Infrastructure

**Good news**: M2M v2 has comprehensive quality metrics:
- Foot skating (weighted composite score)
- Jitter (acceleration magnitude)
- Foot contact validity
- Temporal coherence
- Task-specific metrics (E1-E15)

**These can serve as reward signal**, but **no training loop yet** to use them.

---

## Part 4: Recommended Implementation Roadmap

### Phase 1A: SOAR Baseline (2 weeks, no physics)

**Objective**: Establish SOAR as a solid post-training method for M2M

**Tasks**:
1. ✅ Implement SOAR in `HyMotionM2MTrainer`
2. ✅ Test on `uncond_fm_man_046b` epoch 1000
3. ✅ Evaluate E1-E15 at 5K/10K/20K steps
4. ✅ Ablation: lambda sweep (0.05, 0.1, 0.5, 1.0) × N sweep (1, 2)
5. ✅ Report: Compare base vs SOAR on boundary smoothness, foot skating

**Deliverable**: Publication-ready "SOAR for Motion Generation" analysis

---

### Phase 1B: Physics Validator Scaffold (2 weeks, parallel with 1A)

**Objective**: Create minimal physics validation for motion

**Tasks**:
1. Create `hftrainer/evaluation/physics/` module:
   ```python
   class PhysicsValidator:
       def __init__(self, robot_name='smpl_humanoid'):
           self.sim = IsaacGym(robot_name)  # or MuJoCo
       
       def validate(self, motion_smpl):
           # Retarget SMPL → robot
           # Run physics simulation
           # Extract metrics: foot_contact, foot_skating, ...
           # Return reward
   ```

2. Integrate GMR retargeting for SMPL → robot

3. Light simulation (1-2 frames forward to check physical validity)

4. Expose metrics as numpy array for loss computation

**Deliverable**: `PhysicsValidator` class ready for SOAR integration

---

### Phase 2: SOAR-Physics Fusion (2 weeks)

**Objective**: Integrate physics feedback into SOAR correction

**Tasks**:
1. Modify SOAR correction target to use physics rewards:
   ```python
   v_corr = alpha * v_soar + (1 - alpha) * r_physics * v_soar
   ```

2. Two variants to compare:
   - **Variant A** (soft): Weight correction loss by physics reward
   - **Variant B** (hard): Filter training samples by physics validator

3. Evaluate end-to-end on E1-E15

4. Compare Phase 1A vs Phase 2 improvement

**Deliverable**: "Physics-Aware SOAR for Motion Generation" experiments

---

### Phase 3: DPO Training (Optional, 3 weeks)

**Objective**: If SOAR-Physics shows promise, add DPO for preference-based learning

**Tasks**:
1. Create preference labels:
   ```
   Generate pairs: (HyMotion output, SOAR-corrected output)
   Label: "SOAR-corrected is better" (binary preference)
   ```

2. Implement DPO loss:
   ```python
   L_dpo = -log(sigmoid(beta * (log(π(y_w|x)) - log(π(y_l|x)))))
   ```

3. Post-training DPO on SOAR checkpoint

4. Evaluate on E1-E15

**Deliverable**: DPO training code + results

---

## Part 5: Key Technical Decisions

### 5.1 Physics Simulation Choice

| Simulator | Pros | Cons | Recommendation |
|-----------|------|------|---|
| **IsaacGym** | Parallel, fast, accurate | GPU-heavy, NVIDIA-only | ✅ Use for full validation |
| **MuJoCo** | Simple, CPU-friendly, portable | Slower | ✅ Use for lightweight checks |
| **KIMODO** | Already integrated with M2M | Slower, not designed for loop | ⚠️ Fallback |

**Recommendation**: Start with **lightweight MuJoCo checks** (1-2 frames), upgrade to IsaacGym if needed.

### 5.2 Reward Signal Design

```
Physics Reward r_physics:
  ├─ r_contact = 1 if foot contacts match GT else 0.5
  ├─ r_sliding = 1 - min(1, foot_slide_dist / threshold)
  ├─ r_collision = 1 - collision_count / max_expected
  ├─ r_accel = 1 - min(1, root_accel / threshold)
  └─ r_joint_limits = 1 if all joints in bounds else 0.5
  
  r_physics = geometric_mean([r_contact, r_sliding, r_collision, r_accel, r_joint_limits])
              or alpha*product + (1-alpha)*min (depending on strictness)
```

### 5.3 SOAR Correction Target Variant

**Option A: Additive Weighting (Recommended for Phase 2)**
```python
v_corr_hybrid = (1 - beta) * v_soar + beta * r_physics * v_soar
# Intuition: Reward modulates the magnitude of correction
# When physics score is low, reduce correction; when high, strengthen it
```

**Option B: Replacement (Stronger, Riskier)**
```python
v_corr_hybrid = where(r_physics > 0.8, v_soar, v_current)
# Intuition: Only correct if physics is sufficiently feasible
# Sharp transition could cause training instability
```

**Recommendation**: Start with Option A (additive, smoother), monitor divergence.

---

## Part 6: Critical Insights from Reference Work

### 6.1 SOAR's "Correction Oracle" Can Be Physics-Replaced

**SOAR's key assumption**: The clean target `x_clean` is the correct correction direction.

**Our hypothesis**: For motion, a **physics-validated trajectory** is a better correction oracle than the clean target alone, because:
- Clean target may violate physical constraints (foot sliding, joint limits)
- Physics validator provides domain-specific feedback
- Hybrid approach: Use physics validator to **weight** or **filter** the correction

**Example from PARC**: Their iterative refinement loop proves this concept works:
```
Generated motion → RL tracker → Corrected motion → Better generation
```

We're replacing RL tracker with SOAR + physics validator.

### 6.2 SOAR Doesn't Need New Data

**Critical advantage over RL/DPO**: SOAR uses only existing training data.

For M2M:
- ✅ Existing data: 549K samples from `train_hymotion_400h.json`
- ✅ Quality-filtered: 456K from `high_quality.json`
- ❌ New annotations: Zero required

Physics feedback can be computed on-the-fly during training (no preprocessing).

### 6.3 M2M's Exposure Bias is Worse Than Image Gen

**Why**:
- 50-step ODE vs. 4-10 steps for images
- Temporal data: early errors propagate
- Post-hoc boundaries: hard transitions in both generation and conditioning

**Implication**: SOAR's benefits should be **amplified** for motion.

### 6.4 The _man (Mask-Aware Noise) Variant is Key

**Current M2M setup**: `_man` handles known regions, but generated regions still have full exposure bias.

**SOAR's advantage**: Specifically targets generated regions with dense correction.

**Combined effect**: 
- `_man` solves: "Why does transition between known and generated have a jump?"
- SOAR solves: "Why do generated regions internally have temporal incoherence?"

**Together**: Orthogonal fixes to the two halves of the problem.

---

## Part 7: Anticipated Challenges & Mitigations

### Challenge 1: Physics Validator Overhead

**Problem**: Running physics simulation in training loop slows training.

**Mitigation**:
- Light simulation only (1-2 steps)
- Batch-wise evaluation (not per-sample)
- Async computation (compute while model trains)
- Sparse application (every N steps, not every step)

### Challenge 2: Reward Engineering

**Problem**: Designing good physics reward is non-trivial.

**Mitigation**:
- Start with simple binary reward (pass/fail)
- Gradually move to continuous rewards
- Leverage existing M2M evaluation metrics (foot skating, jitter, etc.)
- Monitor loss curves to detect reward noise

### Challenge 3: SOAR Lambda Tuning

**Problem**: If lambda_soar is too large, SOAR loss dominates and base quality degrades.

**Mitigation**:
- Start conservative (lambda=0.1)
- Monitor both L_base and L_corr during training
- Stop training if L_base increases
- Ablation sweep: lambda ∈ {0.05, 0.1, 0.25, 0.5}

### Challenge 4: Physics Sim ↔ SMPL Mismatch

**Problem**: Physics sim expects robot-specific dynamics; SMPL is kinematic.

**Mitigation**:
- Use SMPL humanoid in IsaacGym (standard setup)
- GMR retargeting handles SMPL → robot mapping
- Validate physics checks on SMPL-space, not robot-space

---

## Part 8: Success Criteria

### Phase 1A: SOAR Baseline
- [ ] SOAR implementation passes smoke tests (gradient flow, loss computation)
- [ ] E1-E15 evaluation: Boundary smoothness improves by 5-10%
- [ ] Foot skating metric improves or unchanged
- [ ] No regression on base quality metrics (GenEval-equivalent for motion)

### Phase 1B: Physics Validator
- [ ] `PhysicsValidator` computes 5+ metrics (foot contact, sliding, accel, limits, collision)
- [ ] Inference speed: <100ms per 50-frame motion (reasonable for post-training loop)
- [ ] Metrics correlate with visual quality (spot-check 10 cases)

### Phase 2: SOAR-Physics Fusion
- [ ] SOAR-Physics loss computes without NaN/Inf
- [ ] E1-E15 evaluation: Further 3-5% improvement over Phase 1A
- [ ] Physics metrics improve (lower foot skating, etc.)
- [ ] Comparable or better than baseline + post-processing

### Phase 3: DPO (Optional)
- [ ] DPO loss converges stably
- [ ] Human evaluation: Preference for DPO-trained model > SOAR-Physics
- [ ] Quantitative metrics: +5% on boundary smoothness or foot skating

---

## Summary Table: Reference Works & Integration Points

| Reference | Key Contribution | Applicability to Physics-Feedback M2M | Integration Point |
|-----------|---|---|---|
| **SOAR** (arXiv:2604.12617) | Exposure bias correction via dense supervision | ✅ **Direct** — framework identical, Phase 1A foundation | Replace clean oracle with physics-validated trajectory |
| **PARC** (SIGGRAPH 2025) | Physics-in-the-loop data augmentation loop | ✅ **Inspirational** — validates iterative refinement concept | Phase 2-3 methodology |
| **PHC** (RSS 2024) | Physics-based humanoid controller | ✅ **Implementation** — can serve as physics validator | Phase 1B: RL tracker |
| **GMR** (ICRA 2026) | SMPL ↔ robot retargeting | ✅ **Critical bridge** — HyMotion output → physics sim | Phase 1B: Retargeting module |
| **ASAP** (RSS 2025) | Sim2Real physics transfer | ✅ **Optional** — for final deployment validation | Phase 3+: Real robot testing |
| **HY-SOAR** (SORA) | Image RL + reward infrastructure | ✅ **Reference** — shows how RL works in diffusion | Phase 3: DPO adaptation |

---

## Conclusion

### The Path Forward

1. **Week 1-2**: SOAR baseline (Phase 1A) — establish foundation
2. **Week 1-4 (parallel)**: Physics validator scaffold (Phase 1B) — build bridge to simulation
3. **Week 3-4**: SOAR-Physics fusion (Phase 2) — hybrid correction
4. **Week 5+ (optional)**: DPO training (Phase 3) — preference-based refinement

### Why This Approach is Feasible

- ✅ **SOAR** requires zero new data (only existing training set)
- ✅ **Physics validator** can reuse existing open-source tools (GMR, IsaacGym)
- ✅ **Integration** is additive, not disruptive (can fall back to SOAR-only if physics causes issues)
- ✅ **Timeline** is realistic (8-12 weeks for full implementation)
- ✅ **Impact** is high (boundary smoothness + foot skating are key M2M weaknesses)

### Key Hypothesis to Test

> **Can physics-validated trajectories serve as better correction oracles than clean training targets in a SOAR-like framework?**

This is novel because:
1. SOAR itself is new (April 2026)
2. Applying SOAR to motion is new
3. Hybrid SOAR-physics is unexplored
4. No prior work combines flow matching post-training with physics feedback

### Next Steps

1. **Read** the full SOAR paper (arXiv:2604.12617)
2. **Implement** SOAR in `HyMotionM2MTrainer` (Phase 1A)
3. **Prototype** physics validator with IsaacGym + MuJoCo (Phase 1B)
4. **Evaluate** on M2M v2 benchmark (E1-E15)
5. **Iterate** based on results

---

## References

### Primary Sources (Analyzed in This Document)

1. **SOAR Paper**: arXiv:2604.12617 (2026-04)
   - Source: `ref_repo/SOAR/SOAR_paper.pdf`
   - Analysis: `ref_repo/SOAR/CLAUDE.md` (comprehensive)

2. **HyMotion M2M Documentation**
   - Source: `hftrainer/models/motion/CLAUDE.md`
   - SOAR applicability: Section 3 of above

3. **Motion + Embodied Intelligence Survey**
   - Source: `docs/temp/survey_motion_gen_embodied_v2_20260508.md`
   - Key projects: PARC, ASAP, VideoMimic, GMR, UH-1

4. **M2M Next-Gen Proposal (CDO-FM)**
   - Source: `docs/temp/hymotion_m2m_next_gen_proposal_20260511.md`
   - Physics-related discussion: Section 1.2.2 (滑步问题根因分析)

### Secondary Sources (Related Work)

- PHC (Physics-based Humanoid Control)
- BeyondMimic (RL + motion generation)
- KIMODO (Motion correction with constraints)
- ProtoMotions (Physics simulation framework)

### Open-Source Implementations Ready-to-Use

- GMR: `github.com/YanjieZe/GMR` (retargeting)
- PARC: `github.com/mshoe/PARC` (physics-based augmentation)
- ASAP: `github.com/LeCAR-Lab/ASAP` (sim-to-real)
- UH-1: `github.com/sihengz02/UH-1` (humanoid dataset)

