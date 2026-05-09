# HyMotion M2M V2 Critical Analysis Report

> Date: 2026-04-16
> Scope: Training and inference design of HyMotion M2M V2 (0.46B, flow matching, VACE conditioning)
> Method: Source code review + cross-project comparison + literature research

---

## Executive Summary

Based on thorough source code review of the trainer (`hymotion_m2m_trainer.py`), pipeline (`hymotion_m2m_pipeline.py`), bundle (`bundle.py`), loss module (`m2m_loss.py`), mask strategies (`universal_mask.py`, `condition_sampler_v2.py`), and architecture (`hymotion_mmdit.py`), this report identifies **16 design problems** organized around three observed failure modes, plus **5 systemic issues** affecting overall quality. Each problem is traced to specific code locations with concrete evidence.

### Three Observed Problems

| Problem | Severity | Root Causes Identified |
|---------|----------|----------------------|
| **P1**: Phase 2 T2M inference far worse than pure T2M | Critical | 6 root causes (RC1-RC6) |
| **P2**: Severe foot sliding, translation-based movement | Critical | 5 root causes (RC7-RC11) |
| **P3**: Keyframe boundary jitter/trembling | Moderate | 5 root causes (RC12-RC16) |

---

## Problem 1: Phase 2 T2M Degradation

**Symptom**: When using the M2M model for pure T2M generation (all mask=1, text-conditioned), results are significantly worse than a dedicated T2M model trained on the same data.

### RC1: M5 Full-Mask Strategy Has Only 5% Training Weight (Critical)

**Location**: `universal_mask.py` strategy weights

```python
# Default weights from CLAUDE.md / universal_mask.py:
# M1: 20%, M2: 12%, M3: 23%, M4: 15%, M5: 5%, M6: 15%, M7: 10%
```

**Analysis**: Pure T2M generation requires `src_mask = all ones` (M5 pattern). With only **5% sampling probability**, the model sees this pattern in roughly 1 out of every 20 training samples. The model spends 95% of training learning partial completion tasks where it can rely on known-region information from the VACE `inactive` channel. When it encounters the all-masked T2M pattern, it has learned to "expect" inactive conditioning signals that are now absent.

**Contrast with KIMODO**: KIMODO uses a **two-phase approach** -- Phase 1 trains pure T2M for 500K steps, Phase 2 adds completion for another 500K steps. This ensures the model fully learns T2M before being fine-tuned for completion. M2M tries to do both simultaneously with a 5/95 split that heavily under-represents T2M.

**Impact**: The model's T2M capacity is severely under-trained. Even with correct text conditioning, the model hasn't learned to generate coherent full-body motion from text alone at sufficient quality.

**Recommendation**: Either (a) increase M5 weight to 15-25% and proportionally reduce others, or (b) adopt KIMODO-style two-phase training, or (c) use curriculum learning that starts with higher M5 ratio and gradually increases mask complexity.

### RC2: Random Initialization of Input/Output Projections (Critical)

**Location**: `CLAUDE.md` Weight Initialization section; `encoders.py`

```
Loaded: 18 transformer blocks (305/308 params from T2M 1.0-Lite)
Random init: input_encoder (201->540), final_layer (201->135) -- 3/308 params
```

**Analysis**: The HunyuanMotion T2M 1.0-Lite pretrained model has `motion_dim=201` (local rot6d + joint positions + translation). M2M uses `motion_dim=135` (rot6d + translation only). The input projection (from 4*D_motion = 4*135=540 to feat_dim=1024) and output projection (from feat_dim to 135) have shape mismatches, so they are **randomly initialized**.

These are not just any layers -- they are the **gateway layers** of the entire architecture:
- `input_encoder`: Maps the concatenated `[x_t, inactive, reactive, mask]` to the transformer's feature space. Random init means the model must learn from scratch what these 540 input channels mean.
- `final_layer`: Maps transformer features back to motion space. Random init means the model must re-learn the entire output distribution.

The 305 pretrained transformer blocks receive randomly projected features from a randomly initialized `input_encoder`, which destroys the pretrained representations in early training. This is the **single most destructive factor** for T2M quality -- the pretrained T2M knowledge is effectively erased.

**Contrast with UMO**: UMO **freezes the entire backbone** (460M params) and only trains a 0.207M adapter. This preserves 100% of T2M capability. M2M re-initializes the I/O projections and fine-tunes everything, destroying pretrained knowledge.

**Recommendation**:
- Use a shape-compatible initialization: zero-pad or repeat the pretrained 201-dim weights to match the new 540/135 shapes, then fine-tune.
- Alternatively, use LoRA adapters on the I/O projections instead of random init.
- Consider freezing transformer blocks for the first N steps (warmup), allowing only I/O projections to train.

### RC3: MAN Training Fundamentally Alters x_t Distribution (Moderate)

**Location**: `hymotion_m2m_trainer.py` lines 220-222

```python
if self.mask_aware_noise and src_mask is not None:
    keep_mask = 1 - src_mask  # (B, L, D), 1=known
    x_t = x_t * src_mask + x1 * keep_mask  # MAN: known=clean, gen=noisy
```

**Analysis**: With mask-aware noise (MAN), the model sees `x_t` with a fundamentally different distribution than standard flow matching:
- **Standard**: `x_t = (1-t)*noise + t*x1` everywhere -- uniform distribution across all dimensions
- **MAN**: `x_t[known] = x1` (clean), `x_t[generate] = (1-t)*noise + t*x1` (noisy)

For M5 (full mask, T2M-equivalent), there are no known regions, so MAN and standard are identical. However, the model's 18 transformer blocks are trained predominantly on the MAN distribution (95% of samples have some known regions). The attention patterns, feature statistics, and learned representations are all optimized for the "mixed clean+noisy" input distribution. When the model encounters the M5 all-noisy input, it's a distributional shift from the pattern it primarily trained on.

**Impact**: Moderate -- this is a secondary effect after RC1 and RC2, but contributes to the distribution gap.

### RC4: cond_mask_prob=1.0 in Unconditioned Variant (High)

**Location**: From CLAUDE.md analysis of training configs

```
cond_mask_prob=1.0 in uncond variant -- text ALWAYS dropped
```

**Analysis**: For the `_uncond` variant, `cond_mask_prob=1.0` means text conditioning is dropped with 100% probability during training. The model uses `null_vtxt_feat` and `null_ctxt_input` (frozen from T2M pretrained checkpoint) for every single training step. This variant **cannot do T2M at all** -- it has no text conditioning pathway.

For the `_caption` variant, `cond_mask_prob` should be set to a reasonable CFG dropout rate (e.g., 0.1-0.2). If it's also set too high, T2M performance will degrade.

**Impact**: For `_uncond` variant, T2M is structurally impossible. For `_caption` variant, need to verify `cond_mask_prob` is appropriately set.

### RC5: VACE Provides Spurious Conditioning in T2M Mode (Low-Moderate)

**Location**: `bundle.py` lines 289-338

```python
inactive = src_motion * (1 - src_mask)  # For M5: all zeros
reactive = src_motion * src_mask        # For M5: all zeros (after zeroing)
vace_context = cat([inactive, reactive, mask], dim=-1)  # (B, L, 3*D)
```

**Analysis**: For T2M (M5 full mask), the VACE context becomes `[zeros, zeros, ones]` -- a 3*135=405 dim vector of mostly zeros with 135 dims of ones. The model must learn that this specific pattern means "generate everything from scratch." However, the model rarely sees this pattern (5% of training), so its behavior for this input configuration is under-determined.

More critically, during the 95% of training with partial masks, the model learns strong associations between VACE patterns and expected output -- these associations don't exist in the T2M case. The model may try to "read" non-existent conditioning signals from the all-zero inactive/reactive channels, producing confused outputs.

**Contrast with dedicated T2M**: A dedicated T2M model has no VACE channels at all. Its entire capacity is devoted to text->motion generation. M2M's 540-dim input (vs. dedicated T2M's ~201-dim) means 75% of the input is wasted on VACE channels that carry no information in T2M mode.

### RC6: Training Data Quality (Moderate)

**Location**: `CLAUDE.md` Training Data Quality Issue

```
549K total -> 85K low quality + 69K borderline = 154K problematic samples (28%)
```

**Analysis**: 28% of training data contains quality issues (foot sliding, jitter, joint jumps). During T2M generation, the model draws from its learned distribution which includes these defects. A dedicated T2M model trained on filtered data would not have this issue.

---

## Problem 2: Severe Foot Sliding

**Symptom**: Generated motion exhibits obvious foot sliding -- the model moves the body via direct translation rather than adjusting foot placement. Feet appear to "skate" across the ground.

### RC7: Motion Representation Lacks Joint Positions (Critical)

**Location**: `universal_mask.py` lines 42-47, `CLAUDE.md` 135-dim Layout

```python
TOTAL_DIM = TRANSL_DIM + 22 * JOINT_ROT_DIM  # 135 = 3 + 22*6
# dims [0:3]  -- translation (absolute)
# dims [3:9]  -- Pelvis rot6d
# ...
# dims [129:135] -- R_Wrist rot6d
# NO joint positions, NO foot contact labels
```

**Analysis**: The 135-dim representation contains ONLY rotation and translation. There are **no world-space joint positions** and **no foot contact labels**. This creates a fundamental problem:

1. **The model cannot directly reason about foot-ground contact.** To determine if a foot is on the ground, one must run forward kinematics (FK) through the entire kinematic chain: root translation -> pelvis rotation -> hip rotation -> knee rotation -> ankle rotation -> foot rotation -> foot position. The model must implicitly learn this entire chain to avoid foot sliding.

2. **Errors in any joint along the chain compound.** A tiny error in pelvis rotation propagates through 5+ joints to produce large foot position errors. The model has no mechanism to "anchor" the foot to a specific world-space position.

3. **Translation is the path of least resistance.** When the model needs to move the character, adjusting the 3 translation dims is far simpler (lower loss) than coordinating rotations across 6+ joints in the kinematic chain to achieve proper foot placement. The model learns to translate because translation is directly supervised and does not require implicit FK reasoning.

**Contrast with KIMODO**: KIMODO uses 333-dim representation with **global joint positions (22x3) + global joint rotations (22x6) + foot contact (4)**. The model can directly supervise foot-ground contact and world-space foot positions. Foot sliding is structurally prevented because the model directly predicts where each foot is in world space.

**Contrast with MoGenDiT**: MoGenDiT uses 201-dim with **local joint positions (22x3)** in addition to rotations. While still local coordinates, the joint positions provide a direct regression target for foot positions, enabling more effective foot contact supervision.

**Impact**: This is the **primary structural cause** of foot sliding. No amount of loss engineering can fully compensate for the absence of foot position in the representation.

### RC8: FK Loss Uses Only Root-Relative Keypoints (Critical)

**Location**: `m2m_loss.py` lines 126-136

```python
# FK loss -- root-relative only:
local_keypoints3d = pred_keypoints3d[:, :, 1:22] - pred_keypoints3d[:, :, 0:1, :]
local_keypoints3d_gt = gt_keypoints3d[:, :, 1:22] - gt_keypoints3d[:, :, 0:1, :]
loss_dict["keypoints3d"] = self.keypoints3d_weight * self.loss_fn(
    local_keypoints3d, local_keypoints3d_gt, reduction="none"
).sum(dim=-1).mean(dim=-1)
```

**Analysis**: The FK loss subtracts the root (Pelvis) position from all joint positions, making it **completely blind to global foot positions**. This loss only ensures that joints are in the correct positions **relative to the pelvis**. It says nothing about:

1. **Where the foot is in world space** -- a globally correct pose with wrong translation will have zero FK loss but severe foot sliding.
2. **Whether the foot is on the ground** -- vertical foot position relative to the ground plane is completely unpenalized.
3. **Whether the foot is stationary when in contact** -- horizontal foot velocity during ground contact (the definition of foot sliding) is not addressed.

The loss effectively says: "make the pose look right locally" but not "make the character's feet touch the ground properly." This is exactly the failure mode observed.

**Why this matters quantitatively**: Foot sliding is primarily a **global position** problem. When a character walks, the foot must be at global position (x, y, 0) during contact. The root-relative FK loss ensures the foot is at the right position relative to the pelvis, but if the pelvis's global translation is wrong by even 2cm, the foot will slide by 2cm -- and this error is **invisible** to the FK loss.

**Recommendation**: Add a **global FK loss** (without root subtraction) or a dedicated **foot contact loss**:
```python
# Global FK loss -- penalizes absolute foot positions
global_foot_pos = pred_keypoints3d[:, :, [10, 11], :]  # L_Foot, R_Foot
global_foot_gt = gt_keypoints3d[:, :, [10, 11], :]
loss_global_foot = F.smooth_l1_loss(global_foot_pos, global_foot_gt)

# Foot contact loss -- penalize foot velocity when in contact
foot_vel = global_foot_pos[:, 1:] - global_foot_pos[:, :-1]
contact = (global_foot_gt[:, :-1, :, 1] < 0.05).float()  # y < 5cm = on ground
loss_contact = (foot_vel.norm(dim=-1) * contact).mean()
```

### RC9: No Foot Contact Loss or Ground Penetration Penalty (Critical)

**Location**: `m2m_loss.py` -- complete absence of foot contact loss

**Analysis**: The loss module has 6 loss terms: velocity, x1, keypoints3d (root-relative), translation, motion_smoothness, fk_consistency. None of these address foot-ground contact:

| Loss Term | Addresses Foot Sliding? | Why Not |
|-----------|------------------------|---------|
| velocity | No | Penalizes flow prediction error, not physical plausibility |
| x1 | No | Per-dim MSE on rotation/translation, no FK awareness |
| keypoints3d | **No** | Root-relative subtraction removes global position info |
| translation | No | Only penalizes root translation error, not foot position |
| smoothness | No | Penalizes temporal jerk, not foot-ground contact |
| fk_consistency | No | Ensures rotation-position consistency in 198-dim, but in local space |

**What's needed (from literature)**:

1. **Foot contact detection** (UnderPressure, ECCV 2022): Learn to predict foot-ground contact labels from motion, then use them as supervision signal.

2. **Foot skating loss** (LODGE, CVPR 2024):
   ```
   L_contact = ||v_foot * contact_label||  -- foot velocity should be 0 when in contact
   L_ground = ||h_foot * contact_label||   -- foot height should be 0 when in contact
   L_penetration = max(0, -h_foot)         -- foot should never go below ground
   ```

3. **Global keypoint loss** (KIMODO, 2025):
   ```
   L_global = ||FK(pred_rotation, pred_translation) - FK(gt_rotation, gt_translation)||
   ```
   Without root subtraction, this directly penalizes global position errors.

### RC10: Translation Dimension Imbalance (Moderate)

**Location**: `m2m_loss.py` lines 83-86

```python
if self.trans_dim_weight != 1.0:
    dim_weights = torch.ones(vel_per_dim.shape[-1], device=vel_per_dim.device)
    dim_weights[:self.trans_dims] = self.trans_dim_weight
    vel_per_dim = vel_per_dim * dim_weights
```

**Analysis**: Translation is 3 dims out of 135 (2.2%). Even with `trans_dim_weight > 1`, the per-dim mean loss is dominated by rotation dims. The model can reduce total loss more effectively by improving rotation predictions (132 dims) than translation predictions (3 dims). This creates an optimization bias where translation quality is sacrificed for rotation quality.

In the context of foot sliding: proper foot placement requires coordinated translation + rotation. When the model is incentivized to prioritize rotation accuracy (because it dominates the loss), translation becomes the "free variable" that absorbs errors -- manifesting as foot sliding.

**Recommendation**: Increase `trans_dim_weight` significantly (e.g., 10-20x) or use a separate translation loss with explicit weight control.

### RC11: Motion Smoothness Loss Doesn't Address Physical Plausibility (Low)

**Location**: `m2m_loss.py` lines 154-166

```python
# Temporal difference loss
pred_motion_vel = pred_x1[:, 1:] - pred_x1[:, :-1]  # (B, L-1, D)
gt_motion_vel = gt_x1[:, 1:] - gt_x1[:, :-1]
smooth_loss = self.loss_fn(pred_motion_vel, gt_motion_vel, reduction="none")
```

**Analysis**: The smoothness loss penalizes deviation in frame-to-frame velocity between prediction and GT. This ensures temporal smoothness but does NOT ensure physical plausibility. A motion can be perfectly smooth yet have severe foot sliding (smooth translation instead of proper foot placement). The loss needs to be augmented with physics-aware terms (foot contact, ground penetration) to address the actual failure mode.

---

## Problem 3: Keyframe Boundary Jitter

**Symptom**: Near condition keyframes (mask boundary between known and generated regions), generated motion shows slight jitter or trembling.

### RC12: MAN Imputation Creates Distribution Discontinuity at Boundaries (Critical)

**Location**: `hymotion_m2m_pipeline.py` lines 278-293

```python
for i in range(n_ode_steps):
    v = fn(t_curr, x)
    x = x + v * dt
    # Imputation: force known regions back
    if rep_mode == 'skip_last' and not is_last_step:
        x = torch.where(keep_mask, x_clean, x)  # HARD replacement
```

**Analysis**: At every ODE step (except the last in `skip_last` mode), known regions are **hard-replaced** with `x_clean`. This creates a sharp discontinuity at mask boundaries:

- Frame k-1 (known, mask=0): `x[k-1] = x_clean[k-1]` -- perfectly clean, zero noise
- Frame k (generated, mask=1): `x[k] = ODE_step(noisy)` -- still partially noisy at intermediate timesteps

The model's velocity prediction at frame k is influenced by the attention to frame k-1 (which is artificially clean), but the model was trained to see a gradual transition in noise levels (the flow matching interpolation `(1-t)*noise + t*clean`). The hard replacement creates a **noise-level cliff** at the boundary that the model never saw during training.

This discontinuity manifests as jitter because:
1. The model produces a velocity field that "expects" gradual noise transition
2. The hard replacement forces an instantaneous transition
3. The mismatch causes oscillating corrections in subsequent ODE steps

### RC13: `skip_last` Mode Creates Final-Step Discontinuity (High)

**Location**: `hymotion_m2m_pipeline.py` lines 281, 292

```python
is_last_step = (i == n_ode_steps - 1)
# ...
elif rep_mode == 'skip_last' and not is_last_step:
    x = torch.where(keep_mask, x_clean, x)
```

**Analysis**: In `skip_last` mode, the final ODE step runs without imputation. This means:
- Step N-1: known regions are replaced with `x_clean` (perfectly clean)
- Step N (final): known regions are NOT replaced, so the model's velocity prediction from step N modifies them freely

The final step's velocity field was computed assuming known regions would stay clean (as they did for all previous steps). When it suddenly gets to modify them, the result is a discontinuity jump at the final step. This is especially visible at mask boundaries where the generated region's last-step correction creates a visible seam with the known region.

**Recommendation**: Use `flow_interp` mode instead of `skip_last`. The `flow_interp` mode replaces known regions with `(1-t)*z0 + t*x_clean` which follows the flow matching interpolation path -- this is exactly what the model expects to see during training. According to CLAUDE.md, flow_interp yields "~40-60% boundary smoothness improvement over skip_last."

### RC14: Post-Hoc Hard Blend Without Feathering (High)

**Location**: From inference data flow documentation

```python
# Post-hoc blend at inference end:
final = original * (1-mask) + model_output * mask  # HARD boundary
```

**Analysis**: After ODE integration, the final output is blended with the original motion using a **binary mask** -- no feathering, no smooth transition, no boundary blending. This creates a hard seam at every mask boundary.

In image inpainting, this problem was solved long ago -- all modern inpainting methods use soft mask boundaries (Gaussian blur on mask edges, alpha blending, etc.). Motion inpainting has the same need for smooth transitions at mask boundaries.

**Evidence from image domain**: BrushNet (ECCV 2024) and PowerPaint (ECCV 2024) both use mask dilation and soft blending at boundaries. The key insight is: **the mask boundary should not be a step function -- it should be a smooth transition zone**.

**Recommendation**: Apply temporal feathering at mask boundaries:
```python
# Feather the mask with Gaussian blur over time dimension
feathered_mask = gaussian_blur_1d(mask, sigma=3)  # 3-frame sigma
final = original * (1 - feathered_mask) + model_output * feathered_mask
```

### RC15: No Boundary-Aware Training Loss (High)

**Location**: `m2m_loss.py` -- no boundary-specific loss term

**Analysis**: During training, the loss treats all generated frames equally -- frames far from mask boundaries are weighted the same as frames right at boundaries. The model has no special incentive to produce smooth transitions at mask edges.

**Contrast with CondMDI (SIGGRAPH 2024)**: CondMDI uses a training scheme where the model explicitly learns to produce smooth transitions at mask boundaries by training with variable-density keyframes. The model learns that frames adjacent to keyframes should smoothly interpolate, not abruptly change.

**Contrast with image inpainting**: PowerPaint's learnable `P_ctxt` prompt specifically encourages context-aware boundary coherence. BrushNet extracts geometric features from mask boundaries to guide boundary-consistent generation.

**Recommendation**: Add a boundary smoothness loss:
```python
# Identify boundary frames (transition from mask=0 to mask=1)
boundary = (mask[:, 1:] != mask[:, :-1]).any(dim=-1).float()  # (B, L-1)
# Penalize large velocity changes at boundaries
vel_pred = pred_x1[:, 1:] - pred_x1[:, :-1]
vel_gt = gt_x1[:, 1:] - gt_x1[:, :-1]
boundary_loss = (F.smooth_l1_loss(vel_pred, vel_gt, reduction='none').mean(-1) * boundary).sum() / boundary.sum().clamp(min=1)
```

### RC16: Euler ODE Solver Accumulates Numerical Error (Moderate)

**Location**: `hymotion_m2m_pipeline.py` lines 278-284

```python
# Manual Euler (NOT midpoint) when replacement guidance active
for i in range(n_ode_steps):
    v = fn(t_curr, x)
    x = x + v * dt  # First-order Euler
```

**Analysis**: When replacement guidance is active (which is the recommended path for MAN models), the pipeline falls back to a **first-order Euler** solver instead of the midpoint solver used in the standard path. Euler has O(dt) local error vs. midpoint's O(dt^2). With 50 steps:
- Euler error: ~O(1/50) = ~0.02 per step, accumulating
- Midpoint error: ~O(1/2500) = ~0.0004 per step

This error accumulation is particularly visible at boundaries where small perturbations get amplified by the hard replacement at each step.

**Recommendation**: Implement midpoint solver compatible with per-step imputation:
```python
# Midpoint with imputation
k1 = fn(t_curr, x) * dt
x_mid = x + 0.5 * k1
if use_replacement:
    x_mid = torch.where(keep_mask, x_clean, x_mid)
k2 = fn(t_curr + 0.5 * dt, x_mid) * dt
x = x + k2
if use_replacement and not is_last_step:
    x = torch.where(keep_mask, x_clean, x)
```

---

## Systemic Issues

### S1: VACE Channel Redundancy with MAN (Architectural)

**Location**: `bundle.py` prepare_vace_input, `CLAUDE.md` cross-project comparison

**Analysis**: With MAN training, known-region information exists in **three places simultaneously**:
1. `x_t` -- known regions are clean (MAN training distribution)
2. `inactive` channel -- known-region values (same as x_t known regions)
3. `mask` -- binary indicator of which regions are known

The `inactive` channel is **redundant** with the x_t signal for MAN models. This wastes 135 dimensions of model capacity on duplicated information. The `no_inactive` ablation mode exists but is not the default.

**Impact**: Model must learn to reconcile redundant signals, wasting capacity. The input dimension is 540 (4*135) when it could be 270 (2*135) with `no_inactive` mode and MAN.

### S2: Training on Unfiltered Data (Data Quality)

**Location**: All training configs use `train_hymotion_400h.json`

549K samples include ~85K low quality + ~69K borderline = 28% problematic data. The model learns from defective motions (jitter, foot sliding, joint jumps) and reproduces these patterns during generation.

**Recommendation**: Use the quality-filtered `high_quality.json` (456K samples) already available at `data/hymotion_m2m_refine_data/data_quality_list/`.

### S3: No Physics-Aware Losses (Fundamental)

The entire loss design operates in **kinematic space** only. No loss term addresses:
- Ground contact forces / reaction forces
- Center of mass trajectory plausibility
- Joint torque limits
- Collision (self-intersection, ground penetration)

Motion generation models like LODGE (CVPR 2024) and PhysDiff (ICLR 2023) have shown that physics-aware losses or post-processing significantly improve realism, especially for foot contact.

### S4: Timestep Sampling Distribution (Training Efficiency)

**Location**: `hymotion_m2m_trainer.py` lines 207-211

```python
if self.bundle.pred_type == 'x1':
    z = torch.randn(B, dtype=x1.dtype, device=device) * 0.8 + (-0.8)
    timesteps = torch.sigmoid(z)  # Logit-normal, biased toward t=0.3
else:
    timesteps = torch.rand(B, dtype=x1.dtype, device=device)  # Uniform
```

For velocity pred_type (the default), timesteps are sampled uniformly. Recent work (Stable Diffusion 3, Rectified Flow) shows that **logit-normal sampling** (more weight on intermediate timesteps) significantly improves generation quality. The x1 pred_type uses logit-normal but velocity doesn't benefit from this improvement.

### S5: Single-Scale Attention / No Sliding Window (Moderate)

The 0.46B MMDiT uses full self-attention over all frames. For 360-frame sequences, this creates O(360^2) attention which:
1. Limits the model to learning primarily global patterns
2. Provides no structural inductive bias for local motion coherence
3. MoGenDiT uses sliding window attention (window=90) which explicitly promotes local coherence

---

## Summary of Recommendations

### Immediate (High Impact, Low Effort)

| # | Recommendation | Addresses | Effort |
|---|---------------|-----------|--------|
| 1 | Increase M5 weight to 20% | P1 (RC1) | Config change |
| 2 | Use `flow_interp` replacement mode | P3 (RC12, RC13) | Config change |
| 3 | Add temporal feathering to post-hoc blend | P3 (RC14) | ~20 lines |
| 4 | Train on quality-filtered data | P1, P2, S2 | Data switch |
| 5 | Increase `trans_dim_weight` to 10-20x | P2 (RC10) | Config change |

### Medium-Term (High Impact, Moderate Effort)

| # | Recommendation | Addresses | Effort |
|---|---------------|-----------|--------|
| 6 | Add global FK foot contact loss | P2 (RC8, RC9) | ~100 lines in loss.py |
| 7 | Add boundary smoothness loss | P3 (RC15) | ~50 lines in loss.py |
| 8 | Implement midpoint ODE with imputation | P3 (RC16) | ~30 lines in pipeline |
| 9 | Shape-compatible initialization for I/O projections | P1 (RC2) | ~50 lines in bundle |
| 10 | Switch to `no_inactive` VACE mode for MAN | S1 | Config + verify |

### Long-Term (Structural Improvements)

| # | Recommendation | Addresses | Effort |
|---|---------------|-----------|--------|
| 11 | Extend representation to include foot contact + joint positions | P2 (RC7) | Repr redesign |
| 12 | Two-phase training (T2M then completion) | P1 (RC1, RC2) | Training redesign |
| 13 | Physics-aware losses or post-processing | P2 (S3) | Research + impl |
| 14 | Adopt CondMDI-style variable-density keyframe training | P3 (RC15) | Mask strategy redesign |

---

## Literature Research: Related Works

### For Problem 1 (T2M Degradation in Multi-Task Models)

| Paper | Venue | Relevance | Key Insight |
|-------|-------|-----------|-------------|
| **UMO** (Brown/MIT/Meta, 2025) | ArXiv | Freeze backbone, tiny adapter | 0.207M adapter preserves 100% T2M; M2M fine-tunes everything and loses it |
| **KIMODO** (NVIDIA, 2025) | ArXiv | Two-phase training | Phase 1: pure T2M 500K steps. Phase 2: add completion. Preserves T2M. |
| **LoRA** (Hu et al., 2022) | ICLR | Parameter-efficient fine-tuning | Add low-rank adapters instead of full fine-tuning to preserve pretrained knowledge |

### For Problem 2 (Foot Sliding)

| Paper | Venue | Repo | Key Technique |
|-------|-------|------|---------------|
| **LODGE** (Li et al., 2024) | CVPR | `ref_repo/LODGE` | Foot Refine Block: contact-aware foot position/velocity loss |
| **UnderPressure** (Mourot et al., 2022) | ECCV | `ref_repo/UnderPressure` | Deep learning foot contact detection + IK-based footskate cleanup |
| **OmniControl** (Xie et al., 2024) | ICLR | `ref_repo/OmniControl` | Per-joint spatial constraints via analytic + realism guidance |
| **KIMODO** (NVIDIA, 2025) | ArXiv | `ref_repo/KIMODO` | Global joint positions + foot contact labels in representation |
| **PhysDiff** (Yuan et al., 2023) | ICLR | N/A | Physics-based motion projection at each diffusion step |
| **GMD** (Karunratanakul et al., 2023) | ICLR | N/A | Guided motion diffusion with spatial constraints via classifier guidance |

### For Problem 3 (Boundary Jitter/Discontinuity)

| Paper | Venue | Repo | Key Technique |
|-------|-------|------|---------------|
| **CondMDI** (Cohan et al., 2024) | SIGGRAPH | `ref_repo/CondMDI` | Variable-density keyframe training; smooth boundaries via training distribution |
| **BrushNet** (Ju et al., 2024) | ECCV | `ref_repo/BrushNet` | Boundary-aware feature extraction for consistent inpainting edges |
| **PowerPaint** (Zhuang et al., 2024) | ECCV | N/A | Learnable task prompts (P_ctxt) for context-aware boundary coherence |
| **MoGenDiT** (Internal) | Internal | `ref_repo/MoGenDiT` | Mask-aware noise + skip_last imputation (same principle, DDPM framework) |
| **SDEdit** (Meng et al., 2022) | ICLR | N/A | Add noise then denoise for smooth blending (limited effectiveness in practice) |

### Downloaded Reference Code

All downloaded to `ref_repo/`:

| Repo | Source | Relevance |
|------|--------|-----------|
| `CondMDI` | [MaxC-CG/diffusion-motion-inbetweening](https://github.com/MaxC-CG/diffusion-motion-inbetweening) | Boundary-aware motion inpainting, variable keyframe density |
| `UnderPressure` | [InterDigitalInc/UnderPressure](https://github.com/InterDigitalInc/UnderPressure) | Foot contact detection + footskate IK cleanup |
| `LODGE` | [Sensible-life/LODGE](https://github.com/Sensible-life/LODGE) | Foot refine block with contact-aware losses |
| `OmniControl` | [jiayng01/OmniControl](https://github.com/jiayng01/OmniControl) | Per-joint spatial constraint via dual guidance |
| `BrushNet` | [TencentARC/BrushNet](https://github.com/TencentARC/BrushNet) | Boundary-consistent image inpainting (transferable principles) |
| `KIMODO` | (pre-existing) | Global representation + two-phase training + imputation |
| `MoGenDiT` | (pre-existing) | Mask-aware noise + DDPM imputation baseline |
| `UMO` | (pre-existing) | Frozen backbone + adapter approach |

---

## Appendix: Code Location Index

| Component | File | Key Lines |
|-----------|------|-----------|
| Training loop | `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | 49-324 |
| Inference pipeline | `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` | 100-372 |
| Model bundle | `hftrainer/models/motion/hymotion_m2m/bundle.py` | 289-338 (VACE) |
| Loss module | `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` | 46-181 |
| Mask strategies | `hftrainer/datasets/motion/motionhub/transforms/universal_mask.py` | 1-777 |
| V2 condition sampler | `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` | 1-255 |
| Encoders (I/O proj) | `hftrainer/models/motion/hymotion_m2m/network/encoders.py` | 1-127 |
| MMDiT architecture | `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` | Full file |
| System documentation | `hftrainer/models/motion/CLAUDE.md` | Full file |
