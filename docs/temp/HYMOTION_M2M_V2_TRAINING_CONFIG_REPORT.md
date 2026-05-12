# HyMotion M2M v2 Training Configuration Analysis

**Date**: 2026-05-12  
**Working Directory**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## Executive Summary

This report provides a detailed breakdown of the HyMotion M2M v2 training configuration, specifically for the `uncond_local` (unconditional) and `caption_local` (caption-conditioned) experiments. Key findings:

1. **Motion Representation**: 198-dim (3 trans + 132 rot6d + 63 position)
2. **Loss Configuration**: Sophisticated multi-term loss with KIMODO-style auxiliary terms
3. **Critical Finding**: **t² timestep-dependent weighting is applied to FK-related losses** (fk_consistency, joint_pos, joint_vel)
4. **Mask Sampling**: v3 universal Rank-K Boolean Tensor Prior (uncond) vs v2 Tier-2 templates (caption)

---

## 1. Configuration Files Overview

### 1.1 Unconditioned Local Model

**File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py` (46B parameters, 198-dim)

```python
# Key settings
model = dict(
    pred_type='velocity',
    uncondition_mode=True,
    text_encoder=None,
    cond_mask_prob=0.0,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
)

# Condition sampler: v3 (Universal Rank-K Boolean Tensor Prior)
sampler_version='v3'
editing_prob=0.15
```

**Launch**: 
```bash
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py 8 --auto-resume
# or Taiji (64 GPUs): python tools/taiji_submit.py m2m_v2_uncond_local ...
```

### 1.2 Caption-Conditioned Local Model

**File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py` (46B parameters, 198-dim)

```python
# Key settings
model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,  # CFG: 10% of samples get null embeddings
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
)

# Data-specific settings
train_dataloader = dict(
    batch_size=20,  # V100-32GB: text tokens (128×4096) add ~6GB
    # Uses pre-extracted Qwen3+CLIP embeddings
)

# Condition sampler: v2 (Tier-2 hard-coded templates)
tier2_prob=0.4
editing_prob=0.15
```

**Launch**:
```bash
python tools/train.py configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py
# or: bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py 8
```

---

## 2. Complete Loss Configuration

### 2.1 Base M2MLoss (from `_base_hymotion_m2m_v2_046b.py`)

**Class**: `M2MLoss` (location: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`)

```python
losses_cfg=dict(
    loss_type='smooth_l1',
    # ─────────────────────────────────────────────────────────────
    # PRIMARY LOSSES
    # ─────────────────────────────────────────────────────────────
    velocity_weight=1.0,              # Main flow-matching loss
    x1_weight=0.0,                    # Disabled (only used for x1 pred_type)
    motion_smoothness_weight=0.5,     # Frame-to-frame smoothness penalty
    
    # ─────────────────────────────────────────────────────────────
    # SECONDARY/DISABLED LOSSES
    # ─────────────────────────────────────────────────────────────
    keypoints3d_weight=0.0,           # Disabled (FK-based supervision)
    translation_weight=0.0,           # Disabled
    fk_consistency_weight=0.0,        # Disabled (using KIMODO aux instead)
    
    # ─────────────────────────────────────────────────────────────
    # DIMENSION REWEIGHTING
    # ─────────────────────────────────────────────────────────────
    trans_dim_weight=5.0,             # Upweight translation dims
                                      # [0:3] by 5× to compensate
                                      # for 3/135 dimension imbalance
    
    # ─────────────────────────────────────────────────────────────
    # WARMUP SCHEDULES
    # ─────────────────────────────────────────────────────────────
    fk_consistency_warmup_steps=2000,
)

velocity_loss_reduction='element_mean'  # or 'component_mean' (KIMODO-style)
```

**Loss Computation Details** (from `M2MLoss.forward()`):

1. **Velocity Loss**:
   - Compares predicted velocity (`pred_vel = pred - x0`) vs GT velocity (`gt_vel = x1 - x0`)
   - Applied per-dimension with `trans_dim_weight=5.0` scaling for translation dims
   - Masked by `data_mask_temporal` (padding aware) and optional `generation_mask`
   - **No timestep weighting**

2. **Motion Smoothness Loss** (weight=0.5):
   - Temporal smoothness: penalizes frame-to-frame velocity deviation
   - `smooth_loss = ||pred_x1[t+1:] - pred_x1[t:-1]|| - ||gt_x1[t+1:] - gt_x1[t:-1]||`
   - Mask requires both frame `t` and `t+1` to be valid
   - **No timestep weighting**

3. **FK Consistency Loss** (weight=0.0, disabled in base; handled by KIMODO aux):
   - Not computed here when KIMODO aux is active (avoids duplicate computation)

---

### 2.2 KIMODO-Style Auxiliary Losses

**Class**: `KimodoStyleAuxLoss` (location: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`)

This is the **critical new loss term** in v2. All three terms use **t² timestep weighting**.

```python
kimodo_aux_loss_cfg=dict(
    # ─────────────────────────────────────────────────────────────
    # WEIGHT MAGNITUDES (in denormalized metres space)
    # ─────────────────────────────────────────────────────────────
    joint_pos_weight=50.0,            # Global joint position loss (KIMODO γ₃)
    joint_vel_weight=500.0,           # Global joint velocity loss (KIMODO γ₄)
    fk_consistency_weight=1500.0,     # FK consistency loss (KIMODO γ₇)
    
    # ─────────────────────────────────────────────────────────────
    # LOSS TYPE AND TIMESTEP WEIGHTING
    # ─────────────────────────────────────────────────────────────
    loss_type='smooth_l1',
    timestep_squared_weighting=True,  # *** CRITICAL: t² weighting enabled ***
    
    # ─────────────────────────────────────────────────────────────
    # WARMUP SCHEDULES (linear from 0 to weight over N steps)
    # ─────────────────────────────────────────────────────────────
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
)
```

#### 2.2.1 Joint Position Loss (aux_joint_pos)

**Purpose**: Supervise global joint positions from FK (prevents foot skating/pelvis cheating)

**Computation** (lines 288-296 in `kimodo_aux_loss.py`):
```python
per_pt = smooth_l1_loss(pred_world, gt_world, reduction="none")  # (B,L,22,3)
per_frame = per_pt.mean(dim=(-1, -2))  # Average over joints & xyz → (B, L)

# *** t² WEIGHTING APPLIED HERE ***
if t_sq is not None:
    per_frame = per_frame * t_sq.unsqueeze(-1)  # (B, L) * (B, 1)

base = _temporal_mean_masked(per_frame, data_mask_temporal)
loss = joint_pos_weight * warmup * base
```

**Interpretation**:
- Base loss (no t² weighting) ~ O(1e-4) (cm-level pred-vs-GT joint position)
- With t² weighting (E[t²]=1/3), effective loss ~ 3× weaker
- **Upweight at t=1 (clean endpoint)**: t²=1
- **Downweight at t=0 (pure noise)**: t²~0

#### 2.2.2 Joint Velocity Loss (aux_joint_vel)

**Purpose**: Supervise global joint velocities from FK (detects velocity mismatches immediately)

**Computation** (lines 301-315 in `kimodo_aux_loss.py`):
```python
pred_vel = pred_world[:, 1:] - pred_world[:, :-1]    # (B, L-1, 22, 3)
gt_vel = gt_world[:, 1:] - gt_world[:, :-1]
per_pt = smooth_l1_loss(pred_vel, gt_vel, reduction="none")
per_frame = per_pt.mean(dim=(-1, -2))  # (B, L-1)

# *** t² WEIGHTING APPLIED HERE ***
if t_sq is not None:
    per_frame = per_frame * t_sq.unsqueeze(-1)

vel_mask = data_mask_temporal[:, 1:] * data_mask_temporal[:, :-1]  # both frames valid
base = _temporal_mean_masked(per_frame, vel_mask)
loss = joint_vel_weight * warmup * base
```

**Interpretation**:
- Base loss ~ O(1e-6) (mm/frame; T=360 sequences)
- Only valid when both frame `t` and `t+1` are in non-padded region
- **Strong discriminator for slipping**: velocity error at every joint, cannot cheat with pelvis

#### 2.2.3 FK Consistency Loss (aux_fk_consistency)

**Purpose**: Enforce that predicted position channels [135:198] match FK-derived positions from pred rotation/translation

**Computation** (lines 320-331 in `kimodo_aux_loss.py`):
```python
pred_pos_chan = pred_denorm[..., 135:]           # (B, L, 63)
fk_pos = _scheme_d_relative(pred_world)          # (B, L, 63)
per_pt = smooth_l1_loss(pred_pos_chan, fk_pos, reduction="none")
per_frame = per_pt.mean(dim=-1)  # (B, L)

# *** t² WEIGHTING APPLIED HERE ***
if t_sq is not None:
    per_frame = per_frame * t_sq.unsqueeze(-1)

base = _temporal_mean_masked(per_frame, data_mask_temporal)
loss = fk_consistency_weight * warmup * base
```

**Interpretation**:
- Base loss ~ O(1.4e-6) (mm-level intra-pred consistency on already FK-consistent representation)
- Teaches model explicit FK equivalence map inside 198-dim space
- Enables position-only inference (e.g., end-effector conditions) without IK
- Much larger nominal weight (1500 vs 50) because base value is ~70× smaller

---

### 2.3 t² Timestep Weighting Implementation

**Implementation Location**: `KimodoStyleAuxLoss.forward()` (lines 280-283)

```python
# Optional t² re-weighting (matches existing motion198_fk_loss).
if self.timestep_squared_weighting and timesteps is not None:
    t_sq = (timesteps.to(pred_world.device).to(pred_world.dtype) ** 2)  # (B,)
else:
    t_sq = None
```

**Also computed in**: `motion198_fk_loss()` in `compute_198dim.py` (lines 190-192)

```python
if timesteps is not None:
    t_sq = (timesteps ** 2).unsqueeze(-1)  # (B, 1)
    loss = loss * t_sq
```

**Timestep Range**: 
- Sampled uniformly in [0, 1] during training (from `hymotion_m2m_trainer.py` line 229)
- t=0: pure noise (x_t ≈ x0), t=1: clean data (x_t ≈ x1)
- E[t²] = ∫₀¹ t² dt = 1/3 ≈ 0.333

---

## 3. Loss Weight Interpretation & Relative Magnitudes

The config comment (lines 95-117 in `_base_hymotion_m2m_v2_046b.py`) provides detailed weight justification:

| Loss Term | Weight | Base Value | E[t²] Factor | Effective | % of loss_velocity |
|-----------|--------|------------|--------------|-----------|-------------------|
| **velocity** | 1.0 | ~0.025 (norm space) | N/A | ~0.025 | **100%** |
| **aux_joint_pos** | 50.0 | ~1e-4 (m) | 1/3 | ~5.0e-3 | **~14%** |
| **aux_joint_vel** | 500.0 | ~1e-6 (m/f) | 1/3 | ~1.0e-3 | **~4%** |
| **aux_fk_consistency** | 1500.0 | ~1.4e-6 (m) | 1/3 | ~2.1e-3 | **~7%** |
| **smoothness** | 0.5 | varies | N/A | varies | **~1%** |

**Key Insight**: The nominal weights (50, 500, 1500) are NOT KIMODO's γ values (which were ~1-10 in normalized space). Instead, they are calibrated to:
1. Account for denormalized space (metres vs. normalized units)
2. Compensate for t² averaging (E[t²]=1/3)
3. Target meaningful fractions of velocity loss

---

## 4. Motion Condition Sampling Configuration

### 4.1 Unconditioned Local (uncond_local_046b)

**Sampler Version**: `v3` (Universal Rank-K Boolean Tensor Prior)

```python
dict(
    type='PrepareM2Mv2Condition',
    key='motion',
    sampler_version='v3',           # ← v3 sampler
    editing_prob=0.15,              # 15% editing mode
    corruptor_names=[
        'jitter', 'joint_jump', 'sliding',
        'limb_candy_wrapper', 'wrist_candy_wrapper',
    ],
    max_corruptions=2,
),
```

**v3 Sampler Characteristics** (from `prepare_m2m_v2.py` line 107-110):
- Universal Rank-K Boolean Tensor Prior
- Covers **any structured motion-completion mask**: arbitrary period, joint subset, channel subset
- Replaces v2's Tier-2 hand-coded templates
- Reference: `docs/design/mask_prior_rank_k.md`
- Default values from `condition_sampler_v3.DEFAULT_*_WEIGHTS` (not shown in config, uses hardcoded defaults)

### 4.2 Caption-Conditioned Local (caption_local_046b)

**Sampler Version**: `v2` (Tier-2 templates)

```python
dict(
    type='PrepareM2Mv2Condition',
    key='motion',
    tier2_prob=0.4,                 # 40% of samples use Tier-2 patterns
    editing_prob=0.15,              # 15% editing mode
    corruptor_names=[
        'jitter', 'joint_jump', 'sliding',
        'limb_candy_wrapper', 'wrist_candy_wrapper',
    ],
    max_corruptions=2,
    tier2_weights={
        'pure_gen': 0.40,           # T2M: 40% of Tier2 = 16% global
        'inbetween': 0.15,
        'prefix': 0.10,
        'keyframes': 0.10,
        'end_effector': 0.08,
        'trajectory': 0.07,
        'foot_ground': 0.05,
        'edit_repair': 0.05,
    },
),
```

**Tier-2 Templates Coverage**:
- **pure_gen** (40%): Full sequence generation (T2M-like)
- **inbetween**: Inbetween-frame completion
- **prefix**: Begin-phrase continuation
- **keyframes**: Sparse keyframe-based completion
- **end_effector**: End-effector trajectory constraint
- **trajectory**: Root trajectory constraint
- **foot_ground**: Ground contact constraint
- **edit_repair**: Corruption repair

**Editing Mode** (both configs):
- **Probability**: 15% (`editing_prob=0.15`)
- **Corruption Pipeline**: Load original .npz file, apply random corruptions (jitter, joint_jump, etc.)
- **Mask Perturbation**: Apply over-masking on corrupted regions to be conservative

---

## 5. Model Architecture & Data Flow

### 5.1 Motion Representation (198-dim layout)

```
Dims [0:3]      → Translation (SMPL trans)
Dims [3:9]      → Root joint 6D rotation
Dims [9:135]    → 21 body joints × 6D rotation (row-major)
                  [Total: 1 + 21 = 22 joints]
Dims [135:198]  → 21 joints × 3D position (Scheme D: XZ rel-pelvis, Y absolute)
                  [Pelvis position is [0, pelvis_y, 0], redundant with trans, dropped]
```

### 5.2 Trainer Data Pipeline (from `hymotion_m2m_trainer.py`)

```
1. Load & normalize motion (src_motion, tgt_motion)
2. Zero out src_motion in mask=1 regions (completion mode)
3. Prepare padding & masks
4. Sample timesteps t ~ U(0,1)
5. Create x_t = (1-t)*x0 + t*x1 (flow matching interpolation)
6. Apply mask-aware noise (MAN): keep x_t[known] = x1[known] clean
7. Build VACE context: [x_t, reactive, mask] (3×D-dim input)
8. Forward through HyMotionMMDiT transformer
9. Compute losses (velocity + smoothness + KIMODO aux terms)
```

**Key**: Trainer handles t² weighting in KIMODO aux loss computation (passes `timesteps` to `_compute_kimodo_aux_loss()`)

---

## 6. Noise Scheduler & Flow Matching

```python
noise_scheduler_cfg=dict(method='euler'),
infer_noise_scheduler_cfg=dict(validation_steps=50),
```

- **Training**: Euler scheduler (simple ODE integration)
- **Inference**: 50 validation steps with Euler scheduler

---

## 7. Training Hyperparameters

### 7.1 Optimization

```python
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=[0.9, 0.99],
    weight_decay=0.0,
)
lr_scheduler = None  # No scheduling (constant learning rate)
```

### 7.2 Batch & Acceleration

```python
train_dataloader = dict(
    batch_size=28,  # uncond; caption config overrides to 20
    num_workers=4,
    persistent_workers=False,
    shuffle=True,
)

accelerator = dict(
    mixed_precision='no',           # Full FP32
    gradient_accumulation_steps=1,
)

train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=10,
    max_grad_norm=1.0,              # Gradient clipping
)
```

### 7.3 Checkpointing & EMA

```python
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1, iter_interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=100, save_last=True),
    ema=dict(type='EMAHook', decay=0.999, update_interval=1),
)
```

- **EMA Decay**: 0.999 (very slow exponential moving average)
- **Checkpoint**: Save every 10 epochs, keep last 100

---

## 8. Text Encoding (Caption-Conditioned Only)

### 8.1 Caption-Conditioned Config

```python
model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,             # CFG dropout: 10% null embeddings
    text_encoder=dict(),            # Qwen3-8B (CPU-based, pre-extracted)
)

# Pre-extracted embeddings
dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
```

### 8.2 Text Embedding Dimensions

```python
ctxt_input_dim=4096,       # CLIP context embeddings
vtxt_input_dim=768,        # Qwen3 VLM embeddings
max_text_len=128,          # Fixed sequence length (matches HY-Motion 1.0)
```

### 8.3 Null Embedding Handling

- When `text_ctxt_raw_length==0`, replace with learned null embeddings (`bundle.null_vtxt_feat`, `bundle.null_ctxt_input`)
- Ensures consistency with CFG dropout distribution during training

---

## 9. Pretrained Weights & Transfer Learning

```python
load_from = dict(
    _delete_=True,
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
```

- **Source**: HY-Motion T2M 1.0 pretrained (motion-to-motion generation baseline)
- **Loaded Components**: Motion transformer (18 transformer blocks)
- **Re-initialized**: Input/output layers (adapted for M2M conditioning)
- **Model Size**: 0.46B parameters (slim variant)

---

## 10. Summary: Loss Configuration Checklist

### Enabled Losses

- [x] **velocity** (weight=1.0) — Main flow-matching loss
- [x] **motion_smoothness** (weight=0.5) — Frame-to-frame smoothness
- [x] **aux_joint_pos** (weight=50.0, t² weighted) — Global joint position supervision
- [x] **aux_joint_vel** (weight=500.0, t² weighted) — Global joint velocity supervision
- [x] **aux_fk_consistency** (weight=1500.0, t² weighted) — FK consistency enforcement

### Disabled Losses

- [ ] **x1** (weight=0.0) — Not used (velocity prediction mode)
- [ ] **keypoints3d** (weight=0.0) — Disabled (redundant with aux losses)
- [ ] **translation** (weight=0.0) — Disabled
- [ ] **fk_consistency (M2MLoss)** (weight=0.0) — Disabled (using KIMODO aux instead)

### Dimension Reweighting

- **trans_dim_weight=5.0** — Upweight translation dims [0:3] by 5× (compensates for 3/135 imbalance)
- **velocity_loss_reduction='element_mean'** — Equal importance to all elements

---

## 11. Key Files Reference

| File | Purpose |
|------|---------|
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py` | Uncond config (v3 sampler) |
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py` | Caption config (v2 sampler) |
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` | Base config (shared) |
| `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | Trainer (loss computation) |
| `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` | M2MLoss implementation |
| `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py` | KIMODO auxiliary losses (t² weighting) |
| `hftrainer/datasets/motion/motionhub/transforms/compute_198dim.py` | Motion 135→198 conversion, FK loss |
| `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` | Condition sampling (v2/v3) |

---

## 12. Reproduction Notes

### To Run Unconditioned Training (8 GPUs local)
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py 8 --auto-resume
```

### To Run Caption-Conditioned Training (8 GPUs local)
```bash
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py 8
```

### To Run on Taiji (64 GPUs × 8 nodes)
```bash
python tools/taiji_submit.py m2m_v2_uncond_local configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py --host_num 8
```

---

## Appendix: t² Weighting Mathematical Justification

From the config comment (lines 105-110 in `_base_hymotion_m2m_v2_046b.py`):

> Combined with t² re-weighting (E[t²]=1/3) the raw base is ~3× weaker.
> The weights below target a meaningful fraction of loss_velocity (≈ 0.025 in normalised space):
> 
> - joint_pos:       50      ⇒ ≈ 5.0e-3   (~14% of loss_velocity)
> - joint_vel:       500     ⇒ ≈ 1.0e-3   (~ 4% of loss_velocity)
> - fk_consistency:  1500    ⇒ ≈ 2.1e-3   (~ 7% of loss_velocity)

**Effect of t² weighting**:
- At t=0 (pure noise): loss is 0 (no signal to supervise)
- At t=1 (clean data): loss is full weight
- Average impact: 1/3 of nominal weight

This is **intentional design** to down-weight supervision when the prediction is mostly noise, since FK on pure noise is uninformative.

