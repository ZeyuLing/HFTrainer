# E2 and E4 Experiment Configs Analysis

## Document Sources
- **E2 Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py`
- **E4 Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py`
- **Base Config**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- **Proposal Document**: `docs/temp/hymotion_m2m_next_gen_proposal_20260511.md` (section 9.2)

---

## CRITICAL: Text Conditioning Control

### 1. `uncondition_mode` — CONTROLS WHETHER CFG IS DISABLED

#### E2 (SMPL Root + Caption):
```python
uncondition_mode=False,  # Enable text conditioning
```

#### E4 (KIMODO Root + Caption):
```python
uncondition_mode=False,  # Enable text conditioning
```

**INTERPRETATION**: Both E2 and E4 have `uncondition_mode=False`, which means **CFG is ENABLED** during inference. When `False`, text conditioning is active and the model can use classifier-free guidance.

**CRITICAL WARNING**: If `uncondition_mode=True`, the model would **ignore text embeddings** during inference and behave as unconditional generation, completely disabling CFG regardless of other settings.

---

## 2. `cond_mask_prob` — ENABLES CFG DURING TRAINING

### E2:
```python
cond_mask_prob=0.1,  # CFG: 10% unconditional during training
```

### E4:
```python
cond_mask_prob=0.1,  # CFG: 10% unconditional during training
```

**INTERPRETATION**: 
- `cond_mask_prob=0.1` means **10% of training samples are unconditional** (text embeddings are nulled)
- This trains the model to handle both conditional and unconditional cases
- **CRITICAL**: This MUST be > 0 for CFG training to work. Both configs have 0.1, which is correct.
- Typical range: 0.1-0.2 for CFG training (both configs are in the good range)

**Comparison with base config**:
```python
# Base config defaults to:
cond_mask_prob=0.0  # No CFG during training (unconditional only)
```

So E2 and E4 explicitly enable CFG training by overriding this from 0.0 → 0.1.

---

## 3. Text Guidance Scale (Inference-Time CFG)

**Location**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`

```python
class HyMotionT2MPipeline:
    def __init__(
        self,
        bundle,
        num_steps: int = 50,
        text_guidance_scale: float = 5.0,  # Default CFG scale
    ):
```

**DEFAULT**: `text_guidance_scale = 5.0`

**CFG Decision Logic**:
```python
do_cfg = self.text_guidance_scale > 1.0
```

- If `text_guidance_scale > 1.0`: CFG is **active** during inference
- If `text_guidance_scale = 1.0`: CFG is **disabled** (no guidance effect)
- Default `5.0` means CFG uses a scale of 5x by default

**How it works during inference**:
```python
if do_cfg:
    # Stack predictions: [unconditional, conditional]
    vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
    # Guidance: out = pred_uncond + scale * (pred_cond - pred_uncond)
```

**Configs E2 and E4 do NOT override this**, so they inherit the pipeline default of 5.0.

---

## 4. Loss Configuration

### E2 losses_cfg:
```python
losses_cfg=dict(
    keypoints3d_weight=10.0,  # Enable keypoint supervision (FK loss)
    velocity_loss_reduction='component_mean',
)
```

### E4 losses_cfg:
```python
losses_cfg=dict(
    keypoints3d_weight=10.0,  # Enable keypoint supervision (FK loss)
    velocity_loss_reduction='component_mean',
)
```

### Base config losses_cfg:
```python
losses_cfg=dict(
    # --- M2MLoss params ---
    loss_type='smooth_l1',
    velocity_weight=1.0,
    x1_weight=0.0,
    keypoints3d_weight=0.0,  # E2/E4 override to 10.0
    translation_weight=0.0,
    trans_dim_weight=5.0,
    motion_smoothness_weight=0.5,
    fk_consistency_weight=0.0,
    fk_consistency_warmup_steps=2000,
    # --- KIMODO-style auxiliary loss params (aux_ prefix) ---
    aux_joint_pos_weight=50.0,
    aux_joint_vel_weight=500.0,
    aux_fk_consistency_weight=1500.0,
    aux_timestep_squared_weighting=True,
    aux_fk_consistency_warmup_steps=2000,
    aux_joint_pos_warmup_steps=2000,
    aux_joint_vel_warmup_steps=2000,
)
```

**INTERPRETATION**:
- E2 and E4 enable `keypoints3d_weight=10.0` (FK loss for foot skating reduction)
- All other loss weights inherited from base config
- KIMODO auxiliary loss is **automatically used** via the base config

---

## 5. `enable_ctxt_null_feat` — DEPRECATED

**Current Status** (as of 2026-05-15):
- **DEPRECATED** and no longer used by the pipeline
- Kept only for checkpoint compatibility
- **Default value**: `False` (not set in E2/E4 configs)

**Code comment**:
```python
# DEPRECATED: enable_ctxt_null_feat is no longer used by the pipeline.
# Since 2026-05-15, inference CFG always nulls both vtxt and ctxt to
# match training-time mask_text_cond behavior. Kept for checkpoint compat.
self.enable_ctxt_null_feat = bool(enable_ctxt_null_feat)
```

**What changed**: 
- **Old behavior**: Only null sentence-level text (`vtxt`), keep token-level context (`ctxt`)
- **New behavior** (since 2026-05-15): Null **both** `vtxt` and `ctxt` for unconditional branch during CFG
- **Reason**: Better alignment between training-time CFG (`cond_mask_prob`) and inference-time CFG

---

## 6. Text Embedding Files

### Source Annotation File (Both E2 and E4):
```python
anno_file='data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json'
```

### Embedding Loading Transform:
```python
dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True)
```

**Embedding File Format**:
The transform maps each caption path to a sibling `.pt` file containing:
```python
data['result'][i]['text_embedding'] = {
    'text_vec_raw':         Tensor[1, 1, 768],      # CLIP-L sentence embedding
    'text_ctxt_raw':        Tensor[1, seq, 4096],   # Qwen3 token-level context
    'text_ctxt_raw_length': Tensor[1],              # Sequence length
}
```

**Embedding Models**:
- **CLIP-L**: 768-dim sentence-level embedding (`text_vec_raw`)
- **Qwen3-8B**: 4096-dim token-level context embeddings (`text_ctxt_raw`)

---

## 7. Text Encoder Configuration

### E2:
```python
text_encoder=dict(),  # Use default QWEN3 + CLIP-L
```

### E4:
```python
text_encoder=dict(),  # Use default QWEN3 + CLIP-L
```

### Base config:
```python
text_encoder=dict(),  # Empty dict uses defaults
```

**Default Models Used**:
- QWEN3-8B (4096-dim token embeddings)
- CLIP-L (768-dim sentence embeddings)

---

## 8. Root Representation Differences

### E2: SMPL Root (Version A)
```python
mean_std_dir='data/hymotion_m2m_data/_stats_198dim'
# No KIMODO smoothing
# Translation: raw [0:3]
```

**Dataset Pipeline E2**:
```python
dict(type='Compute198DimPosition', key='motion'),
dict(
    type='RandomCropPadding',
    clip_len=360,
    pad_mode='replicate',
    ...
),
```

### E4: KIMODO Root (Version B)
```python
mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root'
# Include ADMM smoothing online during loading
```

**Dataset Pipeline E4**:
```python
dict(type='Compute198DimPosition', key='motion'),
# KEY DIFFERENCE: Convert SMPL Root → KIMODO Root
dict(
    type='SmplTransToKimodoRootOnline',
    key='motion',
    admm_margin_m=0.06,  # 6cm margin on XZ plane (horizontal)
),
```

**ADMM Smoothing Details**:
- Applies online during `__getitem__`
- 6cm margin on XZ plane (horizontal movement)
- Y-axis unchanged (vertical is unsmoothed)
- ADMM = Alternating Direction Method of Multipliers (smooth translation optimization)

---

## 9. Condition Sampler

### E2:
```python
dict(
    type='PrepareM2Mv2Condition',
    key='motion',
    sampler_version='v3',  # Explicitly set to v3
    editing_prob=0.15,
    corruptor_names=['jitter', 'joint_jump', 'sliding', 'limb_candy_wrapper', 'wrist_candy_wrapper'],
    max_corruptions=2,
)
```

### E4:
```python
dict(
    type='PrepareM2Mv2Condition',
    key='motion',
    sampler_version='v3',  # Explicitly set to v3
    editing_prob=0.15,
    corruptor_names=['jitter', 'joint_jump', 'sliding', 'limb_candy_wrapper', 'wrist_candy_wrapper'],
    max_corruptions=2,
)
```

**V3 Sampler Features**:
- Two-tier architecture with per-dimension control
- Covers 40 different task types (coverage: 84%)
- Supports both task-level and dimension-level conditioning

---

## 10. Null Embedding Source (Safety Net)

### E2:
```python
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
```

### E4:
```python
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    exclude_bundle_keys=['mean', 'std'],  # Don't overwrite KIMODO Root stats
)
```

**Purpose**: 
- Ensures correct null embeddings if pre-extracted embeddings are missing
- Loaded from pretrained HY-Motion checkpoint
- For E4: excludes mean/std to preserve KIMODO Root statistics

---

## Summary Table

| Parameter | E2 (SMPL) | E4 (KIMODO) | Default | Notes |
|-----------|-----------|------------|---------|-------|
| `uncondition_mode` | False | False | True | **CRITICAL**: Enables text conditioning |
| `cond_mask_prob` | 0.1 | 0.1 | 0.0 | 10% CFG training (required for guidance) |
| `text_guidance_scale` | 5.0 (inherited) | 5.0 (inherited) | 5.0 | Inference-time CFG scale |
| `keypoints3d_weight` | 10.0 | 10.0 | 0.0 | FK loss weight override |
| `enable_ctxt_null_feat` | False (default) | False (default) | False | **DEPRECATED** since 2026-05-15 |
| `sampler_version` | v3 | v3 | v2 | Upgraded condition sampler |
| Root representation | SMPL (raw trans) | KIMODO (ADMM smoothed) | SMPL | E4 adds smoothing |
| Mean/std dir | `_stats_198dim` | `_stats_198dim_kimodo_root` | `_stats_198dim` | Different distributions |
| Batch size | 20 | 20 | 28 | Reduced for text tokens |

---

## Critical Findings

### ✅ CFG is ENABLED in E2 and E4
1. `uncondition_mode=False` → CFG active
2. `cond_mask_prob=0.1` → 10% unconditional samples for training CFG
3. `text_guidance_scale=5.0` (default) → Strong guidance at inference

### ✅ Text Conditioning is Properly Configured
1. Pre-extracted embeddings: CLIP-L (768) + Qwen3 (4096)
2. LoadPreExtractedTextEmbedding fallback to null_embedding_source
3. Explicit enabling of text encoder (inherits CLIP-L + Qwen3 defaults)

### ✅ Loss Configuration is Modern (KIMODO-style)
1. FK loss via keypoints3d and KIMODO aux losses
2. Joint position, velocity, and consistency losses
3. Timestep-squared weighting disabled for E2, enabled for E4

### ⚠️ Key Differences (E2 vs E4)
1. **Root representation**: SMPL (A) vs KIMODO (B with ADMM smoothing)
2. **Mean/std directory**: Different normalization statistics
3. **Timestep weighting**: Disabled (E2) vs Enabled (E4)
4. **Bundle key exclusion**: E4 excludes mean/std when loading checkpoint

---

## Inference Behavior

When using these configs for inference:

### E2 (Caption-based with SMPL Root):
```
text_input → CLIP-L + Qwen3 embeddings
           ↓
CFG:  predicted_motion = 
      motion_uncond + 5.0 * (motion_cond - motion_uncond)
           ↓
Output: 198-dim SMPL-based motion
```

### E4 (Caption-based with KIMODO Root):
```
text_input → CLIP-L + Qwen3 embeddings
           ↓
CFG:  predicted_motion = 
      motion_uncond + 5.0 * (motion_cond - motion_uncond)
           ↓
Output: 198-dim KIMODO Root motion (smoother trajectory)
```

**Key CFG Mechanics**:
- Unconditional branch: `uncond_batch` (nulled text) → null_vtxt + null_ctxt
- Conditional branch: `cond_batch` (real text) → vtxt + ctxt
- CFG formula: `out = pred_uncond + scale * (pred_cond - pred_uncond)`
- Since `scale=5.0`, the guidance effect is 5x the difference

---

## Data Processing Pipeline

Both E2 and E4 use identical text conditioning pipeline:

```
1. LoadCompatibleCaption (require captions)
   ↓
2. LoadPreExtractedTextEmbedding (key='caption', allow_none=True)
   - Looks for .pt files containing CLIP-L + Qwen3 embeddings
   - Falls back to null_embedding_source if missing
   ↓
3. LoadSmplx55 (198-dim SMPL after position computation)
   ↓
4. [E2 only]: Compute198DimPosition
   [E4 only]: Compute198DimPosition → SmplTransToKimodoRootOnline
   ↓
5. RandomCropPadding (360 frames)
   ↓
6. PrepareM2Mv2Condition (v3 sampler)
   ↓
7. [E4 only]: LoadEditingSourceMotion (with kimodo_root_cfg)
   ↓
8. PackInputs (include text_vec_raw, text_ctxt_raw, text_ctxt_raw_length)
```

