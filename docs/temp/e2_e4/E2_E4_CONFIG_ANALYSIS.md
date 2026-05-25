# HyMotion M2M v2 — E2 & E4 Caption-Conditioned Experiment Configs Analysis

Generated: 2026-05-15

## Summary

Both E2 (SMPL Root baseline) and E4 (KIMODO Root with ADMM smoothing) are caption-conditioned M2M training experiments using **Classifier-Free Guidance (CFG)** during training. They inherit from a common base config and differ primarily in their root representation and motion preprocessing.

---

## CRITICAL TEXT-CONDITIONING SETTINGS

### 1. **uncondition_mode** — CFG Control (🟢 ENABLED TEXT CONDITIONING)

| Setting | E2 | E4 | Implication |
|---------|-----|-----|------------|
| **uncondition_mode** | `False` | `False` | ✅ **Text conditioning is ENABLED during inference** |
| Base default | `True` | `True` | Base would have CFG disabled (uncondition_mode=True means ignore text) |

**CRITICAL**: When `uncondition_mode=False`, CFG is available for use at inference time if `text_guidance_scale > 1.0`.

### 2. **cond_mask_prob** — CFG Training Signal (🟢 ENABLED CFG TRAINING)

| Setting | E2 | E4 | Interpretation |
|---------|-----|-----|------------|
| **cond_mask_prob** | `0.1` (10%) | `0.1` (10%) | ✅ **CFG enabled during training** |
| Base default | `0.0` | `0.0` | Base would NOT have CFG training |
| Range for CFG training | Typical: 0.1-0.2 | Typical: 0.1-0.2 | Per CLAUDE.md: "10% unconditional during training" |

**Mechanism**: During training, 10% of batch gets `mask_text_cond()` applied, forcing the model to learn from unconditional (null text) guidance signals. This teaches the model to use text information effectively for CFG.

### 3. **text_guidance_scale** — CFG Guidance Strength (🔧 PIPELINE DEFAULT: 1.0)

| Setting | E2 | E4 | Default | Notes |
|---------|-----|-----|---------|-------|
| **text_guidance_scale** | Not in config | Not in config | `1.0` (in pipeline) | Set at inference time via `HyMotionM2MPipeline` |
| Pipeline location | `hymotion_m2m_pipeline.py` line 86 | Same | Default init | No guidance (scale=1.0 means pure model) |
| Activation rule | Scale > 1.0 AND uncondition_mode=False | Same | See code line 221 | **Only active if scale > 1.0 AND text enabled** |

**Inference CFG formula** (pipeline line 277):
```python
if text_guidance_scale > 1.0 and not bundle.uncondition_mode:
    x_pred = pred_basic + text_guidance_scale * (pred_text - pred_basic)
```

To enable CFG at inference, caller must set `text_guidance_scale > 1.0` (e.g., 7.5, 10.0).

---

## TEXT EMBEDDING & DATASET CONFIGURATION

### 4. **Text Embedding Source**

**E2 Config** (line 65):
```python
dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
```

**E4 Config** (line 80):
```python
dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
```

**Interpretation**:
- Loads pre-computed embeddings from file (not computing on-the-fly)
- Key is `'caption'` (text captions from dataset)
- `allow_none=True` = gracefully skip if embedding file missing
- **File location**: Determined by `LoadPreExtractedTextEmbedding` transform (typically in per-sample annotation or motion data directory)

### 5. **Text Embedding Format** (from PackInputs)

**Both E2 & E4** (lines 102-103):
```python
'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length',
```

**Dual-encoder text representation**:
- `text_vec_raw`: Sentence-level embedding (likely CLIP-L, 768-dim)
- `text_ctxt_raw`: Token-level contextual embeddings (likely Qwen3, 4096-dim)
- `text_ctxt_raw_length`: Sequence length of token embeddings (for masking)

**Default text encoder** (line 47):
```python
text_encoder=dict(),  # Use default QWEN3 + CLIP-L
```

Empty dict means use factory defaults: Qwen3 (4096-dim) for token-level context, CLIP-L (768-dim) for sentence embedding.

### 6. **Dataset Annotation File**

**Both E2 & E4** (line 62):
```python
anno_file='data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json',
```

**Characteristics**:
- `train_hymotion_400h_hq_permo_motionfix_editing_20260514.json` = high-quality filtered dataset
- `_hq_` = high-quality curated subset
- `_permo_` = includes PerMo (Performance Motion) editing pairs
- `_motionfix_` = motion quality corrected
- `_editing_` = supports editing paradigm (reactive channel for corrupted motion repair)

---

## LOSS CONFIGURATION

### 7. **losses_cfg** — Base M2M Loss + KIMODO Auxiliary Loss

**Base Config** (`_base_hymotion_m2m_v2_046b.py` lines 58-95):

**M2M Loss** (primary motion denoising):
```python
losses_cfg=dict(
    loss_type='smooth_l1',              # Smooth L1 loss
    velocity_weight=1.0,                 # Main loss
    x1_weight=0.0,                       # Don't use x1 (final position)
    keypoints3d_weight=0.0,              # No keypoint supervision (base)
    translation_weight=0.0,              # No explicit translation loss
    trans_dim_weight=5.0,                # Emphasize translation dims slightly
    motion_smoothness_weight=0.5,        # Encourage smooth motion
    fk_consistency_weight=0.0,           # Disabled (uses KIMODO aux instead)
    fk_consistency_warmup_steps=2000,
)
```

**KIMODO-Style Auxiliary Loss** (FK-based consistency):
```python
    # aux_joint_pos ≈ γ₃ (FK-derived joint position consistency)
    aux_joint_pos_weight=50.0,           # → ~5.0e-3 of velocity loss
    
    # aux_joint_vel ≈ γ₄ (global joint velocity smoothness)
    aux_joint_vel_weight=500.0,          # → ~1.0e-3 of velocity loss
    
    # aux_fk_consistency ≈ γ₇ (pos-channel ↔ FK rotation consistency)
    aux_fk_consistency_weight=1500.0,    # → ~2.1e-3 of velocity loss
    
    aux_timestep_squared_weighting=True, # Weight by t² to suppress noisy early steps
    aux_fk_consistency_warmup_steps=2000,
    aux_joint_pos_warmup_steps=2000,
    aux_joint_vel_warmup_steps=2000,
```

**E2 Override** (lines 48-54):
```python
losses_cfg=dict(
    keypoints3d_weight=10.0,             # ✅ ENABLE keypoint supervision (vs base 0.0)
    velocity_loss_reduction='component_mean',  # Decompose velocity loss
)
```

**E4 Override** (lines 63-69):
```python
losses_cfg=dict(
    keypoints3d_weight=10.0,             # ✅ ENABLE keypoint supervision (vs base 0.0)
    velocity_loss_reduction='component_mean',  # Decompose velocity loss
)
```

**Difference**:
- Base: `keypoints3d_weight=0.0` (no FK position supervision)
- E2/E4: `keypoints3d_weight=10.0` (strong FK position supervision)
- Both also use `velocity_loss_reduction='component_mean'` for per-component monitoring (translation, root rotation, body rotation, joint position)

---

## 8. **enable_ctxt_null_feat** — DEPRECATED (No Longer Used)

| Setting | E2 | E4 | Status |
|---------|-----|-----|--------|
| **enable_ctxt_null_feat** | Not set (default False) | Not set (default False) | 🔴 **DEPRECATED** |
| **Actual behavior** | Ignored at inference | Ignored at inference | See bundle.py line 187-190 |

**From CLAUDE.md** (bundle.py lines 187-190):
```python
# DEPRECATED: enable_ctxt_null_feat is no longer used by the pipeline.
# Since 2026-05-15, inference CFG always nulls both vtxt and ctxt to
# match training-time mask_text_cond behavior. Kept for checkpoint compat.
self.enable_ctxt_null_feat = bool(enable_ctxt_null_feat)
```

**Current behavior**: CFG pipeline (line 223-226 of hymotion_m2m_pipeline.py) ALWAYS nulls both sentence-level vtxt AND token-level ctxt:
```python
# CFG null-branch construction.  The "silent" CFG branch nulls BOTH
# sentence-level vtxt AND token-level ctxt to match training-time
# mask_text_cond behavior (which masks both vtxt and ctxt).
```

**Conclusion**: This setting is vestigial; all CFG inference uses complete null (both vtxt=null, ctxt=null).

---

## NULL EMBEDDINGS FOR CFG

### 9. **null_embedding_source** — Initialization of Null Text Features

**E2 & E4** (lines 38 & 52):
```python
null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
```

**Purpose**: During checkpoint loading, if `null_vtxt_feat` or `null_ctxt_input` are zero/uninitialized, they are patched from this T2M pretrained checkpoint.

**From bundle.py lines 204-213**:
```python
# Trainable: initialized with small random values. During M2M training,
# these embeddings learn the "no text condition" representation jointly
# with the transformer. This allows CFG to work correctly: when text_available=False,
# the model sees null_embeddings which are distinct from real text embeddings,
# enabling the transformer to learn meaningful text conditioning via the guidance
# signal (pred_with_text - pred_with_null). Frozen null embeddings cause CFG
# to fail because null and real embeddings appear equivalent to the model.
self.null_vtxt_feat = nn.Parameter(torch.randn(1, 1, vtxt_input_dim) * 0.01, requires_grad=True)
self.null_ctxt_input = nn.Parameter(torch.randn(1, 1, ctxt_input_dim) * 0.01, requires_grad=True)
```

**Critical detail**: `requires_grad=True` means null embeddings are **TRAINABLE** during M2M training. They learn the "no text" representation. Do NOT freeze them.

---

## ROOT REPRESENTATION DIFFERENCES

### E2 vs E4 — The Key Distinction

**E2: SMPL Root Baseline**
- Rotation space: `'local'` (SMPL frame, parent-relative)
- Mean/std dir: `'data/hymotion_m2m_data/_stats_198dim'` (SMPL Root stats)
- No ADMM smoothing
- Processing: `LoadSmplx55` → `Compute198DimPosition` → condition sampling

**E4: KIMODO Root with ADMM Online Smoothing**
- Rotation space: `'local'` (SMPL frame for output compatibility)
- Mean/std dir: `'data/hymotion_m2m_data/_stats_198dim_kimodo_root'` (KIMODO Root stats)
- **New step**: `SmplTransToKimodoRootOnline` (line 95-98):
  ```python
  dict(
      type='SmplTransToKimodoRootOnline',
      key='motion',
      admm_margin_m=0.06,  # 6cm margin on XZ plane
  ),
  ```
- ADMM smoothing: Pelvis translation smoothed during dataset __getitem__ with 6cm margin on XZ plane
- Also applied to source motion (line 123-126):
  ```python
  dict(
      type='LoadEditingSourceMotion',
      kimodo_root_cfg=dict(admm_margin_m=0.06),
  ),
  ```

**KIMODO Root 198-dim layout** (E4 config lines 15-19):
```
[0:3]      ADMM smoothed pelvis translation (online smoothing during load)
[3:9]      root joint 6D rotation (continuous)
[9:135]    body (21 non-root joints) 6D rotations
[135:198]  FK-derived joint positions relative to pelvis (21 × 3)
```

**E4 Loss Difference**: Enables `timestep_squared_weighting=True` implicitly (base default) to suppress noisy-FK spikes from ADMM smoothing artifacts.

---

## BATCH SIZE & DATA LOADING

### 10. **Batch Size and Worker Configuration**

**Both E2 & E4** (lines 58-60):
```python
batch_size=20,           # Reduced from base 28 (caption uses higher memory)
num_workers=8,           # Increased from base 4
persistent_workers=True, # Avoid per-epoch worker restart overhead
```

**Rationale**: Caption models have higher VRAM footprint due to dual text encoders (Qwen3 4096 + CLIP 768) per sample. Reduce batch size from 28→20, increase workers to maintain DataLoader prefetch efficiency.

---

## INHERITANCE HIERARCHY

### Base Config Loading

**Both E2 & E4**:
```python
_base_ = './_base_hymotion_m2m_v2_046b.py'
```

**_base_hymotion_m2m_v2_046b.py inherits from**:
```python
_base_ = '../_base_/default_runtime.py'
```

**Override chain**:
1. `default_runtime.py` — global training defaults
2. `_base_hymotion_m2m_v2_046b.py` — M2M v2 base (198-dim motion, MMDiT 0.46B, uncond_mode=True, cond_mask_prob=0.0)
3. `hymotion_m2m_v2_smpl_caption_046b.py` (E2) — SMPL Root + caption (uncond_mode=False, cond_mask_prob=0.1)
4. `hymotion_m2m_v2_kimodo_caption_046b.py` (E4) — KIMODO Root + caption (+ ADMM smoothing)

**Resume Checkpoints**:
- Both load from intermediate caption phase training: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`
- `load_scope='model'` resets optimizer/scheduler (loss config changed)
- E4 uses `exclude_bundle_keys=['mean', 'std']` to prevent SMPL mean/std from overwriting KIMODO mean/std

---

## PIPELINE INFERENCE CONFIGURATION

**From `hymotion_m2m_pipeline.py` (lines 48-103)**:

```python
class HyMotionM2MPipeline:
    def __init__(
        self,
        bundle,
        num_steps: int = 50,
        text_guidance_scale: float = 1.0,      # ← DEFAULT: NO CFG
        replacement_guidance: str = 'none',
        position_constraint_interval: int = 5,
        max_text_len: int = 128,
        sdedit_tau: float = 0.0,
    ):
```

**Key inference parameters**:
- `num_steps=50`: ODE integration steps
- `text_guidance_scale=1.0` (default): **NO guidance** (scale must be > 1.0 to activate)
- `replacement_guidance='none'` (default): No imputation per step (use for standard models; use `'skip_last'` for `_man` variants)
- `max_text_len=128`: Must match trainer's value for attention mask compatibility

**CFG Activation Rule** (line 221):
```python
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode
```

---

## COMPLETE SETTINGS TABLE

| Setting | E2 | E4 | Category |
|---------|-----|-----|----------|
| **uncondition_mode** | False ✅ | False ✅ | Text conditioning |
| **cond_mask_prob** | 0.1 ✅ | 0.1 ✅ | CFG training |
| **text_guidance_scale** | 1.0 (default in pipeline) | 1.0 (default in pipeline) | CFG inference (must set > 1.0 to activate) |
| **enable_ctxt_null_feat** | N/A (deprecated) | N/A (deprecated) | Ignored |
| **null_embedding_source** | checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt | Same | Null embedding init |
| **text_embedding_source** | LoadPreExtractedTextEmbedding | Same | Pre-computed embeddings |
| **text_embedding_dims** | 768 (CLIP-L) + 4096 (Qwen3) | Same | Dual-encoder |
| **losses_cfg.keypoints3d_weight** | 10.0 ✅ | 10.0 ✅ | FK supervision |
| **losses_cfg.velocity_loss_reduction** | 'component_mean' | 'component_mean' | Loss decomposition |
| **losses_cfg.aux_joint_pos_weight** | 50.0 (inherited) | 50.0 (inherited) | KIMODO aux loss |
| **losses_cfg.aux_joint_vel_weight** | 500.0 (inherited) | 500.0 (inherited) | KIMODO aux loss |
| **losses_cfg.aux_fk_consistency_weight** | 1500.0 (inherited) | 1500.0 (inherited) | KIMODO aux loss |
| **rotation_space** | 'local' | 'local' | SMPL compatibility |
| **mean_std_dir** | `_stats_198dim` | `_stats_198dim_kimodo_root` | Motion normalization |
| **ADMM smoothing** | None | 6cm XZ margin | Root preprocessing |
| **batch_size** | 20 | 20 | Training |
| **num_workers** | 8 | 8 | Data loading |
| **load_from** | caption_local_phase2 epoch 3370 | caption_local_phase2 epoch 3370 | Checkpoint resume |

---

## CRITICAL POINTS FOR CFG INFERENCE

### ✅ E2 and E4 ARE CFG-READY

1. **Training**: Both use `cond_mask_prob=0.1`, teaching model to respond to CFG
2. **Model**: Both have `uncondition_mode=False`, enabling CFG at inference
3. **Null embeddings**: Both initialized from T2M pretrained, trainable during M2M training
4. **Text embeddings**: Pre-extracted Qwen3 + CLIP-L, loaded at training time

### 🔴 TO ENABLE CFG AT INFERENCE

Caller must:
1. Instantiate pipeline with `text_guidance_scale > 1.0` (e.g., 7.5 or 10.0)
2. Provide text input (either raw text or pre-extracted embeddings)
3. Ensure `uncondition_mode=False` in checkpoint (✅ both E2/E4 have this)

### ⚠️ KIMODO AUXILIARY LOSS

Both use KIMODO-style auxiliary loss (FK-based consistency) from base config:
- `aux_joint_pos_weight=50.0`
- `aux_joint_vel_weight=500.0`
- `aux_fk_consistency_weight=1500.0`
- `aux_timestep_squared_weighting=True`

This loss is computed post-hoc via FK on denormalized motion. Not critical for CFG but improves motion quality consistency.

---

## REFERENCES

- **Main configs**: `/configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py` (E2), `hymotion_m2m_v2_kimodo_caption_046b.py` (E4)
- **Base config**: `/configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- **Pipeline**: `/hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`
- **Bundle**: `/hftrainer/models/motion/hymotion_m2m/bundle.py`
- **Proposal**: `/docs/temp/hymotion_m2m_next_gen_proposal_20260511.md` (§8.2 E2, §8.3 E4)
- **CLAUDE.md**: Full motion representation and training conventions

