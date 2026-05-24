# HyMotion M2M v2 Caption/Text Training Configs - Complete Analysis

## Summary
Found **16 caption/text-related configs** in `configs/hymotion_m2m_v2/`:
- 14 main caption configs (local/global variants, phases, experiments)
- 3 T2M pretrained variant configs
- 2 SOAR post-training configs for caption models

---

## ARCHITECTURE OVERVIEW

### Base Config (_base_hymotion_m2m_v2_046b.py)
**Purpose**: Foundation config defining core M2M v2 architecture
- **Motion dim**: 198 (3D trans + 132D rot6d + 63D joint positions)
- **VACE mode**: `no_inactive` (input = x_t + reactive + mask = 3×D = 594-dim)
- **uncondition_mode**: `True` (default, caption configs override to False)
- **cond_mask_prob**: 0.0 (default, caption configs override to 0.1)
- **Transformer**: 18 blocks, 1024 feat_dim, 16 heads
- **Text encoders**: QWEN3 (4096-dim context) + CLIP-L (768-dim)
- **Load from**: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (T2M 1.0 pretrained)

---

## CAPTION TRAINING CONFIGS (16 Total)

### GROUP 1: Standard Caption Configs (Local/Global)

#### 1. hymotion_m2m_v2_caption_global_046b.py
- **Purpose**: Caption-conditioned + Global rotation
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py`
- **Size**: 2.8K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key config**:
  - `uncondition_mode=False` (caption enabled)
  - `cond_mask_prob=0.1` (CFG: 10% unconditional)
  - `rotation_space='global'` (world frame)
  - `mean_std_dir='data/hymotion_m2m_data/_stats_198dim_global_rot'`
  - `batch_size=20` (vs base 28 due to text token memory)
- **Data pipeline**:
  - `LoadPreExtractedTextEmbedding` (pre-extracted Qwen3+CLIP embeddings)
  - `LocalToGlobalRotation` applied
  - `PrepareM2Mv2Condition` with tier2_weights (40% pure_gen T2M)
- **Load checkpoint**: Base T2M pretrained
- **Status**: Production config, used as Phase 1 baseline

#### 2. hymotion_m2m_v2_caption_local_046b.py
- **Purpose**: Caption-conditioned + Local rotation (SMPL frame)
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py`
- **Size**: 2.7K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key config** (vs caption_global):
  - `rotation_space='local'` (SMPL frame, no LocalToGlobalRotation)
  - `mean_std_dir='data/hymotion_m2m_data/_stats_198dim'` (standard stats)
  - Everything else identical to caption_global
- **Data pipeline**: Same as caption_global, except NO LocalToGlobalRotation
- **Load checkpoint**: Base T2M pretrained
- **Status**: Production config, used as standard local baseline

---

### GROUP 2: Curriculum Learning Phases (Caption + Local)

#### 3. hymotion_m2m_v2_caption_local_phase1.py
- **Purpose**: Phase 1 - Pure T2M generation (all mask=1)
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py`
- **Size**: 3.1K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences**:
  - `fk_consistency_weight=0.1` (lightweight FC loss during T2M-only phase)
  - **Critical**: Uses `PrepareM2Mv2FullMask` (all samples: mask=1, pure T2M)
  - `mask_aware_noise=False` (MAN has no effect with all mask=1)
  - `max_epochs=50` (short phase)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_183/model.safetensors`
    (pretrained caption_local at epoch 183)
  - Null embedding patch: T2M pretrained
- **Data pipeline**: Pure T2M, no completion/editing
- **Training setup**:
  - Batch size: 20
  - Max grad norm: 10.0 (vs base 2.0)
  - Learns text-to-motion from scratch before seeing editing tasks
- **Status**: Curriculum phase 1, resumed to Phase 2

#### 4. hymotion_m2m_v2_caption_local_phase2.py
- **Purpose**: Phase 2 - Mixed T2M (16%) + Completion (84%)
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py`
- **Size**: 4.4K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences vs Phase 1**:
  - `fk_consistency_weight=0.0` (disabled, KIMODO aux loss active)
  - **Critical**: Uses `PrepareM2Mv2Condition` with **v3 sampler**
  - `sampler_version='v3'` with custom k_weights:
    ```python
    k_weights=(0.16, 0.513, 0.233, 0.065, 0.029)
    # K=0 (pure T2M): 16% (raised from default 10%)
    # K=1..4: proportionally renormalized
    ```
  - `mask_aware_noise=True` (for completion tasks)
  - `max_epochs=10000` (long training)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_50/model.safetensors`
    (Phase 1 best checkpoint)
  - Null embedding patch: T2M pretrained
- **Data pipeline**: Mixed editing + T2M with v3 rank-K sampler
- **Status**: Curriculum phase 2, production training config

#### 5. hymotion_m2m_v2_caption_local_phase2b.py
- **Purpose**: Phase 2b - Component-mean loss reduction
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2b.py`
- **Size**: 4.1K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences vs Phase 2**:
  - **NEW**: `velocity_loss_reduction='component_mean'` (splits 198-dim into 4 groups)
    - Translation: 25% weight
    - Root rotation: 25% weight
    - Body rotation: 25% weight
    - Joint positions: 25% weight
  - `trans_dim_weight=1.0` (reduced from 5.0 to avoid overcorrection to ~55%)
  - Mask sampler: **identical to Phase 2** (v3 with same k_weights)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3320/model.safetensors`
    (Phase 2 best checkpoint)
  - Null embedding patch: T2M pretrained
- **Data pipeline**: Identical to Phase 2 (v3 sampler, same editing_prob=0.15)
- **Status**: Continued training from Phase 2, refining loss balance

---

### GROUP 3: Curriculum Learning Phases (Caption + Global)

#### 6. hymotion_m2m_v2_caption_global_phase1.py
- **Purpose**: Phase 1 - Pure T2M generation (global rotation)
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase1.py`
- **Size**: 3.1K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences** (vs caption_local_phase1):
  - `rotation_space='global'` (world frame)
  - `mean_std_dir='data/hymotion_m2m_data/_stats_198dim_global_rot'`
  - `LocalToGlobalRotation` in pipeline
  - `fk_consistency_weight=0.1` (same as local phase1)
  - `PrepareM2Mv2FullMask` (pure T2M, all mask=1)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_213/model.safetensors`
    (pretrained caption_global at epoch 213)
  - Null embedding patch: T2M pretrained
- **Training setup**:
  - Batch size: 20
  - Max epochs: 50
  - Mask_aware_noise: False
- **Status**: Curriculum phase 1, resumed to Phase 2

#### 7. hymotion_m2m_v2_caption_global_phase2.py
- **Purpose**: Phase 2 - Mixed T2M (16%) + Completion (84%), global rotation
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase2.py`
- **Size**: 3.6K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences** (vs caption_local_phase2):
  - `rotation_space='global'`
  - `mean_std_dir='data/hymotion_m2m_data/_stats_198dim_global_rot'`
  - `LocalToGlobalRotation` in pipeline
  - `fk_consistency_weight=0.0` (disabled)
  - **v3 sampler with identical k_weights** as local phase2
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_global_phase1/checkpoint-epoch_50/model.safetensors`
    (Phase 1 best)
  - Null embedding patch: T2M pretrained
- **Training setup**:
  - Batch size: 20
  - Max grad norm: 10.0
  - Mask_aware_noise: True
- **Status**: Curriculum phase 2, production training

---

### GROUP 4: Experiment Baselines (E2 & E4)

#### 8. hymotion_m2m_v2_smpl_caption_046b.py
- **Purpose**: E2 experiment - SMPL Root + Caption (keypoint supervision)
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py`
- **Size**: 4.5K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key config**:
  - `rotation_space='local'`
  - `mean_std_dir='data/hymotion_m2m_data/_stats_198dim'`
  - `uncondition_mode=False`, `cond_mask_prob=0.1`
  - **NEW**: `keypoints3d_weight=10.0` (enable keypoint supervision)
  - **NEW**: `velocity_loss_reduction='component_mean'` (per-component monitoring)
  - `text_encoder=dict()` (use default QWEN3+CLIP-L)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`
  - Null embedding patch: T2M pretrained
  - `load_scope='model'` (reset optimizer/scheduler due to loss change)
- **Data**:
  - Annotation: `train_hymotion_400h_hq_permo_motionfix_editing_20260514.json`
  - Requires captions (`LoadCompatibleCaption`, `allow_none=False`)
  - v3 sampler (default from base)
  - **NEW**: `LoadEditingSourceMotion` (real Neutral source for PerMo pairs)
- **Training setup**:
  - Batch size: 20
  - Num workers: 8 (increased from 4)
  - Persistent workers: True
- **Status**: Baseline experiment E2

#### 9. hymotion_m2m_v2_kimodo_caption_046b.py
- **Purpose**: E4 experiment - KIMODO Root + Caption (with ADMM smoothing)
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py`
- **Size**: 6.2K (largest caption config)
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences vs E2 (SMPL)**:
  - `mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root'` (KIMODO stats)
  - **NEW**: `SmplTransToKimodoRootOnline` transform with `admm_margin_m=0.06` (6cm margin on XZ)
  - Positions: ADMM smoothed pelvis translation
  - Root: continuous 6D rotation
  - Body: 6D rotations (21 joints)
  - Joint pos: FK-derived, relative to pelvis
  - keypoints3d_weight: 10.0 (same as E2)
  - velocity_loss_reduction: 'component_mean' (same as E2)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`
  - `exclude_bundle_keys=['mean', 'std']` (**critical**: prevent SMPL stats from overwriting KIMODO stats)
  - Null embedding patch: T2M pretrained
  - `load_scope='model'`
- **Data**:
  - Annotation: `train_hymotion_400h_hq_permo_motionfix_editing_20260514.json`
  - Requires captions
  - **NEW**: `SmplTransToKimodoRootOnline` in pipeline
  - **NEW**: `LoadEditingSourceMotion` with `kimodo_root_cfg=dict(admm_margin_m=0.06)`
    (same ADMM smoothing applied to source motion)
- **Training setup**: Same as E2 (batch_size=20, workers=8)
- **Status**: Baseline experiment E4

---

### GROUP 5: PerMo Data Augmentation Variants

#### 10. hymotion_m2m_v2_smpl_caption_permo_046b.py
- **Purpose**: E2+PerMo - SMPL Root + Caption + PerMo dataset
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py`
- **Size**: 4.0K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences vs E2**:
  - **Dataset change**: `anno_file='data/annotation/train_hymotion_400h_permo_caption_20260514.json'` (merged 400h + PerMo)
  - Sample count: 407,552 (400h only) → 414,094 (+6,542 PerMo, +1.6%)
  - Everything else identical to E2 (SMPL Root, caption_local, keypoint supervision)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`
  - Null embedding patch: T2M pretrained
- **Status**: Data augmentation variant, 1.6% more training samples

#### 11. hymotion_m2m_v2_kimodo_caption_permo_046b.py
- **Purpose**: E4+PerMo - KIMODO Root + Caption + PerMo dataset
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_046b.py`
- **Size**: 4.9K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key differences vs E4**:
  - **Dataset change**: `anno_file='data/annotation/train_hymotion_400h_permo_caption_20260514.json'`
  - KIMODO Root + ADMM smoothing (identical to E4)
  - `SmplTransToKimodoRootOnline` with `admm_margin_m=0.06`
  - **NOTE**: Pipeline does NOT include `LoadEditingSourceMotion` section
    (only dumps text vectors, no PerMo editing pairs in this variant)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`
  - `exclude_bundle_keys=['mean', 'std']`
  - Null embedding patch: T2M pretrained
- **Status**: Data augmentation variant with KIMODO Root

---

### GROUP 6: T2M Pretrained Variants

#### 12. hymotion_m2m_v2_046b_t2m_pretrained.py
- **Purpose**: T2M pretrained loading with selective freezing
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_pretrained.py`
- **Size**: 2.4K
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key config**:
  - `t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt'`
  - `t2m_freeze_strategy='encoders'` (default, recommended)
- **Freeze strategies**:
  - `'none'`: all modules trainable (ablation baseline)
  - `'encoders'`: freeze text encoders + timestep encoder
  - `'text_refiner'`: also freeze text_refiner
  - `'blocks'`: freeze transformer blocks only
  - `'full'`: freeze all except input/output layers
- **Reinitialization**:
  - Input encoder: 135→594 (VACE expansion)
  - Final layer: 135→198
- **Status**: Transfer learning framework config

#### 13. hymotion_m2m_v2_046b_t2m_no_freeze.py
- **Purpose**: T2M pretrained loading - NO freeze (ablation)
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_no_freeze.py`
- **Size**: 570B
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key config**:
  - `t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt'`
  - `t2m_freeze_strategy='none'` (all modules trainable)
- **Purpose**: Measure whether pretraining helps vs random init
- **Status**: Ablation study config

#### 14. hymotion_m2m_v2_046b_t2m_full_freeze.py
- **Purpose**: T2M pretrained loading - FULL freeze
- **Path**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_046b_t2m_full_freeze.py`
- **Size**: 687B
- **Base inherits from**: `_base_hymotion_m2m_v2_046b.py`
- **Key config**:
  - `t2m_pretrained_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt'`
  - `t2m_freeze_strategy='full'` (only input/output trainable)
- **Purpose**: Prevent catastrophic forgetting, adapt only VACE-specific components
- **Status**: Conservative transfer learning variant

---

### GROUP 7: SOAR Post-Training Configs

#### 15. hymotion_m2m_v2_caption_global_046b_soar.py
- **Purpose**: SOAR post-training on caption_global
- **Path**: `configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_caption_global_046b_soar.py`
- **Size**: 1.3K
- **Base inherits from**: `../hymotion_m2m_v2_caption_global_046b.py` (caption global config)
- **Key overrides**:
  - `trainer=HyMotionM2MSoarTrainer` (vs standard HyMotionM2MTrainer)
  - `soar_lambda=0.1` (SOAR loss weight)
  - `soar_num_aux=1` (auxiliary trajectories)
  - `soar_K=50` (planning horizon)
  - `soar_cfg_scale=1.0` (no CFG in rollout)
  - `soar_sigma_clamp=0.05` (noise clamping)
  - `lr=2e-5` (lower learning rate for post-training)
  - `batch_size=10` (halved from 20)
  - `max_iters=5000` (iteration-based, not epoch-based)
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_548`
  - Null embedding patch: T2M pretrained
- **Status**: Post-training variant for online trajectory optimization

#### 16. hymotion_m2m_v2_caption_local_046b_soar.py
- **Purpose**: SOAR post-training on caption_local
- **Path**: `configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_caption_local_046b_soar.py`
- **Size**: 1.8K
- **Base inherits from**: `../hymotion_m2m_v2_caption_local_046b.py` (caption local config)
- **Key overrides** (identical to soar_global):
  - `trainer=HyMotionM2MSoarTrainer`
  - `soar_lambda=0.1`
  - `soar_num_aux=1`
  - `soar_K=50`
  - `soar_cfg_scale=1.0` (note: v1 doesn't use CFG in rollout, TODO for ablation E6)
  - `soar_sigma_clamp=0.05`
  - `lr=2e-5`
  - `batch_size=10`
  - `max_iters=5000`
- **Checkpoints**:
  - Load from: `work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_498`
  - Null embedding patch: T2M pretrained
- **Status**: Post-training variant for online trajectory optimization

---

## TRAINING CHAIN SUMMARY

### Caption + Local Path (Primary)
```
1. _base_hymotion_m2m_v2_046b.py
   ↓
   Load: checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt

2. hymotion_m2m_v2_caption_local_046b.py
   (pretrained caption_local checkpoint-epoch_183)

3. hymotion_m2m_v2_caption_local_phase1.py (Pure T2M, 50 epochs)
   ↓
   Load: work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_183

4. hymotion_m2m_v2_caption_local_phase2.py (Mixed, 3370 epochs)
   ↓
   Load: checkpoint-epoch_50 (Phase 1 best)

5. hymotion_m2m_v2_caption_local_phase2b.py (Component-mean loss, continued)
   ↓
   Load: checkpoint-epoch_3320 (Phase 2 best)

6. hymotion_m2m_v2_caption_local_046b_soar.py (Optional: SOAR post-training)
   ↓
   Load: checkpoint-epoch_498 (SFT caption_local best)
```

### Caption + Global Path (Experiment)
```
1. _base_hymotion_m2m_v2_046b.py
   ↓
   Load: checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt

2. hymotion_m2m_v2_caption_global_046b.py
   (pretrained caption_global checkpoint-epoch_213)

3. hymotion_m2m_v2_caption_global_phase1.py (Pure T2M, 50 epochs)
   ↓
   Load: work_dirs/hymotion_m2m_v2_caption_global_046b/checkpoint-epoch_213

4. hymotion_m2m_v2_caption_global_phase2.py (Mixed, continues)
   ↓
   Load: checkpoint-epoch_50 (Phase 1 best)

5. hymotion_m2m_v2_caption_global_046b_soar.py (Optional: SOAR post-training)
   ↓
   Load: checkpoint-epoch_548 (SFT caption_global best)
```

### Experiment Variants (E2 & E4)
```
All start from Phase 2 checkpoint-epoch_3370:
- E2 (SMPL): hymotion_m2m_v2_smpl_caption_046b.py
- E4 (KIMODO): hymotion_m2m_v2_kimodo_caption_046b.py
- E2+PerMo: hymotion_m2m_v2_smpl_caption_permo_046b.py
- E4+PerMo: hymotion_m2m_v2_kimodo_caption_permo_046b.py
```

---

## KEY DIFFERENCES TABLE

| Config | Text Enabled | Rotation Space | Motion Repr | Mask Sampler | K=0 (T2M) | Load From | Notes |
|--------|---|---|---|---|---|---|---|
| caption_local_046b | ✓ | local | SMPL | v2 (tier2) | 16% | T2M pretrained | Standard baseline |
| caption_global_046b | ✓ | global | SMPL (global) | v2 (tier2) | 16% | T2M pretrained | Global rotation variant |
| caption_local_phase1 | ✓ | local | SMPL | full_mask | 100% | caption_local_046b@183 | Pure T2M curriculum |
| caption_local_phase2 | ✓ | local | SMPL | v3 | 16% | phase1@50 | Mixed T2M+completion |
| caption_local_phase2b | ✓ | local | SMPL | v3 | 16% | phase2@3320 | Component-mean loss |
| caption_global_phase1 | ✓ | global | SMPL (global) | full_mask | 100% | caption_global_046b@213 | Pure T2M curriculum |
| caption_global_phase2 | ✓ | global | SMPL (global) | v3 | 16% | phase1@50 | Mixed T2M+completion |
| smpl_caption_046b | ✓ | local | SMPL | v3 | 16% | caption_local_phase2@3370 | E2 baseline + keypoints |
| kimodo_caption_046b | ✓ | local | KIMODO (ADMM) | v3 | 16% | caption_local_phase2@3370 | E4 baseline + ADMM smoothing |
| smpl_caption_permo | ✓ | local | SMPL | v3 | 16% | caption_local_phase2@3370 | E2+PerMo dataset |
| kimodo_caption_permo | ✓ | local | KIMODO (ADMM) | v3 | 16% | caption_local_phase2@3370 | E4+PerMo dataset |
| t2m_pretrained | ✓ | local | SMPL | v3 | 16% | T2M pretrained | Transfer learning framework |
| t2m_no_freeze | ✓ | local | SMPL | v3 | 16% | T2M pretrained | T2M ablation (no freeze) |
| t2m_full_freeze | ✓ | local | SMPL | v3 | 16% | T2M pretrained | T2M ablation (full freeze) |
| caption_local_046b_soar | ✓ | local | SMPL | v3 | 16% | caption_local@498 | SOAR post-training |
| caption_global_046b_soar | ✓ | global | SMPL (global) | v3 | 16% | caption_global@548 | SOAR post-training |

---

## CRITICAL INSIGHTS

### 1. **Curriculum Learning Strategy**
- **Phase 1**: Pure T2M generation (mask=1 everywhere) learns text grounding first
- **Phase 2**: Introduce editing/completion tasks (84% of data) while maintaining high T2M ratio (16%)
- **Phase 2b**: Refine loss balance with component-mean reduction

### 2. **Mask Sampler Evolution**
- **v2 (tier2_prob)**: Used in early caption configs
  - tier2_prob=0.4 → 16% pure_gen (K=0 T2M) globally across all samples
  - Simple weighted task distribution
- **v3 (Rank-K)**: Used in Phase 2 and experiments
  - k_weights=(0.16, 0.513, 0.233, 0.065, 0.029) → K=0 raised to 16%
  - Per-dimension mask rank-based control (more sophisticated)

### 3. **Motion Representations**
- **SMPL Root** (198-dim): Standard SMPL with FK-derived joint positions
- **KIMODO Root** (198-dim): ADMM-smoothed pelvis translation + continuous root rotation
  - **Critical**: Must use `exclude_bundle_keys=['mean', 'std']` when loading SMPL checkpoint
  - ADMM margin: 0.06m (6cm) on XZ plane

### 4. **Checkpoint Loading**
- All caption configs load from T2M 1.0 pretrained initially
- Phase 2 configs resume from Phase 1 best checkpoint (epoch 50)
- **B2-ext fix**: Intermediate checkpoints have all-zero null embeddings
  - Solution: Patch with `null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt'`

### 5. **Memory & Batch Size**
- Base (uncond): batch_size=28, ~30GB on V100-32GB
- Caption configs: batch_size=20 (-28%)
  - Reason: Text tokens (128×4096) add ~6GB per batch
  - v3 sampler: higher overhead

### 6. **Text Conditioning**
- Uses pre-extracted Qwen3+CLIP embeddings
- `LoadPreExtractedTextEmbedding` (late binding: allow_none=True for some configs)
- CFG: `cond_mask_prob=0.1` (10% unconditional during training)

### 7. **Loss Configuration Differences**
```
Base (uncond):
  - fk_consistency_weight=0.0 (disabled, KIMODO aux active)
  - keypoints3d_weight=0.0
  - velocity_loss_reduction=default (element_mean)
  - trans_dim_weight=5.0

Phase 1 (caption):
  - fk_consistency_weight=0.1 (enabled during pure T2M)

Phase 2 (caption):
  - fk_consistency_weight=0.0 (disabled, KIMODO aux active)
  - velocity_loss_reduction=default

E2/E4 experiments:
  - keypoints3d_weight=10.0 (NEW: keypoint supervision)
  - velocity_loss_reduction='component_mean' (NEW: per-component monitoring)

Phase 2b:
  - velocity_loss_reduction='component_mean' (NEW)
  - trans_dim_weight=1.0 (reduced from 5.0)
```

### 8. **T2M Transfer Learning**
- **Framework config**: `hymotion_m2m_v2_046b_t2m_pretrained.py`
- Freeze strategies: 'encoders' (default), 'none', 'text_refiner', 'blocks', 'full'
- Ablations: 'no_freeze' vs 'full_freeze'

---

## DEPENDENCY GRAPH

```
_base_hymotion_m2m_v2_046b.py (foundation)
├── caption_local_046b.py → phase1@183 → phase1.py@50 → phase2.py@3370 → phase2b.py@3320
├── caption_global_046b.py → phase1@213 → phase1.py@50 → phase2.py
├── smpl_caption_046b.py@3370
├── kimodo_caption_046b.py@3370 + exclude_bundle_keys
├── smpl_caption_permo_046b.py@3370
├── kimodo_caption_permo_046b.py@3370 + exclude_bundle_keys
├── t2m_pretrained.py (framework)
├── t2m_no_freeze.py
├── t2m_full_freeze.py
└── SOAR variants
    ├── caption_local_046b_soar.py@498
    └── caption_global_046b_soar.py@548
```

