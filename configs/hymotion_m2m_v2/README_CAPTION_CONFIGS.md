# HyMotion M2M v2 Caption/Text Training Configuration Guide

## Overview

This directory contains **16 caption/text-conditioned configs** for training HyMotion M2M v2 (Motion-to-Motion 0.46B model) with text guidance. This README provides a quick reference; see detailed docs below for in-depth analysis.

## Quick Start

### I just want to train caption-conditioned motion generation

**Start here**: `hymotion_m2m_v2_caption_local_046b.py`
```bash
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py 8
```

### I want curriculum learning (Phase 1 → Phase 2)

1. **Phase 1** (Pure T2M): `hymotion_m2m_v2_caption_local_phase1.py`
   ```bash
   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py 8
   ```
   
2. **Phase 2** (Mixed T2M + Completion): `hymotion_m2m_v2_caption_local_phase2.py`
   ```bash
   bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py 8 \
     --load-from work_dirs/hymotion_m2m_v2_caption_local_phase1/checkpoint-epoch_50/model.safetensors
   ```

### I want to try different motion representations

- **SMPL Root** (standard): Use `hymotion_m2m_v2_caption_local_046b.py` or `hymotion_m2m_v2_smpl_caption_046b.py`
- **KIMODO Root** (ADMM smoothed): Use `hymotion_m2m_v2_kimodo_caption_046b.py`

### I want to add PerMo dataset

- **SMPL + PerMo**: `hymotion_m2m_v2_smpl_caption_permo_046b.py`
- **KIMODO + PerMo**: `hymotion_m2m_v2_kimodo_caption_permo_046b.py`

### I want to use T2M transfer learning

- **Default (freeze encoders)**: `hymotion_m2m_v2_046b_t2m_pretrained.py`
- **Ablation (no freeze)**: `hymotion_m2m_v2_046b_t2m_no_freeze.py`
- **Ablation (full freeze)**: `hymotion_m2m_v2_046b_t2m_full_freeze.py`

---

## What's in Each Config

| Config | Purpose | Training Chain | Key Setting |
|--------|---------|---|---|
| **caption_local_046b** | Standard caption + local rotation | Main baseline | uncondition_mode=False |
| **caption_global_046b** | Caption + global rotation | Experiment variant | rotation_space='global' |
| **caption_local_phase1** | Curriculum Phase 1: Pure T2M | Primary chain step 3 | PrepareM2Mv2FullMask (100% T2M) |
| **caption_local_phase2** | Curriculum Phase 2: Mixed T2M+completion | Primary chain step 4 | v3 sampler, K=0 (16% T2M) |
| **caption_local_phase2b** | Phase 2b: Component-mean loss | Primary chain step 5 | velocity_loss_reduction='component_mean' |
| **caption_global_phase1** | Phase 1 with global rotation | Global baseline step 3 | LocalToGlobalRotation transform |
| **caption_global_phase2** | Phase 2 with global rotation | Global baseline step 4 | Same v3 sampler as local |
| **smpl_caption_046b** | E2 experiment: SMPL + keypoint supervision | Branches at phase2@3370 | keypoints3d_weight=10.0 |
| **kimodo_caption_046b** | E4 experiment: KIMODO Root + ADMM smoothing | Branches at phase2@3370 | SmplTransToKimodoRootOnline |
| **smpl_caption_permo_046b** | E2 + PerMo dataset (+1.6% samples) | Branches at phase2@3370 | Different annotation file |
| **kimodo_caption_permo_046b** | E4 + PerMo dataset | Branches at phase2@3370 | Different annotation file + KIMODO |
| **t2m_pretrained** | T2M transfer learning framework | Optional pre-training | t2m_freeze_strategy='encoders' |
| **t2m_no_freeze** | T2M ablation (no freezing) | Ablation | t2m_freeze_strategy='none' |
| **t2m_full_freeze** | T2M ablation (full freezing) | Ablation | t2m_freeze_strategy='full' |
| **caption_local_046b_soar** | SOAR post-training on caption_local | Optional step 6 | trainer=HyMotionM2MSoarTrainer |
| **caption_global_046b_soar** | SOAR post-training on caption_global | Optional step 5b | soar_lambda=0.1 |

---

## Training Chain Overview

```
                    Base (_base_hymotion_m2m_v2_046b.py)
                              ↓
         T2M Pretrained (checkpoints/HY-Motion-1.0/...)
                              ↓
        ┌──────────────────────┴──────────────────────┐
        ↓                                              ↓
   LOCAL CHAIN                                   GLOBAL CHAIN
   caption_local_046b@183                  caption_global_046b@213
        ↓                                              ↓
   phase1.py@50                                 phase1.py@50
   (Pure T2M, 100%)                           (Pure T2M, 100%)
        ↓                                              ↓
   phase2.py@3370                              phase2.py
   (Mixed: 16% T2M)                           (Mixed: 16% T2M)
        ↓                                              ↓
   phase2b.py@3320                        [SOAR@548 optional]
   (Component-mean loss)
        ↓
  [SOAR@498 optional]

   EXPERIMENT BRANCHES (all start at phase2@3370):
   ├── E2: smpl_caption_046b (keypoint supervision)
   ├── E4: kimodo_caption_046b (ADMM smoothing)
   ├── E2+PerMo: smpl_caption_permo_046b
   └── E4+PerMo: kimodo_caption_permo_046b
```

---

## Key Differences Between Configs

### By Rotation Space
- **local** (SMPL frame): `caption_local_*`, `smpl_caption_*`, `kimodo_caption_*`
- **global** (world frame): `caption_global_*`

### By Motion Representation
- **SMPL Root** (standard): Most configs
  - Stats: `data/hymotion_m2m_data/_stats_198dim`
  - Dim layout: [trans:3, rot6d:132, positions:63]
  
- **KIMODO Root** (ADMM smoothed): `kimodo_caption_*`
  - Stats: `data/hymotion_m2m_data/_stats_198dim_kimodo_root`
  - **Critical**: Use `exclude_bundle_keys=['mean', 'std']` when loading
  - ADMM margin: 0.06m (6cm on XZ plane)

### By Loss Configuration
- **Standard** (most configs): element_mean, fk_consistency=0.0
- **Phase 1**: fk_consistency=0.1 (enabled for pure T2M)
- **Phase 2b**: component_mean with trans_dim_weight=1.0
- **E2/E4**: keypoints3d_weight=10.0 + component_mean

### By Mask Sampler
- **v2 (tier2_prob)**: caption_local_046b, caption_global_046b
  - Simple task distribution: tier2_prob=0.4 → 16% pure T2M
  
- **full_mask**: caption_local_phase1, caption_global_phase1
  - Curriculum Phase 1: 100% pure T2M (mask=1 everywhere)
  
- **v3 (Rank-K)**: phase2, phase2b, E2/E4, SOAR
  - Advanced per-dimension mask ranking
  - k_weights=(0.16, 0.513, 0.233, 0.065, 0.029) → K=0 (16% T2M)

---

## Important Configuration Details

### Text Conditioning
- **Enabled in all caption configs**: `uncondition_mode=False`
- **CFG**: `cond_mask_prob=0.1` (10% unconditional during training)
- **Text encoder**: QWEN3 (4096-dim context) + CLIP-L (768-dim)
- **Embeddings**: Pre-extracted (not on-the-fly)

### Memory & Batch Size
- **Base (uncond)**: batch_size=28 (~30GB V100)
- **Caption configs**: batch_size=20 (-28% due to text tokens)
- **SOAR post-training**: batch_size=10 (additional overhead)

### Checkpoint Loading
All configs follow this pattern:
1. Load base config
2. Inherit T2M 1.0 pretrained weights from base
3. Override `load_from` to resume from previous phase/experiment checkpoint
4. **B2-ext fix**: Patch null embeddings with `null_embedding_source`

**Critical for KIMODO**: Use `exclude_bundle_keys=['mean', 'std']` to prevent SMPL stats from overwriting KIMODO stats.

---

## Common Troubleshooting

### "null_embeddings are all zeros"
- **Problem**: Loading from intermediate checkpoint resets null embeddings
- **Solution**: Use `null_embedding_source` in config (already done in all phase configs)

### "KIMODO Root loss becomes NaN"
- **Problem**: Loaded SMPL stats overwrite KIMODO stats
- **Solution**: Use `exclude_bundle_keys=['mean', 'std']` in load_from dict

### "Out of memory with batch_size=20"
- **Problem**: Text tokens add ~6GB per batch
- **Solution**: Reduce to batch_size=10, or use fewer workers

### "Phase 2 doesn't resume correctly"
- **Problem**: Config mismatch between Phase 1 and Phase 2
- **Solution**: Phase 1 uses full_mask, Phase 2 uses v3 sampler (incompatible sampling)
- **Resolution**: Handled automatically; start Phase 2 with checkpoint-epoch_50 from Phase 1

---

## Detailed Documentation

For complete analysis of all configs, checkpoint chains, and advanced parameters, see:

- **CAPTION_CONFIGS_ANALYSIS.md** (~530 lines)
  - Detailed breakdown of each config
  - Full checkpoint inheritance chains
  - Loss configuration matrix
  - Architecture overview
  
- **CAPTION_CONFIGS_QUICK_REF.md** (~300 lines)
  - Quick reference tables
  - Parameter comparison
  - Checkpoint paths
  - Common pitfalls & solutions

---

## File Structure

```
configs/hymotion_m2m_v2/
├── _base_hymotion_m2m_v2_046b.py          ← Foundation config
├── hymotion_m2m_v2_caption_local_046b.py  ← Main baseline
├── hymotion_m2m_v2_caption_global_046b.py ← Global variant
├── hymotion_m2m_v2_caption_local_phase1.py
├── hymotion_m2m_v2_caption_local_phase2.py
├── hymotion_m2m_v2_caption_local_phase2b.py
├── hymotion_m2m_v2_caption_global_phase1.py
├── hymotion_m2m_v2_caption_global_phase2.py
├── hymotion_m2m_v2_smpl_caption_046b.py   ← E2 baseline
├── hymotion_m2m_v2_kimodo_caption_046b.py ← E4 baseline
├── hymotion_m2m_v2_smpl_caption_permo_046b.py
├── hymotion_m2m_v2_kimodo_caption_permo_046b.py
├── hymotion_m2m_v2_046b_t2m_pretrained.py ← T2M transfer learning
├── hymotion_m2m_v2_046b_t2m_no_freeze.py
├── hymotion_m2m_v2_046b_t2m_full_freeze.py
├── soar/
│   ├── hymotion_m2m_v2_caption_local_046b_soar.py
│   └── hymotion_m2m_v2_caption_global_046b_soar.py
├── README_CAPTION_CONFIGS.md              ← This file
├── CAPTION_CONFIGS_ANALYSIS.md            ← Detailed analysis
└── CAPTION_CONFIGS_QUICK_REF.md           ← Quick reference
```

---

## Citation & Contact

For issues or questions about these configs, refer to:
- Detailed docs: `CAPTION_CONFIGS_ANALYSIS.md`
- Quick ref: `CAPTION_CONFIGS_QUICK_REF.md`

---

**Last updated**: 2026-05-17
**Total caption configs found**: 16
**Status**: All configs documented and analyzed
