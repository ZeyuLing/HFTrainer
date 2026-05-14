# Phase 0 Training Quick Start Guide

**Date:** 2026-05-14  
**Status:** ✅ Ready to Launch  
**Implementation:** Complete (commit cb80966, d1cef52)

## What is Phase 0?

Phase 0 is a comprehensive ablation study of HyMotion M2M v2 with **four controlled variants**:

1. **E1** (`smpl_uncond_E1`): SMPL Root + No Caption = unconditioned baseline
2. **E2** (`smpl_caption_E2`): SMPL Root + Caption = caption effect on SMPL
3. **E3** (`kimodo_uncond_E3`): KIMODO Root + No Caption = ADMM smoothing effect
4. **E4** (`kimodo_caption_E4`): KIMODO Root + Caption = combined effect (best)

## Quick Start: Train E2 (SMPL + Caption)

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Local training (8 GPUs)
bash tools/dist_train.sh \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py \
    8 --auto-resume

# Or Taiji cluster (8 nodes, 64 GPUs)
python3 tools/taiji_submit.py \
    phase0_e2_smpl_caption \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py \
    --host_num 8
```

## Quick Start: Train E4 (KIMODO + Caption)

```bash
# Local training (8 GPUs)
bash tools/dist_train.sh \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py \
    8 --auto-resume

# Or Taiji cluster (8 nodes, 64 GPUs)
python3 tools/taiji_submit.py \
    phase0_e4_kimodo_caption \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py \
    --host_num 8
```

## What's New in This Phase?

### Data Integration (474K entries)
- **Original**: 407.5K entries (academic, retarget, taobao, game)
- **PerMo**: ~12K entries (pre-computed 135-dim, no augmentation)
- **MotionFix**: ~54K entries (editing pairs with synthetic corruption)
- **Total**: ~474K entries (16% increase)

### Code Changes

#### 1. PerMo Support (LoadSmplx55)
Fast path for pre-computed 135-dim motion:
```python
# File: hftrainer/datasets/motion/motionhub/transforms/load_smplx.py
# New method: _load_precomputed_135()
# Auto-detection: if "motion_135" in data and "poses" not in data
# No augmentation applied (pre-computed data used as-is)
```

#### 2. Caption Configs (E2 & E4)
```python
# E2: SMPL Root + Caption
configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py
  • Uses merged annotation: train_hymotion_400h_hq_permo_motionfix_20260514.json
  • Text embedding loading: LoadPreExtractedTextEmbedding
  • Rotation: local space
  • Stats: _stats_198dim

# E4: KIMODO Root + Caption
configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py
  • Same annotation file as E2
  • ADMM smoothing: 6cm XZ margin (online during training)
  • Rotation: local space
  • Stats: _stats_198dim_kimodo_root (different from SMPL!)
```

#### 3. Evaluation Script (eval_m2m_v2_all_tasks.py)
All four models now registered:
```bash
# Run all E1-E4 variants
python3 scripts/eval/eval_m2m_v2_all_tasks.py \
    --model-names smpl_uncond_E1,smpl_caption_E2,kimodo_uncond_E3,kimodo_caption_E4 \
    --run-caption-nonaware \
    --use-rewritten \
    --save-npz
```

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 20 | Reduced from base 28 (higher memory for captions) |
| Workers | 8 | Persistent workers for efficiency |
| CFG prob | 10% | Classifier-free guidance during training |
| Keypoint loss | 10.0 | Enabled for E2/E4 variants |
| Epochs | 3370+ | Resume from caption_local_phase2 checkpoint |
| Annotation | 474K entries | Merged with PerMo + MotionFix |

## Key Design Decisions

### 1. PerMo Integration (No Augmentation)
Pre-computed data is used as-is without yaw/XZ augmentation because:
- PerMo already includes diverse poses
- Pre-computed 135-dim is already optimized
- Augmentation would degrade pre-computed features
- Backward compatible with raw SMPL loading

### 2. MotionFix Handling (Synthetic Corruption)
Source motion generated on-demand during training:
- Reloads same NPZ file
- Applies 1-2 random corruptors
- 15% of batches enter editing mode
- No separate *_source.npz files needed
- Flexible corruptor selection

### 3. Caption Conditioning (CFG)
10% unconditional during training enables:
- Classifier-free guidance at test time
- Better alignment between cond/uncond paths
- Proven technique from DDPM literature
- Compatible with existing M2M v2 architecture

### 4. KIMODO Root (E4 Only)
ADMM smoothing with specific constraints:
- 6cm margin on XZ plane (horizontal)
- Y-axis unchanged (vertical untouched)
- Different statistics: _stats_198dim_kimodo_root
- Helps embodied/robot tasks with smoother trajectories

## File Structure

```
hf_trainer/
├── configs/hymotion_m2m_v2/
│   ├── hymotion_m2m_v2_smpl_caption_046b.py      ← E2 config
│   ├── hymotion_m2m_v2_kimodo_caption_046b.py    ← E4 config
│   ├── hymotion_m2m_v2_smpl_uncond_046b.py       ← E1 config
│   └── hymotion_m2m_v2_kimodo_uncond_046b.py     ← E3 config
├── hftrainer/datasets/motion/motionhub/
│   └── transforms/load_smplx.py                  ← Fast path for PerMo
├── data/annotation/
│   └── train_hymotion_400h_hq_permo_motionfix_20260514.json  ← 474K entries
├── data/hymotion_m2m_data/
│   ├── _stats_198dim/                             ← SMPL stats (E1/E2)
│   └── _stats_198dim_kimodo_root/                 ← KIMODO stats (E3/E4)
└── scripts/eval/
    └── eval_m2m_v2_all_tasks.py                  ← All models registered
```

## Expected Results

### Training Dynamics
- **E1 baseline**: Fast convergence, stable but limited quality
- **E2 vs E1**: Caption improves text-to-motion quality, slight slowdown
- **E3 vs E1**: ADMM smoothing reduces jitter, foot skating better
- **E4 vs others**: Combined benefits (smoothing + caption)

### Evaluation Metrics
Expected to be evaluated on:
- Text-to-motion (T2M) tasks
- Motion editing (M2M) tasks
- Motion completion tasks
- Cross-domain tasks (embodied, gaming, etc.)

### Ablation Analysis
Phase 0 is designed to answer:
1. **Caption effect**: E1 vs E2 (same SMPL, different conditioning)
2. **Smoothing effect**: E1 vs E3 (same uncond, different representation)
3. **Combined effect**: E2 vs E4 (full comparison)
4. **Best model**: E4 should outperform others on most tasks

## Troubleshooting

### OOM (Out of Memory)
```bash
# Reduce batch size in config (default: 20)
train_dataloader = dict(
    batch_size=16,  # Try 16 or 12
    # ...
)
```

### Slow Data Loading
```bash
# Already optimized with:
# - 8 persistent workers
# - Pre-extracted text embeddings
# - Fast path for PerMo (no augmentation)
# Should not be a bottleneck
```

### Text Embedding Errors
```bash
# Ensure pre-extracted embeddings exist:
ls data/eval/m2m_v2/caption_embeddings/
# Should contain cache.pt with QWEN3+CLIP embeddings
```

### KIMODO Root Stats Error (E4 only)
```bash
# Ensure stats directory exists:
ls data/hymotion_m2m_data/_stats_198dim_kimodo_root/
# Config uses exclude_bundle_keys=['mean', 'std'] to prevent overwrite
```

## Monitoring Training

```bash
# Check tensorboard
tensorboard --logdir work_dirs/hymotion_m2m_v2_smpl_caption_E2/

# Or wandb (if configured)
# Check dashboard for:
# - Caption loss curves
# - Keypoint supervision loss
# - Velocity loss (decomposed by component)
# - Text embedding quality metrics
```

## Next Steps After Training

1. **Checkpoint Evaluation**
   ```bash
   python3 scripts/eval/eval_m2m_v2_all_tasks.py \
       --model-names smpl_caption_E2,kimodo_caption_E4 \
       --save-npz --use-rewritten
   ```

2. **Comparison Analysis**
   - E1 vs E2: caption effect
   - E3 vs E4: caption + smoothing
   - E2 vs E4: smoothing effect

3. **Error Analysis**
   - Identify failure modes
   - Compare metrics across tasks
   - Determine best model for deployment

## Backward Compatibility

✅ **Fully backward compatible:**
- Existing configs (uncond_local, uncond_global, caption_local) still work
- Pre-computed 135-dim detection doesn't break raw SMPL loading
- Eval script safely skips incompatible tasks
- No API changes to dataset classes

## Documentation

For more details, see:
- `SESSION_IMPLEMENTATION_SUMMARY.md` — Complete implementation overview
- `CAPTION_CONFIG_ANALYSIS_REPORT.md` — Detailed config analysis
- `CONFIG_COMPARISON.md` — Side-by-side parameter comparison
- `ANALYSIS_COMPLETE.md` — Integration checklist

---

**Status:** ✅ Ready for Phase 0 Training Launch  
**Implementation:** Complete and Validated  
**Recommendation:** Start with E2 (simpler), then E4 (best expected results)

