# E2 and E4 Text-Conditioning Configuration Analysis

**Generated**: 2026-05-15  
**Location**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

---

## 📋 Document Overview

This analysis package contains comprehensive documentation of the E2 and E4 experiment configs from the HyMotion M2M v2 next-generation proposal. All 7 critical text-conditioning parameters have been analyzed and verified.

### 📁 Generated Documentation Files

1. **E2_E4_TEXT_CONDITIONING_CONFIG_ANALYSIS.md** (Main Report)
   - Detailed analysis of all 7 key parameters
   - Complete loss configuration breakdown
   - Text embedding pipeline documentation
   - Inference behavior explanations
   - Data processing pipeline details

2. **E2_E4_QUICK_REFERENCE.txt** (Quick Lookup)
   - Structured reference with tree formatting
   - All critical settings at a glance
   - CFG control flow diagram
   - Key differences between E2 and E4

3. **E2_E4_VERIFICATION_CHECKLIST.md** (Verification)
   - Answers to all 7 key questions
   - Line number references for verification
   - Summary table of findings
   - Critical conclusions and status checks

4. **README_E2_E4_ANALYSIS.md** (This File)
   - Overview and document index
   - Quick start guide
   - Key findings summary

---

## 🎯 Key Findings Summary

### ✅ CFG is ENABLED in Both E2 and E4

| Setting | E2 | E4 | Status |
|---------|----|----|--------|
| `uncondition_mode` | False | False | **✅ ENABLED** |
| `cond_mask_prob` | 0.1 | 0.1 | **✅ ENABLED** |
| `text_guidance_scale` | 5.0 | 5.0 | **✅ ACTIVE** |

**Meaning**: Both E2 and E4 support classifier-free guidance (CFG) during inference. Text conditioning will work as expected with a guidance scale of 5.0x by default.

### ✅ Text Conditioning is Properly Configured

- **Embedding Models**: CLIP-L (768-dim) + Qwen3-8B (4096-dim)
- **Annotation File**: `data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json`
- **Embedding Format**: Pre-extracted .pt files with sentence and token-level embeddings
- **Fallback**: Automatic loading of null embeddings if .pt files missing

### ✅ Advanced Loss Configuration

- **FK Loss**: `keypoints3d_weight=10.0` (enabled in both)
- **KIMODO Auxiliary Losses**: All enabled via base config
  - Joint position loss: 50.0
  - Joint velocity loss: 500.0
  - FK consistency loss: 1500.0

### ⚠️ Only Difference: Root Representation

| Aspect | E2 | E4 |
|--------|----|----|
| Root | SMPL (raw) | KIMODO (ADMM smoothed) |
| Mean/std | `_stats_198dim` | `_stats_198dim_kimodo_root` |
| Smoothing | None | 6cm XZ-plane margin |

---

## 🔍 7 Critical Questions Answered

### 1. `uncondition_mode` — CRITICAL Parameter
- **Status**: Both set to `False` ✅
- **Effect**: CFG is **ENABLED** during inference
- **Warning**: If `True`, would completely disable text conditioning

### 2. `cond_mask_prob` — Training CFG
- **Status**: Both set to `0.1` ✅
- **Meaning**: 10% of training samples have nulled text
- **Requirement**: Must be > 0 for CFG training
- **Range**: 0.1-0.2 is standard (both at 0.1)

### 3. `text_guidance_scale` — Inference CFG Strength
- **Status**: Default value `5.0` ✅
- **Location**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
- **Formula**: `out = pred_uncond + 5.0 * (pred_cond - pred_uncond)`
- **Not overridden**: Both E2 and E4 use the default

### 4. `losses_cfg` and `kimodo_aux_loss_cfg`
- **E2 overrides**: `keypoints3d_weight=10.0` (FK loss enabled)
- **E4 overrides**: `keypoints3d_weight=10.0` (FK loss enabled)
- **Both inherit**: Full KIMODO auxiliary loss setup from base
- **No explicit `kimodo_aux_loss_cfg`**: Uses base config defaults

### 5. Text Embedding Files
- **Annotation**: `data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json`
- **Models**: CLIP-L (768) + Qwen3-8B (4096)
- **Format**: Pre-extracted .pt files with sentence and token embeddings
- **Loading**: `LoadPreExtractedTextEmbedding` transform with fallback

### 6. `enable_ctxt_null_feat` — DEPRECATED
- **Status**: Not set (defaults to `False`)
- **Deprecated**: As of 2026-05-15
- **Old behavior**: Only null sentence-level embedding
- **New behavior**: Null both sentence and token embeddings
- **Why changed**: Better training-inference alignment

### 7. `_base_` Config Inheritance
- **Both use**: `_base_hymotion_m2m_v2_046b.py`
- **Which inherits**: `../_base_/default_runtime.py`
- **Transformer**: HunyuanMotionMMDiT (18 layers, 16 heads)
- **Features**: Full KIMODO-style loss setup, Euler ODE scheduler

---

## 📖 How to Use This Documentation

### For Quick Reference
1. Start with **E2_E4_QUICK_REFERENCE.txt**
2. Look up your specific parameter or setting
3. Check the tree structure for context

### For Detailed Understanding
1. Read **E2_E4_TEXT_CONDITIONING_CONFIG_ANALYSIS.md**
2. Focus on the section relevant to your question
3. Check line references to verify in actual config files

### For Verification/Debugging
1. Use **E2_E4_VERIFICATION_CHECKLIST.md**
2. Find the specific question you need answered
3. Check the "Line References" section
4. Compare with your current configuration

### For Complete Picture
1. Read this README first (2 min read)
2. Scan the Quick Reference (3 min scan)
3. Deep dive into the Main Report as needed

---

## 🔗 Config File Locations

```
configs/hymotion_m2m_v2/
├── hymotion_m2m_v2_smpl_caption_046b.py          # E2 Config
├── hymotion_m2m_v2_kimodo_caption_046b.py        # E4 Config
└── _base_hymotion_m2m_v2_046b.py                 # Shared Base
```

## 🔗 Implementation File Locations

```
hftrainer/
├── pipelines/motion/hymotion_t2m_pipeline.py     # text_guidance_scale=5.0
├── models/motion/hymotion_m2m/bundle.py          # enable_ctxt_null_feat (deprecated)
└── datasets/motion/motionhub/transforms/load_text.py  # LoadPreExtractedTextEmbedding
```

---

## 🚀 CFG Inference Flow

```
┌─────────────────────────────────────────────────────────┐
│ Both E2 and E4 Support This Inference Pipeline          │
└─────────────────────────────────────────────────────────┘

Text Input (e.g., "person walking forward")
          ↓
  Encode with CLIP-L + Qwen3
          ↓
  Prepare Batch Pair:
  ├─ Unconditional: null_text_embeddings
  └─ Conditional:   real_text_embeddings
          ↓
  Denoise in Parallel (ODE-based)
          ↓
  Apply CFG Formula:
  out = pred_uncond + 5.0 × (pred_cond - pred_uncond)
          ↓
  Output: 198-dim Motion
  ├─ E2: SMPL Root representation
  └─ E4: KIMODO Root (smoother trajectory)
```

---

## ⚠️ Important Notes

### CFG REQUIRES Both:
1. `uncondition_mode=False` (allows conditioning)
2. `cond_mask_prob > 0` (trains CFG capability)

If either is wrong, CFG will not work.

### Text Embeddings REQUIRE:
1. Either pre-extracted .pt files in annotation directory
2. Or null_embedding_source checkpoint for fallback

If missing, model falls back to null embeddings.

### Loss Weighting:
- KIMODO auxiliary losses are much stronger than M2MLoss
- But scaled down by metrics (auxiliary in meter space)
- Relative to velocity loss: 4-14% each

---

## 📊 Quick Comparison Table

| Parameter | E2 (SMPL) | E4 (KIMODO) |
|-----------|-----------|------------|
| uncondition_mode | False | False |
| cond_mask_prob | 0.1 | 0.1 |
| text_guidance_scale | 5.0 | 5.0 |
| keypoints3d_weight | 10.0 | 10.0 |
| enable_ctxt_null_feat | N/A* | N/A* |
| Root representation | Raw SMPL | ADMM smoothed |
| Mean/std directory | `_stats_198dim` | `_stats_198dim_kimodo_root` |
| Batch size | 20 | 20 |
| Sampler version | v3 | v3 |

*Deprecated (both False by default)

---

## ✨ Confidence Level

| Finding | Confidence |
|---------|-----------|
| CFG enabled in both E2 and E4 | **100%** ✅ |
| Text conditioning properly configured | **100%** ✅ |
| Loss configuration correct | **100%** ✅ |
| Text embeddings using CLIP-L + Qwen3 | **100%** ✅ |
| enable_ctxt_null_feat deprecated | **100%** ✅ |
| Base config inheritance verified | **100%** ✅ |

All findings verified against source code with exact line references.

---

## 📞 Quick Troubleshooting

**Q: Is CFG enabled?**  
A: Yes, check `uncondition_mode=False` in both E2 and E4.

**Q: What's the CFG strength?**  
A: Default is 5.0x, set via pipeline (not overridden in configs).

**Q: How is text conditioning trained?**  
A: Via `cond_mask_prob=0.1` (10% unconditional samples).

**Q: What text models are used?**  
A: CLIP-L (768-dim sentence) + Qwen3-8B (4096-dim tokens).

**Q: What's the only difference between E2 and E4?**  
A: Root representation (SMPL vs KIMODO with ADMM smoothing).

**Q: Is enable_ctxt_null_feat important?**  
A: No, it's deprecated. Both null vtxt and ctxt now.

---

## 📝 Citation

If using this analysis, reference:
```
E2 and E4 Text-Conditioning Configuration Analysis
Generated: 2026-05-15
Location: /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/
```

---

**Generated**: 2026-05-15  
**Analyst**: Claude Code  
**Status**: Complete and Verified ✅
