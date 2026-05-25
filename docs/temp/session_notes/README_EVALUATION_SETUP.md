# PRISM Evaluation & Inference Setup - Complete Documentation

📍 **Created:** May 18, 2026
🎯 **Purpose:** Comprehensive guide to evaluation/inference scripts, dataset configs, and previous results

---

## 📋 What's Included

This setup includes **three comprehensive documentation files**:

### 1. **PRISM_EVALUATION_AND_INFERENCE_GUIDE.md** (14 KB) - MAIN REFERENCE
   - **8 major sections** covering all aspects
   - Detailed description of each evaluation script
   - Dataset configuration guide
   - Complete model registry
   - Quick start examples
   - Troubleshooting tips

### 2. **QUICK_EVAL_REFERENCE.txt** (8.7 KB) - QUICK LOOKUP
   - 10-section quick reference card
   - Copy-paste command examples
   - File locations at a glance
   - KAFS mode explanations
   - Common troubleshooting

### 3. **DIRECTORY_MAP.md** (12 KB) - FILE STRUCTURE
   - Complete directory tree
   - File location quick reference tables
   - Data flow diagrams
   - Symlink information
   - File size summary
   - Latest models summary

---

## 🚀 Quick Start (Copy-Paste Ready)

### Run PRISM Batch Evaluation on HumanML3D
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/

python scripts/eval/eval_prism_kafs_ablation.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-latest \
    --kafs-mode depth_driven \
    --anno-file data/annotation/test_hml3d.json \
    --output-dir work_dirs/prism_eval_hml3d \
    --num-inference-steps 20 \
    --max-samples 100
```

### Run M2M v2 Multi-Task Evaluation
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --tasks E2 E3 E4 E5 \
    --models caption_local caption_global \
    --max-samples 50 \
    --output-dir work_dirs/m2m_v2_eval_batch
```

### Single Inference with PRISM
```bash
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-latest \
    --prompt "a person walks forward" \
    --output output/test_motion.npz \
    --num-steps 20 \
    --device cuda:0
```

---

## 📂 Key Paths at a Glance

| What | Where |
|------|-------|
| **PRISM Batch Eval Script** | `scripts/eval/eval_prism_kafs_ablation.py` |
| **M2M v2 Batch Eval** | `scripts/eval/eval_m2m_v2_all_tasks.py` |
| **Unified Inference** | `tools/infer.py` |
| **HumanML3D Test** | `data/annotation/test_hml3d.json` |
| **MotionHub T2M Test** | `data/annotation/test_motionhub_t2m.json` |
| **Latest PRISM Model** | `work_dirs/prism_1b_tp2m_multiframe_kt_spectral/` |
| **Latest M2M v2 Model** | `work_dirs/hymotion_m2m_v2_smpl_caption_E2/` |
| **Config: PRISM** | `configs/prism/` (9 variants) |
| **Config: M2M v2** | `configs/hymotion_m2m_v2/` |

---

## 🎯 Finding What You Need

### I want to...

**Run batch inference on HumanML3D or MotionHub**
→ See: **QUICK_EVAL_REFERENCE.txt** Section 1 & 8
→ Script: `scripts/eval/eval_prism_kafs_ablation.py`

**Understand dataset configuration**
→ See: **PRISM_EVALUATION_AND_INFERENCE_GUIDE.md** Section 2
→ Files: `data/annotation/test_*.json` (symlinks)

**Know where previous results are stored**
→ See: **PRISM_EVALUATION_AND_INFERENCE_GUIDE.md** Section 3
→ Locations: `output/`, `work_dirs/`, `eval_results/`

**Get a complete directory structure**
→ See: **DIRECTORY_MAP.md** - Full tree with sizes

**Find the latest models**
→ See: **DIRECTORY_MAP.md** Section "Latest Models Summary"

**Understand PRISM KAFS modes**
→ See: **QUICK_EVAL_REFERENCE.txt** Section 9

**Troubleshoot common issues**
→ See: **QUICK_EVAL_REFERENCE.txt** Section 10
→ Full guide: **PRISM_EVALUATION_AND_INFERENCE_GUIDE.md** Section 8

**Get exact command examples**
→ See: **QUICK_EVAL_REFERENCE.txt** Section 8

---

## 🔬 Technical Highlights

### Evaluation Scripts Summary

| Script | Type | Size | Purpose |
|--------|------|------|---------|
| eval_prism_kafs_ablation.py | Batch Eval | 17 KB | ✓ **RECOMMENDED** - PRISM on H3D/MotionHub |
| eval_m2m_v2_all_tasks.py | Batch Eval | 203 KB | M2M v2 comprehensive multi-task eval |
| infer.py | Single/Batch | 14 KB | ✓ Unified entry point for all models |
| eval_m2m_v2_t2m.py | Task-Specific | 34 KB | M2M v2 T2M evaluation |
| eval_with_motionclip_evaluator.py | Metrics | 24 KB | MotionCLIP metric computation |

### Dataset Configuration

**Test Annotation Files:**
- `test_hml3d.json` - HumanML3D test set (primary)
- `test_motionhub_t2m.json` - MotionHub T2M test (primary)
- Plus 7 additional MotionHub variants (1p, 2p, m2d, pred, recon, s2g)

**Format:** JSON with entries:
```json
{
  "name": "sample_identifier",
  "caption": "text description",
  "motion_path": "path/to/motion.npz"
}
```

### Model Configurations

**PRISM:** 9 config variants in `configs/prism/`
- Latest: `prism_1b_tp2m_multiframe_kt_spectral.py`
- Multiframe recommended over single-frame

**M2M v2:** Multiple variants in `configs/hymotion_m2m_v2/`
- Root representations: SMPL, KIMODO
- Conditioning: unconditioned, caption
- Rotations: local, global
- Phases: Phase 0 (ablations), Phase 1 (pure T2M), Phase 2 (T2M + completion)

### Latest Models (May 18, 2026)

**PRISM:**
```
work_dirs/prism_1b_tp2m_multiframe_kt_spectral/
Size: 11.3 GB | Updated: May 18 07:11
```

**M2M v2 (Best Models):**
```
work_dirs/hymotion_m2m_v2_smpl_caption_E2/
Size: 531.5 GB | Updated: May 18 11:23 ✓ LATEST

work_dirs/hymotion_m2m_v2_kimodo_caption_E4/
Size: 487.2 GB | Updated: May 18 12:12
```

---

## 📊 Storage Summary

| Component | Size | Count |
|-----------|------|-------|
| Evaluation Scripts | ~950 KB | 30+ files |
| Configuration Files | ~40 KB | 50+ configs |
| Test Annotations | ~600+ MB | 16 files |
| **Model Checkpoints (work_dirs)** | **553 TB** | 296 directories |
| **Evaluation Results (output)** | **29 TB** | 100+ directories |
| Checkpoints (pre-trained) | 121.7 GB | Base models |
| **Total Repository** | **~630 TB** | |

---

## ✅ Verification Checklist

Before running evaluations, verify:

- [ ] Test annotation files exist: `data/annotation/test_hml3d.json`
- [ ] Model checkpoint exists: `work_dirs/prism_1b_tp2m_multiframe_kt_spectral/`
- [ ] Caption embedding cache: `data/eval/m2m_v2/caption_embeddings/cache.pt`
- [ ] Evaluation script is executable: `scripts/eval/eval_prism_kafs_ablation.py`
- [ ] Output directory is writable: `work_dirs/` or `output/`

If anything is missing, see troubleshooting section in the reference documents.

---

## 📖 Documentation Structure

```
hf_trainer/
├── PRISM_EVALUATION_AND_INFERENCE_GUIDE.md  [COMPREHENSIVE - Start here]
├── QUICK_EVAL_REFERENCE.txt                 [QUICK LOOKUP - Copy-paste commands]
├── DIRECTORY_MAP.md                         [FILE STRUCTURE - Find what you need]
└── README_EVALUATION_SETUP.md               [THIS FILE - Overview & quick start]
```

---

## 🎓 Learning Path

1. **New to this setup?**
   - Start with this README
   - Then read **QUICK_EVAL_REFERENCE.txt** (5 min read)
   - Try copy-pasting a command from Section 8

2. **Need specific information?**
   - Check **QUICK_EVAL_REFERENCE.txt** Sections 1-7 for specific topics
   - For details, go to **PRISM_EVALUATION_AND_INFERENCE_GUIDE.md**

3. **Lost in directory structure?**
   - Consult **DIRECTORY_MAP.md** for complete file structure

4. **Troubleshooting?**
   - Check **QUICK_EVAL_REFERENCE.txt** Section 10 first
   - Full troubleshooting in **PRISM_EVALUATION_AND_INFERENCE_GUIDE.md** Section 8

---

## 🆘 Common Issues

**❌ "Caption embedding cache not found"**
```bash
# Rebuild:
CUDA_VISIBLE_DEVICES=0 python3 scripts/caption/extract_eval_caption_embeddings.py --force
```

**❌ "Test annotation file not found"**
- Check: `data/annotation/test_hml3d.json` exists as symlink
- Symlink points to: `/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/data/annotation/`

**❌ "Checkpoint not found"**
- Verify: `work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-latest` exists
- List: `ls -la work_dirs/ | grep prism`

**❌ "Import errors when running scripts"**
- Add to path: `export PYTHONPATH=$PYTHONPATH:/path/to/hf_trainer`
- Or: Run from hf_trainer directory: `cd /apdcephfs/.../hf_trainer/`

See **QUICK_EVAL_REFERENCE.txt** Section 10 for more troubleshooting.

---

## 🔗 Related Resources

- Main PRISM guide: `PRISM_EVALUATION_AND_INFERENCE_GUIDE.md`
- Quick reference: `QUICK_EVAL_REFERENCE.txt`
- Directory structure: `DIRECTORY_MAP.md`
- Previous analysis: `PRISM_ANALYSIS_SUMMARY.txt`

---

## 📝 Notes

- All symlinks in `data/annotation/` point to versatilemotion repo
- Latest PRISM checkpoint: May 18 07:11
- Latest M2M v2 checkpoint: May 18 11:23
- Evaluation output uses date stamps: `eval_v2_e9_20260423_manifold/`
- 296 model directories in work_dirs (553 TB total)

---

**Last Updated:** May 18, 2026
**Status:** ✓ Complete and Verified

