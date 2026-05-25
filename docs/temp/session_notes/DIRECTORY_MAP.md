# Complete Directory Map for PRISM Evaluation and Inference

**Last Updated:** May 18, 2026

---

## Directory Structure Overview

```
hf_trainer/ (Base: /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/)
├── scripts/
│   ├── eval/                           [Evaluation Scripts - 950 KB]
│   │   ├── eval_prism_kafs_ablation.py        (MAIN - batch eval for PRISM)
│   │   ├── eval_m2m_v2_all_tasks.py           (MAIN - M2M v2 comprehensive eval)
│   │   ├── eval_m2m_v2_t2m.py
│   │   ├── eval_with_motionclip_evaluator.py
│   │   ├── eval_momask_native_h3d263.py
│   │   ├── momask_infer_h3d_test.py
│   │   ├── compute_kafs_metrics.py
│   │   ├── run_kafs_ablation.sh               (Launcher script)
│   │   ├── run_m2m_v2_eval_latest.sh          (M2M launcher)
│   │   └── ... (30+ other eval scripts)
│   │
│   ├── inference/                      [Inference Scripts - 21 KB]
│   │   └── batch_infer_vermo.py               (VerMo batch inference)
│   │
│   └── ... (caption/, data/, debug/, embodied/, etc.)
│
├── tools/
│   ├── infer.py                        [MAIN - Unified inference entry point]
│   ├── train.py
│   └── ... (6 training/utility scripts)
│
├── configs/
│   ├── prism/                          [PRISM Configurations - 37 KB]
│   │   ├── prism_1b_tp2m_1frame.py     (Single frame baseline)
│   │   ├── prism_1b_tp2m_1frame_kt_dfs.py
│   │   ├── prism_1b_tp2m_1frame_kt_spectral.py
│   │   ├── prism_1b_tp2m_multiframe.py (Multi-frame recommended)
│   │   ├── prism_1b_tp2m_multiframe_kt_dfs.py
│   │   ├── prism_1b_tp2m_multiframe_kt_spectral.py (Latest - May 17)
│   │   ├── prism_mcm_motionhub.py
│   │   ├── prism_mcm_motionhub_16v100.py
│   │   ├── prism_mcm_motionhub_64v100.py
│   │   └── prism_debug_loss_split.py
│   │
│   ├── hymotion_m2m_v2/               [M2M v2 Configurations]
│   │   ├── hymotion_m2m_v2_uncond_local_046b.py
│   │   ├── hymotion_m2m_v2_uncond_global_046b.py
│   │   ├── hymotion_m2m_v2_caption_local_046b.py
│   │   ├── hymotion_m2m_v2_caption_global_046b.py
│   │   ├── hymotion_m2m_v2_*_phase1.py (Phase 1 variants)
│   │   ├── hymotion_m2m_v2_*_phase2.py (Phase 2 variants)
│   │   ├── hymotion_m2m_v2_kimodo_caption_046b.py
│   │   ├── hymotion_m2m_v2_smpl_caption_046b.py
│   │   ├── hymotion_m2m_v2_kimodo_uncond_046b.py
│   │   └── hymotion_m2m_v2_smpl_uncond_046b.py
│   │
│   ├── hymotion_t2m/                 [T2M Configurations]
│   ├── hymotion_m2m/                 [M2M v1 Configurations]
│   ├── vermo/                        [VerMo Multi-task Configs]
│   └── ... (other model configs)
│
├── data/
│   ├── annotation/                    [Test/Train Annotations - 600+ MB]
│   │   ├── test_hml3d.json            (symlink - HumanML3D test)
│   │   ├── test_hml3d_rewritten.json  (symlink - rewritten captions)
│   │   ├── test_motionhub_t2m.json    (symlink - MotionHub T2M test)
│   │   ├── test_motionhub_t2m_rewritten.json
│   │   ├── test_motionhub_1p.json     (1-person)
│   │   ├── test_motionhub_2p.json     (2-person)
│   │   ├── test_motionhub_m2d.json    (motion-to-dance)
│   │   ├── test_motionhub_pred.json   (prediction)
│   │   ├── test_motionhub_recon.json  (reconstruction)
│   │   ├── test_motionhub_s2g.json    (speech-to-gesture)
│   │   ├── train_hymotion_400h_hq_permo_motionfix_20260514.json (212.6 MB)
│   │   └── train_hymotion_400h_hq_permo_motionfix_editing_20260514.json (216.7 MB)
│   │   [Symlink source: /apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/data/annotation/]
│   │
│   ├── eval/
│   │   └── m2m_v2/
│   │       └── caption_embeddings/
│   │           └── cache.pt            (Pre-extracted caption embeddings for eval)
│   │
│   ├── motionhub/                     (Motion data - symlink or local)
│   └── ... (other data directories)
│
├── work_dirs/                          [Model Checkpoints & Training - 553 TB, 296 dirs]
│   ├── hymotion_m2m_v2_smpl_caption_E2/ (531.5 GB - LATEST, May 18 11:23)
│   ├── hymotion_m2m_v2_kimodo_caption_E4/ (487.2 GB - May 18 12:12)
│   ├── hymotion_m2m_v2_kimodo_uncond_E3/ (265.7 GB - May 18 09:54)
│   ├── hymotion_m2m_v2_smpl_uncond_E1/ (553.6 GB - May 18 11:09)
│   │
│   ├── prism_1b_tp2m_multiframe_kt_spectral/ (11.3 GB - May 18 07:11) [LATEST PRISM]
│   ├── prism_kafs_ablation/ (102.2 GB - May 15 15:05)
│   │
│   ├── m2m_v2_eval_four_new_missing_all_20260514_1306_machine1/ (1.4 TB - May 14 15:32)
│   ├── m2m_v2_eval_four_new_missing_all_20260514_1306_machine2/ (1.4 TB - May 14 14:18)
│   │
│   └── ... (290+ other model directories)
│
├── output/                             [Evaluation Results - 29 TB]
│   ├── embodied_t2m_v4/ (568.7 GB - May 14 14:54)
│   ├── embodied_t2m_v3/ (142.1 GB - May 13 16:03)
│   ├── embodied_t2m_v5/ (142.1 GB - May 13 03:12)
│   ├── eval_e14_rerun_20260509/ (98.3 GB - May 9 11:56)
│   ├── embodied_comparison/ (142.1 GB - May 12 20:06)
│   └── ... (100+ other output directories)
│
├── eval_results/                       [Aggregated Evaluation Results]
│   ├── m2m/ (2.65 TB)
│   └── m2m_smoke/ (2.7 GB)
│
├── checkpoints/                        [Pre-trained Checkpoints - 121.7 GB]
│
└── PRISM_EVALUATION_AND_INFERENCE_GUIDE.md [THIS GUIDE]
```

---

## Key File Locations Quick Reference

### 1. Batch Inference Scripts

| Script | Location | Size | Purpose |
|--------|----------|------|---------|
| **eval_prism_kafs_ablation.py** | `scripts/eval/` | 17 KB | Primary PRISM batch eval (HumanML3D + MotionHub) |
| **eval_m2m_v2_all_tasks.py** | `scripts/eval/` | 203 KB | M2M v2 multi-task comprehensive eval |
| **infer.py** | `tools/` | 14 KB | Unified inference entry point (PRISM, T2M, M2M, VerMo) |
| **batch_infer_vermo.py** | `scripts/inference/` | 21 KB | VerMo batch inference |

### 2. Configuration Files

| Config | Location | Use Case |
|--------|----------|----------|
| `prism_1b_tp2m_1frame.py` | `configs/prism/` | Single-frame PRISM baseline |
| `prism_1b_tp2m_multiframe.py` | `configs/prism/` | Multi-frame PRISM (recommended) |
| `prism_1b_tp2m_multiframe_kt_spectral.py` | `configs/prism/` | KT Spectral PRISM (latest May 17) |
| `hymotion_m2m_v2_*.py` | `configs/hymotion_m2m_v2/` | M2M v2 variants |

### 3. Test Annotation Files

| Annotation | Location | Type | Count |
|-----------|----------|------|-------|
| `test_hml3d.json` | `data/annotation/` | symlink | HumanML3D test set |
| `test_motionhub_t2m.json` | `data/annotation/` | symlink | MotionHub T2M test set (primary) |
| `test_motionhub_1p.json` | `data/annotation/` | symlink | MotionHub 1-person |
| `test_motionhub_2p.json` | `data/annotation/` | symlink | MotionHub 2-person |

### 4. Model Checkpoints

| Model | Location | Size | Last Updated |
|-------|----------|------|---------------|
| **prism_1b_tp2m_multiframe_kt_spectral** | `work_dirs/` | 11.3 GB | May 18 07:11 |
| **hymotion_m2m_v2_smpl_caption_E2** | `work_dirs/` | 531.5 GB | May 18 11:23 ✓ LATEST |
| **hymotion_m2m_v2_kimodo_caption_E4** | `work_dirs/` | 487.2 GB | May 18 12:12 |
| **hymotion_m2m_v2_kimodo_uncond_E3** | `work_dirs/` | 265.7 GB | May 18 09:54 |
| **hymotion_m2m_v2_smpl_uncond_E1** | `work_dirs/` | 553.6 GB | May 18 11:09 |

### 5. Evaluation Results

| Output | Location | Size | Last Updated |
|--------|----------|------|---------------|
| embodied_t2m_v4 | `output/` | 568.7 GB | May 14 14:54 |
| m2m_v2_eval_four_new_missing_all | `work_dirs/` | 1.4 TB | May 14 15:32 |
| m2m results | `eval_results/m2m/` | 2.65 TB | Apr 12 23:30 |

### 6. Supporting Data

| Resource | Location | Purpose |
|----------|----------|---------|
| Caption embedding cache | `data/eval/m2m_v2/caption_embeddings/cache.pt` | Pre-computed text embeddings |
| Motion data | `data/motionhub/` | Motion sequences |
| Test annotations | `data/annotation/` | JSON with test samples + captions |

---

## Data Flow Paths

### For PRISM Batch Evaluation on HumanML3D:

```
data/annotation/test_hml3d.json
    ↓ (load samples and captions)
configs/prism/prism_1b_tp2m_multiframe.py (config)
    ↓ + work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-latest (weights)
scripts/eval/eval_prism_kafs_ablation.py
    ↓ (batch inference)
work_dirs/prism_eval_output/
    ├── {sample_1}.npz
    ├── {sample_2}.npz
    └── ...
```

### For M2M v2 Multi-task Evaluation:

```
data/annotation/test_motionhub_*.json (multiple test sets)
    ↓ (load task definitions and captions)
data/eval/m2m_v2/caption_embeddings/cache.pt (pre-computed embeddings)
    ↓ (avoid re-encoding)
configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py
    ↓ + work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-latest
scripts/eval/eval_m2m_v2_all_tasks.py
    ↓ (multi-task inference)
work_dirs/m2m_v2_eval_output/
    ├── E2/{sample_1}.npz, {sample_2}.npz, ...
    ├── E3/{sample_1}.npz, {sample_2}.npz, ...
    └── ...
```

---

## Important Symlinks

Most test annotation files are symlinks to the versatilemotion repo:

```bash
# Check symlink targets:
ls -la data/annotation/test_*.json

# Should show:
test_hml3d.json → /apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/data/annotation/test_hml3d.json
test_motionhub_t2m.json → /apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/data/annotation/test_motionhub_t2m.json
# ... etc
```

---

## File Size Summary

| Component | Size | Count |
|-----------|------|-------|
| **Evaluation Scripts** | ~950 KB | 30+ files |
| **Configuration Files** | ~40 KB | 50+ configs |
| **Test Annotations** | ~600+ MB | 16 files |
| **Model Checkpoints (work_dirs)** | **553 TB** | 296 directories |
| **Evaluation Results (output)** | **29 TB** | 100+ directories |
| **Model Weights (checkpoints)** | 121.7 GB | Pre-trained models |
| **Total Repository Size** | **~630 TB** | |

---

## Latest Models Summary (as of May 18, 2026)

### PRISM
- ✓ Latest: `prism_1b_tp2m_multiframe_kt_spectral/` (11.3 GB)
- Config: `configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py`
- Updated: May 18 07:11

### HyMotion M2M v2
- ✓ Latest: `hymotion_m2m_v2_smpl_caption_E2/` (531.5 GB)
- Config: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py`
- Updated: May 18 11:23
- Also ready: `hymotion_m2m_v2_kimodo_caption_E4/` (487.2 GB, May 18 12:12)

### Test Sets
- ✓ Primary: `data/annotation/test_hml3d.json` (HumanML3D)
- ✓ Primary: `data/annotation/test_motionhub_t2m.json` (MotionHub)
- Also available: 1p, 2p, m2d, recon, s2g variants

---

## Caption Embedding Cache

**Location:** `data/eval/m2m_v2/caption_embeddings/cache.pt`

**Why It Exists:**
- Caption-conditioned models trained with `LoadPreExtractedTextEmbedding` don't include a runtime text encoder
- Cache stores pre-computed embeddings to avoid encoding during eval
- Significantly speeds up evaluation

**If Missing:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/caption/extract_eval_caption_embeddings.py --force
```

---

## Next Steps

1. **To run PRISM batch eval:**
   - See `QUICK_EVAL_REFERENCE.txt` Section 8

2. **To run M2M v2 comprehensive eval:**
   - See `QUICK_EVAL_REFERENCE.txt` Section 8

3. **For detailed configuration info:**
   - See `PRISM_EVALUATION_AND_INFERENCE_GUIDE.md`

4. **For troubleshooting:**
   - See `QUICK_EVAL_REFERENCE.txt` Section 10

