# PRISM Evaluation, Inference, and Dataset Configuration Guide

**Base Path:** `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

---

## 1. EVALUATION/INFERENCE SCRIPTS

### 1.1 Batch Inference Script for PRISM KAFS Ablation

**Location:** `scripts/eval/eval_prism_kafs_ablation.py`

**Purpose:** Batch evaluation script for PRISM model generation under different KAFS (Kinematic-Adaptive Flow Scheduling) modes. Generates motions from PRISM and saves them as per-sample NPZ files for downstream metric computation.

**Key Features:**
- Supports multiple KAFS modes: `none`, `depth_driven`, `uniform`, `random`
- Reads test samples from annotation JSON files
- Generates motions batch-wise
- Saves outputs as per-sample NPZ files
- Supports caption loading (hierarchical or HYMotion format)

**Usage Example:**
```bash
python scripts/eval/eval_prism_kafs_ablation.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --kafs-mode depth_driven \
    --anno-file data/annotation/test_hml3d.json \
    --data-dir data/motionhub \
    --output-dir work_dirs/prism_kafs_ablation \
    --num-inference-steps 50 \
    --max-samples 100
```

**Key Functions:**
- `load_test_samples()` - Loads test samples from annotation files
- `_load_caption()` - Loads captions in hierarchical or HYMotion format
- `generate_motions_batch()` - Batch generation with KAFS mode support
- `save_per_sample_npz()` - Saves generated motions as NPZ

---

### 1.2 Multi-Task M2M v2 Evaluation Script

**Location:** `scripts/eval/eval_m2m_v2_all_tasks.py` (203 KB, executable)

**Purpose:** Comprehensive evaluation of HyMotion M2M v2 across multiple tasks (E1-E15). Evaluates 4 model variants:
- `uncond_local`: No text, local rotation
- `uncond_global`: No text, global rotation
- `caption_local`: Text-conditioned, local rotation
- `caption_global`: Text-conditioned, global rotation

**Key Features:**
- Pre-extracted caption embedding cache support (avoids text encoding at eval time)
- Supports phases 0-2 with different training paradigms
- SMPL and KIMODO root representation variants
- Complex task definitions (E2, E3, E4, E5, etc.)
- Replacement guidance for MAN imputation
- Per-task and per-model evaluation

**Usage Example:**
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --tasks E2 E3 E4 E5 \
    --models caption_local caption_global \
    --max-samples 50 \
    --output-dir work_dirs/m2m_v2_eval
```

**Configuration Options:**
- `--tasks`: Task IDs to evaluate
- `--settings`: A/B/C settings for specific tasks
- `--models`: Model variants to use
- `--max-samples`: Limit number of samples
- `--replacement-guidance`: Handle MAN imputation strategy

---

### 1.3 Unified Inference Entry Point

**Location:** `tools/infer.py` (14 KB, recently updated May 18 12:11)

**Purpose:** Single entry point for inference across all motion pipelines (PRISM, HyMotion T2M, M2M, VerMo, etc.)

**Supported Pipelines:**
- **PRISM T2M:** Text-to-motion generation
- **PRISM MCM:** Motion-to-music dance generation
- **HyMotion T2M:** Text-to-motion (fixed structure)
- **HyMotion M2M:** Motion editing/completion
- **VerMo:** Multi-task motion-language model

**Usage Examples:**

```bash
# PRISM text-to-motion
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --prompt "a person walks forward" \
    --output output/motion.npz

# HyMotion M2M motion editing
python tools/infer.py \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py \
    --checkpoint work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-final \
    --input src_motion.npz \
    --output output/edited.npz

# VerMo multi-task
python tools/infer.py \
    --config configs/vermo/vermo_smoke.py \
    --checkpoint work_dirs/vermo_smoke/checkpoint-iter_10 \
    --task t2m \
    --prompt "a person sits down" \
    --output output/motion.npz
```

**Key Parameters:**
- `--prompt`: Text prompt for generation
- `--input` / `--motion` / `--past-motion` / `--future-motion`: Input motion files
- `--num-steps`: Denoising steps (diffusion)
- `--num-samples`: Number of generated samples
- `--guidance-scale`: CFG scale (default: 5.0)
- `--first-frame-motion`: First frame condition for PRISM
- Device selection: `--device` (cuda/cpu)
- Optional: `--merge-lora` for LoRA adapter merging

---

### 1.4 Batch Inference for VerMo

**Location:** `scripts/inference/batch_infer_vermo.py` (21 KB)

**Purpose:** Batch inference script for VerMo multi-task motion generation

**Handles:** Multiple generation tasks from annotation files

---

### 1.5 Additional Evaluation Scripts in `scripts/eval/`

**Key Scripts:**
- `eval_m2m_v2_t2m.py` (34 KB, executable) - T2M-specific evaluation for M2M v2
- `eval_with_motionclip_evaluator.py` (24 KB) - MotionCLIP metric evaluation
- `eval_momask_native_h3d263.py` (15 KB) - MoMask H3D-263 evaluation
- `momask_infer_h3d_test.py` (13 KB) - MoMask inference on H3D test set
- `eval_prism_kafs_ablation.py` (17 KB) - PRISM KAFS ablation (described above)
- `compute_kafs_metrics.py` (9 KB) - Compute KAFS-specific metrics

**Compilation/Launcher Scripts:**
- `run_kafs_ablation.sh` - Launch KAFS ablation evaluation
- `launch_kafs_single.py` - Single-sample KAFS launcher
- `run_m2m_v2_eval_latest.sh` - Latest M2M v2 evaluation runner

---

## 2. DATASET CONFIGURATIONS

### 2.1 Test Annotation Files (HumanML3D and MotionHub)

**Location:** `data/annotation/`

**Test Set Files:**

| File | Type | Purpose |
|------|------|---------|
| `test_hml3d.json` | symlink | HumanML3D test set annotations |
| `test_hml3d_rewritten.json` | symlink | HumanML3D with rewritten captions |
| `test_motionhub_t2m.json` | symlink | MotionHub T2M test set |
| `test_motionhub_t2m_rewritten.json` | symlink | MotionHub T2M with rewritten captions |
| `test_motionhub_1p.json` | symlink | MotionHub 1-person test set |
| `test_motionhub_2p.json` | symlink | MotionHub 2-person test set |
| `test_motionhub_m2d.json` | symlink | MotionHub motion-to-dance |
| `test_motionhub_pred.json` | symlink | MotionHub prediction task |
| `test_motionhub_recon.json` | symlink | MotionHub reconstruction |
| `test_motionhub_s2g.json` | symlink | MotionHub speech-to-gesture |

**Actual Source Location:**
- Symlinks point to: `/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/data/annotation/`

**Training Annotation Files:**

| File | Size | Updated |
|------|------|---------|
| `train_hymotion_400h_hq_permo_motionfix_20260514.json` | 212.6 MB | May 14 03:30 |
| `train_hymotion_400h_hq_permo_motionfix_editing_20260514.json` | 216.7 MB | May 14 14:13 |

---

### 2.2 PRISM Config Files

**Location:** `configs/prism/`

**Available Configs:**

| Config | Purpose |
|--------|---------|
| `prism_1b_tp2m_1frame.py` | 1B T2P2M with single frame (primary) |
| `prism_1b_tp2m_1frame_kt_dfs.py` | KT DFS variant |
| `prism_1b_tp2m_1frame_kt_spectral.py` | KT Spectral variant |
| `prism_1b_tp2m_multiframe.py` | Multi-frame variant |
| `prism_1b_tp2m_multiframe_kt_dfs.py` | KT DFS multi-frame |
| `prism_1b_tp2m_multiframe_kt_spectral.py` | KT Spectral multi-frame (recent: May 17) |
| `prism_mcm_motionhub.py` | Motion-to-music on MotionHub |
| `prism_mcm_motionhub_16v100.py` | 16-GPU V100 variant |
| `prism_mcm_motionhub_64v100.py` | 64-GPU V100 variant |
| `prism_debug_loss_split.py` | Loss debugging config |

**Latest PRISM Checkpoint:**
- `work_dirs/prism_1b_tp2m_multiframe_kt_spectral/` (Updated May 18 07:11, 11.3 GB)

---

### 2.3 HyMotion M2M v2 Model Registry

**Location:** `scripts/eval/eval_m2m_v2_all_tasks.py` (lines 113-200+)

**Model Variants (V2_MODELS dictionary):**

**Phase 0 - Root Representation Ablations:**
- `kimodo_caption_E4`: KIMODO root + caption (E4 training)
- `smpl_caption_E2`: SMPL root + caption (E2 training)
- `kimodo_uncond_E3`: KIMODO root + unconditioned (E3 training)
- `smpl_uncond_E1`: SMPL root + unconditioned (E1 training)

**Phase 1 & 2 - Mixed Training Variants:**
- `uncond_local/global`: Unconditioned with local/global rotation
- `caption_local/global`: Text-conditioned with local/global rotation
- `caption_local_phase1/2` and `caption_global_phase1/2`: Phase-specific variants

**Model Work Directories:**
- `work_dirs/hymotion_m2m_v2_smpl_uncond_E1/`
- `work_dirs/hymotion_m2m_v2_kimodo_uncond_E3/`
- `work_dirs/hymotion_m2m_v2_smpl_caption_E2/`
- `work_dirs/hymotion_m2m_v2_kimodo_caption_E4/`

---

## 3. PREVIOUS EVALUATION RESULTS

### 3.1 Output Directory Structure

**Primary Location:** `output/` (29 TB)

**Recent Evaluation Runs:**

| Directory | Type | Size | Date |
|-----------|------|------|------|
| `embodied_t2m_v4/` | T2M eval | 568.7 GB | May 14 14:54 |
| `embodied_t2m_v6_test/` | T2M test | 873.5 MB | May 14 05:48 |
| `embodied_t2m_v3/` | T2M eval | 142.1 GB | May 13 16:03 |
| `embodied_t2m_v5/` | T2M eval | 142.1 GB | May 13 03:12 |
| `eval_e14_rerun_20260509/` | E14 eval | 98.3 GB | May 9 11:56 |
| `embodied_comparison/` | Comparison | 142.1 GB | May 12 20:06 |

**Pattern:** Most recent evaluations use directory names with date stamps (e.g., `eval_v2_e9_20260423_manifold`)

---

### 3.2 Work Directory Results

**Primary Location:** `work_dirs/` (553 TB, 296 model directories)

**Recent M2M v2 Evaluations:**
- `m2m_v2_eval_four_new_missing_all_20260514_1306_machine1/` (1.4 TB) - May 14 15:32
- `m2m_v2_eval_four_new_missing_all_20260514_1306_machine2/` (1.4 TB) - May 14 14:18
- `m2m_v2_eval_four_new_20260514_1203_correspond_machine1/` (206.8 GB) - May 14 12:27

**Recent PRISM Evaluations:**
- `prism_1b_tp2m_multiframe_kt_spectral/` (11.3 GB) - May 18 07:11
- `prism_kafs_ablation/` (102.2 GB) - May 15 15:05

**Latest M2M v2 Training Models:**
- `hymotion_m2m_v2_smpl_caption_E2/` (531.5 GB) - May 18 11:23 (latest)
- `hymotion_m2m_v2_kimodo_caption_E4/` (487.2 GB) - May 18 12:12
- `hymotion_m2m_v2_kimodo_uncond_E3/` (265.7 GB) - May 18 09:54
- `hymotion_m2m_v2_smpl_uncond_E1/` (553.6 GB) - May 18 11:09

---

### 3.3 Evaluation Results Storage

**Location:** `eval_results/`

| Directory | Type | Size | Date |
|-----------|------|------|------|
| `m2m/` | M2M results | 2.65 TB | Apr 12 23:30 |
| `m2m_smoke/` | Smoke test | 2.7 GB | Apr 7 00:06 |

**Results Format:** Per-task NPZ files with:
- Generated motions
- Metrics (FID, Diversity, Multimodality, etc.)
- Metadata and embeddings

---

## 4. KEY DATA LOCATIONS

### 4.1 Motion Data

**Location:** `data/motionhub/` (symlink likely or local cache)

**Also:**
- `data/annotation/` - Test/train annotation files
- `data/eval/m2m_v2/caption_embeddings/cache.pt` - Pre-extracted caption embeddings for eval

### 4.2 Checkpoints

**Primary:** `checkpoints/` (121.7 GB)

**Work Directory Checkpoints:** `work_dirs/*/checkpoint-*`

---

## 5. CONFIGURATION PATTERNS

### 5.1 HumanML3D Test Set Configuration

From `eval_prism_kafs_ablation.py`:
```python
--anno-file data/annotation/test_hml3d.json
--data-dir data/motionhub  # or data/humanml3d
```

**Format:** JSON with entries:
- `name`: sample identifier
- `caption`: text description
- `motion_path`: path to motion file

### 5.2 Batch Inference Configuration

**Typical Pattern:**
1. Load annotation JSON (`test_hml3d.json` or `test_motionhub_t2m.json`)
2. For each sample:
   - Load caption (random from pool if hierarchical format)
   - Load motion (if motion-conditioned)
   - Generate with model + diffusion
   - Save as NPZ to output directory

**Output Pattern:** `output_dir/{sample_name}.npz`

---

## 6. QUICK REFERENCE: Running Batch Evaluation

### For PRISM KAFS Ablation:
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/

# Single GPU
python scripts/eval/eval_prism_kafs_ablation.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-latest \
    --kafs-mode depth_driven \
    --anno-file data/annotation/test_hml3d.json \
    --output-dir work_dirs/prism_kafs_eval_test \
    --num-inference-steps 20 \
    --max-samples 10  # For testing
```

### For M2M v2 Full Evaluation:
```bash
python scripts/eval/eval_m2m_v2_all_tasks.py \
    --tasks E2 E3 E4 E5 \
    --models caption_local caption_global \
    --max-samples 100 \
    --output-dir work_dirs/m2m_v2_eval_batch
```

### For Single Inference (PRISM):
```bash
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-latest \
    --prompt "a person walks forward slowly" \
    --output output/test_motion.npz \
    --num-steps 20 \
    --device cuda:0
```

---

## 7. CAPTION EMBEDDING CACHE

**Location:** `data/eval/m2m_v2/caption_embeddings/cache.pt`

**Purpose:** Pre-extracted text embeddings to avoid re-encoding during eval

**Building Cache (if missing):**
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/caption/extract_eval_caption_embeddings.py --force
```

**Format:** PyTorch dict with:
- `cache`: {caption_string: {'text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length'}}
- `meta`: {'llm_type', 'model', ...}

---

## 8. TROUBLESHOOTING & NOTES

1. **Caption Models Missing Text Encoder:**
   - Caption-conditioned models trained with `LoadPreExtractedTextEmbedding` won't have a runtime text_encoder
   - Use pre-extracted embedding cache instead
   - Error will be: "Caption embedding cache not found"

2. **Test Set Annotation Files:**
   - Most are symlinks to `/apdcephfs_cq11/share_1467498/home/zeyuling/versatilemotion/data/annotation/`
   - Actual data path may differ; check symlink target

3. **Checkpoint Structure:**
   - Work dirs store checkpoints in subdirectory format: `work_dirs/{model_name}/checkpoint-{iteration|latest}`
   - Load with: `load_checkpoint(checkpoint_path, map_location='cpu')`

4. **Output File Naming:**
   - Per-sample: `{output_dir}/{sample_name}.npz`
   - Aggregated results: `{output_dir}/metrics.json` or similar

5. **KAFS Modes:**
   - `none`: Standard baseline (no per-joint scaling)
   - `depth_driven`: Per-joint scaling by kinematic tree depth
   - `uniform`: All joints get alpha=1.0 (sanity check)
   - `random`: Random alphas in [0.85, 1.15]

