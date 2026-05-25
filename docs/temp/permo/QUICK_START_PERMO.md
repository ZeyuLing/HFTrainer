# Quick Start: PerMo Integration for Caption Training

## TL;DR - What to do right now

```bash
# Step 1: Fix PerMo annotations and compute stats (30-60 min)
python scripts/data/fix_permo_annotations_and_stats.py

# Step 2: Create merged annotation file (10 sec)
python scripts/data/merge_annotations.py

# Step 3: Test one of the new configs locally (before launching)
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py 8 --auto-resume

# Step 4: Launch on Taiji (when ready)
python tools/taiji_submit.py m2m_v2_smpl_caption_permo_E2plus \
  configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py --host_num 8
```

---

## What's New

### New Config Files
1. **`configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py`**
   - E2 (SMPL Root) + PerMo data
   - 414,094 training samples (407,552 400h + 6,542 PerMo)
   - Same architecture as current E2
   - Recommended starting point

2. **`configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_046b.py`**
   - E4 (KIMODO Root + ADMM smoothing) + PerMo data
   - Better for embodied/robot tasks
   - Same data as E2+PerMo but with trajectory smoothing

### New Scripts
1. **`scripts/data/fix_permo_annotations_and_stats.py`**
   - Fixes PerMo annotation paths
   - Computes 198-dim statistics
   - Creates output: `data/hymotion_data/PerMo/20260513/permo_198dim_stats.npz`

2. **`scripts/data/merge_annotations.py`**
   - Merges 400h + PerMo annotations
   - Creates: `data/annotation/train_hymotion_400h_permo_caption_20260514.json`
   - Can optionally include MotionFix for reference

---

## Step-by-Step Execution

### Step 1: Compute PerMo Statistics (~1 hour)

This loads all PerMo training data and computes mean/std for normalization.

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Run the fix script (automatically computes stats)
python scripts/data/fix_permo_annotations_and_stats.py

# Expected output:
# [INFO] Computing stats from 6542 items...
# [INFO] Loaded 6542 motions, skipped 0
# [INFO] Statistics computed from X frames
# [DONE] Saved data/hymotion_data/PerMo/20260513/permo_198dim_stats.npz
```

**What it does:**
- Verifies all PerMo motion paths
- Loads motion_198 data for each training sample
- Computes per-dimension mean and standard deviation
- Outputs: `permo_198dim_stats.npz` with shapes (1, 198)

**Estimated time**: 30-60 minutes (depends on storage access speed)

### Step 2: Create Merged Annotation (~30 seconds)

```bash
# This merges 400h + PerMo training data
python scripts/data/merge_annotations.py

# Expected output:
# [INFO] Adding base 400h HQ data...
#   └─ 407,552 entries
# [INFO] Adding PerMo training data...
#   └─ 6,542 entries
# [DONE] Saved data/annotation/train_hymotion_400h_permo_caption_20260514.json (8.7 MB)
```

**What it creates:**
- `data/annotation/train_hymotion_400h_permo_caption_20260514.json`
  - 414,094 total entries
  - 407,552 from 400h HQ
  - 6,542 from PerMo train split

### Step 3: Test Locally (Optional but Recommended)

Before launching on Taiji, test that the new config works:

```bash
# Small scale test on local GPUs
bash tools/dist_train.sh \
  configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py \
  8 --auto-resume

# Watch for:
# - DataLoader successfully loads mixed data (400h + PerMo)
# - No path resolution errors
# - Training loss converges normally
# - Can run for 100-500 epochs as sanity check
```

**Expected logs to see:**
```
[INFO] Loading batch...
[INFO] Found 20 samples from mixed datasets
[INFO] Batch contains: 12 from 400h, 8 from PerMo
```

### Step 4: Launch on Taiji

#### Option A: E2+PerMo (SMPL Root - recommended starting point)

```bash
python tools/taiji_submit.py m2m_v2_smpl_caption_permo_E2plus \
  configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py \
  --host_num 8

# Expected: 64 GPUs, ~2 weeks for 10k epochs
```

#### Option B: E4+PerMo (KIMODO Root - smoother trajectories)

```bash
python tools/taiji_submit.py m2m_v2_kimodo_caption_permo_E4plus \
  configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_046b.py \
  --host_num 8

# Expected: 64 GPUs, ~2 weeks for 10k epochs
```

---

## Troubleshooting Quick Reference

### Error: "permo_198dim_stats.npz not found"
**Fix**: Run `fix_permo_annotations_and_stats.py` first

### Error: "annotation file not found"
**Fix**: Run `merge_annotations.py` first

### Error: DataLoader timeout/hangs
**Cause**: Path resolution issues in PerMo annotation
**Fix**: Verify `fix_permo_annotations_and_stats.py` completed successfully
**Check**: 
```python
import json
with open('data/hymotion_data/PerMo/20260513/permo_hymotion_train.json') as f:
    anno = json.load(f)
sample = list(anno['data_list'].values())[0]
print(sample['smplx_path'])  # Should start with ../hymotion_data/
```

### Error: "motion_198 shape mismatch"
**Cause**: Old PerMo NPZ with only motion_135
**Fix**: Regenerate PerMo with:
```bash
python scripts/data/convert_permo_to_hymotion_198dim.py
```

---

## What Changed vs. Current E2

| Aspect | Current E2 | E2+PerMo |
|--------|-----------|----------|
| Dataset | 400h HQ only | 400h HQ + PerMo |
| Train entries | 407,552 | 414,094 (+1.6%) |
| Training epochs | 10k | 10k (same) |
| Per-epoch time | ~8 hrs | ~8-9 hrs (+1-2%) |
| Config file | `hymotion_m2m_v2_smpl_caption_046b.py` | `hymotion_m2m_v2_smpl_caption_permo_046b.py` |
| Annotation | `train_hymotion_400h_hq_20260403.json` | `train_hymotion_400h_permo_caption_20260514.json` |
| Checkpoint | From E2-phase2 | From E2-phase2 |

---

## Expected Results

### Training
- **Loss convergence**: Should be very similar to E2 (slight noise due to data mixing)
- **Speed**: +1-2% slower per epoch (more data to load)
- **Total time**: ~200-240 days on 64 GPUs for 10k epochs

### Model Quality
- **Generalization**: Better (diverse motion sources)
- **Caption diversity**: Better (different annotation style)
- **MotionFix performance**: Use E2+PerMo model + MotionFix val set to evaluate

---

## File Locations Reference

| Purpose | Path |
|---------|------|
| New config (SMPL) | `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py` |
| New config (KIMODO) | `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_046b.py` |
| PerMo annotations | `data/hymotion_data/PerMo/20260513/permo_hymotion_*.json` |
| PerMo stats | `data/hymotion_data/PerMo/20260513/permo_198dim_stats.npz` |
| PerMo motions | `data/hymotion_data/PerMo/PerMo/20260513/motions/` |
| PerMo 198-dim | `data/hymotion_data/PerMo/PerMo/20260513/motions_198/` |
| Merged annotation | `data/annotation/train_hymotion_400h_permo_caption_20260514.json` |
| Merge script | `scripts/data/merge_annotations.py` |
| Fix script | `scripts/data/fix_permo_annotations_and_stats.py` |

---

## Timeline Recommendation

**This week (May 14-20)**:
- [ ] Run `fix_permo_annotations_and_stats.py` (1 hour, patience for compute)
- [ ] Run `merge_annotations.py` (30 seconds)
- [ ] Test locally with one of the new configs
- [ ] Launch E2+PerMo on Taiji

**Next week (May 21-27)**:
- [ ] Monitor E2+PerMo training progress
- [ ] Evaluate intermediate checkpoints
- [ ] Optionally launch E4+PerMo in parallel

**Following week (May 28+)**:
- [ ] Compare E2+PerMo vs. E2-only at same epoch
- [ ] Evaluate on MotionFix val/test sets
- [ ] Prepare for final model release

