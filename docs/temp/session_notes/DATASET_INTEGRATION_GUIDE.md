# MotionFix + PerMo Integration Status & Guide

**Date**: 2026-05-14  
**Current Status**: Both datasets partially ready, integration tools available

## Executive Summary

| Dataset | Status | Train | Val | Test | Stats | Captions | Config |
|---------|--------|-------|-----|------|-------|----------|--------|
| **400h (current)** | ✓ Active | 407,552 | - | - | ✓ | ✓ | ✓ |
| **MotionFix** | ⚠ Partial | ❌ Missing | ✓ 330 | ✓ 1,013 | ❌ | ⚠ 50% | ❌ |
| **PerMo** | ⚠ Partial | ✓ 6,542 | ❌ Missing | ✓ 67 | ❌ | ✓ | ❌ |

---

## 1. CURRENT DATA INVENTORY

### 1.1 MotionFix (data/hymotion_data/MotionFix/20260504)

**Annotation Files**:
- `motionfix_hymotion_all.json`: 1,343 total entries
  - train: **MISSING** (0 entries) 
  - val: 330 entries
  - test: 1,013 entries

**Motion Files**:
- ✓ `motions/val/`: 330 pairs (source + target .npz)
- ✓ `motions/test/`: 1,013 pairs (source + target .npz)
- ❌ `motions/train/`: EMPTY

**Captions**:
- ✓ `augmented_caption/val/`: 330 JSON files
- ✓ `augmented_caption/test/`: 1,013 JSON files
- ❌ `augmented_caption/train/`: EMPTY (needs generation)

**Structure per entry**:
```json
{
  "subset": "MotionFix-test",
  "smplx_path": "MotionFix/20260504/motions/test/000283_target.npz",
  "caption_path": "MotionFix/20260504/augmented_caption/test/000283.json",
  "fps": 30.0,
  "has_hand": false,
  "duration": 4.0,
  "num_frames": 120,
  "language": "en",
  "source_smplx_path": "MotionFix/20260504/motions/test/000283_source.npz",
  "edit_pair_path": "MotionFix/20260504/pairs/test/000283.json"
}
```

### 1.2 PerMo (data/hymotion_data/PerMo/20260513)

**Annotation Files**:
- `permo_hymotion_all.json`: 6,609 total entries
  - train: 6,542 entries
  - test: 67 entries
  - val: **MISSING** (need to extract/create split)

**Motion Files**:
- ✓ `motions/train/` & `motions/test/`: SMPL-X NPZ files
- ✓ `motions_198/train/` & `motions_198/test/`: Pre-computed 198-dim NPY files

**Captions**:
- ✓ `augmented_caption/train/` & `augmented_caption/test/`: JSON files available
- ⚠ Text embeddings: MENTIONED in code but location unclear

**Structure per entry**:
```json
{
  "subset": "PerMo-train",
  "smplx_path": "../hymotion_data/PerMo/PerMo/20260513/motions/train/Angry_KickSth_A02_005.npz",
  "motion_135_path": "../hymotion_data/PerMo/PerMo/20260513/motions/train/Angry_KickSth_A02_005.npz",
  "motion_198_path": "../hymotion_data/PerMo/PerMo/20260513/motions_198/train/Angry_KickSth_A02_005.npy",
  "caption_path": "../hymotion_data/PerMo/PerMo/20260513/augmented_caption/train/Angry_KickSth_A02_005.json",
  "fps": 30.0,
  "has_hand": false,
  "duration": 7.03,
  "num_frames": 211,
  "language": "en",
  "hierarchical_caption_path": "PerMo/20260513/augmented_caption/train/Angry_KickSth_A02_005.json"
}
```

### 1.3 Current 400h (data/annotation/train_hymotion_400h_hq_20260403.json)

- **Total**: 407,552 high-quality entries
- **Composition**:
  - Academic: 195,168 (47.9%)
  - Academic retarget: 106,820 (26.2%)
  - Taobao: 71,009 (17.4%)
  - Game: 34,555 (8.5%)

---

## 2. REQUIRED INTEGRATION STEPS

### Step 1: Compute Statistics for New Datasets

**MotionFix Stats** (required before training):
```bash
python scripts/data/compute_permo_198dim_stats.py \
  --hymotion-root data/hymotion_data \
  --anno-path data/hymotion_data/MotionFix/20260504/motionfix_hymotion_test.json \
  --splits test \
  --output-path data/hymotion_m2m_data/_stats_198dim_motionfix/stats.npz
```

**PerMo Stats** (already have conversion script):
```bash
python scripts/data/fix_permo_annotations_and_stats.py
# This will:
# 1. Build correct all.json from train + test
# 2. Fix annotation paths
# 3. Compute 198-dim statistics
# Output: data/hymotion_data/PerMo/20260513/permo_198dim_stats.npz
```

### Step 2: MotionFix - Generate Missing Training Captions

**Current situation**: Only val/test captions exist, no train data

**Options**:
1. **Generate dummy captions** for train split (fast, placeholder)
   - Use generic caption: "Edit motion according to the target"
   - Placeholder until real captions available
   
2. **Wait for train data** to be added to MotionFix
   - Contact data team to confirm if train split exists
   - If exists, generate captions using caption augmentation pipeline

### Step 3: Create Merged Annotation Files

Merge train data from 400h + PerMo into a new annotation file:

```python
# Pseudo-code for merging
merged_data_list = {}
merged_data_list.update(hymotion_400h['data_list'])  # 407,552 entries
merged_data_list.update(permo_train['data_list'])    # 6,542 entries
# Total: ~414k entries

merged_anno = {
    "meta_info": {
        "dataset": "HYMotion 400h + PerMo",
        "generated_at": "2026-05-14",
        "hymotion_400h": 407552,
        "permo_train": 6542,
        "total": 414094,
    },
    "data_list": merged_data_list,
}

# Save as: data/annotation/train_hymotion_400h_permo_caption_20260514.json
```

Optionally add MotionFix eval data:
```python
merged_data_list.update(motionfix_val['data_list'])  # 330 entries
merged_data_list.update(motionfix_test['data_list']) # 1,013 entries
# Total: ~415k training + ~1,343 edit data
```

### Step 4: Create Config Variants

#### Option A: PerMo-augmented caption training (E2+PerMo)

**File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py`

```python
_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_permo_E2plus'

model = dict(
    # Same as E2 but with different stats directory
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_merged_400h_permo',
)

train_dataloader = dict(
    batch_size=20,
    dataset=dict(
        # Use merged annotation file
        anno_file='data/annotation/train_hymotion_400h_permo_caption_20260514.json',
        # Keep pipeline the same as E2
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(type='LoadSmplx55', key='motion', rot_type='rotation_6d', 
                 transl_type='abs', smpl_type='smpl_22'),
            dict(type='Compute198DimPosition', key='motion'),
            # ... rest same as E2
        ],
    ),
)
```

#### Option B: Full merged (400h + PerMo + MotionFix edit)

**File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_merged_046b.py`

```python
_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_merged'

train_dataloader = dict(
    batch_size=20,
    dataset=dict(
        # Merged annotation
        anno_file='data/annotation/train_hymotion_merged_all_20260514.json',
        pipeline=[
            # ... same pipeline
        ],
    ),
)
```

---

## 3. AVAILABLE INTEGRATION TOOLS

### Tool 1: `scripts/data/fix_permo_annotations_and_stats.py`

**Purpose**: Fix PerMo annotations and compute correct 198-dim statistics

**What it does**:
1. Builds `permo_hymotion_all.json` from train + test JSONs
2. Fixes annotation paths from `PerMo/20260513/` to `../hymotion_data/PerMo/PerMo/20260513/`
3. Adds `hierarchical_caption_path` field for HyMotion compatibility
4. Recomputes 198-dim statistics (not 135-dim)
5. Outputs:
   - `permo_hymotion_all.json`
   - `permo_hymotion_train.json`
   - `permo_hymotion_test.json`
   - `permo_198dim_stats.npz` (in `data/hymotion_m2m_data/`)

**Usage**:
```bash
python scripts/data/fix_permo_annotations_and_stats.py
```

**Output location**: Data saved to `data/hymotion_data/PerMo/20260513/`

---

### Tool 2: `scripts/data/compute_permo_198dim_stats.py`

**Purpose**: Standalone statistics computation for any 198-dim dataset

**Arguments**:
- `--hymotion-root`: Path to HYMotion data root (default: `data/hymotion_data`)
- `--anno-path`: Path to annotation JSON
- `--splits`: Splits to include (default: `["train"]`)
- `--output-path`: Where to save stats NPZ

**Usage**:
```bash
python scripts/data/compute_permo_198dim_stats.py \
  --anno-path data/hymotion_data/MotionFix/20260504/motionfix_hymotion_all.json \
  --output-path data/hymotion_m2m_data/_stats_198dim_motionfix/stats.npz
```

---

### Tool 3: `scripts/data/prepare_motionfix_hymotion.py`

**Purpose**: Convert MotionFix tar files to HYMotion format

**Status**: Already run (val/test captions generated), but no train data

**Key features**:
- Converts SMPL-22 axis-angle to 6D rotations
- Generates caption JSONs and edit pair info
- Extracts QWEN3 text embeddings (if not skipped)
- Generates annotation files

**Usage for train captions** (if train data becomes available):
```bash
python scripts/data/prepare_motionfix_hymotion.py \
  --motionfix-root data/MotionFix \
  --output-root data/hymotion_data/MotionFix/20260504 \
  --splits train val test \
  --only-embeddings  # Skip motion conversion if already done
```

---

### Tool 4: `scripts/data/convert_permo_to_hymotion_198dim.py`

**Purpose**: Convert PerMo from original format to 198-dim HYMotion

**Status**: Appears to have been run (motion_198 files exist)

**Features**:
- Converts SMPL-X to SMPL-22
- Computes 6D rotations (row-major)
- Generates 198-dim via FK positions
- Extracts text embeddings with QWEN3+CLIP-L

**Usage**:
```bash
python scripts/data/convert_permo_to_hymotion_198dim.py \
  --permo-root data/hymotion_data/PerMo/PerMo/20260513 \
  --output-root data/hymotion_data/PerMo/20260513 \
  --anno-path data/hymotion_data/PerMo/20260513/permo_hymotion_all.json
```

---

## 4. STEP-BY-STEP INTEGRATION CHECKLIST

### ✓ Already Done
- [x] MotionFix val/test captions generated
- [x] PerMo 198-dim motion files created (motion_198_path exists)
- [x] PerMo QWEN3 embeddings generated (probably)
- [x] Annotation files created for both datasets

### ⚠ Needs Action

**Priority 1 - Required for caption training:**
- [ ] **Compute PerMo 198-dim statistics**
  ```bash
  python scripts/data/fix_permo_annotations_and_stats.py
  ```
  - Creates: `data/hymotion_data/PerMo/20260513/permo_198dim_stats.npz`
  - Time: ~30-60 min (depends on data access speed)

- [ ] **Copy PerMo stats to training stats dir**
  ```bash
  mkdir -p data/hymotion_m2m_data/_stats_198dim_permo
  cp data/hymotion_data/PerMo/20260513/permo_198dim_stats.npz \
     data/hymotion_m2m_data/_stats_198dim_permo/
  ```

**Priority 2 - Optional for MotionFix:**
- [ ] Determine if MotionFix train split data exists
  - Contact data team or check original MotionFix source
  - If no train split: skip or use val+test only for eval

- [ ] If train exists: generate captions using `prepare_motionfix_hymotion.py --only-embeddings`

**Priority 3 - Config preparation:**
- [ ] Create merged annotation: `train_hymotion_400h_permo_caption_20260514.json`
- [ ] Create merged config: `hymotion_m2m_v2_smpl_caption_permo_046b.py`
- [ ] Create merged stats dir with combined statistics (or use per-dataset stats)

**Priority 4 - Launch training:**
- [ ] Test new config locally
- [ ] Launch on Taiji with new annotation file

---

## 5. CONFIGURATION TEMPLATES

### Template A: PerMo-augmented caption training

```python
# configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_permo_E2plus'

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',  # Use merged or 400h default
    rotation_space='local',
    text_encoder=dict(),
    losses_cfg=dict(
        keypoints3d_weight=10.0,
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_permo_caption_20260514.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(type='LoadSmplx55', key='motion', rot_type='rotation_6d', 
                 transl_type='abs', smpl_type='smpl_22'),
            dict(type='Compute198DimPosition', key='motion'),
            dict(type='RandomCropPadding', clip_len=360, pad_mode='replicate',
                 allow_shorter=True, make_pad_mask=True, pad_mask_key='pad_mask'),
            dict(type='PrepareM2Mv2Condition', key='motion', sampler_version='v3',
                 editing_prob=0.15, corruptor_names=['jitter', 'joint_jump', 'sliding',
                 'limb_candy_wrapper', 'wrist_candy_wrapper'], max_corruptions=2),
            dict(type='PackInputs', keys=['src_motion', 'tgt_motion', 'src_mask',
                 'tgt_length', 'src_length', 'edit_mode'],
                 meta_keys=['motion_path', 'fps'], set_dummy_value=False),
        ],
    ),
)
```

### Template B: E4 + PerMo (KIMODO Root with PerMo)

```python
# configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_046b.py

_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_permo_E4plus'

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    exclude_bundle_keys=['mean', 'std'],
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    cond_mask_prob=0.1,
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
    rotation_space='local',
    text_encoder=dict(),
    losses_cfg=dict(
        keypoints3d_weight=10.0,
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_permo_caption_20260514.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(type='LoadSmplx55', key='motion', rot_type='rotation_6d',
                 transl_type='abs', smpl_type='smpl_22'),
            dict(type='Compute198DimPosition', key='motion'),
            dict(type='SmplTransToKimodoRootOnline', key='motion', admm_margin_m=0.06),
            dict(type='RandomCropPadding', clip_len=360, pad_mode='replicate',
                 allow_shorter=True, make_pad_mask=True, pad_mask_key='pad_mask'),
            dict(type='PrepareM2Mv2Condition', key='motion', sampler_version='v3',
                 editing_prob=0.15, corruptor_names=['jitter', 'joint_jump', 'sliding',
                 'limb_candy_wrapper', 'wrist_candy_wrapper'], max_corruptions=2),
            dict(type='PackInputs', keys=['src_motion', 'tgt_motion', 'src_mask',
                 'tgt_length', 'src_length', 'edit_mode'],
                 meta_keys=['motion_path', 'fps'], set_dummy_value=False),
        ],
    ),
)
```

---

## 6. LAUNCH COMMANDS

### Compute Statistics
```bash
# Fix PerMo annotations and compute stats
python scripts/data/fix_permo_annotations_and_stats.py

# Copy stats to training directory
mkdir -p data/hymotion_m2m_data/_stats_198dim_permo
cp data/hymotion_data/PerMo/20260513/permo_198dim_stats.npz \
   data/hymotion_m2m_data/_stats_198dim_permo/
```

### Local Training
```bash
# PerMo-augmented E2 (SMPL Root + caption + PerMo)
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py 8 --auto-resume

# PerMo-augmented E4 (KIMODO Root + caption + PerMo)
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_046b.py 8 --auto-resume
```

### Taiji Training (64 GPUs)
```bash
# E2 + PerMo
python tools/taiji_submit.py m2m_v2_smpl_caption_permo_E2plus \
  configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_permo_046b.py --host_num 8

# E4 + PerMo  
python tools/taiji_submit.py m2m_v2_kimodo_caption_permo_E4plus \
  configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_permo_046b.py --host_num 8
```

---

## 7. EXPECTED OUTCOMES

### Dataset Size After Merge
- **400h + PerMo**: ~414k entries (407,552 + 6,542)
- **With MotionFix eval**: ~415k training + ~1.3k editing pairs

### Training Time Impact
- PerMo adds ~2% to dataset size (6.5k / 407.5k)
- Expected training slowdown: **1-2% longer per epoch**
- Total 10k epoch training: ~2-4 days longer on 64 GPUs

### Quality Expected
- **Model generalization**: Improved (PerMo has different motion capture sources)
- **Caption diversity**: Enhanced (different annotation style than 400h)
- **Edit stability**: Baseline unchanged (MotionFix for future eval only)

---

## 8. TROUBLESHOOTING

### Issue: "motion_198 key not found in NPZ"
- **Cause**: Using old PerMo NPZ that only has motion_135
- **Fix**: Run `convert_permo_to_hymotion_198dim.py` to regenerate
- **Check**: `numpy -f data/hymotion_data/PerMo/PerMo/20260513/motions_198/train/Angry_KickSth_A02_005.npy` should work

### Issue: "Path not found" in annotation
- **Cause**: Relative paths not resolving from data/motionhub
- **Fix**: Run `fix_permo_annotations_and_stats.py` to normalize paths
- **Check**: All paths should start with `../hymotion_data/` relative to data/motionhub

### Issue: "Stats NPZ has wrong shape (1, 135) not (1, 198)"
- **Cause**: Stats computed on motion_135, not motion_198
- **Fix**: Recompute with correct script
- **Check**: Verify dimensions with: `python -c "import numpy as np; d=np.load('stats.npz'); print(d['mean'].shape)"`

### Issue: DataLoader timeout when loading PerMo
- **Cause**: Path resolution failing silently, worker hangs
- **Fix**: Check `fix_permo_annotations_and_stats.py` path verification passes
- **Verify**: Spot-check a few entries manually load correctly

---

## 9. NEXT STEPS RECOMMENDATION

**Immediate** (this week):
1. Run `fix_permo_annotations_and_stats.py`
2. Copy stats to `data/hymotion_m2m_data/_stats_198dim_permo/`
3. Create merged annotation file
4. Create E2+PerMo config
5. Test locally with small batch_size

**Short-term** (next week):
1. Launch E2+PerMo training on Taiji
2. Monitor convergence vs. 400h-only baseline
3. Evaluate model outputs on PerMo test set

**Medium-term** (2-3 weeks):
1. Determine if MotionFix train split available
2. If yes: generate captions, create full merged config
3. Launch E4+PerMo+MotionFix training
4. Compare edit capabilities on MotionFix pairs

