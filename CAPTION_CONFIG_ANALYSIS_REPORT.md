# HyMotion M2M v2 Caption Training Configs - Complete Report
**Date:** 2026-05-14 | **Status:** Active Experiments E2 & E4

---

## EXECUTIVE SUMMARY

This report documents the current caption training configs for HyMotion M2M v2:
- **E2 (SMPL Caption)**: Standard SMPL Root representation + text conditioning
- **E4 (KIMODO Caption)**: KIMODO Root representation (with ADMM smoothing) + text conditioning

Both use the **same 400-hour dataset** (407,552 high-quality entries) from `train_hymotion_400h_hq_20260403.json`.

**To add PerMo and MotionFix data**, you must:
1. Create annotation files for new datasets (following same structure)
2. Merge into unified annotation JSON
3. Update `anno_file` path in config
4. Adjust batch size/resources as needed

---

## 1. CURRENT ANNOTATION FILE ANALYSIS

### File Location & Metadata
**Path:** `data/annotation/train_hymotion_400h_hq_20260403.json`

**File Structure:**
```json
{
  "meta_info": {
    "dataset": "hymotion_data - train (high quality only)",
    "version": "v1_hq",
    "source": "train_hymotion_400h.json filtered by high_quality.json",
    "generated_at": "2026-04-03 11:29:34",
    "original_count": 549130,
    "filtered_count": 407552
  },
  "data_list": { ... }
}
```

### Dataset Composition (Current 400h)

| Dataset | Count | Percentage | Notes |
|---------|-------|-----------|-------|
| academic | 195,168 | 47.9% | HumanML3D, HumanEva (public datasets) |
| academicretarget | 106,820 | 26.2% | Retargeted academic data |
| taobao | 71,009 | 17.4% | Commercial motion capture |
| game | 34,555 | 8.5% | Game animation sequences |
| **TOTAL** | **407,552** | **100%** | High-quality filtered subset |

### Data Entry Structure (per motion)

```json
{
  "subset": "academic",
  "duration": 10.0,
  "num_frames": 300,
  "smplx_path": "../hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S1_Static_poses.npz",
  "hierarchical_caption_path": "../hymotion_data/Academic/20250916/improved_simple_augmented_caption/HumanML3D-HumanEva/S1_Static_poses.json",
  "fps": 30.0,
  "has_hand": true
}
```

**Key Fields:**
- `subset`: Dataset source category
- `duration`: Motion length in seconds
- `num_frames`: Frame count
- `smplx_path`: Path to SMPL-X 55-dim representation (relative to data_dir)
- `hierarchical_caption_path`: Path to caption annotations JSON (required for caption configs)
- `fps`: Frame rate
- `has_hand`: Boolean for hand tracking presence

---

## 2. E2: SMPL CAPTION CONFIG (`hymotion_m2m_v2_smpl_caption_046b.py`)

### Experiment Overview
**E2**: SMPL Root + Caption Conditioning baseline
- **Representation**: 198-dim SMPL Root (standard motion without ADMM smoothing)
- **Text Conditioning**: Enabled with 10% CFG probability
- **Motion Space**: Local (SMPL frame)
- **Purpose**: Validates text-to-motion with basic SMPL representation

### Configuration

```python
_base_ = './_base_hymotion_m2m_v2_046b.py'
work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_E2'

# Resume from phase2 checkpoint
load_from = dict(
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',  # Reset optimizer/scheduler
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,     # Text conditioning enabled
    cond_mask_prob=0.1,         # 10% CFG during training
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
    rotation_space='local',
    text_encoder=dict(),        # Default QWEN3 + CLIP-L
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
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),  # REQUIRES captions
            dict(type='LoadSmplx55', rot_type='rotation_6d', ...),
            dict(type='Compute198DimPosition', key='motion'),
            dict(type='RandomCropPadding', clip_len=360, ...),
            dict(type='PrepareM2Mv2Condition', sampler_version='v3', ...),
            dict(type='PackInputs', ...),
        ],
    ),
)
```

### Data Pipeline Sequence
1. **LoadCompatibleCaption** → Loads text captions (fail if missing)
2. **LoadSmplx55** → Loads 135-dim motion (3 trans + 22×6 rot6d)
3. **Compute198DimPosition** → FK-derives position channels (135-dim → 198-dim)
4. **RandomCropPadding** → Crops/pads to 360 frames (12 sec @ 30fps)
5. **PrepareM2Mv2Condition** → Sampler v3 + data corruption
6. **PackInputs** → Final packaging

---

## 3. E4: KIMODO CAPTION CONFIG (`hymotion_m2m_v2_kimodo_caption_046b.py`)

### Experiment Overview
**E4**: KIMODO Root + Caption Conditioning
- **Representation**: 198-dim KIMODO Root (with online ADMM smoothing)
- **Text Conditioning**: Enabled with 10% CFG probability
- **Motion Space**: Local (SMPL frame)
- **Smoothing**: 6cm margin on XZ plane (horizontal), Y-axis unsmoothed
- **Purpose**: Better consistency for embodied tasks (robot deployment)

### KIMODO Root Layout (198-dim)
```
[0:3]       ADMM-smoothed pelvis translation (online smoothing)
[3:9]       root joint 6D rotation (continuous)
[9:135]     body (21 non-root joints) 6D rotations
[135:198]   FK-derived joint positions relative to pelvis (21 × 3)
```

### Configuration

```python
_base_ = './_base_hymotion_m2m_v2_046b.py'
work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_E4'

# Resume from phase2 checkpoint
load_from = dict(
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    exclude_bundle_keys=['mean', 'std'],  # Prevent stat overwrite
)

model = dict(
    pred_type='velocity',
    uncondition_mode=False,     # Text conditioning enabled
    cond_mask_prob=0.1,         # 10% CFG during training
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',  # Different stats!
    rotation_space='local',
    text_encoder=dict(),        # Default QWEN3 + CLIP-L
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
        anno_file='data/annotation/train_hymotion_400h_hq_20260403.json',
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),  # REQUIRES captions
            dict(type='LoadSmplx55', rot_type='rotation_6d', ...),
            dict(type='Compute198DimPosition', key='motion'),
            # KEY DIFFERENCE: KIMODO Root conversion with ADMM smoothing
            dict(
                type='SmplTransToKimodoRootOnline',
                key='motion',
                admm_margin_m=0.06,  # 6cm XZ margin
            ),
            dict(type='RandomCropPadding', clip_len=360, ...),
            dict(type='PrepareM2Mv2Condition', sampler_version='v3', ...),
            dict(type='PackInputs', ...),
        ],
    ),
)
```

### Data Pipeline Sequence
1. **LoadCompatibleCaption** → Loads text captions (fail if missing)
2. **LoadSmplx55** → Loads 135-dim motion
3. **Compute198DimPosition** → FK-derives position channels (135-dim → 198-dim)
4. **SmplTransToKimodoRootOnline** → **[UNIQUE TO E4]** ADMM smoothing + KIMODO conversion
5. **RandomCropPadding** → Crops/pads to 360 frames
6. **PrepareM2Mv2Condition** → Sampler v3 + data corruption
7. **PackInputs** → Final packaging

---

## 4. WHAT CHANGES TO ADD PerMo AND MotionFix DATA

### Step 1: Create Annotation Files for New Datasets

You need to create annotation files for PerMo and MotionFix following the same structure.

**File: `data/annotation/train_permo_hq.json`**
```python
{
  "meta_info": {
    "dataset": "PerMo - train (high quality only)",
    "version": "v1_hq",
    "source": "PerMo dataset",
    "generated_at": "2026-05-14",
    "filtered_count": 12345,  # Your count
  },
  "data_list": {
    "permo_001_action1_0.0_10.0": {
      "subset": "permo",
      "duration": 10.0,
      "num_frames": 300,
      "smplx_path": "../permo_data/raw/permo_001_action1.npz",
      "hierarchical_caption_path": "../permo_data/captions/permo_001_action1.json",
      "fps": 30.0,
      "has_hand": true
    },
    # ... more entries
  }
}
```

**File: `data/annotation/train_motionfix_hq.json`**
```python
{
  "meta_info": {
    "dataset": "MotionFix - train (high quality only)",
    "version": "v1_hq",
    "source": "MotionFix dataset",
    "generated_at": "2026-05-14",
    "filtered_count": 54321,  # Your count
  },
  "data_list": {
    "motionfix_001_walk_0.0_10.0": {
      "subset": "motionfix",
      "duration": 10.0,
      "num_frames": 300,
      "smplx_path": "../motionfix_data/raw/motionfix_001_walk.npz",
      "hierarchical_caption_path": "../motionfix_data/captions/motionfix_001_walk.json",
      "fps": 30.0,
      "has_hand": false
    },
    # ... more entries
  }
}
```

### Step 2: Merge Annotation Files

Create a unified file combining all three datasets:

**File: `data/annotation/train_hymotion_400h_permo_motionfix_merged.json`**
```python
{
  "meta_info": {
    "dataset": "HyMotion 400h + PerMo + MotionFix - train (high quality only)",
    "version": "v1_merged",
    "source": "train_hymotion_400h_hq + permo + motionfix",
    "generated_at": "2026-05-14",
    "original_counts": {
      "hymotion_400h": 407552,
      "permo": 12345,
      "motionfix": 54321,
    },
    "total": 474218,
  },
  "data_list": {
    # ... all 407,552 hymotion entries ...
    "academic_HumanML3D-HumanEva_S1_Static_poses_origintime_0.0_3.0": {...},
    # ... all permo entries ...
    "permo_001_action1_0.0_10.0": {...},
    # ... all motionfix entries ...
    "motionfix_001_walk_0.0_10.0": {...},
  }
}
```

### Step 3: Update Config Files

**Option A: Create new config variants (RECOMMENDED)**

Create `hymotion_m2m_v2_smpl_caption_merged_046b.py`:
```python
_base_ = './_base_hymotion_m2m_v2_046b.py'
work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_merged_E2'

train_dataloader = dict(
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_permo_motionfix_merged.json',
        # ... rest inherited from base ...
    ),
)
```

Create `hymotion_m2m_v2_kimodo_caption_merged_046b.py`:
```python
_base_ = './_base_hymotion_m2m_v2_046b.py'
work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_merged_E4'

load_from = dict(
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    exclude_bundle_keys=['mean', 'std'],
)

model = dict(
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim_kimodo_root',
)

train_dataloader = dict(
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_permo_motionfix_merged.json',
        # ... rest inherited from base ...
    ),
)
```

**Option B: If dataset supports multiple annotation files**
```python
train_dataloader = dict(
    dataset=dict(
        anno_file=[
            'data/annotation/train_hymotion_400h_hq_20260403.json',
            'data/annotation/train_permo_hq.json',
            'data/annotation/train_motionfix_hq.json',
        ],
    ),
)
```

### Step 4: Verify Caption Availability

Ensure all PerMo and MotionFix entries have valid captions:
- **If all have captions**: Use `allow_none=False` (current setting)
- **If some missing**: Change to `allow_none=True` in LoadCompatibleCaption
- **Mixed mode**: Consider filtering/validation before adding to annotation file

Current config uses:
```python
dict(type='LoadCompatibleCaption', allow_none=False),  # Requires all entries to have captions
```

If needed, change to:
```python
dict(type='LoadCompatibleCaption', allow_none=True),   # Allows entries without captions
```

### Step 5: Adjust Batch Size & Resources

**Current settings (caption configs):**
```python
batch_size=20       # Reduced from base config (28)
num_workers=8       # Prefetch workers
persistent_workers=True
```

**With merged dataset (407k + 12k + 54k = 474k entries):**
- Expected 16-20% size increase
- **Recommended**: Keep batch_size=20 (already conservative)
- **Alternative**: If OOM, reduce to batch_size=16 or use gradient_accumulation_steps=2

---

## 5. COMPARISON TABLE: Current vs Merged

| Aspect | Current (400h only) | After Merge (400h+PerMo+MotionFix) |
|--------|---------------------|-----------------------------------|
| Total entries | 407,552 | ~474,218 (estimated) |
| Datasets | 4 (academic, retarget, taobao, game) | 6 (+ permo, motionfix) |
| Annotation file | `train_hymotion_400h_hq_20260403.json` | `train_hymotion_400h_permo_motionfix_merged.json` |
| E2 work_dir | `hymotion_m2m_v2_smpl_caption_E2` | `hymotion_m2m_v2_smpl_caption_merged_E2` |
| E4 work_dir | `hymotion_m2m_v2_kimodo_caption_E4` | `hymotion_m2m_v2_kimodo_caption_merged_E4` |
| Batch size | 20 | 20 (or reduce to 16 if OOM) |
| Num workers | 8 | 8-12 |
| Resume checkpoint | phase2 epoch 3370 | Same checkpoint (if available) |

---

## 6. KEY INTEGRATION POINTS

### A. Annotation File Format (CRITICAL)
Each entry in `data_list` must have:
```python
{
  "subset": "dataset_name",           # Classification
  "duration": float,                   # Motion length (seconds)
  "num_frames": int,                   # Frame count
  "smplx_path": "relative/path.npz",  # SMPL-X 55
  "hierarchical_caption_path": "path.json",  # Caption JSON
  "fps": float,                        # Frame rate
  "has_hand": bool,                    # Hand tracking
}
```

### B. Caption Path Validation
- E2 & E4 use `LoadCompatibleCaption(allow_none=False)` → **ALL entries must have captions**
- Verify `hierarchical_caption_path` files exist before training
- Caption JSON format should match existing hymotion captions

### C. Motion Data Validation
- SMPL-X 55-dim format: [trans(3), body_rot(21×3), hand_rot(2×15×3)]
- After LoadSmplx55: converts to [trans(3), 22joints×rot6d(132)]
- After Compute198DimPosition: adds 63-dim positions (198 total)

### D. Mean/Std Normalization
**E2 Config:**
- Uses `_stats_198dim` (SMPL Root statistics)
- If PerMo/MotionFix have different motion ranges, consider recomputing stats

**E4 Config:**
- Uses `_stats_198dim_kimodo_root` (KIMODO Root statistics)
- After ADMM smoothing, different statistics apply
- **Must exclude_bundle_keys=['mean', 'std']** when loading SMPL checkpoint

### E. Caption Encoding
Both use default text encoder (QWEN3 + CLIP-L):
```python
text_encoder=dict(),  # Default configuration
```

Can customize if needed:
```python
text_encoder=dict(
    type='YourEncoder',
    pretrained='path/to/weights',
    # ... other params ...
)
```

---

## 7. IMPLEMENTATION CHECKLIST

- [ ] Create annotation files for PerMo and MotionFix with metadata
- [ ] Verify all entries have valid `smplx_path` and captions
- [ ] Merge annotations into unified file
- [ ] Place merged annotation at `data/annotation/train_hymotion_400h_permo_motionfix_merged.json`
- [ ] Create new config files for merged dataset
- [ ] Verify SMPL-X files exist and are readable
- [ ] Check caption JSON files are accessible
- [ ] Compute mean/std for merged dataset if different from original
- [ ] Test data loading with small subset (verify no errors)
- [ ] Adjust batch_size if needed based on GPU memory
- [ ] Launch training with `--auto-resume` flag
- [ ] Monitor training logs for data loading issues

---

## 8. APPENDIX: Full Config Files

### File 1: hymotion_m2m_v2_smpl_caption_046b.py
[See Section 3 above for full content]

### File 2: hymotion_m2m_v2_kimodo_caption_046b.py
[See Section 4 above for full content]

### File 3: _base_hymotion_m2m_v2_046b.py
[See Base Config documentation - includes model, optimizer, loss config]

---

**Report Generated:** 2026-05-14
**Project:** HyMotion M2M v2 Caption Training
**Status:** Active Experiments (E2 & E4 running)
