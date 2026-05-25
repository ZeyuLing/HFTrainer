# HyMotion M2M v2 Caption Configs - Quick Summary

## 📊 Current Setup

### Two Active Experiments
| Experiment | Config File | Representation | Status |
|-----------|---------|-----------------|--------|
| **E2** | `hymotion_m2m_v2_smpl_caption_046b.py` | SMPL Root (198-dim) | Active |
| **E4** | `hymotion_m2m_v2_kimodo_caption_046b.py` | KIMODO Root (198-dim, ADMM smoothed) | Active |

### Current Dataset
**File:** `data/annotation/train_hymotion_400h_hq_20260403.json`
- **Total Entries:** 407,552 (high-quality)
- **Composition:**
  - academic: 195,168 (47.9%)
  - academicretarget: 106,820 (26.2%)
  - taobao: 71,009 (17.4%)
  - game: 34,555 (8.5%)

---

## 🔑 Key Configuration Differences

### E2 (SMPL Caption)
```python
mean_std_dir = 'data/hymotion_m2m_data/_stats_198dim'
pipeline: [
  LoadCompatibleCaption(allow_none=False),
  LoadSmplx55,
  Compute198DimPosition,
  RandomCropPadding(clip_len=360),
  PrepareM2Mv2Condition(v3),
  PackInputs,
]
```

### E4 (KIMODO Caption) 
```python
mean_std_dir = 'data/hymotion_m2m_data/_stats_198dim_kimodo_root'
load_from: exclude_bundle_keys=['mean', 'std']  # Important!
pipeline: [
  LoadCompatibleCaption(allow_none=False),
  LoadSmplx55,
  Compute198DimPosition,
  SmplTransToKimodoRootOnline(admm_margin_m=0.06),  # ← NEW
  RandomCropPadding(clip_len=360),
  PrepareM2Mv2Condition(v3),
  PackInputs,
]
```

---

## ✅ Annotation Entry Structure

Every motion in the annotation file needs:
```json
{
  "subset": "dataset_name",
  "duration": 10.0,
  "num_frames": 300,
  "smplx_path": "../path/to/motion.npz",
  "hierarchical_caption_path": "../path/to/caption.json",
  "fps": 30.0,
  "has_hand": true
}
```

---

## 🚀 To Add PerMo and MotionFix Data

### 1. Create Annotation Files
Create `train_permo_hq.json` and `train_motionfix_hq.json` with same structure.

### 2. Merge Annotations
Create `train_hymotion_400h_permo_motionfix_merged.json` combining all three.

### 3. Update Config
Create new configs that point to merged annotation file:
```python
# hymotion_m2m_v2_smpl_caption_merged_046b.py
_base_ = './_base_hymotion_m2m_v2_046b.py'
work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_merged_E2'

train_dataloader = dict(
    dataset=dict(
        anno_file='data/annotation/train_hymotion_400h_permo_motionfix_merged.json',
    ),
)
```

### 4. Verify Captions
- All entries must have valid `hierarchical_caption_path`
- Caption JSON files must exist and be accessible
- Currently set to `allow_none=False` → requires captions for all entries

### 5. Adjust Resources
- Current: `batch_size=20, num_workers=8`
- With merged data: ~474k entries (16-20% increase)
- Recommendation: Keep batch_size=20, or reduce to 16 if OOM

---

## 📁 File Locations

**Configs:**
- E2: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py`
- E4: `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py`
- Base: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`

**Data:**
- Current: `data/annotation/train_hymotion_400h_hq_20260403.json`
- Future: `data/annotation/train_hymotion_400h_permo_motionfix_merged.json`

**Checkpoints:**
- Resume from: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`

---

## ⚙️ Common Parameters (Both E2 & E4)

```python
model = dict(
    pred_type='velocity',
    uncondition_mode=False,      # Text conditioning enabled
    cond_mask_prob=0.1,          # 10% CFG probability
    rotation_space='local',
    text_encoder=dict(),          # Default QWEN3 + CLIP-L
    losses_cfg=dict(
        keypoints3d_weight=10.0,
        velocity_loss_reduction='component_mean',
    ),
)

train_dataloader = dict(
    batch_size=20,
    num_workers=8,
    persistent_workers=True,
)
```

---

## 📋 Critical Points

1. **Caption Requirement:** Both E2 and E4 need captions for ALL entries
2. **Motion Format:** SMPL-X 55-dim NPZ files required
3. **Stats Handling:** E4 uses different stats (`_stats_198dim_kimodo_root`) + must exclude mean/std when loading checkpoint
4. **ADMM Smoothing:** E4 only — 6cm margin on XZ plane, Y-axis unsmoothed
5. **Checkpoint Resume:** Both start from phase2 epoch 3370

---

## 🔍 For Full Details
See `CAPTION_CONFIG_ANALYSIS_REPORT.md` for complete configuration files and step-by-step integration guide.

