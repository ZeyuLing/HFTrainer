# HyMotion M2M v2 Caption Configs - Side-by-Side Comparison

## Configuration Parameter Comparison

### Model Configuration

| Parameter | E2 (SMPL) | E4 (KIMODO) | Base Config |
|-----------|-----------|-------------|------------|
| `pred_type` | 'velocity' | 'velocity' | 'velocity' |
| `uncondition_mode` | False | False | **True** |
| `cond_mask_prob` | 0.1 | 0.1 | **0.0** |
| `mean_std_dir` | `_stats_198dim` | `_stats_198dim_kimodo_root` | `_stats_198dim` |
| `rotation_space` | 'local' | 'local' | 'local' |
| `text_encoder` | default | default | default |
| `keypoints3d_weight` | 10.0 | 10.0 | **0.0** |
| `velocity_loss_reduction` | 'component_mean' | 'component_mean' | *N/A* |

---

### Data Pipeline Comparison

| Stage | E2 (SMPL) | E4 (KIMODO) |
|-------|-----------|-------------|
| 1. Captions | LoadCompatibleCaption(allow_none=False) | LoadCompatibleCaption(allow_none=False) |
| 2. Motion Load | LoadSmplx55(rot_type='rotation_6d') | LoadSmplx55(rot_type='rotation_6d') |
| 3. Position | Compute198DimPosition | Compute198DimPosition |
| 4. **Smoothing** | ✗ *None* | ✓ SmplTransToKimodoRootOnline(admm_margin_m=0.06) |
| 5. Cropping | RandomCropPadding(clip_len=360) | RandomCropPadding(clip_len=360) |
| 6. Condition | PrepareM2Mv2Condition(v3) | PrepareM2Mv2Condition(v3) |
| 7. Packaging | PackInputs | PackInputs |

---

### Training Configuration

| Parameter | E2 | E4 | Base |
|-----------|----|----|------|
| `batch_size` | 20 | 20 | 28 |
| `num_workers` | 8 | 8 | 4 |
| `persistent_workers` | True | True | False |
| `shuffle` | *inherited* | *inherited* | True |
| Optimizer | AdamW (1e-4) | AdamW (1e-4) | AdamW (1e-4) |
| Max epochs | *inherited* | *inherited* | 10000 |
| Gradient accumulation | 1 | 1 | 1 |

---

### Checkpoint Loading

| Config | E2 | E4 |
|--------|----|----|
| Load path | `caption_local_phase2/epoch_3370` | `caption_local_phase2/epoch_3370` |
| Load scope | 'model' | 'model' |
| Exclude keys | *none* | `['mean', 'std']` ← **Important!** |
| Null embedding source | `HY-Motion-1.0-Lite/latest.ckpt` | `HY-Motion-1.0-Lite/latest.ckpt` |

---

### Work Directory Setup

| Experiment | Work Dir |
|-----------|----------|
| E2 | `work_dirs/hymotion_m2m_v2_smpl_caption_E2` |
| E4 | `work_dirs/hymotion_m2m_v2_kimodo_caption_E4` |
| **Merged E2** | `work_dirs/hymotion_m2m_v2_smpl_caption_merged_E2` |
| **Merged E4** | `work_dirs/hymotion_m2m_v2_kimodo_caption_merged_E4` |

---

## Motion Representation

### SMPL Root (198-dim) - Used by E2
```
[0:3]       Translation (3)
[3:135]     22 joints × 6D rotations (132) — no smoothing
[135:198]   21 joints × 3D positions (63)
```
**Characteristics:**
- Standard SMPL pelvis translation
- Direct FK-derived positions
- No trajectory smoothing

### KIMODO Root (198-dim) - Used by E4
```
[0:3]       ADMM-smoothed Translation (3) ← Smoothed on XZ plane (6cm margin)
[3:135]     22 joints × 6D rotations (132)
[135:198]   21 joints × 3D positions (63) — derived from smoothed translation
```
**Characteristics:**
- ADMM-smoothed pelvis trajectory (online during __getitem__)
- 6cm margin on XZ plane (horizontal)
- Y-axis unchanged (vertical not smoothed)
- Better consistency for embodied/robot tasks

---

## Data Statistics

### Current Dataset Distribution

```
Total: 407,552 high-quality motions
├── academic          195,168 (47.9%) [HumanML3D, HumanEva]
├── academicretarget  106,820 (26.2%) [Retargeted]
├── taobao             71,009 (17.4%) [Commercial capture]
└── game               34,555 (8.5%)  [Game animations]
```

### With PerMo + MotionFix (Estimated)

```
Total: ~474,218 high-quality motions (+16% increase)
├── academic          195,168 (41.2%)
├── academicretarget  106,820 (22.5%)
├── taobao             71,009 (15.0%)
├── game               34,555 (7.3%)
├── permo            ~12,345 (2.6%)  [Estimated]
└── motionfix        ~54,321 (11.4%) [Estimated]
```

---

## Key Differences Summary

### E2 vs E4

| Aspect | E2 | E4 |
|--------|----|----|
| **Focus** | Baseline SMPL | Robot-ready KIMODO |
| **Smoothing** | None | 6cm XZ margin ADMM |
| **Stats source** | `_stats_198dim` | `_stats_198dim_kimodo_root` |
| **Use case** | General T2M | Embodied/Robot tasks |
| **Complexity** | Lower | Higher |
| **Training stability** | Better (less FK spikes) | Better (smoother trajectories) |

### E2/E4 vs Base Config

| Aspect | Base | E2/E4 |
|--------|------|-------|
| **Text conditioning** | Disabled | **Enabled** |
| **CFG probability** | 0% | **10%** |
| **Keypoint loss** | 0.0 | **10.0** |
| **Batch size** | 28 | 20 (memory for captions) |
| **Caption requirement** | Optional | **Required** |
| **Resume checkpoint** | T2M 1.0 Lite | phase2 epoch 3370 |

---

## Integration Paths

### Path 1: Keep Separate Experiments
```
E2 + E4 (current)
└─ Both use train_hymotion_400h_hq_20260403.json
```

### Path 2: Merge with New Data
```
E2 + E4 (with merged dataset)
├─ E2_merged uses train_hymotion_400h_permo_motionfix_merged.json
└─ E4_merged uses train_hymotion_400h_permo_motionfix_merged.json
```

### Path 3: Mix Single + Merged
```
Current experiments (400h)
├─ E2 (400h only)
├─ E4 (400h only)
├─ E2_merged (400h + new)
└─ E4_merged (400h + new)
```

---

## Code Template: Creating Merged Config

### For E2 (SMPL Caption) with Merged Data

```python
# hymotion_m2m_v2_smpl_caption_merged_046b.py
_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_smpl_caption_merged_E2'

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
    mean_std_dir='data/hymotion_m2m_data/_stats_198dim',
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
        # ← KEY CHANGE: Update annotation file path
        anno_file='data/annotation/train_hymotion_400h_permo_motionfix_merged.json',
    ),
)
```

### For E4 (KIMODO Caption) with Merged Data

```python
# hymotion_m2m_v2_kimodo_caption_merged_046b.py
_base_ = './_base_hymotion_m2m_v2_046b.py'

work_dir = 'work_dirs/hymotion_m2m_v2_kimodo_caption_merged_E4'

load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370',
    load_scope='model',
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    exclude_bundle_keys=['mean', 'std'],  # ← Keep this!
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
        # ← KEY CHANGE: Update annotation file path
        anno_file='data/annotation/train_hymotion_400h_permo_motionfix_merged.json',
    ),
)
```

---

## Launch Commands

### Current (E2 & E4 with 400h)

```bash
# E2
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py 8 --auto-resume

# E4
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py 8 --auto-resume
```

### With Merged Dataset (E2_merged & E4_merged)

```bash
# E2_merged
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_merged_046b.py 8 --auto-resume

# E4_merged
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_merged_046b.py 8 --auto-resume
```

---

**Last Updated:** 2026-05-14
**Scope:** HyMotion M2M v2 Caption Training (E2 & E4)
**Status:** Ready for PerMo + MotionFix integration
