# Checkpoint Paths Report: E2 (SMPL Caption) and E4 (KIMODO Caption)

Generated: 2026-05-15

---

## Executive Summary

This report contains exact checkpoint file paths for two HyMotion M2M v2 training experiments:
- **E2**: SMPL Root rotation baseline with text caption conditioning (Latest: Epoch 330)
- **E4**: KIMODO Root representation with ADMM smoothing + text caption conditioning (Latest: Epoch 270)

Also includes paths to pre-extracted text embedding files used in training.

---

## E2: SMPL Root + Caption Conditioning

### Experiment Details
- **Name**: E2 (SMPL Caption)
- **Purpose**: SMPL root rotation baseline with text caption conditioning (caption_local variant)
- **Key Features**:
  - Rotation space: local (SMPL frame)
  - Text encoder: QWEN3 + CLIP-L
  - CFG (Classifier-Free Guidance): 10% unconditional during training
  - Keypoint supervision weight: 10.0
  - Motion representation: 198-dim (3-dim transl + 22×6 rot6d + 63-dim FK positions)
  - Pred type: velocity

### Configuration
- **Config File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py`
- **Work Directory**: `work_dirs/hymotion_m2m_v2_smpl_caption_E2`
- **Base Config**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- **Initialized From**: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`

### Latest Checkpoint (Epoch 330)

#### Relative Paths
```
work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/model.pt
work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/model.safetensors
work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/optimizer.bin
work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/meta.pt
```

#### Absolute Paths
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/model.pt
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/model.safetensors
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/optimizer.bin
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/meta.pt
```

#### File Sizes
- `model.pt`: 1,845,087,074 bytes (~1.8 GB)
- `model.safetensors`: 1,844,997,584 bytes (~1.8 GB)
- `optimizer.bin`: 3,690,184,570 bytes (~3.7 GB)
- `meta.pt`: 852 bytes

#### Timestamp
- Last Modified: May 15 2026, 16:37 UTC

#### Contains
- 64 distributed process random states (random_states_0.pkl through random_states_63.pkl)
- Distributed training checkpoint metadata (custom_checkpoint_0.pkl)

### All Available Checkpoints
Sorted by epoch (most recent last):
```
checkpoint-epoch_10    checkpoint-epoch_120   checkpoint-epoch_230   checkpoint-epoch_310
checkpoint-epoch_20    checkpoint-epoch_130   checkpoint-epoch_240   checkpoint-epoch_320
checkpoint-epoch_30    checkpoint-epoch_140   checkpoint-epoch_250   checkpoint-epoch_330 ← LATEST
checkpoint-epoch_40    checkpoint-epoch_150   checkpoint-epoch_260
checkpoint-epoch_50    checkpoint-epoch_160   checkpoint-epoch_270
checkpoint-epoch_60    checkpoint-epoch_170   checkpoint-epoch_280
checkpoint-epoch_70    checkpoint-epoch_180   checkpoint-epoch_290
checkpoint-epoch_80    checkpoint-epoch_190   checkpoint-epoch_300
checkpoint-epoch_90    checkpoint-epoch_200
checkpoint-epoch_100   checkpoint-epoch_210
checkpoint-epoch_110   checkpoint-epoch_220
```

---

## E4: KIMODO Root + Caption Conditioning

### Experiment Details
- **Name**: E4 (KIMODO Caption)
- **Purpose**: SMPL motion converted to KIMODO Root representation (with online ADMM smoothing) + text caption conditioning
- **Key Features**:
  - Rotation space: local (SMPL frame)
  - Text encoder: QWEN3 + CLIP-L
  - CFG (Classifier-Free Guidance): 10% unconditional during training
  - Keypoint supervision weight: 10.0
  - Motion representation: 198-dim KIMODO Root
    - [0:3] ADMM smoothed pelvis translation (6cm margin on XZ plane)
    - [3:9] root joint 6D rotation (continuous)
    - [9:135] body (21 non-root joints) 6D rotations
    - [135:198] FK-derived joint positions relative to pelvis
  - Pred type: velocity
  - Timestep squared weighting: True (suppress noisy-FK spikes)

### Configuration
- **Config File**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py`
- **Work Directory**: `work_dirs/hymotion_m2m_v2_kimodo_caption_E4`
- **Base Config**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- **Initialized From**: `work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3370`
- **Mean/Std Directory**: `data/hymotion_m2m_data/_stats_198dim_kimodo_root`
- **ADMM Smoothing Margin**: 0.06m (6cm on horizontal XZ plane)

### Latest Checkpoint (Epoch 270)

#### Relative Paths
```
work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/model.pt
work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/model.safetensors
work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/optimizer.bin
work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/meta.pt
```

#### Absolute Paths
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/model.pt
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/model.safetensors
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/optimizer.bin
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/meta.pt
```

#### File Sizes
- `model.pt`: 1,845,087,074 bytes (~1.8 GB)
- `model.safetensors`: 1,844,997,584 bytes (~1.8 GB)
- `optimizer.bin`: 3,690,184,570 bytes (~3.7 GB)
- `meta.pt`: 852 bytes

#### Timestamp
- Last Modified: May 15 2026, 17:00 UTC

#### Contains
- 64 distributed process random states (random_states_0.pkl through random_states_63.pkl)
- Distributed training checkpoint metadata (custom_checkpoint_0.pkl)

### All Available Checkpoints
Sorted by epoch (most recent last):
```
checkpoint-epoch_10    checkpoint-epoch_110   checkpoint-epoch_200   checkpoint-epoch_250
checkpoint-epoch_20    checkpoint-epoch_120   checkpoint-epoch_210   checkpoint-epoch_260
checkpoint-epoch_30    checkpoint-epoch_130   checkpoint-epoch_220   checkpoint-epoch_270 ← LATEST
checkpoint-epoch_40    checkpoint-epoch_140   checkpoint-epoch_230
checkpoint-epoch_50    checkpoint-epoch_150   checkpoint-epoch_240
checkpoint-epoch_60    checkpoint-epoch_160
checkpoint-epoch_70    checkpoint-epoch_170
checkpoint-epoch_80    checkpoint-epoch_180
checkpoint-epoch_90    checkpoint-epoch_190
checkpoint-epoch_100   checkpoint-epoch_200
```

---

## Pre-Extracted Text Embedding Files

These `.pt` files contain cached text embeddings for caption conditioning used in training.

### Primary Cache

#### Relative Path
```
data/eval/m2m_v2/caption_embeddings/cache.pt
```

#### Absolute Path
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/eval/m2m_v2/caption_embeddings/cache.pt
```

#### Details
- **Size**: 343,440,206 bytes (~328 MB)
- **Last Modified**: May 14 2026, 18:08 UTC
- **Content**: Complete pre-extracted text embeddings cache for all captions
- **Encoder**: QWEN3 (text encoder) + CLIP-L embeddings
- **Used By**: Both E2 and E4 experiments via `LoadPreExtractedTextEmbedding` pipeline component

### Sharded Cache Files (for distributed loading)

#### Directory (Relative)
```
data/eval/m2m_v2/caption_embeddings/shards/
```

#### Directory (Absolute)
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/eval/m2m_v2/caption_embeddings/shards/
```

#### Shard Files
| Shard | Relative Path | Size | Last Modified |
|-------|---------------|------|---------------|
| 0 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_0.pt` | 42 MB | May 14 17:07 |
| 1 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_1.pt` | 42 MB | May 14 17:07 |
| 2 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_2.pt` | 42 MB | May 14 17:07 |
| 3 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_3.pt` | 41 MB | May 14 17:07 |
| 4 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_4.pt` | 41 MB | May 14 18:08 |
| 5 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_5.pt` | 41 MB | May 14 17:07 |
| 6 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_6.pt` | 42 MB | May 14 17:07 |
| 7 | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_7.pt` | 42 MB | May 14 17:07 |

#### Total Size
Approximately 328 MB across 8 shards

### Backup/Historical Embedding File

#### Relative Path
```
data/eval/m2m_v2/caption_embeddings/backup/cache.qwen3_embedding.before_fix.pt
```

#### Absolute Path
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/eval/m2m_v2/caption_embeddings/backup/cache.qwen3_embedding.before_fix.pt
```

#### Details
- **Purpose**: Previous version before QWEN3 embedding fixes (historical reference)
- **Status**: Not used in current experiments

### Embedding Loading in Configs

Both E2 and E4 configs use:
```python
dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True)
```

This loads embeddings from the cache.pt file during training.

---

## Training Data Annotation

Both experiments use the same training annotation file:

### Relative Path
```
data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json
```

### Absolute Path
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json
```

---

## Related Phase Experiments

These configuration files show the progression towards E2/E4:

### Caption Local Phase (Progressive Training)
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase1.py`
  - Intermediate phase 1 checkpoint: epoch 1000+
  
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2.py`
  - Latest intermediate phase: epoch 3370
  - **Used as initialization** for both E2 and E4
  
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py`
  - Variant of local rotation space

### Caption Global Phase
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_phase1.py`
  - Global rotation space variant, phase 1
  
- `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_global_046b.py`
  - Global rotation space variant, final

### SOAR (System of Attention Refinement) Variants
- `configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_caption_local_046b_soar.py`
- `configs/hymotion_m2m_v2/soar/hymotion_m2m_v2_caption_global_046b_soar.py`

---

## Key Differences: E2 vs E4

| Aspect | E2 (SMPL) | E4 (KIMODO) |
|--------|-----------|------------|
| **Root Representation** | SMPL Root (raw) | KIMODO Root (ADMM smoothed) |
| **Translation Smoothing** | None | ADMM smoothing, 6cm margin on XZ plane |
| **Motion Representation** | 198-dim SMPL Root | 198-dim KIMODO Root |
| **Mean/Std Location** | `_stats_198dim` | `_stats_198dim_kimodo_root` |
| **Timestep² Weighting** | False | True |
| **FK Spike Suppression** | No | Yes (via t² weighting) |
| **Latest Checkpoint** | Epoch 330 | Epoch 270 |
| **Use Case** | Standard text-to-motion | Embodied tasks (robot deployment) with consistent root |

---

## Loading Checkpoints

### PyTorch Loading Example
```python
import torch

# Load E2 model
e2_model = torch.load(
    'work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/model.pt',
    map_location='cuda:0',
    weights_only=False
)

# Load E4 model
e4_model = torch.load(
    'work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/model.pt',
    map_location='cuda:0',
    weights_only=False
)

# Load text embeddings
text_embeddings = torch.load(
    'data/eval/m2m_v2/caption_embeddings/cache.pt',
    map_location='cpu'
)
```

### Using HyMotionM2MBundle
```python
from mmengine.config import Config
from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle

# Load E2
cfg_e2 = Config.fromfile('configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py')
bundle_e2 = HyMotionM2MBundle.from_config(cfg_e2.model)
# ... load checkpoint state dict

# Load E4
cfg_e4 = Config.fromfile('configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py')
bundle_e4 = HyMotionM2MBundle.from_config(cfg_e4.model)
# ... load checkpoint state dict with exclude_bundle_keys=['mean', 'std']
```

---

## File Location Summary

| Entity | Type | Relative Path | Size |
|--------|------|---------------|------|
| E2 Model | PyTorch | `work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/model.pt` | 1.8 GB |
| E2 Config | Python | `configs/hymotion_m2m_v2/hymotion_m2m_v2_smpl_caption_046b.py` | ~3 KB |
| E4 Model | PyTorch | `work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/model.pt` | 1.8 GB |
| E4 Config | Python | `configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_caption_046b.py` | ~4 KB |
| Text Embeddings | PyTorch Cache | `data/eval/m2m_v2/caption_embeddings/cache.pt` | 328 MB |
| Embedding Shards (8×) | PyTorch Cache | `data/eval/m2m_v2/caption_embeddings/shards/cache_shard_*.pt` | 328 MB total |
| Training Data | JSON | `data/annotation/train_hymotion_400h_hq_permo_motionfix_editing_20260514.json` | ~2 GB |

---

## Verification

To verify these paths exist:
```bash
# Check E2 checkpoint
ls -lh work_dirs/hymotion_m2m_v2_smpl_caption_E2/checkpoint-epoch_330/model.pt

# Check E4 checkpoint  
ls -lh work_dirs/hymotion_m2m_v2_kimodo_caption_E4/checkpoint-epoch_270/model.pt

# Check text embeddings
ls -lh data/eval/m2m_v2/caption_embeddings/cache.pt

# List all shards
ls -lh data/eval/m2m_v2/caption_embeddings/shards/
```

---

**Report Generated**: 2026-05-15  
**Base Directory**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`
