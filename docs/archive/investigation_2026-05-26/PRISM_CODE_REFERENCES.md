# PRISM Dataset Pipeline: Code Reference Guide

This document provides exact code locations and line numbers for understanding how `caption_path` flows through the PRISM dataset pipeline.

---

## 1. Configuration Files

### Primary Config (T5-Cached)
**File:** `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py`

```
Lines 28-70: Override dataset pipeline
  - Replaces LoadCompatibleCaption with LoadPreExtractedT5Feature
  - Sets feature_dir, data_dir, max_seq_length, allow_none, hidden_dim
```

### Base Config (Core Dataset Config)
**File:** `configs/prism/prism_1b_tp2m_1frame.py`

```
Lines 108-148: train_dataloader dict
  - Line 114: type="MotionHubSingleAgentTextDataset"
  - Line 115: motion_key="smplx"
  - Line 116: data_dir="data/motionhub"
  - Line 117: anno_file="data/annotation/train_hq_motionhub_hymotion.json"
  - Line 118-144: pipeline configuration
```

### Intermediate Base Configs
- `configs/prism/prism_1b_tp2m_multiframe.py` (lines 9-14): Extends 1frame, adds multiframe conditioning
- `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py` (lines 12-19): Extends multiframe, changes joint_pos_mode

---

## 2. Dataset Classes

### MotionHubSingleAgentTextDataset
**File:** `hftrainer/datasets/motion/motionhub/single_agent_text_dataset.py`

```python
Line 8-42: Class definition
  Line 14: caption_key: str = "hierarchical_caption"  # Default parameter
  Line 29: self.caption_key = caption_key             # Store parameter
  
Line 31-41: prepare_data() method ⭐ KEY METHOD
  Line 32: raw_data_info = self.data_list[idx]
  Line 33-40: Create data_info dict
    Line 34-36: Set motion_path = os.path.join(data_dir, motion_path)
    Line 37-39: Set caption_path = os.path.join(data_dir, caption_path) ⭐
  Line 41: return data_info
```

**Key Point:**
- Uses `caption_key` parameter to look up `raw_data_info[f"{self.caption_key}_path"]`
- With default `caption_key="hierarchical_caption"`, looks up `raw_data_info["hierarchical_caption_path"]`

### Parent: MotionHubSingleAgentDataset
**File:** `hftrainer/datasets/motion/motionhub/single_agent_dataset.py`

```python
Line 15-58: __init__ method
  Line 31-57: Initialize parent BaseDataset
  Line 59-113: load_data_list() method
    Line 74: annotations = mmengine.load(self.anno_file)
    Line 87: raw_data_list = annotations["data_list"]
    Line 103-109: Iterate and filter raw_data_list

Line 118-186: __getitem__ method
  Line 132: sample = self.prepare_data(idx)  # Calls prepare_data()
  Line 133: sample = self.pipeline(sample)   # Passes to pipeline
  Line 136-159: Handle pipeline None return (refetch logic)

Line 191-198: Default prepare_data() method (overridden by subclass)
```

---

## 3. Annotation File Format

**File:** `data/annotation/train_hq_motionhub_hymotion.json`

Structure:
```json
{
  "meta_info": {...},
  "data_list": {
    "key1": {
      "smplx_path": "../hymotion_data/...",
      "hierarchical_caption_path": "../hymotion_data/...",  // ⭐ This field
      ...
    }
  }
}
```

---

## 4. Transform Pipeline

### LoadPreExtractedT5Feature
**File:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py`

```python
Line 189-333: Class LoadPreExtractedT5Feature ⭐ KEY TRANSFORM

Line 224-236: __init__ method
  Line 226: self.feature_dir = feature_dir      # "data/t5_feature" (from config)
  Line 227: self.data_dir = data_dir            # "data/motionhub" (from config)
  Line 228: self.max_seq_length = max_seq_length
  Line 229: self.allow_none = allow_none        # True in t5cached config

Line 238-269: _caption_path_to_t5_path() method ⭐ PATH CONVERSION
  Line 245: full_path = os.path.normpath(caption_path)
    INPUT:  "data/motionhub/../hymotion_data/..."
    OUTPUT: "data/hymotion_data/..."             (.. resolved)
  
  Line 246: norm_data_dir = os.path.normpath(self.data_dir)  # "data/motionhub"
  Line 247: data_parent = os.path.dirname(norm_data_dir)     # "data"
  
  Line 250-255: Strip data_dir parent prefix
    rel_path starts as: "hymotion_data/..."     (after removing "data/")
  
  Line 261-263: Remove data_dir basename
    No change (motionhub/ already removed in step 2)
  
  Line 266-267: Change extension
    INPUT:  "hymotion_data/.../file.json"
    OUTPUT: "hymotion_data/.../file.pt"
  
  Line 269: return os.path.join(self.feature_dir, rel_path)
    OUTPUT: "data/t5_feature/hymotion_data/.../file.pt"

Line 271-333: transform() method ⭐ MAIN TRANSFORM CALL
  Line 272: caption_path = results.get('caption_path')
    INPUT: "data/motionhub/../hymotion_data/Academic/.../file.json"
  
  Line 273-276: Check if caption_path exists
  
  Line 278: pt_path = self._caption_path_to_t5_path(caption_path)
    Converts to: "data/t5_feature/hymotion_data/Academic/.../file.pt"
  
  Line 280: if not os.path.exists(pt_path):
    Line 281-282: If not exists AND allow_none=True
      return None  # Triggers dataset refetch
  
  Line 288: data = torch.load(pt_path, ...)  # Load .pt file
  
  Line 294-307: Extract embeddings and captions
  
  Line 316-327: Pad embeddings to max_seq_length
  
  Line 326-327: Build attention mask
  
  Line 329-331: Set results keys
    results['t5_text_embeds'] = padded_emb    # [256, 4096] bf16
    results['t5_text_mask'] = mask             # [256] int64
    results['caption'] = caption               # str
  
  Line 333: return results
```

### Supporting Transforms

**LoadSmplx55:**
**File:** `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py`

- Takes `results['motion_path']` (set by prepare_data())
- Sets `results['motion']` = motion tensor

**RandomCropPadding:**
- Crops/pads motion to 360 frames
- Updates `results['num_frames']`

**PackInputs:**
**File:** `hftrainer/datasets/motion/motionhub/transforms/formatting.py`

- Selects keys to keep: motion, num_frames, caption, t5_text_embeds, t5_text_mask
- Discards intermediate keys like caption_path, motion_path (unless in meta_keys)

---

## 5. Data Flow Diagram (with Line Numbers)

```
┌─ Annotation JSON ─────────────────────────────────────────┐
│ File: data/annotation/train_hq_motionhub_hymotion.json    │
│ {hierarchical_caption_path: "../hymotion_data/..."}      │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ Dataset.__init__ ────────────────────────────────────────┐
│ File: single_agent_dataset.py:74                          │
│ load_data_list() → self.data_list loaded                  │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ Dataset.__getitem__ ─────────────────────────────────────┐
│ File: single_agent_dataset.py:132                         │
│ sample = prepare_data(idx)                                │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ TextDataset.prepare_data ────────────────────────────────┐
│ File: single_agent_text_dataset.py:31-41                  │
│ caption_path = os.path.join(data_dir, hierarchical_...path)
│ → "data/motionhub/../hymotion_data/..."                   │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ Pipeline (mmcv.Compose) ────────────────────────────────┐
│ File: single_agent_dataset.py:133                         │
│ sample = self.pipeline(sample)                            │
│ results['caption_path'] = "data/motionhub/../hymotion..." │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ LoadPreExtractedT5Feature.transform ──────────────────────┐
│ File: load_text.py:271-333                                │
│                                                           │
│ Step 1: caption_path = results.get('caption_path')       │
│   = "data/motionhub/../hymotion_data/..."                │
│                                                           │
│ Step 2: pt_path = _caption_path_to_t5_path(caption_path) │
│   = "data/t5_feature/hymotion_data/.../file.pt"          │
│                                                           │
│ Step 3: Load embeddings from pt_path                     │
│   results['t5_text_embeds'] = padded_emb [256, 4096]    │
│   results['t5_text_mask'] = mask [256]                   │
│   results['caption'] = caption_text                      │
│                                                           │
│ Step 4: return results                                   │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ LoadSmplx55, RandomCropPadding, ... ──────────────────────┐
│ Process motion data                                       │
│ Set results['motion'], results['num_frames']              │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ PackInputs ───────────────────────────────────────────────┐
│ File: formatting.py                                       │
│ Select keys: motion, num_frames, caption, t5_text_embeds, │
│              t5_text_mask                                 │
│ (caption_path discarded unless in meta_keys)             │
└─────────────────────────────────────────┬─────────────────┘
                                          │
                                          ▼
┌─ Final Batch Item ────────────────────────────────────────┐
│ {                                                         │
│   motion: [360, 55],                                     │
│   num_frames: int,                                       │
│   caption: str,                                          │
│   t5_text_embeds: [256, 4096] bf16,                     │
│   t5_text_mask: [256] int64,                            │
│   motion_path: str (meta),                              │
│   fps: float (meta)                                      │
│ }                                                         │
└───────────────────────────────────────────────────────────┘
```

---

## 6. Key Configuration Parameters

### In Config File
```python
# Dataset
data_dir="data/motionhub"
anno_file="data/annotation/train_hq_motionhub_hymotion.json"

# Transform
dict(
    type='LoadPreExtractedT5Feature',
    feature_dir='data/t5_feature',      # Where .pt files are stored
    data_dir='data/motionhub',          # For path normalization
    max_seq_length=256,                 # Padding length
    allow_none=True,                    # Enable refetch on missing .pt
    hidden_dim=4096,                    # T5 embedding dimension
)
```

### At Runtime
```python
# Dataset.__init__ parameters
self.data_dir = "data/motionhub"
self.anno_file = "data/annotation/train_hq_motionhub_hymotion.json"
self.caption_key = "hierarchical_caption"  # Default

# Transform.__init__ parameters
self.feature_dir = "data/t5_feature"
self.data_dir = "data/motionhub"
self.max_seq_length = 256
self.allow_none = True
self.hidden_dim = 4096
```

---

## 7. Refetch Logic Code Locations

**Dataset Refetch Handler:**
**File:** `hftrainer/datasets/motion/motionhub/single_agent_dataset.py:136-159`

```python
Line 136: if sample is None:
Line 137-144: Handle pipeline None return
  Line 138: if not self.refetch:
    raise ValueError(...)
  Line 139-144: if _refetch_depth >= self.max_refetch:
    raise error
  Line 145-159: Pick new random idx and retry
```

**Transform None Return:**
**File:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py:273-285`

```python
Line 273: caption_path = results.get('caption_path')
Line 274: if caption_path is None:
  Line 275: if self.allow_none:
    Line 276: return None  # Trigger refetch

Line 280: if not os.path.exists(pt_path):
  Line 281: if self.allow_none:
    Line 282: return None  # Trigger refetch
```

---

## 8. Testing & Debugging

### To Add Debug Logging

**In `MotionHubSingleAgentTextDataset.prepare_data()`:**
```python
def prepare_data(self, idx: int) -> dict:
    raw_data_info = self.data_list[idx]
    data_info = {
        "motion_path": os.path.join(self.data_dir, raw_data_info[f"{self.motion_key}_path"]),
        "caption_path": os.path.join(self.data_dir, raw_data_info[f"{self.caption_key}_path"]),
    }
    print(f"DEBUG prepare_data[{idx}]:")
    print(f"  raw annotation path: {raw_data_info[f'{self.caption_key}_path']}")
    print(f"  resulting caption_path: {data_info['caption_path']}")
    return data_info
```

**In `LoadPreExtractedT5Feature.transform()`:**
```python
def transform(self, results: Dict) -> Optional[Dict]:
    caption_path = results.get('caption_path')
    pt_path = self._caption_path_to_t5_path(caption_path)
    print(f"DEBUG LoadPreExtractedT5Feature:")
    print(f"  input caption_path: {caption_path}")
    print(f"  computed pt_path: {pt_path}")
    print(f"  exists: {os.path.exists(pt_path)}")
    if not os.path.exists(pt_path):
        if self.allow_none:
            print(f"  returning None (trigger refetch)")
            return None
    # ... rest of method
```

