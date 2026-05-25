# PRISM Dataset `caption_path` - Quick Reference Card

## The Journey of `caption_path`

```
JSON Annotation File
    ↓
Raw value: "../hymotion_data/Academic/.../file.json"
    ↓
Dataset.prepare_data() → os.path.join()
    ↓
"data/motionhub/../hymotion_data/Academic/.../file.json"  ← PASSED TO PIPELINE
    ↓
LoadPreExtractedT5Feature.transform()
    ↓
NORMALIZATION + PATH MAPPING
    ↓
"data/t5_feature/hymotion_data/Academic/.../file.pt"  ← FEATURE FILE LOOKUP
```

---

## Key Facts

| Question | Answer |
|----------|--------|
| **Dataset class?** | `MotionHubSingleAgentTextDataset` |
| **Where is caption_path set?** | `prepare_data()` method (line 37-39) |
| **Is it normalized?** | NO at pipeline entry (contains `..`) |
| **When normalized?** | Inside `LoadPreExtractedT5Feature` (line 245) |
| **What value enters pipeline?** | `"data/motionhub/../hymotion_data/.../*.json"` |
| **What is it mapped to?** | `"data/t5_feature/hymotion_data/.../*.pt"` |
| **What if .pt doesn't exist?** | Returns `None` → triggers dataset refetch |
| **Final output keys?** | `t5_text_embeds`, `t5_text_mask`, `caption` |

---

## File Locations

```
Annotation File:
  → data/annotation/train_hq_motionhub_hymotion.json

Dataset:
  → hftrainer/datasets/motion/motionhub/single_agent_text_dataset.py

Transform:
  → hftrainer/datasets/motion/motionhub/transforms/load_text.py

Configs:
  → configs/prism/prism_1b_tp2m_1frame.py (base)
  → configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py (t5-cached)
```

---

## Code Flow in 4 Steps

### Step 1: Annotation JSON
```json
{
  "hierarchical_caption_path": "../hymotion_data/Academic/.../file.json"
}
```

### Step 2: Dataset.prepare_data() [line 37-39]
```python
caption_path = os.path.join(
    self.data_dir,  # "data/motionhub"
    raw_data_info["hierarchical_caption_path"]  # "../hymotion_data/..."
)
# Result: "data/motionhub/../hymotion_data/..."
```

### Step 3: Pipeline receives results
```python
results = {
    "motion_path": "data/motionhub/../hymotion_data/...",
    "caption_path": "data/motionhub/../hymotion_data/...",
}
```

### Step 4: LoadPreExtractedT5Feature converts [line 278]
```python
pt_path = self._caption_path_to_t5_path(caption_path)
# Converts: "data/motionhub/../hymotion_data/...json"
# To:       "data/t5_feature/hymotion_data/...pt"
```

---

## Path Normalization Details

**Input:**
```
data/motionhub/../hymotion_data/Academic/20250916/human_checked_augmented_caption/file.json
```

**Step by step:**
1. `os.path.normpath()` → `data/hymotion_data/Academic/20250916/human_checked_augmented_caption/file.json`
2. Strip `"data/"` → `hymotion_data/Academic/20250916/human_checked_augmented_caption/file.json`
3. Replace `.json` → `.pt` → `hymotion_data/Academic/20250916/human_checked_augmented_caption/file.pt`
4. Prepend `"data/t5_feature/"` → `data/t5_feature/hymotion_data/Academic/20250916/human_checked_augmented_caption/file.pt`

**Output:**
```
data/t5_feature/hymotion_data/Academic/20250916/human_checked_augmented_caption/file.pt
```

---

## Configuration Override

### Base Config (prism_1b_tp2m_1frame.py)
```python
dataset = dict(
    type="MotionHubSingleAgentTextDataset",
    data_dir="data/motionhub",
    anno_file="data/annotation/train_hq_motionhub_hymotion.json",
    pipeline=[
        dict(type="LoadCompatibleCaption", ...),  # Online encoding
        ...
    ]
)
```

### T5-Cached Override (prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py)
```python
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(
                type='LoadPreExtractedT5Feature',  # ← Replaces LoadCompatibleCaption
                feature_dir='data/t5_feature',
                data_dir='data/motionhub',
                max_seq_length=256,
                allow_none=True,  # ← Enables refetch on missing .pt
            ),
            ...
        ]
    )
)
```

---

## Refetch Behavior

When `allow_none=True` (default in t5cached):

```
LoadPreExtractedT5Feature.transform()
    ↓
pt_path = "data/t5_feature/hymotion_data/.../file.pt"
    ↓
Does pt_path exist?
    ├─ YES → Load embeddings, return results
    └─ NO  → Return None (if allow_none=True)
            ↓
        Dataset catches None
        ↓
        Picks random new sample
        ↓
        Retries (up to max_refetch=100)
```

**Result:** If T5 features aren't pre-extracted yet, dataset will keep fetching different samples until it finds one with pre-extracted features available.

---

## Debug Commands

```bash
# Check annotation file structure
python3 -c "
import json
with open('data/annotation/train_hq_motionhub_hymotion.json') as f:
    data = json.load(f)
sample = list(data['data_list'].values())[0]
print('hierarchical_caption_path:', sample['hierarchical_caption_path'])
"

# Check T5 feature directory structure
ls -la data/t5_feature/hymotion_data/ | head -20

# Count pre-extracted features
find data/t5_feature -name "*.pt" | wc -l

# Check if specific feature exists
ls -l data/t5_feature/hymotion_data/Academic/20250916/human_checked_augmented_caption/ | head
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Many refetches during training | `.pt` files not extracted yet | Run T5 extraction script first |
| File not found errors | Wrong `feature_dir` or `data_dir` config | Check paths in transform config |
| Path doesn't resolve correctly | Relative paths in annotation | Ensure `..` segments in JSON |
| caption_path not in results | Missing from dataset.prepare_data() | Check dataset class inheritance |

---

## Related Transforms

After `LoadPreExtractedT5Feature`:

1. **LoadSmplx55** - Reads `motion_path`, loads motion data
2. **RandomCropPadding** - Crops/pads motion to 360 frames
3. **PackInputs** - Selects final output keys (discards `caption_path`)

---

## Summary

- **What is set?** `results['caption_path']`
- **Where is it set?** `MotionHubSingleAgentTextDataset.prepare_data()` (line 37-39)
- **What value?** `"data/motionhub/../hymotion_data/Academic/.../file.json"` (with literal `..`)
- **Is it resolved?** NO at pipeline entry, YES inside `LoadPreExtractedT5Feature`
- **What does it map to?** `"data/t5_feature/hymotion_data/Academic/.../file.pt"`
- **Why this design?** Allows features to be pre-extracted and cached separately from annotation file

