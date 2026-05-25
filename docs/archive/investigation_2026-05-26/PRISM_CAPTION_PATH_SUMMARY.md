# PRISM Dataset Pipeline: `results['caption_path']` Complete Trace

## TL;DR

**Question:** How is `results['caption_path']` set before it reaches `LoadPreExtractedT5Feature`?

**Answer:** 
1. **Raw annotation value** (from JSON): `../hymotion_data/Academic/.../file.json` (relative path)
2. **After `prepare_data()`**: `data/motionhub/../hymotion_data/Academic/.../file.json` (joined but NOT normalized - still contains `..`)
3. **Passed to pipeline**: Same as (2) - contains literal `..` segments
4. **Inside `LoadPreExtractedT5Feature`**: Gets normalized to `data/hymotion_data/...`, then mapped to T5 feature file `data/t5_feature/hymotion_data/.../file.pt`

---

## Detailed Flow

### 1. Annotation File Structure

**File:** `data/annotation/train_hq_motionhub_hymotion.json`

**Entry Example:**
```json
{
  "subset": "academic",
  "duration": 9.167,
  "num_frames": 275,
  "smplx_path": "../hymotion_data/Academic/20250916/motions/.../S3_Box_1_poses_origintime_13.55_22.7.npz",
  "hierarchical_caption_path": "../hymotion_data/Academic/20250916/human_checked_augmented_caption/.../S3_Box_1_poses_origintime_13.55_22.7.json",
  "fps": 30.0,
  "has_hand": true
}
```

**Key Field:** `hierarchical_caption_path: "../hymotion_data/..."` 
- This is a **relative path** that escapes the `data/motionhub/` directory using `../`
- Points to data outside motionhub (shared with other projects)

---

### 2. Dataset Class: `MotionHubSingleAgentTextDataset`

**File:** `hftrainer/datasets/motion/motionhub/single_agent_text_dataset.py`

**Configuration in Base Config:**
- Dataset type: `MotionHubSingleAgentTextDataset`
- `data_dir: "data/motionhub"`
- `motion_key: "smplx"`
- `caption_key: "hierarchical_caption"` (default)

**Key Method - `prepare_data()`:**
```python
def prepare_data(self, idx: int) -> dict:
    raw_data_info = self.data_list[idx]
    data_info = {
        "motion_path": os.path.join(
            self.data_dir,  # "data/motionhub"
            raw_data_info[f"{self.motion_key}_path"]  # "../hymotion_data/..."
        ),
        "caption_path": os.path.join(
            self.data_dir,  # "data/motionhub"
            raw_data_info[f"{self.caption_key}_path"]  # "../hymotion_data/..."
        ),
    }
    return data_info
```

**Critical Details:**
- The `caption_key` defaults to `"hierarchical_caption"`, so it looks up `raw_data_info["hierarchical_caption_path"]`
- **No normalization** happens here - the path is joined but retains `..` segments
- `os.path.join("data/motionhub", "../hymotion_data/...")` → `"data/motionhub/../hymotion_data/..."`

---

### 3. What Gets Passed to the Pipeline

**At dataset `__getitem__` call:**
```python
results = self.prepare_data(idx)
# results['caption_path'] = "data/motionhub/../hymotion_data/Academic/.../file.json"
# ↑ CONTAINS LITERAL ".." IN THE STRING
```

**Then passed to pipeline:**
```python
results = self.pipeline(results)
# First transform receives:
# results['caption_path'] = "data/motionhub/../hymotion_data/Academic/.../file.json"
```

---

### 4. LoadPreExtractedT5Feature Transform Processing

**File:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py` (lines 271-333)

**Transform Method:**
```python
def transform(self, results: Dict) -> Optional[Dict]:
    caption_path = results.get('caption_path')
    # INPUT: "data/motionhub/../hymotion_data/Academic/.../file.json"
    
    pt_path = self._caption_path_to_t5_path(caption_path)
    # This method converts the caption path to a T5 feature path
    
    if not os.path.exists(pt_path):
        if self.allow_none:
            return None  # Trigger refetch
        raise FileNotFoundError(...)
    
    # Load embeddings and set results keys...
    return results
```

**The `_caption_path_to_t5_path()` Conversion (lines 238-269):**

```python
def _caption_path_to_t5_path(self, caption_path: str) -> str:
    # INPUT: "data/motionhub/../hymotion_data/Academic/.../file.json"
    
    # Step 1: Normalize path (resolve ..)
    full_path = os.path.normpath(caption_path)
    # → "data/hymotion_data/Academic/.../file.json"
    
    norm_data_dir = os.path.normpath(self.data_dir)  # "data/motionhub"
    data_parent = os.path.dirname(norm_data_dir)     # "data"
    
    # Step 2: Strip data_dir parent prefix
    if full_path.startswith(data_parent + '/'):
        rel_path = full_path[len(data_parent) + 1:]
        # → "hymotion_data/Academic/.../file.json"
    
    # Step 3: Remove data_dir basename prefix (if present)
    data_dir_basename = os.path.basename(norm_data_dir)  # "motionhub"
    if rel_path.startswith(data_dir_basename + '/'):
        rel_path = rel_path[len(data_dir_basename) + 1:]
        # → No change (already removed in step 2)
    
    # Step 4: Change extension .json → .pt
    if rel_path.endswith('.json'):
        rel_path = rel_path[:-5] + '.pt'
        # → "hymotion_data/Academic/.../file.pt"
    
    # Step 5: Prepend feature_dir
    return os.path.join(self.feature_dir, rel_path)
    # → "data/t5_feature/hymotion_data/Academic/.../file.pt"
```

**Final T5 Feature Path:**
```
data/t5_feature/hymotion_data/Academic/20250916/human_checked_augmented_caption/.../S3_Box_1_poses_origintime_13.55_22.7.pt
```

**If file exists:** Load embeddings, set `t5_text_embeds`, `t5_text_mask`, `caption`

**If file NOT exists AND `allow_none=True`:** Return `None` → triggers refetch

---

### 5. Path Transformation Summary Table

| Stage | Value | Has `..`? | Normalized? | Notes |
|-------|-------|----------|------------|-------|
| **Annotation JSON** | `../hymotion_data/.../file.json` | YES | NO | Raw relative path |
| **After `os.path.join()` in `prepare_data()`** | `data/motionhub/../hymotion_data/.../file.json` | YES | NO | Joined but NOT normalized |
| **Entered Pipeline** | `data/motionhub/../hymotion_data/.../file.json` | YES | NO | Passed to transforms as-is |
| **After `normpath()` in `_caption_path_to_t5_path()`** | `data/hymotion_data/.../file.json` | NO | YES | Internal to transform |
| **After stripping prefixes** | `hymotion_data/.../file.json` | NO | YES | Ready for feature_dir |
| **Final T5 path** | `data/t5_feature/hymotion_data/.../file.pt` | NO | YES | Feature file lookup path |

---

## Key Architectural Insights

### 1. **Lazy Normalization Pattern**
- Paths are **NOT normalized early** in the dataset pipeline
- Normalization happens **inside the transform** that needs it
- This allows different transforms to interpret paths differently if needed

### 2. **Feature Path Mapping Strategy**
- The T5 feature extraction expects to find .pt files in a specific structure
- Structure mirrors the caption file structure: `data/t5_feature/` + relative caption path with .pt extension
- This design allows features to be cached and reused without re-extraction

### 3. **Refetch Safety (when `allow_none=True`)**
- If T5 pre-extracted features don't exist yet (file not extracted or extraction incomplete):
  - Transform returns `None`
  - Dataset catches this and automatically refetches a different sample
  - Prevents training from stalling on missing features
  - Useful during parallel pre-extraction workflows

### 4. **Relative Path Design**
- Annotation uses relative paths (`../hymotion_data/...`) to reference shared data
- This allows the dataset to be moved or symlinked without breaking paths
- Works as long as directory structure is preserved

---

## Configuration Details

### Base Config
**File:** `configs/prism/prism_1b_tp2m_1frame.py` (lines 108-148)

```python
train_dataloader = dict(
    dataset=dict(
        type="MotionHubSingleAgentTextDataset",
        data_dir="data/motionhub",
        anno_file="data/annotation/train_hq_motionhub_hymotion.json",
        pipeline=[
            dict(type="LoadCompatibleCaption", allow_none=False),  # For non-cached training
            ...
        ],
    ),
)
```

### T5-Cached Override
**File:** `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py` (lines 28-70)

```python
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(
                type='LoadPreExtractedT5Feature',  # Replaces LoadCompatibleCaption
                feature_dir='data/t5_feature',
                data_dir='data/motionhub',
                max_seq_length=256,
                allow_none=True,  # ← Enables refetch on missing .pt
                hidden_dim=4096,
            ),
            dict(type="LoadSmplx55", ...),
            dict(type="RandomCropPadding", ...),
            dict(type="PackInputs", keys=['motion', 'num_frames', 'caption', 't5_text_embeds', 't5_text_mask'], ...),
        ],
    ),
)
```

---

## Debugging Checklist

To verify this behavior during training:

- [ ] **Check dataset logs:** Should show "Loaded X samples from train_hq_motionhub_hymotion.json"
- [ ] **Verify annotation file:** Contains `hierarchical_caption_path` with `../hymotion_data/...` values
- [ ] **Check T5 feature directory:** `data/t5_feature/hymotion_data/` should exist and contain .pt files
- [ ] **Monitor refetch:** If many refetches occur, check if .pt files are being extracted to the right location
- [ ] **Test path mapping:** Run `_caption_path_to_t5_path()` on a sample annotation path to verify output location

---

## References

**Source Files:**
- Dataset: `hftrainer/datasets/motion/motionhub/single_agent_text_dataset.py`
- Parent dataset: `hftrainer/datasets/motion/motionhub/single_agent_dataset.py`
- Transform: `hftrainer/datasets/motion/motionhub/transforms/load_text.py` (lines 189-333)
- Configs: `configs/prism/prism_1b_tp2m_*.py`
- Annotation: `data/annotation/train_hq_motionhub_hymotion.json`

