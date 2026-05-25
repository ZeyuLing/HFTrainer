# PRISM Training Dataset: `results['caption_path']` Trace

## Overview
This document traces how `results['caption_path']` is set before it reaches the `LoadPreExtractedT5Feature` transform in the PRISM training pipeline.

**Config Chain:**
```
prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py
  ↓ (_base_)
prism_1b_tp2m_multiframe_kt_spectral_unified.py
  ↓ (_base_)
prism_1b_tp2m_multiframe.py
  ↓ (_base_)
prism_1b_tp2m_1frame.py
```

## 1. Dataset Configuration

### Location
`configs/prism/prism_1b_tp2m_1frame.py` (lines 108-148)

### Configuration
```python
train_dataloader = dict(
    batch_size=6,
    num_workers=8,
    persistent_workers=False,
    shuffle=True,
    dataset=dict(
        type="MotionHubSingleAgentTextDataset",
        motion_key="smplx",
        data_dir="data/motionhub",
        anno_file="data/annotation/train_hq_motionhub_hymotion.json",
        pipeline=[
            dict(type="LoadCompatibleCaption", allow_none=False),  # NOT used in t5cached config
            dict(type="LoadSmplx55", ...),
            dict(type="RandomCropPadding", ...),
            dict(type="PackInputs", ...),
        ],
    ),
)
```

**For T5-cached training**, the pipeline is replaced in `prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py`:
```python
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(
                type='LoadPreExtractedT5Feature',  # <-- NEW: replaces LoadCompatibleCaption
                feature_dir='data/t5_feature',
                data_dir='data/motionhub',
                max_seq_length=256,
                allow_none=True,
                hidden_dim=4096,
            ),
            dict(type="LoadSmplx55", ...),
            dict(type="RandomCropPadding", ...),
            dict(type="PackInputs", ...),
        ],
    ),
)
```

## 2. Dataset Class: `MotionHubSingleAgentTextDataset`

### Location
`hftrainer/datasets/motion/motionhub/single_agent_text_dataset.py`

### Code
```python
@DATASETS.register_module()
class MotionHubSingleAgentTextDataset(MotionHubSingleAgentDataset):

    def __init__(
        self,
        motion_key: str = "smplx",
        caption_key: str = "hierarchical_caption",  # <-- Key mapping
        data_dir: str = "data/motionhub",
        anno_file: str = "data/motionhub/train.json",
        pipeline: Union[Dict, Any, List[Union[Dict, Any]]] = None,
        refetch=False,
        verbose: bool = True,
    ):
        super().__init__(
            motion_key=motion_key,
            data_dir=data_dir,
            anno_file=anno_file,
            pipeline=pipeline,
            refetch=refetch,
            verbose=verbose,
        )
        self.caption_key = caption_key

    def prepare_data(self, idx: int) -> dict:
        """Convert raw annotation entry to results dict for pipeline.
        
        KEY FUNCTION: This is where caption_path is first set.
        """
        raw_data_info = self.data_list[idx]  # Loaded from anno_file
        data_info = {
            "motion_path": os.path.join(
                self.data_dir, raw_data_info[f"{self.motion_key}_path"]
            ),
            "caption_path": os.path.join(
                self.data_dir, raw_data_info[f"{self.caption_key}_path"]
            ),  # ← Sets caption_path here
        }
        return data_info
```

### Key Point
The `caption_key` parameter defaults to `"hierarchical_caption"`, which means:
- Config looks up: `raw_data_info["hierarchical_caption_path"]`
- **NOT** `raw_data_info.get("hierarchical_caption")`

This is crucial because the annotation file contains `"hierarchical_caption_path"`, not `"hierarchical_caption"`.

## 3. Annotation File Format

### Location
`data/annotation/train_hq_motionhub_hymotion.json`

### Structure
```json
{
  "meta_info": {
    "dataset": "train_hq_motionhub_hymotion (merged)",
    "version": "v1",
    "sources": [...]
  },
  "data_list": {
    "key1": {
      "subset": "academic",
      "duration": 9.167,
      "num_frames": 275,
      "smplx_path": "../hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.npz",
      "hierarchical_caption_path": "../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json",
      "fps": 30.0,
      "has_hand": true
    },
    ...
  }
}
```

### Raw Path Value
```
"hierarchical_caption_path": "../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json"
```

This is a **relative path** that uses `../` to escape the `data/motionhub/` directory.

## 4. Path Resolution in `prepare_data()`

### Raw Annotation Value
```
../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json
```

### Inside `prepare_data()`
```python
caption_path = os.path.join(
    self.data_dir,  # "data/motionhub"
    raw_data_info["hierarchical_caption_path"]  # "../hymotion_data/..."
)
```

### Result
```
data/motionhub/../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json
```

**Important:** This is **NOT normalized** at this point. The path contains `..` segments.

### What Gets Passed to Pipeline
The `results` dict passed to the first transform contains:
```python
{
    "motion_path": "data/motionhub/../hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.npz",
    "caption_path": "data/motionhub/../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json",
}
```

## 5. Path Normalization in `LoadPreExtractedT5Feature`

### Location
`hftrainer/datasets/motion/motionhub/transforms/load_text.py` (lines 238-269)

### The Transform Method
```python
def _caption_path_to_t5_path(self, caption_path: str) -> str:
    """Map caption_path to the corresponding T5 feature .pt path."""
    
    # Step 1: Normalize the path (resolves ..)
    full_path = os.path.normpath(caption_path)
    # Result: "data/hymotion_data/Academic/20250916/human_checked_augmented_caption/..."
    
    norm_data_dir = os.path.normpath(self.data_dir)  # "data/motionhub"
    data_parent = os.path.dirname(norm_data_dir)  # "data"
    
    # Step 2: Strip data_dir parent prefix
    if full_path.startswith(data_parent + '/'):
        rel_path = full_path[len(data_parent) + 1:]
        # Result: "hymotion_data/Academic/20250916/human_checked_augmented_caption/..."
    
    # Step 3: Remove data_dir basename prefix
    data_dir_basename = os.path.basename(norm_data_dir)  # "motionhub"
    if rel_path.startswith(data_dir_basename + '/'):
        rel_path = rel_path[len(data_dir_basename) + 1:]
    
    # Step 4: Change extension .json -> .pt
    if rel_path.endswith('.json'):
        rel_path = rel_path[:-5] + '.pt'
        # Result: "hymotion_data/Academic/20250916/human_checked_augmented_caption/.../S3_Box_1_poses_origintime_13.55_22.7.pt"
    
    # Step 5: Prepend feature_dir
    return os.path.join(self.feature_dir, rel_path)
    # Result: "data/t5_feature/hymotion_data/Academic/20250916/human_checked_augmented_caption/.../S3_Box_1_poses_origintime_13.55_22.7.pt"
```

### Transform Call
```python
def transform(self, results: Dict) -> Optional[Dict]:
    caption_path = results.get('caption_path')
    if caption_path is None:
        if self.allow_none:
            return None  # Trigger refetch
        raise ValueError("LoadPreExtractedT5Feature: 'caption_path' not in results")
    
    pt_path = self._caption_path_to_t5_path(caption_path)  # Path conversion
    
    if not os.path.exists(pt_path):
        if self.allow_none:
            return None  # Trigger refetch — .pt not yet extracted
        raise FileNotFoundError(f"LoadPreExtractedT5Feature: {pt_path} does not exist")
    
    # Load and process embeddings...
    # Sets results['t5_text_embeds'], results['t5_text_mask'], results['caption']
    return results
```

## Summary: Path Transformation

| Stage | Value | Note |
|-------|-------|------|
| **Annotation JSON** | `../hymotion_data/Academic/.../S3_Box_1_poses_origintime_13.55_22.7.json` | Raw relative path with `..` |
| **prepare_data()** | `data/motionhub/../hymotion_data/Academic/.../S3_Box_1_poses_origintime_13.55_22.7.json` | Joined but NOT normalized |
| **Passed to Pipeline** | Same as above | `results['caption_path']` before any transform |
| **After normpath in LoadPreExtractedT5Feature** | `data/hymotion_data/Academic/.../S3_Box_1_poses_origintime_13.55_22.7.json` | `..` segments resolved |
| **After stripping prefixes** | `hymotion_data/Academic/.../S3_Box_1_poses_origintime_13.55_22.7.json` | Ready for feature_dir |
| **Final T5 path** | `data/t5_feature/hymotion_data/Academic/.../S3_Box_1_poses_origintime_13.55_22.7.pt` | Full path to extracted embedding |

## Key Findings

### 1. **No Normalization in `prepare_data()`**
The dataset class does NOT normalize the path after joining. It passes the raw `os.path.join()` result:
```
data/motionhub/../hymotion_data/...
```

### 2. **Normalization Happens in Transform**
The `LoadPreExtractedT5Feature.transform()` method normalizes the path:
```python
full_path = os.path.normpath(caption_path)
# Converts: data/motionhub/../hymotion_data/... → data/hymotion_data/...
```

### 3. **Relative Paths in Annotation**
The annotation file uses relative paths (`../hymotion_data/...`) to reference data outside the `data/motionhub/` directory.

### 4. **Path-based Feature Lookup**
The T5 pre-extracted features are stored using the caption file's relative path as the key:
```
data/t5_feature/hymotion_data/Academic/20250916/human_checked_augmented_caption/.../S3_Box_1_poses_origintime_13.55_22.7.pt
```

### 5. **Error Handling**
If `allow_none=True` (default in T5-cached config) and the .pt file doesn't exist:
- Transform returns `None`
- Dataset triggers refetch (tries another random sample)
- This prevents training stalls when pre-extraction is incomplete

## Configuration for Debugging

To verify this behavior during training, check:
1. **Dataset initialization logs** (verbose=True):
   - Should show "Loaded X samples from data/annotation/train_hq_motionhub_hymotion.json"

2. **First few batches' caption_path values**:
   - Should contain `data/motionhub/../hymotion_data/...` (with `..`)

3. **T5 feature loading logs** (if available):
   - Should show converted .pt paths like `data/t5_feature/hymotion_data/.../...pt`

4. **Feature file existence**:
   - Check if `data/t5_feature/hymotion_data/...` directory exists and contains .pt files
   - If not, training will keep refetching until a valid sample is found or max_refetch is exceeded
