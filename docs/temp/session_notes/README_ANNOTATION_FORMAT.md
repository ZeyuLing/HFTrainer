# HyMotion Annotation Format Documentation

This directory contains comprehensive documentation of the HyMotion dataset annotation JSON format used by the training pipeline.

## 📚 Documentation Files

### Quick Start (5 min read)
**→ [`HYMOTION_ANNOTATION_QUICK_REFERENCE.md`](HYMOTION_ANNOTATION_QUICK_REFERENCE.md)**
- One-page quick reference guide
- Field summary tables
- Code file locations
- Expected data structures after loading

### Complete Specification (20 min read)
**→ [`HYMOTION_ANNOTATION_FORMAT.md`](HYMOTION_ANNOTATION_FORMAT.md)**
- Detailed field-by-field specification
- Two caption file formats (HyMotion & Legacy)
- Data flow through loading pipeline
- Memory structures at each stage
- Embedding path mapping
- Caption extraction logic

### Code Examples (15 min read)
**→ [`HYMOTION_ANNOTATION_EXAMPLES.md`](HYMOTION_ANNOTATION_EXAMPLES.md)**
- 9 concrete JSON examples
- Main annotation file structure
- HyMotion caption files (single & multiple entries)
- Legacy hierarchical format
- Multi-person motion entries
- Full loading pipeline outputs

### Executive Summary (3 min read)
**→ [`ANNOTATION_FORMAT_SUMMARY.txt`](ANNOTATION_FORMAT_SUMMARY.txt)**
- Key findings summary
- Critical implementation details
- All important information in one place

---

## 🎯 At a Glance

### Main Annotation File Structure
```json
{
    "meta_info": {
        "dataset": "hymotion_data - train",
        "version": "v1"
    },
    "data_list": {
        "entry_key_1": {
            "smplx_path": "path/to/motion.npz",  // or list for multi-person
            "num_frames": 275,
            "fps": 30.0,
            "duration": 9.167,
            "has_hand": true,
            "subset": "academic",
            "hierarchical_caption_path": "path/to/caption.json"
            // ... optional audio/speech/metadata fields
        }
    }
}
```

### Hierarchical Caption File (HyMotion Format)
```json
{
    "result": [
        {
            "short_caption": "a person walks forward slowly",
            "short_caption_rewritten": [
                "a person walks forward slowly",
                "a person walks forward in a slow manner"
            ]
        }
    ]
}
```

---

## 🔍 Key Insights

### Required Fields in data_list Entries
- `smplx_path` (string or list) — Motion file path
- `num_frames` (integer) — Total frames
- `fps` (float) — Frames per second
- `duration` (float) — Duration in seconds
- `has_hand` (boolean) — Hand motion included?
- `subset` (string) — Data subset name
- **At least ONE caption field** (see below)

### Caption Fields (Fallback Chain)
1. **`hierarchical_caption_path`** (PREFERRED) — Path to caption JSON file
2. **`caption_path`** (FALLBACK) — Alternative caption path
3. **`caption`** (FALLBACK) — Direct caption string or list

### Optional Fields
- Audio/Music: `music_path`, `genre`, `sr`
- Speech: `audio_path`, `speech_script_path`, `speaker_id`
- Metadata: `language`

---

## 📖 Data Flow

```
1. Load main JSON (train_hymotion_400h.json)
   ├─ Extract meta_info
   └─ Extract data_list (dictionary of entries)

2. For each entry:
   ├─ Resolve relative paths to absolute paths
   ├─ Load caption from hierarchical_caption_path
   ├─ Parse caption JSON (auto-detect format)
   ├─ Extract caption strings + variants
   ├─ Randomly select one caption
   └─ Load pre-extracted embeddings (if .pt exists)

3. Output per sample contains:
   ├─ caption: str (selected caption)
   ├─ caption_list: list[str] (all captions)
   ├─ text_vec_raw: Tensor[768] (CLIP-L, if .pt loaded)
   ├─ text_ctxt_raw: Tensor[seq, 4096] (Qwen3, if .pt loaded)
   └─ ... (motion, fps, duration, etc.)
```

---

## 🛠️ Code Locations

| Component | File |
|-----------|------|
| Base dataset loader | `hftrainer/datasets/motion/motionhub/single_agent_dataset.py` |
| Main dataset class | `hftrainer/datasets/motion/motionhub/multitask_multiagent_dataset.py` |
| Caption loading transforms | `hftrainer/datasets/motion/motionhub/transforms/load_text.py` |

### Key Classes in load_text.py
- **`LoadCompatibleCaption`** — Auto-detects HyMotion vs Legacy format
- **`LoadHYMotionCaption`** — Handles HyMotion format (result array)
- **`LoadHierarchicalCaption`** — Handles Legacy format (macro/meso/micro)
- **`LoadPreExtractedTextEmbedding`** — Loads pre-computed embeddings from .pt files

---

## ⚠️ Critical Implementation Details

### 1. data_list is a DICTIONARY, not an array
```python
# Iteration pattern
for data in data_list.values():
    process(data)
```

### 2. Caption Format Auto-Detection
```python
if "result" in data:              # HyMotion format
elif "macro" in data and ...      # Legacy format
else:                              # Error
```

### 3. HyMotion Caption Extraction
- Prefer `short_caption_rewritten` (list of variants)
- Fallback to `short_caption` (single string)
- Accepts both `"short_caption"` and `"short caption"` (underscore or space)

### 4. Pre-Extracted Embeddings Mapping
```python
CAPTION_TO_QWEN3_DIR = {
    'human_checked_augmented_caption': 'qwen3_augmented',
    'human_checked_caption': 'qwen3_human_checked_short',
    'improved_simple_augmented_caption': 'qwen3_improved_simple_short',
    # ...
}
```

### 5. Multi-Person Support
```python
# Single-person
"smplx_path": "path/to/motion.npz"        # num_person = 1

# Multi-person
"smplx_path": [                            # num_person = len(list)
    "path/to/person0/motion.npz",
    "path/to/person1/motion.npz"
]
```

---

## 📋 Two Caption File Formats

### Format 1: HyMotion (Recommended)
```json
{
    "result": [
        {
            "short_caption": "...",
            "short_caption_rewritten": ["...", "..."]
        }
    ]
}
```
**Used for**: train_hymotion_400h.json dataset

### Format 2: Legacy Hierarchical
```json
{
    "macro": ["...", "..."],
    "meso": ["...", "..."],
    "micro": ["...", "..."]
}
```
**Used for**: Older datasets (HumanML3D, etc.)

---

## 🔗 Loading Pipeline Output

### After `prepare_data()`
```python
{
    "motion_path": str,              # Absolute path
    "caption_path": str,             # Absolute path
    "num_frames": int,
    "fps": float,
    "duration": float,
    "has_hand": bool,
    # ... other fields
}
```

### After `LoadCompatibleCaption`
```python
{
    # ... all above ...
    "caption": str,                  # One selected caption
    "caption_list": [str, ...],      # All captions
    "granularity": str,              # (Legacy format only)
}
```

### After `LoadPreExtractedTextEmbedding` (if .pt found)
```python
{
    # ... all above ...
    "text_vec_raw": Tensor[1, 768],         # CLIP-L embedding
    "text_ctxt_raw": Tensor[seq, 4096],     # Qwen3 embedding
    "text_ctxt_raw_length": Tensor[1],      # Sequence length
    "_text_is_null": False,
}
```

### If no .pt file found
```python
{
    # ... all above ...
    "text_vec_raw": Tensor[1, 768],         # All zeros
    "text_ctxt_raw": Tensor[1, 4096],       # All zeros
    "text_ctxt_raw_length": Tensor(0),      # Zero (signals null)
    "_text_is_null": True,
}
```

---

## 🚀 Getting Started

1. **For a quick understanding**: Read [`HYMOTION_ANNOTATION_QUICK_REFERENCE.md`](HYMOTION_ANNOTATION_QUICK_REFERENCE.md)

2. **For complete details**: Read [`HYMOTION_ANNOTATION_FORMAT.md`](HYMOTION_ANNOTATION_FORMAT.md)

3. **To see concrete examples**: Read [`HYMOTION_ANNOTATION_EXAMPLES.md`](HYMOTION_ANNOTATION_EXAMPLES.md)

4. **For a quick lookup**: See [`ANNOTATION_FORMAT_SUMMARY.txt`](ANNOTATION_FORMAT_SUMMARY.txt)

---

## ❓ Common Questions

**Q: Which caption field should I use?**
A: Use `hierarchical_caption_path` (preferred), fallback to `caption_path`, then `caption`.

**Q: What's the difference between HyMotion and Legacy formats?**
A: HyMotion uses `"result"` array with `short_caption` + `short_caption_rewritten`. Legacy uses `macro`, `meso`, `micro` keys with granularity levels.

**Q: Why is data_list a dictionary instead of an array?**
A: It allows unique key identification for entries and faster lookups.

**Q: Can smplx_path be a list?**
A: Yes, for multi-person motion. Can be string (1 person) or list (2+ people).

**Q: What happens if the pre-extracted embedding .pt file doesn't exist?**
A: The system fills null embeddings (zeros) and sets `_text_is_null: True`.

---

## 📝 Notes

- All paths in the annotation JSON are relative to `data_dir` (typically "data/motionhub")
- Captions are randomly selected during loading for data augmentation
- Pre-extracted embeddings have specific directory mappings (see CAPTION_TO_QWEN3_DIR)
- The loading code handles both old and new caption formats transparently

