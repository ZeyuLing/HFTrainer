# HyMotion Annotation Format - Quick Reference

## Main Annotation File Structure (train_hymotion_400h.json)

```json
{
    "meta_info": {
        "dataset": "hymotion_data - train",
        "version": "v1"
    },
    "data_list": {
        "entry_key_1": {
            // Required motion fields
            "smplx_path": "relative/path/to/motion.npz",      // or list for multi-person
            "num_frames": 275,
            "fps": 30.0,
            "duration": 9.167,
            "has_hand": true,
            
            // Required metadata
            "subset": "academic",
            
            // Caption (at least one required)
            "hierarchical_caption_path": "path/to/caption.json",  // ← PREFERRED
            // "caption_path": "...",                              // ← Fallback
            // "caption": "text" or ["text1", "text2"],           // ← Fallback
            
            // Optional audio/music/speech
            "music_path": "path/to/music.wav",
            "genre": "pop",
            "audio_path": "path/to/audio.wav",
            "speech_script_path": "path/to/script.txt",
            "language": "en"
        },
        "entry_key_2": { ... }
    }
}
```

## Hierarchical Caption File Formats

### Format 1: HyMotion (Recommended for train_hymotion_400h.json)

```json
{
    "result": [
        {
            "short_caption": "a person walks forward slowly",
            "short_caption_rewritten": [
                "a person walks forward slowly",
                "a person walks forward in a slow manner",
                "person walks forward at slow speed"
            ]
        },
        {
            "short_caption": "a person raises both hands",
            "short_caption_rewritten": [ ... ]
        }
    ]
}
```

**Notes:**
- Top key: `"result"` (always a list)
- Each item has `short_caption` (string) and `short_caption_rewritten` (list, optional)
- Accepts both `"short_caption"` and `"short caption"` (underscore or space)
- Rewritten variants provide data augmentation

### Format 2: Legacy Hierarchical (for older datasets)

```json
{
    "macro": [
        "a person walks forward",
        "a person is walking"
    ],
    "meso": [
        "a person walks forward slowly",
        "a person walks forward at normal pace"
    ],
    "micro": [
        "a person walks forward slowly with arms at side",
        "a person walks forward slowly with arms moving"
    ]
}
```

**Notes:**
- Required keys: `"macro"`, `"meso"`, `"micro"`
- Each is a list of captions at that granularity
- All captions flattened + randomly selected with granularity label

## Field Summary

### REQUIRED fields in each data_list entry:
| Field | Type | Example |
|-------|------|---------|
| `smplx_path` | str or list | `"../data/motion.npz"` |
| `num_frames` | int | `275` |
| `fps` | float | `30.0` |
| `duration` | float | `9.167` |
| `has_hand` | bool | `true` |
| `subset` | str | `"academic"` |
| caption (any one) | str | `hierarchical_caption_path` |

### OPTIONAL fields:
- `music_path`, `genre`, `sr` (audio)
- `audio_path`, `speech_script_path`, `speaker_id` (speech)
- `language` (metadata)

## Loading Flow

```
1. Load main JSON
   ├─ Extract meta_info
   └─ Extract data_list (dict of entries)

2. For each entry:
   ├─ Resolve relative paths → absolute paths
   ├─ Load caption from hierarchical_caption_path
   ├─ Parse caption JSON (auto-detect format)
   ├─ Extract caption strings + variants
   ├─ Randomly select one caption
   └─ Load pre-extracted embeddings (if .pt exists)

3. Output per sample:
   ├─ caption: str (selected caption text)
   ├─ caption_list: list[str] (all available captions)
   ├─ text_vec_raw: Tensor[768] (CLIP-L embedding, if .pt loaded)
   ├─ text_ctxt_raw: Tensor[seq, 4096] (Qwen3 embedding, if .pt loaded)
   └─ ... (motion, fps, duration, etc.)
```

## Key Code Locations

| Component | File |
|-----------|------|
| Main dataset class | `hftrainer/datasets/motion/motionhub/multitask_multiagent_dataset.py` |
| Caption loading transforms | `hftrainer/datasets/motion/motionhub/transforms/load_text.py` |
| Base dataset loader | `hftrainer/datasets/motion/motionhub/single_agent_dataset.py` |
| Embedding path mapping | `load_text.py` → `CAPTION_TO_QWEN3_DIR` |

## Pre-Extracted Embedding Mapping

```python
CAPTION_TO_QWEN3_DIR = {
    'human_checked_augmented_caption': 'qwen3_augmented',
    'human_checked_caption': 'qwen3_human_checked_short',
    'improved_simple_augmented_caption': 'qwen3_improved_simple_short',
    'improved_simple_caption': 'qwen3_improved_simple_short',
    'augmented_caption': 'qwen3embedding_augmented',
}
```

**Example:**
- Caption: `.../human_checked_augmented_caption/motion.json`
- Embedding: `.../qwen3_augmented/motion.pt`

## Expected Data Output

After loading through `LoadCompatibleCaption`:
```python
{
    "motion_path": str,                  # Absolute path to .npz
    "num_frames": int,
    "fps": float,
    "duration": float,
    "has_hand": bool,
    "subset": str,
    "caption": str,                      # Selected caption
    "caption_list": list[str],           # All available captions
    "granularity": str,                  # (Legacy format only)
    "granularity_list": list[str],       # (Legacy format only)
    # ... optional audio/music/speech fields
}
```

After `LoadPreExtractedTextEmbedding` (if .pt exists):
```python
{
    # ... all above ...
    "text_vec_raw": Tensor[768],         # CLIP-L
    "text_ctxt_raw": Tensor[seq, 4096],  # Qwen3
    "text_ctxt_raw_length": Tensor[1],   # Seq length
    "_text_is_null": bool,               # True if fallback
}
```

