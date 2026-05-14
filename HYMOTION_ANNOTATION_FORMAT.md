# HyMotion Annotation JSON Format

## 1. Main Annotation File Format (train_hymotion_400h.json)

### Top-Level Structure
```json
{
    "meta_info": { ... },
    "data_list": { ... }
}
```

### 1.1 meta_info Fields

| Field | Type | Description |
|-------|------|-------------|
| `dataset` | string | Dataset name (e.g., "hymotion_data - train") |
| `version` | string | Format version (e.g., "v1") |

### 1.2 data_list Structure

The `data_list` is a **dictionary** (not an array), where:
- **Keys**: unique identifiers (e.g., "academic_HumanML3D-HumanEva_S3_Box_1_poses_origintime_13.55_22.7")
- **Values**: data entry dictionaries

### 1.3 Data Entry Fields (Inside data_list Values)

Each entry in `data_list` can contain:

#### Motion Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `smplx_path` | string or list | ✓ | Path to motion file (SMPLx format .npz). Can be list for multi-person (2+ people). |
| `num_frames` | integer | ✓ | Number of frames in the motion |
| `fps` | float | ✓ | Frames per second (commonly 30.0) |
| `duration` | float | ✓ | Duration in seconds |
| `has_hand` | boolean | ✓ | Whether hand motion is included |

#### Caption Fields (IMPORTANT)
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `hierarchical_caption_path` | string | ~ | **PREFERRED**: Path to hierarchical caption JSON file (see Section 2) |
| `caption_path` | string | ~ | **FALLBACK**: Alternative path if hierarchical_caption_path not available |
| `caption` | string or list | ~ | **FALLBACK**: Direct caption string or list of captions |

**Note**: At least one caption field should be present. The loading code checks in order: `hierarchical_caption_path` → `caption_path` → `caption`.

#### Audio/Music Fields
| Field | Type | Description |
|-------|------|-------------|
| `music_path` | string | Path to music/dance audio file |
| `genre` | string | Music genre (used with music_path) |
| `audio_path` | string | Path to speech/audio file |
| `sr` | integer | Sample rate of audio |

#### Speech Fields
| Field | Type | Description |
|-------|------|-------------|
| `speech_script_path` | string | Path to text file containing speech script |
| `speaker_id` | string | Speaker identifier |

#### Metadata Fields
| Field | Type | Description |
|-------|------|-------------|
| `subset` | string | Data subset name (e.g., "academic") |
| `language` | string | Language code if applicable |

### 1.4 Example Entry
```json
{
    "subset": "academic",
    "duration": 9.166666666666666,
    "num_frames": 275,
    "smplx_path": "../hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.npz",
    "hierarchical_caption_path": "../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json",
    "fps": 30.0,
    "has_hand": true
}
```

---

## 2. Hierarchical Caption File Format

The files referenced by `hierarchical_caption_path` are JSON files with one of two formats:

### Format A: HyMotion Format (Recommended)
Used for HyMotion dataset annotations.

```json
{
    "result": [
        {
            "short_caption": "a person walks forward slowly",
            "short_caption_rewritten": [
                "a person walks forward slowly",
                "a person walks forward in a slow manner"
            ],
            ... (other fields may exist)
        },
        {
            "short_caption": "a person raises both hands above their head",
            "short_caption_rewritten": [ ... ],
            ... 
        }
    ]
}
```

#### HyMotion Format Key Points:
- Top-level key: `"result"` (array of objects)
- Each item in `result` contains:
  - **`short_caption`** (string): Main caption text (space variant: "short caption")
  - **`short_caption_rewritten`** (array of strings, OPTIONAL): Augmented/rewritten variants of the caption

**Note**: The code accepts both underscore and space variants:
- `"short_caption"` or `"short caption"`
- `"short_caption_rewritten"` or `"short caption_rewritten"`

Preference order:
1. `short_caption_rewritten` (if it exists and is non-empty list)
2. `short_caption` (as fallback)

### Format B: Legacy Hierarchical Format
Used for older datasets (HumanML3D, etc.)

```json
{
    "macro": [
        "a person walks forward",
        "a person is walking"
    ],
    "meso": [
        "a person walks forward slowly",
        "a person walks forward at a normal pace"
    ],
    "micro": [
        "a person walks forward slowly with arms at their side",
        "a person walks forward slowly with arms moving"
    ]
}
```

#### Legacy Hierarchical Format Key Points:
- Top-level keys: `"macro"`, `"meso"`, `"micro"` (all required, all arrays)
- Each key contains a list of caption strings at that granularity level
- During loading, all captions are flattened into a single list with granularity labels

### Format Selection

The code uses **LoadCompatibleCaption** transform which auto-detects:

1. If `result` key exists → HyMotion format
2. Else if `macro`, `meso`, `micro` all exist → Legacy Hierarchical format
3. Else → Error

---

## 3. Dataset Loading Code Path

### Files That Load These Annotations

1. **`hftrainer/datasets/motion/motionhub/single_agent_dataset.py`** (Base class)
   - `load_data_list()`: Parses main annotation JSON
   - Expects: `annotations["meta_info"]` and `annotations["data_list"]`
   - Skips multi-person data if `motion_key + "_path"` is not a string

2. **`hftrainer/datasets/motion/motionhub/multitask_multiagent_dataset.py`** (Main user)
   - Inherits from MotionHubSingleAgentDataset
   - `prepare_data()`: Builds final sample dict with paths resolved
   - Checks `hierarchical_caption_path` → `caption_path` for caption

3. **`hftrainer/datasets/motion/motionhub/transforms/load_text.py`** (Caption loading)
   - **LoadHierarchicalCaption**: For legacy format (macro/meso/micro)
   - **LoadHYMotionCaption**: For HyMotion format (result array)
   - **LoadCompatibleCaption**: Auto-detects and handles both formats
   - **LoadPreExtractedTextEmbedding**: Loads pre-computed embeddings from .pt files

4. **Transform Chain**: 
   ```python
   LoadCompatibleCaption → LoadPreExtractedTextEmbedding
   ```
   When both are in pipeline:
   - First: Load caption from JSON (text, granularity, variants)
   - Second: Load pre-extracted embeddings if .pt file exists, else fill nulls

---

## 4. Data Flow During Loading

### Step 1: Load Main Annotation
```
train_hymotion_400h.json
└── mmengine.load() reads JSON
└── Extracts meta_info and data_list
└── For each entry, adds to data_list (as-is, paths not resolved yet)
```

### Step 2: Prepare Sample
```
Raw entry from data_list
└── Join paths with data_dir: "data/motionhub" + relative paths
└── Resolve smplx_path → motion_path (join with data_dir)
└── Resolve hierarchical_caption_path → caption_path
└── Create data_info dict with resolved paths
```

### Step 3: Load Text (Transform)
```
caption_path
└── read_json(caption_path)
└── Detect format (HyMotion vs Legacy)
└── Extract caption strings and variants
└── Randomly select one caption
└── Set results["caption"] and results["caption_list"]
```

### Step 4: Load Embeddings (Transform)
```
caption_path (from results)
└── Map to .pt embedding path via CAPTION_TO_QWEN3_DIR
└── Load torch.load(pt_path)
└── Extract text_vec_raw, text_ctxt_raw, text_ctxt_raw_length
└── Randomly select from result list
└── Set embedding tensors in results
```

---

## 5. Key Data Structures in Memory

### After `prepare_data()` (Before transforms):
```python
{
    "num_person": int,                       # 1 or 2+
    "motion_path": str or list[str],         # Absolute path(s)
    "subset": str,                           # e.g., "academic"
    "fps": float,                            # e.g., 30.0
    "has_hand": bool,
    "duration": float,
    "num_frames": int,
    "caption_path": str,                     # Absolute path to caption JSON
    "music_path": str or None,
    "genre": str or None,
    "audio_path": str or None,
    "speech_script_path": str or None,
    ...
}
```

### After `LoadCompatibleCaption` transform:
```python
{
    ... (all above fields) ...
    "caption": str,                          # Selected single caption
    "caption_list": list[str],               # All available captions
    "granularity": str,                      # (HyMotion format only)
    "granularity_list": list[str],           # (HyMotion format only)
}
```

### After `LoadPreExtractedTextEmbedding` transform (if .pt exists):
```python
{
    ... (all above fields) ...
    "text_vec_raw": Tensor[768],             # CLIP-L embedding
    "text_ctxt_raw": Tensor[seq, 4096],      # Qwen3 context embedding
    "text_ctxt_raw_length": Tensor[1],       # Sequence length
    "_text_is_null": bool,                   # True if fallback to nulls
}
```

---

## 6. Path Mapping for Pre-Extracted Embeddings

From `load_text.py`:

```python
CAPTION_TO_QWEN3_DIR = {
    'human_checked_augmented_caption': 'qwen3_augmented',
    'human_checked_caption': 'qwen3_human_checked_short',
    'improved_simple_augmented_caption': 'qwen3_improved_simple_short',
    'improved_simple_caption': 'qwen3_improved_simple_short',
    'augmented_caption': 'qwen3embedding_augmented',
    # ... (other mappings)
}
```

**Example Mapping**:
```
Caption:   ../hymotion_data/.../human_checked_augmented_caption/S3_Box.json
Embedding: ../hymotion_data/.../qwen3_augmented/S3_Box.pt
```

---

## 7. Required vs. Optional Fields

### REQUIRED (must always be present):
- `smplx_path` (motion file)
- `num_frames`
- `fps`
- `duration`
- `has_hand`
- `subset`
- At least ONE of: `hierarchical_caption_path`, `caption_path`, `caption`

### OPTIONAL:
- Music fields: `music_path`, `genre`, `sr`
- Speech fields: `audio_path`, `speech_script_path`, `speaker_id`
- Other: `language`

---

## 8. Caption Extraction Logic

### HyMotion Format:
```python
result_list = hierarchical_caption["result"]
for item in result_list:
    # Try rewritten variants first
    if "short_caption_rewritten" in item and isinstance(item["short_caption_rewritten"], list):
        caption_list.extend(item["short_caption_rewritten"])
    # Fall back to main short_caption
    elif "short_caption" in item and isinstance(item["short_caption"], str):
        caption_list.append(item["short_caption"])

# Randomly select one
selected_caption = caption_list[random.randint(0, len(caption_list) - 1)]
```

### Legacy Format:
```python
for granularity in ["macro", "meso", "micro"]:
    captions = hierarchical_caption[granularity]
    for caption in captions:
        caption_list.append(caption)
        granularity_list.append(granularity)

# Randomly select one with granularity label
idx = random.randint(0, len(caption_list) - 1)
selected_caption = caption_list[idx]
selected_granularity = granularity_list[idx]
```

