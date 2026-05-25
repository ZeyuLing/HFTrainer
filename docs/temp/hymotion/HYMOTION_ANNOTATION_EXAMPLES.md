# HyMotion Annotation Format - Examples

## Example 1: Main Annotation File (train_hymotion_400h.json)

```json
{
    "meta_info": {
        "dataset": "hymotion_data - train",
        "version": "v1"
    },
    "data_list": {
        "academic_HumanML3D-HumanEva_S3_Box_1_poses_origintime_13.55_22.7": {
            "subset": "academic",
            "duration": 9.166666666666666,
            "num_frames": 275,
            "smplx_path": "../hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.npz",
            "hierarchical_caption_path": "../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json",
            "fps": 30.0,
            "has_hand": true
        },
        "academic_HumanML3D-HumanEva_S1_Static_poses_origintime_0.0_3.0": {
            "subset": "academic",
            "duration": 2.966666666666667,
            "num_frames": 89,
            "smplx_path": "../hymotion_data/Academic/20250916/motions/HumanML3D-HumanEva/S1_Static_poses_origintime_0.0_3.0.npz",
            "hierarchical_caption_path": "../hymotion_data/Academic/20250916/improved_simple_augmented_caption/HumanML3D-HumanEva/S1_Static_poses_origintime_0.0_3.0.json",
            "fps": 30.0,
            "has_hand": true
        }
    }
}
```

### Key Points:
- `data_list` is a **dictionary**, not an array
- Each entry key is unique (e.g., "academic_HumanML3D-HumanEva_S3_Box_1_...")
- Paths are relative to `data_dir` (typically "data/motionhub")
- `hierarchical_caption_path` points to caption JSON files

---

## Example 2: HyMotion Caption File Format (Single Entry)

**File**: `../hymotion_data/Academic/20250916/human_checked_augmented_caption/HumanML3D-HumanEva/S3_Box_1_poses_origintime_13.55_22.7.json`

```json
{
    "result": [
        {
            "short_caption": "a person walks forward slowly",
            "short_caption_rewritten": [
                "a person walks forward slowly",
                "a person walks forward in a slow manner",
                "a person moves forward at a slow pace"
            ],
            "frame_info": {
                "start": 0,
                "end": 275
            },
            "other_fields": "..."
        }
    ]
}
```

### What the loader does:
1. Reads this JSON file
2. Detects format: `"result"` key exists → HyMotion format
3. Iterates through `result` array:
   - Takes `short_caption_rewritten` list if available: `["a person walks forward slowly", "a person walks forward in a slow manner", "a person moves forward at a slow pace"]`
   - Falls back to `short_caption` if rewritten list missing
4. Creates `caption_list = ["a person walks forward slowly", "a person walks forward in a slow manner", "a person moves forward at a slow pace"]`
5. Randomly selects one: `caption = "a person walks forward in a slow manner"`
6. Stores: `results["caption"] = "a person walks forward in a slow manner"` and `results["caption_list"] = [...]`

---

## Example 3: HyMotion Caption File with Multiple Captions

**More realistic example with multiple result items:**

```json
{
    "result": [
        {
            "short_caption": "a person walks forward",
            "short_caption_rewritten": [
                "a person walks forward",
                "the person walks in a forward direction"
            ]
        },
        {
            "short_caption": "a person raises their hands",
            "short_caption_rewritten": [
                "a person raises both hands above head",
                "a person raises hands upward"
            ]
        },
        {
            "short_caption": "a person stands still",
            "short_caption_rewritten": null
        }
    ]
}
```

### What the loader does:
1. Detects HyMotion format (has `"result"` key)
2. Processes each result item:
   - Item 1: Adds 2 captions from `short_caption_rewritten`
   - Item 2: Adds 2 captions from `short_caption_rewritten`
   - Item 3: `short_caption_rewritten` is null, falls back to `short_caption` → adds 1 caption
3. Final `caption_list = ["a person walks forward", "the person walks in a forward direction", "a person raises both hands above head", "a person raises hands upward", "a person stands still"]`
4. Randomly selects one from this list

---

## Example 4: Legacy Hierarchical Format (For Reference)

**File**: Some older caption files may use this format (macro/meso/micro):

```json
{
    "macro": [
        "a person walks forward",
        "a person is walking"
    ],
    "meso": [
        "a person walks forward slowly",
        "a person walks forward at a normal pace",
        "a person walks forward quickly"
    ],
    "micro": [
        "a person walks forward slowly with arms at side",
        "a person walks forward slowly with arms swinging",
        "a person walks forward slowly with hands in pockets"
    ]
}
```

### What the loader does (LoadHierarchicalCaption or LoadCompatibleCaption):
1. Detects format: has `macro`, `meso`, `micro` keys → Legacy format
2. Creates flattened list:
   ```
   caption_list = [
       "a person walks forward",
       "a person is walking",
       "a person walks forward slowly",
       "a person walks forward at a normal pace",
       "a person walks forward quickly",
       "a person walks forward slowly with arms at side",
       "a person walks forward slowly with arms swinging",
       "a person walks forward slowly with hands in pockets"
   ]
   ```
3. Creates granularity list:
   ```
   granularity_list = [
       "macro", "macro",
       "meso", "meso", "meso",
       "micro", "micro", "micro"
   ]
   ```
4. Randomly selects: e.g., `caption = "a person walks forward slowly with arms swinging"`, `granularity = "micro"`

---

## Example 5: Pre-Extracted Embedding File (After LoadPreExtractedTextEmbedding)

**File**: `.../qwen3_augmented/S3_Box_1_poses_origintime_13.55_22.7.pt`

When loaded via torch.load():

```python
{
    "result": [
        {
            "caption": "a person walks forward slowly",
            "text_embedding": {
                "text_vec_raw": torch.Tensor([[[...768 dims...]]])  # shape [1, 1, 768]
                "text_ctxt_raw": torch.Tensor([[[...4096 dims...]]]))  # shape [1, seq_len, 4096]
                "text_ctxt_raw_length": torch.Tensor([[seq_len]]),  # shape [1]
            }
        },
        {
            "caption": "a person walks forward in a slow manner",
            "text_embedding": {
                "text_vec_raw": torch.Tensor([[[...768 dims...]]])
                "text_ctxt_raw": torch.Tensor([[[...4096 dims...]]])
                "text_ctxt_raw_length": torch.Tensor([[seq_len]]),
            }
        }
    ]
}
```

### What the loader does:
1. Maps caption path to embedding path:
   - Input: `.../human_checked_augmented_caption/S3_Box.json`
   - Output: `.../qwen3_augmented/S3_Box.pt`
2. Loads the .pt file with torch.load()
3. Randomly selects from `result` list (e.g., index 0)
4. Extracts embeddings and squeezes batch dimension:
   ```python
   text_vec_raw = emb['text_vec_raw'].squeeze(0)       # [1, 768]
   text_ctxt_raw = emb['text_ctxt_raw'].squeeze(0)     # [seq, 4096]
   text_ctxt_raw_length = emb['text_ctxt_raw_length'].squeeze(0)  # scalar
   ```
5. Stores in results dict:
   ```python
   results['text_vec_raw'] = text_vec_raw
   results['text_ctxt_raw'] = text_ctxt_raw
   results['text_ctxt_raw_length'] = text_ctxt_raw_length
   results['caption'] = "a person walks forward slowly"
   ```

---

## Example 6: Multi-Person Motion Entry

```json
{
    "subset": "taobao",
    "duration": 10.0,
    "num_frames": 300,
    "smplx_path": [
        "../hymotion_data/Taobao/20250916/motions/pair_1/person_0/motion.npz",
        "../hymotion_data/Taobao/20250916/motions/pair_1/person_1/motion.npz"
    ],
    "hierarchical_caption_path": "../hymotion_data/Taobao/20250916/captions/pair_1.json",
    "fps": 30.0,
    "has_hand": true,
    "num_person": 2
}
```

### Key differences:
- `smplx_path` is a **list** instead of string (2 people)
- May have optional `num_person` field explicitly set
- Single `hierarchical_caption_path` for the pair

---

## Example 7: Entry with Optional Fields

```json
{
    "subset": "music_dance",
    "duration": 8.5,
    "num_frames": 255,
    "smplx_path": "../hymotion_data/Music/motions/tiktok_dance_1.npz",
    "hierarchical_caption_path": "../hymotion_data/Music/captions/tiktok_dance_1.json",
    "fps": 30.0,
    "has_hand": true,
    
    "music_path": "../hymotion_data/Music/audio/tiktok_dance_1.wav",
    "genre": "pop",
    "sr": 22050,
    
    "audio_path": "../hymotion_data/Speech/audio/speaker_1_gesture.wav",
    "speech_script_path": "../hymotion_data/Speech/scripts/speaker_1.txt",
    "speaker_id": "speaker_001",
    
    "language": "en"
}
```

### What's present:
- **Required**: smplx_path, num_frames, fps, duration, has_hand, subset, hierarchical_caption_path
- **Music** (optional): music_path, genre, sr
- **Speech** (optional): audio_path, speech_script_path, speaker_id
- **Metadata** (optional): language

---

## Example 8: Full Loading Pipeline Output

**After all transforms process a sample:**

```python
# From MotionHubSingleAgentDataset.prepare_data()
{
    "num_person": 1,
    "motion_path": "/absolute/path/to/motion.npz",
    "subset": "academic",
    "fps": 30.0,
    "has_hand": True,
    "duration": 9.166666666666666,
    "num_frames": 275,
    "caption_path": "/absolute/path/to/caption.json",
    "music_path": None,
    "genre": None,
    "audio_path": None,
    "speech_script_path": None,
    "speaker_id": None,
    "task": <task_object>
}

# After LoadCompatibleCaption transform
{
    # ... all above ...
    "caption": "a person walks forward in a slow manner",  # One selected
    "caption_list": [
        "a person walks forward slowly",
        "a person walks forward in a slow manner",
        "a person moves forward at a slow pace"
    ]
}

# After LoadPreExtractedTextEmbedding transform (if .pt file exists)
{
    # ... all above ...
    "text_vec_raw": Tensor(shape=[1, 768]),          # CLIP-L
    "text_ctxt_raw": Tensor(shape=[seq_len, 4096]),  # Qwen3
    "text_ctxt_raw_length": Tensor(shape=[]),        # Scalar with seq_len
    "_text_is_null": False
}

# If .pt file NOT found, _fill_null_embedding() is called:
{
    # ... all above ...
    "text_vec_raw": Tensor(shape=[1, 768]),          # All zeros
    "text_ctxt_raw": Tensor(shape=[1, 4096]),        # All zeros
    "text_ctxt_raw_length": Tensor(0),               # Zero (signals null)
    "_text_is_null": True
}
```

---

## Example 9: Variant with All Three Caption Options

Some entries might support fallback chains:

```json
{
    "subset": "fallback_test",
    "duration": 5.0,
    "num_frames": 150,
    "smplx_path": "motion.npz",
    "fps": 30.0,
    "has_hand": false,
    
    "hierarchical_caption_path": "path/to/caption.json",  // ← Checked first
    "caption_path": "path/to/caption2.json",              // ← Checked second
    "caption": "direct caption text"                       // ← Checked third
}
```

**Loading order:**
1. Check `hierarchical_caption_path` → if exists and readable, use it
2. Else check `caption_path` → if exists and readable, use it
3. Else use `caption` field directly (string or list)
4. If none available and `allow_none=False` → error

