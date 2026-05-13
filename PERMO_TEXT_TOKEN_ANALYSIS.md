# PerMo Dataset: Text Token Extraction Analysis

## Summary
**PerMo dataset currently does NOT have pre-extracted text tokens.** Only raw captions exist. This extraction pipeline needs to be created.

---

## 1. HyMotion M2M v2 Caption Config — Text Conditioning Architecture

### Config Location
- Base: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- Caption configs: `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_*.py`

### Text Conditioning Flow
```python
# From hymotion_m2m_v2_caption_global_phase2.py (line 42)
dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
```

**Key dimensions:**
- `text_vec_raw`: (1, 1, 768) — CLIP-L text representation (sentence embeddings)
- `text_ctxt_raw`: (1, seq, 4096) — Qwen3-Embedding-8B context tokens
- `text_ctxt_raw_length`: (1,) — actual sequence length

Model expects:
```python
ctxt_input_dim=4096,    # Qwen3 embedding dim
vtxt_input_dim=768,     # CLIP-L embedding dim
```

---

## 2. Text Token Extraction Pipeline

### Transform Implementation
**File:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py`

#### `LoadPreExtractedTextEmbedding` Transform
- **Function**: Loads pre-computed tokens from `.pt` files
- **Mapping**: Caption JSON → Pre-extracted `.pt` embedding files
- **Mapping table** (lines 18-34):
  ```python
  CAPTION_TO_QWEN3_DIR = {
      'augmented_caption': 'qwen3embedding_augmented',
      'human_checked_augmented_caption': 'qwen3_augmented',
      'human_checked_caption': 'qwen3_human_checked_short',
      ...
  }
  ```

#### Expected .pt File Format
```python
{
  'result': [
    {
      'caption': str,
      'text_embedding': {
        'text_vec_raw': Tensor[1, 1, 768],      # CLIP-L
        'text_ctxt_raw': Tensor[1, seq, 4096],  # Qwen3
        'text_ctxt_raw_length': Tensor[1],
      },
      ...
    },
    ...
  ]
}
```

#### Fallback Behavior
When no `.pt` file exists:
- Fills null embeddings (all zeros)
- Sets `text_ctxt_raw_length=0` to mark as null
- Allows training to continue with `LoadCompatibleCaption` for on-the-fly encoding

---

## 3. PerMo Caption Data Status

### Current State
- **Location**: `data/hymotion_data/PerMo/PerMo/20260513/augmented_caption/train/`
- **Format**: HYMotion caption format (result array with `short_caption` key)
- **Sample**:
  ```json
  {
    "result": [
      {
        "short_caption": "The person leaps forward, landing on the same leg each time with an angry, forceful manner."
      }
    ]
  }
  ```

### Missing Pre-Extracted Tokens
```
data/hymotion_data/PerMo/PerMo/20260513/
├── augmented_caption/           ✓ exists (raw captions)
├── qwen3_augmented/             ✗ MISSING (pre-extracted tokens)
├── qwen3embedding_augmented/    ✗ MISSING (pre-extracted tokens)
├── motions/
├── motions_198/
└── pairs/
```

### Contrast with Academic Dataset
Academic dataset (20250916) has pre-extracted tokens:
```
data/hymotion_data/Academic/20250916/
├── human_checked_augmented_caption/
├── qwen3_augmented/              ✓ has .pt files
├── qwen3_human_checked_short/    ✓ has .pt files
├── qwen3embedding_augmented/     ✓ has .pt files
├── qwen3embedding_improved_simple_keywords/  ✓ etc.
└── ...
```

**File count in Academic qwen3_augmented:**
```
HumanML3D-ACCAD: 37 .pt files
HumanML3D-BMLmovi: 40 .pt files
... (12 total dataset sources)
```

**Sample .pt structure** (verified):
```
Keys: ['result']
Result list length: 10  (10 caption variants per motion)
First item keys: ['caption', 'text_embedding', 'start_time', 'end_time', 'version']
text_vec_raw: shape=(1, 1, 768), dtype=float32
text_ctxt_raw: shape=(1, 15, 4096), dtype=float32
text_ctxt_raw_length: shape=(1,), dtype=int64
```

---

## 4. Text Token Extraction Mechanism

### Extraction Tool: `HYTextModel`
**Location**: `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py`

**Configuration**:
```python
model = HYTextModel(
    llm_type='qwen3_embedding',        # Qwen3-Embedding-8B model
    sentence_emb_type='clipl',         # CLIP-L for sentence embeddings
    max_length_llm=512,                # Max token length for Qwen3
    max_length_sentence_emb=77,        # Max CLIP tokens
)
```

**Output**:
```python
vtxt, ctxt, ctxt_len = model.encode(batch)
# vtxt: (B, 1, 768)      — CLIP-L embeddings
# ctxt: (B, seq, 4096)   — Qwen3 context embeddings
# ctxt_len: (B,)         — actual sequence lengths
```

### Extraction Pipeline in Production

**1. MotionFix Dataset** (`scripts/data/prepare_motionfix_hymotion.py`)
- Function: `extract_embeddings()`
- Maps caption JSON paths to qwen3embedding directories
- Loads HYTextModel and processes in batches
- Saves to `.pt` files with exact format expected by `LoadPreExtractedTextEmbedding`
- Supports sharded processing (multi-GPU, multi-node)

**2. Eval Captions** (`scripts/caption/extract_eval_caption_embeddings.py`)
- Pre-extracts embeddings for all eval caption variants
- Stores in `data/eval/m2m_v2/caption_embeddings/cache.pt`
- Keyed by caption text for fast lookup

---

## 5. PerMo Training Implication

### Current Bottleneck
When training M2M v2 caption models on PerMo:
```python
dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
```

If qwen3_augmented dir doesn't exist:
- **Every sample gets null embeddings** (all zeros)
- `LoadCompatibleCaption` fallback is NOT used (already parsed caption data)
- Model trains in **effectively unconditioned mode** during caption phase
- Text embeddings are zeros, defeating caption-conditioned training

### To Enable Caption Training on PerMo
**Must create**: `data/hymotion_data/PerMo/PerMo/20260513/qwen3_augmented/`

**And either**:
1. Use `prepare_motionfix_hymotion.py` pattern as template (sharded extraction)
2. Extend extraction from `extract_eval_caption_embeddings.py`
3. Create dedicated PerMo extraction script

**Estimated cost**:
- PerMo caption count: ~4,000–5,000 unique motions
- Qwen3-Embedding-8B inference: ~100–200 captions/min on V100
- Compute time: 20–50 GPU hours (sharded across multiple GPUs)

---

## 6. Configuration Status for PerMo

### Configs Using Caption Conditioning
- `hymotion_m2m_v2_caption_global_phase1.py` — uses `LoadPreExtractedTextEmbedding`
- `hymotion_m2m_v2_caption_global_phase2.py` — uses `LoadPreExtractedTextEmbedding`
- `hymotion_m2m_v2_caption_local_phase1.py` — uses `LoadPreExtractedTextEmbedding`
- `hymotion_m2m_v2_caption_local_phase2.py` — uses `LoadPreExtractedTextEmbedding`
- `hymotion_m2m_v2_caption_local_phase2b.py` — uses `LoadPreExtractedTextEmbedding`

### PerMo Config Support
- No existing PerMo-specific config found
- Caption configs can be adapted via `anno_file` pointing to PerMo annotation JSON
- But **caption training will be ineffective** without pre-extracted tokens

---

## Conclusion

| Aspect | Status |
|--------|--------|
| **Caption Raw Data** | ✓ Ready (augmented_caption/) |
| **Pre-extracted Tokens** | ✗ **Missing** (qwen3_augmented/) |
| **Transform Support** | ✓ Ready (LoadPreExtractedTextEmbedding) |
| **Extraction Code Available** | ✓ Yes (prepare_motionfix_hymotion.py pattern) |
| **Caption Training Functional** | ✗ **Will fail silently** (null embeddings) |

**User's requirement**: "instruction需要用qwen提取token" — This extraction is **NOT done for PerMo**. The pipeline exists but hasn't been executed.
