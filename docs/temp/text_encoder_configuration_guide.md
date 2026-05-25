# HYTextModel Text Encoder Configuration - Complete Report

## Summary

The `text_encoder` configuration for HYTextModel in the hf_trainer project uses a **lazy-loading, zero-initialization pattern**. Most configs use `text_encoder=dict()` as an empty placeholder, and the actual encoder is only instantiated at inference time when text needs to be encoded.

## HYTextModel Class Definition

**Location**: `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py` (lines 73-117)

### Constructor Parameters

The `HYTextModel.__init__()` accepts these configuration keys:

```python
HYTextModel(
    llm_type: str = "qwen3",                          # LLM encoder type
    max_length_llm: int = 512,                        # Max tokens for LLM
    sentence_emb_type: str = "clipl",                 # Sentence embedding type
    max_length_sentence_emb: int = 77,                # Max tokens for CLIP
    enable_llm_padding: bool = True,                  # Pad LLM to max_length
    torch_dtype: Optional[torch.dtype] = None,        # Precision (fp32, bf16, etc)
)
```

### Supported Model Types

**LLM Encoders** (for contextual text tokens, `ctxt_input_dim=4096`):
- `"qwen3"` → Qwen3-8B (default, 4096-dim hidden state)
- `"qwen3_embedding"` → Qwen3-Embedding-8B (optimized for embedding tasks)
- `"t5"` → T5-v1.1-XXL
- `"distilbert"` → DistilBERT-base-uncased

**Sentence Embeddings** (for sentence-level vectors, `vtxt_input_dim=768`):
- `"clipl"` (default) → CLIP ViT-Large/14 (768-dim)
- `"sentence_transformer"` → all-mpnet-base-v2 (768-dim)

## Actual text_encoder Configurations in Codebase

### Pattern 1: Empty Dict (Lazy Loading at Inference)

Most configs use `text_encoder=dict()` — this is the **standard approach** in this codebase:

```python
# configs/hymotion_t2m/hymotion_t2m_201dim_046b.py
model = dict(
    type='HyMotionT2MBundle',
    ...
    text_encoder=dict(),  # Empty: lazy-load at encode_text() time
)

# configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py
model = dict(
    type='HyMotionM2MBundle',
    ...
    text_encoder=dict(),
)

# configs/hymotion_m2m/_base_hymotion_m2m_046b.py
model = dict(
    type='HyMotionM2MBundle',
    ...
    # Use an empty dict as placeholder so child configs can override
    text_encoder=dict(),
)
```

**How it works** (from `bundle.py` lines 209-332):
1. Bundle stores `self._text_encoder_cfg = deepcopy(text_encoder)` during `__init__`
2. At first `encode_text()` call:
   - Checks if `_text_encoder_cfg is None` (empty dict → falsy)
   - If falsy, raises `RuntimeError("No text_encoder config provided")`
   - If truthy, lazy-instantiates `HYTextModel(**cfg)` on CPU
3. Subsequent calls reuse the same instance

### Pattern 2: Pre-Extracted Embeddings (No text_encoder Needed)

Some configs use **pre-extracted text embeddings** from `.pt` files and skip loading HYTextModel entirely:

```python
# configs/hymotion_m2m/hymotion_m2m_completion_caption_fm_046b.py
model = dict(
    pred_type='velocity',
    uncondition_mode=False,
    # No text_encoder needed: all embeddings are pre-extracted
    # text_encoder=dict() (inherited from base) stays falsy
    cond_mask_prob=0.3,
)

# Data pipeline uses:
pipeline=[
    dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
    # ... motion loading ...
]
```

These configs **inherit `text_encoder=dict()`** from base and never call `encode_text()` — embeddings come from pre-computed `.pt` files.

## Proper Explicit Configuration (If Needed)

If you want to explicitly configure HYTextModel instead of using the default, the full config would be:

```python
# Explicit Qwen3 + CLIP-L (default combination)
text_encoder=dict(
    type='HYTextModel',
    llm_type='qwen3',                   # Use Qwen3-8B for tokens
    max_length_llm=512,                 # Crop to 512 tokens
    sentence_emb_type='clipl',          # Use CLIP-L for sentence vectors
    max_length_sentence_emb=77,         # CLIP max (standard for CLIP)
    enable_llm_padding=True,            # Pad sequences to max_length
    torch_dtype=torch.float32,          # Keep full precision
)

# Alternative: Qwen3-Embedding (optimized for embeddings)
text_encoder=dict(
    type='HYTextModel',
    llm_type='qwen3_embedding',
    max_length_llm=256,
    sentence_emb_type='sentence_transformer',
    max_length_sentence_emb=128,
)
```

## Model Paths

The checkpoint paths are **hardcoded in `text_constants.py`** (`LLM_ENCODER_LAYOUT` and `SENTENCE_EMB_LAYOUT`):

```python
LLM_ENCODER_LAYOUT = {
    "qwen3": {
        "module_path": "checkpoints/Qwen3-8B",
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModelForCausalLM,
    },
    "qwen3_embedding": {
        "module_path": "checkpoints/Qwen3-Embedding-8B",
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModel,
    },
    # ...
}

SENTENCE_EMB_LAYOUT = {
    "clipl": {
        "module_path": "checkpoints/clip-vit-large-patch14",
        "tokenizer_class": CLIPTokenizer,
        "text_encoder_class": CLIPTextModel,
    },
    # ...
}
```

**Actual model files needed** (relative to project root):
- `checkpoints/Qwen3-8B/` — Qwen3 8B LLM
- `checkpoints/clip-vit-large-patch14/` — CLIP-L vision+text model
- `checkpoints/Qwen3-Embedding-8B/` — Alternative LLM

## Output Dimensions

HYTextModel returns a tuple: `(vtxt_raw, ctxt_raw, ctxt_length)`

| Output | Shape | Dim | Source | Purpose |
|--------|-------|-----|--------|---------|
| `vtxt_raw` (sentence embedding) | `(B, 1, 768)` | 768 | CLIP-L pooled | Sentence-level representation for attention |
| `ctxt_raw` (token embeddings) | `(B, max_tokens, 4096)` | 4096 | Qwen3 hidden states | Per-token contextual features |
| `ctxt_length` | `(B,)` | — | Token count | Actual token count (before padding) |

These match the model's expected dimensions:
- `vtxt_input_dim=768`
- `ctxt_input_dim=4096`

## For hymotion_t2m_201dim_046b.py

The config at `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` (line 61) has:
```python
text_encoder=dict(),  # Empty placeholder
```

**To fill this properly**, you would use:
```python
text_encoder=dict(
    type='HYTextModel',
    llm_type='qwen3',
    max_length_llm=512,
    sentence_emb_type='clipl',
    max_length_sentence_emb=77,
)
```

Or leave it empty if you plan to:
1. Not use text conditioning for this model, OR
2. Provide pre-extracted embeddings via dataset pipeline

## Key Implementation Details

### Lazy Loading (from bundle.py, lines 305-332)

```python
@torch.no_grad()
def encode_text(self, text: List[str]) -> Dict[str, Tensor]:
    """Lazy-load text encoder and encode text."""
    device = _get_module_device(self)
    if not hasattr(self, '_text_encoder') or self._text_encoder is None:
        if self._text_encoder_cfg is None:
            raise RuntimeError('No text_encoder config provided; cannot encode text.')
        from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
        cfg = deepcopy(self._text_encoder_cfg)
        cfg.pop('type', None)  # Remove 'type' key before passing to HYTextModel
        self._text_encoder = HYTextModel(**cfg)
        # Keep text encoder on CPU — inference-only, not in trainable graph
    vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
    return {
        'text_vec_raw': vtxt.to(device),
        'text_ctxt_raw': ctxt.to(device),
        'text_ctxt_raw_length': ctxt_len.to(device),
    }
```

**Key points**:
- Text encoder stays on **CPU** to save GPU memory (inference-only)
- Outputs moved to training device after encoding
- 'type' key removed before instantiation (MMEngine convention)
- Encodes text **without gradients** (`@torch.no_grad()`)

### Output Dimensions in Model

HunyuanMotionMMDiT expects:
- `ctxt_input_dim=4096` — Qwen3 token dimension (matches `ctxt_raw`)
- `vtxt_input_dim=768` — CLIP-L sentence dimension (matches `vtxt_raw`)

These are hardcoded in all configs and match the fixed text encoder paths.

## Summary Table

| Config File | text_encoder | Strategy | Encoding at Runtime |
|-------------|--------------|----------|-------------------|
| `hymotion_t2m_201dim_046b.py` | `dict()` | Lazy-load (default) | HYTextModel (Qwen3+CLIP) if `.encode_text()` called |
| `_base_hymotion_m2m_v2_046b.py` | `dict()` | Lazy-load (default) | Same |
| `hymotion_m2m_completion_caption_fm_046b.py` | `dict()` (inherited) | Pre-extracted | From `.pt` files, HYTextModel never loaded |
| Any caption config | `dict()` | Lazy-load | Qwen3+CLIP if text conditioning enabled |

## Recommended Configuration

**For text-conditioned inference with dynamic text**:
```python
model = dict(
    type='HyMotionT2MBundle',  # or HyMotionM2MBundle
    ...
    text_encoder=dict(
        type='HYTextModel',
        llm_type='qwen3',
        max_length_llm=512,
        sentence_emb_type='clipl',
        max_length_sentence_emb=77,
        enable_llm_padding=True,
    ),
    ...
)
```

**For pre-extracted embeddings (faster training, no Qwen3 load)**:
```python
# Leave text_encoder=dict() and use LoadPreExtractedTextEmbedding in pipeline
text_encoder=dict(),
pipeline=[
    dict(type='LoadPreExtractedTextEmbedding', key='caption', allow_none=True),
    ...
]
```

