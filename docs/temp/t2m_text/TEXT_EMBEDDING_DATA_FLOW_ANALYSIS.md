# HyMotion M2M v2 Text Embedding Data Flow Analysis

## Overview
This document traces how text embeddings (`text_vec_raw`, `text_ctxt_raw`) flow through HyMotion M2M v2 training, from dataset loading through model inference.

---

## 1. LoadPreExtractedTextEmbedding Class
**File:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py` (lines 72-187)

### What It Does
Loads pre-extracted Qwen3+CLIP text embeddings from `.pt` files. Maps caption JSON paths to sibling embedding files using `CAPTION_TO_QWEN3_DIR` mapping.

### When `allow_none=True` (Default)
If embeddings are **missing** and `allow_none=True`:
- **Returns:** `_fill_null_embedding(results)` which sets:
  - `text_vec_raw` → `torch.zeros(1, vtxt_dim)` [Default: (1, 768)]
  - `text_ctxt_raw` → `torch.zeros(1, ctxt_dim)` [Default: (1, 4096)]
  - `text_ctxt_raw_length` → `torch.tensor(0)` [Scalar]
  - `_text_is_null` → `True` (marker flag)
- **Does NOT raise an error** — silently falls back to zero-filled tensors

### When `allow_none=False`
If embeddings are **missing** and `allow_none=False`:
- **Raises:** `ValueError` with message: `"LoadPreExtractedTextEmbedding: '<key>_path' not found in results"`

### Keys Set in Data Dict
When embedding file **successfully loads**:
1. `text_vec_raw` — Sentence-level CLIP-L embedding, shape `(1, 768)`
2. `text_ctxt_raw` — Token-level Qwen3 embedding, shape `(seq_len, 4096)` [variable length]
3. `text_ctxt_raw_length` — Actual token sequence length, shape `(scalar)` [e.g., 15 tokens]
4. `caption` — Caption string (for logging/CFG compatibility)

When embedding file **is missing or null**:
- Same keys as above, but with zero-filled tensors (see above)
- `_text_is_null=True` marker

### File Format Expected in `.pt`
```python
data['result'][i] = {
    'caption': str,
    'text_embedding': {
        'text_vec_raw':         Tensor[1, 1, 768],    # CLIP-L (squeezed from batch dim)
        'text_ctxt_raw':        Tensor[1, seq, 4096], # Qwen3 (squeezed from batch dim)
        'text_ctxt_raw_length': Tensor[1],            # Token count (squeezed)
    },
    ...
}
```

### Critical Implementation Detail
The transform **randomly selects** one caption variant from `data['result']` list for data augmentation:
```python
idx = random.randint(0, len(result_list) - 1)
item = result_list[idx]
```

---

## 2. LoadCompatibleCaption Class
**File:** `hftrainer/datasets/motion/motionhub/transforms/load_text.py` (lines 274-376)

### What It Does
Loads captions in a **flexible format-agnostic** manner. Accepts two caption JSON formats:
1. **Hierarchical format:** Contains `"macro"`, `"meso"`, `"micro"` keys
2. **HYMotion format:** Contains `"result"` array with caption variants

### Difference: `allow_none=True` vs `allow_none=False`

#### `allow_none=True`
- If `caption_path` is `None` → **Returns immediately** without raising error
- `results` dict remains **unchanged**
- No caption key is added

#### `allow_none=False`
- If `caption_path` is `None` → **Raises `ValueError`**
- If caption file exists but format doesn't match either expected schema → **Raises `ValueError`** with detailed message:
  ```
  "does not match either format:
   - LoadHierarchicalCaption: requires 'macro', 'meso', 'micro' keys
   - LoadHYMotionCaption: requires 'result' array with 'short_caption' or 'short_caption_rewritten'"
  ```

### Keys Set in Data Dict

#### Hierarchical Format Output:
- `caption` — Randomly selected caption string
- `granularity` — Granularity level of selected caption (`"macro"`, `"meso"`, or `"micro"`)
- `caption_list` — All available captions (for reference)
- `granularity_list` — Corresponding granularity levels

#### HYMotion Format Output:
- `caption` — Randomly selected caption string
- `caption_list` — All available captions

### Key Difference from LoadPreExtractedTextEmbedding
- **LoadCompatibleCaption** loads **raw caption TEXT** only
- **LoadPreExtractedTextEmbedding** loads **pre-computed EMBEDDINGS** (Qwen3 + CLIP-L vectors)
- These are **complementary transforms** in the pipeline — caption can be used as fallback for online encoding if embeddings are unavailable

---

## 3. PackInputs Class
**File:** `hftrainer/datasets/motion/motionhub/transforms/formatting.py` (lines 12-61)

### What It Does
Packs selected fields into the final batch dict for trainer. Converts numpy arrays to torch tensors and filters which keys are passed forward.

### How It Handles Missing Keys

#### When `set_dummy_value=False` (Default)
```python
for k in self.keys:
    value = results.get(k, None)
    if value is not None:
        # Convert and pack
        if isinstance(value, np.ndarray):
            packed[k] = torch.from_numpy(value)
        else:
            packed[k] = value
    else:
        # Key is missing → SKIP IT (don't add to packed dict)
```
- **Missing key is silently omitted** from packed dict
- **No error raised**
- Downstream trainer must handle missing keys gracefully

#### When `set_dummy_value=True`
```python
for k in self.keys:
    value = results.get(k, None)
    if value is not None:
        packed[k] = value  # pack normally
    else:
        packed[k] = self.dummy_value  # set to dummy (e.g., None)
```
- **Missing key is set to `dummy_value`** (default: `None`)
- Used in multi-task training to align batch keys

### Keys Packed
Packs values from three categories:
1. **`keys`** — Main data keys (motion, masks, etc.)
2. **`meta_keys`** — Metadata (paths, names)
3. **`data_keys`** — Additional data fields (captions, embeddings)

### Critical Data Flow
For text embeddings, typical config would be:
```python
keys=['src_motion', 'tgt_motion', 'src_mask', 'tgt_length', 'src_length']
data_keys=['text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length', 'caption']
```
- If embeddings were **loaded successfully** → all keys packed normally
- If embeddings **were null-filled** → still packed (as zero tensors with length=0)
- If embeddings **never seen** → keys omitted, trainer must have fallback logic

---

## 4. Trainer: HyMotionM2MTrainer
**File:** `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (lines 49-244)

### How Trainer Uses Text Embeddings

#### Step 1: Check for Pre-Extracted Embeddings (Line 138)
```python
if batch.get('text_vec_raw') is not None:
    # Pre-encoded text vectors already in batch
    vtxt_input = batch['text_vec_raw'].to(device)  # (B, 1, 768)
    ctxt_raw = batch['text_ctxt_raw']              # List or Tensor
```

#### Step 2: Handle Variable-Length Contexts (Lines 149-162)
Since pre-extracted embeddings are **variable-length** (different captions have different token counts):
- If `ctxt_raw` is a **list** of tensors:
  ```python
  # Pad all to max_text_len (default 128)
  feat_dim = ctxt_raw[0].shape[-1]
  ctxt_padded = ctxt_raw[0].new_zeros(len(ctxt_raw), pad_len, feat_dim)
  for i, t in enumerate(ctxt_raw):
      seq = min(t.shape[0], pad_len)
      ctxt_padded[i, :seq] = t[:seq]
  ctxt_input = ctxt_padded.to(device)
  ```
- If `ctxt_raw` is already a **stacked tensor**:
  ```python
  if cur_len < pad_len:
      ctxt_input = F.pad(ctxt_raw, (0, 0, 0, pad_len - cur_len)).to(device)
  else:
      ctxt_input = ctxt_raw[:, :pad_len].to(device)
  ```

#### Step 3: Build Attention Mask (Line 163)
```python
ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
# Result: (B, pad_len) boolean mask, True = valid token, False = padding
```

#### Step 4: Replace Null Samples with Learned Null Embeddings (Lines 169-178)
```python
null_mask = (ctxt_length == 0)  # Identify null samples
if null_mask.any():
    null_v = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
    null_c = self.bundle.null_ctxt_input.expand_as(ctxt_input)
    vtxt_input = torch.where(null_mask.view(B, 1, 1).expand_as(vtxt_input), null_v, vtxt_input)
    ctxt_input = torch.where(null_mask.view(B, 1, 1).expand_as(ctxt_input), null_c, ctxt_input)
```
**Why?** Zero-filled tensors from `LoadPreExtractedTextEmbedding` are **not valid conditioning**. The model's learned null embeddings (`self.bundle.null_vtxt_feat`, `self.bundle.null_ctxt_input`) are used instead for classifier-free guidance (CFG).

#### Step 5: Apply CFG Dropout (Lines 180-185)
```python
vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    force_mask=False,
    cond_mask_prob=self.bundle.cond_mask_prob,  # CFG dropout rate
    return_text_available=True,
)
```
Randomly drops text during training (classifier-free guidance):
- Some samples get **real text** → use `vtxt_input`, `ctxt_input`
- Some samples get **null embeddings** → use `self.bundle.null_vtxt_feat`, `self.bundle.null_ctxt_input`

#### Step 6: Update Attention Mask for Dropped Samples (Lines 193-197)
```python
if not text_available.all():
    dropped_samples = ~text_available
    ctxt_mask_temporal = ctxt_mask_temporal.clone()
    ctxt_mask_temporal[dropped_samples] = False
    ctxt_mask_temporal[dropped_samples, 0] = True  # Only 1 position valid
```
**Critical fix:** When text is dropped, the attention mask is narrowed to just 1 position (the null embedding) to match inference-time CFG behavior.

#### Step 7: Fallback — Online Encoding (Lines 199-237)
If `text_vec_raw` is **missing** from batch:
```python
elif 'caption' in batch and batch['caption'] is not None:
    # Online text encoding from raw captions
    captions = batch['caption']
    # ... convert to list, handle None entries ...
    with torch.no_grad():
        text_feats = self.bundle.encode_text(captions)
    vtxt_input = text_feats['text_vec_raw'].to(device)
    ctxt_input = text_feats['text_ctxt_raw'].to(device)
    # ... build mask and apply CFG ...
```

#### Step 8: Full Null Embedding Fallback (Lines 239-244)
If neither `text_vec_raw` nor `caption` is available:
```python
else:
    vtxt_input = self.bundle.null_vtxt_feat.expand(B, 1, -1)
    ctxt_input = self.bundle.null_ctxt_input.expand(B, 1, -1)
    ctxt_length = torch.tensor([1], device=device).expand(B)
    ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)
    text_available = torch.zeros(B, dtype=torch.bool, device=device)
```
All samples use learned null embeddings (unconditioned).

---

## 5. Model: HyMotionM2M Bundle Forward Pass
**File:** `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 302-534)

### How Model Uses Text Embeddings

#### Method 1: `encode_text()` (Lines 302-329)
Called for online encoding when pre-extracted embeddings are unavailable:
```python
def encode_text(self, text: List[str]) -> Dict[str, Tensor]:
    # Lazy-load text encoder (Qwen3-8B) on CPU
    device = _get_module_device(self)
    if not hasattr(self, '_text_encoder') or self._text_encoder is None:
        from hftrainer.models.motion.hymotion_m2m.network.text_encoder import HYTextModel
        cfg = deepcopy(self._text_encoder_cfg)
        cfg.pop('type', None)
        self._text_encoder = HYTextModel(**cfg)
    
    vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
    return {
        'text_vec_raw': vtxt.to(device),
        'text_ctxt_raw': ctxt.to(device),
        'text_ctxt_raw_length': ctxt_len.to(device),
    }
```
**Key insight:** Text encoder runs on **CPU** (not GPU) to avoid VRAM exhaustion with 8B LLM.

#### Method 2: `mask_text_cond()` (Lines 331-392)
Applies classifier-free guidance masking:
```python
def mask_text_cond(self, vtxt, ctxt, force_mask=False, cond_mask_prob=0.0, 
                   return_text_available=False):
    bs = vtxt.shape[0]
    text_available = torch.ones(bs, dtype=torch.bool, device=vtxt.device)
    
    if force_mask:
        # Force all samples to null embeddings
        return (
            self.null_vtxt_feat.expand(*vtxt.shape),
            self.null_ctxt_input.expand(*ctxt.shape),
        )
    
    if self.training and cond_mask_prob > 0.0:
        # Randomly drop text for CFG
        mask = torch.bernoulli(torch.ones(bs, device=vtxt.device) * cond_mask_prob)
        text_available = ~mask.squeeze(-1)
        # ... replace with null embeddings ...
    
    if return_text_available:
        return vtxt, ctxt, text_available
    return vtxt, ctxt
```

#### Method 3: `predict_flow()` (Lines 502-534)
Forward pass through transformer with text conditioning:
```python
def predict_flow(self, x_input, ctxt_input, vtxt_input, timesteps,
                 x_mask_temporal=None, ctxt_mask_temporal=None, 
                 mask_density=None):
    return self.motion_transformer(
        x=x_input,
        ctxt_input=ctxt_input,        # (B, L_c, D_c) token embeddings
        vtxt_input=vtxt_input,        # (B, 1, D_v) sentence embedding
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,  # (B, L_c) attention mask
        mask_density=mask_density,
    )
```

### What Happens When Text Inputs Are None or Missing

#### Pre-Forward Check (Trainer Step 8, lines 239-244)
If neither pre-extracted embeddings nor caption is available, the trainer **pre-fills** with learned null embeddings **before calling the model**. The model **never sees None**.

#### At Model Time
The `predict_flow()` method receives:
- `vtxt_input` — Always a tensor (never None), shape `(B, 1, 768)`
- `ctxt_input` — Always a tensor (never None), shape `(B, L_c, 4096)`
- `ctxt_mask_temporal` — Always a mask, shape `(B, L_c)`, may have all False for unconditioned samples

#### When Unconditioned
All `ctxt_input` and `vtxt_input` are filled with `self.bundle.null_vtxt_feat` and `self.bundle.null_ctxt_input` (trainable parameters initialized with small random values).

The model's **cross-attention** layer then:
- Receives the null embeddings
- Applies attention using `ctxt_mask_temporal` (which may be all False)
- Returns predictions conditioned on "no real text"

This allows **classifier-free guidance (CFG)** at inference time:
```
motion = pred_real_text + cfg_scale * (pred_real_text - pred_null)
```

---

## Data Flow Diagram

```
Dataset Pipeline:
├─ caption_path: "data/captions/sample.json"
├─ LoadCompatibleCaption
│  └─ caption: "person walks forward"
├─ LoadPreExtractedTextEmbedding
│  ├─ Maps to: "data/qwen3_augmented/sample.pt"
│  ├─ If found:
│  │  ├─ text_vec_raw: (1, 768)       [CLIP-L]
│  │  ├─ text_ctxt_raw: (seq, 4096)   [Qwen3 tokens]
│  │  └─ text_ctxt_raw_length: scalar [e.g., 15]
│  └─ If not found (allow_none=True):
│     ├─ text_vec_raw: zeros(1, 768)
│     ├─ text_ctxt_raw: zeros(1, 4096)
│     └─ text_ctxt_raw_length: tensor(0)
├─ PackInputs(set_dummy_value=False)
│  └─ packed_dict: all non-None fields
└─ Trainer receives batch

Trainer Processing:
├─ Check: batch.get('text_vec_raw') is not None?
├─ YES:
│  ├─ Load embeddings from batch
│  ├─ Pad ctxt_raw to max_text_len=128
│  ├─ Build attention mask from ctxt_raw_length
│  ├─ Replace null samples (length==0) with learned null embeddings
│  ├─ Apply CFG dropout via mask_text_cond()
│  ├─ Update attention mask for dropped samples
│  └─ → vtxt_input, ctxt_input, ctxt_mask_temporal → Model
├─ NO (else check caption):
│  ├─ Use bundle.encode_text(captions) for online encoding
│  └─ → vtxt_input, ctxt_input, ctxt_mask_temporal → Model
└─ NO (else fallback):
   ├─ Use learned null embeddings
   └─ → vtxt_input, ctxt_input, ctxt_mask_temporal → Model

Model Forward:
├─ predict_flow(
│  ├─ x_input: (B, L, D+3*D)           [motion + VACE context]
│  ├─ ctxt_input: (B, 128, 4096)       [padded token embeddings]
│  ├─ vtxt_input: (B, 1, 768)          [sentence embedding]
│  ├─ ctxt_mask_temporal: (B, 128)     [valid token mask]
│  └─ ...
│  ) → pred: (B, L, D)
└─ Loss computation
```

---

## Key Takeaways

1. **LoadPreExtractedTextEmbedding** with `allow_none=True` gracefully handles missing embeddings by filling with zeros + marking `text_ctxt_raw_length=0` to signal null-ness.

2. **LoadCompatibleCaption** loads raw caption text (separate from embeddings) and supports format flexibility. Useful as fallback for online encoding.

3. **PackInputs** with `set_dummy_value=False` silently skips missing keys; trainers must handle gracefully.

4. **Trainer** has three fallback mechanisms:
   - Level 1: Use pre-extracted embeddings (fastest)
   - Level 2: Online encode caption via Qwen3-8B (slower, more flexible)
   - Level 3: Use learned null embeddings (unconditioned)

5. **Model** never sees None for text inputs; trainer pre-fills with null embeddings before forwarding.

6. **Null samples** (no caption or failed extraction) are replaced with **trainable null embeddings** (`self.bundle.null_vtxt_feat`, `self.bundle.null_ctxt_input`), enabling CFG at both training and inference.

7. **CFG dropout** during training (mask_text_cond) updates attention masks to ensure train/inference distribution matching.

