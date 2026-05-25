# Text Embedding Data Flow - Quick Reference

## 1. LoadPreExtractedTextEmbedding: What Gets Set

### When embeddings are **FOUND**:
```python
results['text_vec_raw']           # (1, 768) CLIP-L sentence embedding
results['text_ctxt_raw']          # (seq_len, 4096) Qwen3 token embeddings  
results['text_ctxt_raw_length']   # scalar: actual token count (e.g., 15)
results['caption']                # str: caption text
results['_text_is_null']          # False
```

### When embeddings are **MISSING** (`allow_none=True`):
```python
results['text_vec_raw']           # (1, 768) zeros
results['text_ctxt_raw']          # (1, 4096) zeros  
results['text_ctxt_raw_length']   # 0 (signals null)
results['_text_is_null']          # True
# NO ERROR RAISED
```

### When embeddings are **MISSING** (`allow_none=False`):
```python
raise ValueError("LoadPreExtractedTextEmbedding: 'caption_path' not found...")
```

---

## 2. LoadCompatibleCaption: What Gets Set

### Behavior with `allow_none=True`:
- Missing path → returns immediately, **no keys added**
- No error

### Behavior with `allow_none=False`:
- Missing path → **raises ValueError**
- Invalid format → **raises ValueError**

### Output when **successful**:
```python
results['caption']          # str: randomly selected caption
results['caption_list']     # list: all available captions

# ONLY for hierarchical format:
results['granularity']      # str: 'macro'|'meso'|'micro'
results['granularity_list'] # list: granularities for each caption
```

---

## 3. PackInputs: Missing Key Behavior

### With `set_dummy_value=False` (DEFAULT):
```python
# Missing key → SILENTLY OMITTED from packed dict
packed_dict = {k: v for k, v in results.items() if v is not None}
```

### With `set_dummy_value=True`:
```python
# Missing key → SET TO dummy_value (default None)
packed_dict[missing_key] = self.dummy_value  # e.g., None
```

---

## 4. Trainer Text Embedding Logic

### Fallback Chain:
```
1. IF batch.get('text_vec_raw') is not None:
     → Use pre-extracted embeddings
     → Pad to max_text_len=128
     → Replace null samples (length==0) with learned nulls
     → Apply CFG dropout
   
2. ELSE IF 'caption' in batch and batch['caption'] is not None:
     → Online encode via bundle.encode_text()
     → Apply CFG dropout
   
3. ELSE:
     → Use learned null embeddings (bundle.null_vtxt_feat, bundle.null_ctxt_input)
     → Set ctxt_mask_temporal all False except position 0
```

### Key Replacements:
```python
# Step 4: Replace zero-filled with learned nulls
null_mask = (ctxt_length == 0)
if null_mask.any():
    vtxt_input = torch.where(
        null_mask.view(B, 1, 1), 
        self.bundle.null_vtxt_feat.expand_as(vtxt_input),
        vtxt_input
    )
    ctxt_input = torch.where(
        null_mask.view(B, 1, 1),
        self.bundle.null_ctxt_input.expand_as(ctxt_input), 
        ctxt_input
    )

# Step 5: Apply CFG dropout
vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    cond_mask_prob=self.bundle.cond_mask_prob
)

# Step 6: Update mask for dropped samples
if not text_available.all():
    dropped = ~text_available
    ctxt_mask_temporal[dropped] = False
    ctxt_mask_temporal[dropped, 0] = True  # Only position 0 valid
```

---

## 5. Model Text Input Guarantee

**The model's `predict_flow()` NEVER receives None:**

- `vtxt_input`: Always (B, 1, 768) tensor
- `ctxt_input`: Always (B, L_c, 4096) tensor  
- `ctxt_mask_temporal`: Always (B, L_c) mask

When unconditioned:
- All embeddings are `bundle.null_vtxt_feat`, `bundle.null_ctxt_input`
- Attention mask may be all False (full padding)
- Model learns "no text" via CFG gradient signal

---

## 6. Real-World Data Paths

### With Pre-Extracted Embeddings:
```
Dataset: caption_path = "data/captions/sample.json"
         ↓
LoadPreExtractedTextEmbedding: Maps to "data/qwen3_augmented/sample.pt"
         ↓
LoadCompatibleCaption: Loads caption text (fallback for online encode)
         ↓
Trainer: Uses pre-extracted embeddings (fast)
```

### Without Pre-Extracted Embeddings:
```
Dataset: caption_path = "data/captions/sample.json"
         ↓
LoadPreExtractedTextEmbedding: Not found → zeros + length=0
         ↓
LoadCompatibleCaption: Loads caption text from JSON
         ↓
Trainer: Detects null (length==0) → Falls back to bundle.encode_text(caption)
         (or uses learned nulls if caption also missing)
```

---

## 7. Critical Gotchas

### ❌ Gotcha 1: Zero-Filled Tensors ≠ Valid Conditioning
```python
# BAD: Just using zero-filled text embeddings directly
# → Model sees same values for all null samples → CFG breaks

# GOOD: Trainer replaces zeros with learned null embeddings
null_mask = (ctxt_length == 0)
vtxt_input = torch.where(null_mask, bundle.null_vtxt_feat, vtxt_input)
```

### ❌ Gotcha 2: Attention Mask Mismatch
```python
# BAD: Original mask built from pre-extracted caption lengths
# When text is dropped via CFG, embeddings become nulls but mask unchanged
# → Model attends to wrong positions

# GOOD: Trainer updates mask for dropped samples
if not text_available.all():
    ctxt_mask_temporal[dropped] = False
    ctxt_mask_temporal[dropped, 0] = True
```

### ❌ Gotcha 3: PackInputs Silently Omits Keys
```python
# BAD: Assuming all keys are always in batch
batch['text_vec_raw'][0]  # KeyError if PackInputs skipped it

# GOOD: Check before using
if batch.get('text_vec_raw') is not None:
    use_embeddings()
elif 'caption' in batch:
    encode_online()
else:
    use_null_embeddings()
```

---

## 8. Configuration Examples

### For Pre-Extracted Embeddings:
```yaml
# dataset config
pipeline:
  - type: LoadPreExtractedTextEmbedding
    key: caption
    allow_none: true
    vtxt_dim: 768
    ctxt_dim: 4096
  - type: LoadCompatibleCaption
    key: caption
    allow_none: true
  - type: PackInputs
    keys: [src_motion, tgt_motion, src_mask, ...]
    data_keys: [text_vec_raw, text_ctxt_raw, text_ctxt_raw_length, caption]
    set_dummy_value: false
```

### For Online Encoding (No Pre-Extracted):
```yaml
pipeline:
  - type: LoadCompatibleCaption
    key: caption
    allow_none: true
  - type: PackInputs
    keys: [src_motion, tgt_motion, src_mask, ...]
    data_keys: [caption]
    set_dummy_value: false
# Trainer falls back to bundle.encode_text(caption)
```

---

## Quick Decision Tree

```
Does batch have 'text_vec_raw'?
├─ YES → Is text_ctxt_raw_length > 0?
│        ├─ YES → Use real embeddings (pre-extracted path)
│        └─ NO → Replace with null_vtxt_feat, null_ctxt_input
│
└─ NO → Does batch have 'caption'?
         ├─ YES → Encode online via bundle.encode_text(caption)
         └─ NO → Use null_vtxt_feat, null_ctxt_input
```

