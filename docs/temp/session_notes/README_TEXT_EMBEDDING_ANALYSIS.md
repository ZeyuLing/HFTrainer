# Text Embedding Data Flow Analysis - Complete Documentation

This repository contains comprehensive analysis of how text embeddings flow through HyMotion M2M v2 training pipeline.

## 📚 Documentation Files

### 1. **TEXT_EMBEDDING_DATA_FLOW_ANALYSIS.md** (Main Reference)
   - **Detailed, comprehensive analysis** of all 5 key components
   - Line-by-line code explanations
   - Before/after states at each stage
   - Expected file formats and data structures
   - Best for: Deep understanding, code review

### 2. **TEXT_EMBEDDING_QUICK_REFERENCE.md** (Quick Lookup)
   - **Fast lookup tables** and decision trees
   - Configuration examples
   - Critical gotchas highlighted
   - Three fallback mechanism chains
   - Best for: Quick checks, debugging, reference

### 3. **TEXT_EMBEDDING_VISUAL_GUIDE.md** (State Diagrams)
   - **ASCII diagrams** showing state machines
   - Flow charts with decision points
   - Shape evolution through pipeline
   - Three data paths visualized
   - Best for: Understanding high-level flow, presentations

## 🔍 Quick Questions Answered

### Q1: What happens when embeddings are missing?
**From LoadPreExtractedTextEmbedding with `allow_none=True`:**
- `text_vec_raw` → `torch.zeros(1, 768)`
- `text_ctxt_raw` → `torch.zeros(1, 4096)`
- `text_ctxt_raw_length` → `0`
- `_text_is_null` → `True`

**No error raised.** See: QUICK_REFERENCE.md § 1

---

### Q2: What's the difference between LoadCompatibleCaption and LoadPreExtractedTextEmbedding?
- **LoadPreExtractedTextEmbedding:** Loads **pre-computed embeddings** (Qwen3 + CLIP)
- **LoadCompatibleCaption:** Loads **raw caption text** only
- They're **complementary** — caption acts as fallback for online encoding

See: MAIN_ANALYSIS.md § 1-2

---

### Q3: Does PackInputs with `set_dummy_value=False` cause errors?
**No.** Missing keys are silently omitted. Trainer must handle gracefully:
```python
if batch.get('text_vec_raw') is not None:
    # use embeddings
elif 'caption' in batch:
    # fallback to online encoding
else:
    # use null embeddings
```

See: QUICK_REFERENCE.md § 3

---

### Q4: How does the trainer handle None text inputs?
**Trainer NEVER passes None to the model.** Three-level fallback:
1. Use pre-extracted embeddings (if available)
2. Online encode caption via Qwen3-8B (if caption available)
3. Use learned null embeddings (always available)

See: MAIN_ANALYSIS.md § 4, Steps 1-8

---

### Q5: When does the model see None?
**Never.** Model always receives tensors for `vtxt_input` and `ctxt_input`, even for unconditioned samples (filled with learned null embeddings).

See: MAIN_ANALYSIS.md § 5

---

## 🎯 The Three Critical Transforms

### 1️⃣ LoadPreExtractedTextEmbedding
- **Maps:** caption JSON → sibling .pt embedding file
- **Output:** text_vec_raw, text_ctxt_raw, text_ctxt_raw_length (or zeros if missing)
- **Key insight:** Gracefully fills zeros when embeddings unavailable

### 2️⃣ LoadCompatibleCaption
- **Maps:** caption JSON → caption text string
- **Output:** caption (randomly selected from variants)
- **Key insight:** Format-agnostic (supports hierarchical + HYMotion formats)

### 3️⃣ PackInputs
- **Filters:** Selects which keys make it into final batch
- **Behavior:** Silently omits missing keys (with `set_dummy_value=False`)
- **Key insight:** Trainer must check before using

## ⚙️ The Trainer's Three-Level Fallback

```
Level 1: Pre-extracted embeddings
├─ Check: batch.get('text_vec_raw') is not None?
├─ YES → Use directly (pad & replace nulls)
└─ NO → Try Level 2

Level 2: Online encoding from caption
├─ Check: 'caption' in batch?
├─ YES → Encode via bundle.encode_text(caption)
└─ NO → Try Level 3

Level 3: Learned null embeddings
└─ Use: bundle.null_vtxt_feat, bundle.null_ctxt_input
```

**Flow diagram:** VISUAL_GUIDE.md § 4

## 🔐 Critical Implementation Details

### Gotcha 1: Zero-Filled ≠ Valid Conditioning
```python
# BAD: Use zeros directly
# → Model sees same value for all null samples → CFG breaks

# GOOD: Replace with learned null embeddings
null_mask = (ctxt_length == 0)
vtxt_input = torch.where(null_mask, bundle.null_vtxt_feat, vtxt_input)
```

### Gotcha 2: Attention Mask Must Update When Text Dropped
```python
# When CFG drops text:
ctxt_mask_temporal[dropped] = False
ctxt_mask_temporal[dropped, 0] = True  # Only 1 position valid
# Ensures inference CFG matches training
```

### Gotcha 3: PackInputs Silently Omits Keys
```python
# DON'T: batch['text_vec_raw'][0]  # May not exist!
# DO:    if batch.get('text_vec_raw') is not None: ...
```

See: QUICK_REFERENCE.md § 7

## 📊 Data Path Visualization

```
Three Paths Through System:

Path A (Fast) ✅
  .json + .pt found → Use embeddings directly

Path B (Flexible) 🟡
  .json only → Online encode via Qwen3-8B

Path C (Unconditioned) 🔴
  Neither → Use learned null embeddings
```

See: VISUAL_GUIDE.md § 8

## 🧪 Configuration Examples

### Pre-Extracted Embeddings Pipeline:
```yaml
pipeline:
  - type: LoadPreExtractedTextEmbedding
    allow_none: true
  - type: LoadCompatibleCaption
    allow_none: true
  - type: PackInputs
    data_keys: [text_vec_raw, text_ctxt_raw, text_ctxt_raw_length]
    set_dummy_value: false
```

### Online Encoding Pipeline:
```yaml
pipeline:
  - type: LoadCompatibleCaption
    allow_none: true
  - type: PackInputs
    data_keys: [caption]
    set_dummy_value: false
```

See: QUICK_REFERENCE.md § 8

## 📍 Code Locations

| Component | File | Lines |
|-----------|------|-------|
| LoadPreExtractedTextEmbedding | load_text.py | 72-187 |
| LoadCompatibleCaption | load_text.py | 274-376 |
| PackInputs | formatting.py | 12-61 |
| Trainer prepare_and_forward | hymotion_m2m_trainer.py | 49-244 |
| Bundle predict_flow | bundle.py | 502-534 |
| Bundle mask_text_cond | bundle.py | 331-392 |
| Bundle encode_text | bundle.py | 302-329 |

## 🚀 Next Steps

1. **Quick question?** → See QUICK_REFERENCE.md
2. **Understanding flow?** → See VISUAL_GUIDE.md
3. **Deep dive?** → See MAIN_ANALYSIS.md
4. **Implementing/debugging?** → Check code locations table above

---

**Version:** 1.0  
**Last Updated:** 2026-05-18  
**Analyzed Components:** HyMotion M2M v2 text embedding pipeline
