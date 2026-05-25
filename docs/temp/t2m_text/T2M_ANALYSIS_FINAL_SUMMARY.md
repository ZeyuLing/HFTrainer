# 🎯 Official HunyuanMotion T2M Training Code Analysis - FINAL SUMMARY

**Generated:** May 15, 2026  
**Status:** ✅ COMPLETE  
**Repository:** `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## 📌 Executive Summary

The official HunyuanMotion T2M (Text-to-Motion) training code is **embedded within the hftrainer repository** as the local implementation, NOT in a separate ref_repo directory as initially hypothesized. The analysis successfully located, extracted, and documented all critical components of the T2M training pipeline.

### Key Discovery

The T2M architecture is a **specialized variant of M2M with the following key differences:**

| Aspect | M2M (Motion-to-Motion) | T2M (Text-to-Motion) |
|--------|----------------------|---------------------|
| **Motion Input Dim** | 135 | 201 |
| **VACE Conditioning** | ✅ Yes (4×motion_dim) | ❌ No |
| **Model Input** | x_t + VACE context | x_t only |
| **Text Conditioning** | ✅ Qwen3 + CLIP-L | ✅ Qwen3 + CLIP-L |
| **Null Embeddings** | ✅ Used for CFG | ✅ Used for CFG |
| **Padding** | 128 max_text_len | 128 max_text_len |

---

## 📁 Official T2M Files Located

### Bundle (Model Architecture & Forward Passes)
**File:** `hftrainer/models/motion/hymotion_t2m/bundle.py`

**Core Classes:**
- `HyMotionT2MBundle(ModelBundle)` — Main model bundle for T2M

**Key Methods:**
- `encode_text(text: List[str])` — Text encoding via HYTextModel
- `mask_text_cond()` — CFG dropout via Bernoulli masking  
- `predict_flow()` — Single forward pass through MMDiT
- `decode_motion_from_latent()` — FK to 3D keypoints
- `normalize_motion()` / `denormalize_motion()` — Motion normalization

### Trainer (Training Loop & Data Processing)
**File:** `hftrainer/trainers/motion/hymotion_t2m_trainer.py`

**Core Class:**
- `HyMotionT2MTrainer(BaseTrainer)` — Official T2M training loop

**Key Methods:**
- `_prepare_and_forward(batch)` — Prepare batch and forward pass
- `_prepare_text_encoding()` — 3-path text conditioning logic
- `train_step()` — Single training step

### Text Encoder (Dual-Encoder System)
**File:** `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py`

**Core Class:**
- `HYTextModel(nn.Module)` — Dual-encoder text embedding

**Encoders:**
- Sentence encoder: CLIP-L (768-dim, normalized)
- LLM encoder: Qwen3-8B (4096-dim, raw)

---

## 🔍 Critical Text Processing Details

### 1. Dual-Encoder Architecture

```
Input Text (List[str])
    ↓
CLIP-L Tokenizer + Encoder
    ↓
vtxt (B, 1, 768) — sentence-level embedding, normalized
    
Input Text
    ↓
Qwen3 Tokenizer + Model
    ↓
ctxt (B, Lc, 4096) — token-level embeddings (LLM)
    ↓
ctxt_length (B,) — actual sequence length
```

**Key Characteristics:**
- **vtxt (Sentence Vector):** Pooled CLIP-L output, L2-normalized to unit sphere
- **ctxt (Context Tokens):** Raw Qwen3 hidden states, NOT normalized
- **ctxt_length:** Pre-padding sequence length (for padding mask)

### 2. Text Padding Convention

**Default:** `max_text_len = 128`

```python
# Padding happens in trainer, not in text encoder
if cur_len < pad_len:
    ctxt_padded = F.pad(ctxt_raw, (0, 0, 0, pad_len - cur_len))  # right-pad
else:
    ctxt_padded = ctxt_raw[:, :pad_len]  # truncate
```

**Padding Mask:** `[True] * ctxt_length + [False] * (pad_len - ctxt_length)`

### 3. Classifier-Free Guidance (CFG) Dropout

**Implementation:** Bernoulli masking on batch dimension

```python
def mask_text_cond(self, vtxt, ctxt, cond_mask_prob=0.1):
    bs = vtxt.shape[0]
    mask = torch.bernoulli(
        torch.ones(bs, device=vtxt.device) * cond_mask_prob
    ).view(bs, 1).bool()
    
    # Where mask=1, replace with null embeddings (unconditional)
    vtxt_masked = where(mask, self.null_vtxt_feat.expand_as(vtxt), vtxt)
    ctxt_masked = where(mask, self.null_ctxt_input.expand_as(ctxt), ctxt)
    
    return vtxt_masked, ctxt_masked
```

**Null Embeddings:**
- `self.null_vtxt_feat` — nn.Parameter (1, 1, 768), zero-initialized
- `self.null_ctxt_input` — nn.Parameter (1, 1, 4096), zero-initialized

### 4. Three Text Conditioning Paths

The trainer implements a sophisticated 3-path text conditioning system:

**Path 1: Pre-Encoded Text from Batch**
```
Batch contains: text_vec_raw, text_ctxt_raw, text_ctxt_raw_length
    → Direct use, no encoding needed
    → Fastest path (pre-computed embeddings)
    → Used in production inference
```

**Path 2: Online Encoding from Captions**
```
Batch contains: caption field
    → call bundle.encode_text(captions)
    → Returns: {text_vec_raw, text_ctxt_raw, text_ctxt_raw_length}
    → Used in training with text conditioning
    → Slower but necessary for caption training
```

**Path 3: Null Embeddings (Unconditional)**
```
When caption is empty or zero-length
    → use bundle.null_vtxt_feat and bundle.null_ctxt_input
    → Implements unconditional generation
    → Also applied via CFG dropout masking
```

### 5. Text Embedding Normalization

**CLIP-L (Sentence Embedding):**
- ✅ L2-normalized to unit vector
- Location: `text_encoder.py` line 254: `normalize(sentence_embeddings, p=2, dim=1)`
- Output shape: `(B, 1, 768)` with norm ≈ 1.0

**Qwen3 (Token Embeddings):**
- ❌ NOT normalized (raw hidden state)
- Output shape: `(B, Lc, 4096)` with arbitrary magnitude
- Pre-processing: `apply_chat_template()` for proper prompt formatting

---

## 🔄 Training Flow (T2M)

### 1. Data Loading
```
Dataset → caption field (optional)
       → motion field (201-dim, T frames)
       → frame padding to max_len=360
       → num_frames preserved (for masking)
```

### 2. Text Conditioning (3 paths)
```
Path 1: Pre-encoded from batch
        ↓ (fastest)
        → vtxt (B,1,768), ctxt (B,128,4096), ctxt_len (B,)

Path 2: Online caption encoding
        ↓
        → extract captions
        → call bundle.encode_text()
        → pad ctxt to 128 dims
        → vtxt (B,1,768), ctxt (B,128,4096), ctxt_len (B,)

Path 3: Unconditional (empty caption or CFG mask)
        ↓
        → use null_vtxt_feat, null_ctxt_input
        → vtxt (B,1,768), ctxt (B,128,4096), all-zeros
```

### 3. CFG Dropout Masking
```
Apply mask_text_cond(vtxt, ctxt, cond_mask_prob=0.1)
    ↓
Bernoulli(p=0.1) for each batch element
    ↓
mask=1 batch elements → replace with null embeddings
    ↓
10% of batch becomes unconditional (for classifier-free guidance)
```

### 4. Transformer Forward
```
predict_flow(
    x_input=x_t,                      # (B, L, 201) motion only
    ctxt_input=ctxt,                  # (B, 128, 4096) token embeddings
    vtxt_input=vtxt,                  # (B, 1, 768) sentence embedding
    timesteps=t,
    x_mask_temporal=padding_mask,
    ctxt_mask_temporal=text_padding_mask
)
    ↓
Output: (B, L, 201) velocity or x1 prediction
```

### 5. Loss Computation
```
pred vs target using SmoothL1 or MSE
    ↓
Weighted by:
  - data_mask_temporal (frame padding)
  - generation_mask (T2M: all 1s)
    ↓
Backprop → optimizer.step()
```

---

## 🆚 M2M vs T2M Comparison

### Text Processing (IDENTICAL)
✅ Both use the exact same text encoding and padding logic
✅ Both use Qwen3-8B (4096-dim) + CLIP-L (768-dim)
✅ Both apply CFG dropout with Bernoulli masking
✅ Both have null embeddings as nn.Parameters
✅ Both pad to max_text_len=128

### Motion Processing (KEY DIFFERENCE)
❌ M2M: 135-dim (absolute transl + rot6d)
❌ T2M: 201-dim (relative transl + rot6d for 33-joint SMPL)

### Model Input (KEY DIFFERENCE)
❌ M2M: x_input = cat([x_t, VACE_context], dim=-1) = (B,L,540)
❌ T2M: x_input = x_t = (B,L,201)

---

## 📊 Text Embedding Dimensions

| Component | Type | Dimension | Notes |
|-----------|------|-----------|-------|
| **vtxt** (sentence) | float32 | (B, 1, 768) | CLIP-L pooled, L2-normalized |
| **ctxt** (tokens) | float32 | (B, 128, 4096) | Qwen3 LLM, NOT normalized |
| **ctxt_length** | int64 | (B,) | Actual pre-padding token count |
| **null_vtxt_feat** | nn.Parameter | (1, 1, 768) | Zero-initialized, fixed |
| **null_ctxt_input** | nn.Parameter | (1, 1, 4096) | Zero-initialized, fixed |

---

## ⚠️ Critical Implementation Details

### 1. Null Embedding Initialization
- Located in: `HyMotionT2MBundle.__init__()`
- Initialized as: `nn.Parameter(torch.zeros(...))`
- **Purpose:** Serve as unconditional conditioning for CFG and empty captions
- **Critical:** Must be loaded from pretrained checkpoint, not random

### 2. Text Padding Order of Operations
```
1. Encode text → vtxt (B,1,768), ctxt (B,Lc,4096), ctxt_len (B,)
2. Compute ctxt_length from actual encoding (pre-padding)
3. Pad ctxt to max_text_len=128 using ctxt_length
4. Build padding_mask from ctxt_length
5. Apply padding_mask in transformer attention
```

### 3. CFG Masking Broadcast
```
mask (B,1) → unsqueeze until matches vtxt/ctxt ndim
    ↓
mask_vtxt (B,1,1) for vtxt (B,1,768)
mask_ctxt (B,1,1) for ctxt (B,128,4096)
    ↓
Element-wise where() operation
```

### 4. Motion Normalization
- Uses Mean/Std buffers loaded from pretrained checkpoint
- Near-zero std dimensions are zeroed after normalization
- Must match between training and inference

---

## 🚀 Key Takeaways for Implementation

### For Text-Conditioned T2M
1. **Use HYTextModel** for robust dual-encoder text encoding
2. **Pad to 128** for text tokens (matches training distribution)
3. **Apply L2-norm** only to sentence embeddings (vtxt), NOT LLM tokens (ctxt)
4. **Implement 3-path conditioning** for flexibility (pre-encoded, online, unconditional)
5. **Use Bernoulli masking** for classifier-free guidance (10% prob typical)

### For Inference
1. **Pre-encode text offline** for speed (Path 1)
2. **Preserve ctxt_length** for proper padding mask generation
3. **Load null embeddings** from pretrained T2M checkpoint
4. **Apply CFG masking** during inference for better results

### For Training
1. **Always normalize motion** using pretrained mean/std
2. **Preserve padding masks** throughout pipeline
3. **Save null embeddings** in checkpoint (bundle-level parameters)
4. **Verify text paths** match training distribution

---

## 📝 File Paths Reference

### Bundle File
```
hftrainer/models/motion/hymotion_t2m/bundle.py
  └─ HyMotionT2MBundle
       ├─ encode_text()
       ├─ mask_text_cond()
       ├─ predict_flow()
       └─ decode_motion_from_latent()
```

### Trainer File
```
hftrainer/trainers/motion/hymotion_t2m_trainer.py
  └─ HyMotionT2MTrainer
       ├─ _prepare_text_encoding()  [3-path logic]
       ├─ _prepare_and_forward()
       └─ train_step()
```

### Text Encoder File
```
hftrainer/models/motion/hymotion_m2m/network/text_encoder.py
  └─ HYTextModel
       ├─ _encode_llm()        [Qwen3]
       ├─ _encode_sentence_emb()  [CLIP-L]
       ├─ encode_pooling()     [mean pooling + L2 norm]
       └─ encode()             [dual-encoder call]
```

---

## ✅ Analysis Completeness

- ✅ Located all official T2M training code files
- ✅ Extracted and documented text encoding logic
- ✅ Mapped CFG dropout implementation details
- ✅ Documented 3-path text conditioning system
- ✅ Compared T2M vs M2M architecture differences
- ✅ Verified text embedding dimensions and normalization
- ✅ Documented null embedding initialization and usage
- ✅ Verified padding and masking conventions

---

## 📋 Related Documentation

This analysis complements existing documentation:
- **M2M Architecture:** See `CLAUDE.md` for motion-to-motion conditioning details
- **Text Encoding Theory:** See `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py` line comments
- **Training Patterns:** See `hftrainer/trainers/motion/hymotion_t2m_trainer.py` for full trainer implementation
- **Configuration:** See `configs/hymotion_t2m/` for training configs

---

**Analysis Status:** ✅ COMPLETE  
**Last Updated:** May 15, 2026  
**Verified Against:** Repository at `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`
