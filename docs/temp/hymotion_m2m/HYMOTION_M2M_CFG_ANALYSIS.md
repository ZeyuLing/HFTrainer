# HyMotion M2M: Complete CFG Dropout & Null Embedding Analysis

**Date:** 2026-05-15  
**Focus:** Classifier-Free Guidance (CFG) dropout mechanism, null embeddings initialization, and text conditioning flow  
**Files Analyzed:** 
- `hftrainer/models/motion/hymotion_m2m/bundle.py` (main analysis)
- `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (training flow)
- `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` (inference flow)

---

## Executive Summary

**Three Critical Questions Answered:**

1. **What does `mask_text_cond` do?**  
   ✅ **REPLACES embeddings** with learned null embeddings via `torch.where()`, NOT zeroing them out.  
   Replaces vtxt with `null_vtxt_feat` and ctxt with `null_ctxt_input` when mask is True.

2. **How does `cond_mask_prob` control masking?**  
   ✅ **Per-sample Bernoulli sampling.** For batch size B, `~cond_mask_prob*100%` of samples are masked per forward pass.  
   Uses `torch.bernoulli(ones(B) * cond_mask_prob)` → independent probabilistic draw for each sample.

3. **Is there a bug causing ALL text embeddings to be masked?**  
   ✅ **NO BUG.** Implementation is mathematically correct:
   - Bernoulli distribution has correct variance and mean
   - Samples are drawn independently per batch item
   - Code correctly expands mask shape to match embeddings
   - No hardcoding or indexing error that could cause 100% masking

---

## 1. The `mask_text_cond` Implementation

### Full Method Code (lines 315–376)

```python
def mask_text_cond(
    self,
    vtxt: Tensor,
    ctxt: Tensor,
    force_mask: bool = False,
    cond_mask_prob: float = 0.0,
    return_text_available: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Apply classifier-free guidance masking to text conditions.

    Args:
        vtxt: Sentence-level text embeddings, shape (B, 1, D_v).
        ctxt: Token-level text embeddings, shape (B, L_c, D_c).
        force_mask: If True, return null embeddings for all samples.
        cond_mask_prob: Probability of masking text (CFG dropout rate).
        return_text_available: If True, also return boolean mask indicating
            which samples have real text (not masked). Shape (B,).

    Returns:
        - If return_text_available=False: (vtxt_masked, ctxt_masked)
        - If return_text_available=True: (vtxt_masked, ctxt_masked, text_available)
          where text_available[b]=True means sample b has real text,
          text_available[b]=False means sample b was masked to null.
    """
    bs = vtxt.shape[0]
    # Track which samples have real (non-masked) text
    text_available = torch.ones(bs, dtype=torch.bool, device=vtxt.device)

    if force_mask:
        text_available.fill_(False)
        result = (
            self.null_vtxt_feat.expand(*vtxt.shape),
            self.null_ctxt_input.expand(*ctxt.shape),
        )
        if return_text_available:
            return result + (text_available,)
        return result

    if self.training and cond_mask_prob > 0.0:
        mask = torch.bernoulli(
            torch.ones(bs, device=vtxt.device) * cond_mask_prob
        ).view(bs, 1).bool()
        # Invert: mask=1 (drop text) -> text_available=0 (no real text)
        text_available = ~mask.squeeze(-1)

        mask_vtxt = mask
        while mask_vtxt.ndim < vtxt.ndim:
            mask_vtxt = mask_vtxt.unsqueeze(-1)
        vtxt = torch.where(
            mask_vtxt, self.null_vtxt_feat.expand_as(vtxt), vtxt
        )
        mask_ctxt = mask
        while mask_ctxt.ndim < ctxt.ndim:
            mask_ctxt = mask_ctxt.unsqueeze(-1)
        ctxt = torch.where(
            mask_ctxt, self.null_ctxt_input.expand_as(ctxt), ctxt
        )

    result = (vtxt, ctxt)
    if return_text_available:
        return result + (text_available,)
    return result
```

### Detailed Operation Flow

#### **Phase 1: Initialization (Line 339–341)**
```
bs = 32                                    # batch size
text_available = [True, True, ..., True]   # shape (32,)
```

#### **Phase 2a: Force Mask Branch (Lines 343–351)**
When `force_mask=True` (e.g., during guidance generation):
```
text_available.fill_(False)                # all False
return (null_vtxt_feat.expand_as(vtxt),    # (32, 1, 768)
        null_ctxt_input.expand_as(ctxt))   # (32, L_c, 4096)
```
**Effect:** All samples get null embeddings. Model sees no text.

#### **Phase 2b: Probabilistic Masking Branch (Lines 353–371)**
Only active during **training** (`self.training=True`) AND `cond_mask_prob > 0`.

**Step 1: Sample Bernoulli mask**
```python
mask = torch.bernoulli(torch.ones(32) * 0.1)  # if cond_mask_prob=0.1
```
- Produces shape (32,) of floats in {0.0, 1.0}
- **Expected value:** E[mask] = 0.1 → ~3 samples masked per 32-sample batch
- **NOT deterministic:** Different calls produce different masks
- **Independent:** Each sample draws independently

**Visual Example (32-sample batch, cond_mask_prob=0.1):**
```
mask = [0, 1, 0, 0, 1, 0, ..., 0, 1, 0]  # shape (32,)
                ↓
            .view(32, 1)
                ↓
          shape (32, 1)
                ↓
text_available = ~mask = [1, 0, 1, 1, 0, 1, ..., 1, 0, 1]  # inverted
```

**Step 2: Expand mask to match embedding dimensions**
```python
mask_vtxt = mask  # shape (32, 1)
while mask_vtxt.ndim < vtxt.ndim:  # vtxt.ndim = 3
    mask_vtxt = mask_vtxt.unsqueeze(-1)
# mask_vtxt now shape (32, 1, 1)

mask_ctxt = mask  # shape (32, 1)
while mask_ctxt.ndim < ctxt.ndim:  # ctxt.ndim = 3
    mask_ctxt = mask_ctxt.unsqueeze(-1)
# mask_ctxt now shape (32, 1, 1)
```

**Step 3: Conditional replacement with torch.where**
```python
vtxt = torch.where(
    mask_vtxt,                                  # shape (32, 1, 1)
    self.null_vtxt_feat.expand_as(vtxt),       # True branch: (32, 1, 768)
    vtxt                                        # False branch: (32, 1, 768)
)
```

**Semantics of `torch.where(condition, true_val, false_val)`:**
```
output[i, j, k] = true_val[i, j, k]  if condition[i, j, k] == True
                  false_val[i, j, k] if condition[i, j, k] == False
```

So:
```
if mask_vtxt[b, 0, 0] == 1 (True):
    vtxt_out[b, :, :] = null_vtxt_feat[0, 0, :]  (broadcast to (1, 768))
else:
    vtxt_out[b, :, :] = vtxt[b, :, :]            (kept as is)
```

#### **Phase 3: Return (Lines 373–376)**
```python
result = (vtxt, ctxt)
if return_text_available:
    return (vtxt, ctxt, text_available)
else:
    return (vtxt, ctxt)
```

### Why This Design Avoids the "100% Masking" Bug

**Concern:** Could the code accidentally mask all ~10% and then apply some condition that cascades to 100%?

**Answer:** No. Here's why:

1. **Independent Draws:**  
   Each sample in the batch gets an independent Bernoulli draw. With `cond_mask_prob=0.1`:
   - P(sample 0 masked) = 0.1
   - P(sample 1 masked) = 0.1
   - P(all 32 samples masked) = 0.1^32 ≈ 10^-31 (astronomically unlikely)

2. **No Reduction or Aggregation:**  
   The mask is never summed, max'd, or aggregated. Each sample is treated independently.

3. **No State Mutation:**  
   The mask is sampled once per call. It doesn't accumulate or trigger any follow-up conditions.

4. **torch.where is Deterministic:**  
   Once the mask is determined, `torch.where` applies it element-wise with no side effects.

---

## 2. Null Embeddings Initialization

### Definition (Lines 212–213)

```python
self.null_vtxt_feat = nn.Parameter(
    torch.randn(1, 1, vtxt_input_dim) * 0.01, 
    requires_grad=True
)
self.null_ctxt_input = nn.Parameter(
    torch.randn(1, 1, ctxt_input_dim) * 0.01, 
    requires_grad=True
)
```

### Breakdown

| Aspect | `null_vtxt_feat` | `null_ctxt_input` |
|--------|---|---|
| **Shape** | (1, 1, 768) | (1, 1, 4096) |
| **Dtype** | float32 | float32 |
| **Initialization** | N(0, 1) × 0.01 → N(0, 0.01²) | N(0, 1) × 0.01 → N(0, 0.01²) |
| **Trainable** | Yes (`requires_grad=True`) | Yes (`requires_grad=True`) |
| **On Device** | Moved to GPU with model | Moved to GPU with model |
| **Role** | Sentence-level null embedding | Token-level null embedding |

### Why Trainable vs. Frozen?

From the code comment (lines 205–211):

> **Trainable:** initialized with small random values. During M2M training, these embeddings learn the "no text condition" representation jointly with the transformer. This allows CFG to work correctly: when text_available=False, the model sees null_embeddings which are distinct from real text embeddings, enabling the transformer to learn meaningful text conditioning via the guidance signal (pred_with_text - pred_with_null). **Frozen null embeddings cause CFG to fail** because null and real embeddings appear equivalent to the model.

**Key Insight:** For CFG to work, the model must learn that:
- Real text → feature vector A
- Null text (no guidance) → feature vector B

If B is frozen at 0 or constant, the model cannot distinguish between:
1. "I was given null text" vs. "I was given text that happened to encode as zeros"

By making null embeddings trainable, the model learns a unique, non-trivial representation for "no guidance," which enables the guidance signal `(prediction_with_text - prediction_with_null)` to be meaningful.

---

## 3. Text Conditioning Flow: Training vs. Inference

### 3.1 Training Flow (Simplified)

```
Batch → _prepare_and_forward()
  ├─ Pre-extracted embeddings: text_vec_raw (B, 1, 768), text_ctxt_raw (B, L_c, 4096)
  │
  ├─ mask_text_cond(text_vec_raw, text_ctxt_raw, cond_mask_prob=0.1)
  │  │  if training and rand() < 0.1:
  │  │    text_vec = null_vtxt_feat.expand_as(text_vec)  # masked
  │  │  else:
  │  │    text_vec = text_vec_raw                         # keep real
  │  │
  │  └─ returns (text_vec, text_ctxt, text_available)
  │
  ├─ model.predict_flow(
  │    motion, text_vec, text_ctxt,
  │    timestep_embed, ...
  │  )
  │
  └─ Compute loss (CFG loss weighting uses text_available flag)
```

**Key Points:**
- `cond_mask_prob` sampled **once per forward pass** (not per epoch or step)
- When masked, ~10% of samples see null embeddings
- Gradient flows through `null_vtxt_feat` and `null_ctxt_input` parameters
- Model learns both "real text" and "null" representations simultaneously

### 3.2 Inference Flow (with CFG)

```
Batch → forward_pass()
  ├─ Pre-extracted embeddings: text_vec (B, 1, 768), text_ctxt (B, L_c, 4096)
  │
  ├─ Guidance scale g = 7.5 (example)
  │
  ├─ Two model forward passes:
  │  │
  │  ├─ Pass 1 (with text):
  │  │   pred_with_text = model.predict_flow(
  │  │       motion, text_vec, text_ctxt, ...
  │  │   )
  │  │
  │  └─ Pass 2 (with null):
  │      pred_with_null = model.predict_flow(
  │          motion, 
  │          null_vtxt_feat.expand_as(text_vec),
  │          null_ctxt_input.expand_as(text_ctxt),
  │          ...
  │      )
  │
  ├─ Compute guided prediction:
  │   pred_guided = pred_with_null + g * (pred_with_text - pred_with_null)
  │              = (1 - g) * pred_with_null + g * pred_with_text
  │
  └─ Use pred_guided for denoising step
```

**Why CFG Works:**
- Model was trained to recognize **two different representations**: real text vs. null
- The difference `(pred_with_text - pred_with_null)` captures the model's learned "text influence"
- Scaling this difference by g amplifies text influence (higher g → stronger text guidance)

---

## 4. Parameter Initialization Semantics

### 4.1 Why N(0, 0.01²)?

```python
torch.randn(1, 1, 768) * 0.01  # ← small random initialization
```

**Rationale:**
1. **Non-zero:** Ensures null embeddings are NOT just zeros, making them distinguishable from silence/padding
2. **Small:** Prevents null embeddings from dominating early training. Small initialization allows gradient descent to find a meaningful "no text" representation
3. **Random:** Each run gets a different starting point, allowing diversity in null embedding learning across runs

**Comparison:**
| Init Scheme | Effect |
|---|---|
| All zeros (0.0) | ❌ CFG broken — model can't distinguish null from zero-padded real text |
| Large random (N(0,1)) | ⚠️ Okay but slower learning — null starts far from text space |
| Small random (N(0,0.01²)) | ✅ Best — null starts near text space, gradient descent finds true null rep |

### 4.2 Broadcasting in mask_text_cond

When `mask=True` for sample b:

```python
# mask_vtxt shape (32, 1, 1), null_vtxt_feat shape (1, 1, 768)
vtxt[b, :, :] = null_vtxt_feat[0, 0, :]  # (1, 768) broadcast to (32, 1, 768)
```

PyTorch automatic broadcasting rules:
```
(1, 1, 768) expanded with (32, 1, 1) shape → (32, 1, 768)
```

---

## 5. Text Embedding Dimensions

### 5.1 Sentence-level (`vtxt`)
- **Shape:** (B, 1, 768)  
- **Semantics:** One embedding per sample, 768-dim
- **Source:** Sentence-level CLIP/text encoder (usually `text_encoder.encode(prompt)`)
- **Used for:** Global semantic understanding of the motion description

### 5.2 Token-level (`ctxt`)
- **Shape:** (B, L_c, 4096)  
- **Semantics:** L_c tokens, 4096-dim per token
- **Source:** Token-level CLIP or BERT embeddings
- **Used for:** Fine-grained alignment between motion frames and text tokens

| Component | Dim | Purpose |
|---|---|---|
| `null_vtxt_feat` | 768 | Learned "no sentence context" |
| `null_ctxt_input` | 4096 | Learned "no token context" |

---

## 6. Integration with Training Loss

### Pseudo-code in trainer

```python
# Training step
text_available_mask = torch.ones(B, dtype=torch.bool)

if enable_cfg_dropout:
    text_vec, text_ctxt, text_available_mask = bundle.mask_text_cond(
        text_vec_raw, text_ctxt_raw,
        cond_mask_prob=0.1,
        return_text_available=True
    )

# Forward pass
pred = model.predict_flow(motion, text_vec, text_ctxt, ...)

# Loss computation (may weight by text_available_mask if desired)
loss = m2m_loss(pred, target)
```

**Note:** The `text_available_mask` tells the trainer which samples were masked. Some implementations use this to:
1. Log separate metrics for masked vs. unmasked samples
2. Apply different loss weights (e.g., lower weight for masked samples during early training)
3. Compute separate guidance-effectiveness metrics

---

## 7. Common Pitfalls & Debug Checklist

| Issue | Cause | Fix |
|---|---|---|
| CFG doesn't improve quality | Frozen null embeddings | Ensure `requires_grad=True` in parameter init |
| All text masked (100%) | Typically trainer logic, not mask_text_cond | Check `cond_mask_prob` is not accidentally set to 1.0; verify Bernoulli isn't replaced with hardcoded True |
| Null embeddings don't converge | Initialization too large or too small | Keep N(0, 0.01²); smaller doesn't help (learning becomes slow) |
| Text conditioning ignored at inference | Model never saw enough masked samples during training | Increase `cond_mask_prob` from 0.0 during training; typical range is 0.1–0.3 |
| Shape mismatch in mask expansion | Bug in while loop | Verify mask expands to match embedding dims (use `.ndim` check) |

---

## 8. Key Equations & Math

### 8.1 Bernoulli Sampling

```
mask ~ Bernoulli(p=cond_mask_prob)

E[mask] = cond_mask_prob
Var[mask] = cond_mask_prob * (1 - cond_mask_prob)

For cond_mask_prob=0.1:
  E[#masked in batch 32] = 3.2
  P(exactly 3 masked) ≈ 0.23
  P(all 32 masked) ≈ 10^-31
```

### 8.2 torch.where Semantics

```
result[i, j, k] = {
    true_val[i, j, k],  if condition[i, j, k] is True
    false_val[i, j, k], if condition[i, j, k] is False
}
```

### 8.3 CFG Guidance Scaling

```
pred_guided = pred_with_null + g * (pred_with_text - pred_with_null)

where:
  g = guidance_scale (e.g., 7.5)
  pred_with_text = model(motion, real_text_embedding, ...)
  pred_with_null = model(motion, null_embedding, ...)

When g=1.0 → no guidance (equivalent to pred_with_text)
When g=0.0 → anti-guidance (equivalent to pred_with_null)
When g>1.0 → amplified text influence
```

---

## 9. Summary Table

| Question | Answer |
|---|---|
| **Does mask_text_cond replace or zero?** | REPLACES with `null_vtxt_feat` and `null_ctxt_input` via `torch.where()` |
| **How does cond_mask_prob work?** | Per-sample Bernoulli sampling; ~cond_mask_prob*100% of batch masked per step |
| **Bug risk (100% masking)?** | NO. Independent draws + no aggregation make 100% masking astronomically unlikely |
| **Are null embeddings trainable?** | YES, both have `requires_grad=True` |
| **Null initialization scale?** | N(0, 0.01²) — small random to allow learning |
| **When is masking applied?** | Only during training when `self.training=True` and `cond_mask_prob > 0` |
| **Inference CFG usage?** | `force_mask=True` replaces with null; two forward passes compute guidance signal |
| **Shape after expansion?** | Mask expands from (B,1) → (B,1,1,1,...) to match embedding tensor dims |

---

## 10. Files for Reference

```
hftrainer/models/motion/hymotion_m2m/
├── bundle.py                  # mask_text_cond (line 315), null init (line 212)
│   └── HyMotionM2MBundle.__init__()  # Parameter defs
│   └── mask_text_cond()               # CFG dropout logic
│
hftrainer/trainers/motion/
├── hymotion_m2m_trainer.py
│   └── _prepare_and_forward()  # Uses mask_text_cond during training
│
hftrainer/pipelines/motion/
└── hymotion_m2m_pipeline.py
    └── forward_pass()         # Uses force_mask=True for inference CFG
```

---

## Appendix: Visual Diagram

### CFG Training & Inference

```
═══════════════════════════════════════════════════════════════════

                    ┌─────────────────────────┐
                    │   HyMotionM2MBundle     │
                    │  (during __init__)      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        null_vtxt_feat (1,1,768)  null_ctxt_input (1,1,4096)
        N(0,0.01²), trainable      N(0,0.01²), trainable

═══════════════════════════════════════════════════════════════════

                    TRAINING PHASE
                    
    ┌─────────────────────────────────────────────────┐
    │ Batch: text_vec_raw (B,1,768), text_ctxt_raw    │
    │        cond_mask_prob = 0.1                     │
    └─────────────┬──────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ mask_text_cond()    │
        │ (training=True)     │
        └────────┬────────────┘
                 │
         ┌───────┴────────┐
         │ Bernoulli(0.1) │  ~10% of batch masked
         └───────┬────────┘
                 │
         ┌───────▼────────────────────────────┐
         │ For masked samples b:              │
         │   text_vec[b] = null_vtxt_feat     │
         │   text_ctxt[b] = null_ctxt_input   │
         │ For unmasked samples:              │
         │   text_vec[b] = text_vec_raw[b]   │
         │   text_ctxt[b] = text_ctxt_raw[b] │
         └───────┬────────────────────────────┘
                 │
                 ▼
         ┌──────────────────┐
         │ model.forward()  │
         │ (with real/null) │
         └────┬──────┬──────┘
              │      │
              ▼      ▼
           loss   gradients
                   ↓ through
            null_vtxt_feat
            null_ctxt_input

═══════════════════════════════════════════════════════════════════

                    INFERENCE PHASE
                    
    ┌──────────────────────────────────────────────┐
    │ Batch: text_vec (B,1,768), text_ctxt (B,L,D) │
    │        guidance_scale = 7.5                  │
    └──────────────┬───────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
    ┌─────────────┐     ┌─────────────┐
    │ Forward +1  │     │ Forward -1  │
    │ (with text) │     │ (with null) │
    └──────┬──────┘     └──────┬──────┘
           │                   │
      pred_with_text      pred_with_null
           │                   │
           └───────┬───────────┘
                   │
                   ▼
        pred_guided = pred_with_null + 7.5 * 
                      (pred_with_text - pred_with_null)
                   │
                   ▼
           (stronger text influence)

═══════════════════════════════════════════════════════════════════
```

---

**End of Analysis Document**
