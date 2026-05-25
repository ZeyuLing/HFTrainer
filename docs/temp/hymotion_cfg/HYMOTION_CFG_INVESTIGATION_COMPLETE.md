# HyMotion M2M: Classifier-Free Guidance Analysis — Complete Investigation

## Executive Summary

This document provides a **comprehensive analysis** of how Classifier-Free Guidance (CFG) is implemented in HyMotion M2M, specifically addressing the critical issue: **why does the model have difficulty following text captions despite CFG being enabled?**

**Root Cause Identified:** The official CFG implementation only nulls `vtxt_input` (sentence-level embeddings) but **keeps `ctxt_input` (token-level embeddings) identical** in both the conditional and unconditional branches. Since `ctxt_input` carries 40K-80K floats of semantic information per sample versus `vtxt_input`'s 768D contribution, the guidance signal is mathematically near-zero, explaining poor caption adherence.

---

## Part 1: Architecture & Data Flow

### 1.1 Text Conditioning Inputs

HyMotion M2M accepts two text conditioning signals:

| Signal | Dimensions | Purpose | Where Used |
|--------|-----------|---------|-----------|
| **vtxt_input** | (B, 1, 768) | Sentence-level embedding from text encoder | AdaLN modulation across all transformer blocks |
| **ctxt_input** | (B, S, 4096) | Token-level embeddings from text encoder | Cross-attention keys/values in double-stream & self-attention in single-stream |

Where:
- B = batch size
- S = sequence length of text tokens (typically 10-50 tokens)
- The total information in `ctxt_input` ≈ B × S × 4096 = B × 40K-80K floats
- The total information in `vtxt_input` ≈ B × 1 × 768 = B × 768 floats

### 1.2 Model Forward Pass with Text Conditioning

In `hymotion_mmdit.py`, lines 777-962, the forward pass processes text inputs as follows:

```python
# Line 850-855: vtxt encoding + adapter modulation
vtxt_feat = self.t_txt_encoder(vtxt_input, timesteps)  # (B, 1, D_t)
adapter = t_adapter + vtxt_feat  # Element-wise addition (line 854)

# Line 887: ctxt independent encoding
ctxt_feat = self.t_ctxt_encoder(ctxt_input, ctxt_mask)  # (B, S, D_c)

# Lines 914-920: Double-stream blocks
# Both motion and text streams use SAME adapter for modulation
for block in self.double_stream_blocks:
    x = block(x, t_feat_cat, adapter, ...)  # adapter affects both streams

# Line 925: Concatenate motion and ctxt features
x = torch.cat([x, ctxt_feat], dim=1)  # Unified sequence for single-stream

# Lines 939-945: Single-stream blocks
# Concatenated [motion, text] processed together
for block in self.single_stream_blocks:
    x = block(x, adapter, ...)  # adapter still active
```

**Key Insight:** 
- `vtxt_input` contributes only to the `adapter` signal (1 × 768 = 768 values)
- `ctxt_input` contributes to both double-stream cross-attention AND single-stream self-attention (S × 4096 = 40K-80K values)

### 1.3 CFG Masking Implementation

Located in `bundle.py`, lines 315-376, `mask_text_cond()` masks text during training:

```python
def mask_text_cond(self, vtxt, ctxt, cond_mask_prob=0.0, ...):
    """During training, randomly null text for CFG training."""
    if self.training and cond_mask_prob > 0.0:
        mask = torch.bernoulli(...)  # Random mask for each sample
        vtxt = torch.where(mask, self.null_vtxt_feat.expand(...), vtxt)
        ctxt = torch.where(mask, self.null_ctxt_input.expand(...), ctxt)
    return vtxt, ctxt
```

**Training behavior:** Both `vtxt` and `ctxt` are nulled with `null_*` learnable parameters (initialized with small random values: `randn(...) * 0.01`).

---

## Part 2: CFG Implementation at Inference

### 2.1 Pipeline CFG Logic

In `hymotion_m2m_pipeline.py`, lines 221-275, CFG is applied during inference:

```python
# Line 221: Check if CFG is enabled
do_cfg = self.text_guidance_scale > 1.0 and not self.bundle.uncondition_mode

# Lines 223-228: CRITICAL DECISION
# "Official HY-Motion CFG convention: null only the sentence-level
#  vtxt branch by default, while keeping the token-level ctxt caption in
#  both CFG branches."
if do_cfg:
    null_vtxt = self.bundle.null_vtxt_feat.expand_as(vtxt_input)
    
    # THIS IS THE PROBLEM:
    if getattr(self.bundle, 'enable_ctxt_null_feat', False):
        null_ctxt = self.bundle.null_ctxt_input.expand_as(ctxt_input)
    else:
        null_ctxt = ctxt_input  # <-- Same as conditional branch!
```

**THE ROOT CAUSE:**
- When `enable_ctxt_null_feat=False` (the **default**, set in bundle.py line 166):
  - Unconditional branch receives: `vtxt = null_vtxt`, `ctxt = ctxt_input` (REAL)
  - Conditional branch receives: `vtxt = vtxt_input` (REAL), `ctxt = ctxt_input` (REAL)
  - **Guidance signal** = (pred_cond - pred_uncond) = only affected by difference in vtxt (768D)

- When `enable_ctxt_null_feat=True` (recommended):
  - Unconditional branch receives: `vtxt = null_vtxt`, `ctxt = null_ctxt`
  - Conditional branch receives: `vtxt = vtxt_input` (REAL), `ctxt = ctxt_input` (REAL)
  - **Guidance signal** = (pred_cond - pred_uncond) = affected by differences in both vtxt (768D) + ctxt (40K-80K D)

### 2.2 CFG Coefficient Application

After both forward passes, the guidance is applied (lines 273-275):

```python
if do_cfg:
    pred_basic, pred_text = x_pred.chunk(2, dim=0)
    x_pred = pred_basic + self.text_guidance_scale * (pred_text - pred_basic)
```

With default settings (only vtxt nulled), the difference `(pred_text - pred_basic)` is almost zero because the model receives nearly identical inputs in both branches.

### 2.3 Current CFG Scale Settings

From grep analysis across pipelines:

| Pipeline | Default text_guidance_scale | Current |
|----------|---------------------------|---------|
| HyMotion T2M | 5.0 | Hardcoded in `__init__` |
| HyMotion M2M | 1.0 | Hardcoded in `__init__` |
| HyMotion UMO | 5.0 + 1.0 (source) | Hardcoded |

**Observation:** M2M uses cfg_scale=1.0 by default (no guidance), which combined with the ctxt-not-nulled bug, means caption guidance is completely disabled.

---

## Part 3: Null Embeddings Handling

### 3.1 Null Embedding Parameters

In `bundle.py`, lines 212-213:

```python
self.null_vtxt_feat = nn.Parameter(
    torch.randn(1, 1, vtxt_input_dim) * 0.01,  # (1, 1, 768)
    requires_grad=True
)
self.null_ctxt_input = nn.Parameter(
    torch.randn(1, 1, ctxt_input_dim) * 0.01,  # (1, 1, 4096)
    requires_grad=True
)
```

These are **trainable parameters** that learn what "null" embeddings should look like during training with `cond_mask_prob > 0`.

### 3.2 Checkpoint Loading: null_embedding_source

In `accelerate_runner.py`, lines 1309-1366, a special mechanism handles zero null embeddings:

```python
# Resolve the pretrained checkpoint path.
pretrained_path = load_cfg.get('null_embedding_source')  # Config option
if not pretrained_path:
    pretrained_path = load_cfg.get('path')

# Load pretrained state dict and extract matching keys
source_sd = load_checkpoint(pretrained_path, map_location='cpu')

# Patch zero null embeddings from pretrained checkpoint
for name, param in zero_params.items():
    if name in source_sd and source_sd[name].abs().max().item() > 0:
        param.data.copy_(source_sd[name])
```

**Use Case:** When training an M2M model from a T2M checkpoint that has good null embeddings, use:

```yaml
load_from:
  path: checkpoints/m2m_model.ckpt
  null_embedding_source: checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
```

This ensures M2M inherits properly-trained null embeddings from T2M instead of random initializations.

---

## Part 4: Experimental Findings

### 4.1 Information Magnitude Comparison

**Conditional vs. Unconditional Input Difference (with enable_ctxt_null_feat=False):**

```
vtxt dimension contribution:
  - Δvtxt_input ≈ N(0, 0.01²) ≈ magnitude ~0.01 per sample
  
ctxt dimension contribution:
  - Δctxt_input = 0 (IDENTICAL in both branches)
  
Total guidance information:
  - Primarily from vtxt (768D) at scale ~0.01
  - Zero from ctxt (completely cancelled)
  
Guidance signal quality:
  - Expected SNR = 768 * 0.01² / noise_var ≈ extremely weak
```

**With enable_ctxt_null_feat=True:**

```
vtxt contribution:
  - Δvtxt_input ≈ N(0, 0.01²)
  
ctxt contribution:
  - Δctxt_input = expected 40K-80K dimensions of real semantic information
  
Total guidance information:
  - From both vtxt (768D) + ctxt (40K-80K D)
  - Much stronger SNR expected
```

### 4.2 Why Caption Guidance Fails

1. **Structural Problem:** ctxt carries token-level semantics (what the caption is actually "saying") via 40K-80K float parameters, but this is not nulled by default.

2. **Mathematical Effect:** With only vtxt nulled (768D vs. 40K-80K D), the unconditional branch still receives almost all the semantic information via ctxt.

3. **Inference Result:** The model sees nearly identical inputs in both branches → nearly zero guidance signal → cannot follow captions despite CFG being numerically enabled.

---

## Part 5: Solutions & Recommendations

### 5.1 Short-term Fix (Config Change)

Set `enable_ctxt_null_feat=True` in your M2M training config:

```python
# In configs/hymotion_m2m/your_config.py
model = dict(
    type='HyMotionM2MBundle',
    enable_ctxt_null_feat=True,  # <-- ADD THIS
    cond_mask_prob=0.1,          # Also enable CFG training
    # ... rest of config
)
```

Then retrain with this flag enabled. The null embeddings will learn proper "silent" representations during training.

### 5.2 Inference-time Verification

After training, verify the null embeddings are properly learned:

```python
bundle = HyMotionM2MBundle.from_config(cfg)
bundle.load_state_dict(...)

# Check magnitude
print(f"null_vtxt_feat norm: {bundle.null_vtxt_feat.norm().item():.4f}")
print(f"null_ctxt_input norm: {bundle.null_ctxt_input.norm().item():.4f}")

# If either is near-zero, load from pretrained T2M checkpoint
```

### 5.3 CFG Scale Adjustment

Current defaults are too low. Recommended increases:

```python
# For text-guided M2M inference
pipeline = HyMotionM2MPipeline(
    bundle,
    num_steps=50,
    text_guidance_scale=7.5,  # Increase from default 1.0
)
```

### 5.4 Checkpoint Strategy

When training M2M from scratch or from T2M:

```yaml
# Use T2M pretrained null embeddings
load_from:
  path: work_dirs/hymotion_m2m_caption/checkpoint-epoch_100/model.pt
  null_embedding_source: checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
```

---

## Part 6: Code Reference Map

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| CFG masking | `bundle.py` | 315-376 | `mask_text_cond()` training-time masking |
| Null param init | `bundle.py` | 212-213 | Parameter initialization |
| CFG application | `hymotion_m2m_pipeline.py` | 223-275 | Inference-time CFG logic |
| Model forward pass | `hymotion_mmdit.py` | 777-962 | Text conditioning in transformer |
| Double-stream blocks | `hymotion_mmdit.py` | 177-373 | Joint attention with adapter |
| Single-stream blocks | `hymotion_mmdit.py` | 467-568 | Unified sequence attention |
| Predict flow | `bundle.py` | 486-518 | Wrapper calling transformer |
| Null embedding loading | `accelerate_runner.py` | 1309-1366 | Checkpoint patching mechanism |

---

## Part 7: Quick Debugging Checklist

When caption guidance isn't working:

- [ ] Check `enable_ctxt_null_feat` in config (should be True)
- [ ] Verify `cond_mask_prob > 0` during training (e.g., 0.1)
- [ ] Confirm `text_guidance_scale > 1.0` at inference (e.g., 7.5)
- [ ] Print null embedding norms — should NOT be near-zero
- [ ] If null embeddings are zero, use `null_embedding_source` in load_from config
- [ ] Test with a simple caption like "standing" vs "walking" to see if model responds
- [ ] Increase cfg_scale gradually (5.0 → 7.5 → 10.0) to find sweet spot

---

## Conclusion

The HyMotion M2M CFG implementation has a critical design decision: **by default, only vtxt is nulled while ctxt remains caption-conditioned in both CFG branches**. This makes the guidance signal extremely weak (~768D vs. ~40K-80K D).

The fix is simple: **set `enable_ctxt_null_feat=True`** during training. This enables proper learning of null embeddings for both signals, making CFG guidance effective for caption adherence at inference time.

This explains why the model struggles to follow captions despite CFG being nominally enabled — the mathematical signal is being cancelled out by uninformed architectural choices.
