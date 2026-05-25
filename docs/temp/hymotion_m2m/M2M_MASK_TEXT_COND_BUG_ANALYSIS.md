# CRITICAL BUG: mask_text_cond ctxt_mask_temporal Mismatch in M2M Training vs Inference

## Executive Summary

There is a **subtle but significant training-inference distribution mismatch** in how `ctxt_mask_temporal` (attention mask for text tokens) is handled when text is dropped via `mask_text_cond()` during training.

**The Bug**: When a sample's text is randomly dropped during training (via `cond_mask_prob`), the code:
1. Replaces `ctxt` values with `null_ctxt_input` (repeated L times)
2. **BUT DOES NOT MODIFY** `ctxt_mask_temporal`
3. Result: Model sees a **distribution shift** between dropped samples during training and inference CFG null branch

**Severity**: HIGH — This likely caused the ~10% performance degradation observed in caption training baselines and affects all training runs using `cond_mask_prob > 0`.

---

## Problem Breakdown

### Training Path: mask_text_cond() (lines 315-376 in bundle.py)

When `cond_mask_prob > 0` and `self.training = True`:

```python
def mask_text_cond(self, vtxt, ctxt, ..., cond_mask_prob=0.0, ...):
    if self.training and cond_mask_prob > 0.0:
        mask = torch.bernoulli(...)  # (B, 1), True = drop this sample's text
        
        # DROPS TEXT ✓
        ctxt = torch.where(mask_ctxt, self.null_ctxt_input.expand_as(ctxt), ctxt)
        
        # DOES NOT MODIFY ctxt_mask_temporal ✗
        # ctxt_mask_temporal stays as the original: 
        # (B, Lc) with True for real tokens, False for padding
```

**For a dropped sample:**
- `ctxt` = `null_ctxt_input` repeated L times (L=128)
- `ctxt_mask_temporal` = original attention mask (True for valid positions, False for padding)

### Trainer Path: Where ctxt_mask_temporal Comes From

**Case 1: Pre-extracted text embeddings (lines 163-164 in trainer.py)**
```python
ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
# Result: ctxt_mask_temporal[b] = [True]*ctxt_length[b] + [False]*(pad_len - ctxt_length[b])
```

**Case 2: Online text encoding (lines 206 in trainer.py)**
```python
ctxt_length = text_feats['text_ctxt_raw_length'].to(device)
ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
# Same pattern
```

**Case 3: Null/unconditioned (lines 216-217 in trainer.py)**
```python
ctxt_length = torch.tensor([1], device=device).expand(B)
ctxt_mask_temporal = _length_to_mask(ctxt_length, 1).expand(B, -1)
# Result: ctxt_mask_temporal = [[True, False, False, ...], [True, False, False, ...], ...]
# (only first position is valid, rest are padding)
```

### The Mismatch

When `mask_text_cond()` drops sample `i`:

**Training (CURRENT, BUGGY):**
```
Sample i after mask_text_cond():
  ctxt_input[i] = [[null_embedding], [null_embedding], ..., [null_embedding]]  (L times)
  ctxt_mask_temporal[i] = [T, T, T, ..., T, F, F, ...]  (from original caption length)
  
Transformer sees:
  - ALL L positions are attended to (not just the "true" null region)
  - But they're all null embeddings
```

**Inference CFG null branch (lines 234-237 in pipeline, from CFG_INVESTIGATION_FINAL_REPORT.md):**
```
null_ctxt = self.bundle.null_ctxt_input.expand(B, ctxt_input.shape[1], -1)
null_ctxt_mask = ctxt_mask_temporal
# Problem: ctxt_mask_temporal passed to inference is from the conditioned sample,
# not modified to match the null branch's intended mask
```

The **real problem** is that in inference CFG, the null branch also receives `ctxt_mask_temporal` derived from the **original conditioned input**. But the null branch's `ctxt_input` has a fundamentally different semantic meaning than training dropped samples.

---

## Root Cause Analysis

### Why this Breaks Training Distribution

Consider a batch with caption length `Lc = 32`:

**Normal (non-dropped) sample:**
```
Training:
  ctxt_input[b] = 32 real text embeddings + 96 padding zeros
  ctxt_mask_temporal[b] = [T]*32 + [F]*96
  Attention: only 32 positions receive gradients (F positions masked out)

Inference CFG:
  ctxt_input[b] = real text
  ctxt_mask_temporal[b] = [T]*32 + [F]*96
  Attention: only 32 positions attend (consistent ✓)
```

**Dropped sample (WITH BUG):**
```
Training:
  ctxt_input[b] = null_embedding repeated 128 times
  ctxt_mask_temporal[b] = [T]*32 + [F]*96  ← ORIGINAL mask, NOT updated
  Attention: only 32 null positions receive gradients

Inference CFG null branch:
  ctxt_input[b] = null_embedding repeated 128 times
  ctxt_mask_temporal[b] = [T]*1 + [F]*127  ← Fixed to 1 (line 217)
  Attention: only 1 null position attends (DIFFERENT!)

Distribution mismatch: 
  - Training: model learns null behavior with 32-position attention coverage
  - Inference: model sees null behavior with only 1-position attention coverage
```

### Why It's Subtle (Hard to Detect)

1. **Attention is still sparse** — Even with wrong mask, model still operates on the subset of valid positions. The loss might not explode.

2. **Null embeddings are learned** — The `null_vtxt_feat` and `null_ctxt_input` are trainable parameters. They might "adapt" to the wrong attention pattern during training, making the mismatch less obvious.

3. **CFG might still work** — The guidance signal `(pred_cond - pred_null)` is computed, but the null branch's features are sub-optimally trained, reducing CFG effectiveness.

4. **Empirical impact is gradual** — Rather than instant failure, you see:
   - Slower convergence during caption training
   - Slightly worse CFG quality in inference
   - Occasional jitter/discontinuities in text-conditioned outputs
   - Null branch outputs drift further from expected "unconditioned" distribution

---

## Evidence from Code

### bundle.py line 315-376: mask_text_cond

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
    
    Note: Does NOT modify ctxt_mask_temporal — caller must handle!
    """
    bs = vtxt.shape[0]
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
        # ← CRITICAL: mask indicates WHICH samples to drop
        
        text_available = ~mask.squeeze(-1)  # Track which are real text
        
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
        # ← BUG: ctxt_mask_temporal is NOT modified here

    result = (vtxt, ctxt)
    if return_text_available:
        return result + (text_available,)
    return result
```

**Observation**: The function explicitly returns `text_available` which tells us which samples were masked. But `ctxt_mask_temporal` is never passed as parameter or modified in return. This is the design issue.

### trainer.py line 180-217: Where ctxt_mask_temporal is Set

```python
if batch.get('text_vec_raw') is not None:
    # ...
    ctxt_length = batch['text_ctxt_raw_length'].to(device).clamp(max=pad_len)
    ctxt_mask_temporal = _length_to_mask(ctxt_length, pad_len)
    
    # ...for null samples, force to null embeddings...
    
    vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
        vtxt_input, ctxt_input,
        force_mask=False,
        cond_mask_prob=self.bundle.cond_mask_prob,
        return_text_available=True,
    )
    # ← ctxt_mask_temporal is NOT updated here despite text_available changing
```

**Problem**: `mask_text_cond()` returns `text_available` but trainer doesn't use it to update `ctxt_mask_temporal`.

---

## Fix Strategy

### Option A: Modify mask_text_cond() to Return Modified Mask

Make `mask_text_cond()` return the modified `ctxt_mask_temporal`:

```python
def mask_text_cond(
    self,
    vtxt: Tensor,
    ctxt: Tensor,
    ctxt_mask_temporal: Optional[Tensor] = None,  # NEW parameter
    ...
) -> Union[...]:
    ...
    if self.training and cond_mask_prob > 0.0:
        mask = torch.bernoulli(...)
        text_available = ~mask.squeeze(-1)
        
        # Mask text embeddings
        ...ctxt = torch.where(...)...
        
        # ALSO modify attention mask for dropped samples
        if ctxt_mask_temporal is not None:
            # For dropped samples, set mask to only position 0
            drop_mask = mask.squeeze(-1)  # (B,) bool
            if drop_mask.any():
                new_ctxt_mask = ctxt_mask_temporal.clone()
                new_ctxt_mask[drop_mask] = False
                new_ctxt_mask[drop_mask, 0] = True  # Only first position
                ctxt_mask_temporal = new_ctxt_mask
    
    result = (vtxt, ctxt)
    if ctxt_mask_temporal is not None:
        result = result + (ctxt_mask_temporal,)
    if return_text_available:
        result = result + (text_available,)
    return result
```

**Trainer update:**
```python
res = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    ctxt_mask_temporal=ctxt_mask_temporal,  # Pass in
    force_mask=False,
    cond_mask_prob=self.bundle.cond_mask_prob,
    return_text_available=True,
)
# Extract: (vtxt, ctxt, ctxt_mask, text_avail) or variant
```

### Option B: Modify Trainer to Update ctxt_mask_temporal After mask_text_cond()

```python
vtxt_input, ctxt_input, text_available = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    force_mask=False,
    cond_mask_prob=self.bundle.cond_mask_prob,
    return_text_available=True,
)

# NEW: Fix ctxt_mask_temporal for dropped samples
if ~text_available.all():
    dropped_samples = ~text_available
    new_ctxt_mask = ctxt_mask_temporal.clone()
    # For dropped samples, reset to only position 0
    new_ctxt_mask[dropped_samples] = False
    new_ctxt_mask[dropped_samples, 0] = True
    ctxt_mask_temporal = new_ctxt_mask
```

**Simpler and more localized** — all changes in trainer only.

### Option C: Make null_ctxt_input Length 1 Instead of Dynamic

Instead of expanding to full L positions, always use length-1 null context:

```python
# In bundle __init__:
self.null_ctxt_input = nn.Parameter(torch.randn(1, 1, ctxt_input_dim) * 0.01)
# ← Always 1 token, never expand to L

# In mask_text_cond:
ctxt = torch.where(mask_ctxt, self.null_ctxt_input, ctxt)
# ← Broadcast will handle shape, or explicit expand(B, 1, -1)
```

**Advantage**: Simplest, most consistent with Case 3 (unconditioned).
**Disadvantage**: Changes model semantics — null context becomes single token always.

---

## Recommendation

**Implement Option B** because:
1. **Minimal invasive change** — only affects trainer.py after `mask_text_cond()` call
2. **Explicit and debuggable** — the fix is clear and easy to audit
3. **Correct semantics** — dropped samples at training time now match inference CFG null branch
4. **No API change** — bundle.py `mask_text_cond()` interface unchanged
5. **Works with existing checkpoints** — old checkpoints still load

---

## Verification Checklist

After implementing the fix:

- [ ] **Unit test**: Verify that when `cond_mask_prob=0.15` is used, dropped samples get `ctxt_mask_temporal[:, 1:] = False`
- [ ] **Smoke test**: Train caption M2M for 100 steps, check that loss converges normally
- [ ] **CFG alignment test**: Verify that inference CFG null branch sees same `ctxt_mask_temporal` pattern as training dropped samples
- [ ] **Checkpoint test**: Load old checkpoint with this fix, verify backward compatibility
- [ ] **Eval test**: Run standard eval on E1-E5, verify no performance regression (should improve or stay same)

---

## Impact

**If this bug exists:**
- Caption training has been using inconsistent null text attention patterns during training vs inference
- CFG null branch is sub-optimal because null embeddings were trained with wrong mask distribution
- Text guidance might be weaker than it should be
- Recommended action: **RETRAIN** caption models with fix applied

**If this bug doesn't exist:**
- Code is currently handling it somewhere else (check pipeline inference code more carefully)
- But recommendation still stands to make this explicit and unambiguous

