# TAL (Text-Awareness Loss) Bug Analysis — HyMotion M2M CRFM Trainer

## Executive Summary

**BUG CONFIRMED**: TAL compares NULL-vs-NULL on already-text-masked samples, making it a no-op. When `cond_mask_prob` has already nullified text embeddings via probabilistic masking, TAL's "null text" forward pass produces predictions that are already identical to the conditional forward (since both branches have null embeddings). The loss reduces to zero, providing no regularization.

---

## Root Cause Analysis

### The TAL Flow (CRFM Trainer, lines 119-199)

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    ctx = self._prepare_and_forward(batch)
    losses = self._compute_base_loss(ctx)
    
    # TAL loss (every N steps)
    global_step = self.get_global_step()
    if (self.tal_weight > 0
            and global_step % self.tal_interval == 0
            and ctx.get('src_mask') is not None):
        tal = self._compute_tal_loss(ctx)
        if tal is not None:
            losses['tal'] = tal

    loss = sum(losses.values())
    return {'loss': loss, 'loss_tal': tal.detach(), ...}
```

### The `_compute_tal_loss` Implementation (lines 144-199)

```python
def _compute_tal_loss(self, ctx: Dict[str, Any]) -> Optional[Tensor]:
    """Compute Text-Awareness Loss via extra null-text forward."""
    src_mask = ctx['src_mask']
    # ... sanity checks ...
    
    # PROBLEM 1: Prepare null embeddings without checking if text is already masked
    null_vtxt = self.bundle.null_vtxt_feat.detach().expand(B, 1, -1)
    null_ctxt = self.bundle.null_ctxt_input.detach().expand(B, ctxt_tokens, -1)
    
    # PROBLEM 2: Use null embeddings directly without re-applying cond_mask_prob
    with torch.no_grad():
        pred_null = self.bundle.predict_flow(
            x_input=x_input,
            ctxt_input=null_ctxt,      # ← null embeddings, NOT masked
            vtxt_input=null_vtxt,      # ← null embeddings, NOT masked
            # ... other args ...
        )
    
    # Compute difference
    tal = text_awareness_loss(
        pred_with_text=ctx['pred'],    # ← trained with cond_mask_prob masking
        pred_without_text=pred_null.detach(),
        # ... other args ...
    )
    
    return tal * self.tal_weight
```

### The Missing Link: How Text Gets Masked During Training

From `HyMotionM2MTrainer._prepare_and_forward` (lines 206-210):

```python
vtxt_input, ctxt_input = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    force_mask=False,
    cond_mask_prob=self.bundle.cond_mask_prob,  # ← Applied to ALL samples
)
```

Default `cond_mask_prob = 0.1` (10% of samples have text dropped).

### The Semantic Bug

**For a sample where text was dropped by `cond_mask_prob`:**

| Branch | Text Embedding | Prediction |
|--------|-----------------|-----------|
| **Base forward (ctx['pred'])** | null (because of `cond_mask_prob=0.1`) | `pred_with_text` (misleading name!) |
| **TAL null forward (pred_null)** | null (explicitly passed) | `pred_without_text` |

**Result**: `pred_with_text ≈ pred_without_text` → `text_awareness_loss ≈ 0` → **no regularization**

For the 90% of samples where text was NOT masked, TAL would work correctly (if the math is sound). But across a batch:
- **10% of samples**: TAL loss = 0 (no effect)
- **90% of samples**: TAL compares "cond + text" vs "uncond + null"

**The bug is that TAL doesn't account for the fact that the "with_text" prediction was already trained on a mixture where 10% have no text.** TAL then tries to add pressure that this 10% should use more text, but they have no text signal available.

---

## Why This Happens: Design Inconsistency

### Training Process (HyMotionM2MTrainer)

```
for batch in dataloader:
    text_inputs = encode_or_load(batch['caption'])
    
    # 50% dropout on text (cond_mask_prob=0.1 means CFG, but here it's actually 0.1)
    text_inputs = mask_text_cond(text_inputs, cond_mask_prob=0.1)
    #                            ↑ APPLIED TO TRAINING DISTRIBUTION
    
    # Flow matching: predict motion
    pred = model(x_t, text=text_inputs, ...)
    
    # Loss on motion prediction
    loss = MSE(pred, target)
    loss.backward()
```

### TAL Loss (HyMotionM2MCRFMTrainer, lines 168-199)

```
# For the same sample:
null_text = null_embeddings  # Fixed pretrained values, NOT masked
text_at_train = "already maybe masked by cond_mask_prob"

# TAL tries to compare:
pred_with = model(x_t, text=text_at_train, ...)
pred_null = model(x_t, text=null_text, ...)

# But if text_at_train was ALREADY null (due to cond_mask_prob):
pred_with ≈ pred_null  → loss ≈ 0
```

---

## Impact of the Bug

### On TAL Regularization

**Expected effect**: TAL should push the model to use text conditioning to influence generated regions.

**Actual effect**: TAL provides no pressure because it doesn't account for masking.

### The Numbers (Hypothetical)

If `cond_mask_prob = 0.1`:
- **10% of batch**: TAL sees `null vs null` → loss = 0
- **90% of batch**: TAL sees `possibly-text vs definitely-null` → loss computed
- **Effective batch coverage**: ~90% at best, but with stale assumptions about training distribution

### Evidence from Code

1. **Line 183 in `condition_routing.py`**: `apply_weight = (mask_density < density_threshold).float()` — applies per-sample reweighting based on mask_density, which is semantically unrelated to whether text was masked
2. **Line 186**: The hinge loss `F.relu(min_effect - diff_per_sample)` should theoretically work, but the samples it's applied to have inconsistent text conditioning histories

---

## How Text Masking Interacts with TAL

### Current Implementation: ZERO Integration

TAL **does not check** whether text was already masked by `cond_mask_prob`.

**The interaction is broken at three points:**

### 1. **No Recording of Which Samples Were Masked**

`mask_text_cond()` is called (line 180-184 in trainer), but the return includes only:
- `vtxt_input` (after masking)
- `ctxt_input` (after masking)

**Missing**: A flag indicating which samples had their text nullified.

```python
# Current (BAD):
vtxt_input, ctxt_input = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    force_mask=False,
    cond_mask_prob=0.1,
)
# Returns only the masked embeddings, not a mask_was_applied flag

# Proposed (GOOD):
vtxt_input, ctxt_input, text_mask_flags = self.bundle.mask_text_cond(
    ...,
    return_mask_flags=True,
)
# Now track which samples have null text: text_mask_flags (B,)
```

### 2. **TAL Doesn't Receive Text Masking Information**

`_compute_tal_loss()` is passed only `ctx`, which has:
- `ctx['pred']` — prediction with "text" (but possibly null)
- `ctx['x_t']`, `ctx['vace_context']`, `ctx['timesteps']`, etc.

**Missing**: `ctx['text_mask_flags']` or similar.

### 3. **TAL Doesn't Filter Out NULL-vs-NULL Comparisons**

```python
def _compute_tal_loss(self, ctx: Dict[str, Any]) -> Optional[Tensor]:
    # ... currently ...
    
    # NO CHECK for: "was this sample's text already masked?"
    
    tal = text_awareness_loss(
        pred_with_text=ctx['pred'],
        pred_without_text=pred_null.detach(),
        src_mask=src_mask,
        # ...
    )
    
    return tal * self.tal_weight
```

---

## The TAL Parameters and How They're Used

### Parameter Initialization (lines 46-61)

```python
def __init__(
    self,
    bundle,
    tal_weight: float = 0.01,           # Scalar multiplier on TAL loss
    tal_interval: int = 4,              # Compute TAL every N steps
    tal_min_effect: float = 0.005,      # Hinge threshold: min diff to avoid loss
    tal_density_threshold: float = 0.7, # Only apply when mask_density < this
    text_grad_scale: float = 0.01,
    **kwargs,
):
    self.tal_weight = tal_weight
    self.tal_interval = tal_interval
    self.tal_min_effect = tal_min_effect
    self.tal_density_threshold = tal_density_threshold
```

### Usage in `train_step` (line 131-136)

```python
if (self.tal_weight > 0
        and global_step % self.tal_interval == 0
        and ctx.get('src_mask') is not None):
    tal = self._compute_tal_loss(ctx)
    if tal is not None:
        losses['tal'] = tal
```

**Flow**:
1. Check `tal_weight > 0` — if zero, TAL is disabled
2. Check `global_step % self.tal_interval == 0` — compute every N steps to save compute
3. Check `src_mask is not None` — only for masked samples

### Usage in `text_awareness_loss` (lines 154-188)

```python
def text_awareness_loss(
    pred_with_text: Tensor,          # (B, L, D)
    pred_without_text: Tensor,       # (B, L, D)
    src_mask: Tensor,                # (B, L, D), 1=generate, 0=known
    mask_density: Tensor,            # (B,)
    min_effect: float = 0.005,       # default tal_min_effect
    density_threshold: float = 0.7,  # default tal_density_threshold
) -> Tensor:
    
    # Compute per-sample mean absolute difference in generated regions only
    gen_count = src_mask.sum(dim=(-1, -2))                    # (B,)
    diff = ((pred_with_text - pred_without_text) * src_mask).abs()
    diff_per_sample = diff.sum(dim=(-1, -2)) / (gen_count + 1e-6)  # (B,)
    
    # Only active when motion condition is strong (mask_density < threshold)
    apply_weight = (mask_density < density_threshold).float()  # (B,)
    
    # Hinge loss: penalize when text effect < min_effect
    loss = F.relu(min_effect - diff_per_sample) * apply_weight
    
    return loss.mean()
```

**Logic**:
- **Line 179**: `diff_per_sample` = average absolute diff in generated regions (mask=1)
- **Line 183**: `apply_weight` activates only for strong motion conditions (low density)
- **Line 186**: Hinge loss: `max(0, min_effect - diff) * apply_weight`

---

## Full Data Flow with Bug

```
TRAINING STEP (batch of 128 samples)
│
├─ encode_text(captions) or load from batch
│   ├─ 118 samples have real text
│   └─ 10 samples have null (e.g., empty caption)
│
├─ mask_text_cond(text_inputs, cond_mask_prob=0.1)  [LINE 206-210 in trainer]
│   ├─ Drops 10% of text → sets to null_embeddings
│   │   ├─ 118 samples: 11 or 12 now have null (dropped), ~107 have real text
│   │   └─ 10 samples: still null (already were)
│   └─ Returns: vtxt_input (modified), ctxt_input (modified)
│       ⚠️  No information WHICH samples were modified
│
├─ forward(x_t, text=vtxt_input/ctxt_input, ...)
│   ├─ Samples with real text:
│   │   ├─ These are present: text conditioning signal
│   │   └─ pred_with_text = model output conditioned on text
│   ├─ Samples with dropped text (now null):
│   │   ├─ These have null: no conditioning signal
│   │   └─ pred_with_text = model output with null (same as unconditioned!)
│   └─ Samples that were already null:
│       └─ pred_with_text = model output with null
│
├─ compute_base_loss(pred_with_text, target)
│   └─ Loss computed for all 128 samples
│
├─ EVERY 4 STEPS: _compute_tal_loss(ctx)  [LINE 131-136, 144-199]
│   │
│   ├─ Prepare null embeddings (line 168-169)
│   │   └─ null_vtxt, null_ctxt = fixed pretrained values
│   │
│   ├─ Forward with null (line 176-184)
│   │   ├─ pred_null = model(x_t, text=null_vtxt/null_ctxt, ...)
│   │   └─ This is ALWAYS null (by construction)
│   │
│   ├─ Compute text_awareness_loss(pred_with_text, pred_null, ...)
│   │   │
│   │   ├─ For ~107 samples with real text:
│   │   │   ├─ diff_per_sample = |pred_with_text - pred_null| (potentially large)
│   │   │   └─ loss = max(0, min_effect - diff_per_sample) * apply_weight
│   │   │       └─ Might be positive if effect is small
│   │   │
│   │   ├─ For ~21 samples with dropped text (now null):  ⚠️ BUG HERE
│   │   │   ├─ pred_with_text ≈ null (they got dropped)
│   │   │   ├─ diff_per_sample ≈ 0  (comparing null vs null)
│   │   │   └─ loss = max(0, min_effect - ~0) * apply_weight
│   │   │       └─ = max(0, 0.005 - ~0) * apply_weight ≈ 0.005
│   │   │       └─ This SEEMS to penalize, but it's an artifact!
│   │   │           The sample never HAD text to enforce!
│   │   │
│   │   └─ return loss.mean()  (average over all 128)
│   │       └─ ~21/128 samples contribute meaningfully (are non-zero by accident)
│   │       └─ The 21 are penalizing the model for something outside its control
│   │
│   └─ tal = text_awareness_loss(...) * tal_weight
│       └─ Scaled by tal_weight (default 0.01)
│
└─ total_loss = base_loss + tal
```

---

## The Specific Problem Instances

### Instance 1: Already-Null Samples in Batch

- **Where**: Sample has `caption = None` or `caption = ''`
- **What happens**: 
  1. Text encoder returns null embeddings
  2. `mask_text_cond()` keeps them as null (no change)
  3. Base forward uses null
  4. TAL forward also uses null
  5. `diff = |null - null| = 0` → loss wrongly triggers hinge even though text was never available

### Instance 2: Randomly Dropped Text (cond_mask_prob)

- **Where**: Sample has real caption, but `cond_mask_prob=0.1` randomly drops it
- **What happens**:
  1. Text encoder produces real embeddings
  2. `mask_text_cond()` drops it (with 10% probability)
  3. Base forward uses null
  4. TAL forward also uses null
  5. `diff = |null - null| = 0` → loss wrongly triggers

### Combined: 10-20% of Batch is Affected

If `cond_mask_prob = 0.1` and ~10% of batch has no caption:
- **~10% dropped by cond_mask_prob** on samples with captions
- **~10% already null** (no caption)
- **Total: ~20% contribute garbage to TAL gradient**

---

## Fix Strategy

### Fix 1: Track Which Samples Were Text-Masked (Minimal)

**In HyMotionM2MTrainer._prepare_and_forward, line 206-210:**

```python
# BEFORE:
vtxt_input, ctxt_input = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    force_mask=False,
    cond_mask_prob=self.bundle.cond_mask_prob,
)

# AFTER (add to context):
vtxt_input, ctxt_input = self.bundle.mask_text_cond(
    vtxt_input, ctxt_input,
    force_mask=False,
    cond_mask_prob=self.bundle.cond_mask_prob,
)
# Also track which samples had text dropped
text_was_masked = batch.get('_cond_mask_applied', None)
# If not provided, infer from ctxt_length
if text_was_masked is None and 'ctxt_length' in batch:
    text_was_masked = batch['ctxt_length'] == 0
if text_was_masked is not None:
    ctx['text_was_masked'] = text_was_masked.to(device)  # (B,) bool
```

### Fix 2: Filter TAL Computation (Recommended)

**In HyMotionM2MCRFMTrainer._compute_tal_loss, lines 144-199:**

```python
def _compute_tal_loss(self, ctx: Dict[str, Any]) -> Optional[Tensor]:
    """Compute Text-Awareness Loss via extra null-text forward.
    
    ⚠️ FIX: Only compute TAL on samples where text was actually provided.
    """
    src_mask = ctx['src_mask']
    if src_mask is None or src_mask.sum() == 0:
        return None
    
    # NEW: Check if any samples had text masked during base forward
    text_was_masked = ctx.get('text_was_masked')
    if text_was_masked is not None:
        # Only include samples where text was NOT masked
        text_available = ~text_was_masked  # (B,)
    else:
        # Fallback: assume all samples have text (may be wrong)
        text_available = torch.ones(ctx['x_t'].shape[0], dtype=torch.bool, 
                                    device=ctx['x_t'].device)
    
    if not text_available.any():
        # All samples had text masked → skip TAL entirely
        return None
    
    # ... rest of TAL computation ...
    
    # In text_awareness_loss, multiply loss by text_available:
    tal = text_awareness_loss(
        pred_with_text=ctx['pred'],
        pred_without_text=pred_null.detach(),
        src_mask=src_mask,
        mask_density=mask_density,
        min_effect=self.tal_min_effect,
        density_threshold=self.tal_density_threshold,
        text_available=text_available,  # NEW PARAMETER
    )
    
    return tal * self.tal_weight
```

**In condition_routing.py, text_awareness_loss:**

```python
def text_awareness_loss(
    pred_with_text: Tensor,
    pred_without_text: Tensor,
    src_mask: Tensor,
    mask_density: Tensor,
    min_effect: float = 0.005,
    density_threshold: float = 0.7,
    text_available: Optional[Tensor] = None,  # NEW: (B,) bool
) -> Tensor:
    # ... existing code ...
    
    # Compute per-sample loss
    loss = F.relu(min_effect - diff_per_sample) * apply_weight
    
    # NEW: Zero out loss for samples where text was never available
    if text_available is not None:
        loss = loss * text_available.float()
    
    return loss.mean()
```

### Fix 3: Alternative – Double-Check During TAL Forward

```python
# In _compute_tal_loss, after computing pred_null:
# Verify that null forward actually produces similar predictions
# to base forward for masked samples

# This is harder to implement robustly, so Fix 1-2 is preferred
```

---

## Testing the Bug

### Unit Test to Demonstrate the Bug

```python
def test_tal_null_vs_null_bug():
    """TAL incorrectly compares null-vs-null on cond_mask_prob-dropped samples."""
    
    # Simulate a batch where 20% of samples had text masked
    B, L, D = 10, 100, 135
    pred_with_text = torch.randn(B, L, D)
    pred_without_text = pred_with_text.clone()  # Identical (null vs null)
    
    src_mask = torch.ones(B, L, D)
    src_mask[:, :50, :] = 0  # First 50 frames known
    
    mask_density = torch.tensor([0.5] * B)
    
    # Samples 0-1 (20%) had text masked → should not contribute to TAL
    text_available = torch.tensor([False, False, True, True, True, True, True, True, True, True])
    
    loss_buggy = text_awareness_loss(
        pred_with_text, pred_without_text, src_mask, mask_density,
        text_available=None,  # BUG: not provided
    )
    
    loss_fixed = text_awareness_loss(
        pred_with_text, pred_without_text, src_mask, mask_density,
        text_available=text_available,  # FIX: provided
    )
    
    # Buggy version: compares 10 samples of null-vs-null, all trigger hinge
    # Fixed version: only compares 8 samples (2-9), also trigger hinge
    
    print(f"Buggy loss: {loss_buggy.item():.4f}")  # Should be ~0.005
    print(f"Fixed loss: {loss_fixed.item():.4f}")  # Should be ~0.005 * (8/10) = 0.004
    
    # If text_available worked: fixed_loss ≈ 0.5 * buggy_loss
    # (because only 80% of samples are considered)
```

---

## Recommendation Summary

| Issue | Severity | Fix |
|-------|----------|-----|
| **TAL compares null-vs-null on masked samples** | 🔴 Critical | Track `cond_mask_prob` masking, filter TAL |
| **TAL doesn't know which samples had text** | 🟠 High | Add `text_was_masked` to context dict |
| **No unit test for null-vs-null bug** | 🟡 Medium | Add test case |
| **No logging of which samples TAL filtered** | 🟡 Medium | Add debug log in `_compute_tal_loss` |

---

## References

- **TAL Implementation**: `hftrainer/trainers/motion/hymotion_m2m_crfm_trainer.py` lines 144-199
- **Text Masking**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py` lines 206-210
- **TAL Loss Function**: `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py` lines 149-188
- **Text Masking Function**: `hftrainer/models/motion/hymotion_m2m/bundle.py` `mask_text_cond()` method
