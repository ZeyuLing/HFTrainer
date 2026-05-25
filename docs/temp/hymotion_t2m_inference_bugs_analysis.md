# HyMotion T2M Inference Pipeline: Bug Analysis Report

**Date:** 2026-05-21  
**Status:** CRITICAL BUGS IDENTIFIED  
**Severity:** 🔴 Breaks all text-to-motion generation

---

## Quick Summary

The custom `generate_motion_from_bundle()` in `scripts/embodied/physflow_eval_demo.py` (lines 187-276) has **3 critical bugs** that completely break text-to-motion generation:

1. **Motion dimension mismatch** (line 198): `motion_dim = 201` should be `motion_dim = bundle.motion_transformer.output_dim` (135)
2. **Sequence padding bug** (line 208): `L_padded = TRAIN_FRAMES` should be `L_padded = max(L, TRAIN_FRAMES)`
3. **Inverted context mask** (line 212): `>= ctxt_len` should be `< ctxt_len` (or use `_length_to_mask()`)

**Result:** Generated motions don't follow text prompts because:
- The model receives 201-dimensional noise but expects 135-dimensional
- The transformer attends to padding tokens instead of real text tokens
- The entire ODE trajectory is corrupted

---

## Detailed Analysis

### Bug #1: Motion Dimension Mismatch (THE SMOKING GUN)

**Location:** `physflow_eval_demo.py:198`

**Current (WRONG):**
```python
motion_dim = 201
```

**Should be (CORRECT):**
```python
motion_dim = bundle.motion_transformer.output_dim  # Will be 135
```

**Why this breaks everything:**

- HyMotion-T2M-1.0-Lite's transformer expects **135-dimensional** input (SMPL-22 representation):
  - Translation: 3 dims
  - 22 joints × 6D rot6d: 132 dims
  - Total: 3 + 132 = 135 dims

- The custom code creates noise in **201-dimensional** space (likely from old codebase)

- What happens during ODE integration:
  ```python
  y0 = torch.randn(B, L_padded, 201, device=device)  # WRONG SHAPE!
  # ...
  v = bundle.predict_flow(x_input=x, ...)  # Model expects (B, L, 135)
  # Broadcasting error or silent truncation/padding
  ```

- The transformer's **input projection layer** (input_encoder) is trained on 135-dim data:
  - If you pass 201-dim noise, it either errors or gets corrupted
  - Even if it somehow runs, the model has never seen this distribution during training
  - The latent space is completely different

**Verification:**
```python
# Check model's actual output dimension:
print(bundle.motion_transformer.output_dim)  # Should print 135
```

**Impact Severity:** 🔴 CRITICAL - Makes entire ODE trajectory invalid

---

### Bug #2: Padding Logic

**Location:** `physflow_eval_demo.py:208`

**Current (WRONG):**
```python
L_padded = TRAIN_FRAMES  # Always 360, never changes
```

**Should be (CORRECT):**
```python
L_padded = max(L, TRAIN_FRAMES)  # 360 minimum, but can be larger
```

**Why this breaks long sequences:**

- HyMotion-T2M was trained with sequences of **360 frames**
- Padding to 360 matches the training distribution
- The official pipeline does: `L_padded = max(L, TRAIN_FRAMES)`

- What happens with custom code:
  ```python
  # User requests 720 frames
  L = 720
  L_padded = TRAIN_FRAMES = 360  # WRONG! Silently truncates!
  
  # ODE runs on 360 frames, output gets truncated to 720[:360]
  # Only 360 frames are generated, not 720
  ```

- What happens with correct code:
  ```python
  # User requests 720 frames
  L = 720
  L_padded = max(720, 360) = 720  # Correct!
  
  # ODE runs on 720 frames
  # Attention patterns see the full 720-frame context
  ```

**For short sequences (L < 360):**
- Custom code: L_padded = 360 (correct by accident)
- Correct code: L_padded = 360 (correct intentionally)
- Both work, but for different reasons

**For long sequences (L ≥ 360):**
- Custom code: L_padded = 360 (WRONG! truncates output)
- Correct code: L_padded = L (CORRECT! generates full length)

**Impact Severity:** 🟠 HIGH - Silently truncates long sequences

---

### Bug #3: Inverted Context Mask

**Location:** `physflow_eval_demo.py:212`

**Current (WRONG):**
```python
ctxt_mask_temporal = torch.arange(max_ctxt_len, device=device).unsqueeze(0) >= ctxt_len.unsqueeze(1)
```

**Should be (CORRECT):**
```python
from hftrainer.models.motion.hymotion_t2m.bundle import _length_to_mask
ctxt_mask_temporal = _length_to_mask(ctxt_len, max_ctxt_len)
```

**Understanding the mask:**

A context mask tells the transformer which tokens are real vs padding:
- `True` = valid token, attend to this
- `False` = padding, don't attend to this

**Example:**
```python
# Text: "a person walks"
# Encoded into 6 tokens, padded to max_len=10
# ctxt_len = 6 (6 real tokens, 4 padding)

# CORRECT mask:
# [T, T, T, T, T, T, F, F, F, F]
# Position 0-5: True (real tokens)
# Position 6-9: False (padding)

# WRONG mask (inverted):
# [F, F, F, F, F, F, T, T, T, T]
# Position 0-5: False (masks out real tokens!)
# Position 6-9: True (attends to padding!)
```

**How the custom code is inverted:**

```python
# Custom (WRONG):
arange >= ctxt_len
# arange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# ctxt_len = 6
# arange >= 6 → [F, F, F, F, F, F, T, T, T, T]  ← INVERTED!

# Official (CORRECT):
arange < ctxt_len
# arange = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# ctxt_len = 6
# arange < 6 → [T, T, T, T, T, T, F, F, F, F]  ← CORRECT!
```

**What happens during generation:**

The transformer's **cross-attention layer** uses this mask:
```
attention_weights = softmax(Q @ K^T + mask)
```

With the inverted mask:
- Real text tokens are masked out (set to -inf)
- Padding tokens are attended to (set to 0)
- The model sees random noise instead of text

**Result:** Text conditioning is completely broken. The model generates motion unconditionally, ignoring the prompt entirely.

**Impact Severity:** 🔴 CRITICAL - Makes text conditioning ineffective

---

## Why Output Doesn't Match Prompts

Combining all three bugs:

1. **Wrong dimension** → ODE explores invalid latent space
2. **Inverted mask** → Text signal is zeroed out
3. **Wrong padding** → Sequence length distribution is wrong

The model receives:
- ✗ Wrong-shaped noise (201 instead of 135)
- ✗ Invalid text conditioning (mask inverted)
- ✗ Wrong sequence context (L_padded always 360)

Output: **Completely random, unconditioned motion**

This explains why "motions don't match text prompts at all" — there's literally no text signal reaching the model!

---

## Official Reference Implementation

**File:** `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`

**Correct implementation (lines 42-182):**

```python
@torch.no_grad()
def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    device = next(self.bundle.motion_transformer.parameters()).device

    # Determine sequence lengths
    tgt_length = batch.get('tgt_length', batch.get('num_frames'))
    if isinstance(tgt_length, Tensor):
        tgt_length = tgt_length.tolist()
    if isinstance(tgt_length, int):
        tgt_length = [tgt_length]

    B = len(tgt_length)
    L = max(tgt_length)

    # ✅ CORRECT: Pad to at least TRAIN_FRAMES
    TRAIN_FRAMES = 360
    L_padded = max(L, TRAIN_FRAMES)

    # ✅ CORRECT: Infer motion dim from transformer output_dim
    motion_dim = batch.get('motion_dim', self.bundle.motion_transformer.output_dim)

    tgt_padding_mask = _length_to_mask(
        torch.tensor(tgt_length, dtype=torch.long, device=device), L_padded
    )

    # Prepare text
    if batch.get('text_vec_raw') is not None:
        vtxt_input = batch['text_vec_raw'].to(device)
        ctxt_input = batch['text_ctxt_raw'].to(device)
        ctxt_length = batch['text_ctxt_raw_length'].to(device)
        # ✅ CORRECT: Use _length_to_mask for context
        ctxt_mask_temporal = _length_to_mask(ctxt_length, ctxt_input.shape[1])
    # ... more code ...
    
    # ODE integration
    y0 = torch.randn(B, L_padded, motion_dim, device=device, dtype=dtype)
    # ... rest of pipeline ...
```

---

## How to Fix

### Option 1: Quick Patch (Recommended)

Replace the `generate_motion_from_bundle()` function in `scripts/embodied/physflow_eval_demo.py` with the corrected version:

```python
def generate_motion_from_bundle(bundle, prompt: str, num_frames: int,
                                 device: torch.device, num_ode_steps: int = 50,
                                 cfg_scale: float = 4.5) -> np.ndarray:
    """Generate motion_135 using a loaded T2M bundle (CORRECTED)."""
    from hftrainer.models.motion.hymotion_t2m.bundle import _length_to_mask

    bundle.eval()
    TRAIN_FRAMES = 360
    
    # FIX #1: Use correct motion dimension
    motion_dim = bundle.motion_transformer.output_dim  # 135, not 201
    
    # Encode text
    text_feats = bundle.encode_text([prompt])
    vtxt_input = text_feats['text_vec_raw'].to(device)
    ctxt_input = text_feats['text_ctxt_raw'].to(device)
    ctxt_len = text_feats['text_ctxt_raw_length'].to(device)

    B = 1
    L = num_frames
    
    # FIX #2: Handle long sequences correctly
    L_padded = max(L, TRAIN_FRAMES)

    # FIX #3: Use correct context mask
    max_ctxt_len = ctxt_input.shape[1]
    ctxt_mask_temporal = _length_to_mask(ctxt_len, max_ctxt_len)

    tgt_padding_mask = _length_to_mask(
        torch.tensor([L], dtype=torch.long, device=device), L_padded
    )

    # CFG setup (unchanged)
    do_cfg = cfg_scale > 1.0
    if do_cfg:
        null_vtxt = bundle.null_vtxt_feat.expand_as(vtxt_input)
        vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
        ctxt_cfg = torch.cat([ctxt_input, ctxt_input], dim=0)
        ctxt_mask_cfg = torch.cat([ctxt_mask_temporal, ctxt_mask_temporal], dim=0)

    def fn(t_val, x):
        # ODE function (unchanged)
        if do_cfg:
            x_double = torch.cat([x, x], dim=0)
            x_pred = bundle.predict_flow(
                x_input=x_double,
                ctxt_input=ctxt_cfg,
                vtxt_input=vtxt_cfg,
                timesteps=t_val.expand(2 * B),
                x_mask_temporal=tgt_padding_mask.repeat(2, 1),
                ctxt_mask_temporal=ctxt_mask_cfg,
            )
        else:
            x_pred = bundle.predict_flow(
                x_input=x,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t_val.expand(B),
                x_mask_temporal=tgt_padding_mask,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )

        if bundle.pred_type == 'x1':
            t_eps = 0.05
            if do_cfg:
                x_pred = (x_pred - torch.cat([x, x], dim=0)) / (1.0 - t_val).clamp_min(t_eps)
            else:
                x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)

        if do_cfg:
            pred_uncond, pred_text = x_pred.chunk(2, dim=0)
            x_pred = pred_uncond + cfg_scale * (pred_text - pred_uncond)

        return x_pred

    # ODE integration (unchanged)
    y0 = torch.randn(B, L_padded, motion_dim, device=device, dtype=torch.float32)
    dt = 1.0 / num_ode_steps
    x = y0
    with torch.no_grad():
        for i in range(num_ode_steps):
            t_val = torch.tensor(i * dt, device=device, dtype=torch.float32)
            v = fn(t_val, x)
            x = x + v * dt

    sampled = x[:, :L, :]
    latent_denorm = bundle.denormalize_motion(sampled)
    motion_201 = latent_denorm[0].cpu().numpy()
    motion_135 = motion_201[:, :135].astype(np.float32)

    return motion_135
```

### Option 2: Use Official Pipeline

Replace the custom implementation entirely with the official `HyMotionT2MPipeline`:

```python
from hftrainer.pipelines.motion import HyMotionT2MPipeline

# Create pipeline
pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=50,
    text_guidance_scale=4.5,
)

# Run inference
batch = {
    'caption': prompt,
    'num_frames': num_frames,
}
result = pipeline(batch)
motion_135 = result['rot6d'].numpy()  # (1, L, 22, 6)
# ... or use result['transl'] for translation ...
```

---

## Verification Checklist

After applying the fix, verify:

- [ ] Model dimension check: `print(bundle.motion_transformer.output_dim)` → should be `135`
- [ ] Text encoding: `text_feats = bundle.encode_text(['a person walks'])` → should have shape `(1, 768)` for vtxt
- [ ] Context mask: Print mask shape, should be `(1, max_ctxt_len)` with mostly `True` and some `False` at end
- [ ] Generate short motion: `generate_motion_from_bundle(bundle, 'walk', 90, device)` → shape `(90, 135)`
- [ ] Generate long motion: `generate_motion_from_bundle(bundle, 'walk', 720, device)` → shape `(720, 135)` NOT `(360, 135)`
- [ ] Visual inspection: Generated motion should now match text prompt (e.g., walking motion for "walk" prompt)

---

## Related Files

- **Official implementation:** `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
- **Bundle implementation:** `hftrainer/models/motion/hymotion_t2m/bundle.py`
- **Training reference:** `scripts/embodied/physflow_trainer.py:358-465`
- **Buggy custom code:** `scripts/embodied/physflow_eval_demo.py:187-276`

---

## Timeline

| Date | Status | Notes |
|------|--------|-------|
| 2026-05-21 | 🔴 IDENTIFIED | All 3 bugs found and documented |
| — | 🟡 READY FOR FIX | Corrected code available in this doc |
| — | ⏳ PENDING | Awaiting merge of corrected implementation |

---

## Questions?

- **Why was 201-dim used?** Likely copied from old HyMotion-T2M-1.0 codebase which used extended representation
- **Why wasn't this caught earlier?** No test cases comparing against official pipeline output
- **Does official pipeline have this bug?** No, official pipeline is correct as-is
- **Will this fix break other code?** No, this only fixes the custom function. Official pipeline is unchanged.

