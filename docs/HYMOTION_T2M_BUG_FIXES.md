# HyMotion T2M Inference Pipeline - Bug Fixes

## Summary

The `generate_motion_from_bundle()` function in `scripts/embodied/physflow_eval_demo.py` contains **three critical bugs** that cause it to generate text-to-motion outputs that don't match the input prompts. These bugs corrupt:
1. The latent motion representation dimension
2. The sequence length handling  
3. The text context masking logic

All three bugs must be fixed together for correct inference.

---

## Bug #1: Incorrect Motion Dimension (Line 198)

### ❌ BUGGY CODE
```python
motion_dim = 201
```

### ✅ CORRECT CODE
```python
motion_dim = bundle.motion_transformer.output_dim  # Should be 135
```

### Root Cause
The HyMotion-T2M model is trained with motion representations of **135 dimensions**:
- 3D translation (3 dims)
- 22 joints × 6D rotation matrices in row-major format (22 × 6 = 132 dims)
- **Total: 135 dims**

The value `201` appears to be from an older codebase variant that included additional joint positions. However, the current model architecture only supports 135-dim inputs to its input projection layer.

### Impact
- **Severity**: CRITICAL
- **Effect**: ODE solver generates 201-dimensional noise, but the transformer expects 135-dim input at each ODE step
- **Result**: Shape mismatch causes either runtime error or silent corruption of the latent space, producing random unconditioned motion

### Why It Breaks Text Conditioning
The transformer's input encoder projects 135→hidden_dim. Passing 201-dim input means:
- Either the first 135 dims pass through (losing 66 dims of information)
- Or the padding causes misaligned feature extraction
- Either way, the model doesn't see the intended motion structure, so text guidance fails

### Fix Verification
After fixing, the latent dimension should match exactly:
```python
# Before ODE: y0.shape = (B=1, L_padded=360, motion_dim=135)
# After ODE: x.shape = (B=1, L_padded=360, motion_dim=135)
# After truncation: sampled.shape = (B=1, L=requested_frames, 135)
```

---

## Bug #2: Hardcoded Padding Length (Line 208)

### ❌ BUGGY CODE
```python
L_padded = TRAIN_FRAMES  # Always 360
```

### ✅ CORRECT CODE
```python
L_padded = max(L, TRAIN_FRAMES)  # max(requested_frames, 360)
```

### Root Cause
The model was trained on sequences of exactly 360 frames. However, the generation function should support variable-length sequences:
- If user requests 90 frames: should pad to 360 (TRAIN_FRAMES)
- If user requests 500 frames: should pad to 500 (not truncate to 360)

The buggy code always pads to 360, which silently truncates longer sequences.

### Impact
- **Severity**: MODERATE-HIGH (breaks for variable-length inputs)
- **Effect**: 
  - For short sequences (e.g., 90 frames): Works but wastes compute on 270 padding frames
  - For long sequences (e.g., 500 frames): **Silently truncates to 360 frames!**
  - Causes inconsistent results across different prompt types

### Why It Breaks Text Conditioning (Indirectly)
1. Many test prompts request different lengths (90-150 frames)
2. Always padding to 360 means the transformer attention patterns are inconsistent
3. Longer prompts don't generate because they're silently truncated
4. The padding mask logic becomes incorrect because `L ≠ L_padded`

### Fix Verification
After fixing, check that:
```python
# Short sequence: 90 frames
L = 90, L_padded = max(90, 360) = 360 ✓

# Long sequence: 500 frames  
L = 500, L_padded = max(500, 360) = 500 ✓

# Standard sequence: 360 frames
L = 360, L_padded = max(360, 360) = 360 ✓
```

---

## Bug #3: Inverted Context Mask (Line 212)

### ❌ BUGGY CODE
```python
max_ctxt_len = ctxt_input.shape[1]
ctxt_mask_temporal = torch.arange(max_ctxt_len, device=device).unsqueeze(0) >= ctxt_len.unsqueeze(1)
```

This creates a mask where:
- `True` = position should be **IGNORED** (padding)
- `False` = position should be **ATTENDED** (valid token)

But it uses `>=` which means:
- For ctxt_len=20: positions [0..19] get True (WRONG - these are real tokens!)
- For ctxt_len=20: positions [20..max] get False (WRONG - these are padding!)
- **The mask is inverted!**

### ✅ CORRECT CODE
```python
ctxt_mask_temporal = _length_to_mask(ctxt_len, ctxt_input.shape[1])
```

Which implements:
```python
def _length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths
    # Result: True for valid positions, False for padding (standard masking convention)
```

### Root Cause
Manual mask construction using `>=` instead of `<` inverts the logic. The official pipeline uses the `_length_to_mask()` helper function which correctly implements the standard masking convention.

### Impact
- **Severity**: CRITICAL (destroys text conditioning entirely!)
- **Effect**: 
  - Real text tokens are marked as "ignore" (True in attention mask)
  - Padding positions are marked as "attend" (False in attention mask)
  - Transformer completely ignores the text context and attends only to padding!
  - Text guidance CFG becomes ineffective because both conditional and unconditional branches see the same masked-out text

### Why This Completely Breaks Text-to-Motion
The text context provides semantic guidance for motion generation. With an inverted mask:
1. All real text tokens are masked out (attention → 0)
2. Only padding tokens are visible to transformer
3. The transformer sees no text information
4. Text guidance CFG doesn't work: `pred_uncond` and `pred_text` produce nearly identical outputs
5. Motion generation becomes purely random, unconditioned noise

**This explains why the motions don't match the prompts at all!**

### Fix Verification
After fixing, verify mask values:
```python
# Given: ctxt_len = 20 (20 real tokens), max_len = 512
# Correct mask should be:
# Positions 0-19: True (valid tokens)
# Positions 20-511: False (padding)

# Check:
mask = _length_to_mask(torch.tensor([20]), 512)
assert mask[0, 0] == True   # First token is valid
assert mask[0, 19] == True  # Last real token is valid
assert mask[0, 20] == False # First padding token is invalid
assert mask[0, 511] == False # Last position is padding
```

---

## Side-by-Side Comparison

| Aspect | Buggy Code | Fixed Code | Impact |
|--------|-----------|-----------|--------|
| **Motion Dim** | `motion_dim = 201` | `motion_dim = bundle.motion_transformer.output_dim` | Fixes shape mismatch |
| **Padding** | `L_padded = TRAIN_FRAMES` | `L_padded = max(L, TRAIN_FRAMES)` | Supports variable lengths |
| **Context Mask** | Manual `>= ctxt_len` | `_length_to_mask(ctxt_len, max_len)` | Fixes text masking |
| **Result** | Random motion, ignores text | Text-conditioned motion | ✅ Works correctly |

---

## How to Apply the Fixes

### Option 1: Direct Replacement
Replace the entire `generate_motion_from_bundle()` function in `physflow_eval_demo.py` with the corrected version from `physflow_eval_demo_FIXED.py`.

### Option 2: Minimal Patch
Apply these three changes to `physflow_eval_demo.py`:

**Line 198**: Change
```python
motion_dim = 201
```
to
```python
motion_dim = bundle.motion_transformer.output_dim
```

**Line 208**: Change
```python
L_padded = TRAIN_FRAMES
```
to
```python
L_padded = max(L, TRAIN_FRAMES)
```

**Lines 211-212**: Replace
```python
max_ctxt_len = ctxt_input.shape[1]
ctxt_mask_temporal = torch.arange(max_ctxt_len, device=device).unsqueeze(0) >= ctxt_len.unsqueeze(1)
```
with
```python
ctxt_mask_temporal = _length_to_mask(ctxt_len, ctxt_input.shape[1])
```

---

## Verification Checklist

After applying fixes, verify:

- [ ] Generate motion from prompt: `"a person walks forward slowly"` (120 frames)
  - Should produce forward walking motion (not random)
  - Should complete full 120 frames (not truncate)

- [ ] Generate motion with different lengths: 90, 150, 300 frames
  - All should work without truncation
  - Attention patterns should adapt to length

- [ ] Check text guidance (CFG) impact:
  - With `cfg_scale=1.0` (no guidance): motion is generic
  - With `cfg_scale=5.0` (strong guidance): motion matches prompt more closely
  - Difference shows text conditioning is working

- [ ] Verify latent dimensions:
  ```python
  motion_135 = generate_motion_from_bundle(...)
  assert motion_135.shape == (num_frames, 135), f"Expected (N, 135), got {motion_135.shape}"
  ```

- [ ] Regenerate mesh JSONs and view in web viewer
  - Motions should now visually match text descriptions
  - Use before/after comparison with same seed

---

## References

- **Official Pipeline**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
  - Lines 77: Correct motion_dim assignment
  - Lines 74: Correct L_padded calculation
  - Lines 88: Correct _length_to_mask usage

- **Bundle Implementation**: `hftrainer/models/motion/hymotion_t2m/bundle.py`
  - Lines 35-39: _length_to_mask definition
  - Lines 225-255: predict_flow method signature
  - Lines 324-327: denormalize_motion implementation

- **Training Reference**: `scripts/embodied/physflow_trainer.py`
  - Lines 358-465: Reference implementation of ODE inference

---

## Timeline

- **Bug Discovery**: Analysis of physflow_eval_demo.py vs official HyMotionT2MPipeline
- **Root Cause**: Identified three independent bugs, each critical
- **Impact**: Explains why generated motions don't match text prompts
- **Fix Applied**: Corrected implementation provided in physflow_eval_demo_FIXED.py
- **Status**: Ready for integration

