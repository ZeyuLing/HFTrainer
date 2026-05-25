# Exact Code Changes Required

## File: `scripts/embodied/physflow_eval_demo.py`

### Function: `generate_motion_from_bundle()` (lines 187-276)

This shows the **exact changes** needed in the function definition and early setup:

---

## CHANGE 1: Docstring (line 190)

### Before:
```python
def generate_motion_from_bundle(bundle, prompt: str, num_frames: int,
                                 device: torch.device, num_ode_steps: int = 50,
                                 cfg_scale: float = 4.5) -> np.ndarray:
    """Generate motion_135 using a loaded T2M bundle.

    Replicates the PhysFlowTrainer.generate_motion() logic.
    """
```

### After:
```python
def generate_motion_from_bundle(bundle, prompt: str, num_frames: int,
                                 device: torch.device, num_ode_steps: int = 50,
                                 cfg_scale: float = 4.5) -> np.ndarray:
    """Generate motion_135 using a loaded T2M bundle.

    Replicates the HyMotionT2MPipeline inference logic with fixes:
    - BUG FIX #1: motion_dim = 135 (not 201)
    - BUG FIX #2: L_padded = max(L, TRAIN_FRAMES) (not hardcoded 360)
    - BUG FIX #3: ctxt_mask_temporal uses < (not >=) for correct masking
    """
```

**Why**: Update docstring to reflect the bug fixes applied.

---

## CHANGE 2: Motion Dimension Setup (lines 196-198)

### Before:
```python
    bundle.eval()
    TRAIN_FRAMES = 360
    motion_dim = 201
```

### After:
```python
    bundle.eval()
    TRAIN_FRAMES = 360
    # FIX #1: Use correct motion dimension from the bundle
    motion_dim = bundle.motion_transformer.output_dim  # Should be 135
```

**Why**: Fix critical bug where motion_dim=201 doesn't match the model's expected 135-dim input.
**Effect**: Prevents shape mismatch in ODE solver and latent space corruption.

---

## CHANGE 3: Padding Length Calculation (lines 206-208)

### Before:
```python
    B = 1
    L = num_frames
    L_padded = TRAIN_FRAMES
```

### After:
```python
    B = 1
    L = num_frames
    # FIX #2: Pad to max of requested length and TRAIN_FRAMES (not hardcoded 360)
    L_padded = max(L, TRAIN_FRAMES)
```

**Why**: Fix bug where sequences longer than 360 frames are silently truncated.
**Effect**: Supports variable-length sequence generation without data loss.

---

## CHANGE 4: Context Masking (lines 210-217)

### Before:
```python
    # Context mask
    max_ctxt_len = ctxt_input.shape[1]
    ctxt_mask_temporal = torch.arange(max_ctxt_len, device=device).unsqueeze(0) >= ctxt_len.unsqueeze(1)

    # Target padding mask
    tgt_padding_mask = _length_to_mask(
        torch.tensor([L], dtype=torch.long, device=device), L_padded
    )
```

### After:
```python
    # FIX #3: Use _length_to_mask for consistent masking (properly handles < logic)
    ctxt_mask_temporal = _length_to_mask(ctxt_len, ctxt_input.shape[1])

    # Target padding mask
    tgt_padding_mask = _length_to_mask(
        torch.tensor([L], dtype=torch.long, device=device), L_padded
    )
```

**Why**: Fix inverted context mask that destroyed text conditioning by ignoring real tokens and attending to padding.
**Effect**: Restores text-to-motion guidance by properly masking the context.

---

## CHANGE 5: Motion Denormalization (lines 271-274)

### Before:
```python
    sampled = x[:, :L, :]
    latent_denorm = bundle.denormalize_motion(sampled)
    motion_201 = latent_denorm[0].cpu().numpy()
    motion_135 = motion_201[:, :135].astype(np.float32)

    return motion_135
```

### After:
```python
    # Truncate to requested length and denormalize
    sampled = x[:, :L, :]
    latent_denorm = bundle.denormalize_motion(sampled)
    motion_135 = latent_denorm[0].cpu().numpy().astype(np.float32)

    return motion_135
```

**Why**: Simplify and clarify final denormalization step (motion_201 variable is no longer needed since motion_dim=135).
**Effect**: Makes code consistent with actual data dimensions.

---

## Summary of Changes

| Line(s) | Before | After | Bug |
|---------|--------|-------|-----|
| 190-193 | Generic docstring | Specific bug fix notes | Documentation |
| 196-198 | `motion_dim = 201` | `motion_dim = bundle.motion_transformer.output_dim` | #1 |
| 208 | `L_padded = TRAIN_FRAMES` | `L_padded = max(L, TRAIN_FRAMES)` | #2 |
| 210-212 | Manual `>= ctxt_len` mask | `_length_to_mask(ctxt_len, ...)` | #3 |
| 271-275 | 201→135 slicing logic | Direct 135-dim handling | Cleanup |

---

## Patch Script

To apply all changes at once, use this sed script:

```bash
cd scripts/embodied/

# Create backup
cp physflow_eval_demo.py physflow_eval_demo.py.backup

# Apply changes (requires careful escaping - better to manually edit or use the FIXED version)
# Manual editing recommended for safety
```

**Recommendation**: Use the provided `physflow_eval_demo_FIXED.py` file to replace the buggy version:

```bash
cp physflow_eval_demo_FIXED.py physflow_eval_demo.py
```

Or manually apply the 5 changes above to your existing file.

---

## Testing the Fix

After applying changes, run:

```python
#!/usr/bin/env python3
import torch
import numpy as np
from scripts.embodied.physflow_eval_demo import generate_motion_from_bundle, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
config_path = '/path/to/config.py'
ckpt_path = '/path/to/checkpoint.pt'
bundle = load_model(config_path, ckpt_path, device, is_physflow_ckpt=True)

# Test 1: Check dimension
prompt = "a person walks forward slowly"
motion = generate_motion_from_bundle(bundle, prompt, num_frames=120, device=device)
assert motion.shape == (120, 135), f"Expected (120, 135), got {motion.shape}"
print(f"✓ Motion dimension correct: {motion.shape}")

# Test 2: Check different lengths work
for nframes in [60, 90, 150, 300]:
    motion = generate_motion_from_bundle(bundle, prompt, num_frames=nframes, device=device)
    assert motion.shape == (nframes, 135), f"Expected ({nframes}, 135), got {motion.shape}"
    print(f"✓ Length {nframes} works: {motion.shape}")

# Test 3: Check text guidance impact
motion_no_cfg = generate_motion_from_bundle(bundle, prompt, num_frames=120, device=device, cfg_scale=1.0)
motion_with_cfg = generate_motion_from_bundle(bundle, prompt, num_frames=120, device=device, cfg_scale=5.0)
diff = np.mean(np.abs(motion_no_cfg - motion_with_cfg))
print(f"✓ CFG impact: mean difference = {diff:.4f}")
assert diff > 0.01, "CFG should produce different results"

print("\n✅ All tests passed! The fix is working correctly.")
```

