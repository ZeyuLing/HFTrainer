# PRISM Motion Deformation Bug Fix - Implementation Guide

## Root Cause Summary

**Verified Bug**: Missing `hidden_states_mask` parameter in inference pipeline
- **Training**: Passes `hidden_states_mask=padding_mask` to transformer
- **Inference**: Never passes `hidden_states_mask` parameter
- **Effect**: Model trained to ignore padded positions, but at inference attends to all positions including padding
- **Result**: Distribution mismatch causes latent corruption, cumulative over 50 denoising steps → deformed motion

## Fixed Files

### 1. hftrainer/pipelines/motion/prism_backend.py (Lines 372-411)

**Changes**: Add motion_mask computation and pass it to both CFG forward passes

**Before** (Broken - Lines 392-410):
```python
noise_pred = current_model(
    hidden_states=latent_model_input,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    attention_kwargs=attention_kwargs,
    is_causal=self.config.is_causal,
)

if do_cfg:
    noise_uncond = current_model(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=negative_prompt_embeds,
        attention_kwargs=attention_kwargs,
        is_causal=self.config.is_causal,
    )
```

**After** (Fixed):
```python
# Compute motion_mask before denoising loop (see implementation below)
# Then pass to both CFG calls:

noise_pred = current_model(
    hidden_states=latent_model_input,
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    hidden_states_mask=motion_mask,  # ← ADDED
    attention_kwargs=attention_kwargs,
    is_causal=self.config.is_causal,
)

if do_cfg:
    noise_uncond = current_model(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=negative_prompt_embeds,
        hidden_states_mask=motion_mask,  # ← ADDED
        attention_kwargs=attention_kwargs,
        is_causal=self.config.is_causal,
    )
```

### 2. Minimal Fix Location in generate_single_segment()

Add motion_mask computation right after timesteps are prepared (around line 358):

```python
# After: self.scheduler.set_timesteps(num_inference_steps, device=device)
#        timesteps = self.scheduler.timesteps

# ADD THIS BLOCK:
# Create motion_mask in latent space [B, T_latent, J]
# All frames are valid (no padding in inference), so mask is all-ones
num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
motion_mask = torch.ones(
    batch_size,
    num_latent_frames,
    num_joints,
    dtype=transformer_dtype,
    device=device
)

# Pass motion_mask to both noise_pred and noise_uncond calls
```

## Validation Test Script

See `tests/motion/test_prism_hidden_states_mask_fix.py` for comprehensive validation including:
1. Verifies mask is passed to transformer
2. Checks mask shape and dtype
3. Validates mask content (all 1s for no-padding case)
4. Confirms both CFG branches receive mask
5. Runs inference and validates output is not NaN/Inf

## Testing Instructions

```bash
# Run comprehensive fix validation
python -m pytest tests/motion/test_prism_hidden_states_mask_fix.py -v

# Or run specific test
python -m pytest tests/motion/test_prism_hidden_states_mask_fix.py::test_hidden_states_mask_passed_to_transformer -v

# Check if fix resolves deformation (qualitative)
python scripts/test_prism_inference_quality.py --prompt "a person walking forward"
```

## Expected Results After Fix

1. **Inference output normalization**: Model predictions should have normal magnitude
2. **Deformation reduction**: Motion output should appear less twisted/deformed
3. **Training/inference consistency**: Attention to padding is now consistent
4. **Quality metrics**: Reduction in jitter, foot skating, and unnatural poses

## Implementation Notes

- Motion mask should be shape `[B, T_latent, J]` where `T_latent = ceil((num_frames-1)/scale_factor) + 1`
- For inference with no padding, mask is all 1.0 (attend to all frames)
- Both CFG branches (text and unconditional) must receive the same mask
- Mask must be on same device and dtype as transformer input

## Verification Checklist

- [ ] Mask parameter is passed to BOTH noise_pred and noise_uncond calls
- [ ] Mask has correct shape `[B, T_latent, J]`
- [ ] Mask values are float (not bool)
- [ ] Motion mask is created before denoising loop
- [ ] Test suite passes all validation checks
- [ ] Inference output is not NaN/Inf
- [ ] Visual quality appears improved (less deformation)

