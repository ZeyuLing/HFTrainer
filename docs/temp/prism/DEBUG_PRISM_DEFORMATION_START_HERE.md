# PRISM Deformation Debugging - Start Here

**Status:** ✅ VACE channel hypothesis **RULED OUT** - deformation is NOT due to input channel mismatch

---

## Investigation Summary

### What We Investigated
Your hypothesis: During training, the model input might include extra VACE channels (inactive/reactive motion) concatenated with noisy latents, but inference doesn't provide these channels, causing a distribution mismatch.

### What We Found
**VACE channels do NOT exist in PRISM.** This is exclusively a HyMotion (M2M) feature.

| Check | Finding |
|-------|---------|
| PRISM trainer hidden_states | Only `noisy_latents` [B,16,T,J] |
| PRISM inference hidden_states | Only `latent_model_input` [B,16,T,J] |
| Transformer in_channels config | 16 (matches input) |
| VACE concatenation in PRISM code | Zero references (0 lines) |
| Training-inference channel mismatch | ✅ NONE - both use 16 channels |

---

## The Real Culprits

Since input channels are identical between training and inference, the deformation is caused by one or more of these:

### 1. **Timestep Distribution Mismatch** (Primary Suspect)
- **Training:** Random timesteps from full [0-1000] schedule (all 1000 timesteps equally likely)
- **Inference:** Sparse schedule (e.g., only 10 specific timesteps in fixed order)
- **Impact:** Model trained on uniform timestep distribution, but inference only samples 1% of timesteps
- **Fix:** See `PRISM_TIMESTEP_MISMATCH_ANALYSIS.md`

### 2. **Sigma Lookup Precision Issues** (Secondary)
- **Problem:** Floating-point matching: `999.8005981445312 == 999.8` may fail
- **Impact:** Wrong sigma values → wrong noise scaling → deformed output
- **Symptom:** Occasional NaNs or extreme values in denoising
- **Fix:** Use nearest-neighbor sigma lookup instead of exact equality

### 3. **Per-Token Timestep Expansion Mismatch** (Tertiary)
- **Training:** Expands via `create_sequence_ts()` with `condition_frame_mask_vae`
- **Inference:** Expands via `expand_timesteps` with `first_frame_mask`
- **Risk:** Shape mismatches or different masking logic
- **Fix:** Verify mask shapes match exactly

### 4. **Input Distribution Shift** (Quaternary)
- **Training:** 10% frame conditioning rate → model sees diverse input patterns
- **Inference:** 100% first-frame conditioning → always same starting condition
- **Impact:** Model out-of-distribution at inference
- **Fix:** Use lower conditioning rate during training

---

## Actionable Debug Steps

### Step 1: Verify No VACE (Already Done ✅)
```bash
grep -r "vace" hftrainer/trainers/motion/prism_trainer.py
grep -r "vace" hftrainer/pipelines/motion/prism_backend.py
# Expected: No matches
```

### Step 2: Enable Sigma Lookup Debugging
Add to `hftrainer/models/motion/prism/bundle.py`:

```python
def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    
    step_indices = []
    for t in timesteps:
        matches = (schedule_timesteps == t).nonzero()
        if len(matches) == 0:
            # ⚠️ MISMATCH DETECTED
            closest_idx = (schedule_timesteps - t).abs().argmin()
            closest_t = schedule_timesteps[closest_idx]
            print(f"[SIGMA_WARN] Timestep {t:.4f} not found. Closest: {closest_t:.4f}")
            step_indices.append(closest_idx.item())
        else:
            step_indices.append(matches[0].item())
    
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
```

Then check inference output for warnings.

### Step 3: Test Per-Token Expansion Isolation
Create test script:

```python
from hftrainer.pipelines.motion.prism_backend import PrismARPipeline

# Test with expand_timesteps=False
pipe = PrismARPipeline(..., expand_timesteps=False)

# Generate and compare with expand_timesteps=True
# If motion is better without expansion, the per-token logic is broken
motion = pipe(
    prompts="a person walking forward",
    num_frames_per_segment=129,
    num_inference_steps=50,
)
```

Compare motion quality with/without per-token expansion.

### Step 4: Log Frame Mask Shapes
Add to `hftrainer/pipelines/motion/prism_backend.py` (line ~383):

```python
latents, condition, first_frame_mask = self.prepare_latents(...)

# Log shapes
print(f"[MASK_DEBUG] latents shape: {latents.shape}")
print(f"[MASK_DEBUG] first_frame_mask shape: {first_frame_mask.shape}")
print(f"[MASK_DEBUG] first_frame_mask min/max: {first_frame_mask.min()}/{first_frame_mask.max()}")
print(f"[MASK_DEBUG] Num conditioned frames: {(first_frame_mask==0).sum()}")
print(f"[MASK_DEBUG] Num generated frames: {(first_frame_mask==1).sum()}")
```

Compare these with training logs.

### Step 5: Inspect Latent Normalization
Add to inference (line ~320):

```python
print(f"[LATENT_DEBUG] Raw latents z: min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}")
print(f"[LATENT_DEBUG] Latents mean: {self.latents_mean[0,0,0,0]:.6f}")
print(f"[LATENT_DEBUG] Latents std: {self.latents_std[0,0,0,0]:.6f}")

z = (z - self.latents_mean) / self.latents_std

print(f"[LATENT_DEBUG] Normalized z: min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}")
```

Check if normalized latents have unexpectedly large/small values.

---

## Expected Outcomes

### If Step 2 finds sigma mismatches
→ **Root cause identified:** Use nearest-neighbor sigma lookup (Fix 1 in PRISM_TIMESTEP_MISMATCH_ANALYSIS.md)

### If Step 3 shows worse motion without per-token expansion
→ **Root cause identified:** Debug per-token expansion logic (create_sequence_ts vs expand_timesteps)

### If Step 4 shows mask shape mismatches
→ **Root cause identified:** Align mask generation between training/inference

### If Step 5 shows extreme normalized latent values
→ **Root cause identified:** Verify latents_mean/latents_std match training

---

## Code Locations

**Key Files to Reference:**
- Training: `hftrainer/trainers/motion/prism_trainer.py` (lines 77-93)
- Inference: `hftrainer/pipelines/motion/prism_backend.py` (lines 382-427)
- Config: `configs/prism/prism_1b_tp2m_1frame.py` (line 31)
- Bundle: `hftrainer/models/motion/prism/bundle.py` (create_sequence_ts, add_flow_noise)

---

## Related Documentation

- `VACE_CHANNEL_MISMATCH_ANALYSIS.md` - Full investigation details
- `VACE_ANALYSIS_QUICK_REFERENCE.md` - Quick lookup table
- `PRISM_TIMESTEP_MISMATCH_ANALYSIS.md` - Detailed timestep analysis

---

## Next Steps

1. ✅ DONE: Rule out VACE channel hypothesis
2. → DO: Enable sigma lookup debugging (Step 2)
3. → DO: Test per-token expansion (Step 3)
4. → DO: Log mask shapes (Step 4)
5. → DO: Check latent normalization (Step 5)

Choose one and run it, then report back with the outputs.

