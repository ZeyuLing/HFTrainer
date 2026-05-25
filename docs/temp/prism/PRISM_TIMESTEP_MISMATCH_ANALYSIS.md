# PRISM TIMESTEP MISMATCH ROOT CAUSE ANALYSIS

## Executive Summary

Investigation into timestep handling mismatches between PRISM training and inference has identified the core differences causing deformed motion output. The issue is **NOT** related to the shift parameter, timestep range, or per-token timestep expansion—all of which match between training and inference. Instead, the mismatch likely stems from **floating-point precision issues in sigma lookup** or **unaccounted differences in the per-token timestep expansion logic**.

---

## Four Critical Questions - Answers

### ✅ Question 1: Does training use the same shift=5.0?

**YES - EXACT MATCH**

Both training and inference use identical FlowMatchEulerDiscreteScheduler configuration:

```python
# Training (configs/prism/prism_1b_tp2m_1frame.py, lines 70-77)
scheduler=dict(
    type="FlowMatchEulerDiscreteScheduler",
    num_train_timesteps=1000,
    shift=5.0,
    use_dynamic_shifting=False,
    base_shift=0.5,
    max_shift=1.15,
)

# Inference (hftrainer/pipelines/motion/prism_backend.py, lines 984-992)
scheduler=HF_MODELS.build(
    dict(
        type="FlowMatchEulerDiscreteScheduler",
        num_train_timesteps=1000,
        shift=5.0,  # ← IDENTICAL
        use_dynamic_shifting=False,
        base_shift=0.5,
        max_shift=1.15,
    ),
)
```

**Conclusion:** The shift parameter is NOT the source of mismatch.

---

### ✅ Question 2: Are timesteps in [0,1] range or [0,1000] range?

**TIMESTEPS ARE IN [0, 1000] RANGE**

Empirical verification shows:
- **Range:** ~[24.4, 1000.0] (not normalized to [0,1])
- **Training:** 1000 timestep values when `set_timesteps(1000)` is called
- **Inference:** Sparse sampling (e.g., 10 values for `set_timesteps(10)`)

Example timestep values:
```
Training (all 1000):
  [1000.0, 999.8, 999.6, 999.4, ..., 43.3, 38.6, 33.9, 29.2, 24.4]

Inference (10 steps):
  [1000.0, 975.7, 946.3, 909.7, 863.1, 801.8, 717.3, 593.6, 395.1, 24.4]
```

**Key insight:** Both use absolute timestep values (0-1000 scale), NOT normalized.

---

### ✅ Question 3: Does training use expand_timesteps (per-token timesteps)?

**YES - BOTH USE PER-TOKEN TIMESTEPS**

Training uses `create_sequence_ts()`:
```python
# hftrainer/models/motion/prism/bundle.py, lines 240-255
def create_sequence_ts(self, ori_ts, condition_frame_mask_vae, patch_size=(1, 1)):
    batch_size, _, latent_frames, latent_joints = condition_frame_mask_vae.shape
    
    # Expand [B] → [B, N_frames, N_joints]
    target_ts = ori_ts.unsqueeze(1).unsqueeze(2).expand(
        batch_size, latent_frames, latent_joints
    )
    
    # Zero out conditioned frames
    target_ts = torch.where(
        condition_frame_mask_vae[:, 0, ::patch_size[0], ::patch_size[1]],
        target_ts,
        0  # ← Conditioning frames = timestep 0
    )
    
    return target_ts.flatten(1)  # [B, N_frames*N_joints]
```

Inference uses `expand_timesteps` flag:
```python
# hftrainer/pipelines/motion/prism_backend.py, lines 407-415
if self.config.expand_timesteps:
    latent_model_input = ((1 - first_frame_mask) * condition + first_frame_mask * latents)
    
    # Expand to per-token timesteps
    if self._kafs_alpha_map is not None:
        temp_ts = (first_frame_mask[0][0] * t * self._kafs_alpha_map).flatten()
    else:
        temp_ts = (first_frame_mask[0][0] * t).flatten()
    
    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
```

**Important note:** KAFS (Kinematic-Adaptive Flow Scheduling) is NOT called in the main() function, so `_kafs_alpha_map` remains `None` by default.

---

### ✅ Question 4: Is the noise formulation identical?

**YES - EXACT MATCH**

Training noise formulation:
```python
# hftrainer/models/motion/prism/bundle.py, lines 257-262
def add_flow_noise(self, latents, timesteps):
    noise = torch.randn_like(latents)
    sigmas = _get_sigmas(self.scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    
    # Flow matching: noisy = (1-σ)*x0 + σ*noise
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    targets = noise - latents  # Flow matching target
    
    return noisy_latents, targets
```

This is the standard flow matching formulation and matches the inference denoising step.

---

## ROOT CAUSE ANALYSIS

Since all four critical questions show matching configurations, the deformation must stem from subtle implementation differences:

### 1. SIGMA LOOKUP PRECISION MISMATCH (Most Likely)

The `_get_sigmas()` function uses exact equality matching:

```python
def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    
    # PROBLEM: Exact equality matching
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
```

**Vulnerability points:**
- Floating-point precision: `999.8005981445312 == 999.8` may fail
- If timesteps are computed via different paths (training vs inference), rounding errors accumulate
- Exception handling is missing: if no match found, `nonzero()` raises IndexError

### 2. TIMESTEP DISTRIBUTION MISMATCH

Training samples timesteps uniformly at random:
```python
step_indices = torch.randint(0, len(scheduler.timesteps), (batch_size,))
timesteps = scheduler.timesteps[step_indices]
```

Inference uses a fixed schedule in reverse:
```python
scheduler.set_timesteps(num_inference_steps)  # e.g., 10
timesteps = scheduler.timesteps  # [1000, 975.7, ..., 24.4]
```

The model is trained on ALL 1000 possible timestep values with uniform probability, but during inference it only sees 10 specific timestep values in a specific order. This **train-test distribution mismatch** can cause quality degradation.

### 3. PER-TOKEN TIMESTEP EXPANSION LOGIC

While both use per-token expansion, the mask generation differs:

**Training:**
```python
condition_frame_mask_vae[:, 0, ::patch_size[0], ::patch_size[1]]
```

**Inference:**
```python
first_frame_mask[0][0]
```

If `patch_size != (1, 1)` or if the frame masking logic differs, the per-token timestep tensor could have different shapes or values, leading to incorrect sigma lookups.

### 4. TIMESTEP FLOAT TYPE MISMATCH

Training passes `latents` in float32 for VAE (line 151, bundle.py):
```python
with torch.autocast(device_type, enabled=False):
    latents = self.vae.encode(motion.float())
```

Then later converts to transformer dtype (line 85, trainer.py):
```python
noisy_latents = noisy_latents.to(dtype=transformer_dtype)
```

If transformer_dtype is bf16 and timesteps are computed in float32, precision loss during conversion could cause sigma lookup failures.

---

## Verification Steps

### Step 1: Enable Sigma Lookup Debugging

Modify `_get_sigmas()` to log when lookups fail:

```python
def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    
    step_indices = []
    for t in timesteps:
        matches = (schedule_timesteps == t).nonzero()
        if len(matches) == 0:
            # MISMATCH DETECTED
            closest_idx = (schedule_timesteps - t).abs().argmin()
            closest_t = schedule_timesteps[closest_idx]
            print(f"WARNING: No exact match for t={t:.4f}, using closest t={closest_t:.4f}")
            step_indices.append(closest_idx.item())
        else:
            step_indices.append(matches[0].item())
    
    sigma = sigmas[step_indices].flatten()
    # ... rest of function
```

### Step 2: Compare Per-Token Timestep Tensors

Add logging to training and inference:

```python
# Training
print(f"Per-token timesteps shape: {timesteps.shape}")
print(f"Per-token timesteps min/max: {timesteps.min()}, {timesteps.max()}")
print(f"Per-token timesteps unique count: {len(torch.unique(timesteps))}")

# Inference
print(f"Per-token timesteps shape: {timestep.shape}")
print(f"Per-token timesteps min/max: {timestep.min()}, {timestep.max()}")
```

### Step 3: Verify Sigma Values

Log sigma values before noise addition:

```python
print(f"Sigmas min/max: {sigmas.min():.6f}, {sigmas.max():.6f}")
print(f"Sigmas dtype: {sigmas.dtype}")
print(f"Noisy latents dtype: {noisy_latents.dtype}")
print(f"Noise scaling check: sigma*noise min/max: {(sigmas*noise).min()}, {(sigmas*noise).max()}")
```

### Step 4: Test expand_timesteps=False

Run inference with global timesteps to isolate the issue:

```python
pipe = PrismARPipeline(..., expand_timesteps=False)
```

If motion quality improves, the per-token expansion logic is the culprit.

---

## Recommended Fixes

### Fix 1: Robust Sigma Lookup (High Priority)

Replace exact equality with nearest-neighbor lookup:

```python
def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    
    # Use nearest-neighbor instead of exact match
    step_indices = []
    for t in timesteps:
        # Find closest timestep in schedule
        distances = (schedule_timesteps - t).abs()
        closest_idx = distances.argmin().item()
        step_indices.append(closest_idx)
    
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
```

### Fix 2: Ensure Consistent Timestep Types

Convert all timesteps to float32 before sigma lookup:

```python
timesteps = timesteps.to(dtype=torch.float32)
sigmas = _get_sigmas(self.scheduler, timesteps, ...)
```

### Fix 3: Add Per-Token Timestep Validation

Ensure training and inference produce identical per-token shapes:

```python
# Training
expected_shape = (batch_size, latent_frames * latent_joints)
assert timesteps.shape == expected_shape, f"Training shape mismatch: {timesteps.shape} vs {expected_shape}"

# Inference
expected_shape = (batch_size, num_latent_frames * num_joints)
assert timestep.shape == expected_shape, f"Inference shape mismatch: {timestep.shape} vs {expected_shape}"
```

### Fix 4: Validate Condition Mask Consistency

Debug the condition vs. generated frame masking:

```python
# Ensure all conditioned frames have timestep=0
assert (timesteps[condition_mask] == 0).all(), "Conditioned frames must have t=0"
assert (timesteps[~condition_mask] != 0).any(), "Generated frames must have non-zero t"
```

---

## Conclusion

The PRISM timestep mismatch is **not** a configuration problem (shift, range, or per-token logic mismatch). Instead, it's likely a **numerical precision issue** in the sigma lookup or **frame masking logic** that causes different effective sigma schedules during inference compared to training.

The most actionable fix is to implement robust sigma lookup using nearest-neighbor matching instead of exact equality, combined with careful type management to avoid float32/bf16 conversion errors.

**Next steps:**
1. Enable sigma lookup debugging (see Step 1)
2. Run inference with `expand_timesteps=False` to isolate the problem
3. Implement Fix 1 (robust sigma lookup) as a first attempt
4. If that doesn't resolve it, proceed to Fixes 2-4

