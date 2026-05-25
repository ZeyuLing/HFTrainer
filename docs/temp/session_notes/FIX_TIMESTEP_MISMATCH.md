# Fix PRISM Timestep Mismatch - Implementation Guide

## Quick Start

The root cause is likely **floating-point precision issues in sigma lookup**. Follow these steps to debug and fix:

---

## Step 1: Apply Robust Sigma Lookup (HIGHEST PRIORITY)

Edit `/apdcephfs/.../hftrainer/models/motion/prism/bundle.py`:

Replace the `_get_sigmas()` function (lines 19-27):

```python
# BEFORE (fragile exact matching)
def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

# AFTER (robust nearest-neighbor matching)
def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    
    # Use nearest-neighbor matching for robustness
    step_indices = []
    for t in timesteps:
        distances = (schedule_timesteps - t).abs()
        closest_idx = distances.argmin().item()
        step_indices.append(closest_idx)
    
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
```

**Why this fixes it:**
- Exact equality fails when `999.8005981445312 != 999.8` due to float precision
- Nearest-neighbor is robust to small rounding errors
- Always finds a valid sigma value (no IndexError crashes)

---

## Step 2: Debug Sigma Lookup (FOR VERIFICATION)

Add logging to track sigma value mismatches:

In `/apdcephfs/.../hftrainer/models/motion/prism/bundle.py`, modify `add_flow_noise()`:

```python
def add_flow_noise(self, latents: torch.Tensor, timesteps: torch.Tensor):
    noise = torch.randn_like(latents)
    
    # DEBUG: Check timestep values
    print(f"[DEBUG add_flow_noise] timesteps dtype: {timesteps.dtype}, shape: {timesteps.shape}")
    print(f"[DEBUG add_flow_noise] timesteps min/max: {timesteps.min():.4f}/{timesteps.max():.4f}")
    
    sigmas = _get_sigmas(self.scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    
    # DEBUG: Check sigma values
    print(f"[DEBUG add_flow_noise] sigmas dtype: {sigmas.dtype}, shape: {sigmas.shape}")
    print(f"[DEBUG add_flow_noise] sigmas min/max: {sigmas.min():.6f}/{sigmas.max():.6f}")
    
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    targets = noise - latents
    return noisy_latents, targets
```

Run training with this logging and save the output. Check if:
- Any timesteps fail to find a match in the schedule
- Sigma values are reasonable [0, 1]
- Sigma dtype matches latents dtype

---

## Step 3: Test expand_timesteps=False (ISOLATION TEST)

Modify the inference call in `prism_backend.py` main() function (line 995):

```python
# CURRENT
expand_timesteps=expand_timesteps,

# TEMPORARY TEST (disable per-token timesteps)
expand_timesteps=False,
```

Generate a motion sample and compare quality:
- If quality improves → per-token expansion logic has a bug
- If quality stays the same → issue is elsewhere

Then revert this change after testing.

---

## Step 4: Verify Timestep Type Consistency

In `/apdcephfs/.../hftrainer/models/motion/prism/bundle.py`, add type validation:

```python
def add_flow_noise(self, latents: torch.Tensor, timesteps: torch.Tensor):
    # Ensure timesteps are float32 for reliable sigma lookup
    if timesteps.dtype != torch.float32:
        print(f"[WARNING] Converting timesteps from {timesteps.dtype} to float32")
        timesteps = timesteps.to(dtype=torch.float32)
    
    noise = torch.randn_like(latents)
    sigmas = _get_sigmas(self.scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    
    # Rest of function...
```

---

## Step 5: Validate Per-Token Timestep Shapes (IF ISSUE PERSISTS)

Add shape assertions in training (prism_trainer.py, line 79):

```python
# After create_sequence_ts call
timesteps = self.bundle.create_sequence_ts(
    timesteps,
    condition_frame_mask_vae,
    self.bundle.transformer.config.patch_size,
)

# Validate shape
batch_size, _, latent_frames, latent_joints = condition_frame_mask_vae.shape
expected_shape = (batch_size, latent_frames * latent_joints)
assert timesteps.shape == expected_shape, \
    f"Per-token timestep shape mismatch: got {timesteps.shape}, expected {expected_shape}"
```

Add similar validation in inference (prism_backend.py, line 415):

```python
timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

# Validate shape
batch_size, channels, frames, joints = latents.shape
expected_shape = (batch_size, frames * joints)
assert timestep.shape == expected_shape, \
    f"Per-token timestep shape mismatch: got {timestep.shape}, expected {expected_shape}"
```

---

## Step 6: Check Patch Size Configuration (IF ISSUE PERSISTS)

Verify that the transformer config has `patch_size=(1, 1)`:

```bash
grep -n "patch_size" /apdcephfs/.../configs/prism/prism_1b_tp2m_1frame.py
```

Expected output:
```
24:        patch_size=(1, 1),
```

If patch_size is different, the per-token expansion logic will need adjustment.

---

## Testing Checklist

After applying fixes, run through this checklist:

- [ ] Training runs without timestep errors
- [ ] Sigma lookup produces valid values [0, 1]
- [ ] Generated motion shows no obvious deformation
- [ ] Per-token timestep shapes are consistent between training and inference
- [ ] All assertions pass

---

## Rollback if Needed

If the fix makes things worse:

1. Revert `_get_sigmas()` to original
2. Check git diff to see what changed
3. Run `git checkout -- hftrainer/models/motion/prism/bundle.py`

---

## Key Takeaway

The most likely cause of PRISM motion deformation is that **floating-point timestep values fail to match** during sigma lookup due to precision limits. Using nearest-neighbor matching instead of exact equality makes the lookup robust and should resolve most deformation issues.

If this doesn't fix the problem, it's likely a deeper issue in the per-token timestep expansion or frame masking logic—in which case Steps 3-6 will help identify the real culprit.

