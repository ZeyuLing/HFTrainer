# PRISM Flow Matching - Complete Code Reference

## 1. NOISING AND TARGET COMPUTATION

### File: `bundle.py` (lines 257-262)
```python
def add_flow_noise(self, latents: torch.Tensor, timesteps: torch.Tensor):
    noise = torch.randn_like(latents)
    sigmas = _get_sigmas(self.scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    targets = noise - latents
    return noisy_latents, targets
```

**What happens:**
- `noise`: Random Gaussian noise (same shape as `latents`)
- `sigmas`: Sigma values from scheduler, reshaped to [B, 1, 1, 1]
- `noisy_latents`: The noised version following: $x_t = (1-\sigma_t)x_0 + \sigma_t\epsilon$
- `targets`: The velocity field the model learns to predict: $v = \epsilon - x_0$

---

## 2. SIGMA RETRIEVAL FUNCTION

### File: `bundle.py` (lines 19-27)
```python
def _get_sigmas(scheduler, timesteps, n_dim: int = 4, dtype=torch.float32):
    device = timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device=device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma
```

**What happens:**
1. Get all sigmas from scheduler: `[σ₀, σ₁, σ₂, ..., σ₁₀₀₀]`
2. Get all timesteps from scheduler: `[0, 10, 20, ..., 999]`
3. For each sampled timestep, find its index in the schedule
4. Extract corresponding sigma values
5. Reshape to match latent dimensionality (4D for motion: `[B, 1, 1, 1]`)
6. Broadcasting will make it `[B, C, T, J]`

---

## 3. TIMESTEP SAMPLING

### File: `prism_trainer.py` (lines 68-75)
```python
step_indices = torch.randint(
    0,
    len(self.bundle.scheduler.timesteps),
    (batch_size,),
    device=latents.device,
)
scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
timesteps = scheduler_timesteps[step_indices]
```

**What happens:**
- Sample `batch_size` random integers from `[0, num_timesteps)`
- Each integer is an index into `scheduler.timesteps`
- If `num_timesteps = 1000` and sampled indices are `[50, 150, 750, ...]`, then `timesteps = [timesteps[50], timesteps[150], timesteps[750], ...]`
- Each sample in the batch gets a potentially different timestep

**Example:**
```
scheduler.timesteps = [0, 10, 20, 30, ..., 990]  # 100 total steps
step_indices = torch.randint(0, 100, (32,))  # 32 batch samples
# Result might be: [5, 87, 2, 45, 12, 99, ...]
timesteps = scheduler.timesteps[[5, 87, 2, 45, 12, 99, ...]]
# Result: [50, 870, 20, 450, 120, 990, ...]
```

---

## 4. COMPLETE TRAINING STEP (SIMPLIFIED)

### File: `prism_trainer.py` (lines 41-118)

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    # Step 1: Encode motion
    motion = batch['motion']  # Shape: [B, T, J*6]
    latents = self.bundle.encode_motion(motion)  # Shape: [B, C, T', J]
    batch_size, _, latent_frames, latent_joints = latents.shape
    
    # Step 2: Create masks
    padding_mask = self.bundle.create_padding_mask(...)  # [B, T', J]
    condition_frame_mask_vae = self.bundle.create_condition_mask(latents, ...)
    
    # Step 3: Encode text
    text_states = self.bundle.encode_prompt(captions, ...)  # [B, seq_len, 512]
    
    # Step 4: Sample timesteps
    step_indices = torch.randint(
        0,
        len(self.bundle.scheduler.timesteps),
        (batch_size,),
        device=latents.device,
    )
    scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]  # [B]
    
    # Step 5: Add flow matching noise
    noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
    # noisy_latents: [B, C, T', J] = (1 - σ) * x₀ + σ * ε
    # targets: [B, C, T', J] = ε - x₀
    
    # Step 6: Apply conditioning (keep conditioned frames clean)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
    
    # Step 7: Create sequence timesteps for patches
    timesteps = self.bundle.create_sequence_ts(
        timesteps,
        condition_frame_mask_vae,
        self.bundle.transformer.config.patch_size,
    )  # Shape: [B, num_patches]
    
    # Step 8: Forward pass
    model_pred = self.bundle.transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=text_states,
        timestep=timesteps,
        hidden_states_mask=padding_mask if num_frames is not None else None,
        encoder_hidden_states_mask=None,
    ).float()  # [B, C, T', J]
    
    # Step 9: Compute MSE loss
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')  # [B, C, T', J]
    
    # Step 10: Apply masks
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask_exp = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask_exp
    
    # Step 11: Split translation and rotation losses
    mse_transl = mse[:, :, :, :1]  # [B, C, T', 1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
    
    mse_rot = mse[:, :, :, 1:]  # [B, C, T', 22]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)
    
    # Step 12: Combine losses
    w_t = self.translation_loss_weight  # default: 0.5
    loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
    
    return {
        'loss': loss,
        'loss_flow': loss.detach(),
        'loss_transl': loss_transl.detach(),
        'loss_rot': loss_rot.detach(),
    }
```

---

## 5. AUDIO-CONDITIONED VARIANT (MCM TRAINER)

### File: `prism_mcm_trainer.py` (lines 142-233)

Same flow matching logic as base trainer, but with additions:

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    # ... (steps 1-7 same as PrismTrainer) ...
    
    # ADDED: Get audio features
    audio_features = self._get_audio_features(batch)  # Optional[Tensor]
    # audio_features can come from:
    #   - Pre-computed: batch['audio_features']
    #   - Raw waveform: self.bundle.encode_audio(waveform)
    #   - None if no audio available
    
    # ADDED: Apply audio dropout for classifier-free guidance
    audio_features = self._apply_audio_dropout(
        audio_features, batch_size, latents.device,
    )
    # Randomly zero out audio features with probability audio_drop_rate
    
    # Different forward call: predict_with_control instead of transformer
    model_pred = self.bundle.predict_with_control(
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        text_states=text_states,
        audio_features=audio_features,  # NEW
        hidden_states_mask=padding_mask if num_frames is not None else None,
    ).float()
    
    # ADDED: Clamp for numerical stability
    model_pred = torch.nan_to_num(model_pred, nan=0.0, posinf=1e4, neginf=-1e4)
    model_pred = model_pred.clamp(-1e4, 1e4)
    
    # ... (rest same as base trainer) ...
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')
    # ... loss computation ...
    
    return {'loss': loss, 'loss_flow': loss.detach()}
```

---

## 6. KEY DIMENSIONS

```
motion:              [B, T_orig, J*6]         # Original motion frames
latents:             [B, C, T', J]            # Encoded latents (downsampled time)
noisy_latents:       [B, C, T', J]            # After adding noise
targets:             [B, C, T', J]            # Velocity targets: ε - x₀
timesteps:           [B]                      # Sampled timestep indices
sigmas:              [B, 1, 1, 1]             # Will broadcast to [B, C, T', J]
model_pred:          [B, C, T', J]            # Model prediction (should match targets)
mse:                 [B, C, T', J]            # Per-element MSE loss
loss:                scalar                   # Final loss value

Where:
  B = batch_size
  C = num_channels (latent dimension)
  T' = downsampled temporal dimension
  J = 23 (1 translation + 22 rotations in 6D)
```

---

## 7. SCHEDULER INTERFACE

```python
# Scheduler provides these attributes:
scheduler.timesteps  # Array: [t₀, t₁, ..., t_{N-1}]
scheduler.sigmas     # Array: [σ₀, σ₁, ..., σ_{N-1}]

# Example (100 total steps):
scheduler.timesteps = [0, 10, 20, 30, ..., 990]
scheduler.sigmas    = [0.999, 0.998, 0.997, ..., 0.001]

# During training:
# 1. Sample random index: idx ∈ [0, 100)
# 2. Get timestep: t = scheduler.timesteps[idx]
# 3. Get sigma: σ = scheduler.sigmas[idx]
# 4. Apply: x_t = (1-σ)x₀ + σε
```

---

## 8. LOSS COMPUTATION DETAIL

```python
# Raw MSE (per-element)
mse = F.mse_loss(model_pred, targets.float(), reduction='none')  # [B, C, T', J]

# Translation (1 joint)
mse_transl = mse[:, :, :, :1]  # [B, C, T', 1]
loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

# Rotation (22 joints)
mse_rot = mse[:, :, :, 1:]  # [B, C, T', 22]
loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

# Weighted combination (prevent translation dilution)
loss = 0.5 * loss_transl + 0.5 * loss_rot

# Instead of: loss = (1.0 / 23) * loss_transl + (22.0 / 23) * loss_rot
# Which would give: loss ≈ 0.043 * loss_transl + 0.957 * loss_rot
```

---

## 9. FRAME CONDITIONING MECHANICS

```python
# Example: 10% of batches condition on first frame(s)

# During training:
condition_frame_mask_vae:  [B, 1, T', J]  # Bool mask
# Shape for batch of 32:
# True  = frame should be noised (train_target)
# False = frame should be conditioned (keep original)

# Apply conditioning:
noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
# If mask[i] = False → use original latent[i]
# If mask[i] = True  → use noised noisy_latents[i]

# Typical pattern:
# [False, True, True, True, ...]  # First frame conditioned
# [False, False, True, True, ...]  # First 2 frames conditioned
# [True, True, True, True, ...]   # All frames to be noised (90% of batches)
```

---

## 10. INFERENCE VS TRAINING

**Training:**
- Timesteps sampled uniformly at random
- Single forward pass per batch
- Multiple noise levels in same batch
- Target is velocity: $v = \epsilon - x_0$

**Inference:**
- Timesteps scheduled linearly from high-noise to low-noise
- Multiple forward passes (one per step)
- Euler integration to iteratively denoise
- Velocity prediction guides the denoising trajectory

