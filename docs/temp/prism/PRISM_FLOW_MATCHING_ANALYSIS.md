# PRISM Flow Matching Training: Complete Analysis

## Repository Location
- **Trainer Code**: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/trainers/motion/`
- **Bundle Code**: `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/prism/bundle.py`

## Files Analyzed
1. `prism_trainer.py` - Main text-to-motion trainer
2. `prism_mcm_trainer.py` - Audio-conditioned trainer
3. `bundle.py` - Noising and encoding logic

---

## 1. HOW NOISE IS ADDED (NOISING EQUATION)

### Location: `bundle.py`, lines 257-262, method `add_flow_noise()`

```python
def add_flow_noise(self, latents: torch.Tensor, timesteps: torch.Tensor):
    noise = torch.randn_like(latents)
    sigmas = _get_sigmas(self.scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    noisy_latents = (1 - sigmas) * latents + sigmas * noise
    targets = noise - latents
    return noisy_latents, targets
```

### The Noising Equation:
$$x_t = (1 - \sigma_t) \cdot x_0 + \sigma_t \cdot \epsilon$$

Where:
- **$x_t$** = `noisy_latents` (the corrupted latent at timestep t)
- **$x_0$** = `latents` (the original clean latent)
- **$\epsilon$** = `noise` (random Gaussian noise, sampled via `torch.randn_like(latents)`)
- **$\sigma_t$** = `sigmas` (sigma schedule value at timestep t)

### Sigma Retrieval: `_get_sigmas()` (lines 19-27)

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

**Key points:**
- Sigmas come from the scheduler (type: `FlowMatchEulerDiscreteScheduler`)
- Sigma values are indexed by matching the sampled timestep to the scheduler's timestep schedule
- Sigmas are reshaped to match the latent's dimensionality (4D for motion latents: `[B, C, T, J]`)

---

## 2. TRAINING TARGET (WHAT THE MODEL PREDICTS)

### Location: `bundle.py`, line 261 and `prism_trainer.py`, line 95

```python
# In bundle.py:
targets = noise - latents

# In prism_trainer.py:
mse = F.mse_loss(model_pred, targets.float(), reduction='none')
```

### The Training Target is: **VELOCITY**

$$v_t = \epsilon - x_0$$

Where:
- **$v_t$** = `targets` (the training target, also called "velocity")
- **$\epsilon$** = the random Gaussian noise
- **$x_0$** = the original clean latent

### Why "Velocity"?
In flow matching, the model predicts the *direction* to move from the noisy sample towards the clean sample. The velocity is the normalized direction of the flow: $v = \epsilon - x_0 = (\epsilon - x_0)$.

This can be rewritten as:
$$v_t = \epsilon - x_0 = (x_t - (1-\sigma_t)x_0) / \sigma_t - x_0 = (x_t - x_0) / \sigma_t$$

The model learns to predict this velocity field, which guides the denoising process during inference.

---

## 3. TIMESTEP/SIGMA SAMPLING DURING TRAINING

### Location: `prism_trainer.py`, lines 68-75

```python
# Randomly sample timestep indices uniformly from the scheduler's timestep schedule
step_indices = torch.randint(
    0,
    len(self.bundle.scheduler.timesteps),
    (batch_size,),
    device=latents.device,
)
# Retrieve the actual timesteps from the scheduler
scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
timesteps = scheduler_timesteps[step_indices]
```

### Timestep Sampling Strategy:
1. **Uniform Random Sampling**: Each training batch samples timestep indices uniformly at random from `[0, num_timesteps)`
2. **Per-Sample Variation**: Each sample in the batch can have a different timestep (allows sampling at different noise levels)
3. **Scheduler-Driven**: The actual timestep values depend on the scheduler's configuration (e.g., `FlowMatchEulerDiscreteScheduler`)
4. **Sigma Schedule**: Once timesteps are selected, sigmas are looked up via `_get_sigmas()`

### Batch-wise vs Sample-wise:
- **Shape of step_indices**: `[batch_size]` — each sample in the batch gets one sampled timestep
- **Different timesteps**: Different samples in the same batch can be trained at different noise levels
- **Efficient training**: This allows the model to learn denoising at all noise levels with a single forward pass

---

## 4. COMPLETE TRAINING STEP (Text-to-Motion)

### Location: `prism_trainer.py`, lines 41-118

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Encode motion to latents
    motion = batch['motion']
    latents = self.bundle.encode_motion(motion)  # [B, C, T, J]
    
    # 2. Create masks (padding + conditioning)
    padding_mask = self.bundle.create_padding_mask(...)
    condition_frame_mask_vae = self.bundle.create_condition_mask(...)
    
    # 3. Encode text prompt
    text_states = self.bundle.encode_prompt(captions, ...)
    
    # 4. Sample timesteps uniformly
    step_indices = torch.randint(0, len(self.bundle.scheduler.timesteps), (batch_size,))
    scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]
    
    # 5. Add flow matching noise
    noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
    #   noisy_latents = (1 - σ) * x₀ + σ * ε
    #   targets = ε - x₀  (velocity)
    
    # 6. Zero out conditioned frames (keep them as clean latents)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
    
    # 7. Create sequence timesteps for patched transformer
    timesteps = self.bundle.create_sequence_ts(
        timesteps, condition_frame_mask_vae, 
        self.bundle.transformer.config.patch_size
    )
    
    # 8. Forward pass through transformer
    model_pred = self.bundle.transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=text_states,
        timestep=timesteps,
        hidden_states_mask=padding_mask,
    )
    
    # 9. Compute MSE loss
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')
    
    # 10. Apply masks (don't penalize conditioned/padded regions)
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask
    
    # 11. Separate translation and rotation losses (translation dilution fix)
    mse_transl = mse[:, :, :, :1]  # First joint = translation (x, y, z)
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
    
    mse_rot = mse[:, :, :, 1:]     # Other joints = rotations (6D representation)
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)
    
    # 12. Weighted combination
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

## 5. KEY MATHEMATICAL SUMMARY

| Component | Equation | Variable Name |
|-----------|----------|----------------|
| **Noising Equation** | $x_t = (1-\sigma_t)x_0 + \sigma_t\epsilon$ | `noisy_latents` |
| **Training Target** | $v_t = \epsilon - x_0$ | `targets` (velocity) |
| **Loss Function** | $L = \text{MSE}(f_\theta(x_t, t), v_t)$ | `mse` |
| **Timestep Sampling** | $t \sim \text{Uniform}(0, T)$ | `step_indices` |
| **Sigma Lookup** | $\sigma_t = \sigma[\text{scheduler.timesteps}[t]]$ | `sigmas` |

---

## 6. SPECIAL FEATURES IN PRISM

### Translation vs Rotation Loss Weighting
The model predicts **23 joint values**:
- **Index 0**: Translation (x, y, z) — 1 joint
- **Indices 1-22**: Rotation (6D representation) — 22 joints

The trainer splits the loss to prevent translation being diluted:
- Translation loss weight: `w_t = 0.5` (default)
- Rotation loss weight: `1.0 - w_t = 0.5`

This prevents the 22 rotation joints (which dominate in count) from overshadowing the 1 translation joint.

### Frame Conditioning
- **Conditioned frames**: First `condition_num_frames` frames are kept as clean latents (not noised)
- **Activation**: Controlled by `frame_condition_rate` (default: 0.1 = 10% of batches)
- **Purpose**: Helps the model learn guided generation (starts from real motion frames)

### Audio-Conditioned Variant (MCM Trainer)
Same flow matching logic, but adds:
- Audio encoding: `self.bundle.encode_audio(waveform)`
- Audio dropout: Random zeroing for classifier-free guidance (default: `audio_drop_rate = 0.1`)
- Control transformer: Separate transformer branch for audio conditioning

---

## 7. SCHEDULER DEPENDENCY

The scheduler (`FlowMatchEulerDiscreteScheduler`) provides:
- `scheduler.timesteps`: Array of timestep indices (e.g., `[0, 10, 20, ..., 999]`)
- `scheduler.sigmas`: Array of sigma values corresponding to each timestep

During training, a random index is sampled, then:
1. Get the timestep: `t = scheduler.timesteps[random_idx]`
2. Get the sigma: `σ = scheduler.sigmas[random_idx]`
3. Apply noising: `x_t = (1-σ)x₀ + σε`
4. Compute velocity: `v = ε - x₀`
5. Predict velocity: `pred_v = model(x_t, t)`
6. Compute loss: `loss = MSE(pred_v, v)`

---

## 8. SUMMARY TABLE

| Aspect | Detail |
|--------|--------|
| **Noising Equation** | $(1-\sigma_t) x_0 + \sigma_t \epsilon$ |
| **Training Target** | Velocity: $\epsilon - x_0$ |
| **Timestep Sampling** | Uniform random from scheduler timesteps |
| **Sigma Schedule** | Retrieved from `FlowMatchEulerDiscreteScheduler` |
| **Loss** | MSE between predicted velocity and actual velocity |
| **Special Feature** | Separate weighting for translation (1 joint) vs rotation (22 joints) |
| **Frame Conditioning** | 10% of batches condition on first N frames (default: 1) |
| **Variant** | MCM trainer adds audio conditioning via control transformer |

