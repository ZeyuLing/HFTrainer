# PRISM Inference Pipeline Code Analysis

## Executive Summary

The PRISM AR (Autoregressive) pipeline generates long motion sequences by:
1. Autoregressively generating multiple segments
2. Using **per-token timesteps** for noise-free condition frame injection
3. Applying **classifier-free guidance (CFG)** during the denoising loop
4. Following an **Euler ODE sampling** approach with FlowMatchEulerDiscreteScheduler

Key files analyzed:
- **prism_backend.py** (750 lines): Main inference pipeline
- **prism_mcm_pipeline.py** (518 lines): MCM audio-conditioned variant
- **transformer_prism.py** (200+ lines): Denoising transformer network

---

## 1. EULER ODE SAMPLING LOOP (Step-by-Step Denoising)

### Location
**File**: `prism_backend.py` lines **276-323**
**Method**: `generate_single_segment()` in `PrismARPipeline` class

### Denoising Loop Code Flow

```python
# Initialize scheduler with timesteps
self.scheduler.set_timesteps(num_inference_steps, device=device)  # Line 261
timesteps = self.scheduler.timesteps                               # Line 262

# Denoising loop over timesteps
for i, t in enumerate(timesteps):                                  # Line 279
    # --- Per-token timestep handling ---
    if self.config.expand_timesteps:                               # Line 283
        # Create per-token timesteps
        latent_model_input = (
            (1 - first_frame_mask) * condition + first_frame_mask * latents
        ).to(transformer_dtype)
        
        # KEY: Expand timestep per token (per position)
        temp_ts = (first_frame_mask[0][0] * t).flatten()          # Line 287
        timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)  # Line 288
    else:
        latent_model_input = latents.to(transformer_dtype)
        timestep = t.expand(latents.shape[0])
    
    # --- Model forward pass ---
    noise_pred = current_model(                                    # Line 293
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        attention_kwargs=attention_kwargs,
        is_causal=self.config.is_causal,
    )
    
    # --- Classifier-Free Guidance (CFG) ---
    if do_cfg:                                                     # Line 301
        noise_uncond = current_model(                              # Line 302
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=negative_prompt_embeds,
            attention_kwargs=attention_kwargs,
            is_causal=self.config.is_causal,
        )
        # CFG formula: noise = noise_uncond + scale * (noise_cond - noise_uncond)
        noise_pred = noise_uncond + current_guidance_scale * (
            noise_pred - noise_uncond
        )                                                          # Line 309-311
    
    # --- Scheduler step (Euler ODE update) ---
    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]  # Line 313
    
    # --- Force-restore condition frames ---
    # After each scheduler step, ensure condition frames remain noise-free
    if first_frame_latents is not None:                            # Line 317
        latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
                                                                    # Line 318

# Final merge (redundant but safe)
if self.config.expand_timesteps and first_frame_latents is not None:
    latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
                                                                    # Line 321-322
```

### ODE Sampling Algorithm Details

| Step | Component | Purpose |
|------|-----------|---------|
| 1 | `scheduler.set_timesteps()` | Generate noise schedule (typically 50 steps from t=1→0) |
| 2 | `expand_timesteps` check | Decide: uniform timesteps OR per-token timesteps |
| 3 | `latent_model_input` prep | Mix condition (first frame) with noisy latents via mask |
| 4 | Timestep expansion | Create `(B, T*J)` timestep tensor: condition frames get t=0 |
| 5 | Model forward (CFG) | Two forward passes: conditioned + unconditional |
| 6 | CFG scaling | Weighted combination of predictions |
| 7 | Scheduler step | Euler ODE integration: `x_{t-1} = x_t + v_t * (t-1 - t)` |
| 8 | Condition restore | Hard-restore condition frames to clean latents |

### Key Properties of the Euler Sampler

- **Integration method**: Euler (first-order ODE solver)
- **Scheduler**: `FlowMatchEulerDiscreteScheduler` from diffusers
- **Default steps**: 50 (configurable via `num_inference_steps`)
- **Noise schedule**: Flow matching (linear interpolation from noise to data)
- **Order**: `scheduler.order` = 1 (warmup steps computation at line 277)

---

## 2. PER-TOKEN TIMESTEPS DURING INFERENCE

### Location
**File**: `prism_backend.py` lines **283-291**
**File**: `prism_mcm_pipeline.py` lines **125-136**

### How Per-Token Timesteps are Created

```python
# In generate_single_segment() at line 283:
if self.config.expand_timesteps:                      # Line 283
    # Prepare latent input by MIXING condition and noisy latents
    latent_model_input = (
        (1 - first_frame_mask) * condition +           # Clean frames
        first_frame_mask * latents                      # Noisy frames
    ).to(transformer_dtype)                            # Line 285-286
    
    # CREATE PER-TOKEN TIMESTEPS
    # first_frame_mask[0][0] is (T*J,) boolean mask
    #   - 0 where condition frames are (first_frame_latents position)
    #   - 1 where generation frames are (rest of sequence)
    temp_ts = (first_frame_mask[0][0] * t).flatten()  # Line 287
    
    # Result:
    #   - Condition positions: temp_ts = 0 * t = 0
    #   - Generation positions: temp_ts = 1 * t = current_timestep
    
    # Expand to batch dimension
    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)  # Line 288
    # Shape: (batch_size, T*J)
```

### MCM Variant (Lines 125-136, prism_mcm_pipeline.py)

```python
for t in timesteps:                                     # Line 124
    if expand_timesteps and first_frame_latents is not None:
        # Replace condition frames with clean latents
        latent_model_input = (
            (1 - first_frame_mask) * condition + first_frame_mask * latents
        ).to(dtype)                                     # Line 128-129
        
        # Per-token timesteps: condition frames get t=0
        temp_ts = (first_frame_mask[0][0] * t).flatten()  # Line 132
        t_batch = temp_ts.unsqueeze(0).expand(batch_size, -1)  # Line 133
    else:
        latent_model_input = latents.to(dtype)
        t_batch = t.unsqueeze(0).expand(batch_size)    # Line 136
```

### Timestep Shape Transformation

```
Initial: t (scalar timestep, e.g., 0.8)
          ↓
first_frame_mask[0][0] * t
          ↓ (element-wise multiplication)
temp_ts: (T*J,) tensor where:
          [0, 0, 0, ..., t, t, t, ...]
                  ↑condition   ↑generation frames
          ↓
timestep: (batch_size, T*J) expanded tensor
```

### Key Insight: Timesteps are Freely Modifiable

**YES, per-token timestep values CAN be modified at inference time.**

The current implementation uses:
- `mask_value * timestep` for each position
- Where `mask_value = 0` → timestep becomes 0 (noise-free)
- Where `mask_value = 1` → timestep becomes `t` (current denoising step)

**You could modify this to:**
- Use different timesteps for different body parts
- Gradually vary timesteps across the sequence
- Use adaptive timesteps based on motion complexity
- Example: `temp_ts = (first_frame_mask[0][0] * t) * (1 - position_fade)`

---

## 3. CLASSIFIER-FREE GUIDANCE (CFG) MECHANISM

### Location
**File**: `prism_backend.py` lines **237, 249, 301-311**
**File**: `prism_mcm_pipeline.py` lines **121, 147-154**

### CFG During Inference

#### Step 1: Initialization (Lines 237, 249)
```python
device = next(self.transformer.parameters()).device
do_cfg = guidance_scale > 1.0  # Line 237

# Encode both positive and negative prompts
prompt_embeds, negative_prompt_embeds = self.encode_prompt(
    prompt=prompt,
    negative_prompt=negative_prompt,
    do_classifier_free_guidance=do_cfg,
    num_motion_per_prompt=1,
    max_sequence_length=max_sequence_length,
    device=device,
)  # Lines 246-253
```

#### Step 2: In Denoising Loop (Lines 301-311)
```python
if do_cfg:                                          # Line 301
    # Two forward passes:
    noise_pred = current_model(                     # Line 293-299
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        attention_kwargs=attention_kwargs,
        is_causal=self.config.is_causal,
    )
    
    noise_uncond = current_model(                   # Line 302-308
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=negative_prompt_embeds,
        attention_kwargs=attention_kwargs,
        is_causal=self.config.is_causal,
    )
    
    # CFG formula: noise = uncond + scale * (cond - uncond)
    noise_pred = noise_uncond + current_guidance_scale * (
        noise_pred - noise_uncond
    )                                               # Line 309-311
```

### CFG Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `guidance_scale` | 5.0 | >1.0 enables CFG | Higher = stronger text control, less diversity |
| `negative_prompt` | Empty string | Any text | What to avoid (e.g., "static motion") |
| `current_guidance_scale` | `guidance_scale` | Configurable per step | Can vary CFG strength across denoising |

### CFG Formulation

Standard classifier-free guidance:
```
ε_θ = ε_θ(x_t, c=∅) + w * [ε_θ(x_t, c=text) - ε_θ(x_t, c=∅)]
```

Where:
- `ε_θ(x_t, c=∅)` = unconditional prediction (`noise_uncond`)
- `ε_θ(x_t, c=text)` = conditional prediction (`noise_pred`)
- `w` = guidance scale (line 281: `current_guidance_scale = guidance_scale`)

### MCM Variant (CFG + Audio)

```python
# Audio is optional; text guidance is standard CFG
do_cfg = guidance_scale > 1.0 and negative_text_states is not None

if do_cfg:
    noise_uncond = bundle.predict_with_control(
        noisy_latents=latent_model_input,
        timesteps=t_batch,
        text_states=negative_text_states,
        audio_features=None,              # NO audio for unconditional
    )
    model_pred = noise_uncond + guidance_scale * (model_pred - noise_uncond)
                                           # Line 154, prism_mcm_pipeline.py
```

---

## 4. EXISTING GUIDANCE/STEERING MECHANISMS

### Current Built-in Mechanisms

#### 4.1 Classifier-Free Guidance (CFG)
- **Status**: ✅ Fully implemented
- **Location**: Lines 301-311 (prism_backend.py)
- **Modifiable**: YES - `guidance_scale` parameter can be adjusted per step
- **Use case**: Text prompt influence control

#### 4.2 Per-Token Timestep Control
- **Status**: ✅ Implemented via `expand_timesteps` flag
- **Location**: Lines 283-291 (prism_backend.py)
- **Modifiable**: YES - timestep values are computed as `mask * t`
- **Use case**: Selective denoising (condition frames vs. generation frames)

#### 4.3 Condition Frame Restoration
- **Status**: ✅ Implemented
- **Location**: Lines 317-318 (prism_backend.py)
- **How**: After each scheduler step, hard-restore condition frames to clean latents
- **Effect**: Ensures first frame remains noise-free during entire denoising process

#### 4.4 Attention Control
- **Status**: ✅ Parameters available
- **Location**: Line 297 `attention_kwargs` parameter
- **Use case**: Can pass attention masks or control attention behavior

### Mechanisms NOT Currently Available

| Feature | Status | Difficulty | Use Case |
|---------|--------|-----------|----------|
| Gradient-based guidance | ❌ Not implemented | Medium | Style/motion quality control |
| Intermediate latent penalties | ❌ Not implemented | Medium | Enforce smoothness/constraints |
| Adaptive timestep scheduling | ❌ Not implemented | Low | Variable denoising per region |
| Joint-specific guidance | ❌ Not implemented | Medium | Control specific body parts |
| Energy/momentum constraints | ❌ Not implemented | High | Physical realism enforcement |

---

## 5. CONDITION FRAMES INJECTION DURING STREAMING

### Location
**File**: `prism_backend.py` lines **240-274** (preparation)
**File**: `prism_backend.py` lines **315-322** (injection in loop)

### Preparation Phase

```python
# Load first frame condition from file or use previous segment
first_frame_motion = None
if first_frame_motion_path is not None:
    first_frame_motion = self.load_condition_pose(first_frame_motion_path)
                                                    # Line 404
    first_frame_motion = first_frame_motion[:, :1] # Line 406, ensure 1 frame

# Encode first frame to latent space
first_frame_latents = None
if first_frame_motion is not None:
    first_frame_latents = self.encode_motion(first_frame_motion)
                                                    # Line 243
    # Shape: (batch_size, C, 1, J)
```

### Latent Preparation (Lines 266-274)

```python
latents, condition, first_frame_mask = self.prepare_latents(
    batch_size=batch_size,
    num_channels_latents=num_channels_latents,
    num_joints=num_joints,
    num_frames=num_frames,
    dtype=transformer_dtype,
    device=device,
    first_frame_latents=first_frame_latents,  # Encoded condition
)
```

#### prepare_latents() Details (Lines 75-126)

```python
def prepare_latents(
    self,
    batch_size: int,
    num_channels_latents: int = 16,
    num_frames: int = 81,
    num_joints: int = 23,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    first_frame_latents: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        latents: Random noise tensor [B, C, T_latent, J]
        condition: Condition tensor with first frame [B, C, T_latent, J]
        first_frame_mask: Binary mask [B, C, T_latent, J]
            - 0 for condition (first frame)
            - 1 for positions to denoise
    """
    
    num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
    shape = (batch_size, num_channels_latents, num_latent_frames, num_joints)
    
    latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)
                                                    # Line 110
    
    # Create mask: 0=condition, 1=denoise
    condition = torch.zeros_like(latents)
    first_frame_mask = torch.ones_like(latents)
    
    if first_frame_latents is not None:
        # Expand batch if needed
        if first_frame_latents.shape[0] == 1 and batch_size > 1:
            first_frame_latents = first_frame_latents.expand(batch_size, -1, -1, -1)
        
        # Set condition for first frame
        condition[:, :, :1, :] = first_frame_latents
        first_frame_mask[:, :, :1, :] = 0.0      # Line 124
    
    return latents, condition, first_frame_mask
```

### Condition Injection in Denoising Loop (Lines 283-318)

```python
# At each denoising step:
for i, t in enumerate(timesteps):
    if self.config.expand_timesteps:
        # Mix condition with noisy latents using mask
        latent_model_input = (
            (1 - first_frame_mask) * condition +      # condition frames (clean)
            first_frame_mask * latents                 # generation frames (noisy)
        ).to(transformer_dtype)
                                                       # Lines 285-286
        
        # Create per-token timesteps
        temp_ts = (first_frame_mask[0][0] * t).flatten()
        # Condition frames: 0 * t = 0 (noise-free timestep)
        # Generation frames: 1 * t = t (current denoising timestep)
        
    # ...forward passes...
    
    # Scheduler step: updates latents
    latents = self.scheduler.step(noise_pred, t, latents, ...)[0]
    
    # FORCE RESTORE condition frames after each step
    if first_frame_latents is not None:
        latents = (
            (1 - first_frame_mask) * condition +
            first_frame_mask * latents
        )                                             # Lines 317-318
```

### Streaming Autoregressive Generation (Lines 412-441)

```python
# Store all motion segments
all_motion_segments = []

# Generate each segment
for seg_idx, prompt in enumerate(prompts):
    # Generate single segment with optional first-frame condition
    motion_vec = self.generate_single_segment(
        prompt=prompt,
        negative_prompt=negative_prompt,
        first_frame_motion=first_frame_motion,  # Condition for this segment
        num_frames=num_frames_this,
        ...
    )
    
    # Store segment
    if seg_idx == 0:
        all_motion_segments.append(motion_vec)
    else:
        # Skip overlapping frames to avoid duplication
        all_motion_segments.append(motion_vec[:, overlap_frames:])
    
    # Extract last frame as condition for NEXT segment
    first_frame_motion = self.extract_last_frame_motion(motion_vec)
                                                       # Line 440
    # This becomes first_frame_latents for next iteration

# Concatenate all segments
full_motion = torch.cat(all_motion_segments, dim=1)
```

### MCM Streaming (Lines 305-372, prism_mcm_pipeline.py)

```python
# Load first-frame condition if provided
first_frame_latents = None
if first_frame_motion_path is not None:
    first_frame_latents = self._load_first_frame(
        first_frame_motion_path, device, dtype
    )                                               # Line 308

# Generate segments autoregressively
for seg_idx in range(num_segments):
    # Generate single segment
    seg_latents = self.generate_single_segment(
        text_states=text_states,
        audio_feat=seg_audio_feat,
        negative_text_states=negative_text_states,
        first_frame_latents=first_frame_latents,  # Condition
        ...
    )                                               # Line 344-354
    
    # Decode this segment
    seg_decoded = self.decode_latents(seg_latents)
    
    # Extract last frame as condition for next segment
    last_frame_motion = seg_decoded[:, -1:, :, :]  # [1, 1, J, D]
    first_frame_latents = self.encode_motion_to_latent(last_frame_motion)
                                                     # Line 372
```

### Key Properties of Condition Injection

| Property | Details |
|----------|---------|
| **Condition Format** | First frame encoded to VAE latent space `[B, C, 1, J]` |
| **Mask Type** | Binary mask: 0=preserve condition, 1=denoise |
| **Injection Method** | Direct tensor blending via `(1-mask)*condition + mask*latents` |
| **Restoration** | Hard-restore after each scheduler step (lines 317-318) |
| **Continuity** | Last frame of segment becomes first frame of next segment |
| **Overlap** | Configurable overlap_frames (default=1) to smooth transitions |

---

## 6. VAE DECODE PATH

### Location
**File**: `prism_backend.py` lines **461-475**
**File**: `prism_mcm_pipeline.py` lines **172-188**

### VAE Decode Process

```python
def decode_motion(self, latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to motion.
    
    Args:
        latents: Latent tensor of shape [B, C, T_latent, J]
    
    Returns:
        Motion tensor of shape [B, T, J, C]
    """
    # Denormalize latents
    latents = latents * self.latents_std.to(latents.device) + self.latents_mean.to(latents.device)
                                                    # Line 470
    
    # Force float32 for VAE
    device_type = latents.device.type
    with torch.autocast(device_type, enabled=False):  # Disable autocast
        motion = self.vae.decode(latents.float())  # Line 474
    
    return motion  # [B, T, J, C]
```

### VAE Configuration (Lines 65-72)

```python
self.latents_mean = torch.tensor(
    vae.config.latents_mean, dtype=dtype, device=device
).view(1, self.vae.config.z_dim, 1, 1)
                                                    # Lines 65-67

self.latents_std = torch.tensor(
    vae.config.latents_std, dtype=dtype, device=device
).view(1, self.vae.config.z_dim, 1, 1)
                                                    # Lines 69-71

self.vae_scale_factor_temporal = vae.config.scale_factor_temporal
                                                    # Line 73
```

### VAE Model Details

**File**: `autoencoder_kl_2d.py` (2D causal VAE for temporal motion)

Key properties:
- **Input format**: `[B, T, J, C]` where T=time, J=joints, C=6D rotation
- **Latent format**: `[B, Z_dim, T_latent, J]` (compressed temporal)
- **Architecture**: Causal 1D convolutions + chunking
- **Scale factor**: Typically 4x temporal compression
- **Frozen decoder**: ✅ VAE is NOT frozen during inference (only during training)

### Frame Count Rounding (Lines 102-103)

```python
num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
```

This ensures:
- Input: 129 frames → Latent: 33 frames (with scale_factor=4)
- Output: 129 frames (upsampled back via decoder)

---

## MODIFICATION POINTS FOR GRADIENT-BASED GUIDANCE

Based on the code analysis, here are the key hooks for adding gradient-based guidance:

### Hook 1: In Denoising Loop (Lines 294-311)

```python
# After model forward pass, before scheduler.step()
noise_pred = current_model(...)  # Line 293-299

# ✅ ADD HERE: Gradient-based guidance computation
if use_gradient_guidance:  # NEW PARAMETER
    with torch.enable_grad():
        latents_opt = latents.detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latents_opt], lr=0.01)
        
        # Your guidance loss computation
        guidance_loss = compute_guidance_loss(latents_opt, ...)
        guidance_loss.backward()
        
        # Modify noise_pred based on gradients
        noise_pred = noise_pred + guidance_scale_grad * latents_opt.grad
```

### Hook 2: Per-Token Timestep Adaptation (Line 287)

```python
# Current: simple element-wise multiplication
temp_ts = (first_frame_mask[0][0] * t).flatten()

# ✅ COULD MODIFY: Adaptive timestep based on motion properties
adaptive_mask = compute_adaptive_timestep_mask(...)  # Your function
temp_ts = (adaptive_mask * t).flatten()
```

### Hook 3: Condition Restoration Strategy (Lines 317-318)

```python
# Current: hard-restore condition frames
if first_frame_latents is not None:
    latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

# ✅ COULD MODIFY: Soft restoration with gradual blending
if first_frame_latents is not None:
    blend_alpha = 1.0 - (i / len(timesteps))  # Decay restoration strength
    latents = (
        (1 - first_frame_mask) * (blend_alpha * condition + (1-blend_alpha) * latents) +
        first_frame_mask * latents
    )
```

---

## SUMMARY TABLE

| Aspect | Location | Modifiable | Current Implementation |
|--------|----------|-----------|----------------------|
| **Euler ODE Loop** | prism_backend.py:276-323 | ✅ YES | 50-step default, configurable |
| **Per-token Timesteps** | prism_backend.py:283-291 | ✅ YES | `(mask * t)`, fully modifiable |
| **CFG Scaling** | prism_backend.py:301-311 | ✅ YES | Per-step scale possible |
| **Condition Frames** | prism_backend.py:240-318 | ✅ YES | Hard-restore, could be soft |
| **Negative Prompt** | prism_backend.py:249-311 | ✅ YES | Empty default, configurable |
| **Gradient Guidance** | — | ❌ NO | Not implemented |
| **Joint-level Control** | — | ❌ NO | Not implemented |
| **VAE Decode** | prism_backend.py:461-475 | ✅ NO (frozen) | Fixed decoder, no gradient |

---

## KEY CODE SNIPPETS FOR MODIFICATION

### To Add Gradient-Based Guidance:
1. Add `requires_grad=True` to latents before scheduler.step()
2. Compute guidance loss (e.g., motion smoothness penalty)
3. Backprop through guidance loss
4. Modify noise_pred based on gradients

### To Modify Per-Token Timesteps:
1. Replace line 287: `temp_ts = (first_frame_mask[0][0] * t).flatten()`
2. with: `temp_ts = custom_mask(first_frame_mask[0][0], t, step_info)`

### To Soften Condition Restoration:
1. Add interpolation factor based on step index
2. Replace hard blend with: `α * condition + (1-α) * latents`

---

## INFERENCE PIPELINE CALL FLOW

```
__call__() [line 330]
  ├─ For each segment:
  │   ├─ generate_single_segment() [line 208]
  │   │   ├─ encode_prompt() [line 246]
  │   │   ├─ scheduler.set_timesteps() [line 261]
  │   │   ├─ prepare_latents() [line 266]
  │   │   ├─ Denoising Loop [line 279]:
  │   │   │   ├─ Per-token timesteps [line 287]
  │   │   │   ├─ Model forward (CFG) [line 293-311]
  │   │   │   ├─ scheduler.step() [line 313]
  │   │   │   └─ Condition restore [line 318]
  │   │   └─ decode_motion() [line 325]
  │   └─ extract_last_frame_motion() [line 440]
  ├─ Concatenate segments [line 446]
  └─ post_process_motion() [line 450]
```

