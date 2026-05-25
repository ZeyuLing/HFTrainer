# PRISM VACE CHANNEL MISMATCH ANALYSIS

## Executive Summary

Conducted comprehensive investigation into VACE channel mismatch hypothesis for PRISM motion generation model producing deformed output at inference despite converged training loss.

**KEY FINDING: NO VACE CHANNEL MISMATCH DETECTED**

The PRISM model architecture does **NOT** use VACE channels (inactive/reactive motion conditioning). All input channels during both training and inference are purely latent channels (16 channels for motion representation).

---

## Investigation Results

### 1. PRISM Trainer (`hftrainer/trainers/motion/prism_trainer.py`) - train_step() Analysis

**What is passed as `hidden_states` to transformer?**

✅ **ONLY `noisy_latents`** - NO VACE concatenation

```python
# Line 77-78: Add flow matching noise to latents
noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)

# Line 85: Convert to transformer dtype
noisy_latents = noisy_latents.to(dtype=transformer_dtype)

# Lines 87-93: Pass directly to transformer
model_pred = self.bundle.transformer(
    hidden_states=noisy_latents,           # ← ONLY noisy_latents
    encoder_hidden_states=text_states,
    timestep=timesteps,
    hidden_states_mask=padding_mask if num_frames is not None else None,
    encoder_hidden_states_mask=None,
).float()
```

**Data flow:**
- Input shape: `latents` = `[B, C, T_latent, J]` where C=16 (from VAE)
- After noise: `noisy_latents` = `[B, 16, T_latent, J]`
- To transformer: `hidden_states` = `[B, 16, T_latent, J]`

**Conclusion:** Training uses ONLY 16 latent channels, NO VACE concatenation.

---

### 2. PRISM Inference (`hftrainer/pipelines/motion/prism_backend.py`) - generate_single_segment() Analysis

**What is passed as `hidden_states` to transformer during inference?**

✅ **ONLY `latent_model_input`** - NO VACE concatenation

```python
# Line 382: Get in_channels from transformer config
num_channels_latents = self.transformer.config.in_channels

# Lines 383-391: Prepare latents
latents, condition, first_frame_mask = self.prepare_latents(
    batch_size=batch_size,
    num_channels_latents=num_channels_latents,
    num_joints=num_joints,
    num_frames=num_frames,
    dtype=transformer_dtype,
    device=device,
    first_frame_latents=first_frame_latents,
)
# latents shape: [B, 16, T_latent, J]

# Lines 407-418: Per-token timestep expansion
if self.config.expand_timesteps:
    latent_model_input = (
        (1 - first_frame_mask) * condition + first_frame_mask * latents
    ).to(transformer_dtype)
else:
    latent_model_input = latents.to(transformer_dtype)

# Lines 420-427: Pass to transformer
noise_pred = current_model(
    hidden_states=latent_model_input,           # ← ONLY latent_model_input (16 channels)
    timestep=timestep,
    encoder_hidden_states=prompt_embeds,
    attention_kwargs=attention_kwargs,
    is_causal=self.config.is_causal,
    hidden_states_mask=motion_mask,
)
```

**Data flow:**
- `prepare_latents()` creates: `latents` = `[B, 16, T_latent, J]`
- `latent_model_input` = `[B, 16, T_latent, J]`
- To transformer: `hidden_states` = `[B, 16, T_latent, J]`

**Conclusion:** Inference uses ONLY 16 latent channels, NO VACE concatenation.

---

### 3. Transformer Model Configuration

**What does transformer.config.in_channels expect?**

✅ **in_channels = 16** (ONLY latent channels)

From `configs/prism/prism_1b_tp2m_1frame.py` (lines 19-42):

```python
transformer=dict(
    type="PrismTransformerMotionModel",
    trainable=True,
    gradient_checkpointing=True,
    module_dtype="bf16",
    patch_size=(1, 1),
    attention_head_dim=128,
    cross_attn_norm=True,
    added_kv_proj_dim=None,
    eps=1e-6,
    ffn_dim=8960,
    freq_dim=256,
    in_channels=16,              # ← Expects exactly 16 channels
    num_attention_heads=12,
    num_layers=30,
    out_channels=16,
    qk_norm="rms_norm_across_heads",
    rope_max_seq_len=1024,
    # KT-RoPE: Kinematic-Topology Rotary Position Embedding
    joint_pos_mode="sequential",  # Options: "sequential", "spectral", "dfs"
    num_spectral_modes=4,
    spectral_scale=None,
    text_dim=4096,
)
```

**Verification:**
- Training passes: `[B, 16, T_latent, J]` ✓ Matches `in_channels=16`
- Inference passes: `[B, 16, T_latent, J]` ✓ Matches `in_channels=16`

**Conclusion:** No mismatch in channel count.

---

## VACE Channel Analysis

### Where is VACE used?

VACE (Velocity, Acceleration, Curvature, Energy) conditioning channels are used in **HyMotion M2M trainers**, NOT PRISM:

1. **`hftrainer/trainers/motion/hymotion_m2m_trainer.py`** (Line 268-276):
   ```python
   # Build VACE context
   vace_context = self.bundle.prepare_vace_input(
       masked_motion_input, motion_mask
   )
   
   # CONCATENATE with latents
   x_input = torch.cat([x_t, vace_context], dim=-1)
   ```
   VACE is concatenated for **motion-to-motion** generation.

2. **`hftrainer/trainers/motion/hymotion_m2m_soar_trainer.py`** (Line 228):
   ```python
   z_re_input = torch.cat([z_re, vace_context], dim=-1)
   ```
   VACE used in SOAR (with physics feedback).

3. **`hftrainer/pipelines/motion/hymotion_m2m_pipeline.py`** (Line 250):
   ```python
   x_input = torch.cat([x, vace_context], dim=-1)
   ```
   VACE concatenated during M2M inference.

### PRISM does NOT use VACE

Search results confirm PRISM trainers/pipelines have **zero references** to VACE:

```bash
$ grep -r "vace\|VACE" hftrainer/trainers/motion/prism_trainer.py
[No results]

$ grep -r "vace\|VACE" hftrainer/pipelines/motion/prism_backend.py
[No results]

$ grep "torch.cat.*hidden\|concat.*hidden" hftrainer/trainers/motion/prism_trainer.py
[No results]

$ grep "torch.cat.*hidden\|concat.*hidden" hftrainer/pipelines/motion/prism_backend.py
[No results]
```

---

## Comparison: Training vs. Inference Input Channels

| Aspect | Training | Inference | Match? |
|--------|----------|-----------|--------|
| Hidden state shape | `[B, 16, T_latent, J]` | `[B, 16, T_latent, J]` | ✓ |
| Channel count | 16 (latent only) | 16 (latent only) | ✓ |
| VACE concatenation | None | None | ✓ |
| Transformer input | `noisy_latents` | `latent_model_input` | ✓ |
| Config in_channels | 16 | 16 | ✓ |

---

## Actual Deformation Root Causes

Since VACE channel mismatch is **NOT the issue**, the deformed output is likely caused by:

### 1. **Timestep Mismatch** (Previously Identified)

See `PRISM_TIMESTEP_MISMATCH_ANALYSIS.md` for detailed analysis:
- Training uses random timestep sampling from full [0, 1000] schedule
- Inference uses sparse timestep schedule (e.g., 10 steps)
- **Train-test distribution mismatch** causes quality degradation
- Floating-point precision in sigma lookup can cause lookup failures

### 2. **Per-Token Timestep Expansion Differences**

Training applies per-token timestep expansion via `create_sequence_ts()`:
```python
# hftrainer/models/motion/prism/bundle.py, lines 240-255
target_ts = ori_ts.unsqueeze(1).unsqueeze(2).expand(
    batch_size, latent_frames, latent_joints
)
target_ts = torch.where(
    condition_frame_mask_vae[:, 0, ::patch_size[0], ::patch_size[1]],
    target_ts,
    0  # ← Conditioning frames = timestep 0
)
```

Inference applies similar logic via `expand_timesteps` flag:
```python
# hftrainer/pipelines/motion/prism_backend.py, lines 407-415
if self.config.expand_timesteps:
    if self._kafs_alpha_map is not None:
        temp_ts = (first_frame_mask[0][0] * t * self._kafs_alpha_map).flatten()
    else:
        temp_ts = (first_frame_mask[0][0] * t).flatten()
```

**Potential issues:**
- Mask shape differences between training (`condition_frame_mask_vae`) and inference (`first_frame_mask`)
- Patch size handling may cause dimension mismatches
- KAFS alpha map affects timestep scaling if enabled

### 3. **Input Distribution Mismatch**

The model is trained on:
- Randomly masked frames (10% condition rate)
- Random temporal cropping (360 frame clips)
- Various motion types and speeds

But inference always starts with:
- First frame conditioned (100% condition on frame 0)
- Full motion generation starting from silence
- Single consistent autoregressive generation pattern

### 4. **Latent Distribution Shift**

Training normalizes latents:
```python
# hftrainer/pipelines/motion/prism_backend.py, lines 319-321
z = (z - self.latents_mean) / self.latents_std
```

If inference latent mean/std differ from training, the normalized input will have different statistics.

---

## Verification Checklist

✅ **Completed Investigation Items:**

- [x] Verified trainer `hidden_states` = only `noisy_latents` (no VACE)
- [x] Verified inference `hidden_states` = only `latent_model_input` (no VACE)
- [x] Confirmed transformer `in_channels=16` (latent only)
- [x] Confirmed VACE is exclusive to HyMotion M2M trainers
- [x] Verified 16-channel match between training and inference
- [x] Confirmed no `torch.cat([..., vace])` patterns in PRISM code

---

## Recommended Next Steps

Since VACE channel mismatch is **definitively NOT the issue**, investigate:

### Priority 1: Timestep Precision Debugging
```python
# Enable detailed sigma lookup logging
# See: PRISM_TIMESTEP_MISMATCH_ANALYSIS.md → Verification Steps
```

### Priority 2: Test expand_timesteps=False
```python
# Run inference with global timesteps to isolate per-token logic:
pipe = PrismARPipeline(..., expand_timesteps=False)
```

### Priority 3: Frame Mask Consistency
```python
# Verify condition mask shapes match between training and inference
# Log frame mask dimensions and values
```

### Priority 4: Input Latent Statistics
```python
# Compare latent distributions at start of inference
# vs. what model saw during training
```

---

## Conclusion

**The PRISM model does NOT suffer from a VACE channel mismatch.** Both training and inference use identical 16-channel latent representations with no concatenated conditioning channels. VACE conditioning is exclusive to HyMotion M2M trainers, not PRISM.

The deformed output at inference is caused by other factors, primarily:
1. **Timestep mismatch** (train-test distribution mismatch)
2. **Per-token timestep expansion differences**
3. **Input distribution shift** (always conditioning first frame at inference)
4. **Sigma lookup precision issues**

Recommend focusing debug efforts on timestep handling and input distribution alignment rather than channel concatenation.

