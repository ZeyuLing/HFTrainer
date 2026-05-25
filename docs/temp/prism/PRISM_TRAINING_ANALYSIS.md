# PRISM Transformer Training Call Analysis

## Executive Summary

This document provides the exact code for how the PRISM transformer is called during training, with detailed analysis of timestep handling, masking, and causality.

---

## 1. TRAINING FORWARD PASS (PrismTrainer.train_step)

**File:** `/hftrainer/trainers/motion/prism_trainer.py` (lines 41-118)

### Complete Training Flow:

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    motion = batch['motion']
    captions = batch['caption']
    num_frames = batch.get('num_frames')

    # Step 1: Encode motion to latent space
    latents = self.bundle.encode_motion(motion)
    batch_size, _, latent_frames, latent_joints = latents.shape

    # Step 2: Create padding mask for variable-length sequences
    padding_mask = self.bundle.create_padding_mask(
        num_frames=num_frames,
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=latents.device,
    )
    
    # Step 3: Encode text prompts
    text_states = self.bundle.encode_prompt(
        captions,
        max_sequence_length=self.max_text_length,
        prompt_drop_rate=self.prompt_drop_rate,
        dtype=next(self.bundle.transformer.parameters()).dtype,
    )
    
    # Step 4: Create condition mask (frames to keep vs. denoise)
    condition_frame_mask_vae = self.bundle.create_condition_mask(
        latents,
        frame_condition_rate=self.frame_condition_rate,
        condition_num_frames=self.condition_num_frames,
    )

    # Step 5: Sample random timesteps per sample in batch
    step_indices = torch.randint(
        0,
        len(self.bundle.scheduler.timesteps),
        (batch_size,),
        device=latents.device,
    )
    scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]  # Shape: [B]

    # Step 6: Add noise according to flow matching
    noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
    
    # ===== CRITICAL: TIMESTEP EXPANSION =====
    # Convert scalar timesteps [B] to per-token timesteps [B, N]
    # where N = (T//p_t) * (J//p_j) is the flattened token sequence length
    timesteps = self.bundle.create_sequence_ts(
        timesteps,
        condition_frame_mask_vae,
        self.bundle.transformer.config.patch_size,
    )
    
    transformer_dtype = next(self.bundle.transformer.parameters()).dtype
    noisy_latents = noisy_latents.to(dtype=transformer_dtype)

    # ===== TRANSFORMER FORWARD PASS =====
    model_pred = self.bundle.transformer(
        hidden_states=noisy_latents,                              # [B, C, T, J]
        encoder_hidden_states=text_states,                        # [B, N_ctx, text_dim]
        timestep=timesteps,                                       # [B, N] EXPANDED!
        hidden_states_mask=padding_mask if num_frames is not None else None,  # [B, T, J] or None
        encoder_hidden_states_mask=None,                          # NOT PASSED
    ).float()

    # Step 7: Compute losses
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')
    # mse shape: [B, C, T', J] where J=23 (token 0=translation, 1-22=rotation)
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask

    # Separate translation (J=0) and rotation (J=1:) to prevent dilution
    mse_transl = mse[:, :, :, :1]           # [B, C, T', 1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

    mse_rot = mse[:, :, :, 1:]              # [B, C, T', 22]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

    w_t = self.translation_loss_weight
    loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
    return {
        'loss': loss,
        'loss_flow': loss.detach(),
        'loss_transl': loss_transl.detach(),
        'loss_rot': loss_rot.detach(),
    }
```

---

## 2. TIMESTEP EXPANSION: create_sequence_ts

**File:** `/hftrainer/models/motion/prism/bundle.py` (lines 240-255)

### Exact Implementation:

```python
def create_sequence_ts(
    self,
    ori_ts: torch.Tensor,                          # Input: [B] scalar timesteps
    condition_frame_mask_vae: torch.Tensor,        # [B, 1, T, J] boolean mask
    patch_size=(1, 1),
) -> torch.Tensor:
    """
    Expands scalar timesteps to per-token timesteps.
    
    For standard diffusion (no conditioning), returns timestep repeated for each token.
    For frame conditioning, uses the condition mask to set conditioned frames to 0.
    """
    batch_size, _, latent_frames, latent_joints = condition_frame_mask_vae.shape
    post_patch_num_frames = latent_frames // patch_size[0]
    post_patch_num_joints = latent_joints // patch_size[1]
    
    # Expand [B] -> [B, 1, 1, 1] -> [B, T', J']
    # ori_ts: [B] 
    # unsqueeze(1).unsqueeze(2): [B] -> [B, 1, 1]
    # expand(...): broadcast to [B, post_patch_num_frames, post_patch_num_joints]
    target_ts = ori_ts.unsqueeze(1).unsqueeze(2).expand(
        batch_size, 
        post_patch_num_frames, 
        post_patch_num_joints
    )
    
    # Apply condition mask:
    # - Where condition_frame_mask_vae is True (frame to denoise): keep the timestep value
    # - Where condition_frame_mask_vae is False (frame is conditioned): set to 0
    target_ts = torch.where(
        condition_frame_mask_vae[:, 0, :: patch_size[0], :: patch_size[1]],
        target_ts,
        0,
    )
    
    # Flatten to sequence: [B, T', J'] -> [B, N] where N = T' * J'
    return target_ts.flatten(1)
```

### Output Shape Transformation:
```
Input ori_ts:        [B]                    # e.g., [2] with values [412, 789]
↓
unsqueeze(1):        [B, 1]                 # [2, 1]
↓
unsqueeze(2):        [B, 1, 1]              # [2, 1, 1]
↓
expand(...):         [B, T', J']            # [2, 8, 24] if T'=8, J'=24
↓
apply mask:          [B, T', J']            # Some positions set to 0
↓
flatten(1):          [B, N]                 # [2, 192] where N = 8*24 = 192

Output timestep:     [B, N]                 # Per-token timesteps!
```

---

## 3. TRANSFORMER FORWARD PASS: PrismTransformerMotionModel

**File:** `/hftrainer/models/motion/prism/network/transformer_prism.py` (lines 232-512)

### Exact Forward Signature and Key Timestep Handling:

```python
def forward(
    self,
    hidden_states: torch.Tensor,                                    # [B, C, T, J]
    timestep: torch.LongTensor,                                    # [B, N] EXPANDED!
    encoder_hidden_states: torch.Tensor,                           # [B, N_ctx, text_dim]
    hidden_states_mask: Optional[torch.Tensor] = None,            # [B, T, J] or None
    encoder_hidden_states_mask: Optional[torch.Tensor] = None,    # None (not used)
    attention_kwargs: Optional[Dict[str, Any]] = None,
    is_causal: bool = False,                                       # NOT USED in training
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Args:
        hidden_states: Motion latent features from VAE encoder.
            Shape: [B, C, T, J] where B=batch, C=channels, T=frames, J=joints.
        
        timestep: Diffusion timesteps. CRITICAL DIFFERENCE:
            Shape: [B] for standard diffusion (scalar per sample)
            Shape: [B, N] for Wan 2.2 TI2V mode (per-token timesteps)
            
            In PRISM training: [B, N] — per-token timesteps after expansion!
            
        hidden_states_mask: Attention mask for motion tokens.
            Shape: [B, T, J]. Values: 1 = visible/valid, 0 = masked/padding.
            
        encoder_hidden_states_mask: Attention mask for text tokens.
            NOT PASSED during PRISM training (None).
            
        is_causal: Boolean flag for causal attention masking.
            NOT USED during training (False).
    """
    
    # ===== CRITICAL CODE: Timestep Detection =====
    # Lines 407-426: Handle both scalar and per-token timesteps
    
    # Handle per-token timesteps for Wan 2.2 TI2V mode
    if timestep.ndim == 2:  # [B, N] shape
        ts_seq_len = timestep.shape[1]  # Extract sequence length N
        timestep = timestep.flatten()   # [B, N] -> [B*N] for embedding
    else:
        ts_seq_len = None  # Standard mode

    # Get timestep embedding (temb), timestep projection, and processed text embeddings
    temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
        timestep,
        encoder_hidden_states,
        timestep_seq_len=ts_seq_len,  # Pass N here!
    )

    # Reshape timestep projection for block modulation
    if ts_seq_len is not None:
        # Wan 2.2 TI2V: per-token modulation 
        # [B, N, 6*inner_dim] -> [B, N, 6, inner_dim]
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        # Standard: global modulation 
        # [B, 6*inner_dim] -> [B, 6, inner_dim]
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
    
    # ===== MASK HANDLING =====
    # Lines 323-374: Patchify and convert masks to attention bias format
    
    if hidden_states_mask is not None:
        # Patchify: [B, T, J] -> [B, T', J'] after min-pooling
        # Convert to attention bias: 1 -> 0.0, 0 -> -inf
        # Final shape: [B, 1, 1, N] for broadcasting in attention
        # If ANY position in a patch is masked, entire patch is masked
        ...
    
    if encoder_hidden_states_mask is not None:
        # NOT USED in training
        ...
    
    # ===== CAUSAL MASKING =====
    # Lines 387-401: NOT USED during training (is_causal=False)
    causal_mask = None
    if is_causal:
        # Only enables if explicitly passed is_causal=True
        ...
    
    # ===== TRANSFORMER BLOCKS FORWARD =====
    # Lines 431-454: Process through transformer blocks with masks
    
    for block in self.blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = torch.utils.checkpoint.checkpoint(
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,           # Per-token or global
                rotary_emb,
                hidden_states_mask,      # Motion mask [B, 1, 1, N]
                encoder_hidden_states_mask,  # None
                causal_mask,             # None
                use_reentrant=False,
            )
        else:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,      # Per-token or global
                rotary_emb=rotary_emb,
                hidden_states_mask=hidden_states_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                causal_mask=causal_mask,
            )
```

---

## 4. EMBEDDING STAGE: Timestep Processing

**File:** `/hftrainer/models/motion/prism/network/embedding.py` (lines 85-133)

### How Timesteps Are Embedded:

```python
class WanTimeTextEmbedding(nn.Module):
    def forward(
        self,
        timestep: torch.Tensor,                        # Flattened: [B*N]
        encoder_hidden_states: torch.Tensor,          # [B, N_ctx, text_dim]
        timestep_seq_len: Optional[int] = None,       # N value
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            - temb: Time embedding
                If timestep_seq_len is None: [B, inner_dim] (scalar mode)
                If timestep_seq_len is not None: [B, N, inner_dim] (per-token mode)
            
            - timestep_proj: Projected timestep for transformer blocks
                If timestep_seq_len is None: [B, 6*inner_dim]
                If timestep_seq_len is not None: [B, N, 6*inner_dim]
            
            - encoder_hidden_states: Projected text embeddings [B, N_ctx, inner_dim]
        """
        
        # Step 1: Apply sinusoidal projection to timesteps
        timestep = self.timesteps_proj(timestep)
        
        # Step 2: Optionally reshape for sequence-level timesteps
        if timestep_seq_len is not None:
            # [B*N] -> [B, N, ...] using unflatten
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))
        
        # Step 3: Handle dtype compatibility
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        
        # Step 4: Compute time embedding through MLP
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        # Output: [B, inner_dim] or [B, N, inner_dim]
        
        # Step 5: Project time embedding with activation
        timestep_proj = self.time_proj(self.act_fn(temb))
        # Output: [B, time_proj_dim] or [B, N, time_proj_dim]
        # where time_proj_dim = 6*inner_dim
        
        # Step 6: Project text embeddings
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        
        return temb, timestep_proj, encoder_hidden_states
```

---

## 5. TRAINING CONFIGURATION SUMMARY

### Parameters Passed to Transformer During Training:

| Parameter | Value | Shape | Usage |
|-----------|-------|-------|-------|
| `hidden_states` | Noisy latents | [B, C, T, J] | Motion input after VAE encode |
| `timestep` | EXPANDED | **[B, N]** | Per-token timesteps (NOT scalar!) |
| `encoder_hidden_states` | Text embeddings | [B, N_ctx, 4096] | Text conditioning |
| `hidden_states_mask` | Padding mask | [B, T, J] or None | Variable-length motion sequences |
| `encoder_hidden_states_mask` | Text mask | None | NOT PASSED |
| `is_causal` | False | - | NOT USED in training |
| `attention_kwargs` | None | - | LoRA scaling (if applicable) |

### Timestep Format Differences:

```python
# BEFORE create_sequence_ts:
timesteps = scheduler_timesteps[step_indices]  # [B] scalar per-sample
# e.g., [412, 789] for batch_size=2

# AFTER create_sequence_ts:
timesteps = self.bundle.create_sequence_ts(...)  # [B, N] per-token
# e.g., [[412, 412, ..., 412],                  # 192 tokens all 412
#        [789, 789, ..., 789]]                  # 192 tokens all 789
# (if 192 = 8 frames * 24 joints per frame)
```

---

## 6. MCM TRAINER VARIANT

**File:** `/hftrainer/trainers/motion/prism_mcm_trainer.py` (lines 142-233)

The MCM trainer follows the **same pattern** as PrismTrainer, with identical transformer call:

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    # ... (same setup as PrismTrainer) ...
    
    noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
    
    timesteps = self.bundle.create_sequence_ts(
        timesteps,
        condition_frame_mask_vae,
        self.bundle.transformer.config.patch_size,  # SAME!
    )
    
    ctrl_dtype = next(self.bundle.control_transformer.parameters()).dtype
    noisy_latents = noisy_latents.to(dtype=ctrl_dtype)
    
    # Forward with control (MCM-specific)
    model_pred = self.bundle.predict_with_control(
        noisy_latents=noisy_latents,
        timesteps=timesteps,                    # [B, N] EXPANDED!
        text_states=text_states,
        audio_features=audio_features,
        hidden_states_mask=padding_mask if num_frames is not None else None,
        encoder_hidden_states_mask=None,
    ).float()
```

**Key observation:** MCM trainer also passes `timesteps` as `[B, N]` per-token format.

---

## 7. ACTUAL TEST CASES FROM CODE

**File:** `/hftrainer/models/motion/prism/network/transformer_prism.py` (lines 553-664)

### Test 1: Basic forward (no mask):
```python
with torch.no_grad():
    hidden_states = torch.randn(batch_size, num_channels, num_frames, num_joints)
    timestep = torch.tensor([0, 1])                           # [B] SCALAR!
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim)
    output = model(
        hidden_states=hidden_states,
        timestep=timestep,                                    # [B] only here!
        encoder_hidden_states=encoder_hidden_states,
    )
```

### Test 2: With masking:
```python
with torch.no_grad():
    hidden_states_mask = torch.zeros(batch_size, num_frames, num_joints)
    hidden_states_mask[0, :12, :] = 1
    hidden_states_mask[1, :16, :] = 1
    
    output = model(
        hidden_states=hidden_states,
        timestep=timestep,                                    # [B] scalar
        encoder_hidden_states=encoder_hidden_states,
        hidden_states_mask=hidden_states_mask,               # [B, T, J]
    )
```

**NOTE:** In the test file, timestep is `[B]` scalar. But in actual training, 
`create_sequence_ts()` expands it to `[B, N]` before passing to the transformer!

---

## CONCLUSIONS

### 1. **Timestep Format During Training:**
- **Input to trainer:** `[B]` scalar, sampled per batch element
- **After `create_sequence_ts`:** `[B, N]` expanded per-token
- **Passed to transformer:** `[B, N]` per-token timesteps
- **Why:** Wan 2.2 TI2V mode enables per-token diffusion modulation

### 2. **Masking During Training:**
- **`hidden_states_mask`:** Passed as `[B, T, J]` when `num_frames` is provided
- **`encoder_hidden_states_mask`:** Always `None` (not used)
- **`is_causal`:** Not used during training (always `False`)

### 3. **Expand_timesteps Status:**
- **Used:** YES, via `create_sequence_ts()`
- **Purpose:** Convert scalar timesteps to per-token for per-frame diffusion modulation

### 4. **Gradient Checkpointing:**
- **Enabled by default** in `PrismTransformerMotionModel.__init__()` (line 218)
- **Used in forward loop** for memory efficiency (lines 434-443)

