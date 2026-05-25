# PRISM Transformer Training - EXACT CODE SNIPPETS

## SECTION 1: The Exact Training Forward Call

**Location:** `/hftrainer/trainers/motion/prism_trainer.py` lines 87-93

```python
model_pred = self.bundle.transformer(
    hidden_states=noisy_latents,
    encoder_hidden_states=text_states,
    timestep=timesteps,
    hidden_states_mask=padding_mask if num_frames is not None else None,
    encoder_hidden_states_mask=None,
).float()
```

**What each parameter is:**
- `hidden_states`: `noisy_latents` - shape `[B, C, T, J]` (motion with added noise)
- `encoder_hidden_states`: `text_states` - shape `[B, N_ctx, 4096]` (encoded text prompts)
- `timestep`: `timesteps` - shape **`[B, N]`** (EXPANDED per-token, NOT scalar!)
- `hidden_states_mask`: `padding_mask if num_frames is not None else None` - shape `[B, T, J]` or `None`
- `encoder_hidden_states_mask`: `None` (always None, text mask not used)

---

## SECTION 2: The Timestep Expansion

**Location:** `/hftrainer/trainers/motion/prism_trainer.py` lines 79-83

```python
timesteps = self.bundle.create_sequence_ts(
    timesteps,
    condition_frame_mask_vae,
    self.bundle.transformer.config.patch_size,
)
```

**Input:** `timesteps` is `[B]` scalar - one timestep per batch element
**Output:** `timesteps` is `[B, N]` - same timestep repeated for each token

---

## SECTION 3: The create_sequence_ts Implementation

**Location:** `/hftrainer/models/motion/prism/bundle.py` lines 240-255

```python
def create_sequence_ts(
    self,
    ori_ts: torch.Tensor,
    condition_frame_mask_vae: torch.Tensor,
    patch_size=(1, 1),
) -> torch.Tensor:
    batch_size, _, latent_frames, latent_joints = condition_frame_mask_vae.shape
    post_patch_num_frames = latent_frames // patch_size[0]
    post_patch_num_joints = latent_joints // patch_size[1]
    target_ts = ori_ts.unsqueeze(1).unsqueeze(2).expand(batch_size, post_patch_num_frames, post_patch_num_joints)
    target_ts = torch.where(
        condition_frame_mask_vae[:, 0, :: patch_size[0], :: patch_size[1]],
        target_ts,
        0,
    )
    return target_ts.flatten(1)
```

**Step-by-step:**
1. `ori_ts.unsqueeze(1).unsqueeze(2)` - shape `[B]` → `[B, 1, 1]`
2. `.expand(batch_size, post_patch_num_frames, post_patch_num_joints)` - broadcast to `[B, T', J']`
3. `torch.where(condition_frame_mask_vae[:, 0, :: patch_size[0], :: patch_size[1]], ...)` - set conditioned frames to 0
4. `.flatten(1)` - shape `[B, T', J']` → `[B, N]` where N = T' × J'

---

## SECTION 4: Transformer Forward Signature

**Location:** `/hftrainer/models/motion/prism/network/transformer_prism.py` lines 232-241

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    hidden_states_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states_mask: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    is_causal: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
```

**Key point:** The `timestep` parameter accepts BOTH:
- `[B]` scalar timesteps (standard diffusion)
- `[B, N]` per-token timesteps (Wan 2.2 TI2V mode - used in PRISM training)

---

## SECTION 5: Timestep Detection in Transformer

**Location:** `/hftrainer/models/motion/prism/network/transformer_prism.py` lines 407-426

```python
# Handle per-token timesteps for Wan 2.2 TI2V mode
if timestep.ndim == 2:  # [B, N] shape - THIS IS PRISM TRAINING!
    ts_seq_len = timestep.shape[1]  # Extract N
    timestep = timestep.flatten()   # [B, N] -> [B*N]
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
    # Wan 2.2 TI2V: per-token modulation [B, N, 6*inner_dim] -> [B, N, 6, inner_dim]
    timestep_proj = timestep_proj.unflatten(2, (6, -1))
else:
    # Standard: global modulation [B, 6*inner_dim] -> [B, 6, inner_dim]
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
```

**Critical logic:**
- When `timestep.ndim == 2` (which is PRISM training), it:
  - Extracts the sequence length N
  - Flattens `[B, N]` → `[B*N]` for embedding
  - Passes `ts_seq_len=N` to embedder to unflatten back to `[B, N, ...]`
  - Unflatten projection to `[B, N, 6, inner_dim]` for per-token modulation

---

## SECTION 6: Hidden States Mask Processing

**Location:** `/hftrainer/models/motion/prism/network/transformer_prism.py` lines 323-357

```python
# Process hidden_states_mask (motion attention mask)
# Patchify the mask to match token sequence length.
# Original shape: [B, T, J] with 1=visible, 0=masked
# Target shape: [B, 1, 1, N] with 0=valid, -inf=masked (for attention bias)
if hidden_states_mask is not None:
    # Step 1: Reshape to separate patch dimensions
    # [B, T, J] -> [B, T//p_t, p_t, J//p_j, p_j]
    hidden_states_mask = hidden_states_mask.reshape(
        batch_size,
        post_patch_num_frames,
        p_t,
        post_patch_num_joints,
        p_j,
    )
    # Step 2: Min pooling across patch dimensions
    # If ANY position in a patch is masked (0), the entire patch is masked
    # [B, T//p_t, p_t, J//p_j, p_j] -> [B, T//p_t, J//p_j]
    hidden_states_mask = hidden_states_mask.amin(dim=(2, 4))

    # Step 3: Flatten to token sequence
    # [B, T//p_t, J//p_j] -> [B, N]
    hidden_states_mask = hidden_states_mask.flatten(1)

    # Step 4: Convert to attention bias format
    # 1 (visible) -> 0.0, 0 (masked) -> -inf (dtype min)
    # Final shape: [B, 1, 1, N] for broadcasting in attention
    hidden_states_mask = (
        (
            (1.0 - hidden_states_mask.float())
            * torch.finfo(hidden_states.dtype).min
        )
        .unsqueeze(1)
        .unsqueeze(2)
    )
```

**What happens:**
1. Input: `[B, T, J]` with 1=visible, 0=masked
2. Reshape to patch layout: `[B, T//p_t, p_t, J//p_j, p_j]`
3. Min-pool: if any token in patch is masked, entire patch is masked
4. Flatten: `[B, N]`
5. Convert to bias: 1 → 0 (attend), 0 → -inf (mask)
6. Output: `[B, 1, 1, N]` for broadcasting

---

## SECTION 7: Transformer Blocks Loop

**Location:** `/hftrainer/models/motion/prism/network/transformer_prism.py` lines 431-454

```python
for block in self.blocks:
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        # Gradient checkpointing: recompute activations during backward
        hidden_states = torch.utils.checkpoint.checkpoint(
            block,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,        # [B, N, 6, inner_dim] for PRISM training
            rotary_emb,
            hidden_states_mask,   # [B, 1, 1, N] or None
            encoder_hidden_states_mask,  # None
            causal_mask,          # None
            use_reentrant=False,
        )
    else:
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=timestep_proj,   # [B, N, 6, inner_dim] for PRISM training
            rotary_emb=rotary_emb,
            hidden_states_mask=hidden_states_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            causal_mask=causal_mask,
        )
```

**For PRISM training:**
- `timestep_proj` is `[B, N, 6, inner_dim]` (per-token)
- `hidden_states_mask` is `[B, 1, 1, N]` (or None)
- `encoder_hidden_states_mask` is `None`
- `causal_mask` is `None`

---

## SECTION 8: Embedding Processing

**Location:** `/hftrainer/models/motion/prism/network/embedding.py` lines 85-133

```python
def forward(
    self,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep_seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process timesteps and text hidden states into embeddings.
    
    For PRISM training:
    - timestep: [B*N] (flattened)
    - timestep_seq_len: N (sequence length)
    """
    # Step 1: Apply sinusoidal projection to timesteps
    timestep = self.timesteps_proj(timestep)

    # Step 2: Optionally reshape for sequence-level timesteps
    if timestep_seq_len is not None:
        # [B*N] -> [B, N, ...]
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

    # Step 6: Project text embeddings
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)

    return temb, timestep_proj, encoder_hidden_states
```

**For PRISM training with `timestep_seq_len=N`:**
1. Input: `timestep` `[B*N]`, `timestep_seq_len=N`
2. After unflatten: `temb` becomes `[B, N, inner_dim]`
3. After projection: `timestep_proj` becomes `[B, N, time_proj_dim]` where `time_proj_dim = 6*inner_dim`
4. Output: per-token embeddings!

---

## SECTION 9: Complete Training Loop

**Location:** `/hftrainer/trainers/motion/prism_trainer.py` lines 41-118

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    motion = batch['motion']
    captions = batch['caption']
    num_frames = batch.get('num_frames')

    # 1. Encode motion
    latents = self.bundle.encode_motion(motion)
    batch_size, _, latent_frames, latent_joints = latents.shape

    # 2. Create padding mask
    padding_mask = self.bundle.create_padding_mask(
        num_frames=num_frames,
        batch_size=batch_size,
        latent_frames=latent_frames,
        latent_joints=latent_joints,
        device=latents.device,
    )
    
    # 3. Encode text
    text_states = self.bundle.encode_prompt(
        captions,
        max_sequence_length=self.max_text_length,
        prompt_drop_rate=self.prompt_drop_rate,
        dtype=next(self.bundle.transformer.parameters()).dtype,
    )
    
    # 4. Create condition mask
    condition_frame_mask_vae = self.bundle.create_condition_mask(
        latents,
        frame_condition_rate=self.frame_condition_rate,
        condition_num_frames=self.condition_num_frames,
    )

    # 5. Sample random timesteps [B]
    step_indices = torch.randint(
        0,
        len(self.bundle.scheduler.timesteps),
        (batch_size,),
        device=latents.device,
    )
    scheduler_timesteps = self.bundle.scheduler.timesteps.to(device=latents.device)
    timesteps = scheduler_timesteps[step_indices]

    # 6. Add flow noise
    noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
    noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
    
    # 7. EXPAND timesteps [B] -> [B, N]
    timesteps = self.bundle.create_sequence_ts(
        timesteps,
        condition_frame_mask_vae,
        self.bundle.transformer.config.patch_size,
    )
    
    transformer_dtype = next(self.bundle.transformer.parameters()).dtype
    noisy_latents = noisy_latents.to(dtype=transformer_dtype)

    # 8. TRANSFORMER FORWARD with expanded timesteps
    model_pred = self.bundle.transformer(
        hidden_states=noisy_latents,
        encoder_hidden_states=text_states,
        timestep=timesteps,  # [B, N] per-token
        hidden_states_mask=padding_mask if num_frames is not None else None,
        encoder_hidden_states_mask=None,
    ).float()

    # 9. Compute loss
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask

    # Separate translation and rotation loss
    mse_transl = mse[:, :, :, :1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

    mse_rot = mse[:, :, :, 1:]
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

## SUMMARY TABLE

| Aspect | Status | Format | Location |
|--------|--------|--------|----------|
| **Timestep Format** | [B, N] per-token | Expanded via create_sequence_ts | prism_trainer.py:79-83 |
| **expand_timesteps Used** | YES | Via create_sequence_ts() | bundle.py:240-255 |
| **hidden_states_mask** | YES | [B, T, J] or None | prism_trainer.py:91 |
| **encoder_hidden_states_mask** | NO | Always None | prism_trainer.py:92 |
| **is_causal** | NO | Not passed (default False) | - |
| **Gradient Checkpointing** | YES | Enabled by default | transformer_prism.py:218 |

