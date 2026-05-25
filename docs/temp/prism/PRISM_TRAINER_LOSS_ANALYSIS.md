# PRISM Trainer Loss Computation Analysis

## Key Files Located

### 1. Main Trainer File
**Path:** `hftrainer/trainers/motion/prism_trainer.py`
- Lines 41-118: Full `train_step()` method

### 2. Config Files
**Base Config:** `configs/prism/prism_1b_tp2m_1frame.py` (179 lines)
**Multi-frame Config:** `configs/prism/prism_1b_tp2m_multiframe.py` (15 lines)
**Debug Config:** `configs/prism/prism_debug_loss_split.py` (177 lines)

---

## Loss Computation Code (PrismTrainer.train_step)

### Data Flow Summary
1. **Motion Encoding** (Line 46): Motion → VAE latents
2. **Padding Mask Creation** (Lines 49-55): Prevents loss on padded frames
3. **Text Encoding** (Lines 56-61): Prompts → text embeddings
4. **Condition Masking** (Lines 62-66): Marks which frames are conditioned
5. **Flow Noise Addition** (Lines 68-85): Adds noise and creates noisy latents
6. **Transformer Forward** (Lines 87-93): Predicts denoised latents
7. **Loss Computation** (Lines 95-112): **MAIN SECTION**

---

## Loss Computation Details (Lines 95-112)

### Line 95: Raw MSE Loss
```python
mse = F.mse_loss(model_pred, targets.float(), reduction='none')
```
- **Shape:** `[B, C, T', J]` where:
  - B = batch size
  - C = latent channels (16)
  - T' = temporal dimension (latent frames)
  - J = 23 (1 translation token + 22 rotation tokens)

### Lines 97-99: Create Full Mask
```python
condition_mask = condition_frame_mask_vae.expand_as(mse).float()
padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
full_mask = condition_mask * padding_mask
```

**Full Mask** combines:
- `condition_mask`: 1 where frames are NOT conditioned (should contribute to loss)
- `padding_mask`: 1 where frames are NOT padded

#### Key Insight: Padding Mask Prevents Loss on Padded Frames
- Line 49-55: Creates padding mask for valid frames only
- Line 98: Expands to match MSE shape
- Result: Loss = 0 for any padded (invalid) frame, regardless of condition

---

## Translation vs Rotation Loss Split

### Current Implementation (Lines 101-112)

#### Translation Loss (Lines 103-105)
```python
mse_transl = mse[:, :, :, :1]           # [B, C, T', 1]
mask_transl = full_mask[:, :, :, :1]
loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
```
- **Selects only J=0** (first token = translation)
- Shape: `[B, C, T', 1]`
- Masked reduction: sum → normalized by sum of mask elements

#### Rotation Loss (Lines 107-109)
```python
mse_rot = mse[:, :, :, 1:]              # [B, C, T', 22]
mask_rot = full_mask[:, :, :, 1:]
loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)
```
- **Selects J=1:23** (remaining 22 tokens = rotation joints)
- Shape: `[B, C, T', 22]`
- Same masked reduction

#### Why Split? (Comment on Line 101-102)
```python
# Separate translation (J=0) and rotation (J=1:) to prevent
# translation loss dilution (1/23 ≈ 4.3% vs 22/23 ≈ 95.7%).
```

**Problem Solved:**
- Without split: `loss = MSE.mean()` computes mean over all 23 tokens
- Translation (1/23) gets only ~4.3% weight
- Rotation (22/23) gets ~95.7% weight
- Translation could become "diluted" in the gradient

**Solution:**
- Compute separate per-token-group means first
- Then blend with configurable weights

---

## Weighted Combination (Lines 111-112)

```python
w_t = self.translation_loss_weight
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

**Configuration Examples:**

| Config | w_t | w_r | Effect |
|--------|-----|-----|--------|
| `translation_loss_weight=0.5` | 0.5 | 0.5 | Equal weight |
| `translation_loss_weight=0.3` | 0.3 | 0.7 | Translation gets 30% |
| `translation_loss_weight=0.04` | 0.04 | 0.96 | Proportional to channels |

**Current Default:** 
- Config line 39: `translation_loss_weight: float = 0.5`
- So loss = 0.5 × loss_transl + 0.5 × loss_rot

---

## Padding Mask Logic (Lines 49-55, 98-99)

### Creation
```python
padding_mask = self.bundle.create_padding_mask(
    num_frames=num_frames,
    batch_size=batch_size,
    latent_frames=latent_frames,
    latent_joints=latent_joints,
    device=latents.device,
)
```
- **Input:** `num_frames` = actual frame counts per batch element
- **Output:** Shape `[B, 1, T', 1]` or similar with 1s for valid, 0s for padded
- Called **once before forward pass** (Line 49-55)

### Application to Loss
```python
padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()  # Line 98
full_mask = condition_mask * padding_mask                         # Line 99
```
- Expands to `[B, C, T', J]` to match MSE shape
- Multiplies with condition_mask to create combined mask
- Effect: Zeroes out loss for padded frames before summation

---

## Output Metrics (Lines 113-118)

```python
return {
    'loss': loss,                    # Combined loss
    'loss_flow': loss.detach(),      # Alias
    'loss_transl': loss_transl.detach(),  # Translation component
    'loss_rot': loss_rot.detach(),        # Rotation component
}
```

All three losses are tracked separately in logs.

---

## Configuration Summary

### File: `configs/prism/prism_1b_tp2m_1frame.py`

**Trainer Hyperparameters (Lines 95-101):**
```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],              # 1 condition frame
    frame_condition_rate=0.1,              # 10% frames are conditioned
    prompt_drop_rate=0.1,                  # 10% prompts dropped
    max_text_length=256,                   # Max tokens for text
    # translation_loss_weight NOT specified = defaults to 0.5
)
```

**Multi-frame variant (Lines 11-14 of multiframe config):**
```python
trainer = dict(
    condition_num_frames=[1, 5, 9],        # Sample 1, 5, or 9 frames
    frame_condition_rate=0.1,              # Same 10%
)
```

**Debug variant (Line 95-102 of debug config):**
```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
    translation_loss_weight=0.5,           # EXPLICIT for debugging
)
```

---

## Motion Representation Format

From `smpl_pose_processor` config (Lines 74-92 of base config):
```python
smpl_pose_processor=dict(
    rot_type="rotation_6d",                # 6D continuous rotation representation
    transl_type="abs_rel",                 # Absolute + relative translation
    smpl_type="smpl_22",                   # 22 SMPL joints
)
```

**Encoding in Latent Space:**
- **Token 0:** Translation (3D position)
- **Tokens 1-22:** Rotation for each of 22 SMPL joints
- Each encoded as 6D rotation + 3D position in latent space

---

## Key Technical Insights

### 1. Padding Mask Design
- **Single source of truth:** `create_padding_mask()` called once (Line 49-55)
- **Purpose:** Mark frames beyond `num_frames` as invalid
- **Usage:** Prevents loss computation on padded/dummy data
- **Implementation:** Binary mask expanded to all dimensions

### 2. Loss Separation Strategy
- **Problem:** Translation is 1/23 of channels → diluted gradient
- **Solution:** Separate MSE → separate means → weighted sum
- **Benefit:** Allows tuning translation vs rotation importance
- **Cost:** Slightly more computation (negligible)

### 3. Condition Masking
- **Different from padding mask:** Controls which frames are used for conditioning
- **Role:** Some frames are "conditioned" (frozen), others are "denoised" (trained)
- **Interaction:** `full_mask = condition_mask * padding_mask`
  - Loss only computed on frames that are:
    1. NOT conditioned (condition_mask=1)
    2. AND NOT padded (padding_mask=1)

### 4. Flow Matching Context
- Uses flow matching diffusion (not traditional diffusion)
- `FlowMatchEulerDiscreteScheduler` with 1000 timesteps
- Transformer predicts "drift" from noise to data at random timesteps

---

## Implementation Checklist for Loss Splitting

✅ **Already Implemented (Lines 95-112):**
- [x] MSE computed on full output
- [x] Padding mask created and applied
- [x] Translation (J=0) and rotation (J=1:) separated
- [x] Per-group masked reduction (mean weighted by mask)
- [x] Configurable weighting via `translation_loss_weight`
- [x] Separate metric logging for both components

✅ **Padding Mask Already Handles:**
- [x] Prevents loss on frames beyond `num_frames`
- [x] Applied to both translation and rotation identically
- [x] Uses dynamic mask creation based on batch

❌ **NOT Implemented (Could Be Added):**
- [ ] Per-channel weighting within translation or rotation
- [ ] Time-dependent weighting (early frames more important?)
- [ ] Adaptive weighting based on running statistics
- [ ] Separate L1/L2 choices per component

