# PRISM Trainer Loss Computation - Quick Start Guide

## Files Located & What They Contain

### 1. **Main Trainer File** 
`hftrainer/trainers/motion/prism_trainer.py` (131 lines total)
- **Lines 14-40:** `PrismTrainer` class initialization with `translation_loss_weight` parameter
- **Lines 41-118:** `train_step()` method with full loss computation
- **Lines 120-130:** `val_step()` method for validation

### 2. **Config Files**
- **`configs/prism/prism_1b_tp2m_1frame.py`** (179 lines) - Base configuration
  - Trainer params at lines 95-101
  - Model at lines 17-93 (transformer, VAE, etc.)
  
- **`configs/prism/prism_1b_tp2m_multiframe.py`** (15 lines) - Multi-frame extension
  - Extends base config with `condition_num_frames=[1, 5, 9]`
  
- **`configs/prism/prism_debug_loss_split.py`** (177 lines) - Debug configuration
  - Explicitly sets `translation_loss_weight=0.5` for testing

---

## Key Code Sections

### Loss Computation (Lines 95-112 of prism_trainer.py)

```python
# Line 103: Compute MSE without reduction
mse = F.mse_loss(model_pred, targets.float(), reduction='none')
# Shape: [B, C=16, T', J=23]
#   - Token 0: Translation
#   - Tokens 1-22: Rotation (22 joints)

# Lines 106-108: Create combined mask
condition_mask = condition_frame_mask_vae.expand_as(mse).float()
padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
full_mask = condition_mask * padding_mask
# full_mask[b,c,t,j] = 1 only if:
#   - Frame NOT padded (based on num_frames)
#   - AND frame NOT conditioned

# Lines 113-115: Translation loss
mse_transl = mse[:, :, :, :1]              # Token 0 only
mask_transl = full_mask[:, :, :, :1]
loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

# Lines 117-119: Rotation loss  
mse_rot = mse[:, :, :, 1:]                 # Tokens 1-22
mask_rot = full_mask[:, :, :, 1:]
loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

# Lines 120-121: Weighted combination
w_t = self.translation_loss_weight  # default=0.5
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

---

## Understanding the Design Choices

### 1. Why Split Translation/Rotation Loss?

**The Problem:**
- Motion has 23 tokens: 1 translation + 22 rotations
- Naive MSE: `mean(mse)` averages equally over all 23
- Translation gets only 1/23 ≈ 4.3% gradient weight
- Rotation gets 22/23 ≈ 95.7% gradient weight
- Translation signal can be diluted/ignored

**The Solution:**
- Compute MSE separately for each component
- Each gets its own normalized mean
- Blend with configurable weights (default 0.5/0.5)
- Allows balancing importance of both

### 2. How Padding Mask Works

**Purpose:** Exclude frames beyond `num_frames` from loss computation

**Flow:**
1. `create_padding_mask()` called with batch's actual frame counts
2. Returns mask with 1s for valid frames, 0s for padded frames
3. Expanded to all dimensions to match MSE tensor
4. Multiplied element-wise with condition mask
5. Loss only computed where mask=1

**Example:**
```
Batch element 0: 32 real frames, padded to 45
  padding_mask = [1,1,...,1(32×), 0,0,...,0(13×)]
  
Batch element 1: 45 real frames, padded to 45
  padding_mask = [1,1,...,1(45×)]
```

### 3. Condition Mask vs Padding Mask

| Mask | Purpose | Type | Value Range |
|------|---------|------|-------------|
| **Padding** | Exclude padded frames | Per-batch | 0=padded, 1=valid |
| **Condition** | Mark frozen conditioning frames | Per-frame | 0=frozen, 1=trainable |

These are **combined** (multiplied) in `full_mask`:
- Loss = 0 if frame is padded **OR** conditioned
- Loss = MSE if frame is valid **AND** unconditioned

---

## Motion Representation Format

### Original SMPL (55D)
```
[tx, ty, tz, R_joint1(6D), R_joint2(6D), ..., R_joint22(6D)]
└─ 3D      └──────────────────────────────────────────────
  translation                 22 joints × 6D rotation
```

### In Latent Space (VAE encoded: 16×23)
```
Token 0: Translation [B, 16, T', 1]
  • Root global position (3D)
  • Encoded in 16 latent channels
  
Tokens 1-22: Joint Rotations [B, 16, T', 22]
  • Token i = Rotation of joint i
  • 6D continuous representation
  • Each in 16 latent channels
```

### Loss Computation Split
```python
# Translation: just token 0
mse[:, :, :, :1]   # Shape [B, 16, T', 1]

# Rotation: tokens 1-22
mse[:, :, :, 1:]   # Shape [B, 16, T', 22]
```

---

## Configuration Reference

### Base Config: `prism_1b_tp2m_1frame.py` (Lines 95-101)

```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],          # Use 1 frame for conditioning
    frame_condition_rate=0.1,          # ~10% of frames are conditioned
    prompt_drop_rate=0.1,              # ~10% prompts randomly dropped
    max_text_length=256,               # Max text tokens
    # translation_loss_weight not set → defaults to 0.5
)
```

### Multi-frame Config: `prism_1b_tp2m_multiframe.py` (Lines 11-14)

```python
trainer = dict(
    condition_num_frames=[1, 5, 9],    # Randomly pick 1, 5, or 9 frames
    frame_condition_rate=0.1,
)
```

### Debug Config: `prism_debug_loss_split.py` (Lines 95-102)

```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
    translation_loss_weight=0.5,       # EXPLICIT for debugging
)
```

---

## Adjusting Loss Weights

### How to Change Translation vs Rotation Importance

**Default (0.5/0.5):**
```python
translation_loss_weight=0.5
# loss = 0.5 * loss_transl + 0.5 * loss_rot
```

**Example: Favor Translation (60%)**
```python
translation_loss_weight=0.6
# loss = 0.6 * loss_transl + 0.4 * loss_rot
```

**Example: Proportional to Channel Count (4%/96%)**
```python
translation_loss_weight=0.04  # 1/23 ≈ 0.0435
# loss = 0.04 * loss_transl + 0.96 * loss_rot
```

**Example: Rotation Only**
```python
translation_loss_weight=0.0
# loss = loss_rot
```

---

## Output Metrics

The trainer returns 4 loss values:

```python
return {
    'loss': loss,                    # Combined (used for backprop)
    'loss_flow': loss.detach(),      # Alias for logging
    'loss_transl': loss_transl.detach(),  # Translation component
    'loss_rot': loss_rot.detach(),        # Rotation component
}
```

All are logged separately in tensorboard/wandb for analysis.

---

## Implementation Checklist

✅ **Already Implemented:**
- [x] MSE computed element-wise (no reduction)
- [x] Padding mask created and applied
- [x] Translation (token 0) and rotation (tokens 1-22) separated
- [x] Per-component masked reduction
- [x] Weighted combination with configurable parameter
- [x] Separate metric logging

✅ **Padding Mask Functionality:**
- [x] Prevents loss on frames beyond num_frames
- [x] Applied identically to translation and rotation
- [x] Dynamic per-batch based on actual frame counts

❌ **Not Implemented (Could Add Later):**
- [ ] Per-channel weighting within components
- [ ] Time-dependent weighting (early frames matter more?)
- [ ] Adaptive weighting from running statistics
- [ ] Separate loss function choice (L1 vs L2) per component

---

## Debugging Tips

### Check if Loss Splitting is Working

In training logs, look for:
- `loss_transl` and `loss_rot` tracked separately
- Both decreasing during training (good)
- If `loss_rot` dominates, consider increasing `translation_loss_weight`
- If `loss_transl` unstable, may indicate translation frames rarely sampled

### Verify Padding Mask

Check that:
1. `num_frames` is passed in batch
2. `create_padding_mask()` returns correct shape
3. Mask values are 0/1 (no NaN, no -1)
4. Loss values at padded positions are zero (not nan)

### Monitor Individual Metrics

```bash
# In tensorboard/wandb, monitor:
loss            # Should decrease smoothly
loss_transl     # Translation loss component
loss_rot        # Rotation loss component
loss_transl/loss_rot  # Ratio should stay balanced
```

---

## References

**Full Documentation:**
- `PRISM_TRAINER_LOSS_ANALYSIS.md` - Comprehensive analysis
- `PRISM_CODE_SECTIONS_REFERENCE.txt` - Line-by-line code with annotations
- `PRISM_LOSS_FLOW_DIAGRAM.txt` - Visual flow diagrams

**Related Files:**
- `hftrainer/models/base_model_bundle.py` - Methods like `create_padding_mask()`
- `hftrainer/trainers/base_trainer.py` - Base trainer class

