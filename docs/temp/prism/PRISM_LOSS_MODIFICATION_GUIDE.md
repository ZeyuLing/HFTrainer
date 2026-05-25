# PRISM Loss Modification Guide: Practical Implementation Patterns

## Quick Reference: Key Files

| File | Purpose | Key Lines |
|------|---------|-----------|
| `hftrainer/trainers/motion/prism_trainer.py` | Trainer class | 95-112: Loss computation |
| `configs/prism/prism_1b_tp2m_1frame.py` | Training config | 95-102: Trainer parameters |
| `configs/prism/prism_debug_loss_split.py` | Debug/test config | 1-3: Verifies loss split |

---

## Common Modifications

### 1. Experiment with Different Loss Weights

**Goal**: Find optimal translation/rotation balance for your use case

**Current Implementation** (lines 111-112):
```python
w_t = self.translation_loss_weight  # 0.5 default
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

**Modification A: Adaptive Weight Scheduling**
```python
# Add to __init__ (after line 39):
self.translation_loss_weight_schedule = translation_loss_weight_schedule or None

# Modify lines 111-112:
w_t = self.translation_loss_weight
if self.translation_loss_weight_schedule:
    # Example: Linear ramp from 0.3 to 0.7 over training
    progress = self.current_step / self.max_steps
    w_t = 0.3 + 0.4 * progress
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

**Modification B: Loss Magnitude-Based Weighting**
```python
# Dynamic weighting based on loss magnitudes (prevents scaling issues)
w_t = loss_transl.detach() / (loss_transl.detach() + loss_rot.detach() + 1e-6)
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

**Config Update** (prism_1b_tp2m_1frame.py):
```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
    translation_loss_weight=0.6,  # Increased from 0.5 for trajectory-focused task
)
```

---

### 2. Add Per-Joint Loss Supervision

**Goal**: Apply different loss weights to different joints (e.g., feet vs hands)

**Current Implementation**: Translation as one token, rotations as 22 tokens

**Modification: Per-Joint Weighting**
```python
# Add to __init__ (after line 39):
self.joint_loss_weights = {
    'translation': 1.0,
    'root': 1.0,           # Joint 1 (pelvis/root)
    'feet': 1.5,           # Joints 10,11 (L/R foot)
    'hands': 0.8,          # Joints 20,21 (L/R hand)
    'others': 1.0,         # Remaining joints
}

# Replace lines 101-109:
# Translation loss (unchanged)
mse_transl = mse[:, :, :, :1]
mask_transl = full_mask[:, :, :, :1]
loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
loss_transl = loss_transl * self.joint_loss_weights['translation']

# Rotation loss (per-joint weighted)
mse_rot = mse[:, :, :, 1:]              # [B, C, T', 22]
mask_rot = full_mask[:, :, :, 1:]

# SMPL joint order: 1=root, 2-3=spine, 4-6=left arm, 7-9=right arm, 10-12=left leg, 13-15=right leg, 16-19=left hand, 20-21=right hand, 22=neck
joint_indices = {
    'root': [0],           # Joint 1
    'feet': [9, 12],       # Joints 10, 13
    'hands': [19, 20, 21], # Joints 20, 21, 22 (indices in rotation tokens)
}

# Apply per-joint weighting
weighted_mse_rot = mse_rot.clone()
for joint_group, indices in joint_indices.items():
    w = self.joint_loss_weights.get(joint_group, 1.0)
    for idx in indices:
        if idx < weighted_mse_rot.shape[3]:
            weighted_mse_rot[:, :, :, idx] *= w

loss_rot = (weighted_mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

w_t = self.translation_loss_weight
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

**Config Update**:
```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
    translation_loss_weight=0.5,
    joint_loss_weights={
        'translation': 1.0,
        'feet': 1.5,        # Emphasize foot placement
        'hands': 0.8,
        'others': 1.0,
    },
)
```

---

### 3. Add Physics-Based Loss Regularization

**Goal**: Encourage physically plausible motion (smooth accelerations, joint limits)

**Current Implementation**: Pure MSE on latent predictions

**Modification: Add Smoothness Constraint**
```python
# Add after line 93 (after transformer output):
model_pred = self.bundle.transformer(...).float()

# Decode to SMPL space to measure acceleration
decoded_motion = self.bundle.decode_motion(model_pred)  # [B, T, 22*3]

# Compute motion smoothness (velocity + acceleration)
motion_vel = torch.diff(decoded_motion, dim=1)
motion_accel = torch.diff(motion_vel, dim=1)

# Smoothness loss: penalize high accelerations
smoothness_loss = F.mse_loss(motion_accel, torch.zeros_like(motion_accel))

# Original MSE loss
mse = F.mse_loss(model_pred, targets.float(), reduction='none')

# [Rest of loss computation...]

# Add smoothness to final loss (line 112):
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
loss = loss + self.smoothness_weight * smoothness_loss
```

**Add to __init__**:
```python
self.smoothness_weight = smoothness_weight or 0.01
```

**Config Update**:
```python
trainer = dict(
    type="PrismTrainer",
    # ... other params ...
    translation_loss_weight=0.5,
    smoothness_weight=0.01,  # Weight for acceleration regularization
)
```

---

### 4. Add Trajectory-Specific Loss

**Goal**: Improve long-horizon generation by explicitly supervising global trajectory

**Current Implementation**: Translation treated as single token

**Modification: Trajectory Consistency Loss**
```python
# Add after line 112:

# Optional: Add trajectory consistency loss
# (prevents drift in long-horizon generation)
if hasattr(self, 'trajectory_weight') and self.trajectory_weight > 0:
    # Compute trajectory (cumulative translation)
    latent_transl = model_pred[:, :, :, :1]  # [B, C, T', 1]
    traj_pred = torch.cumsum(latent_transl, dim=2)  # Cumulative sum over time
    
    target_transl = targets[:, :, :, :1]
    traj_target = torch.cumsum(target_transl, dim=2)
    
    trajectory_loss = F.mse_loss(traj_pred, traj_target)
    loss = loss + self.trajectory_weight * trajectory_loss
```

**Add to __init__**:
```python
self.trajectory_weight = trajectory_weight or 0.0
```

---

### 5. Implement Multi-Scale Loss

**Goal**: Supervise multiple temporal scales (short-term smoothness, long-term trajectory)

**Modification: Hierarchical Loss**
```python
# Replace lines 95-112 with:

mse = F.mse_loss(model_pred, targets.float(), reduction='none')
condition_mask = condition_frame_mask_vae.expand_as(mse).float()
padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
full_mask = condition_mask * padding_mask

# Base loss (current frame)
mse_transl = mse[:, :, :, :1]
loss_transl = (mse_transl * full_mask[:, :, :, :1]).sum() / (full_mask[:, :, :, :1].sum() + 1e-6)

mse_rot = mse[:, :, :, 1:]
loss_rot = (mse_rot * full_mask[:, :, :, 1:]).sum() / (full_mask[:, :, :, 1:].sum() + 1e-6)

loss_frame = self.translation_loss_weight * loss_transl + (1.0 - self.translation_loss_weight) * loss_rot

# Multi-scale loss (downsampled sequences)
loss_multiscale = 0
for scale in [2, 4]:  # 2-frame and 4-frame downsampling
    model_pred_ds = model_pred[:, :, ::scale, :]
    targets_ds = targets[:, :, ::scale, :]
    mask_ds = full_mask[:, :, ::scale, :]
    
    mse_ds = F.mse_loss(model_pred_ds, targets_ds, reduction='none')
    
    mse_transl_ds = mse_ds[:, :, :, :1]
    loss_transl_ds = (mse_transl_ds * mask_ds[:, :, :, :1]).sum() / (mask_ds[:, :, :, :1].sum() + 1e-6)
    
    mse_rot_ds = mse_ds[:, :, :, 1:]
    loss_rot_ds = (mse_rot_ds * mask_ds[:, :, :, 1:]).sum() / (mask_ds[:, :, :, 1:].sum() + 1e-6)
    
    loss_scale = self.translation_loss_weight * loss_transl_ds + (1.0 - self.translation_loss_weight) * loss_rot_ds
    loss_multiscale += loss_scale / len([2, 4])

w_t = self.translation_loss_weight
loss = loss_frame + self.multiscale_weight * loss_multiscale

return {
    'loss': loss,
    'loss_flow': loss.detach(),
    'loss_transl': loss_transl.detach(),
    'loss_rot': loss_rot.detach(),
    'loss_multiscale': loss_multiscale.detach() if self.multiscale_weight > 0 else torch.tensor(0.0),
}
```

**Add to __init__**:
```python
self.multiscale_weight = multiscale_weight or 0.0
```

---

## Testing Your Modifications

### 1. Quick Test: Debug Config
```bash
# Create test config with your modifications
cp configs/prism/prism_debug_loss_split.py configs/prism/prism_test_modification.py

# Edit to reduce training iterations
# In prism_test_modification.py, change:
# train_cfg = dict(
#     by_epoch=False,
#     max_iters=10,  # Just 10 iterations to verify
# )

# Run training
accelerate launch --multi_gpu --num_processes 8 tools/train.py configs/prism/prism_test_modification.py
```

### 2. Verify Loss Logging
Look for in logs:
```
iteration [1/10] loss_flow: 0.234  loss_transl: 0.456  loss_rot: 0.789
iteration [2/10] loss_flow: 0.215  loss_transl: 0.412  loss_rot: 0.701
# Both components should decrease
```

### 3. Check Loss Ratios
If you added per-joint weighting:
```
# Initial logs should show weighted components decreasing
# If loss_rot isn't decreasing, verify joint indices are correct
```

---

## Debugging Common Issues

### Issue 1: Loss diverges after modification
**Cause**: Incorrect mask dimensions or normalization
**Solution**: 
- Add print statements: `print(f"mse shape: {mse.shape}, full_mask shape: {full_mask.shape}")`
- Verify all dimensions match before division
- Check `+ 1e-6` epsilon is present

### Issue 2: One loss component stays constant
**Cause**: Mask is all zeros (dimension mismatch)
**Solution**:
- Print mask: `print(f"mask sum: {full_mask.sum()}")`
- Verify `expand_as()` is working correctly
- Check condition_mask and padding_mask logic

### Issue 3: Training slower after adding loss term
**Cause**: New loss term has wrong magnitude
**Solution**:
- Scale new loss by appropriate weight (0.01-0.1 typically)
- Log all components separately to see their magnitudes
- Use `detach()` on intermediate computations if needed

### Issue 4: Memory error with multi-scale loss
**Cause**: Storing multiple loss scales uses too much memory
**Solution**:
- Reduce number of scales: `for scale in [2, 4]` → `for scale in [4]`
- Reduce sequence length in config
- Ensure intermediate tensors are detached

---

## Verification Checklist

When implementing modifications:

- [ ] Loss computation produces scalar outputs
- [ ] Shapes remain consistent: `[B, C, T', J]` → masked → normalized
- [ ] Mask combined correctly: `full_mask = condition_mask * padding_mask`
- [ ] Normalization uses `mask.sum() + 1e-6` to prevent NaN
- [ ] Logging returns separate loss components
- [ ] Config passes all parameters to trainer
- [ ] Testing on small batch (batch_size=2) before full training
- [ ] Metrics decreasing smoothly (not diverging or staying constant)

---

## Performance Baseline

**Original PRISM Config** (lines 1-177 in prism_debug_loss_split.py):
- Batch size: 2 per GPU × 8 GPUs = 16 total
- Sequence length: 128 frames
- Training iterations: 50 (debug run)
- Expected loss progression: 
  - loss_flow: ~0.5 → ~0.3 over 50 iterations
  - loss_transl and loss_rot: similar magnitude and decrease

Your modifications should maintain:
- Loss decreasing monotonically (not diverging)
- Both components decreasing if multi-component loss
- Training throughput similar or better

---

## Example: Complete Modification (Adaptive Weight + Multi-Scale)

For a production modification, here's a complete example combining Sections 1 and 5:

**File**: `hftrainer/trainers/motion/prism_trainer.py`

```python
def __init__(
    self,
    bundle,
    condition_num_frames: Union[int, List[int]] = 1,
    frame_condition_rate: float = 0.1,
    prompt_drop_rate: float = 0.1,
    max_text_length: int = 128,
    val_prompts: Optional[List[str]] = None,
    num_val_inference_steps: int = 10,
    guidance_scale: float = 5.0,
    translation_loss_weight: float = 0.5,
    multiscale_weight: float = 0.0,  # NEW
    weight_schedule: Optional[str] = None,  # NEW
    **kwargs,
):
    super().__init__(bundle)
    self.condition_num_frames = condition_num_frames
    self.frame_condition_rate = frame_condition_rate
    self.prompt_drop_rate = prompt_drop_rate
    self.max_text_length = max_text_length
    self.val_prompts = val_prompts or ['a person walking forward']
    self.num_val_inference_steps = num_val_inference_steps
    self.guidance_scale = guidance_scale
    self.translation_loss_weight = translation_loss_weight
    self.multiscale_weight = multiscale_weight  # NEW
    self.weight_schedule = weight_schedule  # NEW
    self.current_step = 0  # NEW

def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing code lines 42-93 ...
    
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask
    
    # Base loss
    mse_transl = mse[:, :, :, :1]
    loss_transl_base = (mse_transl * full_mask[:, :, :, :1]).sum() / (full_mask[:, :, :, :1].sum() + 1e-6)
    
    mse_rot = mse[:, :, :, 1:]
    loss_rot_base = (mse_rot * full_mask[:, :, :, 1:]).sum() / (full_mask[:, :, :, 1:].sum() + 1e-6)
    
    # Adaptive weighting
    w_t = self.translation_loss_weight
    if self.weight_schedule == 'linear_ramp' and hasattr(self, 'max_steps'):
        progress = min(self.current_step / self.max_steps, 1.0)
        w_t = 0.3 + 0.4 * progress  # Ramp from 0.3 to 0.7
    
    loss_frame = w_t * loss_transl_base + (1.0 - w_t) * loss_rot_base
    
    # Multi-scale loss
    loss_multiscale = 0
    if self.multiscale_weight > 0:
        for scale in [2, 4]:
            if model_pred.shape[2] > scale:
                model_pred_ds = model_pred[:, :, ::scale, :]
                targets_ds = targets[:, :, ::scale, :]
                mask_ds = full_mask[:, :, ::scale, :]
                
                mse_ds = F.mse_loss(model_pred_ds, targets_ds, reduction='none')
                mse_transl_ds = mse_ds[:, :, :, :1]
                loss_transl_ds = (mse_transl_ds * mask_ds[:, :, :, :1]).sum() / (mask_ds[:, :, :, :1].sum() + 1e-6)
                
                mse_rot_ds = mse_ds[:, :, :, 1:]
                loss_rot_ds = (mse_rot_ds * mask_ds[:, :, :, 1:]).sum() / (mask_ds[:, :, :, 1:].sum() + 1e-6)
                
                loss_scale = w_t * loss_transl_ds + (1.0 - w_t) * loss_rot_ds
                loss_multiscale += loss_scale / 2  # Average over 2 scales
    
    # Final loss
    loss = loss_frame + self.multiscale_weight * loss_multiscale
    self.current_step += 1  # NEW
    
    return {
        'loss': loss,
        'loss_flow': loss.detach(),
        'loss_transl': loss_transl_base.detach(),
        'loss_rot': loss_rot_base.detach(),
        'loss_multiscale': loss_multiscale.detach() if isinstance(loss_multiscale, torch.Tensor) else torch.tensor(0.0),
    }
```

**Config**:
```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1, 5, 9],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
    translation_loss_weight=0.5,
    multiscale_weight=0.1,  # NEW: 10% multi-scale loss weight
    weight_schedule='linear_ramp',  # NEW: Ramp translation weight over training
)
```

This complete example:
1. ✓ Maintains backward compatibility (all new params optional)
2. ✓ Follows existing code style
3. ✓ Includes proper masking and normalization
4. ✓ Supports debug logging via separate loss components
5. ✓ Can be tested with prism_debug_loss_split config

