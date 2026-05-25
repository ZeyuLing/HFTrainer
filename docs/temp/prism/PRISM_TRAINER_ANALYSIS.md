# PRISM Trainer Loss Computation Analysis

## Summary

The PRISM trainer implements **flow-matching motion generation** with a clever **translation/rotation loss split** to prevent translation loss dilution. The motion representation is 135-dimensional (3 translation + 22 joints × 6 rot6d), where translation dominates at 1/23 ≈ 4.3% of dimensions while rotations are 22/23 ≈ 95.7%.

---

## Key Files Located

| File | Path | Purpose |
|------|------|---------|
| **PrismTrainer** | `hftrainer/trainers/motion/prism_trainer.py` | Main trainer with loss computation |
| **Config (multiframe)** | `configs/prism/prism_1b_tp2m_multiframe.py` | Multi-frame conditioning config |
| **Config (1-frame base)** | `configs/prism/prism_1b_tp2m_1frame.py` | Base 1-frame config |
| **Debug config** | `configs/prism/prism_debug_loss_split.py` | Validation config for loss split |

---

## PrismTrainer Implementation

### File: `hftrainer/trainers/motion/prism_trainer.py`

#### 1. **Initialization** (Lines 18-39)

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
    translation_loss_weight: float = 0.5,  # ← KEY PARAMETER
    **kwargs,
):
```

**Key parameter**: `translation_loss_weight` (default=0.5) — controls the balance between translation and rotation loss.

#### 2. **train_step Method** (Lines 41-118)

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    motion = batch['motion']
    captions = batch['caption']
    num_frames = batch.get('num_frames')
```

**Input**:
- `motion`: Raw SMPL motion [B, T, 135] (after VAE encode becomes [B, C, T', J'])
- `captions`: Text prompts for motion
- `num_frames`: Real frame count before padding

##### Step 1: Motion Encoding (Lines 46-47)
```python
latents = self.bundle.encode_motion(motion)
batch_size, _, latent_frames, latent_joints = latents.shape
```
- Encodes motion using VAE
- Output shape: [B, C, T', J] where T' is latent frames, J is latent joints

##### Step 2: Padding Mask Creation (Lines 49-55)
```python
padding_mask = self.bundle.create_padding_mask(
    num_frames=num_frames,
    batch_size=batch_size,
    latent_frames=latent_frames,
    latent_joints=latent_joints,
    device=latents.device,
)
```
- Creates mask to ignore padded frames in loss computation
- Essential for variable-length sequences in batch

##### Step 3: Text Encoding (Lines 56-61)
```python
text_states = self.bundle.encode_prompt(
    captions,
    max_sequence_length=self.max_text_length,
    prompt_drop_rate=self.prompt_drop_rate,
    dtype=next(self.bundle.transformer.parameters()).dtype,
)
```

##### Step 4: Condition Frame Masking (Lines 62-66)
```python
condition_frame_mask_vae = self.bundle.create_condition_mask(
    latents,
    frame_condition_rate=self.frame_condition_rate,
    condition_num_frames=self.condition_num_frames,
)
```
- Randomly masks frames for conditioning (typical: 10% frame condition rate)

##### Step 5: Timestep Selection (Lines 68-75)
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
- Randomly sample timesteps for flow matching

##### Step 6: Add Flow Noise (Lines 77-85)
```python
noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)
noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
timesteps = self.bundle.create_sequence_ts(...)
```
- Add noise for flow matching training
- Preserve condition frames (not noised)

##### Step 7: Model Forward Pass (Lines 87-93)
```python
model_pred = self.bundle.transformer(
    hidden_states=noisy_latents,
    encoder_hidden_states=text_states,
    timestep=timesteps,
    hidden_states_mask=padding_mask if num_frames is not None else None,
    encoder_hidden_states_mask=None,
).float()
```
- Forward pass through transformer
- Output shape: [B, C, T', J]

##### Step 8: **KEY - Loss Computation with Translation/Rotation Split** (Lines 95-112)

```python
# ===== CRITICAL SECTION: LOSS SPLIT =====

# Step 1: Compute MSE on all dimensions
mse = F.mse_loss(model_pred, targets.float(), reduction='none')
# mse shape: [B, C, T', J] where J=23 (token 0=translation, 1-22=rotation)

# Step 2: Create mask for condition and padding
condition_mask = condition_frame_mask_vae.expand_as(mse).float()
padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
full_mask = condition_mask * padding_mask

# Step 3: TRANSLATION LOSS (Joint index 0, first 1 position)
# Dims [0:3] compressed to J=1 representation at latent level
mse_transl = mse[:, :, :, :1]           # [B, C, T', 1]
mask_transl = full_mask[:, :, :, :1]
loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)

# Step 4: ROTATION LOSS (Joint indices 1-22, remaining 22 positions)
mse_rot = mse[:, :, :, 1:]              # [B, C, T', 22]
mask_rot = full_mask[:, :, :, 1:]
loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)

# Step 5: WEIGHTED COMBINATION
w_t = self.translation_loss_weight
loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

**Why this split is necessary**:
- Without split: rotation loss (95.7% of dimensions) overwhelms translation loss (4.3%)
- With split: both losses weighted equally by default (w_t=0.5)
- Prevents translation learning from being diluted

---

## Configuration Files

### File: `configs/prism/prism_1b_tp2m_multiframe.py` (Lines 1-15)

```python
# PRISM 1B text+pose-to-motion, multi-frame conditioning (1/5/9 frames)
#
# Resume from versatilemotion checkpoint (iter=15000):
#   bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_multiframe.py --auto-resume
#
# This stage fine-tunes from the 1-frame pretrained model with multi-frame
# pose conditioning (condition_num_frames=[1, 5, 9], frame_condition_rate=0.1).

_base_ = './prism_1b_tp2m_1frame.py'

trainer = dict(
    condition_num_frames=[1, 5, 9],
    frame_condition_rate=0.1,
)
```

**Key setting**: Uses **multi-frame conditioning** with [1, 5, 9] frames at 10% condition rate

---

### File: `configs/prism/prism_1b_tp2m_1frame.py` (Selected lines)

#### Trainer Configuration (Lines 95-101)

```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
)
```

**Note**: `translation_loss_weight` NOT set, defaults to 0.5

#### Debug Configuration (Lines 95-102 in `prism_debug_loss_split.py`)

```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
    translation_loss_weight=0.5,  # ← EXPLICITLY SET FOR VERIFICATION
)
```

---

## Motion Representation Details

### 135-Dimensional Layout

Based on code comment (Line 96):
```
# mse shape: [B, C, T', J] where J=23 (token 0=translation, 1-22=rotation)
```

At the latent level (after VAE encoding), the representation is compressed to 23 tokens:
- **Token 0**: Translation (3D absolute translation compressed)
- **Tokens 1-22**: Rotation (22 SMPL joints, each represented by rotation_6d)

#### Original 135-dim Motion Space (Before VAE):
```
dims [0:3]     — translation (3 absolute)
dims [3:9]     — joint 0 (Pelvis): rotation_6d
dims [9:15]    — joint 1 (L_Hip): rotation_6d
... (continuing for all 22 joints)
dims [129:135] — joint 21 (R_Wrist): rotation_6d
```

---

## Padding Mask Logic

### Where Padding Mask is Used

**Creation** (Line 49-55):
```python
padding_mask = self.bundle.create_padding_mask(...)
```

**Expansion for Loss** (Line 98-99):
```python
padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
full_mask = condition_mask * padding_mask
```

**Application** (Lines 105, 109):
```python
loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)
```

### Purpose
- Prevents loss computation on padded (fake) frames
- Only valid frames contribute to both translation and rotation losses
- Denominator includes epsilon (1e-6) to avoid division by zero

---

## Loss Return Values

### Output Dictionary (Lines 113-118)

```python
return {
    'loss': loss,                          # Combined weighted loss
    'loss_flow': loss.detach(),            # For monitoring
    'loss_transl': loss_transl.detach(),  # Translation loss component
    'loss_rot': loss_rot.detach(),        # Rotation loss component
}
```

All losses are:
1. **Averaged over both batch and sequence dimensions**
2. **Masked by padding** to ignore padded frames
3. **Separately weighted** - translation gets `w_t`, rotation gets `(1-w_t)`
4. **Detached for logging** - prevent double backprop

---

## Key Design Decisions

### 1. Joint-Level Split
- Translation gets its own loss term with adjustable weight
- Prevents rotation from dominating (95.7% vs 4.3% of dims)
- Default 50-50 split can be tuned via `translation_loss_weight`

### 2. Per-Frame Masking
- Both condition frames AND padded frames are masked
- Ensures only real, unobserved frames contribute to loss
- Critical for variable-length batch handling

### 3. Flow Matching Formulation
- Uses `add_flow_noise` to generate noisy_latents and targets
- Model predicts targets (clean motion) from noisy_latents
- MSE loss measures prediction accuracy

### 4. Padding Mask Application
```python
# Double-check: mask is applied in loss aggregation
loss = (mse * full_mask).sum() / (full_mask.sum() + eps)
# NOT: mse.mean(dim=...)  ← Would average over padded frames!
```

---

## How to Customize the Loss Split

To change translation/rotation loss weighting:

```python
# Option 1: In config
trainer = dict(
    type="PrismTrainer",
    translation_loss_weight=0.3,  # Give 30% weight to translation
)

# Option 2: Runtime
trainer.translation_loss_weight = 0.7  # Give 70% weight to translation

# Interpretation:
# translation_loss_weight=0.0   → Pure rotation loss
# translation_loss_weight=0.5   → Equal weighting (default)
# translation_loss_weight=1.0   → Pure translation loss
```

---

## Motion Encoding Pipeline

The motion representation goes through:
```
Raw SMPL Motion (T, 135)
  ↓
bundle.encode_motion() [via VAE]
  ↓
Latents (B, C, T', J=23)  ← This is what the model operates on
  ↓
add_flow_noise() [Flow Matching]
  ↓
noisy_latents (B, C, T', 23)
targets (B, C, T', 23)
  ↓
model.forward(noisy_latents, ...) → model_pred
  ↓
MSE Loss: L = ||model_pred - targets||²
  ↓
SPLIT LOSS:
  - Loss_transl from mse[:,:,:,:1]   (translation token)
  - Loss_rot from mse[:,:,:,1:]      (rotation tokens)
```

---

## Summary of Key Code Sections

| Section | Lines | Purpose |
|---------|-------|---------|
| **Init** | 18-39 | Define hyperparameters including `translation_loss_weight` |
| **Train Step** | 41-118 | Main training loop |
| **Motion Encode** | 46-47 | Encode to latent space |
| **Padding Mask** | 49-55 | Create mask for padded frames |
| **Forward Pass** | 87-93 | Transformer inference |
| **MSE Loss** | 95 | Compute raw MSE on all dimensions |
| **Mask Creation** | 97-99 | Combine condition + padding masks |
| **Translation Loss** | 103-105 | Extract and compute translation component |
| **Rotation Loss** | 107-109 | Extract and compute rotation component |
| **Weighted Combo** | 111-112 | **Combine with weights** |
| **Return Dict** | 113-118 | Package losses for logging |

---

## To Implement Translation/Rotation Split in Your Code

```python
# Get the VAE-encoded representation [B, C, T', 23]
latents = self.bundle.encode_motion(motion)

# Add noise and get targets
noisy_latents, targets = self.bundle.add_flow_noise(latents, timesteps)

# Get model predictions
model_pred = self.bundle.transformer(noisy_latents, ...)

# Compute MSE loss
mse = F.mse_loss(model_pred, targets.float(), reduction='none')  # [B,C,T',23]

# Extract translation (token 0) and rotation (tokens 1-22)
mse_transl = mse[:, :, :, :1]   # [B, C, T', 1]
mse_rot = mse[:, :, :, 1:]      # [B, C, T', 22]

# Apply padding mask if available
if padding_mask is not None:
    mask_transl = padding_mask.expand_as(mse_transl)
    mask_rot = padding_mask.expand_as(mse_rot)
    loss_transl = (mse_transl * mask_transl).mean()
    loss_rot = (mse_rot * mask_rot).mean()
else:
    loss_transl = mse_transl.mean()
    loss_rot = mse_rot.mean()

# Combine with configurable weight
w_t = translation_loss_weight  # e.g., 0.5
total_loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
```

