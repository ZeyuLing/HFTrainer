# PRISM Trainer: Loss Computation & Translation/Rotation Separation

## Overview

The PRISM trainer implements a **joint-factorized motion generation model** with per-token timestep conditioning. The key technical achievement is splitting MSE loss into separate translation and rotation components to prevent loss dilution.

## Problem Statement: Loss Dilution

The motion representation in PRISM is **2D latent factorized** across time and joints:
- **Shape**: `[B, C, T', J]` where:
  - `B` = batch size
  - `C` = channels (typically 16)
  - `T'` = latent frames (≈ T/4 due to VAE compression)
  - `J` = 23 tokens total
    - **Token 0** = root translation (1 joint)
    - **Tokens 1-22** = rotations (22 joints)

### The Issue
If we apply a single MSE loss across all 23 joints:
- Translation contributes: **1/23 ≈ 4.3%** to total loss
- Rotations contribute: **22/23 ≈ 95.7%** to total loss

This **numerical imbalance** means the model focuses ~95% of gradient updates on rotation, with translation receiving minimal supervision.

## Solution: Per-Channel Loss Weighting

### File Location
`./hftrainer/trainers/motion/prism_trainer.py` (lines 95-112)

### Train Step Loss Computation

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    # Lines 41-93: Encode, add noise, run transformer
    
    # Line 95: Compute per-element MSE loss
    mse = F.mse_loss(model_pred, targets.float(), reduction='none')
    # Shape: [B, C, T', J] where J=23
    
    # Lines 97-99: Create masks
    condition_mask = condition_frame_mask_vae.expand_as(mse).float()
    padding_mask = padding_mask.unsqueeze(1).expand_as(mse).float()
    full_mask = condition_mask * padding_mask
    
    # Lines 101-105: TRANSLATION LOSS
    # Extract first joint (J=0): root translation
    mse_transl = mse[:, :, :, :1]           # [B, C, T', 1]
    mask_transl = full_mask[:, :, :, :1]
    loss_transl = (mse_transl * mask_transl).sum() / (mask_transl.sum() + 1e-6)
    
    # Lines 107-109: ROTATION LOSS
    # Extract remaining 22 joints (J=1:22): rotations
    mse_rot = mse[:, :, :, 1:]              # [B, C, T', 22]
    mask_rot = full_mask[:, :, :, 1:]
    loss_rot = (mse_rot * mask_rot).sum() / (mask_rot.sum() + 1e-6)
    
    # Lines 111-112: WEIGHTED COMBINATION
    w_t = self.translation_loss_weight  # default: 0.5
    loss = w_t * loss_transl + (1.0 - w_t) * loss_rot
    
    return {
        'loss': loss,
        'loss_flow': loss.detach(),
        'loss_transl': loss_transl.detach(),
        'loss_rot': loss_rot.detach(),
    }
```

### Key Design Decisions

1. **Separate Normalization**: Each loss is normalized independently:
   - `loss_transl = sum(mse_transl * mask) / sum(mask_transl)`
   - `loss_rot = sum(mse_rot * mask) / sum(mask_rot)`
   
   This ensures each component has its own scale, avoiding magnitude dominance.

2. **50-50 Weighting** (default `translation_loss_weight=0.5`):
   - Gives equal importance to translation and rotation gradients
   - Allows joint tuning via `translation_loss_weight` parameter

3. **Padding-Aware Masking**:
   - Uses `condition_frame_mask_vae` for per-token timestep conditioning (prefix frames at t=0 don't contribute to loss)
   - Uses `padding_mask` to exclude padded frames
   - Combined mask avoids gradient leakage to ignored regions

## Configuration

### Training Config
File: `configs/prism/prism_1b_tp2m_1frame.py` (lines 95-102)

```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],          # Pose conditioning frames
    frame_condition_rate=0.1,          # 10% of batches get pose conditioning
    prompt_drop_rate=0.1,              # Classifier-free guidance
    max_text_length=256,               # T5 text token limit
    translation_loss_weight=0.5,       # TUNABLE: Loss weighting
)
```

### Multi-Frame Fine-Tuning
File: `configs/prism/prism_1b_tp2m_multiframe.py` (lines 11-14)

```python
trainer = dict(
    condition_num_frames=[1, 5, 9],    # Support 1, 5, or 9 frame conditioning
    frame_condition_rate=0.1,
)
```

### Debug Config
File: `configs/prism/prism_debug_loss_split.py` (lines 1-3)

Explicitly sets `translation_loss_weight=0.5` and trains for 50 iterations to verify loss separation.

## Model Architecture Context

### Latent Space Design
- **2D Factorization**: Joint-wise tokenization enables independent KL per-joint
- **Latent Dimension**: `[T', 23, 16]` (time, joints, features)
- **VAE Output**: AutoencoderKLPrism2DTK (pre-trained from vermo_vae checkpoint)

### Transformer
- **Type**: PrismTransformerMotionModel (1.4B parameters)
- **Layers**: 30 blocks with 2D RoPE (time + joint axes)
- **Text Encoder**: T5-XXL (4096-dim embeddings)
- **Patch Size**: (1, 1) - no spatial patching, per-joint processing

### Scheduler
- **Type**: FlowMatchEulerDiscreteScheduler
- **Timesteps**: 1000 (flow matching continuous scale)
- **Shift**: 5.0 (enables KAFS depth-dependent denoising)

## Per-Token Timestep Conditioning Mechanism

### How Masking Works in Loss
Line 78 in `prism_trainer.py`:
```python
# Prefix frames set to noise-free (condition_frame_mask_vae=True)
noisy_latents = torch.where(condition_frame_mask_vae, noisy_latents, latents)
```

Then in loss computation (line 97):
```python
# condition_mask prevents these prefix frames from contributing to loss gradient
condition_mask = condition_frame_mask_vae.expand_as(mse).float()
```

This ensures:
- **Prefix frames** (t=0, noise-free): Not included in MSE loss
- **Generation frames** (t>0, noisy): Full MSE loss applied
- Both masked together: Translation loss only includes generation frames

## Hyperparameter Tuning

### Primary Lever: `translation_loss_weight`

| Value | Effect |
|-------|--------|
| 0.0 | Pure rotation loss (pre-separation baseline) |
| 0.25 | 25% translation, 75% rotation |
| 0.5 | **Equal weighting** (default, recommended) |
| 0.75 | 75% translation, 25% rotation |
| 1.0 | Pure translation loss (translation-only) |

### Recommended Settings
- **Text-to-Motion**: `translation_loss_weight=0.5` (balance global trajectory + poses)
- **Pose-Conditioned**: `translation_loss_weight=0.5` (maintain trajectory continuity)
- **Physics-Focused**: `translation_loss_weight=0.75` (emphasize physics compliance)

## Evaluation Metrics Logged

The trainer returns per-step metrics:
```python
{
    'loss': combined_loss,           # Weighted combination
    'loss_flow': combined_loss,      # Alias for flow-matching loss
    'loss_transl': translation_loss, # Component 1
    'loss_rot': rotation_loss,       # Component 2
}
```

### Interpretation
- **loss_transl growing much faster than loss_rot**: Translation is undertrained
  - Action: Increase `translation_loss_weight`
- **loss_transl saturating early**: Translation is overfitting
  - Action: Decrease `translation_loss_weight`
- **Both declining smoothly**: Well-balanced training

## Implementation Details

### Dimensions & Indexing
```
Input motion: [B, T, 22*3]   (22 SMPL joints × 3D rotation)
           or [B, T, 1*3]    (root translation)

After VAE encoding: [B, 16, T', 23]  (2D latent grid)
  - Channel dimension (16): latent feature vectors
  - Time dimension (T'):   compressed frames (~T/4)
  - Joint dimension (23):  semantic tokens
    - Index 0:   root translation token
    - Index 1:22: rotation tokens (one per joint)

MSE loss shape: [B, C, T', J] = [B, 16, T', 23]
Split:
  - Translation: [B, C, T', 1]   (J=0 only)
  - Rotation:    [B, C, T', 22]  (J=1:23)
```

### Numerical Stability
- Mask normalization uses `+ 1e-6` epsilon to prevent division by zero
- Both losses normalized independently before weighting
- Prevents loss scaling issues across different batch compositions

## Training Data Pipeline

File: `configs/prism/prism_debug_loss_split.py` (lines 104-142)

```python
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type="MotionHubSingleAgentTextDataset",
        motion_key="smplx",              # SMPL-X 55-joint format
        anno_file="train_hq_motionhub_hymotion.json",
        pipeline=[
            # Load text caption
            dict(type="LoadCompatibleCaption", allow_none=False),
            
            # Load SMPL-X motion and normalize to 22-joint SMPL + rotation 6D
            dict(
                type="LoadSmplx55",
                key="motion",
                rot_type="rotation_6d",  # Continuous 6D rotation representation
                transl_type="abs_rel",   # Absolute + relative translation
                smpl_type="smpl_22",     # Project to 22-joint SMPL skeleton
                transl_aug_prob=0.75,    # Data augmentation
                transl_aug_yaw_deg=180.0,
                transl_aug_offset_std=(1.0, 0.0, 1.0),
            ),
            
            # Crop/pad to fixed length
            dict(
                type="RandomCropPadding",
                clip_len=128,     # Training clip length
                pad_mode="replicate",
                allow_longer=False,
            ),
            
            # Package for model input
            dict(
                type="PackInputs",
                keys=["motion", "num_frames", "caption"],
            ),
        ],
    ),
)
```

### Data Format
- **Motion representation**: Rotation 6D + absolute+relative translation per frame
- **SMPL skeleton**: 22 joints (21 body joints + 1 root/pelvis)
- **Batch size**: 2 (for 8×V100-32GB, ~16 total)
- **Sequence length**: 128 frames (4.27 seconds at ~30 fps)

## Validation & Deployment

### Inference
File: `prism_trainer.py` lines 120-130 (val_step)

```python
def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = PrismPipeline(self.bundle)
    preds = pipeline(
        prompts=self.val_prompts[0],
        num_frames_per_segment=33,      # 1.1 sec segments
        num_inference_steps=self.num_val_inference_steps,  # 10
        guidance_scale=self.guidance_scale,  # 5.0
    )
    return {'preds': preds, 'prompts': self.val_prompts}
```

### Production Deployment
The loss split design ensures:
1. **Balanced gradients**: No single component dominates optimization
2. **Controllable tradeoff**: Tune via `translation_loss_weight`
3. **Stable convergence**: Independent normalization prevents scale issues
4. **Interpretable metrics**: Separate loss curves for debugging

## Summary

The PRISM trainer's loss separation strategy is a **joint-factorization-enabled technique** that addresses a fundamental numerical imbalance in motion generation:

- **Problem**: 1 translation token vs 22 rotation tokens causes 95% of gradients to flow to rotations
- **Solution**: Separate MSE computation + independent normalization + weighted combination
- **Result**: Balanced gradient flow enabling better translation (trajectory) fidelity
- **Tuning**: `translation_loss_weight` parameter provides principled control

This design is uniquely enabled by the 2D joint-factorized latent space, where each token corresponds to a distinct kinematic unit, allowing independent loss supervision.

---

**Files Modified/Created**: 
- No changes to original code required—this analysis documents the existing implementation
- Key file: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/trainers/motion/prism_trainer.py`
- Config files: `configs/prism/prism_*.py`
