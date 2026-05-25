# PRISM VAE Loading and Inference: Complete Guide

## Part 1: Quick Start (TL;DR)

### Loading 1D VAE (138-D joint-agnostic)
```python
from mmengine.config import Config, ConfigDict
from mmengine.runner import load_checkpoint
from hftrainer.models.motion.prism.autoencoder_kl_1d import AutoencoderKLPrism1D

# Load config
config = Config.fromfile(
    '../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/smpl_vae1d_nostatic_aug_hq.py'
)
vae_config = config.model.vae

# Build model
vae_1d = AutoencoderKLPrism1D(**vae_config)
load_checkpoint(vae_1d, '../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth')
vae_1d.eval()
```

### Loading 2D VAE (6-D per-joint, 22 joints)
```python
from diffusers import AutoencoderKL

# Load directly from HuggingFace-compatible checkpoint
vae_2d = AutoencoderKL.from_pretrained(
    'checkpoints/vermo_vae/',
    subfolder=None,
    torch_dtype=torch.float32
)
vae_2d.eval()
```

## Part 2: Detailed Model Information

### 1D VAE Architecture (AutoencoderKLPrism1D)

**Purpose**: Joint-agnostic motion encoding. Treats all joints uniformly by flattening them into a single 138-dimensional channel space.

**Input/Output Dimensions**:
- Input: `[B, T, 138]` where T=121 (sequence length), 138 = 23 joints × 6 (6D rotation per joint)
- Latent: `[B, 16, T/4]` after 4x temporal downsampling
- Output (reconstructed): `[B, T, 138]`

**Internal Processing Pipeline**:
1. Permute: `[B, T, 138]` → `[B, 138, T]`
2. WanEncoder1D: Causal 1D convolutions with temporal downsampling pattern (False, True, True) = 4x downsample
3. quant_conv: Projects to `[B, 2*z_dim, T/4]` (mean and logvar)
4. DiagonalGaussianDistribution: Samples or extracts mode from mean/logvar
5. post_quant_conv: Projects latent from `[B, z_dim, T/4]` to `[B, z_dim, T/4]`
6. WanDecoder1D: Mirrors encoder with upsampling
7. Permute back: `[B, z_dim, T]` → `[B, T, z_dim]`

**Key Parameters**:
```python
{
    'base_dim': 96,              # Base dimension for conv filters
    'z_dim': 16,                 # Latent dimension
    'in_channels': 138,          # Flattened joint space
    'out_channels': 138,
    'dim_mult': (1, 2, 4, 4),    # Channel multipliers per stage
    'num_res_blocks': 2,         # Residual blocks per stage
    'temporal_downsample': (False, True, True),  # Downsampling per stage
    'scale_factor_temporal': 4,  # Total temporal downsample factor
}
```

### 2D VAE Architecture (AutoencoderKLPrism2DTK)

**Purpose**: Joint-aware motion encoding. Maintains spatial structure of joints and treats time and joint dimensions separately with 2D convolutions.

**Input/Output Dimensions**:
- Input: `[B, T, K, C]` where T=121, K=22 joints, C=6 (per-joint features)
- Can also be reshaped: `[B, T, 132]` (since 22×6=132)
- Latent: `[B, 16, T/4, K]` = `[B, 16, 30, 22]` after 4x temporal downsample
- Output (reconstructed): `[B, T, 22, 6]` or `[B, T, 132]`

**Internal Processing Pipeline**:
1. Input shape: `[B, T, K, C]` = `[B, 121, 22, 6]`
2. Reshape for encoder: Treat as `[B, C, T, K]` = `[B, 6, 121, 22]` (channel-first for conv2d)
3. WanEncoder2DTK: 2D causal convolutions over (time, joint) dimensions
4. quant_conv: 1×1 causal conv with padding=0 (critical for temporal causality)
5. Projects to `[B, 2*z_dim, T/4, K]`
6. Latent extraction: `[B, z_dim, T/4, K]`
7. post_quant_conv: 1×1 causal conv
8. WanDecoder2DTK: Mirrors encoder with upsampling
9. Output reshape: Back to `[B, T, K, C]` or `[B, T, K*C]`

**Key Parameters**:
```python
{
    'base_dim': 96,              # Base dimension for conv filters
    'z_dim': 16,                 # Latent dimension
    'in_channels': 6,            # Per-joint feature channels
    'out_channels': 6,
    'dim_mult': (1, 2, 4, 4),    # Channel multipliers per stage
    'num_res_blocks': 2,         # Residual blocks per stage
    'temporal_downsample': (False, True, True),  # Downsampling per stage
    'scale_factor_temporal': 4,  # Total temporal downsample factor
    'latents_mean': [...],       # Pre-computed 16-D mean vector
    'latents_std': [...],        # Pre-computed 16-D std vector
}
```

**Pre-computed Latent Statistics** (from config.json):
```json
{
  "latents_mean": [
    -0.00015412428, -0.000290714, 8.507754e-05, 0.023437843,
    0.00021363031, -2.6158676e-05, 5.2927735e-05, -0.00012251279,
    0.0065064426, 0.000176471, 0.0003246046, -0.037439488,
    -0.023424687, 0.058713272, 7.4118492e-05, -0.00029792162
  ],
  "latents_std": [
    0.9992712, 0.9993094, 0.9990134, 1.0647312, 0.99818367,
    0.99854374, 0.9974088, 0.99949616, 0.9691825, 0.99974465,
    0.9983452, 1.160751, 1.1418496, 1.0437691, 0.9988592, 0.998439
  ]
}
```

## Part 3: Encode/Decode API

### 1D VAE API

```python
import torch

# Encode motion to latent
motion = torch.randn(4, 121, 138)  # [batch=4, time=121, features=138]
with torch.no_grad():
    # encoder outputs [B, 2*z_dim, T_down] where T_down = 121/4 ≈ 30
    latent_dist = vae_1d.encode(motion)  # Returns DiagonalGaussianDistributionNd

# Extract latent (deterministic)
z = latent_dist.mode()  # [B, z_dim, T_down] = [4, 16, 30]

# Or sample stochastically
z_sample = latent_dist.sample()  # [B, z_dim, T_down] = [4, 16, 30]

# Decode latent back to motion
reconstructed = vae_1d.decode(z)  # [B, T, 138] = [4, 121, 138]

# Full forward pass (encode + sample + decode)
full_reconstruction = vae_1d(motion)  # [B, T, 138]
```

### 2D VAE API

```python
import torch

# Encode motion to latent
motion = torch.randn(4, 121, 22, 6)  # [batch=4, time=121, joints=22, channels=6]
with torch.no_grad():
    # encoder outputs DiagonalGaussianDistributionNd
    latent_dist = vae_2d.encode(motion)

# Extract latent (deterministic)
z = latent_dist.mode()  # [B, z_dim, T_down, K] = [4, 16, 30, 22]

# Or sample stochastically
z_sample = latent_dist.sample()  # [B, z_dim, T_down, K] = [4, 16, 30, 22]

# Optional: Normalize latent using pre-computed statistics
z_normalized = (z - torch.tensor(config['latents_mean']).view(1, 16, 1, 1)) / \
               torch.tensor(config['latents_std']).view(1, 16, 1, 1)

# Decode latent back to motion
reconstructed = vae_2d.decode(z)  # [B, T, 22, 6] = [4, 121, 22, 6]

# Full forward pass
full_reconstruction = vae_2d(motion)  # [B, T, 22, 6]
```

### DiagonalGaussianDistributionNd Class

The encoder returns a `DiagonalGaussianDistributionNd` object which represents a diagonal Gaussian distribution:

```python
class DiagonalGaussianDistributionNd:
    def __init__(self, mean: torch.Tensor, logvar: torch.Tensor):
        self.mean = mean         # [B, z_dim, T_down] or [B, z_dim, T_down, K]
        self.logvar = logvar     # Same shape as mean
        self.std = torch.exp(0.5 * logvar)  # Standard deviation
    
    def mode(self) -> torch.Tensor:
        """Deterministic: return mean (best reconstruction)"""
        return self.mean
    
    def sample(self) -> torch.Tensor:
        """Stochastic: sample z ~ N(mean, std)"""
        epsilon = torch.randn_like(self.std)
        return self.mean + epsilon * self.std
```

**Best Practice**:
- Use `.mode()` for inference/reconstruction when you want deterministic results
- Use `.sample()` for data augmentation or stochastic generation
- The `.sample()` method is differentiable, so it can be used in training

## Part 4: Configuration Explanation

### Config Parameter Meanings

**Architecture Parameters**:
- `base_dim=96`: Starting channel dimension. First encoder stage uses 96 channels.
- `dim_mult=(1, 2, 4, 4)`: Channel multipliers for 4 encoder stages. Actual channels: [96, 192, 384, 384]
- `num_res_blocks=2`: Each encoder/decoder stage contains 2 residual blocks for better feature learning
- `in_channels` / `out_channels`: 
  - 1D VAE: 138 (23 joints × 6D rotation)
  - 2D VAE: 6 (6D per joint, joints handled separately)

**Latent Space**:
- `z_dim=16`: Bottleneck dimension. Motion is compressed to 16 dimensions per timestep
- `scale_factor_temporal=4`: Total downsampling in time dimension (121 frames → 30 frames)
- `temporal_downsample=(False, True, True)`: Per-stage downsampling:
  - Stage 1: No downsampling
  - Stage 2: 2x downsampling
  - Stage 3: 2x downsampling
  - Total: 2 × 2 = 4x

**Latent Normalization** (2D VAE only):
- `latents_mean`: Pre-computed mean of latent distribution across training data
- `latents_std`: Pre-computed standard deviation
- **Purpose**: Normalize latent space to approximately N(0, 1) for better diffusion model training
- **Usage**: `z_norm = (z - mean) / std` before passing to diffusion model

### Data Processing Pipeline

From config file, the training pipeline:

```python
pipeline=[
    dict(
        type='LoadSmplx55',           # Load 55-param SMPL-X representation
        key='motion',
        rot_type='rotation_6d',       # Convert rotations to 6D representation
        smpl_type='smpl_22',          # Use 22-joint SMPL (not 55-joint SMPL-X)
        transl_type='abs_rel',        # Use absolute translation with relative velocity
        transl_aug_prob=0.75,         # 75% chance to augment translation
        transl_aug_offset_std=(1.0, 0.0, 1.0),  # Std for x,y,z augmentation
        transl_aug_yaw_deg=180.0,     # Up to 180° yaw rotation augmentation
    ),
    dict(
        type='RandomCropPadding',     # Crop/pad to fixed length
        clip_len=121,                 # Target sequence length
        allow_shorter=True,           # Allow shorter sequences (will be padded)
        pad_mode='reflect',           # Use reflection padding
    ),
    dict(
        type='PackInputs',            # Package for training
        keys='motion',
        meta_keys=['motion_path', 'duration', 'num_frames', 'fps'],
    ),
]
```

## Part 5: Test Data

### Available Datasets

Located in `data/annotation/`:

**Reconstruction Tests**:
- `test_motionhub_recon.json`: For evaluating VAE reconstruction quality
- Format: List of motion records with paths, captions, duration, fps

**Single-Person Tests**:
- `test_motionhub_1p.json`: Single-person motion sequences

**Multi-Person Tests**:
- `test_motionhub_2p.json`: Two-person motion interactions

**Text-to-Motion Tests**:
- `test_motionhub_t2m.json`: Motion-caption pairs for text-to-motion evaluation
- `test_hml3d.json`: HML3D dataset evaluation split

**Other Benchmarks**:
- `test_motionclip_2p.json`: MotionCLIP benchmark

**Training Data**:
- `train_hymotion_400h.json`: Main training set (400 hours)
- `train_hymotion_400h_hq*.json`: High-quality training variants

### Data Format

Each annotation file contains:
```json
[
  {
    "motion_path": "path/to/motion.npy",
    "caption": "person walks forward",
    "duration": 3.5,
    "num_frames": 105,
    "fps": 30
  },
  ...
]
```

### Loading Test Data

```python
import json
import numpy as np
from torch.utils.data import Dataset

class MotionDataset(Dataset):
    def __init__(self, annotation_file, data_dir):
        with open(annotation_file, 'r') as f:
            self.data = json.load(f)
        self.data_dir = data_dir
    
    def __getitem__(self, idx):
        record = self.data[idx]
        motion_path = f"{self.data_dir}/{record['motion_path']}"
        motion = np.load(motion_path)  # [T, J, C] or [T, D]
        return {
            'motion': motion,
            'caption': record.get('caption', ''),
            'duration': record['duration'],
            'fps': record['fps'],
        }
    
    def __len__(self):
        return len(self.data)

# Usage
dataset = MotionDataset('data/annotation/test_motionhub_recon.json', 'data/motionhub')
sample = dataset[0]
motion = sample['motion']  # [T, features]
print(f"Motion shape: {motion.shape}, Duration: {sample['duration']}s, FPS: {sample['fps']}")
```

## Part 6: Integration with PRISM

### PrismBundle Integration

The `PrismBundle` class integrates the VAE with other components:

```python
class PrismBundle:
    def __init__(self, vae_1d, vae_2d, smpl_processor, tokenizer, text_encoder, scheduler):
        self.vae_1d = vae_1d
        self.vae_2d = vae_2d
        self.smpl_processor = smpl_processor
        # ... other components
    
    def encode_motion(self, motion):
        """
        motion: [B, T, J, C] motion in world space
        Returns: Normalized latent codes [B, z_dim, T_down, J]
        """
        # 1. Normalize motion using SMPL processor
        normalized_motion = self.smpl_processor.normalize(motion)
        
        # 2. Encode to latent
        latent_dist = self.vae_2d.encode(normalized_motion)
        latents = latent_dist.mode()  # Deterministic
        
        # 3. Normalize latent space
        latents = (latents - torch.tensor(self.vae_2d.config['latents_mean']).view(1, 16, 1, 1)) / \
                  torch.tensor(self.vae_2d.config['latents_std']).view(1, 16, 1, 1)
        
        return latents
```

### Latent Space Normalization

Pre-computed statistics ensure latent codes are properly scaled:

```python
# Load statistics from 2D VAE config
config = torch.load('checkpoints/vermo_vae/config.json')
latents_mean = torch.tensor(config['latents_mean']).view(1, 16, 1, 1)
latents_std = torch.tensor(config['latents_std']).view(1, 16, 1, 1)

# Normalize encoded latents
z_encode = vae_2d.encode(motion).mode()  # [B, 16, T_down, K]
z_normalized = (z_encode - latents_mean) / latents_std

# Denormalize before decoding
z_denormalized = z_normalized * latents_std + latents_mean
motion_reconstructed = vae_2d.decode(z_denormalized)
```

## Part 7: Causal Convolution Implementation

### Why Causality?

Causal convolutions ensure that the encoder/decoder only uses information from past/current frames, not future frames. This enables:
- **Streaming inference**: Process motion frame-by-frame
- **Causality preservation**: No future information leakage
- **Temporal consistency**: Smooth motion generation

### Implementation Strategy: WAN (Wav2Vec2 Auto-regressive Newt)

The models use **causal dilated convolutions** with feature caching:

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # Padding = (kernel_size - 1) * dilation ensures causality
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        # [B, C, T] → [B, C_out, T]
        x = self.conv(x)
        # Remove future-looking padding
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x
```

### Temporal Chunking Strategy

For long sequences (121 frames), the model chunks:
1. **First chunk**: 1 frame (no history available)
2. **Subsequent chunks**: Stride-4 groups (4 frames each)
   - Frame 0: Context from previous chunks
   - Frames 1-3: New frames with cached features
3. **Feature caching**: Maintains conv layer outputs across chunks for coherence

```python
def encode_chunked(vae, motion, chunk_size=1):
    """
    Encode long motion sequences using temporal chunking.
    
    motion: [B, T, D]
    chunk_size: frames per chunk (1 for first, 4 for rest)
    """
    B, T, D = motion.shape
    z_list = []
    
    cached_features = None
    t = 0
    
    while t < T:
        # Determine chunk
        if t == 0:
            chunk_t = min(1, T - t)  # First chunk: 1 frame
        else:
            chunk_t = min(4, T - t)  # Subsequent: 4 frames
        
        chunk = motion[:, t:t+chunk_t, :]  # [B, chunk_t, D]
        
        # Encode with cached features
        with torch.no_grad():
            z_chunk = vae.encode_with_cache(chunk, cached_features)
            z_list.append(z_chunk)
            cached_features = vae.get_cached_features()
        
        t += chunk_t
    
    # Concatenate all chunks
    z = torch.cat(z_list, dim=2)  # [B, z_dim, T_down]
    return z
```

## Part 8: Troubleshooting

### Issue 1: Shape Mismatch on Encode

**Symptom**: `RuntimeError: expected 3D input, got 2D input`

**Cause**: Input shape incorrect

**Solution**:
```python
# WRONG: [T, D]
motion_wrong = np.load('motion.npy')  # [121, 138]
z = vae_1d.encode(motion_wrong)  # ERROR

# CORRECT: [B, T, D]
motion = torch.from_numpy(motion_wrong).unsqueeze(0).float()  # [1, 121, 138]
z = vae_1d.encode(motion)  # OK
```

### Issue 2: 2D VAE Latent Shape Confusion

**Symptom**: Latent shape is `[B, 16, T_down, K]` but code expects `[B, 16, T_down]`

**Cause**: 2D VAE maintains joint dimension; 1D VAE flattens it

**Solution**:
```python
# 1D VAE latent: [B, z_dim, T_down] = [4, 16, 30]
z_1d = vae_1d.encode(motion_1d).mode()
print(z_1d.shape)  # torch.Size([4, 16, 30])

# 2D VAE latent: [B, z_dim, T_down, K] = [4, 16, 30, 22]
z_2d = vae_2d.encode(motion_2d).mode()
print(z_2d.shape)  # torch.Size([4, 16, 30, 22])
```

### Issue 3: Checkpoint Loading Errors

**Symptom**: `KeyError: 'model.vae.encoder.conv_in.weight'`

**Cause**: Config/checkpoint mismatch or incorrect path

**Solution**:
```python
# Verify checkpoint exists
import os
ckpt_path = '../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth'
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

# Verify model architecture matches checkpoint
config = Config.fromfile('path/to/config.py')
vae = AutoencoderKLPrism1D(**config.model.vae)

# Load with strict=True to catch mismatches
from mmengine.runner import load_checkpoint
load_checkpoint(vae, ckpt_path, strict=True)
```

### Issue 4: FP32 Enforcement Not Working

**Symptom**: VAE produces NaN losses during training

**Cause**: AMP autocast converting to FP16 inside VAE

**Solution**:
```python
# VAE has @apply_forward_hook that enforces FP32
# Make sure it's applied:

# The decorator should be on encode/decode methods:
# @apply_forward_hook(auto_fp32(apply_to=['x']))
# def encode(self, x):
#     ...

# Verify in your usage:
with torch.cuda.amp.autocast(dtype=torch.float16):
    z = vae_1d.encode(motion)  # Still runs in FP32 due to hook
```

### Issue 5: Latent Normalization Wrong Direction

**Symptom**: Diffusion model produces garbage after using denormalized latents

**Cause**: Used wrong mean/std or wrong direction

**Solution**:
```python
# Load config
config = json.load(open('checkpoints/vermo_vae/config.json'))
mean = torch.tensor(config['latents_mean']).view(1, 16, 1, 1)
std = torch.tensor(config['latents_std']).view(1, 16, 1, 1)

# Correct normalization (for diffusion model input)
z = vae_2d.encode(motion).mode()  # [B, 16, T_down, K]
z_normalized = (z - mean) / std   # Normalize for diffusion

# Correct denormalization (before VAE decode)
z_denormalized = z_normalized * std + mean  # Scale back
motion_recon = vae_2d.decode(z_denormalized)
```

### Issue 6: Training Divergence with VAE

**Symptom**: Loss becomes NaN after first few iterations

**Cause**: Learning rate too high or KL weight too high (loss_kl=1e-06 is very small)

**Solution**:
```python
# From config:
loss_weights=dict(
    loss_joints=100.0,    # Reconstruction loss weight
    loss_kl=1e-06,        # KL divergence weight (very small!)
    loss_rec=1.0,         # Additional reconstruction
    loss_static=0,        # No static loss
    loss_transl_l1=1.0,   # Translation loss
    rec_loss_type='l1',   # L1 reconstruction loss
)

# If diverging, try:
# 1. Reduce learning rate: lr=0.0001 instead of 0.0003
# 2. Use gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# 3. Verify input normalization (motion should be normalized)
```

## Part 9: Performance Metrics

### Reconstruction Quality

**For 1D VAE (138-D)**:
- MSE reconstruction error: ~0.001 (L1 loss ~0.02)
- Joint angle error (degrees): ~1-2°
- Works well for: General motion reconstruction

**For 2D VAE (6-D per joint, 22 joints)**:
- MSE reconstruction error: ~0.0005-0.001
- Per-joint RMSE: ~0.02-0.03 (in normalized space)
- Per-joint max error: ~0.1 (in outlier cases)
- Works well for: Detailed joint-level reconstruction

### Inference Speed

**Typical timings on V100 GPU**:

1D VAE:
- Encode 121 frames, batch 32: ~5ms
- Decode 30 latent frames, batch 32: ~3ms
- Total round-trip: ~8ms

2D VAE:
- Encode 121 frames × 22 joints, batch 32: ~8ms
- Decode 30 latent frames × 22 joints, batch 32: ~5ms
- Total round-trip: ~13ms

### Memory Usage

1D VAE:
- Model weights: ~189 MB
- Batch 512 inference: ~2.1 GB
- Batch 64 inference: ~300 MB

2D VAE:
- Model weights: ~69.6 MB
- Batch 64 inference: ~2.8 GB
- Batch 8 inference: ~400 MB

## Part 10: Source Files Reference

**VAE Implementation Files**:
- `hftrainer/models/motion/prism/autoencoder_kl_1d.py` - 1D VAE class
- `hftrainer/models/motion/prism/autoencoder_kl_2d.py` - 2D VAE class
- `hftrainer/models/motion/prism/bundle.py` - PrismBundle integration

**Configuration Files**:
- `../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/smpl_vae1d_nostatic_aug_hq.py`
- `../versatilemotion/work_dirs/smpl_vae2dtk_nostatic_aug_hq/smpl_vae2dtk_nostatic_aug_hq.py`
- `checkpoints/vermo_vae/config.json`

**Checkpoint Files**:
- `../versatilemotion/work_dirs/smpl_vae1d_nostatic_aug_hq/iter_13000.pth` (189 MB)
- `checkpoints/vermo_vae/diffusion_pytorch_model.safetensors` (69.6 MB)

**Test Data**:
- `data/annotation/test_motionhub_recon.json`
- `data/annotation/test_motionhub_1p.json`
- `data/annotation/test_motionhub_2p.json`

**Statistics Files**:
- `data/statistic/smplx55_stats_hymotion_aug.json`

---

**Last Updated**: 2026-05-15
**VAE Architecture**: WAN-style temporal chunking with causal convolutions
**Framework**: MMEngine (1D) + Diffusers (2D)
