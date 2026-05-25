# PRISM Codebase Summary: Architecture & Key Files

Generated: 2026-05-15 | PRISM TMM2026 Motion Generation Framework

## 1. Core Architecture Overview

```
PRISM Motion Generation Pipeline
├── Input: Text + Optional Pose Frames
├── Encoding:
│   ├── Text → T5-XXL Embeddings (4096-dim)
│   └── Motion → Joint-Factorized 2D Latent [T', 23, 16]
│       └── VAE: AutoencoderKLPrism2DTK
│           ├── Causal Temporal Convolutions
│           ├── Joint-Attention (spatial)
│           └── FK Supervision (forward kinematics)
│
├── Denoising: Flow-Matching Transformer
│   ├── Architecture: PrismTransformerMotionModel (1.4B params)
│   ├── Layers: 30 blocks with 2D RoPE
│   ├── Conditions:
│   │   ├── Text states (cross-attention)
│   │   ├── Per-token timestep (prefix frames at t=0)
│   │   └── Padding mask (sequence length aware)
│   │
│   └── Loss Computation: **Translation/Rotation Separation**
│       ├── Token 0 (translation): 1/23 → normalized independently
│       ├── Tokens 1-22 (rotations): 22/23 → normalized independently
│       └── Combined: w_t * loss_transl + (1-w_t) * loss_rot
│
└── Inference:
    ├── Decode latents via VAE
    ├── Convert to SMPL parameters
    └── Optional KAFS: Kinematic-Adaptive Flow Scheduling
        (Denoising schedule varies per joint based on skeletal depth)
```

## 2. File Organization

### Training Framework

| File | Size | Purpose |
|------|------|---------|
| `hftrainer/trainers/motion/prism_trainer.py` | 131 lines | **Loss computation & training step** |
| `hftrainer/trainers/motion/prism_mcm_trainer.py` | - | Motion-Condition-Motion variant |
| `hftrainer/trainers/base_trainer.py` | - | Base class (training loop) |

### Model Bundles

| File | Purpose |
|------|---------|
| `hftrainer/models/motion/prism_bundle.py` | **Main model bundle** (VAE + Transformer + Scheduler) |
| `hftrainer/models/vae/autoencoder_prism2dtk.py` | Joint-factorized VAE |
| `hftrainer/models/transformer/prism_transformer_motion.py` | DiT with 2D RoPE |

### Configurations

| File | Purpose |
|------|---------|
| `configs/prism/prism_1b_tp2m_1frame.py` | **Base config: 1-frame pose conditioning** (5173 lines) |
| `configs/prism/prism_1b_tp2m_multiframe.py` | **Fine-tuning: 1/5/9-frame conditioning** (14 lines, extends base) |
| `configs/prism/prism_debug_loss_split.py` | **Debug config: Verify loss separation** (177 lines) |
| `configs/prism/prism_mcm_motionhub*.py` | Motion-to-Motion with 16-64 V100 setups |

### Data Processing

| File | Purpose |
|------|---------|
| `hftrainer/datasets/motion/motionhub_text_dataset.py` | Text-annotated motion dataset |
| `hftrainer/pipelines/motion_transforms.py` | Data augmentation & preprocessing |

### Inference & Evaluation

| File | Purpose |
|------|---------|
| `hftrainer/pipelines/motion/prism_pipeline.py` | Inference pipeline |
| `hftrainer/evaluation/motion_evaluator.py` | Metrics: FID, R@K, MM-Dist |

## 3. Loss Computation Deep Dive

### Current Implementation (prism_trainer.py, lines 95-112)

**Problem**: Without separation, translation (1/23 ≈ 4%) gets overwhelmed by rotations (22/23 ≈ 96%)

**Solution**: Independent normalization + weighted combination

```python
# Dimensions: [B, C, T', J] where J=23
mse = F.mse_loss(model_pred, targets, reduction='none')

# Separate by joint dimension
mse_transl = mse[:, :, :, :1]    # Translation: [B, C, T', 1]
mse_rot = mse[:, :, :, 1:]       # Rotations:   [B, C, T', 22]

# Normalize independently
loss_transl = (mse_transl * mask).sum() / mask.sum()  # Scale-independent
loss_rot = (mse_rot * mask).sum() / mask.sum()        # Scale-independent

# Combine with tunable weight
w_t = translation_loss_weight  # Default: 0.5
loss = w_t * loss_transl + (1-w_t) * loss_rot
```

### Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `translation_loss_weight` | 0.5 | [0.0, 1.0] | Balance between translation/rotation gradients |
| `condition_num_frames` | [1] | [1, 5, 9] | Pose conditioning: 1-frame (T2M) or multi-frame (TP2M) |
| `frame_condition_rate` | 0.1 | [0.0, 1.0] | Probability of pose conditioning in batch |
| `prompt_drop_rate` | 0.1 | [0.0, 1.0] | Classifier-free guidance: drop rate |

## 4. Latent Space Design

### 2D Joint-Factorized Grid

```
Input Motion: [B, T, 135]
              └─ 22 SMPL joints × 6D rotation + 3D translation (per frame)

After VAE Encoding: [B, 16, T', 23]
                    └─ 16 channels (latent features)
                       └─ T' ≈ T/4 (temporal compression)
                          └─ 23 tokens:
                             ├─ [0]: Root translation (3D)
                             └─ [1:23]: Joint rotations (22 joints)
```

### Why 2D Factorization Matters

1. **Per-Joint KL Regularization**: Each joint has independent posterior
   - CV (coefficient of variation) improves: 0.064 → 0.014 (4.4×)
   
2. **Loss Supervision**: Enables separate translation/rotation control
   
3. **Kinematic-Aware Inference**: KAFS can assign different denoising rates per joint
   - Proximal joints (root, pelvis): Fast denoising (α=0.85)
   - Distal joints (wrists, feet): Slow denoising (α=1.15)

## 5. Per-Token Timestep Conditioning

### How It Works

```python
# Training: Randomly select F prefix frames
if batch_index % 10 < frame_condition_rate:
    # Pose conditioning: keep prefix frames noise-free
    prefix_frames = latents[:, :, :F, :]    # [B, C, F, 23]
    noisy_latents[:, :, :F, :] = prefix_frames  # t=0 (clean)
    
    # Generation frames: normal noising schedule
    noisy_latents[:, :, F:, :] += noise  # t~uniform(0, T_max)

# Loss: Only generation frames contribute
condition_mask[:, :, :F, :] = 0  # Exclude prefix from loss
loss = (mse * condition_mask).sum() / condition_mask.sum()
```

### Unified Training

A single model handles:
- **Text-to-Motion** (F=0, no conditioning)
- **Pose-Conditioned** (F∈{1,5,9}, prefix frames fixed)
- **Sequential Generation** (Autoregressive chaining via prefix)

## 6. Configuration Hierarchy

### Base Config (prism_1b_tp2m_1frame.py)

```
Runtime:
├── checkpoint_hook: interval=2000, max_keep=5
├── logger_hook: interval=1
└── ema_hook: (optional)

Model:
├── Transformer: 30 layers, 1.4B params, bf16
├── VAE: AutoencoderKLPrism2DTK (frozen, fp32)
├── Text Encoder: T5-XXL (frozen, bf16)
└── Scheduler: FlowMatchEulerDiscreteScheduler(shift=5.0)

Data:
├── batch_size: 2 (per GPU)
├── clip_len: 128 frames (~4.3 sec @ 30fps)
├── sequence_length: 256 (text tokens)
└── augmentation: 75% translation yaw rotation

Optimizer:
├── Adam W: lr=3e-4, betas=[0.9, 0.99]
├── Accumulation: 4 steps
└── Precision: mixed (bf16)

Training:
├── FSDP: Full-Shard, TRANSFORMER_BASED_WRAP
├── Epochs: 3 (on MotionHub ~200K sequences)
└── Checkpoints: Load from iter_11000 of text-to-motion pretraining
```

### Multi-Frame Fine-Tuning (prism_1b_tp2m_multiframe.py)

Simple extension:
```python
_base_ = './prism_1b_tp2m_1frame.py'

trainer = dict(
    condition_num_frames=[1, 5, 9],  # Override: support multi-frame
    frame_condition_rate=0.1,
)
```

### Debug Config (prism_debug_loss_split.py)

Minimal setup for quick verification:
```python
batch_size=2, iterations=50
load_from='checkpoint-iter_15000'
```

## 7. Training Data Pipeline

### MotionHub Dataset

| Stage | Operation | Output |
|-------|-----------|--------|
| Load | SMPL-X 55-joint sequence + text | [T, 55*3], caption |
| Transform | Rotation 6D + relative translation | [T, 22*3+3] |
| Normalize | Per-joint standardization | [T, 22*3+3] (norm'd) |
| Crop | Random crop or pad to 128 | [128, 22*3+3] |
| Package | Dict with {motion, num_frames, caption, fps} | Batch-ready |

### Data Augmentation

- **Translation**: 75% probability × (yaw∈[-180°, 180°], offset∈N(0, σ))
- **Cropping**: Random start position for each sample
- **Padding**: Replicate boundary frames for short sequences

## 8. Metrics & Evaluation

### Logged During Training

```python
return {
    'loss': combined,              # Main training signal
    'loss_flow': combined,         # Alias for flow-matching
    'loss_transl': translation_mse,# Translation component
    'loss_rot': rotation_mse,      # Rotation component
}
```

### Validation Metrics

| Metric | Definition | Lower is Better |
|--------|-----------|-----------------|
| FID | Fréchet Inception Distance (motion quality) | ✓ |
| R@K | Recall@K: diversity (% unique motions in top-K retrieval) | ✗ |
| MM-Dist | Multimodal distance (text-motion alignment) | ✓ |
| Diversity | Motion variation (reward model) | ✗ |

### Reported Results (Paper)

| Dataset | FID | R@3 | MM-Dist |
|---------|-----|-----|---------|
| HumanML3D | 0.027 | 0.689 | 0.321 |
| MotionHub | 0.055 | 0.687 | 0.418 |
| BABEL | - | 0.587 | - |

## 9. Deployment Considerations

### Inference Pipeline (prism_pipeline.py)

```python
pipeline = PrismPipeline(bundle)
motion = pipeline(
    prompts="a person walking",
    num_frames_per_segment=33,        # 1.1 sec @ 30fps
    num_inference_steps=10,           # Denoising steps
    guidance_scale=5.0,               # Classifier-free guidance
)
# Output: [1, 99, 134] (batch, frames, 22*6D + 3D)
```

### KAFS Inference Optimization

```python
# Kinematic-Adaptive Flow Scheduling
for joint_idx in range(23):
    depth = kinematic_tree_depth[joint_idx]
    alpha_j = 0.85 + (depth / max_depth) * 0.30
    t_j = global_timestep * alpha_j
    denoised[:, :, :, joint_idx] = denoise_at_t(t_j)
```

**Effect**: 10-15% quality improvement without retraining

## 10. Quick Start Guide

### Setup
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
```

### Train Text-to-Motion
```bash
accelerate launch --multi_gpu --num_processes 8 \
  tools/train.py configs/prism/prism_1b_tp2m_1frame.py \
  --auto-resume
```

### Fine-tune Multi-Frame
```bash
accelerate launch --multi_gpu --num_processes 8 \
  tools/train.py configs/prism/prism_1b_tp2m_multiframe.py \
  --auto-resume
```

### Debug Loss Split
```bash
accelerate launch --multi_gpu --num_processes 8 \
  tools/train.py configs/prism/prism_debug_loss_split.py
# Verify loss_transl and loss_rot both decreasing in logs
```

### Inference
```python
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline

bundle = load_model("checkpoints/prism_1b_tp2m")
pipeline = PrismPipeline(bundle)
motion = pipeline(prompts="a person jumping", num_inference_steps=10)
```

## 11. Key Technical Innovations

### Joint-Factorized Latent Space
- **Impact**: 2.5× FID improvement vs monolithic encoding (isolated)
- **Mechanism**: Per-joint KL regularization produces uniform latent statistics
- **Enables**: KAFS, independent loss weighting, streaming inference

### Per-Token Timestep Conditioning
- **Impact**: Unifies text-to-motion, pose-conditioned, and sequential generation
- **Mechanism**: Prefix frames at t=0 (clean), generation frames at t>0 (noisy)
- **Unique**: Enabled by 2D joint factorization (each token addressable)

### Kinematic-Adaptive Flow Scheduling (KAFS)
- **Impact**: 10-15% quality improvement (no retraining)
- **Mechanism**: Depth-dependent denoising (proximal fast, distal slow)
- **Why it works**: Proximal joints determine global trajectory; distal joints need refinement

### Translation/Rotation Loss Separation
- **Impact**: Balanced gradient flow (prevents 95% dilution)
- **Mechanism**: Independent normalization → weighted combination
- **Tuning**: `translation_loss_weight` parameter [0.0, 1.0]

## 12. Extension Points

### Easy Modifications

1. **Loss Weights**: Change `translation_loss_weight` in config
2. **Inference Steps**: Modify `num_inference_steps` in pipeline
3. **Guidance Scale**: Tune classifier-free guidance strength
4. **Sequence Length**: Adjust `clip_len` in data config

### Medium Complexity

1. Add per-joint loss weighting (Section 2 in modification guide)
2. Implement adaptive weight scheduling (ramp over training)
3. Add multi-scale loss supervision
4. Implement trajectory consistency loss

### High Complexity

1. Modify VAE architecture (change latent dimension, add joint attention)
2. Extend transformer (add body-part-specific heads)
3. Implement KAFS variants (joint-group scheduling)
4. Add physics-based loss regularization

## 13. Troubleshooting

### Problem: Loss not decreasing
- Check `translation_loss_weight` is in [0, 1]
- Verify masking is correct: `print(f"mask sum: {full_mask.sum()}")`
- Ensure learning rate is appropriate (3e-4 default)

### Problem: One loss component constant
- Likely mask dimension mismatch
- Add shape debugging: `print(f"mse: {mse.shape}, mask: {full_mask.shape}")`
- Verify `expand_as()` before multiplying

### Problem: Memory error on multi-GPU
- Reduce batch size: 2 → 1 per GPU
- Reduce sequence length: 128 → 64 frames
- Enable gradient accumulation in accelerator config

### Problem: NaN in loss
- Check padding mask isn't all zeros
- Verify `+ 1e-6` epsilon in normalization
- Ensure no tensor shapes conflict (especially after masking)

---

**Document Summary**:
- Architecture: Joint-factorized VAE + Flow-matching DiT + Per-token timesteps
- Key Innovation: Independent translation/rotation loss (lines 95-112 in trainer)
- Config Hierarchy: Base → Multi-frame → Debug setups
- Extension Strategy: Tunable parameters → modifications → architecture changes

Generated by: Claude Opus Code Analysis | PRISM TMM2026 Research
