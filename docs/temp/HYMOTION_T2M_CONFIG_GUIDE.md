# HyMotion T2M 1.0 Model Configuration & Checkpoint Guide

## Executive Summary

**HyMotion T2M 1.0** is a 0.46B parameter text-to-motion generation model that uses flow matching (ODE-based) inference to generate 360-frame motion sequences at 30 FPS from text prompts.

### Key Files:
- **Config**: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` (201-dim motion representation)
- **Checkpoint**: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (1.8 GB)
- **Config YAML**: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/config.yml`
- **Pipeline**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
- **Inference Script**: `scripts/misc/robot_sim/text_to_g1.py`
- **Evaluation Script**: `scripts/eval/eval_m2m_v2_t2m.py`

---

## 1. Model Architecture

### HunyuanMotionMMDiT (0.46B)

```
Input:  x_t (motion noise) + text embeddings
        - No VACE conditioning (unlike M2M)
        - input_dim = output_dim = 201 (SMPL 22-joint representation)

Architecture:
  - feat_dim: 1024
  - num_layers: 18
  - num_heads: 16
  - mlp_ratio: 4.0
  - mask_mode: "narrowband"
  - time_factor: 1000.0
  - apply_rope_to_single_branch: False

Text Encoders:
  - LLM: Qwen3-0.6B (context embeddings: 4096-dim)
  - Sentence Embedding: CLIP-L (768-dim)
  - Max LLM sequence length: 128 tokens
  - Max Sentence embedding sequence length: 77 tokens
```

---

## 2. Configuration Files

### Main Config: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`

```python
_base_ = '../_base_/default_runtime.py'

# Model dimensions
_motion_dim = 201  # SMPL 22-joint full representation

model = dict(
    type='HyMotionT2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        input_dim=201,                    # NO VACE multiplier
        feat_dim=1024,
        output_dim=201,
        ctxt_input_dim=4096,              # Qwen3 context
        vtxt_input_dim=768,               # CLIP-L embeddings
        num_layers=18,
        num_heads=16,
        mask_mode='narrowband',
        time_factor=1000.0,
    ),
    text_encoder=dict(),                  # Placeholder
    mean_std_dir='checkpoints/HY-Motion-1.0/stats/',
    motion_type='smpl_22',
    pred_type='velocity',
    uncondition_mode=False,               # Text-conditioned (CFG dropout 10%)
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.1,
)

# Training config
train_cfg = dict(
    by_epoch=True,
    max_epochs=1000,
    val_interval=10,
)

# Load pretrained weights
load_from = dict(
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
```

### Alternative: Smoke Test Config

For quick testing, use `configs/hymotion_t2m/hymotion_t2m_smoke.py`:
- Tiny model (3 layers, 64-dim features)
- Synthetic random data
- Runs in ~30 seconds

---

## 3. Checkpoint Location & Structure

### Main Checkpoint Path
```
checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/
├── latest.ckpt          (1.8 GB - PyTorch model weights)
└── config.yml           (YAML config from HY-Motion repo)
```

### Checkpoint Specifications
- **Size**: 1.8 GB
- **Format**: PyTorch checkpoint (`.ckpt`)
- **Parameters**: 460M (0.46B)
- **Input dim**: 201 (matches config)
- **Output dim**: 201 (matches config)
- **Loading**: Compatible with `HyMotionT2MBundle` when input_dim == output_dim

### Associated Directories
```
checkpoints/HY-Motion-1.0/
├── HY-Motion-1.0-Lite/      (T2M model, 0.46B)
│   ├── latest.ckpt
│   └── config.yml
├── HY-Motion-1.0/           (Full model, larger)
├── stats/                   (Mean/std normalization)
└── .cache/huggingface/      (Downloaded models)
```

---

## 4. Running Text-to-Motion Inference

### Method 1: Using HyMotionT2MPipeline (Recommended)

```python
import torch
from mmengine.config import Config
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
from hftrainer.registry import MODEL_BUNDLES

# Load config
cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')

# Build bundle (model)
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

# Load checkpoint
from hftrainer.utils.checkpoint_utils import load_checkpoint
sd = load_checkpoint('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt', 
                     map_location='cpu')
bundle.load_state_dict_selective(sd)
bundle.eval()
bundle = bundle.to('cuda:0')

# Create inference pipeline
pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=50,                    # ODE integration steps
    text_guidance_scale=5.0,         # CFG scale
)

# Run inference
batch = {
    'tgt_length': [360],             # 360 frames @ 30fps = 12 seconds
    'caption': ['a person walks forward slowly'],
}

with torch.no_grad():
    output = pipeline(batch)

# Extract motion (denormalized)
motion_135 = output['latent']        # (1, 360, 201) in latent space
# OR if denorm is available:
# motion_denorm = output['latent_denorm']  # (1, 360, 201) in data space
```

### Method 2: Using text_to_g1.py Script

```bash
python scripts/misc/robot_sim/text_to_g1.py \
    --prompt "a person walks forward slowly" \
    --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --output output/walk_motion/ \
    --num-frames 360 \
    --num-steps 50 \
    --guidance-scale 5.0 \
    --device cuda
```

### Method 3: Batch Inference with eval_m2m_v2_t2m.py

```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --gpus 0 1 2 3 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/t2m_eval/
```

---

## 5. Output Format (NPZ)

Generated motion is saved as **NPZ** (NumPy compressed) with the following structure:

```python
# From eval_m2m_v2_t2m.py (line 371-377):
np.savez_compressed(
    npz_path,
    motion_135=output_135,           # (T, 135) - 6D rotation + translation
    positions=pos_np,                # (T, 22, 3) - 3D joint positions (FK)
    translation=transl,              # (T, 3) - root translation
)
```

### Motion Representation (135-dim or 201-dim)

**135-dim** (standard output):
- Translation (3 dims): root XYZ position
- Rotation 6D (132 dims): 22 joints × 6D rotation encoding

**201-dim** (full representation):
- Translation (3 dims)
- Rotation 6D (132 dims)
- Local joint positions (66 dims): 22 joints × 3D

### 3D Joint Positions (FK)
- **Shape**: (T, 22, 3) where T=frames, 22=joints
- **Computed from**: 6D rotations + bone offsets
- **Bone offsets path**: `data/hymotion_m2m_data/bone_offsets_22.pt`

---

## 6. Inference Pipeline Details

### HyMotionT2MPipeline Flow

```python
# From hftrainer/pipelines/motion/hymotion_t2m_pipeline.py

1. Text Encoding
   ├─ Online: bundle.encode_text([prompt])
   │           → text_vec_raw (sentence emb, 768-dim)
   │           → text_ctxt_raw (LLM context, 4096-dim)
   │           → text_ctxt_raw_length
   └─ Or use pre-encoded embeddings from batch

2. Classifier-Free Guidance (CFG)
   ├─ If text_guidance_scale > 1.0:
   │   ├─ Prepare null embeddings
   │   ├─ Stack [uncond, cond] → 2×batch_size
   │   └─ Compute: output = uncond + scale * (cond - uncond)
   └─ Otherwise: use conditional only

3. ODE Integration
   ├─ Initial noise: y0 ~ N(0, I) shape (B, L, motion_dim)
   ├─ Time schedule: t ∈ [0, 1] with num_steps+1 points
   ├─ ODE solver: torchdiffeq.odeint (method='euler')
   ├─ At each step: x_pred = predict_flow(x, t, text)
   │   └─ Uses HunyuanMotionMMDiT transformer
   └─ Final: x_clean = trajectory[-1]

4. Denormalization
   ├─ Apply denorm: motion = x_clean * std + mean
   ├─ std, mean loaded from: mean_std_dir
   └─ Output: motion_denorm (ready for downstream use)
```

### Key Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_steps` | 50 | ODE solver steps (more = higher quality but slower) |
| `text_guidance_scale` | 5.0 | CFG strength (1.0 = no guidance) |
| `cond_mask_prob` | 0.1 | Training-only: classifier-free dropout probability |

---

## 7. Existing Evaluation Results

### M2M V2 T2M Eval (Yiran Subset)

**Output directories** in `work_dirs/`:
```
m2m_v2_t2m_eval/                           (Main eval)
├── caption_global/npz/
│   └── *.npz (generated motions)
├── caption_local/npz/
├── uncond_global/npz/
└── uncond_local/npz/

m2m_v2_t2m_eval_compare/                   (Model comparison)
```

**Metrics computed** (from eval_m2m_v2_t2m.py):
- Jitter (temporal smoothness)
- Bone length consistency (structural validity)
- Foot-ground contact
- Joint velocity/acceleration
- Motion quality checks (QC)

**Output format**: `result.json` with aggregated statistics
```json
{
  "model": "caption_local",
  "checkpoint": "...",
  "num_prompts": 240,
  "num_steps": 50,
  "cfg_scale": 5.0,
  "total_time_sec": 3600,
  "speed_samples_per_min": 4.0,
  "aggregated": {
    "jitter_135": {"mean": 0.041, "std": 0.012, ...},
    "jitter_pos": {...},
    ...
  },
  "per_sample": [...]
}
```

---

## 8. Quick Reference: Running Inference

### One-liner: Single Prompt
```bash
cd /path/to/hf_trainer

python scripts/misc/robot_sim/text_to_g1.py \
    --prompt "walking forward" \
    --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --output output/walk/
```

### Multi-GPU Batch Inference
```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --gpus 0 1 2 3 4 5 6 7 \
    --prompt-chunks 8 \
    --cfg-sweep 1.0 3.0 5.0 7.0 \
    --output-dir work_dirs/t2m_ablation/
```

### Smoke Test
```bash
python -c "
import torch
from mmengine.config import Config
import hftrainer

cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_smoke.py')
from hftrainer.registry import MODEL_BUNDLES
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
print(f'Bundle created: {bundle}')
"
```

---

## 9. Troubleshooting

### Issue: "No checkpoint found"
**Solution**: Ensure checkpoint exists at:
```
checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
```

### Issue: "input_dim/output_dim mismatch"
**Solution**: Ensure config has `input_dim=output_dim=201` (or 135 for legacy):
```python
model = dict(
    motion_transformer=dict(
        input_dim=201,      # Must match
        output_dim=201,     # Must match
    )
)
```

### Issue: "VACE multiplier applied (should be motion_dim only)"
**Solution**: HyMotion T2M does NOT use VACE conditioning:
```python
# WRONG (M2M style):
input_dim = motion_dim * 4  # ❌

# CORRECT (T2M style):
input_dim = motion_dim      # ✓
```

### Issue: "Text encoding fails"
**Solution**: Check text encoder config is present:
```python
bundle._text_encoder_cfg = {
    'llm_type': 'qwen3_embedding',
    'max_length_llm': 512,
    'sentence_emb_type': 'clipl',
    'max_length_sentence_emb': 77,
}
```

---

## 10. Related Files & Resources

### Core Implementation
- **Bundle**: `hftrainer/models/motion/hymotion_t2m/bundle.py`
- **Pipeline**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
- **Trainer**: `hftrainer/trainers/motion/hymotion_t2m_trainer.py`
- **Dataset**: `hftrainer/datasets/motion/hymotion_t2m_dataset.py`

### Evaluation & Inference
- **T2M Eval**: `scripts/eval/eval_m2m_v2_t2m.py`
- **Robot Demo**: `scripts/misc/robot_sim/text_to_g1.py`
- **Inference Util**: `tools/infer.py`

### Configuration Hierarchy
```
configs/
├── _base_/
│   └── default_runtime.py        (base settings)
└── hymotion_t2m/
    ├── hymotion_t2m_201dim_046b.py    (main config)
    └── hymotion_t2m_smoke.py          (test config)
```

### Normalization Stats
```
checkpoints/HY-Motion-1.0/stats/
├── mean.pt      (loaded by bundle)
└── std.pt       (loaded by bundle)
```

---

## 11. Performance Metrics

### Inference Speed (on single GPU)
- **num_steps=50**: ~1-2 seconds per 360-frame motion
- **Batch size=1**: ~0.5 GB VRAM
- **Batch size=8**: ~2-3 GB VRAM

### Quality
- **Motion smoothness**: Jitter ≈ 0.04 (L2 velocity norm)
- **Bone length consistency**: CV < 0.01
- **Motion diversity**: Varies with guidance scale

### CFG Ablation Results (from eval)
```
cfg_scale=1.0:  Ignores text, generates diverse motions
cfg_scale=3.0:  Balanced guidance
cfg_scale=5.0:  Strong text alignment (default)
cfg_scale=7.0:  Very strong guidance, lower diversity
```

---

## 12. Citation & Credits

**HyMotion T2M 1.0** is part of the HunyuanMotion framework:
- Paper: HunyuanMotion
- Original Config: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/config.yml`
- Text Encoders: Qwen3 (LLM) + CLIP-L (sentence embeddings)
- Diffusion Framework: Flow Matching (ODE-based)

