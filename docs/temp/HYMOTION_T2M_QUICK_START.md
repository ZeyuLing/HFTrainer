# HyMotion T2M 1.0 — Quick Start Guide

## Files Summary

| Component | Path | Size |
|-----------|------|------|
| **Main Config** | `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` | 5.5 KB |
| **Checkpoint** | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` | **1.8 GB** |
| **Pipeline Code** | `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` | 6.3 KB |
| **Bundle Code** | `hftrainer/models/motion/hymotion_t2m/bundle.py` | ~500 lines |
| **Inference Script** | `scripts/misc/robot_sim/text_to_g1.py` | 447 lines |
| **Eval Script** | `scripts/eval/eval_m2m_v2_t2m.py` | 751 lines |

---

## 30-Second Setup

```python
import torch
from mmengine.config import Config
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint

# 1. Load config
cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')

# 2. Build and load model
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
sd = load_checkpoint('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt', map_location='cpu')
bundle.load_state_dict_selective(sd)
bundle.eval().to('cuda')

# 3. Create pipeline
pipeline = HyMotionT2MPipeline(bundle, num_steps=50, text_guidance_scale=5.0)

# 4. Run inference
with torch.no_grad():
    output = pipeline({'tgt_length': [360], 'caption': ['a person walks']})

# 5. Extract motion
motion = output['latent'].cpu().numpy()  # (1, 360, 201)
```

---

## Model Specs at a Glance

```
HunyuanMotionMMDiT (0.46B params)
├── Architecture: Transformer + Flow Matching (ODE)
├── Input: noise + text embeddings (NO VACE)
├── Output: motion (201-dim or 135-dim)
├── Motion length: 360 frames @ 30fps = 12 seconds
├── Text encoders:
│   ├── LLM: Qwen3-0.6B → 4096-dim context
│   └── Sentence: CLIP-L → 768-dim embeddings
└── Guidance: Classifier-free (CFG) with scale ∈ [1.0, 10.0]
```

---

## NPZ Output Format

```python
# Generated motions are saved as NPZ with:
{
    'motion_135': (T, 135),      # 6D rotation + translation
    'positions': (T, 22, 3),     # 3D joint positions (via FK)
    'translation': (T, 3),       # Root translation only
}
```

---

## Key Parameter Cheat Sheet

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `num_steps` | 50 | 10-100 | ODE solver steps (↑ = higher quality, slower) |
| `text_guidance_scale` | 5.0 | 1.0-10.0 | CFG strength (1.0 = no guidance, ignore text) |
| `tgt_length` | 360 | 1-360 | Motion duration in frames |
| `batch_size` | 1 | 1-∞ | Parallel samples (limited by VRAM) |

---

## Common Commands

### Single Prompt Inference
```bash
python scripts/misc/robot_sim/text_to_g1.py \
    --prompt "a person walks forward" \
    --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --output output/walk/ \
    --num-frames 360 --num-steps 50 --guidance-scale 5.0
```

### Batch Inference (Multi-GPU)
```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --gpus 0 1 2 3 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/t2m_eval/
```

### CFG Ablation (Quality vs. Speed)
```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --cfg-sweep 1.0 3.0 5.0 7.0 \
    --prompt-chunks 8 \
    --gpus 0 1 2 3 4 5 6 7
```

---

## File Locations Quick Map

```
project_root/
├── configs/hymotion_t2m/
│   ├── hymotion_t2m_201dim_046b.py    ← MAIN CONFIG
│   └── hymotion_t2m_smoke.py
│
├── checkpoints/HY-Motion-1.0/
│   └── HY-Motion-1.0-Lite/
│       ├── latest.ckpt                ← CHECKPOINT (1.8 GB)
│       └── config.yml
│
├── hftrainer/
│   ├── models/motion/hymotion_t2m/
│   │   ├── __init__.py
│   │   └── bundle.py                  ← Bundle class
│   └── pipelines/motion/
│       └── hymotion_t2m_pipeline.py    ← Inference pipeline
│
└── scripts/
    ├── eval/
    │   └── eval_m2m_v2_t2m.py          ← Batch eval
    └── misc/robot_sim/
        └── text_to_g1.py               ← Single inference
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| "No checkpoint found" | Missing 1.8 GB file | Verify `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` exists |
| "input_dim mismatch" | Config has wrong dims | Ensure `input_dim == output_dim == 201` |
| "VACE applied" | Using M2M config | Use `hymotion_t2m_201dim_046b.py` (NOT M2M) |
| "Text encoding fails" | Missing text encoder | Config auto-injects: `llm_type='qwen3_embedding'` |
| OOM (out of memory) | Batch size too large | Reduce batch or shorten `tgt_length` |

---

## Performance Benchmarks

```
Inference (single GPU, batch_size=1):
  num_steps=50:   ~1-2 seconds per motion ✓ (default)
  num_steps=100:  ~3-4 seconds per motion (higher quality)
  num_steps=20:   ~0.5 seconds per motion (faster, lower quality)

Memory:
  batch_size=1:   ~0.5 GB
  batch_size=4:   ~1.5 GB
  batch_size=8:   ~2.5 GB

Motion quality (metrics from eval):
  jitter:         ~0.041 ± 0.012
  bone_length_cv: < 0.01
  foot_contact:   ~60-70% of frames
```

---

## Text Guidance (CFG) Comparison

| Scale | Behavior | Use Case |
|-------|----------|----------|
| 1.0 | Ignores text, fully random | Diversity testing |
| 3.0 | Balanced, loose text follow | Default for variety |
| 5.0 | Strong text alignment | **DEFAULT** ✓ |
| 7.0 | Very strict text follow | High control needed |
| >10 | Over-constrained (artifacts) | Not recommended |

---

## Output Directories

```
work_dirs/
├── m2m_v2_t2m_eval/              ← Main eval results
│   ├── caption_global/npz/
│   ├── caption_local/npz/
│   ├── uncond_global/npz/
│   └── uncond_local/npz/
│
└── m2m_v2_t2m_eval_compare/      ← Model comparisons
    └── npz/
```

Each contains:
- `*.npz` files with generated motions
- `result.json` with aggregated metrics
- `shard_gpu*.json` with per-GPU results

---

## Links & References

- **Full Guide**: `HYMOTION_T2M_CONFIG_GUIDE.md`
- **Bundle Implementation**: `hftrainer/models/motion/hymotion_t2m/bundle.py`
- **Pipeline Implementation**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
- **Trainer**: `hftrainer/trainers/motion/hymotion_t2m_trainer.py`
- **Evaluation**: `scripts/eval/eval_m2m_v2_t2m.py` (line 57-112 for model definitions)

---

## Version Info

- **HyMotion T2M**: 1.0 (Lite)
- **Model Size**: 0.46B parameters
- **Motion Dim**: 201 (or 135 legacy)
- **Motion Type**: SMPL-22 joints
- **Framework**: PyTorch + torchdiffeq + MMEngine

