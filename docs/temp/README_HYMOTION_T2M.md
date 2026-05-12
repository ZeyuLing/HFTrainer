# HyMotion T2M 1.0 - Model Configuration & Checkpoint Location

## 📍 Start Here

This directory contains **complete documentation** for running HyMotion T2M 1.0 text-to-motion inference. 

### Three Documentation Files Created:

1. **[HYMOTION_T2M_QUICK_START.md](./HYMOTION_T2M_QUICK_START.md)** ⚡
   - 30-second minimal code example
   - Quick parameter reference
   - Common commands
   - Troubleshooting table
   - **START HERE if you want to run inference NOW**

2. **[HYMOTION_T2M_CONFIG_GUIDE.md](./HYMOTION_T2M_CONFIG_GUIDE.md)** 📖
   - 467 lines of comprehensive reference
   - Detailed architecture explanation
   - Full inference pipeline walkthrough
   - Output format documentation
   - Performance metrics
   - **READ THIS for deep understanding**

3. **[HYMOTION_T2M_SUMMARY.txt](./HYMOTION_T2M_SUMMARY.txt)** 📋
   - Structured text format (no markdown)
   - Complete file listing with descriptions
   - All components explained systematically
   - **USE THIS as a detailed checklist**

---

## 🎯 TL;DR - The Essentials

### Config File
```
configs/hymotion_t2m/hymotion_t2m_201dim_046b.py
```

### Checkpoint (1.8 GB)
```
checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
```

### Quick Inference (5 lines of Python)
```python
from mmengine.config import Config
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint

cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
sd = load_checkpoint('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt', map_location='cpu')
bundle.load_state_dict_selective(sd)
bundle.eval().to('cuda')

pipeline = HyMotionT2MPipeline(bundle, num_steps=50, text_guidance_scale=5.0)

with torch.no_grad():
    output = pipeline({'tgt_length': [360], 'caption': ['a person walks']})
motion = output['latent'].cpu().numpy()  # (1, 360, 201)
```

### Command-Line Inference
```bash
python scripts/misc/robot_sim/text_to_g1.py \
    --prompt "a person walks forward" \
    --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --output output/walk/
```

---

## 📂 File Structure

```
project_root/
├── configs/hymotion_t2m/
│   ├── hymotion_t2m_201dim_046b.py    ← MAIN CONFIG (5.5 KB)
│   └── hymotion_t2m_smoke.py          (test config)
│
├── checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/
│   ├── latest.ckpt                    ← CHECKPOINT (1.8 GB)
│   └── config.yml
│
├── hftrainer/
│   ├── models/motion/hymotion_t2m/
│   │   └── bundle.py                  (HyMotionT2MBundle)
│   └── pipelines/motion/
│       └── hymotion_t2m_pipeline.py    (HyMotionT2MPipeline)
│
├── scripts/
│   ├── eval/eval_m2m_v2_t2m.py         (batch inference)
│   └── misc/robot_sim/text_to_g1.py    (single inference)
│
└── work_dirs/
    ├── m2m_v2_t2m_eval/                (existing eval results)
    ├── m2m_v2_t2m_eval_compare/
    └── ...
```

---

## 🚀 Quick Start

### Option 1: Single Prompt
```bash
cd /path/to/project
python scripts/misc/robot_sim/text_to_g1.py \
    --prompt "a person walks forward" \
    --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --output output/walk/
```

### Option 2: Batch Processing (Multi-GPU)
```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --gpus 0 1 2 3 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/t2m_eval/
```

### Option 3: Python API
```python
import torch
from mmengine.config import Config
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint

# Load
cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
sd = load_checkpoint('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt', map_location='cpu')
bundle.load_state_dict_selective(sd)
bundle.eval().to('cuda')

# Infer
pipeline = HyMotionT2MPipeline(bundle, num_steps=50, text_guidance_scale=5.0)
output = pipeline({'tgt_length': [360], 'caption': ['a person walks']})

# Extract
motion = output['latent'].cpu().numpy()  # (1, 360, 201)
positions_3d = output.get('keypoints3d')  # (1, 360, 22, 3)
```

---

## 📊 Model Specifications

| Property | Value |
|----------|-------|
| **Model Type** | HunyuanMotionMMDiT (Flow Matching) |
| **Parameters** | 460M (0.46B) |
| **Motion Dim** | 201 (SMPL 22-joint full representation) |
| **Motion Type** | SMPL with 22 joints |
| **Max Frames** | 360 @ 30fps = 12 seconds |
| **Text Encoders** | Qwen3 (LLM) + CLIP-L (sentence) |
| **Guidance** | Classifier-free guidance (CFG) |
| **Inference** | ODE-based (torchdiffeq.odeint, Euler method) |
| **Checkpoint Size** | 1.8 GB |
| **GPU Memory** | ~0.5 GB (batch_size=1), ~2.5 GB (batch_size=8) |

---

## 🎛️ Key Parameters

```python
# Inference parameters
num_steps = 50                  # ODE solver steps (higher = better quality, slower)
text_guidance_scale = 5.0       # CFG strength (1.0 = no guidance)
tgt_length = 360               # Motion frames
batch_size = 1                 # Parallel samples

# Quality levels
cfg_scale=1.0   → Random (ignores text)
cfg_scale=3.0   → Balanced
cfg_scale=5.0   → Strong (DEFAULT)
cfg_scale=7.0   → Very strict
cfg_scale>10    → Over-constrained (not recommended)

# Speed vs Quality
num_steps=20    → ~0.5s per motion (fast)
num_steps=50    → ~1-2s per motion (DEFAULT)
num_steps=100   → ~3-4s per motion (high quality)
```

---

## 📦 Output Format (NPZ)

Generated motions are saved as compressed NumPy files:

```python
data = np.load('motion.npz')

motion_135 = data['motion_135']    # (T, 135) - main output
                                    # [3 (trans) + 132 (6D rot for 22 joints)]
positions_3d = data['positions']   # (T, 22, 3) - 3D joint positions
translation = data['translation']  # (T, 3) - root motion
```

---

## 🔍 Existing Evaluation Results

Pre-computed T2M evaluations available in `work_dirs/`:

```
work_dirs/
├── m2m_v2_t2m_eval/                (main results)
│   ├── caption_global/npz/         (generated motions)
│   ├── caption_local/npz/
│   ├── uncond_global/npz/
│   └── uncond_local/npz/
│
├── m2m_v2_t2m_eval_cfg_ablation_2860_unpatched/
└── m2m_v2_t2m_eval_compare/
```

Each contains:
- `*.npz` files with generated motions
- `result.json` with aggregated metrics
- `shard_gpu*.json` with per-GPU results

---

## ✅ Verification Checklist

Before running inference, verify:

- [ ] Config exists: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`
- [ ] Checkpoint exists: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (1.8 GB)
- [ ] Pipeline code: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
- [ ] Bundle code: `hftrainer/models/motion/hymotion_t2m/bundle.py`
- [ ] Inference script: `scripts/misc/robot_sim/text_to_g1.py`
- [ ] Eval script: `scripts/eval/eval_m2m_v2_t2m.py`

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "No checkpoint found" | Verify 1.8 GB file at `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` |
| "input_dim mismatch" | Use `hymotion_t2m_201dim_046b.py` config (NOT M2M) |
| "VACE applied" | T2M uses `input_dim = motion_dim` (NOT `motion_dim * 4`) |
| "Text encoding fails" | Config auto-injects text encoder: `llm_type='qwen3_embedding'` |
| Out of memory | Reduce batch_size, tgt_length, or use num_steps=20 |

---

## 📚 Documentation Reference

| Document | Purpose | Length | Format |
|----------|---------|--------|--------|
| **HYMOTION_T2M_QUICK_START.md** | Get started quickly | ~6.7 KB | Markdown tables + code |
| **HYMOTION_T2M_CONFIG_GUIDE.md** | Complete reference | ~13 KB | Detailed sections |
| **HYMOTION_T2M_SUMMARY.txt** | Structured checklist | ~14 KB | Text format |

---

## 🎓 Deep Dive

### Pipeline Architecture

```
Text Input
    ↓
[LLM Encoder] → 4096-dim context embeddings
    ↓
[Sentence Encoder] → 768-dim sentence embeddings
    ↓
[ODE Solver] ← Classifier-Free Guidance
    ├─ Initial noise: y0 ~ N(0, I)
    ├─ Time schedule: t ∈ [0, 1]
    ├─ Transformer: predict_flow(x, t, text)
    └─ Final: x_clean = trajectory[-1]
    ↓
[Denormalization] → motion * std + mean
    ↓
[FK] → 3D joint positions (optional)
    ↓
Output: (T, 201) motion OR (T, 22, 3) positions
```

### Key Components

1. **HyMotionT2MBundle** (`hftrainer/models/motion/hymotion_t2m/bundle.py`)
   - Wraps HunyuanMotionMMDiT transformer
   - Manages text encoding, CFG, denormalization
   - Provides `predict_flow()` for ODE integration

2. **HyMotionT2MPipeline** (`hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`)
   - ODE-based inference using torchdiffeq
   - Handles batch processing and masking
   - Supports classifier-free guidance

3. **Text Encoders**
   - **LLM**: Qwen3-0.6B → 4096-dim context
   - **Sentence**: CLIP-L → 768-dim embeddings
   - Auto-loaded on first text encoding call

---

## 📞 Questions?

Refer to:
1. **Quick Start**: `HYMOTION_T2M_QUICK_START.md`
2. **Full Guide**: `HYMOTION_T2M_CONFIG_GUIDE.md`
3. **Checklist**: `HYMOTION_T2M_SUMMARY.txt`

Or check the implementation files:
- `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` (172 lines)
- `hftrainer/models/motion/hymotion_t2m/bundle.py` (~500 lines)
- `scripts/eval/eval_m2m_v2_t2m.py` (751 lines - comprehensive example)

---

**Status**: ✅ Complete - All HyMotion T2M 1.0 resources documented and indexed.

Generated: 2024-05-12
