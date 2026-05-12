# HyMotion T2M 1.0 — Complete Documentation

This directory contains comprehensive documentation for understanding and running HyMotion T2M (Text-to-Motion) 1.0 inference.

## 📚 Documentation Files

### 1. **T2M_QUICK_REFERENCE.md** ⭐ START HERE
   - **Best for**: Quick lookup, getting started
   - **Contains**: Key facts, basic commands, tips
   - **Time to read**: 5 minutes

### 2. **HYMOTION_T2M_GUIDE.md** 📖 COMPREHENSIVE
   - **Best for**: In-depth understanding
   - **Contains**: Full config details, inference pipeline, output format, limitations
   - **Time to read**: 20 minutes

### 3. **NPZ_FORMAT_DETAILS.md** 📊 TECHNICAL
   - **Best for**: Working with output data
   - **Contains**: NPZ structure, metrics derivation, code references, examples
   - **Time to read**: 15 minutes

---

## 🎯 Quick Start

### Run T2M Inference (30 seconds)

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Single GPU inference on default prompts
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --gpus 0 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/t2m_test
```

### Check Results

```bash
# View aggregated metrics
cat work_dirs/t2m_test/caption_local_phase2/result.json | head -50

# Load generated motion
python << 'PYTHON'
import numpy as np
data = np.load('work_dirs/t2m_test/caption_local_phase2/npz/00000001.npz')
print("NPZ keys:", list(data.keys()))
print("Motion shape:", data['motion_135'].shape)
print("Positions shape:", data['positions'].shape)
PYTHON
```

---

## 🔑 Key Takeaways

| Aspect | Value |
|--------|-------|
| **Model** | HunyuanMotionMMDiT (0.46B params) |
| **Motion Dims** | 201 (transl + rot6d + positions) |
| **Checkpoint** | Pre-trained HY-Motion-1.0-Lite |
| **Inference Script** | `scripts/eval/eval_m2m_v2_t2m.py` |
| **NPZ Output Format** | 3 keys: `motion_135`, `positions`, `translation` |
| **Text Encoders** | Qwen3 (4096-dim) + CLIP-L (768-dim) |
| **CFG Support** | Yes, classifier-free guidance |
| **Multi-GPU** | Yes, with prompt chunking & CFG sweep |

---

## 📁 File Organization

```
/apdcephfs/AILab_DHA/.../hf_trainer/
├── README_T2M.md                          ← You are here
├── T2M_QUICK_REFERENCE.md                 ← 5-min intro
├── HYMOTION_T2M_GUIDE.md                  ← Full guide
├── NPZ_FORMAT_DETAILS.md                  ← Data format specs
│
├── configs/hymotion_t2m/
│   ├── hymotion_t2m_201dim_046b.py        ← Main T2M config
│   └── hymotion_t2m_smoke.py              ← Small test config
│
├── checkpoints/HY-Motion-1.0/
│   └── HY-Motion-1.0-Lite/
│       └── latest.ckpt                    ← 201-dim checkpoint
│
├── scripts/eval/
│   ├── eval_m2m_v2_t2m.py                 ← Main eval script
│   ├── eval_m2m_all_tasks.py              ← Broader eval
│   └── ...other eval scripts
│
├── hftrainer/pipelines/motion/
│   ├── hymotion_t2m_pipeline.py           ← Inference pipeline
│   ├── differentiable_fk.py               ← FK for positions
│   └── hymotion_m2m_pipeline.py           ← M2M pipeline (related)
│
└── work_dirs/
    ├── m2m_v2_t2m_eval/                   ← Existing results
    ├── m2m_v2_t2m_eval_cfg_ablation*/     ← CFG ablations
    └── t2m_test/                          ← Your outputs here
```

---

## 🚀 Common Tasks

### Task 1: Basic Single-GPU Inference

```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --gpus 0 \
    --output-dir work_dirs/t2m_basic
```

**Output**: 
- `work_dirs/t2m_basic/caption_local_phase2/result.json` (metrics)
- `work_dirs/t2m_basic/caption_local_phase2/npz/*.npz` (motion files)

---

### Task 2: Multi-GPU CFG Ablation

```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --cfg-sweep 1.0 2.5 5.0 7.5 \
    --gpus 0 1 2 3 \
    --prompt-chunks 4 \
    --num-steps 50 \
    --output-dir work_dirs/t2m_cfgs
```

**Output Structure**:
```
work_dirs/t2m_cfgs/caption_local_phase2/
├── cfg1/
│   ├── result.json
│   └── npz/*.npz
├── cfg2.5/
│   ├── result.json
│   └── npz/*.npz
├── cfg5/
│   ├── result.json
│   └── npz/*.npz
└── cfg7.5/
    ├── result.json
    └── npz/*.npz
```

---

### Task 3: Speed Up Inference

```bash
# Reduce ODE steps from 50 to 25 (faster but lower quality)
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --num-steps 25 \
    --output-dir work_dirs/t2m_fast
```

---

### Task 4: Load & Analyze Generated Motion

```python
import numpy as np
import json

# Load NPZ
data = np.load('work_dirs/t2m_test/caption_local_phase2/npz/00000001.npz')
motion_135 = data['motion_135']      # (T, 135)
positions = data['positions']        # (T, 22, 3)

# Load metrics
with open('work_dirs/t2m_test/caption_local_phase2/result.json') as f:
    result = json.load(f)

# Get sample metrics
sample = result['per_sample'][0]
print(f"Prompt: {sample['text']}")
print(f"Jitter: {sample['metrics']['jitter_135']:.4f}")
print(f"Arm penetration: {sample['metrics']['arm_penetration_ratio']:.4f}")

# Aggregate metrics
agg = result['aggregated']
print(f"\nAverage metrics across {result['num_prompts']} samples:")
for key, stats in list(agg.items())[:5]:
    print(f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

---

## 💾 Data Pipeline Context

### Current Status

| Component | Status | Details |
|-----------|--------|---------|
| Model Checkpoint | ✅ OK | 201-dim, HY-Motion-1.0-Lite |
| Inference Pipeline | ✅ OK | Full text-to-motion with CFG |
| NPZ Output | ✅ OK | 3 keys (motion_135, positions, translation) |
| Data Training | ⚠️ MISMATCH | Only 135-dim, missing position channel |

### What Works Now

- ✅ Load 201-dim checkpoint
- ✅ Run inference with text conditioning
- ✅ Generate motion with CFG guidance
- ✅ Save NPZ with motion_135 + FK-computed positions
- ✅ Compute all evaluation metrics

### What Doesn't Work

- ❌ Training the position channel (data pipeline outputs 135-dim only)
- ❌ True end-to-end 201-dim training

### How to Fix

The data pipeline (`LoadSmplx55`) needs to output the position channel. See **HYMOTION_T2M_GUIDE.md** section "What Needs to be Fixed" for details.

---

## 📊 Example Result Metrics

Here's what you get in `result.json`:

```python
{
  "model": "caption_local_phase2",
  "num_prompts": 240,
  "cfg_scale": 5.0,
  "speed_samples_per_min": 11.7,
  
  "aggregated": {
    "jitter_135": {
      "mean": 0.010214,
      "std": 0.003501,
      "median": 0.009876,
      "min": 0.004521,
      "max": 0.024391
    },
    "avg_velocity": {...},
    "bone_length_cv_mean": {...},
    "arm_penetration_ratio": {...},
    ...  # 20+ metrics total
  },
  
  "per_sample": [
    {
      "prompt_id": "00000001",
      "text": "A person walks forward.",
      "metrics": {...}
    },
    ...  # 240 samples
  ]
}
```

---

## 🧠 Understanding the Architecture

### Inference Flow

```
Text Input
    ↓
[Qwen3 + CLIP-L encoders]
    ↓
Text Embeddings (4096-dim + 768-dim)
    ↓
[ODE Solver: t=0→1, 50 steps]
    ↓
[HunyuanMotionMMDiT Transformer]
    ↓
Denoised Motion (201-dim)
    ↓
[Denormalize + Extract 135-dim]
    ↓
[Forward Kinematics → Positions (22×3)]
    ↓
Save NPZ: motion_135, positions, translation
```

### Key Hyperparameters

| Param | Value | Impact |
|-------|-------|--------|
| `num_steps` | 50 | Higher = better quality, slower |
| `cfg_scale` | 5.0 | Higher = more text-aligned, less creative |
| `pred_type` | "velocity" | Predicts motion velocity field |
| `cond_mask_prob` | 0.1 | CFG dropout: 10% samples drop text |

---

## 🔗 References

- **Config File**: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`
- **Eval Script**: `scripts/eval/eval_m2m_v2_t2m.py`
- **Pipeline Code**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`
- **Checkpoint**: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
- **Test Data**: `data/eval/t2m/251125_yiran_subset.json` (240 prompts)

---

## ❓ FAQ

**Q: How long does inference take?**
A: ~6 seconds per motion (60 frames) on V100 with 50 ODE steps. Use `--num-steps 25` to halve time.

**Q: What's the difference between motion_135 and positions?**
A: `motion_135` is joint angles (rot6d) + translation. `positions` is 3D world-space coordinates computed via forward kinematics.

**Q: Can I change the motion length?**
A: Yes, the pipeline infers any length via `tgt_length` parameter.

**Q: What's CFG scale?**
A: Classifier-Free Guidance strength. Higher (>5) = more text-aligned. Lower (~1) = more creative.

**Q: Why is motion_201 missing from NPZ?**
A: The training data pipeline only outputs 135-dim. The position channel is computed post-hoc via FK.

**Q: Can I use a different checkpoint?**
A: The config is hard-linked to `latest.ckpt`. You can use `--ckpt-path` in the eval script to override.

---

## 📞 Support

For issues or questions:

1. **Check the docs**: Read T2M_QUICK_REFERENCE.md first
2. **Read the code**: See `scripts/eval/eval_m2m_v2_t2m.py` lines 25-35 for usage examples
3. **Check existing outputs**: Look at `work_dirs/m2m_v2_t2m_eval/*/result.json` for reference metrics

---

**Last Updated**: May 2026
**Checkpoint**: HY-Motion-1.0-Lite (201-dim)
**Model**: HunyuanMotionMMDiT (0.46B)

