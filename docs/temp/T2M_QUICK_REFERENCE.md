# HyMotion T2M 1.0 — Quick Reference

## 🎯 Key Facts

| What | Value |
|------|-------|
| **Config** | `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` |
| **Checkpoint** | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` |
| **Model Size** | 0.46B params (HunyuanMotionMMDiT) |
| **Motion Dimension** | 201 = transl(3) + rot6d(132) + positions(66) |
| **Current Output** | 135-dim (transl + rot6d only, missing positions) |
| **Inference Steps** | 50 (ODE solver: Euler) |
| **Text Encoders** | Qwen3 (4096-dim) + CLIP-L (768-dim) |
| **Guidance** | Classifier-Free Guidance (CFG) at cfg_scale |

---

## 📊 NPZ Output Format

Every generated motion is saved as `.npz` with these keys:

```python
import numpy as np
data = np.load('sample.npz')

# Keys in NPZ
motion_135    # (T, 135) — translation + 6D rotations
positions     # (T, 22, 3) — world-space joint positions
translation   # (T, 3) — same as motion_135[:, :3]

# Example shapes for T=60 frames
print(data['motion_135'].shape)   # (60, 135)
print(data['positions'].shape)    # (60, 22, 3)
```

### Motion Breakdown (201-dim Target)

```
motion_201 = [
  transl           [0:3]     →  3 dims
  rot6d            [3:135]   →  132 dims (22 joints × 6)
  positions        [135:201] →  66 dims (22 joints × 3)
]
```

---

## 🚀 Running Inference

### Minimal Example (Single GPU)

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --gpus 0 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/t2m_test
```

### Multi-GPU CFG Ablation

```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --cfg-sweep 1.0 2.5 5.0 7.5 \
    --gpus 0 1 2 3 \
    --prompt-chunks 4 \
    --num-steps 50 \
    --output-dir work_dirs/t2m_ablation
```

**Output Structure**:
```
work_dirs/t2m_ablation/
├── caption_local_phase2/
│   ├── cfg1/
│   │   ├── result.json
│   │   └── npz/*.npz
│   ├── cfg2.5/
│   │   ├── result.json
│   │   └── npz/*.npz
│   └── cfg5/
│       ├── result.json
│       └── npz/*.npz
```

---

## 📈 Result JSON Structure

**File**: `work_dirs/t2m_test/caption_local_phase2/result.json`

```python
{
    "model": "caption_local_phase2",
    "checkpoint": "...",
    "rotation_space": "local",
    "has_caption": True,
    "num_prompts": 240,
    "cfg_scale": 5.0,
    "num_steps": 50,
    "speed_samples_per_min": 11.7,
    
    "aggregated": {
        "jitter_135": {"mean": 0.01, "std": 0.003, ...},
        "avg_velocity": {"mean": 0.48, ...},
        "bone_length_cv_mean": {"mean": 1e-06, ...},
        ...
    },
    
    "per_sample": [
        {
            "prompt_id": "00000001",
            "text": "A person walks forward.",
            "target_frames": 60,
            "actual_frames": 60,
            "metrics": {
                "jitter_135": 0.0102,
                "inference_time": 5.86,
                "avg_velocity": 0.45,
                ...
            }
        },
        ...
    ]
}
```

### Key Metrics

**Motion Quality**:
- `jitter_135` — Lower is better (smooth motion)
- `avg_velocity`, `avg_acceleration` — Physical realism
- `bone_length_cv_mean` — Should be ~0 (valid skeleton)

**Physical Validity**:
- `foot_ground_*` — Foot-ground contact metrics
- `arm_penetration_ratio` — Should be near 0 (no self-penetration)

**Inference Speed**:
- `inference_time` — Per-sample inference time (seconds)
- `speed_samples_per_min` — Throughput metric

---

## 🔍 Pipeline Code Flow

**File**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`

```python
Input batch → Text encoding → ODE integration → Output denormalization → NPZ save
```

### Steps:

1. **Encode text** (Qwen3 + CLIP-L) → embeddings
2. **Initialize noise** `y0 ~ N(0,1)` shape [B, T, 201]
3. **ODE integration** t=0→1 via Euler solver
4. **CFG scaling** (if cfg_scale > 1.0):
   - Run with + without text → combine predictions
5. **Denormalize** using checkpoint's mean/std
6. **Save NPZ** with motion_135, positions, translation

---

## ⚠️ Known Issues

### Current Limitation

The config declares 201 dims but the **data pipeline only outputs 135 dims** (missing position channel).

```python
# Config says:
_motion_dim = 201  # ✓

# But LoadSmplx55 gives:
output_dim = 3 + 132 = 135  # ✗ Missing 66-dim positions
```

**Impact**: 
- Model can load the 201-dim checkpoint ✓
- Inference works but doesn't train the position channel ✗
- Evaluation NPZ has `positions` (computed via FK) ✓

---

## 📦 Files & Paths Summary

| File/Directory | Purpose |
|---|---|
| `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` | Model + training config |
| `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` | Pre-trained T2M checkpoint (201-dim) |
| `scripts/eval/eval_m2m_v2_t2m.py` | Main inference script |
| `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` | Inference pipeline implementation |
| `hftrainer/pipelines/motion/differentiable_fk.py` | Forward kinematics (NPZ pos generation) |
| `data/eval/t2m/251125_yiran_subset.json` | 240-prompt test set |
| `work_dirs/m2m_v2_t2m_eval/` | Existing evaluation outputs |

---

## 💡 Tips

1. **Speed up inference**: Lower `--num-steps` (e.g., 25 instead of 50)
2. **Better quality**: Increase `--cfg-scale` (e.g., 7.5 instead of 5.0)
3. **Check qual**: Look at `jitter_*` and `arm_penetration_ratio` in result.json
4. **Batch inference**: Use `--prompt-chunks` to split work across GPUs
5. **Repro checkpoint**: Always use `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`

---

## 📚 References

- **Full Guide**: See `HYMOTION_T2M_GUIDE.md` for complete documentation
- **Config**: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`
- **Eval Script**: `scripts/eval/eval_m2m_v2_t2m.py` (line 25-35 for usage examples)

