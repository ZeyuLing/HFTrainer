# HYMotion T2M 1.0 Lite - Quick Reference Card

## Load Model (5 lines)
```python
from mmengine.config import Config
from tools.infer import load_bundle_from_checkpoint

cfg = Config.fromfile("configs/hymotion_t2m/hymotion_t2m_201dim_046b.py")
cfg.model.text_encoder = dict(type='HYTextModel', llm_type='qwen3', max_length_llm=128)
bundle = load_bundle_from_checkpoint(cfg, "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt", "cuda")
```

## Create Pipeline
```python
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
pipeline = HyMotionT2MPipeline(bundle, num_steps=50, text_guidance_scale=5.0)
```

## Generate Motion (4 lines)
```python
batch = {"caption": ["a person walks forward"], "tgt_length": [120]}  # 120 frames @ 30fps = 4s
with torch.no_grad():
    output = pipeline(batch)
motion_201 = output['latent_denorm'][0].cpu().numpy()  # (T, 201)
motion_135 = motion_201[:, :135]  # Extract first 135 dims for motion_135 format
```

## Save Motion
```python
np.savez("motion.npz", motion_135=motion_135.astype(np.float32), fps=np.array(30))
```

---

## API Cheat Sheet

### `HyMotionT2MPipeline.__call__(batch)`

**Input batch dict:**
| Key | Type | Required | Notes |
|-----|------|----------|-------|
| `caption` | List[str] | ✓ (or pre-encoded) | Text prompts to encode |
| `tgt_length` | List[int] | ✓ | Desired motion lengths (frames) |
| `text_vec_raw` | Tensor (B, 1, 768) | ✗ | Pre-encoded sentence (alt to caption) |
| `text_ctxt_raw` | Tensor (B, Lc, 4096) | ✗ | Pre-encoded tokens (with text_vec_raw) |
| `text_ctxt_raw_length` | Tensor (B,) | ✗ | Token counts (with above) |
| `motion_dim` | int | ✗ | Defaults to 201 |

**Output dict:**
| Key | Shape | Dtype | Notes |
|-----|-------|-------|-------|
| `latent` | (B, T, 201) | float32 | Normalized ODE output |
| `latent_denorm` | (B, T, 201) | float32 | Denormalized motion |
| `rot6d` | (B, T, 22, 6) | float32 | Per-joint rotations (row-major) |
| `transl` | (B, T, 3) | float32 | Root translation |
| `root_rotations_mat` | (B, T, 3, 3) | float32 | Root rotation matrices |
| `keypoints3d` | (B, T, 22, 3) or None | float32 | 3D joint positions (if SMPL loaded) |

---

## Motion Dimension Layout (135)

```
[0:3]      Root translation (X, Y, Z)
[3:9]      Root rotation (6D)
[9:15]     Joint 1 rotation (6D)
[15:21]    Joint 2 rotation (6D)
...
[129:135]  Joint 21 rotation (6D)
           
Total: 22 joints × 6D (root + 21 body) = 135 dims
```

---

## Configuration Options

```python
# In pipeline creation:
pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=50,              # ODE steps (25/50/100) — higher = better quality, slower
    text_guidance_scale=5.0,   # CFG scale (1.0 = none, 5.0 = default, 7.5+ = strong)
)

# In batch:
batch = {
    "caption": ["walk forward"],  # Text prompt
    "tgt_length": [120],          # 4 seconds @ 30fps
}

# Post-generation:
from scripts.embodied.batch_t2m_to_embodied import smooth_motion_135
motion_smooth = smooth_motion_135(motion_135)  # Markley quaternion + Savitzky-Golay
```

---

## Performance Benchmarks

| GPU | Motion Length | ODE Steps | Time | Memory |
|-----|---------------|-----------|------|--------|
| A100 | 120 frames | 50 | 6.1s | 300 MB |
| A100 | 360 frames | 50 | 15.5s | 500 MB |
| CPU | 120 frames | 50 | 300s+ | ❌ Not recommended |

---

## Common Patterns

### Batch inference (multiple prompts):
```python
prompts = ["walk forward", "jump", "dance"]
batch = {
    "caption": prompts,
    "tgt_length": [120, 150, 200],  # Different lengths
}
with torch.no_grad():
    output = pipeline(batch)
for i in range(len(prompts)):
    motion_135 = output['latent_denorm'][i, :, :135].cpu().numpy()
    print(f"{prompts[i]}: {motion_135.shape}")
```

### Pre-encode text (faster if reusing):
```python
text_feats = bundle.encode_text(["walk forward"])
batch = {
    "text_vec_raw": text_feats['text_vec_raw'],
    "text_ctxt_raw": text_feats['text_ctxt_raw'],
    "text_ctxt_raw_length": text_feats['text_ctxt_raw_length'],
    "tgt_length": [120],
}
output = pipeline(batch)
```

### Unconditional generation (random motion):
```python
batch = {"tgt_length": [120]}  # No caption
output = pipeline(batch)  # Uses null text embeddings
```

### Different quality settings:
```python
# Fast draft (25 steps): ~3s
pipeline.num_steps = 25
output = pipeline(batch)

# High quality (100 steps): ~15s
pipeline.num_steps = 100
output = pipeline(batch)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `RuntimeError: text_encoder_cfg is None` | Inject config: `cfg.model.text_encoder = dict(...)` |
| Output is NaN | Check mean/std loaded: `print(bundle.mean.shape)` should be `(201,)` |
| Jerky motion | Increase steps: `pipeline.num_steps = 100` OR apply smoothing |
| Very slow first call | Qwen3 LLM lazy-loads (~5s). Subsequent calls are faster. |
| GPU out of memory | Reduce batch size or motion length: `"tgt_length": [60]` |
| Model on CPU instead of GPU | `bundle.to("cuda")` — actually, check: `next(bundle.motion_transformer.parameters()).device` |

---

## Key Parameters Explained

**`num_steps`** — ODE integration steps
- Flow matching: gradually denoise from t=0 (noise) to t=1 (clean motion)
- 25 steps: ~3s, okay quality
- 50 steps: ~6s, good quality ✓
- 100 steps: ~15s, overkill
- Rule of thumb: +50ms per step (A100)

**`text_guidance_scale`** — Classifier-free guidance strength
- 1.0: No guidance (pure unconditional)
- 5.0: Default, follows text
- 7.5+: Very strong, may overfit to prompt
- Affects output diversity vs. prompt adherence

**`tgt_length`** — Motion length in frames
- Minimum: 120 (4s @ 30fps) recommended
- Maximum: 360+ (model was trained on 360-frame sequences)
- Pipeline auto-pads to ≥360 internally, then truncates

---

## Motion Output Format

**denormalized motion (201 dims):**
```
[0:135]    motion_135 (translation + 22×rot6d)
[135:201]  Local joint positions (22×3) [currently unused in output]
```

**How to extract motion_135:**
```python
motion_201 = output['latent_denorm'][0].cpu().numpy()  # (T, 201)
motion_135 = motion_201[:, :135]  # (T, 135)
```

**Manual denormalization (if needed):**
```python
latent = output['latent'][0].cpu().numpy()  # Normalized
motion_denorm = latent * bundle.std.cpu().numpy() + bundle.mean.cpu().numpy()
```

---

## File Locations

| Component | Path |
|-----------|------|
| Config | `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` |
| Checkpoint | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` |
| Mean/Std | `checkpoints/HY-Motion-1.0/stats/Mean.npy`, `Std.npy` |
| Pipeline code | `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` |
| Bundle code | `hftrainer/models/motion/hymotion_t2m/bundle.py` |
| Example usage | `scripts/embodied/batch_t2m_to_embodied.py` (lines 158–232) |

---

## One-Liner Examples

```python
# Load
cfg = Config.fromfile("configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"); cfg.model.text_encoder = dict(type='HYTextModel', llm_type='qwen3', max_length_llm=128); bundle = load_bundle_from_checkpoint(cfg, "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt", "cuda"); pipeline = HyMotionT2MPipeline(bundle, num_steps=50, text_guidance_scale=5.0)

# Generate
motion_135 = pipeline({"caption": ["walk"], "tgt_length": [120]})['latent_denorm'][0, :, :135].cpu().numpy()

# Save
np.savez("motion.npz", motion_135=motion_135.astype(np.float32), fps=np.array(30))
```

