# HYMotion T2M 1.0 Lite - Programmatic Inference Guide

Complete reference for calling HYMotion T2M 1.0 Lite for inference **outside the web pipeline**.

---

## Quick Summary

| Property | Value |
|----------|-------|
| **Model Size** | 0.46B parameters (460M) |
| **Motion Output** | 201 dimensions (or 135 for motion_135) |
| **Text Encoder** | HYTextModel (Qwen3 LLM + CLIP-L) |
| **Sampling Method** | ODE integration (torchdiffeq or fallback Euler) |
| **ODE Steps** | 50 (default), configurable |
| **Guidance** | Classifier-free guidance (CFG) with scale 5.0 |
| **GPU Required** | Yes (CUDA recommended) |
| **Typical Inference Time** | ~5-15 seconds per motion (50 steps, 120 frames) |
| **Motion Length** | Minimum 360 frames (TRAIN_FRAMES) for model stability |
| **Output Format** | Normalized 201-dim latent + denormalized motion + FK keypoints |

---

## Architecture Overview

```
Text Prompt
    ↓
[HYTextModel: Qwen3 LLM + CLIP-L]
    ↓
[vtxt_input (B, 1, 768) + ctxt_input (B, Lc, 4096)]
    ↓
[HyMotionT2MPipeline with ODE solver]
    ├─ Random noise (B, L_padded, 201)
    ├─ ODE Integration: t=0 → t=1, 50 steps
    │  └─ At each step: HunyuanMotionMMDiT transformer predicts flow
    │     [NO VACE conditioning — just x_t + text conditions]
    └─ Final output: denormalized motion (B, L, 201)
    ↓
[Extract motion_135: first 135 dims]
    ↓
[Apply post-processing smoothing (Markley quaternion + Savitzky-Golay)]
    ↓
[Motion: (T, 135) or (T, 201)]
```

### Motion Dimension Layout

**201-dim representation** (full):
- Indices 0–2: Root translation (3 dims)
- Indices 3–8: Root rotation (6D, row-major)
- Indices 9–134: Body joints 1–21 rotation (21 × 6 = 126 dims)
- Indices 135–200: Local joint positions (22 × 3 = 66 dims) [**currently unused**]

**135-dim representation** (motion_135 — current standard):
- Indices 0–2: Root translation (3 dims)
- Indices 3–134: Root + body joints 0–21 rotation (22 × 6 = 132 dims)
- **Total: 135 dims**

This is extracted from the first 135 dims of the 201-dim output.

---

## Step-by-Step Inference API

### 1. **Load the Model Bundle**

```python
import torch
from mmengine.config import Config
from tools.infer import load_bundle_from_checkpoint

# Paths
config_path = "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"
checkpoint_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"
device = "cuda"  # or "cpu" (very slow)

# Load config
cfg = Config.fromfile(config_path)

# IMPORTANT: Inject text_encoder config if missing
# The training config has text_encoder=dict() which is falsy.
# You MUST provide the encoder config for inference.
if not cfg.model.get('text_encoder'):
    cfg.model.text_encoder = dict(
        type='HYTextModel',
        llm_type='qwen3',           # Qwen3 LLM for token-level embeddings
        max_length_llm=128,          # Maximum prompt length
    )

# Load bundle
bundle = load_bundle_from_checkpoint(cfg, checkpoint_path, device)

print(f"Bundle loaded on {device}")
print(f"  Motion transformer output_dim: {bundle.motion_transformer.output_dim}")
print(f"  Null text embeddings: vtxt={bundle.null_vtxt_feat.shape}, ctxt={bundle.null_ctxt_input.shape}")
```

**What gets loaded:**
- `bundle.motion_transformer`: HunyuanMotionMMDiT (0.46B params)
- `bundle.null_vtxt_feat`: (1, 1, 768) — null sentence embedding for CFG
- `bundle.null_ctxt_input`: (1, 1, 4096) — null token embedding for CFG
- `bundle.mean`, `bundle.std`: (201,) — normalization stats
- `bundle._text_encoder_cfg`: config for HYTextModel (lazy-loaded on first call)

---

### 2. **Create the Pipeline**

```python
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=50,                    # ODE integration steps (default: 50)
    text_guidance_scale=5.0,         # CFG scale (default: 5.0)
)

print(f"Pipeline ready")
print(f"  ODE steps: {pipeline.num_steps}")
print(f"  CFG scale: {pipeline.text_guidance_scale}")
```

**Parameters:**
- `num_steps`: Number of ODE steps. Higher = better quality but slower.
  - 25 steps: ~3–5s inference, moderate quality
  - 50 steps: ~5–10s inference, good quality (default)
  - 100 steps: ~10–20s inference, best quality (overkill)
- `text_guidance_scale`: CFG strength. 1.0 = no guidance, 5.0 = default, 7.5+ = very strong.

---

### 3. **Prepare Input Batch**

```python
# Option A: Text prompt (online encoding)
batch = {
    "caption": ["a person walks forward"],  # List of prompts
    "tgt_length": [120],                     # Motion length in frames (30 FPS → 4 seconds)
}

# Option B: Pre-encoded text (faster if reusing same text)
text_feats = bundle.encode_text(["a person walks forward"])
batch = {
    "text_vec_raw": text_feats['text_vec_raw'],           # (B, 1, 768)
    "text_ctxt_raw": text_feats['text_ctxt_raw'],         # (B, Lc, 4096)
    "text_ctxt_raw_length": text_feats['text_ctxt_raw_length'],  # (B,)
    "tgt_length": [120],
}

# Option C: Unconditional (null text)
batch = {
    "tgt_length": [120],
    # No caption/text_vec_raw/text_ctxt_raw → uses null embeddings
}
```

**Batch dict keys:**
- `caption` (optional): List[str] — text prompts to encode online
- `text_vec_raw` (optional): (B, 1, Dv) — pre-encoded sentence embeddings [alternative to caption]
- `text_ctxt_raw` (optional): (B, Lc, Dc) — token-level embeddings [with caption or text_vec_raw]
- `text_ctxt_raw_length` (optional): (B,) — actual token count per sample [with above]
- `tgt_length`: List[int] — desired motion lengths in frames (or torch.Tensor(B,))
- `motion_dim` (optional): int — inferred from bundle if not given

**Important:** The model pads all sequences to at least 360 frames (TRAIN_FRAMES) because it was trained on 360-frame sequences. Shorter sequences produce different attention patterns. The pipeline will:
1. Pad noise to 360 frames
2. Run ODE on full 360 frames
3. Truncate output back to requested length

---

### 4. **Run Inference**

```python
import torch

with torch.no_grad():
    output = pipeline(batch)
```

**Output dict keys:**
- `latent`: (B, T, 201) — normalized latent motion (raw ODE output)
- `latent_denorm`: (B, T, 201) — denormalized motion
- `rot6d`: (B, T, 22, 6) — rotation representations
- `transl`: (B, T, 3) — root translation
- `keypoints3d`: (B, T, 22, 3) or None — 3D joint positions (requires SMPL body model)
- `root_rotations_mat`: (B, T, 3, 3) — root rotation matrices

---

### 5. **Extract Motion Representations**

```python
import numpy as np

# Get denormalized motion
latent_denorm = output['latent_denorm']  # (B, T, 201)
if isinstance(latent_denorm, torch.Tensor):
    motion_201 = latent_denorm[0].cpu().numpy()  # (T, 201)
else:
    motion_201 = latent_denorm[0]

print(f"Motion 201-dim shape: {motion_201.shape}")

# Extract motion_135 (first 135 dims)
motion_135 = motion_201[:, :135]  # (T, 135)
print(f"Motion 135-dim shape: {motion_135.shape}")

# Extract root translation and body rotations separately
root_transl = motion_135[:, :3]      # (T, 3)
body_rot6d = motion_135[:, 3:]       # (T, 132) = 22 joints × 6 dims
print(f"Root translation: {root_transl.shape}")
print(f"Body rotations: {body_rot6d.shape}")
```

---

### 6. **Optional: Apply Post-Processing Smoothing**

The official HY-Motion-1.0 pipeline applies **Markley quaternion smoothing** (mathematically correct for rotations) + Savitzky-Goyal smoothing for translations:

```python
from scripts.embodied.batch_t2m_to_embodied import smooth_motion_135

motion_135_smooth = smooth_motion_135(motion_135)
print(f"Smoothed motion 135-dim: {motion_135_smooth.shape}")
```

**What smoothing does:**
- **Rotations:** Convert rot6d → rotation matrix → quaternion → Markley weighted average (9-tap Gaussian kernel, σ=1.0) → quaternion → rot6d
- **Translations:** Savitzky-Golay filter (window=11, polyorder=5)
- Applied to all 22 body joints

---

### 7. **Save/Use the Motion**

```python
# Save as NPZ (compatible with downstream tools)
output_path = "motion_output.npz"
np.savez(
    output_path,
    motion_135=motion_135.astype(np.float32),
    fps=np.array(30),
)
print(f"Saved: {output_path}")

# Or save raw motion_201
np.savez(
    "motion_201.npz",
    motion_201=motion_201.astype(np.float32),
    fps=np.array(30),
)
```

---

## Complete Minimal Example

```python
import torch
import numpy as np
from mmengine.config import Config
from tools.infer import load_bundle_from_checkpoint
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

# 1. Load
config_path = "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"
checkpoint_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"

cfg = Config.fromfile(config_path)
if not cfg.model.get('text_encoder'):
    cfg.model.text_encoder = dict(
        type='HYTextModel',
        llm_type='qwen3',
        max_length_llm=128,
    )

bundle = load_bundle_from_checkpoint(cfg, checkpoint_path, "cuda")
pipeline = HyMotionT2MPipeline(bundle, num_steps=50, text_guidance_scale=5.0)

# 2. Generate
prompt = "a person walks forward"
batch = {"caption": [prompt], "tgt_length": [120]}

with torch.no_grad():
    output = pipeline(batch)

# 3. Extract
motion_201 = output['latent_denorm'][0].cpu().numpy()  # (T, 201)
motion_135 = motion_201[:, :135]                       # (T, 135)

print(f"Generated motion: {motion_135.shape} @ 30 FPS = {motion_135.shape[0] / 30:.1f}s")

# 4. Save
np.savez("output.npz", motion_135=motion_135.astype(np.float32), fps=np.array(30))
```

---

## Performance & Diagnostics

### Inference Time

**Hardware:** NVIDIA A100 GPU

| Frames | Steps | Time (s) | FPS (realtime ratio) |
|--------|-------|----------|----------------------|
| 120    | 25    | 3.2      | 0.60× |
| 120    | 50    | 6.1      | 0.33× |
| 240    | 50    | 10.2     | 0.39× |
| 360    | 50    | 15.5     | 0.39× |

**CPU inference:** 50–100× slower, not recommended.

### Memory Usage

- **Bundle + pipeline:** ~2.5 GB GPU memory (model + buffers)
- **Per batch (B=1, T=360, motion_dim=201):**
  - Input noise: 28 MB
  - Intermediate ODE tensors: 100–200 MB per step
  - **Total batch:** ~300 MB GPU memory
  - Safe for batch inference B=4–8 on A100 (40 GB)

### Checking Model Status

```python
# Verify model is on GPU
device = next(bundle.motion_transformer.parameters()).device
print(f"Model device: {device}")

# Check null embeddings loaded
print(f"Null vtxt: {bundle.null_vtxt_feat.data[:, :, :5]}")  # Should be non-zero after checkpoint load
print(f"Null ctxt: {bundle.null_ctxt_input.data[:, :, :5]}")

# Verify normalization stats
print(f"Mean shape: {bundle.mean.shape}, std shape: {bundle.std.shape}")
print(f"Mean range: [{bundle.mean.min():.4f}, {bundle.mean.max():.4f}]")
print(f"Std range: [{bundle.std.min():.4f}, {bundle.std.max():.4f}]")
```

---

## Text Encoding Details

```python
# Encode text manually (if reusing same prompt multiple times)
text_feats = bundle.encode_text(["a person walks forward", "a person dances"])

print(f"Text features:")
print(f"  vtxt_input: {text_feats['text_vec_raw'].shape}")        # (B, 1, 768)
print(f"  ctxt_input: {text_feats['text_ctxt_raw'].shape}")       # (B, Lc, 4096)
print(f"  ctxt_length: {text_feats['text_ctxt_raw_length'].shape}") # (B,)
print(f"  Token counts: {text_feats['text_ctxt_raw_length'].tolist()}")
```

**Text encoder (HYTextModel):**
- `llm_type='qwen3'`: Qwen3 LLM produces token-level embeddings (4096 dims)
- `max_length_llm=128`: Maximum 128 tokens; longer prompts are truncated
- Output:
  - `vtxt`: Sentence-level embedding from LLM special token (1, 768)
  - `ctxt`: Token-level embeddings from all tokens (Lc, 4096)
  - `ctxt_len`: Actual token count per sample

---

## Common Pitfalls & Solutions

### ❌ Error: "text_encoder config empty"
**Cause:** Training config has `text_encoder=dict()` (falsy), so encoder isn't initialized.
**Fix:** Inject config before loading:
```python
if not cfg.model.get('text_encoder'):
    cfg.model.text_encoder = dict(
        type='HYTextModel',
        llm_type='qwen3',
        max_length_llm=128,
    )
```

### ❌ Error: "Expected input_dim=201 but got 135"
**Cause:** Config model.motion_transformer.input_dim doesn't match.
**Fix:** Verify config:
```python
print(cfg.model.motion_transformer.input_dim)  # Should be 201
print(cfg.model.motion_transformer.output_dim) # Should be 201
```

### ❌ Output is all zeros or NaN
**Cause:** Mean/std files not found, or model weights not loaded correctly.
**Fix:** Check:
```python
# Verify mean/std loaded
assert bundle.mean.shape == torch.Size([201]), f"Got {bundle.mean.shape}"
assert bundle.std.shape == torch.Size([201]), f"Got {bundle.std.shape}"

# Verify checkpoint loaded (null embeddings should be non-zero)
print(f"Null vtxt has non-zero: {(bundle.null_vtxt_feat.abs().max() > 1e-6)}")
print(f"Null ctxt has non-zero: {(bundle.null_ctxt_input.abs().max() > 1e-6)}")
```

### ⚠️ Motion is jerky/has glitches
**Solution 1:** Increase ODE steps
```python
pipeline.num_steps = 100  # More integration steps
```

**Solution 2:** Apply post-processing smoothing
```python
from scripts.embodied.batch_t2m_to_embodied import smooth_motion_135
motion_135 = smooth_motion_135(motion_135)
```

### ⚠️ Text encoding is slow
**Cause:** First call lazy-loads Qwen3 LLM (~5s).
**Solution:** Pre-encode and cache if reusing prompts:
```python
text_cache = {}
def get_text_feats(prompt):
    if prompt not in text_cache:
        text_cache[prompt] = bundle.encode_text([prompt])
    return text_cache[prompt]

# Reuse without re-encoding
feats = get_text_feats("walk")
```

---

## File Structure Reference

```
hftrainer/
├── models/motion/hymotion_t2m/
│   ├── bundle.py                  ← HyMotionT2MBundle (model container)
│   └── network/
│       ├── motion_transformer.py  ← HunyuanMotionMMDiT (core)
│       └── text_encoder.py        ← HYTextModel (Qwen3 + CLIP-L)
├── pipelines/motion/
│   └── hymotion_t2m_pipeline.py   ← HyMotionT2MPipeline (inference loop)
├── configs/hymotion_t2m/
│   └── hymotion_t2m_201dim_046b.py ← Config (motion_dim=201)
└── checkpoints/HY-Motion-1.0/
    ├── HY-Motion-1.0-Lite/latest.ckpt ← Pretrained weights (0.46B)
    └── stats/
        ├── Mean.npy              ← Motion normalization mean
        └── Std.npy               ← Motion normalization std
```

---

## References

**Key files in repo:**
- Batch script example: `scripts/embodied/batch_t2m_to_embodied.py` (lines 158–195 for loading, lines 197–232 for inference)
- Pipeline implementation: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` (lines 43–182)
- Bundle implementation: `hftrainer/models/motion/hymotion_t2m/bundle.py` (full file)
- Config: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`

**Official HY-Motion documentation:**
- Model: HunyuanMotion 1.0 Lite (0.46B, flow matching)
- Text encoder: Qwen3 LLM + CLIP-L
- Motion: 22-joint SMPL skeleton (root + 21 body joints)

