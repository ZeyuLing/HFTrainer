# HyMotion T2M Bundle — Deep Technical Analysis

**Complete reference for understanding the 6 core components of HyMotion T2M motion generation bundle.**

Based on source code analysis from:
- `/hftrainer/models/motion/hymotion_t2m/bundle.py` (328 lines)
- `/hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` (183 lines)
- `/hymotion_t2m_inference_guide.md` (466 lines)

---

## 1. Motion Normalization & Denormalization

### `normalize_motion()` — Implementation (Lines 311-322)

```python
def normalize_motion(self, motion: Tensor) -> Tensor:
    """Normalize motion using mean/std buffers.
    
    Dims with near-zero std (constant in training data) produce 0 after normalization.
    This matches official HY-Motion-1.0 behavior.
    """
    # Safe division: where std==0, output 0 (those dims are constant)
    safe_std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
    result = (motion - self.mean) / safe_std
    # Zero out dims where std was near-zero
    result = torch.where(self.std.unsqueeze(0) < 1e-3, torch.zeros_like(result), result)
    return result
```

**Key Strategy**: Two-pass approach
1. **Pass 1 (Safe Division)**: For dims with std ≥ 1e-3, compute `(motion - mean) / std`
2. **Pass 2 (Zero-Out Constants)**: For dims with std < 1e-3 (constant dimensions), force result to 0

**Why this matters**: Near-zero std dimensions are essentially constant in the training data (virtually no variation). Computing `(value - mean) / tiny_std` would produce huge outlier values that destabilize training. Setting them to 0 instead is mathematically correct and matches official HY-Motion-1.0 behavior.

**Data shapes**:
- `motion`: (B, T, 201) or (T, 201) — raw motion tensor
- `self.std`: (201,) — per-dimension standard deviation buffer
- `self.mean`: (201,) — per-dimension mean buffer
- **output**: (B, T, 201) or (T, 201) — normalized motion

### `denormalize_motion()` — Implementation (Lines 324-327)

```python
def denormalize_motion(self, motion: Tensor) -> Tensor:
    """Denormalize motion (matching official HY-Motion-1.0: zeros for near-zero std)."""
    std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
    return motion * std + self.mean
```

**Inverse operation** that reverses normalization:
- For near-zero std dims: `motion * 0 + mean` → just the mean value (constant)
- For normal dims: `motion * std + mean` → standard denormalization

**Consistency**: The `std = 0` replacement for near-zero dims ensures that denormalization is mathematically consistent with normalization (if you normalize then denormalize, you get back the mean value for constant dims).

### Mean/Std Loading — Implementation (Lines 131-146)

```python
def _load_mean_std(self, mean_std_dir: Optional[str]) -> None:
    if mean_std_dir is not None and osp.isdir(mean_std_dir):
        mean = torch.from_numpy(
            np.load(osp.join(mean_std_dir, 'Mean.npy'))
        ).float()
        std = torch.from_numpy(
            np.load(osp.join(mean_std_dir, 'Std.npy'))
        ).float()
        # Zero-out near-zero std dims (matching official HY-Motion-1.0)
        # These dims are effectively constant and should produce zero after normalization
        std = torch.where(std < 1e-3, torch.zeros_like(std), std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    else:
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
```

**Loading flow**:
1. **Check if directory exists** with Mean.npy and Std.npy files
2. **Load as numpy** → convert to PyTorch float32 tensors
3. **Zero near-zero std** before registering (applied once at load time)
4. **Register as buffers** (not parameters) so they move to correct device but don't get gradients

**File location**: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/stats/`

---

## 2. Motion Representation: 201-dim vs 135-dim Layout

### Complete 201-dim Layout (SMPL-22 skeleton)

**Dims [0–2]: Root Translation (3)**
- Absolute XYZ translation in world space

**Dims [3–8]: Root Rotation (6D, row-major)**
- Pelvis rotation (joint 0)

**Dims [9–134]: Body Rotations (126D = 21 joints × 6D)**
- Joints 1–21, each with 6D rotation representation
- Row-major layout: `[R00, R01, R10, R11, R20, R21]`

**Dims [135–200]: Local Joint Positions (66D = 22 joints × 3D)**
- **CURRENTLY UNUSED** in motion_135 extraction
- Reserved for future joint position features
- Includes all 22 joints (root + 21 body)

### 135-dim Motion Format (Standard)

**What's included**:
```
motion_135 = motion_201[:, :135]

Dims [0–2]:     Root translation (3)
Dims [3–134]:   Root + 21 body joint rotations (22 × 6 = 132)
─────────────────────────────────────────────────────
Total: 135 dimensions
```

**Why drop 66 dims?**
- Local joint positions (dims 135–200) are typically reconstructed via **forward kinematics** (FK) from SMPL body model
- Storing them explicitly is redundant — they can be perfectly reconstructed from rotations + translation
- This is the standard practice in motion synthesis: store minimal representation, derive derived quantities on-demand

### Layout Mapping (SMPL-22 Skeleton Structure)

```
Skeleton tree:
  0: Pelvis (root)
  ├─ 1: L_Hip     (Dims 9:15)
  │  ├─ 4: L_Knee    (Dims 27:33)
  │  └─ 7: L_Ankle   (Dims 45:51)
  ├─ 2: R_Hip     (Dims 15:21)
  │  ├─ 5: R_Knee    (Dims 33:39)
  │  └─ 8: R_Ankle   (Dims 51:57)
  └─ 3: Spine1    (Dims 21:27)
     └─ 6: Spine2    (Dims 39:45)
        └─ 9: Spine3    (Dims 57:63)
           ├─ 12: Neck   (Dims 75:81)
           ├─ 13: L_Collar (Dims 81:87)
           ├─ 14: R_Collar (Dims 87:93)
           └─ 15: Head   (Dims 93:99)
              ├─ 16: L_Shoulder (Dims 99:105)
              ├─ 17: R_Shoulder (Dims 105:111)
              ├─ 18: L_Elbow   (Dims 111:117)
              ├─ 19: R_Elbow   (Dims 117:123)
              ├─ 20: L_Wrist   (Dims 123:129)
              └─ 21: R_Wrist   (Dims 129:135)
```

---

## 3. Atomic Forward Functions

### `predict_flow()` — Single Transformer Forward Pass

**Signature** (Lines 225-255):
```python
def predict_flow(
    self,
    x_input: Tensor,              # (B, L, motion_dim) noisy motion
    ctxt_input: Tensor,           # (B, Lc, 4096) token-level text embeddings
    vtxt_input: Tensor,           # (B, 1, 768) sentence-level text embeddings
    timesteps: Tensor,            # (B,) diffusion timesteps
    x_mask_temporal: Optional[Tensor] = None,        # (B, L) boolean mask
    ctxt_mask_temporal: Optional[Tensor] = None,     # (B, Lc) boolean mask
) -> Tensor:
    """Single forward pass through the MMDiT transformer."""
    return self.motion_transformer(
        x=x_input,
        ctxt_input=ctxt_input,
        vtxt_input=vtxt_input,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
    )
```

**Return value**: (B, L, motion_dim) — model prediction (velocity or x1 depending on pred_type)

**Parameter details**:
- `x_input`: Noisy motion from ODE solver, shape (B, L, 201)
- `ctxt_input`: Token embeddings from text encoder (Qwen3 LLM), each token is 4096-dim
- `vtxt_input`: Sentence embedding from CLIP-L encoder, per-sample, 768-dim
- `timesteps`: Float in [0, 1] representing ODE time parameter (0=noise, 1=clean)
- `x_mask_temporal`: Boolean mask, (B, L), True=valid frames, False=padded frames
- `ctxt_mask_temporal`: Boolean mask, (B, Lc), True=valid tokens, False=padding

**No VACE conditioning**: Unlike HyMotion M2M, T2M does NOT use VACE (motion+context concatenation). Input is pure motion + text conditions.

### `encode_text()` — Text Encoding with Lazy Loading

**Signature** (Lines 169-191):
```python
@torch.no_grad()
def encode_text(self, text: List[str]) -> Dict[str, Tensor]:
    """Lazy-load text encoder and encode text to vtxt/ctxt.
    
    Returns dict with keys: text_vec_raw, text_ctxt_raw, text_ctxt_raw_length.
    """
    device = _get_module_device(self)
    if not hasattr(self, '_text_encoder') or self._text_encoder is None:
        if self._text_encoder_cfg is None:
            raise RuntimeError('No text_encoder config provided; cannot encode text.')
        from hftrainer.models.motion.hymotion_m2m.network.text_encoder import (
            HYTextModel,
        )
        cfg = deepcopy(self._text_encoder_cfg)
        cfg.pop('type', None)
        self._text_encoder = HYTextModel(**cfg)
    
    vtxt, ctxt, ctxt_len = self._text_encoder.encode(text)
    return {
        'text_vec_raw': vtxt.to(device),           # (B, 1, 768)
        'text_ctxt_raw': ctxt.to(device),          # (B, Lc, 4096)
        'text_ctxt_raw_length': ctxt_len.to(device),  # (B,)
    }
```

**Return values**:
- `text_vec_raw`: (B, 1, 768) — sentence-level embedding from CLIP-L (1 per sample)
- `text_ctxt_raw`: (B, Lc, 4096) — token-level embeddings from Qwen3 LLM (Lc tokens per sample)
- `text_ctxt_raw_length`: (B,) — actual token count for each sample (for masking padding)

**Lazy-loading mechanism**:
1. First call to `encode_text()` instantiates HYTextModel (Qwen3 ~8B + CLIP-L)
2. Loads from config: `type='HYTextModel', llm_type='qwen3', max_length_llm=128`
3. Stored in `self._text_encoder` for reuse
4. Subsequent calls reuse the same encoder instance (no reload overhead)

**Configuration required**:
```python
cfg.model.text_encoder = dict(
    type='HYTextModel',
    llm_type='qwen3',           # Qwen3 LLM for token embeddings
    max_length_llm=128,          # Max prompt length (tokens)
)
```

### `mask_text_cond()` — Classifier-Free Guidance Masking

**Signature** (Lines 193-223):
```python
def mask_text_cond(
    self,
    vtxt: Tensor,
    ctxt: Tensor,
    force_mask: bool = False,
    cond_mask_prob: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    """Apply classifier-free guidance masking to text conditions."""
```

**Three operating modes**:

1. **Force mask** (`force_mask=True`):
   ```python
   return (
       self.null_vtxt_feat.expand(*vtxt.shape),        # Replace with null
       self.null_ctxt_input.expand(*ctxt.shape),       # Replace with null
   )
   ```
   Always returns null embeddings, used for unconditional branch in CFG.

2. **Random masking during training** (`self.training and cond_mask_prob > 0`):
   ```python
   mask = torch.bernoulli(torch.ones(bs) * cond_mask_prob)  # (B,) binary
   # For each sample where mask=1, replace text with null
   vtxt = torch.where(mask_vtxt, self.null_vtxt_feat.expand_as(vtxt), vtxt)
   ctxt = torch.where(mask_ctxt, self.null_ctxt_input.expand_as(ctxt), ctxt)
   ```
   Randomly drops text for CFG training (teaches model to work without text).

3. **Keep all** (inference or training with `cond_mask_prob=0`):
   ```python
   return vtxt, ctxt  # Unchanged
   ```

**Why this matters for CFG**:
- CFG requires both text-conditioned AND unconditional outputs: `pred = uncond + scale * (text − uncond)`
- Training: randomly mask `cond_mask_prob` fraction of batch to get "unconditional" samples
- Inference: use `force_mask=True` for unconditional branch, normal text for conditioned branch

---

## 4. `decode_motion_from_latent()` — Latent to 3D Keypoints

**Signature** (Lines 257-309):

```python
def decode_motion_from_latent(self, latent: Tensor) -> Dict[str, Tensor]:
    """Denormalize latent and run FK to get 3D keypoints.
    
    Returns dict with keys: keypoints3d, rot6d, transl, latent_denorm.
    """
```

**Input**: `latent` (B, T, 201) — normalized motion from ODE solver

**Output dict**:
- `latent_denorm`: (B, T, 201) — denormalized motion (raw space)
- `keypoints3d`: (B, T, 22, 3) — 3D joint positions via FK, or None if body model unavailable
- `rot6d`: (B, T, 22, 6) — rotation representations (22 joints)
- `transl`: (B, T, 3) — root translation
- `root_rotations_mat`: (B, T, 3, 3) — root rotation as 3×3 matrix

**Implementation steps**:

1. **Denormalize**: `latent_denorm = latent * std + mean`
2. **Extract components**: Split into translation, root rot, body rotations
3. **Shape for FK**: Reshape 6D rotations into (B, T, 22, 6) grid
4. **Forward kinematics**: Use SmplxLiteJ24 body model to compute 3D positions
5. **Ground alignment**: Offset translation so lowest joint touches Y=0 (ground level)

**Ground alignment** (Lines 296-301):
```python
if k3d is not None:
    min_y = k3d[:, :, :, 1].min(dim=2)[0].min(dim=1)[0]  # Min Y across joints/frames
    transl[:, :, 1] = transl[:, :, 1] - min_y.unsqueeze(1)  # Shift Y
    k3d[:, :, :, 1] = k3d[:, :, :, 1] - min_y.unsqueeze(1).unsqueeze(1)
```

Ensures motion is foot-grounded (standing level ≈ Y=0).

---

## 5. Text Encoder Configuration & Initialization

### Text Encoder Setup (Bundle init, Lines 99-100)

```python
# ---- text encoder config (lazy-loaded) ----
self._text_encoder_cfg = deepcopy(text_encoder) if text_encoder else None
```

**Must be provided at bundle creation**:
```python
bundle = HyMotionT2MBundle(
    motion_transformer=dict(...),
    text_encoder=dict(
        type='HYTextModel',
        llm_type='qwen3',
        max_length_llm=128,
    ),
    ...
)
```

### Text Encoder Implementation (HYTextModel)

**Two-encoder architecture**:
1. **Qwen3 LLM** (~8B parameters) → token-level embeddings (4096-dim each)
2. **CLIP-L** → sentence-level embedding (768-dim)

**Input**: List of text prompts `["a person walks forward", "dancing motion", ...]`

**Output**:
```python
{
    'text_vec_raw': (B, 1, 768),           # Sentence embedding (CLIP)
    'text_ctxt_raw': (B, Lc, 4096),        # Token embeddings (Qwen3)
    'text_ctxt_raw_length': (B,),          # Token count per sample
}
```

**Configuration impact**:
- `max_length_llm=128`: Truncates prompts to 128 tokens (typical: 8-20 tokens)
- Higher max_length increases memory but allows longer descriptions

---

## 6. Full ODE Inference Pipeline

### Pipeline Implementation (Complete Flow)

**Stage 1: Initialization** (HyMotionT2MPipeline.__call__)

```python
device = next(self.bundle.motion_transformer.parameters()).device

# 1. Determine target sequence length
tgt_length = batch.get('tgt_length')  # e.g. [120] frames
B = len(tgt_length)
L = max(tgt_length)

# 2. Pad to at least 360 frames (training distribution)
TRAIN_FRAMES = 360
L_padded = max(L, TRAIN_FRAMES)

# 3. Create masks
tgt_padding_mask = _length_to_mask(torch.tensor(tgt_length), L_padded)  # (B, L_padded)
```

**Stage 2: Text Encoding**

Three options:
```python
# Option A: Online encoding from caption
if batch.get('caption'):
    text_feats = bundle.encode_text(batch['caption'])
    vtxt_input = text_feats['text_vec_raw']      # (B, 1, 768)
    ctxt_input = text_feats['text_ctxt_raw']     # (B, Lc, 4096)
    ctxt_length = text_feats['text_ctxt_raw_length']

# Option B: Pre-encoded text (faster for reuse)
elif batch.get('text_vec_raw'):
    vtxt_input = batch['text_vec_raw']
    ctxt_input = batch['text_ctxt_raw']
    ctxt_length = batch['text_ctxt_raw_length']

# Option C: Unconditional (null text for text-free generation)
else:
    vtxt_input = bundle.null_vtxt_feat.expand(B, 1, -1)
    ctxt_input = bundle.null_ctxt_input.expand(B, 1, -1)
    ctxt_length = torch.tensor([1], device=device).expand(B)
```

**Stage 3: Classifier-Free Guidance Setup**

```python
do_cfg = text_guidance_scale > 1.0

if do_cfg:
    # Prepare unconditional (null) text
    null_vtxt = bundle.null_vtxt_feat.expand_as(vtxt_input)
    
    # Stack [unconditional, conditional] for single forward pass
    vtxt_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)      # (2B, 1, 768)
    ctxt_cfg = torch.cat([ctxt_input, ctxt_input], dim=0)     # (2B, Lc, 4096)
    ctxt_mask_cfg = torch.cat([ctxt_mask_temporal, ctxt_mask_temporal], dim=0)
```

**Stage 4: ODE Integration**

```python
def fn(t_val: Tensor, x: Tensor) -> Tensor:
    """ODE function: given time t and state x, compute velocity dx/dt"""
    
    if do_cfg:
        # Double batch: unconditional + conditional
        x_double = torch.cat([x, x], dim=0)
        x_pred = bundle.predict_flow(
            x_input=x_double,
            ctxt_input=ctxt_cfg,
            vtxt_input=vtxt_cfg,
            timesteps=t_val.expand(2 * B),
            x_mask_temporal=tgt_padding_mask.repeat(2, 1),
            ctxt_mask_temporal=ctxt_mask_cfg,
        )
        
        # Split predictions and apply CFG
        pred_uncond, pred_text = x_pred.chunk(2, dim=0)
        x_pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)
    else:
        x_pred = bundle.predict_flow(...)
    
    # Convert to velocity if needed (pred_type adjustment)
    if bundle.pred_type == 'x1':
        t_eps = 0.05
        x_pred = (x_pred - x) / (1.0 - t_val).clamp_min(t_eps)
    
    return x_pred

# Initial noise
y0 = torch.randn(B, L_padded, 201, device=device)
t = torch.linspace(0, 1, num_steps + 1, device=device)

# ODE integration (50 steps default)
try:
    from torchdiffeq import odeint
    trajectory = odeint(fn, y0, t, method='euler')
except ImportError:
    # Fallback: simple Euler integration
    trajectory = [y0]
    for i in range(num_steps):
        t_val = torch.tensor(i * dt, device=device)
        v = fn(t_val, trajectory[-1])
        trajectory.append(trajectory[-1] + v * dt)
    trajectory = torch.stack(trajectory)

sampled = trajectory[-1]  # (B, L_padded, 201)
```

**Stage 5: Truncation & Decoding**

```python
# Truncate back to requested length (ODE was run on padded 360+ frames)
sampled = sampled[:, :L, :]  # (B, L, 201) where L = max(tgt_length)

# Decode to motion
result = bundle.decode_motion_from_latent(sampled)
result['latent'] = sampled  # Add raw latent
return result
```

### Output Format

```python
{
    'latent': (B, L, 201),               # Normalized ODE output
    'latent_denorm': (B, L, 201),        # Denormalized motion (raw space)
    'rot6d': (B, L, 22, 6),              # Rotation matrices (6D rep)
    'transl': (B, L, 3),                 # Root translation
    'keypoints3d': (B, L, 22, 3) or None,# 3D joint positions (if FK available)
    'root_rotations_mat': (B, L, 3, 3),  # Root rotation as 3×3 matrix
}
```

---

## Complete Minimal Example

```python
import torch
import numpy as np
from mmengine.config import Config
from tools.infer import load_bundle_from_checkpoint
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline

# 1. Load Model Bundle
config_path = "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py"
checkpoint_path = "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt"

cfg = Config.fromfile(config_path)

# CRITICAL: Inject text encoder config if missing
if not cfg.model.get('text_encoder'):
    cfg.model.text_encoder = dict(
        type='HYTextModel',
        llm_type='qwen3',
        max_length_llm=128,
    )

bundle = load_bundle_from_checkpoint(cfg, checkpoint_path, "cuda")
print(f"Bundle loaded: {bundle.motion_transformer.output_dim}-dim motion")
print(f"  Mean: {bundle.mean.shape}, Std: {bundle.std.shape}")

# 2. Create Pipeline
pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=50,               # ODE integration steps
    text_guidance_scale=5.0,    # CFG strength
)

# 3. Generate Motion
prompt = "a person walks forward"
batch = {
    "caption": [prompt],
    "tgt_length": [120],        # 120 frames @ 30fps = 4 seconds
}

with torch.no_grad():
    output = pipeline(batch)

# 4. Extract Results
motion_201 = output['latent_denorm'][0].cpu().numpy()  # (T, 201)
motion_135 = motion_201[:, :135]                       # (T, 135) - drop joint positions
root_transl = motion_135[:, :3]                        # (T, 3)
body_rot6d = motion_135[:, 3:]                         # (T, 132)

print(f"Generated: {motion_135.shape} @ 30fps = {motion_135.shape[0]/30:.1f}s")

# 5. Save
np.savez(
    "output.npz",
    motion_135=motion_135.astype(np.float32),
    motion_201=motion_201.astype(np.float32),
    fps=np.array(30),
)
print(f"Saved to output.npz")
```

---

## Performance & System Requirements

| Metric | Value |
|--------|-------|
| **Model Size** | 0.46B parameters |
| **Motion Dimensions** | 201 (full) / 135 (standard) |
| **GPU Required** | NVIDIA A100 / 80GB (A40 50GB / RTX 4090 24GB tight) |
| **Typical Inference Time** | 5-15 seconds (50 steps, 120-360 frames) |
| **Memory per Batch** | ~300 MB (B=1, T=360) |
| **Safe Batch Size** | B=4-8 on A100 |
| **ODE Steps** | 25 (3-5s, fast), 50 (5-10s, default), 100 (10-20s, best quality) |

---

## Common Issues & Fixes

### ❌ Text Encoder Config Empty
**Cause**: Training config has `text_encoder=dict()` (falsy)
**Fix**: Inject config before loading:
```python
cfg.model.text_encoder = dict(
    type='HYTextModel',
    llm_type='qwen3',
    max_length_llm=128,
)
```

### ❌ Output is All Zeros
**Cause**: Mean/Std not loaded correctly
**Fix**: Verify bundle initialization:
```python
assert bundle.mean.shape == torch.Size([201])
assert bundle.std.shape == torch.Size([201])
print(f"Mean range: [{bundle.mean.min():.3f}, {bundle.mean.max():.3f}]")
print(f"Std range: [{bundle.std.min():.3f}, {bundle.std.max():.3f}]")
```

### ⚠️ Motion is Jerky
**Solution**: Increase ODE steps
```python
pipeline.num_steps = 100  # vs default 50
```

---

## Key Concepts Summary

| Concept | Details |
|---------|---------|
| **Motion 201-dim** | 3D translation (3) + root rotation (6) + 21 body joints (126) + joint positions (66, unused) |
| **Motion 135-dim** | First 135 dims only: translation + rotations, no joint positions |
| **Normalization** | Per-dim mean subtraction + division by std, with special handling for near-zero std dims |
| **Text Encoding** | Qwen3 LLM (token-level, 4096-dim) + CLIP-L (sentence-level, 768-dim) |
| **Classifier-Free Guidance** | Stack unconditional (null) and conditional outputs, blend with `scale` parameter |
| **ODE Integration** | 50 steps from pure noise → clean motion, conditioned on text |
| **Forward Kinematics** | Convert (rotation, translation) → 3D joint positions using SMPL body model |

