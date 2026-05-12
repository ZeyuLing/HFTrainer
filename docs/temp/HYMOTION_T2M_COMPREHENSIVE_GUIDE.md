# HyMotion T2M 1.0 - Comprehensive Inference & Output Format Guide

## Executive Summary

**HyMotion T2M 1.0** is a 460M-parameter text-to-motion diffusion model that generates 201-dimensional SMPL motion sequences from text prompts. The model uses a **HunyuanMotionMMDiT** transformer with classifier-free guidance and ODE-based inference.

### Key Facts at a Glance
- **Config**: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`
- **Checkpoint**: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (1.8 GB)
- **Motion Output**: 201-dim per frame = [3 (translation) + 22×6 (6D rotations) + 22×3 (joint positions)]
- **NPZ Output Fields**: `motion_135`, `positions`, `translation`
- **Inference Framework**: ODE-based (torchdiffeq.odeint with Euler solver)
- **Guidance**: Classifier-free guidance with configurable scale (default 5.0)

---

## 1. Config File Analysis

### File Location
```
configs/hymotion_t2m/hymotion_t2m_201dim_046b.py
```

### Key Configuration Details

```python
# Motion representation
_motion_dim = 201  # [3 (transl) + 132 (6D rot for 22 joints) + 66 (pos for 22 joints)]

model = dict(
    type='HyMotionT2MBundle',
    motion_transformer=dict(
        type='HunyuanMotionMMDiT',
        trainable=True,
        input_dim=_motion_dim,         # 201 (NO VACE multiplier)
        feat_dim=1024,
        output_dim=_motion_dim,        # 201
        ctxt_input_dim=4096,           # Qwen3 LLM embeddings
        vtxt_input_dim=768,            # CLIP-L sentence embeddings
        num_layers=18,
        num_heads=16,
        mlp_ratio=4.0,
        mlp_act_type='gelu_tanh',
        norm_type='layer',
        qk_norm_type='rms',
        mask_mode='narrowband',
        apply_rope_to_single_branch=False,
        time_factor=1000.0,
    ),
    text_encoder=dict(),  # Placeholder; auto-injected at runtime
    mean_std_dir='checkpoints/HY-Motion-1.0/stats/',
    motion_type='smpl_22',
    pred_type='velocity',
    uncondition_mode=False,    # Text-conditioned
    noise_scheduler_cfg=dict(method='euler'),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.1,        # CFG: 10% of samples drop text
)

# Data pipeline
train_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='LoadCompatibleCaption', allow_none=False),
            dict(
                type='LoadSmplx55',
                key='motion',
                rot_type='rotation_6d',
                transl_type='rel',
                smpl_type='smpl_22',  # 22-joint SMPL model
            ),
            dict(
                type='RandomCropPadding',
                clip_len=360,  # 12 seconds @ 30fps
                pad_mode='replicate',
            ),
        ],
    ),
)

# Load pretrained checkpoint
load_from = dict(
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
```

### Important Notes from Config
- **201-dim output format**: Original HY-Motion-1.0-Lite checkpoint outputs full 201 dims
- **Data pipeline limitation**: Current `LoadSmplx55` outputs 135 dims (3 + 132), not full 201
- **No VACE**: Unlike M2M models, T2M uses `input_dim = motion_dim` (not multiplied by 4)
- **Text encoders injected at runtime**: Config auto-injects `llm_type='qwen3_embedding'` and `sentence_emb_type='clipl'`

---

## 2. Checkpoint Details

### Location
```
checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
```

### Specifications
- **Size**: 1.8 GB
- **Format**: PyTorch state dict
- **Parameters**: 460M (0.46B)
- **Architecture**: HunyuanMotionMMDiT
- **Motion dim**: 201 (matches model input_dim/output_dim)

### Loading Code
```python
from hftrainer.utils.checkpoint_utils import load_checkpoint
from hftrainer.registry import MODEL_BUNDLES
from mmengine.config import Config

cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

sd = load_checkpoint(
    'checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    map_location='cpu'
)
bundle.load_state_dict_selective(sd)
bundle.eval().to('cuda')
```

---

## 3. T2M Inference Pipeline

### Architecture Overview

```
Text Prompt
    ↓
[Encode Text]
├─ LLM Encoder (Qwen3) → 4096-dim context
└─ Sentence Encoder (CLIP-L) → 768-dim vector
    ↓
[Prepare CFG Guidance]
├─ Null context (unconditional)
├─ Compute guidance scale interpolation
    ↓
[ODE Integration]
├─ Initial noise: y₀ ~ N(0, I) with shape (1, 360, 201)
├─ Time schedule: t ∈ [0, 1], num_steps+1 points
├─ Solver: Euler method (step-by-step)
├─ At each step:
│   ├─ Compute predicted flow: bundle.predict_flow(...)
│   ├─ Apply CFG: x_cond - cfg_scale * (x_uncond - x_cond)
│   └─ Update trajectory
├─ Output: x_clean = trajectory[-1] with shape (1, 360, 201)
    ↓
[Denormalization]
├─ Load mean/std from checkpoints/HY-Motion-1.0/stats/
├─ denorm = latent * std + mean
    ↓
[Extract Components]
├─ translation (3): first 3 dims
├─ rot_6d (132): next 132 dims (22 joints × 6 per joint)
├─ positions (66): remaining 66 dims (22 joints × 3 per joint) [OPTIONAL]
    ↓
[FK (Forward Kinematics)]
├─ Convert 6D rotation to axis-angle
├─ Apply skeleton structure (bone_offsets_22.pt)
├─ Compute 3D joint positions
    ↓
Output: motion (201-dim) + positions (22×3) + metrics
```

### Eval Script: `scripts/eval/eval_m2m_v2_t2m.py`

This 751-line script implements parallel T2M inference across multiple GPUs. Key features:

#### Mode A: Per-Model Parallelism (Legacy)
```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local caption_global uncond_local uncond_global \
    --gpus 0 1 2 3 \
    --cfg-scale 5.0 \
    --num-steps 50 \
    --output-dir work_dirs/m2m_v2_t2m_eval/
```

- Each model runs on its own GPU
- One cfg-scale per model
- Output: `<out_dir>/<model>/npz/*.npz`

#### Mode B: CFG-Sweep Parallelism (Recommended)
```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --cfg-sweep 1.0 1.5 2.5 4.0 7.5 \
    --prompt-chunks 8 \
    --gpus 0 1 2 3 4 5 6 7 \
    --num-steps 50 \
    --output-dir work_dirs/m2m_v2_t2m_eval_ablation/
```

- Prompts split into N chunks across workers
- Same model loaded once per worker
- Each worker runs all cfg values sequentially
- Output: `<out_dir>/<model>/cfg{X}/npz/*.npz`
- **Benefit**: Amortizes model loading cost over all configs

---

## 4. NPZ Output Format

### What Gets Saved

The eval script saves compressed NumPy archives at:
```
work_dirs/m2m_v2_t2m_eval/<model>/npz/<prompt_id>.npz
```

### NPZ Structure

```python
import numpy as np

data = np.load('00001401.npz')
print(data.keys())  # ['motion_135', 'positions', 'translation']

# Field 1: Main motion representation
motion_135 = data['motion_135']       # shape: (T, 135)
# Components:
#   - [:, :3]      = translation (3)
#   - [:, 3:135]   = 6D rotations (132 = 22 joints × 6)

# Field 2: 3D joint keypoints (via FK)
positions = data['positions']         # shape: (T, 22, 3)
# - T: number of frames
# - 22: SMPL joints
# - 3: XYZ coordinates

# Field 3: Root translation channel
translation = data['translation']     # shape: (T, 3)
# Redundant copy of motion_135[:, :3]
```

### Actual Data Dimensions (From Sample)
```
Loading: work_dirs/m2m_v2_t2m_eval/caption_local/npz/00001401.npz

motion_135:
  Shape: (60, 135)
  Dtype: float32
  Range: [-0.8892, 1.1765]
  Components: [3 translation + 132 6D rotation]

positions:
  Shape: (60, 22, 3)
  Dtype: float32
  Range: [-0.4378, 1.3356]
  Interpretation: 60 frames, 22 joints, 3D coordinates

translation:
  Shape: (60, 3)
  Dtype: float32
  Range: [-0.0013, 1.1765]
  Note: Same as motion_135[:, :3]
```

### **Critical Discovery: NOT a 201-dim Field**

The NPZ does **NOT** contain a single `motion_201` field. Instead:
- **`motion_135`**: What gets stored (primary output)
- **`positions`**: Computed from FK (secondary)
- **`translation`**: Redundant copy of translation channel

The full 201-dim representation appears to exist only during inference (intermediate tensor in the pipeline), but the eval script only saves the denormalized 135-dim part + FK-computed positions.

---

## 5. Inference Code Flow (From eval_m2m_v2_t2m.py)

### Steps 296-330: Prepare Batch

```python
# For each prompt
text = prompt['text']
T = min(prompt['frames'], 360)  # actual motion length
D = 198  # ← This is the motion dimension used in input
T_PAD = 360

# Create zero tensors (motion will be generated from noise)
src_motion = torch.zeros(1, T_PAD, D, device=device)  # (1, 360, 198)
src_mask = torch.zeros(1, T_PAD, D, device=device)
src_mask[:, :T, :] = 1.0  # Mark valid frames

# Encode text
if model_info['has_caption'] and text:
    text_out = bundle.encode_text([text])
    batch['text_vec_raw'] = text_out['text_vec_raw']      # (1, N, 768)
    batch['text_ctxt_raw'] = text_out['text_ctxt_raw']    # (1, M, 4096)
    batch['text_ctxt_raw_length'] = text_out['text_ctxt_raw_length']

# Run pipeline
batch = {
    'src_motion': src_motion,
    'src_mask': src_mask,
    'src_length': [T],
    'tgt_length': [T],
    'text_vec_raw': ...,
    'text_ctxt_raw': ...,
}

with torch.no_grad():
    output = pipeline(batch)
```

### Steps 327-330: Extract Output

```python
sampled = output['latent']  # Raw tensor from ODE solver
output_denorm = bundle.denormalize_motion(sampled)[0].cpu()  # Denormalize
output_denorm = output_denorm[:T]  # Trim to actual length

# Extract 135-dim motion
output_135 = output_denorm[:, :135].numpy()  # (T, 135)
```

**Key insight**: `output_denorm` has shape `(T, ≥135)` but only the first 135 dims are extracted and saved.

### Steps 371-377: Save NPZ

```python
npz_path = os.path.join(npz_dir, f'{prompt["id"]}.npz')
np.savez_compressed(
    npz_path,
    motion_135=output_135,          # (T, 135): primary output
    positions=pos_np,               # (T, 22, 3): computed from FK
    translation=transl,             # (T, 3): root motion
)
```

---

## 6. Motion135 to Full 201 Mapping

### How 135 Dim Relates to 201 Dim

Based on config and eval script:

```
Full 201-dim representation (theoretical):
├─ [0:3]       Translation (3)
├─ [3:135]     6D Rotations (132) = 22 joints × 6
└─ [135:201]   Local Joint Positions (66) = 22 joints × 3

Stored 135-dim (motion_135 field):
├─ [0:3]       Translation (3)
└─ [3:135]     6D Rotations (132)

Computed 22×3 field (positions):
└─ FK-computed 3D coordinates from rotation + skeleton

Missing from NPZ:
└─ [135:201]   Local joint positions (not explicitly saved)
                (can be reconstructed from FK + translation)
```

### Why Only 135 Dims Saved?

The eval script lines 417-434 check for a `pos_channel` (dims 135-198) but only use it for consistency metrics, not primary storage:

```python
if output_denorm.shape[-1] >= 198:
    pos_channel = output_denorm[:, 135:198].numpy()  # (T, 63)
    # Used for metrics only, not saved to NPZ
```

This suggests the model might output 198 dims (T + 6D*22), but the eval script is conservative and only saves 135 + FK-computed positions.

---

## 7. Existing Eval Results

### Directories in `work_dirs/`

```
work_dirs/
├── m2m_v2_t2m_eval/                          (Main Yiran eval)
│   ├── caption_global/
│   │   ├── npz/       (240 motion files)
│   │   └── result.json
│   ├── caption_local/
│   ├── uncond_global/
│   └── uncond_local/
│
├── m2m_v2_t2m_eval_cfg_ablation_2860_unpatched/  (CFG ablation)
│   ├── cfg1/npz/
│   ├── cfg1.5/npz/
│   ├── cfg2.5/npz/
│   ├── cfg4/npz/
│   └── cfg7.5/npz/
│
├── m2m_v2_t2m_eval_cfg_ablation_v2/
├── m2m_v2_t2m_eval_compare/
└── kimodo_t2m_eval/
```

### Result JSON Structure

```json
{
  "model": "caption_local",
  "checkpoint": ".../latest.ckpt",
  "rotation_space": "local",
  "has_caption": true,
  "num_prompts": 240,
  "num_steps": 50,
  "cfg_scale": 5.0,
  "total_time_sec": 3600.0,
  "speed_samples_per_min": 4.0,
  "aggregated": {
    "jitter_135": {
      "mean": 0.041,
      "std": 0.012,
      "median": 0.038,
      "min": 0.025,
      "max": 0.089
    },
    "jitter_pos": {...},
    "bone_length_cv": {...},
    "foot_contact_time": {...},
    "qc_pass": {...},
    ...
  },
  "per_sample": [
    {
      "prompt_id": "00001401",
      "text": "a person walks forward",
      "target_frames": 120,
      "actual_frames": 120,
      "metrics": {
        "jitter_135": 0.0408,
        "inference_time": 1.32,
        "rot6d_norm_mean": 1.0012,
        ...
      }
    },
    ...
  ]
}
```

---

## 8. Key Inference Parameters

### From eval_m2m_v2_t2m.py

```python
# Line 298-300
T = min(prompt['frames'], 360)  # Actual motion length
D = 198                          # Motion dimension
T_PAD = 360                      # Padded length

# Line 586
num_steps = 50                   # ODE solver steps (default)

# Line 587
cfg_scale = 5.0                  # Text guidance strength (default)

# Quality guide:
# cfg_scale=1.0  → No guidance (ignores text)
# cfg_scale=3.0  → Balanced
# cfg_scale=5.0  → Strong (DEFAULT)
# cfg_scale=7.0  → Very strict
# cfg_scale>10   → Over-constrained (not recommended)

# Speed vs Quality:
# num_steps=20   → ~0.5s per motion
# num_steps=50   → ~1-2s per motion (DEFAULT)
# num_steps=100  → ~3-4s per motion
```

---

## 9. Text Encoding

### Auto-Injection (lines 243-250)

```python
if model_info['has_caption'] and bundle._text_encoder_cfg is None:
    bundle._text_encoder_cfg = {
        'llm_type': 'qwen3_embedding',
        'max_length_llm': 512,
        'sentence_emb_type': 'clipl',
        'max_length_sentence_emb': 77,
    }
```

### Encoding Flow (lines 314-318)

```python
text_out = bundle.encode_text([text])
batch['text_vec_raw'] = text_out['text_vec_raw']          # (1, 77, 768)
batch['text_ctxt_raw'] = text_out['text_ctxt_raw']        # (1, 512, 4096)
batch['text_ctxt_raw_length'] = text_out['text_ctxt_raw_length']  # (1,)
```

### Encoders
- **LLM**: Qwen3-0.6B → (1, 512, 4096)
- **Sentence**: CLIP-L → (1, 77, 768)

---

## 10. Metrics Computed in Eval

From lines 347-415 of eval_m2m_v2_t2m.py:

```python
metrics = {
    # Temporal smoothness
    'jitter_135': compute_jitter_135(output_135),         # L2 velocity
    'jitter_pos': compute_jitter_positions(pos_np),       # 3D smoothness
    
    # Structural consistency
    'bone_length_cv': compute_bone_length_cv(pos_np),     # Bone length variation
    'foot_contact_time': compute_foot_ground_metrics(...),  # Ground contact
    
    # Sanity checks
    'rot6d_norm_mean': float(rot6d_norms.mean()),         # Should ≈ 1.0
    'rot6d_norm_std': float(rot6d_norms.std()),
    'transl_range_x/y/z': float(...),
    
    # Speed & acceleration
    'avg_velocity': float(vel_mag.mean()),
    'max_velocity': float(vel_mag.max()),
    'avg_acceleration': float(acc_mag.mean()),
    'max_acceleration': float(acc_mag.max()),
    
    # Quality checks
    'qc_pass': 1 if result.is_valid else 0,
    'qc_num_failed': len(result.failed_checks),
    
    # Inference
    'inference_time': round(elapsed, 2),
}
```

---

## 11. Quick Reference: Running T2M Inference

### Minimal Python Code
```python
import torch
from mmengine.config import Config
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
from hftrainer.registry import MODEL_BUNDLES
from hftrainer.utils.checkpoint_utils import load_checkpoint

# Load model
cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())
sd = load_checkpoint('checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt')
bundle.load_state_dict_selective(sd)
bundle.eval().to('cuda')

# Infer
pipeline = HyMotionT2MPipeline(bundle, num_steps=50, text_guidance_scale=5.0)
with torch.no_grad():
    output = pipeline({'tgt_length': [360], 'caption': ['a person walks']})

# Extract
motion_135 = output['latent'].cpu().numpy()  # (1, 360, 135) [NOT 201!]
```

### Command-Line Batch Eval
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Single GPU
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --gpus 0 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/t2m_eval_new/

# Multi-GPU with CFG ablation
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local \
    --cfg-sweep 1.0 3.0 5.0 7.0 \
    --prompt-chunks 4 \
    --gpus 0 1 2 3 \
    --num-steps 50 \
    --output-dir work_dirs/t2m_cfg_ablation/
```

---

## Summary Table

| Component | Value |
|-----------|-------|
| **Config** | `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` |
| **Checkpoint** | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (1.8 GB) |
| **Model Type** | HyMotionT2MBundle + HunyuanMotionMMDiT |
| **Parameters** | 460M (0.46B) |
| **Input Motion Dim** | 201 (theoretical) / 198 (used in eval) |
| **Output Motion Dim** | 135 (saved to NPZ) |
| **NPZ Fields** | `motion_135` (T,135), `positions` (T,22,3), `translation` (T,3) |
| **Inference Method** | ODE-based (Euler) |
| **Text Encoders** | Qwen3 (4096) + CLIP-L (768) |
| **Guidance** | Classifier-free (default scale=5.0) |
| **Default Steps** | 50 (ODE integration) |
| **Eval Script** | `scripts/eval/eval_m2m_v2_t2m.py` (751 lines) |
| **Results Location** | `work_dirs/m2m_v2_t2m_eval/` |

