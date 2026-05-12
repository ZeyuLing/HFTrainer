# HyMotion T2M 1.0 Inference & Output Format Guide

## Configuration Overview

**Config file**: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`

### Key Config Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Motion Dimension** | 201 | [transl(3) + 22×rot6d(132) + 22×joint_pos(66)] |
| **Model Architecture** | HunyuanMotionMMDiT (0.46B) | 460M parameters, no VACE conditioning |
| **Feature Dim** | 1024 | Internal transformer hidden dimension |
| **Num Layers** | 18 | Transformer depth |
| **Num Heads** | 16 | Multi-head attention heads |
| **Text Encoders** | Qwen3 (4096-dim) + CLIP-L (768-dim) | For text conditioning |
| **Prediction Type** | velocity | Predicts motion velocity field |
| **Noise Scheduler** | Euler | ODE integration method |
| **Inference Steps** | 50 | Default ODE integration steps (configurable) |
| **CFG Scale** | N/A (text-conditioned) | Text guidance scale in pipeline (not in config) |
| **CFG Dropout** | 10% | Classifier-free guidance: 10% of samples drop text |

### Checkpoint Info

**Path**: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`

- **Model Type**: HyMotionT2MBundle
- **Input/Output Dims**: 
  - `motion_transformer.input_encoder`: [1024, 201]
  - `motion_transformer.final_layer.linear`: [201, 1024]
- **Normalization**: Mean/std stored in checkpoint for 201-dim motion
  - `mean`: shape [201]
  - `std`: shape [201]

---

## Inference Execution

### How to Run T2M Inference

The main evaluation script is: `scripts/eval/eval_m2m_v2_t2m.py`

#### Basic Usage (Single GPU)

```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --gpus 0 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/my_t2m_eval
```

#### Advanced Usage (CFG Ablation - Multi-GPU)

```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --cfg-sweep 1.0 1.5 2.5 4.0 7.5 \
    --prompt-chunks 8 \
    --gpus 0 1 2 3 4 5 6 7 \
    --num-steps 50 \
    --ckpt-path checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --output-dir work_dirs/t2m_eval_cfg_ablation
```

### Inference Pipeline Code

**File**: `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py`

**Key Flow**:

1. **Input**: Batch dict with:
   - `tgt_length`: desired motion sequence length (list of ints)
   - `caption` or pre-encoded text embeddings (`text_vec_raw`, `text_ctxt_raw`, `text_ctxt_raw_length`)
   - Optional `motion_dim` (defaults to transformer's `output_dim=201`)

2. **Text Encoding** (if not pre-encoded):
   - Encodes text via Qwen3 → `text_ctxt_raw` [B, seq_len, 4096]
   - Encodes text via CLIP-L → `text_vec_raw` [B, 1, 768]

3. **ODE Integration** (solver method = "euler" by default):
   - Initial noise: `y0 ~ N(0, 1)` shape [B, T, 201]
   - Solves ODE from t=0 (noise) to t=1 (clean)
   - Supports **Classifier-Free Guidance (CFG)**:
     - If `text_guidance_scale > 1.0`:
       - Runs inference twice: with + without text
       - Combines predictions: `x_pred_uncond + scale * (x_pred_text - x_pred_uncond)`

4. **Output Post-Processing**:
   - Denormalize using mean/std from checkpoint
   - Extract first 135 dims for motion rotation/translation
   - Optionally convert to joint positions via forward kinematics

---

## Output Format

### NPZ File Structure

**Location**: `work_dirs/{model_name}/npz/{prompt_id}.npz`

**Example File**: `work_dirs/m2m_v2_t2m_eval/caption_global/npz/00000001.npz`

#### NPZ Keys and Shapes

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `motion_135` | (T, 135) | float32 | Full motion in 135-dim space: transl(3) + 22×rot6d(132) |
| `positions` | (T, 22, 3) | float32 | World-space joint positions (22 SMPL joints × 3 coords) |
| `translation` | (T, 3) | float32 | Root translation (same as first 3 dims of motion_135) |

**Example Data Shape**:
- Sample motion with T=60 frames at 30 FPS (2 seconds):
  - `motion_135`: (60, 135)
  - `positions`: (60, 22, 3) [pelvis, spine, shoulders, elbows, wrists, hips, knees, ankles, etc.]
  - `translation`: (60, 3) [x, y, z coordinates]

### Result JSON Structure

**Location**: `work_dirs/{model_name}/result.json` (or `cfg{X}/result.json` for CFG ablation)

**Top-Level Keys**:

```python
{
    "model": "caption_global_phase2",           # Model name
    "checkpoint": "/path/to/ckpt.pth",          # Checkpoint path used
    "rotation_space": "local",                  # "local" or "global"
    "has_caption": True,                        # Whether model uses text conditioning
    "num_prompts": 240,                         # Total test samples
    "num_steps": 50,                            # ODE integration steps
    "cfg_scale": 5.0,                           # Classifier-free guidance scale
    "total_time_sec": 1234.5,                   # Total inference time
    "speed_samples_per_min": 11.7,              # Inference speed
    "aggregated": {...},                        # Metrics aggregation (mean/std/median/min/max)
    "per_sample": [...]                         # Per-prompt results
}
```

### Metrics Included in Result JSON

#### Motion Quality Metrics

| Metric | Meaning |
|--------|---------|
| `jitter_135` | Joint velocity jitter in motion space |
| `jitter_pos` | Joint position jitter (differentiable) |
| `avg_velocity` | Mean joint velocity magnitude |
| `max_velocity` | Peak joint velocity magnitude |
| `avg_acceleration` | Mean joint acceleration magnitude |
| `max_acceleration` | Peak joint acceleration magnitude |
| `pelvis_trans_jerk` | Root (pelvis) translational jerk |

#### Physical Validity Metrics

| Metric | Meaning |
|--------|---------|
| `bone_length_cv_mean` | Bone length coefficient of variation (should be ~0) |
| `bone_length_cv_max` | Max bone length variation (should be ~0) |
| `foot_ground_*` | Foot-ground contact metrics (penetration, sliding, etc.) |
| `arm_penetration_ratio` | Wrist penetration into torso (0 = good) |

#### Position Channel Metrics

| Metric | Meaning |
|--------|---------|
| `pos_channel_range` | Value range of extra position channel output |
| `pos_channel_mean` | Mean of position channel output |
| `fk_consistency_mae` | MAE between FK-computed vs. network-predicted positions |

#### Sanity Check Metrics

| Metric | Meaning |
|--------|---------|
| `rot6d_norm_mean` | Mean of 6D rotation vector norms (should be ~1.4 for unit rotations) |
| `rot6d_norm_std` | Std of rotation norms (should be small) |
| `transl_range_x/y/z` | Root translation range per axis |

#### Quality Checker Metrics

| Metric | Meaning |
|--------|---------|
| `qc_pass` | Overall quality check pass (1=valid, 0=invalid) |
| `qc_num_failed` | Count of failed checks |
| `qc_num_borderline` | Count of borderline checks |
| `qc_{checker_name}` | Per-check validity score |

---

## Current Status & Limitations

### ✅ What Works

- **201-dim checkpoint loading**: Latest.ckpt has 201-dim input/output (confirmed in state dict)
- **Text-conditioned T2M generation**: Full Qwen3 + CLIP-L text encoding pipeline
- **CFG-guided inference**: Classifier-free guidance with configurable scales
- **Evaluation infrastructure**: Multi-GPU parallel inference + per-cfg ablation

### ⚠️ Current Pipeline Mismatch

**Issue**: Config says 201 dims, but data pipeline outputs only **135 dims**.

```python
# In config: hymotion_t2m_201dim_046b.py (line 12-17)
_motion_dim = 201  # Model expects 201

# But LoadSmplx55 pipeline only outputs 135 dims
# (transl: 3 + 22×rot6d: 132)
```

**Data Pipeline File**: `hftrainer/models/motion/data/data_transforms.py` (LoadSmplx55)

Currently outputs:
- `transl`: (T, 3)
- `rot6d`: (T, 22, 6) → flattened to (T, 132)
- **Total: 135 dims**

### 🔧 What Needs to be Fixed

To support **true 201-dim training**, LoadSmplx55 needs to output:
- `transl`: (T, 3)
- `rot6d`: (T, 22, 6) → (T, 132)
- `positions`: (T, 22, 3) → (T, 66)  **[NEW]**
- **Total: 3 + 132 + 66 = 201 dims**

---

## How to Run T2M 1.0 Inference

### Step 1: Configure the Model

The config is already set up: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`

Key settings:
- Motion dim: 201
- Model: HunyuanMotionMMDiT (0.46B)
- Checkpoint: Pre-trained HY-Motion-1.0-Lite

### Step 2: Run Inference

**Option A**: Use the eval script directly (recommended):

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Test one prompt
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --gpus 0 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --prompt-file data/eval/t2m/251125_yiran_subset.json \
    --output-dir work_dirs/t2m_test
```

**Option B**: Use the pipeline directly in code:

```python
from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
from mmengine.config import Config
from hftrainer.registry import MODEL_BUNDLES

# Load config
cfg = Config.fromfile('configs/hymotion_t2m/hymotion_t2m_201dim_046b.py')

# Build model
bundle = MODEL_BUNDLES.build(cfg.model.to_dict())

# Create pipeline
pipeline = HyMotionT2MPipeline(
    bundle=bundle,
    num_steps=50,
    text_guidance_scale=5.0,
)

# Prepare batch
batch = {
    'tgt_length': [360],              # Desired sequence length
    'caption': ['A person walks forward.'],
}

# Run inference
with torch.no_grad():
    output = pipeline(batch)

# Extract results
motion_201 = output['latent']  # [1, 360, 201]
motion_135 = bundle.denormalize_motion(motion_201)[0, :, :135]  # [360, 135]
```

### Step 3: Check Output

The eval script outputs:
- **NPZ files**: `work_dirs/{output_dir}/{model}/npz/{id}.npz`
  - Contains: `motion_135`, `positions`, `translation`
- **JSON results**: `work_dirs/{output_dir}/{model}/result.json`
  - Contains: aggregated metrics + per-sample results

---

## Motion Representation Breakdown (201 dims)

```
motion_201[t, :] = [
    transl[0:3],                     # 3 dims — root translation
    rot6d[0:132],                    # 132 dims — 22 joints × 6d rotation
    positions[0:66]                  # 66 dims — 22 joints × 3d world position
]

Total: 3 + 132 + 66 = 201 dims
```

### Conversion to Standard Formats

**From NPZ to SMPL-X**:
```python
# motion_135 → SMPL poses
transl = motion_135[:, :3]
rot6d = motion_135[:, 3:135].reshape(-1, 22, 6)

# Convert 6D rotation to axis-angle (66 dims) for SMPL
from models.motion.components.utils.geometry.rotation_convert import rotation_6d_to_axis_angle
aa = rotation_6d_to_axis_angle(torch.from_numpy(rot6d).float())
poses = aa.reshape(-1, 66)  # Standard SMPL pose representation
```

---

## Existing Eval Outputs

**Location**: `work_dirs/m2m_v2_t2m_eval*`

Existing directories with T2M generation results:
- `m2m_v2_t2m_eval/` — Multi-model comparison
- `m2m_v2_t2m_eval_cfg_ablation_2860_unpatched/` — CFG scale ablation
- `m2m_v2_t2m_eval_cfg_ablation_v2/` — CFG scale ablation (v2)
- `m2m_v2_t2m_eval_compare/` — Comparison variants

Each contains:
- `{model_name}/result.json` — Aggregated metrics
- `{model_name}/npz/*.npz` — Generated motion files

