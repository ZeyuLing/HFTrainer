# KAFS (Kinematic-Adaptive Flow Scheduling) Search Report

## Summary
This report details the comprehensive search for KAFS implementation in the HFTrainer codebase for PRISM T2M (Text-to-Motion) inference.

---

## 1. KAFS Implementation Location

### Primary File: `prism_backend.py`
**Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/pipelines/motion/prism_backend.py`

**Key Components**:
- Class: `PrismARPipeline`
- KAFS Members (Lines 75-78):
  ```python
  # KAFS-Inference: Per-joint adaptive timestep scaling
  # Shape: [num_joints] with values in range [0.85, 1.15] based on kinematic depth
  self._kafs_alpha_map = None
  self._kafs_mode = "none"  # Tracks which KAFS mode is active
  ```

---

## 2. KAFS Method: `set_kafs_alpha()`

**Location**: Lines 134-221 in `prism_backend.py`

**Method Signature**:
```python
def set_kafs_alpha(self, mode: str = "none", alpha_vals: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> None
```

**Supported KAFS Modes**:

1. **"none"** (Line 156-159)
   - Disables KAFS scaling (standard baseline)
   - Sets `_kafs_alpha_map = None`
   - Output: "KAFS: Disabled (standard baseline)"

2. **"depth_driven"** (Lines 161-186)
   - Per-joint scaling based on kinematic tree depth
   - Hardcoded 23-joint SMPL structure with kinematic-based alpha values:
     ```
     Root (depth 0):      0.85 (Translation, Pelvis)
     Legs (depth 1-3):    0.90-1.10
     Spine (depth 1-3):   1.00
     Arms (depth 4-6):    1.10-1.15 (distal = higher)
     ```
   - Joint order: [trans, pelvis, L_hip, R_hip, spine1, L_knee, R_knee, spine2,
                   L_ankle, R_ankle, spine3, L_foot, R_foot, neck,
                   L_collar, R_collar, head, L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist]

3. **"uniform"** (Lines 188-194)
   - All joints get the same alpha (1.0)
   - Should give similar results to baseline for ablation
   - Output: "KAFS: Uniform mode enabled. All alphas = 1.0"

4. **"random"** (Lines 196-202)
   - Random alphas in [0.85, 1.15] for ablation control
   - Reproducible with seed=42
   - Output: "KAFS: Random mode enabled. Alpha range: [min, max]"

5. **"custom"** (Lines 204-218)
   - Use provided alpha_vals tensor
   - Must have shape [..., 23] for 23 joints
   - Accepts torch.Tensor or list
   - Output: "KAFS: Custom mode enabled. Alpha range: [min, max]"

**Usage Example**:
```python
pipeline.set_kafs_alpha(mode="depth_driven", device="cuda")
```

---

## 3. KAFS Application in Inference Loop

**Location**: Lines 375-390 in `generate_single_segment()` method

**Implementation**:
```python
if self.config.expand_timesteps:
    latent_model_input = (
        (1 - first_frame_mask) * condition + first_frame_mask * latents
    ).to(transformer_dtype)
    if self._kafs_alpha_map is not None:
        temp_ts = (first_frame_mask[0][0] * t * self._kafs_alpha_map).flatten()
    else:
        temp_ts = (first_frame_mask[0][0] * t).flatten()
    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
else:
    latent_model_input = latents.to(transformer_dtype)
    timestep = t.expand(latents.shape[0])
```

**KAFS Effect**:
- Multiplies timestep `t` element-wise with `alpha_map`: `t_j = t * alpha_j`
- Only active when `config.expand_timesteps = True`
- Only applied to non-condition frames (first_frame_mask == 1)
- Creates per-joint adaptive timestep scaling

---

## 4. Inference Entry Points

### 4.1 Main Inference Tool
**File**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/tools/infer.py`

**Entry Point Function**: `infer_prism(bundle, args)` (Lines 110-129)

**Usage**:
```bash
python tools/infer.py \
    --config configs/prism/prism_smoke.py \
    --checkpoint work_dirs/prism_smoke/checkpoint-iter_10 \
    --prompt "a person walks forward" \
    --output output/motion.npz
```

**KAFS Integration Status**: ❌ NOT exposed in CLI args
- The `infer_prism()` function does NOT call `set_kafs_alpha()`
- No command-line argument for KAFS mode selection
- Would need modification to enable KAFS in standard inference

### 4.2 Pipeline Wrapper
**File**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/pipelines/motion/prism_pipeline.py`

**Entry Point Class**: `PrismPipeline` (Lines 12-48)

**Access to KAFS**: ✅ Available via backend
```python
self.backend = PrismARPipeline(...)
# To use KAFS:
pipeline.backend.set_kafs_alpha(mode="depth_driven")
```

### 4.3 Direct Backend Usage
**File**: `prism_backend.py` main() function (Lines 734-848)

**Entry Point Function**: `main()`

**Usage**:
```bash
python -m hftrainer.pipelines.motion.prism_backend \
    --trainer_cfg configs/prism/prism_1b_tp2m_1frame.py \
    --trainer_ckpt work_dirs/.../checkpoint.pth \
    --prompts "A person walks;A person runs" \
    --output_path outputs/
```

**KAFS Integration Status**: ❌ NOT exposed in CLI args
- Similar to infer.py, the main() function does NOT use `set_kafs_alpha()`
- Would need modification for KAFS support

---

## 5. PRISM T2M Configuration Files

### Configs Directory
**Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/configs/prism/`

**Files**:
- `prism_1b_tp2m_1frame.py` - Main T2M config with 1-frame conditioning
- `prism_1b_tp2m_multiframe.py` - Multiframe variant
- `prism_mcm_motionhub.py` - MCM (motion-conditioned motion) variant
- `prism_mcm_motionhub_16v100.py` - 16-GPU variant
- `prism_mcm_motionhub_64v100.py` - 64-GPU variant
- `prism_debug_loss_split.py` - Debug config

**KAFS Configuration Status**: ❌ NO KAFS settings in configs
- None of the PRISM config files mention `expand_timesteps`, KAFS, or related settings
- All configs use default `PrismTrainer` without KAFS-specific parameters

### Key Trainer Config (prism_1b_tp2m_1frame.py, Lines 95-101)
```python
trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=256,
)
```

---

## 6. Evaluation Scripts

### T2M Evaluation Script
**Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/eval/eval_m2m_v2_t2m.py`

**Purpose**: Multi-GPU parallel T2M evaluation on Yiran subset
- Evaluates HyMotion M2M models on T2M task
- NOT PRISM-specific
- Supports CFG sweep ablations

**KAFS Status**: ❌ NO KAFS support
- This script evaluates HyMotion, not PRISM
- No KAFS implementation for HyMotion pipelines

---

## 7. How to Enable KAFS in Inference

### Method 1: Direct Backend Usage (Recommended)
```python
from hftrainer.pipelines.motion.prism_pipeline import PrismPipeline
from hftrainer.registry import MODEL_BUNDLES

# Load bundle
bundle = MODEL_BUNDLES.get('PrismBundle').from_config(cfg)
pipeline = PrismPipeline(bundle=bundle)

# Enable KAFS
pipeline.backend.set_kafs_alpha(mode="depth_driven")

# Generate
output = pipeline(
    prompts="a person walks forward",
    num_frames_per_segment=129,
    num_inference_steps=50,
)
```

### Method 2: Modify infer.py
Add KAFS argument and apply it:
```python
parser.add_argument('--kafs-mode', default='none', 
                    help='KAFS mode: none, depth_driven, uniform, random, custom')

# After pipeline creation:
pipeline.backend.set_kafs_alpha(mode=args.kafs_mode)
```

### Method 3: Modify prism_backend.py main()
Add KAFS support to the standalone entry point.

---

## 8. KAFS Technical Details

### Joint Structure (23 SMPL Joints)
```
Index  Joint Name       Kinematic Depth  Alpha
0      Trans            0                0.85
1      Pelvis           0                0.85
2      L_Hip            1                0.90
3      R_Hip            1                0.90
4      Spine1           1                1.00
5      L_Knee           2                1.00
6      R_Knee           2                1.00
7      Spine2           2                1.00
8      L_Ankle          3                1.05
9      R_Ankle          3                1.05
10     Spine3           3                1.00
11     L_Foot           4                1.10
12     R_Foot           4                1.10
13     Neck             2                1.00
14     L_Collar         3                1.05
15     R_Collar         3                1.05
16     Head             3                1.00
17     L_Shoulder       4                1.10
18     R_Shoulder       4                1.10
19     L_Elbow          5                1.12
20     R_Elbow          5                1.12
21     L_Wrist          6                1.15
22     R_Wrist          6                1.15
```

### Alpha Value Interpretation
- **Lower alpha (0.85)**: Slower timestep scaling for root/proximal joints
  - More denoising steps focus on root motion
  - Better stability for body translation

- **Higher alpha (1.15)**: Faster timestep scaling for distal joints
  - Fewer denoising steps for fine-grained details (fingers, wrists)
  - More flexible motion generation for distal joints

### Timestep Transformation
```
Original:     t = [t₀, t₁, ..., tₙ]     (same for all joints)
KAFS:         t'ⱼ = t × αⱼ              (per-joint scaling)
Effect:       Different diffusion schedules per joint
```

---

## 9. Files Summary

### Core KAFS Implementation
| File | Lines | Purpose |
|------|-------|---------|
| prism_backend.py | 75-78 | KAFS member initialization |
| prism_backend.py | 134-221 | `set_kafs_alpha()` method |
| prism_backend.py | 383-384 | KAFS application in denoising loop |

### Inference Entry Points
| File | Function | KAFS Support |
|------|----------|--------------|
| tools/infer.py | `infer_prism()` | ❌ NO |
| tools/infer.py | `main()` CLI | ❌ NO |
| hftrainer/pipelines/motion/prism_pipeline.py | `PrismPipeline.__call__()` | ✅ YES (via backend) |
| hftrainer/pipelines/motion/prism_backend.py | `PrismARPipeline.__call__()` | ✅ YES (direct) |
| hftrainer/pipelines/motion/prism_backend.py | `main()` | ❌ NO |

### Configuration Files
| File | KAFS Settings |
|------|---------------|
| configs/prism/prism_1b_tp2m_1frame.py | ❌ NONE |
| configs/prism/prism_1b_tp2m_multiframe.py | ❌ NONE |
| configs/prism/prism_mcm_motionhub.py | ❌ NONE |

---

## 10. Recommendations

### For KAFS Integration in Evaluation
1. **Add CLI argument to infer.py**
   ```python
   parser.add_argument('--kafs-mode', choices=['none', 'depth_driven', 'uniform', 'random', 'custom'])
   parser.add_argument('--kafs-custom-alpha', help='Custom alpha values as JSON list')
   ```

2. **Apply KAFS after pipeline creation**
   ```python
   pipeline.backend.set_kafs_alpha(mode=args.kafs_mode)
   ```

3. **Document KAFS in config templates**
   - Add optional `expand_timesteps: true` to trainer config
   - Document KAFS alpha ranges

4. **Create evaluation script**
   - Compare baseline vs KAFS modes
   - Generate metrics with/without KAFS

---

## 11. Testing KAFS Functionality

### Verify KAFS is enabled
```python
# Check if KAFS is active
print(pipeline.backend._kafs_mode)  # Should be 'depth_driven'
print(pipeline.backend._kafs_alpha_map.shape)  # Should be [1, 1, 1, 23]
```

### Compare outputs
- Generate same prompt with different KAFS modes
- Compare motion smoothness/quality metrics
- Visualize joint deviations

---

## Search Completion Status

✅ Found KAFS implementation in prism_backend.py
✅ Identified inference entry points
✅ Documented all KAFS modes (none, depth_driven, uniform, random, custom)
✅ Analyzed integration points
✅ Verified config files (no KAFS configs exist)
✅ Documented T2M evaluation scripts
✅ Provided integration recommendations

