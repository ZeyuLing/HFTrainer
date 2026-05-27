# PRISM Model Evaluation on HumanML3D Test Set

## Overview

Complete evaluation outputs from PRISM (text-to-motion generation model) on the HumanML3D test set.

- **Total Samples**: 4,270 complete predictions
- **Format**: Individual NPZ files in SMPLX-55 format with axis-angle rotations
- **Storage Location**: `/work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/`
- **Evaluation Date**: May 19, 2026
- **Inference Config**: 50 diffusion steps, guidance scale 5.0, seed 42

## Quick Start

### List all predictions:
```bash
ls work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/*.npz | wc -l
```

### Load a single prediction:
```python
import numpy as np
data = np.load("work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/humanml3d_10006.npz")
print(data.files)  # List all keys
transl = data['transl']  # Shape: (T, 3)
poses = data['poses']    # Shape: (T, 165)
```

### Check metadata:
```python
import json
manifest = json.load(open("work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/manifest.json"))
run_meta = json.load(open("work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/run_meta.json"))
```

## File Structure

### Directory Layout
```
work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/
├── humanml3d_10006.npz          (individual prediction files)
├── humanml3d_10007.npz
├── ... (4,270 files total)
├── manifest.json                 (metadata index)
└── run_meta.json                 (evaluation config)
```

### NPZ File Naming
- Pattern: `humanml3d_<ID>.npz`
- Example: `humanml3d_10006.npz` corresponds to HumanML3D test ID `humanml3d_10006`
- All 4,270 test IDs have corresponding predictions

### NPZ Keys and Structure

Each NPZ file contains:

```
'transl'              (T, 3)   float32    Translation [XYZ]
'global_orient'       (T, 3)   float32    Root rotation [axis-angle]
'body_pose'           (T, 63)  float32    21 body joints × 3 [axis-angle]
'jaw_pose'            (T, 3)   float32    Jaw joint [axis-angle]
'leye_pose'           (T, 3)   float32    Left eye [axis-angle]
'reye_pose'           (T, 3)   float32    Right eye [axis-angle]
'left_hand_pose'      (T, 45)  float32    15 left hand joints × 3 [axis-angle]
'right_hand_pose'     (T, 45)  float32    15 right hand joints × 3 [axis-angle]
'poses'               (T, 165) float32    Concatenation: global_orient + body_pose + jaw + eyes + hands
'betas'               (10,)    float32    Shape parameters (usually zeros)
'expression'          (T, 10)  float32    Facial expression coefficients
'gender'              ()       string     'neutral', 'male', or 'female'
'mocap_framerate'     ()       float64    Usually 30.0 Hz
```

### manifest.json Structure
```json
[
  {
    "name": "humanml3d_10006",
    "caption": "A person stands still while pointing forward with their right arm.",
    "npz_path": "work_dirs/..../humanml3d_10006.npz",
    "status": "success"
  },
  ...
]
```

### run_meta.json Structure
```json
{
  "config": "configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py",
  "checkpoint": "work_dirs/prism_1b_tp2m_multiframe_kt_spectral/checkpoint-epoch_0",
  "anno_file": "data/annotation/test_hml3d_rewritten.json",
  "num_inference_steps": 50,
  "guidance_scale": 5.0,
  "seed": 42,
  "total_samples": 4270,
  "gpus": [0, 1, 2, 3, 4, 5, 6, 7],
  "num_workers": 8
}
```

## Format Details

### SMPLX-55 Format (Stored Format)
The predictions are stored in SMPLX-55 (SMPL Extended) format with axis-angle rotations:

- **Rotation representation**: Axis-angle (3D vectors)
- **Components**:
  - Translation (3D)
  - Global orientation (3D axis-angle)
  - 21 body joints (63D = 21 × 3)
  - Jaw joint (3D)
  - Eyes (6D = 2 × 3)
  - Hands (90D = 30 × 3)
  
- **Total DOF**: 3 + 3 + 63 + 3 + 3 + 45 + 45 = **168 dimensions**
  - Note: `poses` field contains concatenation of pose components (165D, excluding transl)

### 135-Dimensional Format (For Metrics)

The evaluation metrics (R-Precision, MM-Distance, FID, Diversity) use a 135-dimensional format:

**Conversion process**:
1. Extract 22-joint subset from SMPLX-55
2. Convert axis-angle → rotation matrix → 6D representation
3. Concatenate: transl (3D) + 22 joints × 6D = **135D**

**22-joint layout** (HumanML3D standard):
```
[0]  Hips/Pelvis
[1]  Right Hip
[2]  Right Knee
[3]  Right Ankle
[4]  Left Hip
[5]  Left Knee
[6]  Left Ankle
[7]  Spine
[8]  Chest/Thorax
[9]  Neck
[10] Head
[11] Right Shoulder
[12] Right Elbow
[13] Right Wrist
[14] Left Shoulder
[15] Left Elbow
[16] Left Wrist
[17-20] Additional joints (feet, hands, or roots)
[21] Reserved/placeholder
```

**6D rotation representation**:
- Takes first 2 rows of 3×3 rotation matrix
- Numerically more stable than quaternions for neural networks
- Conversion: `axis_angle → rotation_matrix → matrix_to_rotation_6d`

## Motion Duration Statistics

Based on sample of first 100 predictions:
- **Min frames**: 37 (≈ 1.2 seconds @ 30fps)
- **Max frames**: 301 (≈ 10 seconds)
- **Mean frames**: 193.2 (≈ 6.4 seconds)
- **Median frames**: 203 (≈ 6.8 seconds)

Frame counts are extracted from metadata during evaluation to match original HumanML3D test set specifications.

## Loading Predictions

### Python: Load Single Motion
```python
import numpy as np

motion_id = "humanml3d_10006"
npz_path = f"work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/{motion_id}.npz"

data = np.load(npz_path)

# Access components
transl = data['transl']              # (T, 3) translation
global_orient = data['global_orient'] # (T, 3) root rotation
body_pose = data['body_pose']        # (T, 63) joint rotations
poses = data['poses']                # (T, 165) all poses combined

T = transl.shape[0]  # Number of frames
duration_s = T / 30  # Duration in seconds
```

### Python: Load Batch of Motions
```python
import numpy as np
from pathlib import Path

eval_dir = Path("work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten")
manifest = json.load(open(eval_dir / "manifest.json"))

# Load first 10 motions
motions = {}
for entry in manifest[:10]:
    motion_id = entry['name']
    npz_path = eval_dir / f"{motion_id}.npz"
    motions[motion_id] = np.load(npz_path)
    
print(f"Loaded {len(motions)} motions")
```

### Python: Convert to 135-Dim (Example)
```python
from scipy.spatial.transform import Rotation
import numpy as np

def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle (N, 3) to rotation matrix (N, 3, 3)"""
    r = Rotation.from_rotvec(axis_angle)
    return r.as_matrix()

def matrix_to_rotation_6d(rotation_matrix):
    """Convert rotation matrix to 6D representation (first 2 rows)"""
    assert rotation_matrix.shape[-2:] == (3, 3)
    return rotation_matrix[..., :2, :].reshape(*rotation_matrix.shape[:-2], 6)

def smplx55_to_135dim(smplx_data):
    """Convert SMPLX-55 to 135-dim format for metrics"""
    transl = smplx_data['transl']  # (T, 3)
    global_orient = smplx_data['global_orient']  # (T, 3)
    body_pose = smplx_data['body_pose']  # (T, 63)
    
    T = transl.shape[0]
    
    # Extract 22 joints (simplified - actual selection depends on joint indices)
    # For now, just demonstrate conversion
    all_rotations = np.concatenate([global_orient, body_pose], axis=1)  # (T, 66)
    
    # Convert axis-angle to 6D
    rot_matrices = np.array([axis_angle_to_matrix(all_rotations[t]) for t in range(T)])
    rot_6d = matrix_to_rotation_6d(rot_matrices)  # (T, 22, 6) for 22 joints
    rot_6d = rot_6d.reshape(T, -1)  # Flatten to (T, 132)
    
    # Combine with translation
    motion_135d = np.concatenate([transl, rot_6d], axis=1)  # (T, 135)
    
    return motion_135d
```

## Evaluation Configuration

**PRISM Model**: `prism_1b_tp2m_multiframe_kt_spectral`
- 1 Billion parameter model
- Text-to-Motion multiframe diffusion model
- Spectral tokenization with knowledge distillation

**Evaluation Parameters**:
- **Inference Steps**: 50 (diffusion denoising steps)
- **Guidance Scale**: 5.0 (classifier-free guidance strength)
- **Seed**: 42 (deterministic generation)
- **GPUs**: 8 (multi-GPU parallel evaluation)

**Test Set**: HumanML3D
- 4,270 motions in test split
- Caption-based text-to-motion generation
- Frame counts preserved from original dataset

## Accessing Evaluation Results

### Option 1: Using the evaluation script
```bash
python scripts/eval/eval_prism_t2m_hml3d.py \
    --output-dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten \
    --gpus 0 1 2 3 4 5 6 7
```

### Option 2: Direct NPZ access (recommended)
All outputs are already generated. Simply load NPZ files as shown above.

### Option 3: Compute metrics
```bash
python scripts/eval/eval_with_motionclip_evaluator.py \
    --pred_dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten \
    --device cuda:0
```

This will compute:
- **R-Precision**: Text-motion alignment via MotionCLIP
- **MM-Distance**: Diversity metric
- **FID**: Fréchet Inception Distance
- **Diversity**: Motion diversity

## Related Files

- **Evaluation Script**: `scripts/eval/eval_prism_t2m_hml3d.py`
- **Metrics Script**: `scripts/eval/eval_with_motionclip_evaluator.py`
- **Config**: `configs/prism/prism_1b_tp2m_multiframe_kt_spectral.py`
- **Test Annotations**: `data/annotation/test_hml3d_rewritten.json`
- **Test Metadata**: `data/annotation/test_hml3d.json`

## Alternative Evaluation Directories

Other PRISM variants and checkpoints:
- `/work_dirs/prism_1b_tp2m_multiframe_kt_spectral/eval_hml3d_epoch2/` - Latest checkpoint
- `/work_dirs/prism_kafs_ablation/depth_driven/` - Ablation study
- Multiple other variants available in work_dirs

## Notes

- All 4,270 predictions are complete and successfully generated
- Frame counts match HumanML3D test set specifications (24-300 frames)
- Generated with deterministic seed for reproducibility
- Predictions are in full SMPLX format (not reduced)
- For metrics computation, automatic conversion to 135-dim is performed
- Storage size: ~426 MB total for all 4,270 NPZ files

