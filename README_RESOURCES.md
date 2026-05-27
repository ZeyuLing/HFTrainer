# PRISM Evaluation Resources

This directory contains comprehensive documentation and utilities for accessing and working with PRISM text-to-motion model predictions on the HumanML3D test set.

## Files in This Package

### 1. **PRISM_EVALUATION_GUIDE.md** (Detailed Technical Guide)
Comprehensive markdown guide covering:
- Overview and quick start
- File structure and NPZ format
- NPZ keys and data structures  
- Manifest and run metadata
- SMPLX-55 format details
- 135-dimensional format explanation
- Motion duration statistics
- Python loading examples
- Format conversion code
- Evaluation configuration

**Best for**: Understanding the complete technical structure

---

### 2. **PRISM_COMPLETE_REFERENCE.txt** (Quick Reference)
Plain text reference with quick access commands:
- Executive summary
- Quick access shell commands
- Directory structure
- SMPLX-55 format breakdown
- 135D format specification
- Manifest and metadata structures
- Troubleshooting guide
- Dataset statistics

**Best for**: Fast lookup and copy-paste commands

---

### 3. **prism_predictions_loader.py** (Loading Utility)
Python class for convenient loading and management of PRISM predictions.

**Usage**:
```
loader = PRISMPredictionLoader()
ids = loader.list_available_ids()
data = loader.load_smplx55("humanml3d_10006")
caption = loader.get_caption("humanml3d_10006")
```

---

### 4. **smplx55_to_135dim_converter.py** (Format Conversion Utility)
Converter from SMPLX-55 axis-angle format to 135-dimensional evaluation format.

**Usage**:
```
converter = SMPLX55To135DConverter()
data = np.load("path/to/humanml3d_10006.npz")
motion_135d = converter.convert_smplx55_to_135d(data)
```

---

## Quick Start

### 1. Check the Predictions Exist
```
ls work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten/*.npz | wc -l
# Output: 4270
```

### 2. Load a Single Motion
```
import numpy as np
data = np.load("work_dirs/.../eval_hml3d_rewritten/humanml3d_10006.npz")
print(data['transl'].shape)  # (T, 3)
```

### 3. Compute Metrics
```
python scripts/eval/eval_with_motionclip_evaluator.py \
    --pred_dir work_dirs/prism_1b_tp2m_multiframe_kt_spectral_OLD_row_major_20260521/eval_hml3d_rewritten \
    --device cuda:0
```

---

## Key Information Summary

| Property | Value |
|----------|-------|
| Total Predictions | 4,270 complete samples |
| Storage Location | work_dirs/.../eval_hml3d_rewritten/ |
| Format (Stored) | SMPLX-55 with axis-angle rotations |
| Format (Evaluated) | 135D (3D trans + 22 joints × 6D rot) |
| NPZ File Size | ~100 KB average (426 MB total) |
| Frame Count Range | 37-301 frames (1.2-10 seconds @ 30fps) |
| Inference Steps | 50 diffusion steps |
| Guidance Scale | 5.0 |
| Deterministic Seed | 42 |
| Evaluation Date | May 19, 2026 |
| GPUs Used | 8 (parallel processing) |

---

## Generated Documentation

All reference files are in the working directory:
- PRISM_EVALUATION_GUIDE.md
- PRISM_COMPLETE_REFERENCE.txt
- prism_predictions_loader.py
- smplx55_to_135dim_converter.py
- README_RESOURCES.md (this file)

---

## Notes

✓ All 4,270 predictions are complete
✓ Frame counts match HumanML3D specifications
✓ Generated deterministically (seed=42)
✓ Full SMPLX-55 format (not reduced)
✓ Automatic 135D conversion in evaluation scripts
✓ Metadata fully preserved

---

Generated: 2026-05-27
Last Updated: 2026-05-27

