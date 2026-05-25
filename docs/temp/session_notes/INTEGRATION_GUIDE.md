# NPZ to SMPL-Mesh JSON Conversion: Complete Integration Guide

**Date**: 2026-05-25  
**Status**: ✅ Production Ready  
**Dataset**: PhysFlow v2 Comparison (76 motions, 6,733 frames)

---

## Overview

This guide walks you through the complete NPZ-to-SMPL-mesh conversion pipeline, including:
- Understanding the input/output formats
- Running the conversion
- Integrating with the web viewer
- Troubleshooting and next steps

---

## Quick Start (30 seconds)

If you just want to see the converted files in action:

```bash
# 1. Navigate to the web viewer
cd motion_annot_web/embodied_viz

# 2. Start the Flask app
python3 app.py --port 8095 --data-dir ../..

# 3. Open browser to http://localhost:8095
# The viewer will automatically load all 76 converted JSON files
```

---

## Part 1: Understanding the Conversion Pipeline

### Input: motion_135 NPZ Format

Each NPZ file contains:
- **motion_135**: Shape (T, 135) numpy array, float32
  - T: number of frames (40-120 in this dataset)
  - 135 = 3 (translation) + 22×6 (rot6d)
- **fps**: Integer, typically 30
- **prompt**: String description of the motion (optional)

**Example NPZ structure:**
```python
import numpy as np
data = np.load('finetuned_00_a_person_stands_still_raw.npz', allow_pickle=True)
motion = data['motion_135']      # (120, 135)
fps = int(data['fps'])            # 30
prompt = str(data.get('prompt', ''))  # motion description
```

### Output: SMPL-H Mesh JSON Format

Each JSON file contains the motion in web-ready format:

```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplh",
      "Rh": [[rx, ry, rz]],
      "Th": [[tx, ty, tz]],
      "poses": [[p0, p1, ..., p155]],
      "shapes": [[0, 0, ..., 0]],
      "mocap_framerate": 30
    }],
    // ... more frames
  ]
}
```

**Key Fields:**
- **Rh**: Root orientation (axis-angle, 3 values)
- **Th**: Translation/position (3 values)
- **poses**: 156 parameters for SMPL-H
  - [0:3] Root orientation
  - [3:66] Body joints (21×3)
  - [66:156] Hand joints (30×3, zero-padded)
- **shapes**: 16 shape parameters (all zeros in this dataset)

---

## Part 2: The Conversion Algorithm

### Step 1: Load and Extract

```python
data = np.load(npz_path, allow_pickle=True)
motion = data['motion_135']  # (T, 135)
fps = int(data.get('fps', 30))
T = motion.shape[0]

# Split into translation and rotation
transl = motion[:, :3]                    # (T, 3)
rot6d = motion[:, 3:].reshape(T, 22, 6)  # (T, 22, 6)
```

### Step 2: Convert rot6d to Axis-Angle

HyMotion stores rot6d in row-major format. The conversion:

```python
def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R
    
    # Row-major [R00, R01, R10, R11, R20, R21]
    # → Column-major ordering via index reorder
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    
    # Extract two column vectors
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    # Gram-Schmidt orthogonalization to recover full 3×3 matrix
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    # Stack into 3×3 rotation matrix
    rotmat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    
    # Convert to axis-angle
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    aa_flat = R.from_matrix(rotmat_flat).as_rotvec()
    return aa_flat.reshape(*orig_shape, 3).astype(np.float32)
```

**Why this works:**
- rot6d encodes the first two columns of a 3×3 rotation matrix
- Gram-Schmidt orthogonalization recovers the third column
- The result is a proper rotation matrix (SO(3))
- scipy.spatial.transform.Rotation converts to axis-angle

### Step 3: Map to SMPL-H Structure

```python
# Extract root and body rotations
root_orient = aa[:, 0, :]      # (T, 3) - joint 0
body_pose = aa[:, 1:22, :]     # (T, 21, 3) - joints 1-21

# Build SMPL-H poses (156 parameters)
poses_per_frame = np.zeros((T, 156), dtype=np.float32)
poses_per_frame[:, :3] = root_orient              # Root: indices 0-2
poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)  # Body: indices 3-65
# Hand parameters (66-155) remain zero - no hand data in source

# Shape coefficients (unused, all zeros)
shapes = [[0.0] * 16]
```

### Step 4: Generate JSON

```python
frames = []
for t in range(T):
    frame = [{
        "id": 0,
        "gender": "neutral",
        "smpl_type": "smplh",
        "Rh": [root_orient[t].tolist()],
        "Th": [transl[t].tolist()],
        "poses": [poses_per_frame[t].tolist()],
        "shapes": shapes,
        "mocap_framerate": fps,
    }]
    frames.append(frame)

output = {
    "type": "frames",
    "fps": fps,
    "frames": frames,
}

# Save with compact format
with open(output_path, 'w') as f:
    json.dump(output, f, separators=(',', ':'))
```

---

## Part 3: Running the Conversion

### Using the Batch Script

The conversion has already been completed for the PhysFlow v2 dataset. To convert additional NPZ files:

```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/physflow_v2_compare_iter1000/npz \
    --output-dir output/physflow_v2_compare_iter1000/smpl_mesh \
    --smpl-type smplh \
    --gender neutral \
    --skip-existing
```

**Parameters:**
- `--npz-dir`: Directory containing NPZ files (mutually exclusive with --npz-file)
- `--npz-file`: Single NPZ file to convert (mutually exclusive with --npz-dir)
- `--output-dir`: Where to save JSON files (created if doesn't exist)
- `--smpl-type`: "smpl" (72 params), "smplh" (156 params), or "smplx" (165 params)
- `--gender`: "neutral", "male", or "female"
- `--skip-existing`: Don't re-convert files that already exist

**Example: Single file**
```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-file data/test.npz \
    --output-dir output/test_mesh \
    --smpl-type smplh
```

---

## Part 4: Web Viewer Integration

### Directory Structure

```
motion_annot_web/
  embodied_viz/
    app.py                    # Flask server
    templates/
      index.html              # Main viewer page
      motion_list.html        # Motion listing
    static/
      load_smpl.js           # Three.js SMPL renderer
      css/                   # Styling
    data/
      smpl_mesh/             # Symlink to converted JSON files
```

### Starting the Viewer

```bash
cd motion_annot_web/embodied_viz
python3 app.py --port 8095 --data-dir ../..
```

**Output:**
```
 * Running on http://127.0.0.1:8095
 * Press CTRL+C to quit
```

### Accessing the Viewer

1. Open browser to `http://localhost:8095`
2. The Flask app auto-discovers all JSON files in `data/smpl_mesh/`
3. Select a motion to view
4. Use playback controls to navigate frames

---

## Part 5: Verification & Testing

### Check Converted Files

```bash
# Count files
ls output/physflow_v2_compare_iter1000/smpl_mesh/ | wc -l  # Should be 76

# Check total size
du -sh output/physflow_v2_compare_iter1000/smpl_mesh/  # Should be ~12.9 MB

# View file list
ls -lh output/physflow_v2_compare_iter1000/smpl_mesh/ | head -10
```

### Verify Format

```python
import json
import numpy as np

# Load a file
with open('output/physflow_v2_compare_iter1000/smpl_mesh/pretrained_00_a_person_stands_still_raw.json') as f:
    data = json.load(f)

# Check structure
assert data['type'] == 'frames'
assert data['fps'] == 30
print(f"Total frames: {len(data['frames'])}")

# Check first frame
frame = data['frames'][0][0]
assert len(frame['Rh'][0]) == 3  # Root orientation
assert len(frame['Th'][0]) == 3  # Translation
assert len(frame['poses'][0]) == 156  # SMPL-H
assert len(frame['shapes'][0]) == 16  # Shape coefficients
assert all(s == 0 for s in frame['shapes'][0])  # All zeros

print("✅ File format verified")
```

### Compare with Source NPZ

```python
import numpy as np
import json

# Load original NPZ
npz_data = np.load('output/physflow_v2_compare_iter1000/npz/pretrained_00_a_person_stands_still_raw.npz', allow_pickle=True)
motion_135 = npz_data['motion_135']  # (120, 135)
T = motion_135.shape[0]

# Load converted JSON
with open('output/physflow_v2_compare_iter1000/smpl_mesh/pretrained_00_a_person_stands_still_raw.json') as f:
    json_data = json.load(f)

# Verify dimensions match
assert len(json_data['frames']) == T
print(f"✅ Frame count matches: {T}")

# Check translation values (should be identical)
for t in range(3):
    transl_json = json_data['frames'][t][0]['Th'][0]
    transl_npz = motion_135[t, :3].tolist()
    assert np.allclose(transl_json, transl_npz)
print("✅ Translation values match")

# Check that root orientation is plausible (axis-angle, should be small for standing still)
for t in range(5):
    rh = json_data['frames'][t][0]['Rh'][0]
    norm = np.linalg.norm(rh)
    assert norm < 1.0, f"Root orientation too large: {norm}"
print("✅ Root orientation values plausible")
```

---

## Part 6: Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'scipy'"

**Solution:**
```bash
pip install scipy
```

### Issue: "FileNotFoundError: No such file or directory"

**Solution:** Check that paths are correct:
```bash
# Verify NPZ files exist
ls output/physflow_v2_compare_iter1000/npz/ | head -5

# Verify script exists
ls scripts/embodied/batch_npz_to_smpl_mesh_json.py
```

### Issue: "JSON files too large"

The compact format (`separators=(',', ':')`) is already applied. Files are ~170 KB average.

If you need even smaller files:
- Consider lower precision (float16 instead of float32)
- Quantize values to integers with a scale factor
- Use gzip compression

### Issue: "Web viewer not finding JSON files"

**Solution:** Check symlink:
```bash
# Should show symlink to smpl_mesh directory
ls -la motion_annot_web/embodied_viz/data/

# If missing, create it
cd motion_annot_web/embodied_viz/data
ln -s ../../../output/physflow_v2_compare_iter1000/smpl_mesh smpl_mesh
```

---

## Part 7: Next Steps

### For Analysis

1. **Extract 3D Joint Positions**
   - Use SMPL model forward kinematics
   - Available: `scripts/embodied/batch_npz_to_smpl_joints.py`
   - Outputs skeleton joint positions instead of mesh parameters

2. **Compute Motion Metrics**
   - Velocity, acceleration from joint positions
   - Ground contact detection
   - Motion style classification

3. **Compare Variants**
   - Visualize finetuned vs pretrained
   - Compare raw vs RL-optimized

### For Deployment

1. **Generate Video Previews**
   ```bash
   # Render each JSON as a video using Three.js + ffmpeg
   python3 scripts/render_json_to_video.py \
       --json-dir output/physflow_v2_compare_iter1000/smpl_mesh \
       --output-dir output/preview_videos
   ```

2. **Create Motion Dataset**
   - Index converted files for machine learning
   - Generate embeddings for similarity search
   - Build motion retrieval system

3. **Integrate with Evaluation Pipeline**
   - Use with eval_dashboard
   - Include in score_m2m_refine comparison
   - Export to completion_apps

---

## Part 8: Reference Documentation

**Files delivered with this project:**

| File | Purpose |
|------|---------|
| `scripts/embodied/batch_npz_to_smpl_mesh_json.py` | Main conversion script |
| `scripts/embodied/batch_npz_to_smpl_joints.py` | Joint extraction alternative |
| `NPZ_TO_SMPL_MESH_JSON_CONVERSION_PIPELINE.md` | Detailed specification |
| `NPZ_TO_SMPL_CONVERSION_QUICK_REFERENCE.txt` | Quick lookup guide |
| `CONVERSION_COMPLETE.md` | Status report with statistics |
| `NPZ_FILES_INVENTORY.txt` | List of all 76 files |

---

## Summary

✅ **76 motion files successfully converted**
- Input: motion_135 NPZ format (HyMotion)
- Output: SMPL-H mesh JSON (web-ready)
- Total: 6,733 frames, 12.9 MB
- Success rate: 100%

✅ **Web viewer ready to use**
- Start with: `cd motion_annot_web/embodied_viz && python3 app.py --port 8095`
- Access: `http://localhost:8095`
- All 76 motions available for preview

✅ **Production-ready pipeline**
- Can be applied to additional datasets
- Fully documented and tested
- Ready for integration into evaluation pipelines

---

**Last Updated**: 2026-05-25 12:21 UTC  
**Status**: ✅ Complete and Verified
