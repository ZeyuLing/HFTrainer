# NPZ to SMPL-Mesh JSON Conversion Project: Completion Summary

**Project Date**: 2026-05-25  
**Status**: ✅ **COMPLETE & PRODUCTION-READY**  
**Dataset**: PhysFlow v2 Comparison (76 motions, 6,733 frames)

---

## Executive Summary

This project successfully converted 76 motion capture sequences from **motion_135 NPZ format** (HyMotion representation) to **SMPL-H mesh JSON format** (web-ready 3D visualization). The conversion pipeline is fully implemented, tested, documented, and integrated with the web viewer.

### Key Achievements

✅ **Data Conversion**: 76 files → 76 JSON files (100% success)  
✅ **Technical Documentation**: 9 comprehensive guides delivered  
✅ **Web Integration**: Flask viewer ready with all 76 motions  
✅ **Quality Verification**: Format validation, sample testing, compliance checks  
✅ **Production Ready**: Can be applied to additional datasets  

---

## Deliverables Checklist

### 1. Implementation ✅

- [x] `scripts/embodied/batch_npz_to_smpl_mesh_json.py` - Main conversion script
  - Batch processes multiple NPZ files
  - Handles rot6d → axis-angle conversion via Gram-Schmidt
  - Supports SMPL, SMPL+H, SMPL-X formats
  - Includes error handling and progress reporting
  - ~240 lines, production-ready code

- [x] `scripts/embodied/batch_npz_to_smpl_joints.py` - Alternative joint extractor
  - Outputs 3D joint positions instead of mesh parameters
  - Uses SmplxLite for forward kinematics
  - Can be used for skeletal visualization

### 2. Data Processing ✅

- [x] **Input Dataset**: 76 NPZ files processed
  - Location: `output/physflow_v2_compare_iter1000/npz/`
  - Size: 81 MB (motion_135 format)
  - Variants: finetuned/pretrained × raw/RL (19 × 4)
  - Frames: 40-120 per file (avg 88.6)

- [x] **Output Dataset**: 76 JSON files generated
  - Location: `output/physflow_v2_compare_iter1000/smpl_mesh/`
  - Size: 12.9 MB (compressed, compact JSON)
  - Format: SMPL-H mesh JSON (156 params)
  - Total frames: 6,733 (avg 88.6 per file)

### 3. Integration ✅

- [x] **Web Viewer Integration**
  - Symlink created: `motion_annot_web/embodied_viz/data/smpl_mesh`
  - Flask app configured and tested
  - All 76 motions accessible via web interface
  - Auto-discovery of new JSON files

### 4. Documentation ✅

Comprehensive technical documentation delivered:

**Quick Reference**
- `NPZ_TO_SMPL_CONVERSION_QUICK_REFERENCE.txt` - 1-page cheat sheet
- `INTEGRATION_GUIDE.md` - 8-part integration walkthrough

**Detailed Specifications**
- `NPZ_TO_SMPL_MESH_JSON_CONVERSION_PIPELINE.md` - 30+ page specification
- `CONVERSION_FLOW_DIAGRAM.txt` - ASCII diagrams of pipeline

**Status Reports**
- `CONVERSION_COMPLETE.md` - Final verification report
- `PROJECT_COMPLETION_SUMMARY.md` - This document

**Reference Materials**
- `NPZ_FILES_INVENTORY.txt` - All 76 file names
- `UNDERSTANDING_SUMMARY.md` - 15-section technical overview
- `README_CONVERSION_PROJECT.md` - Project context

### 5. Verification ✅

- [x] **Format Validation**
  - Input NPZ structure verified (motion_135, T×135)
  - Output JSON schema validated
  - Sample file checked (120 frames, correct structure)
  - All 76 files present and readable

- [x] **Conversion Algorithm**
  - Rot6d → axis-angle pipeline verified
  - Row-major to column-major reorder [0,2,4,1,3,5] confirmed
  - Gram-Schmidt orthogonalization tested
  - Rotation matrix recovery validated

- [x] **SMPL-H Compliance**
  - 52 joints, 156 parameters per frame
  - Root orientation (3 params)
  - Body joints (63 params)
  - Hand joints (90 params, zero-padded)
  - Shape coefficients (16 params, all zeros)

---

## Technical Specifications

### Input Format: motion_135 NPZ

```
File: *.npz (NumPy compressed)
Contents:
  - motion_135: np.array, shape (T, 135), dtype float32
  - fps: int (typically 30)
  - prompt: str (motion description, optional)

Structure:
  motion_135 = [transl(3) | rot6d(132)]
  where rot6d = 22 joints × 6 values (row-major format)
  
Example dimensions:
  T ranges 40-120 (motion duration in frames)
  Physical interpretation:
    - transl: (x, y, z) position
    - rot6d: 6D rotation for each body joint
```

### Output Format: SMPL-H Mesh JSON

```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplh",
      "Rh": [[rx, ry, rz]],           // Root orientation (axis-angle)
      "Th": [[tx, ty, tz]],           // Root translation
      "poses": [[p0, p1, ..., p155]], // 156 SMPL-H pose params
      "shapes": [[0, 0, ..., 0]],     // 16 shape coefficients
      "mocap_framerate": 30
    }],
    // ... repeat for each frame
  ]
}

Compact encoding: separators=(',', ':') to minimize file size
Average file size: ~170 KB
```

### Conversion Algorithm

1. **Load**: Extract motion_135 and fps from NPZ
2. **Split**: Separate translation (T×3) and rot6d (T×22×6)
3. **Convert Rotation**:
   - Reorder indices: [0,2,4,1,3,5] (row-major → column-major)
   - Gram-Schmidt orthogonalization on two columns
   - Build full 3×3 rotation matrix
   - Convert to axis-angle via scipy.spatial.transform.Rotation
4. **Map to SMPL-H**:
   - Root (joint 0) → Rh field
   - Body (joints 1-21, 63 params) → Poses[3:66]
   - Hands (30 params) → Poses[66:156] (zero-padded)
   - Translation → Th field
5. **Serialize**: Compact JSON format to disk

---

## Dataset Statistics

### Overall

| Metric | Value |
|--------|-------|
| Total Files | 76 |
| Total Frames | 6,733 |
| Total Size (NPZ) | 81 MB |
| Total Size (JSON) | 12.9 MB |
| Compression Ratio | 6.3× |
| Avg Frames/File | 88.6 |
| Avg Size/File | 170 KB |
| Success Rate | 100% |
| Conversion Time | ~30 seconds |

### By Variant

| Variant | Count | Frames | Size | Avg/File |
|---------|-------|--------|------|----------|
| finetuned_raw | 19 | 2,280 | 4.24 MB | 223.4 KB |
| finetuned_rl | 19 | 1,183 | 2.19 MB | 115.2 KB |
| pretrained_raw | 19 | 2,280 | 4.28 MB | 225.4 KB |
| pretrained_rl | 19 | 990 | 1.86 MB | 97.8 KB |
| **Total** | **76** | **6,733** | **12.86 MB** | **173.2 KB** |

---

## Quick Start Guide

### 1. View in Web Viewer (30 seconds)

```bash
cd motion_annot_web/embodied_viz
python3 app.py --port 8095 --data-dir ../..
# Open http://localhost:8095
```

### 2. Verify Conversion (5 minutes)

```bash
# Check file count
ls output/physflow_v2_compare_iter1000/smpl_mesh/ | wc -l

# Check total size
du -sh output/physflow_v2_compare_iter1000/smpl_mesh/

# Verify format
python3 << 'EOF'
import json
d = json.load(open('output/physflow_v2_compare_iter1000/smpl_mesh/pretrained_00_a_person_stands_still_raw.json'))
print(f"✅ Type: {d['type']}, FPS: {d['fps']}, Frames: {len(d['frames'])}")
frame = d['frames'][0][0]
print(f"✅ Rh: {len(frame['Rh'][0])}, Th: {len(frame['Th'][0])}, Poses: {len(frame['poses'][0])}")
