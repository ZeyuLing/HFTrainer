# Quick Reference — SMPL Mesh Visualization

## File Locations

```
scripts/embodied/
├── batch_npz_to_smpl_mesh_json.py      ← SMPL mesh JSON (annotation)
├── batch_pipeline_to_web.py             ← Robot motion JSON (comparison)
├── batch_t2m_to_embodied.py             ← Full pipeline: T2M → robot motion
├── convert_cache_to_json.py             ← .motion/.pt → JSON
└── pipeline_motion_to_robot.py          ← Chains PyRoki retargeting

motion_annot_web/score_m2m_refine/
├── static/
│   ├── scripts3d/
│   │   ├── load_smpl.js                 ← SkinnedMesh renderer
│   │   ├── create_scene.js
│   │   ├── draw_skeleton.js
│   │   └── export_motion.js
│   ├── assets/
│   │   ├── dump_smplh/                  ← SMPL+H binaries (~1.6GB)
│   │   ├── dump_smplx/                  ← SMPL-X binaries (~2.5GB)
│   │   └── three/                       ← Three.js library
│   └── three/
│       ├── three.module.js
│       ├── jsm/                         ← Three.js modules
│       └── ... (controls, loaders, etc.)
├── templates/
│   ├── vis_smpl_preview.html            ← Standalone SMPL viewer
│   ├── record.html                      ← Annotation UI with SMPL
│   ├── view_record.html
│   ├── admin_review.html
│   └── review_record.html
└── score_m2m_web.py                     ← Flask app
```

---

## Quick Commands

### 1. Convert motion_135 NPZ to SMPL Mesh JSON
```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/embodied_t2m_v4/data/npz \
    --output-dir output/embodied_t2m_v4/data/smpl_mesh \
    --smpl-type smplh \
    --gender neutral \
    --skip-existing
```

**Output:** SMPL JSON with Rh, Th, poses, shapes per frame

### 2. Full Pipeline: T2M → Motion_135 → Robot JSON
```bash
python scripts/embodied/batch_t2m_to_embodied.py \
    --prompts "a person walks forward" "a person jumps" \
    --output-dir output/embodied_v6/ \
    --smooth \
    --max-motions 10
```

**Output:** Robot motion JSON (29 DOF G1 joints)

### 3. Convert Existing NPZ to Robot JSON (skip T2M)
```bash
python scripts/embodied/batch_pipeline_to_web.py \
    --npz-dir work_dirs/motions/npz \
    --output-dir output/web_motions/ \
    --quality-filter \
    --max-motions 50
```

---

## Format Comparison

### SMPL Mesh JSON (for annotation)
- **Use:** score_m2m_refine web app, SMPL body visualization
- **Rendering:** Three.js SkinnedMesh with skinning weights
- **Key fields:** `Rh` (root rotation), `Th` (translation), `poses` (joint angles), `shapes` (16 body shape coefs)
- **Models:** SMPL, SMPL+H (52 joints), SMPL-X (55 joints)

```json
{
  "type": "frames",
  "fps": 30,
  "frames": [[{
    "id": 0,
    "gender": "neutral",
    "smpl_type": "smplh",
    "Rh": [[rx, ry, rz]],
    "Th": [[tx, ty, tz]],
    "poses": [[p0, p1, ...]],  // 1×156 for smplh
    "shapes": [[0, 0, ..., 0]],  // 1×16 shape coefficients
    "mocap_framerate": 30
  }], ...]
}
```

### Robot Motion JSON (for embodied viewer)
- **Use:** G1 humanoid robot simulation, embodied comparison
- **Rendering:** Forward kinematics skeleton from DOF angles
- **Key fields:** `dof_pos` (29 joint angles), `root_pos`, `root_quat`
- **Models:** G1 humanoid (29 DOFs, 33 rigid bodies)

```json
{
  "fps": 50,
  "num_frames": 120,
  "joint_names": ["left_hip_pitch_joint", ...],  // 29 names
  "root_body_index": 0,
  "frames": [{
    "root_pos": [x, y, z],
    "root_quat": [x, y, z, w],
    "dof_pos": [a0, a1, ..., a28]  // 29 joint angles
  }, ...]
}
```

---

## SMPL Asset Files

Location: `motion_annot_web/score_m2m_refine/static/assets/`

### Binary Files per Model
- `v_template.bin` — Neutral mesh template vertices (6890×3 float32)
- `faces.bin` — Triangle indices (13776×3 uint16)
- `skinWeights.bin` — Vertex skin blend weights (6890×4 float32)
- `skinIndice.bin` — Joint influence indices (6890×4 uint16)
- `j_template.bin` — Joint positions (52×3 float32)
- `shapeoffset_*.bin` — 16 body shape offsets per vertex (6890×3 each)
- `keypoints.bin` — COCO keypoints (25 points)

### Model Sizes
- **dump_smplh/** (SMPL+H) — ~1.6GB
- **dump_smplx/** (SMPL-X) — ~2.5GB
- **Gender variants:** dump_smplh_male/, dump_smplh_female/

---

## Key Python Functions

### load_smpl.js (JavaScript)
```javascript
// Load and render SMPL mesh
await load_smpl_with_shapes({
  shapes: [0, 0, ..., 0],        // 16 shape coefficients
  gender: 'neutral',
  smpl_type: 'smplh',
  poses: [p0, p1, ...],          // All joint poses
  Rh: [rx, ry, rz],              // Root rotation
  Th: [tx, ty, tz]               // Root translation
})
```

### batch_npz_to_smpl_mesh_json.py (Python)
```python
rot6d_to_axis_angle_np(rot6d)     # row-major rot6d → axis-angle
convert_single_npz(npz_path, smpl_type, gender)  # NPZ → SMPL JSON
```

### batch_t2m_to_embodied.py (Python)
```python
load_t2m_bundle(args)                # Load HyMotion T2M
run_t2m_inference(bundle, pipeline, text, frames)  # T2M inference
smooth_motion_135(motion_135)         # Markley quaternion smoothing
run_retarget_pipeline(npz_path, output_dir)  # PyRoki retarget
convert_cache_to_json(cache_path, json_path)  # Cache → JSON
```

### convert_cache_to_json.py (Python)
```python
convert_cache_to_json(cache_path, output_path, subsample=1)
```

---

## Integration Points

### T2M → motion_135 → Split Path

```
motion_135 NPZ (T, 135)
└── [transl: 3D] + [22×rot6d: 132D]
    │
    ├─→ Path A: batch_npz_to_smpl_mesh_json.py
    │   └─→ SMPL Mesh JSON (Rh, Th, poses, shapes)
    │       └─→ Web: load_smpl.js → Three.js SkinnedMesh
    │
    └─→ Path B: batch_t2m_to_embodied.py
        ├─→ PyRoki Retargeting
        └─→ Robot Motion JSON (dof_pos, root_pos/quat)
            └─→ Web: Custom Three.js skeleton viewer
```

---

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| SMPL assets 404 | Missing dump_smplh/ binaries | Download/deploy asset files to static/assets/ |
| rot6d conversion NaN | Invalid row-major format | Verify [R00, R01, R10, R11, R20, R21] order |
| Shapes validation error | Non-numeric or wrong size | Pad shapes to 16 elements with 0.0 |
| PyRoki timeout | CPU retarget too slow | Increase timeout or use GPU |
| Motion falls over | Retarget failed | Check foot contact weights in optimization |

---

## Related Documentation

- `SMPL_SOMA_RETARGETING_FIX.md` — SOMA bone retargeting issues
- `FORMAT_SPECIFICATION.md` — Detailed format specifications
- `VERIFICATION_SUMMARY.txt` — Validation results

**Last Updated:** 2026-05-14
