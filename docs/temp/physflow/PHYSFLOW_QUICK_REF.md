# PhysFlow Scripts - Quick Reference

## 🎯 Main Demo Data Generation Script

**`physflow_eval_and_export.py`** ⭐
- **Purpose:** Generate eval demo data with SMPL mesh exports
- **Input:** Original T2M checkpoint + PhysFlow-trained checkpoint
- **Output:** `output/physflow/eval_demo/` with motion_135 NPZ + SMPL mesh JSON
- **Key Data:** Curriculum-based prompts (standing → walking → complex)

```bash
python3 scripts/embodied/physflow_eval_and_export.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval_demo \
    --quick
```

---

## 🔧 Core Supporting Scripts

### Curriculum & Training Setup
- **`physflow_curriculum.py`**: Defines PHYSFLOW_LEVELS (difficulty progression, prompts)
- **`physflow_precompute_text.py`**: Pre-compute text embeddings (Qwen3-8B + CLIP-L)

### Physics Correction Pipeline
- **`physflow_physics_oracle.py`**: MuJoCo PD-tracking wrapper → physics_correct(motion_135)
- **`run_smpl_physics_sim.py`**: Core MuJoCo simulation engine

### Model Training
- **`physflow_trainer.py`**: Main training loop (generate → correct → fine-tune)

### Evaluation
- **`physflow_evaluate.py`**: Comprehensive before/after evaluation

---

## 📦 Format Conversion Scripts

### motion_135 → Web Formats
- **`batch_npz_to_smpl_mesh_json.py`**: motion_135 NPZ → SMPL mesh JSON (Three.js)
- **`batch_npz_to_smpl_joints.py`**: motion_135 NPZ → Joint positions JSON (skeleton-only)

### motion_135 → Robot Retargeting
- **`motion135_to_pyroki_keypoints.py`**: motion_135 → PyRoki keypoints (18-point format)
- **`motion135_to_smplx.py`**: motion_135 → SMPL-X NPZ (GMR retargeting)

### End-to-End Robot Pipeline
- **`pipeline_motion_to_robot.py`**: motion_135 NPZ → PyRoki retargeted robot motion

---

## 📊 Output Directory Structure

```
output/physflow/eval_demo/
├── data/
│   ├── npz/                    # motion_135 format (T, 135)
│   │   ├── original_000_*.npz
│   │   ├── original_001_*.npz
│   │   ├── physflow_000_*.npz
│   │   └── physflow_001_*.npz
│   ├── smpl_mesh/              # Web-viewable kinematic meshes
│   │   ├── original_000_*.json
│   │   └── physflow_000_*.json
│   ├── smpl_mesh_physics/      # Physics-corrected meshes
│   │   ├── original_000_*.json
│   │   └── physflow_000_*.json
│   ├── robot_motion/           # (Optional) PyRoki retargeted
│   └── meta/                   # Per-motion metadata
├── metrics.json                # Summary statistics
├── batch_retarget.sh           # Robot retargeting commands
└── retarget_log.txt
```

---

## 🔄 Data Flow

```
TEXT PROMPTS (curriculum)
    ↓
T2M MODEL (original + trained)
    ↓
GENERATE motion_135 (T, 135)
    ↓
PHYSICS ORACLE: motion_135 → motion_135_phys
    ↓
SAVE:
  • NPZ (motion_135)
  • SMPL mesh JSON (web)
  • Physics-corrected SMPL mesh JSON
    ↓
OPTIONAL: robot retargeting → .motion files
```

---

## 📐 Key Data Formats

### motion_135 (core format)
```python
# Shape: (T, 135)
# [transl(3) + 22*rot6d(132)]
# - transl: root translation (Y-up)
# - rot6d: 22 joints, 6D rotation (row-major), Gram-Schmidt → axis-angle
```

### SMPL Mesh JSON (web format)
```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "Rh": [[rx, ry, rz]],
      "Th": [[tx, ty, tz]],
      "poses": [[p0, p1, ...]],  // 156-dim for SMPL+H
      "shapes": [[s0, s1, ...]], // 16-dim
      "gender": "neutral",
      "smpl_type": "smplh"
    }]
  ]
}
```

### PyRoki Keypoints (robot format)
```python
{
    'positions': (T, 18, 3),           # 18 keypoints
    'orientations': (T, 18, 3, 3),     # Rotation matrices
    'left_foot_contacts': (T, 2),      # Binary contact flags
    'right_foot_contacts': (T, 2),
}
```

---

## ✅ Curriculum Levels Used in Demo

1. **Standing** (3s, 90 frames) - basic stability
2. **Weight shifts** (3s) - balance transitions
3. **Slow walking** (3s) - locomotive motion
4. **Normal walking** (4s) - standard gait
5. **Complex movements** - waves, jumps, turns, etc.

Each level has multiple prompts; quick mode uses 2 prompts per level from levels 0-2.

---

## 🚀 Complete Workflow

```
1. Pre-compute embeddings
   → physflow_precompute_text.py
   → output/physflow/text_embeddings.pt

2. Train PhysFlow (optional)
   → physflow_trainer.py
   → output/physflow/model_final.pt

3. Generate eval demo data
   → physflow_eval_and_export.py
   → output/physflow/eval_demo/

4. (Optional) Export robot motions
   → pipeline_motion_to_robot.py
   → robot .motion files
```

---

## 📁 Key Files to Check

- **Eval demo output:** `/apdcephfs/.../output/physflow/eval_demo/data/npz/`
- **SMPL meshes:** `/apdcephfs/.../output/physflow/eval_demo/data/smpl_mesh/`
- **Metrics:** `/apdcephfs/.../output/physflow/eval_demo/metrics.json`
- **Curriculum prompts:** `physflow_curriculum.py` → `PHYSFLOW_LEVELS`

