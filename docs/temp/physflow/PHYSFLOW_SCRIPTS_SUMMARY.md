# PhysFlow Eval Demo Data Generation Scripts - Complete Reference

## Overview

The PhysFlow eval demo data is generated through a multi-stage pipeline that:
1. Generates motions from Text2Motion (T2M) models (original and PhysFlow-trained)
2. Applies physics corrections via MuJoCo simulation
3. Exports data in multiple formats (NPZ, SMPL mesh JSON, robot motion formats)

**Primary Output Directory:** `output/physflow/eval_demo/`

---

## Core PhysFlow Training Scripts

### 1. **physflow_precompute_text.py** (87 lines)
**Purpose:** Pre-compute text embeddings for curriculum prompts  
**Key Function:** Avoids loading large 8B text encoder during training; pre-encodes all curriculum prompts with Qwen3-8B + CLIP-L  
**Usage:**
```bash
python3 scripts/embodied/physflow_precompute_text.py \
    --output output/physflow/text_embeddings.pt
```
**Output:** Cached embeddings `.pt` file used in training/evaluation  
**Dependencies:** physflow_curriculum.py

---

### 2. **physflow_curriculum.py** (147 lines)
**Purpose:** Adaptive curriculum learning for PhysFlow  
**Key Features:**
- Organizes text prompts by difficulty (standing → walking → complex motions)
- Advances curriculum based on physics correction success rates
- Defines `PHYSFLOW_LEVELS` with prompt sets and success thresholds
- Manages prompt scheduling during training

**Curriculum Levels:**
- Level 0: Standing (static poses, weight shifts)
- Level 1: Slow walking
- Level 2: Complex movements (waves, jumps, turns)
- Level 3-7: Advanced motions

**Key Export:** `PHYSFLOW_LEVELS` dict used by trainer, evaluator, and exporter

---

### 3. **physflow_trainer.py** (993 lines) ⭐ MAIN TRAINING
**Purpose:** Core PhysFlow training loop with physics-grounded learning  
**Training Pipeline:**
1. Generate motion on-policy (current model + text prompt)
2. Correct via MuJoCo PD-tracking physics simulation
3. Fine-tune with flow matching loss against physics-corrected target

**Key Classes:**
- `PhysFlowTrainer`: Main trainer managing iterations, losses, checkpoints
- `load_bundle()`: Loads T2M model (HyMotion checkpoint)
- `motion_135_to_201()`: Converts motion_135 (135-dim) ↔ 201-dim (T2M format)

**Usage:**
```bash
python3 scripts/embodied/physflow_trainer.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --t2m-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --output-dir output/physflow \
    --num-iterations 2000 \
    --lr 2e-5
```

**Output:** `model_final.pt` in output_dir  
**Dependencies:** physflow_curriculum.py, physflow_physics_oracle.py, run_smpl_physics_sim.py

---

### 4. **physflow_physics_oracle.py** (328 lines) ⭐ PHYSICS CORRECTION
**Purpose:** MuJoCo PD-tracking wrapper for physics-corrected motion generation  
**Key Features:**
- Takes motion_135 (T, 135) input
- Runs physics simulation with joint tracking (PD controller)
- Returns physics-corrected motion_135 + detailed stats

**Main API:**
```python
oracle = PhysicsOracle("path/to/smpl_humanoid.xml")
motion_phys, stats = oracle.correct(motion_135_array)
# stats contains: completion_rate, joint_tracking_error, simulated_frames, etc.
if oracle.is_good_quality(stats):
    # Use for training target
```

**Coordinate Transforms:**
- Input/Output: Y-up (HyMotion format)
- Internal: Z-up (MuJoCo format)
- Transforms: `yup_to_zup()`, `zup_to_yup()`

**Dependencies:** run_smpl_physics_sim.py (for physics sim kernels)

---

## Evaluation & Export Scripts

### 5. **physflow_evaluate.py** (598 lines)
**Purpose:** Compare before/after PhysFlow training on curriculum prompts  
**Pipeline:**
1. Generate motions from original model
2. Generate motions from PhysFlow-trained model
3. Physics-correct both versions
4. Compute metrics (completion rate, tracking error, jerk, etc.)
5. Export per-prompt results and metrics.json

**Usage:**
```bash
python3 scripts/embodied/physflow_evaluate.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval \
    --quick  # Quick mode: 2 prompts per level (first 3 levels)
```

**Output Directory Structure:**
```
output_dir/
├── data/
│   ├── npz/              # motion_135 NPZ files
│   ├── smpl_mesh/        # SMPL mesh JSON (kinematic)
│   ├── smpl_mesh_physics/# SMPL mesh JSON (physics-corrected)
│   └── meta/             # per-motion metadata JSON
└── metrics.json          # summary comparison table
```

**Metrics Tracked:** completion_rate, tracking_error, correction_magnitude, jerk, gen_time

---

### 6. **physflow_eval_and_export.py** (517 lines) ⭐ MAIN DEMO EXPORTER
**Purpose:** Generate PhysFlow eval demo data with SMPL mesh exports for website  
**Pipeline:**
1. Generate motions from original model → NPZ + SMPL mesh JSON
2. Generate motions from PhysFlow-trained model → NPZ + SMPL mesh JSON
3. Physics-correct both → SMPL mesh JSON (physics-corrected)
4. Export metrics.json for dashboard

**Usage:** (PRODUCES output/physflow/eval_demo/)
```bash
python3 scripts/embodied/physflow_eval_and_export.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/run_500iter/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval_demo \
    --quick
```

**Output Directory Structure (eval_demo):**
```
output/physflow/eval_demo/
├── data/
│   ├── meta/             # metadata per motion
│   ├── npz/              # motion_135 NPZ (original_* and physflow_*)
│   ├── sim_stats/        # physics correction stats
│   ├── smpl_mesh/        # SMPL mesh JSON (kinematic)
│   ├── smpl_mesh_physics/# SMPL mesh JSON (physics-corrected)
│   └── robot_motion/     # PyRoki retargeted robot motions
├── metrics.json          # summary metrics
├── batch_retarget*.sh    # retargeting shell scripts
└── retarget_log*.txt     # retargeting logs
```

**Key Export Functions:**
- `motion_135_to_smpl_mesh_json()`: Converts motion_135 array → SMPL mesh JSON
- `save_motion()`: Saves NPZ + SMPL mesh JSON + metadata
- `save_physics_motion()`: Saves physics-corrected SMPL mesh JSON

**Dependencies:** physflow_trainer.py, physflow_physics_oracle.py, physflow_curriculum.py, batch_npz_to_smpl_mesh_json.py

---

## Format Conversion Scripts (motion_135 ↔ other formats)

### 7. **batch_npz_to_smpl_mesh_json.py**
**Purpose:** Batch-convert motion_135 NPZ → SMPL mesh JSON for web visualization  
**Input:** motion_135 NPZ files  
**Output:** SMPL mesh JSON (consumable by load_smpl.js)
**Key Process:**
- Extract rot6d from motion_135, reorder [0,2,4,1,3,5]
- Gram-Schmidt to axis-angle
- Generate SMPL poses, shapes, translation
- Format as Three.js compatible JSON

---

### 8. **batch_npz_to_smpl_joints.py**
**Purpose:** Batch-convert motion_135 NPZ → SMPL joint positions JSON  
**Output:** 22 world-space joint positions per frame (skeleton-only, not mesh)
**Use Case:** Lightweight web visualization (skeleton vs mesh)

---

### 9. **motion135_to_pyroki_keypoints.py** (540 lines)
**Purpose:** Convert motion_135 → PyRoki keypoints format for robot retargeting  
**Input:** motion_135 NPZ  
**Output:** .npy dict with:
```python
{
    'positions': (T, 18, 3),           # 18 keypoints (Z-up)
    'orientations': (T, 18, 3, 3),     # rotation matrices
    'left_foot_contacts': (T, 2),      # ankle, toebase contact
    'right_foot_contacts': (T, 2),     # ankle, toebase contact
}
```

**Pipeline:**
1. motion_135 (rot6d) → axis-angle
2. SMPL FK → world-space 22 joints
3. Y-up → Z-up coordinate transform
4. Extract 15 base keypoints + 3 auxiliary
5. Detect foot contacts (velocity + height threshold)

**Used By:** pipeline_motion_to_robot.py (for G1 robot retargeting)

---

### 10. **motion135_to_smplx.py**
**Purpose:** Convert motion_135 → SMPL-X NPZ format for GMR retargeting  
**Output:** NPZ with pose_body (63), root_orient (3), trans (3), betas (10), gender, fps
**Use Case:** Alternative retargeting pipeline (GMR-based, V5)

---

## Robot Motion Generation Scripts

### 11. **pipeline_motion_to_robot.py** (268 lines) ⭐ ROBOT EXPORT
**Purpose:** End-to-end pipeline: motion_135 NPZ → Robot motion (PyRoki V6)  
**Three-stage pipeline:**
1. motion_135 → PyRoki keypoints  (motion135_to_pyroki_keypoints.py)
2. PyRoki keypoints → Retargeted robot motion  (trajectory-level IK optimization)
3. Retargeted NPZ → ProtoMotions .motion  (robot motion cache format)

**Usage:**
```bash
python scripts/embodied/pipeline_motion_to_robot.py \
    --input output/physflow/eval_demo/data/npz/physflow_000_*.npz \
    --output output/physflow/eval_demo/data/robot_motion/ \
    --keep-intermediates
```

**Output:**
- `.motion` files (robot motion cache format)
- `qpos` reference trajectories (PD control targets)
- Intermediate PyRoki keypoint .npy files

**Dependencies:** motion135_to_pyroki_keypoints.py, PyRoki framework

---

### 12. **batch_retarget_parallel.py** (257 lines)
**Purpose:** Parallel batch retargeting using PyRoki (multiprocessing)  
**Motivation:** PyRoki is CPU-only, ~60min per motion → parallelize
**Usage:**
```bash
python3 scripts/embodied/batch_retarget_parallel.py \
    --npz-dir output/embodied_t2m_v4/data/npz \
    --output-dir output/embodied_t2m_v4/data/retarget \
    --workers 4
```

---

## Web Visualization & Rendering Scripts

### 13. **batch_pipeline_to_web.py** (295 lines)
**Purpose:** Batch convert HyMotion eval NPZ → ProtoMotions cache → JSON for web  
**Pipeline:** NPZ → Retarget → JSON
**Output:** Web-compatible motion JSON + cache files

---

### 14. **batch_t2m_to_embodied.py** (1029 lines)
**Purpose:** End-to-end batch pipeline: Text → T2M → Embodied G1 robot  
**Pipeline:**
1. Text prompts → HyMotion T2M (201-dim)
2. Extract motion_135 → save NPZ
3. motion_135 → PyRoki retarget → robot motion
4. .motion → render references + ONNX tracking
5. Generate web JSON + manifest

**Usage:**
```bash
python scripts/embodied/batch_t2m_to_embodied.py \
    --prompt-json output/embodied_comparison_v2/motion_text_mapping.json \
    --output-dir output/embodied_comparison_v3/ \
    --max-motions 5
```

---

## Physics Simulation & Tracking Scripts

### 15. **run_smpl_physics_sim.py** (1171 lines)
**Purpose:** Core physics simulation engine (MuJoCo PD-tracking)  
**Provides:**
- `yup_to_zup()`, `zup_to_yup()`: Coordinate transforms
- `smpl_to_qpos()`: SMPL → MuJoCo joint angles
- `qpos_to_smpl()`: MuJoCo joint angles → SMPL
- Physics simulation loop with collision detection, friction

**Used By:** physflow_physics_oracle.py, physflow_trainer.py

---

### 16. **run_smpl_rl_tracker.py** (1162 lines)
**Purpose:** RL-based motion tracking/correction (alternative to PD controller)  
**Output:** `output/physflow/eval_demo/smpl_mesh_rl/` (RL-corrected SMPL meshes)

---

### 17. **run_g1_rl_tracker_export.py** (729 lines)
**Purpose:** RL-based G1 robot motion tracking  
**Output:** `output/physflow/eval_demo/robot_mesh_rl/` (RL-corrected robot renderings)

---

## Debug & Test Scripts

### 18. **test_e2e_v6.py** (292 lines)
**Purpose:** End-to-end test of PyRoki V6 pipeline (motion_135 → robot motion)

---

### 19. **debug_root_transform.py** (488 lines)
**Purpose:** Debug coordinate transforms (Y-up ↔ Z-up)

---

### 20. **debug_pose_diagnostic.py** (375 lines)
**Purpose:** Diagnose pose/joint issues in physics simulation

---

### 21. **debug_sim_stability.py** (269 lines)
**Purpose:** Test physics simulation stability, contact detection

---

## Helper Scripts (Conversion/Utility)

### 22. **convert_cache_to_json.py**
**Purpose:** Convert cached motion format → JSON

---

### 23. **generate_smpl_mesh_vertices.py** (280 lines)
**Purpose:** Pre-compute SMPL mesh vertices for visualization

---

### 24. **gmr_retarget_headless.py** (260 lines)
**Purpose:** GMR-based retargeting (V5 pipeline)

---

### 25. **gmr_to_protomotions.py** (612 lines)
**Purpose:** Convert GMR-retargeted motion → ProtoMotions .motion format

---

### 26. **rebuild_v4_from_motion.py** (340 lines)
**Purpose:** Reconstruct V4 format from .motion files

---

### 27. **render_tracker_headless.py** (876 lines)
**Purpose:** Headless rendering of tracked motions

---

### 28. **run_tracker_export.py** (703 lines)
**Purpose:** Export tracker outputs (renders + qpos references)

---

## Data Format Reference

### motion_135 Format
```
Shape: (T, 135)
[transl(3) + 22*rot6d(132)]
- transl: (3,) - root translation (Y-up)
- rot6d: (22, 6) - 22 joints with 6D rotation representation (row-major)
```

### SMPL Mesh JSON (web-compatible)
```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplh",
      "Rh": [[rx, ry, rz]],           // 1×3 root orientation
      "Th": [[tx, ty, tz]],           // 1×3 translation
      "poses": [[p0, p1, ...]],       // 1×156 body joint axis-angles
      "shapes": [[s0, s1, ...]],      // 1×16 shape coefficients
      "mocap_framerate": 30
    }],
    ...
  ]
}
```

### PyRoki Keypoint Format
```python
{
    'positions': (T, 18, 3),           # Keypoint positions
    'orientations': (T, 18, 3, 3),     # Rotation matrices per keypoint
    'left_foot_contacts': (T, 2),      # Binary contact flags
    'right_foot_contacts': (T, 2),
}
```

---

## Complete PhysFlow Demo Data Generation Workflow

```
STEP 1: Pre-compute text embeddings
→ physflow_precompute_text.py
→ output/physflow/text_embeddings.pt

STEP 2: Train PhysFlow model
→ physflow_trainer.py
→ output/physflow/model_final.pt

STEP 3: Generate eval demo data
→ physflow_eval_and_export.py
  ├─ Load original + trained models
  ├─ Generate motions on curriculum prompts
  ├─ Physics-correct (oracle.correct())
  ├─ Save motion_135 NPZ + SMPL mesh JSON
  └─ Export metrics.json

OUTPUT: output/physflow/eval_demo/
├── data/
│   ├── npz/              # 20+ motion_135 files
│   ├── smpl_mesh/        # 20+ kinematic meshes
│   ├── smpl_mesh_physics/# 20+ physics-corrected meshes
│   ├── robot_motion/     # PyRoki retargeted robot motions (optional)
│   └── meta/
└── metrics.json
```

---

## Dependencies Graph

```
physflow_eval_and_export.py (MAIN)
├── physflow_trainer.py
│   ├── physflow_curriculum.py (defines prompts/levels)
│   ├── physflow_physics_oracle.py
│   │   └── run_smpl_physics_sim.py (physics kernels)
│   └── run_smpl_physics_sim.py
├── physflow_physics_oracle.py
│   └── run_smpl_physics_sim.py
├── batch_npz_to_smpl_mesh_json.py (export SMPL meshes)
└── physflow_curriculum.py (curriculum prompts)

Optional downstream processing:
pipeline_motion_to_robot.py
├── motion135_to_pyroki_keypoints.py
└── PyRoki trajectory-level IK
```

---

## Key File Locations in Output Directory

```
output/physflow/eval_demo/
├── data/npz/
│   ├── original_000_a_person_stands_still.npz           ← motion_135 format
│   ├── original_001_a_person_shifts_weight_from_le.npz
│   ├── physflow_000_a_person_stands_still.npz           ← trained model output
│   ├── physflow_001_a_person_shifts_weight_from_le.npz
│   └── ... (more prompts)
├── data/smpl_mesh/
│   ├── original_000_a_person_stands_still.json          ← kinematic mesh
│   ├── physflow_000_a_person_stands_still.json
│   └── ... (JSON format for Three.js web viewer)
├── data/smpl_mesh_physics/
│   ├── original_000_a_person_stands_still.json          ← physics-corrected
│   ├── physflow_000_a_person_stands_still.json
│   └── ...
├── data/meta/
│   ├── original_000_a_person_stands_still.json          ← per-motion metadata
│   └── ...
├── metrics.json                                          ← summary statistics
├── batch_retarget.sh                                     ← robot retargeting (optional)
└── retarget_log.txt
```

