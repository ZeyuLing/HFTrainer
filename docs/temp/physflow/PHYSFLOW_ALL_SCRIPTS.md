# Complete Scripts List: scripts/embodied/

**Total Scripts:** 51 Python files in `/scripts/embodied/` directory

---

## Scripts Related to PhysFlow & Demo Data Generation

| Script | Size | Purpose | Key Output |
|--------|------|---------|-----------|
| **physflow_eval_and_export.py** ⭐ | 19.6KB | **PRIMARY**: Generate eval demo data (original + trained model comparison) | eval_demo/data/{npz, smpl_mesh, smpl_mesh_physics, sim_stats, meta} |
| **physflow_evaluate.py** | 22.2KB | Compare original vs trained model with metrics only (no web export) | metrics.json, demos/{before,after,physics}/ |
| **physflow_trainer.py** | 39KB | Main training loop: generate, physics-correct, fine-tune | model_final.pt, training log |
| **physflow_physics_oracle.py** | 11.5KB | MuJoCo physics correction wrapper | motion_135_phys (corrected motion) |
| **physflow_curriculum.py** | 6.3KB | Defines PHYSFLOW_LEVELS (27 curriculum prompts, 5 levels) | PHYSFLOW_LEVELS constant |
| **physflow_precompute_text.py** | 3.6KB | Pre-compute text embeddings for all curriculum prompts | text_embeddings.pt (~226MB) |

---

## Scripts for Motion Format Conversion

| Script | Size | Purpose | Input → Output |
|--------|------|---------|----------------|
| **motion135_to_smplx.py** | 5.1KB | Convert motion_135 to SMPL-X NPZ for GMR retargeting | motion_135 NPZ → SMPL-X NPZ |
| **motion135_to_pyroki_keypoints.py** | ? | Convert motion_135 to PyRoki keypoint format for G1 robot | motion_135 → PyRoki keypoints |
| **batch_npz_to_smpl_mesh_json.py** | ? | Convert motion_135 NPZ to SMPL mesh JSON for Three.js | motion_135 NPZ → SMPL mesh JSON |
| **batch_npz_to_smpl_joints.py** | ? | Extract SMPL joint positions from NPZ | NPZ → joint coordinates |
| **hymotion_to_smplx.py** | ? | Convert HyMotion format to SMPL-X | HyMotion format → SMPL-X |
| **convert_cache_to_json.py** | ? | Convert ProtoMotions cache (.pt/.motion) to Three.js JSON | .pt/.motion → JSON |

---

## Batch Processing Scripts

| Script | Size | Purpose | Workflow |
|--------|------|---------|----------|
| **batch_t2m_to_embodied.py** | 29KB | Full pipeline: text → T2M → motion_135 → Embodied G1 | Text prompts → T2M → PyRoki retarget → JSON + renders |
| **batch_retarget_parallel.py** | ? | Parallel PyRoki retargeting for multiple motions | motion_135 → .motion files (parallel) |
| **batch_pipeline_to_web.py** | ? | Convert pipeline outputs to web-viewable JSON format | .motion → Three.js JSON |
| **_batch_compare.py** | ? | Compare multiple motion versions side-by-side | Multiple motion files → comparison metrics |

---

## Rendering & Visualization Scripts

| Script | Size | Purpose | Output |
|--------|------|---------|--------|
| **render_tracker_headless.py** | ? | Headless rendering of motions (reference + tracked ONNX) | MP4 videos (reference + tracked modes) |
| **run_tracker_export.py** | ? | Export ONNX tracker predictions | Tracker output files |
| **run_g1_rl_tracker_export.py** | ? | Export G1 RL tracker predictions | G1 tracker output |
| **batch_pipeline_to_web.py** | ? | Pipeline outputs → web visualization JSON | Web-ready JSON |

---

## Physics & Simulation Scripts

| Script | Size | Purpose | Simulation Pipeline |
|--------|------|---------|-------------------|
| **run_smpl_physics_sim.py** | ? | MuJoCo PD-tracking physics simulation (Y-up ↔ Z-up conversion, FK, dynamics) | motion_135 → simulated motion (physics-corrected) |
| **physflow_physics_oracle.py** | 11.5KB | Wrapper around physics simulation | Simplified physics API |
| **diagnose_oscillation.py** | ? | Debug oscillation issues in physics simulation | Diagnostic analysis |
| **debug_sim_stability.py** | ? | Test physics simulation stability | Stability metrics |
| **test_pd_standing.py** | ? | Test PD controller for standing poses | PD tuning validation |
| **test_pd_standing_v2.py** | ? | Improved PD standing test | PD tuning v2 |
| **test_mujoco_euler.py** | ? | Test MuJoCo Euler angle handling | Euler angle validation |
| **verify_pipeline_integrity.py** | ? | Verify physics pipeline correctness | Pipeline health check |

---

## Debugging & Diagnostic Scripts

| Script | Size | Purpose | Output |
|--------|------|---------|--------|
| **debug_pose_diagnostic.py** | ? | Diagnose pose/skeleton issues | Diagnostic report |
| **debug_root_transform.py** | ? | Debug root transformation (Y-up ↔ Z-up) | Transform validation |
| **debug_transform_comparison.py** | ? | Compare transform methods side-by-side | Comparison metrics |
| **diag_actuator.py** | ? | Diagnose actuator/controller issues | Actuator diagnostics |
| **check_motion_coords.py** | ? | Verify coordinate system consistency | Coordinate system report |
| **test_height_analysis.py** | ? | Analyze motion height variations | Height statistics |
| **test_single_joint_debug.py** | ? | Debug single joint behavior | Joint diagnostics |
| **test_joint_reorder.py** | ? | Test joint ordering/reordering | Joint order validation |
| **test_body_joint_combos.py** | ? | Test different joint combinations | Combination analysis |
| **test_definitive_fk.py** | ? | Test forward kinematics implementation | FK validation |
| **test_root_rotation_fix.py** | ? | Test root rotation corrections | Rotation fix validation |
| **test_e2e_v6.py** | ? | End-to-end test for V6 pipeline | E2E validation report |

---

## Comparison & Analysis Scripts

| Script | Size | Purpose | Comparison Type |
|--------|------|---------|-----------------|
| **full_smoothness_cmp.py** | ? | Comprehensive smoothness comparison | Smoothness metrics across methods |
| **quick_smoothness_cmp.py** | ? | Quick smoothness comparison | Fast smoothness test |
| **_compare_v4_v5.py** | ? | Compare V4 vs V5 pipeline versions | Pipeline version comparison |
| **_compare_caches.py** | ? | Compare cache files | Cache content comparison |

---

## Retargeting & Format Conversion Scripts

| Script | Size | Purpose | Workflow |
|--------|------|---------|----------|
| **pipeline_motion_to_robot.py** | ? | Complete retargeting pipeline (V6 PyRoki) | motion_135 → PyRoki keypoints → G1 joints → .motion file |
| **gmr_retarget_headless.py** | ? | GMR retargeting (no display) | motion_135 → GMR retarget → output |
| **gmr_to_protomotions.py** | ? | Convert GMR retarget output to ProtoMotions format | GMR output → .motion |
| **batch_retarget_parallel.py** | ? | Parallel retargeting of multiple motions | Multiple motions → retargeted output |

---

## Utility & Preprocessing Scripts

| Script | Size | Purpose | Utility |
|--------|------|---------|---------|
| **generate_smpl_mesh_vertices.py** | ? | Pre-compute SMPL mesh vertices | Mesh vertex cache |
| **rebuild_v4_from_motion.py** | ? | Rebuild V4 motion from existing motion data | Motion rebuild utility |
| **incremental_v4_pipeline.py** | ? | Incremental processing for V4 pipeline | Incremental processing |
| **verify_transform_euler.py** | ? | Verify Euler angle transformations | Transform verification |
| **t2m_verify.py** | ? | Verify T2M model outputs | T2M output validation |

---

## Summary by Category

### 🎯 **For PhysFlow Demo Data Generation (START HERE)**
1. `physflow_precompute_text.py` - Pre-compute embeddings
2. `physflow_eval_and_export.py` ⭐ - Generate eval_demo/
3. `physflow_evaluate.py` - Alternative (metrics-only)

### 🤖 **For Full Embodied Robot Pipeline**
1. `batch_t2m_to_embodied.py` - End-to-end pipeline
2. `pipeline_motion_to_robot.py` - PyRoki V6 retargeting
3. `render_tracker_headless.py` - Video rendering
4. `convert_cache_to_json.py` - Web export

### 📊 **For Physics Correction**
1. `physflow_trainer.py` - Training loop
2. `physflow_physics_oracle.py` - Physics wrapper
3. `run_smpl_physics_sim.py` - Core simulation

### 🔄 **For Format Conversion**
1. `motion135_to_smplx.py` - motion_135 → SMPL-X
2. `motion135_to_pyroki_keypoints.py` - motion_135 → PyRoki
3. `batch_npz_to_smpl_mesh_json.py` - motion_135 → Web JSON

### 🐛 **For Debugging/Validation**
- `test_*.py` (15 scripts) - Unit tests for various components
- `debug_*.py` (5 scripts) - Diagnostic utilities
- `verify_*.py` (2 scripts) - Validation utilities
- `*_cmp.py` (4 scripts) - Comparison utilities

---

## Motion Data Flow Diagram

```
Text Prompts
    ↓
[physflow_trainer.py / batch_t2m_to_embodied.py]
T2M Model (HyMotion)
    ↓
motion_201 (201-dim)
    ↓
motion_135 = motion_201[:, :135]
    ↓
[physflow_physics_oracle.py]
MuJoCo Physics Correction
    ↓
motion_135_phys (Y-up, 30fps)
    ↓
[branch 1: physflow_eval_and_export.py]
SMPL mesh JSON + NPZ files
    ├─→ output/physflow/eval_demo/
    ├─→ data/smpl_mesh/               (kinematic)
    ├─→ data/smpl_mesh_physics/       (physics-corrected)
    └─→ metrics.json
    
[branch 2: batch_t2m_to_embodied.py]
PyRoki Retargeting + G1 Robot
    ├─→ pipeline_motion_to_robot.py
    ├─→ G1 joint targets (qpos)
    ├─→ .motion files
    ├─→ render_tracker_headless.py
    ├─→ convert_cache_to_json.py
    └─→ Three.js JSON + videos
```

---

## Key Statistics

- **Total PhysFlow/Curriculum Scripts:** 6
- **Total Conversion/Format Scripts:** 6
- **Total Batch Processing Scripts:** 4
- **Total Physics/Simulation Scripts:** 8
- **Total Debug/Test Scripts:** 24
- **Total Utility Scripts:** 3

---

## File Sizes Reference

Readable sizes from disk:
- physflow_trainer.py: 39 KB (largest PhysFlow script)
- batch_t2m_to_embodied.py: 29 KB
- physflow_evaluate.py: 22.2 KB
- physflow_eval_and_export.py: 19.6 KB (PRIMARY FOR DEMO)
- physflow_curriculum.py: 6.3 KB
- physflow_precompute_text.py: 3.6 KB

---

## Execution Order for Demo Data Generation

```bash
# Step 1: Pre-compute text embeddings (one-time)
python3 scripts/embodied/physflow_precompute_text.py \
    --output output/physflow/text_embeddings.pt

# Step 2: Generate eval demo (original + trained model)
python3 scripts/embodied/physflow_eval_and_export.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/run_500iter/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval_demo \
    --quick

# Output location:
# output/physflow/eval_demo/
#   ├── data/npz/                    ← motion_135 NPZ files
#   ├── data/smpl_mesh/              ← SMPL mesh JSON
#   ├── data/smpl_mesh_physics/      ← Physics-corrected SMPL mesh JSON
#   ├── data/meta/                   ← Per-motion metadata
#   ├── data/sim_stats/              ← Simulation statistics
#   └── metrics.json
```

