# PhysFlow Eval Demo Data Generation Scripts

**Location:** `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/embodied/`

**Demo Data Output:** `output/physflow/eval_demo/`

## Overview

PhysFlow is a physics-grounded motion generation system that uses MuJoCo physics correction to fine-tune text-to-motion (T2M) models. The evaluation scripts generate demo data by:
1. Running T2M inference on curriculum prompts
2. Physics-correcting the generated motions
3. Converting to motion_135 format (T, 135) = translation(3) + 22×rot6d(132)
4. Exporting as NPZ files and SMPL mesh JSON for web visualization

---

## Core PhysFlow Scripts

### 1. **physflow_trainer.py** (39KB)
**Purpose:** Main training loop for PhysFlow physics correction

**Key Function:** `PhysFlowTrainer` class
- Generates motions on-policy using current model
- Runs MuJoCo physics simulation for correction
- Fine-tunes model with flow matching loss

**Motion Format Used:**
- Input/Output: motion_135 (T, 135) - [transl(3) + 22×rot6d(132)]
- Internal: motion_201 (T, 201) - full HyMotion format

**Key Dependencies:**
- `PhysicsOracle` for physics correction
- `PhysFlowCurriculum` for prompt scheduling
- HyMotion T2M model for generation

---

### 2. **physflow_physics_oracle.py** (11.5KB)
**Purpose:** Wraps MuJoCo physics simulation for motion correction

**Key Functions:**
- `PhysicsOracle.correct(motion_135)` → corrected motion_135
- `decode_motion_135_array()` - motion_135 → SMPL pose + translation
- `encode_to_motion_135()` - SMPL pose + translation → motion_135

**Physics Pipeline:**
```
motion_135 (Y-up)
  ↓ [decode]
SMPL pose + transl (Y-up)
  ↓ [Y→Z-up conversion]
MuJoCo qpos
  ↓ [MuJoCo PD-tracking sim]
Simulated qpos
  ↓ [Z→Y-up conversion]
SMPL pose + transl (Y-up)
  ↓ [encode]
motion_135_phys (Y-up)
```

**Output Metrics:**
- `completion_rate` - fraction of frames successfully simulated
- `tracking_error_rad` - mean joint tracking error
- `joint_tracking_error_rad` - average joint angle error
- `simulated_frames` / `total_frames`

---

### 3. **physflow_curriculum.py** (6.3KB)
**Purpose:** Adaptive prompt scheduling for training

**PHYSFLOW_LEVELS Definition:**
```python
PHYSFLOW_LEVELS = [
    Level 0 "standing":      90 frames, min_success_rate=0.8
    Level 1 "walking":       120 frames, min_success_rate=0.7
    Level 2 "upper_body":    90 frames, min_success_rate=0.6
    Level 3 "transitions":   150 frames, min_success_rate=0.5
    Level 4 "dynamic":       120 frames, min_success_rate=0.4
]
```

**Total Prompts:** 27 curriculum prompts across 5 difficulty levels

**Example Prompts per Level:**
- Level 0: "a person stands still", "a person shifts weight from left to right foot"
- Level 1: "a person walks forward slowly", "a person walks in a straight line"
- Level 2: "a person waves with their right hand", "a person raises both arms above their head"
- Level 3: "a person walks forward then turns around", "a person walks in a small circle"
- Level 4: "a person kicks with their right leg", "a person squats down and stands back up"

---

### 4. **physflow_precompute_text.py** (3.6KB)
**Purpose:** Pre-compute text embeddings for all curriculum prompts

**Usage:**
```bash
python3 scripts/embodied/physflow_precompute_text.py \
    --output output/physflow/text_embeddings.pt \
    --dtype float16 \
    --device cuda
```

**Output:**
- Cache file: `output/physflow/text_embeddings.pt` (~226MB for 27 prompts)
- Embeddings: Qwen3-8B + CLIP-L (vtxt: (1,1,768) + ctxt: (1,512,4096))

**Benefits:**
- Avoids loading 8B text encoder during training (OOM risk)
- ~220MB cache for 27 unique prompts
- Used by both trainer and evaluation scripts

---

## Evaluation & Export Scripts

### 5. **physflow_evaluate.py** (22.2KB)
**Purpose:** Compare original vs PhysFlow-trained model with metrics

**Main Function:** `run_evaluation()`

**Pipeline:**
1. Generate motions from **original model** (before PhysFlow training)
2. Generate motions from **trained model** (after PhysFlow fine-tuning)
3. Compute physics metrics for each:
   - Completion rate
   - Joint tracking error
   - Motion smoothness (jerk)
   - Physics correction magnitude

**Output Structure:**
```
output/physflow/eval/
  ├── metrics.json                 # Summary comparison table
  ├── demos/
  │   ├── before/                  # Original model motions (NPZ)
  │   ├── after/                   # PhysFlow model motions (NPZ)
  │   └── physics/                 # Physics-corrected versions (NPZ)
  └── training_analysis.json       # Training log analysis
```

**Usage:**
```bash
# Full evaluation (all curriculum prompts)
python3 scripts/embodied/physflow_evaluate.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/run_500iter/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval

# Quick test (1 prompt per level)
python3 scripts/embodied/physflow_evaluate.py ... --quick
```

**Metrics Computed:**
- `mean_speed` - translation velocity (smoothness proxy)
- `jerk_magnitude` - 3rd derivative of rotation (smoothness)
- `rotation_range` - range of motion
- `completion_rate` - physics sim success
- `tracking_error_rad` - joint angle deviation

---

### 6. **physflow_eval_and_export.py** (19.6KB) ⭐ **PRIMARY FOR DEMO DATA**
**Purpose:** Full evaluation + SMPL mesh JSON export for website visualization

**This is the script that generates the actual eval_demo data!**

**Main Features:**
1. Generates motions from original & trained models
2. Exports **NPZ files** (motion_135 format)
3. Exports **SMPL mesh JSON** for Three.js web viewer
4. Runs physics correction and exports corrected mesh JSON
5. Extracts simulation statistics

**Output Structure:**
```
output/physflow/eval_demo/
  ├── data/
  │   ├── npz/                     # motion_135 NPZ files
  │   ├── smpl_mesh/               # SMPL mesh JSON (kinematic)
  │   ├── smpl_mesh_physics/       # SMPL mesh JSON (physics-corrected)
  │   ├── meta/                    # Per-motion metadata JSON
  │   ├── sim_stats/               # Physics simulation statistics
  │   └── retarget/                # PyRoki retarget outputs
  ├── metrics.json                 # Summary metrics table
  └── batch_report.json
```

**NPZ File Format:**
```python
motion_135 = npz['motion_135']  # (T, 135) float32
# Layout: [0:3] translation + [3:135] 22×rot6d (row-major)
#   rot6d order: [R00,R01, R10,R11, R20,R21] (row-major)
```

**SMPL Mesh JSON Format:**
```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplh",
      "Rh": [root_orient_axis_angle],     # (3,) rotation in axis-angle
      "Th": [transl],                      # (3,) translation
      "poses": [poses_vector],             # (156,) = root(3) + body(63) + hands(90)
      "shapes": [[betas]],                 # (16,) shape coefficients (zeros)
      "mocap_framerate": 30
    }]
  ]
}
```

**Metrics Exported:**
- `completion_rate` - physics sim success percentage
- `tracking_error_rad` - joint angle deviation
- `correction_magnitude` - how much physics changed the motion
- `jerk` - motion smoothness
- `gen_time` - generation time per motion

**Usage:**
```bash
python3 scripts/embodied/physflow_eval_and_export.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/run_500iter/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval_demo \
    --quick                          # 2 prompts per level, 3 levels
```

**View in Browser:**
```bash
python3 motion_annot_web/embodied_viz/app.py \
    --data-dir output/physflow/eval_demo \
    --port 8095
```

---

## Related Conversion & Utility Scripts

### 7. **motion135_to_smplx.py** (5.1KB)
**Purpose:** Convert motion_135 to SMPL-X NPZ format for GMR retargeting

**Input:** motion_135 NPZ (T, 135)
```
Layout: [0:3] transl + [3:135] 22×rot6d (row-major)
```

**Output:** SMPL-X NPZ with keys:
```
pose_body:           (T, 63)   - body joints in axis-angle
root_orient:         (T, 3)    - root in axis-angle
trans:               (T, 3)    - translation
betas:               (10,)     - shape (zeros)
gender:              "neutral"
mocap_frame_rate:    30
```

**Usage:**
```bash
python3 scripts/embodied/motion135_to_smplx.py \
    input_motion_135.npz \
    output_smplx.npz \
    --fps 30
```

---

### 8. **batch_t2m_to_embodied.py** (29KB)
**Purpose:** Full pipeline: Text → T2M → motion_135 → Embodied G1 Robot

**Comprehensive Pipeline:**
1. T2M inference → motion_135 NPZ
2. Optional post-processing smoothing (Markley quaternion + Savitzky-Golay)
3. PyRoki retargeting (V6) → ProtoMotions .motion file
4. Tracking simulation with ONNX policy
5. Reference + tracked video rendering
6. JSON export for Three.js web viewer
7. SMPL mesh JSON generation

**Motion Format Conversions:**
- Text prompts → motion_201 (201-dim HyMotion)
- motion_201[:, :135] → motion_135 (first 135 dims)
- motion_135 → SMPL pose → PyRoki keypoints → G1 robot actions

**Output Directory Structure:**
```
output/embodied_comparison_v3/
  ├── data/
  │   ├── npz/                  # motion_135 NPZ files
  │   ├── motions/              # Reference JSON (Three.js)
  │   ├── tracked_motions/      # Tracked mode JSON
  │   ├── smpl_mesh/            # SMPL mesh JSON for web viz
  │   ├── caches/               # ProtoMotions .motion files
  │   ├── retarget/             # PyRoki retarget outputs
  │   ├── renders/              # Video renders (reference + tracked)
  │   └── meta/                 # Per-motion metadata
  ├── motion_text_mapping.json
  ├── batch_report.json
  └── metrics.json
```

**Smoothing Implementation:**
- **Translation:** Savitzky-Golay filter (window=11, polyorder=5)
- **Rotation:** Gaussian-weighted Markley quaternion averaging
  - Gaussian kernel: σ=1.0, truncate=4.0 → 9-tap kernel
  - Per-joint smoothing
  - Matches official HY-Motion-1.0 post-processing

**Usage Examples:**
```bash
# From JSON prompts
python scripts/embodied/batch_t2m_to_embodied.py \
    --prompt-json output/embodied_comparison_v2/motion_text_mapping.json \
    --output-dir output/embodied_comparison_v3/ \
    --max-motions 5

# Inline prompts
python scripts/embodied/batch_t2m_to_embodied.py \
    --prompts "a person walks forward" "a person jumps" \
    --output-dir output/embodied_test/

# From existing motion_135 NPZ files
python scripts/embodied/batch_t2m_to_embodied.py \
    --npz-dir work_dirs/.../npz/ \
    --output-dir output/embodied_comparison_v3/
```

---

### 9. **batch_npz_to_smpl_mesh_json.py** (Imported by eval scripts)
**Purpose:** Convert motion_135 NPZ to SMPL mesh JSON

**Key Function:** `convert_single_npz(npz_path, smpl_type="smplh", gender="neutral")`

**Conversion Process:**
1. Load motion_135 (T, 135)
2. Split: transl (T, 3) + rot6d (T, 22, 6)
3. rot6d → rotation matrix (Gram-Schmidt)
4. Rotation matrix → axis-angle
5. Build SMPL pose vector (root + body)
6. Generate frame-by-frame mesh JSON for Three.js

**Output Format:** Same as physflow_eval_and_export.py SMPL mesh JSON

---

## Data Format Summary

### motion_135 (Standard Format)
```python
shape: (T, 135)  # T = number of frames (30 fps)
layout: [0:3] translation + [3:135] 22×rot6d
rot6d order: row-major [R00,R01, R10,R11, R20,R21]
coordinate system: Y-up (matches HyMotion training)
```

### Key Layout Constants:
- **Skeleton:** 22 SMPL joints (pelvis + body, no hands)
- **Joint 0:** Root (pelvis) with orientation + translation
- **Joints 1-21:** Body joints (spine, arms, legs, neck, head)
- **Translation:** m (meters), Y is vertical
- **Rotation:** 6D representation (row-major)

---

## Example: Generate PhysFlow Eval Demo

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Step 1: Pre-compute text embeddings (one-time)
python3 scripts/embodied/physflow_precompute_text.py \
    --output output/physflow/text_embeddings.pt

# Step 2: Generate eval demo (original + trained model comparison)
python3 scripts/embodied/physflow_eval_and_export.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --original-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --trained-ckpt output/physflow/run_500iter/model_final.pt \
    --smpl-xml ref_repo/OmniH2O/phc/phc/data/assets/mjcf/smpl_humanoid.xml \
    --text-cache output/physflow/text_embeddings.pt \
    --output-dir output/physflow/eval_demo \
    --quick                              # Fast: 2 prompts × 3 levels = 6 motions

# Output: output/physflow/eval_demo/
#   ├── data/npz/                    ← motion_135 NPZ files
#   ├── data/smpl_mesh/              ← SMPL mesh JSON (kinematic)
#   ├── data/smpl_mesh_physics/      ← SMPL mesh JSON (physics-corrected)
#   ├── data/meta/                   ← Per-motion metadata
#   ├── metrics.json                 ← Summary table
#   └── data/sim_stats/              ← Simulation statistics
```

---

## Existing Demo Data Structure

**Current Demo Location:** `output/physflow/eval_demo/`

**Contains:**
- 10 motions (5 "original" + 5 "physflow" trained)
- NPZ files with motion_135 data
- SMPL mesh JSON for web visualization
- Simulation statistics (completion rate, tracking error)
- Metrics summary comparing original vs trained model

**Level Distribution (from PHYSFLOW_LEVELS):**
- Level 0 (standing): 1 prompt → 2 motions (original + physflow)
- Level 1 (walking): 1 prompt → 2 motions
- Level 2 (upper_body): 1 prompt → 2 motions
- Level 3 (transitions): 1 prompt → 2 motions
- Level 4 (dynamic): 1 prompt → 2 motions

---

## Script Dependencies Graph

```
physflow_eval_and_export.py ⭐ (MAIN FOR DEMO)
  ├── physflow_trainer.py
  │   ├── physflow_physics_oracle.py
  │   │   └── run_smpl_physics_sim.py
  │   ├── physflow_curriculum.py
  │   └── HyMotion T2M model
  ├── physflow_curriculum.py
  ├── PhysicsOracle
  └── batch_npz_to_smpl_mesh_json.py

physflow_evaluate.py (ALTERNATIVE FOR METRICS-ONLY)
  ├── physflow_trainer.py
  ├── physflow_curriculum.py
  ├── physflow_physics_oracle.py
  └── run_smpl_physics_sim.py

batch_t2m_to_embodied.py (FULL PIPELINE FOR G1 ROBOT)
  ├── HyMotion T2M inference
  ├── pipeline_motion_to_robot.py (PyRoki V6 retargeting)
  ├── render_tracker_headless.py (video rendering)
  ├── convert_cache_to_json.py (Three.js export)
  ├── motion135_to_smplx.py (conversion)
  └── batch_npz_to_smpl_mesh_json.py
```

---

## Quick Reference: Key Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| num_frames | 90-150 | Varies by level | Level-specific in PHYSFLOW_LEVELS |
| num_ode_steps | 20 | 10-100 | Denoising steps in diffusion |
| guidance_scale | 5.0 | 1.0-15.0 | CFG classifier-free guidance |
| text_encoder dtype | float16 | float16/bfloat16/float32 | Trade-off between speed and precision |
| completion_rate threshold | 0.6 | 0.0-1.0 | Physics sim success rate cutoff |
| tracking_error_rad threshold | 0.1 | 0.0-1.0 | Joint angle error tolerance |

---

## Troubleshooting

### Problem: "No 'model_state_dict' in checkpoint"
**Solution:** Check checkpoint format
```python
ckpt = torch.load(path, map_location='cpu')
print(list(ckpt.keys()))  # Check available keys
```

### Problem: OOM when loading text encoder
**Solution:** Use pre-computed text embeddings cache
```bash
# Pre-compute once
python3 scripts/embodied/physflow_precompute_text.py \
    --output output/physflow/text_embeddings.pt

# Then always use --text-cache flag
python3 scripts/embodied/physflow_eval_and_export.py ... \
    --text-cache output/physflow/text_embeddings.pt
```

### Problem: Physics correction fails (low completion rate)
**Possible causes:**
1. Generated motion has extreme values → normalize in preprocessing
2. Contact geometry issues → check MJCF XML
3. PD controller gains too weak/strong → tune in run_smpl_physics_sim.py

### Problem: NaN in generated motion
**Solution:** Check for unstable diffusion
- Reduce num_ode_steps (20 → 10)
- Use float32 dtype instead of float16
- Check text encoder loading

