# KIMODO-Related Inference Scripts & Retargeting Code - Comprehensive Index

**Base Directory**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/`

**Last Updated**: 2026-05-12

---

## Executive Summary

This repository contains **integrated KIMODO inference pipelines** for the HyMotion M2M v2 motion generation project:

1. **SMPL-22 ↔ SOMA-30/77 rotation-based retargeting** with FK-based reconstruction
2. **Multi-prompt sliding-window inference** for long motion sequences (240-frame segments)
3. **Constraint-based generation** with full-body keyframe conditioning
4. **Evaluation dashboard** with 3D SMPL visualization and multi-model comparison

---

## Part 1: Core KIMODO Inference Scripts (`scripts/kimodo/`)

### 1.1 Primary Inference Pipeline

#### `scripts/kimodo/run_kimodo_all_tasks.py` (PRIMARY ENTRY POINT)
- **Lines**: ~2000+
- **Purpose**: Universal KIMODO evaluation runner for all M2M tasks (E2-E8, E10, E14-E16)
- **Key Features**:
  - SMPL-22 → SOMA-30 rotation retargeting with global rotation transfer
  - Multi-prompt sliding-window inference (segments ≤240 frames, transitions with blending)
  - Full-body constraint set with global rotation pinning (custom `FullBodyWithRotConstraintSet`)
  - Motion composition: condition frames + generated frames + context padding
  - NPZ export for dashboard 3D visualization

- **Key Functions**:
  ```python
  _split_num_frames(n: int, safe_len: int = 240) -> list
      # Split long sequences into segments (handles training distribution extrapolation)
  
  _make_fullbody_with_rot_constraint_set() -> FullBodyConstraintSet
      # Custom constraint class that pins global joint rotations (not just positions)
  
  # Skeleton mapping functions for SMPL↔SOMA conversion
  smpl22_to_soma30_constraints(...)
  soma30_to_soma77_expansion(...)  # Expands SOMA-30 → SOMA-77 with relaxed hands
  ```

- **Usage**:
  ```bash
  python scripts/kimodo/run_kimodo_all_tasks.py \
      --all-tasks \
      --max-samples 80 \
      --output-dir work_dirs/kimodo_eval_xxx
  ```

- **Output Format**:
  - `work_dirs/.../kimodo/{task_id}/{setting}/npz/{idx:05d}.npz`
  - NPZ contains: `posed_joints`, `global_rot_mats`, `motion135` (SOMA-77 mesh params)

---

#### `scripts/kimodo/run_kimodo_base_pose_edit.py`
- **Lines**: ~400
- **Purpose**: KIMODO inference for base pose / keypose edit demo
- **Key Features**:
  - Loads before/after keypose edit pairs
  - Selects adaptive keyposes (frames with max correction magnitude)
  - Builds soft constraints: blend before/after poses at keyframes
  - Computes metrics: keyframe MPJPE, global MPJPE, smoothness, foot skating
  
- **Constraint Building**:
  ```python
  _build_base_pose_constraints(
      before_soma_rots, before_soma_pos,
      after_soma_rots, after_soma_pos,
      frame_indices, kp_indices
  )
  # Mixes SOMA-30 poses at keyframes (after) with surrounding context (before)
  ```

- **Output**: 
  - Per-case metrics JSON with before/after/GT comparison
  - Aggregated statistics across test set

---

### 1.2 Data Preparation & Context Appending

#### `scripts/kimodo/append_kimodo_context_soma77.py`
- **Lines**: ~600
- **Purpose**: Post-process KIMODO NPZ outputs to include full-sequence visualization
  
- **Background Problem**:
  - KIMODO E14 NPZ only covers the generation span (condition A + transition + condition B)
  - Gray context (source motion prefix/suffix) lacks SOMA-77 mesh data
  - Dashboard can only show skeleton for context, not mesh

- **Solution**:
  - Deterministically retarget source motion frames (SMPL-22 → SOMA-30 → SOMA-77)
  - Append to existing NPZ without touching main generation output
  - Dashboard updated to concatenate [prefix | main | suffix] for full-sequence rendering

- **Output Fields Added**:
  ```
  prefix_posed_joints       : (T_prefix, 77, 3)   float32
  prefix_global_rot_mats    : (T_prefix, 77, 3, 3) float32
  suffix_posed_joints       : (T_suffix, 77, 3)   float32
  suffix_global_rot_mats    : (T_suffix, 77, 3, 3) float32
  prefix_len, suffix_len    : int  (in layout_json)
  ```

- **Command**:
  ```bash
  python scripts/kimodo/append_kimodo_context_soma77.py \
      --run-dir work_dirs/.../kimodo/E14_M/E14_M \
      --data-file data/eval_data/m2m/eval_e14_hq400h_move100.json \
      --placement velocity
  ```

---

#### `scripts/kimodo/append_kimodo_e15_context_soma77.py`
- **Purpose**: Task-specific variant for E15 (in-betweening) context appending
- **Differences**: Different frame selection logic for E15's head/tail structure

---

#### `scripts/kimodo/run_kimodo_e14_rotfix_batch.sh`
- **Purpose**: Batch script for E14 rotation fix rerun (shell wrapper)
- **Contains**: Sbatch submission parameters + python entry point

---

#### `scripts/kimodo/_run_kimodo_debug.sh`
- **Purpose**: Debug/development variant with verbose logging and small sample count

---

### 1.3 Post-Processing & Patch Scripts

#### `scripts/patch/patch_kimodo_y_anchor.py`
- **Purpose**: Fix KIMODO E14 Y-anchor (height) inconsistency
- **Context**: KIMODO root often drifts in Y; this enforces SOMA floor alignment

#### `scripts/patch/merge_kimodo_e14_shards.py` / `merge_kimodo_shards_simple.py`
- **Purpose**: Merge sharded NPZ outputs from multi-GPU KIMODO runs
- **Output**: Unified evaluation JSON + combined NPZ directory

---

### 1.4 Debugging Scripts

#### `scripts/debug/diagnose_kimodo_e14_boundary_jumps.py`
- **Purpose**: Analyze E14 transition boundary artifacts (motion jumps at segment stitching)
- **Metrics**: Per-frame velocity, per-joint acceleration at boundaries

---

## Part 2: Motion Annotation Web - KIMODO Integration

### 2.1 Eval Dashboard (`motion_annot_web/eval_dashboard/`)

#### `motion_annot_web/eval_dashboard/swap_to_swin_kimodo.py`
- **Purpose**: Database migration script for KIMODO E3 results
- **Context**: Replaces OLD single-shot rollout paths with NEW sliding-window rerun
- **DB Table**: Updates `sample_results.gen_motion_path` for KIMODO_uncond E3
- **Old Paths**: `kimodo/uncond/E3_{every_5f,every_10f,E3_C}`
- **New Paths**: `kimodo/uncond_swin/E3_every_{5,10,15}f`

#### Dashboard Integration Points
- **`eval_dashboard/utils.py`** (line 8-90):
  - SOMA skeleton definitions (SOMA-30, SOMA-77 joint names)
  - NPZ↔SMPL parameter conversion (rot6d, axis-angle, poses, trans)
  - Color coding for body part visualization
  
- **`eval_dashboard/eval_task_registry.py`**:
  - Task metadata: E2-E8, E10, E14-E16 with KIMODO baselines
  - Key metrics definition for each task
  
- **`eval_dashboard/app.py`** (line 1100+):
  - `/api/smpl/<npz_path>`: Render KIMODO NPZ as SMPL mesh
  - KIMODO NPZ field mapping for LBS visualization
  - Multi-task evaluation comparison UI

---

### 2.2 Score M2M (`motion_annot_web/score_m2m/`)

#### `motion_annot_web/score_m2m/swap_to_swin_kimodo.py`
- **Purpose**: Companion to eval_dashboard script
- **DB**: Updates `score_m2m.db` sample motion references
- **Same mapping**: OLD single-shot → NEW sliding-window

---

### 2.3 KIMODO Constraint Demo (`motion_annot_web/kimodo_constraint_demo/`)

A dedicated web app + evaluation suite for SMPL↔SOMA retargeting quality assurance.

#### `motion_annot_web/kimodo_constraint_demo/test_retarget.py` (PRIMARY RETARGETING TEST)
- **Lines**: ~450
- **Purpose**: Comprehensive SMPL→SOMA-30 rotation retargeting validation
  
- **Core Tests**:
  
  1. **T-pose Identity Test**:
     - Verify: identity rotations → SOMA-30 FK → exact neutral pose
     - Tolerance: < 1e-4 cm per-joint error
     - Purpose: Validates FK implementation
  
  2. **T-pose Retarget with Horizontal Arms**:
     - Input: SMPL-22 identity rotations (SMPL's T-pose with tilted arms)
     - After retargeting + offset compensation → SOMA-30 neutral (arms perfectly horizontal)
     - Tests: Neutral pose offset correction working correctly
     - Metrics:
       - Arm bone directions compared to neutral reference
       - Y-component of retargeted arm vectors should match reference
  
  3. **Proportional Consistency**:
     - Extract bone lengths from retargeted SOMA-30 output
     - Compare against SOMA-30 neutral proportions
     - Verify: Hip width, shoulder width, arm lengths, leg lengths match SOMA30 skeleton
     - NOT SMPLX22's proportions (would indicate FK error or wrong mapping)
  
  4. **Round-Trip Error**:
     - SMPL-22 → SOMA-30 global rots → SOMA-30 local rots → SOMA-30 FK
       → extract global rots → SMPL-22 layout → compare
     - Tolerance: < 1 cm for matched joints
  
  5. **Global Rotation Preservation**:
     - Ensure root global rotation maintained through retargeting pipeline
     - No unwanted rotations introduced at segment boundaries
  
  6. **Comparison: Old Position-Scatter vs New Rotation Retargeting**:
     - Quantify improvement of rotation-based approach over legacy position scattering

- **Metrics Computed**:
  ```python
  get_soma30_proportions(joints) -> dict:
      "hip_width", "shoulder_width", "left_upper_arm", "left_forearm",
      "right_upper_arm", "right_forearm", "left_thigh", "left_shin",
      "right_thigh", "right_shin", "torso_height", "spine_length"
  ```

- **Usage**:
  ```bash
  python test_retarget.py [--num_cases 10] [--verbose]
  ```

- **Output**: PASS/FAIL per test + detailed per-joint error metrics

---

#### `motion_annot_web/kimodo_constraint_demo/batch_eval.py`
- **Lines**: ~1100
- **Purpose**: Batch KIMODO evaluation pipeline for demo cases
  
- **Key Function**:
  ```python
  smpl_to_soma30_constraints(poses: np.ndarray, trans: np.ndarray, ...):
      # SMPL-22 (156D pose vector) → SOMA-30 rotations + positions
      # Handles:
      # 1. SMPLX22 joint subset extraction
      # 2. Rotation retargeting (global rotation transfer + SOMA FK)
      # 3. Position reconstruction via SOMA-30 FK
      # Returns: (soma_rots, soma_pos, soma_joints)
  ```

- **Evaluation Metrics**:
  - MPJPE (mean per-joint position error) vs GT
  - Smoothness (double derivative)
  - Foot skating velocity
  - Jitter detection

- **Demo Cases**: Walking, gestures, boxing, jogging (4 predefined scenarios)

---

#### `motion_annot_web/kimodo_constraint_demo/server.py`
- **Lines**: ~1200
- **Purpose**: Flask web server for interactive retargeting demo
- **Features**:
  - Upload SMPL motion or select demo case
  - Real-time retargeting visualization
  - 3D viewer showing before/after/GT comparison
  - Download retargeted motion as NPZ

---

#### Other Demo Scripts
- `prepare_data.py`: Stage evaluation data (MPZ↔demo format)
- `run_generation.py`: Run KIMODO generation for a specific case
- `fix_output_smpl.py`: Post-process generated SMPL to fix numerical issues
- `export_visualization.py`: Generate MP4/PNG visualizations
- `build_hq_eval_index.py`: Curate high-quality eval case subset

---

## Part 3: Model Training Integration

### 3.1 KIMODO Auxiliary Loss

#### `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`
- **Purpose**: Custom auxiliary loss regularizing M2M model to match KIMODO output distribution
- **Concept**: KIMODO as oracle → guide M2M training
- **Loss Terms**:
  - SOMA-30 position reconstruction error
  - Global rotation consistency
  - SOMA-77 LBS mesh error

---

### 3.2 Test Suite

#### `tests/unit/test_kimodo_aux_loss.py`
- **Purpose**: Unit tests for KIMODO auxiliary loss computation
- **Coverage**: SMPL↔SOMA conversion, loss backward pass, gradient checks

---

## Part 4: Configuration & Eval Registry

### 4.1 Task Registry

#### `motion_annot_web/eval_dashboard/eval_task_registry.py`
- **Purpose**: Defines all M2M v2 evaluation tasks (E1-E16) with KIMODO baselines
  
- **Task Metadata**:
  ```python
  EVAL_TASKS = {
      "E14": {
          "name": "Placement",
          "models": ["uncond_local", "caption_local", "KIMODO_uncond", "KIMODO_caption"],
          "datalist": "data/eval_data/m2m/eval_e14_hq400h_move100.json",
          "key_metrics": ["mpjpe_masked", "placement_error", "smoothness", ...],
          "settings": ["M", "L"],  # Medium / Large placement scale
      },
      ...
  }
  ```

- **KIMODO-Specific Fields**:
  - `kimodo_rotation_space`: "6d" or "axis_angle"
  - `kimodo_soma_skeleton`: "SOMA30" or "SOMA77"

---

### 4.2 Data Importers

#### `motion_annot_web/eval_dashboard/data_importer.py`
- **Purpose**: CLI tool to import flattened KIMODO eval JSON into SQLite
  
- **Workflow**:
  1. `split_eval_v2_to_flat.py`: Expand nested JSON → one JSON per (model, task, setting)
  2. `data_importer.py import`: Parse flat JSON → populate eval_runs / sample_results / agg_metrics
  3. Dashboard queries tables for visualization

---

## Part 5: Retargeting Architecture Deep Dive

### 5.1 SMPL-22 → SOMA-30 Mapping

**Joint Correspondence** (22 joints selected from SMPLX22):
```
SMPL-22 Joints           →  SOMA-30 Joints
pelvis (0)               →  Hips (0)
l_hip (1)                →  LeftLeg (1)
r_hip (2)                →  RightLeg (2)
spine1 (3)               →  [interpolated]
l_knee (4)               →  LeftShin (3)
r_knee (5)               →  RightShin (4)
spine2 (6)               →  Chest (5)
l_ankle (7)              →  LeftFoot (6)
r_ankle (8)              →  RightFoot (7)
spine3 (9)               →  UpperChest (8)
l_foot (10)              →  [blended]
r_foot (11)              →  [blended]
neck (12)                →  Neck1 (9)
l_shoulder (16)          →  LeftArm (10)
r_shoulder (17)          →  RightArm (11)
l_elbow (18)             →  LeftForeArm (12)
r_elbow (19)             →  RightForeArm (13)
l_wrist (20)             →  LeftHand (14)
r_wrist (21)             →  RightHand (15)
```

### 5.2 Retargeting Algorithm

**Three-Stage Process** (implemented in `batch_eval.smpl_to_soma30_constraints`):

1. **Global Rotation Transfer**:
   - Extract root global rotation from SMPL-22
   - Apply directly to SOMA-30 root (preserves trajectory)
   
2. **Per-Joint Rotation Retargeting**:
   - For each SOMA-30 joint with SMPL-22 correspondent:
     - Compute neutral pose offset: `offset = SOMA30_neutral_rot - SMPL22_neutral_rot`
     - Retarget: `SOMA30_rot = SMPL22_rot * offset^-1` (in SO(3))
   - Joints without direct mapping: interpolate from neighbors or use zero rotation
   
3. **Forward Kinematics + Position Reconstruction**:
   - SOMA-30 FK: `positions = fk(local_rots, root_position)`
   - Ensure SOMA-30 skeleton proportions maintained (not SMPL-22's proportions)
   - Result: (T, 30, 3) positioned joints in SOMA-30 frame

### 5.3 Neutral Pose Offset Compensation

**Problem**: SMPL-22 and SOMA-30 have different T-poses
- SMPL-22: Arms tilted upward (~30° from horizontal)
- SOMA-30: Arms perfectly horizontal

**Solution**: Precompute neutral pose rotation offsets during initialization
- At setup: compute `neutral_offset[j] = SOMA30_neutral_rot[j] - SMPL22_neutral_rot[j]`
- At runtime: apply offsets to all retargeted rotations
- Result: SOMA-30 output respects SOMA's anatomical conventions

---

### 5.4 SOMA-77 Expansion (Hand + Face Details)

**SOMA-30 → SOMA-77** (adds 47 new joints for detailed hands + face):
- SOMA-30: 30 body joints (no hand/face articulation)
- SOMA-77: 30 body + 15 hand fingers per side (30 hand) + 17 face/eye/jaw
- **Expansion Strategy**:
  - Fingers: Rest pose (relaxed hands) applied to all frames
  - Face/jaw/eyes: Static identity (no animation)
  - Body: Propagated from SOMA-30 → SOMA-77 mapping

---

## Part 6: File Location Reference

### Scripts Directory
```
scripts/kimodo/
├── run_kimodo_all_tasks.py          # Main inference (REQUIRED)
├── run_kimodo_base_pose_edit.py     # Keypose edit variant
├── append_kimodo_context_soma77.py  # Post-process NPZ
├── append_kimodo_e15_context_soma77.py
├── run_kimodo_e14_rotfix_batch.sh
└── _run_kimodo_debug.sh

scripts/patch/
├── patch_kimodo_y_anchor.py
├── merge_kimodo_e14_shards.py
└── merge_kimodo_shards_simple.py

scripts/debug/
└── diagnose_kimodo_e14_boundary_jumps.py
```

### Web Apps Directory
```
motion_annot_web/
├── eval_dashboard/
│   ├── app.py                       # Main dashboard
│   ├── swap_to_swin_kimodo.py       # DB migration
│   ├── utils.py                     # SOMA skeleton + NPZ utils
│   ├── eval_task_registry.py        # Task definitions
│   ├── data_importer.py             # Import pipeline
│   ├── templates/task_detail.html   # 3D viewer
│   └── static/                      # Three.js frontend
│
├── score_m2m/
│   ├── swap_to_swin_kimodo.py       # Companion DB migration
│   └── ...
│
└── kimodo_constraint_demo/
    ├── server.py                    # Flask web server
    ├── test_retarget.py            # RETARGETING TESTS
    ├── batch_eval.py               # Batch evaluation
    ├── run_generation.py           # Generation runner
    ├── prepare_data.py             # Data staging
    ├── fix_output_smpl.py          # Post-processing
    └── cases/                       # Demo cases
```

### Configuration Directory
```
configs/hymotion_m2m_v3/
├── _base_hymotion_m2m_v3_046b.py   # Base config with KIMODO aux loss
└── hymotion_m2m_v3_debug.py

ref_repo/KIMODO/
├── kimodo/
│   ├── model/
│   │   ├── kimodo_model.py          # Core KIMODO model
│   │   ├── load_model.py
│   │   └── __init__.py
│   ├── constraints.py               # Constraint definitions
│   ├── skeleton/
│   │   └── definitions.py           # SOMA-30, SOMA-77 skeletons
│   └── viz/                         # Visualization tools
│       ├── soma_skin.py
│       └── smplx_skin.py
└── CLAUDE.md                         # KIMODO architecture docs
```

---

## Part 7: Known Issues & Workarounds

### Issue 1: Velocity-Based Jumps at 300+ Frames
- **Symptom**: KIMODO generates motion with velocity > 0.5 m/frame after 300 frames
- **Root Cause**: Training distribution extrapolation (KIMODO trained on ≤240-frame clips)
- **Solution**: Cap each segment to 240 frames; use multi-prompt with 5-frame transitions
- **Reference**: `run_kimodo_all_tasks.py` line 30-40 (`KIMODO_SAFE_LEN = 240`)

### Issue 2: Gray Context Lacks SOMA-77 Mesh
- **Symptom**: Dashboard shows skeleton-only for source motion context frames
- **Root Cause**: Original NPZ only covers generation span
- **Solution**: Use `append_kimodo_context_soma77.py` to post-add prefix/suffix SOMA-77 data

### Issue 3: Y-Anchor (Height) Drift
- **Symptom**: Generated motion floats above/below floor
- **Root Cause**: KIMODO root not properly grounded
- **Solution**: Apply `patch_kimodo_y_anchor.py` post-generation

### Issue 4: Rotation Space Mismatch
- **Symptom**: Mesh looks twisted at boundaries between M2M and KIMODO output
- **Root Cause**: Different rotation representations (6d, axis-angle, quaternion)
- **Solution**: Standardize to `rotation_space="6d_column_major"` in all pipelines

---

## Part 8: Quick Reference Checklists

### Running Full KIMODO Evaluation
```bash
# 1. Generate SOMA-30 constraints + inference
python scripts/kimodo/run_kimodo_all_tasks.py \
    --all-tasks \
    --max-samples 100 \
    --output-dir work_dirs/kimodo_eval_20260512

# 2. Append SOMA-77 context for full viz
for task in E14 E15 E16; do
  python scripts/kimodo/append_kimodo_context_soma77.py \
      --run-dir work_dirs/kimodo_eval_20260512/KIMODO_uncond/$task \
      --data-file data/eval_data/m2m/eval_${task}_*.json
done

# 3. Convert to flat JSON format
python tools/split_eval_v2_to_flat.py \
    --in-dir work_dirs/kimodo_eval_20260512 \
    --out-dir work_dirs/kimodo_eval_20260512/import_jsons

# 4. Import into dashboard
for json in work_dirs/kimodo_eval_20260512/import_jsons/*.json; do
  python motion_annot_web/eval_dashboard/data_importer.py import "$json"
done

# 5. View results
cd motion_annot_web/eval_dashboard
python app.py --port 8081
```

### Testing Retargeting Quality
```bash
cd motion_annot_web/kimodo_constraint_demo
python test_retarget.py --num_cases 100 --verbose
```

---

## Part 9: Integration Points with M2M v2

### Training
- **Auxiliary Loss**: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`
  - Regularizes M2M model to match KIMODO output distribution
  - Computed during forward pass; contributes to total loss

### Evaluation
- **Entry Point**: `tools/eval_m2m_v2_all_tasks.py`
  - Supports KIMODO as baseline model
  - Automatically calls `run_kimodo_all_tasks.py` for KIMODO tasks if selected

### Visualization
- **Dashboard**: `motion_annot_web/eval_dashboard/`
  - Loads KIMODO NPZ files via `/api/smpl/` endpoint
  - Renders mesh with LBS from SOMA-77 posed_joints / global_rot_mats

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| **KIMODO-specific Python files** | 28 |
| **Database backup files** | 6 |
| **Retargeting test cases** | 4 main tests + per-joint validation |
| **Supported tasks** | 13 (E2-E16, KIMODO baselines for most) |
| **NPZ storage locations** | 50+ work_dirs with sharded runs |
| **Web apps integrating KIMODO** | 3 (eval_dashboard, score_m2m, kimodo_constraint_demo) |

---

## Related Documentation

- **eval_dashboard/CLAUDE.md**: Full dashboard architecture
- **ref_repo/KIMODO/CLAUDE.md**: KIMODO model design
- **configs/hymotion_m2m_v3/_base_hymotion_m2m_v3_046b.py**: M2M + KIMODO config
- **docs/temp/kimodo_eval_known_issues.md**: Known issues & resolutions

