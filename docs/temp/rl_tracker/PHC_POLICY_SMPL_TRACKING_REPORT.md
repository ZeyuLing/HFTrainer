# PHC Policy Weights & SMPL Humanoid Tracking - Search Report

**Date**: 2026-05-15  
**Repository**: HyMotion hf_trainer  
**Search Focus**: Trained PHC policies, ONNX models, and SMPL humanoid policies

---

## Summary

The codebase contains **both G1 robot and SMPL humanoid trained policies**, but in **different formats and deployment stages**:

| Component | Status | Format | Location |
|-----------|--------|--------|----------|
| **G1 Robot Tracker Policy** | ✅ Deployed | ONNX + Checkpoint | `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/` |
| **SMPL Humanoid Tracker Policy** | ✅ Trained | Checkpoint Only | `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/` |
| **SMPL Humanoid (Terrains)** | ✅ Trained | Checkpoint Only | `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl-terrains/` |
| **PHC Training Configs** | ✅ Available | YAML | `ref_repo/OmniH2O/phc/phc/data/cfg/env/` |
| **SMPL Local Robot** | ✅ Available | Python Module | `ref_repo/OmniH2O/phc/phc/smpllib/` |

---

## 1. G1 Robot Tracker Policy (Ready for Deployment)

### Location
```
ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/
```

### Files
```
├── compiled_models/
│   ├── unified_pipeline.onnx          (22 MB)      ← ONNX export for inference
│   └── unified_pipeline.yaml          (5.8 KB)     ← ONNX metadata
├── last.ckpt                          (228 MB)     ← PyTorch checkpoint
├── resolved_configs.yaml              (47 KB)      ← Full training config
├── resolved_configs_inference.yaml    (45 KB)      ← Inference config
└── experiment_config.py               (20 KB)
```

### ONNX Model Details (G1)

**Path**: `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx`

**Metadata** (`unified_pipeline.yaml`):
```yaml
type: unified_pipeline
dt: 0.02  # 50 Hz control

Robot Structure:
- DOFs: 29 joints (lower body + torso + arms)
  - Leg joints: L/R hip (pitch/roll/yaw), knee, ankle (pitch/roll) = 12 DOF
  - Torso: yaw/roll/pitch = 3 DOF
  - Arms: L/R shoulder (pitch/roll/yaw), elbow, wrist = 14 DOF

- Bodies: 33 total
  - pelvis, head, 30 link bodies (legs, arms, torso)
  - Key bodies: left_ankle_roll_link, right_ankle_roll_link (foot contact)
  - Anchor body: torso_link

Control Parameters (PD):
- Stiffness: [40.2, 99.1, 40.2, ...] (varies by joint)
- Damping: [2.56, 6.31, 2.56, ...] (varies by joint)
- Max effort: 150-500 N·m (varies by joint)
```

**Policy Inputs** (ONNX network):
1. `current_anchor_rot` (1, 4) - anchor body rotation in quaternion [x,y,z,w]
2. `current_dof_pos` (1, 29) - joint positions
3. `current_dof_vel` (1, 29) - joint velocities
4. `current_root_local_ang_vel` (1, 3) - root angular velocity (body frame)
5. `mimic_future_anchor_rot` (1, 4, 4) - future (4 steps) anchor rotation references
6. `mimic_future_dof_pos` (1, 4, 29) - future joint position references
7. `mimic_future_dof_vel` (1, 4, 29) - future joint velocity references
8. `historical_processed_actions` (1, 1, 29) - last action (for action smoothing)

**Policy Outputs**:
1. `actions` (1, 29) - motor commands
2. `joint_pos_targets` (1, 29) - PD target positions
3. `stiffness_targets` (1, 29) - adaptive stiffness
4. `damping_targets` (1, 29) - adaptive damping

**Use in Code**:
- Referenced in `scripts/embodied/run_tracker_export.py` line 56-60:
  ```python
  _DEFAULT_ONNX = "ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx"
  ```
- MJCF model: `ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml`

---

## 2. SMPL Humanoid Tracker Policy (Trained but Not ONNX-Exported)

### Location
```
ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/
```

### Files
```
├── resolved_configs.yaml              (44 KB)      ← Full training config
├── resolved_configs_inference.yaml    (44 KB)      ← Inference config
├── last.ckpt                          (121 MB)     ← PyTorch checkpoint
├── resolved_configs.pt                (36 KB)      ← Pickled config (exact reproducibility)
├── resolved_configs_inference.pt      (37 KB)      ← Pickled inference config
└── assets/
    ├── breakdance.gif                 (5.9 MB)
    └── monkey_walk_backflip.gif       (7.5 MB)
```

### SMPL Model Details

**Key Differences from G1**:

| | G1 Bones | SMPL Humanoid |
|---|---|---|
| **DOF Count** | 29 | ~56 (SMPL full body) |
| **Body Count** | 33 | ~24 (SMPL skeleton) |
| **MJCF Asset** | `g1_holo_compat.xml` | `mjcf/smpl_humanoid.xml` |
| **ONNX Export** | ✅ `unified_pipeline.onnx` | ❌ No ONNX yet |
| **Checkpoint Size** | 228 MB | 121 MB |
| **Training Stage** | ✅ Deployed (frozen) | ✅ Trained, awaiting ONNX export |

**Robot Structure** (from `resolved_configs_inference.yaml`):
```yaml
asset_file_name: mjcf/smpl_humanoid.xml
Body names: [Pelvis, Head, L_Hip, L_Knee, L_Ankle, L_Toe, 
             R_Hip, R_Knee, R_Ankle, R_Toe, Torso, Spine, Chest, Neck,
             L_Shoulder, L_Elbow, L_Wrist, L_Hand,
             R_Shoulder, R_Elbow, R_Wrist, R_Hand]
Foot bodies: [L_Ankle, L_Toe, R_Ankle, R_Toe]
Hand bodies: [L_Hand, R_Hand]
Head: Head
Torso: Torso
Default root height: 0.95
```

**ONNX Export Status**: ⚠️ **No ONNX export yet**
- Checkpoint exists: `last.ckpt` (121 MB, PyTorch format)
- Can export using: `protomotions/deployment/export_bm_tracker_onnx.py`
  ```bash
  python deployment/export_bm_tracker_onnx.py \
    --checkpoint ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt
  ```

---

## 3. SMPL Terrains Tracker Policy

### Location
```
ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl-terrains/
```

**Status**: ✅ Trained, similar to SMPL but with terrain-aware features

**Files**:
- `resolved_configs.yaml` (44 KB)
- `resolved_configs_inference.yaml` (44 KB)
- Checkpoint available (no ONNX export yet)

---

## 4. PHC Training Configs & Scripts

### PHC Environment Configs

**Location**: `ref_repo/OmniH2O/phc/phc/data/cfg/env/`

**Available Configs**:
```
├── h1_im_1.yaml              ← H1 humanoid imitation (variant 1)
├── h1_im_2.yaml              ← H1 humanoid imitation (variant 2)
├── h1_im_3.yaml              ← H1 humanoid imitation (variant 3)
├── h1_im_4.yaml              ← H1 humanoid imitation (variant 4)
├── phc_kp_mcp_iccv.yaml      ← PHC with keypoint + MCP
├── phc_kp_pnn_iccv.yaml      ← PHC with keypoint + PNN
├── phc_prim_iccv.yaml        ← PHC primitive
├── phc_shape_mcp_iccv.yaml   ← PHC shape with MCP
├── phc_shape_pnn_iccv.yaml   ← PHC shape with PNN
└── phc_shape_pnn_train_iccv.yaml  ← PHC shape PNN training
```

### G1 PHC Configs

**Location**: `ref_repo/PHC/phc/data/cfg/env/`

**File**: `env_im_g1_phc.yaml` (lines 1-74)
```yaml
task: HumanoidIm              # Imitation task
motion_file: ""              # (set at runtime)
num_envs: 3072               # Parallel environments
episode_length: 300
obs_v: 6                      # Observation version
enable_debug_vis: False

# G1-specific
has_pnn: True                # Primitive Neural Network
num_prim: 3                  # 3 primitives
actors_to_load: 0

# Physics
default_humanoid_mass: 51.436  # H1 config (51.4 kg)
real_weight: True
kp_scale: 1.0

# Termination
terminationHeight: 0.15
enableEarlyTermination: True
terminationDistance: 0.25

# Key bodies for foot contact
key_bodies: ["left_ankle_roll_link", "right_ankle_roll_link", 
             "left_zero_link", "right_zero_link"]
contact_bodies: ["left_ankle_roll_link", "right_ankle_roll_link"]
```

---

## 5. SMPL Local Robot Library

### Location
```
ref_repo/OmniH2O/phc/phc/smpllib/
```

**Files**:
```
├── smpl_mujoco.py           (22 KB) ← SMPL↔MuJoCo conversion
├── smpl_parser.py           (22 KB) ← SMPL parsing utilities
├── smpl_local_robot.py      (102 KB) ← SMPL robot class (core)
└── smpl_eval.py             (8 KB)  ← SMPL evaluation metrics
```

**`smpl_local_robot.py` - Core Features**:
- Automatic humanoid generation from SMPL/SMPL-H/SMPL-X models
- XML file generation for MuJoCo simulation
- Support for different genders and body shapes
- **Capsule-based and mesh-based models**
- Multiple body shape parameters (currently 49 DOF for SMPL)

**Key Classes**:
```python
class SMPLConverter:
    """Converts between SMPL and robot representations"""
    def __init__(self, model, new_model, smpl_model="smpl")
    # Supports: "smpl", "smplh", "smplx"
    # Has joint weights and body parameters for each
```

---

## 6. Trained Actuator Networks

### Location
```
ref_repo/OmniH2O/resources/actuator_nets/
ref_repo/OmniH2O/legged_gym/resources/actuator_nets/
```

**Model**: `anydrive_v3_lstm.pt` (ActuatorNet for AnyDrive v3)
- 8.6 KB PyTorch model
- Used for actuator dynamics simulation
- Replicated in both locations (symlink or copy)

---

## 7. Script: `run_tracker_export.py`

### Purpose
Export motion tracker predictions to MuJoCo simulation and save as cache files.

### Workflow
```
Reference Motion Cache (.pt)
  ↓
[ONNX Tracker Policy] 
  + [MuJoCo Physics Simulation]
  ↓
Tracked Motion Cache (.pt)
  ↓
[convert_cache_to_json.py]
  ↓
Three.js Visualization
```

### Key Inputs
- **ONNX Model** (default G1): `g1-bones-deploy/compiled_models/unified_pipeline.onnx`
- **MJCF Robot** (default G1): `g1_holo_compat.xml`
- **Reference Motion**: Any `.pt` motion cache from `MotionPlayer`

### Usage
```bash
# Single motion
python scripts/embodied/run_tracker_export.py \
    --motion output/embodied_comparison/data/caches/pipeline_00000.pt \
    --output output/embodied_comparison/data/tracked_caches/tracked_00000.pt

# Batch (all caches in directory)
python scripts/embodied/run_tracker_export.py \
    --motion-dir output/embodied_comparison/data/caches/ \
    --output-dir output/embodied_comparison/data/tracked_caches/ \
    --pattern 'pipeline_*.pt'

# Custom ONNX + MJCF (for SMPL or other robots)
python scripts/embodied/run_tracker_export.py \
    --motion <motion_file> \
    --output <output_file> \
    --onnx <path_to_onnx_model> \
    --mjcf <path_to_mjcf>
```

---

## 8. Next Steps: SMPL ONNX Export

### Status
SMPL humanoid checkpoint **trained but not ONNX-exported**.

### To Export SMPL ONNX Model:

**Command**:
```bash
python deployment/export_bm_tracker_onnx.py \
    --checkpoint ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt \
    --output-dir ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/
```

**What This Does**:
1. Loads PyTorch checkpoint
2. Auto-detects observation keys from checkpoint
3. Exports to ONNX format (~22 MB, similar to G1)
4. Generates `unified_pipeline.yaml` metadata

**After Export**:
```
smpl/
├── last.ckpt                                 (existing)
├── compiled_models/
│   ├── unified_pipeline.onnx                 (NEW)
│   └── unified_pipeline.yaml                 (NEW)
└── resolved_configs*.yaml                    (existing)
```

**Then Use in Tracking**:
```bash
python scripts/embodied/run_tracker_export.py \
    --motion <smpl_motion> \
    --output <tracked_motion> \
    --onnx ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.onnx \
    --mjcf ref_repo/ProtoMotions/protomotions/data/assets/mjcf/smpl_humanoid.xml
```

---

## 9. Policy Comparison: G1 vs SMPL

| Aspect | G1 Tracker | SMPL Tracker |
|--------|-----------|--------------|
| **Deployment Status** | ✅ Production (ONNX ready) | ⏳ Awaiting ONNX export |
| **Format** | ONNX + PyTorch | PyTorch only |
| **DOF** | 29 | ~56 (full SMPL) |
| **Checkpoint Size** | 228 MB | 121 MB |
| **MJCF Asset** | `g1_holo_compat.xml` | `smpl_humanoid.xml` |
| **Stiffness/Damping** | Tuned PD parameters | SMPL default values |
| **Robot Type** | Unitree G1 (quad-like upright) | Generic humanoid |
| **Training Data** | Bones motion (motion tracking) | Multi-source humanoid motion |

---

## 10. PHC vs ProtoMotions Policies

### PHC (Perpetual Humanoid Control)
- **Location**: `ref_repo/PHC/`, `ref_repo/OmniH2O/phc/`
- **Task**: Physics-based humanoid motion synthesis from mocap
- **Features**: PNN (Primitive Neural Network), PMCP (Phase-Manifold Control Policy)
- **Training Configs**: `env_im_g1_phc.yaml`, `phc_*.yaml`
- **Status**: Training configs available, no pre-trained weights in checkpoints/

### ProtoMotions Policies (Motion Trackers)
- **Location**: `ref_repo/ProtoMotions/`
- **Task**: Policy-based motion tracking (FK → physics)
- **Features**: ONNX export, MuJoCo simulation
- **Pre-trained Models**: G1 (ONNX + ckpt), SMPL (ckpt only)
- **Status**: Ready for inference

---

## 11. File Sizes & Storage

| File | Size | Type |
|------|------|------|
| G1 checkpoint | 228 MB | .ckpt |
| G1 ONNX | 22 MB | .onnx |
| SMPL checkpoint | 121 MB | .ckpt |
| SMPL config (YAML) | 44 KB | .yaml |
| G1 config (YAML) | 47 KB | .yaml |
| ActuatorNet | 8.6 KB | .pt |

---

## 12. Key Observations

### ✅ What Exists
1. **G1 robot policy**: Fully deployed ONNX + checkpoint
2. **SMPL humanoid policy**: Trained checkpoint, awaiting ONNX export
3. **PHC configs**: Training configurations for G1 and humanoids
4. **SMPL library**: Automatic humanoid generation from SMPL body model
5. **Tracking script**: End-to-end motion tracking with MuJoCo simulation

### ⚠️ Gaps
1. **No SMPL ONNX export yet**: Checkpoint exists but not compiled to ONNX
2. **No trained SMPL policy weights in checkpoints/**: All pre-trained models are in `ref_repo/`
3. **PHC training configs but no PHC ONNX models**: PHC is RL-based, ProtoMotions is tracker-based
4. **No explicit "policy zoo"**: Policies embedded in multi-purpose training framework

### 🎯 Recommended Actions
1. **Export SMPL ONNX**: Run `export_bm_tracker_onnx.py` on SMPL checkpoint
2. **Document policy I/O**: Create reference table of ONNX inputs/outputs for each robot
3. **Add SMPL support to `run_tracker_export.py`**: Parameterize robot type
4. **Train additional policies**: If needed for other humanoid topologies

---

## References

- **G1 Tracker ONNX**: `ref_repo/ProtoMotions/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx`
- **Tracker Export Script**: `scripts/embodied/run_tracker_export.py`
- **SMPL Robot Library**: `ref_repo/OmniH2O/phc/phc/smpllib/smpl_local_robot.py`
- **ProtoMotions Docs**: `ref_repo/ProtoMotions/CLAUDE.md`
- **PHC Configs**: `ref_repo/OmniH2O/phc/phc/data/cfg/env/env_im_g1_phc.yaml`

