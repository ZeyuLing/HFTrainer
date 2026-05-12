# Unitree G1 Robot Model Files - Complete Inventory

## Project Overview
This is a comprehensive inventory of the **Unitree G1 (29-DOF) robot model** used in the **ProtoMotions embodied AI framework**. The ProtoMotions framework supports multiple physics simulators (IsaacGym, IsaacLab, Newton, Genesis, MuJoCo) and RL algorithms (PPO, AMP, ASE, MaskedMimic).

---

## 1. URDF (Unified Robot Description Format) Files

### Primary URDF File
**Location**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf`

**Robot Name**: `g1_29dof`  
**Type**: Humanoid robot with complete kinematic chain  
**Mesh Directory Reference**: `../mesh/G1/` (relative path in URDF)

---

## 2. MuJoCo XML (MJCF) Configuration Files

ProtoMotions includes **6 MuJoCo configuration variants** for different simulation scenarios:

| Filename | Path | Description | Purpose |
|----------|------|-------------|---------|
| `g1_bm.xml` | `protomotions/data/assets/mjcf/` | Basic model with full meshes | Standard simulation with complete geometry |
| `g1_bm_box_feet.xml` | `protomotions/data/assets/mjcf/` | Basic model with box feet | Simplified foot geometry for faster simulation |
| `g1_bm_no_mesh_box_feet.xml` | `protomotions/data/assets/mjcf/` | Primitive geometries only | Fast CPU simulation without mesh rendering |
| `g1_holo.xml` | `protomotions/data/assets/mjcf/` | Holonomic base variant | Omnidirectional base mobility |
| `g1_holo_compat.xml` | `protomotions/data/assets/mjcf/` | Holonomic compatible | Alternative holonomic configuration |
| `g1_holo_compat_box_feet.xml` | `protomotions/data/assets/mjcf/` | Holonomic with box feet | Combined holonomic + simplified feet |

**Base Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/mjcf/`

---

## 3. Mesh Files (STL Geometry)

### Directory Structure
**Base Directory**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/mesh/G1/`

### Total Mesh Count
**63 STL files** comprising the complete robot geometry.

### Mesh Organization by Body Region

#### Torso and Pelvis (7 files)
- `pelvis.stl` - Main pelvis body
- `pelvis_contour_link.stl` - Pelvis contour visualization
- `torso_link.stl` - Main torso body
- `torso_link_rev_1_0.stl` - Torso revision 1.0
- `torso_constraint_L_link.stl` - Left torso constraint
- `torso_constraint_L_rod_link.stl` - Left constraint rod
- `torso_constraint_R_link.stl` - Right torso constraint
- `torso_constraint_R_rod_link.stl` - Right constraint rod
- `waist_support_link.stl` - Waist support structure
- `waist_yaw_link.stl` - Waist yaw joint
- `waist_yaw_link_rev_1_0.stl` - Waist yaw revision 1.0
- `waist_roll_link.stl` - Waist roll joint
- `waist_roll_link_rev_1_0.stl` - Waist roll revision 1.0
- `waist_constraint_L.stl` - Left waist constraint
- `waist_constraint_R.stl` - Right waist constraint
- `head_link.stl` - Head structure
- `logo_link.stl` - Logo/branding element

#### Left Leg (8 files)
- `left_hip_yaw_link.stl` - Hip yaw rotation joint
- `left_hip_pitch_link.stl` - Hip pitch rotation joint
- `left_hip_roll_link.stl` - Hip roll rotation joint
- `left_knee_link.stl` - Knee joint
- `left_ankle_pitch_link.stl` - Ankle pitch joint
- `left_ankle_roll_link.stl` - Ankle roll joint

#### Right Leg (8 files)
- `right_hip_yaw_link.stl` - Hip yaw rotation joint
- `right_hip_pitch_link.stl` - Hip pitch rotation joint
- `right_hip_roll_link.stl` - Hip roll rotation joint
- `right_knee_link.stl` - Knee joint
- `right_ankle_pitch_link.stl` - Ankle pitch joint
- `right_ankle_roll_link.stl` - Ankle roll joint

#### Left Arm (15 files)
- `left_shoulder_yaw_link.stl` - Shoulder yaw
- `left_shoulder_pitch_link.stl` - Shoulder pitch
- `left_shoulder_roll_link.stl` - Shoulder roll
- `left_elbow_link.stl` - Elbow joint
- `left_wrist_yaw_link.stl` - Wrist yaw
- `left_wrist_pitch_link.stl` - Wrist pitch
- `left_wrist_roll_link.stl` - Wrist roll
- `left_wrist_roll_rubber_hand.stl` - Rubber hand attachment (wrist roll)
- `left_hand_palm_link.stl` - Hand palm
- `left_hand_thumb_0_link.stl` - Thumb segment 0
- `left_hand_thumb_1_link.stl` - Thumb segment 1
- `left_hand_thumb_2_link.stl` - Thumb segment 2
- `left_hand_index_0_link.stl` - Index finger segment 0
- `left_hand_index_1_link.stl` - Index finger segment 1
- `left_hand_middle_0_link.stl` - Middle finger segment 0
- `left_hand_middle_1_link.stl` - Middle finger segment 1
- `left_rubber_hand.stl` - Rubber hand covering

#### Right Arm (15 files)
- `right_shoulder_yaw_link.stl` - Shoulder yaw
- `right_shoulder_pitch_link.stl` - Shoulder pitch
- `right_shoulder_roll_link.stl` - Shoulder roll
- `right_elbow_link.stl` - Elbow joint
- `right_wrist_yaw_link.stl` - Wrist yaw
- `right_wrist_pitch_link.stl` - Wrist pitch
- `right_wrist_roll_link.stl` - Wrist roll
- `right_wrist_roll_rubber_hand.stl` - Rubber hand attachment (wrist roll)
- `right_hand_palm_link.stl` - Hand palm
- `right_hand_thumb_0_link.stl` - Thumb segment 0
- `right_hand_thumb_1_link.stl` - Thumb segment 1
- `right_hand_thumb_2_link.stl` - Thumb segment 2
- `right_hand_index_0_link.stl` - Index finger segment 0
- `right_hand_index_1_link.stl` - Index finger segment 1
- `right_hand_middle_0_link.stl` - Middle finger segment 0
- `right_hand_middle_1_link.stl` - Middle finger segment 1
- `right_rubber_hand.stl` - Rubber hand covering

---

## 4. Joint Specification (29 DOF - Degrees of Freedom)

### Complete Joint List

#### **Lower Body Joints: 12 DOF**

**Left Leg (6 DOF)**
1. `left_hip_pitch_joint` - Axis: Y (pitch) - Range: [-1.5307, 1.5798] rad - Effort: 88 N⋅m
2. `left_hip_roll_joint` - Axis: X (roll) - Range: [-0.5236, 1.2671] rad - Effort: 88 N⋅m
3. `left_hip_yaw_joint` - Axis: Z (yaw) - Range: [-1.2576, 1.2576] rad - Effort: 88 N⋅m
4. `left_knee_joint` - Axis: Y (pitch) - Range: [-0.087267, 2.8798] rad - Effort: 139 N⋅m
5. `left_ankle_pitch_joint` - Axis: Y (pitch) - Range: [-0.87267, 0.5236] rad - Effort: 50 N⋅m
6. `left_ankle_roll_joint` - Axis: X (roll) - Range: [-0.2618, 0.2618] rad - Effort: 50 N⋅m

**Right Leg (6 DOF)**
7. `right_hip_pitch_joint` - Axis: Y (pitch) - Range: [-1.5307, 1.5798] rad - Effort: 88 N⋅m
8. `right_hip_roll_joint` - Axis: X (roll) - Range: [-1.2671, 0.5236] rad - Effort: 88 N⋅m
9. `right_hip_yaw_joint` - Axis: Z (yaw) - Range: [-1.2576, 1.2576] rad - Effort: 88 N⋅m
10. `right_knee_joint` - Axis: Y (pitch) - Range: [-0.087267, 2.8798] rad - Effort: 139 N⋅m
11. `right_ankle_pitch_joint` - Axis: Y (pitch) - Range: [-0.87267, 0.5236] rad - Effort: 50 N⋅m
12. `right_ankle_roll_joint` - Axis: X (roll) - Range: [-0.2618, 0.2618] rad - Effort: 50 N⋅m

#### **Torso Joints: 3 DOF**

13. `waist_yaw_joint` - Axis: Z (yaw) - Range: [-0.52, 0.52] rad - Effort: 88 N⋅m
14. `waist_roll_joint` - Axis: X (roll) - Range: [-0.52, 0.52] rad - Effort: 50 N⋅m
15. `waist_pitch_joint` - Axis: Y (pitch) - Range: [-0.52, 0.52] rad - Effort: 50 N⋅m

#### **Left Arm Joints: 7 DOF**

16. `left_shoulder_pitch_joint` - Axis: Y (pitch) - Range: [-1.5882, 1.5882] rad - Effort: 25 N⋅m
17. `left_shoulder_roll_joint` - Axis: X (roll) - Range: [-1.5882, 2.2515] rad - Effort: 25 N⋅m
18. `left_shoulder_yaw_joint` - Axis: Z (yaw) - Range: [-1.5882, 1.5882] rad - Effort: 25 N⋅m
19. `left_elbow_joint` - Axis: Y (pitch) - Range: [-1.0472, 1.5882] rad - Effort: 25 N⋅m
20. `left_wrist_roll_joint` - Axis: X (roll) - Range: [-1.614429558, 1.614429558] rad - Effort: 25 N⋅m
21. `left_wrist_pitch_joint` - Axis: Y (pitch) - Range: [-1.614429558, 1.614429558] rad - Effort: 5 N⋅m
22. `left_wrist_yaw_joint` - Axis: Z (yaw) - Range: [-1.614429558, 1.614429558] rad - Effort: 5 N⋅m

#### **Right Arm Joints: 7 DOF**

23. `right_shoulder_pitch_joint` - Axis: Y (pitch) - Range: [-1.5882, 1.5882] rad - Effort: 25 N⋅m
24. `right_shoulder_roll_joint` - Axis: X (roll) - Range: [-2.2515, 1.5882] rad - Effort: 25 N⋅m
25. `right_shoulder_yaw_joint` - Axis: Z (yaw) - Range: [-1.5882, 1.5882] rad - Effort: 25 N⋅m
26. `right_elbow_joint` - Axis: Y (pitch) - Range: [-1.0472, 1.5882] rad - Effort: 25 N⋅m
27. `right_wrist_roll_joint` - Axis: X (roll) - Range: [-1.614429558, 1.614429558] rad - Effort: 25 N⋅m
28. `right_wrist_pitch_joint` - Axis: Y (pitch) - Range: [-1.614429558, 1.614429558] rad - Effort: 5 N⋅m
29. `right_wrist_yaw_joint` - Axis: Z (yaw) - Range: [-1.614429558, 1.614429558] rad - Effort: 5 N⋅m

### DOF Breakdown Summary

| Body Region | DOF Count | Joints |
|---|---|---|
| **Legs** | 12 | Hip (pitch, roll, yaw) × 2, Knee × 2, Ankle (pitch, roll) × 2 |
| **Torso/Waist** | 3 | Yaw, Roll, Pitch |
| **Arms** | 14 | Shoulder (pitch, roll, yaw) × 2, Elbow × 2, Wrist (roll, pitch, yaw) × 2 |
| **TOTAL** | **29** | Complete humanoid kinematic chain |

---

## 5. Robot Configuration (Python)

**Location**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/robot_configs/g1.py`

This Python file defines:
- Robot name and asset paths
- Kinematic chain structure extracted from MJCF/URDF
- Per-DOF control parameters (stiffness, damping, effort limits)
- Simulator-specific physics parameters (for IsaacGym, IsaacLab, Newton, Genesis, MuJoCo)
- Joint angle limits and velocity limits
- Mass and inertia properties

---

## 6. Motion Data References

**Motion File Format**: PyTorch tensors (`.pt` files)  
**Example Path**: `data/motion_for_trackers/g1_bones_seed_mini.pt`

Motion files contain:
- Ground truth trajectories (gts)
- Ground truth rotations (grs)
- Ground truth velocities (gvs)
- Ground truth angular velocities (gavs)
- DOF positions (dps)
- DOF velocities (dvs)

---

## 7. Pre-trained Models Reference

**Location**: `data/pretrained_models/motion_tracker/`

Available pre-trained G1 models:
- `g1-bones-deploy/last.ckpt` - Main deployment model
- `soma-bones/last.ckpt` - Alternative SOMA configuration

---

## 8. Key ProtoMotions Features Supporting G1

### Multi-Simulator Support
The same G1 model files work across:
- **IsaacGym** - GPU-accelerated NVIDIA simulator
- **IsaacLab** - Extended IsaacGym framework
- **Newton** - Physics-based simulator (v1.0.0+)
- **Genesis** - Modern differentiable simulator
- **MuJoCo** - CPU-friendly simulator (single env only)

### RL Algorithms
- **PPO** - Proximal Policy Optimization
- **AMP** - Adversarial Motion Priors
- **ASE** - Adaptive Skill Encoding
- **MaskedMimic** - Motion imitation with masks

### Key Components
- **MotionLib** - Motion capture loading (SLERP interpolation for quaternions)
- **PoseLib** - Forward kinematics computation (batched, multi-horizon)
- **SceneLib** - Environment objects and collision meshes
- **Terrain** - Procedural environment generation

---

## 9. Kinematic Structure Summary

### Kinematic Chain Hierarchy
```
pelvis (root)
├── Left Leg
│   ├── left_hip_pitch → left_hip_roll → left_hip_yaw
│   ├── left_knee
│   └── left_ankle_pitch → left_ankle_roll
├── Right Leg
│   ├── right_hip_pitch → right_hip_roll → right_hip_yaw
│   ├── right_knee
│   └── right_ankle_pitch → right_ankle_roll
├── Torso
│   ├── waist_yaw → waist_roll → waist_pitch
│   └── head
├── Left Arm
│   ├── left_shoulder_pitch → left_shoulder_roll → left_shoulder_yaw
│   ├── left_elbow
│   └── left_wrist_roll → left_wrist_pitch → left_wrist_yaw
└── Right Arm
    ├── right_shoulder_pitch → right_shoulder_roll → right_shoulder_yaw
    ├── right_elbow
    └── right_wrist_roll → right_wrist_pitch → right_wrist_yaw
```

### Floating Base
MuJoCo configurations support floating base (XYZ position + quaternion rotation) for free-space humanoid locomotion.

---

## 10. File Access Summary

### Quick Reference Table

| File Type | Count | Base Directory | Full Path Example |
|-----------|-------|-----------------|-------------------|
| URDF | 1 | `protomotions/data/assets/urdf/for_retargeting/` | `./ref_repo/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf` |
| MJCF (XML) | 6 | `protomotions/data/assets/mjcf/` | `./ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_bm.xml` |
| STL Meshes | 63 | `protomotions/data/assets/mesh/G1/` | `./ref_repo/ProtoMotions/protomotions/data/assets/mesh/G1/pelvis.stl` |
| Python Config | 1 | `protomotions/robot_configs/` | `./ref_repo/ProtoMotions/protomotions/robot_configs/g1.py` |
| Motion Data | Multiple | `data/motion_for_trackers/` | `./data/motion_for_trackers/g1_bones_seed_mini.pt` |

---

## 11. Integration with Embodied Pipeline

### Usage in Training
```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --num-envs 4096 \
    --batch-size 16384
```

### Usage in Inference
```bash
python protomotions/inference_agent.py \
    --checkpoint data/pretrained_models/motion_tracker/g1-bones-deploy/last.ckpt \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt \
    --simulator isaacgym --num-envs 16
```

### Cross-Simulator Evaluation (Sim2Sim)
The same model can be deployed across simulators for robustness testing.

---

## Documentation References

- **ProtoMotions CLAUDE.md**: Project overview, architecture, and common commands
- **Robot Configuration Format**: `examples/experiments/format.py`
- **Main Entry Point**: `protomotions/train_agent.py` (comprehensive config system documentation)
- **MdpComponent System**: `protomotions/envs/mdp_component.py`
- **Context Path System**: `protomotions/envs/context_views.py`

---

## Summary

The Unitree G1 model in ProtoMotions is a **complete 29-DOF humanoid** with:
- **1 URDF** for kinematic definition
- **6 MuJoCo XML variants** for different simulation scenarios
- **63 STL mesh files** for complete geometric representation
- **29 actively controlled joints** (12 legs, 3 torso, 14 arms)
- **Simulator-agnostic** configuration supporting 5 different physics engines
- **Production-ready** deployment with pre-trained models

All files are stored relative to the ProtoMotions root directory and use mesh references for modular asset management.

