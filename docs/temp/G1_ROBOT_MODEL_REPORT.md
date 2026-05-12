# UNITREE G1 ROBOT MODEL - COMPLETE FILE PATHS AND SPECIFICATIONS

## EXECUTIVE SUMMARY

The Unitree G1 robot model used by the embodied pipeline in ProtoMotions is a **29-DOF humanoid** with full-body kinematics including legs, torso, and articulated arms with hands. Model files are located in the ProtoMotions reference repository.

---

## 1. ABSOLUTE FILE PATHS

### 1.1 URDF (Unified Robot Description Format)

**Primary URDF File:**
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/urdf/for_retargeting/g1.urdf
```

- **Model Name:** `g1_29dof`
- **Size:** ~100 KB
- **Format:** XML (URDF standard)
- **Purpose:** Robot kinematic structure, link geometry, and inertia definitions
- **Mesh References:** Relative path `../mesh/G1/` pointing to STL files

### 1.2 MuJoCo XML (MJCF - MuJoCo Configuration Format)

Five model variants available:

| Variant | Filename | Path | Purpose |
|---------|----------|------|---------|
| **Basic with Meshes** | `g1_bm.xml` | `.../data/assets/mjcf/g1_bm.xml` | Full mesh geometry |
| **Box Feet** | `g1_bm_box_feet.xml` | `.../data/assets/mjcf/g1_bm_box_feet.xml` | Simplified feet for stability |
| **Primitive Only** | `g1_bm_no_mesh_box_feet.xml` | `.../data/assets/mjcf/g1_bm_no_mesh_box_feet.xml` | No mesh (capsules/spheres only) |
| **Holonomic** | `g1_holo.xml` | `.../data/assets/mjcf/g1_holo.xml` | Omnidirectional base |
| **Holo + Box Feet** | `g1_holo_compat_box_feet.xml` | `.../data/assets/mjcf/g1_holo_compat_box_feet.xml` | Combined variant |

**Full Path Base:**
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/mjcf/
```

### 1.3 Mesh/STL Files

**Directory:**
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/mesh/G1/
```

- **Total Files:** 61 STL mesh files
- **Format:** STL (Stereolithography) - binary/ASCII mesh format
- **Size:** ~15-20 MB total
- **Includes:** Complete body geometry for all links

### 1.4 Configuration Files

**Python Robot Configuration:**
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/robot_configs/g1.py
```

**USD Assets (Isaac Sim):**
```
/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions/protomotions/data/assets/usd/
├── g1_bm/
├── g1_bm_box_feet/
├── g1_holo_compat/
└── g1_holo_compat_box_feet/
```

---

## 2. JOINT SPECIFICATION

### 2.1 Overall DOF Summary

| Component | DOFs | Details |
|-----------|------|---------|
| **Legs** | 12 | 6 per leg (hip×3, knee×1, ankle×2) |
| **Torso** | 3 | Waist (yaw, roll, pitch) |
| **Arms** | 14 | 7 per arm (shoulder×3, elbow×1, wrist×3) |
| **TOTAL** | **29** | Excluding floating base |

### 2.2 Complete Joint Names (29 DOFs)

#### LEFT LEG (6 DOFs)
1. `left_hip_pitch_joint` — Hip forward/back (Y-axis)
2. `left_hip_roll_joint` — Hip abduction/adduction (X-axis)
3. `left_hip_yaw_joint` — Hip rotation (Z-axis)
4. `left_knee_joint` — Knee extension/flexion (Y-axis)
5. `left_ankle_pitch_joint` — Ankle forward/back (Y-axis)
6. `left_ankle_roll_joint` — Ankle inversion/eversion (X-axis)

#### RIGHT LEG (6 DOFs)
7. `right_hip_pitch_joint` — Hip forward/back (Y-axis)
8. `right_hip_roll_joint` — Hip abduction/adduction (X-axis)
9. `right_hip_yaw_joint` — Hip rotation (Z-axis)
10. `right_knee_joint` — Knee extension/flexion (Y-axis)
11. `right_ankle_pitch_joint` — Ankle forward/back (Y-axis)
12. `right_ankle_roll_joint` — Ankle inversion/eversion (X-axis)

#### TORSO (3 DOFs)
13. `waist_yaw_joint` — Torso rotation (Z-axis)
14. `waist_roll_joint` — Lateral flexion (X-axis)
15. `waist_pitch_joint` — Forward/back bending (Y-axis)

#### LEFT ARM (7 DOFs)
16. `left_shoulder_pitch_joint` — Shoulder forward/back (Y-axis)
17. `left_shoulder_roll_joint` — Shoulder abduction/adduction (X-axis)
18. `left_shoulder_yaw_joint` — Shoulder rotation (Z-axis)
19. `left_elbow_joint` — Elbow flexion/extension (Y-axis)
20. `left_wrist_roll_joint` — Wrist pronation/supination (X-axis)
21. `left_wrist_pitch_joint` — Wrist flexion/extension (Y-axis)
22. `left_wrist_yaw_joint` — Wrist radial/ulnar deviation (Z-axis)

#### RIGHT ARM (7 DOFs)
23. `right_shoulder_pitch_joint` — Shoulder forward/back (Y-axis)
24. `right_shoulder_roll_joint` — Shoulder abduction/adduction (X-axis)
25. `right_shoulder_yaw_joint` — Shoulder rotation (Z-axis)
26. `right_elbow_joint` — Elbow flexion/extension (Y-axis)
27. `right_wrist_roll_joint` — Wrist pronation/supination (X-axis)
28. `right_wrist_pitch_joint` — Wrist flexion/extension (Y-axis)
29. `right_wrist_yaw_joint` — Wrist radial/ulnar deviation (Z-axis)

### 2.3 Joint Parameters from MJCF (g1_bm.xml)

All joints specify:
- **Type:** `revolute` (single-axis rotation)
- **Range:** Joint limits in radians
- **Control:** Motor actuators with effort limits
- **Damping:** Friction and damping coefficients
- **Stiffness:** PD control stiffness parameters
- **Armature:** Motor inertia

Example (left_hip_pitch_joint):
```xml
<joint name="left_hip_pitch_joint" 
       axis="0 1 0" 
       limited="true" 
       range="-2.5307 2.8798" 
       actuatorfrcrange="-88 88" 
       stiffness="100" 
       damping="10" 
       armature="0.03" 
       frictionloss="0.03" />
```

---

## 3. MESH FILES (61 TOTAL)

### 3.1 Mesh Organization

**Directory:** `/apdcephfs/.../mesh/G1/`

| Category | Count | Files |
|----------|-------|-------|
| **Body** | 31 | pelvis, torso, waist, head, supports, constraints |
| **Legs** | 12 | hip/knee/ankle links (×2 legs) |
| **Arms** | 18 | shoulders, elbows, wrists, hands (×2 arms) |

### 3.2 Complete Mesh List

#### Pelvis & Torso (12 files)
- `pelvis.stl`
- `pelvis_contour_link.stl`
- `waist_yaw_link.stl`
- `waist_yaw_link_rev_1_0.stl`
- `waist_roll_link.stl`
- `waist_roll_link_rev_1_0.stl`
- `torso_link.stl`
- `torso_link_rev_1_0.stl`
- `waist_support_link.stl`
- `head_link.stl`
- `logo_link.stl`
- `left_rubber_hand.stl`, `right_rubber_hand.stl` (part of hands)

#### Constraints (4 files)
- `torso_constraint_L_link.stl`
- `torso_constraint_L_rod_link.stl`
- `torso_constraint_R_link.stl`
- `torso_constraint_R_rod_link.stl`
- `waist_constraint_L.stl`
- `waist_constraint_R.stl`

#### LEFT LEG (6 files)
- `left_hip_pitch_link.stl`
- `left_hip_roll_link.stl`
- `left_hip_yaw_link.stl`
- `left_knee_link.stl`
- `left_ankle_pitch_link.stl`
- `left_ankle_roll_link.stl`

#### RIGHT LEG (6 files)
- `right_hip_pitch_link.stl`
- `right_hip_roll_link.stl`
- `right_hip_yaw_link.stl`
- `right_knee_link.stl`
- `right_ankle_pitch_link.stl`
- `right_ankle_roll_link.stl`

#### LEFT ARM (8 files)
- `left_shoulder_pitch_link.stl`
- `left_shoulder_roll_link.stl`
- `left_shoulder_yaw_link.stl`
- `left_elbow_link.stl`
- `left_wrist_roll_link.stl`
- `left_wrist_pitch_link.stl`
- `left_wrist_yaw_link.stl`
- `left_hand_palm_link.stl`
- `left_hand_thumb_[0-2]_link.stl` (3 files)
- `left_hand_index_[0-1]_link.stl` (2 files)
- `left_hand_middle_[0-1]_link.stl` (2 files)
- `left_wrist_roll_rubber_hand.stl`

#### RIGHT ARM (8 files)
- `right_shoulder_pitch_link.stl`
- `right_shoulder_roll_link.stl`
- `right_shoulder_yaw_link.stl`
- `right_elbow_link.stl`
- `right_wrist_roll_link.stl`
- `right_wrist_pitch_link.stl`
- `right_wrist_yaw_link.stl`
- `right_hand_palm_link.stl`
- `right_hand_thumb_[0-2]_link.stl` (3 files)
- `right_hand_index_[0-1]_link.stl` (2 files)
- `right_hand_middle_[0-1]_link.stl` (2 files)
- `right_wrist_roll_rubber_hand.stl`

---

## 4. KINEMATIC CHAIN STRUCTURE

```
pelvis (floating base)
├── LEFT LEG
│   ├── left_hip_pitch_link
│   │   └── left_hip_roll_link
│   │       └── left_hip_yaw_link
│   │           └── left_knee_link
│   │               └── left_ankle_pitch_link
│   │                   └── left_ankle_roll_link
│
├── RIGHT LEG
│   ├── right_hip_pitch_link
│   │   └── right_hip_roll_link
│   │       └── right_hip_yaw_link
│   │           └── right_knee_link
│   │               └── right_ankle_pitch_link
│   │                   └── right_ankle_roll_link
│
├── TORSO
│   ├── waist_yaw_link
│   │   └── waist_roll_link
│   │       └── torso_link
│   │           ├── LEFT ARM
│   │           │   ├── left_shoulder_pitch_link
│   │           │   │   └── left_shoulder_roll_link
│   │           │   │       └── left_shoulder_yaw_link
│   │           │   │           └── left_elbow_link
│   │           │   │               └── left_wrist_roll_link
│   │           │   │                   └── left_wrist_pitch_link
│   │           │   │                       └── left_wrist_yaw_link
│   │           │
│   │           └── RIGHT ARM
│   │               ├── right_shoulder_pitch_link
│   │               │   └── right_shoulder_roll_link
│   │               │       └── right_shoulder_yaw_link
│   │               │           └── right_elbow_link
│   │               │               └── right_wrist_roll_link
│   │               │                   └── right_wrist_pitch_link
│   │               │                       └── right_wrist_yaw_link
```

---

## 5. MODEL PHYSICS PROPERTIES

### 5.1 Mass Distribution (from URDF)

| Component | Mass (kg) | Inertia |
|-----------|-----------|---------|
| Pelvis | 3.813 | Specified in URDF |
| Torso | 8.562 | High inertia (trunk is heavy) |
| L/R Hip Pitch | 1.35 × 2 | 2.70 |
| L/R Hip Roll | 1.52 × 2 | 3.04 |
| L/R Hip Yaw | 1.702 × 2 | 3.404 |
| L/R Knee | 1.932 × 2 | 3.864 |
| L/R Ankle | 0.682 × 2 | 1.364 |
| **Legs Total** | ~22 kg | — |
| L/R Shoulder Pitch | 0.718 × 2 | 1.436 |
| L/R Shoulder Roll | 0.643 × 2 | 1.286 |
| L/R Shoulder Yaw | 0.734 × 2 | 1.468 |
| L/R Elbow | 0.6 × 2 | 1.2 |
| L/R Wrist | 0.5+ × 2 | ~1.0 |
| **Arms Total** | ~6 kg | — |

### 5.2 Joint Effort Limits (from MJCF)

| Joint Group | Max Effort (N⋅m) |
|-------------|------------------|
| Hip | 88 |
| Knee | 139 |
| Ankle | 50 |
| Waist | 50 |
| Shoulder | 25 |
| Elbow | 25 |
| Wrist | 5 |

---

## 6. USAGE IN EMBODIED PIPELINE

### 6.1 ProtoMotions Framework Integration

The G1 model is used with:
- **Simulators:** IsaacGym, IsaacLab, Newton, Genesis, MuJoCo
- **Algorithms:** PPO, AMP, ASE, MaskedMimic
- **Tasks:** Motion tracking, imitation learning, real2sim control

### 6.2 Loading in Code

**From Python config:**
```python
# In robot_configs/g1.py
robot_config = G1RobotConfig(
    urdf_path="path/to/g1.urdf",
    mjcf_path="path/to/g1_bm.xml",
    mesh_path="path/to/mesh/G1/",
)
```

**Motion tracking example:**
```bash
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --motion-file data/motion_for_trackers/g1_bones_seed_mini.pt
```

---

## 7. SUMMARY TABLE

| Attribute | Value |
|-----------|-------|
| **Robot Name** | Unitree G1 |
| **Model Variant** | 29-DOF Humanoid |
| **Total Joints** | 29 (+ 1 floating base) |
| **Legs** | 2 × 6-DOF |
| **Torso** | 3-DOF |
| **Arms** | 2 × 7-DOF |
| **URDF Path** | `.../urdf/for_retargeting/g1.urdf` |
| **MJCF Variants** | 5 (basic, box_feet, holo, etc.) |
| **MJCF Path** | `.../mjcf/g1_*.xml` |
| **Mesh Files** | 61 STL files |
| **Mesh Path** | `.../mesh/G1/` |
| **Estimated Total Mass** | ~35-40 kg |
| **Actuators** | 29 motors (all joints) |
| **Sensors** | IMU (on torso) |

---

## 8. REFERENCE DOCUMENTATION

- **Framework:** ProtoMotions3 (GPU-accelerated physics sim + RL)
- **License:** Apache 2.0
- **Repository:** `/apdcephfs/AILab_DHA/.../ref_repo/ProtoMotions/`
- **Config System:** Python-based (experiment files in `examples/experiments/`)
- **Multi-simulator:** IsaacGym/Lab, Newton, Genesis, MuJoCo

