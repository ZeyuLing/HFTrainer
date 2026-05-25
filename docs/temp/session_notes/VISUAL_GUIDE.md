# Visual Guide: SMPL-X Pose Processing

## 1. Joint Hierarchy Visualization

```
SMPL-X 55 Joints (Full Model)
│
├─ Joints 0-21: SMPL Core (22 joints) ◄── SELECTED FOR SMPL-22
│   ├─ 0: Pelvis (root)
│   ├─ 1-4: Left leg
│   ├─ 5-8: Right leg
│   ├─ 9-14: Spine & head
│   ├─ 15-20: Left arm
│   └─ 21: Right shoulder
│
├─ Joints 22-51: SMPL-H Extensions (30 joints)
│   ├─ 22: Jaw
│   ├─ 23-24: Eyes
│   └─ 25-51: Hand articulation (26 fingers/palm)
│
└─ Joints 52-54: SMPL-X Only (3 joints)
    └─ Additional hand/expression joints
```

**SMPL-22 Selection**: `[0, 1, 2, ..., 21]` - first 22 consecutive joints

---

## 2. Rotation Representation Space

```
┌─────────────────────────────────────────────────────────────────┐
│                    Rotation Representations                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Axis-Angle (3D)          Rotation Matrix (3×3)                 │
│  ┌─────────┐              ┌───────────────────┐                 │
│  │ θ_x     │              │ R00  R01  R02     │                 │
│  │ θ_y     │  ─(via)────→ │ R10  R11  R12     │                 │
│  │ θ_z     │     SciPy    │ R20  R21  R22     │                 │
│  │ (rad)   │              │ (9 params)        │                 │
│  └─────────┘              └───────────────────┘                 │
│                                  ↓                               │
│                         ┌────────────────┐                       │
│                         │  6D (Zhou'19)  │                       │
│                         │  Column-major  │                       │
│                         │ [R00,R10,R20   │                       │
│                         │  R01,R11,R21]  │                       │
│                         │  (6 params)    │                       │
│                         └────────────────┘                       │
│                                  ↓ (HyMotion)                    │
│                         ┌────────────────┐                       │
│                         │  6D Rearranged │                       │
│                         │  Row-major     │                       │
│                         │ [R00,R01       │                       │
│                         │  R10,R11       │                       │
│                         │  R20,R21]      │                       │
│                         │  (6 params)    │                       │
│                         └────────────────┘                       │
│        ↓ Alternative         ↓ Alternative      ↓ Alternative   │
│   Quaternion (4D)        Euler (3D)         Matrix (9D)         │
│   [w,x,y,z]              [θ_x,θ_y,θ_z]     [R00..R22]          │
│   normalized             (XYZ order)        flattened           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. The 6D Rearrangement: Column-Major → Row-Major

```
┌────────────────────────────────────────────────────────────────┐
│              Rotation Matrix Memory Layout                      │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Original Matrix:          Column-major 6D:                     │
│  ┌─────────────────┐       ┌──────────────────┐                │
│  │ R00  R01  R02   │       │ [R00, R10, R20,  │                │
│  │ R10  R11  R12   │  →    │  R01, R11, R21]  │                │
│  │ R20  R21  R22   │       │                  │                │
│  └─────────────────┘       └──────────────────┘                │
│       ↓ Values               ↓ Indices                          │
│  [0] [1] [2]               [0]  [1]  [2]  [3] [4] [5]           │
│   ^   ^   ^                                                      │
│   └─Column 0   Col 1 ─┐                                         │
│                        └─→ Stacked!                              │
│                                                                  │
│  HyMotion wants Row-major 6D:                                   │
│  ┌──────────────────────┐                                       │
│  │ [R00, R01,          │                                        │
│  │  R10, R11,          │                                        │
│  │  R20, R21]          │                                        │
│  └──────────────────────┘                                       │
│       ↓ Indices                                                  │
│   [0] [1] [2] [3] [4] [5]                                       │
│                                                                  │
│  Permutation: [0, 3, 1, 4, 2, 5]                                │
│               ↓                                                  │
│  col_major[0]   → row_major[0]   ✓  (R00)                       │
│  col_major[3]   → row_major[1]   ✓  (R01)                       │
│  col_major[1]   → row_major[2]   ✓  (R10)                       │
│  col_major[4]   → row_major[3]   ✓  (R11)                       │
│  col_major[2]   → row_major[4]   ✓  (R20)                       │
│  col_major[5]   → row_major[5]   ✓  (R21)                       │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Complete Data Shape Evolution

```
Step 1: Input Data
┌───────────────────────────────┐
│  SMPL-X Poses                 │
│  Shape: [100, 165]            │
│  (100 frames, 55 joints×3)    │
│  Format: axis-angle (radians) │
└───────────────────────────────┘
         ↓ reshape
┌───────────────────────────────┐
│  [100, 55, 3]                 │
│  T=100, J=55, D=3             │
└───────────────────────────────┘

Step 2: Joint Selection (SMPL-22)
┌───────────────────────────────┐
│  Select indices [0:22]        │
│  [100, 22, 3]                 │
│  T=100, J=22, D=3             │
└───────────────────────────────┘
         ↓ flatten for conversion
┌───────────────────────────────┐
│  [2200, 3]  (T*J, 3)          │
│  Batch all rotations together │
└───────────────────────────────┘

Step 3: Rotation Conversion (to 6D)
┌─────────────────┐   ┌──────────────┐   ┌──────────────┐
│ [2200, 3]       │   │ [2200, 3, 3] │   │ [2200, 6]    │
│ axis-angle      │→→→│ matrix       │→→→│ 6D col-major │
└─────────────────┘   └──────────────┘   └──────────────┘
   (via SciPy)          (internal)          (intermediate)
         
         ↓ rearrange indices [0,3,1,4,2,5]
         
┌──────────────────┐
│ [2200, 6]        │
│ 6D row-major     │
│ (HyMotion style) │
└──────────────────┘

Step 4: Reshape and Return
┌───────────────────────────────┐
│  Reshape to [T, J, 6]         │
│  [100, 22, 6]                 │
└───────────────────────────────┘
         ↓ flatten
┌───────────────────────────────┐
│  Output: [100, 132]           │
│  T=100, J*D=22*6=132          │
│  dtype: float32               │
└───────────────────────────────┘
```

---

## 5. Output Layout for SMPL-22 + 6D

```
┌─────────────────────────────────────────────────────────┐
│  Final Output: [T=100, 132]                             │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Dimension Index → Value Meaning                         │
│  ───────────────────────────────────────────────────     │
│  [0]     → Joint 0, row-major 6D dim 0 (R00)           │
│  [1]     → Joint 0, row-major 6D dim 1 (R01)           │
│  [2]     → Joint 0, row-major 6D dim 2 (R10)           │
│  [3]     → Joint 0, row-major 6D dim 3 (R11)           │
│  [4]     → Joint 0, row-major 6D dim 4 (R20)           │
│  [5]     → Joint 0, row-major 6D dim 5 (R21)           │
│  ───────────────────────────────────────────────────     │
│  [6]     → Joint 1, row-major 6D dim 0 (R00)           │
│  [7]     → Joint 1, row-major 6D dim 1 (R01)           │
│  ...     → ...                                           │
│  [11]    → Joint 1, row-major 6D dim 5 (R21)           │
│  ───────────────────────────────────────────────────     │
│  ...     → (repeat for joints 2-20)                     │
│  ───────────────────────────────────────────────────     │
│  [126]   → Joint 21, row-major 6D dim 0 (R00)          │
│  [127]   → Joint 21, row-major 6D dim 1 (R01)          │
│  [128]   → Joint 21, row-major 6D dim 2 (R10)          │
│  [129]   → Joint 21, row-major 6D dim 3 (R11)          │
│  [130]   → Joint 21, row-major 6D dim 4 (R20)          │
│  [131]   → Joint 21, row-major 6D dim 5 (R21)          │
│                                                           │
│  Pattern: dims_per_joint = 6                            │
│           joint_idx = i // 6                            │
│           dim_idx = i % 6                               │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Comparison: Different Output Configurations

```
╔════════════════════════════════════════════════════════════════╗
║              Output Shape Examples (T=100 frames)              ║
╠════════════════════════════════════════════════════════════════╣
║                                                                 ║
║ ┌─────────────┬──────────────┬─────────────────────────────┐  ║
║ │ out_type    │ rot_type     │ Shape & Notes               │  ║
║ ├─────────────┼──────────────┼─────────────────────────────┤  ║
║ │ SMPL-22     │ axis_angle   │ [100, 66]   (22×3)         │  ║
║ │             │ 6d           │ [100, 132]  (22×6) ◄─ MAIN │  ║
║ │             │ quaternion   │ [100, 88]   (22×4)         │  ║
║ │             │ euler        │ [100, 66]   (22×3)         │  ║
║ ├─────────────┼──────────────┼─────────────────────────────┤  ║
║ │ SMPL-H (52) │ axis_angle   │ [100, 156]  (52×3)         │  ║
║ │             │ 6d           │ [100, 312]  (52×6)         │  ║
║ │             │ quaternion   │ [100, 208]  (52×4)         │  ║
║ │             │ euler        │ [100, 156]  (52×3)         │  ║
║ ├─────────────┼──────────────┼─────────────────────────────┤  ║
║ │ SMPL-X (55) │ axis_angle   │ [100, 165]  (55×3)         │  ║
║ │             │ 6d           │ [100, 330]  (55×6)         │  ║
║ │             │ quaternion   │ [100, 220]  (55×4)         │  ║
║ │             │ euler        │ [100, 165]  (55×3)         │  ║
║ └─────────────┴──────────────┴─────────────────────────────┘  ║
║                                                                 ║
║ Note: These are pose dimensions only (translation separate)    ║
║                                                                 ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 7. Function Call Trace

```
LoadSmplx55.transform()
    ↓
_process_one_person()
    ├─ process_transl()           ← handles translation
    └─ process_smplx_pose()       ← handles rotation ◄─── HERE
        ├─ Reshape to [T, 55, 3]
        ├─ Select joint subset
        ├─ Flatten to [T*J, 3]
        ├─ IF rot_type == "rotation_6d":
        │   ├─ axis_angle_to_rotation_6d()
        │   │   ├─ axis_angle_to_matrix()   [scipy path]
        │   │   └─ matrix_to_rotation_6d()  [stack cols]
        │   └─ Rearrange indices [0,3,1,4,2,5]
        │
        ├─ IF rot_type == "quaternion":
        │   └─ axis_angle_to_quaternion()
        │
        ├─ IF rot_type == "euler":
        │   ├─ axis_angle_to_matrix()
        │   └─ matrix_to_euler()
        │
        └─ Reshape to [T, J*D] and return
    ↓
Concatenate [trans | pose] → [T, D_trans + D_pose]
```

---

## 8. Quick Lookup: Dimension Formula

```
Dimension Calculator for process_smplx_pose
╔═══════════════════════════════════════════════════════════╗
║ Output Dimension = num_joints × dims_per_rotation        ║
╠═══════════════════════════════════════════════════════════╣
║                                                            ║
║  num_joints = {                                          ║
║    22,   if out_type == "smpl_22"                        ║
║    52,   if out_type == "smplh"                          ║
║    55,   if out_type == "smplx_55"                       ║
║  }                                                         ║
║                                                            ║
║  dims_per_rotation = {                                   ║
║    3,    if rot_type == "axis_angle" or "euler"         ║
║    4,    if rot_type == "quaternion"                     ║
║    6,    if rot_type == "rotation_6d"                    ║
║    9,    (not used in this function, but theoretical)    ║
║  }                                                         ║
║                                                            ║
║  Example: SMPL-22 + 6D → 22 × 6 = 132 ✓                 ║
║                                                            ║
╚═══════════════════════════════════════════════════════════╝
```

---

## 9. Edge Cases & Gotchas

```
┌─────────────────────────────────────────────────────────┐
│                     Edge Cases                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│ 1. SMPL-H Input (52 joints)                             │
│    ├─ Padded with zeros after joint 22 (jaw/eyes)      │
│    └─ Result: [T, 52, 3] → treated as SMPL-X           │
│                                                           │
│ 2. Only 6D Gets Rearranged                             │
│    ├─ quaternion, euler, axis_angle: NO rearrangement  │
│    └─ 6d: YES, uses [0,3,1,4,2,5] permutation         │
│                                                           │
│ 3. Output is Always float32                            │
│    └─ Explicit `.astype(np.float32)` at end            │
│                                                           │
│ 4. NaN Checking (in LoadSmplx55.transform)            │
│    └─ Raises error if NaNs found in final tensor       │
│                                                           │
│ 5. SciPy Used for NumPy Path                           │
│    ├─ numpy → scipy.spatial.transform.Rotation         │
│    └─ PyTorch → native torch implementations           │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 10. Code Path Decision Tree

```
                    Input received
                          ↓
                    Determine format
                    /           \
              [T,165]        [T,55,3]  or  [T,52,3]
                /                    \
          Reshape to              Use directly /
          [T,55,3]                Pad if SMPL-H
              ↓                         ↓
        ┌─────────────┐          ┌─────────────┐
        │ out_type?   │──────────│ out_type?   │
        └────┬────────┘          └────┬────────┘
             │                        │
    ┌────┬───┴────┬────┐    ┌────┬───┴────┬────┐
    │    │        │    │    │    │        │    │
  "22" "52" "55" ...  "22" "52" "55" ...
    │    │        │         │    │        │
    ↓    ↓        ↓         ↓    ↓        ↓
  Select        Select    Select
  idx[0:22]     idx[0:52] idx[0:55]
    ↓    ↓        ↓         ↓    ↓        ↓
  [T,22,3]    [T,52,3]   [T,55,3]
      ↓           ↓           ↓
  ┌──────────┐
  │ rot_type?│
  ├──────────┤
  │    │    │ │
  │    │    │ └──→ "euler"
  │    │    └─────→ "quaternion"
  │    └──────────→ "rotation_6d" ◄─── SPECIAL (rearrange!)
  └──────────────→ "axis_angle"
       ↓
    flatten
      ↓
    convert
      ↓
    reshape to [T, J*D]
      ↓
  return float32
```

