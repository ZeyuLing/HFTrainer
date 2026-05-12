# KIMODO Root Representation - Executive Summary

## Quick Answer: What is KIMODO's Root?

KIMODO's root representation is a **5-dimensional encoding** that separates trajectory control from heading control:

```
KIMODO Root (dims 0-5 of 333-dim feature vector):
├── smooth_root_pos [3 dims]: [x_smooth, y_raw, z_smooth]
│   └── XZ plane: ADMM-smoothed trajectory (margin ±0.06m)
│   └── Y axis: Raw pelvis height (NOT smoothed)
└── global_root_heading [2 dims]: [cos(ψ), sin(ψ)]
    └── Explicit heading angle around Y-axis (radians)
```

## Why This Design?

| Aspect | KIMODO Approach | Benefit |
|---|---|---|
| **Smoothing** | Explicit ADMM filter | Animator-friendly trajectory (no jitter) |
| **Heading** | Explicit 2D encoding | Direct heading control without IK |
| **Coordinates** | Global (world-frame) | World-space constraints don't need IK |
| **Flexibility** | Trajectory + Heading separated | Independent control of path and orientation |

---

## Comparison: KIMODO vs HyMotion M2M

### Root Representation Size
- **KIMODO**: 5 dims (smooth_root[3] + heading[2])
- **HyMotion**: 6 dims (abs_trans[3] + rel_trans[3])

### Root Quality
- **KIMODO**: Smoothed XZ, raw Y → High-quality stable trajectories
- **HyMotion**: All raw → Noisier but more faithful to source

### Heading Representation
- **KIMODO**: Explicit [cos ψ, sin ψ] → Can directly constrain
- **HyMotion**: Implicit in motion → Must solve inverse problem

### Joint Rotations
- **KIMODO**: Global (27×6=162 dims) → Direct world-space control
- **HyMotion**: Local/relative (22×6=132 dims) → Requires FK for world positions

---

## How to Convert Between KIMODO and SMPL

### SMPL → KIMODO
```
1. FK: local_rots + raw_pos → global_rots + global_pos
2. Smooth: raw_pelvis_pos → smooth_root_pos (ADMM, margin ±0.06m)
3. Extract: hip_vector → heading [cos ψ, sin ψ]
4. Localize: global_pos - [smooth_root_x, 0, smooth_root_z]
5. Encode: global_rots → 6D continuous
```

### KIMODO → SMPL
```
1. Decode: 6D rotations → rotation matrices
2. IFK: global_rots → local_rots via parent^T @ child
3. Restore: local_pos + [smooth_root_x, 0, smooth_root_z]
4. Extract: root_pos from first joint position
```

**⚠️ Key Property**: Conversion is **lossy** (smoothing removes high-freq noise)

---

## The 333-Dimension Feature Vector

```
[0:3]       smooth_root_pos           (3 dims)
[3:5]       global_root_heading       (2 dims)
[5:86]      local_joints_positions    (27×3 = 81 dims)
[86:248]    global_rot_data           (27×6 = 162 dims)
[248:329]   velocities                (27×3 = 81 dims)
[329:333]   foot_contacts             (4 dims)
```

### Critical Components

**Smooth Root Position [0:3]**
- Heavily smoothed XZ trajectory (ADMM algorithm, margin ±0.06m)
- Raw Y height (preserves vertical motion)
- Removes foot skating and jitter

**Global Root Heading [3:5]**
- Encoded as [cos(ψ), sin(ψ)]
- ψ = heading angle (computed from hip vector)
- Avoids discontinuities at ±π

**Local Joint Positions [5:86]**
- Relative to smooth_root in XZ
- Absolute in Y (individual joint heights)
- Enables position translation via root movement

**Global Rotations [86:248]**
- **World-frame** (NOT local/relative like SMPL)
- 6D continuous representation per joint
- Enables direct world-space constraint imputation

**Velocities [248:329]**
- Derived from positions: vel = fps × (pos[t] - pos[t-1])
- Not constrained during diffusion
- Used in smoothness loss computation

**Foot Contacts [329:333]**
- Binary flags: [L_heel, L_toe, R_heel, R_toe]
- Computed from position & velocity
- Threshold: height < 0.15m AND velocity < 0.10 m/s

---

## Constraint System: Why It Matters

KIMODO supports **direct imputation** of constraints at every diffusion step:

```
Constraint Imputation (before each denoising step):

x_t = x_t * (1 - motion_mask) + observed_motion * motion_mask

Where:
  motion_mask[t, dims] = True if constrained
  observed_motion[t, dims] = target value
  x_t = noisy prediction

Model input: concat([x_t, motion_mask]) ← Tells model which dims are constrained
```

### Supported Constraints
1. **smooth_root_2d** → Constrain trajectory (XZ plane)
2. **root_y_pos** → Constrain height
3. **global_root_heading** → Constrain heading angle
4. **global_joints_rots** → Constrain joint orientations
5. **global_joints_positions** → Constrain world positions

All can be mixed and matched during diffusion.

---

## Key Implementation Details

### ADMM Smooth Root Algorithm
- **Objective**: Minimize ||x - x_original||² + λ||∇²x||²
- **Constraints**: ||x[t]|| ≤ 0.06 m per frame (margin)
- **Method**: Multigrid ADMM (coarse→fine resolution)
- **Result**: Smooth stable trajectory, removes jitter

### FK/IFK Rotation Conversion
```python
# Forward (Local → Global)
global[0] = local[0]
for j in 1..21:
    global[j] = global[parent[j]] @ local[j]

# Inverse (Global → Local)
local[0] = global[0]
for j in 1..21:
    local[j] = global[parent[j]]^T @ global[j]
```

### 6D Rotation Format
- Row-major (training): [R[0,0], R[0,1], R[1,0], R[1,1], R[2,0], R[2,1]]
- Column-major (library): [R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1]]
- Reorder indices: _ROW_TO_COL = [0,2,4,1,3,5] and _COL_TO_ROW = [0,3,1,4,2,5]

---

## Implementation Files

### KIMODO Source
- `kimodo/motion_rep/reps/kimodo_motionrep.py` ← Main implementation
- `kimodo/motion_rep/smooth_root.py` ← ADMM smoother
- `kimodo/motion_rep/feature_utils.py` ← Helpers (heading, velocity)

### HyMotion Integration
- `hftrainer/datasets/motion/motionhub/transforms/fk_utils.py` ← FK/IFK, row/col conversion
- `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` ← SMPL loading

---

## Practical Conversion Code

```python
# SMPL → KIMODO
from kimodo import KimodoMotionRep
kimodo_rep = KimodoMotionRep(skeleton=soma30, fps=30)
kimodo_features = kimodo_rep(
    local_joint_rots=local_rots[None],  # [1,T,J,3,3]
    root_positions=abs_trans[None],      # [1,T,3]
    to_normalize=False
)  # → [1,T,333]

# KIMODO → SMPL
output = kimodo_rep.inverse(
    features=kimodo_features,
    is_normalized=False
)
# → local_rot_mats[T,J,3,3], root_positions[T,3]
```

---

## Summary: Why KIMODO's Design Works

1. **Smoothing**: ADMM filter removes jitter while respecting motion bounds
2. **Explicit heading**: Direct control without IK complexity
3. **Global rotations**: World-space constraints don't need inverse kinematics
4. **Separated root**: Trajectory and heading controlled independently
5. **Compact encoding**: 5 dims (smooth root + heading) vs 6 dims (abs + rel)

This design makes KIMODO ideal for **animator-friendly constraint-based motion generation**.

---

## References

Full technical details available in:
- **KIMODO_ROOT_ANALYSIS.md** (8000+ lines) - Complete breakdown
- **KIMODO_QUICK_REFERENCE.txt** (400+ lines) - Visual reference guide

