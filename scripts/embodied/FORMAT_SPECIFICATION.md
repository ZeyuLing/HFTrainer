# HY-Motion 1.0 Format Specification & Verification Guide

## Quick Reference

### Motion Dimension Layout

```
motion_201 (201 dimensions total)
├─ [0:3]       Translation (3D)
├─ [3:9]       Pelvis rot6d (6D) ← Global orientation
├─ [9:135]     Joints 1-21 rot6d (126D) ← Local rotations, 6D each
└─ [135:201]   Joint positions (66D) ← Retained in motion_201 only

motion_135 (135 dimensions) - USED BY PIPELINE
├─ [0:3]       Translation (3D)
└─ [3:135]     22 × rot6d (132D)
```

### Data Flow Diagram

```
HyMotion T2M Model (Diffusion Transformer)
    ↓
    Latent output (normalized, shape B×T×201)
    ↓
    Denormalization: x_denorm = x_norm × std + mean
    ↓
    motion_201 (T, 201)
    ↓
    Extract: motion_135 = motion_201[:, :135]
    ↓
    [Optional] Smoothing (Savitzky-Golay)
    ↓
    motion_135 NPZ file
    ↓
    Reshape: rot6d (T, 22, 6)
    ↓
    Gram-Schmidt: rot6d → rotmat (3×3)
    ↓
    Axis-angle: rotmat → aa (3D)
    ↓
    Split: root_orient (T, 3) + pose_body (T, 63)
    ↓
    SMPL-X NPZ file
    ↓
    [Rest of pipeline: GMR, ProtoMotions, rendering, etc.]
```

---

## Rot6d Format: THE CRITICAL DETAIL

### Why Row-Major vs Column-Major Matters

**HyMotion outputs 6D rotations in ROW-MAJOR order:**
```
6D vector: [e0, e1, e2, e3, e4, e5]

This represents a 3×3 rotation matrix in row-major order:
R = [ e0  e1 ]  ← Row 0, columns 0-1
    [ e2  e3 ]  ← Row 1, columns 0-1
    [ e4  e5 ]  ← Row 2, columns 0-1
```

**But Gram-Schmidt expects COLUMN-MAJOR layout:**
```
We need to extract:
Column 1: [e0, e2, e4]  ← a1
Column 2: [e1, e3, e5]  ← a2
```

**Reordering [0, 2, 4, 1, 3, 5] converts row-major to column-major:**
```
Before: [0, 1, 2, 3, 4, 5]  (row-major)
After:  [0, 2, 4, 1, 3, 5]  (column-major)
        ↓  ↓  ↓  ↓  ↓  ↓
        a1_0 a1_1 a1_2 a2_0 a2_1 a2_2
```

### Gram-Schmidt Process (Step-by-Step)

```python
# CRITICAL: Apply reorder FIRST
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]

# Extract columns
a1 = rot6d[..., :3]    # First 3 elements (now first column)
a2 = rot6d[..., 3:6]   # Last 3 elements (now second column)

# Step 1: Normalize first column
b1 = a1 / ||a1||

# Step 2: Gram-Schmidt orthogonalization
b2 = a2 - (b1 · a2) * b1
b2 = b2 / ||b2||

# Step 3: Cross product for third column
b3 = b1 × b2

# Result: 3×3 rotation matrix
R = [b1 | b2 | b3]  (columns stacked as [b1, b2, b3])
```

### Example: Numerical Verification

```python
import numpy as np

# Example rot6d (row-major from HyMotion)
rot6d_orig = np.array([1.0, 0.1, 0.05, 0.98, -0.1, 1.0])

# Step 1: Reorder to column-major
rot6d_reordered = rot6d_orig[[0, 2, 4, 1, 3, 5]]
# Result: [1.0, 0.05, -0.1, 0.1, 0.98, 1.0]
#         └────────────┬─────────────┘  └─────────────┬──────────┘
#              First column a1        Second column a2

# Step 2: Gram-Schmidt
a1 = rot6d_reordered[:3]  # [1.0, 0.05, -0.1]
a2 = rot6d_reordered[3:]  # [0.1, 0.98, 1.0]

b1 = a1 / np.linalg.norm(a1)  # Normalize
# b1 ≈ [0.9914, 0.0495, -0.0990]

dot = np.sum(b1 * a2)  # Dot product
b2 = a2 - dot * b1     # Orthogonalize
b2 = b2 / np.linalg.norm(b2)  # Normalize
# b2 ≈ [0.0993, 0.9709, 0.0995]

b3 = np.cross(b1, b2)  # Cross product
# b3 ≈ [0.1872, -0.1089, 0.9764]

# Final 3×3 matrix
R = np.stack([b1, b2, b3], axis=-1)
# Shape: (3, 3)
```

**Verification:** R should be orthonormal (R^T @ R ≈ I)

---

## SMPL-H Joint Ordering

### 22 Joints: Pelvis + 21 Body Joints

```
Index  Joint Name              Type
────────────────────────────────────
  0    Pelvis                 ROOT (global orientation + translation)
  
  LEGS (8 joints)
  1    Left Hip               Hinge
  2    Left Knee              Hinge
  3    Left Ankle             Hinge
  4    Left Foot              End-effector
  5    Right Hip              Hinge
  6    Right Knee             Hinge
  7    Right Ankle            Hinge
  8    Right Foot             End-effector
  
  SPINE (4 joints)
  9    Spine1 (Abdomen)       Ball
 10    Spine2 (Chest)         Ball
 11    Spine3 (Upper Chest)   Ball
 12    Neck                   Ball
 13    Head                   End-effector
  
  ARMS (8 joints)
 14    Left Shoulder          Ball
 15    Left Arm (Upper)       Ball
 16    Left Elbow (Forearm)   Ball
 17    Left Wrist/Hand        End-effector
 18    Right Shoulder         Ball
 19    Right Arm (Upper)      Ball
 20    Right Elbow (Forearm)  Ball
 21    Right Wrist/Hand       End-effector
```

### Motion Data to SMPL-X NPZ Conversion

```
motion_135 (T, 135)
└─ Reshape: (T, 22, 6) rotations
   └─ Process each joint rot6d → rotmat → axis-angle
      └─ Split: Joint 0 (pelvis) vs Joints 1-21 (body)
         └─ SMPL-X NPZ:
            ├─ root_orient: (T, 3)    [Joint 0 axis-angle]
            ├─ pose_body:   (T, 63)   [Joints 1-21 axis-angle, reshaped]
            └─ trans:       (T, 3)    [Translation from motion_135[:, :3]]

SMPL-X NPZ Fields:
┌──────────────────────────────────────────────────────┐
│ pose_body        │ (T, 63)    │ 21×3 body joints   │
│ root_orient      │ (T, 3)     │ Root/pelvis joint  │
│ trans            │ (T, 3)     │ Root translation   │
│ betas            │ (10,)      │ Shape params (0s)  │
│ gender           │ str        │ "neutral"          │
│ mocap_frame_rate │ int        │ FPS (usually 30)   │
└──────────────────────────────────────────────────────┘
```

---

## Translation & Global Orientation

### Translation (First 3 Dimensions)

```
Source: motion_135[:, :3]
Format: 3D Cartesian coordinates [x, y, z]
Space:  World space (absolute position)
Unit:   Meters (typically)

Processing: DIRECT COPY → trans field in SMPL-X NPZ
No transformation or scaling applied
```

### Global Body Orientation (Rot6d for Pelvis/Joint 0)

```
Source: motion_135[:, 3:9]
Format: 6D continuous rotation (row-major)
Meaning: Absolute orientation of pelvis in world space
NOT a local rotation relative to a parent joint

Processing:
1. Reorder [0,2,4,1,3,5] (row-major → column-major)
2. Apply Gram-Schmidt orthogonalization
3. Extract 3×3 rotation matrix
4. Convert to axis-angle representation
5. Store as root_orient in SMPL-X NPZ

Result: root_orient (T, 3) axis-angle representation
```

### Coordinate System

```
Standard SMPL-H Coordinate System:
┌────────────────────────────┐
│         +Y (up)            │
│           ↑                │
│           │     +Z (forward, initial facing)
│           │    ↗            
│    ───────┼────────  +X (right)
│          /                 │
│         /                  │
│        +                   │
└────────────────────────────┘

- Y-axis: Vertical (up)
- X-axis: Lateral (right when facing +Z)
- Z-axis: Anterior-posterior (forward)
- Origin: Pelvis (root joint)
```

---

## Denormalization Formula

### Standard Z-Score Normalization

```
During training: x_normalized = (x_original - mean) / std

During inference (denormalization):
    x_original = x_normalized * std + mean

Where:
- x_normalized: Output from diffusion model (B, T, 201)
- x_original: Denormalized motion (B, T, 201)
- mean: Pre-computed mean from training data (201,)
- std: Pre-computed standard deviation (201,)
```

### Safety Measures

```python
# Avoid division by zero or near-zero std values
std = np.where(std < 1e-3, 1.0, std)

# This ensures dimensions with very small std get scaling factor of 1.0
# (i.e., no scaling applied, preserve original values)
```

### Bundle Statistics

```
The bundle contains:
- bundle.mean: (201,) tensor
- bundle.std: (201,) tensor

These are loaded from the official HY-Motion-1.0 checkpoint
and must NOT be locally computed.

Location: Loaded via load_bundle_from_checkpoint(cfg, ckpt_path, device)
```

---

## Smoothing Filter (Optional, Local Addition)

### Savitzky-Golay Filter

```python
# Translation (dimensions 0:3)
window_length = 7        # ~0.23s @ 30fps
polyorder = 3            # Cubic polynomial

# Rot6d (dimensions 3:135)
window_length = 5        # ~0.17s @ 30fps
polyorder = 3            # Cubic polynomial

# Applied separately to preserve different characteristics:
# - Translation: Wider window for stable root trajectory
# - Rot6d: Narrower window to preserve pose detail
```

### Why Different Windows?

```
Root Translation (0:3):
- Uses wider window (7 frames)
- Rationale: Root trajectory should be smooth
- Wider smoothing preserves overall motion path
- Reduces jitter in character position

Joint Rotations (3:135):
- Uses narrower window (5 frames)
- Rationale: Pose details must be preserved
- Narrower smoothing removes only frame-to-frame noise
- Keeps fine motor control and gestures

Combined Effect:
- Smooth, stable character trajectories
- Preserved pose details and gestures
- Reduced diffusion model artifacts (jitter)
```

### Enabling/Disabling Smoothing

```bash
# Enable smoothing (default)
python batch_t2m_to_embodied.py --prompts "..." --smooth

# Disable smoothing
python batch_t2m_to_embodied.py --prompts "..." --no-smooth
```

---

## Complete Code Example: motion_135 → SMPL-X

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

def motion135_to_smplx(motion_135_npz_path, output_npz_path):
    """Convert motion_135 NPZ to SMPL-X NPZ"""
    
    # Load motion_135
    data = np.load(motion_135_npz_path)
    motion = data['motion_135']  # (T, 135)
    T = motion.shape[0]
    
    # Extract translation and rot6d
    transl = motion[:, :3]               # (T, 3)
    rot6d = motion[:, 3:].reshape(T, 22, 6)  # (T, 22, 6)
    
    # CRITICAL: Reorder from row-major to column-major
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    
    # Gram-Schmidt orthogonalization
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    rotmat = np.stack([b1, b2, b3], axis=-1)  # (T, 22, 3, 3)
    
    # Convert to axis-angle
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    r = R.from_matrix(rotmat_flat)
    aa = r.as_rotvec().reshape(T, 22, 3)  # (T, 22, 3)
    
    # Split root and body
    root_orient = aa[:, 0, :]          # (T, 3)
    pose_body = aa[:, 1:22, :].reshape(T, -1)  # (T, 63)
    
    # Save SMPL-X NPZ
    np.savez(
        output_npz_path,
        pose_body=pose_body.astype(np.float32),
        root_orient=root_orient.astype(np.float32),
        trans=transl.astype(np.float32),
        betas=np.zeros(10, dtype=np.float32),
        gender="neutral",
        mocap_frame_rate=np.array(30),
    )
    
    print(f"Saved SMPL-X NPZ: {output_npz_path}")
```

---

## Verification Checklist

```
Format Specifications:
  [ ] Motion layout: 201 dims = 3 + 6 + 126 + 66
  [ ] motion_135: first 135 dims extracted correctly
  [ ] Rot6d: 6D continuous rotation format
  [ ] Gram-Schmidt: reorder [0,2,4,1,3,5] applied
  
Joint Ordering:
  [ ] 22 SMPL-H joints
  [ ] Pelvis at index 0 (global orientation)
  [ ] Joints 1-21: local rotations
  [ ] Split: root_orient (T,3) + pose_body (T,63)
  
Data Conversion:
  [ ] rot6d → rotation matrix (3×3)
  [ ] Rotation matrix → axis-angle (3D)
  [ ] Translation: direct copy from motion_135[:, :3]
  [ ] Output format: pose_body, root_orient, trans, betas, gender, fps
  
Coordinate System:
  [ ] Y-up convention
  [ ] Z forward (initial facing)
  [ ] X right (cross product)
  
Denormalization:
  [ ] Formula: x = x_norm × std + mean
  [ ] std safety: clamp std < 1e-3 to 1.0
  [ ] Statistics from official checkpoint
  
Optional Enhancements:
  [ ] Smoothing filter disabled by default
  [ ] Can enable with --smooth flag
  [ ] Parameters: translation (7, 3), rot6d (5, 3)
```

---

## References

- **Official Repo**: https://github.com/Tencent-Hunyuan/HY-Motion-1.0
- **Paper**: arXiv:2512.23464
- **SMPL Documentation**: https://smpl.is.tue.mpg.de/
- **Rot6d Paper**: Zhou et al., "On the Continuity of Rotation Representations in Neural Networks"

---
