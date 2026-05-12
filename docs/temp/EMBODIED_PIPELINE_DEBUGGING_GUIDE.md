# Embodied Pipeline Debugging Guide

## Quick Diagnosis Flowchart

```
SYMPTOM: Foot sliding
├─→ Check: Are feet below Z=0 before correction?
│   └─→ Bug #1 or #2: Ground correction failing
├─→ Check: Do joints match adjusted root_pos Z?
│   └─→ Bug #1: Double-failure (GMR outputs feet-below, then FK corrects)
└─→ Check: Are foot body indices wrong?
    └─→ Bug #2: Using indices [7,13] but G1 has feet at different indices

SYMPTOM: Ground penetration (feet below Z=0)
├─→ Check: Is offset_to_ground disabled in GMR?
│   └─→ Bug #1: Pipeline passes --no-offset-to-ground
└─→ Check: Did FK correction make things worse?
    └─→ Bug #6: Overcorrection from wrong starting position

SYMPTOM: Deformed poses (limbs bent wrong)
├─→ Check: Is rot6d layout correct?
│   └─→ Bug #5: Potential row-major/column-major mismatch
├─→ Check: Are rotations numerically valid?
│   └─→ Bug #10: Non-unit quaternions
└─→ Check: Are joint limits violated?
    └─→ Bug #7: No clamping to mechanical limits

SYMPTOM: Joints at limits (elbows/knees locked)
├─→ No clamping in pipeline
│   └─→ Bug #7: Add joint limit validation
└─→ GMR IK not respecting limits
    └─→ Check GMR configuration

SYMPTOM: Inconsistent frame conversion errors
├─→ Position and rotation don't match
│   └─→ Bug #4: Inconsistent frame transform conventions
└─→ Verify GMR's rot_offset is applied correctly
    └─→ Check GMR's IK config: smplx_to_g1.json
```

---

## Debugging Commands

### 1. Visualize Intermediate Pipeline Stages

```bash
# Run step-by-step with intermediates
python scripts/embodied/pipeline_motion_to_robot.py \
    --input motion_135.npz \
    --output robot_cache.pt \
    --keep-intermediates

# Check output files
ls -lh robot_cache_smplx.npz
ls -lh robot_cache_gmr.pkl
ls -lh robot_cache.pt
```

### 2. Inspect SMPL-X NPZ (after step 1)

```python
import numpy as np

# Load and inspect
data = np.load('robot_cache_smplx.npz', allow_pickle=True)
print("Keys:", data.files)
print("pose_body:", data['pose_body'].shape)
print("root_orient:", data['root_orient'].shape)
print("trans:", data['trans'].shape)

# Check ranges
trans = data['trans']
print(f"Translation X range: [{trans[:, 0].min():.3f}, {trans[:, 0].max():.3f}]")
print(f"Translation Y range: [{trans[:, 1].min():.3f}, {trans[:, 1].max():.3f}]")
print(f"Translation Z range (height): [{trans[:, 2].min():.3f}, {trans[:, 2].max():.3f}]")
```

### 3. Inspect GMR PKL (after step 2)

```python
import pickle

with open('robot_cache_gmr.pkl', 'rb') as f:
    gmr = pickle.load(f)

print("Keys:", gmr.keys())
print("root_pos shape:", gmr['root_pos'].shape)
print("root_rot shape:", gmr['root_rot'].shape)
print("dof_pos shape:", gmr['dof_pos'].shape)
print("fps:", gmr['fps'])

# Check for ground penetration (feet below Z=0)
root_z = gmr['root_pos'][:, 2]
print(f"\nRoot Z (height) range: [{root_z.min():.3f}, {root_z.max():.3f}]")
print(f"  Min root Z: {root_z.min():.4f}m")

# Check quaternion norms (should be ~1.0)
import numpy as np
root_rot = gmr['root_rot']  # xyzw format
norms = np.linalg.norm(root_rot, axis=1)
print(f"\nRoot quaternion norms: min={norms.min():.6f}, max={norms.max():.6f}")
if not np.allclose(norms, 1.0, atol=1e-6):
    print("  WARNING: Non-unit quaternions detected!")

# Check DOF ranges
dof_pos = gmr['dof_pos']
for i in range(dof_pos.shape[1]):
    dof_min = dof_pos[:, i].min()
    dof_max = dof_pos[:, i].max()
    print(f"DOF {i:2d}: [{dof_min:7.3f}, {dof_max:7.3f}]")
```

### 4. Inspect ProtoMotions Cache (final .pt file)

```python
import torch
import numpy as np

cache = torch.load('robot_cache.pt', weights_only=False)

print("Keys:", cache.keys())
for k, v in cache.items():
    if isinstance(v, np.ndarray):
        print(f"{k:15s}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.4f}, {v.max():.4f}]")
    else:
        print(f"{k:15s}: {v}")

# Check body positions for ground penetration
body_pos = cache['body_pos']  # (T, 33, 3)
min_z_per_frame = body_pos[:, :, 2].min(axis=1)  # Min Z for each frame
print(f"\nBody minimum Z per frame: min={min_z_per_frame.min():.4f}, max={min_z_per_frame.max():.4f}")
print(f"  Frames below Z=-0.01: {(min_z_per_frame < -0.01).sum()}")

# Check quaternion validity
body_rot = cache['body_rot']  # (T, 33, 4)
norms = np.linalg.norm(body_rot, axis=2)  # (T, 33)
print(f"\nBody quaternion norms: min={norms.min():.6f}, max={norms.max():.6f}")
if not np.allclose(norms, 1.0, atol=1e-5):
    print("  WARNING: Non-unit quaternions detected!")
```

### 5. Check G1 MJCF Body Indices

```python
import xml.etree.ElementTree as ET

mjcf_path = "ref_repo/ProtoMotions/protomotions/data/assets/mjcf/g1_holo_compat.xml"
tree = ET.parse(mjcf_path)
root = tree.getroot()

bodies = root.findall(".//body")
for i, body in enumerate(bodies):
    name = body.get("name", f"unnamed_{i}")
    if "foot" in name.lower() or "ankle" in name.lower():
        print(f"Body index {i}: {name}")

print("\nAll body indices:")
for i, body in enumerate(bodies):
    name = body.get("name", f"unnamed_{i}")
    print(f"{i:2d}: {name}")
```

### 6. Verify rot6d Conversion (for Bug #5)

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# Create a known rotation: 45° around Z axis
angle = np.pi / 4
known_rot = R.from_rotvec([0, 0, angle])
known_mat = known_rot.as_matrix()  # Shape (3, 3)

print("Known rotation matrix:")
print(known_mat)

# Assume row-major layout: [R00, R01, R10, R11, R20, R21]
rot6d_row_major = np.array([
    known_mat[0, 0], known_mat[0, 1],
    known_mat[1, 0], known_mat[1, 1],
    known_mat[2, 0], known_mat[2, 1],
])

print("\nrot6d (row-major):", rot6d_row_major)

# Test the reorder [0,2,4,1,3,5]
rot6d_reordered = rot6d_row_major[[0, 2, 4, 1, 3, 5]]
print("After reorder [0,2,4,1,3,5]:", rot6d_reordered)
print("Expected (column-major):", np.array([
    known_mat[0, 0], known_mat[1, 0], known_mat[2, 0],
    known_mat[0, 1], known_mat[1, 1], known_mat[2, 1],
]))

# Reconstruct via Gram-Schmidt
a1 = rot6d_reordered[:3]
a2 = rot6d_reordered[3:6]
b1 = a1 / np.linalg.norm(a1)
b2 = a2 - np.dot(b1, a2) * b1
b2 = b2 / np.linalg.norm(b2)
b3 = np.cross(b1, b2)

reconstructed = np.stack([b1, b2, b3], axis=-1)
print("\nReconstructed rotation matrix:")
print(reconstructed)

# Compare with original
print("\nDifference from original:")
print(np.abs(reconstructed - known_mat).max())
```

### 7. Test Frame Conversions (for Bug #4)

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

# GMR's rot_offset: [0.5, -0.5, -0.5, -0.5] (wxyz)
rot_offset_wxyz = np.array([0.5, -0.5, -0.5, -0.5])
rot_offset_xyzw = rot_offset_wxyz[[1, 2, 3, 0]]  # Convert to xyzw

print("rot_offset (wxyz):", rot_offset_wxyz)
print("rot_offset (xyzw):", rot_offset_xyzw)

# Test with identity quaternion (standing)
q_identity = np.array([0, 0, 0, 1])  # xyzw

rot_offset = R.from_quat(rot_offset_xyzw)
q_rot = R.from_quat(q_identity)

# Right multiply: q_corrected = q * rot_offset.inv()
q_corrected = q_rot * rot_offset.inv()
print("\nIdentity quaternion after right-multiply by rot_offset.inv():")
print(q_corrected.as_quat())

# Position conversion: apply rot_offset.inv() as active rotation
pos = np.array([1, 0, 0])  # X direction in SMPL-X
pos_converted = rot_offset.inv().apply(pos)
print("\n[1, 0, 0] in SMPL-X converted to MuJoCo:")
print(pos_converted)
print("Expected from rotation: Z should have the X value")

# Check consistency: if we rotate a direction by R, 
# and rotate a frame by R, they should be consistent
direction = np.array([1, 0, 0])
direction_rotated = rot_offset.inv().apply(direction)
print("\n[1, 0, 0] rotated by inverse offset:", direction_rotated)
print("Should align with what frame conversion produces")
```

---

## MuJoCo Body Index Reference

### Finding Body Indices from MJCF

```python
import mujoco

# Load model
model = mujoco.MjModel.from_xml_path("g1_holo_compat.xml")

print(f"Total bodies: {model.nbody}")
print("\nBody names and indices:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"{i:2d}: {name}")

# Find foot bodies
print("\nFoot bodies:")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and ("foot" in name.lower() or "ankle" in name.lower()):
        print(f"{i:2d}: {name}")
```

---

## Joint Limit Extraction

```python
import mujoco

model = mujoco.MjModel.from_xml_path("g1_holo_compat.xml")

print("Joint limits (first 7 = base + 6 DOF, rest = actuators):")
for i in range(min(36, model.nq)):  # G1 has 36 DOF total
    jnt_type = model.jnt_type[i]
    limited = model.jnt_limited[i]
    
    if limited:
        range_min = model.jnt_range[i, 0]
        range_max = model.jnt_range[i, 1]
        print(f"DOF {i:2d}: type={jnt_type}, range=[{range_min:7.3f}, {range_max:7.3f}]")
    else:
        print(f"DOF {i:2d}: type={jnt_type}, unlimited")
```

---

## Systematic Debugging Checklist

- [ ] **Step 1: Verify input motion_135**
  - Check NPZ keys and shapes
  - Verify translation ranges (height should be 0-2m)
  - Plot motion to ensure it looks reasonable

- [ ] **Step 2: Verify SMPL-X conversion**
  - Check rotation matrices are valid (det=1, orthogonal)
  - Check axis-angle values are reasonable (<2π)
  - Check translation is unchanged

- [ ] **Step 3: Check GMR output before FK correction**
  - Are feet below ground (Z < 0)?
  - Are quaternions normalized?
  - Are joint angles within valid ranges?

- [ ] **Step 4: Verify MuJoCo FK**
  - Load MJCF and extract body indices
  - Verify foot_body_indices [7, 13] are correct
  - Run FK on known poses and check outputs

- [ ] **Step 5: Check FK ground correction**
  - Is correction actually being applied?
  - Are feet at ground after correction?
  - Did root Z change as expected?

- [ ] **Step 6: Validate final cache**
  - Check for ground penetration
  - Verify quaternions are unit
  - Check velocities look reasonable

---

## Common Error Messages & Solutions

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `KeyError: 'motion_135'` | Wrong NPZ key or corrupted input | Check NPZ with Bug #8 guidance |
| `Index out of bounds: [7, 13]` | Wrong foot body indices | Use MuJoCo script to find correct indices |
| `RuntimeError: data.xpos[bi + 1] out of range` | +1 offset too large | Check if world body is at 0 or implicit |
| `Quaternion not normalized` | Numerical error in frame conversion | Add normalization after conversions |
| `Feet penetrate ground` | Ground correction not applied | Check --fk-ground-correction flag |
| `Robot too tall` | FK correction overcorrecting | Check Bug #1 and #6 |
| `Deformed limbs` | Rot6d layout wrong | Verify Bug #5 with known poses |

---

## Performance Impact of Fixes

| Bug | Fix Complexity | Time Cost | Quality Impact |
|-----|----------------|-----------|-----------------|
| #1 | High (logic restructure) | Minimal | High (foot sliding gone) |
| #2 | Low (find correct indices) | Minimal | High (ground contact fixed) |
| #3 | Low (verify offset) | Minimal | High (FK accuracy improved) |
| #4 | Medium (verify math) | Minimal | High (pose alignment) |
| #5 | Medium (add test) | Minimal | High (limb orientation) |
| #6 | Medium (add iteration) | +10-20ms per motion | Medium (better stability) |
| #7 | Low (add clamping) | Minimal | Medium (prevent violations) |
| #8 | Low (add validation) | Minimal | Low (error catching) |
| #9 | Low (fix edge case) | Minimal | Low (smoothness) |
| #10 | Low (add normalize) | Minimal | Low (numerical stability) |

