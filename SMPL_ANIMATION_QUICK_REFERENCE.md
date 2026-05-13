# SMPL Animation - Quick Reference Guide

## Answers to Your Specific Questions

### 1. How are `poses`, `Rh`, `Th` used to animate frame-by-frame?

```javascript
// From score.html lines 657-673
const bones = state.model_mesh[sp.id];
const mesh = bones[0].parent;

// Set root POSITION (translation)
mesh.position.x = sp.Th[0][0];
mesh.position.y = sp.Th[0][1];
mesh.position.z = sp.Th[0][2];

// Set bone ROTATIONS (axis-angle → quaternion)
for (let i = 0; i < bones.length; i++) {
  const idx = 3 * i;  // Each joint = 3 values
  const axis = new THREE.Vector3(poses[idx], poses[idx+1], poses[idx+2]);
  const angle = axis.length();  // magnitude = angle
  axis.normalize();  // direction = axis
  bones[i].quaternion.setFromAxisAngle(axis, angle);
}
// Three.js GPU LBS automatically deforms mesh vertices
```

### 2. What format are `poses` in?

**Axis-Angle** (3D vectors):
```
poses = [
  rx0, ry0, rz0,    # Joint 0 rotation (axis-angle)
  rx1, ry1, rz1,    # Joint 1 rotation
  ...
  rx51, ry51, rz51  # Joint 51 rotation (for SMPL+H)
]

Total: 52 joints × 3 = 156 dimensions

Each value: sqrt(rx² + ry² + rz²) = rotation angle (radians)
            [rx, ry, rz] / angle = rotation axis (normalized)
```

### 3. What is `Rh`?

**Root Rotation** (same as poses[0:3]):
- First joint rotation in axis-angle format
- Same value stored in both `Rh` and `poses[0:3]`
- Used for bone[0] (pelvis/root bone)

### 4. What is `Th`?

**Root Translation** (absolute world position):
- 3D position of the mesh root in world space
- Set directly: `mesh.position = {x: Th[0], y: Th[1], z: Th[2]}`
- Independent of bones; affects whole mesh

## Data Format at Every Stage

### Stage 1: NPZ File
```python
# motion_135 format (embodied motion)
motion_135[t] = [
  tx, ty, tz,              # translation (3)
  r6d_0, ..., r6d_0,       # joint 0 rot6d (6 values)
  r6d_1, ..., r6d_1,       # joint 1 rot6d (6 values)
  ...
]  # Total: 3 + 22*6 = 135 dims
```

### Stage 2: Server Conversion (motion_utils.py)
```python
# Input: motion_135 (T, 135)

rot6d = motion[:, 3:135].reshape(T, 22, 6)        # Extract rotations
axis_angle = rotation_6d_to_axis_angle(rot6d)     # Convert 6D → 3D
poses_66 = axis_angle.reshape(T, 66)              # 22 joints * 3
poses_156 = zeros((T, 156))
poses_156[:, :66] = poses_66                      # Pad for hands
Rh = poses_156[:, :3]                             # First 3 = root rotation
Th = motion[:, :3]                                # First 3 of original = translation
```

### Stage 3: JSON Response (/api/smpl)
```json
{
  "frames": [
    [
      {
        "Rh": [[rx, ry, rz]],
        "Th": [[tx, ty, tz]],
        "poses": [[p0, p1, ..., p155]],
        "shapes": [[0, 0, ..., 0]],
        "gender": "neutral",
        "smpl_type": "smplh"
      }
    ],
    ...
  ]
}
```

### Stage 4: Frontend Animation
```javascript
// Extract frame data
sp = data.frames[frame_index][0];

// Apply transforms
mesh.position.set(sp.Th[0][0], sp.Th[0][1], sp.Th[0][2]);

// Animate each bone
posesArr = sp.poses[0];  // 156-element array
for (i = 0; i < 52; i++) {
  axis_angle = [posesArr[3*i], posesArr[3*i+1], posesArr[3*i+2]];
  angle = sqrt(axis_angle[0]² + axis_angle[1]² + axis_angle[2]²);
  axis = axis_angle / angle;  // normalize
  bones[i].quaternion.setFromAxisAngle(axis, angle);
}
```

## File Reference Guide

### Server-Side (Python)

| File | Purpose | Key Functions |
|------|---------|---|
| `motion_annot_web/score_m2m/score_m2m_web.py` | Web API | `/api/smpl` endpoint |
| `motion_annot_web/score_m2m/motion_utils.py` | Data conversion | `_smpl_from_motion135()`, `_smpl_from_original_npz()`, `rotation_6d_to_axis_angle()` |

### Client-Side (JavaScript)

| File | Purpose | Key Functions |
|------|---------|---|
| `motion_annot_web/score_m2m/templates/score.html` | Main UI & animation loop | `loadMotion()`, `updateFrame()`, `playLoop()` |
| `motion_annot_web/score_m2m/static/scripts3d/load_smpl.js` | Mesh loading | `load_smpl_with_shapes()` |

### Embodied Export

| File | Purpose |
|------|---------|
| `scripts/embodied/batch_npz_to_smpl_joints.py` | Convert motion_135 NPZ → JSON (joint positions) |

## Animation Pipeline (Step-by-Step)

```
1. User loads NPZ file → sets window.NPZ_PATH
2. Frontend calls loadMotion()
3. loadMotion() → fetch('/api/smpl?path=...')
4. Server reads NPZ, converts motion_135 → axis-angle
5. Server returns JSON with frames[]
6. Frontend calls load_smpl_with_shapes() to create mesh
7. updateFrame() called for frame 0 (sets initial pose)
8. User clicks play → playLoop() starts
9. playLoop() increments frame counter every 33.33ms (for 30fps)
10. updateFrame() called each frame:
    - Set mesh.position = Th
    - For each bone: quaternion = setFromAxisAngle(poses[3*i:3*i+3])
11. Three.js GPU shader applies LBS deformation automatically
12. renderer.render() displays deformed mesh
```

## Key Insights

### 1. Axis-Angle Encoding
```javascript
// Why magnitude = angle?
axis_angle = [0, 0, π/2]  // Rotate 90° around Z
magnitude = sqrt(0² + 0² + (π/2)²) = π/2 ≈ 1.5708 radians
direction = [0, 0, 1]  // unit Z-axis

// Why is this useful?
// - Compact: 3 floats/joint (vs 9 for matrix, 4 for quaternion)
// - Differentiable: used in motion generation
// - Interpolable: smooth motion between frames
```

### 2. GPU Skinning (Automatic)
```
When you update bones[i].quaternion:
  ↓
Three.js recomputes world transforms (cascade)
  ↓
GPU shader applies LBS: v_deformed = Σ(weight[i] * transform[i] * v_template)
  ↓
Mesh vertices automatically deform (NO manual updates!)
```

### 3. Dual Representation of Root
```
Rh = poses[0:3]  // Both refer to same thing
```
- `Rh` is just extracted for clarity
- In actual application, only `poses` is used in the loop

### 4. Playback Timing
```javascript
baseIntervalTime = 1000 / fps  // ms per frame
playLoop() checks: elapsed_time >= baseIntervalTime / playbackSpeed
```
- Decouples browser refresh rate (60 FPS) from motion playback (30 FPS)

## Common Issues & Solutions

### Issue: Mesh not rotating
**Solution:** Check if `poses` format is correct (should be axis-angle, not 6D or quaternion)

### Issue: Motion jittery
**Solution:** Likely playback timing issue. Check `baseIntervalTime` calculation

### Issue: Mesh deformed incorrectly
**Solution:** May be because `Th` is in wrong coordinate system. Verify NPZ source.

### Issue: "poses not found in npz" error
**Solution:** NPZ doesn't have `poses` key. Check if it's motion_135 format (server should auto-convert)

## Testing the Animation

```javascript
// In browser console:
visState.currentFrame = 0;
updateFrame();  // Jump to frame 0
visState.currentFrame = 10;
updateFrame();  // Jump to frame 10
visState.isPlaying = true;  // Start playback
visState.isPlaying = false;  // Stop playback
```

---

## Summary Table

| Concept | Meaning | Example |
|---------|---------|---------|
| **axis-angle** | 3D vector where magnitude=angle, direction=axis | `[0, 0, 1.57]` = 90° around Z |
| **poses** | All joint rotations in axis-angle format | `[rx0, ry0, rz0, rx1, ry1, rz1, ...]` |
| **Rh** | Root rotation (first 3 values of poses) | `[rx0, ry0, rz0]` |
| **Th** | Root translation (world position) | `[tx, ty, tz]` |
| **shapes** | SMPL body shape parameters | `[b0, b1, ..., b15]` (unused in web) |
| **bones** | Three.js bone hierarchy | `bones[0]` = root, `bones[1]` = child, ... |
| **LBS** | GPU shader that deforms mesh | Automatic when quaternions update |
| **SkinnedMesh** | Three.js mesh with skeletal animation | Created by `load_smpl_with_shapes()` |

