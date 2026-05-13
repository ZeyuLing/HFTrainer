# SMPL Animation - Visual & Detailed Guide

## Visual: Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NPZ FILE (motion_135)                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ [tx, ty, tz] + [rot6d_j0 (6)] + [rot6d_j1 (6)] + ... (22 joints)  │  │
│  │       ↑                                                             │  │
│  │   Translation                   6D rotation vectors (compact)       │  │
│  │                                                                      │  │
│  │  Total: 3 + 22×6 = 135 dimensions                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │   /api/smpl Endpoint          │
                    │   (score_m2m_web.py)          │
                    └───────────────────────────────┘
                                    ↓
        ┌───────────────────────────────────────────────────┐
        │  motion_utils.py Conversion                       │
        ├───────────────────────────────────────────────────┤
        │ 1. Split: transl (3) + rot6d (22×6)              │
        │ 2. Convert: rot6d → axis-angle (3D vectors)      │
        │ 3. Extract: Rh = poses[0:3]                       │
        │ 4. Pad:    poses_66 → poses_156 (hand padding)   │
        └───────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      JSON Response (frames[])                            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ {                                                                   │  │
│  │   "frames": [                                                       │  │
│  │     [{"Rh": [[rx0,ry0,rz0]],                                       │  │
│  │       "Th": [[tx,ty,tz]],           ← Root transforms             │  │
│  │       "poses": [[p0,...,p155]],     ← All joint rotations         │  │
│  │       "gender": "neutral"}]                                        │  │
│  │   ],                                                                │  │
│  │   "fps": 30                                                         │  │
│  │ }                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │  Browser (score.html)         │
                    └───────────────────────────────┘
                                    ↓
                ┌─────────────────────────────────┐
                │ load_smpl_with_shapes()         │
                ├─────────────────────────────────┤
                │ Create mesh + bone hierarchy    │
                │ Setup GPU skinning (LBS)        │
                │ Return: {bones, mesh}           │
                └─────────────────────────────────┘
                                    ↓
                ┌─────────────────────────────────┐
                │ updateFrame() [Called per frame]│
                ├─────────────────────────────────┤
                │ 1. mesh.position = Th           │
                │ 2. For each bone[i]:            │
                │    axis = poses[3*i:3*i+3]      │
                │    angle = |axis|               │
                │    quaternion.setFromAxisAngle()│
                └─────────────────────────────────┘
                                    ↓
                ┌─────────────────────────────────┐
                │  renderer.render()              │
                ├─────────────────────────────────┤
                │ GPU LBS Shader:                 │
                │ v_deformed = Σ(weight[i] ×     │
                │              transform[i] ×     │
                │              v_template)        │
                └─────────────────────────────────┘
                                    ↓
                          ┌─────────────────┐
                          │ 3D Animation!   │
                          └─────────────────┘
```

---

## Visual: Axis-Angle Representation

### What is Axis-Angle?

```
3D Vector Encoding:
┌──────────────────────────────────────────────┐
│ axis_angle = [rx, ry, rz]                   │
├──────────────────────────────────────────────┤
│ magnitude = √(rx² + ry² + rz²) = rotation   │
│                                    angle     │
│                                  (radians)   │
│                                              │
│ direction = [rx, ry, rz] / magnitude =      │
│            rotation axis (normalized)        │
└──────────────────────────────────────────────┘

Examples:

1) Rotate 90° around Z-axis:
   axis_angle = [0, 0, π/2]
   magnitude = 1.5708 rad = 90°
   direction = [0, 0, 1]  (Z-axis)

2) Rotate 45° around X-axis:
   axis_angle = [π/4, 0, 0]
   magnitude = 0.7854 rad = 45°
   direction = [1, 0, 0]  (X-axis)

3) Rotate 0° (identity):
   axis_angle = [0, 0, 0]
   magnitude = 0
   direction = undefined (identity)
```

### Why Axis-Angle?

```
Format Comparison:

              Dims   Compact   Diff   Intuitive
─────────────────────────────────────────────
Rotation      3      ✓ Yes    ✓ Yes   ✓ Yes
Matrix
            
Matrix       9      ✗ No      ✓ Yes   ✗ No
representation

Quaternion    4      ◐ ~      ✓ Yes   ◐ ~
             
Euler Angles  3      ✓ Yes    ✗ No    ✗ Gimbal
             
Axis-Angle    3      ✓ Yes    ✓ Yes   ✓ Yes  ← SMPL Choice
```

---

## Visual: Skeletal Hierarchy (SMPL+H)

```
                           ┌──────────────┐
                           │  bones[0]    │
                           │ (Pelvis/Root)│
                           └──────┬───────┘
                                  │
                  ┌───────────────┼───────────────┐
                  │               │               │
        ┌─────────▼────────┐ ┌───▼──────┐ ┌─────▼────────┐
        │   bones[1]       │ │bones[3]  │ │ bones[4]     │
        │  (L_Hip)         │ │(Spine1)  │ │ (R_Hip)      │
        └────┬─────────────┘ │          │ └──┬───────────┘
             │               └───┬──────┘    │
             │                   │           │
        ┌────▼─────┐         ┌───▼──────┐   ┌──▼────────┐
        │bones[4]  │         │bones[6]  │   │ bones[5]  │
        │(L_Knee)  │         │(Spine2)  │   │ (R_Knee)  │
        └────┬─────┘         └───┬──────┘   └──┬────────┘
             │                   │             │
             │                   │             │
        ┌────▼─────┐         ┌───▼──────┐   ┌──▼────────┐
        │bones[7]  │         │bones[9]  │   │ bones[8]  │
        │(L_Ankle) │         │(Spine3)  │   │ (R_Ankle) │
        └────┬─────┘         └───┬──────┘   └──┬────────┘
             │                   │             │
             │                   │             │
        ┌────▼──────┐        ┌───┼──────────────┼──────────┐
        │bones[10]  │        │   │              │          │
        │(L_Foot)   │        │   │              │          │
        └───────────┘    ┌───▼───┐  ┌────────┐  ┌─▼─────────┐
                         │bones  │  │ bones  │  │ bones     │
                         │[12]   │  │ [13]   │  │ [14]      │
                         │(Neck) │  │(L_Collar)│ │(R_Collar)│
                         └───┬───┘  └────┬───┘  └────┬──────┘
                             │           │          │
                        ┌────▼────┐  ┌──▼─────┐  ┌──▼──────┐
                        │bones[15]│  │bones   │  │ bones   │
                        │(Head)   │  │[16]    │  │ [17]    │
                        └─────────┘  │(L_Shldr)  │(R_Shldr)│
                                     └────┬──┘   └────┬────┘
                                          │          │
                                          └──────┬───┘
                                                 │
                        (Continues to arms & wrists)

Key Points:
- Each bone has ONE parent (except root)
- Each bone can have MULTIPLE children
- Transforms cascade DOWN the tree
- Total: 52 bones for SMPL+H
```

---

## Visual: Per-Frame Animation

### What Happens Each Frame

```
Frame n at time t:
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  1. Fetch data from frames[n]                           │
│     sp = frames[n][0]  // SMPL data for frame n         │
│                                                          │
│  2. Set Root Transform                                  │
│     ┌──────────────────────────────────┐               │
│     │ mesh.position = {                │               │
│     │   x: sp.Th[0][0],  // Translation│               │
│     │   y: sp.Th[0][1],                │               │
│     │   z: sp.Th[0][2]                 │               │
│     │ }                                 │               │
│     │                                   │               │
│     │ NOTE: Th is independent of bones!│               │
│     │ It moves the entire mesh.         │               │
│     └──────────────────────────────────┘               │
│                                                          │
│  3. Animate Bones (Rotations)                           │
│     ┌──────────────────────────────────┐               │
│     │ for i = 0 to 51:                 │               │
│     │   idx = 3 * i                     │               │
│     │   axis_angle = [                 │               │
│     │     sp.poses[0][idx],     // rx  │               │
│     │     sp.poses[0][idx+1],   // ry  │               │
│     │     sp.poses[0][idx+2]    // rz  │               │
│     │   ]                               │               │
│     │   angle = ||axis_angle||          │               │
│     │   axis.normalize()                │               │
│     │                                   │               │
│     │   bones[i].quaternion =           │               │
│     │     setFromAxisAngle(axis, angle) │               │
│     └──────────────────────────────────┘               │
│                                                          │
│  4. GPU Skinning (AUTOMATIC)                            │
│     ┌──────────────────────────────────┐               │
│     │ For each vertex v:                │               │
│     │   v_deformed = 0                  │               │
│     │   for each bone b influencing v: │               │
│     │     w = skinWeight[v][b]         │               │
│     │     T = boneTransform[b]          │               │
│     │     v_deformed +=                 │               │
│     │       w * T * v_template[v]      │               │
│     │                                   │               │
│     │ (Executed by GPU shader!)        │               │
│     └──────────────────────────────────┘               │
│                                                          │
│  5. Render Frame                                        │
│     renderer.render(scene, camera)                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Visual: How LBS (Linear Blend Skinning) Works

```
Before Skinning (T-Pose):
┌─────────────────────┐
│ ○ shoulder          │
│ │                   │
│ ├─ ○ elbow          │  Each vertex assigned to
│ │  │                │  nearby bones with weights
│ │  └─ ○ wrist       │
└─────────────────────┘

After Updating bones[i].quaternion:

Bone Hierarchy Updated:
bones[0].quaternion = ...
bones[1].quaternion = ...  ← Pelvis rotates
└─ bones[3].quaternion = ...
   └─ bones[16].quaternion = ... ← Shoulder rotates
      └─ bones[18].quaternion = ... ← Elbow rotates
         └─ bones[20].quaternion = ... ← Wrist rotates

GPU LBS Deformation:
Each vertex blends transforms from nearby bones:

v_orig = [shoulder vertex position in T-pose]
weights = {
  bone_0 (pelvis): 0.05,    // Far influence
  bone_16 (shoulder): 0.90, // Primary
  bone_18 (elbow): 0.05     // Secondary
}

v_deformed = 
  0.05 * pelvis_transform * v_orig +
  0.90 * shoulder_transform * v_orig +
  0.05 * elbow_transform * v_orig

Result: Smooth mesh deformation following skeleton!
```

---

## Visual: Data Types at Each Stage

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Python NumPy Arrays (motion_utils.py)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ rot6d: np.ndarray (T, 22, 6)    ← 6D rotation (compact)    │
│   Type: float32                                              │
│   Layout: [r00 r01 r10 r11 r20 r21] per joint              │
│   Size: T * 22 * 6 = T * 132 values                         │
│                                                              │
│ axis_angle: np.ndarray (T, 22, 3) ← 3D rotation (expanded)  │
│   Type: float32                                              │
│   Layout: [rx ry rz] per joint                              │
│   Size: T * 22 * 3 = T * 66 values                          │
│                                                              │
│ poses_156: np.ndarray (T, 156)                              │
│   Type: float32 (or list after tolist())                    │
│   Layout: [r00 r01 r02 r10 r11 r12 ... r51,2]             │
│   Size: T * 156 values (padded for SMPL+H)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Stage 2: JSON Serialization (JSON response)                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ frames: [                                                    │
│   [                  ← List of persons (usually 1)          │
│     {                                                        │
│       "Rh": [[rx, ry, rz]],             ← Root rotation    │
│       "Th": [[tx, ty, tz]],             ← Root translation │
│       "poses": [[p0, p1, ..., p155]],   ← All rotations   │
│       "shapes": [[b0, ..., b15]],       ← Shape params    │
│       ...                                                    │
│     }                                                        │
│   ]                                                          │
│ ]                                                            │
│                                                              │
│ Note: All numbers are JSON floats (64-bit doubles)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Stage 3: JavaScript Typed Arrays (score.html)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ posesArr: Float64Array (from JSON)                          │
│   Length: 156                                                │
│   posesArr[i] = floating-point rotation component           │
│                                                              │
│ For frame update:                                            │
│   sp.poses[0]  ← Single frame data as array/list            │
│   sp.Th[0]     ← Translation as [x, y, z]                   │
│   sp.Rh[0]     ← Root rotation as [rx, ry, rz]             │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Three.js Objects (JavaScript runtime)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ axis = THREE.Vector3(rx, ry, rz)       ← 3D vector        │
│   .length() → rotation angle (radians)                      │
│   .normalize() → rotation axis (unit vector)                │
│                                                              │
│ bones[i].quaternion                                          │
│   Type: THREE.Quaternion {x, y, z, w}                      │
│   Set via: quaternion.setFromAxisAngle(axis, angle)         │
│                                                              │
│ mesh.position                           ← THREE.Vector3    │
│ mesh.bind(skeleton)                     ← SkinnedMesh      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Visual: Conversion Formulas

### Rot6D → Axis-Angle

```
Input: 6D rotation (row-major layout)
  rot6d = [R00, R01, R10, R11, R20, R21]

Step 1: Reorder columns
  col1 = [rot6d[0], rot6d[2], rot6d[4]]  // [R00, R10, R20]
  col2 = [rot6d[1], rot6d[3], rot6d[5]]  // [R01, R11, R21]

Step 2: Gram-Schmidt orthogonalization
  b1 = col1 / ||col1||              // First column (normalized)
  b2 = col2 - (col2·b1)*b1          // Orthogonalize
  b2 = b2 / ||b2||                  // Normalize
  b3 = b1 × b2                      // Cross product (third column)

Step 3: Form rotation matrix
  R = [b1 | b2 | b3]  (3×3 matrix)

Step 4: Matrix → Axis-Angle
  θ = arccos((tr(R) - 1) / 2)       // Rotation angle
  axis = [R[2,1] - R[1,2],          // Axis direction
          R[0,2] - R[2,0],
          R[1,0] - R[0,1]] / (2*sin(θ))

Output: axis_angle = axis * θ  (3D vector)
```

### Axis-Angle → Quaternion

```
Input: axis_angle = [rx, ry, rz]

Step 1: Extract magnitude and direction
  θ = √(rx² + ry² + rz²)        // Rotation angle (radians)
  axis = [rx, ry, rz] / θ       // Rotation axis (normalized)

Step 2: Axis-angle to quaternion
  qx = axis.x * sin(θ/2)
  qy = axis.y * sin(θ/2)
  qz = axis.z * sin(θ/2)
  qw = cos(θ/2)

Output: q = (qx, qy, qz, qw)  (quaternion)

Special case: If θ ≈ 0
  q = (0, 0, 0, 1)  (identity quaternion)
```

---

## Summary Diagram: Information Flow

```
                    NPZ File
                      ↓
        ┌─────────────────────────────┐
        │  motion_135 (135 dims/frame)│
        │  ┌───┬───────────────────┐  │
        │  │3  │132 (22*6 rot6d)   │  │
        │  │Th │Rh + 21 joints     │  │
        │  └───┴───────────────────┘  │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │ motion_utils.py conversion  │
        │ rot6d[22,6] → aa[22,3]     │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │ JSON frames (156 dims/frame)│
        │ ┌────────────────────────┐  │
        │ │ Th[3] + poses[156]     │  │
        │ │ = Th + Rh + 51 joints  │  │
        │ └────────────────────────┘  │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  Browser score.html         │
        │  loadMotion() / updateFrame()│
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  load_smpl_with_shapes()    │
        │  SkinnedMesh + Skeleton     │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │ updateFrame() per frame:    │
        │ - Set mesh.position = Th    │
        │ - bones[i].q = axisAngle    │
        └─────────────────────────────┘
                      ↓
        ┌─────────────────────────────┐
        │  GPU LBS Shader             │
        │  Auto-deform vertices       │
        └─────────────────────────────┘
                      ↓
               Animated Mesh!
```

---

## Testing Checklist

- [ ] Verify NPZ has `motion_135` key (135 dims per frame)
- [ ] Check server logs: rotation conversion succeeds
- [ ] Verify JSON response has `frames` with `poses` (156 dims)
- [ ] Check bone count matches SMPL+H (52 bones)
- [ ] Test frame 0: initial pose displays correctly
- [ ] Test frame n: mesh rotates smoothly
- [ ] Verify FPS setting matches motion file
- [ ] Check camera follows root position (if enabled)
- [ ] Verify playback speed adjusts (1.0x, 0.5x, 2.0x)

