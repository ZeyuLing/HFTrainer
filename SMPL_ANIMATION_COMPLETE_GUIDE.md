# SMPL Mesh Animation Complete Data Flow Analysis

## Executive Summary

This document traces the complete data flow from NPZ files through the web server to Three.js rendering, explaining how SMPL skeletal animation works in the codebase.

### Key Finding: Axis-Angle Representation
The **`poses`** parameter uses **axis-angle format (3D vectors)**, where:
- **Magnitude** = rotation angle (in radians)
- **Direction** = rotation axis (normalized unit vector)
- Three values per joint: `[axis_x, axis_y, axis_z]`

---

## 1. Data Flow Overview

```
NPZ File (motion_135 or original format)
    ↓
/api/smpl endpoint (score_m2m_web.py)
    ↓
motion_utils.py conversion
    ↓
JSON response with frames
    ↓
Frontend score.html
    ↓
load_smpl_with_shapes() → Three.js SkinnedMesh
    ↓
updateFrame() → bone rotation animation
    ↓
renderer.render()
```

---

## 2. Server-Side: NPZ to JSON Conversion

### File: `motion_annot_web/score_m2m/motion_utils.py`

#### Input Format 1: motion_135 (Embodied/Generated Motion)
**Source:** `scripts/embodied/batch_npz_to_smpl_joints.py`

```python
motion_135: (T, 135)
├─ [0:3]    → translation (T, 3)      # Absolute world position Th
└─ [3:135]  → rot6d for 22 joints     # (T, 22*6) raw 6D rotations

# Internally in motion_utils.py:
rot6d = motion[:, 3:135].reshape(T, 22, 6)  # (T, 22, 6)
axis_angle = rotation_6d_to_axis_angle(rot6d)  # (T, 22, 3)
poses_66 = axis_angle.reshape(T, 66)  # 22 * 3 = 66 dims for body
poses_flat = zeros((T, 156))
poses_flat[:, :66] = poses_66  # pad with zeros for hand joints (SMPL+H)
Rh = poses_flat[:, :3]  # root rotation (same as joint 0)
```

**Rotation Space Handling:**
- `rotation_space="local"` (default): Direct rot6d → axis-angle
- `rotation_space="global"`: Convert global rot6d → local rot6d first, then axis-angle
- Uses `global_to_local_rot6d()` function in motion_utils.py

#### Input Format 2: Original NPZ (poses/trans/betas)
**Source:** Manual SMPL poses (e.g., from mocap)

```python
poses: (T, N*3) where N ∈ {22, 52, 55} for SMPL/SMPL+H/SMPL-X
       Values are already axis-angle (not 6D)
trans: (T, 3)   → translation Th
betas: (1, 16)  → shape parameters (face/body)

# Server truncates to SMPL-22 (66 dims) if needed:
if poses.shape[1] > 66:
    poses = poses[:, :66]
```

#### Output JSON Format
**Returned by `/api/smpl` endpoint:**

```json
{
  "type": "frames",
  "fps": 30.0,
  "num_frames": T,
  "gender": "neutral",
  "smpl_type": "smplh",
  "frames": [
    [
      {
        "id": 0,
        "gender": "neutral",
        "smpl_type": "smplh",
        "Rh": [[rx, ry, rz]],              // Root rotation (1x3)
        "Th": [[tx, ty, tz]],              // Root translation (1x3)
        "poses": [[p0,p1,p2,...,p155]],    // All joint rotations (1x156)
        "shapes": [[b0,b1,...,b15]],       // Shape parameters (1x16)
        "mocap_framerate": 30.0
      }
    ],
    [
      { ... frame 2 ... }
    ],
    ...
  ]
}
```

**Key Fields:**
- **`Rh`**: Root joint rotation (axis-angle, 3D) - **is part of poses[0:3]**
- **`Th`**: Root joint translation (3D world position)
- **`poses`**: All 52 joint rotations for SMPL+H (52 * 3 = 156 dims)
  - dims [0:3] = root rotation (same as Rh)
  - dims [3:156] = 51 child joints
- **`shapes`**: SMPL shape blendshape coefficients (unused in web, set to zeros)

---

## 3. Frontend: JavaScript Loading & Animation

### File: `motion_annot_web/score_m2m/static/scripts3d/load_smpl.js`

#### Function Signature
```javascript
async function load_smpl_with_shapes(params, gender_param) {
  // params = {
  //   shapes: [0, 0, ..., 0],     // 16-element array
  //   gender: 'neutral',           // 'neutral' | 'male' | 'female'
  //   poses: null,                 // NOT used here (data comes at runtime)
  //   Rh: null,                    // NOT used here
  //   Th: null,                    // NOT used here
  //   smpl_type: 'smplh',          // 'smpl' | 'smplh' | 'smplx'
  //   framerate: 30                // NOT used for playback
  // }
  
  return { bones, skeleton, mesh };  // bones[i] is Three.js Bone
}
```

**Important:** `load_smpl_with_shapes()` only creates the **static mesh structure and bone hierarchy**. It does NOT animate.

#### Bone Hierarchy Construction
```javascript
// Build skeleton from keypoints (joint template positions)
var rootBone = new THREE.Bone();
rootBone.position.set(keypoints[0], keypoints[1], keypoints[2]);
var bones = [rootBone];

for (let i = 1; i < keypoints.length / 3; i++) {
  const bone = new THREE.Bone();
  const parentIndex = edges[i];  // parent bone index
  bone.position.set(
    keypoints[3*i] - keypoints[3*parentIndex],     // local X
    keypoints[3*i+1] - keypoints[3*parentIndex+1],  // local Y
    keypoints[3*i+2] - keypoints[3*parentIndex+2]   // local Z
  );
  bones.push(bone);
  bones[parentIndex].add(bone);  // parent → child hierarchy
}

var skeleton = new THREE.Skeleton(bones);
var mesh = new THREE.SkinnedMesh(geometry, material);
mesh.add(bones[0]);
mesh.bind(skeleton);  // Bind geometry to skeleton for GPU skinning
```

**Skeleton Topology:**
- SMPL: 24 joints
- SMPL+H: 52 joints (body + hands)
- SMPL-X: 55 joints (body + hands + face)

---

## 4. Frontend: Per-Frame Animation Loop

### File: `motion_annot_web/score_m2m/templates/score.html`

#### Fetching Data
```javascript
// Line 720-721
const url = '/api/smpl?path=' + window.NPZ_PATH + '&rotation_space=' + encodeURIComponent(window.ROTATION_SPACE || 'local');
fetch(url).then(r => r.json()).then(data => {
  state.infos = data.frames;        // Extract frames list
  state.total_frame = frames.length;
});
```

#### Creating Mesh (Lines 807-850)
```javascript
frames[0].forEach(d => {
  load_smpl_with_shapes({
    shapes: new Array(16).fill(0),
    gender: d.gender || 'neutral',
    smpl_type: d.smpl_type || 'smpl'
  }).then(result => {
    state.scene.add(result.mesh);
    state.model_mesh[d.id] = result.bones;  // Save bones for animation
  });
});
```

#### Animation Loop: updateFrame() (Lines 635-692)
**Called every frame, sets bone rotations:**

```javascript
function updateFrame() {
  const state = visState;
  const f = state.currentFrame;
  const info = state.infos[f];  // Get data for frame f
  
  if (info && Array.isArray(info)) {
    info.forEach(sp => {
      const bones = state.model_mesh[sp.id];
      const mesh = bones[0].parent;  // Root bone's parent
      
      // 1. SET ROOT TRANSLATION
      mesh.position.x = sp.Th[0][0];
      mesh.position.y = sp.Th[0][1];
      mesh.position.z = sp.Th[0][2];
      
      // 2. ANIMATE BONE ROTATIONS (axis-angle → quaternion)
      const posesArr = sp.poses[0];  // 156-element array (or 69 for SMPL)
      let poses_offset = posesArr.length === 69 ? -3 : 0;
      const maxBone = Math.min(bones.length, Math.floor((posesArr.length - poses_offset) / 3));
      
      for (let i = 0; i < maxBone; i++) {
        const idx = poses_offset + 3 * i;
        if (idx < 0 || idx + 2 >= posesArr.length) continue;
        
        // Extract 3D axis-angle vector
        const axis = new THREE.Vector3(
          posesArr[idx],
          posesArr[idx + 1],
          posesArr[idx + 2]
        );
        
        // angle = magnitude of axis vector
        const angle = axis.length();
        
        if (angle > 1e-8) {
          // Normalize axis to unit vector
          axis.normalize();
          // Convert axis-angle to quaternion and apply
          bones[i].quaternion.setFromAxisAngle(axis, angle);
        } else {
          // Zero rotation
          bones[i].quaternion.set(0, 0, 0, 1);  // identity quaternion
        }
      }
    });
  }
  
  // Update camera follow (if enabled)
  if (state.cameraFollow && state.cameraFollowOffset && state.infos) {
    const t = state.infos[f][0].Th[0];
    const center = new THREE.Vector3(t[0], t[1], t[2]);
    state.controls.target.copy(center);
    state.camera.position.copy(center.clone().add(state.cameraFollowOffset));
  }
}
```

#### Playback Loop (Lines 694-715)
```javascript
function playLoop(now) {
  const state = visState;
  if (state.isPlaying && now - state.lastFrameTime >= (state.baseIntervalTime / state.playbackSpeed)) {
    state.currentFrame = (state.currentFrame + 1) % state.total_frame;
    state.lastFrameTime = now;
    updateFrame();  // Update bone rotations for new frame
  }
  if (state.isPlaying) {
    state.animationId = requestAnimationFrame(playLoop);
  }
}
```

---

## 5. Axis-Angle to Quaternion Conversion

### The Core Animation Mechanism

**Axis-Angle Representation:**
```
axis_angle_3d = [rx, ry, rz]

magnitude = sqrt(rx² + ry² + rz²)  // This is the rotation angle in radians
direction = [rx, ry, rz] / magnitude  // Unit vector (rotation axis)

// Example: rotate 90° around Z-axis
axis_angle = [0, 0, π/2]
magnitude = π/2 ≈ 1.5708 radians
direction = [0, 0, 1]
```

**Three.js Conversion:**
```javascript
const axis = new THREE.Vector3(rx, ry, rz);
const angle = axis.length();           // Extract magnitude
axis.normalize();                       // Make it unit vector
bones[i].quaternion.setFromAxisAngle(axis, angle);
```

**Why This Format?**
- Compact: 3 floats per joint (vs 9 for matrices or 4 for quaternions)
- Efficient: Direct from SMPL models (standard export format)
- Interpretable: Magnitude is angle, direction is axis
- Differentiable: Used in motion generation pipelines

---

## 6. GPU Skinning: LBS (Linear Blend Skinning)

### How Vertices Move with Bones

**Setup (load_smpl.js, Lines 254-273):**
```javascript
geometry.setIndex(new THREE.BufferAttribute(faces, 1));
geometry.setAttribute('position', new THREE.BufferAttribute(v_template, 3));
geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, NUM_SKIN_WEIGHTS));
geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, NUM_SKIN_WEIGHTS));

const material = new THREE.MeshStandardMaterial({
  color: gender_color[gender],
  side: THREE.DoubleSide
});

var mesh = new THREE.SkinnedMesh(geometry, material);
mesh.bind(skeleton);  // CRITICAL: connects geometry to skeleton
```

**Runtime (Three.js GPU, automatic):**
When you update `bones[i].quaternion`, Three.js automatically:
1. Recomputes bone world transforms (cascade from root to leaves)
2. GPU shader applies LBS: `v_skinned = Σ weight[i] * bone[i].transform * v_template`
3. SkinnedMesh geometry deforms accordingly

**No manual mesh updates needed!**

---

## 7. Data Format Summary Table

| Aspect | Value |
|--------|-------|
| **`poses` Format** | Axis-angle (3D vectors) |
| **Dimensions per Joint** | 3 (rx, ry, rz) |
| **Total Joints SMPL+H** | 52 |
| **Total Dims in `poses`** | 156 (52 * 3) |
| **`Rh` (root rotation)** | Same as poses[0:3] |
| **`Th` (root translation)** | 3D world position |
| **Quaternion (internal)** | 4D (x, y, z, w) - computed from axis-angle |

---

## 8. Animation Timing

### Playback Configuration
```javascript
state.baseIntervalTime = 1000.0 / rfps;  // milliseconds per frame
state.playbackSpeed = 1.0;               // default, can be adjusted

// Frame advance condition:
if (now - state.lastFrameTime >= (state.baseIntervalTime / state.playbackSpeed)) {
  state.currentFrame++;
  updateFrame();
}
```

### Example: 30 FPS Motion
```
fps = 30
baseIntervalTime = 1000 / 30 ≈ 33.33 ms
playLoop() called every ~16ms (typical 60 FPS browser)
Frame advances when accumulated time ≥ 33.33 ms
```

---

## 9. Complete Data Transformation Pipeline

### Step-by-Step Example (1 Frame)

**Input NPZ (motion_135):**
```
motion_135[frame_t] = [
  tx, ty, tz,                    # translation (3 values)
  r6d_0_0, r6d_0_1, ..., r6d_0_5,  # joint 0 rot6d (6 values)
  r6d_1_0, ..., r6d_1_5,           # joint 1 rot6d
  ... (22 joints total)
]
# Total: 3 + 22*6 = 135 values
```

**Server Processing (motion_utils.py):**
```python
# 1. Extract and reshape
translation = motion_135[0:3]              # (3,)
rot6d_flat = motion_135[3:135]             # (132,)
rot6d = rot6d_flat.reshape(22, 6)          # (22, 6)

# 2. Convert rot6d → axis-angle
axis_angle = rotation_6d_to_axis_angle(rot6d)  # (22, 3)

# 3. Flatten and pad
poses_66 = axis_angle.reshape(66)          # (66,)
poses_156 = zeros(156)
poses_156[0:66] = poses_66                 # (156,) with hand padding

# 4. Extract root rotation
Rh = poses_156[0:3]                        # Same as axis_angle[0]

# 5. JSON response
{
  "Rh": [[Rh_x, Rh_y, Rh_z]],
  "Th": [[tx, ty, tz]],
  "poses": [[p0, p1, ..., p155]],
  ...
}
```

**Frontend (score.html):**
```javascript
// 1. Receive JSON
sp = data.frames[frame_t][0];

// 2. Set root transform
mesh.position.set(sp.Th[0][0], sp.Th[0][1], sp.Th[0][2]);

// 3. Set bone rotations
for (let i = 0; i < 52; i++) {
  const idx = 3 * i;
  const axis = new THREE.Vector3(
    sp.poses[0][idx],
    sp.poses[0][idx+1],
    sp.poses[0][idx+2]
  );
  const angle = axis.length();
  if (angle > 1e-8) {
    axis.normalize();
    bones[i].quaternion.setFromAxisAngle(axis, angle);
  }
}

// 4. GPU skinning (automatic)
renderer.render(scene, camera);  // Three.js updates mesh vertices via LBS
```

---

## 10. Key Implementation Details

### Pose Array Length Handling (Line 661)
```javascript
let poses_offset = posesArr.length === 69 ? -3 : 0;
```
- **69 dims**: Old format (22 joints * 3 - root excluded) → offset -3 to add root back
- **156 dims**: SMPL+H format (52 joints * 3 with root included) → offset 0

### Skeletal Hierarchy Storage
```javascript
bones[0] = root bone
bones[0].add(bones[1]);  // L_Hip child of root
bones[0].add(bones[2]);  // R_Hip child of root
bones[1].add(bones[4]);  // L_Knee child of L_Hip
// ... builds tree structure
```

### Three.js Automatic Updates
- Updating `bones[i].quaternion` automatically cascades transforms
- No need to manually update:
  - Bone world position
  - Skinned mesh vertex positions
  - Normal vectors
- GPU LBS shader handles all vertex deformation

---

## 11. Rotation Space: Local vs Global

### Local Rotations (default, used in web)
- Each joint rotation is **relative to parent joint**
- Used in SMPL model definition
- Can be applied directly to bone quaternions

### Global Rotations (alternative)
- Each joint rotation is **absolute in world space**
- Must convert to local before applying to bones
- Conversion function: `global_to_local_rot6d()` in motion_utils.py

**Conversion Example:**
```python
# Global rot6d (world space)
global_rot6d[joint_i]

# Get parent's global rotation
parent_rot = global_rot6d[parent_i]
parent_rot_inv = inverse(parent_rot)

# Convert to local
local_rot = parent_rot_inv @ global_rot6d[joint_i]
```

---

## 12. Embodied Motion Pipeline

### Source: `scripts/embodied/batch_npz_to_smpl_joints.py`

**NPZ Format:**
```python
data = {
  'motion_135': (T, 135),  # motion_135 format
  'fps': 30
}
```

**Conversion Process:**
```python
# 1. Split motion_135
transl = motion[:, :3]              # (T, 3)
rot6d = motion[:, 3:135].reshape(T, 22, 6)  # (T, 22, 6)

# 2. Convert rot6d → axis-angle
aa = rot6d_to_axis_angle_np(rot6d)  # (T, 22, 3)

# 3. Run FK to compute joint positions (if needed)
global_orient = aa[:, 0, :]         # root rotation
body_pose = aa[:, 1:22, :].reshape(T, -1)  # 21 joints * 3

joints, _, _ = model.fk(
  transl=transl_t,
  global_orient=global_orient_t,
  body_pose=body_pose_t,
)  # (1, T, 22, 3)

# 4. Save JSON with frame data
{
  "fps": 30,
  "num_frames": T,
  "frames": [
    {"joints": [[x,y,z]*22]},  # per-frame joint positions
    ...
  ]
}
```

**Note:** This script produces **joint positions**, not SMPL poses. Used for skeleton-only visualization, not full mesh rendering.

---

## 13. Summary: How SMPL Animation Actually Works

1. **Static Setup** (load_smpl_with_shapes):
   - Load SMPL mesh (6890 vertices for SMPL+H)
   - Create bone hierarchy (52 bones for SMPL+H)
   - Bind geometry to skeleton for GPU skinning

2. **Per-Frame Animation** (updateFrame):
   - Receive frame data: `Th` (root translation), `poses` (axis-angle rotations)
   - Set mesh root position: `mesh.position = Th`
   - For each bone:
     - Extract 3D axis-angle from `poses`
     - Convert to quaternion: `setFromAxisAngle(axis.normalize(), angle)`
     - Three.js automatically propagates transforms down skeleton

3. **GPU Rendering** (renderer.render):
   - Three.js GPU shader uses LBS to deform vertices
   - For each vertex: `v_deformed = Σ(weight[i] * bone[i].transform * v_original)`
   - Result: smooth mesh deformation following skeleton

4. **Playback Loop** (playLoop):
   - Increment frame counter at specified FPS
   - Call updateFrame() for each new frame
   - Browser requestAnimationFrame handles timing

---

## References

**Key Files:**
- Server: `motion_annot_web/score_m2m/motion_utils.py` (conversion logic)
- Server: `motion_annot_web/score_m2m/score_m2m_web.py` (API endpoint)
- Client: `motion_annot_web/score_m2m/static/scripts3d/load_smpl.js` (mesh loading)
- Client: `motion_annot_web/score_m2m/templates/score.html` (animation loop)
- Embodied: `scripts/embodied/batch_npz_to_smpl_joints.py` (data export)

**Three.js Documentation:**
- `SkinnedMesh`: GPU-accelerated mesh deformation
- `Quaternion.setFromAxisAngle()`: Axis-angle to quaternion conversion
- `Skeleton`: Bone hierarchy management
