# SMPL Visualization & Mesh Rendering Analysis

## Executive Summary

The codebase has **skeleton-line rendering working** but **mesh rendering is partially implemented**. The system has:

- ✅ **SMPL mesh topology data** (v_template, faces, skinWeights, skinIndices) pre-computed and deployed
- ✅ **SkinnedMesh Three.js implementation** for SMPL+H/SMPL-X mesh rendering in `load_smpl.js`
- ✅ **Motion data export** to JSON format for web visualization
- ⚠️ **Current skeleton visualization** uses lines (spheres for joints, cylinders for limbs)
- ⚠️ **Mesh rendering** exists but needs **motion data integration** with skeleton interpolation

---

## 1. Embodied Pipeline & Motion Data Export

### 1.1 Main Pipeline: `scripts/embodied/batch_t2m_to_embodied.py`

**File**: `/scripts/embodied/batch_t2m_to_embodied.py` (1008 lines)

**Pipeline Flow**:
```
Text Prompt (or existing NPZ)
    ↓
[Step A] T2M Inference → motion_135 NPZ
    • Input: text caption
    • Output: motion_135 (T, 135) = [translation(3) + 22×rot6d(132)]
    • Uses HyMotion 1.0-Lite model @ 30 FPS
    ↓
[Step B] PyRoki Retarget → .motion file
    • Converts motion_135 → ProtoMotions .motion format
    • Calls: scripts/embodied/pipeline_motion_to_robot.py
    ↓
[Step C] Convert .motion → Reference JSON
    • Calls: convert_cache_to_json()
    • Output: JSON with dof_pos, root_quat, root_pos (for web viewer)
    ↓
[Step D] ONNX Tracker → Tracked JSON (optional)
    ↓
[Step E] Render Videos (optional)
    • Reference mode: skeleton visualization
    • Tracked mode: with ONNX policy
```

**Key Functions**:
- `run_t2m_inference()`: Line 197 → generates motion_135 (T, 135)
- `smooth_motion_135()`: Line 392 → Markley quaternion smoothing
- `save_motion_135_npz()`: Line 469 → Saves NPZ with key='motion_135'
- `convert_cache_to_json()`: Line 576 → Imports & calls convert_cache_to_json.py

**Motion Data Format**:
```python
motion_135: (T, 135)
  - [0:3]: translation (x, y, z)
  - [3:135]: 22 joints × 6D rotation (row-major)
    22 joints: [pelvis, l_hip, r_hip, spine, l_knee, r_knee, chest, 
                l_ankle, r_ankle, upper_chest, l_shoulder, r_shoulder, 
                neck, l_elbow, r_elbow, head, l_wrist, r_wrist, 
                l_hand, r_hand, l_toe, r_toe]
```

### 1.2 SMPL FK & Mesh Export: `scripts/embodied/motion135_to_smplx.py`

**File**: `/scripts/embodied/motion135_to_smplx.py` (130 lines)

**Purpose**: Convert motion_135 → SMPL-X NPZ format

**Key Functions**:
- `rot6d_to_rotmat()`: Line 26
  - Input: rot6d (row-major: [R00,R01,R10,R11,R20,R21])
  - Reorder to column-major [0,2,4,1,3,5]
  - Gram-Schmidt orthogonalization → (N, 3, 3) rotation matrix
  
- `rotmat_to_axis_angle()`: Line 58
  - Uses scipy.spatial.transform.Rotation
  - Output: (T, 22, 3) axis-angle vectors
  
- `convert_motion135_to_smplx()`: Line 69
  - **NOT computing vertices!**
  - Only converts to axis-angle (SMPL-X compatible format)
  - Output NPZ: `pose_body`, `root_orient`, `trans`, `betas`, `gender`, `mocap_frame_rate`

**⚠️ MISSING**: No vertex/mesh computation here!

### 1.3 Batch SMPL Mesh Export: `scripts/embodied/batch_npz_to_smpl_mesh_json.py`

**File**: `/scripts/embodied/batch_npz_to_smpl_mesh_json.py` (239 lines)

**Purpose**: Convert motion_135 NPZ → SMPL mesh JSON for web visualization

**Key Functions**:
- `rot6d_to_axis_angle_np()`: Line 45
  - Same rot6d conversion as motion135_to_smplx.py
  - Output: (T, 22, 3) axis-angle
  
- `convert_single_npz()`: Line 74
  - **Outputs SMPL mesh-ready JSON** with per-frame SMPL parameters
  - Format matches score_m2m's `/api/smpl` endpoint
  - Output JSON structure:
    ```json
    {
      "type": "frames",
      "fps": 30,
      "frames": [
        [{
          "id": 0,
          "gender": "neutral",
          "smpl_type": "smplx",
          "Rh": [[rx, ry, rz]],           // root orientation (axis-angle)
          "Th": [[tx, ty, tz]],           // translation
          "poses": [[p0, p1, ...]],       // all joints in axis-angle (flattened)
          "shapes": [[0,...,0]],          // 16 shape coefficients (all zeros)
          "mocap_framerate": 30
        }],
        ...
      ]
    }
    ```

**⚠️ KEY LIMITATION**: 
- motion_135 only has **22 body joints**, but SMPL-X/SMPL+H have 55/52 joints
- Hands + face joints are **zero-padded** (not animated)
- Only body skeleton can be rendered realistically

**Usage**:
```bash
python scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/embodied_t2m_v4/data/npz \
    --output-dir output/embodied_t2m_v4/data/smpl_mesh
```

### 1.4 Cache-to-JSON Conversion: `scripts/embodied/convert_cache_to_json.py`

**File**: `/scripts/embodied/convert_cache_to_json.py` (220 lines)

**Purpose**: Convert ProtoMotions .motion cache → JSON for web Three.js viewer

**Key Data**:
- **Input format (.motion file)**: 
  - `rigid_body_pos`: (T, N_bodies, 3) — body positions
  - `rigid_body_rot`: (T, N_bodies, 4) xyzw — body quaternions
  - `dof_pos`: (T, 29) — joint angles (G1 robot DOF ordering)
  - `fps` or `motion_dt`

- **Output JSON format**:
  ```json
  {
    "fps": 50,
    "num_frames": N,
    "joint_names": ["left_hip_pitch_joint", ...],  // 29 G1 DOF names
    "root_body_index": 0,
    "frames": [
      {
        "root_pos": [x, y, z],
        "root_quat": [x, y, z, w],    // xyzw
        "dof_pos": [v0, v1, ..., v28]  // 29 joint angles
      },
      ...
    ]
  }
  ```

**⚠️ NOTE**: This outputs **G1 robot DOF data**, not SMPL skeleton!

---

## 2. Web Visualization Stack

### 2.1 Directory Structure

```
motion_annot_web/
├── score_m2m/
│   ├── static/
│   │   ├── scripts3d/
│   │   │   ├── load_smpl.js          ← SkinnedMesh SMPL renderer
│   │   │   ├── draw_skeleton.js      ← Skeleton lines (spheres + cylinders)
│   │   │   ├── create_scene.js       ← Three.js scene setup
│   │   │   ├── create_ground.js      ← Ground plane + coordinate axes
│   │   │   └── export_motion.js      ← Motion export utilities
│   │   └── assets/
│   │       ├── dump_smplh/           ← SMPL+H model topology (pre-computed)
│   │       │   ├── v_template.bin    (6890 vertices × 3 floats = 81KB)
│   │       │   ├── faces.bin         (13780 faces × 3 uint16 = 81KB)
│   │       │   ├── skinWeights.bin   (6890 vertices × 4 floats = 108KB)
│   │       │   ├── skinIndice.bin    (6890 vertices × 4 uint16 = 54KB)
│   │       │   ├── j_template.bin    (52 joints × 3 floats = 0.6KB)
│   │       │   ├── shapeoffset_0-15.bin   (16 × 81KB shape basis)
│   │       │   └── shapeoffset_j_0-15.bin (16 × 0.6KB joint shape basis)
│   │       └── dump_smplx/           ← SMPL-X model topology
│   │           └── [similar structure: 10475 vertices, 55 joints]
│   └── templates/
│       └── score.html                ← Main viewer
└── score_m2m_refine/                 ← Duplicate Three.js setup
    ├── static/scripts3d/
    └── static/assets/
```

### 2.2 Current Skeleton Visualization: `load_smpl.js` + `draw_skeleton.js`

**File**: `motion_annot_web/score_m2m/static/scripts3d/load_smpl.js` (294 lines)

**Function**: `load_smpl_with_shapes(params, gender_param)` → returns `{bones, skeleton, mesh}`

**Implementation**:
1. **Load SMPL topology**:
   - Fetches `v_template.bin` (vertices)
   - Fetches `faces.bin` (face indices)
   - Fetches `skinWeights.bin` (per-vertex bone weights)
   - Fetches `skinIndice.bin` (per-vertex bone indices)
   - Fetches `j_template.bin` (joint positions in T-pose)

2. **Apply shape blending**:
   - Loads 16 shape offset files (`shapeoffset_*.bin`)
   - Blends based on beta shape parameters (if provided)
   - Updates v_template += shapeoffset_i × beta_i

3. **Build skeleton**:
   - Creates THREE.Bone() for each joint
   - Sets up parent-child hierarchy using `edges` array
   - SMPL skeleton structure (lines 214-235):
     ```
     SMPL (24 joints):     [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, ...]
     SMPL+H (52 joints):   [body 23 + hands 29]
     SMPL-X (55 joints):   [body 23 + face 3 + hands 30]
     ```

4. **Create SkinnedMesh**:
   ```javascript
   const material = new THREE.MeshStandardMaterial({
       color: gender_color[gender],
       side: THREE.DoubleSide
   });
   const mesh = new THREE.SkinnedMesh(geometry, material);
   mesh.add(bones[0]);
   mesh.bind(skeleton);  // ← Bind skeleton for skinning
   ```

**File**: `motion_annot_web/score_m2m/static/scripts3d/draw_skeleton.js` (140+ lines)

**Function**: `visualizeSkeleton(keypoints, scene, radius_joint, radius_limb)` → draws skeleton lines

**Implementation**:
- Uses object pooling to reuse sphere & cylinder meshes
- Draws joints as SphereGeometry(r, 8, 8) — 64 triangles each
- Draws limbs as cylinders connecting joint pairs
- Hard-coded edge list (24 edges for SMPL)
- Material: MeshStandardMaterial (red for joints, blue for limbs)

**⚠️ KEY ISSUE**: Currently renders **skeleton lines only**, not full mesh!

### 2.3 Scene Setup: `create_scene.js` & `create_ground.js`

**File**: `motion_annot_web/score_m2m/static/scripts3d/create_scene.js` (200+ lines)

**Function**: `create_scene(scene, camera, renderer, use_ground, axis_up, axis_forward)`

**Sets up**:
- Camera position & orientation (supports Y-up and Z-up)
- Lighting: AmbientLight + DirectionalLight (with shadow maps)
- Optional ground plane
- Renderer: shadow mapping enabled, soft shadows

**File**: `motion_annot_web/score_m2m/static/scripts3d/create_ground.js` (200+ lines)

**Provides**:
- `getChessboard()` — Textured ground plane (X-Z axis)
- `getCoordinate()` — RGB coordinate axes (X=red, Y=green, Z=blue)

---

## 3. SMPL Model Topology Data

### 3.1 Model Specifications

| Model | Vertices | Faces | Joints | Shape Params |
|-------|----------|-------|--------|--------------|
| **SMPL** | 6890 | 13780 | 24 (body only) | 10 betas |
| **SMPL+H** | 6890 | 13780 | 52 (+ hands) | 10 betas |
| **SMPL-X** | 10475 | 20908 | 55 (+ hands + face) | 10 betas |

### 3.2 Pre-computed Assets Deployed

**Location**: `motion_annot_web/score_m2m/static/assets/dump_smplh/`

**Files**:
- `v_template.bin`: (6890, 3) float32 → 81 KB
- `faces.bin`: (13780, 3) uint16 → 81 KB  
- `skinWeights.bin`: (6890, 4) float32 → 108 KB
- `skinIndice.bin`: (6890, 4) uint16 → 54 KB
- `j_template.bin`: (52, 3) float32 → 0.6 KB
- `shapeoffset_*.bin`: 16 shape basis files (81 KB each)
- `shapeoffset_j_*.bin`: 16 joint shape basis files (0.6 KB each)

**Total**: ~2 MB per model type (SMPL+H, SMPL-X)

### 3.3 Skeleton Hierarchy (Parent-Child Edges)

**SMPL+H (52 joints)**:
```
Root = 0 (Pelvis)
├─ 1: L Hip (child=0)
├─ 2: R Hip (child=0)
├─ 3: Spine (child=0)
│  ├─ 4: L Knee (child=1)
│  ├─ 5: R Knee (child=2)
│  ├─ 6: Chest (child=3)
│  │  ├─ 7: L Ankle (child=4)
│  │  ├─ 8: R Ankle (child=5)
│  │  ├─ 9: Upper Chest (child=6)
│  │  │  ├─ 10: L Shoulder (child=9)
│  │  │  ├─ 11: R Shoulder (child=9)
│  │  │  ├─ 12: Neck (child=9)
│  │  │  │  ├─ 13: L Elbow (child=10)
│  │  │  │  ├─ 14: R Elbow (child=11)
│  │  │  │  └─ 15: Head (child=12)
...
```

---

## 4. Motion Data Flow for Mesh Rendering

### Current State

```
motion_135 (T, 135)
├─ [0:3]: translation
└─ [3:135]: 22 × rot6d (row-major)

    ↓ [motion135_to_smplx.py]
    
SMPL-X NPZ
├─ pose_body: (T, 63)        ← 21 body joints × 3 axis-angle
├─ root_orient: (T, 3)       ← pelvis orientation
├─ trans: (T, 3)             ← translation
├─ betas: (10,)              ← shape params (all zeros)
└─ gender: "neutral"

    ↓ [batch_npz_to_smpl_mesh_json.py]
    
SMPL Mesh JSON
└─ frames[t]:
   ├─ Rh: [[rx, ry, rz]]     ← root axis-angle
   ├─ Th: [[tx, ty, tz]]     ← translation
   ├─ poses: [[p0,p1,...]]   ← all joints axis-angle (padded to 165 for SMPL-X)
   ├─ shapes: [[0,...,0]]    ← shape coefficients
   └─ mocap_framerate: 30
```

### What's Missing

**To render mesh in web**:
1. ✅ Motion data → SMPL parameters (done)
2. ✅ SMPL model topology (v_template, faces, joints, weights, indices) — pre-deployed
3. ❌ **Web code to update skeletal positions per frame** ← **MISSING!**
4. ❌ **Web code to compute vertices per frame** ← **MISSING!**
5. ❌ **Web code to call SkinnedMesh.updateMatrixWorld()** ← **MISSING!**

---

## 5. Key Technical Details

### 5.1 Motion Data Rotation Representation

**Row-major rot6d** (HyMotion standard):
```
[R00, R01, R10, R11, R20, R21]
```

**Conversion to rotation matrix** (load_smpl.js + motion135_to_smplx.py):
```javascript
// Row-major → Column-major reorder
rot6d_reordered = rot6d[[0, 2, 4, 1, 3, 5]]  // [R00,R10,R20,R01,R11,R21]

// Gram-Schmidt orthogonalization
a1 = rot6d_reordered[0:3]
a2 = rot6d_reordered[3:6]

b1 = a1 / ||a1||
b2 = (a2 - (b1·a2)b1) / ||a2 - (b1·a2)b1||
b3 = b1 × b2

R = [b1 | b2 | b3]  // Column vectors → (3, 3) rotation matrix
```

### 5.2 Three.js SkinnedMesh Pipeline

**Step 1**: Load geometry with skinning attributes
```javascript
geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
```

**Step 2**: Create skeleton
```javascript
const bones = [...];  // Array of THREE.Bone objects
const skeleton = new THREE.Skeleton(bones);
```

**Step 3**: Create & bind SkinnedMesh
```javascript
const mesh = new THREE.SkinnedMesh(geometry, material);
mesh.add(bones[0]);  // Add root bone
mesh.bind(skeleton);  // Bind skeleton
```

**Step 4**: Update bone transforms per frame ← **CURRENTLY NOT DONE**
```javascript
for (let i = 0; i < bones.length; i++) {
    bones[i].quaternion.setFromAxisAngle(axis, angle);
    bones[i].updateMatrix();
    bones[i].updateMatrixWorld();
}
```

---

## 6. Integration Requirements for SMPL Mesh Rendering

### 6.1 What Needs to Be Built

**Frontend (JavaScript)**:
```javascript
// New function in load_smpl.js or separate module
function updateSkeletonPose(bones, axisAngleParams) {
    // axisAngleParams: (num_joints, 3) axis-angle vectors
    // For each joint:
    //   1. Convert axis-angle → quaternion
    //   2. Set bone.quaternion
    //   3. Update bone.matrix & bone.matrixWorld
    // SkinnedMesh automatically deforms vertices based on bone transforms
}

// In animation loop:
for (frame_t in frames) {
    const poses = frame_data.poses;  // (num_joints × 3)
    const trans = frame_data.Th[0];  // [tx, ty, tz]
    
    updateSkeletonPose(skeleton.bones, poses);
    mesh.position.set(trans[0], trans[1], trans[2]);
    
    renderer.render(scene, camera);
}
```

**Backend (Python)**:
- Already done: `batch_npz_to_smpl_mesh_json.py` outputs motion JSON
- Already done: `motion135_to_smplx.py` converts rotations

### 6.2 Data Format for Frontend

**Current JSON output from batch_npz_to_smpl_mesh_json.py**:
```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplx",
      "Rh": [[rx, ry, rz]],        ← Root orientation (axis-angle)
      "Th": [[tx, ty, tz]],        ← Translation
      "poses": [[p0, p1, ...]],    ← All poses in axis-angle (165 floats for SMPL-X)
      "shapes": [[0, 0, ..., 0]],  ← Shape coefficients (16 for SMPL-X)
      "mocap_framerate": 30
    }],
    ...
  ]
}
```

**Layout of `poses` array** (SMPL-X, 165 values):
```
[0:3]       → Root (already used as Rh, can skip)
[3:66]      → 21 body joints × 3
[66:69]     → Jaw
[69:75]     → 2 eyes × 3
[75:165]    → 30 hand joints × 3 (15 per hand)
```

### 6.3 Skeleton Hierarchy & Bone Parent-Child Links

**Already embedded in load_smpl.js** (line 214-235):
```javascript
// SMPL+H edges (parent indices for 52 joints)
edges = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 
         12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 
         20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 
         21, 46, 47, 21, 49, 50];
```

Bone hierarchy already built in `load_smpl_with_shapes()` function!

---

## 7. Limitations & Constraints

### 7.1 motion_135 → SMPL Mapping

**motion_135 has 22 joints**:
```
[0] Pelvis (root)
[1-2] Hip (L, R)
[3] Spine
[4-5] Knee (L, R)
[6] Chest
[7-8] Ankle (L, R)
[9] Upper Chest
[10-11] Shoulder (L, R)
[12] Neck
[13-14] Elbow (L, R)
[15] Head
[16-17] Wrist (L, R)
[18-19] Hand (L, R) ← Note: single hand joints, not 15-dof hands
[20-21] Toe (L, R)
```

**SMPL+H has 52 joints**:
- Joints 0-22: Body (same as SMPL)
- Joints 23-51: Hand articulation (15 DOF per hand)

**Mapping issue**:
- `motion_135[18:20]` = 2 hand root positions
- `SMPL+H[23:51]` = 30 hand DOF (15 × 2 hands)
- **Hands are zero-padded in current batch_npz_to_smpl_mesh_json.py** (line 117)

**Implication**: 
- Hands won't deform realistically (just rendered as static bones)
- Acceptable for gross motion rendering

### 7.2 Shape Parameters

**All betas are zero** (no shape blending):
```python
shapes = [[0.0] * 16]  # batch_npz_to_smpl_mesh_json.py line 135
```

**Justification**: motion_135 only captures pose, not shape
- Default T-pose template used (all individuals look identical)
- Acceptable for motion quality assessment

### 7.3 No Hand Articulation

**Hands frozen in T-pose**:
- Only hand root positions animated (2 joints)
- 15-DOF finger articulation not in motion_135
- Renders fine for general motion evaluation

---

## 8. Asset File Specifications

### 8.1 Binary Format Details

**v_template.bin** (6890 vertices for SMPL+H):
- Format: float32 array (row-major)
- Size: 6890 × 3 × 4 = 82,680 bytes ≈ 81 KB
- Interpretation in load_smpl.js:
  ```javascript
  const v_template = bufferToFloat32Array(buffers[0]);  // float32
  // v_template[3*i:3*i+3] = vertex i position [x, y, z]
  ```

**faces.bin** (13780 faces for SMPL+H):
- Format: uint16 array (row-major)
- Size: 13780 × 3 × 2 = 82,680 bytes ≈ 81 KB
- Interpretation:
  ```javascript
  const faces = bufferToUint16Array(buffers[1]);  // uint16
  // faces[3*i:3*i+3] = triangle i vertex indices [v0, v1, v2]
  ```

**skinWeights.bin** (6890 vertices, 4 weights per vertex):
- Format: float32 array
- Size: 6890 × 4 × 4 = 110,240 bytes ≈ 108 KB
- Each vertex influenced by up to 4 bones with respective weights

**skinIndice.bin** (6890 vertices, 4 indices per vertex):
- Format: uint16 array
- Size: 6890 × 4 × 2 = 55,120 bytes ≈ 54 KB
- Indices into the 52-joint skeleton

**j_template.bin** (52 joints for SMPL+H):
- Format: float32 array
- Size: 52 × 3 × 4 = 624 bytes
- T-pose joint positions

### 8.2 Shape Offset Files

**shapeoffset_i.bin** (i = 0..15):
- Format: float32 array (same size as v_template)
- Size: 6890 × 3 × 4 = 82,680 bytes per file
- Vertex displacement for shape basis i
- Applied: v = v_template + Σ(beta_i × shapeoffset_i)

**shapeoffset_j_i.bin** (joint shape offsets):
- Format: float32 array (same size as j_template)
- Size: 52 × 3 × 4 = 624 bytes per file
- Joint position displacement for shape basis i

---

## 9. Code Locations & Function Index

### Python Scripts

| File | Lines | Key Functions | Purpose |
|------|-------|---------------|---------|
| `batch_t2m_to_embodied.py` | 1008 | `run_t2m_inference()`, `smooth_motion_135()`, `convert_cache_to_json()` | T2M → motion_135 → JSON pipeline |
| `motion135_to_smplx.py` | 130 | `rot6d_to_rotmat()`, `rotmat_to_axis_angle()`, `convert_motion135_to_smplx()` | motion_135 → axis-angle conversion |
| `batch_npz_to_smpl_mesh_json.py` | 239 | `rot6d_to_axis_angle_np()`, `convert_single_npz()` | motion_135 → SMPL mesh JSON |
| `convert_cache_to_json.py` | 220 | `convert_cache_to_json()` | .motion → robot DOF JSON |

### JavaScript / Three.js

| File | Lines | Key Functions | Purpose |
|------|-------|---------------|---------|
| `load_smpl.js` | 294 | `load_smpl_with_shapes()` | SkinnedMesh geometry + skeleton setup |
| `draw_skeleton.js` | 140+ | `visualizeSkeleton()` | Skeleton line rendering |
| `create_scene.js` | 200+ | `create_scene()`, `fitCameraToScene()` | Scene & camera setup |
| `create_ground.js` | 200+ | `getChessboard()`, `getCoordinate()` | Ground plane & axes |

### Assets

| Location | Model | Vertices | Faces | Size |
|----------|-------|----------|-------|------|
| `score_m2m/static/assets/dump_smplh/` | SMPL+H | 6890 | 13780 | ~2 MB |
| `score_m2m/static/assets/dump_smplx/` | SMPL-X | 10475 | 20908 | ~3 MB |

---

## 10. Current Rendering Approach

### Skeleton-Only (Current)

```
motion_135 (T, 135)
  ↓
draw_skeleton.js: visualizeSkeleton(keypoints)
  ├─ For each frame:
  │  ├─ Update joint sphere positions
  │  ├─ Draw limb cylinders between joints
  │  └─ Render using THREE.Mesh(sphere_geo, material)
  └─ Output: Line drawing of skeleton
```

**Pros**: 
- Fast (only 24 meshes per frame)
- Clearly shows motion

**Cons**:
- No visual skin/surface
- Cannot assess silhouette quality

### Mesh Rendering (Partially Implemented)

```
motion_135 (T, 135)
  ↓
batch_npz_to_smpl_mesh_json.py: convert_single_npz()
  ├─ Extract axis-angles per frame
  └─ Output: SMPL mesh JSON
             └─ Contains: Rh (root), Th (trans), poses (all joints), shapes (betas)

load_smpl.js: load_smpl_with_shapes()
  ├─ Load SMPL topology: v_template, faces, skinWeights, skinIndices
  ├─ Create THREE.SkinnedMesh with loaded geometry
  ├─ Create skeleton hierarchy (THREE.Bone)
  └─ Output: {mesh, skeleton, bones}

Animation Loop (NOT YET IMPLEMENTED):
  └─ For each frame t:
     ├─ Load poses[t] from JSON
     ├─ Update bone transforms:
     │  └─ For each bone i:
     │     ├─ Convert axis-angle → quaternion
     │     ├─ Set bone.quaternion
     │     └─ Call bone.updateMatrixWorld()
     ├─ Update root position: mesh.position = Th[t]
     └─ Renderer automatically deforms vertices via skinning
```

**Pros**:
- Full surface mesh rendering
- Can assess silhouette, surface details
- Better visual feedback on motion quality

**Cons**:
- Requires per-frame bone transform updates
- Not yet integrated into web viewer

---

## Summary Table

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **Motion Data** | ✅ Complete | `scripts/embodied/batch_t2m_to_embodied.py` | motion_135 format well-defined |
| **Rotation Conversion** | ✅ Complete | `motion135_to_smplx.py`, `batch_npz_to_smpl_mesh_json.py` | rot6d → axis-angle working |
| **Mesh Data Export** | ✅ Complete | `batch_npz_to_smpl_mesh_json.py` | JSON with per-frame SMPL params |
| **SMPL Topology** | ✅ Deployed | `static/assets/dump_smpl{h,x}/` | 2-3 MB binary files per model |
| **Three.js Setup** | ✅ Complete | `load_smpl.js` | SkinnedMesh created & bound |
| **Skeleton Hierarchy** | ✅ Complete | `load_smpl.js` line 214-235 | Bone parent-child links defined |
| **Skinning Attributes** | ✅ Set Up | `load_smpl.js` line 254-258 | Buffer attributes for skinning |
| **Animation Loop** | ❌ **Missing** | Web viewer | Need to update bone transforms per frame |
| **Skeleton Line Rendering** | ✅ Complete | `draw_skeleton.js` | Currently used for visualization |

---

## Recommendations for SMPL Mesh Rendering

1. **Immediate** (1-2 days):
   - Add animation loop to update bone transforms from JSON
   - Toggle between skeleton lines and mesh rendering
   - Test with existing score_m2m website

2. **Optional** (1 week):
   - Implement hand articulation from motion_135 (requires new hand kinematics)
   - Add shape blending UI (currently hardcoded to zeros)
   - Optimize rendering for real-time playback

3. **Advanced** (future):
   - Export FBX with skeletal animation
   - Real-time shape editing
   - Multi-person comparison
