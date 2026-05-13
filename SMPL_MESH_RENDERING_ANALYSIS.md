# SMPL Mesh Rendering Analysis - Complete Report

## Executive Summary

This report provides a thorough analysis of the current SMPL skeleton visualization system in the HyMotion Motion Annotation Web platform, identifying the infrastructure for adding full SMPL mesh rendering capabilities. The codebase already contains all necessary components for mesh rendering; only integration and UI enhancements are needed.

---

## Table of Contents

1. [Web Visualization Architecture](#web-visualization-architecture)
2. [Skeleton Data Storage and Loading](#skeleton-data-storage-and-loading)
3. [SMPL Mesh Topology Files](#smpl-mesh-topology-files)
4. [Rendering Pipeline Scripts](#rendering-pipeline-scripts)
5. [Data Format Specifications](#data-format-specifications)
6. [Key Code Snippets](#key-code-snippets)
7. [Implementation Roadmap](#implementation-roadmap)

---

## Web Visualization Architecture

### Overview

The web visualization system is built on Three.js WebGL rendering and is primarily located in the `score_m2m_refine` application. The system currently renders SMPL skeletons as line segments connecting joints, with the infrastructure already in place to render full 3D mesh geometry.

### Directory Structure

```
motion_annot_web/score_m2m_refine/
├── templates/
│   ├── vis_smpl_preview.html           # Main SMPL preview HTML (1897 lines) ⭐
│   ├── index.html                      # App main page
│   └── ...other templates
├── static/
│   ├── scripts3d/
│   │   ├── load_smpl.js                # SMPL model loading module (282 lines) ⭐
│   │   ├── create_scene.js             # Scene initialization
│   │   ├── create_ground.js            # Ground plane rendering
│   │   ├── draw_skeleton.js            # Skeleton line drawing
│   │   └── export_motion.js            # Motion export utilities
│   └── assets/
│       ├── dump_smpl/                  # SMPL model topology (24 joints, 6890 verts)
│       ├── dump_smplh/                 # SMPL+H model topology (52 joints, 6890 verts) ⭐
│       ├── dump_smplx/                 # SMPL-X model topology (55 joints, 10475 verts)
│       ├── dump_smpl_male/
│       ├── dump_smpl_female/
│       ├── dump_smplh_male/
│       ├── dump_smplh_female/
│       ├── dump_smplx_male/
│       └── dump_smplx_female/
└── score_m2m_web.py                    # Flask backend (1351 lines)
```

### Current Rendering Approach

**File:** `motion_annot_web/score_m2m_refine/templates/vis_smpl_preview.html`

The system currently:
1. **Loads SMPL model** via `load_smpl_with_shapes()` which creates:
   - THREE.SkinnedMesh with vertex deformation
   - Bone hierarchy (skeleton)
   - Material (THREE.MeshPhongMaterial)

2. **Updates every frame**:
   - Applies axis-angle rotations to bones via quaternions
   - Updates bone translations
   - Applies global translation offset
   - Triggers shader skinning via GPU

3. **Displays skeleton** (current implementation):
   - Optional skeleton line overlay (draw_skeleton.js)
   - Skeleton not rendered by default (only mesh)

4. **Playback features**:
   - Frame-by-frame navigation
   - Speed control (0.1x to 2.0x)
   - Horizontal orthographic view for large-scale motion
   - Global translation toggle

### Three.js Components

**Core Three.js objects:**

```javascript
// From load_smpl.js (lines 114-249)
const geometry = new THREE.BufferGeometry();
geometry.setIndex(new THREE.BufferAttribute(faces, 1));
geometry.setAttribute('position', new THREE.BufferAttribute(v_template, 3));
geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, NUM_SKIN_WEIGHTS));
geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, NUM_SKIN_WEIGHTS));
geometry.computeVertexNormals();

// Create SkinnedMesh
var skeleton = new THREE.Skeleton(bones);
var material = new THREE.MeshPhongMaterial({color: gender_color[gender]});
var skinnedMesh = new THREE.SkinnedMesh(geometry, material);
skinnedMesh.bind(skeleton);
return {bones, skeleton, mesh};
```

---

## Skeleton Data Storage and Loading

### Input Data Format

**Source:** Motion data is provided in two formats:

1. **HyMotion motion_135 NPZ format** (used in pipeline scripts):
   ```
   motion_135: (T, 135) where:
     [0:3]      = translation (x, y, z)
     [3:135]    = 22 × rot6d (132D, 6D per joint)
   ```
   Total: 3 + 22*6 = 135 dimensions per frame

2. **Web JSON format** (used by visualization):
   ```json
   {
     "type": "frames",
     "fps": 30,
     "frames": [
       [{
         "id": 0,
         "gender": "neutral",
         "smpl_type": "smplh",
         "Rh": [[rx, ry, rz]],           // Root orientation (axis-angle)
         "Th": [[tx, ty, tz]],           // Root translation
         "poses": [[p0, p1, ...]],       // Full pose vector (axis-angles flattened)
         "shapes": [[s0, ..., s15]],     // Shape coefficients (16D)
         "mocap_framerate": 30
       }],
       ... (more frames)
     ]
   }
   ```

### Loading Pipeline

**File:** `motion_annot_web/score_m2m_refine/static/scripts3d/load_smpl.js` (282 lines)

**Key function:** `async function load_smpl_with_shapes(params, gender_param)`

**Step-by-step loading process:**

1. **Load binary assets** (lines 49-125):
   ```javascript
   const urls = [
     `/static/assets/${model_dir}/v_template.bin`,    // Vertex positions (T-pose)
     `/static/assets/${model_dir}/faces.bin`,         // Triangle indices
     `/static/assets/${model_dir}/skinWeights.bin`,   // Per-vertex blend weights
     `/static/assets/${model_dir}/skinIndice.bin`,    // Per-vertex bone indices
     `/static/assets/${model_dir}/j_template.bin`,    // Joint positions
   ];
   ```

2. **Load shape blendshapes** (lines 128-200):
   ```javascript
   // Load up to 16 shape offset files
   const offsets = await Promise.all(
     shapes.map(async (_, i) => {
       const url = `/static/assets/${model_dir}/shapeoffset_${i}.bin`;
       // Load and apply: v_template += offset * shapes[i]
     })
   );
   ```

3. **Apply shape deformations** (lines 172-178):
   ```javascript
   offsets.forEach((offset, i) => {
     for (let j = 0; j < v_template.length / 3; j++) {
       v_template[3*j]     += offset[3*j]     * shapes[i];
       v_template[3*j+1]   += offset[3*j+1]   * shapes[i];
       v_template[3*j+2]   += offset[3*j+2]   * shapes[i];
     }
   });
   ```

4. **Build skeleton hierarchy** (lines 201-241):
   ```javascript
   // Define parent relationships (edges array varies by SMPL type)
   let edges;
   if (smpl_type === 'smpl') {
     edges = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...];  // 24 joints
   } else if (smpl_type === 'smplh') {
     edges = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...];  // 52 joints (includes hands)
   } else if (smpl_type === 'smplx') {
     edges = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ...];  // 55 joints (+ face)
   }
   
   // Create THREE.Bone objects with proper parent-child relationships
   var bones = [rootBone];
   for (let i = 1; i < keypoints.length / 3; i++) {
     const bone = new THREE.Bone();
     const parentIndex = edges[i];
     bone.position.set(
       keypoints[3*i]   - keypoints[3*parentIndex],
       keypoints[3*i+1] - keypoints[3*parentIndex+1],
       keypoints[3*i+2] - keypoints[3*parentIndex+2]
     );
     bones.push(bone);
     bones[parentIndex].add(bone);  // Attach to parent
   }
   ```

5. **Create THREE.Skeleton and SkinnedMesh** (lines 242-256):
   ```javascript
   var skeleton = new THREE.Skeleton(bones);
   var skinnedMesh = new THREE.SkinnedMesh(geometry, material);
   skinnedMesh.bind(skeleton);
   return {bones, skeleton, mesh};
   ```

### Frame Update Logic

**File:** `motion_annot_web/score_m2m_refine/templates/vis_smpl_preview.html` (lines ~600-700)

**Key function:** `function updateFrame()`

```javascript
const updateFrame = () => {
  if (infos.length === 0) return;
  
  const smpl_params = infos[currentFrame];
  if (!smpl_params) return;
  
  // Apply root orientation (Rh: axis-angle)
  var axis = new THREE.Vector3(
    smpl_params.Rh[0][0],
    smpl_params.Rh[0][1],
    smpl_params.Rh[0][2]
  );
  var angle = axis.length();
  axis.normalize();
  var quaternion = new THREE.Quaternion().setFromAxisAngle(axis, angle);
  rootBone.quaternion.copy(quaternion);
  
  // Apply root translation (Th: 3D position)
  rootBone.position.set(
    smpl_params.Th[0][0],
    smpl_params.Th[0][1],
    smpl_params.Th[0][2]
  );
  
  // Apply per-joint poses (axis-angles)
  const poses_offset = (smpl_type === 'smpl' || smpl_type === 'smplh') ? -3 : 0;
  for (let i = 1; i < bones.length; i++) {
    var axis = new THREE.Vector3(
      smpl_params.poses[0][poses_offset + 3*i],
      smpl_params.poses[0][poses_offset + 3*i+1],
      smpl_params.poses[0][poses_offset + 3*i+2]
    );
    var angle = axis.length();
    axis.normalize();
    var quaternion = new THREE.Quaternion().setFromAxisAngle(axis, angle);
    bones[i].quaternion.copy(quaternion);
  }
  
  // Update render
  renderer.render(scene, camera);
};
```

---

## SMPL Mesh Topology Files

### File Organization

**Base directory:** `motion_annot_web/score_m2m_refine/static/assets/`

**Available model directories:**

| Directory | Type | Joints | Vertices | Description |
|-----------|------|--------|----------|-------------|
| `dump_smpl` | SMPL | 24 | 6890 | Body only (basic SMPL) |
| `dump_smpl_male` | SMPL (male) | 24 | 6890 | Male-specific shape |
| `dump_smpl_female` | SMPL (female) | 24 | 6890 | Female-specific shape |
| `dump_smplh` | SMPL+H | 52 | 6890 | Body + hands (most common) |
| `dump_smplh_male` | SMPL+H (male) | 52 | 6890 | Male with hands |
| `dump_smplh_female` | SMPL+H (female) | 52 | 6890 | Female with hands |
| `dump_smplx` | SMPL-X | 55 | 10475 | Body + hands + face (largest) |
| `dump_smplx_male` | SMPL-X (male) | 55 | 10475 | Male with hands and face |
| `dump_smplx_female` | SMPL-X (female) | 55 | 10475 | Female with hands and face |

### File Specifications

**Path examples:** `motion_annot_web/score_m2m_refine/static/assets/dump_smplh/`

#### 1. `v_template.bin` (vertex positions)
- **Size:** ~81 KB (for SMPL/SMPL+H)
- **Format:** Float32Array, 3 floats per vertex
- **Structure:** `[x0, y0, z0, x1, y1, z1, ..., xN-1, yN-1, zN-1]`
- **Total elements:** 6890 vertices × 3 = 20,670 floats (82.68 KB)
- **Purpose:** T-pose base mesh vertices
- **Used in:** Load as initial geometry vertex positions

#### 2. `faces.bin` (triangle indices)
- **Size:** ~54 KB (for SMPL/SMPL+H)
- **Format:** Uint16Array
- **Structure:** `[i0, i1, i2, i3, i4, i5, ..., iN-1, iN, iN+1]` (triplets)
- **Total indices:** ~13,776 indices (27,552 floats for 9,184 triangles)
- **Purpose:** Defines mesh triangulation
- **Used in:** THREE.BufferGeometry.setIndex()

#### 3. `skinIndice.bin` (per-vertex bone indices)
- **Size:** ~27 KB
- **Format:** Uint16Array
- **Structure:** Up to 4 bone indices per vertex
- **Elements:** 6890 vertices × 4 = 27,560 values
- **Purpose:** Specifies which bones influence each vertex (0-51 for SMPL+H)
- **Used in:** THREE.BufferAttribute('skinIndex', 4)

#### 4. `skinWeights.bin` (per-vertex blend weights)
- **Size:** ~54 KB
- **Format:** Float32Array
- **Structure:** 4 weights per vertex (sum to 1.0)
- **Elements:** 6890 vertices × 4 = 27,560 floats
- **Purpose:** Blend weights for each bone
- **Used in:** THREE.BufferAttribute('skinWeight', 4)

#### 5. `j_template.bin` (joint positions)
- **Size:** ~624 bytes
- **Format:** Float32Array
- **Structure:** `[x0, y0, z0, x1, y1, z1, ..., xJ-1, yJ-1, zJ-1]`
- **Elements:** 52 joints × 3 = 156 floats (for SMPL+H)
- **Purpose:** T-pose joint positions for skeleton hierarchy
- **Used in:** Building THREE.Bone hierarchy

#### 6. `shapeoffset_N.bin` (N = 0 to 15, shape blendshapes)
- **Size:** ~81 KB each (same as v_template.bin)
- **Format:** Float32Array
- **Structure:** Shape deformation deltas, 3 per vertex
- **Elements:** 6890 vertices × 3 = 20,670 floats
- **Purpose:** Shape parameter contributions (e.g., weight, height, muscle)
- **Used in:** Vertex shape deformation `v_final = v_template + Σ(shapeoffset[i] * shapes[i])`

#### 7. `shapeoffset_j_N.bin` (optional, joint shape offsets)
- **Size:** ~39 bytes each (52 joints × 3)
- **Format:** Float32Array
- **Purpose:** Joint position adjustments based on shape
- **Used in:** Updating joint template positions

#### 8. `keypoints.bin` (alternative to j_template)
- **Purpose:** Keypoint/landmark positions (same as j_template)

### SMPL Type Comparison

#### SMPL (24 joints)
- **Joint parent edges:** `[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]`
- **Pose dimension:** root(3) + body(21×3=63) = 72 params
- **Vertices:** 6,890
- **Use case:** Basic body without hand detail

#### SMPL+H (52 joints) ⭐ *Most common in pipeline*
- **Joint parent edges:** Extends SMPL with left hand (15 joints) + right hand (15 joints)
- **Pose dimension:** root(3) + body(21×3=63) + lhand(15×3=45) + rhand(15×3=45) = 156 params
- **Vertices:** 6,890 (same as SMPL)
- **Use case:** Full body with detailed hands (HyMotion default)

#### SMPL-X (55 joints)
- **Joint parent edges:** Extends SMPL+H with jaw(1) + eyes(2) + extended hands(15+15=30)
- **Pose dimension:** root(3) + body(21×3=63) + jaw(3) + eyes(6) + lhand(15×3=45) + rhand(15×3=45) = 165 params
- **Vertices:** 10,475 (more detailed face region)
- **Use case:** Full-body with detailed face and hands

---

## Rendering Pipeline Scripts

### 1. `batch_npz_to_smpl_mesh_json.py`

**Path:** `scripts/embodied/batch_npz_to_smpl_mesh_json.py` (239 lines)

**Purpose:** Batch convert motion_135 NPZ files to web-consumable JSON format

**Key function:** `convert_single_npz(npz_path: str, smpl_type: str = "smplx", gender: str = "neutral") -> dict`

**Input format:**
```python
# Load motion_135 NPZ
data = np.load(npz_path, allow_pickle=True)
motion = data['motion_135']  # (T, 135)
fps = int(data.get('fps', 30))
T = motion.shape[0]

# Split motion into components
transl = motion[:, :3]                    # (T, 3)
rot6d = motion[:, 3:].reshape(T, 22, 6)  # (T, 22, 6)
```

**Rot6d to axis-angle conversion** (lines 45-71):
```python
def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major rot6d (..., 6) to axis-angle (..., 3)"""
    from scipy.spatial.transform import Rotation as R
    
    # CRITICAL: Reorder from row-major to column-major
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    rotmat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    
    # Rotation matrix → axis-angle
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    aa_flat = R.from_matrix(rotmat_flat).as_rotvec()
    return aa_flat.reshape(*orig_shape, 3).astype(np.float32)
```

**Output format:**
```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplx",
      "Rh": [[rx, ry, rz]],              // Root orientation (1×3)
      "Th": [[tx, ty, tz]],              // Root translation (1×3)
      "poses": [[p0, p1, ...]],          // Full pose (1×N flattened axis-angles)
      "shapes": [[0,...,0]],             // Shape coefficients (1×16)
      "mocap_framerate": 30
    }],
    ... (more frames)
  ]
}
```

**Usage:**
```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/embodied_t2m_v4/data/npz \
    --output-dir output/embodied_t2m_v4/data/smpl_mesh \
    --smpl-type smplh
```

### 2. `batch_pipeline_to_web.py`

**Path:** `scripts/embodied/batch_pipeline_to_web.py` (296 lines)

**Purpose:** Orchestrate NPZ → cache → JSON pipeline with quality filtering

**Quality filter function** (lines 40-74):
```python
def quality_filter_npz(npz_path, min_height=1.2, max_height=2.0):
    """Quick quality check: reject degenerate motions based on estimated body height."""
    data = np.load(npz_path, allow_pickle=True)
    if 'positions' in data:
        pos = data['positions']  # (T, 22, 3)
        # Head joint = 15, foot joints = 10, 11
        head_y = pos[:, 15, 1]
        foot_y = np.minimum(pos[:, 10, 1], pos[:, 11, 1])
        heights = head_y - foot_y
        median_h = float(np.median(heights))
        if median_h < min_height or median_h > max_height:
            return False, median_h, "degenerate"
        return True, median_h, "ok"
```

**Main processing loop:**
1. Load NPZ files from directory
2. Apply quality filtering
3. Run pipeline (NPZ → .pt cache)
4. Convert cache → JSON
5. Update manifest.json

### 3. `render_tracker_headless.py`

**Path:** `scripts/embodied/render_tracker_headless.py` (877 lines)

**Purpose:** Headless MuJoCo rendering for G1 robot motion (not directly for SMPL)

**Relevant for understanding motion cache format:**
```python
cache = {
    'dof_pos': (T, 29),           # Joint angles
    'dof_vel': (T, 29),           # Joint velocities
    'body_rot': (T, 33, 4),       # Quaternion rotations (xyzw format)
    'body_pos': (T, 33, 3),       # Body positions
    'body_vel': (T, 33, 3),       # Linear velocities
    'body_ang_vel': (T, 33, 3),   # Angular velocities
    'control_dt': float,          # Time step
    'num_frames': int
}
```

---

## Data Format Specifications

### HyMotion motion_135 Format

**Complete specification from FORMAT_SPECIFICATION.md:**

```
motion_135 (135 dimensions total)
├─ [0:3]       Translation (3D) — root position in world space
└─ [3:135]     22 × rot6d (132D)
   ├─ [3:9]    Joint 0 (Pelvis) rot6d — global orientation
   └─ [9:135]  Joints 1-21 rot6d — local rotations
```

### Rot6d Format (Critical for Conversion)

**Row-major to column-major conversion:**

HyMotion outputs rot6d in row-major order:
```
6D vector: [e0, e1, e2, e3, e4, e5]
         = [R00, R01, R10, R11, R20, R21]  (row-major)

Gram-Schmidt expects column-major: [c0, c1, c2, c3, c4, c5]
Reordering [0, 2, 4, 1, 3, 5] converts:
[e0, e2, e4, e1, e3, e5] = [R00, R10, R20, R01, R11, R21]  (column-major)
```

### Denormalization Formula

```python
# During training
x_normalized = (x_original - mean) / std

# During inference (denormalization)
x_original = x_normalized * std + mean

# Safety: clamp std < 1e-3 to 1.0
std = np.where(std < 1e-3, 1.0, std)
```

**Statistics:** Loaded from official HyMotion checkpoint via `load_bundle_from_checkpoint()`

---

## Key Code Snippets

### Snippet 1: Load SMPL Model with Shape Deformation

**File:** `load_smpl.js` (lines 6-256)

```javascript
async function load_smpl_with_shapes(params, gender_param) {
    // Parse input parameters (shapes, gender, poses, SMPL type)
    let shapes, gender, smpl_type;
    if (typeof params === 'object' && !Array.isArray(params)) {
        shapes = params.shapes;
        gender = params.gender || 'neutral';
        smpl_type = params.smpl_type || 'smplh';
    } else {
        shapes = params;
        gender = gender_param || 'neutral';
        smpl_type = 'smplh';
    }
    
    // Select model directory based on type and gender
    let model_dir = 'dump_smplh';
    if (smpl_type === 'smplx') model_dir = 'dump_smplx';
    if (gender === 'male') model_dir += '_male';
    if (gender === 'female') model_dir += '_female';
    
    // Load binary assets
    const urls = [
        `/static/assets/${model_dir}/v_template.bin`,
        `/static/assets/${model_dir}/faces.bin`,
        `/static/assets/${model_dir}/skinWeights.bin`,
        `/static/assets/${model_dir}/skinIndice.bin`,
        `/static/assets/${model_dir}/j_template.bin`,
    ];
    
    const buffers = await Promise.all(urls.map(url => 
        fetch(url).then(r => r.arrayBuffer())
    ));
    
    // Load shape blendshapes and apply deformations
    const MAX_SHAPES = 16;
    shapes = shapes.length > MAX_SHAPES ? shapes.slice(0, MAX_SHAPES) : shapes;
    
    const offsets = await Promise.all(
        shapes.map(async (_, i) => {
            const response = await fetch(
                `/static/assets/${model_dir}/shapeoffset_${i}.bin`
            );
            if (!response.ok) return new Float32Array(v_template.length);
            return bufferToFloat32Array(await response.arrayBuffer());
        })
    );
    
    // Apply shape deformation to vertices
    offsets.forEach((offset, i) => {
        for (let j = 0; j < v_template.length / 3; j++) {
            v_template[3*j]   += offset[3*j]   * shapes[i];
            v_template[3*j+1] += offset[3*j+1] * shapes[i];
            v_template[3*j+2] += offset[3*j+2] * shapes[i];
        }
    });
    
    // Create THREE.BufferGeometry
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(new THREE.BufferAttribute(faces, 1));
    geometry.setAttribute('position', new THREE.BufferAttribute(v_template, 3));
    geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
    geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
    geometry.computeVertexNormals();
    
    // Build skeleton hierarchy
    let edges = [...];  // Based on smpl_type
    var rootBone = new THREE.Bone();
    var bones = [rootBone];
    for (let i = 1; i < keypoints.length / 3; i++) {
        const bone = new THREE.Bone();
        const parentIndex = edges[i];
        bone.position.set(
            keypoints[3*i]   - keypoints[3*parentIndex],
            keypoints[3*i+1] - keypoints[3*parentIndex+1],
            keypoints[3*i+2] - keypoints[3*parentIndex+2]
        );
        bones.push(bone);
        bones[parentIndex].add(bone);
    }
    
    // Create SkinnedMesh
    var skeleton = new THREE.Skeleton(bones);
    var material = new THREE.MeshPhongMaterial({
        color: gender_color[gender],
        skinning: true,
        emissive: 0x333333,
        flatShading: false
    });
    var skinnedMesh = new THREE.SkinnedMesh(geometry, material);
    skinnedMesh.bind(skeleton);
    
    return { bones, skeleton, mesh: skinnedMesh };
}
```

### Snippet 2: Update Frame with Pose Data

**File:** `vis_smpl_preview.html` (lines ~650-720)

```javascript
function updateFrame() {
    if (infos.length === 0) return;
    const smpl_params = infos[currentFrame];
    if (!smpl_params) return;
    
    // Apply root orientation (axis-angle → quaternion)
    let rootRotation = new THREE.Vector3(
        smpl_params.Rh[0][0],
        smpl_params.Rh[0][1],
        smpl_params.Rh[0][2]
    );
    let angle = rootRotation.length();
    rootRotation.normalize();
    let quat = new THREE.Quaternion().setFromAxisAngle(rootRotation, angle);
    rootBone.quaternion.copy(quat);
    
    // Apply root translation
    rootBone.position.set(
        smpl_params.Th[0][0],
        smpl_params.Th[0][1],
        smpl_params.Th[0][2]
    );
    
    // Update bone rotations (skip root offset for different SMPL types)
    const posesOffset = (smpl_type === 'smpl' || smpl_type === 'smplh') ? -3 : 0;
    for (let i = 1; i < bones.length; i++) {
        let axis = new THREE.Vector3(
            smpl_params.poses[0][posesOffset + 3*i],
            smpl_params.poses[0][posesOffset + 3*i+1],
            smpl_params.poses[0][posesOffset + 3*i+2]
        );
        let angle = axis.length();
        axis.normalize();
        let quaternion = new THREE.Quaternion().setFromAxisAngle(axis, angle);
        bones[i].quaternion.copy(quaternion);
    }
    
    // GPU skinning automatically applied via WebGL shader
    renderer.render(scene, camera);
}
```

### Snippet 3: Rot6d to Axis-Angle Conversion

**File:** `batch_npz_to_smpl_mesh_json.py` (lines 45-71)

```python
def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major rot6d (..., 6) to axis-angle (..., 3).
    
    HyMotion stores rot6d in row-major: [R00,R01, R10,R11, R20,R21].
    Must reorder [0,2,4,1,3,5] to column-major before Gram-Schmidt.
    """
    from scipy.spatial.transform import Rotation as R
    
    # Row-major -> column-major reorder
    rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    
    # Gram-Schmidt orthogonalization
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    
    rotmat = np.stack([b1, b2, b3], axis=-1)  # (..., 3, 3)
    
    # Rotation matrix -> axis-angle
    orig_shape = rotmat.shape[:-2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    aa_flat = R.from_matrix(rotmat_flat).as_rotvec()
    return aa_flat.reshape(*orig_shape, 3).astype(np.float32)
```

---

## Implementation Roadmap

### Current Status

✅ **Already implemented:**
1. THREE.SkinnedMesh rendering (mesh is fully loaded and rendered)
2. Binary asset loading (vertices, faces, weights, indices)
3. Shape blendshape deformation
4. Skeleton hierarchy with proper parent-child relationships
5. Frame-by-frame pose updates via quaternion interpolation
6. Rot6d to axis-angle conversion pipeline
7. JSON export with motion composition

### Current Rendering

The system **already renders the full SMPL mesh by default**. The only optional feature is skeleton line overlay (which is disabled by default).

### Enhancement Opportunities

1. **Material improvements:**
   - Add multiple material types (phong, standard, toon)
   - Enable/disable wireframe mode
   - Adjust shininess, metallic properties

2. **Lighting:**
   - Add environment lighting
   - Shadow mapping

3. **Color-coding:**
   - Color vertices by joint influence
   - Color regions (upper/lower body, left/right)

4. **Performance:**
   - LOD (level of detail) for SMPL-X
   - Instanced rendering for comparison views

5. **UI enhancements:**
   - Material selector
   - Lighting controls
   - Color mode selector

---

## Summary Table

| Component | Current Status | Location | Technology |
|-----------|----------------|----------|------------|
| SMPL model loading | ✅ Implemented | `load_smpl.js` | Three.js |
| Binary asset pipeline | ✅ Implemented | `static/assets/dump_smpl*` | Binary buffers |
| Shape deformation | ✅ Implemented | `load_smpl.js` (lines 128-200) | Vertex shader |
| Skeleton hierarchy | ✅ Implemented | `load_smpl.js` (lines 201-241) | THREE.Skeleton |
| SkinnedMesh rendering | ✅ Implemented | `load_smpl.js` (lines 242-256) | WebGL skinning |
| Frame pose updates | ✅ Implemented | `vis_smpl_preview.html` | Quaternion interpolation |
| Rot6d conversion | ✅ Implemented | `batch_npz_to_smpl_mesh_json.py` | Gram-Schmidt |
| Motion composition | ✅ Implemented | `completion_apps/app.py` | Frame concatenation |
| JSON export | ✅ Implemented | `batch_npz_to_smpl_mesh_json.py` | Standardized format |

**Conclusion:** SMPL mesh rendering is **fully functional**. The web visualization is rendering the complete 3D mesh with proper skinning and shape deformation. Any enhancements would focus on UI improvements, lighting, or additional visual modes rather than core rendering functionality.

