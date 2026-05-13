# SMPL Mesh Visualization Infrastructure — Comprehensive Summary

## Overview
This document summarizes the embodied motion visualization infrastructure: from motion generation through SMPL mesh rendering in web viewers. The system converts text prompts → HyMotion T2M → motion_135 NPZ → robot retargeting → ProtoMotions cache → web-viewable JSON and SMPL mesh visualization.

---

## 1. `batch_npz_to_smpl_mesh_json.py` — NPZ to SMPL Mesh JSON Converter

**File Path:** `scripts/embodied/batch_npz_to_smpl_mesh_json.py`

**Purpose:** Convert motion_135 NPZ files directly to SMPL mesh-ready JSON for Three.js web visualization.

### Input Format
- **NPZ file structure:**
  - `motion_135`: (T, 135) array
    - [0:3] = translation (x, y, z)
    - [3:135] = 22 × rot6d (row-major format)
      - Layout: [R00, R01, R10, R11, R20, R21] per joint
  - `fps`: Frames per second (default 30)

### Output Format
Produces JSON matching the `/api/smpl` format used by score_m2m web viewer:

```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",          // or "male", "female"
      "smpl_type": "smplh",         // "smpl", "smplh", or "smplx"
      "Rh": [[rx, ry, rz]],         // 1×3 root orientation (axis-angle)
      "Th": [[tx, ty, tz]],         // 1×3 translation
      "poses": [[p0, p1, ...]],     // 1×N flattened body joint axis-angles
      "shapes": [[0,...,0]],        // 1×16 shape coefficients (zeros = neutral shape)
      "mocap_framerate": 30
    }],
    ...
  ]
}
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `rot6d_to_axis_angle_np(rot6d)` | Convert row-major rot6d (…, 6) → axis-angle (…, 3) via Gram-Schmidt orthogonalization + scipy Rotation |
| `convert_single_npz(npz_path, smpl_type, gender)` | Main converter: extract motion_135, split transl/rot6d, build per-frame SMPL pose arrays |
| `main()` | Batch processor with --skip-existing, --quality-filter options |

### SMPL Type Support
- **SMPL** (24 joints): 72 pose params [root(3) + body(23×3=69)]
- **SMPL+H** (52 joints, default): 156 pose params [root(3) + body(21×3=63) + hands(30×3=90)]
- **SMPL-X** (55 joints): 165 pose params [root(3) + body(21×3=63) + jaw(3) + eyes(6) + hands(90)]

**Note:** motion_135 contains only 22 body joints (1 root + 21 body). Hand/jaw/eye joints are zero-padded.

### Usage Example
```bash
# Batch convert directory
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/embodied_t2m_v4/data/npz \
    --output-dir output/embodied_t2m_v4/data/smpl_mesh \
    --smpl-type smplh \
    --gender neutral

# Single file
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-file data/embodied_debug/v6_e2e_test/npz/wave_hand.npz \
    --output-dir data/embodied_debug/v6_e2e_test/smpl_mesh
```

---

## 2. `batch_pipeline_to_web.py` — Full ProtoMotions Pipeline to Web

**File Path:** `scripts/embodied/batch_pipeline_to_web.py`

**Purpose:** Full end-to-end pipeline from motion_135 NPZ → ProtoMotions retarget cache → JSON for web visualization.

### Pipeline Steps

```
motion_135 NPZ 
    ↓ [Step 1: run_pipeline()]
ProtoMotions cache (.pt or .motion)
    ↓ [Step 2: convert_cache_to_json()]
JSON for Three.js viewer
```

### Key Components

| Component | Function |
|-----------|----------|
| `quality_filter_npz()` | Reject degenerate motions (estimated body height < 1.2m or > 2.0m) using joint positions |
| `run_pipeline()` | Spawns `pipeline_motion_to_robot.py` (V6 PyRoki retargeting, ~60-70 min per motion on CPU) |
| `convert_cache_to_json()` | Calls `convert_cache_to_json.py` to convert .pt/.motion → JSON |

### Does it call `batch_npz_to_smpl_mesh_json.py`?
**No.** This pipeline produces **robot motion JSON** (G1 DOF angles), not SMPL mesh JSON. It has its own conversion path:
1. motion_135 NPZ → PyRoki retargeting → ProtoMotions .motion
2. .motion → JSON with robot joint angles + rigid body positions/rotations

**Different format:** Robot JSON has `dof_pos` (29 joint angles), `root_pos`, `root_quat`. SMPL mesh JSON has `Rh`, `Th`, `poses`, `shapes`.

### Output Structure
```
output_dir/
├── motions/
│   ├── pipeline_00000.json          (reference motion)
│   ├── pipeline_00001.json
│   └── manifest.json                (motion metadata list)
├── tracked_motions/
│   └── pipeline_*.json              (ONNX-tracked version)
├── caches/
│   └── pipeline_*.pt                (intermediate cache)
└── manifest.json                    (overall metadata)
```

### Usage Example
```bash
python scripts/embodied/batch_pipeline_to_web.py \
    --npz-dir work_dirs/all_tasks_after_fix_20260421/uncond_local/E2_B/npz/ \
    --output-dir output/embodied_comparison/data/motions/ \
    --quality-filter \
    --max-motions 20
```

---

## 3. `batch_t2m_to_embodied.py` — Full Text-to-Motion Pipeline (V6)

**File Path:** `scripts/embodied/batch_t2m_to_embodied.py` (1008 lines)

**Purpose:** Complete end-to-end pipeline: text prompts → HyMotion T2M inference → motion_135 NPZ → PyRoki retargeting → web JSONs + metadata.

### Full Pipeline Steps

```
Text Prompts
    ↓ [A] T2M Inference (HyMotion)
motion_135 NPZ
    ↓ [B] PyRoki Retarget Pipeline (V6)
ProtoMotions .motion cache
    ↓ [C] Robot JSON conversion
Reference JSON (robot DOF + rigid body pose)
    ↓ [D] ONNX Tracker (optional)
Tracked JSON
    ↓ [E] Video Rendering (optional)
Reference + Tracked videos
    ↓ [F] Metrics extraction
Per-motion metadata
    ↓ [G] Manifest generation
motion_text_mapping.json + batch_report.json
```

### Does it generate SMPL mesh JSONs?
**No.** This pipeline generates **robot motion JSONs** (G1 humanoid DOF angles), not SMPL mesh JSONs. The SMPL mesh format is used in score_m2m_refine web app for annotation, not in this embodied robot comparison.

### Key Input Modes

| Mode | Purpose |
|------|---------|
| `--prompt-json` | Load from motion_text_mapping.json (multiple prompts from file) |
| `--prompts` | Inline text prompts (command line) |
| `--npz-dir` | Use existing motion_135 NPZ files (skip T2M inference) |

### Step [A]: T2M Inference

**Function:** `load_t2m_bundle()`, `run_t2m_inference()`

- Loads HyMotion T2M 1.0-Lite checkpoint
- Generates 201-dim motion vectors (T, 201)
- Extracts first 135 dims → motion_135 (T, 135)
- Optionally applies **Markley quaternion smoothing** (sigma=1.0, 9-tap Gaussian kernel)

**Smoothing strategy:**
- Translation: Savitzky-Golay filter (window=11, polyorder=5)
- Rotation: Quaternion space with Markley weighted averaging
  - rot6d → rotation matrix → quaternion → Markley avg → rotation matrix → rot6d
  - Preserves rotational manifold (unlike direct rot6d filtering)

### Step [B]: PyRoki Retargeting (V6)

**Function:** `run_retarget_pipeline()`

Spawns `pipeline_motion_to_robot.py` which chains:
1. `motion135_to_pyroki_keypoints.py` — motion_135 → SMPL FK → PyRoki keypoints
2. `batch_retarget_to_g1_from_keypoints.py` — PyRoki trajectory-level optimization (jaxls, 800 iterations)
3. `convert_pyroki_retargeted_robot_motions_to_proto.py` — Retargeted NPZ → ProtoMotions .motion

**Optimization weights:**
- Local bone alignment: 1.0
- Global keypoint alignment: 4.0
- Foot contact cost: 30.0
- Joint smoothness: 4.0
- Root smoothness: 1.0
- Joint velocity limit: 50.0

### Step [C-D]: JSON Conversion & Tracking

**Function:** `convert_cache_to_json()` (from convert_cache_to_json.py)

Converts .motion file to JSON:
```json
{
  "fps": 50,
  "num_frames": N,
  "joint_names": ["left_hip_pitch_joint", ...],  // 29 G1 joints
  "root_body_index": 0,
  "frames": [
    {
      "root_pos": [x, y, z],
      "root_quat": [x, y, z, w],
      "dof_pos": [v0, v1, ..., v28]
    },
    ...
  ]
}
```

### Output Directory Structure
```
output_root/
├── data/
│   ├── motions/                  (reference robot JSONs)
│   │   ├── motion_0000.json
│   │   └── manifest.json
│   ├── tracked_motions/          (ONNX-tracked JSONs)
│   ├── caches/                   (intermediate .motion/.pt)
│   ├── npz/                      (motion_135 NPZ files)
│   ├── renders/                  (reference + tracked videos)
│   ├── retarget/                 (PyRoki output dir per motion)
│   └── meta/                     (per-motion metadata.json)
├── motion_text_mapping.json      (prompt → motion_id)
└── batch_report.json
```

### Key Configuration

| Param | Default | Purpose |
|-------|---------|---------|
| `--config` | configs/hymotion_t2m/hymotion_t2m_201dim_046b.py | T2M config |
| `--checkpoint` | checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt | T2M weights |
| `--num-steps` | 100 | ODE denoising steps |
| `--guidance-scale` | 5.0 | Classifier-free guidance scale |
| `--device` | cuda | Inference device |
| `--smooth` | True | Apply Markley smoothing to motion_135 |
| `--render-width` | 640 | Video render width |
| `--render-height` | 480 | Video render height |

### Metrics Extraction

**Function:** `extract_metrics_from_cache()`

Extracts per-motion quality metrics:
- `num_frames`, `fps`, `duration_s`
- Root height statistics (mean, std, min, max)
- Joint velocity (max, mean)
- Fall detection (height < 0.3m)

### Usage Example
```bash
# Full pipeline from text
python scripts/embodied/batch_t2m_to_embodied.py \
    --prompt-json output/embodied_comparison_v2/motion_text_mapping.json \
    --output-dir output/embodied_comparison_v3/ \
    --max-motions 5 \
    --smooth \
    --render-width 1280 \
    --render-height 960

# From existing NPZ files (skip T2M)
python scripts/embodied/batch_t2m_to_embodied.py \
    --npz-dir work_dirs/.../npz/ \
    --output-dir output/embodied_comparison_v3/
```

---

## 4. `convert_cache_to_json.py` — ProtoMotions Cache to JSON

**File Path:** `scripts/embodied/convert_cache_to_json.py`

**Purpose:** Convert ProtoMotions .pt cache or .motion files to JSON for Three.js web visualization.

### Input Formats (Dual Support)

**Old .pt cache format:**
- `body_pos`: (T, 33, 3) — rigid body positions
- `body_rot`: (T, 33, 4) — rigid body rotations [x, y, z, w]
- `dof_pos`: (T, 29) — joint angles
- `control_dt`: scalar — control timestep
- `num_frames`: scalar

**New .motion format (PyRoki V6 output):**
- `rigid_body_pos`: (T, N_bodies, 3)
- `rigid_body_rot`: (T, N_bodies, 4) [xyzw]
- `dof_pos`: (T, 29)
- `motion_dt` or `fps`: timing info

### Output JSON Structure
```json
{
  "fps": 50,
  "num_frames": 120,
  "joint_names": [
    "left_hip_pitch_joint", "left_hip_roll_joint", ...,
    "right_wrist_yaw_joint"
  ],
  "root_body_index": 0,
  "frames": [
    {
      "root_pos": [0.0, 0.9, 0.0],
      "root_quat": [0.0, 0.0, 0.0, 1.0],
      "dof_pos": [0.1, -0.2, 0.05, ..., -0.1]
    },
    ...
  ]
}
```

### G1 Robot Joint Names (29 DOFs)
```
Left leg (6):   left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll
Right leg (6):  right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
Waist (3):      waist_yaw, waist_roll, waist_pitch
Left arm (7):   left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow, left_wrist_roll, left_wrist_pitch, left_wrist_yaw
Right arm (7):  right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow, right_wrist_roll, right_wrist_pitch, right_wrist_yaw
```

---

## 5. SMPL Mesh Visualization in Web — score_m2m_refine

**Base Path:** `motion_annot_web/score_m2m_refine/`

### SMPL Mesh Assets

**Location:** `motion_annot_web/score_m2m_refine/static/assets/`

| Directory | Model | Vertices | Joints | Notes |
|-----------|-------|----------|--------|-------|
| `dump_smplh/` | SMPL+H (default) | 6890 | 52 | ~1.6GB, main model |
| `dump_smplx/` | SMPL-X | 10475 | 55 | ~2.5GB, includes hand/face joints |
| `dump_smplh_male/` | SMPL+H (male) | 6890 | 52 | Gender-specific |
| `dump_smplh_female/` | SMPL+H (female) | 6890 | 52 | Gender-specific |

**Binary Files per Model Directory:**

```
dump_smplh/
├── v_template.bin             (6890×3 vertices, float32)
├── faces.bin                  (13776 triangles, uint16)
├── skinWeights.bin            (6890×4 skin blend weights, float32)
├── skinIndice.bin             (6890×4 joint influence indices, uint16)
├── j_template.bin             (52×3 joint positions, float32)
├── keypoints.bin              (25 keypoints in COCO order, float32)
├── shapeoffset_0.bin          (6890×3 shape coeff 0 offset, float32)
├── shapeoffset_1.bin
├── ...
├── shapeoffset_15.bin         (16 total shape coefficients)
├── shapeoffset_j_0.bin        (shape offsets for joint positions)
└── ...
```

### Three.js SMPL Loader (`load_smpl.js`)

**File:** `motion_annot_web/score_m2m_refine/static/scripts3d/load_smpl.js` (~12.4KB)

**Main Export:**
```javascript
async function load_smpl_with_shapes(params, gender_param)
```

**Parameters (object format):**
```javascript
{
  shapes: [float, ...],         // 16-element shape coefficients array
  gender: "neutral|male|female", // Default: "neutral"
  poses: [float, ...],          // Optional: SMPL pose parameters
  Rh: [rx, ry, rz],             // Optional: root rotation (axis-angle)
  Th: [tx, ty, tz],             // Optional: root translation
  smpl_type: "smpl|smplh|smplx", // Default: "smplh"
  framerate: int                // Default: 30
}
```

**Rendering Pipeline:**
1. Load binary assets (v_template, faces, skinWeights, skinIndice, j_template)
2. Load 16 shape offset files (shapeoffset_*.bin)
3. Compute deformed vertex positions: `v_shaped = v_template + sum(shape_coeff[i] * offset[i])`
4. Apply skinning weights to blend rigid transformations
5. Create Three.js SkinnedMesh with geometry + skeleton
6. Animate skeleton joints from SMPL pose parameters

### Web Viewer Integration

**SMPL preview template:** `motion_annot_web/score_m2m_refine/templates/vis_smpl_preview.html`

**Usage flow:**
1. Fetch motion JSON with SMPL parameters (Rh, Th, poses, shapes)
2. Create Three.js scene with camera, lighting, ground plane
3. Load SMPL mesh via `load_smpl_with_shapes()`
4. Update skeleton bone rotations each frame from motion JSON poses
5. Render with OrbitControls for interactive viewing

**Color scheme by gender:**
- Neutral: #ffffff (white)
- Male: #6495ED (cornflower blue)
- Female: #FF6B81 (light coral)

### Score M2M Web App Structure

**Flask app:** `motion_annot_web/score_m2m_refine/score_m2m_web.py`

**Routes:**
- `/api/smpl` — API endpoint returning SMPL JSON for motion
- Templates using `load_smpl_with_shapes()`:
  - `record.html` — annotation interface with SMPL preview
  - `view_record.html` — view annotated motion
  - `admin_review.html` — admin review with SMPL mesh
  - `review_record.html` — reviewer interface
  - `vis_smpl_preview.html` — standalone SMPL motion viewer

---

## 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TEXT PROMPTS / EXISTING MOTION_135 NPZ                                       │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
        [Path A: T2M Inference]   [Path B: Existing NPZ]
                │                         │
       batch_t2m_to_embodied.py   (skip T2M)
                │                         │
        ┌───────┴─────────┐              │
        │ HyMotion T2M    │              │
        │ (1.0-Lite)      │              │
        └───────┬─────────┘              │
                │                         │
                │ [motion_135 + smoothing]
                │                         │
                └──────────────┬──────────┘
                               ↓
                    MOTION_135 NPZ (T, 135)
                  [3D transl + 22×rot6d]
                               │
                ┌──────────────┼──────────────┐
                │              │              │
       ┌────────▼────────┐    │         ┌────▼──────────────┐
       │ SMPL MESH PATH  │    │         │ ROBOT MOTION PATH │
       │ (annotation)    │    │         │ (embodied)        │
       └────────┬────────┘    │         └────┬──────────────┘
                │              │              │
    batch_npz_to_smpl_        │     batch_pipeline_to_web.py
    mesh_json.py             │     batch_t2m_to_embodied.py
                │              │              │
                │              │    ┌─────────▼────────┐
                │              │    │ PyRoki Retarget  │
                │              │    │ (V6 trajectory)  │
                │              │    └─────────┬────────┘
                │              │              │
                │              │    ┌─────────▼────────────┐
                │              │    │ ProtoMotions Cache   │
                │              │    │ .motion / .pt file   │
                │              │    └─────────┬────────────┘
                │              │              │
    ┌───────────▼──────┐   │    ┌──────────▼──────────────┐
    │ SMPL JSON        │   │    │ ROBOT JSON              │
    │ {Rh, Th, poses,  │   │    │ {root_pos, root_quat,   │
    │  shapes, gender} │   │    │  dof_pos, joint_names}  │
    └───────────┬──────┘   │    └──────────┬──────────────┘
                │              │              │
                │              │              │
    ┌───────────▼──────┐   │    ┌──────────▼──────────────┐
    │ Web Viewer       │   │    │ Web Viewer              │
    │ (score_m2m)      │   │    │ (embodied comparison)   │
    │                  │   │    │                         │
    │ load_smpl.js     │   │    │ dof-based viewer        │
    │ → SkinnedMesh    │   │    │                         │
    └──────────────────┘   │    └─────────────────────────┘
                           │
                    [Database Storage]
                    Fixed rule manifest
```

---

## 7. Integration Points & API Contracts

### SMPL Mesh JSON Format (for annotation)
```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [{
      "id": 0,
      "gender": "neutral",
      "smpl_type": "smplh",
      "Rh": [[rx, ry, rz]],
      "Th": [[tx, ty, tz]],
      "poses": [[p0, p1, ...]],
      "shapes": [[0, 0, ..., 0]],
      "mocap_framerate": 30
    }],
    ...
  ]
}
```

### Robot Motion JSON Format (for embodied viewer)
```json
{
  "fps": 50,
  "num_frames": 120,
  "joint_names": ["left_hip_pitch_joint", ...],
  "root_body_index": 0,
  "frames": [
    {
      "root_pos": [x, y, z],
      "root_quat": [x, y, z, w],
      "dof_pos": [a0, a1, ..., a28]
    },
    ...
  ]
}
```

### Key File Conversions

| Input | Script | Output | Use Case |
|-------|--------|--------|----------|
| motion_135 NPZ | batch_npz_to_smpl_mesh_json.py | SMPL JSON | Annotation web UI |
| motion_135 NPZ | batch_pipeline_to_web.py | Robot JSON | Embodied comparison |
| .pt cache | convert_cache_to_json.py | Robot JSON | Embodied web viewer |
| SMPL JSON | load_smpl.js | Three.js SkinnedMesh | Browser rendering |
| Robot JSON | (custom viewer) | Three.js 3D skeleton | Browser rendering |

---

## 8. Embodied Web Viewer Status

**Current Implementation:**
- ✅ `batch_t2m_to_embodied.py` generates robot motion JSONs + videos
- ✅ `convert_cache_to_json.py` supports both old .pt and new .motion formats
- ✅ G1 humanoid DOF angles in JSON (29 joints)
- ✅ Rigid body positions/rotations included (33 bodies for collision mesh)
- ✅ Metrics extraction (height, velocity, fall detection)

**Web Viewer for Robot Motion:**
- Not currently in score_m2m_refine (that uses SMPL mesh format)
- Robot motion JSON format is ready, awaiting custom Three.js viewer implementation
- Could integrate with existing Three.js infrastructure in `motion_annot_web/score_m2m_refine/static/three/`

---

## 9. File References

### Python Scripts
- `scripts/embodied/batch_npz_to_smpl_mesh_json.py` — 239 lines
- `scripts/embodied/batch_pipeline_to_web.py` — 296 lines
- `scripts/embodied/batch_t2m_to_embodied.py` — 1008 lines
- `scripts/embodied/convert_cache_to_json.py` — 220 lines
- `scripts/embodied/pipeline_motion_to_robot.py` — 50+ lines (chains 3 PyRoki scripts)

### Web Assets
- `motion_annot_web/score_m2m_refine/static/scripts3d/load_smpl.js` — 12.4KB (SkinnedMesh loader)
- `motion_annot_web/score_m2m_refine/static/assets/dump_smplh/` — ~1.6GB (SMPL+H binaries)
- `motion_annot_web/score_m2m_refine/static/assets/dump_smplx/` — ~2.5GB (SMPL-X binaries)
- `motion_annot_web/score_m2m_refine/templates/vis_smpl_preview.html` — 73KB (SMPL viewer template)

### Supporting Files
- `motion_annot_web/score_m2m_refine/score_m2m_web.py` — Flask app with SMPL endpoints
- `motion_annot_web/score_m2m_refine/templates/record.html`, `view_record.html`, `admin_review.html` — annotation interfaces

---

## 10. Troubleshooting & Common Issues

### SMPL Assets Missing
**Error:** "SMPL 模型资源未部署: v_template.bin 返回 404"
**Solution:** Ensure `motion_annot_web/score_m2m_refine/static/assets/dump_smplh/` exists with binary files

### Invalid Shapes Array
**Error:** "shapes contains NaN/null/undefined values"
**Solution:** Ensure shape coefficients are numeric and padded to 16 elements

### Rot6d Conversion Fails
**Error:** Non-orthogonal matrix after Gram-Schmidt in rot6d_to_axis_angle_np()
**Solution:** Check input rot6d is truly row-major format [R00, R01, R10, R11, R20, R21]

### PyRoki Retarget Timeout
**Error:** "PIPELINE TIMEOUT (>7200s)"
**Solution:** PyRoki CPU optimization takes ~60-70 min per motion; increase timeout or use GPU

---

**Document Generated:** 2026-05-14
**Version:** 1.0 (comprehensive)
