# Embodied Motion Visualization Website Analysis
## Port 8097 Architecture & Data Flow

---

## 1. WEB SERVER SETUP

### Active Server Process
- **Port**: 8097
- **Directory**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4`
- **Server Type**: Python's built-in `http.server` module
- **Command**: `python3 -m http.server 8097`
- **Running PID**: 2239696 (as of May 13)

### How to Start
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4
python3 -m http.server 8097
# Server runs on: http://<hostname>:8097/
```

### Website URL
- **Entry point**: `http://<hostname>:8097/index.html`
- **Serves directory structure as-is** (no routing logic needed)

---

## 2. WEBSITE FRONTEND

### Main HTML File
**Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/index.html`

**Size**: 67.2 KB

**Key Features**:
- Single-page application (SPA) with Three.js for 3D visualization
- **Three synchronized 3D viewers**:
  1. **SMPL Skeleton** (left panel): Original human skeleton from T2M model (22 joints, SMPL format, 30 FPS)
  2. **Reference Robot** (middle panel): Retargeted to Unitree G1 humanoid via GMR (29 DOFs, 50 FPS)
  3. **Physics Tracked** (right panel): ONNX policy tracking in MuJoCo simulation (shows actual robot dynamics)

- **Motion Gallery Sidebar**: Filterable list of 114 generated motions
- **Playback Controls**: Play/pause, frame stepping, seek bar, speed control (0.25x - 2x)
- **Quality Assessment UI**:
  - Stage 1 Filter: Kinematic quality (reference motion validity)
  - Stage 2 Filter: Physics stability (tracking success in MuJoCo)
  - Motion status badges: "Stable" (physics OK), "Fell @frame" (fell), "Bad Gen" (kinematic fail)

### 3D Rendering Details
- **Framework**: Three.js (v0.168.0) via CDN
- **Camera Controls**: OrbitControls with mouse + keyboard shortcuts
- **Mesh Format**: STL files loaded from `/meshes/` subdirectory
- **G1 Robot Model**:
  - 33 bodies, 29 DOFs (MuJoCo compatible)
  - Body hierarchy defined in code (BODY_PARENT array)
  - 65 STL mesh files per body
  - Material colors: Dark (0x333333) and Silver (0xb3b3b3)

### UI Color Scheme
```javascript
--bg: #0f1117           // Dark background
--surface: #1a1d27      // Panel background
--accent: #6c8cff       // Primary blue
--accent2: #4ecdc4      // Teal
--success: #4ecdc4      // Pass indicator
--danger: #ff6b6b       // Fall indicator
--purple: #a78bfa       // SMPL skeleton color
```

---

## 3. DATA LOADING FLOW

### Initial Data Load (async)
```javascript
async loadAllData()
  ↓
  fetch('data/motions/manifest.json')        // Motion list
  fetch('motion_text_mapping.json')          // Text prompts
  fetch('data/tracked_caches/tracker_summary.json')  // Physics results
  ↓
  Build quality classification for each motion
  Sort by: success > fell > badgen
  Render motion gallery
```

### Per-Motion Data Load (on selection)
```javascript
selectMotion(id)
  ↓
  fetch(`data/motions/${id}.json`)           // Reference robot motion
  fetch(`data/tracked_motions/${id}.json`)   // Physics-tracked motion
  fetch(`data/smpl_joints/${id}.json`)       // SMPL skeleton joint positions
  ↓
  Parse JSON frames
  Update all 3 viewer poses
  Build trajectory visualization
  Display physics insights
```

---

## 4. DATA DIRECTORY STRUCTURE

### V4 (Current - 114 motions)
```
output/embodied_t2m_v4/
├── index.html                           # Main website file
├── motion_text_mapping.json             # Text prompts for each motion
├── meshes/                              # 65 STL robot part files
│   ├── pelvis.STL
│   ├── left_hip_pitch_link.STL
│   ├── ... (all G1 robot parts)
│   └── right_rubber_hand.STL
└── data/
    ├── motions/                         # Reference robot JSON (50 FPS, 29 DOFs)
    │   ├── manifest.json                # Motion list with metadata
    │   ├── v4_walk_001.json             # Frame-by-frame robot motion
    │   ├── v4_walk_002.json
    │   └── ... (114 .json files)
    │
    ├── caches/                          # .pt cache files (intermediate)
    │   ├── v4_*.pt                      # ProtoMotions cache (T, 33, 3/4)
    │   └── ... (114 .pt files)
    │
    ├── tracked_motions/                 # Physics-tracked robot JSON (50 FPS)
    │   ├── v4_walk_001.json
    │   └── ... (114 .json files)
    │
    ├── tracked_caches/                  # .pt cache files (physics output)
    │   ├── tracker_summary.json         # Physics simulation results
    │   └── v4_*.pt
    │
    ├── smpl_joints/                     # SMPL skeleton JSON (30 FPS, 22 joints)
    │   ├── v4_walk_001.json
    │   └── ... (114 .json files)
    │
    ├── npz/                             # Original NPZ motion files
    └── meta/                            # Motion metadata
```

### V5 (In Progress - 115 motions)
```
output/embodied_t2m_v5/
├── data/
│   ├── motions/                         # Reference robot JSON
│   ├── caches/                          # .pt cache files
│   ├── tracked_motions/                 # Physics-tracked JSON
│   ├── meta/                            # Motion metadata
│   └── (no smpl_joints or separate tracked_caches)
└── comparison_report.json               # V4 vs V5 comparison
```

---

## 5. JSON DATA FORMATS

### Reference Motion JSON: `data/motions/{id}.json`
```json
{
  "fps": 50.0,
  "num_frames": 199,
  "joint_names": [
    "left_hip_pitch_joint",     // DOF 0
    "left_hip_roll_joint",      // DOF 1
    ...
    "right_wrist_yaw_joint"     // DOF 28
  ],
  "root_body_index": 0,
  "frames": [
    {
      "root_pos": [x, y, z],           // Pelvis position (m)
      "root_quat": [x, y, z, w],       // Pelvis quaternion (xyzw order!)
      "dof_pos": [v0, v1, ..., v28]    // 29 joint angles (radians)
    },
    ...
  ]
}
```
- **Size**: ~200-800 KB per motion (depending on frame count)
- **Frames stored as**: Full precision floats, compact JSON encoding
- **Root**: Pelvis (body index 0 in MuJoCo MJCF)
- **DOF ordering**: Matches `DOF_JOINT_NAMES` array (12 leg DOFs + 3 waist + 8 per arm)

### SMPL Skeleton JSON: `data/smpl_joints/{id}.json`
```json
{
  "fps": 30,
  "num_frames": 120,
  "joint_names": [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", ..., "R_Wrist"
  ],
  "joint_parents": [-1, 0, 0, 0, 1, 2, 3, ...],
  "bones": [[0,1], [0,2], [0,3], ...],
  "frames": [
    {
      "joints": [                      // 22 joint positions
        [x, y, z],                     // Pelvis (Y-up coords)
        [x, y, z],                     // L_Hip
        ...
        [x, y, z]                      // R_Wrist
      ]
    },
    ...
  ]
}
```
- **30 FPS** (vs 50 FPS for robot) → mapped via `robotFrameToSMPLFrame()`
- **Y-up coordinate system** (vs Z-up for MuJoCo)
- **22 SMPL joints** (original T2M output, before retargeting)

### Physics Tracker Summary: `data/tracked_caches/tracker_summary.json`
```json
[
  {
    "id": "v4_walk_002",
    "status": "success",          // or "fell"
    "num_frames": 199,
    "fall_frame": null,           // Frame where fall happened (or null)
    "root_height_min": 0.745,     // Minimum root height during sim
    "duration_s": 3.98,
    "sim_time_s": 0.495,          // Actual simulation time
    "output_path": "output/embodied_t2m_v4/data/tracked_caches/v4_walk_002.pt"
  },
  ...
]
```

### Manifest: `data/motions/manifest.json`
```json
{
  "motions": [
    {
      "id": "v4_walk_002",
      "text": "A person walks forward slowly.",
      "num_frames": 199,
      "fps": 50,
      "duration_s": 3.98,
      "fell": false,                // Stage 2 physics result
      "fall_frame": null,
      "root_height_mean": 0.692,   // Stage 1 kinematic quality
      "max_joint_velocity": 37.03
    },
    ...
  ]
}
```

### Motion Text Mapping: `motion_text_mapping.json`
```json
{
  "motions": [
    {
      "motion_id": "v4_walk_001",
      "text": "A person walks forward.",
      "duration_frames": 90
    },
    ...
  ]
}
```

---

## 6. GENERATION PIPELINE (NPZ → JSON)

### Scripts Path
`/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/scripts/embodied/`

### Main Pipeline Scripts

#### 1. **batch_pipeline_to_web.py**
```bash
python scripts/embodied/batch_pipeline_to_web.py \
  --npz-dir <input_npz_dir> \
  --output-dir output/embodied_t2m_v4/data/motions/ \
  --cache-dir output/embodied_t2m_v4/data/caches/ \
  --skip-existing
```

**Process**:
1. Load NPZ files (SMPL motion_135: T x 22 x 3 positions)
2. **Quality filter**: Reject if body height outside [1.2m, 2.0m] range
3. Run `pipeline_motion_to_robot.py` on each → ProtoMotions .pt cache
4. Convert cache → JSON via `convert_cache_to_json()`
5. Write manifest.json

**Output**: 
- JSON files in `data/motions/`
- .pt caches in `data/caches/`
- manifest.json with motion metadata

#### 2. **convert_cache_to_json.py**
```python
convert_cache_to_json(cache_path, output_path, subsample=1)
```

**Process**:
1. Load .pt cache file (PyTorch)
   - `dof_pos`: (T, 29) joint angles
   - `body_pos`: (T, 33, 3) body positions
   - `body_rot`: (T, 33, 4) quaternions (xyzw)
   - `control_dt`: time step (typically 0.02 → 50 FPS)

2. Extract root (body 0 = pelvis) and DOF data
3. Format as JSON frames:
   ```
   "root_pos": body_pos[t, 0],
   "root_quat": body_rot[t, 0],
   "dof_pos": dof_pos[t]
   ```
4. Write compact JSON (no pretty-print)

**Output**: ~200-800 KB JSON per motion

#### 3. **pipeline_motion_to_robot.py**
Converts SMPL 22-joint skeleton to G1 robot via GMR retargeting.

#### 4. **run_tracker_export.py**
Runs ONNX policy in MuJoCo simulator to generate physics-tracked motions.

```bash
python scripts/embodied/run_tracker_export.py \
  --motion-dir output/embodied_t2m_v4/data/caches/ \
  --output-dir output/embodied_t2m_v4/data/tracked_caches/ \
  --pattern 'v4_*.pt'
```

**Process**:
1. Load reference .pt cache
2. Run MuJoCo sim with ONNX tracking policy
3. Export tracked trajectory → tracked .pt cache
4. Write tracker_summary.json with fall info

#### 5. **batch_npz_to_smpl_joints.py**
Extracts SMPL joint positions from NPZ files → JSON.

---

## 7. MESH ASSETS

### STL File Location
`/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/output/embodied_t2m_v4/meshes/`

### Body Mesh Mapping (from index.html)
```javascript
BODY_MESH_CONFIG = [
  /* 0  pelvis */              {meshes: [{file:'pelvis.STL', color:0x333333}, ...]},
  /* 1  head (virtual) */      null,
  /* 2  left_hip_pitch_link */ {meshes: [{file:'left_hip_pitch_link.STL', color:0x333333}]},
  /* 3  left_hip_roll_link */  {meshes: [{file:'left_hip_roll_link.STL', color:0xb3b3b3}]},
  ...
  /* 32 right_rubber_hand */   null,
];
```

### Total Files: 65 STL files (for 33 bodies, some null, some with multiple parts)

### Loading Flow
```javascript
preloadAllMeshes()
  ↓
  Find all unique .stl filenames from BODY_MESH_CONFIG
  ↓
  Load each via THREE.STLLoader
  ↓
  Cache geometries in _geometryCache{}
  ↓
  Create viewers and attach meshes to body groups
```

### Mesh Material
- **Rendering**: THREE.MeshPhongMaterial (3D look)
- **Colors**: 0x333333 (dark) or 0xb3b3b3 (silver)
- **Special**: Physics-tracked robot gets slight green tint (0x2d8877 blend)
- **Shadows**: Enabled for depth perception

---

## 8. THREE.JS SKELETON HIERARCHY

### Coordinate Systems
- **MuJoCo**: Z-up (0, 0, 1)
- **Three.js**: Y-up (0, 1, 0)
- **Conversion**: `coordAdapter.rotation.x = -π/2` applies 90° X-axis rotation

### Forward Kinematics (FK)
```javascript
setPose(viewer, frame)
  1. Set root position: bodyGroups[0].position = frame.root_pos
  2. Set root rotation: bodyGroups[0].quaternion = frame.root_quat (xyzw!)
  3. For each body i>0:
     - Apply body frame quaternion (offset rotation from MJCF)
     - Apply DOF joint rotation: Q_final = Q_bodyFrame * exp(ω * θ)
  4. Render scene
```

### Body Hierarchy
- **Root**: Pelvis (body 0)
- **Chain**: Body → Parent → ... → Root
- **DOF mapping**: `DOF_TO_BODY[d]` gives body index for DOF d

---

## 9. PLAYBACK & SYNCHRONIZATION

### Frame Timing
```javascript
animate(now)
  ↓
  if isPlaying:
    dt = (now - lastFrameTime) / 1000
    framesToAdvance = dt * FPS * playSpeed
    currentFrame += floor(framesToAdvance)
  ↓
  setPose(refViewer, data[currentFrame])
  setPose(trackedViewer, data[currentFrame])
  setSMPLPose(smplViewer, data[smplFrameIndex])  // FPS mapping
  ↓
  Update progress bar, frame info, fall indicator
  ↓
  requestAnimationFrame(animate)  // 60 Hz refresh
```

### Frame Rate Mapping (SMPL vs Robot)
```javascript
robotFrameToSMPLFrame(robotFrame, smplData):
  time = robotFrame / 50  // 50 Hz robot
  smplFrame = floor(time * 30)  // 30 Hz SMPL
  return min(smplFrame, smplData.num_frames - 1)
```

### Camera Synchronization
```javascript
syncCameras = true  →  All 3 viewers mirror reference camera offset
```

---

## 10. QUALITY CLASSIFICATION LOGIC

### Stage 1: Kinematic Filtering
```javascript
"badgen" criteria:
  - Reference motion height mean < 0.5 m (crouched/underground)
  - Reference motion marked as "fell" in manifest
```

### Stage 2: Physics Testing
```javascript
"success" criteria:
  - Passed Stage 1 kinematic check
  - Physics tracker status = "success" (didn't fall)

"fell" criteria:
  - Passed Stage 1 kinematic check
  - Physics tracker status = "fell" (fell at some frame)
  - Marked with fall_frame from tracker_summary.json
```

### Motion Gallery Filters
- **"Quality"** (default): Shows passed Stage 1 (excludes badgen)
- **"Stable"**: Shows "success" status
- **"Fell"**: Shows "fell" status
- **"Bad Gen"**: Shows "badgen" status
- **"All"**: Shows everything

---

## 11. PHYSICS INSIGHTS GENERATOR

### Display Logic
```javascript
getPhysicsInsight(id, trackerInfo)
  ↓
  if badgen:
    "Stage 1 filtered: kinematically implausible pose (crouched/underground)"
  ↓
  if success && root_height_min > 0.7:
    "Excellent stability — root height never dropped below 0.7m"
  ↓
  if fell && fall_frame == 0:
    "Immediate fall at frame 0 — initially unstable"
  ↓
  if fell && fall_frame < 20:
    "Early fall — abrupt transition or unrealistic acceleration"
  ↓
  if fell && fall_frame < 100:
    "Mid-sequence fall — dynamic quality issue"
  ↓
  else:
    "Late fall — challenging dynamic transition"
```

### Display Location
- Appears in insight panel (bottom-left of tracked viewer) when motion selected
- Shows quantitative fall frame + qualitative interpretation

---

## 12. KEYBOARD SHORTCUTS & CONTROLS

| Key | Action |
|-----|--------|
| **Space** | Play/Pause |
| **←** | Previous frame |
| **→** | Next frame |
| **N** | Next motion (filtered list) |
| **P** | Previous motion |
| **R** | Reset camera (in viewer) |
| **G** | Toggle grid |
| **Mouse** | Orbit camera (drag), Zoom (scroll) |

### UI Buttons
- **Play/Pause**: Toggle playback
- **←/→**: Frame step
- **Speed**: 0.25x, 0.5x, 1x, 2x playback
- **Loop**: Repeat at end (default: on)
- **Follow**: Auto-track motion (default: on)
- **Sync Cam**: Sync all 3 cameras (default: on)

---

## 13. LOADING & PERFORMANCE NOTES

### Initial Load Time
1. Download index.html (~67 KB)
2. Download Three.js library (CDN, ~200 KB)
3. Preload 65 STL meshes (in parallel, ~30-50 MB total)
   - Async loading with progress spinner
4. Download manifest.json (~50 KB)
5. Download text mapping (~150 KB)
6. Download tracker summary (~100 KB)
7. **Total**: ~30-50 MB for assets + metadata

### Per-Motion Load Time
1. Download reference motion JSON (~200-800 KB)
2. Download tracked motion JSON (~200-800 KB)
3. Download SMPL JSON (~80-300 KB)
4. **Total**: ~500 KB - 2 MB per motion
5. **Time**: ~500ms - 2s on decent network

### Caching
- STL geometries cached in memory after first load
- Motion data NOT cached (re-fetches on each selection)
- IndexedDB NOT used

---

## 14. COMPARISON WITH V5

### V5 Differences
1. **No separate SMPL viewer**: SMPL data may not be pre-generated
2. **No smpl_joints/ directory**: Would break current visualization
3. **Merged tracked data**: May combine reference + tracked into single JSON
4. **Additional metadata**: comparison_report.json available

### To Enable V5 on Port 8097
```bash
# Current: V4
# cd output/embodied_t2m_v4 && python3 -m http.server 8097

# To switch to V5:
# cd output/embodied_t2m_v5 && python3 -m http.server 8097
# BUT: Will fail because V5 is missing index.html and smpl_joints/
```

### Required for V5 Support
1. Copy `index.html` to V5 output directory
2. Generate SMPL JSON files for V5 motions
3. OR modify HTML to make SMPL viewer optional

---

## 15. HOW THE WEBSITE READS DATA

### Request Flow (Browser → Server)
```
Browser (http://hostname:8097/)
  ↓
HTTP Server (python3 -m http.server 8097)
  ├── GET /index.html → Serve HTML
  ├── GET /motion_text_mapping.json → Serve JSON
  ├── GET /data/motions/manifest.json → Serve JSON
  ├── GET /data/tracked_caches/tracker_summary.json → Serve JSON
  ├── GET /meshes/pelvis.STL → Serve binary
  ├── GET /data/motions/v4_walk_001.json → Serve JSON
  ├── GET /data/tracked_motions/v4_walk_001.json → Serve JSON
  └── GET /data/smpl_joints/v4_walk_001.json → Serve JSON
```

### No Backend Processing
- Pure static file serving
- All logic runs in browser (JavaScript)
- No API endpoints or database queries
- No authentication or access control

### Client-Side Processing
1. Parse JSON motion frames
2. Compute forward kinematics
3. Render with Three.js
4. Handle playback timing
5. Track user interactions

---

## 16. FILE SIZE STATISTICS

### V4 Complete Directory
```
embodied_t2m_v4/
├── index.html                    67 KB
├── motion_text_mapping.json     150 KB
├── meshes/                     ~30 MB (65 STL files)
├── data/
│   ├── motions/               ~1.3 GB (114 JSON files)
│   ├── caches/                ~48 MB (114 .pt files)
│   ├── tracked_motions/       ~1.3 GB (114 JSON files)
│   ├── tracked_caches/        ~48 MB (114 .pt files)
│   ├── smpl_joints/           ~1.2 GB (114 JSON files)
│   └── meta/                  ~91 MB (metadata)
└── TOTAL:                     ~2.3 GB
```

### Per-Motion Average
- Motion JSON: ~12 MB
- SMPL JSON: ~10 MB
- .pt cache: ~400 KB
- Together: ~22 MB per motion × 114 = 2.5 GB

---

## 17. DEPLOYMENT CHECKLIST

To set up embodied visualization website:

✅ 1. Generate motion data via `batch_pipeline_to_web.py`
✅ 2. Create reference robot JSON in `data/motions/`
✅ 3. Create manifest.json with motion metadata
✅ 4. Run physics tracker via `run_tracker_export.py`
✅ 5. Generate SMPL JSON from `batch_npz_to_smpl_joints.py`
✅ 6. Copy STL meshes to `meshes/` subdirectory
✅ 7. Copy index.html to output root
✅ 8. Copy motion_text_mapping.json to root
✅ 9. Start HTTP server: `python3 -m http.server 8097`
✅ 10. Access: `http://hostname:8097/index.html`

---

## 18. KNOWN ISSUES & LIMITATIONS

1. **Mesh Loading**: If meshes not found, ghost skeleton appears (groups only)
2. **FPS Mismatch**: SMPL (30 Hz) vs Robot (50 Hz) causes frame skipping
3. **No Fallback**: V5 setup will fail if SMPL data missing
4. **No Compression**: Full JSON stored (could use binary format + gzip)
5. **Single Server**: Only one version (V4 or V5) can run at a time
6. **No Authentication**: Anyone with network access can view

---

## SUMMARY

The embodied motion visualization website on **port 8097** is a **static file server** (`python3 -m http.server 8097`) serving:

1. **Frontend**: Single HTML page with Three.js 3D visualization
2. **Data**: Pre-computed JSON motion files + STL robot meshes
3. **Flow**: Browser downloads HTML → loads motion list → user selects motion → fetches 3 JSON files → renders in 3D

**Key components**:
- `/apdcephfs/.../output/embodied_t2m_v4/index.html` - UI & JavaScript logic
- `/apdcephfs/.../output/embodied_t2m_v4/data/motions/*.json` - Reference robot motions
- `/apdcephfs/.../output/embodied_t2m_v4/data/tracked_motions/*.json` - Physics-simulated motions  
- `/apdcephfs/.../output/embodied_t2m_v4/data/smpl_joints/*.json` - Original human skeleton
- `/apdcephfs/.../output/embodied_t2m_v4/meshes/*.STL` - G1 robot geometry

**Data generation pipeline**:
NPZ → `pipeline_motion_to_robot.py` → .pt cache → `convert_cache_to_json.py` → JSON

**Serving**:
```bash
cd output/embodied_t2m_v4
python3 -m http.server 8097
# http://hostname:8097/
```

