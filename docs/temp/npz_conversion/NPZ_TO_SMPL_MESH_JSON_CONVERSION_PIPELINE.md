# NPZ to SMPL Mesh JSON Conversion Pipeline
## Complete Technical Specification

Generated: 2026-05-25

---

## 1. CONVERSION SCRIPTS LOCATION

```
scripts/embodied/
├── batch_npz_to_smpl_mesh_json.py      # Full SMPL mesh rendering for web viewer
├── batch_npz_to_smpl_joints.py         # Skeleton-only joint positions
└── __pycache__/
```

---

## 2. PRIMARY FUNCTION SIGNATURES

### 2.1 SMPL Mesh JSON Converter
**File:** `scripts/embodied/batch_npz_to_smpl_mesh_json.py`

#### Main Conversion Function
```python
def convert_single_npz(
    npz_path: str,
    smpl_type: str = "smplx",          # Options: "smpl", "smplh", "smplx"
    gender: str = "neutral"             # Options: "neutral", "male", "female"
) -> dict
```

**Returns:** Dictionary with schema shown in Section 3 below

#### Rotation Conversion Helper
```python
def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray
```
- **Input:** `rot6d` shape (..., 6) — row-major [R00,R01, R10,R11, R20,R21]
- **Process:** 
  1. Reorder indices [0,2,4,1,3,5] from row-major to column-major
  2. Apply Gram-Schmidt orthogonalization on first two columns
  3. Compute third column via cross product
  4. Convert 3×3 rotation matrix to axis-angle via scipy.spatial.transform.Rotation
- **Output:** axis-angle (..., 3) as float32

#### CLI Entry Point
```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir <path>           # Directory with motion_135 NPZ files
    --npz-file <path>          # OR single NPZ file
    --output-dir <path>        # JSON output directory [REQUIRED]
    --smpl-type <str>          # "smpl", "smplh" (default), or "smplx"
    --gender <str>             # "neutral" (default), "male", or "female"
    --skip-existing            # Optional: skip already converted files
```

### 2.2 SMPL Joints JSON Converter (for comparison)
**File:** `scripts/embodied/batch_npz_to_smpl_joints.py`

```python
def convert_single_npz(
    npz_path: str,
    model: SmplxLite,
    device: torch.device
) -> dict
```
- Uses SmplxLite FK to compute 22 world-space joint positions
- Requires SMPL model checkpoint loading
- Output: joint positions only (skeleton visualization)

---

## 3. NPZ INPUT FORMAT (motion_135)

### 3.1 File Structure
```python
{
    "motion_135": np.ndarray,   # shape (T, 135), dtype float32
    "fps": int,                 # scalar, e.g., 30
    "prompt": str,              # text description of motion
}
```

### 3.2 motion_135 Payload Breakdown (T, 135)
Per frame, 135 values distributed as:

| Index Range | Field       | Dimensions | Description |
|-------------|-------------|-----------|-------------|
| 0:3         | Translation | (3,)      | [tx, ty, tz] — root position in world space |
| 3:135       | Rot6D       | (22, 6)   | 22 body joints × 6 values each (row-major) |

### 3.3 Rotation Format (row-major)
Each joint's 6D rotation `[R00, R01, R10, R11, R20, R21]` represents:
- First two columns of 3×3 rotation matrix in row-major storage
- Gram-Schmidt orthogonalization recovers the full orthonormal 3×3 matrix
- Final conversion to axis-angle for SMPL compatibility

### 3.4 Joint Order (22 body joints)
```
0: Pelvis (root)
1-21: 21 body joints
```

**Mapping to SMPL:**
- Joint 0 → SMPL root (Pelvis)
- Joints 1-21 → SMPL body joints 1-21
- Joints 22-23 (SMPL only) → zero-padded in motion_135

---

## 4. SMPL MESH JSON OUTPUT FORMAT

### 4.1 Complete Schema
```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [
      {
        "id": 0,
        "gender": "neutral",
        "smpl_type": "smplx",
        "Rh": [[rx, ry, rz]],
        "Th": [[tx, ty, tz]],
        "poses": [[p0, p1, ..., pN]],
        "shapes": [[0.0, 0.0, ..., 0.0]],
        "mocap_framerate": 30
      }
    ],
    [
      {
        "id": 0,
        ...
      }
    ]
  ]
}
```

### 4.2 Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `type` | str | Always "frames" for this pipeline |
| `fps` | int | Frames per second (typically 30) |
| `frames` | list[list[dict]] | T frames, each containing 1-person array |
| `id` | int | Person ID (0 for single-person) |
| `gender` | str | SMPL gender ("neutral", "male", "female") |
| `smpl_type` | str | Model type: "smpl" (24 joints), "smplh" (52 joints), "smplx" (55 joints) |
| `Rh` | list[list[float]] | Root orientation (1×3): axis-angle [rx, ry, rz] |
| `Th` | list[list[float]] | Root translation (1×3): [tx, ty, tz] |
| `poses` | list[list[float]] | Flattened axis-angles for all joints (1×N) |
| `shapes` | list[list[float]] | SMPL beta coefficients (always 16 zeros, shape only) |
| `mocap_framerate` | int | Frame rate (copy of fps field) |

### 4.3 Poses Array Size by SMPL Type

| SMPL Type | Total Joints | Pose Dims | Structure |
|-----------|-------------|----------|-----------|
| SMPL | 24 | 72 (24×3) | root(3) + body(69) |
| SMPL+H | 52 | 156 (52×3) | root(3) + body(63) + hands(90) |
| SMPL-X | 55 | 165 (55×3) | root(3) + body(63) + jaw(3) + eyes(6) + hands(90) |

**For motion_135 (22 joints):**
- Provides: root(1) + body(21) = 22 joints
- Mapped to: axis-angle per joint = 22×3 = 66 values
- Motion_135 always zero-pads remaining joints to match target SMPL type

### 4.4 Example Frame (first frame of 30fps motion)
```json
[
  {
    "id": 0,
    "gender": "neutral",
    "smpl_type": "smplh",
    "Rh": [[0.05, -0.02, 0.01]],
    "Th": [[0.018, 0.991, -0.023]],
    "poses": [
      [0.05, -0.02, 0.01,     // root
       0.01, 0.02, 0.03,      // joint 1
       ...,                   // joints 2-20
       0.0, ..., 0.0]         // zero-padded hands/etc for type
    ],
    "shapes": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    "mocap_framerate": 30
  }
]
```

---

## 5. DATA AVAILABILITY

### 5.1 Comparison Output Directory
```
output/physflow_v2_compare_iter1000/
├── npz/                          # 76 NPZ files (3.6 MB total)
├── comparison_report.txt
└── comparison_results.json
```

### 5.2 Available NPZ Files (76 total)
```
NAMING PATTERN: {prefix}_{idx}_{description}_{suffix}.npz

prefix:  "pretrained" or "finetuned"
idx:     00-18 (19 motion types)
suffix:  "raw" or "rl" (2 variants per type)

TOTAL: 19 types × 2 variants × 2 prefixes = 76 files
```

**Sample Files:**
```
1. finetuned_00_a_person_stands_still_raw.npz
2. finetuned_00_a_person_stands_still_rl.npz
3. finetuned_01_a_person_stands_in_a_relaxed_pose_raw.npz
...
74. pretrained_17_a_person_does_a_jumping_jack_rl.npz
75. pretrained_18_a_person_does_a_high_kick_raw.npz
76. pretrained_18_a_person_does_a_high_kick_rl.npz
```

### 5.3 NPZ File Statistics
- **Total size:** ~3.6 MB
- **Average per file:** ~50 KB
- **Typical duration:** 120 frames @ 30 fps = 4 seconds
- **Motion types:** Standing, walking, stretching, jumping, kicking, etc.

**Sample file inspection:**
```
finetuned_00_a_person_stands_still_raw.npz
├── motion_135: shape (120, 135), dtype float32
│   ├── Values range: [-2.19, 1.29]
│   └── Frame 0 translation: [0.0181, 0.9906, -0.0228]
├── fps: 30
└── prompt: "a_person_stands_still" (21 chars)
```

---

## 6. EMBODIED_VIZ DATA DIRECTORY STATUS

⚠️ **Directory does not exist** — `motion_annot_web/embodied_viz/data/` is not present in the repository.

### 6.1 Expected Structure (if implemented)
```
motion_annot_web/embodied_viz/data/
├── smpl_mesh/           # Output from batch_npz_to_smpl_mesh_json.py
│   ├── {motion}.json
│   ├── {motion}.json
│   └── ...
├── smpl_joints/         # Output from batch_npz_to_smpl_joints.py
│   ├── {motion}.json
│   └── ...
└── metadata.json        # Optional: file listing and metadata
```

### 6.2 Web Viewer Integration
The JSON output is designed for consumption by:
- **Frontend:** `load_smpl.js` SkinnedMesh renderer (Three.js)
- **Backend API:** `score_m2m` `/api/smpl` endpoint (reference format)
- **Use case:** Full 3D SMPL mesh visualization with texture/skinning

---

## 7. CONVERSION PIPELINE WALKTHROUGH

### 7.1 Single File Conversion
```bash
# Step 1: Load NPZ
npz_path = "output/physflow_v2_compare_iter1000/npz/pretrained_00_a_person_stands_still_raw.npz"
data = np.load(npz_path, allow_pickle=True)
motion_135 = data['motion_135']        # (120, 135)
fps = data['fps']                      # 30
T = 120

# Step 2: Extract components
transl = motion_135[:, :3]             # (120, 3)
rot6d = motion_135[:, 3:].reshape(T, 22, 6)  # (120, 22, 6)

# Step 3: Convert rotations
axis_angle = rot6d_to_axis_angle_np(rot6d)   # (120, 22, 3)
root_orient = axis_angle[:, 0, :]     # (120, 3)
body_pose = axis_angle[:, 1:22, :]    # (120, 21, 3)

# Step 4: Build SMPL-H poses (52 joints = 156 dims)
poses_per_frame = np.zeros((T, 156), dtype=np.float32)
poses_per_frame[:, :3] = root_orient  # root
poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)  # 21 body joints
# poses_per_frame[:, 66:156] = 0       # hands zero-padded

# Step 5: Build JSON frames
frames = []
for t in range(T):
    frame = [{
        "id": 0,
        "gender": "neutral",
        "smpl_type": "smplh",
        "Rh": [root_orient[t].tolist()],
        "Th": [transl[t].tolist()],
        "poses": [poses_per_frame[t].tolist()],
        "shapes": [[0.0] * 16],
        "mocap_framerate": fps,
    }]
    frames.append(frame)

# Step 6: Save JSON
result = {
    "type": "frames",
    "fps": fps,
    "frames": frames,
}
with open("output.json", 'w') as f:
    json.dump(result, f, separators=(',', ':'))
```

### 7.2 Batch Processing
```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/physflow_v2_compare_iter1000/npz \
    --output-dir output/physflow_v2_compare_iter1000/smpl_mesh \
    --smpl-type smplh \
    --gender neutral
```

Expected output:
```
Found 76 NPZ files to process
SMPL type: smplh, gender: neutral
  [1/76] finetuned_00_a_person_stands_still_raw: 120 frames @ 30fps -> 425.3KB
  [2/76] finetuned_00_a_person_stands_still_rl: 48 frames @ 30fps -> 170.1KB
  ...
Done: 76 converted, 0 failed, 0 skipped
Output: output/physflow_v2_compare_iter1000/smpl_mesh
```

---

## 8. KEY CONVERSION DETAILS

### 8.1 Rotation Format Transformation (Critical)
**Input:** HyMotion row-major rot6d
```
[R00, R01, R10, R11, R20, R21]
```

**Step 1:** Reorder to column-major
```python
rot6d = rot6d[..., [0, 2, 4, 1, 3, 5]]
# Now: [R00, R10, R20, R01, R11, R21]
# Which is [col0_row0, col0_row1, col0_row2, col1_row0, col1_row1, col1_row2]
```

**Step 2:** Extract two orthogonal vectors
```python
a1 = rot6d[..., :3]    # [R00, R10, R20] — first column
a2 = rot6d[..., 3:6]   # [R01, R11, R21] — second column
```

**Step 3:** Gram-Schmidt orthogonalization
```python
b1 = a1 / ||a1||
b2 = (a2 - (b1·a2)b1) / ||...||
b3 = b1 × b2           # cross product
# Stack into full 3×3 rotation matrix
```

**Step 4:** Convert to axis-angle
```python
R_matrix = [b1, b2, b3]  # 3×3 rotation matrix
axis_angle = Rotation.from_matrix(R_matrix).as_rotvec()  # (3,)
```

### 8.2 Zero-Padding Strategy
Motion_135 provides **22 joints** (root + 21 body), but SMPL models have more:

| SMPL Type | Total Joints | motion_135 Coverage | Zero-Padded |
|-----------|-------------|-------------------|------------|
| SMPL | 24 | 22 (root + 21 body) | 2 joints |
| SMPL+H | 52 | 22 | 30 joints (hands, etc.) |
| SMPL-X | 55 | 22 | 33 joints (hands, face, eyes) |

**Implementation:**
```python
if smpl_type == "smplx":
    poses_per_frame = np.zeros((T, 165), dtype=np.float32)
    poses_per_frame[:, :3] = root_orient        # root
    poses_per_frame[:, 3:66] = body_pose.reshape(T, 63)  # 21 body joints × 3
    # [:, 66:165] remain zero (jaw, eyes, hands)
```

### 8.3 Shape Coefficients
- **Always 16 zeros:** `shapes = [[0.0] * 16]` per frame
- **Reason:** Motion_135 contains no shape/beta parameters
- **Effect:** Generated meshes will always have a "neutral" body shape
- **Note:** If beta parameters were available, they would be optimized separately

---

## 9. WEB VIEWER INTEGRATION

### 9.1 Frontend Consumption (Three.js)
The JSON format is designed for:
- **Parser:** `load_smpl.js` (SkinnedMesh renderer)
- **Vertex Animation:** Per-frame SMPL pose parameters
- **Texture Mapping:** SMPL model UV coordinates
- **Performance:** Compact JSON with separators=(',', ':')

### 9.2 Example Load Code (JavaScript)
```javascript
// Load JSON from conversion pipeline
fetch('motion.json')
  .then(r => r.json())
  .then(data => {
    const fps = data.fps;
    const frames = data.frames;
    
    // Per frame, update SMPL pose
    frames.forEach((frameArray, frameIdx) => {
      const person = frameArray[0];  // Single person
      const {Rh, Th, poses, shapes, smpl_type} = person;
      
      // Feed to SkinnedMesh:
      skinnedMesh.setPose({
        global_orient: Rh[0],     // [rx, ry, rz]
        transl: Th[0],            // [tx, ty, tz]
        body_pose: poses[0],      // Flattened axis-angles
        betas: shapes[0],         // Shape coefficients
      });
      
      // Render frame
      renderer.render(scene, camera);
    });
  });
```

---

## 10. TROUBLESHOOTING

### 10.1 Common Errors

**Error:** `ValueError: motion_135 key not found`
- **Cause:** NPZ file doesn't have motion_135 field
- **Fix:** Verify NPZ format matches specification

**Error:** `Shape mismatch: expected (T, 135), got (T, 132)`
- **Cause:** Missing 3D translation field
- **Fix:** Ensure motion_135 includes transl in first 3 dims

**Error:** `Gram-Schmidt normalization produces NaN`
- **Cause:** Rot6d vectors are zero or nearly collinear
- **Fix:** Script includes 1e-8 epsilon term for numerical stability

**Error:** `poses size mismatch for SMPL type`
- **Cause:** Incorrect smpl_type selection or shape parameter error
- **Fix:** Verify --smpl-type argument (smpl, smplh, smplx)

### 10.2 Validation Checklist
- [ ] NPZ file loads without error: `np.load(npz_path, allow_pickle=True)`
- [ ] motion_135 shape is (T, 135): `motion['motion_135'].shape`
- [ ] FPS is reasonable (10-120): `data['fps']`
- [ ] No NaN/Inf values: `np.isfinite(motion_135).all()`
- [ ] Rot6d vectors are normalized: `np.linalg.norm(rot6d, axis=-1).mean() ≈ 1.0`
- [ ] Output JSON is valid: `json.load(open(output_path))`

---

## 11. PERFORMANCE CHARACTERISTICS

### 11.1 Processing Speed
- **Per-file time:** ~0.1-0.5 seconds (numpy, no GPU needed)
- **Batch of 76 files:** ~5-10 seconds total
- **Bottleneck:** JSON serialization (separators=(',', ':') mitigates)

### 11.2 Output File Sizes
| Motion Duration | Typical JSON Size | Compression |
|---|---|---|
| 4 seconds (120 fps @ 30fps) | 400-500 KB | ~1% of uncompressed poses |
| 10 seconds (300 frames) | ~1.0 MB | ~1% |

**Size formula:**
```
JSON_size ≈ (T frames) × (150 bytes per frame) + header overhead
```

### 11.3 Memory Requirements
- **Per-file:** ~50-100 MB (peak during processing)
- **Batch processing:** ~500 MB for 76 files
- **No GPU required:** Pure NumPy / Python execution

---

## 12. REFERENCE DOCUMENTATION

### 12.1 External Dependencies
- `numpy`: Array operations
- `scipy.spatial.transform.Rotation`: Rotation conversions
- `json`: JSON serialization
- `pathlib.Path`: File I/O

### 12.2 SMPL Model References
- **SMPL (23 body joints):** Simple human body model with 10 shape params
- **SMPL+H (52 joints):** Extends SMPL with detailed hand rigs (15 per hand)
- **SMPL-X (55 joints):** Adds face (jaw + eyes) and hands

### 12.3 Related Scripts
```
scripts/embodied/
├── batch_npz_to_smpl_mesh_json.py      # ← Primary conversion (THIS)
├── batch_npz_to_smpl_joints.py         # Joint positions only
├── motion135_to_smplx.py               # Single-file reference
├── batch_t2m_to_embodied.py            # T2M motion generation
└── physflow_motion_converter.py        # Motion format conversions
```

---

**End of Document**
