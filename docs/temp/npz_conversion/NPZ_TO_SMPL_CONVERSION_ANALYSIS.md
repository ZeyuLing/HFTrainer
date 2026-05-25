# NPZ to SMPL Mesh JSON Conversion Pipeline - Complete Analysis

**Generated:** 2026-05-25  
**Analysis Scope:** Complete conversion pipeline for 3D web viewer visualization

---

## 1. CONVERSION FUNCTION SIGNATURES

### Primary Function: `convert_single_npz()`

**Location:** `scripts/embodied/batch_npz_to_smpl_mesh_json.py` (lines 74-165)

```python
def convert_single_npz(npz_path: str, smpl_type: str = "smplx",
                        gender: str = "neutral") -> dict:
    """Convert a single motion_135 NPZ to SMPL mesh JSON format.
    
    The motion_135 format: [transl(3) + 22*rot6d(132)] per frame.
    SMPL-X uses 55 joints (22 body + 3 face + 30 hands), but motion_135
    only has 22 body joints. We zero-pad the rest.
    
    Returns:
        dict with keys: type, fps, frames
    """
```

**Parameters:**
- `npz_path: str` - Path to input NPZ file
- `smpl_type: str` - SMPL model variant: `"smplx"` (default), `"smplh"`, or `"smpl"`
- `gender: str` - Character gender: `"neutral"` (default), `"male"`, or `"female"`

**Return Type:** `dict` - JSON-serializable dictionary with fields:
- `type: str` - Always `"frames"`
- `fps: int` - Framerate (typically 30)
- `frames: list` - Array of frame arrays (see output format below)

### Rot6D to Axis-Angle Converter: `rot6d_to_axis_angle_np()`

**Location:** Lines 45-71

```python
def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major rot6d (..., 6) to axis-angle (..., 3).
    
    HyMotion stores rot6d in row-major: [R00,R01, R10,R11, R20,R21].
    Must reorder [0,2,4,1,3,5] to column-major before Gram-Schmidt.
    """
```

**Process:**
1. Reorder row-major rot6d to column-major: `[0,2,4,1,3,5]`
2. Apply Gram-Schmidt orthogonalization
3. Compute cross product for third column
4. Convert 3×3 rotation matrix → axis-angle using scipy `Rotation.from_matrix().as_rotvec()`
5. Return shape `(..., 3)` as `float32`

---

## 2. INPUT NPZ FORMAT (motion_135)

### NPZ File Structure

**Keys Required:**
- `motion_135` - Main motion data (type: `np.ndarray`)
- `fps` - Frame rate (type: `int` or `np.int64`)
- `prompt` - Text description (type: `str`)

### motion_135 Array Format

**Shape:** `(T, 135)` where `T` = number of frames

**Content Layout (135 values per frame):**
```
Bytes/Values 0-2:     Translation (transl)  → 3 floats [tx, ty, tz]
Bytes/Values 3-134:   22 Joint Rotations    → 22 × 6 floats each
                      (rot6d per joint)
```

**Joint Structure (22 joints total):**
- Joint 0: Root (Pelvis)
- Joints 1-21: Body joints (21 body joints)
  - Spine hierarchy
  - Left/right arms
  - Left/right legs

**Rot6D Encoding (per joint):**
- 6 floats per joint encoding first two columns of 3×3 rotation matrix
- Row-major order: `[R00, R01, R10, R11, R20, R21]`
- Requires orthogonalization before use (see `rot6d_to_axis_angle_np()`)

### Example Data

**File:** `output/physflow_v2_compare_iter1000/npz/pretrained_00_a_person_stands_still_raw.npz`

```
Keys: ['motion_135', 'fps', 'prompt']
motion_135 shape: (120, 135)  # 120 frames, 135 values per frame
fps: 30
prompt: "a person stands still"
```

**First frame sample (motion_135[0]):**
```
[0.0181, 0.9906, -0.0228,  # Translation: (0.0181, 0.9906, -0.0228)
 0.9993, -0.0053, 0.0038,   # Joint 0 rot6d (row-major)
 0.9994, -0.0362, -0.0347,  # Joint 1 rot6d
 0.9997, 0.0146, -0.0129,   # Joint 2 rot6d
 ...                         # Joints 3-21 rot6d (18 more sets)
]
```

---

## 3. OUTPUT JSON FORMAT (SMPL Mesh JSON)

### Top-Level Structure

```json
{
  "type": "frames",
  "fps": 30,
  "frames": [
    [frame_0_person_0],
    [frame_1_person_0],
    ...
    [frame_T_person_0]
  ]
}
```

### Frame Structure

Each element in `frames` is an **array of person objects** (typically 1 person):

```json
[{
  "id": 0,                          // Person ID in scene
  "gender": "neutral",              // "neutral" | "male" | "female"
  "smpl_type": "smplh",             // "smpl" | "smplh" | "smplx"
  "Rh": [[rx, ry, rz]],             // Root orientation (1×3 axis-angle)
  "Th": [[tx, ty, tz]],             // Root translation (1×3)
  "poses": [[p0, p1, p2, ...]],     // Full body poses (1×N axis-angle, flattened)
  "shapes": [[0, 0, ..., 0]],       // Shape coefficients (1×16, all zeros)
  "mocap_framerate": 30
}]
```

### Field Details

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `Rh` | list[list[float]] | `[1, 3]` | Root joint axis-angle (not double-wrapped) |
| `Th` | list[list[float]] | `[1, 3]` | Root translation (XYZ in meters) |
| `poses` | list[list[float]] | `[1, N]` | Flattened axis-angle for all joints |
| `shapes` | list[list[float]] | `[1, 16]` | SMPL β parameters (always zeros) |

### Pose Vector Size by SMPL Type

| SMPL Type | Total Joints | Joints Structure | Pose Vector Size |
|-----------|--------------|------------------|-----------------|
| `smpl` | 24 | root(1) + body(23) | **72** (3+69) |
| `smplh` | 52 | root(1) + body(23) + hands(28) | **156** (3+69+84) |
| `smplx` | 55 | root(1) + body(23) + jaw(1) + eyes(2) + hands(30) | **165** (3+69+3+6+90) |

**Note:** motion_135 has only **22 joints** (root + 21 body):
- Mapped to SMPL positions [0:22] in the pose vector
- Remaining SMPL joints padded with zeros
- **No hand or facial animation** (all hand/eye params = 0)

### Pose Vector Layout (for smplh example, 156 params)

```
Index   Joint Name           Type           Size
0-2     Root (Hips)          Axis-angle     3
3-5     Spine0                Axis-angle     3
6-8     Spine1                Axis-angle     3
9-11    Spine2                Axis-angle     3
12-14   Neck                  Axis-angle     3
15-17   Head                  Axis-angle     3
18-20   Left Shoulder         Axis-angle     3
21-23   Left Elbow            Axis-angle     3
24-26   Left Wrist            Axis-angle     3
27-29   Right Shoulder        Axis-angle     3
30-32   Right Elbow           Axis-angle     3
33-35   Right Wrist           Axis-angle     3
36-38   Left Hip              Axis-angle     3
39-41   Left Knee             Axis-angle     3
42-44   Left Ankle            Axis-angle     3
45-47   Right Hip             Axis-angle     3
48-50   Right Knee            Axis-angle     3
51-53   Right Ankle           Axis-angle     3
54-56   Left Toe              Axis-angle     3
57-59   Right Toe             Axis-angle     3
60-62   (padded, zeros)       Axis-angle     3
63-65   (padded, zeros)       Axis-angle     3
66-155  Left Hand (15 joints) Axis-angle     90
156-245 Right Hand (15 joints)Axis-angle     90
```

### Actual Mapping from motion_135

```
motion_135 joints (22):           → SMPL model (52 for smplh)
0: Root                           → poses[0:3]      (Hips)
1: Spine0                         → poses[3:6]
2: Spine1                         → poses[6:9]
3: Spine2                         → poses[9:12]
4: Neck                           → poses[12:15]
5: Head                           → poses[15:18]
6: LeftShoulder                   → poses[18:21]
7: LeftElbow                      → poses[21:24]
8: LeftWrist                      → poses[24:27]
9: RightShoulder                  → poses[27:30]
10: RightElbow                    → poses[30:33]
11: RightWrist                    → poses[33:36]
12: LeftHip                        → poses[36:39]
13: LeftKnee                       → poses[39:42]
14: LeftAnkle                      → poses[42:45]
15: RightHip                       → poses[45:48]
16: RightKnee                      → poses[48:51]
17: RightAnkle                     → poses[51:54]
18: LeftToe                        → poses[54:57]
19: RightToe                       → poses[57:60]
20: (unused)                       → zeros
21: (unused)                       → zeros
(none)                            → poses[66:156] = zeros (hand joints)
```

---

## 4. BATCH CONVERSION CLI

**Script:** `scripts/embodied/batch_npz_to_smpl_mesh_json.py`

### Usage

```bash
# Batch convert all NPZ files in a directory
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/physflow_v2_compare_iter1000/npz \
    --output-dir output/physflow_v2_compare_iter1000/smpl_mesh \
    --smpl-type smplh \
    --gender neutral \
    --skip-existing

# Convert single file
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-file data/embodied_debug/v6_e2e_test/npz/wave_hand.npz \
    --output-dir data/embodied_debug/v6_e2e_test/smpl_mesh \
    --smpl-type smplx
```

### Arguments

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--npz-dir` | str | - | * | Directory of motion_135 NPZ files (mutually exclusive with `--npz-file`) |
| `--npz-file` | str | - | * | Single NPZ file to convert (mutually exclusive with `--npz-dir`) |
| `--output-dir` | str | - | ✓ | Output directory for JSON files |
| `--smpl-type` | str | `"smplh"` | - | SMPL model: `"smpl"`, `"smplh"` (recommended), or `"smplx"` |
| `--gender` | str | `"neutral"` | - | Character gender: `"neutral"`, `"male"`, or `"female"` |
| `--skip-existing` | flag | False | - | Skip already converted files |

### Output

**Filename Pattern:** `{input_stem}.json`

**Example:**
- Input: `pretrained_00_a_person_stands_still_raw.npz`
- Output: `pretrained_00_a_person_stands_still_raw.json`

**Console Output Example:**
```
Found 76 NPZ files to process
SMPL type: smplh, gender: neutral
  [1/76] pretrained_00_a_person_stands_still_raw: 120 frames @ 30fps -> 245.3KB
  [2/76] pretrained_01_a_person_stands_in_a_relaxed_pose_raw: 120 frames @ 30fps -> 246.1KB
  ...
Done: 76 converted, 0 failed, 0 skipped
Output: output/physflow_v2_compare_iter1000/smpl_mesh
```

---

## 5. NPZ FILES AVAILABLE IN COMPARISON OUTPUT

**Directory:** `output/physflow_v2_compare_iter1000/npz/`

### Statistics

- **Total NPZ files:** 76
- **Naming convention:** `{model_variant}_{id}_{description}_{motion_type}.npz`

### Breakdown by Variant

| Model | Raw Motion | RL Motion | Total |
|-------|-----------|-----------|-------|
| `pretrained_` | 19 files | 19 files | 38 |
| `finetuned_` | 19 files | 19 files | 38 |
| **Total** | **38** | **38** | **76** |

### Naming Pattern Examples

```
pretrained_00_a_person_stands_still_raw.npz
pretrained_00_a_person_stands_still_rl.npz
pretrained_01_a_person_stands_in_a_relaxed_pose_raw.npz
pretrained_01_a_person_stands_in_a_relaxed_pose_rl.npz
...
finetuned_17_a_person_does_a_jumping_jack_raw.npz
finetuned_17_a_person_does_a_jumping_jack_rl.npz
```

### Motion Types

- **`raw`** - Raw motion output from generation model
- **`rl`** - Motion after reinforcement learning refinement

### Motion IDs (0-18, 19 motions per variant)

Each variant has 19 motion sequences:
- ID 0: "a person stands still"
- ID 1: "a person stands in a relaxed pose"
- ID 2: "a person shifts weight from left to right"
- ID 3: "a person walks forward at a normal pace"
- ... and 15 more

---

## 6. EMBODIED_VIZ DATA DIRECTORY STATE

**Base Path:** `motion_annot_web/embodied_viz/`

### Current Directory Structure

```
embodied_viz/
├── ARCHITECTURE.md                      (23.6 KB)
├── ARCHITECTURE_ANALYSIS.md            (15.4 KB)
├── DEPLOYMENT_GUIDE.md                 (20.7 KB)
├── DIAGNOSTICS.md                      (11.7 KB)
├── FEATURE_COMPLETION_REPORT.md        (12.9 KB)
├── FINAL_SUMMARY.txt                   (10.3 KB)
├── GO_LIVE_CHECKLIST.md                (7.6 KB)
├── IMPLEMENTATION_COMPLETE.md          (12.7 KB)
├── IMPLEMENTATION_STATUS.md            (14.2 KB)
├── INDEX.md                            (3.6 KB)
├── PHYSICS_SIM_TRACE.md                (13.6 KB)
├── PRODUCTION_READINESS.txt            (17.1 KB)
├── QUICK_START.md                      (13.6 KB)
├── README.md                           (10.6 KB)
├── README_COMPARISON.md                (9.0 KB)
├── START_HERE.md                       (9.0 KB)
├── VARIANT_COMPARISON_README.md        (18.1 KB)
├── VIEWER_CODE_WALKTHROUGH.md          (17.4 KB)
├── app.py                              (28.8 KB - Flask app)
├── static/ -> ../score_m2m_refine/static  (symlink)
├── templates/                          (19.6 KB directory)
└── test_implementation.py               (8.4 KB)

**⚠️ NOTE: No `data/` subdirectory exists**
```

### Missing Components

- **No `data/` directory** - Must be created for JSON conversions
- **No pre-converted SMPL mesh JSON files** - Ready for batch conversion
- **Static assets** - Linked via symlink to `../score_m2m_refine/static`

### Web App Configuration

**File:** `motion_annot_web/embodied_viz/app.py` (28.8 KB)

Likely serves:
- Flask application for 3D visualization
- Routes for loading motion JSON files
- Asset serving from `static/` symlink
- HTML templates from `templates/`

### Templates Available

**Directory:** `motion_annot_web/embodied_viz/templates/` (19.6 KB)

Contains HTML templates for web viewer rendering.

---

## 7. CONVERSION WORKFLOW RECOMMENDATION

### Step 1: Create Data Directory

```bash
mkdir -p motion_annot_web/embodied_viz/data/smpl_mesh
```

### Step 2: Run Batch Conversion

```bash
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/physflow_v2_compare_iter1000/npz \
    --output-dir motion_annot_web/embodied_viz/data/smpl_mesh \
    --smpl-type smplh \
    --gender neutral \
    --skip-existing
```

### Step 3: Verify Output

```bash
# Check converted files
ls -lh motion_annot_web/embodied_viz/data/smpl_mesh/ | head -10

# Validate JSON structure
python3 -c "import json; d = json.load(open('motion_annot_web/embodied_viz/data/smpl_mesh/pretrained_00_a_person_stands_still_raw.json')); print('Keys:', d.keys()); print('Frames:', len(d['frames'])); print('FPS:', d['fps'])"
```

### Step 4: Expected Output Size

- **Per file:** ~200-300 KB (depending on motion length)
- **Total for 76 files:** ~15-23 MB
- **Compact JSON format** uses `separators=(',', ':')` for minimal size

---

## 8. KEY TRANSFORMATION DETAILS

### Translation (T_h)

- **Source:** motion_135[:, 0:3]
- **Target:** JSON `"Th": [[tx, ty, tz]]` (root translation)
- **Interpretation:** Position of root joint in world space (meters)

### Rotation (R_h)

- **Source:** motion_135[:, 3:9] (first 6 values = root joint rot6d)
- **Process:**
  1. Reorder from row-major [R₀₀, R₀₁, R₁₀, R₁₁, R₂₀, R₂₁] to column-major
  2. Apply Gram-Schmidt to complete 3×3 matrix
  3. Convert to axis-angle [rx, ry, rz]
- **Target:** JSON `"Rh": [[rx, ry, rz]]` (root orientation)

### Body Poses

- **Source:** motion_135[:, 9:135] (22 joints × 6 rot6d = 132 values, split into 21 body joints)
- **Process:** Same rot6d→axis-angle conversion per joint
- **Target:** JSON `"poses": [[[p0, p1, ..., pN]]]` where each triplet is axis-angle
- **Structure:**
  - First 3 elements: root (duplicated from Rh for compatibility)
  - Next 63 elements: 21 body joints × 3
  - Remaining elements: zeros for SMPL joints not in motion_135

### Shape Coefficients (β)

- **Source:** None (motion_135 doesn't contain shape)
- **Target:** `"shapes": [[0, 0, ..., 0]]` (16 zeros for SMPL-H/X)
- **Interpretation:** Neutral body shape (no personalization)

---

## 9. POTENTIAL ISSUES & MITIGATIONS

| Issue | Cause | Mitigation |
|-------|-------|-----------|
| Rot6D orthogonalization fails | Numerical precision in rot6d data | Check input NPZ values in valid range |
| JSON files too large | Many frames + high precision | Use `separators=(',', ':')` (already done) |
| Missing hand animation | motion_135 only has body joints | Expected; hands remain neutral pose |
| Frontend load fails | Missing `/data/` directory structure | Pre-create directory and mount in Flask |
| Axis-angle singularities | Gimbal lock near π rotations | scipy.spatial.transform handles robustly |

---

## SUMMARY

✅ **Conversion Pipeline Status:** Ready to execute

- **Input:** 76 NPZ files (motion_135 format, 22 joints, 135 values/frame)
- **Function:** `convert_single_npz()` with rot6d→axis-angle transformation
- **Output:** SMPL mesh JSON (full 52-165 joint skeleton, web-compatible)
- **SMPL Type:** Default `smplh` (52 joints, hand-capable)
- **Directory:** Create `motion_annot_web/embodied_viz/data/` for storage
- **Expected Size:** ~15-23 MB for all 76 files
- **Next Step:** Run batch conversion script to populate web viewer data

