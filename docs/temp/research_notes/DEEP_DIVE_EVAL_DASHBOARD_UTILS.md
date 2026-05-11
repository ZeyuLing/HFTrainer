# Deep Dive: `motion_annot_web/eval_dashboard/utils.py` & `score_m2m` Integration

## Executive Summary

**`score_m2m_web.py` uses only 2 functions from `eval_dashboard/utils.py`:**
- `load_npz_smpl_params()` — lines 251-319 indirect calls
- `load_npz_positions()` — line 1618 indirect call

The minimal dependency set consists of **3 core hftrainer modules** and **1 PyTorch asset file**. Everything else in utils.py is NOT called by score_m2m.

---

## 1. Full Function Index of `utils.py` (1035 lines)

### Functions & Line Counts

| # | Function Signature | Lines | Used by score_m2m? | Dependencies |
|----|----|----|----|----|
| 1 | `load_npz_positions(npz_path, rotation_space="local")` | 108-224 | ✅ YES (line 1618) | `_compute_fk_positions()` |
| 2 | `load_source_motion_positions(npz_path)` | 226-271 | ❌ NO | hftrainer transforms + `_compute_fk_positions()` |
| 3 | `_compute_fk_positions(motion, rotation_space="local")` | 273-315 | ✅ Indirect (called by 1) | torch, hftrainer.pipelines.motion.differentiable_fk |
| 4 | `_simple_position_extract(motion)` | 317-333 | ✅ Indirect (fallback in 3) | numpy only |
| 5 | `load_npz_smpl_params(npz_path, rotation_space, target_faces)` | 335-378 | ✅ YES (lines 287, 294) | Multiple: `_smpl_from_*()` functions |
| 6 | `_get_soma_skin()` | 384-418 | ✅ Indirect (called by 7) | torch, numpy |
| 7 | `_smpl_from_kimodo_lbs(data, target_faces=None)` | 420-755 | ✅ Indirect (called by 5) | torch, base64, scipy.spatial, fast_simplification (optional) |
| 8 | `_mesh_from_smpl22_positions(positions, fps=30)` | 757-868 | ❌ NO | numpy only |
| 9 | `_smpl_from_original_npz(data)` | 870-936 | ✅ Indirect (called by 5) | numpy only |
| 10 | `_smpl_from_motion135(data, rotation_space="local")` | 938-1022 | ✅ Indirect (called by 5) | hftrainer.models.motion.components.utils.geometry.rotation_convert |
| 11 | `format_metric(value, metric_name="")` | 1024-1036 | ❌ NO | None (pure Python) |

**Module-level cache:** `_soma_cache = {}` (line 381)

### Data Structures (Non-function)

Lines 10-105: Skeleton definitions (parents, edges, joint names) — all pure data, ~95 lines

---

## 2. Detailed Function Analysis for score_m2m Usage

### 2.1 `load_npz_smpl_params()` — **PRIMARY ENTRY POINT**

**Location:** lines 335-378  
**Called by score_m2m:** Line 287, 294 in `read_smpl_from_npz()`  
**Signature:**
```python
def load_npz_smpl_params(
    npz_path: str,
    rotation_space: str = "local",
    target_faces: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
```

**Logic Flow:**
```
load_npz_smpl_params()
  ├─ Check if NPZ has KIMODO format (posed_joints + global_rot_mats)
  │   └─ YES → call _smpl_from_kimodo_lbs(data, target_faces)
  │           Returns: {"type": "mesh_sequence", "vertices_b64": ..., "faces_b64": ..., ...}
  │
  ├─ Check if NPZ has original SMPL format (poses + trans)
  │   └─ YES → call _smpl_from_original_npz(data)
  │           Returns: {"frames": [...], "num_frames": ..., ...}
  │
  └─ Check if NPZ has motion_135 (eval output)
      └─ YES → call _smpl_from_motion135(data, rotation_space)
              Returns: {"frames": [...], "num_frames": ..., ...}
```

**External Calls:**
1. `_smpl_from_kimodo_lbs(data, target_faces)` — line 359
2. `_smpl_from_original_npz(data)` — line 371
3. `_smpl_from_motion135(data, rotation_space)` — line 375

**Key Variable Usages:**
- `data["posed_joints"]`, `data["global_rot_mats"]` — KIMODO detection
- `data["motion_135"]` — eval motion format detection
- `data["poses"]`, `data["trans"]` — original SMPL format

---

### 2.2 `load_npz_positions()` — **SECONDARY ENTRY POINT**

**Location:** lines 108-224  
**Called by score_m2m:** Line 1618 in `/api/pair_npz/` handler  
**Signature:**
```python
def load_npz_positions(npz_path: str, rotation_space: str = "local") -> Optional[Dict[str, Any]]:
```

**Logic Flow:**
```
load_npz_positions()
  ├─ Try to load direct "positions" field (SMPL-22 joint positions, (T,22,3))
  │   └─ YES → return {"positions": [...], "edges": SMPL22_EDGES, "fps": 30, ...}
  │
  ├─ Check KIMODO format: "posed_joints" + "global_rot_mats"
  │   └─ YES → extract joints, ground-normalize, return {"positions": [...], ...}
  │           Also handles prefix_/suffix_ context for stitched KIMODO E14 timelines
  │
  └─ Try FK computation on motion / motion_denorm
      └─ YES → call _compute_fk_positions(motion, rotation_space)
              Returns: {"positions": [...], ...}
```

**Key Features:**
- Does NOT ground-normalize SMPL-22 positions (frontend does it via `canonicalizeGround()`)
- Detects KIMODO via shape[1] >= 23 AND has global_rot_mats
- Handles layout_json for E14/E15/E8 dynamic context visualization
- Loads both "positions" AND "posed_joints" fields (different formats)

---

### 2.3 `_compute_fk_positions()` — **FK CORE**

**Location:** lines 273-315  
**Signature:**
```python
def _compute_fk_positions(
    motion: np.ndarray, 
    rotation_space: str = "local"
) -> Optional[Dict[str, Any]]:
```

**Dependencies:**
```python
import torch
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk
```

**Logic:**
```
1. Load bone_offsets_22.pt from PROJECT_ROOT/data/hymotion_m2m_data/
   └─ If not found, fallback to _simple_position_extract()

2. Convert numpy motion (T, 135) → torch tensor
   └─ Unsqueeze if 1D

3. Call motion135_to_fk(motion_t, bone_offsets, rotation_space)
   └─ Returns: (world_positions, world_rotations, ...)

4. Convert positions back to numpy, return dict with:
   ├─ "positions": (T, 22, 3) list
   ├─ "edges": SMPL22_EDGES
   ├─ "fps": 30
   ├─ "joint_names": SMPL22_JOINT_NAMES
   └─ "num_frames": T
```

**Critical File:** `bone_offsets_22.pt`
- **Location:** `data/hymotion_m2m_data/bone_offsets_22.pt`
- **Type:** PyTorch tensor (1.476 KB)
- **Content:** 22 bone offsets (child - parent in T-pose), shape (22, 3)
- **Loaded:** `torch.load(..., map_location='cpu').float()`
- **Fallback:** If not found, uses `_simple_position_extract()` which just extracts translation

---

### 2.4 `_smpl_from_motion135()` — **EVAL MOTION → SMPL CONVERSION**

**Location:** lines 938-1022  
**Signature:**
```python
def _smpl_from_motion135(data, rotation_space: str = "local") -> Optional[Dict[str, Any]]:
```

**External Imports:**
```python
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
)
from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
    global_to_local_rot6d,  # fallback if rotation_utils.py unavailable
)
```

**Logic:**
```
1. Extract motion_135 from NPZ: (T, 135) array
   ├─ trans: [0:3]        → (T, 3) absolute translation
   └─ rot6d: [3:135]      → (T, 22, 6) 22-joint 6D rotations

2. Convert rot6d → axis-angle
   ├─ If rotation_space == "global":
   │   ├─ Convert global_rot6d → local_rot6d via global_to_local_rot6d()
   │   └─ Then rot6d → axis-angle
   │
   └─ If rotation_space == "local":
       └─ Direct rot6d → axis-angle

3. Reshape axis-angle to (T, 66), pad to (T, 156) for SMPL+H
   └─ First 66 dims = body joints
   └─ Next 90 dims = hand joints (zeroed out)

4. Build per-frame SMPL param dicts:
   ├─ "id": 0
   ├─ "gender": "neutral"
   ├─ "smpl_type": "smplh"
   ├─ "Rh": root rotation (3,)
   ├─ "Th": translation (3,)
   ├─ "poses": full 156D
   ├─ "shapes": zeros (1, 16)
   └─ "mocap_framerate": 30

5. Return: {"frames": [frame_dict], "num_frames": T, "fps": 30, ...}
```

**Important Notes:**
- Uses zero betas (no body shape info in eval NPZ)
- Pads rot6d to SMPL+H (52 joints) even though M2M only has 22 joints body
- Calls `rotation_utils.py` locally first if available, else imports from hftrainer

---

### 2.5 `_smpl_from_kimodo_lbs()` — **SOMA-77 MESH SKINNING**

**Location:** lines 420-755 (336 lines — most complex function!)

**Signature:**
```python
def _smpl_from_kimodo_lbs(data, target_faces: Optional[int] = None) -> Dict[str, Any]:
```

**What is SOMA-77?**
- KIMODO output skeleton = SOMA-77 (77 joints instead of SMPL-22)
- First 22 joints = SMPL-22 body (compatible)
- Remaining 55 joints = fingers (5 per hand × 5 fingers) + face (3 joints)
- See `SOMA_JOINT_NAMES` (lines 65-83), `SOMA_PARENTS_77` (lines 91-97)

**External Dependencies:**
```python
import torch
import base64
from scipy.spatial import cKDTree  # for barycentric interp
import fast_simplification  # optional, for mesh decimation
```

**Assets Required:**
```
ref_repo/KIMODO/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz
```
Contains:
- `bind_rig_transform` — (77, 4, 4) inverse bind matrices
- `bind_vertices` — (V, 3) template mesh vertices
- `faces` — (F, 3) triangle indices
- `lbs_indices` — (V, 8) which joints influence each vertex
- `lbs_weights` — (V, 8) weight per joint influence

**Complex Logic Sections:**

**A. Prefix/Suffix Stitching (lines 440-484)**
- KIMODO E14 outputs can have prefix_/suffix_ SOMA-77 data for context
- Concatenates prefix + main + suffix into one timeline
- Applies overlap removal based on `prefix_main_overlap`, `suffix_main_overlap`
- Result: `full_gr` (T_full, 77, 3, 3), `full_pj` (T_full, 77, 3)

**B. LBS Skinning (lines 490-524)**
- Build FK transforms from `global_rot_mats` + `posed_joints`
- Linear Blend Skinning: `verts = sum_j(LBS_weight[v,j] * T_j @ bind_verts[v])`
- Vectorized on GPU (or CPU if no GPU): ~millions of ops

**C. Ground Normalization (lines 506-524)**
- For stitched timelines, use ONLY the main span's min_y as floor reference
- Prevents "floating" when context frames crouch/penetrate
- Anchor logic: `main_mask = (logical_idx >= prefix_len) & (logical_idx < prefix_len + main_len)`

**D. Frame Subsampling (lines 525-546)**
- If T > max_frames (720 for stitched, 300 for single), subsample with stride
- For stitched timelines, preserve exact boundary frames at prefix_len and prefix_len+main_len
- Non-naive subsampling: ensures no frame-skip at seam

**E. Post-LBS Mesh Decimation (lines 547-727)**
- If fast_simplification available: quadric simplification
  - Target: 18000 faces (tuned 2026-04-27 to avoid mesh artifacts at thin parts)
  - Agg param: 4 (less aggressive than default 7)
  - Inherit LBS weights via barycentric interpolation on 8 nearest faces
- Fallback (no fast_simplification): uniform vertex subsampling every Nth vertex
- Output: optimized verts_np + faces_out

**F. Payload Encoding (lines 728-737)**
- Round verts to 3 decimal places (~1mm precision)
- Binary base64 encoding: 10x faster than JSON .tolist()
- Output: "vertices_b64", "faces_b64" (decoder on frontend: Float32Array)

**Return Dict:**
```python
{
    "type": "mesh_sequence",
    "vertices_b64": base64_str,
    "faces_b64": base64_str,
    "num_frames": T_eff,
    "num_vertices": V',
    "num_faces": F',
    "fps": fps // stride,
    "stride": stride,
    "skeleton_type": "soma",
    "prefix_len_full": prefix_len,
    "main_len_full": main_len,
    "suffix_len_full": suffix_len,
}
```

---

### 2.6 `_get_soma_skin()` — **SOMA ASSET LOADER**

**Location:** lines 384-418

**Caching Strategy:**
- Module-level `_soma_cache = {}` (line 381)
- Only loads ONCE per process, stores in `_soma_cache["ready"]`
- Converts numpy arrays → torch tensors on CPU

**Asset Path:**
```
os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),     # eval_dashboard/
        '..',                          # motion_annot_web/
        '..',                          # hf_trainer/
        'ref_repo', 'KIMODO',
        'kimodo', 'kimodo', 'assets', 'skeletons', 'somaskel77',
        'skin_standard.npz'
    )
)
```

**Returns Dict with Keys:**
- `"ready": True`
- `"device": torch.device("cpu")`
- `"bind_rig_inv"`: (77, 4, 4) inverse bind transforms
- `"bind_verts"`: (V, 3)
- `"faces"`: (F, 3) as int32 numpy
- `"lbs_indices"`: (V, 8) as torch.long
- `"lbs_weights"`: (V, 8) as torch.float

---

## 3. Trace of `read_smpl_from_npz()` Calls (score_m2m_web.py lines 251-319)

**Location in score_m2m:** lines 251-349

**Full Call Chain:**
```
read_smpl_from_npz(npz_path, rotation_space, target_faces)
  │
  ├─ (Lines 268-311) If _eval_load_npz_smpl_params available:
  │   ├─ Build cache_key from file mtime/size + rotation_space + target_faces
  │   ├─ Check cache: _kimodo_cache_get(cache_key)
  │   ├─ If NOT cached:
  │   │   ├─ Call: _eval_load_npz_smpl_params(npz_path, rotation_space, target_faces)
  │   │   │  └─ This is load_npz_smpl_params() from utils.py
  │   │   ├─ If result.type == "mesh_sequence":
  │   │   │   ├─ Cache it: _kimodo_cache_put(cache_key, result)
  │   │   │   └─ Return result directly (KIMODO SOMA mesh)
  │   │   └─ Else if "frames" in result:
  │   │       └─ Wrap: {"type": "frames", "frames": result["frames"]}
  │   │
  │   └─ If TypeError (old utils signature): retry without target_faces param
  │
  └─ (Lines 314-349) Fallback: parse NPZ locally
      ├─ Load poses, trans, betas, gender, framerate
      ├─ Reshape poses to (T, 66) if oversized
      ├─ Build frame dicts with Rh, Th, poses, shapes
      └─ Return {"type": "frames", "frames": [...]}
```

**Key Variables Used from eval utils:**
- `load_npz_smpl_params(npz_path, rotation_space, target_faces)` — main call
- All indirect calls flow through here to `_smpl_from_*()` functions

---

## 4. External hftrainer Dependencies in score_m2m → eval_dashboard Flow

### Minimal Set for score_m2m Usage

**THREE core hftrainer modules ONLY:**

1. **`hftrainer.pipelines.motion.differentiable_fk`**
   - Import: `from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk`
   - Called by: `_compute_fk_positions()` (line 282 of utils.py)
   - What: Forward kinematics for 135-dim motion → world positions + rotations
   - Signature: `motion135_to_fk(motion_t, bone_offsets, rotation_space) → (world_pos, world_rot, trans, local_rotmat)`
   - Uses: torch

2. **`hftrainer.datasets.motion.motionhub.transforms.fk_utils`**
   - Import: `from hftrainer.datasets.motion.motionhub.transforms.fk_utils import SMPL22_PARENTS` + rotation converters
   - Called by: `motion135_to_fk()` (via differentiable_fk.py line 125)
   - Functions used:
     - `SMPL22_PARENTS` — constant list of 22 parent indices
     - `global_to_local_rot6d_torch()` — converts global rot6d to local
     - `local_to_global_rot6d_torch()` — inverse
   - Rotation convention: **row-major** (matches M2M training data)

3. **`hftrainer.models.motion.components.utils.geometry.rotation_convert`**
   - Import: `from hftrainer.models.motion.components.utils.geometry.rotation_convert import rotation_6d_to_axis_angle`
   - Called by: `_smpl_from_motion135()` (line 954 of utils.py)
   - Function: `rotation_6d_to_axis_angle()` — convert 6D rotation to 3D axis-angle
   - Rotation convention: **column-major** (different from hftrainer's row-major!)
   - Workaround in code: reorder [0,2,4,1,3,5] to convert between conventions

### Additional Dependencies (NOT on hftrainer, but score_m2m may import)

From score_m2m_web.py itself:
- Line 126: `rotation_6d_to_axis_angle` (pre-warming)
- Line 129: `global_to_local_rot6d_torch` (pre-warming)
- Line 977: `axis_angle_to_matrix`, `matrix_to_axis_angle` (for rigid alignment)
- Line 1076: `m2m_eval_metrics` (for metrics collection, NOT used by eval_dashboard.utils)

---

## 5. `bone_offsets_22.pt` Details

**File Location:** `data/hymotion_m2m_data/bone_offsets_22.pt`

**File Size:** 1,476 bytes

**How It's Loaded:**
```python
# In _compute_fk_positions(), lines 287-291
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
bone_offsets_path = os.path.join(
    _project_root, 'data', 'hymotion_m2m_data', 'bone_offsets_22.pt'
)
bone_offsets = torch.load(bone_offsets_path, map_location='cpu').float()
```

**Resolution:**
- `__file__` = `.../motion_annot_web/eval_dashboard/utils.py`
- dirname(3) = `.../` (project root, where data/ lives)
- Final path = `./data/hymotion_m2m_data/bone_offsets_22.pt`

**Content:**
- PyTorch tensor, dtype float32
- Shape: (22, 3) — 22 joints, 3D offsets (X, Y, Z)
- Semantics: relative position from parent joint in T-pose
- Loaded with `map_location='cpu'` for CPU compatibility

**Fallback If Missing:**
- Calls `_simple_position_extract(motion)` (line 294)
- Just returns root translation, no proper FK
- Result quality: poor (only 1 joint active instead of 22)

---

## 6. KIMODO SOMA-77 Mesh Handling in Depth

### Skeleton Structure

**SOMA-77 vs SMPL-22:**
```
SMPL-22 (body only):
  0-21: pelvis + 21 body joints

SOMA-77 (full-body with fingers + face):
  0-21: same as SMPL-22 (first 22 aligned)
  22-24: jaw + eyes (3 face joints)
  25-46: fingers (44 joints)
           ├─ L hand: 5 fingers × 3 joints = 15 joints
               ├─ L_Index: 25-27
               ├─ L_Middle: 28-30
               ├─ L_Pinky: 31-33
               ├─ L_Ring: 34-36
               └─ L_Thumb: 37-39
               ├─ R_Index: 40-42
               ├─ R_Middle: 43-45
               ├─ R_Pinky: 46-48
               ├─ R_Ring: 49-51
               └─ R_Thumb: 52-54
  55-76: context/reserved (22 extra joints for future)
```

**Why This Matters for score_m2m:**
- KIMODO outputs SOMA-77 mesh + skeleton
- If score_m2m tries to render with SMPL-22 rig, wrists/hands vanish
- Solution: use SOMA_EDGES_77 + SOMA_JOINT_NAMES when rendering

### Asset Path

**skin_standard.npz location:**
```
ref_repo/KIMODO/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz
```

**Relative from eval_dashboard/utils.py:**
```python
os.path.normpath(os.path.join(
    os.path.dirname(__file__),   # eval_dashboard/
    '..', '..',                  # → hf_trainer/
    'ref_repo', 'KIMODO',
    'kimodo', 'kimodo', 'assets', 'skeletons', 'somaskel77',
    'skin_standard.npz'
))
```

**NPZ Contents:**
- `bind_rig_transform` — numpy (77, 4, 4) float32
- `bind_vertices` — numpy (V, 3) float32 [V ≈ 18056]
- `faces` — numpy (F, 3) int32 [F ≈ 36108]
- `lbs_indices` — numpy (V, 8) int64
- `lbs_weights` — numpy (V, 8) float32

---

## 7. Rotation Convention Chaos 🎭

### Three Different Conventions!

**1. Training Data / M2M Network (row-major rot6d)**
```
Order: [R00, R01, R10, R11, R20, R21]
        col0  col0  col1  col1  col2  col2
Example 6D = [1, 0, 0, 1, 0, 0] (identity first 2 cols of 3×3)
Indices: [0, 1, 2, 3, 4, 5]
```

**2. rotation_convert.py (column-major rot6d)**
```
Order: [R00, R10, R20, R01, R11, R21]
        col0  col0  col0  col1  col1  col1
Example 6D = [1, 0, 0, 0, 1, 0] (identity first 2 cols, but swapped)
Indices: [0, 2, 4, 1, 3, 5] — reorder from row-major!
```

**3. geometry.py (row-major, M2M-native)**
```
Order: same as training data [R00, R01, R10, R11, R20, R21]
Used by: motion135_to_fk in differentiable_fk.py
```

### Conversion Chains in utils.py

**For `_smpl_from_motion135()` (line 965-971):**
```python
# Get rot6d in row-major from motion_135
rot6d = motion[:, 3:135].reshape(T, 22, 6)  # row-major from training data

if rotation_space == "global":
    # Call hftrainer's global_to_local_rot6d (row-major input expected)
    local_rot6d = global_to_local_rot6d(rot6d)  # row-major
    # Now convert to column-major for rotation_convert API
    local_colmajor = local_rot6d[:, [0, 2, 4, 1, 3, 5]]  # reorder
else:
    # Already local rot6d (row-major)
    rot6d_colmajor = rot6d[:, [0, 2, 4, 1, 3, 5]]  # convert to column-major

# Call rotation_convert (expects column-major)
axis_angle = rotation_6d_to_axis_angle(rot6d_colmajor)  # col-major in → axis-angle out
```

**For `_compute_fk_positions()` (line 302):**
```python
# motion135 is row-major
# Call motion135_to_fk directly (expects row-major per differentiable_fk.py docstring)
world_pos, world_rot, trans, local_rotmat = motion135_to_fk(
    motion_t, bone_offsets, rotation_space
)
# motion135_to_fk internally:
#   1. Extracts rot6d from motion (row-major, line 121)
#   2. If rotation_space == 'global':
#        └─ Calls global_to_local_rot6d_torch (row-major) from fk_utils.py
#   3. Calls rot6d_to_rotmat_row_major (uses geometry.py, row-major)
```

**Key Insight:**
- `differentiable_fk.motion135_to_fk()` handles row-major natively → no reordering needed
- `rotation_convert.rotation_6d_to_axis_angle()` expects column-major → reorder before call

---

## 8. score_m2m_web.py References to eval_dashboard

**ALL references (24 total mentions):**

| Line | Context | Purpose |
|------|---------|---------|
| 53 | `EVAL_DASHBOARD_DIR = ...` | Path to eval_dashboard module |
| 54 | `EVAL_DASHBOARD_DB = ...` | Path to eval_dashboard.db SQLite |
| 56 | Comment | Explanation |
| 104-105 | Comment | Explain dynamic import strategy |
| 106-107 | `_eval_load_npz_smpl_params = None` | Cache for function pointer |
| 111-115 | `spec_from_file_location(...)` | Dynamic module loading |
| 116-117 | `getattr(_mod, "load_npz_smpl_params")` | Extract function |
| 118-121 | Logging | Log successful loads |
| 126 | Pre-warm import | rotation_convert (NOT from utils) |
| 129 | Pre-warm import | fk_utils (NOT from utils) |
| 136 | Warning log | If import fails |
| 255-259 | Docstring | Explain what utils.load_npz_smpl_params supports |
| 268 | `if _eval_load_npz_smpl_params is not None:` | Check before using |
| 287 | `result = _eval_load_npz_smpl_params(...)` | CALL 1 with all 3 params |
| 294 | `result = _eval_load_npz_smpl_params(...)` | CALL 2 fallback (old signature) |
| 296 | Log warning | If call raised exception |
| 299 | `if isinstance(result, dict) and result.get("type") == "mesh_sequence":` | Check type |
| 300 | Cache put | Cache KIMODO result |
| 302 | Wrap frames | If SMPL frames result |
| 902, 914 | Comments | Reference KIMODO implementation |
| 1203 | Comment | Reference /api/source_motions |
| 1210 | Comment | Reference eval_dashboard URL env var |
| 1601 | `if _eval_load_npz_positions is None:` | Check availability |
| 1618 | `result = _eval_load_npz_positions(...)` | CALL 3 for skeleton rendering |
| 1671 | Comment | Reference setting derivation |
| 1673 | Comment | Reference per-setting datalist |
| 2208-2251 | Admin sync functions | Sync from eval_dashboard.db (NOT utils.py) |

**Critical Finding:** Lines 2208-2251 (admin sync) use `eval_dashboard.db` (database), NOT `utils.py` functions.

---

## 9. Summary: Minimal Inline Set for score_m2m

If you need to inline eval_dashboard code into score_m2m without external import:

### Functions to Inline (4 total)

1. **`load_npz_smpl_params(npz_path, rotation_space, target_faces)`** — 44 lines
   - Entry point for all SMPL conversions

2. **`_smpl_from_kimodo_lbs(data, target_faces)`** — 336 lines (!) 
   - Most complex; handles SOMA-77 LBS + decimation

3. **`_smpl_from_motion135(data, rotation_space)`** — 85 lines
   - Eval motion rot6d → axis-angle

4. **`_smpl_from_original_npz(data)`** — 67 lines
   - Original SMPL format support

5. **`_compute_fk_positions(motion, rotation_space)`** — 43 lines
   - FK entry point (calls hftrainer differentiable_fk)

6. **Helper: `_get_soma_skin()`** — 35 lines
   - SOMA asset loader (caches)

7. **Helper: `_simple_position_extract(motion)`** — 17 lines
   - Fallback if FK unavailable

### External Dependencies (Cannot Inline)

1. **torch** library — entire FK system depends on it
2. **hftrainer.pipelines.motion.differentiable_fk.motion135_to_fk()** — core FK
3. **hftrainer.datasets.motion.motionhub.transforms.fk_utils**:
   - `SMPL22_PARENTS` (constant)
   - `global_to_local_rot6d_torch()` (if rotation_space == 'global')
4. **hftrainer.models.motion.components.utils.geometry.rotation_convert.rotation_6d_to_axis_angle()** — rot6d→axis-angle conversion
5. **bone_offsets_22.pt** asset file (1.476 KB)
6. **Optional:** `fast_simplification`, `scipy.spatial` (for mesh decimation)

### Exact Lines to Copy

**Core:**
- Lines 10-33: SMPL22_PARENTS (skeleton topology)
- Lines 35-40: SMPL22_JOINT_NAMES
- Lines 42-43: SMPL22_EDGES
- Lines 65-102: SOMA skeleton defs (for KIMODO support)
- Lines 335-378: `load_npz_smpl_params()` **[MAIN]**
- Lines 938-1022: `_smpl_from_motion135()` **[MAIN]**
- Lines 870-936: `_smpl_from_original_npz()` **[MAIN]**
- Lines 273-315: `_compute_fk_positions()` **[MAIN]**

**KIMODO Support (if needed):**
- Lines 380-382: _soma_cache definition
- Lines 384-418: `_get_soma_skin()` **[HELPER]**
- Lines 420-755: `_smpl_from_kimodo_lbs()` **[MAIN - HUGE]**

**Optional:**
- Lines 317-333: `_simple_position_extract()` (fallback)
- Lines 1024-1036: `format_metric()` (not used by score_m2m)

---

## 10. Imports Required in Inlined Code

```python
# Bare minimum (always needed)
import os
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# For FK paths
import torch
from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

# For rotation conversion
from hftrainer.datasets.motion.motionhub.transforms.fk_utils import (
    SMPL22_PARENTS,
    global_to_local_rot6d_torch,
)
from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
    rotation_6d_to_axis_angle,
)

# Optional (for KIMODO mesh decimation)
import base64
try:
    import fast_simplification
    from scipy.spatial import cKDTree
except ImportError:
    pass
```

---

## 11. No Other Hidden Dependencies

Checked:
- ✅ `load_npz_positions()` — no calls to functions not in utils.py (except hftrainer)
- ✅ `_compute_fk_positions()` — only calls hftrainer.pipelines.motion.differentiable_fk
- ✅ `_smpl_from_motion135()` — imports hftrainer locally, no utils.py utils calls
- ✅ `_smpl_from_kimodo_lbs()` — only uses numpy, torch, base64, scipy, fast_simplification
- ✅ No circular imports within utils.py itself

**Conclusion:** You can safely extract these functions into score_m2m with only hftrainer external imports.

