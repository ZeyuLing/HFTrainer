# NPZ → SMPL Mesh JSON Conversion - Quick Reference

## 🔄 Conversion Pipeline at a Glance

```
INPUT (NPZ)                PROCESSING                OUTPUT (JSON)
┌─────────────┐           ┌──────────────────┐       ┌──────────────┐
│motion_135   │──────────→│rot6d_to_axis_angle│──────→│SMPL mesh JSON│
│22 joints    │           │Gram-Schmidt       │       │52+ joints    │
│135 params   │           │orthogonalization  │       │156+ params   │
└─────────────┘           └──────────────────┘       └──────────────┘
```

---

## 📋 Function Signatures (Copy-Paste Ready)

### Main Conversion Function
```python
def convert_single_npz(npz_path: str, smpl_type: str = "smplx",
                        gender: str = "neutral") -> dict:
    """Returns: {"type": "frames", "fps": 30, "frames": [...]}"""
```

### Rotation Converter
```python
def rot6d_to_axis_angle_np(rot6d: np.ndarray) -> np.ndarray:
    """Converts (..., 6) → (..., 3) axis-angle"""
```

---

## 📦 Input Format: motion_135 NPZ

| Field | Shape | Content |
|-------|-------|---------|
| `motion_135` | `(T, 135)` | **3 transl + 22×6 rot6d** |
| `fps` | scalar | Frame rate (usually 30) |
| `prompt` | string | Motion description |

**Breakdown of 135 values per frame:**
```
[0:3]      → Translation (tx, ty, tz)
[3:135]    → 22 joints × 6 rot6d values each
           = 132 rot6d values
```

---

## 📤 Output Format: SMPL Mesh JSON

**Per Frame Structure:**
```json
[{
  "id": 0,
  "gender": "neutral",
  "smpl_type": "smplh",
  "Rh": [[rx, ry, rz]],           // 1×3 root axis-angle
  "Th": [[tx, ty, tz]],           // 1×3 translation
  "poses": [[p₀, p₁, ..., pₙ]],   // 1×N pose params (axis-angle)
  "shapes": [[0, 0, ..., 0]],     // 1×16 (always zeros)
  "mocap_framerate": 30
}]
```

**Pose Vector Sizes by SMPL Type:**
| Type | Joints | Pose Size | Structure |
|------|--------|-----------|-----------|
| `smpl` | 24 | 72 | root(3) + body(69) |
| `smplh` | 52 | 156 | root(3) + body(69) + hands(84) |
| `smplx` | 55 | 165 | root(3) + body(69) + jaw(3) + eyes(6) + hands(90) |

---

## 🚀 CLI Command (Ready to Run)

```bash
# Create output directory
mkdir -p motion_annot_web/embodied_viz/data/smpl_mesh

# Batch convert all 76 files
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-dir output/physflow_v2_compare_iter1000/npz \
    --output-dir motion_annot_web/embodied_viz/data/smpl_mesh \
    --smpl-type smplh \
    --gender neutral \
    --skip-existing

# Convert single file
python3 scripts/embodied/batch_npz_to_smpl_mesh_json.py \
    --npz-file output/physflow_v2_compare_iter1000/npz/pretrained_00_a_person_stands_still_raw.npz \
    --output-dir motion_annot_web/embodied_viz/data/smpl_mesh
```

---

## 📊 Available NPZ Files (76 total)

**Location:** `output/physflow_v2_compare_iter1000/npz/`

```
Variants:
├── pretrained_*_*_raw.npz     (19 files)
├── pretrained_*_*_rl.npz      (19 files)
├── finetuned_*_*_raw.npz      (19 files)
└── finetuned_*_*_rl.npz       (19 files)

Motion types:
├── raw    = raw generation model output
└── rl     = after RL refinement

Naming: {variant}_{id}_{description}_{type}.npz
Example: pretrained_00_a_person_stands_still_raw.npz
```

---

## 💾 Data Directory Status

```
motion_annot_web/embodied_viz/
├── app.py                    ✅ (Flask web viewer)
├── templates/                ✅ (HTML templates)
├── static/ → symlink         ✅ (assets from score_m2m_refine)
└── data/                     ❌ (NEEDS TO BE CREATED)
    └── smpl_mesh/            ❌ (output directory for JSON files)
```

**Action Required:** Create `motion_annot_web/embodied_viz/data/smpl_mesh/`

---

## 🔍 Verification Commands

```bash
# Check conversion success
ls -lh motion_annot_web/embodied_viz/data/smpl_mesh/ | head -5

# Inspect JSON structure
python3 -c "
import json
with open('motion_annot_web/embodied_viz/data/smpl_mesh/pretrained_00_a_person_stands_still_raw.json') as f:
    data = json.load(f)
    print(f'Type: {data[\"type\"]}')
    print(f'FPS: {data[\"fps\"]}')
    print(f'Frames: {len(data[\"frames\"])}')
    print(f'Pose size: {len(data[\"frames\"][0][0][\"poses\"][0])}')
"

# Check file sizes
du -sh motion_annot_web/embodied_viz/data/smpl_mesh/
```

---

## ⚙️ Transformation Details

### Translation Path
```
NPZ motion_135[t, 0:3]  →  [tx, ty, tz]  →  JSON Th[t]
```

### Rotation Path
```
NPZ motion_135[t, 3:9]      (rot6d for joint 0)
    ↓ [reorder: 0,2,4,1,3,5]
column-major rot6d
    ↓ [Gram-Schmidt orthogonalization]
3×3 rotation matrix
    ↓ [Rotation.from_matrix().as_rotvec()]
axis-angle [rx, ry, rz]
    ↓
JSON Rh[t]
```

### Body Joints Path
```
For each of 22 joints:
  NPZ motion_135[t, 3+6i : 9+6i]  (rot6d for joint i)
    ↓ [same rot6d→axis-angle process]
  axis-angle [px, py, pz]
    ↓
  JSON poses[t, 3+3i : 6+3i]
```

---

## 📈 Expected Output Stats

| Metric | Value |
|--------|-------|
| Input files | 76 NPZ files |
| Output files | 76 JSON files |
| Per-file size | ~200-300 KB |
| Total size | ~15-23 MB |
| Frames per file | ~100-150 frames |
| Framerate | 30 fps |
| Processing time | ~5-10 minutes (total) |

---

## 🎯 Key Points

✅ **motion_135 encoding:** [3 transl + 22×6 rot6d]  
✅ **rot6d format:** Row-major, needs Gram-Schmidt + orthogonalization  
✅ **Output:** SMPL mesh JSON for 3D web viewer (load_smpl.js)  
✅ **SMPL-H default:** 52 joints, 156-param poses (good web support)  
✅ **Hand joints:** Zero-padded (motion_135 only has body)  
✅ **Shape coefficients:** Always 16 zeros (no body shape variation)  
✅ **JSON serialization:** Compact format (`separators=(',', ':')`)  

---

## 🔗 Related Files

- **Main script:** `scripts/embodied/batch_npz_to_smpl_mesh_json.py`
- **Web viewer:** `motion_annot_web/embodied_viz/app.py`
- **Input data:** `output/physflow_v2_compare_iter1000/npz/`
- **Output directory:** `motion_annot_web/embodied_viz/data/smpl_mesh/`

