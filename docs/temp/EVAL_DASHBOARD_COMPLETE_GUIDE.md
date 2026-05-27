# Motion Annotation Web — Eval Dashboard Complete Guide

**Version**: 2.0  
**Last Updated**: 2026-05-27  
**Status**: ✓ Tested and Verified

---

## Executive Summary

The **Eval Dashboard** (`motion_annot_web/eval_dashboard/`) is a Flask web application for visualizing and comparing HyMotion M2M v2 evaluation results:

- **Database**: SQLite with 5 tables (models, eval_runs, sample_results, agg_metrics, baselines)
- **Visualization**: 3D SMPL meshes in browser (Three.js)
- **Data Format**: NPZ files (motion_135: 135D rotation vectors)
- **Port**: 8081 (default)
- **Data Import**: CLI tool (`data_importer.py`) + optional web UI

---

## Quick Start (3 Steps)

### 1. Convert Evaluation Results

Convert nested `summary.json` → flat JSON format:

```bash
bash /tmp/full_eval_import_workflow.sh \
    work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
    hymotion_m2m_v2_overfit_100
```

This script:
- ✓ Finds all `summary.json` files in eval_overfit/ subdirs
- ✓ Converts to flat JSON (one per eval mode)
- ✓ Verifies NPZ files exist
- ✓ Backs up existing database

**Output**: 8 JSON files in `/tmp/eval_imports_<timestamp>/`

### 2. Import to Database

```bash
cd motion_annot_web/eval_dashboard
python3 data_importer.py import /tmp/eval_imports_<timestamp>/hymotion_m2m_v2_overfit_100__E14_keyframe_periodic.json
```

**Output**:
```json
{
  "status": "ok",
  "model": "hymotion_m2m_v2_overfit_100",
  "model_id": 44,
  "run_id": 5182,
  "task_id": "E14",
  "setting": "keyframe_periodic",
  "num_metrics": 6,
  "num_samples": 100
}
```

### 3. View Results

```bash
# Start server
python3 app.py --port 8081

# Open browser
# http://localhost:8081/task/E14
```

---

## Data Flow

```
work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/
  keyframe_periodic/
    summary.json (nested format)
    motionfix_test_006202.npz
    motionfix_test_006202.npz
    ... (100 total)

  keyframe_pos/
  keyframe_rot/
  text_frame/
  ... (8 modes total)
```

### Conversion Pipeline

```
summary.json
├─ mode: "keyframe_periodic"
├─ checkpoint: "work_dirs/.../checkpoint-epoch_2750"
├─ n_samples: 100
├─ mean_mpjpe_mm: 36.42
├─ per_sample: [
│   {
│     "key": "motionfix_test_006202",
│     "mpjpe": 34.34,
│     "mpjre": 12.12,
│     "T": 120
│   },
│   ...
│ ]
└─ (nested structure, one summary per model)

                    ↓ (convert)

flat.json (one per eval run)
├─ model: "hymotion_m2m_v2_overfit_100"
├─ checkpoint: "work_dirs/.../checkpoint-epoch_2750"
├─ task_id: "E14"
├─ setting: "keyframe_periodic"
├─ aggregated: { mean_mpjpe_mm: 36.42, ... }
├─ per_sample: [
│   {
│     "prompt_id": "motionfix_test_006202",
│     "_npz_path": "/full/path/to/motionfix_test_006202.npz",  ← KEY FIELD
│     "mpjpe": 34.34,
│     "mpjre": 12.12,
│     "T": 120
│   },
│   ...
│ ]
└─ (flat, one per evaluation run)

                    ↓ (data_importer.py)

eval_dashboard.db (SQLite)
├─ models table:
│   id=44, name="hymotion_m2m_v2_overfit_100", checkpoint="...", epoch=2750
├─ eval_runs table:
│   id=5182, model_id=44, task_id="E14", setting="keyframe_periodic", num_samples=100
├─ sample_results table:
│   eval_run_id=5182, sample_idx=0, prompt_id="motionfix_test_006202", 
│   motion_path="/full/path/to/npz", metrics_json='{mpjpe: 34.34, ...}'
└─ agg_metrics table:
    run_id=5182, metric_name="mean_mpjpe_mm", value=36.42

                    ↓ (Flask /api/smpl/<path>)

NPZ File (motion_135 format)
├─ motion_135: shape (T, 135)
│   = [tx, ty, tz, r6d_j0, r6d_j1, ..., r6d_j21]
│   = [3D translation + 22 joints × 6D rotation]
└─ (immutable, read-only from Flask)

                    ↓ (utils.py load_npz_smpl_params)

SMPL Parameters (JSON)
├─ frames: [
│   {
│     "Rh": [rx, ry, rz],           ← root rotation (axis-angle)
│     "Th": [tx, ty, tz],           ← root translation
│     "poses": [rx0, ry0, rz0, ...], ← 156D (22 joints + hand zeros)
│     "shapes": [0, 0, ..., 0],     ← 16D shape (unused)
│     "gender": "neutral"
│   },
│   ... (T frames)
│ ]
├─ num_frames: 120
├─ fps: 30
└─ (frontend-ready format)

                    ↓ (Three.js rendering)

3D Mesh in Browser
├─ SMPL body model
├─ Per-frame skeleton animation
├─ Interactive camera control
└─ Real-time 60 FPS
```

---

## Key Concepts

### NPZ File Format

**Motion135** (135-dimensional motion vector):

```
Frame i = [
  tx, ty, tz,                     ← 3D translation (root position)
  r6d_0[6], r6d_1[6], ..., r6d_21[6]  ← 22 joints × 6D rotation
]

Shape: (T, 135) where T = num frames
Total: 3 + 22×6 = 3 + 132 = 135
```

**6D Rotation Representation** (column-major):

```
6D vector encodes first two columns of 3×3 rotation matrix:
- Elements 0-2: First column (normalized)
- Elements 3-5: Second column (normalized and orthogonalized)
- Third column: Computed as cross product

Conversion to axis-angle (3D):
1. Build 3×3 matrix from 6D
2. Extract axis-angle vector via Rodrigues' formula
```

**Reordering** (important!):

```
row-major (NPZ storage): [R00, R01, R10, R11, R20, R21]
                   ↓
column-major (needed): [R00, R10, R20, R01, R11, R21]
                   ↓
Reorder: [0, 2, 4, 1, 3, 5]
```

### SMPL Skeleton

```
22 joints (SMPL body skeleton):
0: pelvis (root)
1-3: spine (spine1, spine2, spine3)
4-5: l_leg (l_hip, l_knee, l_ankle, l_foot)
6-9: r_leg (r_hip, r_knee, r_ankle, r_foot)
10-12: l_arm (l_collar, l_shoulder, l_elbow, l_wrist)
13-15: r_arm (r_collar, r_shoulder, r_elbow, r_wrist)
16: neck
17: head

Parent hierarchy:
[
  -1,    # 0: pelvis (no parent)
  0,     # 1-3: spine (parent: pelvis)
  0, 0,
  1,     # 4-5: l_leg (parent: spine1)
  1,
  ... (hierarchical)
]

Local vs Global:
- Global: absolute world coordinates (rare)
- Local: relative to parent joint (normal, required for SMPL)
```

---

## Database Schema

### `models` table

```
Column            | Type    | Description
------------------+---------+-------------------------------
id                | INTEGER | PRIMARY KEY
name              | TEXT    | model name (e.g., "hymotion_m2m_v2_overfit_100")
checkpoint_path   | TEXT    | full checkpoint directory path
rotation_space    | TEXT    | "local" or "global"
has_caption       | BOOLEAN | whether model supports text conditioning
epoch             | INTEGER | checkpoint epoch number
```

**Example**:
```
id=44, name="hymotion_m2m_v2_overfit_100", 
checkpoint_path="work_dirs/hymotion_m2m_v2_overfit_100_v2/checkpoint-epoch_2750",
rotation_space="local", has_caption=true, epoch=2750
```

### `eval_runs` table (★ CORE)

```
Column              | Type    | Description
--------------------+---------+-------------------------------
id                  | INTEGER | PRIMARY KEY
model_id            | INTEGER | FOREIGN KEY → models.id
task_id             | TEXT    | "E14", "E15", etc. (evaluation task)
setting             | TEXT    | "M", "L", "keyframe_periodic", etc
timestamp           | TEXT    | import timestamp
num_samples         | INTEGER | number of samples (100)
result_json_path    | TEXT    | path to imported JSON file
notes               | TEXT    | human annotation (optional)
```

**Example**:
```
id=5182, model_id=44, task_id="E14", setting="keyframe_periodic",
timestamp="2026-05-27 17:42:50", num_samples=100,
result_json_path="/tmp/eval_imports/hymotion_m2m_v2_overfit_100__E14_keyframe_periodic.json",
notes="Batch import: all modes"
```

### `sample_results` table (★ DATA)

```
Column          | Type    | Description
----------------+---------+-------------------------------
id              | INTEGER | PRIMARY KEY
eval_run_id     | INTEGER | FOREIGN KEY → eval_runs.id
sample_idx      | INTEGER | index within run (0-99)
prompt_id       | TEXT    | sample identifier (e.g., "motionfix_test_006202")
text_caption    | TEXT    | optional text description
motion_path     | TEXT    | path to NPZ file (★ must exist!)
gen_motion_path | TEXT    | optional generated motion path
num_frames      | INTEGER | motion length (T)
metrics_json    | TEXT    | JSON string containing all metrics
```

**Example**:
```
eval_run_id=5182, sample_idx=0, prompt_id="motionfix_test_006202",
motion_path="work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/motionfix_test_006202.npz",
num_frames=120,
metrics_json='{"mpjpe": 34.338, "mpjre": 12.118, "trans_error_mm": 34.346, "T": 120, "has_text": true}'
```

### `agg_metrics` table

```
Column      | Type    | Description
------------+---------+-------------------------------
id          | INTEGER | PRIMARY KEY
run_id      | INTEGER | FOREIGN KEY → eval_runs.id
metric_name | TEXT    | "mean_mpjpe_mm", "std_mpjpe_mm", etc
value       | REAL    | metric value
```

**Example**:
```
run_id=5182, metric_name="mean_mpjpe_mm", value=36.417
run_id=5182, metric_name="std_mpjpe_mm", value=17.475
run_id=5182, metric_name="mean_mpjre_deg", value=8.898
...
```

### `baselines` table

```
Column          | Type    | Description
----------------+---------+-------------------------------
id              | INTEGER | PRIMARY KEY
name            | TEXT    | baseline name (e.g., "KIMODO", "MoGenDIT")
task_id         | TEXT    | "E14", "E15", etc
setting         | TEXT    | "M", "L", etc
metrics_json    | TEXT    | JSON: {mpjpe: ..., mpjre: ..., ...}
skeleton_type   | TEXT    | "SMPL22", "SOMA77", etc
source          | TEXT    | "paper", "checkpoint", etc
```

---

## API Endpoints

### `/api/smpl/<path:npz_path>` — SMPL Mesh

**Purpose**: Return per-frame SMPL parameters for mesh rendering

**Query Parameters**:
- `rotation_space`: "local" (default) or "global"

**Request**:
```bash
curl "http://localhost:8081/api/smpl/work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/motionfix_test_006202.npz?rotation_space=local"
```

**Response** (200 OK):
```json
{
  "frames": [
    {
      "Rh": [0.1234, -0.0567, 0.0123],
      "Th": [0.0, 1.0, 0.0],
      "poses": [0.123, -0.056, ..., 0.0],
      "shapes": [0, 0, ..., 0],
      "gender": "neutral"
    },
    { ... 119 more frames ... }
  ],
  "num_frames": 120,
  "fps": 30
}
```

**Errors**:
- 404: NPZ file not found at path
- 500: NPZ corrupted or rotation conversion failed

**Caching**:
- Server: LRU cache (1024 entries)
- Browser: ETag-based (mtime-aware)
- Performance: 150-300ms (first load), 1ms (cached)

### `/api/npz/<path:npz_path>` — Skeleton Positions

**Purpose**: Return FK joint positions for skeleton overlay

**Response**:
```json
{
  "positions": [
    [[0.0, 1.0, 0.0], [0.1, 0.9, 0.1], ...],  ← T=120 frames
    ...
  ],
  "edges": [[0, 1], [1, 4], [1, 5], ...],
  "joint_names": ["pelvis", "l_hip", "r_hip", ...],
  "fps": 30
}
```

### `/task/<task_id>` — Task Summary

**Purpose**: Show aggregated metrics, radar chart, sample table

**URL**:
```
http://localhost:8081/task/E14
```

**Page Features**:
- Metric table (all models for this task)
- Radar chart (multi-model comparison)
- Sample browser (click to view 3D)

### `/compare` — Multi-Model Comparison

**Purpose**: Overlay N runs side-by-side

**URL**:
```
http://localhost:8081/compare?runs=5182,5183,5184
```

**Features**:
- Side-by-side 3D viewers
- Metric radar charts
- Statistics comparison

### `/viewer?path=<npz>` — Standalone Viewer

**Purpose**: View any NPZ file without database

**URL**:
```
http://localhost:8081/viewer?path=work_dirs/.../motionfix_test_006202.npz
```

---

## Manual Conversion Guide

If you don't want to use the automated script, here's the manual process:

### Step 1: Load summary.json

```python
import json

with open("work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/summary.json") as f:
    summary = json.load(f)

print(f"Mode: {summary['mode']}")
print(f"Samples: {summary['n_samples']}")
print(f"Mean MPJPE: {summary['mean_mpjpe_mm']:.2f} mm")
print(f"Mean MPJRE: {summary['mean_mpjre_deg']:.2f}°")
```

### Step 2: Create flat JSON structure

```python
flat = {
    "model": "hymotion_m2m_v2_overfit_100",
    "checkpoint": summary["checkpoint"],
    "task_id": "E14",
    "setting": summary["mode"],  # e.g., "keyframe_periodic"
    "aggregated": {
        "mean_mpjpe_mm": summary["mean_mpjpe_mm"],
        "std_mpjpe_mm": summary.get("std_mpjpe_mm"),
        "mean_mpjre_deg": summary["mean_mpjre_deg"],
        "std_mpjre_deg": summary.get("std_mpjre_deg"),
        "mean_trans_error_mm": summary.get("mean_trans_error_mm"),
        "n_samples": summary["n_samples"],
    },
    "per_sample": [
        {
            "prompt_id": s["key"],
            "_npz_path": f"work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/{summary['mode']}/{s['key']}.npz",
            "mpjpe": s.get("mpjpe"),
            "mpjre": s.get("mpjre"),
            "trans_error_mm": s.get("trans_error_mm"),
            "T": s.get("T"),
            "has_text": s.get("has_text", False),
        }
        for s in summary["per_sample"]
    ],
}
```

### Step 3: Verify NPZ files

```bash
# Check one sample
ls -lh work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/motionfix_test_006202.npz

# Verify all exist
for sample in work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/*.npz; do
    if [ ! -f "$sample" ]; then
        echo "MISSING: $sample"
    fi
done
```

### Step 4: Save flat JSON

```python
import json
from pathlib import Path

output_dir = Path("/tmp/eval_imports")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "hymotion_m2m_v2_overfit_100__E14_keyframe_periodic.json"
with open(output_file, 'w') as f:
    json.dump(flat, f, indent=2)

print(f"✓ Saved to {output_file}")
```

### Step 5: Import to database

```bash
cd motion_annot_web/eval_dashboard
python3 data_importer.py import /tmp/eval_imports/hymotion_m2m_v2_overfit_100__E14_keyframe_periodic.json
```

---

## Troubleshooting

### ❌ "404 Not Found" on 3D Viewer

**Symptom**: Clicking "View 3D" shows 404 error

**Cause**: `_npz_path` in database doesn't match actual file

**Diagnosis**:
```bash
# Check if file exists
ls -lh "work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/motionfix_test_006202.npz"

# Check database path
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db << EOF
SELECT motion_path FROM sample_results WHERE eval_run_id = 5182 LIMIT 1;
