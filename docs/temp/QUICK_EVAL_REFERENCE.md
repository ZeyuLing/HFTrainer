# Eval Dashboard Quick Reference

## TL;DR — Get Running in 2 Minutes

```bash
# 1. Convert & import evaluation results
bash /tmp/full_eval_import_workflow.sh \
    work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
    hymotion_m2m_v2_overfit_100

# 2. Start dashboard
cd motion_annot_web/eval_dashboard && python3 app.py --port 8081

# 3. Open browser
# http://localhost:8081/task/E14
```

---

## Key Concepts

### What is eval_dashboard?

A Flask web app that:
- Stores evaluation metrics in SQLite database
- Serves NPZ motion files as JSON (SMPL parameters + skeleton)
- Renders 3D SMPL meshes in browser (Three.js)
- Compares multiple models/tasks with radar charts

### Port

- Default: **8081**
- Start: `python3 app.py --port 8081`
- Access: `http://localhost:8081`

### Database

- SQLite: `motion_annot_web/eval_dashboard/eval_dashboard.db`
- 5 tables: models, eval_runs, agg_metrics, sample_results, baselines
- Core table: **eval_runs** — one row per (model, task, setting)

### NPZ Files

Motion data format:
- Shape: `(T, 135)` where T = num frames
- Contents: `[tx, ty, tz, rot6d_j0[6], rot6d_j1[6], ..., rot6d_j21[6]]`
- Stored in eval_overfit dirs: `work_dirs/.../eval_overfit/{mode}/*.npz`

### Data Flow

```
summary.json                   (nested, model-level)
    ↓ (convert)
flat.json                      (per eval_run)
    ↓ (data_importer.py)
eval_dashboard.db
    ↓ (frontend query)
/api/smpl/<path>               (Flask endpoint)
    ↓ (load_npz_smpl_params)
SMPL poses + translation       (JSON)
    ↓ (Three.js)
3D mesh in browser
```

---

## Import Workflow

### Step 1: Prepare Flat JSON

Convert nested summary.json → flat JSON for each eval mode:

```bash
python3 << 'PYEOF'
import json

# Load summary
with open("work_dirs/.../eval_overfit/keyframe_periodic/summary.json") as f:
    summary = json.load(f)

# Create flat JSON
flat = {
    "model": "hymotion_m2m_v2_overfit_100",
    "checkpoint": summary["checkpoint"],
    "task_id": "E14",
    "setting": "keyframe_periodic",
    "aggregated": {
        "mean_mpjpe_mm": summary["mean_mpjpe_mm"],
        "mean_mpjre_deg": summary["mean_mpjre_deg"],
        "n_samples": summary["n_samples"],
    },
    "per_sample": [
        {
            "prompt_id": s["key"],
            "_npz_path": f"work_dirs/.../eval_overfit/keyframe_periodic/{s['key']}.npz",
            "mpjpe": s.get("mpjpe"),
            "mpjre": s.get("mpjre"),
            "trans_error_mm": s.get("trans_error_mm"),
            "T": s.get("T"),
        }
        for s in summary["per_sample"]
    ],
}

with open("/tmp/flat.json", 'w') as f:
    json.dump(flat, f)
PYEOF
```

**Key fields**:
- `prompt_id` — sample name
- `_npz_path` — absolute path to NPZ file (MUST EXIST)
- `mpjpe`, `mpjre` — metrics
- Aggregated: summary stats

### Step 2: Verify NPZ Files

```bash
# Check a sample
ls -lh work_dirs/.../eval_overfit/keyframe_periodic/motionfix_test_006202.npz

# Count all
find work_dirs/.../eval_overfit -name "*.npz" | wc -l
```

### Step 3: Import to Database

```bash
cd motion_annot_web/eval_dashboard
python3 data_importer.py import /tmp/flat.json --notes "My eval run"
```

### Step 4: Verify

```bash
sqlite3 eval_dashboard.db "SELECT COUNT(*) FROM eval_runs;"
```

---

## Pages & URLs

| URL | Purpose |
|-----|---------|
| `/` | Homepage (task overview) |
| `/task/E14` | Task detail (metrics table + radar + samples) |
| `/compare?runs=1,2,3` | Multi-model comparison |
| `/viewer?path=<npz>` | Standalone NPZ viewer |
| `/data` | Database browser |
| `/import` | Web upload for JSON |

---

## API Endpoints

### `/api/smpl/<path>` — SMPL Parameters

Returns per-frame poses + translations for mesh rendering.

```bash
curl "http://localhost:8081/api/smpl/work_dirs/.../keyframe_periodic/motionfix_test_006202.npz?rotation_space=local"
```

Response:
```json
{
  "frames": [
    {"Rh": [rx, ry, rz], "Th": [tx, ty, tz], "poses": [...], "shapes": [...], "gender": "neutral"},
    ...
  ],
  "num_frames": 120,
  "fps": 30
}
```

### `/api/npz/<path>` — Skeleton Positions

Returns FK joint positions for skeleton rendering.

```bash
curl "http://localhost:8081/api/npz/work_dirs/.../keyframe_periodic/motionfix_test_006202.npz?rotation_space=local"
```

Response:
```json
{
  "positions": [[[x, y, z], ...], ...],
  "edges": [[0, 1], [1, 4], ...],
  "joint_names": ["pelvis", "l_hip", ...],
  "fps": 30
}
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| 404 on 3D view | NPZ path doesn't exist | Check `_npz_path` in DB |
| Motion looks twisted | Rotation space mismatch | Verify `rotation_space=local` |
| No mesh, only skeleton | SMPL import failed | `pip install -e hftrainer` |
| Slow page load | Cache miss on large NPZ | Repeat view will be fast (cached) |
| Metrics show NaN | Evaluation incomplete | Check NPZ file integrity |

---

## Performance

- Server cache: LRU, 1024 entries, key=(path, rot_space, mtime)
- Browser cache: ETag-based, 1-day max-age
- First load: 150-300ms
- Repeat load: ~1ms

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 1100 | Flask routes + API |
| `data_importer.py` | 241 | JSON import CLI |
| `db_manager.py` | 271 | SQLite CRUD |
| `utils.py` | varies | NPZ conversion (FK, SMPL) |
| `eval_dashboard.db` | N/A | SQLite database |

---

## Common Commands

```bash
# Start server
cd motion_annot_web/eval_dashboard && python3 app.py --port 8081

# Import results
python3 data_importer.py import /tmp/flat.json

# Check database
sqlite3 eval_dashboard.db "SELECT * FROM models LIMIT 5;"

# Count samples
sqlite3 eval_dashboard.db "SELECT COUNT(*) FROM sample_results;"

# Verify NPZ
python3 << 'PYEOF'
import numpy as np
data = np.load("work_dirs/.../keyframe_periodic/sample.npz")
print(f"Keys: {list(data.keys())}")
print(f"motion_135 shape: {data['motion_135'].shape}")
PYEOF
```

---

## Next Steps

1. Run conversion script for each eval mode
2. Import all flat JSONs to database
3. Start Flask server
4. Navigate to `/task/E14` to view results
5. Click samples to see 3D SMPL meshes

See `EVAL_DASHBOARD_SETUP_GUIDE.md` for detailed walkthrough and troubleshooting.
