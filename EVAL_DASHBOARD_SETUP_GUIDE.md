# Motion Annotation Web Eval Dashboard — Complete Setup & Troubleshooting Guide

Last Updated: 2026-05-27
Dashboard URL: http://localhost:8081
Default Port: 8081
Database: motion_annot_web/eval_dashboard/eval_dashboard.db

---

## Quick Start (5 minutes)

### 1. Start the Dashboard Server

```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081
```

Expected output:
```
 * Running on http://127.0.0.1:8081
 * Press CTRL+C to quit
```

Open browser: http://localhost:8081

### 2. Import Evaluation Results

From the work_dirs directory:

```bash
# Navigate to dashboard
cd motion_annot_web/eval_dashboard

# Import evaluation results
python3 data_importer.py import \
    /tmp/eval_imports/hymotion_m2m_v2_overfit_100__E14_M.json \
    --notes "Initial import: keyframe_periodic evaluation, 100 samples"
```

Expected output:
```
Imported: hymotion_m2m_v2_overfit_100__E14_M.json
  - Model: hymotion_m2m_v2_overfit_100
  - Task: E14, Setting: M
  - Samples: 100
  - Mean MPJPE: 36.42 mm
```

### 3. View Results

In the browser, navigate to:
- Task View: /task/E14 — See aggregated metrics + radar chart
- Sample Viewer: Click any sample to view 3D SMPL mesh
- Compare Multiple: /compare — Overlay different models/runs

---

## Converting Evaluation Results: Full Workflow

### Step 1: Prepare Summary JSON -> Flat JSON

The dashboard expects flat JSON format, but evaluation outputs are nested. You must convert them first.

Option A: Manual conversion (recommended for understanding)

```python
import json
from pathlib import Path

# Load summary
summary_path = "work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/summary.json"
with open(summary_path) as f:
    summary = json.load(f)

# Build flat JSON
flat = {
    "model": "hymotion_m2m_v2_overfit_100",
    "checkpoint": summary["checkpoint"],
    "task_id": "E14",
    "setting": "M",
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
            "_npz_path": f"work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/{s['key']}.npz",
            "mpjpe": s.get("mpjpe"),
            "mpjre": s.get("mpjre"),
            "trans_error_mm": s.get("trans_error_mm"),
            "T": s.get("T"),
            "has_text": s.get("has_text", False),
        }
        for s in summary["per_sample"]
    ],
}

# Save
output_path = "/tmp/eval_imports/hymotion_m2m_v2_overfit_100__E14_M.json"
Path("/tmp/eval_imports").mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(flat, f, indent=2)

print(f"Saved to {output_path}")
```

### Step 2: Verify NPZ Files Exist

```bash
# Check one sample
ls -lh work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/motionfix_test_006202.npz

# Count total
find work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic -name "*.npz" | wc -l
```

CRITICAL: NPZ files MUST exist at the paths specified in _npz_path. If missing, 3D viewer will show 404 errors.

### Step 3: Import to Database

```bash
cd motion_annot_web/eval_dashboard

# Single import
python3 data_importer.py import /tmp/eval_imports/hymotion_m2m_v2_overfit_100__E14_M.json

# Batch import (all eval_imports)
for json_file in /tmp/eval_imports/*.json; do
    python3 data_importer.py import "$json_file" --notes "Batch import"
done
```

### Step 4: Verify Import Success

```bash
sqlite3 eval_dashboard.db << EOF
SELECT m.name, COUNT(*) as runs FROM eval_runs r 
JOIN models m ON r.model_id = m.id 
GROUP BY m.name;
EOF
```

---

## Performance & Caching

Server-Side LRU Cache:
- maxsize: 1024 entries
- Cache key: (npz_path, rotation_space, file_mtime)
- First load (miss): 150-300ms
- Repeat loads (hit): 1ms

Browser Cache:
- /api/smpl/... -> ETag-based (mtime-aware)
- /api/npz/... -> 1-day cache

---

## Common Issues & Fixes

### Issue: "404 Not Found" when clicking 3D viewer

Cause: _npz_path in database doesn't match actual file location.

Fix:
```bash
# Verify NPZ exists
ls -lh "work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/keyframe_periodic/motionfix_test_006202.npz"

# Check database _npz_path
sqlite3 eval_dashboard.db << EOF
SELECT sample_idx, value FROM sample_results 
WHERE run_id = 1 AND metric_name = '_npz_path' 
LIMIT 3;
EOF
```

### Issue: Motion looks twisted/broken

Cause: Rotation space mismatch (stored as global, displayed as local).

Fix: Verify NPZ contains rotation_space hint and update import if needed.

### Issue: Skeleton visible but no mesh

Cause: SMPL module import failed.

Fix:
```bash
pip install -e /path/to/hftrainer
```

---

## API Endpoints

### /api/smpl/<path:npz_path> - SMPL Mesh Frames

Returns per-frame SMPL parameters (poses, translation, shape, gender).

Query parameters:
- rotation_space: "local" or "global"

### /api/npz/<path:npz_path> - Skeleton Positions

Returns forward kinematics joint positions for skeleton rendering.

---

## Database Schema

models table:
- id (INTEGER PRIMARY KEY)
- name (TEXT) - model name
- checkpoint_path (TEXT)
- rotation_space (TEXT)
- has_caption (BOOLEAN)
- epoch (INTEGER)

eval_runs table (CORE):
- id (INTEGER PRIMARY KEY)
- model_id (FOREIGN KEY)
- task_id (TEXT) - "E14", "E15", etc
- setting (TEXT) - "M", "L", etc
- num_samples (INTEGER)
- result_json_path (TEXT)

sample_results table:
- run_id (FOREIGN KEY)
- sample_idx (INTEGER)
- metric_name (TEXT)
- value (REAL or TEXT)

CRITICAL: _npz_path must point to actual NPZ files!

---

## Production Checklist

- [ ] Database backed up
- [ ] NPZ files readable from server
- [ ] Flask running on correct port
- [ ] SMPL module installed
- [ ] Database integrity checked
- [ ] Server firewall allows port 8081

---

## Next Steps

1. Convert evaluation results to flat JSON format
2. Verify NPZ files exist at specified paths
3. Import to database using data_importer.py
4. Start Flask server on port 8081
5. View results in browser at /task/E14

See QUICK_START_DATA_FLOW.md for technical details on rotation conversion pipeline.
