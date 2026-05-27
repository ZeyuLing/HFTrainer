# Eval Dashboard - Complete Reference

## Overview

The **Eval Dashboard** is a web-based visualization system for motion generation model evaluation results.

### Key Features
- 📊 Metrics comparison across models
- 🎬 Interactive 3D SMPL mesh visualization
- 📈 Radar charts for visual metric comparison
- 🔄 Multi-model side-by-side comparison
- 📁 SQLite database for evaluation runs

---

## Quick Start (3 Steps)

### 1. Import Results
```bash
bash full_eval_import_workflow.sh \
    work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
    my_model_name
```

### 2. Start Server
```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081
```

### 3. Open Browser
**http://localhost:8081/task/E14**

---

## System Architecture

```
Evaluation Results
    ↓
NPZ + summary.json
    ↓
prepare_eval_import.py (convert)
    ↓
Flat JSON Format
    ↓
data_importer.py (import)
    ↓
SQLite Database
    ↓
Flask Server
    ↓
Browser (Three.js)
```

---

## Core Scripts

### prepare_eval_import.py (4.7K)
Converts `summary.json` (nested format) to flat JSON (import format)

```bash
python3 prepare_eval_import.py \
    --summary-json work_dirs/.../summary.json \
    --output-dir ./import_jsons \
    --model-name my_model \
    --mode-dir work_dirs/.../keyframe_periodic \
    --setting keyframe_periodic
```

### full_eval_import_workflow.sh (4.5K)
Automated 5-step workflow: validate → convert → backup → import → verify

```bash
bash full_eval_import_workflow.sh <eval_dir> <model_name>
```

---

## Dashboard Components

### Pages
- `/` - Home (overview)
- `/task/E14` - Task detail (metrics + 3D viewer)
- `/compare` - Multi-model comparison
- `/data` - Database browser
- `/viewer` - Standalone NPZ player

### APIs
- `/api/smpl/<path>` - SMPL mesh frames
- `/api/npz/<path>` - Skeleton positions
- `/api/runs/<id>/samples` - Sample list
- `/api/baselines` - Baseline CRUD

---

## Database Schema

### sample_results Table
```sql
CREATE TABLE sample_results (
    id INTEGER PRIMARY KEY,
    eval_run_id INTEGER,
    sample_idx INTEGER,
    prompt_id TEXT,
    gen_motion_path TEXT,      -- FULL PATH to NPZ file
    num_frames INTEGER,
    metrics_json TEXT
);
```

**Critical**: `gen_motion_path` must be absolute path

### models Table
```sql
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    checkpoint_path TEXT,
    rotation_space TEXT,
    has_caption BOOLEAN,
    epoch INTEGER
);
```

### eval_runs Table
```sql
CREATE TABLE eval_runs (
    id INTEGER PRIMARY KEY,
    model_id INTEGER,
    task_id TEXT,
    setting TEXT,
    timestamp TEXT,
    num_samples INTEGER
);
```

---

## JSON File Formats

### Input: summary.json
```json
{
  "mode": "keyframe_periodic",
  "checkpoint": "work_dirs/.../checkpoint-epoch_2750",
  "n_samples": 100,
  "mean_mpjpe_mm": 36.42,
  "per_sample": [
    {
      "key": "motion_id_001",
      "mpjpe": 34.34,
      "T": 120,
      "has_text": true
    }
  ]
}
```

### Output: flat_import.json
```json
{
  "model": "hymotion_m2m_v2_overfit_100",
  "checkpoint": "work_dirs/.../checkpoint-epoch_2750",
  "task_id": "E14",
  "setting": "keyframe_periodic",
  "aggregated": {
    "mean_mpjpe_mm": 36.42,
    "n_samples": 100
  },
  "per_sample": [
    {
      "prompt_id": "motion_id_001",
      "_npz_path": "/absolute/path/to/motion_id_001.npz",
      "mpjpe": 34.34,
      "T": 120,
      "has_text": true
    }
  ]
}
```

---

## Files Reference

### Scripts (Project Root)
- `prepare_eval_import.py` - JSON converter
- `full_eval_import_workflow.sh` - Workflow automator

### Dashboard (motion_annot_web/eval_dashboard/)
- `app.py` - Flask server (~1100 lines)
- `data_importer.py` - Import logic (~240 lines)
- `db_manager.py` - SQLite wrapper (~600 lines)
- `eval_dashboard.db` - Database (auto-created)

### Documentation (Project Root)
- `README_EVAL_DASHBOARD.md` - This file
- `EVAL_IMPORT_USAGE_GUIDE.md` - Detailed usage
- `START_HERE.md` - Quick start
- `EVAL_DASHBOARD_SETUP_GUIDE.md` - Server setup

---

## Troubleshooting

### Motion Not Visible
```bash
# Check path exists
ls work_dirs/.../motion_id_001.npz

# Check database path
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \
  "SELECT gen_motion_path FROM sample_results LIMIT 1;"
```

### Models Not Appearing
```bash
# Restore backup
cp motion_annot_web/eval_dashboard/eval_dashboard.db.bak_* \
   motion_annot_web/eval_dashboard/eval_dashboard.db

# Check database
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \
  "SELECT COUNT(*) FROM eval_runs;"
```

### Slow Performance
```bash
# Disable cache
DISABLE_CACHE=1 python3 app.py --port 8081

# Add indexes
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db << EOF
CREATE INDEX IF NOT EXISTS idx_eval_runs_task ON eval_runs(task_id, setting);
CREATE INDEX IF NOT EXISTS idx_sample_results_run ON sample_results(eval_run_id);
EOF
```

---

## Database Status

Current state:
- **Models**: 38
- **Eval Runs**: 351
- **Samples**: 83,384+

Test imports verified:
- hymotion_m2m_v2_overfit_100_test_import: 6 runs, 600 samples
- test_model_single: 1 run, 100 samples
- test_workflow: 8 runs, 800 samples

---

## Next Steps

1. Run workflow script: `bash full_eval_import_workflow.sh <dir> <name>`
2. Verify import: `http://localhost:8081/task/E14`
3. Click samples for 3D visualization
4. Use `/compare` for multi-model analysis
5. Add baselines via `/api/baselines`

For detailed workflows, see `EVAL_IMPORT_USAGE_GUIDE.md`

