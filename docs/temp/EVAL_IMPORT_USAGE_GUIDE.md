# Eval Dashboard - Complete Import & Usage Guide

## Quick Start (5 Minutes)

### 1. Prepare Your Evaluation Data

Make sure you have an `eval_overfit` directory with the following structure:

```
eval_overfit/
├── keyframe_periodic/
│   ├── summary.json
│   ├── motion1.npz
│   ├── motion2.npz
│   └── ... (100 NPZ files)
├── keyframe_pos/
│   ├── summary.json
│   └── ... (100 NPZ files)
├── keyframe_rot/
├── style_edit/
├── text_frame/
├── text_lower/
├── text_upper/
└── trans_only/
```

**Important**: Each mode directory must contain:
- `summary.json`: Aggregated metrics and per-sample metadata
- `*.npz` files: Actual motion data (named to match `summary.json` entries)

### 2. Run the Automated Workflow

```bash
bash full_eval_import_workflow.sh \
    work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
    my_model_name
```

This will:
- ✓ Validate the eval directory
- ✓ Convert all summary.json files to flat JSON format
- ✓ Back up the database
- ✓ Import all evaluation modes to the dashboard

### 3. Start the Dashboard

```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081
```

Then open: **http://localhost:8081/task/E14**

---

## Manual Import (For Advanced Users)

### Convert a Single Mode

```bash
python3 prepare_eval_import.py \
    --summary-json work_dirs/.../keyframe_periodic/summary.json \
    --output-dir ./import_jsons \
    --model-name my_model \
    --mode-dir work_dirs/.../keyframe_periodic \
    --setting keyframe_periodic \
    --task-id E14
```

This creates a flat JSON file ready for import:
```json
{
  "model": "my_model",
  "checkpoint": "work_dirs/.../checkpoint-epoch_2750",
  "task_id": "E14",
  "setting": "keyframe_periodic",
  "aggregated": {
    "mean_mpjpe_mm": 36.42,
    "std_mpjpe_mm": 17.47,
    ...
  },
  "per_sample": [
    {
      "prompt_id": "motion_id_001",
      "_npz_path": "work_dirs/.../keyframe_periodic/motion_id_001.npz",
      "mpjpe": 34.34,
      "mpjre": 12.12,
      "trans_error_mm": 34.35,
      "T": 120,
      "has_text": true
    },
    ...
  ]
}
```

### Import to Dashboard

```bash
cd motion_annot_web/eval_dashboard

# Back up database
cp eval_dashboard.db eval_dashboard.db.bak_$(date +%Y%m%d_%H%M%S)

# Import
python3 data_importer.py import ./import_jsons/my_model__E14_keyframe_periodic.json \
    --notes "Import description"
```

---

## Data Flow

```
evaluation_output/
├── summary.json (nested format)
├── *.npz files (motion data)
    ↓
prepare_eval_import.py
    ↓
flat_import.json (one per mode)
    ↓
data_importer.py
    ↓
eval_dashboard.db (SQLite)
├── models table
├── eval_runs table
├── agg_metrics table
├── sample_results table
    ↓
browser (/task/E14)
    ↓
/api/smpl/<npz_path>
    ↓
Three.js SMPL mesh visualization
```

---

## JSON File Formats

### Input: summary.json (from eval_m2m_v2_all_tasks.py)

```json
{
  "mode": "keyframe_periodic",
  "checkpoint": "work_dirs/.../checkpoint-epoch_2750",
  "n_samples": 100,
  "mean_mpjpe_mm": 36.42,
  "std_mpjpe_mm": 17.47,
  "mean_mpjre_deg": 8.90,
  "std_mpjre_deg": 3.66,
  "mean_trans_error_mm": 35.32,
  "per_sample": [
    {
      "key": "motion_id_001",
      "mpjpe": 34.34,
      "mpjre": 12.12,
      "trans_error_mm": 34.35,
      "T": 120,
      "has_text": true
    },
    ...
  ]
}
```

### Output: flat_import.json (for data_importer.py)

```json
{
  "model": "hymotion_m2m_v2_overfit_100",
  "checkpoint": "work_dirs/.../checkpoint-epoch_2750",
  "task_id": "E14",
  "setting": "keyframe_periodic",
  "aggregated": {
    "mean_mpjpe_mm": 36.42,
    "std_mpjpe_mm": 17.47,
    "mean_mpjre_deg": 8.90,
    "std_mpjre_deg": 3.66,
    "mean_trans_error_mm": 35.32,
    "n_samples": 100
  },
  "per_sample": [
    {
      "prompt_id": "motion_id_001",
      "_npz_path": "/full/path/to/motion_id_001.npz",
      "mpjpe": 34.34,
      "mpjre": 12.12,
      "trans_error_mm": 34.35,
      "T": 120,
      "has_text": true
    },
    ...
  ]
}
```

**Key differences:**
- `aggregated`: Separated from per_sample items
- `per_sample` items have `_npz_path` (full path to NPZ file)
- No nested structure; flat list of metrics

---

## Dashboard Navigation

### Main Pages

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | Task overview + model stats |
| Task Detail | `/task/E14` | Metrics table + 3D viewer + radar chart |
| Compare | `/compare` | Multi-run comparison |
| Data Explorer | `/data` | Raw database tables |
| Viewer | `/viewer` | Standalone NPZ player |

### Task Detail Page (`/task/E14`)

1. **Metrics Table**: Shows all runs for the task, sorted by "BEST MODEL"
2. **3D Viewer**: Click any sample to view 3D SMPL mesh
3. **Radar Chart**: Visual comparison of key metrics across models
4. **Baseline**: Compare against paper baselines (KIMODO, MoGenDIT)

### 3D Viewer Controls

- **Play/Pause**: Spacebar
- **Seek**: Click on timeline
- **Speed**: Slider at bottom
- **Rotation**: Drag mouse
- **Zoom**: Scroll wheel

---

## Troubleshooting

### Issue: "Motion not visible in 3D viewer"

**Cause**: NPZ file path is incorrect or file doesn't exist

**Fix**:
```bash
# Verify NPZ path exists
ls work_dirs/.../keyframe_periodic/motion_id_001.npz

# Check database entry
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \
  "SELECT gen_motion_path FROM sample_results LIMIT 1;"

# Should output the full path to an existing file
```

### Issue: "Models not appearing on dashboard"

**Cause**: Database wasn't backed up before import, or import failed

**Fix**:
```bash
# Restore from backup
cp motion_annot_web/eval_dashboard/eval_dashboard.db.bak_* \
   motion_annot_web/eval_dashboard/eval_dashboard.db

# Check database integrity
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \
  "SELECT COUNT(*) FROM models; SELECT COUNT(*) FROM eval_runs;"

# Re-run import if needed
```

### Issue: "Import appears frozen"

**Cause**: The dashboard is slow when importing large batches

**Fix**:
```bash
# Kill the process and check logs
ps aux | grep data_importer

# Import a single file to test
python3 motion_annot_web/eval_dashboard/data_importer.py \
    import ./import_jsons/my_model__E14_keyframe_periodic.json

# Should complete in < 5 seconds
```

---

## Database Schema (SQLite)

### models table

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | Primary key |
| name | TEXT | Model name (e.g., "hymotion_m2m_v2_overfit_100") |
| checkpoint_path | TEXT | Path to model checkpoint |
| rotation_space | TEXT | "local" or "global" |
| has_caption | BOOLEAN | Whether model is text-conditioned |
| epoch | INTEGER | Training epoch (extracted from checkpoint) |

### eval_runs table

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | Primary key |
| model_id | INTEGER | Foreign key to models |
| task_id | TEXT | Task ID (E14, E15, etc.) |
| setting | TEXT | Setting name (keyframe_periodic, text_frame, etc.) |
| timestamp | TEXT | ISO timestamp |
| num_samples | INTEGER | Number of evaluated samples |
| result_json_path | TEXT | Path to import JSON |
| notes | TEXT | Import notes |

### sample_results table

| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER | Primary key |
| eval_run_id | INTEGER | Foreign key to eval_runs |
| sample_idx | INTEGER | Index in the run (0-99) |
| prompt_id | TEXT | Motion ID (for querying) |
| gen_motion_path | TEXT | **Full path to NPZ file** (used by 3D viewer) |
| num_frames | INTEGER | Duration in frames |
| metrics_json | TEXT | JSON string with all metrics |

---

## Performance Tips

### Optimize For Many Models

If importing 20+ models, disable the Flask cache:

```bash
cd motion_annot_web/eval_dashboard
DISABLE_CACHE=1 python3 app.py --port 8081
```

### Batch Import Multiple Models

```bash
for model_dir in work_dirs/*/eval_overfit; do
    model_name=$(basename $(dirname "$model_dir"))
    bash full_eval_import_workflow.sh "$model_dir" "$model_name"
done
```

### Export Metrics to CSV

```bash
cd motion_annot_web/eval_dashboard
sqlite3 eval_dashboard.db \
  ".mode csv" \
  ".output metrics_export.csv" \
  "SELECT m.name, r.setting, m.mean_mpjpe_mm, m.std_mpjpe_mm \
   FROM eval_runs r JOIN models m ON r.model_id=m.id;"
```

---

## Files Reference

### Scripts (in project root)

- **prepare_eval_import.py**: Convert summary.json → flat JSON
- **full_eval_import_workflow.sh**: Automated 5-step import workflow

### Dashboard (motion_annot_web/eval_dashboard/)

- **app.py**: Flask server with API endpoints
- **data_importer.py**: Database import logic
- **db_manager.py**: SQLite wrapper
- **eval_dashboard.db**: SQLite database (auto-created)

### Generated Files

- **import_jsons_YYYYMMDD_HHMMSS/**: Temporary directory with flat JSON files
- **eval_dashboard.db.bak_***: Database backups

---

## Next Steps

1. ✓ Import your first model using `full_eval_import_workflow.sh`
2. ✓ Verify on dashboard: http://localhost:8081/task/E14
3. ✓ Click samples to view 3D motions
4. ✓ Use `/compare` page to compare multiple models
5. ✓ Customize baselines in `/api/baselines` endpoint

