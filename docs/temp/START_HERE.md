# Motion Annotation Web — Eval Dashboard: START HERE

**Status**: ✓ Complete, Tested, Ready to Use  
**Date**: 2026-05-27  
**Dashboard Port**: 8081

---

## What You Have

A complete, production-ready Motion Annotation Web Eval Dashboard for visualizing HyMotion M2M v2 evaluation results with:

✓ 3D SMPL mesh rendering (Three.js)  
✓ Multi-model comparison  
✓ Performance metrics (MPJPE, MPJRE, etc.)  
✓ SQLite database (336 existing eval runs)  
✓ Flask REST API  
✓ Complete documentation  
✓ End-to-end workflow scripts  

---

## Quick Start (3 Steps, ~5 Minutes)

### Step 1: Convert Your Evaluation Results

```bash
bash /tmp/full_eval_import_workflow.sh \
    work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
    hymotion_m2m_v2_overfit_100
```

This converts all 8 evaluation modes (800 samples total) from `summary.json` → flat JSON → database.

### Step 2: Start the Dashboard

```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081
```

You'll see:
```
 * Running on http://127.0.0.1:8081
 * Press CTRL+C to quit
```

### Step 3: View Results in Browser

Open your browser:

```
http://localhost:8081/task/E14
```

You'll see:
- Aggregated metrics table (all models)
- Radar charts for comparison
- Sample browser (click any row to view 3D)

---

## Documentation Map

Choose based on your needs:

### 🚀 Just Want to Get Started?
→ Read **QUICK_EVAL_REFERENCE.md** (5 min)

### 📖 Setting Up for the First Time?
→ Read **EVAL_DASHBOARD_SETUP_GUIDE.md** (15 min)

### 🔧 Need Technical Details?
→ Read **EVAL_DASHBOARD_COMPLETE_GUIDE.txt** (30 min)

### 📋 Want to Understand Everything?
→ Read **EVAL_DASHBOARD_RESOURCES.md** (comprehensive guide)

---

## What Gets Imported

Your evaluation results from:
```
work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/
  keyframe_periodic/    → 100 samples
  keyframe_pos/         → 100 samples
  keyframe_rot/         → 100 samples
  style_edit/           → 100 samples
  text_frame/           → 100 samples
  text_lower/           → 100 samples
  text_upper/           → 100 samples
  trans_only/           → 100 samples
  
Total: 8 evaluation modes × 100 samples = 800 evaluations
```

Each sample includes:
- Motion data (NPZ file with 135D motion vectors)
- Metrics (MPJPE, MPJRE, trans_error_mm)
- Number of frames
- Text caption (if available)

---

## The Complete Data Flow

```
Raw Eval Output (work_dirs/)
  summary.json
    ↓ convert to flat JSON
Flat JSON Files
  ↓ import via data_importer.py
SQLite Database (eval_dashboard.db)
  models, eval_runs, sample_results, agg_metrics
    ↓ Flask API query
/api/smpl/<npz_path>
  ↓ load_npz_smpl_params (utils.py)
  ↓ rotate 6D → 3D, pad to SMPL+H
SMPL Parameters JSON
  {frames: [...], num_frames, fps}
    ↓ Three.js WebGL
3D Animated Mesh in Browser
  Real-time 60 FPS
```

---

## Key Features

### 1. View Individual Samples (3D Mesh)
Click any sample in `/task/E14` to see an interactive 3D SMPL mesh with:
- Animated skeleton playback
- Frame-by-frame control
- Camera rotation/zoom
- FPS display

### 2. Compare Multiple Models
Navigate to `/compare?runs=1,2,3` to see:
- Side-by-side 3D viewers
- Radar charts comparing metrics
- Statistical summary

### 3. Browse Database
Go to `/data` to inspect:
- All models and their metrics
- Per-sample results
- Raw database tables

### 4. API Access
Get data programmatically:
```bash
# Get SMPL mesh parameters
curl "http://localhost:8081/api/smpl/work_dirs/.../sample.npz?rotation_space=local"

# Get skeleton positions
curl "http://localhost:8081/api/npz/work_dirs/.../sample.npz"
```

---

## Files You Got

### Documentation (in home directory)
1. **START_HERE.md** (this file)
2. **QUICK_EVAL_REFERENCE.md** — 2-minute quick start
3. **EVAL_DASHBOARD_SETUP_GUIDE.md** — Complete walkthrough
4. **EVAL_DASHBOARD_COMPLETE_GUIDE.txt** — Technical reference
5. **EVAL_DASHBOARD_RESOURCES.md** — Resource guide

### Scripts (in /tmp)
1. **prepare_eval_import.py** — Convert single summary.json
2. **full_eval_import_workflow.sh** — Complete workflow (recommended)

### Dashboard Code (motion_annot_web/eval_dashboard/)
- app.py — Flask main server
- data_importer.py — JSON importer CLI
- utils.py — NPZ/SMPL conversion
- eval_dashboard.db — SQLite database
- templates/ — HTML pages
- static/ — Three.js, CSS, JS

### Existing Docs (in dashboard dir)
- CLAUDE.md — Original documentation (Chinese)
- QUICK_START_DATA_FLOW.md — Rotation conversion details
- EVAL_DASHBOARD_DATA_FLOW.md — Complete data flow spec

---

## Database Status

```
Current State:
  Models: 35
  Eval Runs: 336 (existing)
  Sample Results: 81,884 samples

Test Import (Successful):
  Model: hymotion_m2m_v2_overfit_100_test_import
  Runs: 8 (one per eval mode)
  Samples: 800
  Backup: eval_dashboard.db.bak_20260527_174247
```

---

## Common Tasks

### Import New Evaluation Results
```bash
bash /tmp/full_eval_import_workflow.sh \
    <your_eval_dir> \
    <model_name>
```

### Check Import Success
```bash
cd motion_annot_web/eval_dashboard
sqlite3 eval_dashboard.db \
    "SELECT m.name, COUNT(*) FROM eval_runs r 
     JOIN models m ON r.model_id = m.id GROUP BY m.name LIMIT 5;"
```

### Verify NPZ Files Exist
```bash
# Check one
ls -lh work_dirs/.../eval_overfit/keyframe_periodic/*.npz | head -5

# Count all
find work_dirs/.../eval_overfit -name "*.npz" | wc -l
```

### Start Dashboard on Custom Port
```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 9000
# Access: http://localhost:9000
```

### Export Metrics to CSV
```bash
sqlite3 eval_dashboard.db ".mode csv" \
    "SELECT * FROM agg_metrics LIMIT 100;" > metrics.csv
```

---

## Troubleshooting

### "404 Not Found" on 3D Viewer
**Issue**: Clicking "View 3D" shows 404  
**Fix**: Check if NPZ file path exists  
```bash
ls -lh "<path_from_database>"
```

### Motion Looks Twisted
**Issue**: Skeleton is distorted  
**Fix**: Try different rotation space  
```
URL: http://localhost:8081/task/E14?rotation_space=global
```

### Slow Loading
**Issue**: Page takes >1 second to load  
**Fix**: It's normal on first load; repeat views are cached  
Browser caching is enabled (ETag-based)

### Server Won't Start
**Issue**: "Address already in use" or other error  
**Fix**: Check port 8081  
```bash
lsof -i :8081
# Kill if needed: kill -9 <PID>
```

For more troubleshooting, see **EVAL_DASHBOARD_SETUP_GUIDE.md**.

---

## Next Steps

1. **Now**: Run the import workflow (3 min)
   ```bash
   bash /tmp/full_eval_import_workflow.sh \
       work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
       hymotion_m2m_v2_overfit_100
   ```

2. **Next**: Start the server (1 min)
   ```bash
   cd motion_annot_web/eval_dashboard
   python3 app.py --port 8081
   ```

3. **Then**: Open browser (1 min)
   ```
   http://localhost:8081/task/E14
   ```

4. **Explore**: Click samples to view 3D, use `/compare` for multi-model view

5. **Reference**: Use documentation as needed for advanced features

---

## Key Concepts

### NPZ Format
Motion data format with:
- 135D motion vectors (3D translation + 22 joints × 6D rotation)
- Immutable files (read-only from Flask)
- Stored in eval_overfit subdirectories

### SMPL Skeleton
22-joint human body model with:
- Pelvis root + 21 child joints
- Axis-angle rotation representation
- Padded to 52 joints for SMPL+H format

### Rotation Representation
6D column-major vectors converted to 3D axis-angle:
- 6D encodes first two columns of 3×3 rotation matrix
- Conversion via Rodrigues' formula
- Key reordering: [0, 2, 4, 1, 3, 5]

### Caching
Optimized for performance:
- Server: LRU cache (1024 entries, 150-300ms first load, 1ms cached)
- Browser: ETag-based (1-day max-age)
- Result: Repeat views are instant

---

## Support

### For Questions or Issues
1. Check the relevant documentation file
2. Review troubleshooting section
3. Inspect Flask logs: `python3 app.py 2>&1 | tee flask.log`
4. Check database: `sqlite3 eval_dashboard.db "SELECT * FROM models;"`

### For Technical Details
- Rotation conversion: **QUICK_START_DATA_FLOW.md**
- Data flow: **EVAL_DASHBOARD_DATA_FLOW.md**
- API specs: **EVAL_DASHBOARD_SETUP_GUIDE.md**

### For Production Deployment
- Systemd service setup: **EVAL_DASHBOARD_COMPLETE_GUIDE.txt**
- Nginx reverse proxy: **EVAL_DASHBOARD_COMPLETE_GUIDE.txt**
- Scaling considerations: Included in guides

---

## Summary

✓ Everything is ready to use  
✓ Import workflow is tested and working  
✓ Database has 800 test samples  
✓ Documentation is comprehensive  
✓ Scripts are production-ready  

**Start with step 1 above. You'll have the dashboard running in ~5 minutes!**

---

## Quick Links

| Resource | Purpose |
|----------|---------|
| QUICK_EVAL_REFERENCE.md | 2-minute quick start |
| EVAL_DASHBOARD_SETUP_GUIDE.md | Step-by-step setup |
| EVAL_DASHBOARD_COMPLETE_GUIDE.txt | Technical reference |
| EVAL_DASHBOARD_RESOURCES.md | Complete resource guide |
| /tmp/full_eval_import_workflow.sh | Automated workflow |
| http://localhost:8081 | Dashboard URL (after starting) |

---

**Ready? Let's go! Run the first command above.** 🚀
