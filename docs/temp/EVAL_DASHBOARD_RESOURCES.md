# Eval Dashboard Resources — Complete Package

## Overview

This package contains everything needed to understand, configure, and deploy the Motion Annotation Web Eval Dashboard for visualizing HyMotion M2M v2 evaluation results.

---

## Documentation Files

### 1. **QUICK_EVAL_REFERENCE.md** (Quick Read)
**Purpose**: 2-minute quick start  
**Contents**:
- TL;DR setup (3 commands)
- Key concepts overview
- API endpoint summary
- Common commands reference
- Troubleshooting matrix

**Use when**: You just want to get started quickly

---

### 2. **EVAL_DASHBOARD_SETUP_GUIDE.md** (Practical Guide)
**Purpose**: Complete step-by-step walkthrough  
**Contents**:
- 5-minute quick start
- Full conversion workflow (Step 1-4)
- NPZ file format specifications
- Database schema documentation
- Common issues & fixes (4 detailed solutions)
- Performance & caching details
- Production checklist

**Use when**: You're setting up the system for the first time

---

### 3. **EVAL_DASHBOARD_COMPLETE_GUIDE.txt** (Reference)
**Purpose**: Comprehensive technical reference  
**Contents**:
- Executive summary
- 3-step quick start
- NPZ file format deep-dive
- Database schema (all tables)
- API endpoints (all routes)
- Data flow diagram
- Manual import process
- Troubleshooting section
- Performance optimization
- Production deployment
- Key files index

**Use when**: You need detailed technical reference or troubleshooting

---

## Existing Documentation (In Dashboard Directory)

### 4. **motion_annot_web/eval_dashboard/CLAUDE.md**
**Original Documentation**
- Project overview (in Chinese)
- 5 Flask applications in the motion_annot_web suite
- Eval dashboard port (8081) and purpose
- Data flow and import process

### 5. **motion_annot_web/eval_dashboard/QUICK_START_DATA_FLOW.md**
**Technical Spec (600+ lines)**
- Complete motion_135 → SMPL conversion pipeline
- 6D rotation reordering explanation ([0,2,4,1,3,5])
- Skeleton positions computation
- SMPL+H padding (22→52 joints)
- Caching strategy (LRU + HTTP ETag)
- Common issues & fixes

### 6. **motion_annot_web/eval_dashboard/EVAL_DASHBOARD_DATA_FLOW.md**
**Detailed Technical Guide (600+ lines)**
- Flask endpoint documentation (/api/npz/, /api/smpl/)
- KIMODO format support
- Complete rotation space conversion
- mtime-based cache invalidation
- File index with line numbers

---

## Scripts & Tools

### 1. **/tmp/prepare_eval_import.py**
**Purpose**: Convert single summary.json to flat JSON  
**Usage**:
```bash
python3 /tmp/prepare_eval_import.py \
    --summary-json work_dirs/.../keyframe_periodic/summary.json \
    --output-dir /tmp/eval_imports \
    --model-name "my_model" \
    --npz-base-path work_dirs/.../eval_overfit
```

**Output**: Flat JSON file ready for import

---

### 2. **/tmp/full_eval_import_workflow.sh** (★ MAIN SCRIPT)
**Purpose**: Complete end-to-end workflow  
**Usage**:
```bash
bash /tmp/full_eval_import_workflow.sh \
    work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
    hymotion_m2m_v2_overfit_100
```

**Does**:
1. Finds all summary.json files
2. Converts each to flat JSON
3. Backs up existing database
4. Imports all to database (one command per mode)

**Output**:
- 8 flat JSON files (one per mode)
- Database imported with 800 samples total
- Backup created: `eval_dashboard.db.bak_<timestamp>`

---

## Database State (Current)

### Current Status
```
Models: 35
Eval Runs: 336
Sample Results: 81,884
Latest Run: E9 (depth & smoothing variations)
```

### Successfully Imported (Test Run)
```
Model: hymotion_m2m_v2_overfit_100_test_import
Model ID: 44
Runs: 8 (one per eval mode)
  - Run 5182: keyframe_periodic (100 samples)
  - Run 5183: keyframe_pos (100 samples)
  - Run 5184: keyframe_rot (100 samples)
  - Run 5185: style_edit (100 samples)
  - Run 5186: text_frame (100 samples)
  - Run 5187: text_lower (100 samples)
  - Run 5188: text_upper (100 samples)
  - Run 5189: trans_only (100 samples)

Total Samples: 800
Backup: eval_dashboard.db.bak_20260527_174247
```

---

## Quick Reference Commands

### Start Dashboard
```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081
# Access: http://localhost:8081
```

### Import Results
```bash
# One mode
python3 data_importer.py import /tmp/eval_imports/model__E14_mode.json

# Batch all modes
for json in /tmp/eval_imports/*.json; do
    python3 data_importer.py import "$json"
done
```

### Database Queries
```bash
# List all models
sqlite3 eval_dashboard.db "SELECT name FROM models LIMIT 10;"

# Count samples by model
sqlite3 eval_dashboard.db \
    "SELECT m.name, COUNT(*) FROM eval_runs r 
     JOIN models m ON r.model_id = m.id GROUP BY m.name;"

# View latest runs
sqlite3 eval_dashboard.db \
    "SELECT r.id, m.name, r.task_id, r.num_samples 
     FROM eval_runs r JOIN models m ON r.model_id = m.id 
     ORDER BY r.id DESC LIMIT 5;"
```

### Verify NPZ Files
```bash
# Check one
ls -lh work_dirs/.../eval_overfit/keyframe_periodic/motionfix_test_006202.npz

# Count all
find work_dirs/.../eval_overfit -name "*.npz" | wc -l

# Check for missing (all should exist)
for npz in $(sqlite3 eval_dashboard.db "SELECT DISTINCT motion_path FROM sample_results LIMIT 100;"); do
    [ ! -f "$npz" ] && echo "MISSING: $npz"
done
```

---

## Data Structure

### Evaluation Directory Layout
```
work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit/
├── keyframe_periodic/
│   ├── summary.json (nested format, 100 samples)
│   └── *.npz (100 motion files)
├── keyframe_pos/
├── keyframe_rot/
├── style_edit/
├── text_frame/
├── text_lower/
├── text_upper/
└── trans_only/

Total: 8 modes × 100 samples = 800 evaluations
```

### Conversion Process
```
summary.json (nested)
    ↓ convert
flat.json (per eval_run)
    ↓ import
eval_dashboard.db (SQLite)
    ↓ query
/api/smpl/<path> (Flask)
    ↓ transform
SMPL parameters (JSON)
    ↓ render
3D mesh (Three.js browser)
```

---

## Key Concepts

### NPZ Format
- **Type**: NumPy compressed array
- **Contents**: motion_135 (135D motion vectors)
- **Structure**: [T, 135] where T = num frames
- **Data**: [3D translation] + [22 joints × 6D rotation]

### Rotation Representation
- **Input**: 6D column-major (motion_135)
- **Output**: 3D axis-angle (SMPL)
- **Key Reordering**: [0, 2, 4, 1, 3, 5]
- **Conversion**: 6D → 3×3 matrix → axis-angle

### SMPL Skeleton
- **Joints**: 22 body parts (SMPL standard)
- **Format**: Axis-angle per joint
- **Hierarchy**: Pelvis root → 21 child joints
- **Padding**: 22 → 52 joints (add hand zeros)

### Caching Strategy
- **Server**: LRU cache, 1024 entries, key=(path, rot_space, mtime)
- **Browser**: ETag-based, 1-day max-age
- **Performance**: 150-300ms (first), 1ms (cached)

---

## Troubleshooting Matrix

| Problem | Cause | Solution |
|---------|-------|----------|
| 404 on 3D viewer | NPZ path invalid | Verify motion_path in database |
| Motion twisted | Rotation space mismatch | Check rotation_space parameter |
| No mesh | SMPL module missing | pip install -e hftrainer |
| Slow loading | Cache miss on large NPZ | Repeat view will be faster |
| Metrics NaN | NPZ corrupted | Verify file integrity |
| Server crash | Module import failed | Check Flask logs |

---

## Testing Verification

### Import Test (Completed)
✓ Workflow script executed successfully  
✓ 8 evaluation modes converted  
✓ 800 samples imported to database  
✓ All NPZ paths verified  
✓ Database backed up before import  
✓ Metrics properly stored in JSON format  
✓ Sample query returned correct data  

### Sample Query Results
```
Run 5182 (keyframe_periodic):
  - prompt_id: motionfix_test_006202
  - mpjpe: 34.338 mm
  - mpjre: 12.118°
  - trans_error_mm: 34.346 mm
  - T: 120 frames
  - has_text: true
```

---

## Recommended Reading Order

1. **First**: Read QUICK_EVAL_REFERENCE.md (5 min)
2. **Then**: Run /tmp/full_eval_import_workflow.sh (2 min)
3. **Next**: Start server and browse /task/E14 (1 min)
4. **Reference**: Use EVAL_DASHBOARD_SETUP_GUIDE.md for troubleshooting
5. **Deep Dive**: Read QUICK_START_DATA_FLOW.md for technical details

---

## File Locations

### Documentation
- QUICK_EVAL_REFERENCE.md → $HOME/
- EVAL_DASHBOARD_SETUP_GUIDE.md → $HOME/
- EVAL_DASHBOARD_COMPLETE_GUIDE.txt → $HOME/
- EVAL_DASHBOARD_RESOURCES.md (this file) → $HOME/

### Scripts
- /tmp/prepare_eval_import.py
- /tmp/full_eval_import_workflow.sh

### Dashboard Code
- motion_annot_web/eval_dashboard/app.py
- motion_annot_web/eval_dashboard/data_importer.py
- motion_annot_web/eval_dashboard/utils.py
- motion_annot_web/eval_dashboard/db_manager.py
- motion_annot_web/eval_dashboard/eval_dashboard.db

---

## Support & Next Steps

### To Start Using Dashboard
```bash
# 1. Import your evaluation results
bash /tmp/full_eval_import_workflow.sh \
    work_dirs/YOUR_EVAL/eval_overfit \
    your_model_name

# 2. Start server
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081

# 3. Open browser
# http://localhost:8081/task/E14
```

### For Troubleshooting
1. Check QUICK_EVAL_REFERENCE.md (common issues)
2. Refer to EVAL_DASHBOARD_SETUP_GUIDE.md (detailed fixes)
3. Read QUICK_START_DATA_FLOW.md (rotation transforms)
4. Inspect Flask logs: python3 app.py 2>&1 | tee flask.log

### For Production Deployment
1. Review EVAL_DASHBOARD_COMPLETE_GUIDE.txt
2. Run production checklist
3. Configure systemd service
4. Set up Nginx reverse proxy
5. Enable SSL certificates

---

## Summary

✓ Complete documentation package provided  
✓ End-to-end workflow script tested  
✓ Database import verified (800 samples)  
✓ Quick reference materials included  
✓ Troubleshooting guide provided  
✓ Technical specifications documented  

**You're ready to use the Eval Dashboard!**
