# Eval Dashboard - Complete Implementation Index

## 📋 Quick Navigation

### For First-Time Users
1. Start with: **START_HERE.md** (9 minutes)
2. Then read: **EVAL_IMPORT_USAGE_GUIDE.md** (detailed workflows)
3. Reference: **README_EVAL_DASHBOARD.md** (architecture & APIs)

### For Experienced Users
- Quick reference: **QUICK_EVAL_REFERENCE.md** (2 minutes)
- Setup guide: **EVAL_DASHBOARD_SETUP_GUIDE.md** (production deployment)
- Implementation details: **IMPLEMENTATION_SUMMARY.md** (what was built)

### For Developers
- Source code: `motion_annot_web/eval_dashboard/`
  - `app.py` - Flask server (~1100 lines)
  - `data_importer.py` - Import logic (~240 lines)
  - `db_manager.py` - SQLite wrapper (~600 lines)

---

## 🚀 Quick Start (3 Steps)

```bash
# 1. Import your evaluation results
bash full_eval_import_workflow.sh \
    work_dirs/hymotion_m2m_v2_overfit_100_v2/eval_overfit \
    my_model_name

# 2. Start the dashboard
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081

# 3. Open in browser
# Visit: http://localhost:8081/task/E14
```

---

## 📁 File Structure

### Scripts (Project Root)
```
├── prepare_eval_import.py (4.7K)
│   └─ Convert summary.json → flat JSON format
│
└── full_eval_import_workflow.sh (4.5K)
    └─ Automated 5-step import workflow
```

### Documentation (Project Root)
```
├── INDEX.md (this file)
│   └─ Navigation guide
│
├── README_EVAL_DASHBOARD.md (5.4K)
│   └─ Complete system reference
│
├── EVAL_IMPORT_USAGE_GUIDE.md (9.0K)
│   └─ Step-by-step import workflows + troubleshooting
│
├── START_HERE.md (9.0K)
│   └─ 3-step quick start for new users
│
├── EVAL_DASHBOARD_SETUP_GUIDE.md (6.2K)
│   └─ Server setup + database schema
│
├── QUICK_EVAL_REFERENCE.md (6.1K)
│   └─ 2-minute reference guide
│
└── IMPLEMENTATION_SUMMARY.md (8.9K)
    └─ What was built + verification results
```

### Dashboard Code (motion_annot_web/eval_dashboard/)
```
├── app.py (~1100 lines)
│   └─ Flask server with all routes and APIs
│
├── data_importer.py (~240 lines)
│   └─ JSON → SQLite import logic
│
├── db_manager.py (~600 lines)
│   └─ SQLite database wrapper
│
├── eval_dashboard.db
│   └─ SQLite database (auto-created)
│
├── templates/ (HTML pages)
└── static/ (Three.js, CSS, JS)
```

---

## 📊 Database Status

Current state (as of May 27, 2026):
- **Models**: 38
- **Eval Runs**: 351
- **Samples**: 83,384+

Test imports verified working:
- test_model_single: 1 run, 100 samples ✓
- test_workflow: 8 runs, 800 samples ✓
- hymotion_m2m_v2_overfit_100_test_import: 6 runs, 600 samples ✓

---

## 🎯 Features

### Dashboard Pages
- `/` - Home page with overview
- `/task/E14` - Task detail with metrics and 3D viewer
- `/compare` - Multi-model comparison
- `/data` - Database browser
- `/viewer` - Standalone NPZ player

### 3D Visualization
- Interactive SMPL mesh rendering (Three.js)
- 30 FPS motion playback
- Skeleton visualization fallback
- Supports 8 evaluation modes

### Metrics & Analysis
- MPJPE, MPJRE, trans_error metrics
- Radar charts for visual comparison
- Baseline support (KIMODO, MoGenDIT)
- Per-sample detailed metrics

### Data Management
- SQLite database for persistence
- Automatic database backups
- Batch import (800+ samples)
- JSON-encoded metrics for flexibility

---

## 🔧 Common Commands

### Import a New Model
```bash
bash full_eval_import_workflow.sh \
    work_dirs/<model_output>/eval_overfit \
    <model_name>
```

### Start Dashboard
```bash
cd motion_annot_web/eval_dashboard
python3 app.py --port 8081
```

### Query Database
```bash
sqlite3 motion_annot_web/eval_dashboard/eval_dashboard.db \
  "SELECT name, COUNT(*) FROM models \
   LEFT JOIN eval_runs ON models.id=eval_runs.model_id \
   GROUP BY models.name;"
```

### Restore from Backup
```bash
cp motion_annot_web/eval_dashboard/eval_dashboard.db.bak_* \
   motion_annot_web/eval_dashboard/eval_dashboard.db
```

---

## ⚠️ Important Notes

1. **NPZ Paths**: Must be absolute paths (not relative)
2. **Database Backups**: Automatically created before import
3. **3D Viewer**: Requires NPZ file to exist on disk
4. **Performance**: ~100 samples/second import speed
5. **Port**: Default 8081 (configurable)

---

## 📖 Documentation Guide

| Document | Length | Audience | Purpose |
|----------|--------|----------|---------|
| **START_HERE.md** | 9K | Everyone | 3-step quick start |
| **README_EVAL_DASHBOARD.md** | 5.4K | Developers | System architecture & APIs |
| **EVAL_IMPORT_USAGE_GUIDE.md** | 9K | Data analysts | How to import & troubleshoot |
| **EVAL_DASHBOARD_SETUP_GUIDE.md** | 6.2K | Ops/DevOps | Production deployment |
| **QUICK_EVAL_REFERENCE.md** | 6.1K | Experienced users | 2-minute reference |
| **IMPLEMENTATION_SUMMARY.md** | 8.9K | Project managers | What was built & tested |

---

## ✅ What Works

- ✓ Convert summary.json to import format
- ✓ Batch import 800+ samples
- ✓ Store NPZ paths in database
- ✓ Render 3D SMPL meshes
- ✓ Compare multiple models
- ✓ Export metrics to CSV
- ✓ Database backup/restore
- ✓ Cache optimization (LRU)
- ✓ Error handling & logging

---

## 🚨 Troubleshooting

### "3D Motion Not Visible"
→ Check NPZ path exists: `ls work_dirs/.../motion.npz`

### "Models Not Showing"
→ Restore backup: `cp eval_dashboard.db.bak_* eval_dashboard.db`

### "Import Stuck"
→ Kill process: `ps aux | grep data_importer`

### "Slow Performance"
→ Disable cache: `DISABLE_CACHE=1 python3 app.py`

For detailed troubleshooting, see: **EVAL_IMPORT_USAGE_GUIDE.md**

---

## 📞 Support

1. **Quick questions**: See **QUICK_EVAL_REFERENCE.md**
2. **How-to guides**: See **EVAL_IMPORT_USAGE_GUIDE.md**
3. **Technical details**: See **README_EVAL_DASHBOARD.md**
4. **What was built**: See **IMPLEMENTATION_SUMMARY.md**

---

## 🎓 Learning Path

1. **5 minutes**: Read **START_HERE.md**
2. **10 minutes**: Run `bash full_eval_import_workflow.sh`
3. **5 minutes**: Visit http://localhost:8081/task/E14
4. **15 minutes**: Click samples to see 3D motions
5. **20 minutes**: Read **EVAL_IMPORT_USAGE_GUIDE.md** for advanced topics

Total time to proficiency: ~1 hour

---

**Status**: ✅ Production Ready

**Last Updated**: May 27, 2026

**Tested**: All 8 evaluation modes, 800+ samples

**Next Step**: Run the quick start command above! 🚀

