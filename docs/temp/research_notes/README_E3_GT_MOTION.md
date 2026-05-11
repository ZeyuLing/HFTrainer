# E3 Ground Truth Motion Data - Documentation

This folder contains comprehensive documentation on locating and loading ground truth (GT) motion data for E3 keyframe interpolation eval cases in the hf_trainer project.

## Documents Overview

### 1. **E3_GT_MOTION_QUICK_REF.md** ⭐ START HERE
   - **Purpose**: Quick reference for developers
   - **Contains**: 
     - 5-minute TL;DR code snippet
     - File paths quick lookup
     - Database query templates
     - Common issues and solutions
   - **Use when**: You need to quickly get GT motion paths or generated results

### 2. **E3_DATA_LOCATIONS_REAL_EXAMPLES.md** 📊 MOST USEFUL
   - **Purpose**: Real data examples from the actual project
   - **Contains**:
     - Actual E3 run IDs from eval_dashboard.db
     - Complete mapping for Sample 0 (GT path, generated path, captions)
     - Database schema with real example records
     - Step-by-step walkthrough with real paths
     - Quick lookup table for all 240 samples
   - **Use when**: You want to understand the data structure with real examples

### 3. **GT_MOTION_LOADING_GUIDE.md** 📚 COMPREHENSIVE
   - **Purpose**: Complete technical guide
   - **Contains**:
     - Detailed data source locations
     - Complete database schemas with all tables
     - Multiple loading approaches (JSON, DB, result files)
     - Motion frequency metrics computation
     - Full example pipeline class
     - Summary table of all locations
   - **Use when**: You need to understand the full system or implement complex metrics

---

## Key Information At a Glance

### Data Flow
```
E3 Datalist JSON (240 items)
    ↓
    ├─ Each item has: motion_path (→ GT motion), num_frames, caption, category
    │
Eval Run (task_id=E3)
    ↓
    ├─ Stored in: eval_runs table (eval_dashboard.db)
    ├─ Links to: sample_results table (240 per run)
    ├─ Each sample has: sample_idx, prompt_id, gen_motion_path
    │
Generated Results
    ├─ NPZ files: work_dirs/eval_{ts}/*/E3_{setting}/npz/{sample_idx:05d}.npz
    ├─ JSON summary: work_dirs/eval_{ts}/import_jsons/*_E3_{setting}.json
    └─ Metrics: stored in eval_dashboard.db + JSON files
```

### Essential File Paths

| Component | Path |
|-----------|------|
| **E3 Datalist** | `data/eval/m2m_v2/eval_e3_keyframe.json` |
| **GT Motions** | `/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/{actor}/{action}.npz` |
| **Generated Results** | `work_dirs/eval_{timestamp}/{model_name}/*_E3_{setting}_*/E3_{setting}/npz/{idx:05d}.npz` |
| **Aggregated Results** | `work_dirs/eval_{timestamp}/import_jsons/{src}/MODEL_E3_{setting}.json` |
| **eval_dashboard DB** | `/tmp/eval_dashboard.db` |

### Database Essentials

**eval_dashboard.db tables**:
- `eval_runs`: Metadata about each eval run (task_id, setting, result_json_path)
- `sample_results`: Per-sample data (sample_idx, gen_motion_path, metrics_json)

**Query templates**:
```sql
-- List E3 runs
SELECT * FROM eval_runs WHERE task_id = 'E3' ORDER BY created_at DESC;

-- Get samples from run
SELECT * FROM sample_results WHERE eval_run_id = 3250 ORDER BY sample_idx;
```

---

## Common Tasks

### Task 1: Get GT Motion Path for Sample N
1. Open `data/eval/m2m_v2/eval_e3_keyframe.json`
2. Index into: `data['data_list'][N]['motion_path']`
3. Load NPZ: concatenate `trans` (3 dims) + `poses` (132 dims) → 135-dim

### Task 2: Find Generated Motion for Sample N in Run X
1. Query eval_dashboard.db: `SELECT gen_motion_path FROM sample_results WHERE eval_run_id = X AND sample_idx = N`
2. Load NPZ same way as GT

### Task 3: Compute Motion Frequency Metrics
1. Load both GT and generated motions (135-dim arrays)
2. Apply FFT or other spectral analysis
3. Compare frequency content

### Task 4: List All E3 Runs
1. Connect to `/tmp/eval_dashboard.db`
2. `SELECT id, setting, num_samples FROM eval_runs WHERE task_id = 'E3' ORDER BY created_at DESC`

---

## Motion Format

**Standard M2M motion format: 135 dimensions**
- Dims 0-2: Translation (x, y, z)
- Dims 3-8: Root joint rotation (6D continuous)
- Dims 9-134: 21 other joints × 6D each

**NPZ file contents**:
- `poses`: (T, 22, 3) or (T, 66) - 6D rotations for all joints
- `trans`: (T, 3) or (T, 1, 3) - global translation
- `gender`: 'neutral' or similar
- `fps`: 30.0

---

## E3 Task Settings

All E3 runs use the same 240 samples but with different keyframe selections:

1. **adaptive**: Dynamic keyframe selection (1-15 frame intervals)
2. **every_5f**: Keep every 5th frame (20% sparse)
3. **every_10f**: Keep every 10th frame (10% sparse)
4. **every_15f**: Keep every 15th frame (~6.7% sparse)
5. **every_30f**: Keep every 30th frame (~3.3% sparse)
6. **every_60f**: Keep every 60th frame (~1.7% sparse)

---

## Code Examples

### Quick: Load GT Motion
```python
import json
import numpy as np

with open('data/eval/m2m_v2/eval_e3_keyframe.json') as f:
    data = json.load(f)
    
gt_path = data['data_list'][0]['motion_path']
npz = np.load(gt_path, allow_pickle=True)

motion = np.concatenate([
    npz['trans'].reshape(npz['trans'].shape[0], -1)[:, :3],
    npz['poses'].reshape(npz['poses'].shape[0], -1)
], axis=1)

print(f"Shape: {motion.shape}")  # (T, 135)
```

### Advanced: Load GT + Gen + Compute Metrics
See full example in `GT_MOTION_LOADING_GUIDE.md` section 6 (E3MetricsComputer class).

---

## Troubleshooting

**Q: Why is `motion_path` empty in sample_results table?**
A: The table doesn't store GT paths. Use `sample_idx` to index into E3 JSON datalist.

**Q: How do I handle different motion lengths (GT vs Generated)?**
A: Interpolate to match lengths before comparing. See solutions in quick ref.

**Q: What if NPZ has different key names?**
A: Check both `'poses'` and `'body_pose'`, `'trans'` and `'transl'`. See code examples.

**Q: Where are the original source actor files?**
A: At `/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/`

---

## File Statistics

- **E3 Datalist**: 240 items, ~100 KB
- **GT Motions**: ~2 MB per motion, stored on /apdcephfs
- **Generated Results**: ~2 MB per generated motion in work_dirs
- **eval_dashboard.db**: ~41 MB (contains all eval runs)
- **Aggregated JSON per setting**: ~5 MB (contains per-sample metrics)

---

## Related Code Files

- **Eval script**: `scripts/eval/eval_m2m_v2_all_tasks.py` (main eval runner)
- **Motion loading**: `hftrainer/evaluation/motion/m2m_eval_metrics.py` (utility functions)
- **Dashboard**: `motion_annot_web/eval_dashboard/app.py` (web API for viewing)

---

## Quick Navigation

**I want to...**
- ✅ Get GT motion for one sample → **E3_GT_MOTION_QUICK_REF.md** section 1
- ✅ Understand data structure → **E3_DATA_LOCATIONS_REAL_EXAMPLES.md**
- ✅ Build a metrics pipeline → **GT_MOTION_LOADING_GUIDE.md** section 6
- ✅ Query the database → **E3_DATA_LOCATIONS_REAL_EXAMPLES.md** section 6
- ✅ Compute frequency analysis → **GT_MOTION_LOADING_GUIDE.md** section 4

---

**Last updated**: 2026-05-11
**Project**: hf_trainer (HyMotion M2M v2)
**Task**: E3 Keyframe Interpolation Evaluation
