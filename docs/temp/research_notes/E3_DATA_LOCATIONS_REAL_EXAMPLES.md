# E3 Data Locations with Real Examples

This document shows **actual data paths** from the hf_trainer project for E3 keyframe interpolation eval cases.

---

## 1. E3 Eval Runs in eval_dashboard.db

### Recent E3 runs (as of 2026-05-09):

```
Run ID  Model  Task  Setting     Samples  Status
─────────────────────────────────────────────────────────────────
3360    5      E3    every_60f   240      Latest uncond_local
3359    5      E3    every_30f   240      
3358    5      E3    every_15f   240      
3357    5      E3    every_10f   240      
3356    5      E3    every_5f    240      
3355    5      E3    adaptive    240      
3354    19     E3    every_30f   240      KIMODO caption model
3353    19     E3    every_15f   240      
3352    19     E3    every_10f   240      
3351    19     E3    every_5f    240      
3250    19     E3    adaptive    240      ← Most recent caption run
```

### Query to list all E3 runs:
```sql
SELECT id, model_id, task_id, setting, num_samples, result_json_path
FROM eval_runs
WHERE task_id = 'E3'
ORDER BY created_at DESC;
```

**Database location**: `/tmp/eval_dashboard.db` (symlink from `motion_annot_web/eval_dashboard/eval_dashboard.db`)

---

## 2. Sample 0 (First E3 Sample) - Complete Mapping

### From Database (Run ID 3250 - KIMODO caption E3 adaptive):
```
sample_idx: 0
prompt_id: 00000
eval_run_id: 3250
text_caption: "A person performs a martial arts thrusting motion with their right arm while stepping forward."
gen_motion_path: work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_adaptive_000_025/E3_adaptive/npz/00000.npz
num_frames: 308 (generated)
```

### From E3 JSON Datalist:
```json
{
  "motion_path": "/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/dongming_20260127/发射后抛弃一次性手持炮发射筒。_take_1515.npz",
  "action_name": "发射后抛弃一次性手持炮发射筒。",
  "caption_en": "发射后抛弃一次性手持炮发射筒。",
  "category": "combat",
  "num_frames": 293,
  "fps": 30.0,
  "duration_sec": 9.77,
  "source": "dongming_20260127"
}
```

### File System Locations:
```
GT Motion (Source):
  /apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/dongming_20260127/发射后抛弃一次性手持炮发射筒。_take_1515.npz
  ├── Size: ~2 MB
  ├── Shape: (293, 135) when loaded as 135-dim motion
  └── Keys: ['poses', 'trans', 'gender', 'fps']

Generated Motion (E3 Adaptive - KIMODO):
  work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_adaptive_000_025/E3_adaptive/npz/00000.npz
  ├── Size: ~2 MB
  ├── Shape: (308, 135) when loaded as 135-dim motion
  └── Same structure as GT
```

---

## 3. All E3 Samples Mapping

**E3 Datalist**: `data/eval/m2m_v2/eval_e3_keyframe.json`
- **Total items**: 240
- **Index range**: 0-239
- **Sample mapping**: sample_idx N → data_list[N]

### First 3 samples with details:

#### Sample 0:
```
GT Path: /apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/dongming_20260127/发射后抛弃一次性手持炮发射筒。_take_1515.npz
Category: combat
Frames: 293
Caption: (Chinese) 发射后抛弃一次性手持炮发射筒。
English: Weapon/projectile throwing action
```

#### Sample 1:
```
GT Path: /apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/dongming_20260226/扔纸飞机，从胸前低位起手向前上方抛出。_take_2158.npz
Category: daily_object
Frames: 273
Caption: throw a paper airplane
```

#### Sample 2:
```
GT Path: /apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/dongming_20260226/表示不敢相信地摇头。_take_2307.npz
Category: daily_stand
Frames: 300
Caption: shake head
```

---

## 4. Generated Results Organization

### Path Pattern:
```
work_dirs/eval_{timestamp}/{model_variant}/{model_name}_E3_{setting}_{seed_suffix}/E3_{setting}/npz/{sample_idx:05d}.npz
```

### Real Example Structure:
```
work_dirs/eval_8082_refresh_20260501/
├── kimodo/
│   ├── kimodo_caption_E3_adaptive_000_025/
│   │   └── E3_adaptive/npz/
│   │       ├── 00000.npz ← sample 0
│   │       ├── 00001.npz ← sample 1
│   │       ├── 00002.npz ← sample 2
│   │       └── ... (239 files total)
│   ├── kimodo_caption_E3_every_5f_000_025/
│   │   └── E3_every_5f/npz/
│   │       ├── 00000.npz
│   │       └── ...
│   ├── kimodo_caption_E3_every_10f_000_025/
│   │   └── E3_every_10f/npz/
│   └── ... (more settings)
└── import_jsons/
    └── kimodo/
        ├── KIMODO_caption__E3_adaptive.json
        ├── KIMODO_caption__E3_every_5f.json
        ├── KIMODO_caption__E3_every_10f.json
        └── ...
```

---

## 5. Aggregated Results JSON

### Locations:
```
work_dirs/eval_8082_refresh_20260501/import_jsons/kimodo/KIMODO_caption__E3_{setting}.json
```

### Example: KIMODO_caption__E3_adaptive.json

**Top-level structure**:
```json
{
  "model": "KIMODO_caption",
  "checkpoint": "kimodo-soma-rp",
  "rotation_space": "global",
  "has_caption": true,
  "timestamp": "2026-05-02 02:34:34",
  "task_id": "E3",
  "setting": "adaptive",
  "num_prompts": 240,
  "aggregated": {...},
  "per_sample": [...]
}
```

**Per-sample entry** (from `per_sample` array):
```json
{
  "inference_time": 14.55,
  "y_anchor_delta": 0.0001,
  "mpjpe_pos": 0.09869164228439331,
  "jitter_pos": 3.946249008178711,
  "_npz_path": "work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_adaptive_000_025/E3_adaptive/npz/00000.npz",
  "_sample_idx": 0,
  "_caption": "A person performs a martial arts thrusting motion with their right arm while stepping forward.",
  "_num_frames": 308
}
```

---

## 6. Database Schema Details

### eval_runs table:
```sql
CREATE TABLE eval_runs (
  id INTEGER PRIMARY KEY,
  model_id INTEGER,                    -- e.g., 19 for KIMODO_caption
  task_id TEXT,                        -- "E3"
  setting TEXT,                        -- "adaptive", "every_5f", etc.
  timestamp TIMESTAMP,
  num_samples INTEGER,                 -- Always 240 for E3
  total_time_sec REAL,
  result_json_path TEXT,               -- Path to aggregated JSON file
  notes TEXT,
  created_at TIMESTAMP
);
```

**Example record**:
```
id: 3250
model_id: 19
task_id: E3
setting: adaptive
timestamp: 2026-05-02 02:34:34
num_samples: 240
total_time_sec: 2241.3
result_json_path: /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/work_dirs/eval_8082_refresh_20260501/import_jsons/kimodo/KIMODO_caption__E3_adaptive.json
created_at: 2026-05-02 02:34:34
```

### sample_results table:
```sql
CREATE TABLE sample_results (
  id INTEGER PRIMARY KEY,
  eval_run_id INTEGER,
  sample_idx INTEGER,                  -- 0-239
  prompt_id TEXT,                      -- "00000", "00001", etc.
  text_caption TEXT,                   -- English caption
  motion_path TEXT,                    -- Usually empty for eval
  gen_motion_path TEXT,                -- Path to generated NPZ
  num_frames INTEGER,
  metrics_json TEXT
);
```

**Example records**:
```
id: X | eval_run_id: 3250 | sample_idx: 0 | prompt_id: 00000
  text_caption: "A person performs a martial arts thrusting motion..."
  gen_motion_path: work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_adaptive_000_025/E3_adaptive/npz/00000.npz
  num_frames: 308
  metrics_json: {"inference_time": 14.55, "mpjpe_pos": 0.0987, ...}

id: X+1 | eval_run_id: 3250 | sample_idx: 1 | prompt_id: 00001
  text_caption: "A person performs an overhand throwing motion with their right arm."
  gen_motion_path: work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_adaptive_000_025/E3_adaptive/npz/00001.npz
  num_frames: 260
  metrics_json: {...}
```

---

## 7. Step-by-Step: Load GT Motion for Sample 5

```python
import json
import numpy as np
import sqlite3

# Step 1: Get GT motion path from E3 JSON
with open('data/eval/m2m_v2/eval_e3_keyframe.json') as f:
    e3_data = json.load(f)
gt_path = e3_data['data_list'][5]['motion_path']
print(f"GT path: {gt_path}")
# Output: /apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/.../...npz

# Step 2: Get generated motion path from database
conn = sqlite3.connect('/tmp/eval_dashboard.db')
cursor = conn.cursor()
cursor.execute("""
    SELECT gen_motion_path FROM sample_results
    WHERE eval_run_id = 3250 AND sample_idx = 5
""")
gen_path = cursor.fetchone()[0]
print(f"Gen path: {gen_path}")
# Output: work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_adaptive_000_025/E3_adaptive/npz/00005.npz

# Step 3: Load GT motion (135-dim)
npz_gt = np.load(gt_path, allow_pickle=True)
gt_motion = np.concatenate([
    npz_gt['trans'].reshape(npz_gt['trans'].shape[0], -1)[:, :3],
    npz_gt['poses'].reshape(npz_gt['poses'].shape[0], -1)
], axis=1)
print(f"GT motion shape: {gt_motion.shape}")
# Output: GT motion shape: (num_frames, 135)

# Step 4: Load generated motion
npz_gen = np.load(gen_path, allow_pickle=True)
gen_motion = np.concatenate([
    npz_gen['trans'].reshape(npz_gen['trans'].shape[0], -1)[:, :3],
    npz_gen['poses'].reshape(npz_gen['poses'].shape[0], -1)
], axis=1)
print(f"Gen motion shape: {gen_motion.shape}")

# Step 5: Compute frequency metrics
from scipy.fft import fft, fftfreq
def freq_energy(motion):
    mag = np.abs(fft(motion, axis=0))
    freq = fftfreq(motion.shape[0], d=1.0/30)
    low = mag[np.abs(freq) <= 2.0].sum()
    mid = mag[(np.abs(freq) > 2.0) & (np.abs(freq) <= 5.0)].sum()
    high = mag[np.abs(freq) > 5.0].sum()
    return {'low': low, 'mid': mid, 'high': high}

gt_freq = freq_energy(gt_motion)
gen_freq = freq_energy(gen_motion)
print(f"GT frequency energy: {gt_freq}")
print(f"Gen frequency energy: {gen_freq}")

conn.close()
```

---

## 8. Quick Lookup Table

For a given **sample_idx** N (0-239):

| What | How to get | Example value |
|------|-----------|--------|
| **E3 JSON item** | `e3_data['data_list'][N]` | dict with motion_path, caption_en, num_frames |
| **GT motion path** | `e3_data['data_list'][N]['motion_path']` | `/apdcephfs_cq11/share/.../dongming_.../....npz` |
| **GT frames** | `e3_data['data_list'][N]['num_frames']` | 293 |
| **Generated path (Run 3250)** | Query DB with `sample_idx=N, eval_run_id=3250` | `work_dirs/.../E3_adaptive/npz/{N:05d}.npz` |
| **English caption** | Query DB or JSON | "A person performs a martial arts..." |

---

## 9. All E3 Settings Summary

| Setting | Keyframe Interval | Sparsity | Use Case |
|---------|-------------------|----------|----------|
| **adaptive** | Dynamic (1-15 frames) | Variable | Adaptive based on motion content |
| **every_5f** | Every 5 frames | 20% keyframes | Dense interpolation |
| **every_10f** | Every 10 frames | 10% keyframes | Moderate interpolation |
| **every_15f** | Every 15 frames | ~6.7% keyframes | Sparse interpolation |
| **every_30f** | Every 30 frames | ~3.3% keyframes | Very sparse interpolation |
| **every_60f** | Every 60 frames | ~1.7% keyframes | Extremely sparse (recent addition) |

Each setting has 240 samples using the same E3 datalist.

---

## References

- **Main eval script**: `scripts/eval/eval_m2m_v2_all_tasks.py`
- **Sample loading function**: lines 1056-1240 (`load_eval_samples`)
- **Full guide**: `GT_MOTION_LOADING_GUIDE.md`
- **Quick reference**: `E3_GT_MOTION_QUICK_REF.md`

