# HF Trainer: E3 Keyframe Interpolation Ground Truth Motion Loading Guide

## Overview
For E3 keyframe interpolation eval cases, the ground truth (source) motion data is stored in separate location from the generated results. This guide explains how to locate and load GT motions for computing motion frequency metrics.

---

## 1. Data Source Locations

### 1.1 E3 Eval Dataset Definition
**File**: `data/eval/m2m_v2/eval_e3_keyframe.json` (primary)
- Also available: `eval_e3_keyframe_v2.json`, `eval_e3_keyframe_v2_rewritten.json`, `eval_e3_keyframe_rewritten.json`

**Schema of each data item**:
```json
{
  "motion_path": "/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/dongming_20260127/..._take_XXXX.npz",
  "action_name": "Action description (Chinese)",
  "caption_en": "English caption",
  "category": "combat|daily_object|...",
  "num_frames": 293,
  "fps": 30.0,
  "duration_sec": 9.77,
  "source": "dongming_20260127"
}
```

### 1.2 Ground Truth Motion File Paths
The `motion_path` field contains **absolute paths** to the original NPZ motion files:
```
/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/{source_actor}/{action_name}_{take_id}.npz
```

These are the **unmodified, ground truth motions** used as the source for E3 keyframe interpolation.

### 1.3 Generated Results Location
Generated outputs are stored in `work_dirs/`:
```
work_dirs/eval_{timestamp}/{model_name}/{model_variant}_E3_{setting}_000_{suffix}/E3_{setting}/npz/{sample_idx:05d}.npz
```

Examples:
- `work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_adaptive_000_025/E3_adaptive/npz/00000.npz`
- `work_dirs/eval_8082_refresh_20260501/kimodo/kimodo_caption_E3_every_10f_000_025/E3_every_10f/npz/00001.npz`

---

## 2. Database Locations and Schemas

### 2.1 eval_dashboard.db Schema
**Location**: `motion_annot_web/eval_dashboard/eval_dashboard.db` (symlink to `/tmp/eval_dashboard.db`)

**Key Tables**:

#### `eval_runs` - Evaluation Run Metadata
```sql
CREATE TABLE eval_runs (
  id INTEGER PRIMARY KEY,
  model_id INTEGER,          -- Model variant ID
  task_id TEXT,              -- "E3"
  setting TEXT,              -- "adaptive", "every_10f", "every_15f", "every_30f", "every_5f"
  timestamp TIMESTAMP,
  num_samples INTEGER,       -- Number of samples (240 for E3)
  total_time_sec REAL,
  result_json_path TEXT,     -- Path to aggregated results JSON
  notes TEXT,
  created_at TIMESTAMP
);
```

**Example Query**:
```sql
SELECT id, model_id, task_id, setting, num_samples, result_json_path
FROM eval_runs
WHERE task_id = 'E3'
ORDER BY created_at DESC
LIMIT 5;
```

#### `sample_results` - Per-Sample Results
```sql
CREATE TABLE sample_results (
  id INTEGER PRIMARY KEY,
  eval_run_id INTEGER,
  sample_idx INTEGER,        -- 0-239 for E3
  prompt_id TEXT,            -- "00000", "00001", etc.
  text_caption TEXT,
  motion_path TEXT,          -- ⚠️ Usually empty in this table
  gen_motion_path TEXT,      -- Path to generated NPZ
  num_frames INTEGER,
  metrics_json TEXT          -- Per-sample metrics
);
```

**Note**: The `motion_path` field is typically empty. Instead, use `prompt_id` to index into the E3 JSON datalist.

### 2.2 score_m2m.db Schema
**Location**: `motion_annot_web/score_m2m/score_m2m.db`

This is a different database used for human annotation scoring, not directly useful for GT motion lookup (it stores human scores, not source motion paths).

---

## 3. How to Load GT Motion for Each E3 Sample

### 3.1 Loading Process (Python)

```python
import json
import os
import numpy as np
from pathlib import Path

# Step 1: Load the E3 datalist JSON
EVAL_DATA_DIR = 'data/eval/m2m_v2'
E3_JSON = os.path.join(EVAL_DATA_DIR, 'eval_e3_keyframe.json')

with open(E3_JSON) as f:
    e3_data = json.load(f)

data_list = e3_data['data_list']

# Step 2: For a given sample index (0-239), get the corresponding item
sample_idx = 0  # Example: first sample
item = data_list[sample_idx]

# Step 3: Extract the motion path (absolute path)
motion_path = item['motion_path']
print(f"Sample {sample_idx} GT motion: {motion_path}")

# Step 4: Load the NPZ file
npz_data = np.load(motion_path, allow_pickle=True)

# The motion data is stored in one of these keys:
# 'poses', 'trans' (or 'transl'), 'gender', 'fps'
poses = npz_data['poses']              # (T, 66) for SMPL-22 (6D rotation per joint)
trans = npz_data['trans']              # (T, 3) or (T, 1, 3) translation

# Flatten to 135-dim representation (3 for trans + 22*6 for rot6d)
# Already done by motion loading functions
gt_motion_135 = np.concatenate([
    trans.reshape(trans.shape[0], -1)[:, :3],  # First 3 dims of translation
    poses.reshape(poses.shape[0], -1)          # All 132 dims of rot6d (22 joints * 6)
], axis=1)
# Result shape: (T, 135)
```

### 3.2 Accessing from eval_dashboard Database

```python
import sqlite3

db_path = '/tmp/eval_dashboard.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query for a specific E3 task run
E3_RUN_ID = 3250  # Example: adaptive setting run

# Get per-sample info
cursor.execute("""
    SELECT sample_idx, prompt_id, text_caption, gen_motion_path
    FROM sample_results
    WHERE eval_run_id = ?
    ORDER BY sample_idx
""", (E3_RUN_ID,))

samples = cursor.fetchall()

# Now cross-reference with E3 JSON
with open('data/eval/m2m_v2/eval_e3_keyframe.json') as f:
    e3_data = json.load(f)
data_list = e3_data['data_list']

for sample_idx, prompt_id, caption, gen_path in samples:
    e3_item = data_list[sample_idx]
    gt_motion_path = e3_item['motion_path']
    
    print(f"Sample {sample_idx:05d}:")
    print(f"  GT motion: {gt_motion_path}")
    print(f"  Generated: {gen_path}")
    
conn.close()
```

### 3.3 Using the Result JSON Files

Each eval run produces an aggregated result JSON that includes per-sample metadata:

**File structure**:
```
work_dirs/eval_{timestamp}/import_jsons/kimodo/KIMODO_caption__E3_{setting}.json
```

**Example**: `work_dirs/eval_8082_refresh_20260501/import_jsons/kimodo/KIMODO_caption__E3_adaptive.json`

```python
import json

result_json = 'work_dirs/eval_8082_refresh_20260501/import_jsons/kimodo/KIMODO_caption__E3_adaptive.json'

with open(result_json) as f:
    results = json.load(f)

# results['per_sample'] contains per-sample metrics
per_sample = results['per_sample']  # List of 240 dicts

# For each sample, cross-reference with E3 JSON to get GT motion
with open('data/eval/m2m_v2/eval_e3_keyframe.json') as f:
    e3_data = json.load(f)
data_list = e3_data['data_list']

for sample_idx, sample_metrics in enumerate(per_sample):
    e3_item = data_list[sample_idx]
    gt_motion_path = e3_item['motion_path']
    gen_motion_path = sample_metrics['_npz_path']  # From result JSON
    
    print(f"Sample {sample_idx:05d}:")
    print(f"  GT: {gt_motion_path}")
    print(f"  Gen: {gen_motion_path}")
    print(f"  Metrics: {list(sample_metrics.keys())}")
```

---

## 4. Motion Frequency Metrics: Loading and Computing

### 4.1 Loading Both GT and Generated Motions

```python
import numpy as np

def load_motion_from_npz(npz_path):
    """Load motion from NPZ file and return 135-dim representation."""
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # Handle different key names
    poses_key = 'poses' if 'poses' in npz_data else 'body_pose'
    trans_key = 'trans' if 'trans' in npz_data else 'transl'
    
    poses = npz_data[poses_key].astype(np.float32)  # (T, 66) rot6d
    trans = npz_data[trans_key].astype(np.float32)  # (T, 3) or (T, 1, 3)
    
    # Flatten translation to (T, 3)
    if trans.ndim == 3:
        trans = trans[:, 0, :]
    
    # Concatenate to 135-dim
    motion_135 = np.concatenate([
        trans,              # First 3 dims
        poses.reshape(poses.shape[0], -1)  # 132 dims (22*6)
    ], axis=1)
    
    return motion_135.astype(np.float32)

# Example usage
gt_motion = load_motion_from_npz('/path/to/gt/motion.npz')
gen_motion = load_motion_from_npz('/path/to/generated/motion.npz')
```

### 4.2 Computing Motion Frequency Metrics

Frequency-based metrics typically measure temporal smoothness and spectral properties:

```python
def compute_motion_frequency_spectrum(motion, fps=30):
    """
    Compute frequency spectrum of motion.
    
    Args:
        motion: (T, D) motion matrix
        fps: frames per second
        
    Returns:
        frequencies: FFT frequencies
        magnitude: FFT magnitude spectrum
    """
    from scipy.fft import fft, fftfreq
    
    # Apply FFT along time axis for each DOF
    fft_result = fft(motion, axis=0)
    magnitude = np.abs(fft_result)
    
    # Get frequencies
    frequencies = fftfreq(motion.shape[0], d=1.0/fps)
    
    # Take only positive frequencies
    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    magnitude = magnitude[positive_freq_idx]
    
    return frequencies, magnitude

def compute_jitter_metric(motion, fps=30):
    """Compute jitter as acceleration magnitude (2nd derivative)."""
    if motion.shape[0] < 3:
        return np.nan
    
    # Compute velocity (1st derivative)
    velocity = np.diff(motion, axis=0)
    
    # Compute acceleration (2nd derivative)
    acceleration = np.diff(velocity, axis=0)
    
    # Compute magnitude
    jitter = np.linalg.norm(acceleration, axis=1)
    
    return jitter.mean(), jitter.std()
```

### 4.3 Example Metrics Computation Pipeline

```python
import json

# Configuration
E3_JSON = 'data/eval/m2m_v2/eval_e3_keyframe.json'
E3_RUN_ID = 3250  # Example run ID
RESULT_JSON = 'work_dirs/eval_8082_refresh_20260501/import_jsons/kimodo/KIMODO_caption__E3_adaptive.json'

# Load E3 datalist
with open(E3_JSON) as f:
    e3_data = json.load(f)
data_list = e3_data['data_list']

# Load result JSON
with open(RESULT_JSON) as f:
    results = json.load(f)

# Compute metrics for each sample
metrics_by_sample = []

for sample_idx, per_sample_metrics in enumerate(results['per_sample']):
    e3_item = data_list[sample_idx]
    gt_path = e3_item['motion_path']
    gen_path = per_sample_metrics['_npz_path']
    
    # Load motions
    gt_motion = load_motion_from_npz(gt_path)
    gen_motion = load_motion_from_npz(gen_path)
    
    # Align lengths (interpolate gen to GT length if needed)
    if gen_motion.shape[0] != gt_motion.shape[0]:
        from scipy.interpolate import interp1d
        t_gen = np.linspace(0, 1, gen_motion.shape[0])
        t_gt = np.linspace(0, 1, gt_motion.shape[0])
        gen_motion = np.array([
            interp1d(t_gen, gen_motion[:, d])(t_gt)
            for d in range(gen_motion.shape[1])
        ]).T
    
    # Compute frequency spectrum
    frequencies, mag_gt = compute_motion_frequency_spectrum(gt_motion, fps=30)
    _, mag_gen = compute_motion_frequency_spectrum(gen_motion, fps=30)
    
    # Compute jitter
    gt_jitter_mean, gt_jitter_std = compute_jitter_metric(gt_motion)
    gen_jitter_mean, gen_jitter_std = compute_jitter_metric(gen_motion)
    
    metrics_by_sample.append({
        'sample_idx': sample_idx,
        'gt_jitter_mean': gt_jitter_mean,
        'gen_jitter_mean': gen_jitter_mean,
        'gt_jitter_std': gt_jitter_std,
        'gen_jitter_std': gen_jitter_std,
    })

# Save results
output_json = 'frequency_metrics_e3.json'
with open(output_json, 'w') as f:
    json.dump(metrics_by_sample, f, indent=2)
```

---

## 5. Data File Locations Summary

| Component | Location | Notes |
|-----------|----------|-------|
| **E3 Datalist** | `data/eval/m2m_v2/eval_e3_keyframe.json` | Primary source of GT motion paths |
| **GT Motion NPZ Files** | `/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/{actor}/{action}.npz` | Absolute paths from datalist |
| **Generated Results** | `work_dirs/eval_{timestamp}/{model_name}/*_E3_{setting}_*/E3_{setting}/npz/{idx:05d}.npz` | Per-model per-setting outputs |
| **Aggregated Results** | `work_dirs/eval_{timestamp}/import_jsons/{import_src}/{MODEL}_E3_{setting}.json` | Contains per-sample metrics |
| **eval_dashboard DB** | `motion_annot_web/eval_dashboard/eval_dashboard.db` | Metadata (eval_runs, sample_results tables) |
| **score_m2m DB** | `motion_annot_web/score_m2m/score_m2m.db` | Human annotation scores (less useful for GT lookup) |

---

## 6. Example: Complete Pipeline for Computing Metrics

```python
#!/usr/bin/env python3
"""
Complete pipeline to compute motion frequency metrics for E3 samples.
"""

import json
import sqlite3
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq

class E3MetricsComputer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.e3_json_path = self.project_root / 'data/eval/m2m_v2/eval_e3_keyframe.json'
        self.db_path = self.project_root / 'motion_annot_web/eval_dashboard/eval_dashboard.db'
        
        # Load E3 datalist once
        with open(self.e3_json_path) as f:
            self.e3_data = json.load(f)
        self.data_list = self.e3_data['data_list']
    
    def load_motion(self, npz_path):
        """Load NPZ and convert to 135-dim motion."""
        try:
            npz_data = np.load(npz_path, allow_pickle=True)
            poses = npz_data['poses'].astype(np.float32)
            trans = npz_data['trans'].astype(np.float32)
            if trans.ndim == 3:
                trans = trans[:, 0, :]
            motion = np.concatenate([trans, poses.reshape(poses.shape[0], -1)], axis=1)
            return motion
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return None
    
    def get_gt_motion_path(self, sample_idx):
        """Get GT motion path from E3 datalist."""
        if 0 <= sample_idx < len(self.data_list):
            return self.data_list[sample_idx]['motion_path']
        return None
    
    def get_samples_for_run(self, run_id):
        """Query eval_dashboard DB for samples in a given run."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sample_idx, prompt_id, gen_motion_path
            FROM sample_results
            WHERE eval_run_id = ?
            ORDER BY sample_idx
        """, (run_id,))
        samples = cursor.fetchall()
        conn.close()
        return samples
    
    def compute_frequency_energy(self, motion, fps=30):
        """Compute energy in different frequency bands."""
        fft_result = fft(motion, axis=0)
        magnitude = np.abs(fft_result)
        frequencies = fftfreq(motion.shape[0], d=1.0/fps)
        
        # Energy in different bands
        low_freq_mask = np.abs(frequencies) <= 2.0
        mid_freq_mask = (np.abs(frequencies) > 2.0) & (np.abs(frequencies) <= 5.0)
        high_freq_mask = np.abs(frequencies) > 5.0
        
        energy_low = magnitude[low_freq_mask].sum()
        energy_mid = magnitude[mid_freq_mask].sum()
        energy_high = magnitude[high_freq_mask].sum()
        
        return {
            'low': float(energy_low),
            'mid': float(energy_mid),
            'high': float(energy_high),
        }
    
    def compute_metrics_for_run(self, run_id, output_path=None):
        """Compute metrics for all samples in a run."""
        samples = self.get_samples_for_run(run_id)
        
        results = []
        for sample_idx, prompt_id, gen_path in samples:
            gt_path = self.get_gt_motion_path(sample_idx)
            
            if not gt_path:
                print(f"Skipping {sample_idx}: no GT path found")
                continue
            
            gt_motion = self.load_motion(gt_path)
            gen_motion = self.load_motion(gen_path)
            
            if gt_motion is None or gen_motion is None:
                print(f"Skipping {sample_idx}: failed to load motions")
                continue
            
            # Compute frequency metrics
            gt_freq = self.compute_frequency_energy(gt_motion)
            gen_freq = self.compute_frequency_energy(gen_motion)
            
            results.append({
                'sample_idx': int(sample_idx),
                'prompt_id': str(prompt_id),
                'gt_motion_path': str(gt_path),
                'gen_motion_path': str(gen_path),
                'gt_frequency_energy': gt_freq,
                'gen_frequency_energy': gen_freq,
                'shape_gt': gt_motion.shape,
                'shape_gen': gen_motion.shape,
            })
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        return results

# Usage
if __name__ == '__main__':
    computer = E3MetricsComputer('/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer')
    
    # Compute for E3 adaptive setting run
    metrics = computer.compute_metrics_for_run(
        run_id=3250,
        output_path='e3_frequency_metrics_adaptive.json'
    )
    
    print(f"Computed metrics for {len(metrics)} samples")
```

---

## Summary

To compute motion frequency metrics for E3 keyframe interpolation eval cases:

1. **Locate source GT motions**: Index the E3 JSON datalist (`data/eval/m2m_v2/eval_e3_keyframe.json`) by sample index (0-239)
2. **Get motion paths**: Each datalist item has an absolute `motion_path` field pointing to the NPZ file
3. **Load motions**: Use `np.load()` to read NPZ and concatenate poses + trans to 135-dim representation
4. **Find generated results**: Query `eval_runs` and `sample_results` from `eval_dashboard.db` to locate generated NPZ files
5. **Compute metrics**: Apply FFT or other spectral analysis on both GT and generated motions
6. **Store results**: Save metrics to JSON for downstream analysis

