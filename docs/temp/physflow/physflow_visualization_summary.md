# PhysFlow Visualization/Comparison System - Code Analysis Summary

## Overview

The PhysFlow system generates 4-panel comparisons through a **comparison report + NPZ file export** workflow, rather than a monolithic 4-panel rendering function. The visualization is designed to be viewed through external 3D web viewers.

---

## Key Visualization Script

### **Main File: `physflow_visualize_compare.py`**
**Location:** `scripts/embodied/physflow_visualize_compare.py` (21.3 KB, modified May 25 11:26)

**Purpose:** Generate motions from both original pretrained model and PhysFlow fine-tuned model, then runs RL physics simulation on each.

**Outputs:**
1. **NPZ files** for 3D viewer comparison (raw and RL-corrected pairs)
2. **Physics metrics comparison** (completion ratio, tracking error, root height)
3. **Summary report** (text and JSON)

### **4-Panel Structure Concept**

While the code doesn't create literal 4 matplotlib/panel subplots in Python, it produces a **conceptual 4-panel comparison** through:

```
Panel 1: Pretrained Model (Raw Generation)
Panel 2: Pretrained Model (RL Corrected)
Panel 3: Fine-tuned Model (Raw Generation)  
Panel 4: Fine-tuned Model (RL Corrected)
```

These are stored as separate NPZ files and rendered externally by `motion_annot_web/embodied_viz` web viewer.

---

## Generated Comparison Files

### **Output Directory**
`output/physflow_v2_compare/` (as of May 25 15:31)

**Structure:**
```
├── comparison_report.txt          # Human-readable comparison
├── comparison_results.json        # Structured metrics (JSON)
├── launch_compare.sh              # Helper script
├── run_compare.sh                 # Helper script
└── npz/                           # Motion files
    ├── pretrained_00_a_person_stands_still_raw.npz
    ├── pretrained_00_a_person_stands_still_rl.npz
    ├── finetuned_00_a_person_stands_still_raw.npz
    ├── finetuned_00_a_person_stands_still_rl.npz
    └── ... (2x2 = 4 files per prompt, 19 prompts = 76 NPZ files)
```

### **Latest Comparison Output**

**File:** `output/physflow_v2_compare/comparison_report.txt`

**Report Structure:**
```
PhysFlow Visualization Comparison Report
================================================================================
- Per-prompt comparison table (19 prompts)
  * Columns: Prompt | Pretrained (status, completion, height) | Fine-tuned | Δ
  * Example: "a person stands still | fell c=0.41 h=0.26 | fell c=0.88 h=0.27 | +0.47"

- Summary statistics:
  * Total prompts: 19
  * Pretrained: avg completion 0.438, success rate 0/19 (0.0%)
  * Fine-tuned: avg completion 0.500, success rate 2/19 (10.5%)
  * Improvement: +0.062 completion, +10.5% success rate

- Per-category breakdown:
  * Standing (n=3): 0.282 → 0.493 (Δ=+0.212)
  * Walking (n=7): 0.446 → 0.549 (Δ=+0.103)
  * Upper body (n=3): 0.448 → 0.398 (Δ=-0.050)
  * Dynamic (n=6): 0.502 → 0.498 (Δ=-0.004)
```

---

## Core Visualization Functions

### **1. Motion Generation**
**Function:** `load_bundle_and_generate()`
- Takes: config path, checkpoint path, text cache, prompts, num_frames
- Returns: List of motion_135 arrays (T, 135)
- Uses: T2M model diffusion sampling (50 ODE steps, CFG scale 5.0)

### **2. RL Physics Evaluation**
**Function:** `run_rl_physics_evaluation()`
- Takes: List of motion_135 arrays, prompts, output_dir, label
- Runs: RLPhysicsOracle.correct() on each motion
- Saves: 2 NPZ files per motion
  - `{label}_{i:02d}_{prompt}_raw.npz` - Generated motion
  - `{label}_{i:02d}_{prompt}_rl.npz` - RL-corrected motion
- Returns: List of stats dicts with:
  - `status`: 'success' or 'fell'
  - `completion_ratio`: float (0-1)
  - `root_height_min`: float (meters)
  - `npz_raw`, `npz_rl`: file paths

### **3. Comparison Report Generation**
**Function:** `print_comparison_report()`
- Accepts: pretrained_stats, finetuned_stats, prompts, output_path
- Outputs:
  1. Human-readable report (stdout + file)
  2. JSON with all metrics and file paths
  3. Per-category statistical breakdown

---

## Related Visualization Scripts

### **`physflow_eval_demo.py`** (19.7 KB, modified May 21 14:54)
**Purpose:** Evaluation demo with optional baseline comparison

**Outputs:**
- NPZ motion files
- SMPL mesh JSON files for 3D web viewer
- 3-way comparison: baseline → V5 model → V5+RL correction

**Key Function:** `motion_135_to_mesh_json()`
- Converts motion_135 → SMPL-X mesh format (55 joints)
- Exports as JSON for three.js/Babylon web visualization

### **`batch_pipeline_to_web.py`** (8.7 KB)
**Purpose:** Batch pipeline for converting NPZ files → web-ready formats

**Flow:** NPZ → ProtoMotions cache → JSON for web viewer

---

## Data Format: motion_135

All visualizations use the **motion_135** format:
```
Shape: (T, 135) where T = number of frames

Components:
- [0:3]       → Translation (root XYZ position)
- [3:9]       → Root rotation (6D rotation representation)
- [9:135]     → Body pose (21 joints × 6D representation)

Total: 3 + 6 + (21 × 6) = 135 dimensions
```

---

## NPZ File Contents

**Standard NPZ save (line 232-233 in physflow_visualize_compare.py):**
```python
np.savez(npz_file, motion_135=motion_135, fps=30, prompt=prompt)
```

**NPZ contents:**
```
{
  'motion_135': np.ndarray (T, 135) - motion data
  'fps': int - 30 fps
  'prompt': str - text prompt
  [optional] 'rl_status': str - 'success' or 'fell'
}
```

---

## Comparison Metrics (JSON Output)

**File:** `output/physflow_v2_compare/comparison_results.json`

**Structure:**
```json
{
  "prompts": ["a person stands still", ...],
  "pretrained_stats": [
    {
      "status": "fell",
      "total_ref_frames": 120,
      "total_sim_steps": 200,
      "actual_sim_steps": 81,
      "fall_frame": 80,
      "root_height_min": 0.262,
      "completion_ratio": 0.405,
      "control_dt": 0.02,
      "oracle_time_s": 0.890,
      "npz_raw": "output/physflow_v2_compare/npz/pretrained_00_a_person_stands_still_raw.npz",
      "npz_rl": "output/physflow_v2_compare/npz/pretrained_00_a_person_stands_still_rl.npz",
      "prompt": "a person stands still",
      "label": "pretrained"
    },
    ...
  ],
  "finetuned_stats": [...],
  "summary": {
    "pretrained_avg_completion": 0.438,
    "finetuned_avg_completion": 0.500,
    "improvement_completion": 0.062,
    "pretrained_success_rate": 0.0,
    "finetuned_success_rate": 0.105
  }
}
```

---

## Test Prompts (19 curriculum levels)

**File:** Lines 406-431 in `physflow_visualize_compare.py`

```python
TEST_PROMPTS = [
    # Standing (easy - level 0)
    "a person stands still",
    "a person stands in a relaxed pose",
    "a person shifts weight from left to right foot",
    
    # Walking (medium - level 1)
    "a person walks forward at a normal pace",
    "a person walks in a small circle",
    "a person walks forward slowly",
    "a person walks with long strides",
    
    # Upper body (medium - level 2)
    "a person waves with their right hand",
    "a person raises both arms above their head",
    "a person claps their hands together",
    "a person stretches arms to the sides",
    
    # Transitions (hard - level 3)
    "a person walks and then stops",
    "a person walks forward then turns around",
    "a person jogs slowly then walks",
    
    # Dynamic (hardest - level 4)
    "a person kicks with their right leg",
    "a person squats down and stands back up",
    "a person jumps in place",
    "a person does a jumping jack",
    "a person does a high kick",
]
```

---

## Viewing the Visualizations

**Reference (line 366 in physflow_visualize_compare.py):**
```
Use motion_annot_web/embodied_viz to view side-by-side.
```

**External tools:**
- `motion_annot_web` - Web-based motion annotation/visualization tool
- `embodied_viz` - 3D embodied motion viewer
- Web frameworks: three.js or Babylon.js render SMPL meshes from JSON

---

## RLPhysicsOracle Integration

**File:** `scripts/embodied/physflow_rl_oracle.py` (39 KB)

**Usage in comparison:**
```python
oracle = RLPhysicsOracle()
motion_135_rl, stats = oracle.correct(motion_135)
```

**Outputs:**
- `motion_135_rl`: RL-corrected motion (typically shorter, falls earlier)
- `stats`: Dictionary with physics metrics
  - `status`: 'success' or 'fell'
  - `completion_ratio`: (actual_sim_steps / total_sim_steps)
  - `root_height_min`: minimum root height during simulation
  - `duration_s`: simulation elapsed time

---

## Usage Example

```bash
# Generate 4-panel comparison (pretrained vs fine-tuned)
python3 scripts/embodied/physflow_visualize_compare.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --pretrained-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --finetuned-ckpt output/physflow_v2_train_blend50/model_iter500.pt \
    --text-cache output/physflow_v2_test/text_embeddings.pt \
    --output-dir output/physflow_v2_compare \
    --device cuda:0 \
    --num-frames 120 \
    --seed 42
```

**Output files:**
```
output/physflow_v2_compare/
├── comparison_report.txt          # Metrics summary
├── comparison_results.json        # Detailed results
└── npz/
    ├── pretrained_00_*.npz        # 4 files per prompt
    ├── pretrained_00_*_rl.npz     # (raw + RL corrected × 2)
    ├── finetuned_00_*.npz
    ├── finetuned_00_*_rl.npz
    └── ... (76 total NPZ files for 19 prompts)
```

---

## Key Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| **Main comparison script** | `physflow_visualize_compare.py` | 1-570 | 4-panel generation |
| **Motion generation** | `physflow_visualize_compare.py` | 36-191 | Load T2M + generate |
| **Physics evaluation** | `physflow_visualize_compare.py` | 194-257 | RL oracle integration |
| **Report generation** | `physflow_visualize_compare.py` | 260-400 | Metrics + JSON export |
| **RL Physics Oracle** | `physflow_rl_oracle.py` | 1-200+ | Physics correction |
| **Evaluation demo** | `physflow_eval_demo.py` | 1-499 | 3-way comparison variant |
| **Mesh JSON export** | `physflow_eval_demo.py` | 62-114 | motion_135 → web format |

---

## Summary

**The PhysFlow visualization system:**
1. ✅ **Generates** 2 motion sets (pretrained + fine-tuned)
2. ✅ **Corrects** each with RL physics simulation (→ 4 variants)
3. ✅ **Exports** NPZ files for external 3D viewer
4. ✅ **Reports** comparison metrics in text + JSON
5. ✅ **Organizes** by curriculum level (standing → walking → dynamic)

**External rendering:**
- NPZ files are viewed via `motion_annot_web/embodied_viz`
- No internal matplotlib/panel 4-subplot rendering
- Designed for web-based 3D SMPL mesh visualization
