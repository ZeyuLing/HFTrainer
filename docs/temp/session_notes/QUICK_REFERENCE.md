# PhysFlow 4-Panel Visualization - Quick Reference

## TL;DR

The 4-panel visualization system in PhysFlow compares motions through:
- **Panel 1**: Pretrained model (raw generation)
- **Panel 2**: Pretrained model (RL physics corrected)
- **Panel 3**: Fine-tuned model (raw generation)
- **Panel 4**: Fine-tuned model (RL physics corrected)

---

## Main Files to Know

| File | What It Does | Key Functions |
|------|-------------|-----------------|
| `scripts/embodied/physflow_visualize_compare.py` | Generates all 4 panels & comparison report | `load_bundle_and_generate()`, `run_rl_physics_evaluation()`, `print_comparison_report()` |
| `output/physflow_v2_compare/comparison_report.txt` | Human-readable comparison metrics | — |
| `output/physflow_v2_compare/comparison_results.json` | Machine-readable metrics in JSON | — |
| `output/physflow_v2_compare/npz/` | 76 NPZ motion files (4 per prompt) | — |

---

## How It Works

### 1️⃣ Generate Motions
```
Phase 1: Load pretrained model → generate 19 prompts
Phase 2: Load fine-tuned model → generate same 19 prompts
```

### 2️⃣ Apply RL Physics Correction
```
Phase 3: For each motion set (pretrained + finetuned):
  - Run RLPhysicsOracle.correct()
  - Saves 2 NPZ files per motion:
    * {label}_{idx}_{prompt}_raw.npz
    * {label}_{idx}_{prompt}_rl.npz
```

### 3️⃣ Generate Comparison Report
```
Phase 4: Compare metrics between pretrained & finetuned
  - completion_ratio
  - root_height_min
  - fall status
  - per-category breakdown
```

---

## Output Files Generated

```
output/physflow_v2_compare/
├── comparison_report.txt         # Summary table + statistics
├── comparison_results.json       # Detailed metrics (JSON)
└── npz/                          # 76 motion files
    ├── pretrained_00_a_person_stands_still_raw.npz
    ├── pretrained_00_a_person_stands_still_rl.npz
    ├── finetuned_00_a_person_stands_still_raw.npz
    ├── finetuned_00_a_person_stands_still_rl.npz
    ├── pretrained_01_a_person_stands_in_a_relaxed_pose_raw.npz
    ├── pretrained_01_a_person_stands_in_a_relaxed_pose_rl.npz
    └── ... (4 files × 19 prompts = 76 total)
```

---

## Latest Results (May 25, 2026)

```
Prompt                              | Pretrained  | Fine-tuned  | Improvement
─────────────────────────────────── | ─────────── | ─────────── | ────────────
a person stands still               | c=0.41      | c=0.88      | +0.47 ✅
a person walks forward at a pace    | c=0.38      | c=0.88      | +0.50 ✅
a person walks forward then turns   | c=0.32      | c=0.67      | +0.35 ✅
a person jumps in place             | c=0.40      | c=0.79      | +0.40 ✅
a person squats down and stands up  | c=0.23      | c=0.40      | +0.18 ✅

Summary:
  Pretrained avg completion:   0.438
  Fine-tuned avg completion:   0.500
  Improvement:                 +0.062 (+14%)
  
Per-category gains:
  • Standing:     +0.212 ⭐ (best improvement)
  • Walking:      +0.103
  • Upper body:   -0.050 (slight regression)
  • Dynamic:      -0.004 (neutral)
```

---

## Test Prompts (19 Curriculum Levels)

**Standing (3):**
- "a person stands still"
- "a person stands in a relaxed pose"
- "a person shifts weight from left to right foot"

**Walking (4):**
- "a person walks forward at a normal pace"
- "a person walks in a small circle"
- "a person walks forward slowly"
- "a person walks with long strides"

**Upper Body (4):**
- "a person waves with their right hand"
- "a person raises both arms above their head"
- "a person claps their hands together"
- "a person stretches arms to the sides"

**Transitions (3):**
- "a person walks and then stops"
- "a person walks forward then turns around"
- "a person jogs slowly then walks"

**Dynamic (5):**
- "a person kicks with their right leg"
- "a person squats down and stands back up"
- "a person jumps in place"
- "a person does a jumping jack"
- "a person does a high kick"

---

## Key Metrics Explained

| Metric | Meaning | Range | Good Value |
|--------|---------|-------|-----------|
| `completion_ratio` | % of motion completed before fall | 0-1 | >0.8 |
| `root_height_min` | Lowest point during simulation | meters | >0.3 |
| `status` | Final result | 'success' or 'fell' | 'success' |
| `actual_sim_steps` | Steps taken before falling | int | ~200 |
| `total_sim_steps` | Total expected steps | int | 200 |

---

## 4-Panel Data Flow

```
                      Pretrained Model                    Fine-tuned Model
                           |                                    |
                      T2M Generation (ODE)                T2M Generation (ODE)
                           |                                    |
                    motion_135 (T, 135)                 motion_135 (T, 135)
                           |                                    |
            ┌──────────────┴────────────────┐   ┌──────────────┴────────────────┐
            |                               |   |                               |
        Save NPZ                      Run RL Oracle                    Save NPZ                      Run RL Oracle
     (Panel 1 data)                       |                         (Panel 3 data)                       |
            |                   motion_135_rl (shorter)                     |                   motion_135_rl (shorter)
            |                             |                                 |                             |
        pretrained_*.npz          Save NPZ (Panel 2 data)           finetuned_*.npz         Save NPZ (Panel 4 data)
            |                             |                                 |                             |
            └─────────────────────────────┴─────────────────────────────────┴──────────────────────────┘
                                          |
                             Generate Comparison Report
                                          |
                    comparison_report.txt + comparison_results.json
```

---

## Usage

### Run Full Comparison
```bash
python3 scripts/embodied/physflow_visualize_compare.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --pretrained-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --finetuned-ckpt output/physflow_v2_train_blend50/model_iter500.pt \
    --text-cache output/physflow_v2_test/text_embeddings.pt \
    --output-dir output/physflow_v2_compare \
    --device cuda:0
```

### View Results
```bash
# Text report
cat output/physflow_v2_compare/comparison_report.txt

# JSON metrics
cat output/physflow_v2_compare/comparison_results.json

# Motion files
ls output/physflow_v2_compare/npz/
```

---

## External Viewing Tools

The NPZ files are designed to be viewed via:
- `motion_annot_web` - Web-based motion visualization
- `embodied_viz` - 3D embodied motion viewer
- Uses SMPL mesh rendering (three.js/Babylon.js)

---

## Related Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `physflow_eval_demo.py` | 3-way demo (baseline vs V5 vs V5+RL) | NPZ + SMPL mesh JSON |
| `batch_pipeline_to_web.py` | Batch NPZ → web format conversion | JSON files for web viewer |
| `physflow_rl_oracle.py` | Physics correction engine | Corrected motions + stats |

---

## Data Format: motion_135

All visualizations use **motion_135** (shape: T×135):
```
[0:3]      → Translation (X, Y, Z)
[3:9]      → Root rotation (6D)
[9:135]    → Body pose (21 joints × 6D each)

Total: 3 + 6 + (21×6) = 135 dimensions
```

**NPZ file contents:**
```python
{
  'motion_135': np.ndarray (T, 135),
  'fps': 30,
  'prompt': str,
  'rl_status': 'success' | 'fell'  # optional
}
```

---

## Code Locations

| Component | File | Lines |
|-----------|------|-------|
| Main script | `physflow_visualize_compare.py` | 1-570 |
| Motion generation | `physflow_visualize_compare.py` | 36-191 |
| RL evaluation | `physflow_visualize_compare.py` | 194-257 |
| Report generation | `physflow_visualize_compare.py` | 260-400 |
| Test prompts | `physflow_visualize_compare.py` | 406-431 |
| Main function | `physflow_visualize_compare.py` | 434-569 |

---

## Summary

✅ **Main visualization system:** `physflow_visualize_compare.py`
✅ **4-Panel concept:** Pretrained/Finetuned × Raw/RL-corrected
✅ **Output:** NPZ motion files + comparison report (text + JSON)
✅ **Metrics:** completion_ratio, root_height_min, status
✅ **Viewing:** External web viewer (embodied_viz)
✅ **Latest results:** +6.2% avg completion improvement, +21.2% for standing motions
