# PhysFlow 4-Panel Visualization System - Complete Index

## 📋 Documentation Files Created

This analysis has generated 3 comprehensive reference documents:

### 1. **QUICK_REFERENCE.md** ⚡
**Best for:** Quick lookup, visual overview, command-line usage
- TL;DR summary
- 4-panel structure diagram
- Latest results table
- Test prompts list
- Key metrics explained
- Quick usage commands

### 2. **PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md** 🔧
**Best for:** Developers, code review, implementation details
- Main comparison pipeline (4 phases)
- Core function signatures with code examples
- 4-panel ASCII visualization
- Comparison report sample output
- JSON structure with examples
- NPZ file format specification
- Command-line usage with full arguments

### 3. **physflow_visualization_summary.md** 📚
**Best for:** Comprehensive understanding, architecture study
- System overview
- Key visualization script details
- Generated comparison files structure
- Core functions documentation
- Related visualization scripts
- Data format deep-dive
- Comparison metrics breakdown
- Usage examples
- Code locations table
- RLPhysicsOracle integration guide

---

## 🎯 Quick Start

### I want to...

**...run a comparison myself**
→ See `QUICK_REFERENCE.md` "Usage" section or `PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md` "Command-line Usage"

**...understand the 4-panel concept**
→ Read `QUICK_REFERENCE.md` "TL;DR" section (30 seconds) or see the ASCII diagram in "4-Panel Data Flow"

**...view the latest results**
→ Check `QUICK_REFERENCE.md` "Latest Results" section or read `output/physflow_v2_compare/comparison_report.txt`

**...understand the code structure**
→ Read `physflow_visualization_summary.md` "Core Visualization Functions" or see implementation in `PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md`

**...modify/extend the visualization**
→ Study `PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md` code examples and function signatures

---

## 📍 Main Code Files

| File | Purpose | Lines | Last Modified |
|------|---------|-------|---|
| **scripts/embodied/physflow_visualize_compare.py** | 4-panel comparison generator | 570 | May 25 11:26 |
| **scripts/embodied/physflow_rl_oracle.py** | Physics correction engine | 39 KB | May 25 07:16 |
| **scripts/embodied/physflow_eval_demo.py** | Evaluation demo (3-way comparison) | 499 | May 21 14:54 |
| **scripts/embodied/batch_pipeline_to_web.py** | Batch NPZ → web conversion | ~150 | May 14 01:19 |

---

## 📊 Output Structure

```
output/physflow_v2_compare/
├── comparison_report.txt              # Human-readable summary
├── comparison_results.json            # Machine-readable metrics
├── launch_compare.sh                  # Helper script
├── run_compare.sh                     # Helper script
└── npz/                               # Motion files
    ├── pretrained_00_*.npz            # Panel 1: Pretrained raw
    ├── pretrained_00_*_rl.npz         # Panel 2: Pretrained + RL
    ├── finetuned_00_*.npz             # Panel 3: Fine-tuned raw
    ├── finetuned_00_*_rl.npz          # Panel 4: Fine-tuned + RL
    └── ... (76 total files: 19 prompts × 4 panels)
```

---

## 🎬 The 4 Panels Explained

```
Panel 1                           Panel 2
Pretrained Model                Pretrained + RL Correction
(Raw Generation)                (Physics-Corrected)
        ↓                               ↓
Generate with T2M       →      Run RL Oracle.correct()
motion_135 (T, 135)            motion_135_rl (shorter)

Panel 3                           Panel 4
Fine-tuned Model                Fine-tuned + RL Correction
(Raw Generation)                (Physics-Corrected)
        ↓                               ↓
Generate with T2M       →      Run RL Oracle.correct()
(+ fine-tuned weights)         motion_135_rl (shorter)

Metrics Tracked:
• completion_ratio: % of motion completed before fall (0-1)
• root_height_min: Lowest point during simulation (meters)
• status: 'success' or 'fell'
• delta: improvement from pretrained to fine-tuned
```

---

## 📈 Latest Results Summary

**Improvement from Pretrained → Fine-tuned:**

```
Standing motions:    +0.212 ⭐ BEST improvement
Walking motions:     +0.103
Transitions:         +0.053
Dynamic motions:     -0.004 (slight regression)
Upper body:          -0.050 (slight regression)

Overall:
  Avg completion improved from 0.438 → 0.500 (+6.2%)
  Success rate improved from 0/19 → 2/19 (+10.5%)
```

---

## 🔑 Key Functions

| Function | File | Purpose | Returns |
|----------|------|---------|---------|
| `load_bundle_and_generate()` | physflow_visualize_compare.py | Generate motions from T2M model | List[np.ndarray] motion_135 |
| `run_rl_physics_evaluation()` | physflow_visualize_compare.py | Run RL physics correction + metrics | List[dict] stats |
| `print_comparison_report()` | physflow_visualize_compare.py | Generate comparison report | None (prints to file) |
| `RLPhysicsOracle.correct()` | physflow_rl_oracle.py | Physics-correct a motion | (motion_135_rl, stats dict) |
| `motion_135_to_mesh_json()` | physflow_eval_demo.py | Convert to SMPL mesh JSON | dict for web viewer |

---

## 📂 Data Formats

### motion_135 Array
```python
Shape: (T, 135) where T = number of frames
[0:3]       → Translation (X, Y, Z)
[3:9]       → Root rotation (6D representation)
[9:135]     → Body pose (21 joints × 6D each)
```

### NPZ File Contents
```python
{
  'motion_135': np.ndarray (T, 135) - motion kinematics
  'fps': int - frames per second (30)
  'prompt': str - text prompt used for generation
  'rl_status': str - 'success' or 'fell' (optional)
}
```

### Stats Dictionary
```python
{
  'status': 'fell' | 'success',
  'completion_ratio': float (0-1),
  'root_height_min': float (meters),
  'fall_frame': int,
  'total_sim_steps': int (200),
  'actual_sim_steps': int (varies),
  'oracle_time_s': float,
  'npz_raw': str (path),
  'npz_rl': str (path),
  # ... + additional metrics
}
```

---

## 🚀 Running a Comparison

```bash
# Full 4-panel comparison (pretrained vs fine-tuned)
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

**Output timing:**
- Phase 1 (Pretrained generation): ~5-10 minutes
- Phase 2 (Fine-tuned generation): ~5-10 minutes
- Phase 3 (RL physics evaluation): ~30-60 minutes
- Phase 4 (Report generation): < 1 minute
- **Total: ~1-2 hours**

---

## 👁️ Viewing Results

**Text Report:**
```bash
cat output/physflow_v2_compare/comparison_report.txt
```

**JSON Metrics:**
```bash
cat output/physflow_v2_compare/comparison_results.json | python3 -m json.tool
```

**Motion Files:**
```bash
ls -lh output/physflow_v2_compare/npz/
```

**3D Visualization:**
Use `motion_annot_web` or `embodied_viz` external viewers to render SMPL meshes

---

## 🧪 Test Prompts (19 Curriculum Levels)

**Standing (Level 0):** 3 prompts
- "a person stands still"
- "a person stands in a relaxed pose"
- "a person shifts weight from left to right foot"

**Walking (Level 1):** 4 prompts
- "a person walks forward at a normal pace"
- "a person walks in a small circle"
- "a person walks forward slowly"
- "a person walks with long strides"

**Upper Body (Level 2):** 4 prompts
- "a person waves with their right hand"
- "a person raises both arms above their head"
- "a person claps their hands together"
- "a person stretches arms to the sides"

**Transitions (Level 3):** 3 prompts
- "a person walks and then stops"
- "a person walks forward then turns around"
- "a person jogs slowly then walks"

**Dynamic (Level 4):** 5 prompts
- "a person kicks with their right leg"
- "a person squats down and stands back up"
- "a person jumps in place"
- "a person does a jumping jack"
- "a person does a high kick"

---

## 📚 Related Documentation

- `README_EMBODIED_MOTION_VISUALIZATION.md` - Overview of embodied motion system
- `PHYSFLOW_QUICK_REFERENCE.md` - General PhysFlow reference
- `NPZ_TO_SMPL_QUICK_REFERENCE.md` - NPZ to SMPL conversion guide
- `METRICS_IMPLEMENTATION_REFERENCE.md` - Metrics calculations

---

## 🔗 Integration Points

**Input dependencies:**
- T2M model config: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`
- Pretrained checkpoint: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
- Fine-tuned checkpoint: `output/physflow_v2_train_blend50/model_iter500.pt`
- Text embeddings: `output/physflow_v2_test/text_embeddings.pt`

**Output integrations:**
- NPZ files → `motion_annot_web` viewer
- Mesh JSON → three.js/Babylon.js web viewer
- Metrics → reporting dashboards

---

## 📖 How to Use These Docs

1. **For a quick overview:** Start with `QUICK_REFERENCE.md`
2. **For implementation details:** Read `PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md`
3. **For deep understanding:** Study `physflow_visualization_summary.md`
4. **For actual code:** Check `scripts/embodied/physflow_visualize_compare.py` (main file)
5. **For latest results:** Read `output/physflow_v2_compare/comparison_report.txt`

---

## ✅ Checklist for Understanding

- [ ] I understand what the 4 panels represent
- [ ] I know where the main script is located
- [ ] I can find the comparison outputs
- [ ] I understand the motion_135 format
- [ ] I know what metrics are tracked
- [ ] I can read and interpret the comparison report
- [ ] I understand the workflow (generate → RL correct → report)
- [ ] I know how to run a comparison myself

---

## 🎯 Key Takeaways

✅ **PhysFlow generates 4-panel comparisons by:**
1. Generating motions with both pretrained and fine-tuned models
2. Applying RL physics correction to each
3. Exporting to NPZ files for external 3D viewers
4. Generating comparison reports (text + JSON)

✅ **Main script:** `scripts/embodied/physflow_visualize_compare.py` (570 lines)

✅ **Output:** `output/physflow_v2_compare/` with 76 NPZ files + reports

✅ **Metrics:** completion_ratio, root_height_min, status

✅ **Latest improvement:** +6.2% avg completion, +21.2% for standing

