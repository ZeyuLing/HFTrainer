# PhysFlow Visualization System - Navigation Index

**Quick Links to Key Resources**

---

## 📚 Documentation Files (Start Here!)

| File | Purpose | Best For |
|------|---------|----------|
| **PHYSFLOW_VISUALIZATION_GUIDE.md** | Complete walkthrough with examples | First-time users, understanding the system |
| **QUICK_REFERENCE.md** | One-page summary of the 4-panel system | Quick lookup, running comparisons |
| **PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md** | Detailed code breakdown with line numbers | Developers, modifying the code |
| **physflow_visualization_summary.md** | Technical analysis of the architecture | Understanding design decisions |
| **comparison_report.txt** | Latest results from May 25, 2026 | Viewing actual performance metrics |

---

## 🚀 Quick Start (3 Steps)

### 1. Understand the System (5 min)
Read: `QUICK_REFERENCE.md`
- What is 4-panel comparison?
- Where are the files?
- What do the metrics mean?

### 2. View Latest Results (2 min)
```bash
cat output/physflow_v2_compare/comparison_report.txt
```
Shows:
- Pretrained vs Fine-tuned performance
- Per-prompt improvements
- Category-level breakdown

### 3. Load and Analyze NPZ Files (10 min)
```python
import numpy as np
npz = np.load('output/physflow_v2_compare/npz/pretrained_00_a_person_stands_still_raw.npz')
motion_135 = npz['motion_135']  # Shape: (120, 135)
```
See: **PHYSFLOW_VISUALIZATION_GUIDE.md** → "Examining Individual NPZ Files"

---

## 📁 Directory Structure

```
hf_trainer/
├── scripts/embodied/
│   ├── physflow_visualize_compare.py ......... Main script (570 lines)
│   ├── physflow_rl_oracle.py ................ Physics engine (39 KB)
│   ├── physflow_eval_demo.py ................ Alternative demo
│   └── compare_runtime_steps.py ............. Runtime comparison
│
├── output/physflow_v2_compare/
│   ├── comparison_report.txt ................ Text results ← VIEW THIS
│   ├── comparison_results.json .............. JSON metrics
│   ├── run_compare.sh ....................... Bash script to re-run
│   ├── npz/ ................................ 76 motion files
│   │   ├── pretrained_00_*_raw.npz
│   │   ├── pretrained_00_*_rl.npz
│   │   ├── finetuned_00_*_raw.npz
│   │   └── finetuned_00_*_rl.npz
│   │
│   └── ... (19 prompts × 4 files = 76 total)
│
└── Documentation/ (THIS DIRECTORY)
    ├── PHYSFLOW_VISUALIZATION_GUIDE.md (comprehensive)
    ├── QUICK_REFERENCE.md (quick lookup)
    ├── PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md (code details)
    ├── physflow_visualization_summary.md (analysis)
    └── PHYSFLOW_VIZ_INDEX.md (you are here)
```

---

## 🎯 What is the 4-Panel Comparison?

```
                  PRETRAINED MODEL          FINE-TUNED MODEL
                        ↓                           ↓
    PHASE 1 & 2: T2M Diffusion Generation (50 ODE steps)
                        ↓                           ↓
            motion_135 (120 frames)    motion_135 (120 frames)
                  ↙             ↘            ↙             ↘
            Save                Run      Save              Run
            Raw              Physics    Raw            Physics
            ↓                 Sim        ↓              Sim
                                ↓                        ↓
        PANEL 1           PANEL 2  PANEL 3        PANEL 4
     Pretrained       Pretrained+RL Fine-tuned   Fine-tuned+RL
        Raw            Corrected     Raw          Corrected
    [NPZ File]        [NPZ File]   [NPZ File]   [NPZ File]
```

Each panel is saved as a separate NPZ file with motion data + metrics.

---

## 📊 Latest Performance (May 25, 2026)

### Summary
```
Metric                  Pretrained    Fine-tuned    Improvement
────────────────────────────────────────────────────────────────
Avg Completion Ratio       0.438         0.500        +0.062 (+14%)
Success Rate (0/19)        0/19          2/19         +2 (+10.5%)
Best Category            Walking       Standing      +0.212 (+21.2%)
```

### Top Improvements
- "a person stands still": 0.41 → 0.88 (+0.47) ✅
- "a person walks forward": 0.38 → 0.88 (+0.50) ✅
- "a person jumps in place": 0.40 → 0.79 (+0.40) ✅

### Detailed Report
→ See: `output/physflow_v2_compare/comparison_report.txt`

---

## 🔍 Finding Specific Information

### "How do I..."

| Task | File | Section |
|------|------|---------|
| Understand the 4-panel concept | QUICK_REFERENCE.md | "TL;DR" |
| View the latest results | comparison_report.txt | — |
| Load motion data in Python | PHYSFLOW_VISUALIZATION_GUIDE.md | "Examining Individual NPZ Files" |
| Modify generation parameters | PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md | "Main Function" |
| Understand motion_135 format | QUICK_REFERENCE.md | "Data Format: motion_135" |
| Compare raw vs RL-corrected | PHYSFLOW_VISUALIZATION_GUIDE.md | "Compare Raw vs RL-Corrected" |
| See what files are generated | PHYSFLOW_VISUALIZATION_GUIDE.md | "Where to Find Everything" |
| Re-run the full comparison | PHYSFLOW_VISUALIZATION_GUIDE.md | "Running the Full Comparison" |
| Understand the metrics | QUICK_REFERENCE.md | "Key Metrics Explained" |

---

## 💾 Data Formats

### NPZ File Structure
```python
{
  'motion_135': np.ndarray (T, 135),      # T frames × 135 dims
  'fps': 30,                               # Frame rate
  'prompt': str,                           # Text prompt
  'rl_status': 'success' or 'fell'        # Physics outcome
}
```

### motion_135 Layout (135 dimensions)
```
[0:3]      → Translation (X, Y, Z)
[3:9]      → Root rotation (6D)
[9:135]    → Body pose (21 joints × 6D)
────────────────────────────────────────
Total: 3 + 6 + (21 × 6) = 135 dims
```

### Key Metrics
| Metric | Meaning | Range | Good |
|--------|---------|-------|------|
| completion_ratio | % motion completed before fall | 0-1 | >0.8 |
| root_height_min | Lowest point in simulation | meters | >0.3 |
| status | Did it succeed? | success/fell | success |
| actual_sim_steps | Frames before fall | int | ~200 |

---

## 🔧 Main Components

### 1. Motion Generation
**File:** `scripts/embodied/physflow_visualize_compare.py`
**Function:** `load_bundle_and_generate()` (lines 36-191)
**Output:** motion_135 arrays (120, 135)

### 2. Physics Simulation
**File:** `scripts/embodied/physflow_rl_oracle.py`
**Function:** `RLPhysicsOracle.correct()`
**Input/Output:** motion_135 → motion_135_rl + stats

### 3. Comparison Generation
**File:** `scripts/embodied/physflow_visualize_compare.py`
**Function:** `print_comparison_report()` (lines 260-400)
**Output:** .txt + .json reports

### 4. Results Storage
**Location:** `output/physflow_v2_compare/`
**Files:** 76 NPZ files (4 per prompt) + 2 report files

---

## 🎬 Test Prompts (19 Categories)

### Standing (3)
- "a person stands still" ⭐ Best improvement
- "a person stands in a relaxed pose"
- "a person shifts weight from left to right foot"

### Walking (7)
- "a person walks forward at a normal pace" ⭐ Good improvement
- "a person walks in a small circle"
- "a person walks forward slowly"
- "a person walks with long strides"
- "a person walks and then stops"
- "a person walks forward then turns around"
- "a person jogs slowly then walks"

### Upper Body (3)
- "a person waves with their right hand"
- "a person raises both arms above their head"
- "a person claps their hands together"
- "a person stretches arms to the sides"

### Dynamic (5)
- "a person kicks with their right leg"
- "a person squats down and stands back up"
- "a person jumps in place" ⭐ Good improvement
- "a person does a jumping jack"
- "a person does a high kick"

---

## ⚙️ Running the Comparison

### Option 1: Quick Script (Recommended)
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
bash output/physflow_v2_compare/run_compare.sh
```

### Option 2: Full Command
```bash
python3 scripts/embodied/physflow_visualize_compare.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --pretrained-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --finetuned-ckpt output/physflow_v2_train_blend50/model_iter500.pt \
    --text-cache output/physflow_v2_test/text_embeddings.pt \
    --output-dir output/physflow_v2_compare \
    --device cuda:0
```

**Runtime:** ~15-20 minutes on GPU

---

## 🔗 External Viewing

The NPZ files can be viewed with external tools:

1. **motion_annot_web** - General motion visualization
   ```bash
   cd /path/to/motion_annot_web
   python3 app.py --npz-dir /path/to/hf_trainer/output/physflow_v2_compare/npz/
   ```

2. **embodied_viz** - 3D SMPL mesh viewer
   ```bash
   Specialized viewer for embodied motion with physics visualization
   ```

---

## ❓ Common Questions

**Q: What's in each NPZ file?**
A: motion_135 array + fps + prompt. See "Data Formats" above.

**Q: Why 4 files per prompt?**
A: Pretrained (raw) + Pretrained (RL) + Fine-tuned (raw) + Fine-tuned (RL)

**Q: What does RL-corrected mean?**
A: Motion was run through MuJoCo physics simulator for stability correction.

**Q: How is "success" determined?**
A: completion_ratio > 0.99 (99% of motion completed without falling)

**Q: Can I modify parameters?**
A: Yes, see PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md → "Main Function"

**Q: How long does the full comparison take?**
A: ~15-20 minutes on GPU (Phases 1-2: ~10 min, Phases 3-4: ~5-10 min)

---

## 📝 File Modification History

| File | Date | Size | Purpose |
|------|------|------|---------|
| physflow_visualize_compare.py | May 25 11:26 | 21.3 KB | Main script |
| physflow_rl_oracle.py | May 25 07:16 | 39 KB | Physics engine |
| comparison_report.txt | May 25 11:31 | 3.2 KB | Latest results |
| comparison_results.json | May 25 11:31 | 40 KB | Structured metrics |
| QUICK_REFERENCE.md | May 25 16:07 | 8.9 KB | Quick lookup |
| PHYSFLOW_VISUALIZATION_GUIDE.md | May 25 16:10 | ~15 KB | Comprehensive guide |

---

## 🚦 Status Checklist

- ✅ Pretrained model available
- ✅ Fine-tuned model available
- ✅ T2M config available
- ✅ Text embeddings cache available
- ✅ Physics oracle implemented
- ✅ Main visualization script complete
- ✅ Comparison report generated
- ✅ 76 NPZ files created
- ✅ Metrics computed
- ✅ Documentation created

---

## 📖 Reading Order

**For Different Audiences:**

### 🟢 First-time users
1. QUICK_REFERENCE.md (3 min)
2. PHYSFLOW_VISUALIZATION_GUIDE.md § "What is 4-Panel?" (5 min)
3. View output: `cat output/physflow_v2_compare/comparison_report.txt` (2 min)

### 🟡 Developers
1. PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md (10 min)
2. PHYSFLOW_VISUALIZATION_GUIDE.md § "Key Code Locations" (5 min)
3. Review: `scripts/embodied/physflow_visualize_compare.py` (30 min)

### 🔴 Researchers
1. physflow_visualization_summary.md (10 min)
2. PHYSFLOW_VISUALIZATION_GUIDE.md § "Statistical Summary" (5 min)
3. Load and analyze JSON: `cat output/physflow_v2_compare/comparison_results.json` (10 min)

---

**Last Updated:** May 25, 2026  
**System:** PhysFlow Motion Visualization + RL Physics Correction  
**Status:** ✅ Complete and Documented
