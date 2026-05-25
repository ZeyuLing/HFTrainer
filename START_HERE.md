# 🚀 PhysFlow Visualization System - START HERE

**Welcome!** This document guides you through the PhysFlow 4-panel visualization system.

---

## ⚡ 30-Second Overview

PhysFlow generates a **4-panel motion comparison** that shows:

```
Pretrained Raw  →  Pretrained+RL  |  Fine-tuned Raw  →  Fine-tuned+RL
(Panel 1)           (Panel 2)     |    (Panel 3)        (Panel 4)
```

Each panel is a **motion file (NPZ)** containing:
- `motion_135` array: 120 frames × 135 dimensions
- `fps`: 30
- `prompt`: text description
- `rl_status`: whether it succeeded or fell

**Latest Results (May 25, 2026):**
- Pretrained: 0.438 avg completion
- Fine-tuned: 0.500 avg completion
- **Improvement: +6.2% (+14%)**
- **Best category: Standing motions (+21.2%)**

---

## 📚 Quick Navigation

### Choose Your Path

**🟢 I just want to understand what this is (5 minutes)**
→ Read: [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)

**🟡 I want to understand HOW it works (15 minutes)**
→ Read: [`PHYSFLOW_VISUALIZATION_GUIDE.md`](PHYSFLOW_VISUALIZATION_GUIDE.md) → "How It Works"

**🔴 I want to look at the code (30 minutes)**
→ Read: [`PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md`](PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md)

**📊 I want to analyze the results (20 minutes)**
→ Command: `cat output/physflow_v2_compare/comparison_report.txt`

**🗺️ I'm looking for something specific**
→ See: [`PHYSFLOW_VIZ_INDEX.md`](PHYSFLOW_VIZ_INDEX.md) (master index)

---

## 🎯 The 4-Panel Concept Explained

Think of it like this:

```
We have TWO MODELS:
  • Pretrained: Original HY-Motion-1.0
  • Fine-tuned: PhysFlow fine-tuned version

For EACH model, we generate the SAME 19 motions:
  1. Generate motion (raw) → Save as NPZ file
  2. Run physics simulator → Make it realistic → Save as NPZ file

Result: 4 files per motion (2 models × 2 states)
Total: 76 files (4 × 19 prompts)
```

### Why Do This?

To answer: **"Does fine-tuning improve physics-realistic motion?"**

By comparing:
- ✅ Pretrained (raw) vs Fine-tuned (raw) → Shows fine-tuning quality
- ✅ Both corrected versions → Shows max possible quality

---

## 📂 Where Are The Files?

```
hf_trainer/
├── output/physflow_v2_compare/
│   ├── comparison_report.txt         ← Read this! (human-readable)
│   ├── comparison_results.json       ← Structured data
│   └── npz/ (76 motion files)
│       ├── pretrained_00_*_raw.npz
│       ├── pretrained_00_*_rl.npz
│       ├── finetuned_00_*_raw.npz
│       └── finetuned_00_*_rl.npz
│
├── scripts/embodied/
│   ├── physflow_visualize_compare.py  ← Main script
│   └── physflow_rl_oracle.py          ← Physics engine
│
└── Documentation/ (5 files, 50+ KB)
    ├── START_HERE.md                  (this file)
    ├── QUICK_REFERENCE.md             (recommended first read)
    ├── PHYSFLOW_VISUALIZATION_GUIDE.md
    ├── PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md
    ├── PHYSFLOW_VIZ_INDEX.md
    ├── physflow_visualization_summary.md
    └── DOCUMENTATION_MANIFEST.txt
```

---

## 🚀 Quick Start (3 Steps, 10 minutes)

### Step 1: View the Results

```bash
cat output/physflow_v2_compare/comparison_report.txt
```

This shows:
- **Per-prompt comparison table** (19 rows)
  - Pretrained vs Fine-tuned
  - Improvement delta
  - Examples: standing still (+0.47), walking forward (+0.50)

- **Summary statistics**
  - Average completion ratio
  - Success rate
  - Category breakdown

### Step 2: Understand the Data

```bash
# How many NPZ files were generated?
ls output/physflow_v2_compare/npz/ | wc -l
# Output: 76

# List files for one prompt
ls -1 output/physflow_v2_compare/npz/ | grep "stands_still"
```

### Step 3: Load in Python (Optional)

```python
import numpy as np

# Load one panel
npz = np.load('output/physflow_v2_compare/npz/pretrained_00_a_person_stands_still_raw.npz')

# Extract data
motion_135 = npz['motion_135']  # Shape: (120, 135)
fps = npz['fps']                # 30
prompt = npz['prompt']          # 'a person stands still'

# Inspect
print(f"Frames: {motion_135.shape[0]}")
print(f"Duration: {motion_135.shape[0] / fps:.2f} seconds")
```

---

## 📊 Latest Results Summary

### Top Improvements

| Motion | Pretrained | Fine-tuned | Δ | Status |
|--------|-----------|-----------|---|--------|
| stands still | 0.41 | 0.88 | +0.47 | ✅ |
| walks forward | 0.38 | 0.88 | +0.50 | ✅ |
| jumps in place | 0.40 | 0.79 | +0.40 | ✅ |
| turns around | 0.32 | 0.67 | +0.35 | ✅ |
| squats | 0.23 | 0.40 | +0.18 | ✅ |

### Category Breakdown

| Category | Pretrained | Fine-tuned | Improvement |
|----------|-----------|-----------|------------|
| Standing | 0.282 | 0.493 | **+0.212** ⭐ BEST |
| Walking | 0.446 | 0.549 | +0.103 |
| Transitions | 0.494 | 0.631 | +0.137 |
| Upper Body | 0.448 | 0.398 | -0.050 |
| Dynamic | 0.502 | 0.498 | -0.004 |

**Overall:** +6.2% (+14%) improvement in average completion ratio

---

## 🔑 Key Concepts

### What is motion_135?

All motions use the **motion_135** format:
```
Shape: (T, 135) where T = number of frames

Layout:
  [0:3]    → Translation (X, Y, Z position)
  [3:9]    → Root rotation (6D)
  [9:135]  → Body pose (21 joints × 6D)
```

### What do the metrics mean?

| Metric | Meaning | Good Value |
|--------|---------|-----------|
| **completion_ratio** (c) | % of motion before falling | > 0.8 |
| **root_height_min** (h) | Lowest point in sim | > 0.3 m |
| **status** | Did it succeed? | 'success' |

Example from report:
```
a person stands still | fell c=0.41 h=0.26 | fell c=0.88 h=0.27 | +0.47
                        └─ Pretrained      └─ Fine-tuned        └─ Δ
```

### What does "RL" mean?

**RL** = Reinforcement Learning physics correction

- **Raw**: Motion as generated by model (may be unrealistic)
- **RL**: Motion after physics simulator (more stable)

---

## 📖 Documentation Roadmap

### For First-Time Users
1. **This file** (START_HERE.md) - 5 min
2. [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) - 3 min
3. [`PHYSFLOW_VISUALIZATION_GUIDE.md`](PHYSFLOW_VISUALIZATION_GUIDE.md) - 10 min

Total: **18 minutes** to full understanding

### For Developers
1. [`PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md`](PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md) - 10 min
2. `scripts/embodied/physflow_visualize_compare.py` - 30 min

Total: **40 minutes** to code understanding

### For Researchers
1. [`physflow_visualization_summary.md`](physflow_visualization_summary.md) - 10 min
2. `output/physflow_v2_compare/comparison_results.json` - 10 min

Total: **20 minutes** to results analysis

---

## ❓ Common Questions

**Q: What exactly are the 4 panels?**
A: See the diagram in "The 4-Panel Concept Explained" above.

**Q: Why are the NPZ files different sizes?**
A: Physics simulation stops when character falls. Smaller = fell earlier = worse.

**Q: Can I visualize the motions?**
A: Yes, use `motion_annot_web` or `embodied_viz` tools with NPZ files.

**Q: How do I run the comparison again?**
A: `bash output/physflow_v2_compare/run_compare.sh` (15-20 min on GPU)

**Q: Where's the actual visualization code?**
A: `scripts/embodied/physflow_visualize_compare.py` (main logic)

**Q: What's the file structure of NPZ files?**
A: 
```python
{
  'motion_135': np.ndarray (120, 135),
  'fps': 30,
  'prompt': str,
  'rl_status': 'success' | 'fell'
}
```

---

## ✅ Status

- ✅ System: **Complete and tested**
- ✅ Results: **Generated May 25, 2026**
- ✅ Documentation: **Comprehensive**
- ✅ Files: **76 NPZ + 2 reports**
- ✅ Code: **Functional**

---

## 🎯 Next Steps

Choose one:

1. **Want to understand the system?**
   → Read: [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md)

2. **Want to see the code?**
   → Read: [`PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md`](PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md)

3. **Want to analyze results?**
   → Command: `cat output/physflow_v2_compare/comparison_report.txt`

4. **Want a comprehensive guide?**
   → Read: [`PHYSFLOW_VISUALIZATION_GUIDE.md`](PHYSFLOW_VISUALIZATION_GUIDE.md)

5. **Want to find something specific?**
   → See: [`PHYSFLOW_VIZ_INDEX.md`](PHYSFLOW_VIZ_INDEX.md)

---

## 📞 Need Help?

All documentation is in this directory:
- `QUICK_REFERENCE.md` - Quick overview
- `PHYSFLOW_VISUALIZATION_GUIDE.md` - Detailed walkthrough
- `PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md` - Code details
- `PHYSFLOW_VIZ_INDEX.md` - Master index
- `physflow_visualization_summary.md` - Technical analysis
- `DOCUMENTATION_MANIFEST.txt` - Complete manifest

---

**Last Updated:** May 25, 2026  
**Status:** ✅ Complete  
**Ready to explore?** Start with [`QUICK_REFERENCE.md`](QUICK_REFERENCE.md) →
