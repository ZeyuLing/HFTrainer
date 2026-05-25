# PhysFlow 4-Panel Visualization - Complete Walkthrough

**Last Updated:** May 25, 2026  
**Status:** ✅ Complete and Tested

---

## 🎯 What is the 4-Panel Comparison?

The PhysFlow system generates a **4-panel motion comparison** that shows:

| Panel | Content | Status |
|-------|---------|--------|
| **Panel 1** | Pretrained Model → Raw Motion | Generated |
| **Panel 2** | Pretrained Model → RL Physics Corrected | Generated |
| **Panel 3** | Fine-tuned Model → Raw Motion | Generated |
| **Panel 4** | Fine-tuned Model → RL Physics Corrected | Generated |

These four versions are generated for **19 test prompts** covering different motion categories.

---

## 📊 Latest Results (May 25, 2026)

### Overall Performance

```
                     Pretrained  →  Fine-tuned    Δ
Average Completion:    0.438    →     0.500     +0.062 (+14%)
Success Rate:          0/19     →     2/19      +10.5%
```

### Best Improvements

| Motion Type | Pretrained | Fine-tuned | Gain |
|-------------|-----------|-----------|------|
| Standing   | 0.282     | 0.493     | **+0.212** ⭐ |
| Walking    | 0.446     | 0.549     | **+0.103** |
| Upper Body | 0.448     | 0.398     | -0.050 |
| Dynamic    | 0.502     | 0.498     | -0.004 |

### Standout Examples

- **"a person stands still"**: 0.41 → 0.88 (**+0.47**) ✅
- **"a person walks forward at normal pace"**: 0.38 → 0.88 (**+0.50**) ✅
- **"a person jumps in place"**: 0.40 → 0.79 (**+0.40**) ✅

---

## 📁 Where to Find Everything

```
hf_trainer/
├── scripts/embodied/
│   ├── physflow_visualize_compare.py    ← Main visualization script
│   ├── physflow_rl_oracle.py            ← Physics correction engine
│   └── physflow_eval_demo.py            ← Alternative 3-way demo
│
├── output/physflow_v2_compare/
│   ├── comparison_report.txt            ← Human-readable results
│   ├── comparison_results.json          ← Machine-readable metrics
│   ├── npz/
│   │   ├── pretrained_00_*_raw.npz     ← Panel 1 data
│   │   ├── pretrained_00_*_rl.npz      ← Panel 2 data
│   │   ├── finetuned_00_*_raw.npz      ← Panel 3 data
│   │   ├── finetuned_00_*_rl.npz       ← Panel 4 data
│   │   └── ... (76 total: 4 files × 19 prompts)
│   ├── run_compare.sh                   ← Run full comparison
│   └── launch_compare.sh                ← Helper script
│
└── Documentation/
    ├── QUICK_REFERENCE.md               ← Start here!
    ├── PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md
    └── physflow_visualization_summary.md
```

---

## 🔧 How It Works (4-Phase Pipeline)

### Phase 1: Load Pretrained Model & Generate

```bash
# Loads HY-Motion-1.0 pretrained model
# Generates 120-frame motions for 19 test prompts
# Uses T2M diffusion (50 ODE steps, CFG scale 5.0)
```

**Output:** 19 motion arrays, each shape (120, 135)

### Phase 2: Load Fine-tuned Model & Generate

```bash
# Loads PhysFlow fine-tuned model (blend50, iter 500)
# Generates same 19 prompts with fine-tuned weights
```

**Output:** 19 motion arrays, each shape (120, 135)

### Phase 3: Run RL Physics Simulation

```bash
# For EACH motion (38 total: 19 pretrained + 19 finetuned):
#   1. Save raw motion → {label}_{idx}__{prompt}_raw.npz
#   2. Run RLPhysicsOracle.correct() → motion_135_rl
#   3. Save corrected → {label}_{idx}_{prompt}_rl.npz
#   4. Collect physics metrics (completion_ratio, height, status)
```

**Output:** 76 NPZ files (4 per prompt)

### Phase 4: Generate Comparison Report

```bash
# Combine all metrics from Phase 3
# Generate:
#   - comparison_report.txt (human-readable table + stats)
#   - comparison_results.json (structured metrics)
#   - Per-category breakdown (standing/walking/upper_body/dynamic)
```

**Output:** Summary statistics and per-prompt comparison

---

## 🎬 Test Prompts (19 Total)

### Category: Standing (3)
1. "a person stands still" ✅ **Best improvement**
2. "a person stands in a relaxed pose"
3. "a person shifts weight from left to right foot"

### Category: Walking (4)
4. "a person walks forward at a normal pace" ✅ **Good improvement**
5. "a person walks in a small circle"
6. "a person walks forward slowly"
7. "a person walks with long strides"

### Category: Upper Body (3)
8. "a person waves with their right hand"
9. "a person raises both arms above their head"
10. "a person claps their hands together"
11. "a person stretches arms to the sides"

### Category: Transitions (3)
12. "a person walks and then stops"
13. "a person walks forward then turns around"
14. "a person jogs slowly then walks"

### Category: Dynamic (5)
15. "a person kicks with their right leg"
16. "a person squats down and stands back up"
17. "a person jumps in place" ✅ **Good improvement**
18. "a person does a jumping jack"
19. "a person does a high kick"

---

## 📊 Understanding the Metrics

| Metric | Meaning | Range | Good Value |
|--------|---------|-------|-----------|
| **completion_ratio** (c) | Fraction of motion completed before falling | 0.0 - 1.0 | > 0.8 |
| **root_height_min** (h) | Lowest height during simulation | meters | > 0.3 |
| **status** | Did it complete without falling? | 'success' or 'fell' | 'success' |
| **actual_sim_steps** | Frames simulated before falling | int | ~200 |
| **total_sim_steps** | Expected total frames | int | 200 |

### Example Row from Report
```
a person stands still | fell c=0.41 h=0.26 | fell c=0.88 h=0.27 | +0.47
                        └─ Pretrained      └─ Fine-tuned        └─ Improvement
```

---

## 💾 Data Format: motion_135

All visualizations use **motion_135** format:

```
Shape: (T, 135) where T = number of frames

Layout:
  [0:3]      → Translation (root X, Y, Z position)
  [3:9]      → Root rotation (6D rotation representation)
  [9:135]    → Body pose (21 joints × 6D each)

Total: 3 + 6 + (21 × 6) = 135 dimensions
```

### NPZ File Contents

```python
{
  'motion_135': np.ndarray (T, 135),     # Motion data
  'fps': 30,                              # Frame rate
  'prompt': 'a person stands still',      # Text prompt
  'rl_status': 'success' | 'fell'        # Physics result (optional)
}
```

---

## 🚀 Running the Full Comparison

### Quick Start (Using Saved Script)

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
bash output/physflow_v2_compare/run_compare.sh
```

### Manual Command

```bash
python3 scripts/embodied/physflow_visualize_compare.py \
    --t2m-config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \
    --pretrained-ckpt checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \
    --finetuned-ckpt output/physflow_v2_train_blend50/model_iter500.pt \
    --text-cache output/physflow_v2_test/text_embeddings.pt \
    --output-dir output/physflow_v2_compare \
    --device cuda:0 \
    --num-frames 120
```

**Runtime:** ~15-20 minutes on GPU
**Output:** 76 NPZ files + 2 report files

---

## 📖 Viewing & Analyzing Results

### View Human-Readable Report

```bash
cat output/physflow_v2_compare/comparison_report.txt
```

**Output:**
```
PhysFlow Visualization Comparison Report
================================================================================

Prompt                                   |      Pretrained      |      Fine-tuned      | Δ         
-----------------------------------------------------------------------------------------------
a person stands still                    |  fell c=0.41 h=0.26  |  fell c=0.88 h=0.27  | +0.47     
a person stands in a relaxed pose        |  fell c=0.21 h=0.29  |  fell c=0.20 h=0.29  | -0.01     
...

SUMMARY STATISTICS
  Total prompts evaluated: 19
  Pretrained:
    Avg completion:   0.438
    Success rate:     0/19 (0.0%)
  Fine-tuned (PhysFlow blend50, iter 500):
    Avg completion:   0.500
    Success rate:     2/19 (10.5%)
  Improvement:
    Avg completion:   +0.062
    Success rate:     +2 (+10.5%)

PER-CATEGORY BREAKDOWN
  standing (n=3): pretrained=0.282 → finetuned=0.493 (Δ=+0.212)
  walking (n=7): pretrained=0.446 → finetuned=0.549 (Δ=+0.103)
  ...
```

### View Machine-Readable JSON

```bash
cat output/physflow_v2_compare/comparison_results.json | python3 -m json.tool | head -100
```

### Count NPZ Files

```bash
ls output/physflow_v2_compare/npz/ | wc -l
# Output: 76
```

### Verify 4-Panel Structure

```bash
# List first 4 files for one prompt
ls -1 output/physflow_v2_compare/npz/*person_stands_still*

# Should show:
# finetuned_00_a_person_stands_still_raw.npz
# finetuned_00_a_person_stands_still_rl.npz
# pretrained_00_a_person_stands_still_raw.npz
# pretrained_00_a_person_stands_still_rl.npz
```

---

## 🔬 Examining Individual NPZ Files

### Load a Single NPZ File

```python
import numpy as np

# Load Panel 1: Pretrained Raw
npz = np.load('output/physflow_v2_compare/npz/pretrained_00_a_person_stands_still_raw.npz')
print(npz.files)  # ['motion_135', 'fps', 'prompt']

motion_135 = npz['motion_135']      # Shape: (120, 135)
fps = npz['fps']                     # 30
prompt = npz['prompt']               # 'a person stands still'

print(f"Motion shape: {motion_135.shape}")
print(f"Duration: {motion_135.shape[0] / fps:.2f} seconds")
```

### Compare Raw vs RL-Corrected

```python
import numpy as np

# Load both versions
raw = np.load('output/physflow_v2_compare/npz/pretrained_00_a_person_stands_still_raw.npz')
rl_corrected = np.load('output/physflow_v2_compare/npz/pretrained_00_a_person_stands_still_rl.npz')

motion_raw = raw['motion_135']
motion_rl = rl_corrected['motion_135']

print(f"Raw motion frames: {motion_raw.shape[0]}")
print(f"RL-corrected frames: {motion_rl.shape[0]}")
print(f"→ RL may be shorter due to early termination (falling)")

# Compare root position
root_pos_raw = motion_raw[:, :3]
root_pos_rl = motion_rl[:, :3]

print(f"\nRoot height (Y) - Raw: min={root_pos_raw[:, 1].min():.3f}, max={root_pos_raw[:, 1].max():.3f}")
print(f"Root height (Y) - RL:  min={root_pos_rl[:, 1].min():.3f}, max={root_pos_rl[:, 1].max():.3f}")
```

---

## 🎯 Key Code Locations

### Main Visualization Pipeline

**File:** `scripts/embodied/physflow_visualize_compare.py`

| Component | Lines | Purpose |
|-----------|-------|---------|
| `load_bundle_and_generate()` | 36-191 | Generate motions with T2M model |
| `run_rl_physics_evaluation()` | 194-257 | Run physics sim, save NPZ files |
| `print_comparison_report()` | 260-400 | Generate comparison metrics |
| `TEST_PROMPTS` | 406-431 | 19 test prompts across categories |
| `main()` | 434-569 | 4-phase pipeline orchestration |

### Physics Correction Engine

**File:** `scripts/embodied/physflow_rl_oracle.py`

| Component | Purpose |
|-----------|---------|
| `RLPhysicsOracle.correct()` | Runs MuJoCo physics sim on motion |
| Returns | (motion_135_rl, stats_dict) |

---

## 📈 Statistical Summary

### Metrics Across All 19 Prompts

```
Pretrained Model:
  • Avg completion_ratio:   0.438
  • Success rate:           0/19 (0.0%)
  • Motions that "fell":    19/19 (100%)

Fine-tuned Model:
  • Avg completion_ratio:   0.500
  • Success rate:           2/19 (10.5%)
  • Motions that "fell":    17/19 (89.5%)
  • Successful motions:     "a person walks forward then turns around"
                            + 1 other

Improvement:
  • Avg completion delta:   +0.062 (+14%)
  • Success rate delta:     +2 (+10.5%)
  • Best category:          Standing (+0.212, +21.2%)
  • Worst category:         Upper Body (-0.050, -5.0%)
```

---

## 🎬 External Visualization

The NPZ files are designed to be viewed through external tools:

### Option 1: motion_annot_web

```bash
cd /path/to/motion_annot_web
python3 app.py --npz-dir /path/to/hf_trainer/output/physflow_v2_compare/npz/
```

### Option 2: embodied_viz

Specialized 3D viewer for embodied motions with SMPL mesh rendering.

---

## ❓ FAQ

### Q: Why are there 4 files per prompt?

**A:** This is the "4-panel" concept:
1. **Raw** motion from pretrained model
2. **Physics-corrected** version of #1
3. **Raw** motion from fine-tuned model
4. **Physics-corrected** version of #3

Together they show: How much does fine-tuning improve physics compliance?

### Q: Why are some RL-corrected files smaller?

**A:** The physics simulation stops when the character falls. Smaller file = it fell earlier = worse physics stability.

### Q: What does "completion_ratio" mean?

**A:** It's the fraction of the original motion that completed before the character fell:
- 1.0 = completed entire motion without falling ✅
- 0.5 = fell halfway through
- 0.1 = fell very quickly

### Q: How is "success" determined?

**A:** A motion is successful if `completion_ratio > 0.99` (99% of frames completed).

### Q: Can I re-generate with different parameters?

**A:** Yes! Edit these in `physflow_visualize_compare.py`:
- Line ~50: `NUM_SAMPLES` (number of generations per prompt)
- Line ~30: `ODE_STEPS` (diffusion steps, 50 is current)
- Line ~40: `CFG_SCALE` (conditional guidance scale, 5.0 is current)

---

## 📋 Checklist: Is Everything Set Up?

- ✅ Pretrained model: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
- ✅ Fine-tuned model: `output/physflow_v2_train_blend50/model_iter500.pt`
- ✅ T2M config: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`
- ✅ Text cache: `output/physflow_v2_test/text_embeddings.pt`
- ✅ Physics oracle: `scripts/embodied/physflow_rl_oracle.py`
- ✅ Main script: `scripts/embodied/physflow_visualize_compare.py`
- ✅ Output directory: `output/physflow_v2_compare/`
- ✅ NPZ files: `output/physflow_v2_compare/npz/` (76 files)
- ✅ Reports: `comparison_report.txt` + `comparison_results.json`

---

## 📚 Related Documentation

For more details, see:
- `QUICK_REFERENCE.md` - Quick overview
- `PHYSFLOW_VISUALIZATION_CODE_REFERENCE.md` - Detailed code breakdown
- `physflow_visualization_summary.md` - Technical analysis

---

**Generated:** May 25, 2026  
**System:** PhysFlow Motion Generation + RL Physics Correction  
**Status:** ✅ Complete and Tested
