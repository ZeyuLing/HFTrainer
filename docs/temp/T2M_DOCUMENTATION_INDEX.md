# HyMotion T2M 1.0 Documentation Index

## 📋 What I've Created For You

I've created a comprehensive documentation set for understanding and running HyMotion T2M 1.0. **Start here:**

### 1️⃣ **README_T2M.md** — Entry Point
   - Overview of all documentation
   - Quick start commands
   - FAQ section
   - 10 minutes to read

### 2️⃣ **T2M_QUICK_REFERENCE.md** — Cheat Sheet
   - Key facts table
   - Running commands (single GPU, multi-GPU)
   - Result JSON structure
   - Troubleshooting tips
   - 5 minutes to read

### 3️⃣ **HYMOTION_T2M_GUIDE.md** — Complete Technical Guide
   - Configuration parameters (detailed)
   - Inference execution details
   - Output format specifications
   - Current status & limitations
   - How-to guide
   - 20 minutes to read

### 4️⃣ **NPZ_FORMAT_DETAILS.md** — Data Format Specification
   - NPZ file structure (3 keys explained)
   - Expected vs. current format
   - Metric derivations from data
   - Code examples
   - Data loading recipes
   - 15 minutes to read

---

## 🎯 Your Questions Answered

### Q1: How to run HyMotion T2M 1.0 inference?

**Answer**: Use the eval script:

```bash
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --gpus 0 \
    --num-steps 50 \
    --cfg-scale 5.0 \
    --output-dir work_dirs/t2m_test
```

See **T2M_QUICK_REFERENCE.md** for more variations.

---

### Q2: What is the output format?

**Answer**: NPZ files with **3 keys**:

| Key | Shape | Content |
|-----|-------|---------|
| `motion_135` | (T, 135) | Transl(3) + 6D rotations(132) |
| `positions` | (T, 22, 3) | World-space joint positions |
| `translation` | (T, 3) | Root translation (redundant) |

**Example**:
```python
import numpy as np
data = np.load('sample.npz')
motion_135 = data['motion_135']     # (T, 135)
positions = data['positions']       # (T, 22, 3)
```

See **NPZ_FORMAT_DETAILS.md** for complete specifications.

---

### Q3: Does the output have both motion_135 and positions fields?

**Answer**: YES ✅

Current NPZ output contains:
- ✅ `motion_135` (135-dim: transl + rot6d)
- ✅ `positions` (computed via Forward Kinematics)
- ✅ `translation` (redundant copy of first 3 dims)

**Missing**:
- ❌ `motion_201` (would be transl + rot6d + positions flattened)

The 201-dim checkpoint exists, but the data pipeline only trains on 135 dims. Position channel is computed post-hoc.

See **HYMOTION_T2M_GUIDE.md** section "Current Status & Limitations" for details.

---

## 📚 Quick Navigation

| Need | Read This | Time |
|------|-----------|------|
| Get started fast | **README_T2M.md** + **T2M_QUICK_REFERENCE.md** | 15 min |
| Run single GPU | **T2M_QUICK_REFERENCE.md** line "Minimal Example" | 2 min |
| Run multi-GPU CFG | **T2M_QUICK_REFERENCE.md** line "Multi-GPU" | 5 min |
| Understand NPZ format | **NPZ_FORMAT_DETAILS.md** section "Current NPZ Keys" | 10 min |
| Load & process data | **NPZ_FORMAT_DETAILS.md** section "Loading & Processing NPZ" | 5 min |
| Deep dive technical | **HYMOTION_T2M_GUIDE.md** | 30 min |
| See examples | **README_T2M.md** section "Common Tasks" | 10 min |
| Check metrics | **NPZ_FORMAT_DETAILS.md** section "Metric Derivations" | 8 min |

---

## 🔍 Key Facts Summary

| Aspect | Value |
|--------|-------|
| **Config File** | `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py` |
| **Checkpoint** | `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt` (201-dim) |
| **Model Size** | 0.46B parameters |
| **Motion Dim** | 201 (but data pipeline = 135) |
| **Inference Script** | `scripts/eval/eval_m2m_v2_t2m.py` |
| **Pipeline Code** | `hftrainer/pipelines/motion/hymotion_t2m_pipeline.py` |
| **Default ODE Steps** | 50 (Euler method) |
| **Text Encoders** | Qwen3 (4096-dim) + CLIP-L (768-dim) |
| **CFG Support** | Yes (Classifier-Free Guidance) |
| **Multi-GPU** | Yes (via prompt chunking) |
| **NPZ Keys** | 3 (motion_135, positions, translation) |
| **Test Data** | 240 prompts in `data/eval/t2m/251125_yiran_subset.json` |

---

## 🚀 Quick Start (Copy & Paste)

### Minimal Command

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
python scripts/eval/eval_m2m_v2_t2m.py \
    --models caption_local_phase2 \
    --gpus 0 \
    --output-dir work_dirs/t2m_quick
```

### Check Results

```bash
# View metrics
cat work_dirs/t2m_quick/caption_local_phase2/result.json | head -50

# Load motion
python -c "
import numpy as np
data = np.load('work_dirs/t2m_quick/caption_local_phase2/npz/00000001.npz')
print(f'motion_135: {data[\"motion_135\"].shape}')
print(f'positions: {data[\"positions\"].shape}')
"
```

---

## ⚠️ Known Limitations

### Data Pipeline Mismatch

The config declares 201-dim, but the data pipeline outputs only 135-dim:

```
Config: _motion_dim = 201
Output: 3 (transl) + 132 (rot6d) = 135
Missing: 66 (positions channel)
```

**Impact**:
- ✅ Checkpoint loads fine (201-dim exists in file)
- ✅ Inference works (network processes 201-dim)
- ✅ Evaluation outputs positions (computed via FK)
- ❌ Training doesn't learn position channel

**To fix**: Extend `LoadSmplx55` in data pipeline to output position channel. See **HYMOTION_T2M_GUIDE.md** for details.

---

## 📦 File Structure

```
Working Directory: /apdcephfs/AILab_DHA/.../hf_trainer/

Documentation:
├── README_T2M.md ⭐ START HERE
├── T2M_QUICK_REFERENCE.md (5-min cheat sheet)
├── HYMOTION_T2M_GUIDE.md (complete technical guide)
├── NPZ_FORMAT_DETAILS.md (data format specs)
└── T2M_DOCUMENTATION_INDEX.md (this file)

Code:
├── configs/hymotion_t2m/hymotion_t2m_201dim_046b.py
├── scripts/eval/eval_m2m_v2_t2m.py
├── hftrainer/pipelines/motion/hymotion_t2m_pipeline.py
└── hftrainer/pipelines/motion/differentiable_fk.py

Checkpoints:
├── checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
└── data/hymotion_m2m_data/bone_offsets_22.pt

Data:
├── data/eval/t2m/251125_yiran_subset.json (240 prompts)
└── data/motionhub/ (training data)

Results:
├── work_dirs/m2m_v2_t2m_eval/ (existing results)
├── work_dirs/m2m_v2_t2m_eval_cfg_ablation*/
└── work_dirs/t2m_test/ (your outputs)
```

---

## 💡 Pro Tips

1. **Read in order**: README_T2M → T2M_QUICK_REFERENCE → HYMOTION_T2M_GUIDE
2. **Run first, understand later**: Copy the minimal command and run it
3. **Check output early**: Look at `result.json` to understand metrics
4. **Use existing results**: Compare with `work_dirs/m2m_v2_t2m_eval/*/result.json`
5. **Debug with code**: Use the Python examples in NPZ_FORMAT_DETAILS.md

---

## 🔗 Critical Paths to Remember

```bash
# Main inference script
scripts/eval/eval_m2m_v2_t2m.py

# Config
configs/hymotion_t2m/hymotion_t2m_201dim_046b.py

# Pipeline code
hftrainer/pipelines/motion/hymotion_t2m_pipeline.py

# Checkpoint
checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt

# Test prompts
data/eval/t2m/251125_yiran_subset.json

# Your outputs
work_dirs/t2m_test/
```

---

## ✅ Checklist: Have You Read?

- [ ] README_T2M.md (overview & quick start)
- [ ] T2M_QUICK_REFERENCE.md (facts & commands)
- [ ] This file (documentation index)
- [ ] Run the minimal command: `python scripts/eval/eval_m2m_v2_t2m.py ...`
- [ ] Load a sample NPZ: `np.load('work_dirs/t2m_test/.../00000001.npz')`
- [ ] Check result.json: `cat work_dirs/t2m_test/.../result.json`

Once done, you're ready to:
- ✅ Run inference
- ✅ Understand outputs
- ✅ Analyze metrics
- ✅ Debug issues

---

## 📞 Troubleshooting

### "Module not found" error
→ Check you're in the correct working directory
→ Verify Python path: `sys.path.insert(0, ...)`

### "Checkpoint not found"
→ Verify checkpoint exists: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
→ Use `--ckpt-path` to override

### "Out of memory"
→ Reduce batch size (not directly exposed, edit config)
→ Reduce `--num-steps` (default 50, try 25)
→ Reduce prompt chunks

### "Metrics look wrong"
→ Check NPZ file: `np.load(..., allow_pickle=True)`
→ Verify shapes match expectations in NPZ_FORMAT_DETAILS.md
→ Check bone_offsets file exists: `data/hymotion_m2m_data/bone_offsets_22.pt`

---

## 📊 What You Get

Running the inference produces:

```
work_dirs/t2m_test/
└── caption_local_phase2/
    ├── result.json                 # Aggregated metrics (20+ stats)
    ├── npz/
    │   ├── 00000001.npz            # motion_135, positions, translation
    │   ├── 00000002.npz
    │   └── ... (240 total)
    └── per_sample_chunk*.json      # (if multi-chunk mode)
```

**result.json contains**:
- `aggregated`: Mean/std/median/min/max for 20+ metrics
- `per_sample`: Per-motion metadata + individual metrics
- Metadata: model, checkpoint, cfg_scale, num_steps, speed, etc.

**Each NPZ contains**:
- `motion_135`: (T, 135) motion representation
- `positions`: (T, 22, 3) joint positions
- `translation`: (T, 3) root translation

---

## 🎓 Learning Path

**Beginner** (30 min):
1. Read README_T2M.md
2. Run minimal command
3. Load and inspect one NPZ file

**Intermediate** (1-2 hours):
1. Read T2M_QUICK_REFERENCE.md
2. Try multi-GPU inference
3. Run CFG ablation
4. Compare metrics

**Advanced** (3+ hours):
1. Read HYMOTION_T2M_GUIDE.md cover to cover
2. Read NPZ_FORMAT_DETAILS.md
3. Study `eval_m2m_v2_t2m.py` source code
4. Understand the data pipeline issue

---

**Created**: May 2026
**Last Updated**: Today
**Status**: Complete & Ready to Use

For questions, refer back to the documentation files. Everything is explained there.

