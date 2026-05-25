# PRISM Resources Index

**Generated:** 2026-05-18  
**Project:** hf_trainer  
**Base Path:** `/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

## Quick Navigation

### 🚀 Start Here
1. **For Quick Lookup:** [`PRISM_QUICK_REFERENCE.txt`](PRISM_QUICK_REFERENCE.txt) (6.1 KB)
2. **For Detailed Info:** [`PRISM_CHECKPOINT_AND_INFERENCE_GUIDE.md`](PRISM_CHECKPOINT_AND_INFERENCE_GUIDE.md) (14 KB)
3. **For Complete Inventory:** [`PRISM_FILES_MANIFEST.txt`](PRISM_FILES_MANIFEST.txt) (15 KB)

---

## Key Resources at a Glance

### Checkpoints
| Path | Size | Type | Status |
|------|------|------|--------|
| `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000` | 27 GB | Multi-frame T2M | ✅ Latest |
| `work_dirs/prism_1b_tp2m_1frame/checkpoint-iter_11000` | 27 GB | 1-frame T2M | ✅ Base |

### Inference Entry Point
| File | Size | Purpose |
|------|------|---------|
| `tools/infer.py` | 15 KB | Universal inference script |

### Configuration Files
| Location | Count | Types |
|----------|-------|-------|
| `configs/prism/` | 10 files | T2M, MCM, Debug variants |

### Evaluation Scripts
| File | Purpose |
|------|---------|
| `scripts/eval/eval_prism_t2m_hml3d.py` | Multi-GPU HML3D evaluation |
| `scripts/eval/run_prism_eval_fixed.sh` | Bash wrapper for eval |

---

## Recommended Workflow

### 1. Single Inference
```bash
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --prompt "a person walks forward" \
    --output output/motion.npz
```

### 2. Evaluation on Test Set
```bash
python scripts/eval/eval_prism_t2m_hml3d.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --output-dir eval_output/ \
    --gpus 0 1 2 3 4 5 6 7
```

### 3. With First-Frame Conditioning
```bash
python tools/infer.py \
    --config configs/prism/prism_1b_tp2m_multiframe.py \
    --checkpoint work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000 \
    --prompt "a person waves" \
    --first-frame-motion condition.npz \
    --output output/motion.npz
```

---

## Documentation Map

### Core Documentation (Newly Created)
- **PRISM_CHECKPOINT_AND_INFERENCE_GUIDE.md** - Comprehensive 11-section guide
- **PRISM_QUICK_REFERENCE.txt** - One-page quick lookup
- **PRISM_FILES_MANIFEST.txt** - Complete file inventory

### Related Existing Documentation
- **PRISM_EVALUATION_AND_INFERENCE_GUIDE.md** - Evaluation details
- **PRISM_TRAINER_QUICK_START.md** - Training basics
- **PRISM_CODEBASE_SUMMARY.md** - Architecture overview
- **README_PRISM_ANALYSIS.md** - Analysis and insights
- **PRISM_ACTION_PLAN.md** - Implementation plan

---

## Common Questions

**Q: Which checkpoint should I use?**  
A: Use `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000` - it's the latest and most capable.

**Q: How do I run inference?**  
A: Use `tools/infer.py` with the config and checkpoint paths above.

**Q: How many GPUs do I need?**  
A: 1 GPU minimum for inference. For evaluation, use 8 GPUs for faster processing.

**Q: Where's the eval script?**  
A: `scripts/eval/eval_prism_t2m_hml3d.py` for HML3D test set evaluation.

**Q: What's the difference between model variants?**  
A: See PRISM_FILES_MANIFEST.txt Section 2 for detailed config comparison.

---

## File Structure

```
project_root/
├── PRISM_CHECKPOINT_AND_INFERENCE_GUIDE.md    (← Start here for details)
├── PRISM_QUICK_REFERENCE.txt                  (← Start here for quick lookup)
├── PRISM_FILES_MANIFEST.txt                   (← Start here for inventory)
├── README_PRISM_RESOURCES.md                  (← You are here)
│
├── configs/prism/                             (10 config files)
├── tools/
│   ├── infer.py                               (Main inference script)
│   └── train.py
├── scripts/
│   ├── eval/
│   │   ├── eval_prism_t2m_hml3d.py           (Evaluation script)
│   │   ├── run_prism_eval_fixed.sh
│   │   └── eval_prism_kafs_ablation.py
│   └── debug/
│       ├── diagnose_prism_jitter.py
│       └── quick_prism_jitter_test.py
│
└── work_dirs/
    ├── prism_1b_tp2m_1frame/
    │   └── checkpoint-iter_11000/            (Base model - 27 GB)
    ├── prism_1b_tp2m_multiframe/
    │   └── checkpoint-iter_15000/            (Latest - 27 GB) ✅
    └── prism_mcm_motionhub/                  (MCM variants)
```

---

## Key Architecture Info

- **Model Type:** Diffusion-based Motion Transformer
- **Parameters:** ~1 Billion
- **Layers:** 30
- **Attention Heads:** 12
- **Text Encoding:** T5-XXL (4096-dim)
- **Motion Representation:** SMPL-X with rotation_6d
- **Conditioning:** Text + Multi-frame pose (1, 5, 9 frames)

---

## Training Stages

```
Stage 1: 1-Frame Base Model (iter_11000)
         ↓ (fine-tuned)
Stage 2: Multi-Frame Model (iter_15000) ← RECOMMENDED
         ↓ (used as pretrained)
Stage 3: MCM Control Transformer (optional)
```

---

## Support Resources

- **For Checkpoint Issues:** See Section 6 in PRISM_FILES_MANIFEST.txt
- **For Config Details:** See Section 4 in PRISM_CHECKPOINT_AND_INFERENCE_GUIDE.md
- **For Command Examples:** See PRISM_QUICK_REFERENCE.txt Sections 4-6
- **For Troubleshooting:** See PRISM_FILES_MANIFEST.txt Section 12

---

## Version Info

| Component | Version |
|-----------|---------|
| Checkpoint Latest | iter_15000 |
| Model Type | PrismBundle |
| PyTorch Required | 2.0+ |
| Python Required | 3.9+ |

---

**Last Updated:** 2026-05-18 23:40 UTC  
**Status:** ✅ Complete and Verified

For the most current information, see [`PRISM_CHECKPOINT_AND_INFERENCE_GUIDE.md`](PRISM_CHECKPOINT_AND_INFERENCE_GUIDE.md).
