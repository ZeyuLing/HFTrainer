# PhysFlow Phase 0 — Quick Start Guide

**Status**: 🟢 Ready to Launch  
**Last Updated**: 2026-05-26

---

## TL;DR — Launch Phase 0 in 2 Commands

### Step 1: Verify Setup (5 minutes)

```bash
python3 scripts/embodied/launch_physflow_phase0.py --config c0 --dry-run
```

Expected output:
```
✓ CUDA Available: True
✓ ProtoMotions: True
✓ HyMotion Checkpoint: True
✓ MuJoCo: True
Dry-run mode: Environment verified, config loaded, ready to run
```

### Step 2: Run Phase 0 Baseline (2-4 hours)

```bash
python3 scripts/embodied/physflow_evaluate.py \
    --config configs/experiments/physflow_phase0/phase0_baseline_c0.py \
    --num-samples 200 \
    --compute-fid \
    --compute-r-precision \
    --compute-ppr
```

---

## What is Phase 0?

Establishes **baseline metrics** for PhysFlow:
- **C0**: Pure T2M generation (no RL correction)
- **Expected metrics**: FID ~0.50-0.60, PPR ~30-50%
- **Purpose**: Foundation for measuring improvement in Phases 1-3

---

## Phase 0 Outputs

Results saved to: `results/physflow_phase0/c0_baseline_t2m/`

```
├── metrics.json              # FID, R-Prec, Diversity, PPR
├── generated_motions.pkl     # Generated motion data
├── experiment_metadata.json  # Experiment info and timestamp
└── videos/                   # Motion visualizations (if enabled)
```

---

## Key Metrics Explained

| Metric | Meaning | Expected Value |
|--------|---------|-----------------|
| **FID** | Generation quality (lower=better) | 0.50-0.60 |
| **R-Prec** | Text-motion alignment | 0.30-0.40 |
| **Diversity** | Multimodality | 0.70-0.80 |
| **PPR** | Physics plausibility rate | 30-50% |

---

## Phase 0 Success Criteria

✅ **Gate to Phase 1**: `PPR > 0.25 AND FID < 1.0`

If both conditions met → Proceed to Phase 1 (Direction B)  
If not met → Debug and retry Phase 0

---

## Files Structure

```
configs/experiments/physflow_phase0/
  └── phase0_baseline_c0.py          ← Phase 0 config

scripts/embodied/
  ├── launch_physflow_phase0.py       ← Launcher (use this!)
  ├── physflow_evaluate.py            ← Metrics computation
  └── physflow_trainer.py             ← Main trainer (Phase 1+)

results/physflow_phase0/
  └── c0_baseline_t2m/                ← Output directory (auto-created)

docs/
  ├── PHYSFLOW_STATUS_2026-05-26.md   ← Full status
  ├── PHASE0_LAUNCH_READY.md          ← Detailed readiness
  └── DOCUMENTATION_INDEX.md          ← All docs index
```

---

## Troubleshooting

### "CUDA not available"
```bash
# Check CUDA
nvidia-smi

# If no GPU, can still run on CPU (very slow)
python3 scripts/embodied/launch_physflow_phase0.py --config c0 --dry-run
```

### "HyMotion checkpoint not found"
```bash
# Verify checkpoint exists
ls -lh checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt

# If missing, download from HuggingFace
# Instructions in PHASE0_LAUNCH_READY.md
```

### "ProtoMotions import error"
```bash
# Update ProtoMotions
cd ref_repo/ProtoMotions
git pull origin main
cd ../..

# Verify import
python3 -c "import sys; sys.path.insert(0, 'ref_repo/ProtoMotions'); from protomotions.components.motion_lib import MotionLib; print('OK')"
```

### "Out of memory"
```bash
# Reduce batch size in config or command line
python3 scripts/embodied/physflow_evaluate.py \
    --config configs/experiments/physflow_phase0/phase0_baseline_c0.py \
    --num-samples 50 \
    --batch-size 8
```

---

## Typical Timeline

| Step | Time | Command |
|------|------|---------|
| 1. Verify setup | 5 min | `launch_physflow_phase0.py --dry-run` |
| 2. Run Phase 0 | 2-4 hours | `physflow_evaluate.py --config c0 --num-samples 200` |
| 3. Analyze results | 30 min | Check `metrics.json` and `experiment_metadata.json` |
| 4. Check gate | 5 min | Verify `PPR > 0.25 AND FID < 1.0` |

**Total Wall-Clock Time**: 2.5-4.5 hours

---

## What Next?

### If Phase 0 Gate Passes ✅

```bash
# Proceed to Phase 1: Direction B (Gen→RL)
# Read: docs/PHYSFLOW_STATUS_2026-05-26.md → Phase 1 section
# Expected: Improved RL tracker on T2M-generated motions
```

### If Phase 0 Gate Fails ❌

```bash
# Debug and retry Phase 0
# Check PHASE0_LAUNCH_READY.md → Troubleshooting section
# Possible issues:
#   - T2M model needs fine-tuning
#   - Physics simulator configuration
#   - Metrics computation error
```

---

## Documentation

For detailed information:
- **Status**: `PHYSFLOW_STATUS_2026-05-26.md`
- **Readiness**: `PHASE0_LAUNCH_READY.md`
- **Index**: `DOCUMENTATION_INDEX.md`
- **Full spec**: `docs/temp/physflow_experiment_spec.md`

---

## Support

**Issues during Phase 0?**

1. Check `PHASE0_LAUNCH_READY.md` → Risk Mitigation section
2. Review launcher output: `results/physflow_phase0/*/phase0_*.log`
3. Check environment: `python3 scripts/embodied/launch_physflow_phase0.py --dry-run`

---

**Ready?** Run: `python3 scripts/embodied/launch_physflow_phase0.py --config c0 --dry-run`

🟢 All systems go!
