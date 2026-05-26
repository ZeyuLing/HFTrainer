# PhysFlow Quick Reference Card 🚀

**Last Updated**: 2026-05-26  
**Status**: 🟢 Phase 0 Complete + Phase 1 Ready

---

## Phase 0 Baseline (COMPLETE ✅)

### Results Summary
```
PPR:      0.331  (gate > 0.25)  ✓
FID:      0.537  (gate < 1.0)   ✓
Diversity: 0.716 (expected 0.70-0.80) ✓
```

### Files
- Config: `configs/experiments/physflow_phase0/phase0_baseline_c0.py`
- Results: `results/physflow_phase0/c0_baseline_t2m/metrics.json`
- Report: `PHASE0_RESULTS.md`

### Gate Status
✅ **PASSED** — Ready for Phase 1

---

## Phase 1 Direction B (READY 🟢)

### What It Does
```
T2M Generator → RL Policy Correction → Physics-Valid Motions
```

### Expected Results
```
PPR:       0.331 → 0.43-0.53  (+10-20%)
FID:       0.537 → < 0.70     (acceptable increase)
Diversity: 0.716 → > 0.70     (maintained)
```

### Gate Criteria (ALL must pass)
```
✓ PPR improvement ≥ 10%
✓ FID < 0.70
✓ Diversity > 0.70
✓ Training stable (loss converges)
```

---

## Launch Phase 1 (ONE COMMAND)

```bash
python3 scripts/embodied/launch_physflow_phase1.py \
    --config c1 \
    --direction b \
    --phase0-baseline results/physflow_phase0/c0_baseline_t2m/metrics.json \
    --num-train-steps 200000
```

**Runtime**: 2-4 hours on V100 GPU

---

## Phase 1 Key Hyperparameters

```
Algorithm:        PPO
Learning Rate:    1e-4 (RL), 5e-5 (T2M)
Reward Weights:   Physics 0.5, Tracking 0.3, Smooth 0.1, Text 0.1
Num Envs:         16
Horizon:          300 steps
Entropy Coef:     0.01
Training Steps:   200k
```

---

## Phase 1 Timeline

| Milestone | Time | PPR | FID |
|-----------|------|-----|-----|
| Start | 0h | 0.331 | 0.537 |
| 50k steps | 0.5h | ~0.350 | ~0.545 |
| 100k steps | 1h | ~0.370 | ~0.555 |
| 150k steps | 1.5h | ~0.400 | ~0.575 |
| 200k steps | 2h | 0.43-0.53 | 0.65-0.70 |

---

## Monitor Training

```bash
# TensorBoard (optional)
tensorboard --logdir runs/physflow_phase1_c1 --port 6006

# Check results after training
cat results/physflow_phase1/c1_direction_b_gen_rl/metrics.json
```

---

## After Phase 1 Completes

### If Gate Passes ✓
```
Proceed to Phase 2 (Bidirectional Training)
Expected: PPR > 0.50, FID < 0.65
```

### If Gate Fails ✗
```
Debug using PHASE1_LAUNCH_GUIDE.md troubleshooting section
Retry with adjusted hyperparameters
```

---

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `PHASE0_RESULTS.md` | Phase 0 results & analysis | 312 |
| `PHASE1_LAUNCH_GUIDE.md` | Phase 1 launch instructions | 380 |
| `SESSION_COMPLETION_2026-05-26.md` | Session summary | 418 |
| `phase1_direction_b_c1.py` | Phase 1 config | 267 |

---

## Critical Watchpoints

🔴 **PPR Not Improving** (< +5% after 50k)
→ Increase physics weight to 0.6-0.7

🔴 **FID Increasing Too Much** (> 0.75)
→ Reduce physics weight to 0.4, increase text to 0.15

🔴 **Training Diverges** (Loss explodes)
→ Reduce learning rates by 2x, check reward scaling

🔴 **Training Too Slow** (< 5k steps/hour)
→ Increase num_envs to 32, check GPU

---

## Repository Status

```
✓ Working directory: clean
✓ All work committed
✓ Ready for Phase 1 launch
```

Latest commit: `2072191` (Session completion summary)

---

## Quick Commands

```bash
# Verify Phase 1 config
python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('config', 'configs/experiments/physflow_phase1/phase1_direction_b_c1.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('✓ Config OK')"

# View Phase 0 baseline
cat results/physflow_phase0/c0_baseline_t2m/metrics.json | python3 -m json.tool

# Prepare Phase 1 environment
mkdir -p results/physflow_phase1/c1_direction_b_gen_rl/
cp results/physflow_phase0/c0_baseline_t2m/metrics.json results/physflow_phase1/phase0_baseline_reference.json

# Launch Phase 1
python3 scripts/embodied/launch_physflow_phase1.py --config c1 --direction b --phase0-baseline results/physflow_phase0/c0_baseline_t2m/metrics.json --num-train-steps 200000
```

---

## Project Status

### ✅ Phase 0: COMPLETE
- Baseline established
- Gate: PASSED
- Metrics: PPR 0.331, FID 0.537, Div 0.716

### 🟢 Phase 1: READY TO LAUNCH
- Configuration: ✓
- Launch guide: ✓
- Hyperparameters: ✓
- Documentation: ✓

### Phase 2 (Preview)
- Bidirectional Training: RL ↔ T2M
- Expected improvement: PPR > 0.50, FID < 0.65

---

## Next Action

```bash
# Execute Phase 1 immediately
python3 scripts/embodied/launch_physflow_phase1.py --config c1 --direction b --phase0-baseline results/physflow_phase0/c0_baseline_t2m/metrics.json --num-train-steps 200000
```

**Expected Duration**: 2-4 hours  
**Success Criteria**: PPR ≥ 0.364, FID < 0.70, Div > 0.70

---

**Project**: PhysFlow  
**Phase**: Phase 0 Complete + Phase 1 Ready  
**Date**: 2026-05-26  
**Status**: 🟢 Ready to Launch

