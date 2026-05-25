# Loss Spike Fixes — Quick Start for Validation

**Status**: ✅ Ready to launch  
**Time**: May 13, 2026

---

## TL;DR

Two P0 fixes for loss spikes have been implemented and committed:
1. **Fix 1**: max_grad_norm=2.0 (prevents gradient clipping crush)
2. **Fix 2**: Spike detection with 0.3× downweighting (reduces spike propagation)

Expected impact: 60-75% spike reduction across E1/E2/E4.

---

## Launch Validation (3 Steps)

### 1️⃣ Start Training (2-3 hours)
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python3 -m torch.distributed.launch --nproc_per_node=8 scripts/train.py \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py
```

### 2️⃣ Analyze Results (2 min, after training completes)
```bash
python3 scripts/analysis/extract_loss_curves.py \
    work_dirs/hymotion_m2m_v2_uncond_local_046b_validation \
    --output docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md
```

### 3️⃣ Review Report
Open `docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md` and check:
- ✅ Spike Frequency Reduction (target: <30% of steps)
- ✅ Spike Magnitude Reduction (target: peak <50% of baseline)
- ✅ Learning Stability (target: no NaN/Inf)
- ✅ Model Convergence (target: loss <0.025)
- ✅ Detection Active (target: 20-40% of steps trigger)

**Result**: All 5 pass → Ready for production training  
**Result**: Any fail → Debug per LOSS_SPIKE_FIXES_VALIDATION_PLAN.md

---

## Monitoring During Training

Every 30 min, check:
```bash
# Latest logs
tail -50 work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/work_dir*/train.log

# Epoch progress
ls work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/work_dir*/checkpoints/epoch_*.pth | wc -l

# Errors?
grep -i "error\|warning" work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/work_dir*/train.log | tail -5
```

---

## Expected Metrics (During Training)

| Metric | Expected | Red Flag |
|--------|----------|----------|
| Epoch 1 loss | 0.02-0.03 | >0.05 |
| Epoch 10 loss | <0.025 | >0.03 |
| Spikes/epoch | 100-200 detected | 0 or >500 |
| Training time | ~2.5 hrs | >4 hrs |

---

## Key Files

| File | Purpose |
|------|---------|
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py` | Validation config (ready to use) |
| `scripts/analysis/extract_loss_curves.py` | Analysis script (executable) |
| `docs/LOSS_SPIKE_FIXES_VALIDATION_PLAN.md` | Detailed validation strategy |
| `docs/LOSS_SPIKE_FIXES_SUMMARY.md` | Full technical summary |
| `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py:24-275` | Fix 2 implementation |
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py:225` | Fix 1 (max_grad_norm=2.0) |

---

## What Was Done (This Session)

✅ Verified both fixes implemented and committed  
✅ Updated validation config (loads T2M pretrained weights)  
✅ Created analysis script (parses logs, tests success criteria)  
✅ Generated 5 comprehensive documentation files  
✅ Pre-flight checklist complete  

**Everything is ready. You can launch immediately.**

---

## Success Criteria Details

### 1. Spike Frequency Reduction
- **Baseline**: 11-47% of steps show >20% loss jump
- **Target**: <30% of steps above threshold
- **Measure**: `Analysis script → Spike Frequency section`

### 2. Spike Magnitude Reduction
- **Baseline**: Max spike 0.81x-8.2x above mean
- **Target**: Max spike <50% of baseline peak
- **Measure**: `Analysis script → Peak Spike Magnitude section`

### 3. Learning Stability
- **Target**: No NaN/Inf in loss_velocity across all epochs
- **Measure**: Log parsing for error patterns

### 4. Model Convergence
- **Baseline**: Epoch 485 at ~0.025
- **Target**: Epoch 10 loss_velocity <0.025
- **Measure**: Extract epoch_10 loss_velocity from report

### 5. Detection Active
- **Target**: 20-40% of steps trigger spike downweighting
- **Measure**: Count downweight applications in logs

---

## If Validation Fails

1. Review `docs/LOSS_SPIKE_FIXES_VALIDATION_PLAN.md` (Debugging section)
2. Check which criterion failed
3. Common issues:
   - **High spike frequency?** → Threshold may need tuning
   - **No detection?** → Check logs for feature activation
   - **NaN/Inf?** → Usually indicates numerical instability elsewhere
4. Options:
   - Adjust `spike_detection_std_threshold` (default 2.0)
   - Adjust `spike_downweight_factor` (default 0.3)
   - Disable feature: `spike_downweight_enabled=False`
   - Revert max_grad_norm to 1.0

---

## Questions?

- **Implementation details?** → See `docs/LOSS_SPIKE_FIXES_VERIFICATION.md`
- **Why these thresholds?** → See `docs/LOSS_SPIKE_FIXES_SUMMARY.md`
- **Debugging help?** → See `docs/LOSS_SPIKE_FIXES_VALIDATION_PLAN.md`
- **Quick reference?** → See `docs/NEXT_STEPS_VALIDATION_TRAINING.md`

---

## Timeline

| Step | Duration | Status |
|------|----------|--------|
| Launch training | 0 min | ⏳ Ready |
| Training runs | 150 min | ⏳ Pending |
| Analysis | 2 min | ⏳ Pending |
| Review | 10 min | ⏳ Pending |
| **Decision** | **3.5 hrs** | **⏳ Target: All 5 pass** |

---

**READY TO LAUNCH** ✅

Next action: Run Step 1️⃣ command above.

