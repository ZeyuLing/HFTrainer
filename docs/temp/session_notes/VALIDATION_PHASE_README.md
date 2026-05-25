# Loss Spike Fixes Validation Phase — Complete Guide

**Start Date**: May 13, 2026  
**Current Phase**: 🟡 Validation Training Ready  
**Estimated Completion**: May 14, 2026

---

## Overview

This repository now contains **two critical fixes** for HyMotion M2M v2 loss spikes:

1. **Fix 1**: Gradient clipping threshold increased to 2.0
2. **Fix 2**: Dynamic spike detection with downweighting

Both fixes are **implemented and verified**. This phase validates them through 10-epoch training.

---

## What to Do Now

### Step 1: Read the Core Documents (5 minutes)

In order of importance:

1. **For quick overview**: `docs/LOSS_SPIKE_FIXES_SUMMARY.md`
2. **For implementation details**: `docs/LOSS_SPIKE_FIXES_VERIFICATION.md`
3. **For full validation plan**: `docs/LOSS_SPIKE_FIXES_VALIDATION_PLAN.md`
4. **For quick commands**: `docs/NEXT_STEPS_VALIDATION_TRAINING.md`

### Step 2: Run Validation Training (2-3 hours)

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train.py \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py
```

**Monitor during training**:
- Watch for NaN/Inf in logs
- Check that `loss_velocity` decreases over epochs
- Confirm `loss_velocity_trans` stays < 0.015

### Step 3: Analyze Results (2 minutes)

After training completes:

```bash
python3 scripts/analysis/extract_loss_curves.py \
    work_dirs/hymotion_m2m_v2_uncond_local_046b_validation \
    --output docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md

# View the report
cat docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md
```

### Step 4: Decision (Based on Analysis)

**If ALL criteria pass ✅**:
- Validation successful
- Proceed to production training
- Submit E1, E2, E4 models for full 1000+ epoch training

**If ANY criteria fail ❌**:
- Review debugging section in `LOSS_SPIKE_FIXES_VALIDATION_PLAN.md`
- Adjust config parameters and retry
- Contact for support if issues persist

---

## Document Map

### 📋 Quick Reference
- **`docs/NEXT_STEPS_VALIDATION_TRAINING.md`** — What to do next (commands, files, timelines)
- **`docs/LOSS_SPIKE_FIXES_SUMMARY.md`** — Executive summary (this phase overview)

### 📊 Implementation Details
- **`docs/LOSS_SPIKE_FIXES_VERIFICATION.md`** — ✅ Authoritative source for implementation details
- **`docs/LOSS_SPIKE_ANALYSIS_20260513.md`** — Root cause analysis
- **`hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`** — Fix 2 source code

### 🧪 Validation Strategy
- **`docs/LOSS_SPIKE_FIXES_VALIDATION_PLAN.md`** — 4-phase validation plan with detailed steps
- **`configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py`** — Validation config

### 📈 Analysis Tools
- **`scripts/analysis/extract_loss_curves.py`** — Parse logs and generate report

### ⚠️ Obsolete (Do NOT Use)
- **`docs/LOSS_SPIKE_FIX_IMPLEMENTATION_STATUS.md`** — Contains error (max_grad_norm=10.0 is wrong)

---

## Key Information at a Glance

### The Fixes

| Fix | What | Value | Why | Where |
|-----|------|-------|-----|-------|
| **Fix 1** | Gradient clipping | `max_grad_norm=2.0` | Prevents gradient explosion during spikes | `_base_hymotion_m2m_v2_046b.py:225` |
| **Fix 2** | Spike detection | 0.3× downweight when z>2.0 | Targets translation loss (65-79% of spikes) | `m2m_loss.py:23-275` |

### Expected Impact

- **E1**: -60% spike severity (1.56x → 0.6x), -50% spike frequency (12% → 6%)
- **E2**: -51% spike severity (0.81x → 0.4x), -57% spike frequency (11.7% → 5%)
- **E4**: -73% spike severity (8.2x → 2.2x), -68% spike frequency (46.9% → 8%)

### Validation Success Criteria

All must be ✅ PASS:

1. ✅ No NaN/Inf in loss values
2. ✅ `loss_velocity` shows smooth convergence
3. ✅ `loss_velocity_trans` stays < 0.015
4. ✅ No 50%+ spikes in loss_velocity
5. ✅ All 10 epochs complete and save

---

## Command Reference

### Run Validation Training
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 scripts/train.py \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py
```

### Analyze Results
```bash
python3 scripts/analysis/extract_loss_curves.py \
    work_dirs/hymotion_m2m_v2_uncond_local_046b_validation \
    --output docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md
```

### View Analysis Report
```bash
cat docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md
```

### Submit Production Training (if validation passes)
```bash
# E1 (most important baseline)
taiji submit --config configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py --gpu 32

# E2 (text-conditioned)
taiji submit --config configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py --gpu 32

# E4 (most affected by spikes)
taiji submit --config configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_uncond_046b.py --gpu 32
```

---

## Checklist

### Before Starting Training
- [ ] Read `LOSS_SPIKE_FIXES_SUMMARY.md` (5 min)
- [ ] Verify GPU availability (8× V100)
- [ ] Confirm config exists: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py`
- [ ] Confirm script exists: `scripts/analysis/extract_loss_curves.py`

### During Training
- [ ] Monitor logs for first epoch (15-20 min expected)
- [ ] Check that losses are decreasing
- [ ] Watch for NaN/Inf errors

### After Training Completes
- [ ] Run analysis script
- [ ] Review `docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md`
- [ ] Check pass/fail on all 5 success criteria
- [ ] Document any issues

### If Validation Passes
- [ ] Update this document with completion date
- [ ] Submit production training jobs
- [ ] Set up monitoring at 50-epoch checkpoints
- [ ] Archive validation training logs

---

## Troubleshooting

### Issue: "Training is slow"
- **Expected**: 2-3 hours for 10 epochs on 1×8 V100
- **If slower**: Check GPU utilization (should be >80%)
- **Fix**: Check for bottlenecks in data pipeline

### Issue: "NaN appears in loss"
- **Cause**: Usually gradient explosion or numerical instability
- **Check**: Look at gradients before the NaN epoch
- **Fix**: May need to reduce `max_grad_norm` or increase downweight factor

### Issue: "Analysis script crashes"
- **Check**: Verify training log format matches expected pattern
- **Fix**: Manual analysis: `tail -100 work_dirs/.../train.log | grep loss`

---

## Timeline

| Task | Expected | Actual |
|------|----------|--------|
| Read documentation | 30 min | — |
| Run validation training | 2-3 hours | — |
| Analyze results | 15 min | — |
| **Total** | **3-4 hours** | — |

---

## FAQ

**Q: Do I need to do anything before running validation?**
A: No, both fixes are already in place. Just run the validation config.

**Q: What if I want to test the fixes separately?**
A: You can disable Fix 2 by setting `spike_downweight_enabled=False` in the config, but testing both together is recommended.

**Q: Can I use a different config for validation?**
A: Yes, but use the provided `hymotion_m2m_v2_uncond_local_046b_validation.py` for consistency with this guide.

**Q: How long before we move to production training?**
A: After validation passes, 1-2 hours max. No additional setup needed.

**Q: What if validation fails?**
A: See detailed debugging guide in `LOSS_SPIKE_FIXES_VALIDATION_PLAN.md`.

---

## Support

If you encounter issues:

1. **First**: Check `LOSS_SPIKE_FIXES_VALIDATION_PLAN.md` debugging section
2. **Second**: Review actual training logs: `work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/*/train.log`
3. **Third**: Compare your config with `_base_hymotion_m2m_v2_046b.py`

---

## Success Indicators

Once validation completes successfully, you'll see:

✅ `docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md` generated  
✅ All 5 success criteria marked PASS  
✅ Loss curves showing smooth convergence  
✅ No NaN/Inf in output  
✅ Ready to submit production training  

---

**Document Version**: 1.0  
**Created**: May 13, 2026  
**Last Updated**: May 13, 2026, 17:50 UTC

**Next Action**: Run validation training → Analyze → Decide → Proceed to production (if pass)
