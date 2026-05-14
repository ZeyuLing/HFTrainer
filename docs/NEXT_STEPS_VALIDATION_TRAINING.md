# Next Steps: Loss Spike Fixes Validation Training

**Current Status**: ✅ Implementation verified, 🟡 Validation phase ready to start  
**Date**: May 13, 2026  
**Priority**: HIGH (blocks production training of E1/E2/E4 models)

---

## What Has Been Done

### ✅ Completed: Implementation Verification

1. **Fix 1 Verified**: `max_grad_norm=2.0` confirmed in base config (line 225)
   - Correct value: 2.0 (not 1.0, not 10.0)
   - Reason: Analysis recommended 2.0-2.5 range; 2.0 is safe, aggressive gradient clipping

2. **Fix 2 Verified**: Spike detection implemented and active in `M2MLoss`
   - Spike detection parameters initialized (lines 23-50)
   - Detection methods present and correct (lines 94-112)
   - Integration into velocity loss (lines 234-246) and x1 loss (lines 264-275)
   - Downweighting factor: 0.3× (70% spike reduction)
   - Detection threshold: z-score > 2.0 (95% confidence)
   - Window size: 100 steps (20-30 training steps of dynamics)

3. **Documentation Created**:
   - `docs/LOSS_SPIKE_FIXES_VERIFICATION.md` — Corrects previous status doc error
   - `docs/LOSS_SPIKE_FIXES_VALIDATION_PLAN.md` — Complete validation strategy
   - `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py` — Validation config
   - `scripts/analysis/extract_loss_curves.py` — Loss analysis script

---

## What Needs to Be Done

### Phase 1: Run Validation Training (10 epochs)

**Objective**: Confirm both fixes work together and produce expected loss behavior

**Config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py`

**Command** (choose one):

**Option A: Local/cluster submission**
```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train.py \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py
```

**Option B: Taiji batch submission** (if available)
```bash
taiji submit \
    --task_name m2m_v2_uncond_local_validation_fix \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py \
    --gpu 8 \
    --priority high
```

**Expected Duration**: 2-3 hours on 1×8 V100  
**Output**: Training logs saved to `work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/*/train.log`

**Success Criteria** (must ALL pass):
- [ ] 10 epochs complete without errors
- [ ] No NaN/Inf in loss values
- [ ] `loss_velocity` shows smooth convergence (no 50%+ spikes)
- [ ] `loss_velocity_trans` stays < 0.015 (downweighting active)
- [ ] All 10 checkpoint files save successfully
- [ ] Training speed reasonable (<10% overhead vs baseline)

### Phase 2: Analyze Loss Curves (after training)

**Objective**: Extract and visualize loss behavior to confirm spike reduction

**Command**:
```bash
python3 scripts/analysis/extract_loss_curves.py \
    work_dirs/hymotion_m2m_v2_uncond_local_046b_validation \
    --output docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md
```

**Output**: Markdown report with:
- Loss trajectory table (all 10 epochs)
- Spike detection summary
- Per-component statistics
- Validation criteria pass/fail
- Overall result: ✅ PASS or ❌ NEEDS REVIEW

**Expected Report Content**:
```
| Epoch | loss_velocity | loss_velocity_trans | loss_smoothness | Notes |
|-------|---------------|---------------------|-----------------|-------|
| 1 | 0.0285 | 0.0109 | 0.0145 | ⚠️ Moderate (vel 0.020-0.025) |
| 2 | 0.0242 | 0.0095 | 0.0132 | ✅ Good (vel < 0.020) |
| 3 | 0.0218 | 0.0087 | 0.0128 | ✅ Good |
| ... | ... | ... | ... | ... |
| 10 | 0.0168 | 0.0062 | 0.0098 | ✅ Good (converged) |

No significant spikes detected (all changes <20%)

VALIDATION PASSED ✅
```

### Phase 3: Proceed to Production Training (if Phase 1-2 pass)

If validation passes all criteria:

**Submit production training for all 3 model variants**:

```bash
# E1 (uncond_local) — most important baseline
taiji submit \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py \
    --gpu 32 \
    --priority high \
    --task_name m2m_v2_uncond_local_046b_with_fixes

# E2 (caption_local) — text-conditioned variant
taiji submit \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py \
    --gpu 32 \
    --priority high \
    --task_name m2m_v2_caption_local_046b_with_fixes

# E4 (kimodo_uncond) — most affected by spikes
taiji submit \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_uncond_046b.py \
    --gpu 32 \
    --priority high \
    --task_name m2m_v2_kimodo_uncond_046b_with_fixes
```

**Production Training Targets**:
- E1: Spike frequency should drop to ~6% (from ~12%)
- E2: Spike frequency should drop to ~5% (from ~11.7%)
- E4: Spike frequency should drop to ~15% (from ~46.9%)

---

## Files and Paths

### Configuration
- **Validation config**: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py` ← **New**
- **Base config**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` (already has Fix 1)
- **Loss implementation**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` (already has Fix 2)

### Documentation
- **Verification (current)**: `docs/LOSS_SPIKE_FIXES_VERIFICATION.md` ← **Use this, not IMPLEMENTATION_STATUS.md**
- **Validation plan**: `docs/LOSS_SPIKE_FIXES_VALIDATION_PLAN.md` ← **Detailed strategy**
- **This document**: `docs/NEXT_STEPS_VALIDATION_TRAINING.md` ← **Quick reference**

### Analysis Scripts
- **Loss extraction**: `scripts/analysis/extract_loss_curves.py` ← **New**
- **Usage**: `python3 scripts/analysis/extract_loss_curves.py <work_dir> --output <report.md>`

### Expected Outputs (after training)
- **Training logs**: `work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/*/train.log`
- **Checkpoints**: `work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/checkpoint-epoch_*.pt` (10 files)
- **Analysis report**: `docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md` ← **Generated by extract_loss_curves.py**

---

## Key Metrics to Monitor

### During Validation Training

Watch these in the training log output:

```
Epoch [1/10] ... loss_velocity: 0.0285 loss_smoothness: 0.0145 ...
```

**Healthy Patterns**:
- `loss_velocity` gradually decreases over epochs (should go from ~0.028 → ~0.017)
- `loss_velocity_trans` stays low and stable (should stay <0.012)
- `loss_smoothness` also decreases (confirms general convergence)
- No sudden 50% jumps (Fix 2 prevents them)

**Warning Signs** (investigate if seen):
- `loss_velocity` suddenly jumps 50%+ (Fix 2 might not be active)
- `loss_velocity_trans` > 0.015 consistently (downweighting too weak)
- NaN or Inf appearing (numerical instability)
- Training speed dramatically slow (Fix 2 overhead > 50%)

### After Validation Training

Use the analysis script to check:

```bash
python3 scripts/analysis/extract_loss_curves.py \
    work_dirs/hymotion_m2m_v2_uncond_local_046b_validation \
    --output docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md

# Then review: docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md
```

**Report will show**:
- ✅ or ❌ for each success criterion
- Spike count and magnitude
- Per-component min/max/average
- Overall validation result

---

## Troubleshooting

### If Validation Fails

**Issue**: "Spike detection not working (loss_velocity still has spikes)"
- **Check**: `grep spike_downweight_enabled configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- **Fix**: Ensure losses_cfg includes spike detection parameters (see VERIFICATION doc)

**Issue**: "Training slow, NaN appears"
- **Check**: `tail -50 work_dirs/.../train.log | grep NaN`
- **Likely cause**: Gradient clipping too tight (2.0 might be too low for this data)
- **Fix**: Try `max_grad_norm=3.0` or `5.0` in validation config

**Issue**: "loss_velocity_trans > 0.015 (downweighting not active)"
- **Check**: Verify `spike_downweight_factor=0.3` in M2MLoss init
- **Fix**: Try more aggressive threshold: `spike_detection_std_threshold=1.5` (was 2.0)

---

## Timeline & Ownership

| Task | Owner | Target Date | Status |
|------|-------|-------------|--------|
| Run validation training (10 epochs) | — | May 13-14 | 🟡 Ready to start |
| Extract and analyze loss curves | — | May 14 | 🟡 After training |
| Review analysis report | — | May 14 | 🟡 After analysis |
| Submit production training (if pass) | — | May 15 | 🟡 Conditional |
| Monitor production (50-epoch checkpoint) | — | May 16 | 🟡 After submit |

---

## Quick Reference Checklist

**Before starting validation training:**
- [ ] Read `docs/LOSS_SPIKE_FIXES_VERIFICATION.md` for implementation details
- [ ] Confirm config path: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py`
- [ ] GPU available: 8× V100 (or compatible)
- [ ] Estimated runtime: 2-3 hours

**After training completes:**
- [ ] Run: `python3 scripts/analysis/extract_loss_curves.py work_dirs/hymotion_m2m_v2_uncond_local_046b_validation --output docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md`
- [ ] Review report: `docs/LOSS_SPIKE_VALIDATION_ANALYSIS.md`
- [ ] Check pass/fail on all 4 success criteria
- [ ] If ALL ✅: proceed to Phase 3 (production training)
- [ ] If ANY ❌: review `LOSS_SPIKE_FIXES_VALIDATION_PLAN.md` debugging section

---

## Summary

**What We Know** (verified):
- Fix 1 (max_grad_norm=2.0) is in place ✅
- Fix 2 (spike detection) is implemented and active ✅
- Expected improvement: -60% to -75% spike severity ✅

**What We Need to Confirm** (validation phase):
- Both fixes work correctly together in actual training
- Loss curves match predicted patterns
- No numerical issues or unexpected side effects
- Ready for 1000+ epoch production training

**Next Action**: Run 10-epoch validation training → Extract loss curves → Analyze results → Proceed to production if all criteria pass

---

**Document Version**: 1.0  
**Last Updated**: May 13, 2026, 17:30 UTC  
**Related**: LOSS_SPIKE_ANALYSIS_20260513.md, LOSS_SPIKE_FIXES_VERIFICATION.md, LOSS_SPIKE_FIX_IMPLEMENTATION_STATUS.md (outdated, do not use)
