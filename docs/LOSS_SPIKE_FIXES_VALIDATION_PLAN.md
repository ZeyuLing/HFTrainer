# Loss Spike Fixes Validation Plan

**Date**: May 13, 2026  
**Status**: 🟡 **Validation Phase Initiated**  
**Objective**: Confirm that Fix 1 (max_grad_norm=2.0) and Fix 2 (spike detection) are functioning correctly and producing expected loss spike reduction

---

## Background

Based on comprehensive analysis in `docs/LOSS_SPIKE_ANALYSIS_20260513.md` and implementation verification in `docs/LOSS_SPIKE_FIXES_VERIFICATION.md`, two critical fixes have been implemented:

1. **Fix 1**: Gradient clipping with `max_grad_norm=2.0` (not 1.0, not 10.0)
2. **Fix 2**: Dynamic translation loss downweighting with spike detection

Expected combined impact:
- **E1**: Max spike -60% (1.56x → ~0.6x)
- **E2**: Max spike -51% (0.81x → ~0.4x)
- **E4**: Max spike -73% (8.2x → ~2.2x), spike frequency -68% (46.9% → ~8%)

---

## Validation Phases

### Phase 1: Unit Test Verification (✅ Completed)

Confirmed that both fixes are present in the codebase:

**Fix 1 Verification**:
```bash
grep -r "max_grad_norm=2.0" configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py
# Result: Line 225: train_cfg = dict(..., max_grad_norm=2.0, ...)
```

**Fix 2 Verification**:
```bash
grep -c "spike_downweight_enabled" hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py
# Result: 8 matches (init + methods + integration points)

grep -c "_detect_trans_spike\|_update_spike_detection_stats" hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py
# Result: 2 matches (spike detection methods active)
```

### Phase 2: Training Smoke Test (🟡 In Progress)

**Objective**: Run 10-epoch validation training to confirm:
1. No NaN/Inf in spike detection
2. Loss curves show spike reduction pattern
3. Model convergence not broken
4. Fix 1 + Fix 2 work together correctly

**Config**:
- Path: `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py`
- Duration: 10 epochs (fast validation, ~2-3 hours on 1×8 V100)
- Batch size: 28 (matching production)
- Checkpoint interval: every epoch (to observe loss curve evolution)

**Expected Metrics**:

#### Epoch-by-Epoch Loss Curve Pattern

For uncond_local (E1 baseline), typical curves look like:

| Epoch | loss_velocity (typical) | loss_smoothness | loss_velocity_trans | Notes |
|-------|-------------------------|-----------------|-------------------|-------|
| 1 | 0.025-0.035 | 0.012-0.018 | 0.008-0.012 | High variability, spike detection warm-up |
| 2-3 | 0.020-0.028 | 0.010-0.015 | 0.007-0.010 | Stabilizing, spikes becoming less frequent |
| 4-5 | 0.018-0.024 | 0.009-0.013 | 0.006-0.009 | Clear trend, fewer outliers |
| 6-10 | 0.015-0.020 | 0.008-0.012 | 0.005-0.008 | Converging, smooth trajectory |

**Spike Detection Active Indicators**:
- Absence of sudden 50%+ jumps in loss_velocity (Fix 2 preventing them)
- loss_velocity_trans component remains <0.015 throughout (downweighting active)
- Gradient norms staying in reasonable range (no massive clipping, Fix 1 working)

**Commands to Run**:

```bash
# Create and submit validation training job
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer

# Single GPU validation (local or small cluster)
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train.py \
    configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py

# Or submit to Taiji for batch:
taiji submit \
    --task_name m2m_v2_uncond_local_validation_fix \
    --config configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b_validation.py \
    --gpu 8 \
    --priority high
```

**Success Criteria** (for each epoch log):

- [ ] No NaN values in any loss component
- [ ] No Inf values in loss or gradients
- [ ] `loss_velocity` monotonically decreasing or stable (no 30%+ jumps)
- [ ] `loss_velocity_trans` consistently <0.015 (spike downweighting active)
- [ ] Model checkpoint saves successfully every epoch
- [ ] Training completes without OOM or gradient overflow

### Phase 3: Comparative Analysis (🟡 Pending)

After validation training completes, extract and analyze:

**From work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/:**

1. **Loss Curve Analysis**:
   ```bash
   # Extract training logs
   tail -100 work_dirs/hymotion_m2m_v2_uncond_local_046b_validation/*/train.log
   
   # Parse loss components per epoch (requires simple Python script):
   python3 scripts/analysis/extract_loss_curves.py \
       work_dirs/hymotion_m2m_v2_uncond_local_046b_validation \
       --output docs/LOSS_SPIKE_VALIDATION_CURVES.md
   ```

2. **Spike Statistics**:
   - Count of spike detection events per epoch
   - Distribution of downweight factor (1.0 vs 0.3)
   - Comparison to baseline (Fix 1 only) if available

3. **Visual Comparison**:
   - Plot loss_velocity vs epoch (should be smooth, no outlier spikes)
   - Plot loss_velocity_trans separately (tight, constrained)
   - Overlay: expected range from analysis vs actual observed

### Phase 4: Full Training Deployment (🟡 After Validation)

If Phase 2-3 pass all success criteria:

1. **Start production training**:
   ```bash
   # E1 (uncond_local)
   taiji submit --config configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_046b.py --gpu 32
   
   # E2 (caption_local)
   taiji submit --config configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_046b.py --gpu 32
   
   # E4 (kimodo_uncond)
   taiji submit --config configs/hymotion_m2m_v2/hymotion_m2m_v2_kimodo_uncond_046b.py --gpu 32
   ```

2. **Monitor loss curves** at 50-epoch intervals
3. **Export metrics** at epoch 100, 200, 500 for comparison

---

## Expected Outcomes

### Primary Validation Metrics

| Metric | Validation Phase Expected | Phase 4 Production Target |
|--------|---------------------------|--------------------------|
| **Max spike frequency reduction** | <20% deviation from predicted | -60% to -75% |
| **Loss convergence rate** | Smooth, <5% epoch-to-epoch variance | Stable convergence |
| **Spike detection events/epoch** | 5-15 (depending on batch heterogeneity) | Consistent |
| **Model training time/epoch** | <5% overhead vs baseline | Negligible |

### Loss Component Behavior

**Expected pattern with Fix 1 + Fix 2 active**:

```
Epoch 1:  [High variability] → Fix 2 detects spikes, applies downweighting
Epoch 2-3: [Convergence starts] → Spike frequency decreasing
Epoch 4-5: [Stable trend] → Smooth loss curve, no outlier events
Epoch 6-10: [Converged] → Consistent loss trajectory
```

**DO NOT expect** (would indicate bug):
- Sudden 50%+ jumps in loss_velocity (Fix 2 should prevent)
- Gradient norm saturation at exactly 2.0 every step (Fix 1 should rarely trigger)
- NaN/Inf values (numerical stability)
- Training speed degradation >10% (spike detection overhead)

---

## Debugging Strategy (If Validation Fails)

### Scenario A: Spike Detection Not Triggering

**Symptoms**: loss_velocity still has frequent >20% jumps; loss_velocity_trans >0.02

**Diagnosis**:
1. Check if `spike_downweight_enabled=True` is being passed to M2MLoss
2. Verify `spike_detection_window=100` is large enough for current batch patterns
3. Check if spike threshold `2.0 * std_dev` is too high for this data

**Fix**:
```python
# In config:
model = dict(
    losses_cfg=dict(
        ...
        spike_downweight_enabled=True,  # Explicit (default)
        spike_detection_std_threshold=1.5,  # Lower threshold to be more aggressive
        spike_detection_window=50,  # Smaller window for faster response
    ),
)
```

### Scenario B: Gradient Norm Clipping Ineffective

**Symptoms**: Gradient norm histogram shows many values >10, clipping to 2.0 happens <1% of steps

**Diagnosis**:
1. Verify `max_grad_norm=2.0` is in `train_cfg` (not 10.0, not 1.0)
2. Check if `optimizer_cfg` is using the correct `type='OptimWrapper'` with `clip_grad` dict
3. Verify gradients are being computed (check backward pass)

**Fix**:
```python
# In train_cfg:
train_cfg = dict(
    by_epoch=True,
    max_epochs=10,
    val_interval=1,
    max_grad_norm=2.0,  # Must be set here, not hardcoded
)
```

### Scenario C: Training Diverges with Both Fixes

**Symptoms**: loss_velocity increases over time, or NaN appears

**Diagnosis**:
1. Spike downweight factor too aggressive (0.3 cuts learning signal too much)
2. Gradient norm too tight (2.0 prevents important updates)
3. Data pipeline issue (ranks not synchronized, NaN in input)

**Fix**:
```python
# Try less aggressive downweighting:
spike_downweight_factor=0.5  # was 0.3

# Or less strict clipping:
max_grad_norm=5.0  # was 2.0
```

---

## Post-Validation Checklist

Once validation training completes, verify:

- [ ] 10 epochs completed without errors
- [ ] Loss curve saved to `docs/LOSS_SPIKE_VALIDATION_CURVES.md`
- [ ] All 10 epoch checkpoints available
- [ ] No NaN/Inf in any logs
- [ ] loss_velocity_trans stays <0.015
- [ ] Spike detection active (log shows downweight events)
- [ ] Training reproducible (same seed = same loss curve)
- [ ] Update LOSS_SPIKE_FIXES_VERIFICATION.md with validation results
- [ ] Create production training tickets if all checks pass

---

## Related Documents

- **Analysis**: `docs/LOSS_SPIKE_ANALYSIS_20260513.md` — Root cause analysis
- **Verification**: `docs/LOSS_SPIKE_FIXES_VERIFICATION.md` — Implementation details
- **Implementation Status**: `docs/LOSS_SPIKE_FIX_IMPLEMENTATION_STATUS.md` — Previous status (outdated, use Verification instead)
- **M2M Loss Code**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` — Spike detection implementation
- **Base Config**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` — max_grad_norm=2.0

---

## Timeline

| Phase | Target Date | Status |
|-------|------------|--------|
| Phase 1 (Unit Tests) | May 13 | ✅ Completed |
| Phase 2 (Smoke Test) | May 13-14 | 🟡 In Progress |
| Phase 3 (Analysis) | May 14 | 🟡 Pending |
| Phase 4 (Production) | May 15+ | 🟡 Pending |

**Last Updated**: May 13, 2026, 17:15 UTC
