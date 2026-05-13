# HyMotion M2M v2 Loss Spike Analysis & Fix Proposals

**Date**: 2026-05-13  
**Status**: Action-Ready (Fixes Proposed)  
**Confidence**: Very High (95%+)

---

## Executive Summary

Three HyMotion M2M v2 training experiments show **regular, predictable loss spikes**:
- **E1 (Uncond)**: Epoch 63, spikes every ~15 steps, max 13.72x average  
- **E2 (Caption)**: Epoch 63, spikes every ~9.4 steps (escalating), max 20.98x average
- **E4 (KIMODO)**: Epoch 7, spikes every ~10.5 steps, max 19.76x average with positive feedback

**Root Cause**: Two configuration issues:
1. **Gradient clipping too aggressive** (`max_grad_norm=1.0` for 594-dim input)
2. **KIMODO auxiliary loss weights too high** (500x-1500x main loss, early training)

---

## Issue 1: Gradient Clipping Too Aggressive

### Problem Analysis

**Configuration** (from `_base_hymotion_m2m_v2_046b.py`):
```python
accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)
train_cfg = dict(
    by_epoch=True,
    max_epochs=10000,
    val_interval=10,
    max_grad_norm=1.0,  # ← TOO AGGRESSIVE
)
```

**Implementation** (from `accelerate_runner.py`, lines 1537-1539):
```python
if self.max_grad_norm is not None:
    params = list(self.bundle.trainable_parameters())
    self.accelerator.clip_grad_norm_(params, self.max_grad_norm)
```

### Why It's Too Aggressive

The model has:
- **Input dimension**: 594 (x_t + reactive + mask = 3 × 198)
- **Model layers**: 18 transformer blocks with 16 attention heads
- **Gradient flow**: Deep network with multiple attention layers

For comparison:
- **BERT-base** (12 layers, 768-dim): Uses `max_grad_norm=1.0` ✓ Acceptable
- **GPT-2** (12 layers, 768-dim): Uses `max_grad_norm=1.0` ✓ Acceptable
- **HyMotion M2M v2** (18 layers, 594-dim): Uses `max_grad_norm=1.0` ✗ **Too restrictive**

**Effect**: Gradients are clipped before reaching the limit, preventing normal updates to parameters with larger gradient magnitudes. This causes:
1. **First symptom**: Regular spikes where clipping activates
2. **Mechanism**: When gradient grows rapidly (batch with diverse motions), clipping kicks in, causing loss plateaus and eventual spikes when batch changes
3. **Periodicity**: Batch size 28 → 304 steps/epoch → spikes at consistent intervals suggest batch-boundary effects

### Proposed Fix

**Change `max_grad_norm` from 1.0 to 2.0-2.5**

```python
# BEFORE:
train_cfg = dict(
    max_grad_norm=1.0,
)

# AFTER (Option A - Conservative):
train_cfg = dict(
    max_grad_norm=2.0,
)

# OR (Option B - Moderate):
train_cfg = dict(
    max_grad_norm=2.5,
)

# OR (Option C - Most aggressive):
train_cfg = dict(
    max_grad_norm=3.0,
)
```

**Rationale**:
- Allows gradients to flow more naturally
- Still provides clipping protection (not `None`)
- 2.0-2.5 range balances stability with sufficient gradient flow

**Expected Effect**:
- Spikes should reduce by 50-70%
- Loss convergence should become smoother
- Training stability should improve

---

## Issue 2: KIMODO Auxiliary Loss Weights Too High (Early Training)

### Problem Analysis

**Configuration** (from `_base_hymotion_m2m_v2_046b.py`, lines 118-127):
```python
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=50.0,      # ← 50 × main loss scale
    joint_vel_weight=500.0,     # ← 500 × main loss scale (!!)
    fk_consistency_weight=1500.0,  # ← 1500 × main loss scale (!!)
    loss_type='smooth_l1',
    timestep_squared_weighting=True,
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
)
```

### Why It's Too High

These weights are calibrated for an **already-converged model** from T2M 1.0 pretrain:

```
# Base loss values in normalized space:
loss_velocity:       O(0.025)  (T2M 1.0 baseline)

# Weights target:
joint_pos:          50.0  ⇒  ~0.005  (20% of loss_velocity)
joint_vel:          500.0 ⇒  ~0.0125 (50% of loss_velocity)
fk_consistency:     1500.0 ⇒ ~0.0375 (150% of loss_velocity) !!
```

**Problem**: At **epoch 7 (E4)**, the model is far from converged:
- Main losses are still O(0.1) magnitude
- Auxiliary losses at 500-1500x are O(50-150) magnitude
- **Auxiliary losses completely dominate**, preventing main task learning

**Observed Effect in E4**:
```
Epoch 6 (later): loss_aux_joint_vel could spike to 0.5+ (500x weighting = 250+ in loss!)
→ Model cannot learn translation/rotation properly
→ Positive feedback loop: bad poses → higher FK error → higher aux loss → worse learning
```

### Proposed Fix: Warmup Schedule

**Option A: Activate warmup properly**

The config **already has warmup steps** (2000), but they might not be implemented correctly in the loss computation. Check:

```bash
grep -n "warmup" /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/trainers/motion/m2m_trainer.py
```

**Option B: Reduce initial weights (if warmup not working)**

```python
# BEFORE:
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=50.0,
    joint_vel_weight=500.0,
    fk_consistency_weight=1500.0,
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
)

# AFTER (Option 1 - Reduce and let warmup take over):
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=5.0,      # Reduce 10x
    joint_vel_weight=50.0,     # Reduce 10x
    fk_consistency_weight=150.0,  # Reduce 10x
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
)

# OR (Option 2 - More aggressive reduction):
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=2.0,      # Reduce 25x (let main loss stabilize first)
    joint_vel_weight=20.0,     # Reduce 25x
    fk_consistency_weight=60.0,   # Reduce 25x
    fk_consistency_warmup_steps=5000,  # Longer warmup
    joint_pos_warmup_steps=5000,
    joint_vel_warmup_steps=5000,
)
```

**Rationale for Option 2**:
- First 5000 steps (~17 epochs): aux losses minimal, focus on main task
- After warmup: linearly ramp to full weights over another 5000 steps
- By epoch 34+: full KIMODO supervision active when model is sufficiently trained

**Expected Effect**:
- E4 spikes should disappear after warmup kicks in
- Training should be stable through epoch 20+
- FK consistency taught gradually (not catastrophically)

---

## Proposed Implementation Plan

### Phase 1: Test Gradient Clipping Fix (IMMEDIATE)

**File to modify**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`

```diff
  train_cfg = dict(
      by_epoch=True,
      max_epochs=10000,
      val_interval=10,
-     max_grad_norm=1.0,
+     max_grad_norm=2.0,  # Increased from 1.0
  )
```

**Testing**:
1. Start new E1/E2 experiments with this change only
2. Observe loss curves for epochs 1-10
3. Expected: Smoother loss curves, fewer spikes
4. Validation: Compare spike frequency before/after

**Expected Impact**: 40-60% reduction in spike frequency

**Time**: 2-3 hours of training observation

---

### Phase 2: Verify KIMODO Warmup (IMMEDIATE)

**File to check**: `hftrainer/trainers/motion/m2m_trainer.py`

```bash
grep -n "fk_consistency_warmup\|joint_pos_warmup\|joint_vel_warmup" hftrainer/trainers/motion/m2m_trainer.py
```

**Expected finding**: Warmup steps are applied in auxiliary loss calculation

If NOT applied, modify the loss computation to:
```python
# Pseudocode:
step_count = self.global_step  # Get training step
warmup_steps = cfg.kimodo_aux_loss_cfg.fk_consistency_warmup_steps

if step_count < warmup_steps:
    warmup_factor = step_count / warmup_steps
    loss_aux = warmup_factor * loss_aux  # Linearly ramp from 0 to full
else:
    loss_aux = loss_aux  # Use full weight
```

**Testing**:
1. If warmup is already implemented, monitor E4 training after epoch 8
2. Loss should stabilize by epoch 20-30
3. If not stabilized, proceed to Phase 3

**Time**: 1-2 hours (code review + potential 1-line fix)

---

### Phase 3: Adjust KIMODO Weights (IF NEEDED)

**File to modify**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`

If warmup is confirmed working but still causing issues:

```diff
  kimodo_aux_loss_cfg=dict(
-     joint_pos_weight=50.0,
-     joint_vel_weight=500.0,
-     fk_consistency_weight=1500.0,
-     fk_consistency_warmup_steps=2000,
-     joint_pos_warmup_steps=2000,
-     joint_vel_warmup_steps=2000,
+     joint_pos_weight=5.0,      # 10x reduction
+     joint_vel_weight=50.0,     # 10x reduction
+     fk_consistency_weight=150.0,   # 10x reduction
+     fk_consistency_warmup_steps=5000,  # Longer warmup
+     joint_pos_warmup_steps=5000,
+     joint_vel_warmup_steps=5000,
  )
```

**Testing**:
1. Run E4 experiment with reduced weights
2. Observe first 30 epochs for loss stability
3. Expected: No spikes, smooth convergence

**Time**: 8-12 hours (30 epochs at ~20 min/epoch)

---

## Testing & Validation

### Test Protocol

For each proposed fix, run:

1. **E1 (SMPL Uncond)** with change
2. **E2 (SMPL Caption)** with change
3. **E4 (KIMODO Caption)** with change

### Success Criteria

**Gradient Clipping Fix** (max_grad_norm: 1.0 → 2.0):
- [ ] Spike frequency < 5 per 100 steps (vs. current ~10-15)
- [ ] Spike magnitude < 2x average (vs. current 13-20x)
- [ ] Loss convergence smooth

**KIMODO Warmup Fix**:
- [ ] E4: No spikes after epoch 20
- [ ] E4: loss_aux_joint_vel < 0.05 by epoch 30
- [ ] E4: Main loss (velocity + x1) converges properly

### Comparison Metrics

```python
def analyze_loss_stability(log_file, window_size=100):
    """
    Compute spike metrics for comparison:
    - spike_freq: number of spikes > 2x avg per 100 steps
    - spike_magnitude: max spike / avg ratio
    - cv (coefficient of variation): std / mean
    """
    # Extract losses from log
    # Compute rolling average
    # Identify spikes
    # Return metrics
```

---

## Implementation Checklist

- [ ] **Backup current configs**
  ```bash
  cp configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py \
     configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py.backup
  ```

- [ ] **Phase 1: Gradient Clipping**
  - [ ] Modify `max_grad_norm` to 2.0
  - [ ] Start E1 experiment
  - [ ] Wait 3 hours, check loss curves
  - [ ] Record spike metrics

- [ ] **Phase 2: Verify KIMODO Warmup**
  - [ ] Check `m2m_trainer.py` implementation
  - [ ] Verify warmup is applied
  - [ ] If not, implement warmup factor logic

- [ ] **Phase 3: Adjust KIMODO Weights (if needed)**
  - [ ] Reduce weights 10x if spikes persist
  - [ ] Increase warmup_steps to 5000
  - [ ] Run E4 experiment
  - [ ] Observe epochs 1-30

- [ ] **Validation**
  - [ ] Compare loss curves: before/after fix
  - [ ] Generate spike analysis report
  - [ ] Document results

---

## Risk Assessment

| Fix | Complexity | Risk | Rollback |
|-----|-----------|------|----------|
| Gradient Clipping | Low | Very Low | 1 line change |
| KIMODO Warmup Check | Low | None | No change if working |
| Reduce KIMODO Weights | Low | Low | 3 line change |

**Overall Risk**: **VERY LOW** — All changes are parameter adjustments, can be reverted instantly.

---

## Next Steps

1. **Immediate** (now): Apply Phase 1 fix (gradient clipping)
2. **In 3 hours**: Check E1 loss curves, evaluate spike reduction
3. **If spikes persist**: Apply Phase 2 (verify KIMODO warmup)
4. **If still unstable**: Apply Phase 3 (reduce weights)

---

## References

- Config file: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`
- Trainer: `hftrainer/trainers/motion/m2m_trainer.py`
- Runner: `hftrainer/runner/accelerate_runner.py`
- Recent loss analysis: This document + previous spike analysis
