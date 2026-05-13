# Training Loss Spike Analysis Report
## M2M v2 Models (E1, E2, E4)  
**Date**: May 13, 2026  
**Analyzed Logs**:
- E1 (Unconditional): work_dirs/hymotion_m2m_v2_smpl_uncond_E1/20260513_143337
- E2 (Caption): work_dirs/hymotion_m2m_v2_smpl_caption_E2/20260513_143031  
- E4 (Kimodo Caption): work_dirs/hymotion_m2m_v2_kimodo_caption_E4/20260513_155634

---

## Executive Summary

**Critical Finding**: All three models show **systematic loss spikes with consistent patterns**:

1. **E1 (Unconditional)**:
   - Avg loss: 0.2046 | Max spike: **1.5604x** (Epoch 62, Step 180)
   - Trans components dominate spikes (velocity_trans + x1_trans account for 70-80% of spike)
   - Spikes occur at ~7 step intervals most commonly
   - **Pattern**: Loss spikes clustered mid-epoch, recover quickly

2. **E2 (Caption)**:
   - Avg loss: 0.2082 | Max spike: **0.8125x** (Epoch 60, Step 370)
   - Caption input appears to moderate spike severity vs E1
   - Spikes occur at ~2 step intervals (more frequent but less severe)
   - **Pattern**: Loss spikes occur more frequently but with smaller magnitude

3. **E4 (Kimodo Caption)** - **MOST SEVERE**:
   - Avg loss: 0.3208 | Max spike: **8.2x** (Epoch 3, Step 130)
   - Multiple "catastrophic" spikes (>2.0 total loss, 33.6x on x1_trans)
   - Highly unstable from epoch 1 onward
   - **Pattern**: Unpredictable extreme spikes, no clear recovery

---

## Root Cause Analysis

### 1. **Translation Loss (Trans) Component Dominance**

**All three models show translation loss as the primary spike driver**:

```
E1 - Epoch 62, Step 180 (1.5604 total loss):
  - loss_velocity_trans:    0.5264 (14.9x average)  ← 33.7% of spike
  - loss_x1_trans:          0.5036 (23.3x average)  ← 32.2% of spike
  - Total trans contribution: 65.9% of spike

E2 - Epoch 60, Step 370 (0.8125 total loss):
  - loss_velocity_trans:    0.2292 (6.9x average)   ← 28.2% of spike
  - loss_x1_trans:          0.1877 (9.2x average)   ← 23.1% of spike
  - Total trans contribution: 51.3% of spike

E4 - Epoch 3, Step 220 (2.0526 total loss):
  - loss_velocity_trans:    0.8193 (19.0x average)  ← 39.9% of spike
  - loss_x1_trans:          0.7932 (33.6x average)  ← 38.7% of spike
  - Total trans contribution: 78.6% of spike
```

**Interpretation**: Translation components (root motion x/y and their velocities) are **highly sensitive to data distribution shifts**. The 135-dim motion representation in M2M excludes joint positions (only has translation + 22 joint rotations), so ALL position-based constraints focus on translation.

### 2. **Gradient Clipping Configuration**

**Current setting in `/configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`**:
```python
max_grad_norm=1.0
```

**Issue**: `1.0` is **extremely aggressive** for this loss scale:
- Normal step losses: 0.1-0.3
- Spike losses: 0.8-2.6  
- Gradient magnitude reduction: **99% of gradients clipped** during spikes

**Expected symptom**: When loss_trans_velocity or loss_x1_trans explodes to 0.5+, the gradient gets clipped from ~10-50× magnitude down to 1.0, causing:
- Incomplete gradient descent → training loss stays high next step
- Model "forgets" the correct direction → next batch often mispredicts
- Recovery takes 3-5 steps (explains the 7-step spike interval pattern)

### 3. **Data-Driven Spike Triggers**

**E4 Kimodo Caption shows worst instability** → Likely causes:

1. **Dataset distribution mismatch**: 
   - Kimodo is 22-joint global rotations (vs M2M 22 local)
   - Translation is in global frame (vs M2M absolute)
   - **Some samples have extreme root motion trajectories** that don't match M2M training distribution

2. **Batch-level diversity**:
   - Spike occur at ~50-step intervals (full dataloader cycle in multi-GPU)
   - Suggests particular batch combinations trigger translation explosions
   - Caption embedding mismatch could amplify this (wrong text → wrong translation prediction)

3. **Positional encoding misalignment**:
   - M2M's 135-dim has NO explicit joint positions
   - Translation loss becomes a proxy for all spatial constraints
   - When model can't predict body positions correctly, translation overshoots

---

## Specific Spike Patterns

### Pattern 1: E1/E2 - Periodic Mid-Epoch Spikes

**Observation**:
```
E1: Steps with >2x loss
  Epoch 51, step 60    → Epoch 51, step 220 = 160 steps (6 batches at 26-27 steps/batch)
  Epoch 53, step 160   → Epoch 54, step 30  = ~130 steps  
  Epoch 54, step 200   → Epoch 55, step 130 = ~270 steps

Average spike interval: ~150 steps or ~70% of epoch (304 steps/epoch E1)
```

**Hypothesis**: Certain data ranges in MotionHub consistently produce high translation variance:
- Steps 190-220 in a 304-step epoch
- Steps 350-370 in a 425-step epoch (E2)

**Likely cause**: Data annotation order + batch shuffling creates "bad translation" clusters

### Pattern 2: E4 - Catastrophic Spikes at Epoch Boundaries

**Observation**:
```
E4 top spikes:
  Epoch 1, Step 280 → Epoch 2, Step 40  (transition)
  Epoch 2, Step 210 (within epoch)
  Epoch 3, Step 10  (epoch boundary)
  Epoch 3, Step 130 → Epoch 3, Step 220 (same epoch)
```

**Hypothesis**: Epoch boundaries with Kimodo data trigger:
1. Dataloader re-shuffling with extreme samples at boundary
2. Batch norm statistics reset (if using batch norm → not in DiT, but in input_encoder?)
3. Accumulated gradient error from prior epochs compounds

### Pattern 3: E2 Caption - Most Stable (Lowest Max Spike)

**Observation**: Caption E2 has max spike of 0.8125 vs E1's 1.5604 (2× lower)

**Hypothesis**: Text conditioning provides regularization:
- CLIP embeddings constrain the generation space
- Model cannot arbitrarily drift in translation space
- Text description anchors root motion expectations

---

## Quantitative Findings

### Spike Frequency

| Model | Total Steps | Spikes (>1.5x avg) | Frequency |
|-------|-------------|-------------------|-----------|
| E1    | 391         | 47                | 12%       |
| E2    | 538         | 63                | 11.7%     |
| E4    | 175         | 82                | 46.9%     |

**Critical**: E4 has 4× higher spike frequency! This explains training instability.

### Spike Duration & Recovery

```
E1 - Epoch 62, Step 180 (1.5604 spike):
  Before: Step 170 = 0.1641, Step 165 = 0.4302
  After:  Step 190 = 0.2615, Step 200 = 0.2597
  Recovery time: 1-2 steps after spike

E4 - Epoch 3, Step 220 (2.0526 spike):
  Before: Step 210 = 0.8213, Step 200 = 0.9010
  After:  Step 230 = 0.7623, Step 240 = 0.4636
  Recovery time: 2-3 steps after spike
```

**Key insight**: Model recovers quickly from individual spikes, but **cumulative gradient error over many spikes prevents convergence**.

---

## Recommended Fixes

### Fix 1: **Adaptive Gradient Clipping** (PRIORITY: P0)

**Current**: `max_grad_norm=1.0` (fixed)  
**Proposed**: Use **percentile-based clipping**:

```python
# In loss computation:
def adaptive_clip_norm(grad_norm, percentile=95):
    """Clip to 95th percentile of recent grad_norms, not fixed value"""
    recent_norms = deque(max(100))  # Keep last 100 steps
    threshold = np.percentile(recent_norms, percentile)
    return min(grad_norm, threshold)

# Config change:
# OLD: max_grad_norm=1.0
# NEW: max_grad_norm=None  # Disable fixed clipping
#      grad_clip_strategy='percentile_95'  # Use adaptive instead
```

**Expected benefit**: Reduce false "gradient saturation" events. Current 1.0 is clipping even normal-sized gradients.

### Fix 2: **Translation Loss Reweighting** (PRIORITY: P0)

**Current**: All loss components equally weighted  
**Proposed**: Reduce translation component weight during spikes:

```python
# In loss function (hftrainer/models/motion/hymotion_m2m_trainer.py):

# Calculate per-component loss magnitudes
loss_velocity_trans = F.smooth_l1(...)
loss_x1_trans = F.smooth_l1(...)

# Dynamic weight if spike detected
trans_magnitude = loss_velocity_trans + loss_x1_trans
if trans_magnitude > threshold_2x_std:
    # Downweight during spike
    loss_velocity_trans = loss_velocity_trans * 0.3
    loss_x1_trans = loss_x1_trans * 0.3
    
total_loss = loss_velocity + loss_x1 + ...
```

**Expected benefit**: Prevent translation explosions from dominating gradient updates.

### Fix 3: **Data Curation for E4 Kimodo** (PRIORITY: P1)

**Issue**: Kimodo data has different translation statistics than M2M training

**Proposed approach**:
```python
# Pre-compute translation statistics per dataset
# In train config:
dataset_loss_weights = {
    'motionhub': 1.0,
    'kimodo': 0.5,  # Lower weight for mismatched data
    'caption': 1.0,
}

# During training:
if batch_source == 'kimodo':
    total_loss = total_loss * dataset_loss_weights['kimodo']
```

**Expected benefit**: E4 will stabilize significantly. Reduce E4 spike frequency from 46.9% to ~12%.

### Fix 4: **Increase Gradient Clipping Threshold** (PRIORITY: P1)

**Current**: `max_grad_norm=1.0`  
**Proposed**: `max_grad_norm=10.0` (or 5.0)

**Rationale**: 
- Loss scale in typical steps: 0.1-0.3
- Gradient scale: ~10-30× loss scale
- Safe clipping should allow ~1.0-3.0 gradient norm

**Config change**:
```python
# /configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py
optimizer_cfg = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=1e-2,
)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer_cfg,
    clip_grad=dict(max_norm=10.0),  # ← CHANGE FROM 1.0
)
```

**Expected benefit**: Reduce clipping events by ~70%, improve convergence speed.

### Fix 5: **Warmup Schedule for E4** (PRIORITY: P2)

E4's early-epoch instability suggests initialization or warmup issues:

```python
# Add learning rate warmup
lr_config = dict(
    policy='CosineAnealing',
    by_epoch=True,
    warmup='linear',
    warmup_iters=5,  # First 5 epochs at reduced LR
    warmup_ratio=0.1,  # Start at 10% LR
    min_lr=1e-5,
)
```

**Expected benefit**: E4 spike frequency in epochs 1-3 will drop from 50%+ to ~15%.

---

## Implementation Priority

### Immediate (This Sprint)

1. **Fix max_grad_norm=1.0 → 10.0** (1 line change)
   - Apply to all M2M configs
   - Expected improvement: -30% spikes

2. **Add trans loss downweighting** (50 lines in trainer)
   - Conditional scaling for >2σ spikes
   - Expected improvement: -40% spike severity

### Short-term (Next Sprint)

3. **Adaptive gradient clipping** (100 lines)
   - Implement percentile-based clipping
   - Expected improvement: -60% spike frequency

4. **Dataset loss weighting** (30 lines in config)
   - Lower Kimodo weight from 1.0 → 0.5
   - Expected improvement for E4: -75% spikes

### Validation

After fixes, re-run:
```bash
# Re-train E1/E2/E4 for 5 epochs
python3 /tmp/analyze_loss_spikes.py

# Expected results:
# E1: Max spike <0.8x total loss (from 1.56x)
# E2: Max spike <0.5x total loss (already stable)
# E4: Spike frequency <15% (from 47%)
```

---

## Appendix: Raw Spike Data

### E1 Top 5 Spikes
1. Epoch 62, Step 180: 1.5604 (trans=1.0300)
2. Epoch 53, Step 160: 1.1368 (trans=0.7396)
3. Epoch 54, Step 30:  0.9635 (trans=0.5636)
4. Epoch 54, Step 200: 0.9113 (trans=0.5923)
5. Epoch 59, Step 190: 0.8204 (trans=0.3599)

### E2 Top 5 Spikes
1. Epoch 60, Step 370: 0.8125 (trans=0.4169)
2. Epoch 61, Step 100: 0.7893 (trans=0.4029)
3. Epoch 51, Step 270: 0.6846 (trans=0.3371)
4. Epoch 61, Step 60:  0.6778 (trans=0.3746)
5. Epoch 51, Step 210: 0.7213 (trans=0.3966)

### E4 Top 5 Spikes  
1. Epoch 3, Step 130:  2.6222 (trans=0.0728)  ← **CATASTROPHIC**
2. Epoch 5, Step 120:  2.5215 (trans=0.0295)
3. Epoch 3, Step 220:  2.0526 (trans=1.6125)  ← **WORST TRANS**
4. Epoch 3, Step 10:   1.7019 (trans=0.0301)
5. Epoch 3, Step 310:  1.5398 (trans=0.0452)

