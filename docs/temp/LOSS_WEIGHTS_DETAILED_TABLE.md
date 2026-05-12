# HyMotion M2M v2 Loss Configuration - Detailed Weights Table

## Loss Configuration Summary Table

| Loss Term | Class | Weight | Enabled | Timestep Weighting | Base Value | E[t²] | Effective | % of velocity |
|-----------|-------|--------|---------|--------------------|-----------:|------:|----------:|---------------:|
| **velocity** | M2MLoss | 1.0 | ✓ | No (t=1 fixed) | ~0.025 | 1.0 | ~0.025 | **100%** |
| **motion_smoothness** | M2MLoss | 0.5 | ✓ | No | varies | 1.0 | varies | **~1-2%** |
| **aux_joint_pos** | KimodoStyleAuxLoss | 50.0 | ✓ | **Yes (t²)** | ~1e-4 m | 1/3 | ~5.0e-3 | **~14%** |
| **aux_joint_vel** | KimodoStyleAuxLoss | 500.0 | ✓ | **Yes (t²)** | ~1e-6 m/f | 1/3 | ~1.0e-3 | **~4%** |
| **aux_fk_consistency** | KimodoStyleAuxLoss | 1500.0 | ✓ | **Yes (t²)** | ~1.4e-6 m | 1/3 | ~2.1e-3 | **~7%** |
| keypoints3d | M2MLoss | 0.0 | ✗ | No | - | - | 0 | 0% |
| translation | M2MLoss | 0.0 | ✗ | No | - | - | 0 | 0% |
| x1 | M2MLoss | 0.0 | ✗ | No | - | - | 0 | 0% |
| fk_consistency (M2MLoss) | M2MLoss | 0.0 | ✗ | No | - | - | 0 | 0% |

---

## Loss Computation Details

### 1. Velocity Loss (M2MLoss)
```python
# Code: hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py (line 176-195)
weight: 1.0
formula: loss_velocity = velocity_weight * smooth_l1(pred_vel, gt_vel)
where:
  - pred_vel = pred - x0  (flow-matching prediction)
  - gt_vel = x1 - x0      (ground truth velocity)
  - applied per-dimension with trans_dim_weight=5.0 for dims [0:3]
timestep_weighting: NONE (constant weight across all t)
mask: data_mask_temporal (padding aware) + optional generation_mask (MAN)
```

### 2. Motion Smoothness Loss (M2MLoss)
```python
# Code: hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py (line 249-261)
weight: 0.5
formula: loss_smooth = smooth_l1(pred_x1[1:] - pred_x1[:-1], gt_x1[1:] - gt_x1[:-1])
where:
  - penalizes frame-to-frame velocity changes
  - computed on denoised x1 space (not flow velocity)
timestep_weighting: NONE
mask: both frame t and t+1 must be valid (smooth_mask = mask[1:] * mask[:-1])
```

### 3. Auxiliary Joint Position Loss (KimodoStyleAuxLoss)
```python
# Code: hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py (line 288-296)
weight: 50.0
formula: loss = 50.0 * warmup * smooth_l1(pred_world, gt_world) * t²
where:
  - pred_world, gt_world: (B, L, 22, 3) global joint positions from FK
  - reduced to (B, L) by averaging over joints and xyz
timestep_weighting: YES - per_frame *= t_sq.unsqueeze(-1)
warmup: linear from 0 to 50.0 over 2000 steps
mask: data_mask_temporal (padding aware)
purpose: prevents foot skating and pelvis cheating
```

### 4. Auxiliary Joint Velocity Loss (KimodoStyleAuxLoss)
```python
# Code: hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py (line 301-315)
weight: 500.0
formula: loss = 500.0 * warmup * smooth_l1(pred_vel, gt_vel) * t²
where:
  - pred_vel = pred_world[:, 1:] - pred_world[:, :-1]  (temporal diff)
  - gt_vel = gt_world[:, 1:] - gt_world[:, :-1]
  - reduced to (B, L-1) by averaging over joints and xyz
timestep_weighting: YES - per_frame *= t_sq.unsqueeze(-1)
warmup: linear from 0 to 500.0 over 2000 steps
mask: vel_mask = data_mask_temporal[:, 1:] * data_mask_temporal[:, :-1]
purpose: strong discriminator for slipping (velocity error at every joint)
```

### 5. Auxiliary FK Consistency Loss (KimodoStyleAuxLoss)
```python
# Code: hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py (line 320-331)
weight: 1500.0
formula: loss = 1500.0 * warmup * smooth_l1(pred_pos_chan, fk_pos) * t²
where:
  - pred_pos_chan = pred_denorm[..., 135:]  (predicted position channels)
  - fk_pos = FK(pred_rotation, pred_translation) converted to Scheme D
  - reduced to (B, L) by averaging over 63 position dims
timestep_weighting: YES - per_frame *= t_sq.unsqueeze(-1)
warmup: linear from 0 to 1500.0 over 2000 steps
mask: data_mask_temporal (padding aware)
purpose: teaches explicit FK equivalence map; enables position-only inference
```

---

## t² Timestep Weighting Details

### Implementation
```python
# hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py (line 280-283)
if self.timestep_squared_weighting and timesteps is not None:
    t_sq = (timesteps.to(pred_world.device).to(pred_world.dtype) ** 2)  # (B,)
else:
    t_sq = None
```

### Timestep Sampling
```python
# hftrainer/trainers/motion/hymotion_m2m_trainer.py (line 229)
timesteps = torch.rand(B, dtype=x1.dtype, device=device)  # U(0, 1)
```

### Mathematical Analysis
- **Timestep range**: t ∈ [0, 1] (uniform sampling)
- **t=0**: x_t ≈ x0 (pure noise) → t²≈0 → loss ≈ 0
- **t=1**: x_t ≈ x1 (clean data) → t²=1 → loss = full weight
- **Expected value**: E[t²] = ∫₀¹ t² dt = [t³/3]₀¹ = 1/3 ≈ 0.333
- **Effect**: Average impact is ~1/3 of nominal weight

### Rationale (from config comment)
> "Combined with t² re-weighting (E[t²]=1/3) the raw base is ~3× weaker."
> 
> This is intentional: FK on pure noise is uninformative. Down-weight
> supervision when the prediction is mostly noise.

---

## Dimension-Specific Reweighting

### Translation Dimension Upweighting
```python
# M2MLoss: trans_dim_weight=5.0
# Applied to: velocity loss, x1 loss (if enabled)

trans_dim_weight: 5.0    # dims [0:3]
other dims:       1.0    # dims [3:198]

Rationale: Compensate for 3-dim translation vs 132-dim rotation imbalance
Effect: Translation dims contribute 5× more strongly to velocity loss
```

---

## Loss Reduction Modes

### element_mean (default)
```python
# All elements (B, L, D) treated equally
# Per-dimension smooth_l1, then average over all valid elements
# Respects padding (data_mask_temporal) and generation masks (MAN)

result = sum(loss_all_elements * combined_mask) / clamp(sum(combined_mask), 1.0)
```

### component_mean (optional, not used in current config)
```python
# KIMODO-style semantic reduction:
# 1. Compute loss for each semantic component (trans, root_rot, body_rot, joint_pos)
# 2. Reduce each component separately
# 3. Average across components

This prevents large components (body rot) from swallowing small ones (translation)
```

---

## Warmup Schedules

All three KIMODO auxiliary losses use **linear warmup from 0 to full weight**:

```python
def _warmup(weight: float, warmup_steps: int, global_step: Optional[int]) -> float:
    if weight == 0.0 or warmup_steps <= 0 or global_step is None:
        return weight
    if global_step >= warmup_steps:
        return weight
    return weight * (float(global_step) / float(warmup_steps))
```

| Loss | Warmup Steps | Warmup Schedule |
|-----|--------------|-----------------|
| aux_joint_pos | 2000 | 0 → 50.0 over steps 0-2000 |
| aux_joint_vel | 2000 | 0 → 500.0 over steps 0-2000 |
| aux_fk_consistency | 2000 | 0 → 1500.0 over steps 0-2000 |

**Purpose**: Gradually introduce strong physical constraints to prevent training instability in early epochs.

---

## Loss Weighting Justification (from config comments)

### Base Values in Normalized vs Denormalized Space

From an already-converged checkpoint (~1 cm joint error):

| Loss | Denormalized Space | Normalized Space |
|-----|------|------|
| velocity | ~0.025 | ~0.025 |
| joint_pos | ~1e-4 m (cm-level) | - |
| joint_vel | ~1e-6 m/frame (mm-level) | - |
| fk_consistency | ~1.4e-6 m (mm-level) | - |

### Weight Calibration Process

1. **Measure base loss value** in denormalized metres
2. **Account for t² averaging** (E[t²]=1/3) → multiply effective weight by 3
3. **Target meaningful fractions** of velocity loss:
   - aux_joint_pos: 50 → ~5.0e-3 (~14% of velocity)
   - aux_joint_vel: 500 → ~1.0e-3 (~4% of velocity)
   - aux_fk_consistency: 1500 → ~2.1e-3 (~7% of velocity)

**Result**: ~25% of total training signal from KIMODO auxiliary losses

---

## Motion Representation (198-dim)

```
Dim Range   Component                           Count
-----------+------------------------------------+-------
[0:3]       Translation (SMPL trans)            3 dims
[3:9]       Root joint 6D rotation              1 joint × 6 = 6 dims
[9:135]     Body joints 6D rotation            21 joints × 6 = 126 dims
[135:198]   Joint positions (Scheme D)         21 joints × 3 = 63 dims
-----------+------------------------------------+-------
TOTAL:      198 dims
```

**Scheme D Position Encoding**:
- X: relative to pelvis (not absolute)
- Y: absolute world height
- Z: relative to pelvis
- Pelvis position omitted (redundant with translation)

---

## Configuration Files Reference

| File | Lines | Section |
|------|-------|---------|
| `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` | 58-127 | Complete loss config + comments |
| `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` | 8-277 | M2MLoss implementation |
| `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py` | 124-333 | KimodoStyleAuxLoss implementation |
| `hftrainer/datasets/motion/motionhub/transforms/compute_198dim.py` | 146-204 | motion198_fk_loss + t² weighting |
| `hftrainer/trainers/motion/hymotion_m2m_trainer.py` | 356-359 | KIMODO aux loss computation call |

