# HyMotion M2M v2 Flow Matching Loss Computation — Complete Analysis

## Executive Summary

The flow matching loss in HyMotion M2M v2 is highly configurable with two distinct reduction modes:

1. **`element_mean` (default)**: Computes a single average over all D dimensions
2. **`component_mean` (KIMODO-style, Scheme-D)**: Splits into semantic components, averages each component separately, then averages across components

The `trans_dim_weight` parameter **scales dimensions [0:3] WITHIN the same loss function** — it does NOT create a separate logged component. All component losses are logged with the key `loss_{component_name}`.

---

## Question 1: Is velocity loss averaged over all 198 dims or split into components?

### Answer: **Depends on `velocity_loss_reduction` parameter**

#### Mode 1: `element_mean` (Default)
- **Single average over all 198 dimensions**
- Applied in: `M2MLoss._masked_motion_loss()` when `velocity_loss_reduction == "element_mean"`

**File**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py:71-80`

```python
if self.velocity_loss_reduction == "element_mean":
    if generation_mask is not None:
        gen_mask = generation_mask.to(per_dim.device).to(per_dim.dtype)
        combined = gen_mask * data_mask.unsqueeze(-1)
        mask_sum = torch.clamp(combined.sum(), min=1.0)
        return (per_dim * combined).sum() / mask_sum    # Single average
    
    per_frame = per_dim.mean(dim=-1)  # Average over D
    mask_sum = torch.clamp(data_mask.sum(), min=1.0)
    return (per_frame * data_mask).sum() / mask_sum     # Single scalar loss
```

**Result**: One scalar `loss_velocity` that treats all 198 dims equally (after trans_dim_weight scaling).

---

#### Mode 2: `component_mean` (KIMODO-style, Scheme-D)
- **Splits into 4 semantic components, averages each separately, then averages the component averages**
- Applied in: `M2MLoss._masked_motion_loss()` when `velocity_loss_reduction == "component_mean"`

**File**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py:82-104`

```python
# KIMODO-style semantic reduction: each representation component
# first gets its own valid-cell mean, then active components are
# averaged.  This prevents large components (e.g. body rot6d) from
# swallowing small but important ones such as translation/root.
comp_losses = []
for start, end in self._motion_components(per_dim.shape[-1]):
    comp = per_dim[..., start:end]
    # ... apply mask ...
    denom = comp_mask.sum()
    if torch.gt(denom.detach(), 0):
        comp_losses.append((comp * comp_mask).sum() / denom)

if not comp_losses:
    return per_dim.sum() * 0.0
return torch.stack(comp_losses).mean()  # Average the component averages
```

**Component Structure** (lines 55-60):

```python
@staticmethod
def _motion_components(dim: int):
    if dim >= 198:
        return ((0, 3), (3, 9), (9, 135), (135, 198))    # For 198-dim
    if dim >= 135:
        return ((0, 3), (3, 9), (9, 135))                # For 135-dim
    return ((0, dim),)
```

**Component breakdown for 198-dim**:
- **Component 1**: dims [0:3] — Translation (3 dims)
- **Component 2**: dims [3:9] — Root rotation 6D (6 dims)
- **Component 3**: dims [9:135] — Body rotations (126 dims = 21 joints × 6D)
- **Component 4**: dims [135:198] — Position channels (63 dims = 21 joints × 3D)

**Result**: Still ONE `loss_velocity`, but internally computed as:
```
loss_velocity = (loss_trans + loss_root_rot + loss_body_rot + loss_pos) / 4
```

Each component gets its own average loss first, **preventing the large body_rot (126 dims) from dominating the small translation (3 dims)**.

**When enabled**: Via config `velocity_loss_reduction='component_mean'` (see `configs/hymotion_m2m_v2/loss_component_mean/*.py`)

---

## Question 2: What per-component losses are logged?

### Loss Dict Keys — Exact Output

**File**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:394-401`

```python
def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
    ctx = self._prepare_and_forward(batch)
    losses = self._compute_base_loss(ctx)          # Returns loss_dict
    loss = sum(losses.values())                    # Aggregate
    result = {'loss': loss}                        # Total loss
    for k, v in losses.items():
        result[f'loss_{k}'] = v.detach()           # Log each component as loss_{k}
    return result
```

### Possible Keys in Logged Output

#### Main Flow Matching Loss (from `M2MLoss.forward()`):

**File**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py:135-222`

1. **`velocity`** → logged as `loss_velocity` (line 147)
   - Weight: `velocity_weight` (default 1.0)
   - Reduction: `element_mean` or `component_mean`
   - Applied trans_dim_weight scaling: ✓

2. **`x1`** → logged as `loss_x1` (line 159)
   - Weight: `x1_weight` (default 0.0 → usually NOT logged)
   - Reduction: Same as velocity (element_mean or component_mean)
   - Applied trans_dim_weight scaling: ✓

3. **`keypoints3d`** → logged as `loss_keypoints3d` (line 171)
   - Weight: `keypoints3d_weight` (default 0.0 → usually NOT logged)
   - Computed from FK on 22 joints

4. **`translation`** → logged as `loss_translation` (line 181)
   - Weight: `translation_weight` (default 0.0 → usually NOT logged)
   - Shape: (B, L, 3)

5. **`smoothness`** → logged as `loss_smoothness` (line 205)
   - Weight: `motion_smoothness_weight` (default 0.5)
   - Computed as smooth_l1(pred_x1[1:] - pred_x1[:-1], gt_x1[1:] - gt_x1[:-1])
   - Penalizes frame-to-frame acceleration

6. **`fk_consistency`** → logged as `loss_fk_consistency` (line 218)
   - Weight: `fk_consistency_weight` (default 0.0 → usually NOT logged)
   - Linear warm-up over `fk_consistency_warmup_steps`
   - Compares pos channels [135:198] vs FK-derived positions

#### KIMODO-Style Auxiliary Losses (from `KimodoStyleAuxLoss.forward()`):

**File**: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py:201-333`

These are SEPARATE from the main M2MLoss and added via `aux_losses.update()` (trainer line 360, 389):

7. **`aux_joint_pos`** → logged as `loss_aux_joint_pos` (line 296)
   - Weight: `joint_pos_weight` (default 50.0 in base config)
   - Computed from: smooth_l1(FK global joint positions, GT global joint positions)
   - Shape: (B, L, 22, 3) → (B, L) → scalar
   - Warm-up: `joint_pos_warmup_steps` (default 2000)

8. **`aux_joint_vel`** → logged as `loss_aux_joint_vel` (line 315)
   - Weight: `joint_vel_weight` (default 500.0 in base config)
   - Computed from: smooth_l1(temporal derivative of FK positions)
   - Shape: (B, L-1, 22, 3) → (B, L-1) → scalar
   - Warm-up: `joint_vel_warmup_steps` (default 2000)

9. **`aux_fk_consistency`** → logged as `loss_aux_fk_consistency` (line 331)
   - Weight: `fk_consistency_weight` (default 1500.0 in base config)
   - Computed from: smooth_l1(pos channels [135:198], FK-derived rel-pelvis positions)
   - Shape: (B, L, 63) → (B, L) → scalar
   - Warm-up: `fk_consistency_warmup_steps` (default 2000)
   - NOTE: This is SEPARATE from `loss_fk_consistency` (M2MLoss version)

### Summary Table — What Gets Logged in Default Config

| Key | Source | Default Weight | Logged? |
|-----|--------|-----------------|---------|
| `loss_velocity` | M2MLoss | 1.0 | ✅ YES |
| `loss_x1` | M2MLoss | 0.0 | ❌ NO |
| `loss_keypoints3d` | M2MLoss | 0.0 | ❌ NO |
| `loss_translation` | M2MLoss | 0.0 | ❌ NO |
| `loss_smoothness` | M2MLoss | 0.5 | ✅ YES |
| `loss_fk_consistency` | M2MLoss | 0.0 | ❌ NO |
| `loss_aux_joint_pos` | KimodoStyleAuxLoss | 50.0 | ✅ YES |
| `loss_aux_joint_vel` | KimodoStyleAuxLoss | 500.0 | ✅ YES |
| `loss_aux_fk_consistency` | KimodoStyleAuxLoss | 1500.0 | ✅ YES |

---

## Question 3: How is `trans_dim_weight=5.0` applied?

### Answer: **Scales dims [0:3] within the combined loss, NOT a separate component**

#### Implementation

**File**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py:142-149`

```python
if pred_vel is not None and gt_vel is not None:
    # velocity loss: (B, L, D) -> scalar
    # Apply per-dimension weighting: upweight translation dims (first trans_dims)
    # to compensate for the 3/135 dimension ratio imbalance
    vel_per_dim = self.loss_fn(pred_vel, gt_vel, reduction="none")  # (B, L, D)
    if self.trans_dim_weight != 1.0:
        dim_weights = torch.ones(vel_per_dim.shape[-1], device=vel_per_dim.device)
        dim_weights[:self.trans_dims] = self.trans_dim_weight            # ← Scale first 3 dims
        vel_per_dim = vel_per_dim * dim_weights                         # ← Element-wise multiply
    loss_dict["velocity"] = self.velocity_weight * self._masked_motion_loss(
        vel_per_dim, data_mask_temporal, generation_mask
    )
```

#### Behavior

1. Create a per-dimension weight vector: `[5.0, 5.0, 5.0, 1.0, 1.0, ..., 1.0]` (shape D)
   - First 3 elements (translation) = 5.0
   - Rest (rotation + position) = 1.0

2. Element-wise multiply the loss tensor: `vel_per_dim *= dim_weights`
   - Translation dimensions now contribute 5× to the loss

3. **NO separate loss component is logged** — still just one `loss_velocity`

4. The scaling is applied **before** the reduction (element_mean or component_mean)

#### Applied to Both Velocity and X1

The same scaling is applied to `loss_x1`:

**File**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py:154-161`

```python
if pred_x1 is not None and gt_x1 is not None:
    x1_per_dim = self.loss_fn(pred_x1, gt_x1, reduction="none")  # (B, L, D)
    if self.trans_dim_weight != 1.0:
        dim_weights = torch.ones(x1_per_dim.shape[-1], device=x1_per_dim.device)
        dim_weights[:self.trans_dims] = self.trans_dim_weight
        x1_per_dim = x1_per_dim * dim_weights
    loss_dict["x1"] = self.x1_weight * self._masked_motion_loss(
        x1_per_dim, data_mask_temporal, generation_mask
    )
```

#### When It's Disabled

**File**: `configs/hymotion_m2m_v2/loss_component_mean/hymotion_m2m_v2_uncond_local_046b_component_mean.py`

```python
model = dict(
    losses_cfg=dict(
        velocity_loss_reduction='component_mean',
        trans_dim_weight=1.0,    # ← Disabled (= 1.0 = no scaling)
    ),
)
```

**Reason**: In KIMODO-style component reduction, translation already gets its own semantic slot (component 1), so the 5× scaling would be redundant. Each component is already balanced.

---

## Configuration Examples

### Default (element_mean with trans_dim_weight=5.0)

**File**: `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py:58-71`

```python
losses_cfg=dict(
    loss_type='smooth_l1',
    velocity_weight=1.0,
    x1_weight=0.0,
    keypoints3d_weight=0.0,
    translation_weight=0.0,
    trans_dim_weight=5.0,              # ← Scale trans 5×
    motion_smoothness_weight=0.5,
    fk_consistency_weight=0.0,
    fk_consistency_warmup_steps=2000,
),
```

**Loss computation**:
```
loss_velocity = mean_over_all_dims(vel_per_dim * [5, 5, 5, 1, 1, ..., 1])
```

### KIMODO-style (component_mean with trans_dim_weight=1.0)

**File**: `configs/hymotion_m2m_v2/loss_component_mean/hymotion_m2m_v2_uncond_local_046b_component_mean.py`

```python
losses_cfg=dict(
    velocity_loss_reduction='component_mean',
    trans_dim_weight=1.0,              # ← No scaling
),
```

**Loss computation**:
```
loss_velocity = mean([
    mean(vel_per_dim[0:3]),           # Translation component
    mean(vel_per_dim[3:9]),           # Root rotation
    mean(vel_per_dim[9:135]),         # Body rotation
    mean(vel_per_dim[135:198]),       # Position
])
```

---

## KIMODO Auxiliary Losses — Complete Details

### Why Separate?

The KIMODO-style auxiliary losses (`aux_joint_pos`, `aux_joint_vel`, `aux_fk_consistency`) are:
- Computed **post-hoc from normalized x1** via differentiable FK
- **NOT** part of the main flow-matching loss
- Operate in **denormalized world-space** (metres), not normalized space
- Include **t² re-weighting** to down-weight pure-noise samples

### Integration

**File**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:352-360`

```python
# ---- KIMODO-style auxiliary losses ----
# Operate on (pred_x1, gt_x1) in normalised space; computed via
# FK on rotation+translation channels.  Padding-aware; ignores
# generation_mask by design (KIMODO supervises every frame).
aux_losses = self._compute_kimodo_aux_loss(
    pred_x1_for_smooth, x1, timesteps, tgt_padding_mask
)
if aux_losses:
    losses.update(aux_losses)
```

The KIMODO losses are merged into the main `losses` dict and aggregated via `sum(losses.values())`.

### t² Re-weighting

**File**: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py:279-283`

```python
# Optional t² re-weighting (matches existing motion198_fk_loss).
if self.timestep_squared_weighting and timesteps is not None:
    t_sq = (timesteps.to(pred_world.device).to(pred_world.dtype) ** 2)  # (B,)
else:
    t_sq = None
```

**Effect**: Multiplies each term by t² ∈ [0, 1], down-weighting early diffusion steps where x_t is pure noise.

---

## Complete Loss Aggregation Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ trainer.train_step(batch)                                       │
│ → _prepare_and_forward(batch) → _compute_base_loss(ctx)        │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   M2MLoss.forward()        _compute_kimodo_aux_loss()
   (m2m_loss)               (kimodo_aux_loss)
        │                         │
        ├─ loss_velocity          ├─ aux_joint_pos
        ├─ loss_smoothness        ├─ aux_joint_vel
        ├─ loss_fk_consistency    └─ aux_fk_consistency
        ├─ (loss_x1) [if enabled]
        ├─ (loss_keypoints3d) [if enabled]
        └─ (loss_translation) [if enabled]
        │                         │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ losses.update(aux_dict) │
        │ Combined loss_dict      │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────────────┐
        │ loss = sum(losses.values())     │
        │ result['loss'] = loss (scalar)  │
        └────────────┬────────────────────┘
                     │
        ┌────────────▼────────────────────────────┐
        │ for k, v in losses.items():             │
        │   result[f'loss_{k}'] = v.detach()      │
        │                                         │
        │ → loss_loss_velocity                    │
        │ → loss_loss_smoothness                  │
        │ → loss_loss_aux_joint_pos               │
        │ → loss_loss_aux_joint_vel               │
        │ → loss_loss_aux_fk_consistency          │
        │ etc.                                    │
        └─────────────────────────────────────────┘
```

---

## Summary — Key Takeaways

| Question | Answer | File | Lines |
|----------|--------|------|-------|
| **Q1: Single or split velocity loss?** | Depends on `velocity_loss_reduction`: `element_mean` = single average; `component_mean` = split into 4 components, each averaged separately | `m2m_loss.py` | 71-104 |
| **Q2: What gets logged?** | 6–9 keys depending on weights: `loss_velocity`, `loss_smoothness`, `loss_aux_joint_pos`, `loss_aux_joint_vel`, `loss_aux_fk_consistency` (KIMODO), plus optional `loss_x1`, `loss_keypoints3d`, `loss_translation`, `loss_fk_consistency` | `m2m_loss.py`: 135–222; `kimodo_aux_loss.py`: 201–333 | `hymotion_m2m_trainer.py`: 399–400 |
| **Q3: How is trans_dim_weight applied?** | Scales dims [0:3] by 5.0× **within** the velocity loss (before reduction), NO separate component logged | `m2m_loss.py` | 142–149, 154–161 |
| **Disabled by component_mean?** | YES — when `velocity_loss_reduction='component_mean'`, set `trans_dim_weight=1.0` because translation already has its own semantic slot | `loss_component_mean/*.py` | - |

---

## File Reference Summary

### Primary Loss Definition
- **`hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`** (223 lines)
  - M2MLoss class with element_mean / component_mean modes
  - velocity, x1, smoothness, keypoints3d, translation, fk_consistency losses
  - trans_dim_weight scaling (lines 142–149, 154–161)
  - Component structure definition (lines 55–60)

### KIMODO Auxiliary Losses
- **`hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`** (334 lines)
  - KimodoStyleAuxLoss class (post-hoc FK-based losses)
  - aux_joint_pos, aux_joint_vel, aux_fk_consistency
  - t² re-weighting, warm-up scheduling
  - Denormalization and FK computation

### Trainer Integration
- **`hftrainer/trainers/motion/hymotion_m2m_trainer.py`** (596 lines)
  - _compute_base_loss() aggregates M2MLoss + KimodoStyleAuxLoss (lines 288–392)
  - train_step() logs all loss components (lines 394–401)
  - _compute_kimodo_aux_loss() calls the auxiliary loss module (lines 447–484)
  - _compute_fk_consistency_loss() computes the M2MLoss variant (lines 486–521)

### Configuration Examples
- **`configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`** (245 lines)
  - Default: element_mean, trans_dim_weight=5.0 (lines 58–71)
  - KIMODO aux weights: joint_pos=50, joint_vel=500, fk_consistency=1500 (lines 118–127)

- **`configs/hymotion_m2m_v2/loss_component_mean/*.py`** (4 files)
  - component_mean mode with trans_dim_weight=1.0

