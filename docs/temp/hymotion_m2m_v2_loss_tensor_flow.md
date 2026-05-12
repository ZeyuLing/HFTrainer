# HyMotion M2M v2 — Loss Computation Tensor Flow

## Velocity Loss Detailed Tensor Shapes

### Input
```
pred_vel: (B, L, D)  where D=198  [predicted velocity]
gt_vel:   (B, L, D)              [ground truth velocity]
data_mask_temporal: (B, L)       [1=valid frame, 0=padded]
generation_mask: (B, L, D) opt   [1=generate region, 0=known]
```

---

## Pathway 1: `element_mean` (Default)

### Step 1: Compute per-dimension loss
```python
vel_per_dim = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
# Shape: (B, L, D) = (B, L, 198)
# Each element is loss between corresponding pred/gt pairs
```

### Step 2: Apply trans_dim_weight scaling
```python
if trans_dim_weight != 1.0:  # trans_dim_weight = 5.0 (default)
    dim_weights = torch.ones(D)
    dim_weights[:3] = 5.0   # dims 0,1,2 (translation)
    # dim_weights = [5.0, 5.0, 5.0, 1.0, 1.0, ..., 1.0]  shape (198,)
    
    vel_per_dim = vel_per_dim * dim_weights  # broadcast (B,L,D) * (D,)
    # Shape stays (B, L, D), but dims 0:3 now 5× larger
```

### Step 3: Apply mask and average
```python
# Convert mask to float on same device as vel_per_dim
data_mask = data_mask_temporal.to(vel_per_dim.device).to(vel_per_dim.dtype)
# Shape: (B, L)

# Average over time first (stay per-batch for now)
per_frame = vel_per_dim.mean(dim=-1)  # (B, L) — mean over all D dims
# For each (b, t): per_frame[b, t] = mean(vel_per_dim[b, t, :])

# Weight by temporal mask
weighted = per_frame * data_mask  # (B, L) * (B, L) = (B, L)

# Sum and divide by number of valid frames
mask_sum = torch.clamp(data_mask.sum(), min=1.0)  # scalar ≥ 1.0
loss_velocity = weighted.sum() / mask_sum  # scalar
```

### Step 4: Apply weight
```python
loss_dict["velocity"] = velocity_weight * loss_velocity  # scalar (1.0 * scalar = scalar)
```

**Result**: Single scalar `loss_velocity`

---

## Pathway 2: `component_mean` (KIMODO-style)

### Step 1: Compute per-dimension loss (same)
```python
vel_per_dim = F.smooth_l1_loss(pred_vel, gt_vel, reduction="none")
# Shape: (B, L, D) = (B, L, 198)
```

### Step 2: NO trans_dim_weight scaling
```python
# When velocity_loss_reduction='component_mean':
#   trans_dim_weight = 1.0 (disabled)
# So vel_per_dim stays unchanged
```

### Step 3: Component-wise reduction
```python
# Component boundaries
_motion_components(198) returns:
    ((0, 3), (3, 9), (9, 135), (135, 198))
#   comp1    comp2   comp3     comp4

comp_losses = []
for start, end in components:
    # Extract component slice
    comp = vel_per_dim[..., start:end]  # (B, L, comp_size)
    
    # Apply mask (same structure as element_mean)
    comp_mask = data_mask.unsqueeze(-1).expand_as(comp)  # (B, L, comp_size)
    # comp_mask broadcasts: (B, L, 1) → (B, L, comp_size)
    
    # Compute component average
    denom = comp_mask.sum()
    if denom > 0:
        comp_loss = (comp * comp_mask).sum() / denom  # scalar
        comp_losses.append(comp_loss)

# comp_losses is now a list of 4 scalars (one per component)
```

### Component Detail Example (Component 1: Translation [0:3])
```
comp = vel_per_dim[..., 0:3]           # (B, L, 3) — translation losses
comp_mask = data_mask.unsqueeze(-1)    # (B, L, 1) → broadcast to (B, L, 3)
comp_loss = (comp * comp_mask).sum() / comp_mask.sum()
# Averaging: (B,L,3) summed over all → scalar
```

### Step 4: Average components
```python
loss_velocity = torch.stack(comp_losses).mean()
# input: tensor([comp1_loss, comp2_loss, comp3_loss, comp4_loss])  shape (4,)
# output: scalar = (comp1_loss + comp2_loss + comp3_loss + comp4_loss) / 4
```

**Result**: Still single scalar `loss_velocity`, but internally:
```
loss_velocity = (loss_trans + loss_root_rot + loss_body_rot + loss_pos) / 4
```

---

## With `generation_mask` (Mask-Aware Loss)

### Element-wise masking
```
src_mask: (B, L, D)  where 1 = generation region, 0 = known region
combined = generation_mask * data_mask.unsqueeze(-1)
# (B, L, D) * (B, L, 1) → (B, L, D)
# Only generation regions of valid frames contribute

loss_velocity = (vel_per_dim * combined).sum() / torch.clamp(combined.sum(), min=1.0)
```

### Component-wise with generation_mask
```
for start, end in components:
    comp = vel_per_dim[..., start:end]
    comp_mask = (
        generation_mask[..., start:end] * data_mask.unsqueeze(-1)
    )
    comp_loss = (comp * comp_mask).sum() / comp_mask.sum()
    comp_losses.append(comp_loss)

loss_velocity = torch.stack(comp_losses).mean()
```

---

## Loss Dictionary Aggregation

### M2MLoss Output
```python
loss_dict = {
    "velocity": scalar,              # always computed
    "smoothness": scalar or None,    # if motion_smoothness_weight > 0
    "x1": scalar or None,            # if x1_weight > 0
    "keypoints3d": scalar or None,   # if keypoints3d_weight > 0
    "translation": scalar or None,   # if translation_weight > 0
    "fk_consistency": scalar or None # if fk_consistency_weight > 0 AND global_step
}
```

### KIMODO Auxiliary Loss Output
```python
aux_dict = {
    "aux_joint_pos": scalar or None,          # if joint_pos_weight > 0
    "aux_joint_vel": scalar or None,          # if joint_vel_weight > 0
    "aux_fk_consistency": scalar or None,     # if fk_consistency_weight > 0
}
```

### Combined Loss Dict
```python
losses.update(aux_dict)  # Merge auxiliary into main dict
# losses now contains all enabled components
```

### Total Loss
```python
loss = sum(losses.values())  # Single scalar aggregating all components
```

### Logging
```python
result = {'loss': loss}
for k, v in losses.items():
    result[f'loss_{k}'] = v.detach()

# result = {
#     'loss': scalar,
#     'loss_velocity': scalar,
#     'loss_smoothness': scalar,
#     'loss_aux_joint_pos': scalar,
#     'loss_aux_joint_vel': scalar,
#     'loss_aux_fk_consistency': scalar,
# }
```

---

## KIMODO Auxiliary Loss Details

### `aux_joint_pos`: Global FK Joint Positions

```
Input:  pred_x1_norm (B, L, 198), gt_x1_norm (B, L, 198)

Step 1: Denormalize to world space
    pred_denorm = pred_x1_norm * std + mean  # (B, L, 198)
    pred_135 = pred_denorm[..., :135]        # (B, L, 135) — trans+rot only
    
Step 2: Run FK to get world positions
    pred_world = fk(pred_135)  # (B, L, 22, 3) — global joint positions
    gt_world = fk(gt_135)
    
Step 3: Compute loss
    per_pt = smooth_l1(pred_world, gt_world, reduction="none")  # (B, L, 22, 3)
    per_frame = per_pt.mean(dim=(-1, -2))  # (B, L) — mean over joints and xyz
    
Step 4: Apply t² weighting
    if timestep_squared_weighting:
        per_frame = per_frame * (t² from timesteps)
    
Step 5: Apply temporal mask and average
    per_frame = per_frame * data_mask  # (B, L) * (B, L)
    loss = per_frame.sum() / data_mask.sum()  # scalar
    
Step 6: Apply warm-up and weight
    if global_step < joint_pos_warmup_steps:
        w = global_step / joint_pos_warmup_steps
    loss = joint_pos_weight * w * loss
    
Output: scalar loss_aux_joint_pos
```

### `aux_joint_vel`: Global FK Joint Velocities

```
Input:  pred_x1_norm (B, L, 198), gt_x1_norm (B, L, 198)

Step 1-2: Denormalize and FK (same as aux_joint_pos)
    pred_world = fk(pred_135)  # (B, L, 22, 3)
    gt_world = fk(gt_135)
    
Step 3: Compute temporal derivative
    pred_vel = pred_world[:, 1:] - pred_world[:, :-1]  # (B, L-1, 22, 3)
    gt_vel = gt_world[:, 1:] - gt_world[:, :-1]
    
Step 4: Compute loss
    per_pt = smooth_l1(pred_vel, gt_vel, reduction="none")  # (B, L-1, 22, 3)
    per_frame = per_pt.mean(dim=(-1, -2))  # (B, L-1)
    
Step 5: Apply t² weighting
    per_frame = per_frame * t²  # if enabled
    
Step 6: Apply VELOCITY mask (both endpoints must be valid)
    vel_mask = data_mask[:, 1:] * data_mask[:, :-1]  # (B, L-1)
    per_frame = per_frame * vel_mask
    loss = per_frame.sum() / vel_mask.sum()  # scalar
    
Step 7: Apply warm-up and weight
    w = warmup_schedule(...)
    loss = joint_vel_weight * w * loss
    
Output: scalar loss_aux_joint_vel
```

### `aux_fk_consistency`: Intra-prediction Consistency

```
Input:  pred_x1_norm (B, L, 198)

Step 1: Denormalize
    pred_denorm = pred_x1_norm * std + mean  # (B, L, 198)
    pred_135 = pred_denorm[..., :135]        # (B, L, 135) — trans+rot
    pred_pos_chan = pred_denorm[..., 135:]   # (B, L, 63) — stored pos
    
Step 2: Run FK to get FK-derived positions
    pred_world = fk(pred_135)  # (B, L, 22, 3) — world positions
    fk_pos = scheme_d_relative(pred_world)   # (B, L, 63) — rel-pelvis pos
    
Step 3: Compute consistency loss
    per_pt = smooth_l1(pred_pos_chan, fk_pos, reduction="none")  # (B, L, 63)
    per_frame = per_pt.mean(dim=-1)  # (B, L)
    
Step 4: Apply t² weighting
    per_frame = per_frame * t²  # if enabled
    
Step 5: Apply temporal mask
    per_frame = per_frame * data_mask  # (B, L)
    loss = per_frame.sum() / data_mask.sum()  # scalar
    
Step 6: Apply warm-up and weight
    w = warmup_schedule(...)
    loss = fk_consistency_weight * w * loss
    
Output: scalar loss_aux_fk_consistency

NOTE: This is pure intra-prediction (no GT needed), so generation_mask
      is NOT applied here — KIMODO supervises all frames uniformly.
```

---

## Summary: Shape Transitions

### Element-Mean Path
```
(B, L, 198) ──smooth_l1──→ (B, L, 198)          [per-dim loss]
(B, L, 198) ──scale by [5,5,5,1,...]──→ (B, L, 198)  [if trans_dim_weight]
(B, L, 198) ──mean over D──→ (B, L)             [per-frame average]
(B, L) ──* data_mask──→ (B, L)                 [temporal weighting]
(B, L) ──sum / count──→ scalar                  [final loss]
```

### Component-Mean Path
```
(B, L, 198) ──smooth_l1──→ (B, L, 198)         [per-dim loss]
(B, L, 198) ──split [0:3], [3:9], [9:135], [135:198]──→ 4 × (B, L, size_i)
    ∀ component_i:
        (B, L, size_i) ──mean + mask + sum──→ scalar_i
4 scalars ──mean──→ scalar                    [final loss]
```

### KIMODO Joint Pos Path (example)
```
(B, L, 198) ──denorm──→ (B, L, 198)
(B, L, 135) ──fk──→ (B, L, 22, 3)            [world joint pos]
(B, L, 22, 3) ──smooth_l1──→ (B, L, 22, 3)  [per-joint loss]
(B, L, 22, 3) ──mean over (J, xyz)──→ (B, L)  [per-frame average]
(B, L) ──* t²──→ (B, L)                      [t² down-weight]
(B, L) ──* data_mask + sum / count──→ scalar [final loss]
```

