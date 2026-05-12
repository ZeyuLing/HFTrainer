# HyMotion M2M v2 Loss Computation — Quick Reference

## 3 Main Questions Answered

### 1️⃣ **Velocity Loss: Single Average or Split?**

| Mode | Behavior | File | Config |
|------|----------|------|--------|
| **`element_mean`** (default) | Single average over all 198 dims | `m2m_loss.py:71-80` | Default |
| **`component_mean`** (KIMODO) | Split into 4 components, average each separately | `m2m_loss.py:82-104` | `loss_component_mean/*.py` |

**Component Structure** (when `component_mean`):
```
[0:3]      = Translation (3 dims)         → avg → component 1
[3:9]      = Root rot 6D (6 dims)         → avg → component 2  
[9:135]    = Body rot 6D (126 dims)       → avg → component 3
[135:198]  = Position (63 dims)           → avg → component 4
                    ↓
        loss_velocity = mean([comp1, comp2, comp3, comp4])
```

---

### 2️⃣ **What Gets Logged?**

**Default config logging** (9 keys):

| Key | Source | Weight | Logged? | Notes |
|-----|--------|--------|---------|-------|
| **`loss_velocity`** | M2MLoss | 1.0 | ✅ | Main flow loss |
| **`loss_smoothness`** | M2MLoss | 0.5 | ✅ | Frame-to-frame diff |
| **`loss_aux_joint_pos`** | KimodoAux | 50.0 | ✅ | FK global positions |
| **`loss_aux_joint_vel`** | KimodoAux | 500.0 | ✅ | FK joint velocities |
| **`loss_aux_fk_consistency`** | KimodoAux | 1500.0 | ✅ | pos ↔ FK consistency |
| `loss_x1` | M2MLoss | 0.0 | ❌ | Usually disabled |
| `loss_keypoints3d` | M2MLoss | 0.0 | ❌ | Usually disabled |
| `loss_translation` | M2MLoss | 0.0 | ❌ | Usually disabled |
| `loss_fk_consistency` | M2MLoss | 0.0 | ❌ | Usually disabled (KIMODO takes over) |

**How logged** (`hymotion_m2m_trainer.py:399-400`):
```python
for k, v in losses.items():
    result[f'loss_{k}'] = v.detach()  # → loss_velocity, loss_smoothness, etc.
```

---

### 3️⃣ **How does `trans_dim_weight=5.0` work?**

**Answer**: Scales dims [0:3] **within** the velocity loss, creates **NO separate component**

**Implementation** (`m2m_loss.py:142-149`):
```python
vel_per_dim = self.loss_fn(pred_vel, gt_vel, reduction="none")  # (B, L, D)
if self.trans_dim_weight != 1.0:
    dim_weights = torch.ones(vel_per_dim.shape[-1])
    dim_weights[:self.trans_dims] = self.trans_dim_weight        # [5.0, 5.0, 5.0, 1.0, ...]
    vel_per_dim = vel_per_dim * dim_weights                     # Element-wise multiply
loss_dict["velocity"] = self.velocity_weight * self._masked_motion_loss(...)
```

**Result**:
- Translation dims contribute 5× to loss → no separate `loss_velocity_trans` logged
- Still just one `loss_velocity` total
- Applied **before** element_mean or component_mean reduction

**Disabled when?**
```python
# In component_mean mode, translation already has semantic slot → no scaling needed
velocity_loss_reduction='component_mean'
trans_dim_weight=1.0  # Disabled
```

---

## KIMODO Auxiliary Losses (3 Extra Terms)

These are **NOT** part of M2MLoss — they're computed post-hoc via FK:

### `aux_joint_pos` (Weight: 50.0)
- **What**: Smooth-L1 between FK-derived global joint positions and GT positions
- **Why**: Suppresses foot-slipping by directly supervising world-space leg positions
- **Formula**: `smooth_l1(FK(pred_x1[:135])[global], GT_kp[global])`
- **Shape**: (B, L, 22, 3) → scalar
- **Warm-up**: Over 2000 steps

### `aux_joint_vel` (Weight: 500.0)
- **What**: Smooth-L1 on temporal derivative of FK positions (d/dt of joint positions)
- **Why**: Velocity-level supervision directly punishes slipping trajectories
- **Formula**: `smooth_l1(d/dt FK(pred_x1[:135])[global], d/dt GT_kp[global])`
- **Shape**: (B, L-1, 22, 3) → scalar
- **Warm-up**: Over 2000 steps

### `aux_fk_consistency` (Weight: 1500.0)
- **What**: Intra-prediction consistency: pos channels [135:198] vs FK-derived rel-pelvis positions
- **Why**: Teaches model explicit FK equivalence so inference-time pos-only conditioning works
- **Formula**: `smooth_l1(pred_x1[135:198], FK_scheme_d_relative(pred_x1[:135]))`
- **Shape**: (B, L, 63) → scalar
- **Warm-up**: Over 2000 steps

**t² Weighting**: Each term multiplied by t² ∈ [0, 1] to down-weight pure-noise samples

---

## Loss Aggregation Pipeline

```
┌─ M2MLoss.forward() ──────────────────┐   ┌─ KimodoStyleAuxLoss.forward() ─┐
│                                       │   │                                 │
│ velocity (1.0)                        │   │ aux_joint_pos (50.0)            │
│ smoothness (0.5)                      │   │ aux_joint_vel (500.0)           │
│ [fk_consistency] (0.0, disabled)      │   │ aux_fk_consistency (1500.0)     │
│                                       │   │                                 │
└─────────────────┬─────────────────────┘   └────────────┬────────────────────┘
                  │                                      │
                  └──────────┬───────────────────────────┘
                             │
                  ┌──────────▼───────────┐
                  │ losses.update()      │
                  │ Combined dict        │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────────────┐
                  │ loss = sum(losses.values())  │
                  │ Scalar total loss            │
                  └──────────┬───────────────────┘
                             │
                  ┌──────────▼────────────────────────┐
                  │ Log each key as loss_{k}         │
                  │                                  │
                  │ loss_velocity                    │
                  │ loss_smoothness                  │
                  │ loss_aux_joint_pos               │
                  │ loss_aux_joint_vel               │
                  │ loss_aux_fk_consistency          │
                  └──────────────────────────────────┘
```

---

## Default Config Snippet

```python
# From configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py

losses_cfg=dict(
    loss_type='smooth_l1',
    velocity_weight=1.0,                      # ← Main flow loss
    x1_weight=0.0,
    keypoints3d_weight=0.0,
    translation_weight=0.0,
    trans_dim_weight=5.0,                     # ← Scale translation 5×
    motion_smoothness_weight=0.5,             # ← Frame-to-frame smoothing
    fk_consistency_weight=0.0,                # ← Disabled (KIMODO replaces it)
    fk_consistency_warmup_steps=2000,
),
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=50.0,                    # ← FK joint position loss
    joint_vel_weight=500.0,                   # ← FK joint velocity loss
    fk_consistency_weight=1500.0,             # ← Intra-pred consistency
    loss_type='smooth_l1',
    timestep_squared_weighting=True,          # ← t² down-weighting
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
),
```

---

## Component-Mean Config Snippet

```python
# From configs/hymotion_m2m_v2/loss_component_mean/hymotion_m2m_v2_uncond_local_046b_component_mean.py

model = dict(
    losses_cfg=dict(
        velocity_loss_reduction='component_mean',    # ← Split into 4 components
        trans_dim_weight=1.0,                        # ← Disabled (semantic slot)
    ),
)
```

---

## File Quick Links

| File | Purpose | Key Lines |
|------|---------|-----------|
| `m2m_loss.py` | Main loss definition | 55-60 (components), 71-104 (reduction), 142-149 (scaling) |
| `kimodo_aux_loss.py` | KIMODO auxiliary losses | 124-167 (class), 288-315 (joint_pos/vel), 320-331 (fk_cons) |
| `hymotion_m2m_trainer.py` | Trainer integration | 288-392 (loss compute), 394-401 (logging), 447-484 (KIMODO call) |
| Config base | Default settings | 58-71 (losses), 118-127 (KIMODO) |
| Config component_mean | Component reduction | 10-12 (settings) |

