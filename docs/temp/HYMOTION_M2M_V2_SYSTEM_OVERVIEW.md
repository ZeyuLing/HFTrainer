# HyMotion M2M v2 System Overview & Quick Start Guide

## 📋 Document Map

You now have **two comprehensive reference documents:**

1. **`HYMOTION_M2M_V2_CRITICAL_FILES.md`** (14 KB)
   - Deep dive into each component
   - Loss function explanations
   - KIMODO auxiliary loss rationale
   - Motion representation semantics

2. **`HYMOTION_M2M_V2_LINE_REFERENCE.md`** (7.1 KB)
   - Exact line numbers for every critical element
   - Quick lookup tables
   - Key instantiation sequences

---

## 🔑 Key Facts at a Glance

### Motion Representation
- **198 dimensions total:**
  - `[0:3]` Translation (3D world position)
  - `[3:135]` 22 joints × 6D rot6d = 132D (row-major format)
  - `[135:198]` 21 joints × 3D position = 63D (XZ rel-pelvis, Y absolute)

### The Three Loss Tracks (Phase 2b Config)

#### 1. **M2MLoss** (Main Velocity Loss)
```python
# Lines 27–41 in caption config
losses_cfg=dict(
    loss_type='smooth_l1',
    velocity_weight=1.0,
    velocity_loss_reduction='component_mean',  # ← v2 v1 change: equal 25% to each component
    trans_dim_weight=1.0,                       # ← adjusted to 1.0 with component_mean
    motion_smoothness_weight=0.5,
    fk_consistency_weight=0.0,                  # ← DISABLED: replaced by KIMODO
)
```

**Components (4 groups, each gets 25% weight):**
1. trans (3D)
2. root_rot (6D)
3. body_rot (126D)
4. joint_pos (63D)

#### 2. **KimodoStyleAuxLoss** (Foot Skating Suppression)
```python
# Lines 44–53 in caption config
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=50.0,         # Suppress pelvis cheating
    joint_vel_weight=500.0,        # Main skating killer
    fk_consistency_weight=1500.0,  # Enforce pos↔rot consistency
    loss_type='smooth_l1',
    timestep_squared_weighting=True,
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
)
```

**Three auxiliary terms:**
1. **joint_pos (γ₃):** Global joint positions must match GT (in denormalised metres)
2. **joint_vel (γ₄):** Global joint velocities must match GT (catches slipping)
3. **fk_consistency (γ₇):** Pos channels must satisfy FK(rot6d[:, :, 1:], trans)

#### 3. **Smoothness Regularization**
```python
motion_smoothness_weight=0.5  # Temporal consistency term (existing M2MLoss feature)
```

---

## 🎯 Critical Config Decisions: Phase 2b Changes

### Change 1: `velocity_loss_reduction='component_mean'`
- **Why:** Under element_mean, 126-dim body_rot drowns out 3-dim translation
  - translation: ~1.5% of loss under element_mean
  - translation: 25% of loss under component_mean
- **Math:** Each component first averaged independently, then meta-averaged
- **Impact:** Better translation supervision, but requires trans_dim_weight adjustment

### Change 2: `trans_dim_weight=1.0` (down from base 5.0)
- **Why:** component_mean already gives translation 25% (not 1.5%), so per-dim upweighting overcorrects
  - 5.0 × 25% = 125% → overcorrection to ~55%
  - 1.0 × 25% = 25% → correct balance
- **Impact:** Avoids learning instability from over-supervision

### Change 3: KIMODO Auxiliary Losses Enabled
- **Primary change:** `fk_consistency_weight=0.0` (M2MLoss) → use KIMODO instead
- **Why:** KIMODO runs on denormalised metres (FK world space), not normalised latent
- **Impact:** Direct world-space supervision prevents foot skating

---

## 🔗 Code Instantiation Flow

```
Config File (caption_local_phase2b.py, lines 17-131)
  ├─ _base_ = '_base_hymotion_m2m_v2_046b.py'
  │   ├─ Transformer: HunyuanMotionMMDiT, 18 layers, 594→198
  │   ├─ Rotation space: 'local' SMPL
  │   ├─ Mean/Std path: 'data/hymotion_m2m_data/_stats_198dim'
  │   └─ VACE mode: 'no_inactive' (x_t + reactive + mask = 3×198)
  │
  └─ Phase 2b Overrides (lines 21-131)
     ├─ losses_cfg (lines 27-41)
     │   └─ velocity_loss_reduction='component_mean'
     │   └─ M2MLoss instantiated at bundle line 121
     │
     ├─ kimodo_aux_loss_cfg (lines 44-53)
     │   └─ KimodoStyleAuxLoss instantiated at bundle line 129
     │
     └─ Load from checkpoint epoch 3320 (line 125)
        └─ Patch null embeddings from T2M pretrained (line 129)

↓ Training Step ↓

Bundle.forward() → Trainer computes:
  1. M2MLoss (velocity + smoothness)
  2. KimodoStyleAuxLoss (joint_pos + joint_vel + fk_consistency)
  3. Combined loss = loss_velocity + aux_losses
```

---

## 📐 Motion Dimensionality: Detailed Breakdown

### Full 198-Dim Layout (with indices)

```
[0]       X translation
[1]       Y translation
[2]       Z translation
────────────────────────
[3:9]     Root (Pelvis) rotation 6D
────────────────────────
[9:135]   Body rotations 6D × 21 joints
          [9:15]    Joint 1 (L_Hip)
          [15:21]   Joint 2 (R_Hip)
          ... (19 more joints)
          [129:135] Joint 21 (R_Wrist)
────────────────────────
[135:198] Joint positions (21 × 3)
          [135:137]  Joint 1 X, Y, Z (rel-pelvis for X,Z; absolute for Y)
          [137:140]  Joint 2 X, Y, Z
          ... (19 more)
          [195:198]  Joint 21 X, Y, Z
```

### What Gets Normalized?
- **Input:** Each dimension normalized by per-dim mean/std from training data
  - Mean: `data/hymotion_m2m_data/_stats_198dim/Mean.npy` (198,)
  - Std: `data/hymotion_m2m_data/_stats_198dim/Std.npy` (198,)
  - Loaded in Bundle `_load_mean_std()` (lines 149–163)

- **Denormalization in decode** (bundle.py lines 434–435):
  ```python
  std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
  latent_denorm = latent * std + self.mean
  ```

---

## 🦴 SMPL-22 Kinematic Tree (fk_utils.py lines 29-52)

### Joint Indices (0-indexed)
```
0: Pelvis (root)
├─ 1: L_Hip
│  ├─ 4: L_Knee
│  │  └─ 7: L_Ankle
│  │     └─ 10: L_Foot ← Skating target
│  └─ (chain continues)
│
├─ 2: R_Hip
│  ├─ 5: R_Knee
│  │  └─ 8: R_Ankle
│  │     └─ 11: R_Foot ← Skating target
│  └─ (chain continues)
│
├─ 3: Spine1
│  └─ 6: Spine2
│     └─ 9: Spine3
│        ├─ 12: Neck
│        │  └─ 15: Head
│        ├─ 13: L_Collar → 16: L_Shoulder → 18: L_Elbow → 20: L_Wrist
│        └─ 14: R_Collar → 17: R_Shoulder → 19: R_Elbow → 21: R_Wrist
```

### Foot Skating Prevention (via KIMODO)
- **joint_vel loss** on ALL 22 joints' world-space velocities
- **Especially critical for joints 10 & 11** (feet): any pelvis-without-leg translation causes immediate velocity error

---

## 📊 Loss Weight Magnitudes Explained (Phase 2b)

### M2MLoss Velocity Components
Under `component_mean` reduction with `trans_dim_weight=1.0`:
```
Component          | Dims | Weight in Loss | Proportion
───────────────────┼──────┼────────────────┼────────────
Translation        | 3    | 25%            | 25% (was 1.5% under element_mean)
Root Rotation      | 6    | 25%            | 25%
Body Rotation      | 126  | 25%            | 25% (previously dominant)
Joint Position     | 63   | 25%            | 25%
───────────────────┴──────┴────────────────┴────────────
TOTAL velocity weight = 1.0 (in loss_dict)
```

### KIMODO Auxiliary Loss Magnitudes
(In denormalised metres; smooth_l1 quadratic regime, O(1e-4) base values)
```
Loss Term           | Weight | Reasoning
────────────────────┼────────┼─────────────────────────────────
joint_pos           | 50     | ≈ 5e-3 (≈14% of velocity loss)
joint_vel           | 500    | ≈ 1e-3 (≈4% of velocity loss) — main skating killer
fk_consistency      | 1500   | ≈ 2.1e-3 (≈7% of velocity loss) — 70× tighter than joint_pos
────────────────────┴────────┴─────────────────────────────────
timestep_squared_weighting: True → modulate by (t/1000)²
```

---

## 🚀 Quick Action: Reproducing Phase 2b Training

```bash
# Launch Phase 2b caption training (epoch 3320 + 10000)
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2b.py 8

# Or unconditioned cmean (epoch 2900 + 10000)
bash tools/dist_train.sh configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_cmean.py 8
```

### What Happens on Resume
1. Load checkpoint from phase2 (epoch 3320, caption) or uncond_046b (epoch 2900, uncond)
2. **Bundle patches null embeddings** from T2M pretrained (load_from line 129)
   - Reason: safetensors doesn't store bundle-level params, so intermediate ckpts have all-zero nulls
3. Continue training with new loss configuration
4. Save checkpoints every 10 epochs in phase2b work_dir

---

## 🔍 Where to Look for What

| Question | File | Lines |
|----------|------|-------|
| How are m2m_loss and kimodo_aux_loss created? | bundle.py | 120–129 |
| What are the 4 components in component_mean? | m2m_loss.py | 54–60, 106 |
| Why do feet skate? (and how to fix?) | kimodo_aux_loss.py | 1–48, 13–38 |
| What's the 198-dim layout? | _base_config.py | 13–14 |
| How does denormalization work? | bundle.py | 434–435 |
| What's the kinematic tree? | fk_utils.py | 29–52 |
| How to extract ft joint indices? | fk_utils.py | 40–41 |
| Phase 2b loss settings? | caption_phase2b.py | 27–53 |
| Rotation space (local vs global)? | bundle.py | 96–99 |

---

## ⚠️ Critical Notes

1. **Component_mean is fragile:** If you change `trans_dim_weight` without component_mean, translation will get buried again
2. **KIMODO and M2M fk_consistency are mutually exclusive:** Only one should be active
3. **Compute198DimPosition MUST come before LocalToGlobalRotation:** FK requires local rot
4. **Null embeddings need T2M patching:** Safetensors checkpoints lose bundle-level parameters
5. **Foot skating is a multi-term problem:** All three KIMODO losses (joint_pos, joint_vel, fk_consistency) work together

---

## 📚 Reading Order

**For understanding the system:**
1. This file (system overview)
2. Motion representation section
3. Loss weights explained section
4. CRITICAL FILES document for deep dives

**For debugging:**
1. Line reference for exact code locations
2. CRITICAL FILES for detailed logic
3. Config files for current hyperparameters

**For modifications:**
1. Understand which loss component you want to change
2. Look up its weight in configs and bundle
3. Check if it interacts with other components (e.g., component_mean + trans_dim_weight)
4. Review the instantiation flow to ensure your change propagates correctly

