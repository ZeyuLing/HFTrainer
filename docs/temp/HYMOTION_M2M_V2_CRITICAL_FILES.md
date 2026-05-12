# HyMotion M2M v2 System — Critical Files Report

## Executive Summary
This report documents the critical files, configurations, and code paths for the HyMotion M2M v2 motion-to-motion editing system. The system uses a 198-dimensional motion representation and includes KIMODO-style auxiliary losses for foot-skating suppression.

---

## 1. HyMotionM2MBundle Class

**File:** `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/hymotion_m2m/bundle.py`

### Key Attributes and Methods

#### `__init__` Instantiation (Lines 60–142)
- **Registered module:** `motion_transformer` (HunyuanMotionMMDiT)
- **m2m_loss instantiation** (Lines 120–121):
  ```python
  from hftrainer.models.motion.hymotion_m2m.network.m2m_loss import M2MLoss
  self.m2m_loss = M2MLoss(**(losses_cfg or {}))
  ```
  - Accepts `losses_cfg` dict with keys:
    - `loss_type` (default: 'smooth_l1')
    - `velocity_weight` (default: 1.0)
    - `velocity_loss_reduction` ('element_mean' or 'component_mean')
    - `trans_dim_weight` (default: 1.0)
    - `motion_smoothness_weight`
    - `fk_consistency_weight`
    - etc.

- **kimodo_aux_loss instantiation** (Lines 126–129):
  ```python
  from hftrainer.models.motion.hymotion_m2m.network.kimodo_aux_loss import (
      KimodoStyleAuxLoss,
  )
  self.kimodo_aux_loss = KimodoStyleAuxLoss(**(kimodo_aux_loss_cfg or {}))
  ```
  - Accepts `kimodo_aux_loss_cfg` dict with keys:
    - `joint_pos_weight` (default: 0.0)
    - `joint_vel_weight` (default: 0.0)
    - `fk_consistency_weight` (default: 0.0)
    - `loss_type` (default: 'smooth_l1')
    - `timestep_squared_weighting` (default: False)
    - Warmup steps for each loss component

- **mean/std buffers** (Lines 149–163):
  - Loaded from `mean_std_dir`:
    - `Mean.npy` → registered as `self.mean` buffer
    - `Std.npy` → registered as `self.std` buffer
    - Std clamped to avoid div-by-zero: `std = torch.where(std < 1e-3, torch.ones_like(std), std)`

- **rotation_space attribute** (Lines 96–99):
  - Stored as `self.rotation_space`
  - Must be 'local' or 'global'
  - v2 configs use 'local'

#### `normalize_motion()` Method (Lines 481–483)
```python
def normalize_motion(self, motion: Tensor) -> Tensor:
    """Normalize motion using mean/std buffers."""
    return (motion - self.mean) / self.std
```

#### `denormalize_motion()` Method (Lines 485–488)
```python
def denormalize_motion(self, motion: Tensor) -> Tensor:
    """Denormalize motion."""
    std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
    return motion * std + self.mean
```

#### `get_bone_offsets()` Method (Lines 491–528)
- Returns `(22, 3)` tensor of bone offsets (kinematic tree rest positions)
- **Primary path:** Computes from body_model.J_template if available:
  ```python
  J_template = self.body_model.J_template[:22].clone()
  offsets = torch.zeros(22, 3, ...)
  offsets[0] = J_template[0]
  for j in range(1, 22):
      parent = SMPL22_PARENTS[j]
      offsets[j] = J_template[j] - J_template[parent]
  ```
- **Fallback path:** Loads pre-computed from `data/hymotion_m2m_data/bone_offsets_22.pt`

#### Body Model Property (Lines 166–179)
- Lazy-loads SmplxLiteJ24 model
- Called at training device inference time to compute FK for auxiliary losses

#### Key Forward Functions
- `encode_text()` (Lines 186–213): Lazy-load text encoder
- `mask_text_cond()` (Lines 215–276): Classifier-free guidance masking
- `prepare_padding()` (Lines 278–329): Pad src/tgt motions to same length
- `prepare_vace_input()` (Lines 331–384): Build VACE context (3D input)
- `predict_flow()` (Lines 386–418): Single transformer forward pass
- `decode_motion_from_latent()` (Lines 420–479): FK decode to 3D keypoints

---

## 2. Current Running Configs

### Caption Config (Latest Phase 2b)
**File:** `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2b.py`

**Loss Configuration (Lines 27–53):**
```python
losses_cfg=dict(
    loss_type='smooth_l1',
    velocity_weight=1.0,
    x1_weight=0.0,
    keypoints3d_weight=0.0,
    translation_weight=0.0,
    velocity_loss_reduction='component_mean',  # ← Key: splits into 4 components
    trans_dim_weight=1.0,                       # ← Avoids overcorrection
    motion_smoothness_weight=0.5,
    fk_consistency_weight=0.0,                  # ← Disabled; replaced by KIMODO
    fk_consistency_warmup_steps=2000,
),
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=50.0,
    joint_vel_weight=500.0,
    fk_consistency_weight=1500.0,
    loss_type='smooth_l1',
    timestep_squared_weighting=True,
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
),
```

**Training Parameters (Lines 56–107):**
- Batch size: 20
- Clip length: 360 frames
- Mask sampler: v3 Rank-K (k_weights = (0.16, 0.513, 0.233, 0.065, 0.029))
- Editing probability: 0.15
- Corruptors: jitter, joint_jump, sliding, limb_candy_wrapper, wrist_candy_wrapper
- Max corruptions: 2

**Resume Point (Line 125):**
```
work_dirs/hymotion_m2m_v2_caption_local_phase2/checkpoint-epoch_3320/model.safetensors
```

### Unconditioned Config (Latest cmean)
**File:** `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_cmean.py`

**Loss Configuration (Lines 28–51):**
- Identical to caption phase 2b **except:**
  - `uncondition_mode=True`
  - `text_encoder=None`
  - `cond_mask_prob=0.0` (no text dropout)
  - Motion smoothness weight: 0.5
  - **Same KIMODO weights as caption**

**Resume Point (Line 114):**
```
work_dirs/hymotion_m2m_v2_uncond_local_046b/checkpoint-epoch_2900/model.safetensors
```

---

## 3. Base Config Structure

**File:** `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py`

### Motion Representation (Line 21)
- **198-dimensional motion layout:**
  - `[0:3]` — translation (SMPL trans)
  - `[3:135]` — 22 joints × 6D rot6d (row-major, 132 dims)
  - `[135:198]` — 21 joints × 3D position (XZ rel pelvis, Y absolute, 63 dims)

### Model Configuration (Lines 23–52)
- **Transformer:** HunyuanMotionMMDiT
  - Input dim: 594 (= 198 × 3: x_t + reactive + mask)
  - Output dim: 198
  - 18 layers, 16 heads
  - VACE mode: 'no_inactive' (v2 slim VACE)

### Loss Configuration (Lines 58–127)
**Main M2MLoss:**
- `velocity_loss_reduction='element_mean'` (in base; overridden to 'component_mean' in v2)
- `trans_dim_weight=5.0` (in base; overridden to 1.0 in phase 2b/cmean)

**KIMODO Auxiliary Loss (Lines 118–127):**
```python
kimodo_aux_loss_cfg=dict(
    joint_pos_weight=50.0,
    joint_vel_weight=500.0,
    fk_consistency_weight=1500.0,
    loss_type='smooth_l1',
    timestep_squared_weighting=True,
    fk_consistency_warmup_steps=2000,
    joint_pos_warmup_steps=2000,
    joint_vel_warmup_steps=2000,
)
```

### Data Pipeline (Lines 157–197)
1. `LoadCompatibleCaption` — Load caption or set None
2. `LoadSmplx55` — Load motion as rot6d + absolute translation, smpl_22 format
3. **`Compute198DimPosition`** — **CRITICAL:** Computes position channels via FK (must come BEFORE LocalToGlobalRotation)
4. `RandomCropPadding` — Clip to 360 frames, pad with replicate mode
5. `PrepareM2Mv2Condition` — Sample corruptions (v3 Rank-K sampler)
6. `PackInputs` — Pack into batch dict

---

## 4. SMPL-22 Kinematic Tree

**File:** `hftrainer/datasets/motion/motionhub/transforms/fk_utils.py` (Lines 28–52)

### SMPL22_PARENTS List
```python
SMPL22_PARENTS: List[int] = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip
    0,   # 2: R_Hip
    0,   # 3: Spine1
    1,   # 4: L_Knee
    2,   # 5: R_Knee
    3,   # 6: Spine2
    4,   # 7: L_Ankle
    5,   # 8: R_Ankle
    6,   # 9: Spine3
    7,   # 10: L_Foot
    8,   # 11: R_Foot
    9,   # 12: Neck
    9,   # 13: L_Collar
    9,   # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
]
```

### Key Joint Indices
- **Feet (skating targets):**
  - Joint 10: L_Foot
  - Joint 11: R_Foot
- **Ankles (FK chain parents):**
  - Joint 7: L_Ankle (parent of L_Foot)
  - Joint 8: R_Ankle (parent of R_Foot)

### Rotation Convention Notes (Lines 6–18)
- **Row-major rot6d:** `[R00, R01, R10, R11, R20, R21]` (used in training data)
- **Column-major rot6d:** Used in `rotation_convert.py` (different convention)
- **Bundle decoding** uses row-major natively via `geometry.py` functions
- Conversion path in FK utils: row-major ↔ col-major ↔ matrix ↔ FK/IK

---

## 5. Loss Functions & Auxiliary Losses

### M2MLoss Class
**File:** `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` (Lines 1–200+)

**Key Methods:**
- `_motion_components()` (Lines 54–60): Defines semantic component ranges:
  - For 198-dim: `((0, 3), (3, 9), (9, 135), (135, 198))`
    - trans (3), root_rot (6), body_rot (126), joint_pos (63)
  - For 135-dim: `((0, 3), (3, 9), (9, 135))`
  - For <135: `((0, dim),)`

- `_masked_motion_loss()` (Lines 62–104): Applies masking with optional component_mean reduction
- `_masked_motion_loss_with_components()` (Lines 108–142): Same but returns per-component logs

**Component names** (Line 106):
```python
_COMP_NAMES = ('trans', 'root_rot', 'body_rot', 'joint_pos')
```

### KimodoStyleAuxLoss Class
**File:** `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py` (Lines 1–150+)

**Key Functions:**
- `_fk_global_positions()` (Lines 70–85): Runs FK to get (B, L, 22, 3) world-space joints
- `_scheme_d_relative()` (Lines 88–103): Converts world pos to 198-dim layout (XZ rel-pelvis, Y absolute)
- `_temporal_mean_masked()` (Lines 106–121): Averages per-frame losses under mask

**Three Loss Terms:**
1. **joint_pos (γ₃):** Smooth-L1 on global joint positions vs GT → suppresses pelvis cheating
2. **joint_vel (γ₄):** Smooth-L1 on global joint velocities → suppresses foot skating
3. **fk_consistency (γ₇):** Smooth-L1 on pos-channel ↔ FK(pred_rot/trans) consistency → teaches explicit FK equivalence inside 198-dim

---

## 6. Critical Config Differences: Phase 2b vs Base

| Parameter | Base (046b) | Phase 2b | Reason |
|-----------|-----------|---------|--------|
| `velocity_loss_reduction` | 'element_mean' | 'component_mean' | Equal 25% weight to 4 semantic components |
| `trans_dim_weight` | 5.0 | 1.0 | Avoid overcorrection under component_mean |
| All KIMODO weights | Same (50/500/1500) | Same (50/500/1500) | Proven weight ratios |
| Motion smoothness weight | 0.5 | 0.5 | Smoothness regularization enabled |
| Cond mask prob (caption) | — | 0.1 | 10% text dropout for robustness |
| Batch size (caption) | 28 | 20 | Memory; V100 limit with caption embeddings |

---

## 7. Motion Representation Details

### 198-Dimensional Layout
```
Offset Range | Channels | Meaning
[0:3]        | 3        | Translation (world-space, meters)
[3:135]      | 132      | 22 joints × 6D rot6d (row-major)
[135:198]    | 63       | 21 joints × 3D position
             |          | - [135:198] packed as (j, xyz) = (j, x, y, z)
             |          | - X, Z relative to pelvis (XZ plane)
             |          | - Y absolute (world frame)
             |          | - Pelvis joint (j=0) excluded
```

### Normalization
- Each dimension normalized by per-dim mean/std from training distribution
- Stored in `data/hymotion_m2m_data/_stats_198dim/Mean.npy` and `Std.npy`
- **Denormalization in decode** (bundle.py line 434–435):
  ```python
  latent_denorm = latent * std + self.mean
  ```

### FK Consistency Semantics
- The 63-dim position channels are **supposed to** satisfy:
  ```
  FK(rot6d[:, :, 1:], trans) → world_pos
  scheme_d_relative(world_pos) ≡ position_channels[135:198]
  ```
- **fk_consistency loss** enforces this intra-prediction equivalence
- Allows position-only edits at inference (hand trajectory, end-effector constraints) without IK

---

## 8. File Paths Summary

| Purpose | File Path |
|---------|-----------|
| Bundle class | `hftrainer/models/motion/hymotion_m2m/bundle.py` |
| M2M Loss | `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` |
| KIMODO Aux Loss | `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py` |
| FK/IK Utils | `hftrainer/datasets/motion/motionhub/transforms/fk_utils.py` |
| Caption Phase 2b Config | `configs/hymotion_m2m_v2/hymotion_m2m_v2_caption_local_phase2b.py` |
| Uncond cmean Config | `configs/hymotion_m2m_v2/hymotion_m2m_v2_uncond_local_cmean.py` |
| Base Config (v2 0.46B) | `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` |
| Bone offsets (fallback) | `data/hymotion_m2m_data/bone_offsets_22.pt` |
| Mean/Std normalization | `data/hymotion_m2m_data/_stats_198dim/Mean.npy` / `Std.npy` |

---

## 9. Key Takeaways for v2 System

1. **Motion is 198-dim:** trans(3) + rot6d×22(132) + pos×21(63)
2. **Three loss tracks:**
   - **M2MLoss (velocity):** Main gradient signal with component_mean reduction
   - **KimodoAuxLoss (joint_pos/vel/fk_consistency):** Foot-skating suppression via world-space supervision
   - Both run post-hoc during training, weights configured separately

3. **Rotation space:** Local SMPL (not global) — `rotation_space='local'`
4. **VACE mode:** 'no_inactive' (v2 slim) → model input = x_t + reactive + mask = 3×198 = 594-dim
5. **Component_mean is critical:** Prevents body rotations (126-dim) from drowning out translation (3-dim)
6. **FK consistency teaches position-rotation equivalence:** Enables position-only editing at inference

