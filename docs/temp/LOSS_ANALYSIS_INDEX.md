# HyMotion M2M v2 Loss Analysis — Complete Documentation Index

## 📄 Generated Documentation Files

This analysis consists of 4 comprehensive documents in `docs/`:

1. **`hymotion_m2m_v2_loss_analysis.md`** (Complete Technical Reference)
   - In-depth explanation of all 3 questions
   - Full code snippets with line numbers
   - Configuration examples
   - KIMODO auxiliary loss details
   - Complete aggregation flow

2. **`hymotion_m2m_v2_loss_quick_ref.md`** (Quick Reference Card)
   - One-page summary of key answers
   - Tables showing logged keys
   - Component structure visualization
   - Default config snippets
   - Fast lookup reference

3. **`hymotion_m2m_v2_loss_tensor_flow.md`** (Implementation Details)
   - Step-by-step tensor shape transitions
   - Detailed pathways for element_mean and component_mean
   - KIMODO auxiliary loss computation flows
   - Mask-aware loss details

4. **`LOSS_ANALYSIS_INDEX.md`** (This File)
   - Quick navigation guide
   - Exact file paths and line numbers
   - Q&A reference

---

## 🎯 Quick Answers to Your 3 Questions

### Q1: Is velocity loss single average or split?

| Answer | Location | Lines |
|--------|----------|-------|
| **Depends on `velocity_loss_reduction` parameter** | - | - |
| Element-mean: Single average over all 198 dims | `m2m_loss.py` | 71-80 |
| Component-mean: Split into 4 semantic components | `m2m_loss.py` | 82-104 |
| Component structure definition | `m2m_loss.py` | 55-60 |

### Q2: What per-component losses are logged?

| Loss Key | Source | Default Weight | Logged? | Lines |
|----------|--------|-----------------|---------|-------|
| `loss_velocity` | M2MLoss | 1.0 | ✅ YES | 147 |
| `loss_smoothness` | M2MLoss | 0.5 | ✅ YES | 205 |
| `loss_aux_joint_pos` | KimodoAuxLoss | 50.0 | ✅ YES | 296 |
| `loss_aux_joint_vel` | KimodoAuxLoss | 500.0 | ✅ YES | 315 |
| `loss_aux_fk_consistency` | KimodoAuxLoss | 1500.0 | ✅ YES | 331 |
| `loss_x1` | M2MLoss | 0.0 | ❌ NO | 159 |
| `loss_keypoints3d` | M2MLoss | 0.0 | ❌ NO | 171 |
| `loss_translation` | M2MLoss | 0.0 | ❌ NO | 181 |
| `loss_fk_consistency` | M2MLoss | 0.0 | ❌ NO | 218 |

**Logging code**: `hymotion_m2m_trainer.py:399-400`

### Q3: How is trans_dim_weight=5.0 applied?

| Aspect | Answer | Location | Lines |
|--------|--------|----------|-------|
| **What it does** | Scales dims [0:3] by 5× within velocity loss | `m2m_loss.py` | 142-149 |
| **How it works** | Element-wise multiply before reduction | `m2m_loss.py` | 145-146 |
| **Separate component?** | NO — still just one `loss_velocity` logged | - | - |
| **Applied to x1 loss too?** | YES, same mechanism | `m2m_loss.py` | 154-161 |
| **When disabled?** | When `velocity_loss_reduction='component_mean'` | Config | - |
| **Config example** | `trans_dim_weight=1.0` | `loss_component_mean/*.py` | 11 |

---

## 📍 File Structure and Line Number Reference

### Primary Loss Files

#### `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` (223 lines)
```
Lines 1-53       — Imports and M2MLoss.__init__()
Lines 54-60      — _motion_components() — Component boundaries
                   [0:3], [3:9], [9:135], [135:198]

Lines 62-104     — _masked_motion_loss()
  71-80          —   element_mean mode: single average
  82-104         —   component_mean mode: split reduction

Lines 106-222    — forward()
  142-149        —   Velocity loss with trans_dim_weight scaling
  154-161        —   X1 loss with same scaling
  171-177        —   Keypoints3D loss
  181-187        —   Translation loss
  205-207        —   Smoothness loss (frame-to-frame velocity)
  218-220        —   FK consistency loss
```

#### `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py` (334 lines)
```
Lines 1-48       — Module docstring and overview

Lines 50-122     — Utility functions
  60-62          —   _safe_std()
  65-67          —   _denormalize_198()
  70-85          —   _fk_global_positions()
  88-103         —   _scheme_d_relative()
  106-121        —   _temporal_mean_masked()

Lines 124-192    — KimodoStyleAuxLoss class definition
  156-184        —   __init__()
  194-199        —   _warmup()

Lines 201-333    — forward()
  268-277        —   FK computation on both pred and GT
  288-296        —   aux_joint_pos loss
  301-315        —   aux_joint_vel loss
  320-331        —   aux_fk_consistency loss
```

#### `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (596 lines)
```
Lines 1-48       — Imports and helper functions

Lines 49-287     — _prepare_and_forward()
                   Data loading, normalization, x_t construction

Lines 288-392    — _compute_base_loss()
  310-350        —   Velocity prediction type: compute x1 and velocities
  356-360        —   Call _compute_kimodo_aux_loss()
  361-389        —   X1 prediction type
  385-389        —   Call _compute_kimodo_aux_loss()

Lines 394-401    — train_step() ← LOGGING HAPPENS HERE
  399-400        —   for k, v in losses.items():
                      result[f'loss_{k}'] = v.detach()

Lines 403-445    — _compute_fk_keypoints()
                   Runs SMPL body model on predicted x1

Lines 447-484    — _compute_kimodo_aux_loss()
                   Calls KimodoStyleAuxLoss.forward()

Lines 486-521    — _compute_fk_consistency_loss()
                   Computes FK consistency for M2MLoss
```

### Configuration Files

#### `configs/hymotion_m2m_v2/_base_hymotion_m2m_v2_046b.py` (245 lines)
```
Lines 1-15       — File header and 198-dim layout explanation

Lines 23-136     — model dict
  58-71          —   losses_cfg with velocity_weight, trans_dim_weight, etc.
  118-127        —   kimodo_aux_loss_cfg with joint_pos_weight, etc.

Lines 138-143    — trainer dict

Lines 145-202    — train_dataloader with data pipeline

Lines 205-227    — Optimizer, scheduler, accelerator, train_cfg

Lines 229-234    — Hooks (logging, checkpoint, EMA)

Lines 236-240    — Load from T2M pretrained weights

Lines 242-245    — Val config (None)
```

#### `configs/hymotion_m2m_v2/loss_component_mean/*.py` (4 files)
- All inherit from `_base_hymotion_m2m_v2_046b.py`
- Override: `velocity_loss_reduction='component_mean'`
- Override: `trans_dim_weight=1.0`

Example: `loss_component_mean/hymotion_m2m_v2_uncond_local_046b_component_mean.py` (14 lines)
```
Lines 1-6        — _base inheritance and comment
Lines 8-13       — model dict override
  10-12          —   losses_cfg override
```

---

## 🔗 Cross-References

### Component Structure (Where Used)
- **Definition**: `m2m_loss.py:55-60`
- **Used in element_mean**: `m2m_loss.py:87`
- **Used in component_mean**: `m2m_loss.py:87`

### trans_dim_weight Parameter
- **Default value**: `_base_hymotion_m2m_v2_046b.py:64` (5.0)
- **Applied to velocity**: `m2m_loss.py:142-149`
- **Applied to x1**: `m2m_loss.py:154-161`
- **Disabled in component_mean**: `loss_component_mean/*.py:11` (1.0)

### Velocity Loss Reduction Mode
- **Parameter name**: `velocity_loss_reduction`
- **Default**: "element_mean"
- **Options**: "element_mean" or "component_mean"
- **Validation**: `m2m_loss.py:37-41`
- **Applied**: `m2m_loss.py:71` (if statement)

### Loss Logging
- **Trainer logging**: `hymotion_m2m_trainer.py:399-400`
- **Result keys**: `f'loss_{k}'` where k ∈ losses dict
- **Aggregation**: `hymotion_m2m_trainer.py:397` (sum)

### KIMODO Auxiliary Losses
- **Class definition**: `kimodo_aux_loss.py:124-192`
- **Integrated in trainer**: `hymotion_m2m_trainer.py:356-360, 385-389`
- **Configuration**: `_base_hymotion_m2m_v2_046b.py:118-127`

---

## 📚 Document Navigation

### For Learning Order
1. Start with **Quick Reference** (`hymotion_m2m_v2_loss_quick_ref.md`)
   - Get high-level overview in 5 minutes
2. Read **Tensor Flow** (`hymotion_m2m_v2_loss_tensor_flow.md`)
   - Understand implementation details
3. Study **Complete Analysis** (`hymotion_m2m_v2_loss_analysis.md`)
   - Deep dive with full code snippets

### For Problem Solving
- "What losses are being logged?" → Quick Ref Table
- "How does element_mean work?" → Tensor Flow Pathway 1
- "Why is translation scaled 5×?" → Complete Analysis Q3
- "What are KIMODO aux losses?" → Complete Analysis KIMODO section
- "How are losses combined?" → Tensor Flow Loss Dict section

### For Implementation
- "Need to change loss weights?" → Config Base file, lines 58-71 or 118-127
- "Need to change reduction mode?" → Config Loss Component Mean, line 10
- "Need to understand loss computation?" → Trainer file, lines 288-392
- "Need to debug specific loss?" → m2m_loss.py or kimodo_aux_loss.py

---

## ⚡ Key Insights

### 1. Two Loss Reduction Modes
- **element_mean** (default): Treats all 198 dims equally (after 5× trans scaling)
- **component_mean** (KIMODO): Balances 4 semantic components independently
  - Prevents 126D body rotations from dominating 3D translation

### 2. trans_dim_weight Strategy
- **Purpose**: Compensate for dimension imbalance (3 vs 195 non-trans dims)
- **Implementation**: Element-wise multiply (not separate component)
- **Effect**: Translation contribution = 5 × expected value from proportion
- **Disabled in component_mean**: Because components already balanced

### 3. KIMODO Auxiliary Losses Are Not Main Flow Loss
- Separate from M2MLoss
- Computed post-hoc via FK in world space (denormalized metres)
- Suppresses foot-slipping and intra-prediction inconsistency
- t² re-weighting down-weights pure noise early in diffusion

### 4. All Losses Eventually Aggregated
- M2MLoss produces 1-6 losses (velocity always, smoothness usually)
- KimodoAuxLoss adds 3 auxiliary losses
- Trainer sums all, logs each separately
- Final `loss = sum(all_components)`

---

## 🔍 Key Line Numbers at a Glance

| Concept | File | Line |
|---------|------|------|
| Component boundaries | m2m_loss.py | 55-60 |
| element_mean reduction | m2m_loss.py | 71-80 |
| component_mean reduction | m2m_loss.py | 82-104 |
| trans_dim_weight for velocity | m2m_loss.py | 142-149 |
| trans_dim_weight for x1 | m2m_loss.py | 154-161 |
| Velocity loss stored | m2m_loss.py | 147 |
| Smoothness loss stored | m2m_loss.py | 205 |
| FK consistency loss stored | m2m_loss.py | 218 |
| aux_joint_pos stored | kimodo_aux_loss.py | 296 |
| aux_joint_vel stored | kimodo_aux_loss.py | 315 |
| aux_fk_consistency stored | kimodo_aux_loss.py | 331 |
| KIMODO call in trainer | hymotion_m2m_trainer.py | 356-360 |
| Loss logging | hymotion_m2m_trainer.py | 399-400 |
| Default config losses | _base_hymotion_m2m_v2_046b.py | 58-71 |
| Default config KIMODO | _base_hymotion_m2m_v2_046b.py | 118-127 |
| Component-mean override | loss_component_mean/*.py | 10-12 |

