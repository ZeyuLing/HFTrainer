# HyMotion M2M v2 — Quick Reference: Exact Line Numbers

## Bundle File: hftrainer/models/motion/hymotion_m2m/bundle.py

| Element | Lines | Key Detail |
|---------|-------|-----------|
| Class definition | 51 | `class HyMotionM2MBundle(ModelBundle)` |
| `__init__` signature | 60–84 | Full init with all params |
| Motion transformer build | 88 | `self._build_modules({'motion_transformer': motion_transformer})` |
| Rotation space validation | 96–99 | Assert local/global check |
| **m2m_loss instantiation** | **120–121** | `self.m2m_loss = M2MLoss(**(losses_cfg or {}))` |
| **kimodo_aux_loss instantiation** | **126–129** | `self.kimodo_aux_loss = KimodoStyleAuxLoss(**(kimodo_aux_loss_cfg or {}))` |
| _load_mean_std method | 149–163 | Load/register mean/std buffers |
| mean buffer registration | 159 | `self.register_buffer('mean', mean)` |
| std buffer registration | 160 | `self.register_buffer('std', std)` |
| body_model property | 166–179 | Lazy-load SmplxLiteJ24 |
| encode_text method | 186–213 | Text encoding with lazy loading |
| mask_text_cond method | 215–276 | CFG masking |
| prepare_padding method | 278–329 | Padding alignment |
| prepare_vace_input method | 331–384 | VACE context construction |
| predict_flow method | 386–418 | Single transformer forward |
| decode_motion_from_latent method | 420–479 | FK decode (CRITICAL) |
| normalize_motion method | 481–483 | (motion - mean) / std |
| denormalize_motion method | 485–488 | motion * std + mean |
| **get_bone_offsets method** | **491–528** | Bone offset computation |

---

## Config Files

### Caption Phase 2b: hymotion_m2m_v2_caption_local_phase2b.py

| Element | Lines | Value |
|---------|-------|-------|
| Base config inheritance | 17 | `_base_ = './_base_hymotion_m2m_v2_046b.py'` |
| Work dir | 19 | `'work_dirs/hymotion_m2m_v2_caption_local_phase2b'` |
| M2M loss config | 27–41 | `dict(...)` |
| **velocity_loss_reduction** | **35** | **'component_mean'** |
| **trans_dim_weight** | **36** | **1.0** |
| **KIMODO aux loss config** | **44–53** | Full KIMODO config |
| **joint_pos_weight** | **45** | **50.0** |
| **joint_vel_weight** | **46** | **500.0** |
| **fk_consistency_weight** | **47** | **1500.0** |
| **timestep_squared_weighting** | **49** | **True** |
| Batch size | 57 | 20 |
| Clip length | 71 | 360 |
| Mask sampler version | 83 | 'v3' |
| v3 k_weights | 91 | (0.16, 0.513, 0.233, 0.065, 0.029) |
| Mask aware noise | 112 | True |
| Resume checkpoint | 125 | epoch_3320/model.safetensors |

### Uncond cmean: hymotion_m2m_v2_uncond_local_cmean.py

| Element | Lines | Value |
|---------|-------|-------|
| Base config inheritance | 17 | `_base_ = './_base_hymotion_m2m_v2_046b.py'` |
| uncondition_mode | 23 | True |
| text_encoder | 24 | None |
| cond_mask_prob | 25 | 0.0 |
| **KIMODO aux loss config** | **42–51** | Identical to caption |
| Mask sampler version | 77 | 'v3' |
| Resume checkpoint | 114 | epoch_2900/model.safetensors |

### Base Config: _base_hymotion_m2m_v2_046b.py

| Element | Lines | Value/Description |
|---------|-------|-----------|
| Motion dim constant | 21 | `_motion_dim = 198` |
| Motion layout comment | 13–14 | [0:3] trans, [3:135] rot6d, [135:198] pos |
| Input dim to transformer | 32 | 594 (= 198 × 3) |
| Output dim from transformer | 34 | 198 |
| Transformer type | 26 | HunyuanMotionMMDiT |
| Num layers | 37 | 18 |
| Num heads | 38 | 16 |
| VACE mode | 132 | 'no_inactive' |
| Mean/std path | 54 | 'data/hymotion_m2m_data/_stats_198dim' |
| **M2MLoss base config** | **58–71** | velocity_loss_reduction='element_mean', trans_dim_weight=5.0 |
| **KIMODO aux loss base** | **118–127** | Same weights: 50/500/1500 |
| **Compute198DimPosition** | **168** | **CRITICAL pipeline transform** |
| Data pipeline comment | 166–167 | Must come BEFORE LocalToGlobalRotation |
| Batch size (base) | 146 | 28 |
| Clip length | 171 | 360 frames |
| Load T2M pretrained | 237–239 | 'checkpoints/HY-Motion-1.0/...' |

---

## Loss Files

### m2m_loss.py

| Element | Lines | Description |
|---------|-------|-----------|
| M2MLoss class def | 8 | `class M2MLoss(nn.Module)` |
| __init__ params | 9–23 | Full parameter list |
| velocity_loss_reduction options | 37–41 | Assert 'element_mean' or 'component_mean' |
| _motion_components method | 54–60 | Component ranges for different dims |
| **For 198-dim** | **57** | **((0,3), (3,9), (9,135), (135,198))** |
| **trans, root_rot, body_rot, joint_pos ranges** | **57** | **3, 6, 126, 63 dims** |
| _masked_motion_loss method | 62–104 | Main loss reduction logic |
| _masked_motion_loss_with_components | 108–142 | Same but returns per-component logs |
| Component names | 106 | ('trans', 'root_rot', 'body_rot', 'joint_pos') |
| forward method | 144–237+ | Full loss computation |

### kimodo_aux_loss.py

| Element | Lines | Description |
|---------|-------|-----------|
| Purpose docstring | 1–48 | Foot skating suppression via 3 aux losses |
| _fk_global_positions function | 70–85 | FK → (B,L,22,3) world positions |
| _scheme_d_relative function | 88–103 | World pos → 198-dim layout |
| _temporal_mean_masked function | 106–121 | Per-frame loss averaging under mask |
| KimodoStyleAuxLoss class | 124 | `class KimodoStyleAuxLoss(nn.Module)` |
| Three loss terms doc | 13–38 | joint_pos (γ₃), joint_vel (γ₄), fk_consistency (γ₇) |

---

## FK Utils: fk_utils.py

| Element | Lines | Description |
|---------|-------|-----------|
| Docstring notes | 1–18 | Row-major vs column-major rot6d conventions |
| **SMPL22_PARENTS definition** | **29–52** | Full 22-joint parent array |
| **Pelvis** | **30** | 0: -1 (root) |
| **L_Foot** | **40** | 10: 7 (parent = L_Ankle) |
| **R_Foot** | **41** | 11: 8 (parent = R_Ankle) |
| NUM_JOINTS constant | 54 | 22 |
| _ROW_TO_COL order | 61 | [0, 2, 4, 1, 3, 5] |
| _COL_TO_ROW order | 62 | [0, 3, 1, 4, 2, 5] |
| local_to_global_rot6d numpy | 83–108 | Numpy FK (dataset transforms) |
| global_to_local_rot6d numpy | 111–136 | Numpy IFK |
| global_to_local_rot6d_torch | 144–176 | Torch IFK (inference) |
| local_to_global_rot6d_torch | 179–208 | Torch FK (inference) |

---

## Key Instantiation Sequences

### HyMotionM2MBundle Initialization
```
__init__ (line 60)
  ↓ _build_modules (line 88)
  ↓ _load_mean_std (line 117) → self.mean, self.std buffers
  ↓ M2MLoss (line 121) → self.m2m_loss
  ↓ KimodoStyleAuxLoss (line 129) → self.kimodo_aux_loss
  ↓ body_model lazy-load (line 166) → SmplxLiteJ24
```

### Loss Weight Chain (Phase 2b)
```
Config losses_cfg:
  velocity_loss_reduction = 'component_mean' (line 35)
  trans_dim_weight = 1.0 (line 36)
  motion_smoothness_weight = 0.5 (line 37)
  fk_consistency_weight = 0.0 (line 39) — DISABLED

Config kimodo_aux_loss_cfg:
  joint_pos_weight = 50.0 (line 45)
  joint_vel_weight = 500.0 (line 46)
  fk_consistency_weight = 1500.0 (line 47)
  timestep_squared_weighting = True (line 49)
```

### Motion 198-Dim Breakdown
```
[0:3]       → translation (3)
[3:9]       → root rot6d (6)
[9:135]     → body rot6d (126) = 21 joints × 6
[135:198]   → joint positions (63) = 21 joints × 3
Total: 3 + 6 + 126 + 63 = 198
```

