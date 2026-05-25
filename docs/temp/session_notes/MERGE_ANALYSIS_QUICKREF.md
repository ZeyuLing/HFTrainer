# KimodoStyleAuxLoss ↔ M2MLoss — Quick Reference (TL;DR)

**Status**: Analysis complete (PLANNING ONLY, no implementation)  
**Recommendation**: Keep separate classes, use unified config dict  
**Est. Implementation Time**: 3 days (non-critical path)

---

## ⚠️ Core Finding

**DO NOT merge into single class.** Reasons:
1. Fundamentally different loss computation (FK-based vs representation-based)
2. Parameter explosion (27 total params)
3. Spike detection doesn't generalize
4. FK redundancy (both need it, computed differently)
5. Would create unmaintainable 300+ line god class

---

## 📊 Parameter Comparison at a Glance

| Category | KimodoStyleAuxLoss | M2MLoss | Overlap |
|----------|-------------------|---------|---------|
| `loss_type` | smooth_l1, l1 | smooth_l1, l1, mse | ✅ 100% |
| `fk_consistency_weight` | ✅ | ✅ | ✅ 100% |
| `fk_consistency_warmup_steps` | ✅ | ✅ | ✅ 100% |
| **Joint-level warmups** | 3 variants | ❌ | ❌ 0% |
| **Motion_dim guard** | ✅ | ❌ | ❌ 0% |
| **t² reweighting** | ✅ | ❌ | ❌ 0% |
| **Spike detection** | ❌ | ✅ (4 params) | ❌ 0% |
| **Per-component reduction** | ❌ | ✅ | ❌ 0% |
| **TOTAL OVERLAP** | | | ~20% |

---

## 🔧 Proposed Design: Option B (RECOMMENDED)

### Architecture
```
Single Config Dict
    ├── m2m: { velocity, x1, keypoints3d, translation, smoothness, spike_detect }
    ├── fk_consistency: { weight, warmup_steps }
    └── kimodo: { joint_pos, joint_vel, t²_weight, motion_dim, warmup_steps }

Trainer
    ├── self.m2m_loss = M2MLoss(**config.m2m + config.fk_consistency + config.loss_type)
    └── self.kimodo_loss = KimodoStyleAuxLoss(**config.kimodo + config.fk_consistency + config.loss_type)
        (or None if disabled)
```

### Config Example
```yaml
loss_config:
  loss_type: "smooth_l1"
  
  m2m:
    velocity_weight: 1.0
    x1_weight: 1.0
    # ... 8 more params ...
    spike_downweight: { enabled: true, factor: 0.3, ... }
  
  fk_consistency:
    weight: 0.0
    warmup_steps: 1000
  
  kimodo:
    enabled: false  # Set to true to enable
    joint_pos_weight: 0.0
    joint_vel_weight: 0.0
    # ... 4 more params ...
```

**Benefits**:
- ✅ Zero impact on existing M2M code
- ✅ Single config entry point
- ✅ Easy to enable/disable KIMODO (weights=0 or enabled=false)
- ✅ Each loss owns its logic (no entanglement)
- ✅ Testable independently

---

## 📋 KIMODO External Data Needs

**Required by forward()**:
```python
# Denormalization
mean: Tensor              # (D,) — from trainer
std: Tensor               # (D,) — from trainer

# Geometry
bone_offsets: Tensor      # (22, 3) — SMPL-22 offsets
rotation_space: str       # 'local' or 'global'

# Masks & Scheduling
data_mask_temporal: Tensor        # (B, L) — padding mask
timesteps: Optional[Tensor]       # (B,) — for t² reweighting
global_step: Optional[int]        # for warmup

# Input motion (normalized!)
pred_x1_norm, gt_x1_norm: Tensor  # (B, L, 198+)
```

**Computation Pipeline**:
```
Denorm(x1_norm, mean, std)
    → Extract [0:135] (translation + rot6d)
    → FK (bone_offsets, rotation_space)
    → (B, L, 22, 3) world positions
    → 3 losses: joint_pos, joint_vel, fk_consistency
```

---

## 🎯 What Truly Overlaps?

### Shared Parameters (3)
1. **loss_type**: smooth_l1/l1 (KIMODO doesn't support mse)
2. **fk_consistency_weight**: exact same semantics
3. **fk_consistency_warmup_steps**: exact same warmup convention

### Unique to KIMODO (8)
- joint_pos_weight, joint_vel_weight
- joint_pos_warmup_steps, joint_vel_warmup_steps
- fk_consistency_warmup_steps (already listed as shared!)
- motion_dim (198 guard)
- timestep_squared_weighting (t² reweighting)

### Unique to M2M (14)
- velocity_weight, x1_weight, keypoints3d_weight, translation_weight
- motion_smoothness_weight
- fk_loss_start_step
- trans_dim_weight, trans_dims
- velocity_loss_reduction
- spike_downweight_enabled, spike_downweight_factor
- spike_detection_std_threshold, spike_detection_window

---

## 💥 Why Spike Detection Doesn't Generalize

**M2M Spike Detection** (translation-level):
- Detects outlier translation loss frames
- Applicable to: velocity_trans, x1_trans, translation losses
- Not applicable to: rotation, joint position (different scale/distribution)

**Would need per-loss detectors for KIMODO**:
- Separate spike detection for joint_pos
- Separate spike detection for joint_vel
- But: M2M's window-based rolling stats assume "close distribution"
- Translation loss and joint position loss have very different scales and noise characteristics

**Verdict**: Don't try to use same detector for all losses. Keep per-loss or disable for KIMODO.

---

## 🔄 FK Computation Opportunity

**If both losses enabled**:
- M2M needs FK for `fk_consistency_loss` (pre-computed by trainer)
- KIMODO needs FK for `joint_pos/joint_vel` (computed internally)

**Could optimize**:
```python
# Trainer computes once
world_pos_pred = fk(pred_135)  # (B, L, 22, 3)
world_pos_gt = fk(gt_135)
ctx["world_pos_pred"] = world_pos_pred
ctx["world_pos_gt"] = world_pos_gt

# M2M uses for consistency loss
ctx["fk_consistency_loss"] = compute_fk_consistency(world_pos_pred, world_pos_gt)

# KIMODO reuses instead of recomputing
kimodo_loss(world_pos_pred=ctx["world_pos_pred"], ...)
```

---

## 📝 Implementation Roadmap

| Phase | Task | Duration | Priority |
|-------|------|----------|----------|
| 1 | Config schema definition (YAML/dataclass) | 0.5 day | 🔴 High |
| 2 | Trainer instantiation (both losses) | 0.5 day | 🔴 High |
| 3 | Context preparation (provide mean/std/bone_offsets) | 0.5 day | 🔴 High |
| 4 | Unit tests + smoke tests | 1 day | 🟡 Medium |
| 5 | Documentation + examples | 0.5 day | 🟢 Low |
| **TOTAL** | | **3 days** | |

---

## ✅ Checklist for Future Implementation

- [ ] Define `LossConfig` dataclass with m2m/kimodo/fk_consistency sections
- [ ] Add config validation (e.g., loss_type consistency)
- [ ] Modify trainer `__init__` to instantiate both losses (with enable/disable logic)
- [ ] Modify trainer `_compute_loss` to call both losses and merge dicts
- [ ] Ensure ctx contains: mean, std, bone_offsets, rotation_space
- [ ] Optional: cache FK results if both losses enabled
- [ ] Unit tests: each loss separately, both enabled, both disabled
- [ ] Integration test: 10-step training loop
- [ ] Update CLAUDE.md with config schema and usage examples

---

## 🚫 Common Pitfalls to Avoid

| Pitfall | Risk | Mitigation |
|---------|------|-----------|
| loss_type mismatch (m2m=smooth_l1, kimodo=l1) | 🔴 High | Config validation, single loss_type param |
| Mean/std not provided to KIMODO | 🔴 High | Assert in forward(), provide in ctx |
| Rotation space mismatch | 🔴 High | Store in ctx, validate consistency |
| Spike detection applied to joint losses | 🟡 Medium | Keep spike detection translation-only |
| FK computed twice | 🟡 Medium | Cache world_pos in ctx if both enabled |
| motion_dim < 198 not handled | 🟡 Medium | KIMODO already has `enabled` property, use it |
| Wrong position channel layout (Scheme-D) | 🔴 High | Document in CLAUDE.md, no changes needed (handled by KIMODO) |

---

## 📚 Reference Files

- **Full Analysis**: `KIMODO_M2M_MERGE_ANALYSIS.md` (512 lines)
  - Section 1: KimodoStyleAuxLoss complete interface
  - Section 2: M2MLoss complete interface
  - Section 3: Detailed comparison matrix
  - Section 4: Feasibility analysis
  - Section 5: Recommended unified config structure
  - Sections 6-9: Design rationale, questions answered, conclusion

- **KIMODO Source**: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`
  - 334 lines, 3 loss terms, internal FK computation

- **M2M Source**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py`
  - 348 lines, 6+ loss terms, external FK dependency

- **M2M CLAUDE.md**: Section "Motion Representation" and "Loss Types"

