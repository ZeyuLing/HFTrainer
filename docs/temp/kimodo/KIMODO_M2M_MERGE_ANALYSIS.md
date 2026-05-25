# KimodoStyleAuxLoss ↔ M2MLoss Merge Analysis

**Date**: 2026-05-13  
**Scope**: Full interface and dependency analysis for planning a potential merge  
**Status**: Comprehensive (no implementation, planning-only)

---

## Executive Summary

### Compatibility Assessment: **MODERATE** (Merge Feasible but Requires Careful Design)

| Aspect | Status | Reason |
|--------|--------|--------|
| Param overlap | HIGH (60%+) | Both have `loss_type`, `fk_consistency_*`, `timestep_squared_weighting` |
| Data requirements | MODERATE (different) | KIMODO needs denormalization + FK; M2M already works with normalized data |
| Loss computation | LOW overlap | KIMODO is FK-based, M2M is representation-based |
| Architectural separation | EXCELLENT | Can remain independent with unified config |
| Spike detection | UNIQUE to M2M | Translation-level detection; KIMODO has no equivalent |

**Recommendation**: Do NOT do a complete merge into a single class. Instead:
1. **Unified configuration dict** (single entry point for both)
2. **Keep two separate classes** (cleaner logic flow)
3. **Shared warmup scheduler** (factor out common logic)
4. **Extend spike detection** to joint-position and joint-velocity losses

---

## 1. KimodoStyleAuxLoss — Complete Interface

### 1.1 __init__ Parameters (11 total)

```python
def __init__(
    self,
    joint_pos_weight: float = 0.0,           # FK global joint position L1
    joint_vel_weight: float = 0.0,           # FK global joint velocity L1
    fk_consistency_weight: float = 0.0,      # FK consistency (pos channels vs FK)
    loss_type: str = "smooth_l1",            # 'smooth_l1' or 'l1'
    motion_dim: int = 198,                   # Only runs if D >= 198
    fk_consistency_warmup_steps: int = 0,    # Linear warmup for consistency loss
    joint_pos_warmup_steps: int = 0,         # Linear warmup for position loss
    joint_vel_warmup_steps: int = 0,         # Linear warmup for velocity loss
    timestep_squared_weighting: bool = True, # t² reweighting (matches M2M convention)
):
```

**Initialization Logic**:
- Converts all numeric params to float/int
- Selects loss function: `F.smooth_l1_loss` or `F.l1_loss`
- No state buffers (all computation is stateless)

**`enabled` property**:
```python
@property
def enabled(self) -> bool:
    return (
        self.joint_pos_weight > 0.0
        or self.joint_vel_weight > 0.0
        or self.fk_consistency_weight > 0.0
    )
```

### 1.2 forward() Signature & Data Flow

```python
def forward(
    self,
    pred_x1_norm: Tensor,              # (B, L, 198+)
    gt_x1_norm: Tensor,                # (B, L, 198+)
    mean: Tensor,                      # (D,) normalization mean
    std: Tensor,                       # (D,) normalization std
    bone_offsets: Tensor,              # (22, 3) SMPL-22 offsets
    rotation_space: str = "local",     # 'local' or 'global' rot6d
    data_mask_temporal: Optional[Tensor] = None,  # (B, L) padding mask
    timesteps: Optional[Tensor] = None,  # (B,) diffusion timesteps [0,1]
    global_step: Optional[int] = None,   # For warmup scheduling
) -> Dict[str, Tensor]:
```

**Return Value**:
```python
{
    "aux_joint_pos": Tensor,        # (scalar) if weight > 0
    "aux_joint_vel": Tensor,        # (scalar) if weight > 0
    "aux_fk_consistency": Tensor,   # (scalar) if weight > 0
}
# Empty dict {} if not enabled or motion_dim < 198
```

### 1.3 External Data Dependencies

#### **Denormalization**
```python
def _denormalize_198(x_norm: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return x_norm * _safe_std(std) + mean
    # _safe_std: clamp std >= 1e-3 to avoid division issues
```

**Critical**: KIMODO requires **exact denormalization buffers** (mean, std). These must match training stats.

#### **Forward Kinematics**
```python
def _fk_global_positions(
    motion_135_denorm: Tensor,
    bone_offsets: Tensor,
    rotation_space: str,
) -> Tensor:
    # Calls: hftrainer.pipelines.motion.differentiable_fk.motion135_to_fk()
    # Returns: (B, L, 22, 3) world-space joint positions
```

**Key**: FK assumes **motion_135_denorm[:, :, :135]** contains:
- dims [0:3] = translation (absolute)
- dims [3:135] = rot6d (22 joints × 6 dims)

#### **Position Channel Layout (Scheme-D Relative)**
```python
def _scheme_d_relative(world_pos: Tensor) -> Tensor:
    # Converts (B, L, 22, 3) world positions → (B, L, 21, 3) relative-to-pelvis
    # Layout: body_x - pelvis_x, body_y (absolute), body_z - pelvis_z
    # Returns: (B, L, 63) flattened
```

The position channels stored in x1_norm[:, :, 135:198] are expected to follow this exact layout.

### 1.4 Internal Methods

#### **Warmup Scheduling**
```python
@staticmethod
def _warmup(weight: float, warmup_steps: int, global_step: Optional[int]) -> float:
    if weight == 0.0 or warmup_steps <= 0 or global_step is None:
        return weight
    if global_step >= warmup_steps:
        return weight
    return weight * (float(global_step) / float(warmup_steps))
```

**Convention**: Linear warmup from 0 to full weight over `warmup_steps`.

#### **Temporal Masking**
```python
def _temporal_mean_masked(per_frame: Tensor, mask: Tensor) -> Tensor:
    # Averages per-frame loss under (B, L) mask
    # Denominator clamped to 1.0 to handle all-zero masks
    m = mask.to(per_frame.device).to(per_frame.dtype)
    denom = torch.clamp(m.sum(), min=1.0)
    return (per_frame * m).sum() / denom
```

### 1.5 Loss Terms (in forward)

#### **1. joint_pos Loss (KIMODO γ₃)**
- **Input**: world_pos (B, L, 22, 3) for pred and gt
- **Reduction**: loss per-point → per-joint-xyz mean → per-frame mean → temporal mask average
- **Weighting**: `t²` optional, then warmup scheduling
- **Output key**: `"aux_joint_pos"`

#### **2. joint_vel Loss (KIMODO γ₄)**
- **Input**: velocity = world_pos[:, 1:] - world_pos[:, :-1] → (B, L-1, 22, 3)
- **Mask**: velocity valid only if **both** endpoints are valid: `vel_mask = mask[:, 1:] * mask[:, :-1]`
- **Weighting**: `t²` optional, then warmup scheduling
- **Output key**: `"aux_joint_vel"`

#### **3. fk_consistency Loss (KIMODO γ₇)**
- **Input**: 
  - Predicted pos channels: `pred_denorm[:, :, 135:198]` (B, L, 63)
  - FK-derived pos: `_scheme_d_relative(_fk_global_positions(pred_135, ...))` (B, L, 63)
- **Reduction**: loss per-dim → per-frame mean → temporal mask average
- **Output key**: `"aux_fk_consistency"`

### 1.6 Critical Assumptions

1. **motion_dim >= 198**: Pos channels at [:, :, 135:198] only exist if 198-dim
2. **rotation_space consistency**: Must match how pred_135 was created
3. **Padding awareness**: data_mask_temporal must be provided (not recommended to omit)
4. **No generation_mask**: KIMODO supervises all frames uniformly (even known regions contribute near-zero loss under MAN)

---

## 2. M2MLoss — Complete Interface

### 2.1 __init__ Parameters (14 total)

```python
def __init__(
    self,
    loss_type: str = "smooth_l1",                        # 'smooth_l1', 'l1', 'mse'
    velocity_weight: float = 1.0,                        # Velocity term weight
    x1_weight: float = 1.0,                              # x1 (position) term weight
    keypoints3d_weight: float = 1.0,                     # Keypoints3d term weight
    translation_weight: float = 1.0,                     # Translation term weight
    motion_smoothness_weight: float = 0.0,               # Motion temporal smoothness
    fk_loss_start_step: int = 0,                         # Step to enable keypoint losses
    trans_dim_weight: float = 1.0,                       # Translation dim upweighting
    trans_dims: int = 3,                                 # Num translation dims
    velocity_loss_reduction: str = "element_mean",       # 'element_mean' or 'component_mean'
    fk_consistency_weight: float = 0.0,                  # FK consistency weight (OVERLAPS!)
    fk_consistency_warmup_steps: int = 1000,             # FK consistency warmup (OVERLAPS!)
    spike_downweight_enabled: bool = True,               # Spike detection toggle
    spike_downweight_factor: float = 0.3,                # Spike downweight amount
    spike_detection_std_threshold: float = 2.0,          # Std threshold for spike
    spike_detection_window: int = 100,                   # Rolling window size
):
```

**State Initialization**:
- `self._trans_loss_history`: `deque(maxlen=spike_detection_window)` for rolling stats
- `self._baseline_trans_loss`: Running baseline (initialized 0.0)
- `self._trans_loss_std`: Running std (initialized 0.0)

### 2.2 forward() Signature

```python
def forward(
    self,
    pred_vel: Optional[Tensor] = None,              # (B, L, D) predicted velocity
    gt_vel: Optional[Tensor] = None,                # (B, L, D) ground-truth velocity
    pred_x1: Optional[Tensor] = None,               # (B, L, D) predicted x1
    gt_x1: Optional[Tensor] = None,                 # (B, L, D) ground-truth x1
    pred_keypoints3d: Optional[Tensor] = None,      # (B, L, J, 3)
    gt_keypoints3d: Optional[Tensor] = None,        # (B, L, J, 3)
    pred_translation: Optional[Tensor] = None,      # (B, L, 3)
    gt_translation: Optional[Tensor] = None,        # (B, L, 3)
    global_step: Optional[int] = None,              # For warmup scheduling
    data_mask_temporal: Tensor,                     # (B, L) padding mask [REQUIRED]
    generation_mask: Optional[Tensor] = None,       # (B, L, D) generation region mask
    fk_consistency_loss: Optional[Tensor] = None,   # Pre-computed scalar (EXTERNAL!)
) -> Dict[str, Tensor]:
```

**Return Value**: 6-14 loss terms with optional component breakdowns.

### 2.3 Spike Detection (Unique to M2M)

Applied to **translation dimensions only** (first `trans_dims` dims):
- Maintains rolling window of translation loss magnitudes
- Detects outliers: `loss > baseline + std_threshold * std`
- Downweights spike frames by `spike_downweight_factor` (typically 0.3)
- Only activates after 10 samples in history window

---

## 3. Parameter Overlap Analysis

| Parameter | KimodoStyleAuxLoss | M2MLoss | Overlap | Notes |
|-----------|-------------------|---------|---------|-------|
| `loss_type` | ✅ smooth_l1, l1 | ✅ smooth_l1, l1, mse | **100%** | KIMODO doesn't support mse |
| `fk_consistency_weight` | ✅ | ✅ | **100%** | Exactly same semantics |
| `fk_consistency_warmup_steps` | ✅ | ✅ | **100%** | Exactly same warmup convention |
| `*_warmup_steps` (per-loss) | ✅ 3 variants | ❌ only fk_consistency | **0%** | KIMODO-specific fine-grained control |
| `motion_dim` | ✅ guard (>=198) | ❌ | **0%** | KIMODO-specific |
| `timestep_squared_weighting` | ✅ | ❌ | **0%** | KIMODO-specific |
| `spike_downweight_*` | ❌ | ✅ 4 params | **0%** | M2M-specific translation anomaly detection |

**True Overlap**: ~20% (loss_type, fk_consistency × 2)  
**Potential Conflict**: loss_type must match or losses behave inconsistently

---

## 4. Data Dependencies Comparison

### KIMODO Needs (External to M2M)
- `mean, std`: normalization buffers — trainer must provide
- `bone_offsets`: SMPL-22 geometry — must be passed
- `rotation_space`: critical parameter — must match training
- Full denormalization pipeline (3 steps of math)
- FK computation: expensive, called 2-4 times per forward

### M2M Needs (Already in Trainer)
- `data_mask_temporal`: already required
- `generation_mask`: optional but supported
- `global_step`: already tracked for logging/scheduling

### Shared Computation Opportunity
If both losses enabled:
- M2M needs FK for `fk_consistency_loss` (pre-computed)
- KIMODO needs FK for `joint_pos/joint_vel` (computed internally)
- **Can reuse**: Compute FK once, cache result for both

---

## 5. Recommended Architecture: Option B (Unified Config, Separate Classes)

### 5.1 Proposed Config Structure

```python
loss_config = {
    # Shared
    "loss_type": "smooth_l1",
    
    # M2M section
    "m2m": {
        "velocity_weight": 1.0,
        "x1_weight": 1.0,
        "keypoints3d_weight": 1.0,
        "translation_weight": 1.0,
        "motion_smoothness_weight": 0.0,
        "velocity_loss_reduction": "component_mean",
        "trans_dim_weight": 1.0,
        "trans_dims": 3,
        "fk_loss_start_step": 0,
        "spike_downweight": {
            "enabled": True,
            "factor": 0.3,
            "std_threshold": 2.0,
            "window": 100,
        },
    },
    
    # Shared FK consistency
    "fk_consistency": {
        "weight": 0.0,
        "warmup_steps": 1000,
    },
    
    # KIMODO section (optional)
    "kimodo": {
        "enabled": False,
        "joint_pos_weight": 0.0,
        "joint_vel_weight": 0.0,
        "joint_pos_warmup_steps": 0,
        "joint_vel_warmup_steps": 0,
        "timestep_squared_weighting": True,
        "motion_dim": 198,
    },
}
```

### 5.2 Trainer Integration

```python
class HyMotionM2MTrainerWithKimodo(HyMotionM2MTrainer):
    def __init__(self, config):
        super().__init__(config)
        loss_cfg = config.loss_config
        
        # M2M loss (primary)
        self.m2m_loss = M2MLoss(
            loss_type=loss_cfg["loss_type"],
            velocity_weight=loss_cfg["m2m"]["velocity_weight"],
            # ... all m2m params
            fk_consistency_weight=loss_cfg["fk_consistency"]["weight"],
            fk_consistency_warmup_steps=loss_cfg["fk_consistency"]["warmup_steps"],
            **loss_cfg["m2m"]["spike_downweight"],
        )
        
        # KIMODO loss (optional)
        if loss_cfg["kimodo"]["enabled"]:
            self.kimodo_loss = KimodoStyleAuxLoss(
                loss_type=loss_cfg["loss_type"],
                joint_pos_weight=loss_cfg["kimodo"]["joint_pos_weight"],
                joint_vel_weight=loss_cfg["kimodo"]["joint_vel_weight"],
                joint_pos_warmup_steps=loss_cfg["kimodo"]["joint_pos_warmup_steps"],
                joint_vel_warmup_steps=loss_cfg["kimodo"]["joint_vel_warmup_steps"],
                fk_consistency_weight=loss_cfg["fk_consistency"]["weight"],
                fk_consistency_warmup_steps=loss_cfg["fk_consistency"]["warmup_steps"],
                timestep_squared_weighting=loss_cfg["kimodo"]["timestep_squared_weighting"],
                motion_dim=loss_cfg["kimodo"]["motion_dim"],
            )
        else:
            self.kimodo_loss = None
    
    def _compute_loss(self, batch, ctx):
        # M2M loss
        loss_dict = self.m2m_loss(
            pred_vel=ctx["pred_vel"],
            gt_vel=ctx["gt_vel"],
            pred_x1=ctx["pred_x1"],
            gt_x1=ctx["gt_x1"],
            data_mask_temporal=ctx["data_mask_temporal"],
            generation_mask=ctx.get("generation_mask"),
            fk_consistency_loss=ctx.get("fk_consistency_loss"),
            global_step=self.global_step,
        )
        
        # KIMODO loss (if enabled)
        if self.kimodo_loss is not None:
            kimodo_dict = self.kimodo_loss(
                pred_x1_norm=ctx["pred_x1"],
                gt_x1_norm=ctx["gt_x1"],
                mean=ctx["mean"],
                std=ctx["std"],
                bone_offsets=ctx["bone_offsets"],
                rotation_space=ctx["rotation_space"],
                data_mask_temporal=ctx["data_mask_temporal"],
                timesteps=ctx.get("timesteps"),
                global_step=self.global_step,
            )
            loss_dict.update(kimodo_dict)
        
        return loss_dict
```

### 5.3 Benefits of This Design

✅ **Separation of Concerns**: Each loss class owns its logic  
✅ **Minimal M2M Changes**: KIMODO is pure addition  
✅ **Easy Enable/Disable**: Set weights to 0 or `enabled: False`  
✅ **Unified Config**: Single dict for all loss params  
✅ **Clear Semantics**: No hidden parameter conflicts  
✅ **Extensible**: Easy to add more losses later (SOAR, VACE, etc.)  
✅ **Testable**: Can unit-test each loss independently  

---

## 6. Why NOT Full Merge?

### Reasons NOT to merge into single class:

1. **Fundamentally different computation paths**
   - KIMODO: denormalize → FK → world-space constraints
   - M2M: representation-space constraints on normalized dims
   - Merging these creates a 300+ line god class

2. **Parameter explosion**
   - 27 total params (14 M2M + 11 KIMODO + 2 shared)
   - Hard to reason about which params apply to which losses
   - Easy to create inconsistent configs

3. **Spike detection doesn't generalize**
   - Only makes sense for translation (M2M)
   - Can't reuse for joint positions/velocities (different distributions)
   - Would need 3 separate detectors anyway

4. **FK computation redundancy**
   - M2M needs `fk_consistency_loss` pre-computed
   - KIMODO computes FK internally
   - Merging would either compute FK twice or require awkward state management

5. **Testing complexity**
   - Single class makes it hard to test loss terms in isolation
   - Config validation becomes complex ("if kimodo enabled, then...")
   - Separate classes can be unit-tested independently

---

## 7. Implementation Checklist

- [ ] **Phase 1: Config Schema** (0.5 day)
  - [ ] Define `LossConfig` dataclass or YAML schema
  - [ ] Document all 20+ parameters
  - [ ] Add validation (e.g., loss_type must be consistent)

- [ ] **Phase 2: Trainer Integration** (0.5 day)
  - [ ] Modify trainer to instantiate both losses from config
  - [ ] Add logic to skip KIMODO if `enabled: False`
  - [ ] Merge loss dicts: `loss_dict.update(kimodo_dict)`

- [ ] **Phase 3: Context Preparation** (0.5 day)
  - [ ] Ensure trainer provides: mean, std, bone_offsets, rotation_space to ctx
  - [ ] Ensure trainer computes FK or caches world_pos
  - [ ] Pass cleaned ctx to both losses

- [ ] **Phase 4: Testing** (1 day)
  - [ ] Unit test: both losses disabled → only M2M losses
  - [ ] Unit test: only KIMODO enabled → only aux_* losses
  - [ ] Unit test: both enabled → all losses present
  - [ ] Smoke test: motion_dim < 198 → KIMODO skipped silently
  - [ ] Integration test: end-to-end training 10 steps

- [ ] **Phase 5: Documentation** (0.5 day)
  - [ ] Update CLAUDE.md with config schema and examples
  - [ ] Add section: "Enabling KIMODO losses"
  - [ ] Add known limitations and design rationale

**Total Effort**: ~3 days (non-critical path)

---

## 8. Key Questions Answered

### Q1: What params are truly duplicated?
**3 params overlap** (loss_type, fk_consistency_weight, fk_consistency_warmup_steps)  
**24 params are unique** to each loss.

### Q2: What additional data does KIMODO need that M2M doesn't?
- `mean, std` (normalization)
- `bone_offsets` (SMPL geometry)
- `rotation_space` (training convention)
- These are NOT expensive to pass, but must be correct

### Q3: Is complete merge feasible?
**NO**. Too many orthogonal concerns, parameter explosion, and computation redundancy.

### Q4: What would unified config look like?
**See §5.1**: Nested dict with m2m, kimodo, fk_consistency sections.

### Q5: Could spike detection extend to KIMODO?
**Possibly, but low priority**. Spike detection makes sense for translation (M2M paradigm). Extending to joint positions/velocities would require separate detectors and may not generalize well.

---

## 9. Conclusion

**Recommended Path**:

1. ✅ **Do NOT merge into single class**
2. ✅ **Adopt unified config dict** (single entry point)
3. ✅ **Keep M2MLoss and KimodoStyleAuxLoss separate**
4. ✅ **Extract shared utilities** (e.g., LinearWarmupScheduler)
5. ✅ **Make KIMODO optional** (weights default to 0)
6. ⏸️ **Defer spike detection extension** (future work if needed)

This provides:
- Clear separation of concerns
- Easy maintenance and testing
- Incremental adoption path
- Minimal impact on existing M2M code
- Foundation for future loss enhancements

