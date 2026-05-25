# Smoke Test Infrastructure Analysis & rot6d Convention Coverage

## Executive Summary

After analyzing the smoke test infrastructure and the three inference pipelines (HyMotion M2M, PRISM, VerMo), I have identified:

1. **The data flow from training to inference** - All three models train on rot6d data produced by **LoadSmplx55** transform
2. **Critical discovery in LoadSmplx55** - Lines 88-93 show HyMotion M2M uses **row-major rot6d convention** during training, NOT column-major
3. **Smoke test coverage gaps** - PRISM/VerMo configs referenced in smoke tests do NOT exist yet; only HyMotion M2M v1/v2 are actually tested
4. **Inference pipeline alignment** - All three pipelines assume column-major rot6d convention (with explicit comments stating this)
5. **The mismatch problem** - This creates a training-to-inference incompatibility that the smoke tests currently don't catch

---

## Part 1: LoadSmplx55 Transform — The Data Source

### Location
`hftrainer/datasets/motion/motionhub/transforms/load_smplx.py:218-496`

### Critical Code Section (Lines 88-93)
```python
elif rot_type == "rotation_6d":
    out = axis_angle_to_rotation_6d(aa_flat).reshape(T, J, 6)
    # axis_angle_to_rotation_6d outputs column-major: [R00,R10,R20, R01,R11,R21]
    # HyMotion convention is row-major: [R00,R01, R10,R11, R20,R21]
    # Rearrange: col_major[0,3,1,4,2,5] -> row_major
    out = out[:, :, [0, 3, 1, 4, 2, 5]]
    D = 6
```

**What this means:**
- `axis_angle_to_rotation_6d()` produces **column-major** rot6d: `[R00,R10,R20, R01,R11,R21]`
- But LoadSmplx55 then **permutes it to row-major**: `[R00,R01, R10,R11, R20,R21]` via indexing `[0,3,1,4,2,5]`
- This row-major data is what flows into HyMotion M2M training

### Usage in Configs

**HyMotion M2M (v1 & v2):**
```python
dict(
    type='LoadSmplx55',
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs_rel',
    smpl_type='smpl_22',
    transl_aug_prob=0.75,
    transl_aug_yaw_deg=180.0,
    transl_aug_offset_std=(1.0, 0.0, 1.0),
)
```
Source: `configs/hymotion_m2m/_base_hymotion_m2m_046b.py` (same for v2)

**PRISM:**
```python
dict(
    type="LoadSmplx55",
    key="motion",
    rot_type="rotation_6d",
    transl_type="abs_rel",
    smpl_type="smpl_22",
    transl_aug_prob=0.75,
    transl_aug_yaw_deg=180.0,
    transl_aug_offset_std=(1.0, 0.0, 1.0),
)
```
Source: `configs/prism/prism_1b_tp2m_1frame.py` (and other PRISM configs)

**VerMo:**
```python
dict(
    type='LoadSmplx55',
    key='motion',
    rot_type='rotation_6d',
    transl_type='abs_rel',
    smpl_type='smpl_22',
    transl_aug_prob=0.75,
    transl_aug_yaw_deg=180.0,
    transl_aug_offset_std=(1.0, 0.0, 1.0),
)
```
Source: `configs/vermo/_base_vermo_pretrain_wavtokenizer.py` (and other VerMo configs)

**Key finding:** All three models train using **LoadSmplx55 with rot_type="rotation_6d"**, which means they all train on **row-major rot6d** (after the permutation on line 93).

---

## Part 2: Smoke Test Infrastructure

### Structure
File: `tests/smoke/test_task_startup.py:172-275`

### Test Cases Defined

| Case | Config Path | Required Assets | Type |
|------|-------------|-----------------|------|
| **prism** | `configs/prism/prism_smoke.py` | tiny_tokenizer, tiny_t5_encoder, smpl_stats.json | ❌ **NOT FOUND** |
| **prism_mcm** | `configs/prism/prism_mcm_smoke.py` | tiny_tokenizer, tiny_t5_encoder, smpl_stats.json | ❌ **NOT FOUND** |
| **vermo** | `configs/vermo/vermo_smoke.py` | tiny_tokenizer, tiny_llama, smpl_stats.json | ✅ **EXISTS** |
| **hymotion_m2m** | `configs/hymotion_m2m/hymotion_m2m_smoke.py` | (none) | ✅ **EXISTS** |
| **hymotion_m2m_v2** | `configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py` | (none) | ✅ **EXISTS** |
| **hymotion_t2m** | `configs/hymotion_t2m/hymotion_t2m_smoke.py` | (none) | ✅ **EXISTS** |

### What Each Smoke Test Does

All follow the same pattern (line 279-332):
1. Load config file and apply common overrides (`_set_common_smoke_overrides`):
   - `max_iters = 1` — only 1 training iteration
   - `val_interval = 999999` — skip validation
   - Save checkpoints with `interval=1` (every step)
2. Run `tools/train.py` with the modified config → produces checkpoint
3. Find latest checkpoint
4. Run `tools/infer.py` with checkpoint → validate output file exists and is non-empty

### Customizations per Model

| Model | Customization |
|-------|---|
| PRISM | `batch_size=1, num_workers=0, mixed_precision='no'` |
| PRISM MCM | `batch_size=1, num_workers=0, mixed_precision='no'` |
| VerMo | `batch_size=1, num_workers=0, num_samples=2, tasks=['t2m'], mixed_precision='no'` |
| HyMotion M2M | `batch_size=1, num_workers=0, num_samples=2, mixed_precision='no'` |
| HyMotion M2M v2 | `batch_size=1, num_workers=0, num_samples=2, mixed_precision='no'` |
| HyMotion T2M | `batch_size=1, num_workers=0, num_samples=2, mixed_precision='no'` |

### Existing Configs (3 out of 6)

#### 1. VerMo Smoke Config (`configs/vermo/vermo_smoke.py:1-96`)
```python
smpl_pose_processor=dict(
    type='SMPLPoseProcessor',
    smpl_model=None,
    smooth_model=None,
    do_normalize=True,
    stats_file='tests/assets/motion/smpl_stats.json',  # ← Uses test stats
    rot_type='rotation_6d',
    transl_type='abs_rel',
    smpl_type='smpl_22',
)
```
Uses synthetic toy dataset (`VermoToyDataset`) — does NOT call LoadSmplx55.

#### 2. HyMotion M2M Smoke Config (`configs/hymotion_m2m/hymotion_m2m_smoke.py:1-96`)
```python
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='HyMotionM2MSyntheticDataset',
        num_samples=8,
        max_frame=16,
        motion_dim=135,  # ← Fixed 135-dim motion
        mask_ratio=0.5,
    ),
)
```
Uses synthetic dataset — does NOT call LoadSmplx55. Motion data is generated directly without any rot6d transform.

#### 3. HyMotion M2M v2 Smoke Config (`configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py:1-100`)
```python
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    shuffle=True,
    dataset=dict(
        type='HyMotionM2MSyntheticDataset',
        num_samples=8,
        max_frame=16,
        motion_dim=198,  # ← 198-dim for v2
        mask_ratio=0.5,
    ),
)
```
Uses synthetic dataset — does NOT call LoadSmplx55.

---

## Part 3: Inference Pipeline rot6d Handling

### PRISM Inference Pipeline

**File:** `hftrainer/pipelines/motion/prism_backend.py:694-706`

```python
# Line 694: Denormalize motion
x_dec = self.smpl_processor.denormalize(x_dec)

# Lines 700-706: rot6d post-processing
pred_poses = rearrange(pred_poses, "b t (j d)-> (b t) j d", d=6)
# Training data already uses column-major 6D convention [R00,R10,R20,R01,R11,R21]
# (matrix_to_rotation_6d uses _stack_cols01 → columns of rotation matrix).
# rotation_6d_to_axis_angle expects column-major input — no permutation needed.
pred_poses = rotation_6d_to_axis_angle(pred_poses)
```

**Assumption:** Column-major rot6d input (NO permutation applied before `rotation_6d_to_axis_angle`)

### VerMo Inference Pipeline

**File:** `hftrainer/pipelines/motion/vermo_backend.py:263-273`

```python
# Line 263: Denormalize motion
x_dec = self.smpl_processor.denormalize(x_dec)

# Lines 269-273: rot6d post-processing
pred_poses = rearrange(pred_poses, "p t (j d)-> (p t) j d", d=6)
# Training data already uses column-major 6D convention [R00,R10,R20,R01,R11,R21]
# (matrix_to_rotation_6d uses _stack_cols01 → columns of rotation matrix).
# rotation_6d_to_axis_angle expects column-major input — no permutation needed.
pred_poses = rotation_6d_to_axis_angle(pred_poses)
```

**Assumption:** Column-major rot6d input (NO permutation applied)

### PRISM MCM Inference Pipeline

**File:** `hftrainer/pipelines/motion/prism_mcm_pipeline.py:498-506`

```python
# Line 498: Denormalize motion
x_dec = bundle.smpl_pose_processor.denormalize(x_dec)

# Lines 503-506: rot6d post-processing
pred_poses = rearrange(pred_poses, 'b t (j d) -> (b t) j d', d=6)
# Training data already uses column-major 6D convention [R00,R10,R20,R01,R11,R21]
# (matrix_to_rotation_6d uses _stack_cols01 → columns of rotation matrix).
# rotation_6d_to_axis_angle expects column-major input — no permutation needed.
pred_poses = rotation_6d_to_axis_angle(pred_poses)
```

**Assumption:** Column-major rot6d input (NO permutation applied)

---

## Part 4: The Training-to-Inference Mismatch

### The Problem

| Stage | Convention | Source |
|-------|-----------|--------|
| **Training Data Loading** | Row-major | `LoadSmplx55` line 93: `out[:, :, [0,3,1,4,2,5]]` permutation |
| **Training Model Input** | Row-major | Direct from LoadSmplx55 (no further transformation) |
| **Model Output (at inference)** | Row-major | Model learned to output same convention as training input |
| **Inference Post-processing** | **Expects Column-major** | All three pipelines assume column-major (see comments above) |
| **Rotation conversion** | Column-major expected | `rotation_6d_to_axis_angle()` designed for column-major |

### The Risk

If you train PRISM/VerMo with row-major rot6d from LoadSmplx55, then run inference:

1. Model outputs row-major rot6d (because it learned from row-major training data)
2. Inference pipeline calls `rotation_6d_to_axis_angle(row_major_6d)` directly
3. The function **interprets row-major as if it were column-major**
4. Result: **incorrect axis-angle conversion** → wrong SMPL-X rotations

### For HyMotion M2M

HyMotion M2M uses a different post-processing path:
- Model outputs velocity in motion_135 format
- Inference doesn't call `rotation_6d_to_axis_angle()` directly
- Instead, it likely reconstructs positions through the motion_135 representation
- **But the model was trained on row-major rot6d data**, so any downstream code assuming column-major convention will break

---

## Part 5: Smoke Test Coverage Status

### Current Coverage

✅ **Caught:** HyMotion M2M (v1 & v2) training startup
- Uses synthetic data, bypasses rot6d altogether
- Cannot catch LoadSmplx55 convention errors

✅ **Caught:** VerMo training startup  
- Uses toy dataset, no real motion loading
- Cannot catch LoadSmplx55 convention errors

❌ **NOT Caught:** PRISM training → inference chain
- Configs `prism_smoke.py` and `prism_mcm_smoke.py` **do not exist**
- No smoke test validates PRISM model output → inference pipeline post-processing

❌ **NOT Caught:** VerMo training → inference chain
- VerMo training smoke test uses toy dataset (no LoadSmplx55)
- Smoke inference runs, but output isn't validated for correctness
- Cannot detect rot6d convention mismatch

❌ **NOT Caught:** rot6d convention consistency across LoadSmplx55 + inference pipelines
- Smoke tests don't validate that training rot6d matches inference assumptions
- A `rot6d_convention` parameter change in LoadSmplx55 would NOT be caught

---

## Part 6: Recommendations

### If you add `rot6d_convention` parameter to LoadSmplx55:

1. **Add default:** `rot6d_convention='row'` (to maintain backward compatibility with HyMotion M2M training)

2. **Create missing PRISM smoke configs:**
   ```python
   # configs/prism/prism_smoke.py
   _base_ = '../_base_/default_runtime.py'
   
   model = dict(
       type='PrismBundle',
       transformer=dict(...tiny config...),
       smpl_pose_processor=dict(
           rot_type='rotation_6d',
           transl_type='abs_rel',
           smpl_type='smpl_22',
       ),
   )
   
   train_dataloader = dict(
       dataset=dict(
           type='MotionHubSingleAgentTextDataset',
           pipeline=[
               dict(type='LoadSmplx55', rot_type='rotation_6d', ...),
               ...
           ],
       ),
   )
   ```

3. **Add smoke test validation:**
   ```python
   def _validate_prism_rot6d(_result, output_path: Path):
       """Verify rot6d inference produces valid SMPL poses."""
       # Load NPZ output
       data = np.load(output_path)
       motion = data['motion']  # [1, T, 22, 6] or similar
       
       # Check that rotation_6d_to_axis_angle produces valid rotations
       # (no NaNs, reasonable angle magnitudes, etc.)
       from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
           rotation_6d_to_axis_angle
       )
       poses_6d = motion[..., 6:].reshape(-1, 22, 6)
       poses_aa = rotation_6d_to_axis_angle(poses_6d)
       
       assert not np.any(np.isnan(poses_aa)), "NaN values in axis-angle output"
       assert np.all(np.abs(poses_aa) < 2*np.pi), "Axis-angle magnitudes out of range"
   ```

4. **Add rot6d convention round-trip test:**
   ```python
   def test_rot6d_convention_consistency():
       """Verify LoadSmplx55 convention matches inference pipeline."""
       # Generate test motion with LoadSmplx55
       transform = LoadSmplx55(rot_type='rotation_6d')
       
       # Round-trip through inference pipeline
       # and verify output matches input (up to model noise)
   ```

5. **Update PRISM/VerMo smoke customizations to set `rot_type` consistently:**
   ```python
   def _customize_prism(cfg: Config, has_cuda: bool):
       # Ensure LoadSmplx55 uses the same convention as inference expects
       pipeline = cfg.train_dataloader.dataset.pipeline
       for step in pipeline:
           if step.get('type') == 'LoadSmplx55':
               step['rot_type'] = 'rotation_6d'  # explicit
   ```

---

## Summary Table: What Each Pipeline Does with rot6d

| Pipeline | Training Input | Training Type | Inference Output | Inference Assumes | Smoke Test Covers |
|----------|---|---|---|---|---|
| **HyMotion M2M** | Row-major (LoadSmplx55 + permutation) | Motion-to-motion velocity | Not converted to axis-angle | N/A (stays in motion_135) | ✅ (but synthetic data) |
| **PRISM** | Row-major (LoadSmplx55 + permutation) | Text-to-motion diffusion | Model output rot6d | Column-major (explicit comment) | ❌ (config missing) |
| **VerMo** | Row-major (LoadSmplx55 + permutation) | Multi-task LM | Model output rot6d | Column-major (explicit comment) | ✅ Training only |

The mismatch: PRISM & VerMo **train on row-major** but **assume column-major at inference**.

---

## Conclusion

Adding a `rot6d_convention` parameter to LoadSmplx55 is worthwhile, but **the current smoke tests will NOT catch a mismatch** because:

1. PRISM smoke configs don't exist yet
2. VerMo/HyMotion smoke tests use synthetic data (bypass LoadSmplx55)
3. No round-trip validation between LoadSmplx55 output → inference pipeline input

To make the addition safe, you must:
- Create PRISM smoke configs with real (tiny) motion data
- Update the smoke test to validate rot6d convention consistency end-to-end
- Document the current training-to-inference mismatch and why it works (or doesn't)
