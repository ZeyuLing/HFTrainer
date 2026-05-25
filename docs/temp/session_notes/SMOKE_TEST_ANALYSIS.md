# Smoke Test Infrastructure Analysis for HyMotion M2M, PRISM, and VerMo

## Executive Summary

The smoke test infrastructure (`tests/smoke/test_task_startup.py`) follows a **parametrized approach** with 6 test cases. It exercises training for **1 iteration** followed by **inference**, validating that model architecture and dataset loading work end-to-end. 

### Key Finding: Synthetic vs. Real Data Pipelines
- **Smoke tests use SYNTHETIC datasets** (random motion generation) — they do **NOT use LoadSmplx55**
- **Real training uses LoadSmplx55** with real SMPL-X files and dataset pipelines
- This creates a **gap** in detecting issues specific to LoadSmplx55 behavior

---

## Test Structure

### Test File Location
- **File:** `tests/smoke/test_task_startup.py` (333 lines)
- **Test Function:** `test_train_and_infer_startup()` (parametrized)
- **Decorator:** `@pytest.mark.parametrize('case', SMOKE_CASES)`

### Test Cases (6 total)

| Case ID | Config | Dataset Type | Dataset Class | rot_type | Motion Dim | Real Data? |
|---------|--------|--------------|----------------|----------|------------|-----------|
| `prism` | `configs/prism/prism_smoke.py` | Toy | **MISSING** | - | - | ❌ No config exists |
| `prism_mcm` | `configs/prism/prism_mcm_smoke.py` | Toy | **MISSING** | - | - | ❌ No config exists |
| `vermo` | `configs/vermo/vermo_smoke.py` | Toy | `VermoToyDataset` | rot_dim=6 | 22*6+6=138 | ❌ Synthetic |
| `hymotion_m2m` | `configs/hymotion_m2m/hymotion_m2m_smoke.py` | Synthetic | `HyMotionM2MSyntheticDataset` | - | 135 | ❌ Synthetic |
| `hymotion_m2m_v2` | `configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py` | Synthetic | `HyMotionM2MSyntheticDataset` | - | 198 | ❌ Synthetic |
| `hymotion_t2m` | `configs/hymotion_t2m/hymotion_t2m_smoke.py` | Synthetic | ? | - | ? | ❌ Synthetic |

**Status:**
- ✅ 4 configs exist: vermo, hymotion_m2m, hymotion_m2m_v2, hymotion_t2m
- ❌ 2 configs MISSING: prism_smoke.py, prism_mcm_smoke.py (test will skip if not found)

---

## Test Execution Flow

```
test_train_and_infer_startup()
    ├─ Load config from disk
    ├─ Set common smoke overrides
    │  ├─ max_iters = 1 (train for exactly 1 iteration)
    │  ├─ val_interval = 999999 (no validation)
    │  ├─ checkpoint save at interval=1
    │  ├─ batch_size → 1 (customized per case)
    │  ├─ num_workers = 0 (sync loading)
    │  └─ mixed_precision = 'no'
    │
    ├─ Run customize_cfg() (case-specific customizations)
    │  └─ HyMotion M2M: num_samples=2, batch_size=1
    │  └─ VerMo: tasks=['t2m'], num_samples=2
    │  └─ PRISM: (would customize if config existed)
    │
    ├─ TRAIN PHASE (timeout: 900s per case)
    │  ├─ Execute: tools/train.py config.py
    │  ├─ Load dataset (synthetic, random data)
    │  ├─ Run 1 training iteration
    │  ├─ Save checkpoint
    │  └─ Assert checkpoint exists and is non-empty
    │
    └─ INFER PHASE (timeout: 900s per case)
       ├─ Load checkpoint
       ├─ Execute: tools/infer.py --config --checkpoint + task-specific args
       ├─ Generate motion from learned model (minimal)
       └─ Assert output file created and non-empty (.npz or .txt)
```

---

## Dataset Pipeline Analysis

### Synthetic Datasets (Used in Smoke Tests)

#### 1. **HyMotionM2MSyntheticDataset** (135-dim and 198-dim)
**Location:** `hftrainer/datasets/motion/hymotion_m2m_dataset.py`

```python
class HyMotionM2MSyntheticDataset(Dataset):
    """Generates random src/tgt motion pairs with proper shapes and masks."""
    
    def __getitem__(self, idx):
        src_motion = torch.randn(L, D)      # Random [16, 135] or [16, 198]
        tgt_motion = torch.randn(L, D)
        src_mask = torch.zeros(L, D)
        src_mask[split:] = 1.0              # Mask 50% for generation
        
        return {
            'src_motion': src_motion,
            'tgt_motion': tgt_motion,
            'src_mask': src_mask,
            'tgt_length': L,
            'src_length': L,
        }
```

**Key Point:** No LoadSmplx55 involved; random Gaussian tensors.

#### 2. **VermoToyDataset** (22*6+6=138-dim, rotation_6d)
**Location:** `hftrainer/datasets/motion/vermo_toy_dataset.py`

```python
class VermoToyDataset(PipelineDataset):
    def __init__(self, num_samples=8, num_frames=17, num_joints=22, rot_dim=6):
        self.motion = torch.randn(
            num_samples, num_frames, 
            num_joints * rot_dim + 6,  # = 138-dim
            generator=torch.Generator().manual_seed(seed),
        )
```

**Key Point:** 
- Pre-generated random motion in rotation_6d format (never goes through LoadSmplx55)
- `rot_dim=6` is hardcoded; this ALWAYS uses 6D rotation, not row/column-major variant

---

### Real Training Datasets (NOT Used in Smoke Tests)

#### PRISM, VerMo (Production)
**Class:** `MotionhubMultiTaskMultiAgentDataset`
**Pipeline Stages:**
```python
pipeline=[
    dict(type="LoadHierarchicalCaption", allow_none=True),
    dict(
        type="LoadSmplx55",           # ← LOADS REAL SMPL-X FILES
        key="motion",
        rot_type="rotation_6d",
        transl_type="abs_rel",
        smpl_type="smpl_22",
    ),
    dict(type="LoadAudio", key="audio", target_sr=16000, allow_none=True),
    dict(type="RandomCropPadding", clip_len=512),
    dict(type="PackInputs", keys=["motion", "num_frames", "caption", "audio"]),
]
```

#### HyMotion M2M (Production)
**Class:** `MultiTaskM2MDataset`
**Pipeline:** Uses `LoadSmplx55` to read SMPL-X .npz files

---

## LoadSmplx55 Implementation Details

**File:** `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` (496 lines)

### Function: `process_smplx_pose()` (Lines 16–104)

This is the **critical function** for your `rot6d_convention` parameter:

```python
def process_smplx_pose(
    pose_55_axis_angle: np.ndarray,  # [T, 165] or [T, 55, 3]
    rot_type: str,                    # "axis_angle" | "rotation_6d" | "quaternion" | "euler"
    out_type: str,                    # "smpl_22" | "smplh" | "smplx_55"
) -> np.ndarray:
    """Convert SMPL-X 55-joint axis-angle to target rotation representation."""
    
    # ... reshape to [T, 55, 3] ...
    
    if rot_type == "rotation_6d":
        out = axis_angle_to_rotation_6d(aa_flat).reshape(T, J, 6)
        # ⚠️ CURRENT BEHAVIOR: Column-major → Row-major conversion (lines 90–93)
        out = out[:, :, [0, 3, 1, 4, 2, 5]]  # HARDCODED!
        D = 6
```

### Current Behavior (Hardcoded)
- Line 89: `axis_angle_to_rotation_6d()` outputs **column-major**: `[R00, R10, R20, R01, R11, R21]`
- Lines 92–93: Rearrange to **row-major**: `[R00, R01, R10, R11, R20, R21]`
- **No parameter control; always row-major after conversion**

### Your Proposed `rot6d_convention` Parameter
Would need to:
1. Add `rot6d_convention: str = "row"` parameter to `LoadSmplx55.__init__()`
2. Conditionally apply rearrangement based on the parameter
3. Pass convention through config pipeline

---

## Test Coverage Analysis

### Coverage Matrix: Will smoke tests catch a `rot6d_convention` change?

| Task | Smoke Test Exists? | Dataset | Uses LoadSmplx55? | Catches Regression? |
|------|------------------|---------|------------------|-------------------|
| **PRISM** | ❌ NO | None/Synthetic | ❌ NO | ❌ NO |
| **PRISM-MCM** | ❌ NO | None/Synthetic | ❌ NO | ❌ NO |
| **VerMo** | ✅ YES | `VermoToyDataset` | ❌ NO | ❌ NO |
| **HyMotion M2M** | ✅ YES | `HyMotionM2MSyntheticDataset` | ❌ NO | ❌ NO |
| **HyMotion M2M v2** | ✅ YES | `HyMotionM2MSyntheticDataset` | ❌ NO | ❌ NO |
| **HyMotion T2M** | ✅ YES | Synthetic | ❌ NO | ❌ NO |

**Conclusion:** ❌ **Smoke tests will NOT catch rot6d_convention regressions** because they bypass LoadSmplx55 entirely.

---

## Gaps in Test Coverage

### Gap 1: Missing PRISM Smoke Configs
- Test expects: `configs/prism/prism_smoke.py` and `configs/prism/prism_mcm_smoke.py`
- Current state: **Neither file exists**
- Test behavior: Will skip PRISM tests with message "Missing required local asset"
- Impact: PRISM model training startup is NOT smoke-tested at all

### Gap 2: Synthetic vs. Real Data Pipeline Mismatch
**Real training pipeline:**
```
SMPL-X NPZ files → LoadSmplx55 → process_smplx_pose() → rot_6d conversion → model
```

**Smoke test pipeline:**
```
Random Gaussian tensors (pre-generated) → model
```

- `rot6d_convention` parameter affects `process_smplx_pose()` **only**
- Synthetic data never flows through `process_smplx_pose()`
- ❌ Smoke tests will NOT exercise your new parameter

### Gap 3: VerMo RotDim Configuration
- `VermoToyDataset` hardcodes `rot_dim=6`
- If you add `rot6d_convention`, the toy dataset should accept it as a parameter
- Currently: No way to test both row-major and column-major in smoke tests

### Gap 4: No Parametrized Convention Testing
- Smoke tests only run once per task
- They don't test alternative `rot6d_convention` values
- Would need separate test cases to validate both "row" and "column" behaviors

---

## Specific Recommendations for Your Change

### ✅ To Ensure Full Coverage of `rot6d_convention`:

1. **Add PRISM smoke configs** (if not already planned):
   - Create `configs/prism/prism_smoke.py`
   - Create `configs/prism/prism_mcm_smoke.py`
   - Use synthetic data or tiny real datasets

2. **Update smoke test datasets to reference real motion data**:
   - Option A: Use real motion files (slow, requires assets)
   - Option B: Use `LoadSmplx55` with precomputed NPZ cache
   - Option C: Create a "mid-level" smoke dataset that exercises LoadSmplx55

3. **Add integration tests** (not just smoke tests):
   - Create `tests/integration/test_loadsmplx55_convention.py`
   - Test `rot6d_convention="row"` and `rot6d_convention="column"` explicitly
   - Verify bit-for-bit equivalence with expected outputs
   - Use small real motion files from `tests/assets/`

4. **Update VermoToyDataset**:
   - Add `rot6d_convention: str = "row"` parameter
   - Document that it controls the output rotation format
   - Have toy dataset generate matching output

5. **Smoke test execution**:
   - ✅ Will still pass (synthetic data, no behavior change)
   - ❌ Will NOT catch LoadSmplx55 regressions
   - ✅ Integration tests will catch those

---

## Config File Locations Summary

| File | Lines | Purpose |
|------|-------|---------|
| `tests/smoke/test_task_startup.py` | 333 | Main test orchestration |
| `configs/vermo/vermo_smoke.py` | 97 | VerMo toy config |
| `configs/hymotion_m2m/hymotion_m2m_smoke.py` | 97 | HyMotion M2M 135-dim config |
| `configs/hymotion_m2m_v2/hymotion_m2m_v2_smoke.py` | 101 | HyMotion M2M 198-dim config |
| `configs/hymotion_t2m/hymotion_t2m_smoke.py` | ~100 | HyMotion T2M config |
| `hftrainer/datasets/motion/hymotion_m2m_dataset.py` | 96 | M2M synthetic dataset |
| `hftrainer/datasets/motion/vermo_toy_dataset.py` | 70 | VerMo toy dataset |
| `hftrainer/datasets/motion/motionhub/transforms/load_smplx.py` | 496 | **LoadSmplx55 + process_smplx_pose()** |

---

## Concrete Impact Assessment

### If you add `rot6d_convention="row"` (default) to LoadSmplx55:

1. **Smoke tests:** ✅ Still pass (use synthetic data)
2. **Real training (PRISM, VerMo, HyMotion):** ❓ Depends on default choice
   - If default="row": ✅ No change to current behavior
   - If default="column": ❌ All real training breaks silently (different motion representation)

3. **Integration tests needed:** ✅ To validate rot6d conversion logic

### If you update PRISM/VerMo configs to use `rot6d_convention="column"`:

1. **Smoke tests:** ❌ Don't catch change (no LoadSmplx55)
2. **Real training:** ❓ Requires parallel testing
3. **Smoke config missing:** ❌ Can't validate PRISM at all

---

## Summary Table: What Catches What?

| Regression Type | Caught by Smoke Test? | Requires? |
|-----------------|----------------------|-----------|
| Model architecture bug | ✅ YES | Any dataset ok |
| Trainer loop bug | ✅ YES | Any dataset ok |
| Inference code regression | ✅ YES | Trained checkpoint |
| **LoadSmplx55 rot6d conversion bug** | ❌ NO | Real SMPL-X pipeline |
| **rot6d_convention parameter bug** | ❌ NO | Integration test |
| PRISM startup | ❌ NO (config missing) | PRISM smoke config needed |
| VerMo startup | ✅ YES | Synthetic data ok |
| HyMotion M2M startup | ✅ YES | Synthetic data ok |

