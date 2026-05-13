# HyMotionT2MBundle Mean/Std Normalization Stats: Complete Loading Analysis

**Date**: 2026-05-13  
**Repo Path**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer`

---

## Executive Summary

HyMotionT2MBundle loads normalization stats via **`_load_mean_std()`** method with **three possible sources** (in order of preference):

1. **Mean.npy/Std.npy files** from a directory path (if `mean_std_dir` is provided and exists)
2. **Fallback to zero-mean/unit-std** (if `mean_std_dir` is None or invalid)
3. **NOT stored in checkpoint files** (contrary to initial hypothesis) — checkpoint `model_state_dict` contains `mean` and `std` keys, but these are loaded by the checkpoint loading system separately, not by `_load_mean_std()`

---

## 1. Code Analysis: HyMotionT2MBundle._load_mean_std()

**File**: `hftrainer/models/motion/hymotion_t2m/bundle.py` (lines 131-145)

```python
def _load_mean_std(self, mean_std_dir: Optional[str]) -> None:
    if mean_std_dir is not None and osp.isdir(mean_std_dir):
        mean = torch.from_numpy(
            np.load(osp.join(mean_std_dir, 'Mean.npy'))
        ).float()
        std = torch.from_numpy(
            np.load(osp.join(mean_std_dir, 'Std.npy'))
        ).float()
        # Clamp std to avoid div-by-zero
        std = torch.where(std < 1e-3, torch.ones_like(std), std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    else:
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
```

### Loading Logic

| Condition | Behavior | Result |
|-----------|----------|--------|
| `mean_std_dir` provided AND directory exists AND contains `Mean.npy` + `Std.npy` | Load from files + clamp std ≥ 1e-3 | Proper normalization buffers |
| `mean_std_dir` is None | Skip file loading | Registers zero mean, unit std (no normalization) |
| `mean_std_dir` provided but NOT a valid directory | Skip file loading | Registers zero mean, unit std (no normalization) |
| `mean_std_dir` provided but missing Mean.npy or Std.npy | **FileNotFoundError raised** | Training/inference fails |

### Std Clamping

```python
std = torch.where(std < 1e-3, torch.ones_like(std), std)
```

Any dimension with `std < 0.001` is replaced with 1.0 to prevent division-by-zero in normalization/denormalization.

---

## 2. Checkpoint Analysis: Do .ckpt Files Store Mean/Std?

**Finding**: YES, but NOT loaded by `_load_mean_std()`.

**Evidence**: Inspected `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`

```
Checkpoint structure:
  'model_state_dict': {
    'null_vtxt_feat': tensor(shape [1, 1, 768]),
    'null_ctxt_input': tensor(shape [1, 1, 4096]),
    'mean': tensor(shape [201]),       ← Present in checkpoint
    'std': tensor(shape [201]),        ← Present in checkpoint
    'motion_transformer.input_encoder.weight': ...,
    'motion_transformer.ctxt_encoder.weight': ...,
    ... (308 transformer keys total)
  },
  'epoch': int,
  'global_step': int
}
```

**Key Point**: The `mean` and `std` tensors in the checkpoint are loaded **by the generic model-loading system** (via `ModelBundle.load_state_dict_selective()` or `trainer.load_checkpoint()`), **NOT by `_load_mean_std()`**, which only runs during `__init__()` before checkpoint loading.

**Data Flow**:
```
1. HyMotionT2MBundle.__init__(mean_std_dir=...) 
   → _load_mean_std() registers buffers (either from files or zeros)
   
2. Later: trainer.load_checkpoint(path)
   → checkpoint['model_state_dict']['mean'] and ['std'] overwrite the buffers
   → Those checkpoint values persist for inference/training
```

**Implication for T2M**: If loading a T2M checkpoint with `mean_std_dir=None`, the `__init__` registers zero/unit buffers, but they are immediately overwritten during checkpoint load. The final result uses the checkpoint's mean/std values (201-dim for T2M-201).

---

## 3. M2M Bundle Comparison

**File**: `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 149-163)

**Identical implementation** to T2M:

```python
def _load_mean_std(self, mean_std_dir: Optional[str]) -> None:
    if mean_std_dir is not None and osp.isdir(mean_std_dir):
        # Load from files
    else:
        # Register zero/unit defaults
```

### Key Difference: No M2M v2 bundle

The repo does NOT have `hymotion_m2m_v2/bundle.py`. M2M v2 uses the same `HyMotionM2MBundle` class as M2M v1, but with different config and training setup.

**M2M v2 stats are NOT compatible with T2M**:
- M2M v2 uses **198-dim** representation (3 trans + 132 rot6d + 63 position)
- T2M uses **201-dim** representation (3 rel trans + 198 rot6d)
- Different normalization statistics directories:
  - T2M: `checkpoints/HY-Motion-1.0/stats/` or config-specific paths
  - M2M v1: `data/hymotion_m2m_data/_stats/` (135-dim)
  - M2M v2: `data/hymotion_m2m_data/_stats_198dim` (198-dim)

---

## 4. All mean_std_dir Paths Referenced in Configs

### Grep Results (55 config files)

| `mean_std_dir` Value | Used By | Dimensions | Exists? |
|---|---|---|---|
| `checkpoints/HY-Motion-1.0/stats/` | T2M (T2M-201dim, smoke) | 201-dim | ❌ **Not found** (checkpoint stores inline) |
| `data/hymotion_m2m_data/_stats` | M2M v1 (all text-free variants) | 135-dim | ✅ **Exists** |
| `data/hymotion_m2m_data/_stats_global_rot` | M2M (globalrot variants), DiT | 135-dim | ✅ **Exists** |
| `data/hymotion_m2m_data/_stats_198dim` | M2M v2 (local, uncond, caption) | 198-dim | ✅ **Exists** |
| `data/hymotion_m2m_data/_stats_198dim_global_rot` | M2M v2 (globalrot variants) | 198-dim | ✅ **Exists** |
| `data/hymotion_m2m_data/_stats_201dim` | (No config uses this currently) | 201-dim | ✅ **Exists** |
| `data/hymotion_m2m_data/_stats_201dim_global_rot` | (No config uses this currently) | 201-dim | ✅ **Exists** |
| `/apdcephfs_cq11/share_1467498/home/zeyuling/HY-Motion-1.0/stats/` | UMO-201dim | 201-dim | ❌ **Not found** (symlink to checkpoint) |
| `None` | Smoke test configs | N/A (identity norm) | N/A |

### Directory Structure

```
data/hymotion_m2m_data/
├── _stats/                     # 135-dim (M2M v1)
│   ├── Mean.npy (1208 bytes)
│   └── Std.npy  (1208 bytes)
├── _stats_global_rot/          # 135-dim with global rotation
│   ├── Mean.npy (1336 bytes)
│   └── Std.npy  (1336 bytes)
├── _stats_198dim/              # 198-dim (M2M v2 local)
│   ├── Mean.npy (920 bytes)
│   └── Std.npy  (920 bytes)
├── _stats_198dim_global_rot/   # 198-dim with global rotation
│   ├── Mean.npy (920 bytes)
│   └── Std.npy  (920 bytes)
├── _stats_201dim/              # 201-dim (T2M-compatible, currently unused)
│   ├── Mean.npy (1864 bytes)
│   └── Std.npy  (1864 bytes)
└── _stats_201dim_global_rot/   # 201-dim global rotation (currently unused)
    ├── Mean.npy (1864 bytes)
    └── Std.npy  (1864 bytes)
```

---

## 5. Checkpoint Loading Utility Analysis

**File**: `hftrainer/utils/checkpoint_utils.py`

### Functions

| Function | Purpose | Relevant to Mean/Std? |
|----------|---------|----------------------|
| `find_latest_checkpoint()` | Locate latest `.ckpt` in work_dir | No |
| `_unwrap_legacy_checkpoint()` | Extract `state_dict` from nested formats | **Yes** — extracts mean/std that were saved in checkpoint |
| `load_checkpoint()` | Load `.ckpt` / `.safetensors` / `pytorch_model.bin` | **Yes** — loads files containing state_dict |

### Key Code (lines 62-78)

```python
def _unwrap_legacy_checkpoint(data: dict) -> dict:
    """Unwrap legacy checkpoint formats to a flat/nested state_dict."""
    if 'state_dict' in data and isinstance(data['state_dict'], dict):
        return data['state_dict']  # MMEngine format
    if 'model_state_dict' in data and isinstance(data['model_state_dict'], dict):
        return data['model_state_dict']  # PyTorch Lightning / HunyuanMotion format
    return data
```

**Note**: This function extracts the state_dict (which includes `mean` and `std` keys if they were saved), but it does **NOT** handle them specially. They are treated as regular buffer tensors and loaded by the generic `load_state_dict()` system.

### Implication

- Checkpoints saved by `save_checkpoint()` with `use_safetensors=True` will **NOT preserve mean/std** (safetensors only supports flat tensor dicts, bundle-level buffers are excluded)
- Checkpoints saved as `.pt` (PyTorch pickle) **WILL preserve mean/std** if they were registered as buffers during training
- Inference relies on:
  - **Option 1**: Load from `mean_std_dir` files (called in `__init__` before checkpoint load)
  - **Option 2**: Restore from checkpoint (if checkpoint was saved with buffers intact)
  - **Option 3**: Use zero/unit defaults (if no `mean_std_dir` and checkpoint has no mean/std)

---

## 6. T2M Config Analysis

**File**: `configs/hymotion_t2m/hymotion_t2m_201dim_046b.py`

```python
model = dict(
    type='HyMotionT2MBundle',
    ...
    mean_std_dir='checkpoints/HY-Motion-1.0/stats/',  # Line 62
    ...
)

load_from = dict(
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
```

### Problem

The config references `mean_std_dir='checkpoints/HY-Motion-1.0/stats/'`, but:
- **This directory does NOT exist** in the repo
- The actual checkpoint is at `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
- The T2M training/inference relies on **fallback mechanism**:
  1. Try to load from `checkpoints/HY-Motion-1.0/stats/` → **Fails** (no such directory)
  2. Register zero/unit defaults → `mean = zeros(1), std = ones(1)`
  3. Later: `load_checkpoint()` overwrites these with checkpoint's inline `mean` (201-dim) and `std` (201-dim)

### Successful Paths

**Solution 1**: Use `mean_std_dir=None` (all T2M smoke tests do this)
```python
model = dict(
    ...
    mean_std_dir=None,  # Skip file loading, rely on checkpoint inline values
    ...
)
```

**Solution 2**: Create the stats directory and populate it
```bash
# Extract mean/std from T2M checkpoint and save to directory
mkdir -p checkpoints/HY-Motion-1.0/stats/
# (script to extract and save Mean.npy / Std.npy)
```

**Solution 3**: Use existing 201-dim stats (if training new model)
```python
mean_std_dir='data/hymotion_m2m_data/_stats_201dim'  # 201-dim stats exist
```

---

## 7. Cross-Project Stats Reusability

### Can M2M v2 stats be reused for T2M?

**NO** — dimensional mismatch:
- M2M v2: 198-dim (3 trans + 132 rot6d + 63 position)
- T2M: 201-dim (3 rel trans + 6*33 rot6d where 33 joints)

Even if dimensions matched numerically, the **semantic meaning differs**:
- M2M v2: Uses both rotation AND position channels
- T2M: Uses only rotation (no explicit position)
- Normalization statistics are distribution-specific; using wrong stats produces incorrect scaling

### Stats Available

| Model | Dimension | Stats Dir | Status |
|-------|-----------|-----------|--------|
| M2M v1 | 135 | `_stats/` | ✅ Used |
| M2M v1 globalrot | 135 | `_stats_global_rot/` | ✅ Used |
| M2M v2 local | 198 | `_stats_198dim/` | ✅ Used |
| M2M v2 globalrot | 198 | `_stats_198dim_global_rot/` | ✅ Used |
| T2M (optional) | 201 | `_stats_201dim/` | ⚠️ Exists but unused |
| T2M globalrot (optional) | 201 | `_stats_201dim_global_rot/` | ⚠️ Exists but unused |

---

## 8. Summary of All Possible Stats Sources for HyMotionT2MBundle

### Priority Order (what code checks)

1. **`mean_std_dir` parameter (if provided and is a valid directory)**
   - Location: Config parameter
   - T2M path: `checkpoints/HY-Motion-1.0/stats/` (intended but doesn't exist)
   - T2M fallback: `data/hymotion_m2m_data/_stats_201dim/` (exists but not currently used)
   - Files required: `Mean.npy`, `Std.npy`
   - Dimensions: Must match model's motion_dim (T2M=201)

2. **Checkpoint's inline `mean` and `std` buffers (loaded after __init__)**
   - Source: `load_from.path` checkpoint file
   - For T2M: `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
   - Dimensions: 201-dim (from HY-Motion-1.0 pretraining)
   - When used: Always, after `_load_mean_std()` (if checkpoint is loaded)
   - Priority: **Overrides** `mean_std_dir` files if both exist

3. **Hardcoded zero/unit defaults (if no valid mean_std_dir)**
   - `mean = torch.zeros(1)` (shape [1])
   - `std = torch.ones(1)` (shape [1])
   - Effect: **No normalization** (multiplying/dividing by 1, adding 0)
   - Used by: Smoke test configs
   - ⚠️ **Problematic for real models**: Causes dimension mismatch in loss computation when model outputs 201-dim but mean/std are 1-dim

---

## 9. Critical Issues & Recommendations

### Issue #1: T2M Config References Non-Existent Stats Directory

**Status**: ⚠️ **NOT CRITICAL** (checkpoint fallback works)

**Current behavior**:
- Config specifies `mean_std_dir='checkpoints/HY-Motion-1.0/stats/'`
- Directory does not exist → `osp.isdir()` returns False
- Falls back to zero/unit defaults
- Checkpoint load overwrites with inline values

**Recommendation**:
```python
# Option A: Fix config to use actual stats or None
mean_std_dir=None,  # Rely on checkpoint inline stats

# Option B: Create stats directory from checkpoint
# (requires extraction utility)

# Option C: Point to existing 201-dim stats
mean_std_dir='data/hymotion_m2m_data/_stats_201dim/',
```

### Issue #2: Safetensors Checkpoints Lose Mean/Std

**Status**: ⚠️ **MODERATE** (affects model.safetensors but not model.pt)

**Current behavior**:
- `.safetensors` format cannot store bundle-level buffers
- If model is saved as `.safetensors`, mean/std are lost
- Loading from `.safetensors` + `mean_std_dir=None` → zero/unit defaults

**Recommendation**:
- Keep mean/std files on disk (`_stats_*/Mean.npy`)
- Always specify `mean_std_dir` in configs
- Or: Save models as `.pt` format to preserve buffers

### Issue #3: Dimension Mismatch When mean/std=zeros(1)/ones(1)

**Status**: 🔴 **CRITICAL** (causes training/inference failures if not caught)

**Symptom**: If `mean_std_dir=None` and checkpoint is not loaded, then:
```python
# In normalize_motion():
normalized = (motion - self.mean) / self.std  # motion: (B, T, 201), mean/std: (1,)
# Broadcasting: normalized = (B, T, 201) - (1,) → ERROR or wrong computation

# In denormalize_motion():
denormalized = motion * std + mean  # Same issue
```

**Prevention**: Always either:
1. Provide `mean_std_dir` pointing to correct-dimension stats files
2. Load a checkpoint with matching-dimension mean/std buffers
3. Explicitly check dimension matching in code

---

## 10. Recommended Configuration for T2M

### Scenario A: Fine-tuning from HY-Motion-1.0 Pretrained

```python
model = dict(
    type='HyMotionT2MBundle',
    ...
    mean_std_dir=None,  # Don't load from files
    motion_type='smpl_33',  # or 'smpl_22' depending on task
    ...
)

load_from = dict(
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
```

**Why**: Checkpoint's inline mean/std (201-dim) are correct and preserved.

### Scenario B: Training from Scratch

```python
model = dict(
    type='HyMotionT2MBundle',
    ...
    mean_std_dir='data/hymotion_m2m_data/_stats_201dim/',  # Pre-computed 201-dim stats
    motion_type='smpl_33',
    ...
)

load_from = dict(
    path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
    load_scope='model',
)
```

**Why**: Stats are available on disk from initialization; checkpoint mean/std will overwrite if present.

### Scenario C: Smoke Test / Debug

```python
model = dict(
    type='HyMotionT2MBundle',
    ...
    mean_std_dir=None,  # Identity normalization for quick testing
    ...
)

# Don't load checkpoint, or load one without mean/std
```

**Why**: Quick iteration without waiting for stats I/O.

---

## Conclusion

HyMotionT2MBundle has **robust fallback handling** for mean/std normalization:

1. **Primary source**: `mean_std_dir` files (Mean.npy / Std.npy)
2. **Secondary source**: Checkpoint's inline buffers
3. **Tertiary source**: Hardcoded zero/unit defaults (identity normalization)

For **production T2M inference**, ensure:
- ✅ Either `mean_std_dir` points to valid 201-dim stats directory
- ✅ Or checkpoint being loaded contains mean/std buffers (HY-Motion-1.0 does)
- ✅ Or explicitly accept identity normalization (for testing only)

The current T2M config (`hymotion_t2m_201dim_046b.py`) functions correctly because it loads from HY-Motion-1.0 checkpoint, which includes inline mean/std values.

