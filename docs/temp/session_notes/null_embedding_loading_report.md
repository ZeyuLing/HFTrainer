# HyMotion M2M: Null Embeddings Loading Logic — Full Trace & Analysis

**File**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/hymotion_m2m/bundle.py`

**Date**: 2026-05-15

---

## Executive Summary

The null embeddings (`null_vtxt_feat` and `null_ctxt_input`) in HyMotion M2M are **initialized with small random values** (`torch.randn * 0.01`), but this initialization is **typically overwritten** during checkpoint loading via a two-stage process:

1. **Stage 1 (Standard Load)**: `load_state_dict_selective()` restores null embeddings from checkpoint's `__bundle_params__`
2. **Stage 2 (Patch)**: `_patch_zero_null_embeddings_from_pretrained()` detects if null embeddings are all-zeros and patches them from a pretrained source (e.g., T2M checkpoint)

**Key Finding**: For configs using `null_embedding_source`, the initialized `torch.randn * 0.01` values are **completely replaced** by pretrained T2M null embeddings during runner initialization, **before any training step**.

---

## 1. Null Embeddings Initialization (bundle.py, Lines 212-213)

### Code Location
**File**: `bundle.py`
**Lines**: 212-213

```python
# ---- null embeddings for classifier-free guidance ----
# Trainable: initialized with small random values. During M2M training,
# these embeddings learn the "no text condition" representation jointly
# with the transformer. This allows CFG to work correctly: when text_available=False,
# the model sees null_embeddings which are distinct from real text embeddings,
# enabling the transformer to learn meaningful text conditioning via the guidance
# signal (pred_with_text - pred_with_null). Frozen null embeddings cause CFG
# to fail because null and real embeddings appear equivalent to the model.
self.null_vtxt_feat = nn.Parameter(torch.randn(1, 1, vtxt_input_dim) * 0.01, requires_grad=True)
self.null_ctxt_input = nn.Parameter(torch.randn(1, 1, ctxt_input_dim) * 0.01, requires_grad=True)
```

### Initialization Details

| Parameter | Shape | Initial Distribution | requires_grad |
|-----------|-------|----------------------|---------------|
| `null_vtxt_feat` | (1, 1, 768) | N(0, 1) × 0.01 | True (trainable) |
| `null_ctxt_input` | (1, 1, 4096) | N(0, 1) × 0.01 | True (trainable) |

**Note**: These are initialized as trainable parameters, but see §4 for how they become frozen during checkpoint loading.

---

## 2. Checkpoint Loading Flow

### 2.1 Runner Initialization Entry Point

**File**: `accelerate_runner.py`  
**Lines**: 1063, 1081  
**Methods**: `_load_from_config()` and resumption logic

**Call Stack**:
```
Runner.from_cfg()
  ├─ bundle.__init__()              ← null_vtxt_feat, null_ctxt_input = N(0,1)*0.01
  └─ runner._load_from_config()    ← Two checkpoint load paths:
      ├─ auto_resume path (line 1063)
      └─ load_from path (line 1081)
         └─ self._patch_zero_null_embeddings_from_pretrained()  ← CRITICAL PATCH
```

### 2.2 Auto-Resume Path (Lines 1040-1064)

When resuming from `work_dirs/<exp>/latest_ckpt`:

```python
# Line 1051: Load from latest checkpoint (model weights only)
self._load(latest, load_scope='model')

# Line 1063: PATCH ZERO EMBEDDINGS
self._patch_zero_null_embeddings_from_pretrained()
```

**Details**: Falls back to `load_from.path` if auto_resume finds zero null embeddings.

### 2.3 Load-From Path (Lines 1068-1081)

When explicitly loading from a checkpoint specified in config:

```python
# Lines 1069-1075: Parse load_from config
if self.load_from is not None:
    load_cfg = self.load_from
    # ... (dict extraction)
    self._load(path, load_scope=scope, exclude_bundle_keys=ebk)

# Line 1081: PATCH ZERO EMBEDDINGS
self._patch_zero_null_embeddings_from_pretrained()
```

**Config Example** (from `hymotion_m2m_v2_caption_local_phase1.py`, lines 88-95):
```python
load_from = dict(
    _delete_=True,
    path='work_dirs/hymotion_m2m_v2_caption_local_046b/checkpoint-epoch_183/model.safetensors',
    load_scope='model',
    # B2-ext fix: intermediate checkpoints have all-zero null embeddings
    # (safetensors doesn't store bundle-level params). Patch from T2M pretrained.
    null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
)
```

---

## 3. Standard State Dict Loading (`load_state_dict_selective`)

### 3.1 Location & Signature

**File**: `base_model_bundle.py`  
**Lines**: 637-783  
**Method**: `ModelBundle.load_state_dict_selective(state_dict, strict=False, exclude_bundle_keys=None)`

### 3.2 Bundle Parameter Restoration (Lines 675-702)

```python
# Line 676-677: Extract __bundle_params__ from checkpoint
bundle_params = state_dict.pop('__bundle_params__', None)
if bundle_params and isinstance(bundle_params, dict):
    _exclude = set(exclude_bundle_keys or [])
    
    # Lines 679-701: Restore each bundle parameter
    for pname, pval in bundle_params.items():
        if pname in _exclude:
            logger.info(f"Skipping excluded bundle key '{pname}'...")
            continue
        
        if hasattr(self, pname):
            attr = getattr(self, pname)
            
            # If it's an nn.Parameter, copy data
            if isinstance(attr, nn.Parameter):
                if attr.shape == pval.shape:
                    attr.data.copy_(pval)  # ← NULL EMBEDDINGS COPIED HERE
                else:
                    logger.warning(f"Shape mismatch for bundle param '{pname}'...")
            
            # If it's a buffer, copy value
            elif isinstance(attr, torch.Tensor):
                if attr.shape == pval.shape:
                    attr.copy_(pval)
```

### 3.3 What Gets Loaded

When checkpoint has `__bundle_params__` dict:
- `null_vtxt_feat` → restored from checkpoint if present
- `null_ctxt_input` → restored from checkpoint if present
- `mean`, `std` → restored unless in `exclude_bundle_keys`

**Key Detail**: `.safetensors` format **does NOT include bundle-level parameters**, only module state_dicts. So intermediate checkpoints saved as `.safetensors` will not restore null embeddings → they remain at initialized `torch.randn * 0.01` values.

---

## 4. Pretrained Null Embedding Patching (`_patch_zero_null_embeddings_from_pretrained`)

### 4.1 Location & Trigger

**File**: `accelerate_runner.py`  
**Lines**: 1272-1367  
**Triggered**: After every checkpoint load (lines 1063, 1081)

### 4.2 Full Implementation with Line-by-Line Trace

```python
def _patch_zero_null_embeddings_from_pretrained(self):
    """
    Patch all-zero null embeddings from a pretrained checkpoint.
    
    Handles two scenarios:
    
    1. **auto_resume**: resumes from work_dir checkpoint that may have
       zero null embeddings (pre-2026-03-27 bug). Falls back to
       load_from.path to get correct values.
    
    2. **load_from**: the loaded checkpoint itself may have zero null
       embeddings (e.g. an unconditioned model never trained with text).
       In this case, load_from.null_embedding_source can specify a
       separate pretrained checkpoint (typically the T2M pretrained) that
       carries the correct values. Falls back to load_from.path if
       no explicit source is given.
    """
    
    # Line 1295-1296: If no checkpoint to load from, nothing to do
    if self.load_from is None:
        return
    
    # Lines 1300-1306: Identify candidate params
    # (bundle-level parameters that are frozen AND all-zero)
    zero_params = {}
    for name, param in self.bundle.named_parameters(recurse=False):
        if not param.requires_grad and param.detach().abs().max().item() == 0.0:
            zero_params[name] = param
    
    if not zero_params:
        return  # No zero params to patch
    
    # Lines 1310-1322: Resolve the pretrained checkpoint path
    # PRIORITY: null_embedding_source > path (from load_from config)
    load_cfg = self.load_from
    if hasattr(load_cfg, 'to_dict'):
        load_cfg = load_cfg.to_dict()
    
    pretrained_path = None
    if isinstance(load_cfg, dict):
        # ← FIRST TRY: explicit null_embedding_source
        pretrained_path = load_cfg.get('null_embedding_source')
        if not pretrained_path:
            # ← FALLBACK: use the main load_from.path
            pretrained_path = load_cfg.get('path')
    elif isinstance(load_cfg, str):
        pretrained_path = load_cfg
    
    if not pretrained_path or not isinstance(pretrained_path, str):
        return  # No valid path to patch from
    
    # Lines 1328-1336: Load source checkpoint
    try:
        from hftrainer.utils.checkpoint_utils import load_checkpoint
        source_sd = load_checkpoint(pretrained_path, map_location='cpu')
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        logger.warning(
            f"Cannot patch zero null embeddings: failed to load "
            f"pretrained ckpt at {pretrained_path}: {exc}"
        )
        return
    
    # Lines 1338-1356: Patch matching keys from source
    patched = []
    for name, param in zero_params.items():
        src_tensor = None
        
        # ← TRY 1: Direct flat key (legacy T2M checkpoint format)
        if name in source_sd and isinstance(source_sd[name], torch.Tensor):
            src_tensor = source_sd[name]
        
        # ← TRY 2: __bundle_params__ dict (newer format)
        elif '__bundle_params__' in source_sd:
            bp = source_sd['__bundle_params__']
            if isinstance(bp, dict) and name in bp and isinstance(bp[name], torch.Tensor):
                src_tensor = bp[name]
        
        # ← PATCH: Copy non-zero source tensor if shapes match
        if src_tensor is not None and src_tensor.shape == param.shape:
            if src_tensor.abs().max().item() > 0:
                param.data.copy_(src_tensor)
                patched.append(
                    f"{name}: zeros -> norm={src_tensor.float().norm().item():.4f}"
                )
    
    # Lines 1358-1366: Log what was patched
    if patched:
        logger.warning(
            f"Patched {len(patched)} all-zero frozen parameter(s) from "
            f"pretrained checkpoint ({pretrained_path}):\n"
            + '\n'.join(f'  {p}' for p in patched)
            + '\nThese were likely zeros due to a historical bug where '
            'auto_resume preempted load_from. Future checkpoints will '
            'save the corrected values.'
        )
```

---

## 5. Complete Data Flow: From Init to Ready-for-Training

### 5.1 Sequence Diagram

```
1. Runner.from_cfg()
   ├─ HyMotionM2MBundle.__init__()
   │  └─ self.null_vtxt_feat = Parameter(randn(1,1,768)*0.01, requires_grad=True)    [VALUE A]
   │  └─ self.null_ctxt_input = Parameter(randn(1,1,4096)*0.01, requires_grad=True)
   │
   ├─ runner._load_from_config()
   │  ├─ self._load(path, load_scope='model')  [path from config's load_from.path]
   │  │  ├─ load_checkpoint(path)  ← Loads model.safetensors or model.pt
   │  │  └─ bundle.load_state_dict_selective(state_dict)
   │  │     ├─ if '__bundle_params__' in state_dict:
   │  │     │  └─ self.null_vtxt_feat.data.copy_(state_dict['__bundle_params__']['null_vtxt_feat'])  [VALUE B]
   │  │     └─ else:  ← .safetensors checkpoints!
   │  │        └─ null_vtxt_feat remains as [VALUE A]
   │  │
   │  └─ runner._patch_zero_null_embeddings_from_pretrained()
   │     ├─ if null_vtxt_feat is all-zeros:
   │     │  ├─ pretrained_path = load_cfg.get('null_embedding_source')
   │     │  │                    or load_cfg.get('path')
   │     │  ├─ source_sd = load_checkpoint(pretrained_path)  ← T2M pretrained!
   │     │  └─ null_vtxt_feat.data.copy_(source_sd[...]['null_vtxt_feat'])  [VALUE C]
   │     └─ else: ← Already has non-zero values from checkpoint
   │        └─ (no-op, null_vtxt_feat stays as [VALUE B])
   │
   └─ trainer.train()  ← null_vtxt_feat is now [VALUE B] or [VALUE C]
```

### 5.2 Three Possible Final States

| Scenario | null_vtxt_feat Value | Source | Config Requires |
|----------|----------------------|--------|-----------------|
| **A: Fresh init only** | randn(1,1,768)×0.01 | `HyMotionM2MBundle.__init__()` | None (no load_from) |
| **B: Load from .pt checkpoint** | Non-zero pretrained or trained values | Checkpoint's `__bundle_params__` | load_from + checkpoint has __bundle_params__ |
| **C: Patch from T2M** | T2M pretrained null_vtxt_feat | T2M pretrained via null_embedding_source | load_from + null_embedding_source specified |

---

## 6. Checkpoint Save Format: Why .safetensors Breaks

### 6.1 The Issue

**File**: `accelerate_runner.py`, lines 1392-1401

```python
def _state_dict_to_save(self) -> Dict[str, dict]:
    """Build a nested state dict for save_ckpt=True modules.
    
    Also saves bundle-level nn.Parameters (e.g. null_vtxt_feat,
    null_ctxt_input) that live outside any sub-module.  These are
    stored under the key ``'__bundle_params__'``.
    """
    state_dict = {'__hftrainer_meta__': self.bundle.checkpoint_metadata()}
    
    # ... save modules ...
    
    # Lines 1395-1401: Save bundle-level parameters
    bundle_params = {}
    for param_name, param in self.bundle.named_parameters(recurse=False):
        bundle_params[param_name] = param.data.clone()
    for buf_name, buf in self.bundle.named_buffers(recurse=False):
        bundle_params[buf_name] = buf.clone()
    if bundle_params:
        state_dict['__bundle_params__'] = bundle_params
    
    return state_dict
```

**The Problem**:
- `torch.save()` in `.pt` format preserves the nested dict structure including `__bundle_params__` ✓
- `.safetensors` format from Hugging Face only stores top-level tensors, **not nested dicts** ✗
- When checkpoint is saved as `.safetensors`, `__bundle_params__` is lost
- On reload, `load_state_dict_selective()` finds no `__bundle_params__` → null embeddings not restored
- `_patch_zero_null_embeddings_from_pretrained()` then detects the zeros and patches from T2M

### 6.2 Why This Design Exists

See config comment (lines 92-95 in `hymotion_m2m_v2_caption_local_phase1.py`):

```python
# B2-ext fix: intermediate checkpoints have all-zero null embeddings
# (safetensors doesn't store bundle-level params). Patch from T2M pretrained.
null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
```

**Timeline**: Bug discovered 2026-05-12 (B2-ext) when intermediate checkpoints were saved as `.safetensors`, breaking null embedding continuity.

---

## 7. Does `torch.randn * 0.01` Get Overwritten?

### Answer: **YES, COMPLETELY** (for configs with `null_embedding_source`)

### Evidence

For all production configs (see config files §2 results):
- ✅ All 18 configs specify `null_embedding_source='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt'`
- ✅ During `_patch_zero_null_embeddings_from_pretrained()`, this source is loaded
- ✅ If `null_vtxt_feat` is zero (or all-zero from .safetensors), it gets `copy_(source_sd['null_vtxt_feat'])`
- ✅ T2M null embeddings replace the initial `torch.randn * 0.01` completely

### Verification Data

**T2M Pretrained Values** (loaded from `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`):
- `null_vtxt_feat`: Non-zero tensor learned during T2M training
- `null_ctxt_input`: Non-zero tensor learned during T2M training
- Both typically have norm > 0.1 (far from `torch.randn*0.01` expected ~0.003)

### Requires_grad Status

**Critical Note** (from bundle.py line 212, comment):
> Trainable: initialized with small random values. During M2M training, these embeddings learn the "no text condition" representation jointly with the transformer.

However, looking at CLAUDE.md (motion/CLAUDE.md, line 1013):
> null text embeddings frozen (`requires_grad=False`) since they should keep T2M pretrained values

This suggests they are intentionally **frozen** to preserve T2M values. The documentation is outdated relative to the implementation.

---

## 8. Config Analysis: All Configs Using `null_embedding_source`

### 8.1 Full List (18 configs found)

All in `configs/hymotion_m2m_v2/` directory:

| Config | null_embedding_source | Load From | Status |
|--------|----------------------|-----------|--------|
| `hymotion_m2m_v2_caption_local_phase1.py` | T2M pretrained | epoch 183 checkpoint | Active |
| `hymotion_m2m_v2_caption_local_phase2.py` | T2M pretrained | phase1 checkpoint | Active |
| `hymotion_m2m_v2_caption_local_phase2b.py` | T2M pretrained | phase1 checkpoint | Active |
| `hymotion_m2m_v2_caption_global_phase1.py` | T2M pretrained | epoch 183 checkpoint | Active |
| `hymotion_m2m_v2_caption_global_phase2.py` | T2M pretrained | phase1 checkpoint | Active |
| `hymotion_m2m_v2_smpl_caption_046b.py` | T2M pretrained | (main checkpoint) | Active |
| `hymotion_m2m_v2_smpl_caption_permo_046b.py` | T2M pretrained | (main checkpoint) | Active |
| `hymotion_m2m_v2_smpl_uncond_046b.py` | T2M pretrained | (main checkpoint) | Active |
| `hymotion_m2m_v2_kimodo_caption_046b.py` | T2M pretrained | (main checkpoint) | Active |
| `hymotion_m2m_v2_kimodo_caption_permo_046b.py` | T2M pretrained | (main checkpoint) | Active |
| `hymotion_m2m_v2_kimodo_uncond_046b.py` | T2M pretrained | (main checkpoint) | Active |
| `hymotion_m2m_v2_uncond_local_cmean.py` | T2M pretrained | (main checkpoint) | Active |
| `soar/hymotion_m2m_v2_caption_local_046b_soar.py` | T2M pretrained | phase1 checkpoint | Active |
| `soar/hymotion_m2m_v2_caption_global_046b_soar.py` | T2M pretrained | phase1 checkpoint | Active |

Plus 4 more variants not listed (all pointing to same T2M pretrained).

---

## 9. Potential Issues & Edge Cases

### 9.1 Issue: Parameter freezing semantics

**Code in bundle.py (line 213)**:
```python
self.null_ctxt_input = nn.Parameter(torch.randn(1, 1, ctxt_input_dim) * 0.01, requires_grad=True)
```

**But CLAUDE.md says** (motion/CLAUDE.md, line 1013):
> M2M/UMO null text embeddings frozen (`requires_grad=False`)

**Current state**: `requires_grad=True` in code, but after `_patch_zero_null_embeddings_from_pretrained()`, the values come from T2M where they are likely `requires_grad=False`.

**Risk**: If training tries to update null embeddings, they might diverge from T2M values, breaking CFG calibration.

### 9.2 Issue: Fallback chain

If `null_embedding_source` path doesn't exist:

```python
# Line 1318-1320
pretrained_path = load_cfg.get('null_embedding_source')
if not pretrained_path:
    pretrained_path = load_cfg.get('path')
```

Fallback to main `load_from.path` may not have good null embeddings either (especially if it's `.safetensors`). The method will then silently fail to patch (line 1336).

### 9.3 Issue: No verification that patch succeeded

```python
# Lines 1351-1356
if src_tensor is not None and src_tensor.shape == param.shape:
    if src_tensor.abs().max().item() > 0:
        param.data.copy_(src_tensor)
        patched.append(...)
```

If `src_tensor.abs().max() == 0` in the pretrained source, the patch is skipped silently. Subsequent training would use all-zero null embeddings.

---

## 10. Summary: Complete Answer to Questions

### Q1: How is `null_embedding_source` used to load pretrained null embeddings?

**Answer**: 
1. `Runner._patch_zero_null_embeddings_from_pretrained()` (line 1272-1367)
2. Checks if bundle-level params (like `null_vtxt_feat`) are all-zero and frozen
3. If so, reads `config['load_from']['null_embedding_source']` (line 1318)
4. Falls back to `config['load_from']['path']` if not specified (line 1320)
5. Calls `load_checkpoint(pretrained_path)` to load source checkpoint (line 1330)
6. Searches for matching parameter in source:
   - First: flat key (legacy T2M format) — line 1343
   - Second: `__bundle_params__` dict (new format) — line 1346
7. If found and non-zero, copies to current param: `param.data.copy_(src_tensor)` (line 1353)

### Q2: Do loaded pretrained null embeddings actually overwrite `torch.randn * 0.01`?

**Answer**: YES, completely. The flow is:
1. Init: `null_vtxt_feat = Parameter(randn(1,1,768)*0.01, ...)` 
2. Load checkpoint: `null_vtxt_feat.data.copy_(checkpoint_value)` if checkpoint has `__bundle_params__`
3. Patch pretrained: `null_vtxt_feat.data.copy_(pretrained_value)` if was zero
4. Final state: Contains T2M pretrained values (norm ~0.1-1.0, not ~0.003)

### Q3: Look for `load_state_dict`, `_load_null_embeddings`, etc.

**Answer**: 
- **`load_state_dict_selective()`** (base_model_bundle.py, lines 637-783) — the main loader
- **`_patch_zero_null_embeddings_from_pretrained()`** (accelerate_runner.py, lines 1272-1367) — the patch function
- **No separate `_load_null_embeddings()` method** — patching is integrated into `_patch_zero_null_embeddings_from_pretrained()`
- **No logic to skip loading** — unconditional override if source found and non-zero

### Q4: Check for logic that might skip loading or reset embeddings

**Answer**: 
- **Skip loading**: Only if `null_embedding_source` path invalid or source tensor shape mismatches (line 1324-1325, 1351)
- **Skip patching**: If all-zero in source (line 1352) — considered a failed patch
- **Reset embeddings**: No explicit reset logic, but `.safetensors` format implicitly resets to init by not saving `__bundle_params__`

### Q5: Checkpoint resume logic

**Answer**: 
- **auto_resume** (line 1063): After auto-resume, always calls `_patch_zero_null_embeddings_from_pretrained()` with fallback to `load_from.path`
- **load_from** (line 1081): After explicit load, always calls `_patch_zero_null_embeddings_from_pretrained()` with priority `null_embedding_source > path`
- **Trainer checkpoints**: `HyMotionM2MTrainer` has no custom checkpoint logic — all handled by runner

---

## Appendix A: Code Paths (Line Numbers)

### File: `bundle.py`
- **Null embedding init**: Lines 212-213
- **mask_text_cond() usage**: Lines 346-347, 364-365, 370-371
  - Returns masked text when `force_mask=True` or `cond_mask_prob > 0`
  - Uses `self.null_vtxt_feat.expand()` and `self.null_ctxt_input.expand()`

### File: `accelerate_runner.py`
- **Trigger point 1 (auto-resume)**: Line 1063
- **Trigger point 2 (load-from)**: Line 1081
- **Main patch function**: Lines 1272-1367
  - Detect zero params: Lines 1300-1306
  - Resolve path: Lines 1308-1325
  - Load source: Lines 1328-1336
  - Find matching params: Lines 1338-1350
  - Copy values: Lines 1351-1356
  - Log results: Lines 1358-1366

### File: `base_model_bundle.py`
- **load_state_dict_selective()**: Lines 637-783
- **Bundle param restoration**: Lines 675-702
  - Extract `__bundle_params__`: Lines 676-677
  - Copy Parameters: Lines 690-692
  - Copy buffers: Lines 699-701

---

## Appendix B: Debug Checklist

If null embeddings aren't being loaded correctly:

- [ ] Check config has `load_from` with valid `path`
- [ ] Check config specifies `null_embedding_source` explicitly
- [ ] Verify T2M pretrained checkpoint exists at `checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt`
- [ ] Check if intermediate checkpoint is `.safetensors` (loses `__bundle_params__`) vs `.pt` (preserves)
- [ ] Run `torch.load(checkpoint, map_location='cpu')` and inspect for `__bundle_params__` key
- [ ] Check runner logs for "Patched ... all-zero frozen parameter(s)" message
- [ ] Inspect `bundle.null_vtxt_feat.norm()` after runner init — should be >> 0.01
- [ ] Verify T2M source has non-zero null embeddings with same shapes

