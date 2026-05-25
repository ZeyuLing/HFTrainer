# DTYPE Configuration for FSDP Training in hf_trainer (Comprehensive Summary)

## Executive Summary

Model dtype in hf_trainer is configured through a **three-layer system**:

1. **Config-level specification** (files like `prism_1b_tp2m_1frame.py`)
2. **Bundle-level instantiation** (`base_model_bundle.py` / `_build_modules()`)
3. **FSDP-level handling** (`accelerate_runner.py` with auto-fallback)

The current PRISM config uses **bf16 for transformer/text_encoder and fp32 for VAE**, with FSDP configured to run in `mixed_precision='no'` (no automatic mixed precision).

---

## 1. CONFIG-LEVEL DTYPE SPECIFICATION

### Current PRISM Config Chain

```
prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py
  ↓ _base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified.py'
prism_1b_tp2m_multiframe_kt_spectral_unified.py
  ↓ _base_ = './prism_1b_tp2m_multiframe.py'
prism_1b_tp2m_multiframe.py
  ↓ _base_ = './prism_1b_tp2m_1frame.py'
prism_1b_tp2m_1frame.py (ULTIMATE BASE CONFIG)
```

### Dtype Specifications in `prism_1b_tp2m_1frame.py` (Lines 23-97)

```python
model = dict(
    type='PrismBundle',
    
    transformer=dict(
        type='PrismTransformerMotionModel',
        from_pretrained={...},
        module_dtype="bf16",  # ← TRANSFORMER DTYPE: BFLOAT16
        ...
    ),
    
    vae=dict(
        type='AutoencoderKLPrism2DTK',
        from_pretrained={...},
        module_dtype="fp32",  # ← VAE DTYPE: FLOAT32
        ...
    ),
    
    text_encoder=dict(
        type='UMT5EncoderModel',
        from_pretrained={...},
        module_dtype="bf16",  # ← TEXT ENCODER DTYPE: BFLOAT16
        ...
    ),
    
    # Other modules (scheduler, tokenizer, smpl_pose_processor) 
    # do NOT have module_dtype specs
    ...
)
```

### Supported Dtype Aliases

From `base_model_bundle.py` lines 46-56:

```python
_DTYPE_ALIASES = {
    'fp32': torch.float32,
    'float32': torch.float32,
    'torch.float32': torch.float32,
    'fp16': torch.float16,
    'float16': torch.float16,
    'torch.float16': torch.float16,
    'bf16': torch.bfloat16,
    'bfloat16': torch.bfloat16,
    'torch.bfloat16': torch.bfloat16,
}
```

**Supported strings**: `'fp32'`, `'float32'`, `'fp16'`, `'float16'`, `'bf16'`, `'bfloat16'`, or a `torch.dtype` object

---

## 2. BUNDLE-LEVEL DTYPE APPLICATION

### Flow: Config → Instantiation → .to(dtype)

**File**: `base_model_bundle.py`, lines 394-492 (method `_build_modules()`)

#### Step 1: Extract dtype spec from config (Line 422)

```python
module_dtype_spec = sub_cfg.pop('module_dtype', None)
```

#### Step 2: Build the module (Line 451)

```python
module = HF_MODELS.build(sub_cfg)  # Instantiates model in default dtype (usually fp32)
```

#### Step 3: Resolve dtype string → torch.dtype (Lines 464-466)

```python
if isinstance(module, nn.Module):
    if module_dtype_spec is not None:
        module_dtype = self._resolve_module_dtype(module_dtype_spec)  # Line 465
        module = module.to(dtype=module_dtype)  # Line 466: Apply dtype casting
```

#### Dtype Resolution Logic (Lines 224-234)

```python
@classmethod
def _resolve_module_dtype(cls, dtype_spec) -> torch.dtype:
    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    if isinstance(dtype_spec, str):
        dtype = cls._DTYPE_ALIASES.get(dtype_spec)
        if dtype is not None:
            return dtype
    raise ValueError(
        "module_dtype must be one of: fp32, fp16, bf16, float32, float16, "
        "bfloat16, torch.float32, torch.float16, torch.bfloat16, or torch.dtype."
    )
```

### Key Points

- **When**: Module dtype is applied **immediately after model construction**, before trainable/frozen config, before LoRA, before gradient checkpointing
- **What**: Uses PyTorch's standard `.to(dtype=...)` which recursively converts all parameters and buffers
- **Type coercion**: ONLY works on `nn.Module` instances. For non-nn.Module objects (tokenizers, schedulers), an error is raised if dtype is specified (line 487-490)
- **Non-trainable modules**: Dtype applies regardless of `trainable=True/False`

---

## 3. ACCELERATOR AND FSDP CONFIGURATION

### FSDP Setup in accelerate_runner.py

**File**: `hftrainer/runner/accelerate_runner.py`, lines 204-246

#### Step 1: Extract accelerator config (Lines 205-208)

```python
accel_cfg = cfg.get('accelerator', {})
accel_cfg = copy.deepcopy(accel_cfg)
if hasattr(accel_cfg, 'to_dict'):
    accel_cfg = accel_cfg.to_dict()
```

#### Step 2: Build FSDP plugin (Lines 210-217)

```python
fsdp_plugin = None
fsdp_cfg = accel_cfg.pop('fsdp_plugin', None)
if fsdp_cfg is not None:
    from accelerate import FullyShardedDataParallelPlugin
    if hasattr(fsdp_cfg, 'to_dict'):
        fsdp_cfg = fsdp_cfg.to_dict()
    fsdp_plugin = FullyShardedDataParallelPlugin(**fsdp_cfg)
```

#### Step 3: Auto-fallback mechanism for unsupported bf16 (Lines 228-237)

```python
requested_mp = accel_cfg.get('mixed_precision', 'no')
if requested_mp == 'bf16':
    import torch as _torch
    if _torch.cuda.is_available() and not _torch.cuda.is_bf16_supported():
        logger.warning(
            "bf16 mixed precision requested but not supported on this device. "
            "Falling back to fp16."
        )
        accel_cfg['mixed_precision'] = 'fp16'
```

#### Step 4: Create Accelerator with plugins (Lines 239-246)

```python
accelerator = Accelerator(
    mixed_precision=accel_cfg.get('mixed_precision', 'no'),
    gradient_accumulation_steps=accel_cfg.get('gradient_accumulation_steps', 1),
    log_with=accel_cfg.get('log_with', 'tensorboard'),
    project_dir=run_dir,
    fsdp_plugin=fsdp_plugin,
    deepspeed_plugin=deepspeed_plugin,
)
```

### Current PRISM FSDP Configuration

From `prism_1b_tp2m_1frame.py` lines 158-173:

```python
accelerator = dict(
    mixed_precision="no",  # ← NO AUTO MIXED PRECISION
    gradient_accumulation_steps=1,
    fsdp_plugin=dict(
        sharding_strategy="FULL_SHARD",           # Shard all params across GPUs
        backward_prefetch="BACKWARD_PRE",         # Prefetch during backward
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",
        transformer_cls_names_to_wrap=["WanTransformerBlockWithMask"],
        state_dict_type="FULL_STATE_DICT",        # Save full params (not sharded)
        sync_module_states=True,                  # Sync init state across ranks
        use_orig_params=True,                     # Use original param references
        cpu_offload=False,                        # No CPU offloading
    ),
)
```

---

## 4. KEY FINDINGS: MODEL DTYPE VS MIXED PRECISION

### Model Dtype (Lines 23-97 in config)

- **Applied to**: Individual modules (transformer, vae, text_encoder)
- **Applied at**: Bundle instantiation time, via `.to(dtype=...)`
- **Scope**: Affects weights and computations of that specific module
- **Current values**:
  - `transformer` (PrismTransformerMotionModel): **bf16**
  - `vae` (AutoencoderKL): **fp32**
  - `text_encoder` (UMT5Encoder): **bf16**

### Mixed Precision (Lines 158-173 in config)

- **Applied to**: Entire Accelerator/FSDP context
- **Applied at**: During training loop via `accelerator.prepare()` and autocast contexts
- **Current value**: **'no'** (disabled)
- **How it works**:
  - `'no'`: All computations use model dtype as-is
  - `'fp16'`: Wrap computations in fp16 autocast (requires loss scaling)
  - `'bf16'`: Wrap computations in bf16 autocast (no loss scaling needed)

### The Relationship

```
Model Dtype (bf16 for transformer) + Mixed Precision (no)
  ↓
Transformer runs in bf16 natively, no additional autocast
  ↓
Loss computation, backward pass all happen in bf16
  ↓
Gradients are bf16, optimizer updates in bf16
```

---

## 5. HOW TO CHANGE DTYPE

### Option A: Change Model Dtype (Recommended for specific modules)

**File**: `configs/prism/prism_1b_tp2m_1frame.py` (base config)

```python
model = dict(
    type='PrismBundle',
    transformer=dict(
        type='PrismTransformerMotionModel',
        from_pretrained={...},
        module_dtype="fp16",  # ← CHANGE from "bf16" to "fp16"
        ...
    ),
    vae=dict(
        type='AutoencoderKLPrism2DTK',
        from_pretrained={...},
        # module_dtype="fp32",  # Keep as is
        ...
    ),
    ...
)
```

**Pros**:
- Precise control per module
- Can keep VAE in fp32 while transformer in fp16
- Matches PRISM's intended design

**Cons**:
- VAE already in fp32 (no impact for it)
- Still accumulates fp16 errors during generation

### Option B: Enable Mixed Precision at Accelerator Level (if device supports)

**File**: `configs/prism/prism_1b_tp2m_1frame.py` (base config)

```python
accelerator = dict(
    mixed_precision="fp16",  # ← CHANGE from "no" to "fp16"
    gradient_accumulation_steps=1,
    fsdp_plugin=dict(...),  # FSDP config unchanged
)
```

**Pros**:
- Lightweight change
- Consistent with HuggingFace ecosystem
- Auto-fallback already handles unsupported devices (lines 228-237)

**Cons**:
- VAE stays fp32 (good for numerical stability)
- Transformer computations wrapped in fp16 autocast
- Requires loss scaling for fp16 (Accelerate handles this automatically)

### Option C: Child Config Override (Don't modify base)

**File**: `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py`

```python
_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified.py'

model = dict(
    transformer=dict(
        module_dtype="fp16",  # Override base
    ),
)

accelerator = dict(
    mixed_precision="fp16",  # Override base
)
```

**Pros**:
- Non-destructive to base configs
- Easy to test variants
- Child config takes precedence over inherited values

**Cons**:
- Creates config fragment that needs inheritance to work

---

## 6. AUTO-FALLBACK MECHANISM

### What It Does (Lines 228-237 of accelerate_runner.py)

If Accelerator is requested with `mixed_precision='bf16'` but the device doesn't support bf16:

```python
# Check device capability
if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
    # Auto-downgrade to fp16
    accel_cfg['mixed_precision'] = 'fp16'
    logger.warning("bf16 mixed precision requested but not supported on this device. Falling back to fp16.")
```

### When It Applies

- **Only for Accelerator mixed_precision string** (`'bf16'` → `'fp16'`)
- **NOT for model module_dtype** (if config says `module_dtype="bf16"`, device will try to use bf16 regardless)

### How to Check Device Support

```python
import torch
print(f"BF16 supported on this GPU: {torch.cuda.is_bf16_supported()}")
# Output: True on A100, H100, etc. | False on V100, older GPUs

print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## 7. COMPLETE CONFIGURATION FLOW DIAGRAM

```
┌─────────────────────────────────────────────┐
│  prism_1b_tp2m_1frame.py                    │
│  ├─ model.transformer.module_dtype="bf16"   │
│  ├─ model.vae.module_dtype="fp32"           │
│  ├─ model.text_encoder.module_dtype="bf16"  │
│  └─ accelerator.mixed_precision="no"        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  accelerate_runner.py                       │
│  - Extract accelerator config               │
│  - Check bf16 device support (auto-fallback)│
│  - Create Accelerator instance              │
│  - Create FSDP plugin                       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  base_model_bundle._build_modules()         │
│  For each sub-module:                       │
│  1. Extract module_dtype spec               │
│  2. HF_MODELS.build(sub_cfg)                │
│  3. _resolve_module_dtype(spec)             │
│  4. module.to(dtype=resolved_dtype)         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  PrismBundle Instance                       │
│  ├─ transformer: bf16                       │
│  ├─ vae: fp32                               │
│  ├─ text_encoder: bf16                      │
│  └─ (FSDP wrapping applied by Accelerator)  │
└─────────────────────────────────────────────┘
```

---

## 8. PRACTICAL STEPS TO CHANGE DTYPE TO FP16

### Step 1: Identify the config file to modify

For PRISM:
```bash
# Current chain
configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py
  → prism_1b_tp2m_multiframe_kt_spectral_unified.py
  → prism_1b_tp2m_multiframe.py
  → prism_1b_tp2m_1frame.py  ← MODIFY THIS (base config)
```

### Step 2: Decide which dtype approach to use

**Option A (Module-level, recommended for PRISM)**:
```python
# Edit: configs/prism/prism_1b_tp2m_1frame.py
model = dict(
    transformer=dict(
        module_dtype="fp16",  # Changed from "bf16"
        ...
    ),
    # VAE stays fp32, text_encoder stays bf16
)
```

**Option B (Accelerator-level)**:
```python
# Edit: configs/prism/prism_1b_tp2m_1frame.py
accelerator = dict(
    mixed_precision="fp16",  # Changed from "no"
    ...
)
```

### Step 3: Create a test script

```python
import torch
from hftrainer.models.motion.prism.bundle import PrismBundle
from mmengine import Config

cfg = Config.fromfile('configs/prism/prism_1b_tp2m_1frame.py')
bundle = PrismBundle.from_config(cfg.model)

# Verify dtype
print(f"Transformer dtype: {next(bundle.transformer.parameters()).dtype}")
print(f"VAE dtype: {next(bundle.vae.parameters()).dtype}")
print(f"Text encoder dtype: {next(bundle.text_encoder.parameters()).dtype}")
```

### Step 4: Run training

```bash
python tools/train.py configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py
```

---

## 9. LIMITATIONS AND CONSTRAINTS

### Cannot Pass Custom MixedPrecision Policy

The Accelerator supports only **string values**:
- `'no'` (default)
- `'fp16'`
- `'bf16'`

**NOT supported**: Custom `MixedPrecision` policies via PyTorch AMP

```python
# ❌ This doesn't work in the current code
custom_policy = torch.amp.MixedPrecision(...)  
accelerator = Accelerator(mixed_precision=custom_policy)  # Error
```

**Why**: Accelerate's `Accelerator` class only accepts string names or None. Custom policies would need to be implemented in the training loop's autocast context manually.

### Module Dtype Only for nn.Module

Non-nn.Module objects (tokenizers, schedulers) cannot have `module_dtype` applied:

```python
# ❌ This raises ValueError
scheduler=dict(
    type='FlowMatchEulerDiscreteScheduler',
    module_dtype="fp32",  # Error: not an nn.Module
)

# ✅ This is fine (no dtype spec)
scheduler=dict(
    type='FlowMatchEulerDiscreteScheduler',
)
```

---

## 10. SUMMARY TABLE

| Aspect | Current Value | How to Change | File |
|--------|---------------|---------------|------|
| **Transformer dtype** | bf16 | Set `model.transformer.module_dtype="fp16"` | prism_1b_tp2m_1frame.py |
| **VAE dtype** | fp32 | Set `model.vae.module_dtype="fp16"` (not recommended) | prism_1b_tp2m_1frame.py |
| **Text encoder dtype** | bf16 | Set `model.text_encoder.module_dtype="fp16"` | prism_1b_tp2m_1frame.py |
| **Mixed precision** | no | Set `accelerator.mixed_precision="fp16"` | prism_1b_tp2m_1frame.py |
| **FSDP strategy** | FULL_SHARD | Change `fsdp_plugin.sharding_strategy` | prism_1b_tp2m_1frame.py |
| **Auto-fallback** | Enabled | (Auto, line 228-237 of accelerate_runner.py) | N/A |

---

## References

**Files Examined**:
- `configs/prism/prism_1b_tp2m_1frame.py` (lines 23-97, 158-173)
- `hftrainer/models/base_model_bundle.py` (lines 394-492, 224-234)
- `hftrainer/runner/accelerate_runner.py` (lines 204-246, 228-237)
- `configs/_base_/default_runtime.py` (lines 21-24)

**Key Methods**:
- `ModelBundle._build_modules()` — dtype application point
- `ModelBundle._resolve_module_dtype()` — dtype string resolution
- `accelerate_runner._create_accelerator()` — FSDP setup and bf16 auto-fallback

