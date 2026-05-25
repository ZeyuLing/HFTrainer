# DTYPE Configuration — Quick Reference Guide

## Quick Answer: Where Are Dtypes Configured?

**Config Level**: `configs/prism/prism_1b_tp2m_1frame.py` (base config)
- Lines 23-97: `model.{transformer,vae,text_encoder}.module_dtype`
- Lines 158-173: `accelerator.mixed_precision`

**Code Level**: 
- Bundle instantiation: `hftrainer/models/base_model_bundle.py` lines 464-466
- FSDP setup: `hftrainer/runner/accelerate_runner.py` lines 228-237

---

## Current PRISM Dtype Configuration

```
transformer:    bf16 (bfloat16)
vae:            fp32 (float32)
text_encoder:   bf16 (bfloat16)
mixed_precision: no  (disabled)
```

---

## Change Transformer from BF16 to FP16

### Quick Edit (Config File)

**File**: `configs/prism/prism_1b_tp2m_1frame.py`

Find this:
```python
transformer=dict(
    type='PrismTransformerMotionModel',
    from_pretrained={...},
    module_dtype="bf16",  # ← Line 23
    ...
)
```

Change to:
```python
transformer=dict(
    type='PrismTransformerMotionModel',
    from_pretrained={...},
    module_dtype="fp16",  # ← CHANGED
    ...
)
```

---

## Enable Mixed Precision (FP16)

### Quick Edit (Config File)

**File**: `configs/prism/prism_1b_tp2m_1frame.py`

Find this:
```python
accelerator = dict(
    mixed_precision="no",  # ← Line 160
    ...
)
```

Change to:
```python
accelerator = dict(
    mixed_precision="fp16",  # ← CHANGED
    ...
)
```

**Note**: Device without BF16 support will auto-fallback to FP16 if you use `mixed_precision="bf16"` (see accelerate_runner.py lines 228-237).

---

## Verify Dtype After Loading

```python
import torch
from hftrainer.models.motion.prism.bundle import PrismBundle
from mmengine import Config

# Load config
cfg = Config.fromfile('configs/prism/prism_1b_tp2m_1frame.py')
bundle = PrismBundle.from_config(cfg.model)

# Check dtypes
print("Transformer:", next(bundle.transformer.parameters()).dtype)
print("VAE:", next(bundle.vae.parameters()).dtype)
print("Text Encoder:", next(bundle.text_encoder.parameters()).dtype)

# Expected output:
# Transformer: torch.bfloat16  (or torch.float16 if you changed it)
# VAE: torch.float32
# Text Encoder: torch.bfloat16  (or torch.float16 if you changed it)
```

---

## Supported Dtype Values

| Alias | PyTorch Type | Usage |
|-------|--------------|-------|
| `'fp32'` or `'float32'` | `torch.float32` | ✅ Use |
| `'fp16'` or `'float16'` | `torch.float16` | ✅ Use |
| `'bf16'` or `'bfloat16'` | `torch.bfloat16` | ✅ Use |
| `torch.float32` | torch.float32 | ✅ Use |
| `torch.float16` | torch.float16 | ✅ Use |
| `torch.bfloat16` | torch.bfloat16 | ✅ Use |

---

## Device BF16 Support Check

```python
import torch

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    has_bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {gpu_name}")
    print(f"BF16 Support: {has_bf16}")
else:
    print("CUDA not available")

# Expected:
# GPU: NVIDIA A100-SXM4-40GB  (or similar)
# BF16 Support: True  (A100, H100) or False (V100, older)
```

---

## Model Dtype vs Mixed Precision

| Feature | Scope | Applied At | Example |
|---------|-------|------------|---------|
| **Model dtype** | Per-module | Bundle init | `module_dtype="bf16"` |
| **Mixed precision** | Entire training | Accelerator init | `mixed_precision="bf16"` |

**Both are independent** — you can have:
- Model bf16 + Mixed precision no ← Current PRISM
- Model bf16 + Mixed precision fp16
- Model fp32 + Mixed precision fp16
- etc.

---

## Practical Scenarios

### Scenario 1: GPU doesn't support BF16 (e.g., V100)

**Problem**: Config says `module_dtype="bf16"` but GPU doesn't support it

**Solution**:
```python
# Option A: Change config to fp16
model = dict(
    transformer=dict(
        module_dtype="fp16",  # Changed from "bf16"
    ),
)

# Option B: Let auto-fallback handle mixed_precision
accelerator = dict(
    mixed_precision="bf16",  # Auto-fallback will convert to "fp16"
)
```

### Scenario 2: Want to use all FP16 training

**Change in config**:
```python
# Option 1: Module-level
model = dict(
    transformer=dict(module_dtype="fp16"),
    text_encoder=dict(module_dtype="fp16"),
    # vae stays fp32 for numerical stability
)

# Option 2: Accelerator-level (simpler)
accelerator = dict(
    mixed_precision="fp16",
)
```

### Scenario 3: Keep VAE in FP32, transformer in FP16

**Already done in current PRISM config** — VAE has `module_dtype="fp32"`, transformer has `module_dtype="bf16"`. Just change transformer to FP16:

```python
model = dict(
    transformer=dict(module_dtype="fp16"),  # Changed
    # vae stays fp32, text_encoder can be fp16 or bf16
)
```

---

## Common Mistakes

❌ **Mistake 1**: Specifying dtype on non-nn.Module
```python
scheduler=dict(
    type='SomeScheduler',
    module_dtype="fp32",  # ERROR: not an nn.Module
)
```
✅ **Fix**: Only use `module_dtype` on nn.Module objects (transformers, encoders, etc.)

---

❌ **Mistake 2**: Assuming mixed_precision applies to model_dtype
```python
# These are INDEPENDENT
model.transformer.module_dtype="bf16"  # Module is bf16
accelerator.mixed_precision="fp32"     # But accelerator uses fp32 autocast
# Result: Conflict and unexpected behavior
```
✅ **Fix**: Understand model_dtype and mixed_precision work at different levels

---

❌ **Mistake 3**: Not checking device support before using bf16
```python
# Config always says bf16, but:
# - V100, older GPUs: no bf16 support → runtime error
# - A100, H100: bf16 support → works fine
```
✅ **Fix**: Check with `torch.cuda.is_bf16_supported()` or rely on auto-fallback

---

## Key Files for Reference

| File | Lines | What |
|------|-------|------|
| `configs/prism/prism_1b_tp2m_1frame.py` | 23-97 | Model dtype config |
| `configs/prism/prism_1b_tp2m_1frame.py` | 158-173 | FSDP + mixed precision config |
| `hftrainer/models/base_model_bundle.py` | 422-466 | How dtype is applied |
| `hftrainer/models/base_model_bundle.py` | 46-56 | Dtype alias mapping |
| `hftrainer/models/base_model_bundle.py` | 224-234 | Dtype resolution logic |
| `hftrainer/runner/accelerate_runner.py` | 228-237 | Auto-fallback for unsupported bf16 |

---

## Config Inheritance Path

When using `prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py`:

```
prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py  (T5-cached variant)
  ↓ inherits _base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified.py'
prism_1b_tp2m_multiframe_kt_spectral_unified.py  (KT-RoPE spectral)
  ↓ inherits _base_ = './prism_1b_tp2m_multiframe.py'
prism_1b_tp2m_multiframe.py  (Multi-frame)
  ↓ inherits _base_ = './prism_1b_tp2m_1frame.py'
prism_1b_tp2m_1frame.py  ← BASE CONFIG (contains dtype specs)
```

**To modify dtype**:
- Option A: Edit the base config `prism_1b_tp2m_1frame.py` (affects all variants)
- Option B: Override in child config `prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py` (only this variant)

---

## Testing Your Changes

```bash
# 1. Verify dtype in config
python3 -c "from mmengine import Config; cfg = Config.fromfile('configs/prism/prism_1b_tp2m_1frame.py'); print(cfg.model.transformer.module_dtype)"

# 2. Load bundle and check actual dtype
python3 << 'PYEOF'
import torch
from hftrainer.models.motion.prism.bundle import PrismBundle
from mmengine import Config

cfg = Config.fromfile('configs/prism/prism_1b_tp2m_1frame.py')
bundle = PrismBundle.from_config(cfg.model)

# Print actual dtypes
for name in ['transformer', 'vae', 'text_encoder']:
    module = getattr(bundle, name, None)
    if module is not None:
        dtype = next(module.parameters()).dtype
        print(f"{name:15s}: {dtype}")
PYEOF

# 3. Run training to verify it works
python tools/train.py configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py --gpu 0
```

---

## One-Liner: Change All to FP16

Edit `configs/prism/prism_1b_tp2m_1frame.py`:

```python
# Replace this section (around line 23-62):
transformer=dict(..., module_dtype="bf16", ...),
text_encoder=dict(..., module_dtype="bf16", ...),

# With this:
transformer=dict(..., module_dtype="fp16", ...),  # ← bf16 → fp16
text_encoder=dict(..., module_dtype="fp16", ...),  # ← bf16 → fp16

# And optionally (around line 160):
# Replace:
accelerator = dict(mixed_precision="no", ...)
# With:
accelerator = dict(mixed_precision="fp16", ...)
```

