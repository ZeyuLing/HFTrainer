# PRISM Transformer Attention Architecture Trace

## Overview
This document traces how the PRISM transformer model constructs its attention blocks, focusing on the `use_fp32_upcast_attention` parameter and its propagation through the architecture.

---

## Part 1: Configuration Inheritance Chain

### Config File Path: `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v3.py`
**File**: `/repo/configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v3.py`
**Lines**: 1-64

**Base**: `./prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py` (line 27)

Key settings in v3:
- **Line 32**: Sets `module_dtype='fp32'` for transformer (traditional AMP recipe)
- **Line 48**: Sets `use_fp16_autocast=True` in trainer
- Purpose: V100 GPU compatibility with traditional AMP (not FSDP)

---

### Config File Path: `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py`
**File**: `/repo/configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py`
**Lines**: 1-73

**Base**: `./prism_1b_tp2m_multiframe_kt_spectral_unified.py` (line 15)

Key settings:
- Disables text encoder/tokenizer (uses pre-extracted T5 features instead)
- Loads `LoadPreExtractedT5Feature` instead of online T5 encoding
- Uses pre-extracted null embeddings for prompt dropout

---

### Config File Path: `configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py`
**File**: `/repo/configs/prism/prism_1b_tp2m_multiframe_kt_spectral_unified.py`
**Lines**: 1-66

**Base**: `./prism_1b_tp2m_multiframe.py` (line 12)

Key settings:
- **Lines 14-19**: Transformer config with KT-RoPE spectral_unified mode
- **Line 41**: FSDP auto_wrap_policy wraps `WanTransformerBlockWithMask`
- Still inherits transformer model definition from base

---

### Config File Path: `configs/prism/prism_1b_tp2m_multiframe.py`
**File**: `/repo/configs/prism/prism_1b_tp2m_multiframe.py`
**Lines**: 1-15

**Base**: `./prism_1b_tp2m_1frame.py` (line 9)

Key settings:
- Adds multi-frame conditioning: `condition_num_frames=[1, 5, 9]`
- Inherits core transformer definition from base

---

### Config File Path: `configs/prism/prism_1b_tp2m_1frame.py` ← MAIN MODEL DEFINITION
**File**: `/repo/configs/prism/prism_1b_tp2m_1frame.py`
**Lines**: 1-185

**THIS IS THE FULL TRANSFORMER CONFIGURATION**

#### Transformer Model Configuration (Lines 17-43):

```python
model = dict(
    type="PrismBundle",
    transformer=dict(
        type="PrismTransformerMotionModel",
        trainable=True,
        gradient_checkpointing=True,
        module_dtype="bf16",
        patch_size=(1, 1),
        attention_head_dim=128,
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=16,
        num_attention_heads=12,
        num_layers=30,
        use_fp32_upcast_attention=True,  # ← KEY PARAMETER
        out_channels=16,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        # KT-RoPE: Kinematic-Topology Rotary Position Embedding
        joint_pos_mode="sequential",  # Options: "sequential", "spectral", "dfs"
        num_spectral_modes=4,  # Number of Laplacian eigenvector modes (if spectral)
        spectral_scale=None,  # Scaling for spectral coordinates (None = num_joints)
        text_dim=4096,
    ),
    # ... other modules (VAE, tokenizer, text_encoder, scheduler, etc.)
)
```

**Critical Finding**: 
- **Line 34**: `use_fp32_upcast_attention=True` is explicitly set in the config
- This is the **PRIMARY** control for softmax overflow prevention
- **Default value in code**: True (defined in `PrismTransformerMotionModel.__init__`)

---

## Part 2: Model Implementation

### File: `hftrainer/models/motion/prism/network/transformer_prism.py`

**Class**: `PrismTransformerMotionModel(WanTransformer3DModel)`
**Location**: Lines 82-516

#### `__init__` Method (Lines 133-222):

```python
def __init__(
    self,
    # ... other params ...
    use_fp32_upcast_attention: bool = True,  # ← DEFAULT: True
) -> None:
```

**Line 154**: Parameter defined with **default=True**

#### Attention Block Construction (Lines 195-209):

```python
self.blocks = nn.ModuleList(
    [
        WanTransformerBlockWithMask(
            inner_dim,              # = num_attention_heads * attention_head_dim
            ffn_dim,
            num_attention_heads,
            qk_norm,
            cross_attn_norm,
            eps,
            added_kv_proj_dim,
            use_fp32_upcast_attention,  # ← PASSED DIRECTLY TO EACH BLOCK
        )
        for _ in range(num_layers)  # 30 blocks created
    ]
)
```

**Finding**: 
- **Line 205**: The `use_fp32_upcast_attention` parameter is **explicitly passed** to each `WanTransformerBlockWithMask` instance
- This happens for **ALL 30 blocks** (num_layers=30)
- The value comes directly from the config via the model's constructor parameter

#### Forward Pass (Lines 236-516):

The forward method doesn't directly create attention blocks — it just uses them:

```python
for block in self.blocks:
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        # Gradient checkpointing
        hidden_states = torch.utils.checkpoint.checkpoint(
            block,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            rotary_emb,
            hidden_states_mask,
            encoder_hidden_states_mask,
            causal_mask,
            use_reentrant=False,
        )
    else:
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=timestep_proj,
            rotary_emb=rotary_emb,
            hidden_states_mask=hidden_states_mask,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            causal_mask=causal_mask,
        )
```

**Finding**: Blocks are called with mask support (lines 450-458)

---

## Part 3: Block Implementation

### File: `hftrainer/models/motion/prism/network/block_with_mask.py`

**Class**: `WanTransformerBlockWithMask(nn.Module)`
**Location**: Lines 32-277

#### `__init__` Method (Lines 76-146):

```python
def __init__(
    self,
    dim: int,
    ffn_dim: int,
    num_heads: int,
    qk_norm: str = "rms_norm_across_heads",
    cross_attn_norm: bool = False,
    eps: float = 1e-6,
    added_kv_proj_dim: Optional[int] = None,
    use_fp32_upcast_attention: bool = True,  # ← DEFAULT: True (Line 85)
):
```

**Line 85**: Parameter default is **True**

#### Self-Attention Setup (Lines 89-107):

```python
attn1_processor = (
    WanAttnProcessorFP32Upcast()         # ← If use_fp32_upcast_attention is True
    if use_fp32_upcast_attention 
    else WanAttnProcessor()               # ← Otherwise use standard processor
)
self.attn1 = WanAttention(
    dim=dim,
    heads=num_heads,
    dim_head=dim // num_heads,
    eps=eps,
    cross_attention_dim_head=None,  # Self-attention
    processor=attn1_processor,       # ← Processor with or without FP32 upcast
)
```

**Finding**: 
- **Line 97-99**: If `use_fp32_upcast_attention=True`, uses `WanAttnProcessorFP32Upcast()`
- **Line 98**: If False, uses standard `WanAttnProcessor()`
- This is the **conditional branch** where the parameter has its effect

#### Cross-Attention Setup (Lines 109-124):

```python
attn2_processor = (
    WanAttnProcessorFP32Upcast()         # ← If use_fp32_upcast_attention is True
    if use_fp32_upcast_attention 
    else WanAttnProcessor()               # ← Otherwise use standard processor
)
self.attn2 = WanAttention(
    dim=dim,
    heads=num_heads,
    dim_head=dim // num_heads,
    eps=eps,
    added_kv_proj_dim=added_kv_proj_dim,  # For I2V
    cross_attention_dim_head=dim // num_heads,  # Enables cross-attention
    processor=attn2_processor,             # ← Processor with or without FP32 upcast
)
```

**Finding**: 
- **Lines 113-115**: Same conditional logic for cross-attention
- Both self-attention AND cross-attention respect the `use_fp32_upcast_attention` flag

---

## Part 4: FP32 Upcast Implementation

### File: `hftrainer/models/motion/prism/network/attention_fp32_upcast.py`

**Class**: `WanAttnProcessorFP32Upcast(WanAttnProcessor)`
**Location**: Lines 37-260

#### Design (Lines 37-72):

```python
class WanAttnProcessorFP32Upcast(WanAttnProcessor):
    """
    WanAttnProcessor with automatic fp32 upscaling for softmax stability.
    """
    
    _use_fp32_upcast = True
    _supported_precisions = (torch.float16, torch.bfloat16)  # ← Applies to both fp16 and bf16
    
    def __init__(self, use_fp32_upcast: bool = True):
        super().__init__()
        self._use_fp32_upcast = use_fp32_upcast
```

#### Execution in `__call__` Method (Lines 74-239):

**Key Logic (Lines 104-118)**:

```python
autocast_fp16_active = (
    torch.is_autocast_enabled()
    and torch.get_autocast_gpu_dtype() == torch.float16
)
should_upcast = (
    self._use_fp32_upcast
    and (
        hidden_states.dtype in self._supported_precisions
        or autocast_fp16_active
    )
)
```

**Upcast path (Lines 204-229)**:

```python
query_fp32 = query.to(torch.float32)
key_fp32 = key.to(torch.float32)
value_fp32 = value.to(torch.float32)
attn_mask_fp32 = attention_mask.to(torch.float32) if attention_mask is not None else None

with torch.cuda.amp.autocast(enabled=False):  # ← Disable autocast to preserve fp32
    hidden_states = dispatch_attention_fn(
        query_fp32,
        key_fp32,
        value_fp32,
        attn_mask=attn_mask_fp32,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
# Cast back to original dtype
hidden_states = hidden_states.to(original_dtype)
```

**Finding**: 
- Upcasts Q, K, V to fp32 for attention computation
- Disables autocast during SDPA to preserve fp32 precision
- Casts output back to original dtype (fp16 or bf16)
- Applies to both main attention and image branch (for I2V)
- Attention mask is also upcast to fp32

---

## Summary Table

| Component | File | Line(s) | Key Finding |
|-----------|------|---------|-------------|
| **Config** | `prism_1b_tp2m_1frame.py` | 34 | `use_fp32_upcast_attention=True` explicitly set |
| **Model Init** | `transformer_prism.py` | 154 | Default parameter: `bool = True` |
| **Block Construction** | `transformer_prism.py` | 205 | Parameter passed to ALL 30 blocks |
| **Block Init** | `block_with_mask.py` | 85 | Default parameter: `bool = True` |
| **Self-Attention** | `block_with_mask.py` | 97-99 | Conditional processor selection |
| **Cross-Attention** | `block_with_mask.py` | 113-115 | Conditional processor selection |
| **FP32 Upcast** | `attention_fp32_upcast.py` | 204-229 | Actual upcast implementation |

---

## Answer to Specific Questions

### Q1: How does `PrismTransformerMotionModel` construct transformer blocks?

**A**: 
- Uses `nn.ModuleList` with list comprehension (lines 195-209)
- Creates 30 `WanTransformerBlockWithMask` instances (one per layer)
- Passes 8 parameters to each block, including `use_fp32_upcast_attention`
- All blocks are stored in `self.blocks` for use in forward pass

### Q2: Does it use `WanTransformerBlockWithMask`? Does it pass `use_fp32_upcast_attention`?

**A**: 
- ✓ **YES** - Uses `WanTransformerBlockWithMask` (line 197)
- ✓ **YES** - Explicitly passes `use_fp32_upcast_attention` (line 205)
- No wrapper, kwargs, or defaults — direct parameter passing

### Q3: Full config inheritance chain for `use_fp32_upcast_attention`

**A**: 
```
prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v3.py
  ↓ (_base_ at line 27)
prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py
  ↓ (_base_ at line 15)
prism_1b_tp2m_multiframe_kt_spectral_unified.py
  ↓ (_base_ at line 12)
prism_1b_tp2m_multiframe.py
  ↓ (_base_ at line 9)
prism_1b_tp2m_1frame.py  ← SETS use_fp32_upcast_attention=True (line 34)
```

**No override in the three v3/v2/unified configs** — all inherit the explicit `True` from base.

### Q4: Does config rely on default `True` or is it explicit?

**A**: 
- **Explicit**: Set in `prism_1b_tp2m_1frame.py` line 34
- Not relying on code defaults — it's in the config dict
- v3 config doesn't override, so v3 also gets `use_fp32_upcast_attention=True`

### Q5: How are `WanTransformerBlockWithMask` instances created?

**A**: 
- **Direct instantiation**, no kwargs unpacking (lines 197-206)
- Parameters are passed positionally and by name
- Same parameters for all 30 blocks
- No dynamic config forwarding

---

## Architecture Diagram

```
Config: prism_1b_tp2m_1frame.py
    ↓
    use_fp32_upcast_attention: True (line 34)
    ↓
PrismTransformerMotionModel.__init__()
    ↓ (line 205)
    for _ in range(num_layers=30):  
        WanTransformerBlockWithMask(
            ...
            use_fp32_upcast_attention=True  ← passed here
        )
    ↓
WanTransformerBlockWithMask.__init__()
    ↓ (lines 97-99)
    if use_fp32_upcast_attention:
        attn1_processor = WanAttnProcessorFP32Upcast()
    else:
        attn1_processor = WanAttnProcessor()
    ↓ (lines 113-115)
    if use_fp32_upcast_attention:
        attn2_processor = WanAttnProcessorFP32Upcast()
    else:
        attn2_processor = WanAttnProcessor()
    ↓
    self.attn1 = WanAttention(..., processor=attn1_processor)
    self.attn2 = WanAttention(..., processor=attn2_processor)
    ↓
During forward():
    WanAttnProcessorFP32Upcast.__call__()
        ↓ (lines 204-229)
        Upcast Q, K, V to fp32
        Disable autocast
        Run attention in fp32
        Cast output back to original dtype
```

