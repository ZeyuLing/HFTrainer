# PRISM Attention Architecture - Complete Code References

## Critical File Locations and Line Numbers

### 1. Configuration Definition
**File**: `configs/prism/prism_1b_tp2m_1frame.py`
**Lines**: 17-43

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
        use_fp32_upcast_attention=True,  # ← LINE 34: KEY PARAMETER
        out_channels=16,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        joint_pos_mode="sequential",
        num_spectral_modes=4,
        spectral_scale=None,
        text_dim=4096,
    ),
    # ... other modules
)
```

**Key Facts**:
- Explicitly set to `True` in base config
- Not overridden in v3/v2/unified variants
- Controls softmax precision for all 30 transformer blocks

---

### 2. Model Initialization
**File**: `hftrainer/models/motion/prism/network/transformer_prism.py`
**Lines**: 133-222

#### Parameter Definition:
```python
# Line 154
use_fp32_upcast_attention: bool = True,  # ← DEFAULT VALUE
```

#### Block Construction:
```python
# Lines 195-209
self.blocks = nn.ModuleList(
    [
        WanTransformerBlockWithMask(
            inner_dim,                    # Line 198
            ffn_dim,                      # Line 199
            num_attention_heads,          # Line 200
            qk_norm,                      # Line 201
            cross_attn_norm,              # Line 202
            eps,                          # Line 203
            added_kv_proj_dim,            # Line 204
            use_fp32_upcast_attention,    # Line 205 ← CRITICAL: PASSED TO EACH BLOCK
        )
        for _ in range(num_layers)
    ]
)
```

**Key Facts**:
- Creates 30 blocks (num_layers=30 from config)
- Line 205 passes the parameter to each block
- No dynamic config dict unpacking — direct parameter passing
- Same parameter value for all blocks

---

### 3. Block Class Definition
**File**: `hftrainer/models/motion/prism/network/block_with_mask.py`
**Lines**: 76-146

#### Constructor:
```python
# Lines 76-146
def __init__(
    self,
    dim: int,                              # Line 78
    ffn_dim: int,                          # Line 79
    num_heads: int,                        # Line 80
    qk_norm: str = "rms_norm_across_heads", # Line 81
    cross_attn_norm: bool = False,         # Line 82
    eps: float = 1e-6,                     # Line 83
    added_kv_proj_dim: Optional[int] = None, # Line 84
    use_fp32_upcast_attention: bool = True,  # Line 85 ← DEFAULT: True
):
    super().__init__()
```

#### Self-Attention Processor Selection:
```python
# Lines 97-107
attn1_processor = (
    WanAttnProcessorFP32Upcast()           # Line 98 ← IF TRUE
    if use_fp32_upcast_attention 
    else WanAttnProcessor()                # Line 100 ← IF FALSE
)
self.attn1 = WanAttention(
    dim=dim,
    heads=num_heads,
    dim_head=dim // num_heads,
    eps=eps,
    cross_attention_dim_head=None,         # Self-attention marker
    processor=attn1_processor,             # Line 106 ← Processor selection
)
```

#### Cross-Attention Processor Selection:
```python
# Lines 113-124
attn2_processor = (
    WanAttnProcessorFP32Upcast()           # Line 114 ← IF TRUE
    if use_fp32_upcast_attention 
    else WanAttnProcessor()                # Line 116 ← IF FALSE
)
self.attn2 = WanAttention(
    dim=dim,
    heads=num_heads,
    dim_head=dim // num_heads,
    eps=eps,
    added_kv_proj_dim=added_kv_proj_dim,
    cross_attention_dim_head=dim // num_heads,  # Cross-attention marker
    processor=attn2_processor,             # Line 123 ← Processor selection
)
```

**Key Facts**:
- Parameter received as positional arg (position 8)
- Conditional logic at lines 97-99 and 113-115
- Affects BOTH self-attention and cross-attention
- Determines which attention processor is used

---

### 4. FP32 Upcast Processor
**File**: `hftrainer/models/motion/prism/network/attention_fp32_upcast.py`
**Lines**: 37-260

#### Class Definition:
```python
# Lines 37-72
class WanAttnProcessorFP32Upcast(WanAttnProcessor):
    _use_fp32_upcast = True                # Line 61
    _supported_precisions = (torch.float16, torch.bfloat16)  # Line 62
    
    def __init__(self, use_fp32_upcast: bool = True):
        super().__init__()
        self._use_fp32_upcast = use_fp32_upcast  # Line 72
```

#### Upcast Decision Logic:
```python
# Lines 104-118
autocast_fp16_active = (
    torch.is_autocast_enabled()
    and torch.get_autocast_gpu_dtype() == torch.float16
)
should_upcast = (
    self._use_fp32_upcast                  # Line 112
    and (
        hidden_states.dtype in self._supported_precisions  # Line 115
        or autocast_fp16_active            # Line 116
    )
)
```

#### Upcast Execution:
```python
# Lines 204-229
query_fp32 = query.to(torch.float32)       # Line 209
key_fp32 = key.to(torch.float32)           # Line 210
value_fp32 = value.to(torch.float32)       # Line 211
attn_mask_fp32 = attention_mask.to(torch.float32) if attention_mask is not None else None  # Line 214

with torch.cuda.amp.autocast(enabled=False):  # Line 216
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
hidden_states = hidden_states.to(original_dtype)  # Line 228
```

**Key Facts**:
- Applies to fp16 and bf16 inputs
- Also handles `autocast(fp16)` context
- Upcasts Q, K, V, and attention mask to fp32
- Disables autocast during SDPA to prevent re-downcasting
- Casts output back to original dtype

---

## Complete Config Inheritance Chain

### Chain with Line References:

1. **prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v3.py** (Line 27)
   - `_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py'`
   - Overrides: `module_dtype='fp32'`, `use_fp16_autocast=True`
   - Does NOT override `use_fp32_upcast_attention`

2. **prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached.py** (Line 15)
   - `_base_ = './prism_1b_tp2m_multiframe_kt_spectral_unified.py'`
   - Overrides: `tokenizer=None`, `text_encoder=None`
   - Does NOT override `use_fp32_upcast_attention`

3. **prism_1b_tp2m_multiframe_kt_spectral_unified.py** (Line 12)
   - `_base_ = './prism_1b_tp2m_multiframe.py'`
   - Overrides: `joint_pos_mode`, `num_spectral_modes`, `spectral_scale`
   - Does NOT override `use_fp32_upcast_attention`

4. **prism_1b_tp2m_multiframe.py** (Line 9)
   - `_base_ = './prism_1b_tp2m_1frame.py'`
   - Overrides: `condition_num_frames`, `frame_condition_rate`
   - Does NOT override `use_fp32_upcast_attention`

5. **prism_1b_tp2m_1frame.py** (Line 34)
   - ✓ **SETS: `use_fp32_upcast_attention=True`**
   - This is where the parameter is first (and only) defined
   - All configs above inherit this value

**Result**: All configs in the chain use `use_fp32_upcast_attention=True`

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ CONFIG FILE: prism_1b_tp2m_1frame.py (Line 34)                 │
│ use_fp32_upcast_attention=True                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │ (config dict passed to build())
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ PrismTransformerMotionModel.__init__()                          │
│ Line 154: use_fp32_upcast_attention: bool = True                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─ Store as self.config.use_fp32_upcast_attention
                     │
                     └─ For each of 30 layers (Line 207):
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ WanTransformerBlockWithMask.__init__()                          │
│ Line 205: use_fp32_upcast_attention passed as positional arg 8  │
│ Line 85: use_fp32_upcast_attention: bool = True (default)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ├─ Lines 97-99: if use_fp32_upcast_attention:
                     │    attn1_processor = WanAttnProcessorFP32Upcast()
                     │  else:
                     │    attn1_processor = WanAttnProcessor()
                     │
                     ├─ Lines 113-115: if use_fp32_upcast_attention:
                     │    attn2_processor = WanAttnProcessorFP32Upcast()
                     │  else:
                     │    attn2_processor = WanAttnProcessor()
                     │
                     ├─ Line 106: self.attn1 = WanAttention(..., processor=attn1_processor)
                     └─ Line 123: self.attn2 = WanAttention(..., processor=attn2_processor)
                        ↓
┌─────────────────────────────────────────────────────────────────┐
│ During block.forward() call:                                    │
│                                                                 │
│ attn1_processor.__call__() / attn2_processor.__call__()         │
│ WanAttnProcessorFP32Upcast.__call__()                          │
│ (from attention_fp32_upcast.py)                                 │
│                                                                 │
│ Lines 104-118: Detect if upcast needed (check dtype/autocast)  │
│ Lines 209-214: Upcast Q, K, V, mask to fp32                    │
│ Line 216: with torch.cuda.amp.autocast(enabled=False):         │
│   Line 217-226: Run attention in fp32                          │
│ Line 228: Cast output back to original dtype                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: Where `use_fp32_upcast_attention` is Used

| Location | Purpose | Impact |
|----------|---------|--------|
| Config line 34 | User-facing control | Determines behavior for all blocks |
| transformer_prism.py line 205 | Parameter passing | Passed to each of 30 blocks |
| block_with_mask.py line 97-99 | Self-attention processor | Chooses between upcast/standard |
| block_with_mask.py line 113-115 | Cross-attention processor | Chooses between upcast/standard |
| attention_fp32_upcast.py line 112 | Upcast gate | Controls actual fp32 conversion |
| attention_fp32_upcast.py line 209-228 | Upcast execution | Performs Q, K, V, mask conversion |

---

## Testing the Setting

To verify `use_fp32_upcast_attention` is working:

```python
# In training code:
model = build_model(config)  # config has use_fp32_upcast_attention=True

# Check first block
first_block = model.transformer.blocks[0]
print(type(first_block.attn1.processor))  
# Should print: <class 'WanAttnProcessorFP32Upcast'>

print(type(first_block.attn2.processor))  
# Should print: <class 'WanAttnProcessorFP32Upcast'>
```

To disable it in a custom config:
```python
transformer=dict(
    type="PrismTransformerMotionModel",
    # ... other params ...
    use_fp32_upcast_attention=False,  # ← Set to False to disable
)
```

Then verify:
```python
print(type(first_block.attn1.processor))  
# Should print: <class 'WanAttnProcessor'> (standard, no upcast)
```

---

## Precision Handling in v3 Config

**v3 Config Note** (`prism_1b_tp2m_multiframe_kt_spectral_unified_t5cached_v3.py`):

- **Line 32**: `module_dtype='fp32'` — keeps model parameters in fp32
- **Line 48**: `use_fp16_autocast=True` — enables manual fp16 autocast in trainer
- **Line 34** (inherited from base): `use_fp32_upcast_attention=True` — upcasts attention internally

This is a traditional AMP (Automatic Mixed Precision) recipe:
1. fp32 params (no FSDP bf16 issues)
2. fp16 autocast for linear ops (V100 tensor cores)
3. fp32 attention softmax (prevents overflow)
4. fp32 loss computation (stable gradients)

Result: All 30 blocks use `WanAttnProcessorFP32Upcast` for softmax stability.

