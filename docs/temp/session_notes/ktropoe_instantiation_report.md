# MotionWanRotaryPosEmbed Instantiation Analysis

## Summary
This report documents how `MotionWanRotaryPosEmbed` is instantiated in the PRISM transformer and how to pass new KT-RoPE parameters through the config system.

---

## 1. MotionWanRotaryPosEmbed Instantiation

### Location
**File**: `/apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/hftrainer/models/motion/prism/network/transformer_prism.py`

### Exact Instantiation Line
**Line 164-166** in `PrismTransformerMotionModel.__init__()`:
```python
self.rope = MotionWanRotaryPosEmbed(
    attention_head_dim, patch_size, rope_max_seq_len
)
```

### Parameters Passed
1. **`attention_head_dim`** (int)
   - Current value: `128` (from config)
   - Used to split between temporal and spatial dimensions
   - Split logic in `motion_rope.py` line 84-85:
     ```python
     j_dim = attention_head_dim // 2        # Spatial/joint dimension
     t_dim = attention_head_dim - j_dim     # Temporal dimension (gets remainder if odd)
     ```

2. **`patch_size`** (Tuple[int, int])
   - Current value: `(1, 1)` (from config)
   - Format: `(patch_frames, patch_joints)`
   - Used to compute number of patches in each dimension
   - Assertion on line 155: `assert patch_size[-1] == 1` (joint patching not supported)

3. **`rope_max_seq_len`** (int)
   - Current value: `1024` (from config)
   - Maximum sequence length for pre-computing RoPE frequencies

4. **`theta`** (float) - **Currently Hardcoded!**
   - Value: `10000.0`
   - This parameter is **NOT** configurable via the config system
   - Defined in `motion_rope.py` line 74 with default value

---

## 2. How MotionWanRotaryPosEmbed is Used

### In Forward Pass
**File**: `transformer_prism.py`, **Line 307** in `forward()`:
```python
rotary_emb = self.rope(hidden_states)
```

**Input**: `hidden_states` with shape `[B, C, T, J]` where:
- B = batch size
- C = channels
- T = num_frames
- J = num_joints

**Output**: Tuple `(freqs_cos, freqs_sin)` with shape `(1, N, 1, attention_head_dim)` where:
- N = (T // p_t) * (J // p_j) = sequence length after patchification

### Usage in Transformer Blocks
**Line 433-438**: Passed to each transformer block:
```python
hidden_states = torch.utils.checkpoint.checkpoint(
    block,
    hidden_states,
    encoder_hidden_states,
    timestep_proj,
    rotary_emb,  # ← Rotary embeddings passed here
    hidden_states_mask,
    encoder_hidden_states_mask,
    causal_mask,
    use_reentrant=False,
)
```

---

## 3. MotionWanRotaryPosEmbed Initialization Parameters

### `__init__()` Signature (motion_rope.py, lines 69-75)
```python
def __init__(
    self,
    attention_head_dim: int,
    patch_size: Tuple[int, int],
    max_seq_len: int,
    theta: float = 10000.0,  # ← DEFAULT PARAMETER
):
```

### Current Fixed Initialization (motion_rope.py, lines 96-106)
```python
for dim in [t_dim, j_dim]:
    freq_cos, freq_sin = get_1d_rotary_pos_embed(
        dim,
        max_seq_len,
        theta,  # ← Currently hardcoded as 10000.0
        use_real=True,
        repeat_interleave_real=True,
        freqs_dtype=freqs_dtype,
    )
```

---

## 4. Config File Structure

### Primary Config File
**Path**: `configs/prism/prism_1b_tp2m_1frame.py`

### Transformer Configuration Section (Lines 18-42)
```python
model = dict(
    type="PrismBundle",
    transformer=dict(
        type="PrismTransformerMotionModel",
        trainable=True,
        gradient_checkpointing=True,
        module_dtype="bf16",
        
        # RoPE Parameters:
        patch_size=(1, 1),              # ← Patchification
        rope_max_seq_len=1024,          # ← Max sequence length
        attention_head_dim=128,         # ← Head dimension (split between T and J)
        
        # Other model parameters
        cross_attn_norm=True,
        added_kv_proj_dim=None,
        eps=1e-6,
        ffn_dim=8960,
        freq_dim=256,
        in_channels=16,
        num_attention_heads=12,
        num_layers=30,
        out_channels=16,
        qk_norm="rms_norm_across_heads",
        
        # NEW KT-RoPE Parameters (Currently Added in Config But Not Used):
        joint_pos_mode="sequential",    # Options: "sequential", "spectral", "dfs"
        num_spectral_modes=4,           # For spectral mode
        spectral_scale=None,            # Scaling for spectral coordinates
        
        text_dim=4096,
    ),
    # ... vae, tokenizer, text_encoder, scheduler configs ...
)
```

### Multi-Frame Config
**Path**: `configs/prism/prism_1b_tp2m_multiframe.py`
- Inherits from `prism_1b_tp2m_1frame.py`
- Overrides trainer settings:
  ```python
  trainer = dict(
      condition_num_frames=[1, 5, 9],
      frame_condition_rate=0.1,
  )
  ```

---

## 5. Checkpoint Resume Paths

### Main Training
- **Working directory**: `work_dirs/prism_1b_tp2m_1frame/`
- **Pre-migrated checkpoint**: `work_dirs/prism_1b_tp2m_1frame/checkpoint-iter_11000/`
- **Resume command**: `bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_1frame.py --auto-resume`

### Multi-Frame Fine-tuning
- **Training directory**: `work_dirs/prism_1b_tp2m_multiframe/`
- **Resume command**: `bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_multiframe.py --auto-resume`

### MCM Training (Motion Control Module)
- **Load from**: `work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000`
- **Save to**: `work_dirs/prism_mcm_motionhub/`
- **Config file**: `configs/prism/prism_mcm_motionhub.py` (lines 1-11)
  ```python
  load_from = dict(
      _delete_=True,
      path="work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000",
      load_scope="model",
  )
  ```

---

## 6. How Config Parameters Flow to Instantiation

### Flow Diagram
```
config file (prism_1b_tp2m_1frame.py)
    ↓
transformer config dict {
    type: "PrismTransformerMotionModel",
    attention_head_dim: 128,
    patch_size: (1, 1),
    rope_max_seq_len: 1024,
    ...
}
    ↓
Model builder (via registry)
    ↓
PrismTransformerMotionModel.__init__()
    ↓
Line 164-166: self.rope = MotionWanRotaryPosEmbed(
    attention_head_dim,      # = 128
    patch_size,              # = (1, 1)
    rope_max_seq_len         # = 1024
)
```

### Config Parameters Received by `__init__` (lines 132-149)
```python
@register_to_config
def __init__(
    self,
    patch_size: Tuple[int] = (1, 1),
    num_attention_heads: int = 40,
    attention_head_dim: int = 128,
    in_channels: int = 16,
    out_channels: int = 16,
    text_dim: int = 4096,
    freq_dim: int = 256,
    ffn_dim: int = 13824,
    num_layers: int = 40,
    cross_attn_norm: bool = True,
    qk_norm: Optional[str] = "rms_norm_across_heads",
    eps: float = 1e-6,
    added_kv_proj_dim: Optional[int] = None,
    rope_max_seq_len: int = 1024,
    pos_embed_seq_len: Optional[int] = None,
) -> None:
```

---

## 7. How to Add New KT-RoPE Parameters

### Step 1: Update Config File
Add new parameters to `configs/prism/prism_1b_tp2m_1frame.py`:
```python
model = dict(
    type="PrismBundle",
    transformer=dict(
        type="PrismTransformerMotionModel",
        # ... existing params ...
        rope_max_seq_len=1024,
        
        # NEW KT-RoPE Parameters
        rope_theta=10000.0,              # Rotation frequency base
        joint_pos_mode="sequential",     # "sequential", "spectral", "dfs"
        num_spectral_modes=4,            # Number of Laplacian eigenvectors
        spectral_scale=None,             # Scaling factor for spectral coords
        kinematic_tree_path=None,        # Path to kinematic tree definition
        
        # ... other params ...
    ),
)
```

### Step 2: Update `PrismTransformerMotionModel.__init__`
Modify `transformer_prism.py` lines 132-149:
```python
@register_to_config
def __init__(
    self,
    # ... existing params ...
    rope_max_seq_len: int = 1024,
    
    # NEW KT-RoPE Parameters
    rope_theta: float = 10000.0,
    joint_pos_mode: str = "sequential",
    num_spectral_modes: int = 4,
    spectral_scale: Optional[int] = None,
    kinematic_tree_path: Optional[str] = None,
    
    pos_embed_seq_len: Optional[int] = None,
) -> None:
```

### Step 3: Pass Parameters to RoPE Instantiation
Modify `transformer_prism.py` lines 164-166:
```python
# RoPE for encoding temporal and joint positions
self.rope = MotionWanRotaryPosEmbed(
    attention_head_dim=attention_head_dim,
    patch_size=patch_size,
    rope_max_seq_len=rope_max_seq_len,
    theta=rope_theta,  # ← Pass theta
    joint_pos_mode=joint_pos_mode,  # ← New param
    num_spectral_modes=num_spectral_modes,  # ← New param
    spectral_scale=spectral_scale,  # ← New param
    kinematic_tree_path=kinematic_tree_path,  # ← New param
)
```

### Step 4: Update `MotionWanRotaryPosEmbed.__init__`
Modify `motion_rope.py` lines 69-75:
```python
def __init__(
    self,
    attention_head_dim: int,
    patch_size: Tuple[int, int],
    max_seq_len: int,
    theta: float = 10000.0,  # ← Now configurable
    joint_pos_mode: str = "sequential",
    num_spectral_modes: int = 4,
    spectral_scale: Optional[int] = None,
    kinematic_tree_path: Optional[str] = None,
):
```

---

## 8. Current RoPE Implementation Details

### Location
**File**: `motion_rope.py`

### Key Functions
1. **`__init__` (lines 69-111)**
   - Pre-computes 1D RoPE frequencies for temporal and spatial dimensions
   - Uses `get_1d_rotary_pos_embed()` from diffusers
   - Concatenates temporal and joint frequencies

2. **`forward` (lines 113-179)**
   - Takes input shape `[B, C, T, J]`
   - Computes number of patches: `ppf = T // p_t`, `ppj = J // p_j`
   - Slices and expands pre-computed frequencies
   - Returns `(freqs_cos, freqs_sin)` with shape `(1, N, 1, attention_head_dim)`

### Dimension Split Logic (lines 84-85)
```python
j_dim = attention_head_dim // 2    # Second half: spatial/joint
t_dim = attention_head_dim - j_dim # First half + remainder: temporal
```

For `attention_head_dim=128`:
- `j_dim = 64` (joint dimension)
- `t_dim = 64` (temporal dimension)

---

## 9. Integration with Transformer Blocks

### Where RoPE is Used
**File**: `network/block_with_mask.py` (referenced in transformer_prism.py line 75)

RoPE frequencies are passed to attention layers within each `WanTransformerBlockWithMask`.

### Causal Masking Integration
See `transformer_prism.py` lines 381-395 for frame-level causal attention mask that works alongside RoPE.

---

## 10. Quick Reference: What to Modify for KT-RoPE

| File | Lines | Change |
|------|-------|--------|
| `configs/prism/prism_1b_tp2m_1frame.py` | 36-40 | Add new KT-RoPE config params |
| `network/transformer_prism.py` | 132-149 | Add params to `__init__` signature |
| `network/transformer_prism.py` | 164-166 | Pass params to `MotionWanRotaryPosEmbed()` |
| `network/motion_rope.py` | 69-75 | Update `__init__` signature |
| `network/motion_rope.py` | 76-111 | Implement KT-RoPE logic |

---

## 11. Testing Locations

### Unit Test for RoPE
**File**: `motion_rope.py` lines 182-369
- Tests initialization
- Tests forward pass shapes
- Tests different patch sizes
- Tests frequency value ranges
- Tests consistency

### Integration Tests
**File**: `transformer_prism.py` lines 509-744
- Test 1: Basic forward pass (no mask)
- Test 2: With hidden_states_mask
- Test 3: With encoder_hidden_states_mask
- Test 4: With both masks
- Test 5: With patch_size > 1
- Test 6: Mask patchify logic

---

## 12. Related Architecture Components

### Condition Embeddings (Lines 175-181)
```python
self.condition_embedder = WanTimeTextEmbedding(...)
```
- Processes timestep and text conditioning
- Separate from RoPE but used in same forward pass

### Transformer Blocks (Lines 186-199)
```python
self.blocks = nn.ModuleList([
    WanTransformerBlockWithMask(...) 
    for _ in range(num_layers)
])
```
- 30 layers by default
- Each uses RoPE from `self.rope`

---

## Summary

**Key Takeaway**: To add new KT-RoPE parameters, you need to:

1. **Add to config file** with default values
2. **Update `PrismTransformerMotionModel.__init__`** signature to accept new params
3. **Pass params to `MotionWanRotaryPosEmbed()`** instantiation
4. **Update `MotionWanRotaryPosEmbed.__init__`** to accept and use new params
5. **Implement KT-RoPE logic** in the `forward()` method

The parameter `theta` (RoPE frequency base) is particularly important and currently hardcoded at `10000.0`. Making it configurable requires changes at all 4 points above.

