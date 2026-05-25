# KT-RoPE Implementation Code Snippets

## Quick Implementation Guide

This file contains exact code snippets to add KT-RoPE parameter support to the PRISM transformer.

---

## 1. Config File Changes

### File: `configs/prism/prism_1b_tp2m_1frame.py`

**Location**: After line 35 (after `rope_max_seq_len=1024,`)

**Add these lines**:
```python
rope_theta=10000.0,              # Base frequency for rotary embeddings
joint_pos_mode="sequential",     # Position encoding: "sequential", "spectral", "dfs"
num_spectral_modes=4,            # Number of Laplacian eigenvectors (for spectral mode)
spectral_scale=None,             # Scaling for spectral coordinates (None = num_joints)
```

**Complete section should look like**:
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
        out_channels=16,
        qk_norm="rms_norm_across_heads",
        rope_max_seq_len=1024,
        
        # NEW KT-RoPE Parameters
        rope_theta=10000.0,
        joint_pos_mode="sequential",
        num_spectral_modes=4,
        spectral_scale=None,
        
        text_dim=4096,
    ),
    # ... rest of config
)
```

---

## 2. Transformer Model Changes

### File: `hftrainer/models/motion/prism/network/transformer_prism.py`

#### Change 2a: Update `__init__` signature

**Location**: Lines 132-149 in the `__init__` method

**Replace the current signature with**:
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
    # NEW KT-RoPE Parameters
    rope_theta: float = 10000.0,
    joint_pos_mode: str = "sequential",
    num_spectral_modes: int = 4,
    spectral_scale: Optional[int] = None,
) -> None:
```

#### Change 2b: Pass parameters to RoPE instantiation

**Location**: Lines 164-166

**Replace**:
```python
self.rope = MotionWanRotaryPosEmbed(
    attention_head_dim, patch_size, rope_max_seq_len
)
```

**With**:
```python
self.rope = MotionWanRotaryPosEmbed(
    attention_head_dim=attention_head_dim,
    patch_size=patch_size,
    rope_max_seq_len=rope_max_seq_len,
    theta=rope_theta,
    joint_pos_mode=joint_pos_mode,
    num_spectral_modes=num_spectral_modes,
    spectral_scale=spectral_scale,
)
```

---

## 3. RoPE Module Changes

### File: `hftrainer/models/motion/prism/network/motion_rope.py`

#### Change 3a: Update imports (if needed)

**Location**: Top of file after existing imports

```python
from typing import Tuple, Optional
import numpy as np
```

#### Change 3b: Update `__init__` signature

**Location**: Lines 69-75

**Replace the `__init__` method signature with**:
```python
def __init__(
    self,
    attention_head_dim: int,
    patch_size: Tuple[int, int],
    max_seq_len: int,
    theta: float = 10000.0,
    joint_pos_mode: str = "sequential",
    num_spectral_modes: int = 4,
    spectral_scale: Optional[int] = None,
):
```

#### Change 3c: Store KT-RoPE parameters

**Location**: After line 111 (after registering buffers), add**:
```python
        # Store configuration for KT-RoPE
        self.joint_pos_mode = joint_pos_mode
        self.num_spectral_modes = num_spectral_modes
        self.spectral_scale = spectral_scale
```

#### Change 3d: Implement KT-RoPE logic (if using non-sequential mode)

**Location**: In the `forward()` method, after line 178 (before returning)

**Add before the return statement**:
```python
        # Apply KT-RoPE transformations if using non-sequential mode
        if self.joint_pos_mode == "spectral":
            # TODO: Apply spectral decomposition to joint positions
            # This would modify freqs_cos_j and freqs_sin_j based on
            # Laplacian eigenvectors of the kinematic tree
            pass
        elif self.joint_pos_mode == "dfs":
            # TODO: Apply depth-first search ordering to joint positions
            # This would reorder joint indices based on kinematic tree traversal
            pass
        # "sequential" mode uses default behavior (no modification)
```

---

## 4. Testing Code

### Unit test for new parameters

**File**: Add to the `if __name__ == "__main__":` section in `motion_rope.py`

```python
# ==================== Test KT-RoPE Parameters ====================
print("\n[Test KT-RoPE] Testing new KT-RoPE parameters")
print("-" * 50)

rope_kt = MotionWanRotaryPosEmbed(
    attention_head_dim=attention_head_dim,
    patch_size=patch_size,
    max_seq_len=max_seq_len,
    theta=5000.0,  # Different theta
    joint_pos_mode="sequential",
    num_spectral_modes=8,
    spectral_scale=num_joints,
)

hidden_states = torch.randn(batch_size, num_channels, num_frames, num_joints)
cos_out, sin_out = rope_kt(hidden_states)

print(f"  theta: 5000.0 (changed from default 10000.0)")
print(f"  joint_pos_mode: sequential")
print(f"  num_spectral_modes: 8")
print(f"  spectral_scale: {num_joints}")
print(f"  Output shape: {cos_out.shape}")

assert cos_out.shape == (1, expected_seq_len, 1, attention_head_dim)
print("✓ KT-RoPE parameters test passed!")
```

---

## 5. Integration test in transformer

### File: In `transformer_prism.py` `if __name__ == "__main__":` section

**Add a new test**:
```python
# ==================== Test 7: KT-RoPE Parameters ====================
print("\n" + "-" * 40)
print("Test 7: KT-RoPE with different theta parameter")

model_kt_rope = PrismTransformerMotionModel(
    patch_size=(1, 1),
    attention_head_dim=128,
    cross_attn_norm=True,
    added_kv_proj_dim=None,
    eps=1e-6,
    ffn_dim=8960,
    freq_dim=256,
    in_channels=num_channels,
    num_attention_heads=12,
    num_layers=4,
    out_channels=num_channels,
    qk_norm="rms_norm_across_heads",
    rope_max_seq_len=1024,
    # NEW KT-RoPE Parameters
    rope_theta=5000.0,  # Different from default
    joint_pos_mode="sequential",
    num_spectral_modes=4,
    spectral_scale=None,
    text_dim=text_dim,
).to(device=device, dtype=dtype)
model_kt_rope.eval()

with torch.no_grad():
    hidden_states = torch.randn(
        batch_size, num_channels, num_frames, num_joints
    ).to(device=device, dtype=dtype)
    timestep = torch.tensor([0, 1]).to(device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(batch_size, text_seq_len, text_dim).to(
        device=device, dtype=dtype
    )
    
    output = model_kt_rope(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
    )
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"rope_theta: 5000.0")
    assert output.shape == hidden_states.shape, "Output shape mismatch!"
    print("✓ Test 7 passed!")
```

---

## 6. Command to Resume Training with New Config

```bash
# Make sure config is updated first, then:
bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_1frame.py --auto-resume
```

---

## 7. Verify Parameter Flow

### Debug script to verify parameters are passed correctly

**File**: Create `test_ktropoe_params.py`

```python
import torch
from hftrainer.models.motion.prism.network.transformer_prism import PrismTransformerMotionModel

# Create model with custom KT-RoPE parameters
model = PrismTransformerMotionModel(
    attention_head_dim=128,
    patch_size=(1, 1),
    rope_max_seq_len=1024,
    rope_theta=5000.0,
    joint_pos_mode="spectral",
    num_spectral_modes=8,
    spectral_scale=22,
    num_attention_heads=12,
    num_layers=2,  # Small for testing
    in_channels=16,
    text_dim=4096,
)

# Verify parameters were stored
print("Transformer Config:")
print(f"  rope_theta: {model.config.rope_theta}")
print(f"  joint_pos_mode: {model.config.joint_pos_mode}")
print(f"  num_spectral_modes: {model.config.num_spectral_modes}")
print(f"  spectral_scale: {model.config.spectral_scale}")

print("\nRoPE Module:")
print(f"  rope.joint_pos_mode: {model.rope.joint_pos_mode}")
print(f"  rope.num_spectral_modes: {model.rope.num_spectral_modes}")
print(f"  rope.spectral_scale: {model.rope.spectral_scale}")

# Test forward pass
batch_size = 2
num_channels = 16
num_frames = 16
num_joints = 22

hidden_states = torch.randn(batch_size, num_channels, num_frames, num_joints)
timestep = torch.tensor([0, 1])
text_states = torch.randn(batch_size, 10, 4096)

output = model(
    hidden_states=hidden_states,
    timestep=timestep,
    encoder_hidden_states=text_states,
)

print(f"\nForward pass successful!")
print(f"  Input shape: {hidden_states.shape}")
print(f"  Output shape: {output.shape}")
```

**Run with**:
```bash
python test_ktropoe_params.py
```

---

## 8. Configuration Comparison Table

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `attention_head_dim` | int | 128 | Head dimension (split: 64 temporal, 64 spatial) |
| `patch_size` | Tuple[int,int] | (1, 1) | Patch size (frames, joints) |
| `rope_max_seq_len` | int | 1024 | Max sequence length for RoPE precomputation |
| **`rope_theta`** | float | **10000.0** | **Base frequency for rotary embeddings** |
| **`joint_pos_mode`** | str | **"sequential"** | **Position encoding: "sequential", "spectral", "dfs"** |
| **`num_spectral_modes`** | int | **4** | **Number of Laplacian eigenvectors (spectral mode)** |
| **`spectral_scale`** | Optional[int] | **None** | **Scaling for spectral coordinates** |

*Bold = New KT-RoPE parameters*

---

## 9. Troubleshooting

### Issue: "TypeError: __init__() got unexpected keyword argument 'rope_theta'"

**Solution**: Make sure you updated the `__init__` signature in `transformer_prism.py` (Step 2a above)

### Issue: "AttributeError: 'MotionWanRotaryPosEmbed' object has no attribute 'joint_pos_mode'"

**Solution**: Make sure you added the parameter storage code (Step 3c above) in the RoPE `__init__`

### Issue: Parameters not being loaded from config

**Solution**: 
1. Check that `@register_to_config` decorator is present (line 131 in transformer_prism.py)
2. Verify config file has the new parameters
3. Clear any cached configs: `rm -rf ~/.cache/huggingface/*`

---

## 10. Next Steps

After implementing these changes:

1. **Run unit tests**:
   ```bash
   cd hftrainer/models/motion/prism/network
   python motion_rope.py  # Run RoPE unit tests
   python transformer_prism.py  # Run transformer integration tests
   ```

2. **Run verification script**:
   ```bash
   python test_ktropoe_params.py
   ```

3. **Start training**:
   ```bash
   bash tools/taiji_dist_train.sh configs/prism/prism_1b_tp2m_1frame.py --auto-resume
   ```

4. **Implement full KT-RoPE logic** (if using spectral/dfs modes):
   - Modify `motion_rope.py` forward() method to apply actual kinematic tree-based transformations
   - Add kinematic tree loading and processing
   - Implement Laplacian eigenvector computation (for spectral mode)
   - Implement DFS traversal ordering (for dfs mode)

