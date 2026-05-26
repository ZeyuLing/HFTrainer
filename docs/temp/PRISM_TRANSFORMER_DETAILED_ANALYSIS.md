# PRISM Transformer Motion Model - Detailed Technical Analysis

**Date**: May 27, 2026  
**Scope**: Forward pass implementation with spectral RoPE and per-token timesteps  
**Model**: `PrismTransformerMotionModel` in `hftrainer/models/motion/prism/network/transformer_prism.py`

---

## Executive Summary

The PRISM transformer processes motion sequences [B, C, T, J] through a sophisticated pipeline that combines:

1. **Patch embedding** - spatial tokenization along temporal and joint dimensions
2. **Spectral RoPE** - kinematic tree-aware rotary position embeddings
3. **Per-token timesteps** - Wan 2.2 TI2V mode for frame-level diffusion control
4. **Adaptive layer normalization** - timestep-modulated output scaling

This analysis covers the complete forward pass (lines 236-517) with detailed shape transformations and implementation details.

---

## Part 1: Forward Method Overview

### Location
- **File**: `hftrainer/models/motion/prism/network/transformer_prism.py`
- **Lines**: 236-517
- **Class**: `PrismTransformerMotionModel`

### 13-Step Processing Pipeline

```
Input Motion [B, C, T, J]
     ↓
[1] Extract dimensions & patch params
     ↓
[2] Compute RoPE (joint_pos_mode='spectral_unified')
     ↓
[3] Patch embedding: [B, C, T, J] → [B, N, inner_dim]
     ↓
[4] Process hidden_states_mask (motion masking)
     ↓
[5] Process encoder_hidden_states_mask (text masking)
     ↓
[5b] Build causal attention mask (if is_causal=True)
     ↓
[6] Process timestep & text conditions
     ↓
[7] Transformer blocks loop (self-attn + cross-attn + FFN)
     ↓
[8] Output adaptive layer norm
     ↓
[9] Output projection
     ↓
[10] Unpatchify: [B, N, C*p_t*p_j] → [B, C, T, J]
     ↓
Output Motion [B, C, T, J]
```

---

## Part 2: Detailed Shape Transformations

### Step 1: Dimension Extraction (Lines 304-311)

```python
batch_size, num_channels, num_frames, num_joints = hidden_states.shape
p_t, p_j = self.config.patch_size  # typically (2, 1)
post_patch_num_frames = num_frames // p_t     # T' = T / p_t
post_patch_num_joints = num_joints // p_j     # J' = J / p_j
```

**Example** (typical PRISM config):
- Input: `hidden_states = [2, 16, 256, 22]`  (batch=2, channels=16, frames=256, joints=22)
- Patch size: `(2, 1)`
- Output: `T' = 256/2 = 128`, `J' = 22/1 = 22`
- Total tokens: `N = 128 × 22 = 2816`

### Step 2: RoPE Computation (Line 317)

```python
rotary_emb = self.rope(hidden_states)
```

See Part 3 below for detailed spectral RoPE computation.

**Output shape**: `[1, N, 1, attention_head_dim]` where N = post_patch_num_frames × post_patch_num_joints

### Step 3: Patch Embedding (Lines 322-324)

**Stage 3a**: Conv2d patch embedding
```python
hidden_states = self.patch_embedding(hidden_states)
# [B, C, T, J] → [B, inner_dim, T', J']
# Conv kernel (p_t, p_j) with stride (p_t, p_j)
```

**Stage 3b**: Flatten and transpose
```python
hidden_states = hidden_states.flatten(2).transpose(1, 2)
# [B, inner_dim, T', J'] → [B, inner_dim*T'*J']
#                        → [B, N, inner_dim]
```

**Concrete example** (typical config):
- After Conv: `[2, 768, 128, 22]`  (inner_dim=768)
- After flatten: `[2, 768, 2816]`
- After transpose: `[2, 2816, 768]`

### Step 4: Motion Masking (Lines 332-361)

Converts shape `[B, T, J]` (1=visible, 0=masked) → `[B, 1, 1, N]` (0=attend, -inf=masked)

**Algorithm**:
```python
# Step 4.1: Reshape to separate patch dimensions
# [B, T, J] → [B, T', p_t, J', p_j]
hidden_states_mask = hidden_states_mask.reshape(
    batch_size, post_patch_num_frames, p_t, post_patch_num_joints, p_j
)

# Step 4.2: Min-pool across patch dimensions
# If ANY element in a patch is masked, the entire patch is masked
# [B, T', p_t, J', p_j] → [B, T', J']
hidden_states_mask = hidden_states_mask.amin(dim=(2, 4))

# Step 4.3: Flatten to token sequence
# [B, T', J'] → [B, N]
hidden_states_mask = hidden_states_mask.flatten(1)

# Step 4.4: Convert to attention bias format
# 1 (visible) → 0, 0 (masked) → -inf
# Add dimensions: [B, N] → [B, 1, 1, N]
hidden_states_mask = (
    (1.0 - hidden_states_mask.float()) * torch.finfo(dtype).min
).unsqueeze(1).unsqueeze(2)
```

### Step 5: Text Masking (Lines 370-378)

Converts text mask `[B, N_ctx]` → `[B, 1, 1, N_ctx]`:
```python
encoder_hidden_states_mask = (
    (1.0 - encoder_hidden_states_mask.float()) * torch.finfo(dtype).min
).unsqueeze(1).unsqueeze(2)
```

### Step 5b: Causal Mask (Lines 392-405)

Frame-level causal masking: tokens can only attend to tokens in the current frame or earlier frames.

```python
seq_len = hidden_states.shape[1]  # N = T' × J'
# Frame index for each token: frame_idx[i] = i // J'
frame_idx = torch.arange(seq_len, device=device) // post_patch_num_joints

# Build mask: causal_mask[i, j] = -inf if frame_idx[j] > frame_idx[i]
causal_mask = (
    (frame_idx.unsqueeze(0) > frame_idx.unsqueeze(1))
    .to(dtype) * torch.finfo(dtype).min
).unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
```

**Example** (N=4 joints per frame, 2 frames):
```
Token positions:  0 1 2 3 | 4 5 6 7  (| separates frames)
Frame indices:    0 0 0 0 | 1 1 1 1

Causal mask (abbreviated, -inf shown as X):
       0 1 2 3 4 5 6 7
    0 [0 0 0 0 X X X X]
    1 [0 0 0 0 X X X X]
    2 [0 0 0 0 X X X X]
    3 [0 0 0 0 X X X X]
    4 [0 0 0 0 0 0 0 0]
    5 [0 0 0 0 0 0 0 0]
    6 [0 0 0 0 0 0 0 0]
    7 [0 0 0 0 0 0 0 0]
```

### Step 6: Timestep & Text Conditioning (Lines 411-422)

**Stage 6a**: Handle per-token timesteps (Wan 2.2 TI2V)
```python
if timestep.ndim == 2:
    # Per-token timesteps: [B, N] → flatten to [B*N]
    ts_seq_len = timestep.shape[1]
    timestep = timestep.flatten()
else:
    # Standard timesteps: [B] (stay as is)
    ts_seq_len = None
```

**Stage 6b**: Embed timesteps and text
```python
temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
    timestep,
    encoder_hidden_states,
    timestep_seq_len=ts_seq_len,
)
```

This calls `WanTimeTextEmbedding.forward()` which:
1. Projects timesteps through sinusoidal encoding
2. Applies MLP embedder
3. Projects through SiLU + linear layer for block modulation
4. Projects text embeddings with GELU-tanh activation (fp32 to prevent overflow)

See Part 4 for detailed embedding processing.

**Stage 6c**: Reshape timestep projection for block modulation
```python
if ts_seq_len is not None:
    # Per-token: [B, N, 6*inner_dim] → [B, N, 6, inner_dim]
    timestep_proj = timestep_proj.unflatten(2, (6, -1))
else:
    # Global: [B, 6*inner_dim] → [B, 6, inner_dim]
    timestep_proj = timestep_proj.unflatten(1, (6, -1))
```

The factor of 6 comes from the number of scale-shift parameters across all norm layers in the block.

### Step 7: Transformer Blocks (Lines 435-458)

Each block receives:
- `hidden_states`: [B, N, inner_dim]
- `encoder_hidden_states`: [B, N_ctx, text_dim]
- `temb`: [B, 6, inner_dim] or [B, N, 6, inner_dim]
- `rotary_emb`: [1, N, 1, head_dim]
- Masks as previously prepared

The block performs:
1. Adaptive layer norm (timestep-modulated)
2. Self-attention with RoPE
3. Cross-attention to text
4. FFN with adaptive layer norm

### Step 8: Output Adaptive Layer Norm (Lines 463-486)

Uses timestep embedding for adaptive scaling and shifting.

**Case 1: Per-token timesteps** (ts_seq_len != None)
```python
# temb: [B, N, inner_dim]
# scale_shift_table: [2, inner_dim] (learnable)
shift, scale = (
    self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
).chunk(2, dim=2)
# Output: shift [B, N, inner_dim], scale [B, N, inner_dim]

hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
```

**Case 2: Global timestep** (ts_seq_len == None)
```python
# temb: [B, inner_dim]
shift, scale = (
    self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)
).chunk(2, dim=1)
# Output: shift [B, 1, inner_dim], scale [B, 1, inner_dim]

hidden_states = self.norm_out(hidden_states) * (1 + scale) + shift
```

### Step 9: Output Projection (Line 491)

```python
hidden_states = self.proj_out(hidden_states)
# [B, N, inner_dim] → [B, N, C*p_t*p_j]
```

### Step 10: Unpatchify (Lines 498-509)

Inverse of patch embedding process.

**Stage 10a**: Reshape to separate patch and spatial dims
```python
# [B, N, C*p_t*p_j] → [B, T', J', p_t, p_j, C]
hidden_states = hidden_states.reshape(
    batch_size,
    post_patch_num_frames,
    post_patch_num_joints,
    p_t,
    p_j,
    -1,  # C
)
```

**Stage 10b**: Permute to interleave patches
```python
# [B, T', J', p_t, p_j, C] → [B, C, T', p_t, J', p_j]
hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
```

**Stage 10c**: Flatten back to original shape
```python
# [B, C, T', p_t, J', p_j] → [B, C, T'*p_t, J'*p_j] = [B, C, T, J]
output = hidden_states.flatten(4, 5).flatten(2, 3)
```

---

## Part 3: Spectral RoPE with Kinematic Tree Awareness

### Overview

When `joint_pos_mode='spectral_unified'`, RoPE uses the SMPL-22 kinematic tree structure to compute per-joint position scalars from graph Laplacian eigenvectors.

### File & Location
- **File**: `hftrainer/models/motion/prism/network/motion_rope.py`
- **Class**: `MotionWanRotaryPosEmbed`
- **Initialization**: Lines 385-465 (spectral_unified mode)
- **Forward**: Lines 577-671

### Kinematic Tree Structure (SMPL-22)

```
Pelvis (0)
├─ Spine (1-3)
├─ LeftHip (4)
│  └─ LeftKnee (5)
│     └─ LeftAnkle (6)
│        └─ LeftFoot (7)
├─ RightHip (8)
│  └─ RightKnee (9)
│     └─ RightAnkle (10)
│        └─ RightFoot (11)
├─ LeftShoulder (12)
│  └─ LeftElbow (13)
│     └─ LeftWrist (14)
├─ RightShoulder (15)
│  └─ RightElbow (16)
│     └─ RightWrist (17)
└─ Head (18-21)

Total: 22 body joints + 1 translation token = 23 channels
```

### Spectral_Unified Initialization (Lines 414-451)

**Step 1**: Build kinematic tree adjacency (lines 414-416)
```python
# Compute adjacency matrix for SMPL-22 tree structure
# A[i,j] = 1 if i and j are directly connected
```

**Step 2**: Compute graph Laplacian (line 416)
```python
# L = D - A where D is degree matrix
# Eigendecomposition: L = U @ Λ @ U^T
```

**Step 3**: Use Laplacian eigenvectors for position encoding (lines 420-422)
```python
# For each joint i (22 joints, not including translation):
# pos[i] = L2_norm(eigenvector_column_i)
# This gives a scalar position that respects kinematic relationships
```

**Step 4**: Compute RoPE frequencies (lines 440-451)
```python
# inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
# For standard RoPE: inv_freq[k] = 1 / 10000^(2k/d)
# where d = attention_head_dim

# For spectral RoPE:
# rope_freq[j, k] = inv_freq[k] * pos[j]
# This multiplies the base RoPE frequencies by per-joint positions
```

### Translation Token Special Handling

Translation token (index 0, representing global motion) gets **identity RoPE**:
```python
# rot_dim = attention_head_dim
# cos[0, :, rot_dim//2:] = 1.0
# sin[0, :, rot_dim//2:] = 0.0
# (first half set to cos/sin with zero frequency)
```

### Forward Pass (Lines 577-671)

**Input**: `hidden_states = [B, C, T, J]` (C includes channel dimension)

**Step 1**: Extract motion shape
```python
batch_size, num_channels, num_frames, num_joints = hidden_states.shape
num_tokens = (num_frames // self.patch_t) * (num_joints // self.patch_j)
```

**Step 2**: Compute RoPE for current sequence length
```python
# Get cached cos/sin buffers or compute them
pos = torch.arange(num_tokens, device=device, dtype=torch.long)
freqs = einsum('..., f -> ... f', pos, self.inv_freq)
emb = torch.cat([freqs, freqs], dim=-1)
cos = emb.cos()[None, :, None, :]  # [1, num_tokens, 1, rot_dim]
sin = emb.sin()[None, :, None, :]
```

**Step 3**: Apply per-joint scaling
```python
# For translation token (j=0): scale by 1.0 (identity)
# For body joints (j=1-22): scale by pos[j] from Laplacian eigenvectors
# Result: [1, num_tokens, 1, head_dim]
```

**Output**: `rotary_emb = [1, N, 1, attention_head_dim]`

### Why Spectral RoPE Matters

Traditional sequential RoPE assigns positions 0, 1, 2, ..., T-1 (temporal) or 0, 1, ..., J-1 (joint indices). This treats all joints equally.

**Spectral RoPE advantages**:
- **Kinematic awareness**: Joints in the same limb chain get similar positions
- **Hierarchical encoding**: Leaf joints (feet, hands) get different frequencies than root joints (pelvis)
- **Natural structure**: Respects SMPL skeleton topology in positional encoding

**Example**: If Laplacian eigenvectors produce:
- Pelvis pos ≈ 0.1 (root)
- Foot pos ≈ 0.8 (leaf)
- Hand pos ≈ 0.7 (leaf)

Then attention can more naturally learn that feet and hands have similar movement patterns but both differ from pelvis.

---

## Part 4: Timestep Embedding with Per-Token Timesteps

### File & Location
- **File**: `hftrainer/models/motion/prism/network/embedding.py`
- **Class**: `WanTimeTextEmbedding`
- **Lines**: 85-140

### Architecture

```
Timestep (discrete, 0-999)
    ↓
Sinusoidal Projection (Timesteps class)
    ├─ Frequency encoding: sin(t * 2π * base^(2k/d))
    └─ Output: [B*N, time_freq_dim]
    ↓
MLP Embedding (TimestepEmbedding class)
    └─ Output: [B*N, dim] or [B, N, dim]
    ↓
SiLU Activation + Linear Projection
    └─ Output: timestep_proj [B, time_proj_dim] or [B, N, time_proj_dim]
    ↓
Used for:
- Adaptive layer norm scaling/shifting
- Block-level conditioning
```

### Per-Token Timestep Handling (Lines 115-117)

When `timestep_seq_len` is provided (Wan 2.2 TI2V mode):

```python
# Input: timestep [B*N] (flattened)
# Step 1: Apply sinusoidal projection
timestep = self.timesteps_proj(timestep)  # [B*N, time_freq_dim]

# Step 2: Reshape if per-token mode
if timestep_seq_len is not None:
    # [B*N, ...] → [B, N, ...]
    timestep = timestep.unflatten(0, (-1, timestep_seq_len))

# Step 3: Apply MLP
temb = self.time_embedder(timestep)
# If per-token: [B, N, time_freq_dim] → [B, N, dim]
# If global: [B, time_freq_dim] → [B, dim]
```

### Per-Token vs Global Timesteps

**Standard (per-batch) timestep**:
- Shape: `[B]` e.g., `[2]` for batch of 2
- Meaning: entire motion sequence gets same diffusion timestep
- All B*N tokens conditioned on same t
- Used for standard diffusion sampling

**Per-token timesteps** (Wan 2.2 TI2V):
- Shape: `[B, N]` e.g., `[2, 2816]` for batch of 2 with 2816 tokens
- Meaning: each token position can have different t
- Use case: Condition frames get t=0 (no noise), generation frames get t>0 (noisy)
- Enables **noise-free frame injection** during autoregressive generation

### Implementation Details

**Text Embedding Processing** (Lines 131-138):

```python
with torch.amp.autocast('cuda', dtype=torch.float32):
    encoder_hidden_states = self.text_embedder(encoder_hidden_states)
# Cast back to match temb dtype
encoder_hidden_states = encoder_hidden_states.type_as(temb)
```

The FP32 upcast prevents overflow in GELU-tanh activation (computes x^3):
- In fp16: when |x| > 40.3, x^3 overflows
- In fp32: x^3 is computed safely, result cast back to fp16

### Output Shapes

For typical config (B=2, T=256, J=22, p_t=2, p_j=1, N=2816):

**Global timesteps** (standard):
- Input timestep: `[2]`
- Output temb: `[2, 128]` (inner_dim=128)
- Output timestep_proj: `[2, 768]` (time_proj_dim=768)

**Per-token timesteps** (Wan 2.2 TI2V):
- Input timestep: `[2, 2816]` after unflatten
- Output temb: `[2, 2816, 128]`
- Output timestep_proj: `[2, 2816, 768]`

---

## Part 5: Critical Implementation Details

### 5.1 Patch Min-Pooling Masking

When masking motion with variable lengths, the mask min-pools across patch dimensions (lines 344-345):

```python
hidden_states_mask = hidden_states_mask.amin(dim=(2, 4))
```

**Implication**: If ANY token position within a patch is masked, the entire patch is masked.

**Potential Issue**: 
- Input mask [B, T, J] with 1=valid, 0=masked
- If last few frames are masked but few joints in earlier frames are also masked, min-pooling may over-mask
- Recommendation: Ensure mask is contiguous (e.g., frames 0-N are valid, N+1-T are masked)

### 5.2 RoPE Precision

RoPE is computed in FP32 then applied:
- Position frequencies: computed as fp32 scalars
- cos/sin buffers: stored/computed as fp32
- Applied to attention scores: converted to model dtype (fp16/bf16)

**Why FP32?**: Prevents precision loss in high-frequency components when d_head > 64

### 5.3 Adaptive Layer Norm Reshaping

Final layer norm uses timestep embedding for per-token or global scale/shift:

```python
# Per-token case:
shift, scale = (scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
# Result: [B, N, 1, 2*inner_dim] → chunks to [B, N, inner_dim] each

# Global case:
shift, scale = (scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
# Result: [B, 1, 2*inner_dim] → chunks to [B, 1, inner_dim] each
```

The factor of 2 comes from concatenating shift and scale parameters: `[2*inner_dim]` → `[2, inner_dim]`

### 5.4 Causal Masking with Joint Tokens

Frame-level causality (lines 395-397):

```python
frame_idx = torch.arange(seq_len) // post_patch_num_joints
```

If post_patch_num_joints = 22:
- Tokens 0-21 → frame 0
- Tokens 22-43 → frame 1
- Tokens 44-65 → frame 2

All tokens within a frame can attend to each other (no joint-level causality).

---

## Part 6: Potential Issues & Risk Assessment

### Issue 1: Translation Token Index Mismatch
**Location**: RoPE forward pass assumes translation token at position 0  
**Risk**: HIGH if VAE output includes translation channel but RoPE excludes it  
**Mitigation**: Verify channel ordering in patch embedding output

### Issue 2: Patch Min-Pooling Masking Behavior
**Location**: Lines 344-345, motion mask processing  
**Risk**: MEDIUM - over-masks if masks aren't frame-aligned  
**Mitigation**: Document that input masks should be contiguous (valid frames then padding)

### Issue 3: Per-Token Timestep Sequence Length Validation
**Location**: Lines 411-416, timestep shape handling  
**Risk**: MEDIUM - silent failure if ts_seq_len doesn't match N  
**Mitigation**: Add runtime assertion: `assert ts_seq_len * batch_size == timestep.numel()`

### Issue 4: Spectral Coordinate Sign Determinism
**Location**: `motion_rope.py` line 422, L2 norm of eigenvectors  
**Risk**: LOW - eigenvectors have arbitrary sign, but L2 norm is always positive  
**Current Status**: Already handled correctly in codebase

### Issue 5: FP32 RoPE to FP16 Conversion
**Location**: RoPE cos/sin buffers applied to fp16 attention  
**Risk**: LOW - cos/sin are bounded [-1, 1], safe to convert  
**Current Status**: Already handled correctly

### Issue 6: RoPE Application Shape Matching
**Location**: Rotary attention kernel expects `[1, N, 1, head_dim]`  
**Risk**: MEDIUM - shape mismatches would cause broadcasting errors  
**Mitigation**: Add shape assertions in attention computation

---

## Part 7: Verification Checklist

### Forward Pass Shapes

| Stage | Input Shape | Output Shape | Notes |
|-------|------------|--------------|-------|
| Input | [B, C, T, J] | - | e.g., [2, 16, 256, 22] |
| RoPE | [B, C, T, J] | [1, N, 1, head_dim] | N = (T/p_t) × (J/p_j) |
| Patch Emb | [B, C, T, J] | [B, N, inner_dim] | Flattened tokens |
| Motion Mask | [B, T, J] → [B, 1, 1, N] | - | 1→0, 0→-inf |
| Timestep Emb (global) | [B] | [B, inner_dim] | Standard diffusion |
| Timestep Emb (per-token) | [B, N] | [B, N, inner_dim] | TI2V mode |
| Transformer Out | [B, N, inner_dim] | [B, N, inner_dim] | 30-40 blocks |
| Unpatch | [B, N, C*p_t*p_j] | [B, C, T, J] | Inverse of patch |

### Debug Prints to Add

```python
# In forward method:
print(f"Motion shape: {hidden_states.shape}")
print(f"Patch size: {p_t}x{p_j}")
print(f"Tokens N: {hidden_states.shape[1]}")
print(f"RoPE shape: {rotary_emb.shape}")
print(f"Timestep shape: {timestep.shape}")
print(f"Timestep per-token? {timestep.ndim == 2}")
print(f"Output shape: {output.shape}")
```

### Test Cases

1. **Basic forward** (no masks, global timestep)
   - Input: [2, 16, 64, 22], timestep [2]
   - Verify output: [2, 16, 64, 22]

2. **Per-token timesteps** (TI2V mode)
   - Input: [2, 16, 64, 22], timestep [2, 176] (64/p_t × 22/p_j)
   - Verify output: [2, 16, 64, 22]

3. **With motion masking**
   - Input: [2, 16, 64, 22], mask [2, 64, 22] (first 32 frames valid, rest masked)
   - Verify masked tokens don't contribute to output

4. **Causal masking**
   - Input: [2, 16, 64, 22], is_causal=True
   - Verify frame-level causality enforced

---

## References

### Code Files
- `hftrainer/models/motion/prism/network/transformer_prism.py` - Main model
- `hftrainer/models/motion/prism/network/motion_rope.py` - Spectral RoPE
- `hftrainer/models/motion/prism/network/embedding.py` - Timestep/text embeddings
- `hftrainer/models/motion/prism/network/block_with_mask.py` - Transformer block

### Key Classes
- `PrismTransformerMotionModel` - Main entry point
- `MotionWanRotaryPosEmbed` - RoPE with kinematic tree awareness
- `WanTimeTextEmbedding` - Timestep + text conditioning
- `WanTransformerBlockWithMask` - Individual transformer block

### External References
- Wan: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py
- DiT: https://arxiv.org/abs/2212.09748
- RoPE: https://arxiv.org/abs/2104.09864
- Laplacian Eigenmaps: https://arxiv.org/abs/cond-mat/0306021

---

**Analysis Complete**  
**Generated**: 2026-05-27  
**Status**: Ready for technical review and implementation verification
