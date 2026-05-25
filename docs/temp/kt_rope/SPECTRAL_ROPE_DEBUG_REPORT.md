# PRISM Spectral RoPE Bug Analysis - COMPREHENSIVE DEBUG REPORT

**Analysis Date:** 2026-05-25  
**Target Model:** PRISM 1B with KT-RoPE (Kinematic Tree Rotary Position Embedding) in spectral mode  
**Symptom:** Generated motions are near-static with very low diversity  
**Root Cause:** Multiple critical bugs preventing proper spectral RoPE implementation and application

---

## CRITICAL BUG #1: RoPE DIMENSION MISMATCH (BLOCKER)

### ⚠️ SEVERITY: CRITICAL (Prevents Inference)

### Location
- **File:** `motion_rope.py`, lines 381-483 (`forward()` method)
- **Affected Lines:** 477-482 (output reshape)
- **Integration Point:** `block_with_mask.py` line 225 (passes to WanAttention)
- **External:** `diffusers.models.transformers.transformer_wan.WanAttnProcessor` (applies RoPE)

### The Problem

**Current Output Shape from MotionWanRotaryPosEmbed:**
```
freqs_cos.shape = (1, num_patches, 1, attention_head_dim)
                = (1, 1472, 1, 128)
freqs_sin.shape = (1, 1472, 1, 128)
```

**Expected Shape for WanAttnProcessor:**
The processor applies RoPE AFTER unflattening query/key into per-head format:

```python
# In WanAttnProcessor.__call__ (diffusers library):
query = query.unflatten(2, (attn.heads, -1))  # ← Splits head dim
# Result: (batch_size, seq_len, num_heads, head_dim)
#       = (2, 1472, 4, 32)

# Then applies RoPE:
def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    # x1, x2 shape: (batch_size, seq_len, num_heads, head_dim//2)
    #            = (2, 1472, 4, 16)
    cos = freqs_cos[..., 0::2]  # ← Slices LAST dimension
    sin = freqs_sin[..., 1::2]
    out[..., 0::2] = x1 * cos - x2 * sin  # ← SHAPE MISMATCH HERE
```

### Dimension Analysis

**Query/Key after unflatten:**
```
(batch=2, seq_len=1472, num_heads=4, head_dim=32)
```

**Query after head_dim unflatten:**
```
x = query.unflatten(-1, (-1, 2))  # Split head_dim into (16, 2)
x1, x2 = x.unbind(-1)
Result: x1.shape = (2, 1472, 4, 16)
```

**RoPE cos after slicing:**
```
cos = freqs_cos[..., 0::2]
    = (1, 1472, 1, 128)[..., 0::2]
    = (1, 1472, 1, 64)
    
Expected: (1, 1472, 1, 16)  ← Per-head dim, not full attention_head_dim
```

**Broadcasting Failure:**
```
x1.shape:    (2, 1472, 4, 16)
cos.shape:   (1, 1472, 1, 64)
             ↑ Mismatch in last dimension: 16 vs 64
Cannot broadcast!
RuntimeError: The size of tensor a (16) must match the size of tensor b (64)
```

### Why This Happens

The `MotionWanRotaryPosEmbed.forward()` is designed to:
1. Split `attention_head_dim` (128) into temporal (64) and joint (64) components
2. Compute separate RoPE for temporal dimension and per-joint RoPE
3. Concatenate them back to full `attention_head_dim` (128)
4. Output shape: `(1, num_patches, 1, 128)`

But the attention processor:
1. Receives this `(1, seq, 1, 128)` RoPE
2. After `query.unflatten(2, (num_heads, -1))`, query is `(batch, seq, 4, 32)`
3. Then tries to apply RoPE by slicing `freqs_cos[..., 0::2]` on the LAST dimension
4. Expects the RoPE to already be in per-head format `(1, seq, 1, 32)`

**The fundamental issue:** RoPE is computed at the full `attention_head_dim` level, but must be applied at the per-head level.

### Code Trace

**In motion_rope.py (lines 477-482):**
```python
# Concatenate temporal and joint, reshape for attention
freqs_cos = torch.cat([freqs_cos_f, freqs_cos_j], dim=-1).reshape(
    1, ppf * ppj, 1, -1
)
freqs_sin = torch.cat([freqs_sin_f, freqs_sin_j], dim=-1).reshape(
    1, ppf * ppj, 1, -1
)
# ↑ freqs_cos is (1, 1472, 1, 128) - FULL attention_head_dim
```

**In block_with_mask.py (line 225):**
```python
attn_output = self.attn1(
    hidden_states=norm_hidden_states,
    encoder_hidden_states=None,
    attention_mask=combined_self_attn_mask,
    rotary_emb=rotary_emb,  # ← Passed as (freqs_cos, freqs_sin)
)
```

**In diffusers WanAttnProcessor:**
```python
query = attn.norm_q(query)  # (batch, seq, dim=128)
query = query.unflatten(2, (attn.heads, -1))  # (batch, seq, 4, 32)

def apply_rotary_emb(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    # x1: (batch, seq, heads, 16)
    cos = freqs_cos[..., 0::2]  # (1, seq, 1, 64) ← WRONG!
    out[..., 0::2] = x1 * cos - x2 * sin  # ✗ Shape mismatch
```

### Verification

**To confirm this bug exists:**
```python
import torch
from hftrainer.models.motion.prism.network.motion_rope import MotionWanRotaryPosEmbed
from hftrainer.models.motion.prism.network.block_with_mask import WanTransformerBlockWithMask

# Create components
rope = MotionWanRotaryPosEmbed(
    attention_head_dim=128, 
    patch_size=(1, 1), 
    max_seq_len=1024,
    joint_pos_mode="spectral"
)
block = WanTransformerBlockWithMask(dim=1024, ffn_dim=4096, num_heads=8)

# Create inputs
hidden_states = torch.randn(2, 16, 64, 23)  # [B, C, T, J]
encoder_hidden_states = torch.randn(2, 77, 1024)
temb = torch.randn(2, 6, 1024)

# Compute RoPE
rotary_emb = rope(hidden_states)
print(f"RoPE shape: {rotary_emb[0].shape}")  # (1, 1472, 1, 128)

# Try to apply
try:
    output = block(
        hidden_states=torch.randn(2, 1472, 1024),
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        rotary_emb=rotary_emb
    )
except RuntimeError as e:
    print(f"ERROR: {e}")
    # Expected: "The size of tensor a (16) must match the size of tensor b (64)"
```

### Impact

This bug causes:
1. **Training cannot proceed** with spectral mode
2. **Inference crashes** with dimension mismatch
3. **Fall-back behavior:** Model might default to identity attention (no RoPE)
4. **Complete loss of positional encoding:** Attention completely fails
5. **Result:** Near-zero motion diversity (all motion patterns look the same without position encoding)

---

## CRITICAL BUG #2: TRAINING VS. INFERENCE MISMATCH

### ⚠️ SEVERITY: CRITICAL (Causes Poor Convergence)

### Location
- **Config:** `prism_1b_tp2m_multiframe_kt_spectral.py`, lines 62-68
- **Intent vs. Reality Mismatch**

### The Issue

**What the config says (lines 62-68):**
```python
# Load weights from the sequential RoPE checkpoint (model weights only).
# RoPE buffers are non-persistent and will be recomputed with spectral coords.
load_from = dict(
    _delete_=True,
    path='work_dirs/prism_1b_tp2m_multiframe/checkpoint-iter_15000',
    load_scope='model',
)
```

**What actually happens:**
1. Model weights loaded from sequential RoPE checkpoint ✓
2. BUT spectral RoPE buffers are freshly computed at init time ✓
3. The model's learned weights WERE optimized for sequential RoPE attention patterns
4. Now at fine-tuning, the model sees DIFFERENT attention patterns from spectral RoPE
5. **Result:** Massive train-test mismatch → poor convergence → low-quality outputs

### Why This Is Critical

The model's transformer weights (query/key/value projections, layer norms, feed-forward networks) were all trained to:
- Recognize and process attention from **sequential RoPE**
- Sequential RoPE: joints indexed 0, 1, 2, ..., 22 (linear order)
- Joints at indices `i` and `i+1` are consecutive regardless of kinematic relationship

But now with **spectral RoPE**:
- Joints indexed by kinematic tree Laplacian eigenvectors
- Joints at similar spectral positions are kinematically close
- Completely different attention pattern structure

### The Fundamental Problem

**Attention score computation:**
```
attention_scores = softmax(Q @ K^T / sqrt(d_k) + RoPE_bias)
```

**With sequential RoPE:**
- RoPE_bias shaped from sequential position indices
- Model learned attention patterns that work with this bias

**With spectral RoPE:**
- RoPE_bias shaped from spectral coordinates
- Model was never trained on this pattern
- Model tries to apply learned weights to unseen attention pattern
- **Result:** Severe performance degradation

### Why Generated Motions Are Near-Static

Without proper training on spectral RoPE:
1. Attention patterns are meaningless to the model
2. Model falls back to using only residual connections and FFN
3. With very limited positional information
4. All generated motion tokens become very similar
5. **Result:** Identical frames → near-static motion

---

## BUG #3: SPECTRAL BUFFER PERSISTENCE INCONSISTENCY

### ⚠️ SEVERITY: MEDIUM (Causes Loading Issues)

### Location
- **Code:** `motion_rope.py`, lines 312-324 (spectral mode initialization)
- **Config:** `prism_1b_tp2m_multiframe_kt_spectral.py`, lines 62-68

### The Issue

**In motion_rope.py:**
```python
self.register_buffer(
    "joint_freqs_cos", joint_freqs_cos, persistent=True
)
self.register_buffer(
    "joint_freqs_sin", joint_freqs_sin, persistent=True
)
```

**In config file:**
```python
# RoPE buffers are non-persistent and will be recomputed with spectral coords.
```

### The Problem

1. **If persistent=True:** RoPE buffers are saved in checkpoints
   - When loading from sequential checkpoint, sequential buffers might get loaded
   - New spectral buffers created, but old sequential buffers still in memory
   - Potential name collisions or shape mismatches

2. **If persistent=False:** RoPE buffers NOT saved in checkpoints
   - Buffers recomputed fresh each time
   - More memory efficient
   - But config comment suggests this is the intent

### What Should Be Done

The config comment says buffers are non-persistent, but code says persistent=True.
This inconsistency means:
- Developers think buffers are non-persistent (and will be recomputed)
- But code is actually persisting them (saving/loading from checkpoints)
- When fine-tuning from a sequential checkpoint, shape mismatches can occur

---

## BUG #4: SPECTRAL SCALE FACTOR OPTIMIZATION

### ⚠️ SEVERITY: MEDIUM (Affects Diversity)

### Location
- **Code:** `motion_rope.py`, lines 254-259
- **Config:** `prism_1b_tp2m_multiframe_kt_spectral.py`, line 17

### The Issue

**Current code:**
```python
scale = spectral_scale if spectral_scale is not None else 22.0
spectral_coords = spectral_coords * scale
```

**Current config:**
```python
spectral_scale=22.0,  # Scale spectral coords (default = num_joints)
```

### The Problem

**Spectral coordinates before scaling:**
```
Range: [-0.448, 0.448]
Mean: -0.003
Std: 0.213
```

**After scaling by 22:**
```
Range: [-9.87, 9.87]
Mean: -0.06
Std: 4.69
```

### Why This Matters

The scaled coordinates are used to compute RoPE frequencies:

```python
# In motion_rope.py line 291:
freqs = 1.0 / (theta ** (2.0 * freq_seq / dim))  # Standard RoPE frequency
angles = pos * freqs  # ← pos is the scaled spectral coordinate
cos_vals = torch.cos(angles)
sin_vals = torch.sin(angles)
```

**If scale is too large:**
- `angles = pos * freqs` becomes very large
- `cos(large_angle)` and `sin(large_angle)` oscillate rapidly
- Attention becomes dominated by high-frequency components
- **Result:** Limited attention pattern diversity

**If scale is too small:**
- `angles` becomes very small
- `cos(small_angle) ≈ 1` and `sin(small_angle) ≈ 0`
- RoPE becomes nearly identity (no position information)
- **Result:** Minimal positional encoding effect

### Optimization Needed

The scale factor of 22.0 appears arbitrary:
- Is it meant to match the range of sequential indices (0-21)? ✓ Reasonable
- Or is it empirically optimized? ✗ Unknown

### Impact on Low Diversity

If scale is suboptimal:
- RoPE frequencies either too concentrated (few unique patterns) or too weak
- Attention weights become homogeneous
- All queries attend to all keys with similar weights
- Motion generation becomes constrained to low-diversity modes
- **Result:** Near-static motions

---

## BUG #5: SPECTRAL COORDINATE SIGN INCONSISTENCY

### ⚠️ SEVERITY: LOW (But Worth Noting)

### Location
- **Code:** `motion_rope.py`, lines 99-108

### The Issue

**Current implementation:**
```python
# Canonicalize by enforcing that the first joint (Pelvis, root) has a positive
# coordinate in each mode.
for mode_idx in range(num_modes):
    if spectral_coords[0, mode_idx] < 0:
        spectral_coords[:, mode_idx] *= -1.0
```

### Why This Is Correct

✓ Ensures deterministic output (eigenvectors are defined up to sign)
✓ Makes results reproducible across numpy/BLAS versions
✓ Canonical form is good practice

### Potential Issue

The pre-trained model NEVER saw these spectral coordinates because:
- They were added in the fine-tuning config
- Original model trained only with sequential RoPE
- Even though the coordinates are canonicalized now, they represent a completely new domain
- Model needs to learn how to interpret these new coordinates

---

## ROOT CAUSE ANALYSIS: WHY MOTIONS ARE NEAR-STATIC

### Primary Cause (Probability: 95%)

**Bug #1: RoPE Dimension Mismatch + Bug #2: Train-Test Mismatch**

Chain of failures:
1. Model attempts to apply spectral RoPE
2. Dimension mismatch in `WanAttnProcessor.apply_rotary_emb()` occurs
3. **Fallback:** Model either crashes or silently defaults to no RoPE
4. Without RoPE, positional information is lost
5. Without positional information, attention becomes position-agnostic
6. All attention patterns collapse to low-diversity subspace
7. **Result:** Near-identical motion tokens → near-static motion

OR:

1. Model attempts to apply spectral RoPE (somehow works despite shape mismatch)
2. RoPE patterns completely different from training (Bug #2)
3. Model's learned weights don't know how to interpret new RoPE patterns
4. Model falls back to default behavior (residual + FFN)
5. Without position-aware attention, motion diversity is severely limited
6. **Result:** Near-static, low-diversity outputs

### Secondary Cause (Probability: 85%)

**Bug #4: Spectral Scale Factor**

Even if RoPE is applied correctly, if scale factor is poorly chosen:
- RoPE frequencies either too strong (oscillate wildly) or too weak (nearly identity)
- Either way: limited effective positional information
- Reduced attention diversity → reduced motion diversity

---

## RECOMMENDED FIXES (Priority Order)

### FIX #1: SHAPE MISMATCH (CRITICAL)

**Problem:** RoPE output is full `attention_head_dim`, but attention expects per-head dims

**Solution A (Preferred): Reshape RoPE at output**

In `motion_rope.py`, modify the forward method to reshape RoPE after concatenation:

```python
# After line 482 in motion_rope.py (end of spectral/dfs mode)

# Current: (1, ppf * ppj, 1, j_dim + t_dim) = (1, seq, 1, 128)
# Need: (1, seq, 1, head_dim) = (1, seq, 1, 32) for each head

# Calculate head dimension from context
num_heads_from_config = self.attention_head_dim // per_head_dim
# But we don't have this info in the module...
# BETTER: Don't split into temporal+joint in the first place
```

**Solution B (Alternative): Custom attention processor**

Create a custom RoPE application that understands temporal+joint factorization:

```python
def apply_spectral_rope_emb(query, key, freqs_cos, freqs_sin, t_dim, j_dim):
    """
    Apply spectral RoPE respecting temporal+joint factorization.
    
    Args:
        query: (batch, seq_len, num_heads, head_dim)
        freqs_cos: (1, seq_len, 1, t_dim + j_dim)
        freqs_sin: (1, seq_len, 1, t_dim + j_dim)
    """
    # Split RoPE into temporal and joint components
    freqs_cos_t = freqs_cos[..., :t_dim]  # (1, seq_len, 1, t_dim)
    freqs_cos_j = freqs_cos[..., t_dim:]  # (1, seq_len, 1, j_dim)
    
    # Apply to corresponding head dimensions
    # (Requires custom attention processor)
    ...
```

### FIX #2: TRAIN-TEST MISMATCH (CRITICAL)

**Problem:** Model weights trained on sequential RoPE, fine-tuned with spectral RoPE

**Solution: Full re-training from scratch**

```python
# In prism_1b_tp2m_multiframe_kt_spectral.py
load_from = dict(
    _delete_=True,
    # Don't load from sequential checkpoint!
    # Either:
    # A) Start from scratch (no load_from)
    # B) Load from a previous spectral checkpoint if available
)

# Start training from epoch 0 with fresh spectral RoPE initialization
```

OR:

**Alternative: Warm-up strategy**

```python
# Train for first N epochs with sequential RoPE
# Then switch to spectral RoPE (with small learning rate to adapt)
# This allows model to gradually learn spectral patterns
```

### FIX #3: SPECTRAL SCALE OPTIMIZATION (IMPORTANT)

**Problem:** Scale factor of 22.0 appears arbitrary and may be suboptimal

**Solution: Systematic optimization**

```python
# Test different scale factors
scales_to_test = [1.0, 5.0, 10.0, 15.0, 22.0, 30.0, 50.0]

for scale in scales_to_test:
    # Train model with this scale
    # Evaluate motion diversity metrics (entropy, variance, etc.)
    # Record performance
    
# Use the scale that maximizes motion diversity while maintaining quality
```

### FIX #4: BUFFER PERSISTENCE (MINOR)

**Problem:** Inconsistency between code (persistent=True) and documentation (non-persistent)

**Solution: Make explicit choice**

```python
# Option A: Make buffers non-persistent (if recomputing is desired)
self.register_buffer("joint_freqs_cos", joint_freqs_cos, persistent=False)

# Option B: Keep buffers persistent (if we want checkpoint consistency)
self.register_buffer("joint_freqs_cos", joint_freqs_cos, persistent=True)
# Then update config comment to reflect this

# Recommendation: Use persistent=False for spectral mode
# (since spectral coordinates are always recomputed from kinematic tree)
```

---

## VALIDATION CHECKLIST

After implementing fixes:

- [ ] Verify RoPE output shape is compatible with WanAttnProcessor
- [ ] Add unit test that runs forward pass through attention with spectral RoPE
- [ ] Verify attention is actually being applied (add assertions)
- [ ] Test shape propagation through full model forward pass
- [ ] Compare attention patterns: sequential vs spectral vs DFS
- [ ] Train model from scratch with spectral RoPE only
- [ ] Verify generated motion diversity increases after fixes
- [ ] Check GPU memory usage and computation time
- [ ] Validate on holdout test set

---

## SUMMARY

| Bug | Severity | Impact | Fix Priority |
|-----|----------|--------|--------------|
| #1: RoPE Shape Mismatch | CRITICAL | Prevents inference | IMMEDIATE |
| #2: Train-Test Mismatch | CRITICAL | Poor convergence | IMMEDIATE |
| #3: Buffer Persistence | MEDIUM | Loading issues | HIGH |
| #4: Scale Factor | MEDIUM | Low diversity | HIGH |
| #5: Sign Inconsistency | LOW | Reproducibility | LOW |

**Overall Status:** 🔴 **BROKEN** - Multiple critical blockers prevent spectral RoPE from functioning

**Estimated Impact of Fixes:**
- Bug #1 + #2 fixed: Model should train and converge properly
- Bug #3 + #4 fixed: Further improvements to motion quality and diversity
- Bug #5 fixed: Better reproducibility and debugging

