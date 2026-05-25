# KT-RoPE Implementation Guide for PRISM

This document provides guidance for implementing KT-RoPE in the PRISM codebase based on the paper description.

---

## Overview

KT-RoPE (Kinematic-Tree Rotary Position Embeddings) augments standard 2D RoPE with explicit kinematic tree structure. The implementation modifies the joint-axis positional encoding computation in the DiT's attention modules.

---

## Mathematical Definition

For joint $j$ at depth $d_j$ in the SMPL tree with parent $p_j$:

```
m_j^KT-RoPE = m_j + β_d·f_d(d_j) + β_p·f_p(j,p_j) + β_s·f_s(j)
```

Where:
- `m_j` = standard joint-index positional encoding (originally in [0, K-1])
- `f_d(d_j)` = tree depth encoding
- `f_p(j,p_j)` = parent-child relationship encoding
- `f_s(j)` = sibling structure encoding
- `β_d, β_p, β_s` = learnable or fixed weighting parameters

---

## Implementation Components

### 1. SMPL Kinematic Tree Structure

Standard SMPL has 23 joints with the following hierarchy:

```
Depth  Joint Name(s)
─────  ──────────────────────────────────────
0      Pelvis (root)
1      L_Hip, R_Hip, Spine1
2      L_Knee, R_Knee, Spine2
3      L_Ankle, R_Ankle, Spine3
4      L_Foot, R_Foot, Neck
5      L_Collar, R_Collar, Head
6      L_Shoulder, R_Shoulder, L_Hand, R_Hand (wrists)
```

Maximum depth: 6 (wrists)

### 2. Depth Encoding Function

```python
def encode_depth(joint_idx, depth_dict, embedding_dim):
    """
    Encode joint depth as positional encoding.
    
    Args:
        joint_idx: Joint index in [0, K-1]
        depth_dict: Dict mapping joint_idx -> depth in kinematic tree
        embedding_dim: Dimension of depth encoding
    
    Returns:
        Tensor of shape (embedding_dim,) with depth information
    """
    depth = depth_dict[joint_idx]
    max_depth = 6  # Maximum depth in SMPL
    
    # Normalized depth: [0, 1]
    normalized_depth = depth / max_depth
    
    # Frequency-based encoding (similar to positional encoding)
    encoding = []
    for i in range(embedding_dim):
        if i % 2 == 0:
            # Even dimensions: sin
            encoding.append(torch.sin(
                normalized_depth * torch.pi * 
                (2 ** (i / embedding_dim))
            ))
        else:
            # Odd dimensions: cos
            encoding.append(torch.cos(
                normalized_depth * torch.pi * 
                (2 ** ((i-1) / embedding_dim))
            ))
    
    return torch.stack(encoding)
```

### 3. Parent-Child Encoding Function

```python
def encode_parent_relationship(joint_idx, parent_dict, embedding_dim):
    """
    Encode parent-child relationships.
    
    Args:
        joint_idx: Joint index
        parent_dict: Dict mapping joint_idx -> parent_idx
        embedding_dim: Dimension of parent encoding
    
    Returns:
        Tensor of shape (embedding_dim,)
    """
    parent_idx = parent_dict[joint_idx]
    
    # Use parent index as a frequency modifier
    parent_factor = (parent_idx + 1) / 23  # Normalize to [1/23, 1]
    
    encoding = []
    for i in range(embedding_dim):
        freq = 2 ** (i / embedding_dim)
        if i % 2 == 0:
            encoding.append(torch.sin(parent_factor * torch.pi * freq))
        else:
            encoding.append(torch.cos(parent_factor * torch.pi * freq))
    
    return torch.stack(encoding)
```

### 4. Sibling Structure Encoding

```python
def encode_sibling_structure(joint_idx, sibling_dict, embedding_dim):
    """
    Encode sibling/bilateral structure.
    
    Sibling pairs (e.g., L_Hip vs R_Hip) share a parent and should
    receive correlated but offset encodings.
    
    Args:
        joint_idx: Joint index
        sibling_dict: Dict mapping joint_idx -> (parent, sibling_offset)
        embedding_dim: Dimension of encoding
    
    Returns:
        Tensor of shape (embedding_dim,)
    """
    if joint_idx not in sibling_dict:
        return torch.zeros(embedding_dim)
    
    parent_idx, sibling_offset = sibling_dict[joint_idx]
    
    # Sibling offset: +1 for right, -1 for left
    offset_factor = sibling_offset / 2.0
    
    encoding = []
    for i in range(embedding_dim):
        freq = 2 ** (i / embedding_dim)
        if i % 2 == 0:
            encoding.append(torch.sin(offset_factor * torch.pi * freq))
        else:
            encoding.append(torch.cos(offset_factor * torch.pi * freq))
    
    return torch.stack(encoding)
```

### 5. Main KT-RoPE Integration

```python
def compute_kt_rope(joint_axis_pos, t, head_dim):
    """
    Compute KT-RoPE for joint-axis positional encoding.
    
    This replaces the standard RoPE computation for the joint axis.
    The temporal axis RoPE remains unchanged.
    
    Args:
        joint_axis_pos: Position along joint axis [0, K-1]
        t: Position along temporal axis
        head_dim: Dimension per attention head
    
    Returns:
        Rotary matrix for this (t, joint) position
    """
    # Standard 2D RoPE base frequencies
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Temporal position component (standard RoPE)
    t_emb = t * inv_freq
    t_cos = torch.cos(t_emb)
    t_sin = torch.sin(t_emb)
    
    # Joint position component (augmented with tree structure)
    # Encode tree structure
    joint_depth_emb = encode_depth(joint_axis_pos, DEPTH_DICT, head_dim//2)
    joint_parent_emb = encode_parent_relationship(joint_axis_pos, PARENT_DICT, head_dim//2)
    joint_sibling_emb = encode_sibling_structure(joint_axis_pos, SIBLING_DICT, head_dim//2)
    
    # Combine with learned weights
    β_d, β_p, β_s = 0.5, 0.3, 0.2  # Learnable parameters
    joint_base = (joint_axis_pos / 23) * inv_freq
    joint_aug = joint_base + β_d * joint_depth_emb[:head_dim//2] + \
                β_p * joint_parent_emb[:head_dim//2] + \
                β_s * joint_sibling_emb[:head_dim//2]
    
    joint_cos = torch.cos(joint_aug)
    joint_sin = torch.sin(joint_aug)
    
    # Construct rotary matrices
    # For query-key attention, use both temporal and joint components
    cos_emb = torch.cat([t_cos, joint_cos])  # [head_dim]
    sin_emb = torch.cat([t_sin, joint_sin])  # [head_dim]
    
    # Return as rotation matrix
    return cos_emb, sin_emb  # To be used in rotating q and k
```

### 6. Integration into DiT Attention

```python
class KT_RoPE_Attention(nn.Module):
    """Attention module with KT-RoPE."""
    
    def forward(self, q, k, v, time_pos, joint_pos):
        """
        Args:
            q, k, v: Query, key, value of shape [batch, seq_len, num_heads, head_dim]
            time_pos: Temporal positions [seq_len] or [batch, seq_len]
            joint_pos: Joint positions [seq_len] or [batch, seq_len]
                      (repeats for each frame in sequence)
        """
        batch, seq_len, num_heads, head_dim = q.shape
        
        # Apply KT-RoPE to queries and keys
        for t_idx in range(seq_len):
            t = time_pos[t_idx] if time_pos.dim() > 1 else time_pos[t_idx]
            j = joint_pos[t_idx] if joint_pos.dim() > 1 else joint_pos[t_idx]
            
            cos_emb, sin_emb = compute_kt_rope(j, t, head_dim)
            
            # Rotate query and key
            q[:, t_idx] = apply_rotation(q[:, t_idx], cos_emb, sin_emb)
            k[:, t_idx] = apply_rotation(k[:, t_idx], cos_emb, sin_emb)
        
        # Standard attention computation
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = attn @ v
        
        return output
```

---

## SMPL Kinematic Tree Mappings

Create these dictionaries based on SMPL structure:

```python
# Joint depth in kinematic tree
DEPTH_DICT = {
    0: 0,   # Pelvis
    1: 1,   # L_Hip
    2: 1,   # R_Hip
    3: 1,   # Spine1
    # ... continue for all 23 joints
    22: 6,  # R_Wrist (or R_Hand depending on indexing)
}

# Parent joint for each joint
PARENT_DICT = {
    0: -1,  # Pelvis has no parent (root)
    1: 0,   # L_Hip parent is Pelvis
    2: 0,   # R_Hip parent is Pelvis
    3: 0,   # Spine1 parent is Pelvis
    # ... continue
    22: 20, # R_Wrist parent is R_Shoulder
}

# Sibling relationships (bilateral pairs)
SIBLING_DICT = {
    1: (0, -1),   # L_Hip: parent=Pelvis, left sibling
    2: (0, 1),    # R_Hip: parent=Pelvis, right sibling
    4: (0, -1),   # L_Knee: parent=L_Hip, left sibling
    # ... continue for all bilateral pairs
}
```

---

## Configuration Parameters

For the full KT-RoPE design (based on ablation results):

```yaml
# KT-RoPE Configuration
kt_rope:
  enabled: true
  encoding_type: "depth+parent"  # Options: depth_only, depth+parent, full
  
  # Tree encoding parameters
  depth_encoding:
    enabled: true
    dimension: 32  # Per-head, so total contribution is 32
    temperature: 1.0
  
  parent_encoding:
    enabled: true
    dimension: 32
    temperature: 1.0
  
  sibling_encoding:
    enabled: false  # Disabled due to diminishing returns
    dimension: 32
    temperature: 1.0
  
  # Learnable weighting parameters
  weights:
    beta_d: 0.5    # Depth weight
    beta_p: 0.3    # Parent weight
    beta_s: 0.2    # Sibling weight (unused if disabled)
    learnable: true  # Make these learnable during training
```

---

## Expected Results

Based on paper ablations:

| Configuration | HumanML3D FID | MotionHub FID | BABEL FID | Improvement |
|---------------|---------------|---------------|-----------|-------------|
| Baseline (standard RoPE) | 0.141 | 0.055 | 0.168 | — |
| Depth-only | 0.136 | 0.051 | 0.164 | 3.6%-7.2% |
| Depth+parent | 0.134 | 0.050 | 0.161 | 5.4%-9.4% |
| Full (+sibling) | 0.132 | 0.052 | 0.162 | 3.6%-6.4% |
| With KAFS | 0.128 | 0.049 | 0.158 | 10.9% combined |

---

## Testing Checklist

- [ ] Verify depth encoding matches SMPL tree structure
- [ ] Test parent-child relationship encoding for bilateral pairs
- [ ] Validate positional encoding dimensions match head_dim
- [ ] Check gradient flow through tree encoding functions
- [ ] Verify no NaN or Inf values in positional embeddings
- [ ] Compare against baseline RoPE on test set
- [ ] Ablate each component (depth, parent, sibling) separately
- [ ] Test on multiple datasets (HumanML3D, MotionHub, BABEL)
- [ ] Verify compatibility with existing KAFS implementation
- [ ] Profile training time (should be negligible overhead)

---

## Notes for Implementation

1. **Compatibility**: KT-RoPE only modifies the joint-axis RoPE; temporal-axis RoPE remains standard
2. **Backward Compatibility**: Existing checkpoints trained with standard RoPE can be fine-tuned with KT-RoPE
3. **Learnable Parameters**: β_d, β_p, β_s should be learnable parameters for best results
4. **Computational Cost**: Minimal overhead; encoding functions can be precomputed and cached
5. **Generalization**: Works with any skeleton structure if SMPL tree mappings are provided
6. **Combined with KAFS**: Apply KT-RoPE during training, then apply KAFS at inference for best results

---

## References

- Paper: PRISM TMM2026 (Section 3.1 and ablation in Section 4)
- Standard RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- SMPL: "Keep It SMPL: Automatic Estimation and Optimization of 3D Human Body Shape"

