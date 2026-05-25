# MMDiT Text Embedding Implementation - Code Reference

## File Structure

```
hftrainer/models/motion/hymotion_m2m/network/
├── hymotion_mmdit.py          # Main MMDiT transformer (THIS IS WHERE TEXT IS USED)
├── hymotion_dit.py            # Text-free variant for comparison
├── token_refiner.py           # SingleTokenRefiner (optional text preprocessing)
├── modulate.py                # AdaLN implementation (ModulateDiT)
├── attention.py               # Attention computation
├── encoders.py                # Input encoders (ctxt, vtxt, timestep)
├── bricks.py                  # Basic layers (norm, activation)
└── positional_encoding.py     # RoPE implementation
```

---

## Quick Code Lookup Table

| Question | File | Lines | Code |
|----------|------|-------|------|
| **Where is ctxt_encoder?** | hymotion_mmdit.py | 706 | `self.ctxt_encoder = nn.Linear(ctxt_input_dim → feat_dim)` |
| **Where is vtxt_encoder?** | hymotion_mmdit.py | 708 | `self.vtxt_encoder = MLPEncoder(vtxt_input_dim → feat_dim)` |
| **Where is adapter created?** | hymotion_mmdit.py | 855 | `adapter = timestep_feat + vtxt_feat` |
| **Where does ctxt flow?** | hymotion_mmdit.py | 887-890 | `ctxt_feat = ctxt_encoder; [text_refiner]; double_blocks` |
| **Joint attention implementation** | hymotion_mmdit.py | 286-327 | `q,k,v = concat([motion, text]); attention(q,k,v)` |
| **T→M blocking** | hymotion_mmdit.py | 1169 | `base[:,:,motion_len:,:motion_len] = float("-inf")` |
| **Text refiner application** | hymotion_mmdit.py | 889-890 | `if hasattr(self, "text_refiner"): ctxt_feat = self.text_refiner(...)` |
| **ModulateDiT definition** | modulate.py | 10-19 | `class ModulateDiT` |
| **How adapter is used** | hymotion_mmdit.py | 209-229 | `(shift,scale,gate) = self.motion_mod(adapter).chunk(6)` |

---

## Code Walkthrough: Forward Pass

### Step 1: Input Encoding (hymotion_mmdit.py, lines 824-856)

```python
def forward(self, x, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal, ...):
    device = get_module_device(self)
    
    # Encode motion
    if pre_encoded_motion is not None:
        motion_feat = pre_encoded_motion
    else:
        motion_feat = self.input_encoder(x)  # Linear: input_dim → feat_dim
        
    # Encode context text (token-level)
    ctxt_feat = self.ctxt_encoder(ctxt_input.float())  # Linear: 4096 → feat_dim
    
    # OPTIONAL: Refine text with self-attention
    if hasattr(self, "text_refiner"):
        ctxt_feat = self.text_refiner(x=ctxt_feat, t=timesteps, mask=(ctxt_key_padding_mask == 0).to(device))
    
    # Encode global text (sentence-level)
    timestep_feat = self.timestep_encoder(timesteps)  # (B,) → (B, 1, feat_dim)
    vtxt_feat = self.vtxt_encoder(vtxt_input.float())  # MLPEncoder: 768 → feat_dim
    
    # ⭐ CRITICAL: Combine vtxt with timestep for modulation
    adapter = timestep_feat + vtxt_feat  # (B, 1, feat_dim)
```

**Key Points:**
- `ctxt_feat`: Remains as sequence of tokens, will be used as cross-attention K,V
- `adapter`: Single vector per batch, used for AdaLN shift/scale/gate parameters
- `vtxt` is **NOT** kept as tokens; it's absorbed into adapter

### Step 2: Build Attention Masks (hymotion_mmdit.py, lines 857-910)

```python
# Convert boolean masks to additive format (0=valid, -inf=masked)
motion_key_padding_mask = self._canonical_mask(x_mask_temporal).to(device)
ctxt_key_padding_mask = self._canonical_mask(ctxt_mask_temporal).to(device)
seq_key_padding_mask = torch.cat((motion_key_padding_mask, ctxt_key_padding_mask), dim=1)

# Build sequence mask (causal/narrowband if specified)
if self.mask_mode is None:
    seq_mask = None
elif self.mask_mode == "causal":
    # Causal mask: only attend to past
    seq_mask = torch.triu(torch.full((motion_len, motion_len), float("-inf")), diagonal=1)
elif self.mask_mode == "narrowband":
    # Local attention window
    window = int(round(self.narrowband_length))
    idx = torch.arange(motion_len, device=device)
    dist = (idx[None, :] - idx[:, None]).abs()
    band = dist <= window
    seq_mask = torch.full((motion_len, motion_len), float("-inf"), device=device)
    seq_mask = seq_mask.masked_fill(band, 0.0)

# Build double-stream block attention mask
attn_mask_double = self._build_dmm_attn_mask_shared(
    bsz=bsz, motion_len=motion_len, text_len=text_len,
    dtype=mask_dtype, key_padding_mask=seq_key_padding_mask,
    attn_mask=seq_mask, device=device,
)
```

**Key Points:**
- Masks ensure padding positions are ignored
- Motion can attend to text (M→T allowed)
- Text CANNOT attend to motion (T→M blocked) — see next section

### Step 3: Double-Stream Blocks (hymotion_mmdit.py, lines 912-920)

```python
for i_layer, mod in enumerate(self.double_blocks):
    motion_feat, ctxt_feat = mod(
        motion_feat=motion_feat,      # (B, L_motion, feat_dim)
        text_feat=ctxt_feat,          # ⭐ ctxt_input is here as features
        adapter=adapter,              # (B, 1, feat_dim) for modulation only
        attn_mask=attn_mask_double,
    )
```

**What happens inside MMDoubleStreamBlock:**

```python
# hymotion_mmdit.py, lines 177-373, MMDoubleStreamBlock.forward()

# ===== MOTION STREAM =====
# Generate modulation parameters from adapter
(motion_shift_msa, motion_scale_msa, motion_gate_msa,
 motion_shift_mlp, motion_scale_mlp, motion_gate_mlp) = self.motion_mod(adapter).chunk(6, dim=-1)

# Apply layer norm + adaptive modulation
motion_modulated = self.motion_norm1(motion_feat)
motion_modulated = modulate(motion_modulated, shift=motion_shift_msa, scale=motion_scale_msa)

# Project to Q, K, V
motion_qkv = self.motion_qkv(motion_modulated)
motion_q, motion_k, motion_v = rearrange(motion_qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads)
motion_q = self.motion_q_norm(motion_q).to(motion_v)
motion_k = self.motion_k_norm(motion_k).to(motion_v)

# ===== TEXT STREAM =====
# Same structure: norm + modulate + project
text_modulated = self.text_norm1(text_feat)
text_modulated = modulate(text_modulated, shift=text_shift_msa, scale=text_scale_msa)
text_qkv = self.text_qkv(text_modulated)
text_q, text_k, text_v = rearrange(text_qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads)
text_q = self.text_q_norm(text_q).to(text_v)
text_k = self.text_k_norm(text_k).to(text_v)

# ===== JOINT ATTENTION ⭐ =====
# CONCATENATE motion and text Q, K, V
q = torch.cat((motion_q, text_q), dim=1)  # (B, L_motion+L_text, H, D)
k = torch.cat((motion_k, text_k), dim=1)
v = torch.cat((motion_v, text_v), dim=1)

# Single scaled dot-product attention
ret = attention(q, k, v, mode="torch", drop_rate=dropout_p, attn_mask=attn_mask, ...)

if isinstance(ret, tuple):
    attn_output, attn_w = ret
else:
    attn_output = ret

# Split back into motion and text portions
motion_attn_output = attn_output[:, :motion_len, ...]
text_attn_output = attn_output[:, motion_len:, ...]

# Residual connection with gating (adapter-modulated)
motion_feat = motion_feat + apply_gate(self.motion_out_proj(motion_attn_output), gate=motion_gate_msa)
text_feat = text_feat + apply_gate(self.text_out_proj(text_attn_output), gate=text_gate_msa)

# MLP with residual
motion_feat = motion_feat + apply_gate(
    self.motion_mlp(modulate(self.motion_norm2(motion_feat), shift=motion_shift_mlp, scale=motion_scale_mlp)),
    gate=motion_gate_mlp,
)
text_feat = text_feat + apply_gate(
    self.text_mlp(modulate(self.text_norm2(text_feat), shift=text_shift_mlp, scale=text_scale_mlp)),
    gate=text_gate_mlp,
)
```

**Key Points:**
- ✅ Motion queries attend to text keys/values
- ✅ Text queries attend to motion (BUT BLOCKED by mask)
- ✅ Modulation (shift/scale/gate) from adapter applies to BOTH streams
- ✅ Text embeddings directly used as K,V for cross-attention

### Step 4: Single-Stream Blocks (hymotion_mmdit.py, lines 922-945)

```python
# Concatenate motion and text for single-stream processing
split_len = motion_feat.shape[1]
x = torch.cat((motion_feat, ctxt_feat), 1)  # (B, L_motion+L_text, feat_dim)

# Build single-stream attention mask
attn_mask_single = self._build_smm_attn_mask_shared(
    bsz=bsz, split_len=split_len, total_len=total_len,
    dtype=mask_dtype, key_padding_mask=seq_key_padding_mask,
    attn_mask=seq_mask, device=device,
)
# Note: This also has T→M blocking at line 1229

# Process through single-stream blocks
for i_layer, mod in enumerate(self.single_blocks):
    x = mod(
        x=x,
        split_len=split_len,
        adapter=adapter,              # ⭐ adapter still used for modulation
        attn_mask=attn_mask_single,
    )
```

**What happens inside MMSingleStreamBlock:**

```python
# hymotion_mmdit.py, lines 467-568, MMSingleStreamBlock.forward()

# Generate modulation parameters (factor=3: shift, scale, gate)
shift_msa, scale_msa, gate_msa = self.modulation(adapter).chunk(3, dim=-1)

# Apply norm + modulation
x_modulated = modulate(self.norm(x), shift_msa, scale_msa)

# Fused linear: computes QKV + MLP_hidden simultaneously
if self.elementwise_attn_output_gate:
    qkv, mlp_hidden = torch.split(self.linear1(x_modulated), [4*feat_dim, mlp_hidden_dim])
    q, k, v, g = rearrange(qkv, "B L (K H D) -> K B L H D", K=4, H=num_heads)
else:
    qkv, mlp_hidden = torch.split(self.linear1(x_modulated), [3*feat_dim, mlp_hidden_dim])
    q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=num_heads)
    g = None

q = self.q_norm(q).to(v)
k = self.k_norm(k).to(v)

# Split Q/K into motion and text portions for RoPE
q1, q2 = q[:, :split_len, ...], q[:, split_len:, ...]
k1, k2 = k[:, :split_len, ...], k[:, split_len:, ...]

# Apply RoPE (only to motion if apply_rope_to_single_branch=True)
if self.apply_rope_to_single_branch:
    q1, k1 = self.rotary_emb.apply_rotary_emb(q1, k1)

q = torch.cat((q1, q2), dim=1)
k = torch.cat((k1, k2), dim=1)

# Joint attention with T→M blocking via attn_mask
ret = attention(q, k, v, mode="torch", attn_mask=attn_mask, ...)

if isinstance(ret, tuple):
    attn_output, attn_w = ret
else:
    attn_output = ret

# Fused output: [attn_out || mlp_act(mlp_hidden)] → linear2
output = self.linear2(torch.cat((attn_output, self.mlp_act(mlp_hidden)), 2))

# Residual + gate (from adapter modulation)
return x + apply_gate(output, gate=gate_msa)
```

### Step 5: Output Extraction (hymotion_mmdit.py, lines 947-962)

```python
# Extract motion portion only (discard text)
x = x[:, :split_len, ...]

# Remove start token if inserted
if self.insert_start_token:
    x = x[:, 1:, ...]

# Long skip connection
if self.with_long_skip_connection:
    x = self.long_skip_net(origin_feat, timestep_feat) + x

# Final layer with adapter modulation
predicted_res = self.final_layer(x, adapter)
return predicted_res
```

---

## Critical Code Sections

### 1. Attention Mask for T→M Blocking

**File:** hymotion_mmdit.py, lines 1112-1170

```python
def _build_dmm_attn_mask_shared(self, bsz, motion_len, text_len, dtype, 
                                key_padding_mask, attn_mask, device):
    """
    Builds attention mask with T→M blocking.
    
    Pattern:
        motion_q: [allowed to see all]
        text_q:   [allowed to see text_k only, NOT motion_k]
    """
    total_len = motion_len + text_len
    base = torch.zeros((bsz, 1, total_len, total_len), dtype=dtype, device=device)
    
    # Apply sequence mask (causal/narrowband)
    if attn_mask is not None:
        base[:, :, :motion_len, :motion_len] += attn_mask.view(1, 1, motion_len, motion_len)
    
    # Apply padding mask
    if key_padding_mask is not None:
        base = base + key_padding_mask.view(bsz, 1, 1, total_len)
    
    # ⭐ BLOCK TEXT FROM ATTENDING TO MOTION (T→M)
    base[:, :, motion_len:, :motion_len] = float("-inf")
    
    return base  # (B, 1, total_len, total_len)
```

**Why this matters:**
- Text queries (indices [motion_len:]) cannot attend to motion keys (indices [:motion_len])
- This prevents noisy motion from corrupting text representations during diffusion
- Motion CAN still attend to text (M→T allowed)

### 2. ModulateDiT (Adaptive Layer Norm)

**File:** modulate.py, lines 10-19

```python
class ModulateDiT(nn.Module):
    """Generates shift, scale, gate parameters from adapter."""
    
    def __init__(self, feat_dim: int, factor: int, act_type: str = "silu"):
        super().__init__()
        self.act = get_activation_layer(act_type)()
        # Maps feat_dim → (factor * feat_dim)
        # factor=6: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.linear = nn.Linear(feat_dim, factor * feat_dim, bias=True)
        # ⭐ ZERO-INITIALIZE for stable training
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.act(x))
```

**Usage in double-stream:**
```python
mods = self.motion_mod(adapter).chunk(6, dim=-1)  # (B, 1, feat_dim) → 6×(B, 1, feat_dim/6)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods
```

### 3. Token Refiner (Optional Text Preprocessing)

**File:** hymotion_mmdit.py, lines 716-721

```python
# In __init__
if text_refiner_module != "" and text_refiner_module is not None:
    text_refiner_cfg.update(input_dim=feat_dim, feat_dim=feat_dim, num_heads=num_heads)
    self._text_refiner_cfg = text_refiner_cfg.copy()
    self.text_refiner = SingleTokenRefiner(**text_refiner_cfg)
```

**In forward (line 889-890):**
```python
ctxt_feat = self.ctxt_encoder(ctxt_input.float())

# Optional: Self-attention refinement
if hasattr(self, "text_refiner"):
    ctxt_feat = self.text_refiner(
        x=ctxt_feat,  # (B, L_text, feat_dim)
        t=timesteps,  # (B,) diffusion timestep
        mask=(ctxt_key_padding_mask == 0).to(device)  # (B, L_text) valid positions
    )
```

**What it does (token_refiner.py, lines 176-192):**
```python
def forward(self, x: Tensor, t: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    # 1. Encode timestep
    timestep_aware_representations = self.timestep_encoder(t)
    
    # 2. Pool context (mean of valid tokens)
    if mask is None:
        context_aware_representations = x.mean(dim=1)
    else:
        mask_float = mask.float().unsqueeze(-1)
        denom = mask_float.sum(dim=1).clamp_min(1e-6)
        context_aware_representations = (x * mask_float).sum(dim=1) / denom
    
    # 3. Combine timestep + context
    context_aware_representations = self.context_encoder(context_aware_representations).unsqueeze(1)
    c = timestep_aware_representations + context_aware_representations  # (B, 1, feat_dim)
    
    # 4. Refine text with self-attention conditioned on timestep+context
    x = self.input_embedder(x)
    x = self.individual_token_refiner(x, c, mask)  # Self-attention layers
    
    return x
```

---

## Key Encoder Implementations

### ctxt_encoder (Line 706)

```python
self.ctxt_encoder = nn.Linear(
    in_features=ctxt_input_dim,  # 4096 (Qwen3 token embeddings)
    out_features=feat_dim         # typically 1024
)

# Usage in forward:
ctxt_feat = self.ctxt_encoder(ctxt_input.float())
# Input:  (B, L_text, 4096)
# Output: (B, L_text, feat_dim)
```

**Purpose:** Simple linear projection preserving token sequence

### vtxt_encoder (Line 708)

```python
self.vtxt_encoder = MLPEncoder(
    in_dim=vtxt_input_dim,        # 768 (CLIP sentence embedding)
    feat_dim=feat_dim,             # typically 1024
    num_layers=2,
    act_type="silu"
)

# Usage in forward:
vtxt_feat = self.vtxt_encoder(vtxt_input.float())
# Input:  (B, 1, 768)
# Output: (B, 1, feat_dim)
```

**Purpose:** Non-linear processing + scaling to hidden dimension

### timestep_encoder (Line 710)

```python
self.timestep_encoder = TimestepEmbeddingEncoder(
    embedding_dim=feat_dim,
    feat_dim=feat_dim,
    time_factor=time_factor  # controls frequency scale
)

# Usage in forward:
timestep_feat = self.timestep_encoder(timesteps)
# Input:  (B,) - diffusion timesteps
# Output: (B, 1, feat_dim)
```

**Purpose:** Sinusoidal positional encoding + MLP projection

---

## Data Type Conversions

Note the `.float()` calls in forward pass:

```python
# Line 852
ctxt_feat = self.ctxt_encoder(ctxt_input.float())

# Line 852
vtxt_feat = self.vtxt_encoder(vtxt_input.float())

# Line 887
ctxt_feat = self.ctxt_encoder(ctxt_input.float())
```

**Why:** Ensures inputs are in float32 even if loaded as float16 (mixed precision)

---

## Testing the Implementation

From hymotion_mmdit.py, lines 1499-1536:

```python
if __name__ == "__main__":
    from configs._base_.model_network_base import MOTION_MODEL_CONFIG
    
    network_module_cfg = MOTION_MODEL_CONFIG["1.04B_narrowband"]["network_module_args"]
    network_module_cfg = dict(network_module_cfg)
    
    bsz, seq_len, text_seq_len, input_dim = 1, 360, 128, 201
    network_module_cfg["input_dim"] = input_dim
    MMDiT = HunyuanMotionMMDiT(**network_module_cfg)
    
    # Generate random inputs
    x = torch.randn(bsz, seq_len, input_dim)                    # Motion
    ctxt_condition = torch.randn(bsz, text_seq_len, 4096)       # ← Qwen3 (4096-dim)
    vtxt_condition = torch.randn(bsz, 1, 768)                   # ← CLIP (768-dim)
    timesteps = torch.randint(0, 1000, (bsz,))
    
    # Create masks
    length = torch.arange(seq_len).unsqueeze(0).repeat(bsz, 1)
    ctxt_length = torch.arange(text_seq_len).unsqueeze(0).repeat(bsz, 1)
    x_mask_temporal = length < 100
    ctxt_mask_temporal = ctxt_length < 50
    
    # Forward pass
    output = MMDiT(
        x=x,
        ctxt_input=ctxt_condition,     # ← Both text inputs used here
        vtxt_input=vtxt_condition,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
    )
    
    assert output.shape == (bsz, seq_len, input_dim)
    print(f"✓ Output shape: {output.shape}")
```

---

## Summary: Where Text Is Used

| Text Type | Variable | Encoding | Usage |
|-----------|----------|----------|-------|
| **ctxt** (Qwen3 4096D) | `ctxt_input` | `nn.Linear(4096→D)` | K,V in double-stream joint attention |
| **ctxt** (refined) | `ctxt_feat` | `text_refiner()` | Optional self-attention before main blocks |
| **vtxt** (CLIP 768D) | `vtxt_input` | `MLPEncoder(768→D)` | AdaLN modulation (shift/scale/gate) |
| **timestep** | `timesteps` | Sinusoidal+MLP | AdaLN modulation (combined with vtxt) |

**Flow:** ctxt → attention K,V; vtxt → modulation adapter; both enable cross-modal control
