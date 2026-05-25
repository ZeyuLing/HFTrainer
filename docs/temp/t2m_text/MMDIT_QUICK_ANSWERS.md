# MMDiT Text Embeddings - Quick Answers

## Your 5 Questions, Answered

### (a) How ctxt_input (4096-dim token-level Qwen3) enters the transformer

**SHORT ANSWER:** 
- Via **joint attention in double-stream blocks (first 1/3 of layers)**
- Becomes **Keys and Values** for motion queries to attend to
- Each motion query can see every text token via attention mechanism

**DETAILED:**
```
ctxt_input (B, L_text, 4096)
    ↓ (Line 706)
ctxt_encoder: Linear(4096 → feat_dim)
    ↓ 
ctxt_feat (B, L_text, feat_dim)
    ↓ (Optional, Line 889-890)
[text_refiner: self-attention refinement]
    ↓ (Line 915)
→ Double-stream block (text_feat parameter)
    ↓ (Lines 260-327)
    ├─ Layer Norm + Modulation (from adapter)
    ├─ Project to K, V (text becomes keys/values)
    └─ CONCATENATE with motion Q, K, V
        ↓ (Line 289)
        → Single attention matrix: [motion_q, text_q] × [motion_k, text_k]^T
        → Result: Motion attends to text tokens
```

**Layers Used:** `self.double_blocks` (first 1/3 of `num_layers`)  
**Mechanism:** Concatenated joint attention (not traditional cross-attention)  
**Can motion attend to text?** ✅ YES  
**Can text attend to motion?** ⚠️ YES in math, but BLOCKED by attention mask

---

### (b) How vtxt_input (768-dim sentence-level CLIP) enters

**SHORT ANSWER:**
- Via **AdaLN (Adaptive Layer Normalization) modulation**
- Combined with timestep: `adapter = timestep_encoder(t) + vtxt_encoder(vtxt)`
- Used to generate shift, scale, gate parameters for ALL transformer layers
- **NOT used for cross-attention**

**DETAILED:**
```
vtxt_input (B, 1, 768)
    ↓ (Line 708)
vtxt_encoder: MLPEncoder(768 → feat_dim)
    ↓
vtxt_feat (B, 1, feat_dim)
    ↓ (Line 855)
timestep_feat (B, 1, feat_dim) + vtxt_feat (B, 1, feat_dim)
    ↓
adapter (B, 1, feat_dim) ← Single vector per batch
    ↓ (Lines 209-229, 493-497, and many more)
    → ModulateDiT(adapter).chunk(6) 
        → [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
    → Applied to:
        ├─ Motion stream: modulate(motion_feat, shift, scale)
        ├─ Text stream: modulate(text_feat, shift, scale)
        └─ Output gating: apply_gate(output, gate)
    → Used in EVERY layer (double + single blocks)
```

**Layers Used:** ALL layers (double + single blocks)  
**Mechanism:** Broadcast conditioning (same adapter vector repeated)  
**Information type:** Global context (no per-token information)  
**Data flow:** NOT as tokens, but as continuous modulation parameters

---

### (c) Is there a "text_refiner" that processes text before transformer?

**SHORT ANSWER:**
- ✅ **YES: `SingleTokenRefiner`**
- Located in `hftrainer/models/motion/hymotion_m2m/network/token_refiner.py`
- Applied after `ctxt_encoder`, before double-stream blocks
- Optional (configurable via `text_refiner_module` parameter)

**DETAILED:**
```python
# Enabled by default (line 617)
text_refiner_module: str = "hymotion/network/token_refiner.SingleTokenRefiner"
text_refiner_cfg: dict = {"num_layers": 2}

# Instantiation (lines 718-721)
if text_refiner_module != "" and text_refiner_module is not None:
    self.text_refiner = SingleTokenRefiner(
        input_dim=feat_dim, 
        feat_dim=feat_dim, 
        num_heads=num_heads,
        num_layers=2
    )

# Application (lines 889-890)
if hasattr(self, "text_refiner"):
    ctxt_feat = self.text_refiner(x=ctxt_feat, t=timesteps, mask=...)
```

**What it does:**
1. Encodes diffusion timestep
2. Pools context (mean of valid text tokens)
3. Combines timestep + pooled context
4. Self-attention refinement over text tokens (2 layers by default)

**Purpose:** Refine text representations with timestep awareness before cross-attention

**Result:** Enhanced text embeddings that are timestep-aware and self-refined

---

### (d) Are there separate projection layers for vtxt and ctxt?

**SHORT ANSWER:**
- ✅ **YES, completely separate:**
  - `ctxt_encoder`: `nn.Linear(4096 → feat_dim)` — simple linear
  - `vtxt_encoder`: `MLPEncoder(768 → feat_dim)` — 2-layer MLP

**DETAILED:**
```python
# ctxt encoder (line 706)
self.ctxt_encoder = nn.Linear(in_features=ctxt_input_dim, out_features=feat_dim)
# Input:  (B, L_text, 4096) - Qwen3 embeddings, token-level
# Output: (B, L_text, feat_dim) - Preserves sequence structure
# Design: Simple linear projection (preserves information)

# vtxt encoder (line 708)
self.vtxt_encoder = MLPEncoder(
    in_dim=vtxt_input_dim,        # 768
    feat_dim=feat_dim,             # typically 1024
    num_layers=2, 
    act_type="silu"
)
# Input:  (B, 1, 768) - CLIP embeddings, sentence-level
# Output: (B, 1, feat_dim)
# Design: 2-layer MLP (non-linear processing, global aggregation)

# timestep encoder (line 710)
self.timestep_encoder = TimestepEmbeddingEncoder(
    embedding_dim=feat_dim,
    feat_dim=feat_dim,
    time_factor=time_factor
)
# Input:  (B,) - integer timesteps
# Output: (B, 1, feat_dim)
# Design: Sinusoidal embedding + MLP (adds positional information)
```

**Why different architectures?**
- **ctxt:** Needs to preserve token information → simple linear projection
- **vtxt:** Single vector for global conditioning → can use non-linear MLP
- **timestep:** Positional information about diffusion progress → sinusoidal embedding

---

### (e) What happens if ctxt or vtxt are all zeros or all the same value?

**SHORT ANSWER:**
- ❌ **NO attention collapse**
- ⚠️ **Information flow degrades**, but architecture remains stable

**DETAILED ANALYSIS:**

#### Scenario 1: All-Zero ctxt_input
```python
ctxt_input = torch.zeros(B, L_text, 4096)
ctxt_feat = ctxt_encoder(ctxt_input)  # → all zeros (B, L_text, feat_dim)

# In double-stream attention:
text_qkv = text_qkv_proj(text_feat)  # → zeros
text_q, text_k, text_v = zeros

# Concatenated attention matrix:
# Q = [motion_q, 0]
# K = [motion_k, 0]
# V = [motion_v, 0]

# Attention: Q × K^T → zeros for motion-to-text attention
# But: Motion-to-motion attention continues (diagonal terms are motion_q × motion_k)
# Result: Motion self-attention works, text contributes zero
```

**Impact:** ⚠️ **Partial degradation — text provides no information, but system is stable**

#### Scenario 2: All-Zero vtxt_input
```python
vtxt_input = torch.zeros(B, 1, 768)
vtxt_feat = vtxt_encoder(vtxt_input)  # → zeros (B, 1, feat_dim)

# Adapter formation:
adapter = timestep_feat + 0  # → adapter ≈ timestep_feat only

# Modulation:
modulation_output = motion_mod(adapter)
# motion_mod is zero-initialized, so early in training:
# modulation_output ≈ 0 → shift ≈ 0, scale ≈ 0, gate ≈ 0

# Modulation effect:
motion_modulated = motion * (1 + scale) + shift ≈ motion * 1.0 + 0 ≈ motion
# Result: Layers become nearly identity transforms
```

**Impact:** ⚠️ **Minimal effect — vtxt is conditioning signal, becomes timestep-only**

#### Scenario 3: Constant ctxt_input (all same value)
```python
ctxt_input = torch.ones(B, L_text, 4096) * c  # Every token is identical
ctxt_feat = ctxt_encoder(ctxt_input)  # All tokens project to same embedding

# QKV projection:
text_q, text_k, text_v  # All queries/keys/values are identical
# Attention: Q × K^T where Q = K
# Softmax(Q × K^T / sqrt(D)) = uniform distribution

# Result: Text tokens attend uniformly to each other
# Softmax is numerically stable (outputs all 0.25 for 4 tokens, etc.)
# No NaN or Inf values
```

**Impact:** ⚠️ **No collapse, but text carries zero discriminative information**

#### Scenario 4: Constant vtxt_input
```python
vtxt_input = torch.ones(B, 1, 768) * c  # Constant global embedding
vtxt_feat = vtxt_encoder(vtxt_input)  # Same output

adapter = timestep_feat + same_vector
# Modulation becomes: batch-uniform per timestep
# But timestep still varies across diffusion steps
```

**Impact:** ⚠️ **No collapse, modulation is uniform per batch but varies with timestep**

#### SUMMARY TABLE:

| Input | Math Status | Information Flow | Training Stability | Recommendation |
|-------|-------------|-----------------|-------------------|-----------------|
| **Zero ctxt** | ✅ Stable | ❌ Blocked (K,V = 0) | ✅ Stable | ✅ Safe to experiment |
| **Zero vtxt** | ✅ Stable | ⚠️ Reduced (timestep-only) | ✅ Stable | ✅ Acceptable fallback |
| **Const ctxt** | ✅ Stable | ❌ No-op (uniform attn) | ✅ Stable | ⚠️ Reduces model capacity |
| **Const vtxt** | ✅ Stable | ⚠️ Reduced | ✅ Stable | ⚠️ Reduces conditioning |

#### Why No Collapse?

1. **Motion has self-attention:** Even with zero text, motion queries can attend to motion keys (diagonal of attention matrix). Softmax is well-defined.

2. **Timestep always varies:** The timestep signal is always present and varies across the diffusion schedule. Model doesn't lose all conditioning.

3. **Softmax is numerically robust:** Even with uniform values, `softmax` outputs valid probabilities (normalized to 1.0 per row). No NaN/Inf unless mathematical operations fail (e.g., log(0), but softmax avoids this).

4. **Residual connections:** Each layer is residual (y = x + f(x)), so layers can pass through unchanged if necessary.

5. **Zero-initialized modulation:** Modulation starts as identity transform (shift=0, scale=0, gate=0 after zero-init + SiLU), so network doesn't completely break.

---

## Summary: Text Information Pathways

```
┌─────────────────────────────────────────────────────────────┐
│                  TWO SEPARATE TEXT PATHWAYS                 │
└─────────────────────────────────────────────────────────────┘

PATHWAY 1: CTXT (Token-Level Information)
  ctxt_input (4096-dim) 
    ↓ Linear encoder
  ctxt_feat (B, L_text, D)
    ↓ [Optional text_refiner]
  ctxt_refined
    ↓ 
  DOUBLE-STREAM BLOCKS (1/3 of layers):
    Joint attention: Motion queries attend to text keys/values
    Result: Token-level semantic information flows to motion
  
  SINGLE-STREAM BLOCKS (2/3 of layers):
    Concatenated: [motion | text] processed together
    T→M blocked: Text cannot influence motion through attention


PATHWAY 2: VTXT (Global Conditioning)
  vtxt_input (768-dim)
    ↓ MLPEncoder
  vtxt_feat (B, 1, D)
    ↓ Add timestep_feat
  adapter (B, 1, D)
    ↓
  ModulateDiT (ALL layers):
    → Shift parameters
    → Scale parameters  
    → Gate parameters
    Result: Global context + timestep awareness applied everywhere


COMBINED EFFECT:
  ✓ Motion has access to semantic guidance (via ctxt cross-attention)
  ✓ Motion has access to global conditioning (via vtxt modulation)
  ✓ Motion responds to diffusion progress (via timestep modulation)
  ✗ Text protected from noisy motion (T→M blocked)
  ✗ No information loss (all ctxt tokens flow through)
```

---

## Files to Read

1. **Main architecture:** `hymotion_mmdit.py` (main transformer)
2. **Double-stream blocks:** `hymotion_mmdit.py` lines 50-373 (MMDoubleStreamBlock)
3. **Single-stream blocks:** `hymotion_mmdit.py` lines 376-569 (MMSingleStreamBlock)
4. **Text refiner:** `token_refiner.py` lines 133-192 (SingleTokenRefiner)
5. **Modulation:** `modulate.py` lines 10-47 (ModulateDiT and helper functions)
6. **Attention masking:** `hymotion_mmdit.py` lines 1112-1230 (mask building)
7. **Forward pass:** `hymotion_mmdit.py` lines 777-962 (HunyuanMotionMMDiT.forward)

---

## Bottom Line

✅ **YES, the transformer DOES use text embeddings.**

- **ctxt (4096-dim Qwen3):** Used in double-stream blocks as keys/values for cross-attention
- **vtxt (768-dim CLIP):** Used in all layers as modulation conditioning (shift/scale/gate)
- **Both pathways are essential:** ctxt provides semantic detail, vtxt provides global context
- **Architecture is stable:** No attention collapse even with zero/constant inputs

