# HyMotion M2M: Text Embedding Flow — Training & Inference

**Comprehensive trace of text embeddings from input to model prediction and loss**

---

## Part 1: Training Flow

### 1.1 Data Loading & Pre-extraction

```
Sample from DataLoader
├─ Raw data:
│  ├─ prompt (str): "a person walks forward"
│  ├─ motion (T, 22, 3): raw SMPL-22 motion frames
│  └─ [optional metadata]
│
└─ In collate_fn or trainer preprocessing:
   ├─ text_encoder.encode(prompt)
   │  └─ Returns: (text_vec_raw, text_ctxt_raw)
   │     ├─ text_vec_raw (B, 1, 768): sentence embedding
   │     └─ text_ctxt_raw (B, L_c, 4096): token embeddings
   │
   └─ Store in batch dict for training
```

### 1.2 _prepare_and_forward() in HyMotionM2MTrainer

**Key method signature:**
```python
def _prepare_and_forward(self, batch, cond_mask_prob: float = 0.0):
    """
    Prepare batch data and forward through model.
    
    Args:
        batch: Contains motion, text_vec_raw, text_ctxt_raw, etc.
        cond_mask_prob: CFG dropout rate (e.g., 0.1 for 10%)
    """
```

**Processing steps:**

#### **Step 1: Extract embeddings from batch**
```python
# Assuming batch is a dict with pre-extracted embeddings
text_vec_raw = batch['text_vec']      # shape (B, 1, 768)
text_ctxt_raw = batch['text_ctxt']    # shape (B, L_c, 4096)
motion = batch['motion']               # shape (B, T, 22*3) or (B, T, 66)
motion_mask = batch['motion_mask']     # shape (B, T) or None
text_mask = batch['text_mask']         # shape (B, L_c) or None
```

#### **Step 2: Apply CFG dropout (mask_text_cond)**
```python
if cond_mask_prob > 0.0:
    text_vec, text_ctxt, text_available = bundle.mask_text_cond(
        text_vec_raw,
        text_ctxt_raw,
        force_mask=False,
        cond_mask_prob=cond_mask_prob,
        return_text_available=True  # Track which samples were masked
    )
else:
    text_vec = text_vec_raw
    text_ctxt = text_ctxt_raw
    text_available = torch.ones(B, dtype=torch.bool)

# After mask_text_cond (assuming cond_mask_prob=0.1, batch_size=32):
# text_vec shape (32, 1, 768) — ~3 samples replaced with null_vtxt_feat
# text_ctxt shape (32, L_c, 4096) — ~3 samples replaced with null_ctxt_input
# text_available shape (32,) — [True, False, True, ...] mask
```

**Detailed mask application:**

```
Before mask_text_cond:
┌─────────────────────────────────────────────────────────┐
│ text_vec_raw:                                           │
│ ┌─────────┬─────────┬─────────┬─────────┐              │
│ │ sample0 │ sample1 │ sample2 │sample31 │ (B, 1, 768)  │
│ │(real)   │(real)   │(real)   │(real)   │              │
│ └─────────┴─────────┴─────────┴─────────┘              │
└─────────────────────────────────────────────────────────┘

Bernoulli mask (cond_mask_prob=0.1):
┌─────────────────────────────────────────────────────────┐
│ [0, 1, 0, 0, 1, 0, ..., 0, 1, 0]  (32,)               │
│  ↓ reshaped to (32, 1) ↓                               │
│ [[0], [1], [0], [0], [1], [0], ..., [1], [0]]         │
│  ↓ unsqueezed to (32, 1, 1) ↓                          │
│ [[[0]], [[1]], [[0]], ..., [[1]], [[0]]]               │
└─────────────────────────────────────────────────────────┘

After torch.where(mask, null_vtxt_feat, text_vec_raw):
┌─────────────────────────────────────────────────────────┐
│ text_vec (output):                                      │
│ ┌─────────┬──────────┬─────────┬─────────┐             │
│ │ sample0 │ null(!)  │ sample2 │sample31 │ (B, 1, 768) │
│ │(real)   │(masked)  │(real)   │(real)   │             │
│ └─────────┴──────────┴─────────┴─────────┘             │
│            ↑         ↑                                  │
│            replaced with null_vtxt_feat[0,0,:]         │
│            broadcasted to (1, 768)                     │
└─────────────────────────────────────────────────────────┘

text_available = ~mask:
┌─────────────────────────────────────────────────────────┐
│ [True, False, True, True, False, True, ..., False, True] │
│  S0      S1     S2    S3     S4     S5  ...    S29  S31 │
└─────────────────────────────────────────────────────────┘
```

#### **Step 3: Normalize motion**
```python
motion = (motion - motion_mean) / motion_std  # Normalize by dataset statistics
```

#### **Step 4: Forward through model**
```python
# Model takes:
#   - motion: (B, T, 66) normalized motion
#   - text_vec: (B, 1, 768) sentence embedding (or null)
#   - text_ctxt: (B, L_c, 4096) token embeddings (or null)
#   - timestep: diffusion timestep
#   - [optional masks]
#
# Model returns:
#   - pred: (B, T, 66) predicted velocity or noise

pred = model.predict_flow(
    motion=motion,
    text_vec=text_vec,
    text_ctxt=text_ctxt,
    timestep=timestep,
    motion_mask=motion_mask,
    text_mask=text_mask,
)
```

**Inside model.predict_flow():**
```
Cross-attention in MMDiT:
├─ text_vec (B, 1, 768) → projected to hidden_dim → cross-attn with motion
├─ text_ctxt (B, L_c, 4096) → projected to hidden_dim → cross-attn with motion
└─ Motion tokens attend to text (real or null)
```

#### **Step 5: Compute loss**
```python
loss = m2m_loss(
    pred=pred,
    target=target,
    text_available=text_available,  # Flag for masked samples
)

# Loss computation may:
# - Weight masked samples lower (training signal stronger when text given)
# - Log separate metrics for text_available=True vs. False
# - Apply auxiliary losses (e.g., motion smoothness, FK consistency)
```

### 1.3 Gradient Flow During Training

```
┌──────────────────────────────────────┐
│ Loss = MSE(pred, target)             │
│ (potentially weighted by text_avail) │
└──────────────┬───────────────────────┘
               │
               ▼ backward()
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
   Model weights   Text embeddings
        │             │
        │             ├─ text_vec_raw (frozen from text encoder)
        │             ├─ text_ctxt_raw (frozen from text encoder)
        │             │
        │             └─ null_vtxt_feat ✓ (TRAINABLE)
        │                null_ctxt_input ✓ (TRAINABLE)
        │
        └─ Gradient accumulates into null_vtxt_feat
           (whenever a sample was masked and contributed to loss)
```

**Key:** Gradients flow into `null_vtxt_feat` and `null_ctxt_input` whenever a masked sample was used. Over many training steps, these parameters learn the optimal "no text" representation.

---

## Part 2: Inference Flow (with CFG)

### 2.1 Pipeline setup

```
Sample from val/test set
├─ prompt (str): "a person walks forward"
├─ motion_cond (T_cond, 66): conditioning frames
└─ [optional metadata]

↓

Inference pipeline loads:
├─ Pre-extracted embeddings (cached or computed)
│  ├─ text_vec: (1, 1, 768) sentence embedding
│  └─ text_ctxt: (1, L_c, 4096) token embeddings
│
└─ Model in eval mode (no gradients, dropout disabled)
```

### 2.2 Denoising loop with CFG

**High-level structure:**
```python
def sample_with_cfg(
    motion_cond,
    text_vec,
    text_ctxt,
    guidance_scale=7.5,
    num_steps=50,
):
    x_t = torch.randn(...)  # Initial noise
    
    for step in num_steps:
        # ─────────────────────────────────────────────────
        # TWO forward passes: with text and with null
        # ─────────────────────────────────────────────────
        
        # Forward pass 1: Real text
        noise_pred_with_text = model.predict_flow(
            motion=torch.cat([motion_cond, x_t], dim=1),  # Full context
            text_vec=text_vec,          # Real text
            text_ctxt=text_ctxt,        # Real tokens
            timestep=t,
        )
        
        # Forward pass 2: Null text (CFG)
        noise_pred_null = model.predict_flow(
            motion=torch.cat([motion_cond, x_t], dim=1),
            text_vec=bundle.null_vtxt_feat.expand_as(text_vec),  # FORCE NULL
            text_ctxt=bundle.null_ctxt_input.expand_as(text_ctxt),  # FORCE NULL
            timestep=t,
        )
        
        # ─────────────────────────────────────────────────
        # Compute guided prediction
        # ─────────────────────────────────────────────────
        noise_pred_guided = (
            noise_pred_null + guidance_scale * 
            (noise_pred_with_text - noise_pred_null)
        )
        # Equivalently:
        # noise_pred_guided = (1 - guidance_scale) * noise_pred_null 
        #                   + guidance_scale * noise_pred_with_text
        
        # ─────────────────────────────────────────────────
        # Denoising step
        # ─────────────────────────────────────────────────
        x_t = denoise_step(x_t, noise_pred_guided, alpha, beta, t)
    
    return x_t  # Denoised motion
```

### 2.3 Detailed CFG Mechanism

**Without CFG (guidance_scale = 1.0):**
```
noise_pred_guided = noise_pred_null + 1.0 * (noise_pred_with_text - noise_pred_null)
                  = noise_pred_with_text
                  → Model prediction conditioned on text (no guidance)
```

**With strong CFG (guidance_scale = 7.5):**
```
noise_pred_guided = noise_pred_null + 7.5 * (noise_pred_with_text - noise_pred_null)
                  = -6.5 * noise_pred_null + 7.5 * noise_pred_with_text
                  → Amplified text influence + suppressed null influence
                  → Stronger adherence to prompt
```

**Visual comparison:**

```
Noise space at denoising step t:

                    Typical region of x_t
                         ╱╲
           ╱─────────────╱  ╲─────────────╲
          ╱               ╲  ╱             ╲
         │   noise_pred_null        noise_pred_with_text   │
         │        ↑                       ↑                 │
         │        │                       │                 │
         │        │ (model sees null)    │ (model sees text)│
         └────────┴───────────────────────┴─────────────────┘

noise_pred_with_text - noise_pred_null = "text direction"
                ↑
      (the vector that moved model prediction
       when given text instead of null)

With guidance_scale = 7.5:
  We move 7.5× further in the "text direction"
  → Stronger text influence → Better prompt adherence
```

### 2.4 Why force_mask=True in Inference?

In the inference pipeline, when computing the null prediction:

```python
null_vtxt, null_ctxt = bundle.mask_text_cond(
    text_vec,
    text_ctxt,
    force_mask=True,  # ← Always use null embeddings
    return_text_available=False,
)
```

This calls the method's first branch:
```python
if force_mask:
    text_available.fill_(False)
    result = (
        self.null_vtxt_feat.expand(*vtxt.shape),      # (1,1,768)
        self.null_ctxt_input.expand(*ctxt.shape),     # (1,L_c,4096)
    )
    return result
```

**Semantics:** "Override input embeddings; use learned null instead."

---

## Part 3: Why Trainable Null Embeddings Matter

### 3.1 The CFG Training-Inference Mismatch Problem

**Scenario 1: Frozen null (all zeros)**

```
Training:
├─ mask_text_cond: replaces ~10% with zeros
├─ Model learns: "motion X → real embedding E → feature F_real"
│                "motion X → zero embedding 0 → feature F_zero"
└─ F_real ≠ F_zero (model distinguishes)

Inference:
├─ Pass 1: motion X → real embedding E → feature F_real
├─ Pass 2: motion X → zero embedding 0 → feature F_zero
├─ CFG signal: g * (F_real - F_zero) = large value (different!)
└─ ✓ Works okay (but suboptimal)

PROBLEM: But what if training text encodes as near-zero (e.g., short text)?
├─ Then real text embedding ≈ zero embedding
├─ F_real ≈ F_zero → CFG signal ≈ 0 → guidance fails!
└─ ✗ CFG breaks for "simple" prompts
```

**Scenario 2: Trainable null (learned)**

```
Training:
├─ mask_text_cond: replaces ~10% with learnable null_vtxt_feat
├─ Model learns: "motion X → real embedding E → feature F_real"
│                "motion X → null embedding N → feature F_null"
│
│ During gradient descent:
│ - Maximize difference: F_real - F_null
│ - null_vtxt_feat learns to be "most different" from typical text
│
└─ null_vtxt_feat → unique representation (not zeros, not typical text)

Inference:
├─ Pass 1: motion X → real embedding E → feature F_real
├─ Pass 2: motion X → null embedding N (same N as training!) → feature F_null
├─ CFG signal: g * (F_real - F_null) = meaningful value
└─ ✓ CFG always works, even for edge-case prompts
```

### 3.2 Mathematical Justification

**Training loss (simplified):**
```
L = ||pred(motion, text) - target||²
  + λ * ||pred(motion, null) - target||²  (scaled down if text_available=False)

∂L / ∂null_vtxt_feat = ???

Gradient depends on:
- How different null and text representations are
- Model's ability to distinguish them
- Loss feedback from both branches

Over many steps:
- null_vtxt_feat evolves to maximize the distinction
- This maximizes the CFG signal (F_real - F_null)
```

---

## Part 4: Text Mask Integration

### 4.1 Optional: Per-token masking

Some configs include `text_mask` (shape `(B, L_c)`):

```python
text_mask = batch.get('text_mask')  # (B, L_c) or None

# Indicates which tokens are padding/valid
# Example: [1, 1, 1, 0, 0] for 3-token prompt + 2 padding

# Passed to model:
pred = model.predict_flow(
    ...,
    text_mask=text_mask,  # Cross-attention respects this
)

# In cross-attention:
# attn_weights = attn(motion, text_ctxt)
# attn_weights = attn_weights.masked_fill(~text_mask, -inf)
# attention = softmax(attn_weights)  # Ignores masked tokens
```

### 4.2 Interaction with mask_text_cond

Important: `mask_text_cond` and `text_mask` are independent.

```
mask_text_cond:
- Controls whether sample receives real or null embeddings
- Probabilistic (CFG dropout)
- Sample-level decision

text_mask:
- Indicates which tokens are valid (padding mask)
- Deterministic (from tokenization)
- Token-level decision

Both applied:
├─ mask_text_cond determines (real or null) whole tensor
├─ text_mask determines which tokens attend
└─ Orthogonal mechanisms
```

---

## Part 5: Common Debug Scenarios

### Scenario A: "Text doesn't improve motion quality"

**Diagnosis checklist:**

1. **Is mask_text_cond active?**
   ```python
   # Check config
   cond_mask_prob = 0.1  # Should be > 0 for training
   ```

2. **Are null embeddings trained?**
   ```python
   # Check gradient flow
   if not bundle.null_vtxt_feat.requires_grad:
       print("BUG: Null embeddings frozen!")
   ```

3. **Verify gradient updates:**
   ```python
   # Log parameter norms during training
   print(f"null_vtxt_feat norm: {bundle.null_vtxt_feat.norm():.4f}")
   # Should change over training iterations
   ```

4. **Check model architecture:**
   ```python
   # Ensure text_vec and text_ctxt are actually used in cross-attention
   # (Not just passed but ignored)
   ```

### Scenario B: "CFG causes motion degradation"

**Diagnosis:**

1. **Guidance scale too high?**
   ```python
   # Try lower guidance_scale
   guidance_scale = 7.5  # Try 3.0, 5.0 first
   ```

2. **Null embeddings poorly trained?**
   ```python
   # Check: was cond_mask_prob > 0 during training?
   # If cond_mask_prob = 0.0, null embeddings never see gradient
   ```

3. **Mismatch in null embedding usage?**
   ```python
   # Ensure inference uses EXACT same null_vtxt_feat/null_ctxt_input
   # as trained (not reinitialized)
   ```

### Scenario C: "All samples masked (100%)"

**Root cause:** NOT in mask_text_cond; check trainer logic:

```python
# Incorrect trainer code:
for batch in dataloader:
    cond_mask_prob = 1.0  # ← BUG: masks everything!
    
# Correct trainer code:
cond_mask_prob = 0.1  # 10% masking
for batch in dataloader:
    # ... no override per batch
```

---

## Part 6: Reference Implementation Flow

### Full Training Iteration (pseudo-code)

```python
# 1. Load batch with pre-extracted text embeddings
batch = dataloader.next()
text_vec_raw = batch['text_vec']      # (32, 1, 768)
text_ctxt_raw = batch['text_ctxt']    # (32, L_c, 4096)
motion = batch['motion']               # (32, T, 66)

# 2. Apply CFG dropout
text_vec, text_ctxt, text_available = bundle.mask_text_cond(
    text_vec_raw,
    text_ctxt_raw,
    cond_mask_prob=0.1,
    return_text_available=True,
)
# Expected: ~3 samples masked, text_available = [T,T,F,T,F,...]

# 3. Normalize
motion = (motion - motion_mean) / motion_std

# 4. Forward
pred = model(motion, text_vec, text_ctxt, timestep, ...)
# pred shape (32, T, 66)

# 5. Loss
loss = mse_loss(pred, target)

# 6. Backward
loss.backward()
# Gradients flow into:
#   - model weights
#   - null_vtxt_feat (for ~3 masked samples)
#   - null_ctxt_input (for ~3 masked samples)

# 7. Optimizer step
optimizer.step()
optimizer.zero_grad()
```

### Full Inference Iteration (pseudo-code)

```python
# 1. Load sample with text embeddings
text_vec = batch['text_vec']           # (1, 1, 768)
text_ctxt = batch['text_ctxt']         # (1, L_c, 4096)
motion_cond = batch['motion_cond']     # (1, T_cond, 66)

# 2. Initialize noise
x_t = torch.randn(1, T_gen, 66)

# 3. Denoising loop
for t in reversed(range(T)):
    # Pass 1: with text
    pred_text = model(
        torch.cat([motion_cond, x_t]),
        text_vec,
        text_ctxt,
        t,
    )
    
    # Pass 2: with null (use force_mask)
    null_v, null_c = bundle.mask_text_cond(
        text_vec, text_ctxt,
        force_mask=True,
    )
    pred_null = model(
        torch.cat([motion_cond, x_t]),
        null_v,
        null_c,
        t,
    )
    
    # CFG
    pred_guided = pred_null + 7.5 * (pred_text - pred_null)
    
    # Denoise step
    x_t = denoise(x_t, pred_guided, t)

# 4. Denormalize and return
motion_gen = x_t * motion_std + motion_mean
return motion_gen
```

---

## Summary Table

| Phase | Component | Shape | Trainable | Value |
|---|---|---|---|---|
| **Training** | text_vec (real) | (B, 1, 768) | No (from encoder) | Real embedding |
| **Training** | text_vec (masked) | (B, 1, 768) | Yes | null_vtxt_feat |
| **Training** | null_vtxt_feat | (1, 1, 768) | **Yes** | Learned |
| **Inference (CFG)** | text_vec (real) | (1, 1, 768) | No | Real embedding |
| **Inference (CFG)** | text_vec (null) | (1, 1, 768) | No (loaded from checkpoint) | null_vtxt_feat |

---

**End of Text Flow Document**
