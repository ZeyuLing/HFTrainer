# HyMotion M2M: Complete CFG Data Flow with Diagrams

## Part 1: Normal Forward Pass (Training & Unconditioned Inference)

```
TEXT INPUTS
-----------
caption_raw: "person walks"
                    ↓
         Text Encoder (PERMO/LLaMA)
                    ↓
         ┌─────────────────────┐
         │ TWO OUTPUTS         │
         └─────────────────────┘
         ↓                      ↓
    vtxt_input             ctxt_input
    (B, 1, 768)        (B, S, 4096)
    sentence-level      token-level
                    
MOTION PATHWAY
--------------
motion_t + noise ──→ Model(motion, ctxt, vtxt, t)
                    ↓
                    ├─ vtxt → adapter (modulation)
                    ├─ ctxt → cross-attention keys/values
                    └─ double-stream + single-stream blocks
                    ↓
                x_pred (predicted motion)
```

## Part 2: CFG Forward Pass - WITH BUG (enable_ctxt_null_feat=False, DEFAULT)

```
BATCH CONCATENATION
────────────────────
Unconditional branch:
  motion_uncond = [motion_noise, motion_noise]
  vtxt_uncond = [null_vtxt, null_vtxt]
  ctxt_uncond = [ctxt_input, ctxt_input]  ← REAL VALUES!
  
Conditional branch (concatenated):
  motion_cond = [motion_noise, motion_noise]
  vtxt_cond = [vtxt_input, vtxt_input]    ← REAL VALUES
  ctxt_cond = [ctxt_input, ctxt_input]    ← SAME!

UNIFIED FORWARD PASS
────────────────────
Model([motion_uncond, motion_cond],
      ctxt=[ctxt_uncond, ctxt_cond],  ← Both receive IDENTICAL ctxt!
      vtxt=[vtxt_uncond, vtxt_cond],
      t=t_batch)
      
OUTPUTS
───────
x_pred shape: (2*B, L, D_motion)
split into:
  pred_uncond = x_pred[:B]    ← From null_vtxt + REAL ctxt
  pred_cond   = x_pred[B:]    ← From REAL vtxt + REAL ctxt
  
GUIDANCE COMPUTATION (THE PROBLEM)
──────────────────────────────────
difference = (pred_cond - pred_uncond)
           = Model(vtxt=REAL) - Model(vtxt=null)
           = Model(ctxt=REAL) - Model(ctxt=REAL)  ← CANCELS OUT!
           
  Only vtxt differs (768D)
  ctxt is IDENTICAL in both branches
  Guidance signal ≈ only 768D worth of information
  
GUIDANCE APPLICATION
────────────────────
x_final = pred_uncond + scale * (pred_cond - pred_uncond)
        = pred_uncond + 7.5 * (very_small_difference)
        ≈ pred_uncond  (guidance almost has no effect!)
```

## Part 3: CFG Forward Pass - FIXED (enable_ctxt_null_feat=True, RECOMMENDED)

```
BATCH CONCATENATION
────────────────────
Unconditional branch:
  motion_uncond = [motion_noise, motion_noise]
  vtxt_uncond = [null_vtxt, null_vtxt]      ← NULLED
  ctxt_uncond = [null_ctxt, null_ctxt]      ← NULLED ✓
  
Conditional branch (concatenated):
  motion_cond = [motion_noise, motion_noise]
  vtxt_cond = [vtxt_input, vtxt_input]      ← REAL
  ctxt_cond = [ctxt_input, ctxt_input]      ← REAL

UNIFIED FORWARD PASS
────────────────────
Model([motion_uncond, motion_cond],
      ctxt=[null_ctxt, ctxt_input],        ← DIFFERENT!
      vtxt=[null_vtxt, vtxt_input],        ← DIFFERENT!
      t=t_batch)
      
OUTPUTS
───────
x_pred shape: (2*B, L, D_motion)
split into:
  pred_uncond = x_pred[:B]    ← From null_vtxt + null_ctxt
  pred_cond   = x_pred[B:]    ← From REAL vtxt + REAL ctxt
  
GUIDANCE COMPUTATION (FIXED)
────────────────────────────
difference = (pred_cond - pred_uncond)
           = Model(vtxt=REAL, ctxt=REAL) - Model(vtxt=null, ctxt=null)
           
  Both vtxt AND ctxt differ
  Total information: 768D (vtxt) + 40K-80K D (ctxt)
  Guidance signal ≈ 40K-80K D worth of information ✓
  
GUIDANCE APPLICATION
────────────────────
x_final = pred_uncond + scale * (pred_cond - pred_uncond)
        = pred_uncond + 7.5 * (large_semantic_difference)
        ≠ pred_uncond  (guidance now works!) ✓
```

## Part 4: Information Flow Through Transformer Blocks

```
DOUBLE-STREAM BLOCKS (shared adapter modulation)
─────────────────────────────────────────────────

Input:
  motion_stream: (B, L_m, D_m)
  text_stream:   (B, L_t, D_t)  ← from ctxt_input encoding
  adapter:       (B, 1, D_a)    ← from vtxt_input + timestep

Block Processing:
  ┌─────────────┐
  │ Motion Q/K/V│ ─────┐
  └─────────────┘      │
                  Joint Attention
  ┌─────────────┐      │
  │ Text Q/K/V  │ ─────┤
  │ (from ctxt) │      │
  └─────────────┘      │
                       ↓
              Concatenated attention
              (motion queries can attend to text keys/values)
              
  adapter → AdaLN modulation applied to all features
  
Output:
  motion_stream': (B, L_m, D_m)  ← motion influenced by text
  text_stream':   (B, L_t, D_t)


SINGLE-STREAM BLOCKS (after concatenation)
──────────────────────────────────────────

Input:
  unified: torch.cat([motion', text'], dim=1)  ← (B, L_m+L_t, D)
  adapter: (B, 1, D_a)  ← still from vtxt
  
Block Processing:
  Unified Self-Attention
  (all positions can attend to all other positions)
  
  adapter → AdaLN modulation applied to all features
  
Output:
  unified': (B, L_m+L_t, D)
  
Final:
  Extract motion part: unified'[:, :L_m]  ← only motion output used
```

## Part 5: Why CFG Effectiveness Differs by 40K+ Orders of Magnitude

```
INFORMATION MAGNITUDE ANALYSIS
──────────────────────────────

With enable_ctxt_null_feat=False (DEFAULT, BROKEN):
═══════════════════════════════════════════════════
Unconditional input size:
  - motion: normalized, ≈ standard scale
  - vtxt: null_vtxt (learned, typically ≈ 0.1-0.5 norm)
  - ctxt: REAL embeddings (40K-80K float semantic content)
  
Conditional input size:
  - motion: normalized, ≈ standard scale
  - vtxt: REAL embeddings (≈ 1.0 norm typically)
  - ctxt: REAL embeddings (40K-80K float semantic content)
  
Guidance signal difference:
  ΔInput = Δmtion + Δvtxt + Δctxt
         = 0         + (real - null)  + 0
         = ~768D worth of small values (null_vtxt ≈ 0.1-0.5)
  
Expected SNR: 
  SNR ∝ (768 * 0.1²) / noise_var ≈ 10 / noise_var ≈ very weak


With enable_ctxt_null_feat=True (RECOMMENDED, FIXED):
═════════════════════════════════════════════════════
Unconditional input size:
  - motion: normalized
  - vtxt: null_vtxt (≈ 0.1-0.5 norm)
  - ctxt: null_ctxt (≈ 0.1-0.5 norm)
  
Conditional input size:
  - motion: normalized
  - vtxt: REAL embeddings (≈ 1.0 norm)
  - ctxt: REAL embeddings (semantic content, ≈ 1.0-2.0 norm)
  
Guidance signal difference:
  ΔInput = Δmotion + Δvtxt + Δctxt
         = 0        + (real - null)  + (real - null)
         = ~768D + ~40K-80K D of semantic information ✓
  
Expected SNR:
  SNR ∝ ((768 + 40K-80K) * 1.0²) / noise_var ≈ 40K / noise_var ≈ much stronger!


MAGNITUDE COMPARISON
───────────────────
Factor difference ≈ 40K-80K / 768 ≈ 50-100× stronger guidance signal
```

## Part 6: Step-by-Step Inference Execution

```
STEP 1: Load Model & Setup
──────────────────────────
bundle = HyMotionM2MBundle.from_config(cfg)
pipeline = HyMotionM2MPipeline(bundle, text_guidance_scale=7.5)


STEP 2: Process Caption
────────────────────────
caption = "person walks forward"
         ↓
    Text Encoder
         ↓
    vtxt_input (1, 1, 768) ← sentence embedding
    ctxt_input (1, S, 4096) ← token embeddings (S=~20 tokens typically)


STEP 3: Initialize Noise
────────────────────────
x_t = torch.randn(1, 360, 135)  ← (B, T, D_motion)


STEP 4: ODE Integration with CFG
────────────────────────────────
for t in reversed(t_schedule):
    
    # Prepare batch: [uncond, cond]
    x_input_batch = torch.cat([x_t, x_t], dim=0)  # (2, T, D)
    
    # Create CFG masks
    ctxt_batch = torch.cat([null_ctxt, ctxt_input], dim=0)  # (2, S, 4096)
    vtxt_batch = torch.cat([null_vtxt, vtxt_input], dim=0)  # (2, 1, 768)
    
    # Forward pass (with enable_ctxt_null_feat=True):
    x_pred = model.predict_flow(
        x_input=x_input_batch,
        ctxt_input=ctxt_batch,    # ← Different in both branches!
        vtxt_input=vtxt_batch,    # ← Different in both branches!
        timesteps=t.expand(2)
    )
    
    # Split predictions
    pred_uncond, pred_cond = x_pred.chunk(2, dim=0)
    
    # Apply guidance
    pred_guided = pred_uncond + 7.5 * (pred_cond - pred_uncond)
    
    # Update x_t using ODE solver
    x_t = ode_solver_step(pred_guided, t, x_t)


STEP 5: Denormalize Output
──────────────────────────
motion_output = bundle.denormalize_motion(x_t)  # (1, 360, 135)
```

## Part 7: Configuration Comparison

```
TRAINING CONFIG (Before vs After Fix)
═════════════════════════════════════

BEFORE (broken):
────────────────
model = dict(
    type='HyMotionM2MBundle',
    enable_ctxt_null_feat=False,  # DEFAULT
    cond_mask_prob=0.0,           # No CFG training!
    # ... motion_transformer config
)

AFTER (fixed):
──────────────
model = dict(
    type='HyMotionM2MBundle',
    enable_ctxt_null_feat=True,   # ← CRITICAL FIX
    cond_mask_prob=0.1,           # ← Enable CFG training
    # ... motion_transformer config
)


INFERENCE CODE (Before vs After)
════════════════════════════════

BEFORE (weak guidance):
───────────────────────
pipeline = HyMotionM2MPipeline(
    bundle,
    num_steps=50,
    text_guidance_scale=1.0  # No effect!
)

AFTER (strong guidance):
────────────────────────
pipeline = HyMotionM2MPipeline(
    bundle,
    num_steps=50,
    text_guidance_scale=7.5  # Now works!
)
```

## Summary

The entire CFG mechanism can be understood as:

1. **During Training:** Both `vtxt` and `ctxt` randomly masked (if `enable_ctxt_null_feat=True`)
2. **During Inference:** 
   - Without fix: Only `vtxt` differs between branches → weak guidance
   - With fix: Both `vtxt` and `ctxt` differ → strong guidance

The fix is literally one line in the config: `enable_ctxt_null_feat=True`
