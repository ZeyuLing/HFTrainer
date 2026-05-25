# Caption Not Follow: Root Cause Analysis

**Date**: 2026-05-15  
**Models affected**: HyMotion M2M v2 caption models (E2, E4)  
**Severity**: CRITICAL — text conditioning is effectively non-functional  
**Status**: Root cause confirmed; training fix proposed

---

## 1. Problem Statement

Caption-conditioned HyMotion M2M v2 models (E2 at epoch 90, E4) produce near-static motions that completely ignore caption text input. Changing the caption from "a person walks forward" to "a person sits down" produces nearly identical output. CFG (Classifier-Free Guidance) with `guidance_scale=5.0` has negligible effect because the text-conditioned and null-conditioned predictions are virtually identical.

---

## 2. Architecture Background

### Text Conditioning Path (vtxt → adapter → ModulateDiT)

```
CLIP-L sentence embedding (768-dim)
  → MLPEncoder (Linear(768→1024) → SiLU → Linear(1024→1024))
  → vtxt_feat (1024-dim)
  → adapter = timestep_feat + vtxt_feat   ← THIS IS WHERE TEXT ENTERS
  → ModulateDiT in every block: silu(adapter) → Linear → (shift, scale, gate)
  → modulate: x * (1 + scale) + shift
  → gate: x * gate
```

### Text Conditioning Path (ctxt → cross-attention in double_blocks)

```
Qwen3 token embeddings (4096-dim, variable length)
  → Linear(4096→1024) → ctxt_feat
  → SingleTokenRefiner (timestep-conditioned self-attention)
  → Double blocks: joint motion+text attention (Motion→Text allowed, Text→Motion blocked)
```

### CFG Formula

```
v_guided = v_null + guidance_scale × (v_text − v_null)
```

When `v_text ≈ v_null`, the guidance signal is zero regardless of `guidance_scale`.

---

## 3. Root Cause: vtxt_encoder (MLPEncoder) Collapse

### 3.1 The Smoking Gun

The 2-layer MLPEncoder maps **all** 768-dim CLIP-L embeddings to nearly the same point in 1024-dim space. Diagnostic data from `scripts/debug/diag_caption_raw_clip_similarity.py`:

| Metric | Raw (768-dim CLIP-L) | Encoded (1024-dim, after MLPEncoder) | Change |
|--------|---------------------|--------------------------------------|--------|
| cos(caption, null) mean | **0.099** | **0.983** | +0.884 |
| Inter-caption cos mean | **0.343** | **0.992** | +0.650 |
| Angular diversity (mean pair angle) | **71.0°** | **7.1°** | 90% destroyed |

**Interpretation**: In raw CLIP-L space, captions are well-separated (cos ≈ 0.1 to null, cos ≈ 0.34 between captions, ~71° average angle). After passing through the trained MLPEncoder, everything collapses to within 7° of each other, with cos > 0.98 to null. The encoder has learned to **destroy** text information.

### 3.2 Why the MLP Collapses: Bias Dominance

The second layer's bias dominates the output. The "zero-input path" (pure bias propagation) produces `‖output‖ = 40.0`, while actual caption inputs produce `‖output‖ = 33.7–46.5`. The bias contributes 86–119% of the output norm, and `cos(encoded, bias_output)` ≈ 0.97–0.99.

```
vtxt_encoder(zeros) → ‖output‖ = 40.03  (pure bias path)
vtxt_encoder(CLIP)  → ‖output‖ = 37.56  (mean across captions)
vtxt_encoder(null)  → ‖output‖ = 26.07  (null_vtxt_feat, norm=10.1)
```

**The signal-to-bias ratio is catastrophically low.** The second Linear layer has `‖bias‖ = 16.9` and the SiLU activation compresses the first layer's output range, making the bias path dominant.

### 3.3 Weight Analysis Confirms Rank Collapse

SVD analysis of the MLPEncoder weights:

| Layer | σ_max | σ_min | Condition Number |
|-------|-------|-------|-----------------|
| Linear₁ (768→1024) | 36.70 | 0.046 | **806** |
| Linear₂ (1024→1024) | 43.51 | 0.000143 | **303,706** |

The second layer has a condition number of **303,706**, meaning it has essentially collapsed to a low-rank mapping. Most of the 1024 output dimensions are dominated by a few singular vectors — exactly the bias direction. The MLPEncoder does not have the capacity to preserve the rich 768-dim CLIP-L structure.

### 3.4 Cascading Effect Through Architecture

Since `adapter = timestep_feat + vtxt_feat`, and `vtxt_feat(text) ≈ vtxt_feat(null)`:

1. **Adapter is timestep-dominated**: `‖timestep_feat‖ ≈ 130–170` vs `‖vtxt_feat‖ ≈ 26–38`, so vtxt contributes only 15–23% of adapter norm.
2. **cos(adapter_text, adapter_null) ≈ 0.9993** for all timesteps — the model literally cannot distinguish text from null through the adapter.
3. **All ModulateDiT outputs** (shift/scale/gate for every block) are driven by this adapter, so `cos(modulation_text, modulation_null) > 0.997` across all blocks.
4. **CFG guidance signal** is effectively zero: `‖v_text − v_null‖` is negligible compared to `‖v_null‖`.

### 3.5 The ctxt Path (Qwen3 tokens → cross-attention) Also Fails

Even though Motion→Text attention weights are non-zero (0.01–0.85 per block), the attention output is gated by `motion_gate_msa` values that are very small (0.03–0.29), and these gates are driven by the same collapsed adapter. The text features in double_blocks do receive different ctxt tokens for text vs null, but the gating mechanism suppresses this difference before it can affect motion features.

---

## 4. Why Did This Happen?

### 4.1 Training Dynamics

During training with `cond_mask_prob=0.1`, the model sees text 90% of the time and null 10%. The loss gradient is dominated by the unconditional path because:

1. **The vtxt_encoder starts small**: null_vtxt_feat is initialized at `torch.randn * 0.01` (norm ≈ 0.27 initially), so the vtxt pathway contributes almost nothing to the adapter at init.
2. **The timestep encoder dominates from the start**: The sinusoidal embedding + MLP produces large-norm features immediately.
3. **The model has no incentive to differentiate**: The motion reconstruction loss (MSE + keypoints3d_weight=10) is minimized by accurately predicting the flow velocity, which the model can do primarily via the strong unconditioned pathway. Adding text signal through the vtxt pathway provides marginal benefit and may even increase training loss variance.
4. **The MLP collapses rather than learns**: As training progresses, the second layer's large bias and high condition number mean that gradient updates primarily affect the bias direction, not the signal subspace. The encoder converges to a near-constant function.

### 4.2 Comparison with T2M (Where Text Works)

In HyMotion T2M, text conditioning works because:
- The vtxt pathway uses the same MLPEncoder architecture BUT the model was initialized from a pretrained text-motion model where vtxt already had meaningful gradients
- T2M does not have the strong VACE conditioning channel that M2M has, so text is the only conditioning signal — the model MUST use it
- null_vtxt_feat is initialized to `torch.zeros` (not random), creating a cleaner separation

### 4.3 Parent Model Analysis

The parent model (`checkpoint-epoch_3370`) shows partial collapse:
- Text/null adapter ratio is 38% (vs E2's 7%)
- Some text signal passes through, but it's already degraded

This means the collapse **started in the parent training** and **worsened during E2 fine-tuning**, likely because E2 training with different data or learning rate further collapsed the already-weak text pathway.

---

## 5. Diagnostic Scripts and Evidence

| Script | What it measures | Key finding |
|--------|-----------------|-------------|
| `scripts/debug/diag_caption_raw_clip_similarity.py` | Raw vs encoded CLIP-L similarity, MLP bias dominance, adapter contribution | **vtxt_encoder collapses cos from 0.10 to 0.98** |
| `scripts/debug/diag_caption_text_branch_trace.py` | Per-layer activation tracing through full MMDiT | cos(text, null) > 0.997 at all modulation points |
| `scripts/debug/diag_caption_attention_gates.py` | Attention weights and gate values in double_blocks | M→T attention flows but gets gated down (0.03–0.29) |
| `scripts/debug/diag_caption_ode_velocity.py` | Per-step ODE velocity with CFG | cfg_diff norm negligible |

---

## 6. Proposed Training Fixes

### Fix A: Decouple vtxt from timestep in adapter (RECOMMENDED)

**Problem**: `adapter = timestep_feat + vtxt_feat` allows timestep to dominate.

**Solution**: Use separate modulation channels for timestep and text:

```python
# Option A1: Concatenation-based adapter (requires doubling ModulateDiT input dim)
adapter = torch.cat([timestep_feat, vtxt_feat], dim=-1)
# ModulateDiT: Linear(2048 → 6*feat_dim) instead of Linear(1024 → 6*feat_dim)

# Option A2: Multiplicative gating
adapter = timestep_feat * (1 + vtxt_gate(vtxt_feat))  # vtxt_gate: Linear(1024→1024, tanh)

# Option A3: Separate modulation (like PixArt-α)
# timestep controls shift/scale/gate for motion stream
# vtxt controls a separate cross-attention or additional modulation
```

**Pros**: Architecturally prevents timestep from swamping text signal.  
**Cons**: Requires architecture change + retraining from scratch. Option A1 doubles ModulateDiT parameter count.

### Fix B: Scale vtxt_feat to match timestep_feat norm (QUICK FIX)

**Problem**: `‖vtxt_feat‖ ≈ 37` vs `‖timestep_feat‖ ≈ 155` — 4× difference.

**Solution**: Add a learned or fixed scaling factor:

```python
# In bundle or transformer init:
self.vtxt_scale = nn.Parameter(torch.tensor(4.0))  # or fixed: 4.0

# In forward:
adapter = timestep_feat + self.vtxt_scale * vtxt_feat
```

**Pros**: Minimal code change, can be applied to existing checkpoints.  
**Cons**: Doesn't address the MLP collapse root cause; scaling a collapsed signal just amplifies noise.

### Fix C: Replace MLPEncoder with larger/better text encoder (MEDIUM EFFORT)

**Problem**: 2-layer MLP (768→1024→1024) with condition number 300K cannot preserve CLIP-L structure.

**Solution**:

```python
# Option C1: Deeper MLP with residual connections
class ResMLPEncoder(nn.Module):
    def __init__(self, in_dim, feat_dim, num_layers=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, feat_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, feat_dim),
                nn.SiLU(),
                nn.Linear(feat_dim, feat_dim),
            ) for _ in range(num_layers)
        ])
    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = x + block(x)  # residual preserves input information
        return x

# Option C2: Use CLIP-L embedding directly (skip MLP, just project dim)
# vtxt_encoder = nn.Linear(768, 1024)  # simple projection preserves structure
```

**Pros**: Preserves input structure via residual connections or direct projection.  
**Cons**: Requires retraining. Option C2 may lose the "learned null" flexibility.

### Fix D: Increase cond_mask_prob + contrastive text loss (TRAINING REGIME)

**Problem**: With cond_mask_prob=0.1, the model sees null only 10% of the time, giving insufficient contrast signal.

**Solution**:

```python
# In config:
cond_mask_prob = 0.3  # or even 0.5 for early training

# Additionally, add a contrastive loss to force text/null separation:
# L_contrastive = -log(sigmoid(cos(v_text, gt_flow))) - log(sigmoid(-cos(v_null, gt_flow)))
```

**Pros**: Works with existing architecture; forces the model to learn text discrimination.  
**Cons**: Higher cond_mask_prob reduces text conditioning examples, potentially making the collapse worse if the architecture issue isn't also addressed.

### Fix E: Freeze timestep_encoder, train only vtxt_encoder (DIAGNOSTIC)

**Problem**: Both encoders are trained jointly; timestep encoder grows large while vtxt shrinks.

**Solution**: Freeze timestep_encoder and only train vtxt_encoder for the first N epochs to establish a meaningful text signal, then unfreeze.

**Pros**: Forces vtxt_encoder to learn discriminative features.  
**Cons**: Experimental; may not converge well if timestep encoder needs adaptation too.

### Fix F: Initialize null_vtxt_feat to zeros + normalize vtxt_feat (HYGIENE)

**Problem**: null_vtxt_feat starts at `randn * 0.01` (norm ≈ 0.27), but during training it can drift to norm ≈ 10.1 (as observed), approaching the CLIP embedding space and blurring the text/null boundary.

**Solution**:

```python
# Initialize null to zeros (like T2M)
self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim), requires_grad=False)
# Freeze it — null should always be at origin

# Normalize vtxt_feat to unit sphere before adding to timestep
vtxt_feat = F.normalize(vtxt_encoder(vtxt_input), dim=-1) * self.vtxt_scale
```

**Pros**: Ensures clean separation between text and null in the vtxt_feat space.  
**Cons**: Requires retraining.

---

## 7. Recommended Action Plan

### Phase 1: Quick Validation (1-2 days)

1. **Apply Fix B + F together**: Scale vtxt_feat by 4x AND freeze null_vtxt_feat to zeros.
2. **Retrain E2 from parent checkpoint** for 100 epochs with these changes.
3. **Run diagnostic script** to verify cos(enc_text, enc_null) drops below 0.8.

### Phase 2: Architecture Fix (1 week)

1. **Apply Fix A1 (concatenation adapter)** + Fix C2 (simple linear projection) + Fix F.
2. **Retrain from scratch** with `cond_mask_prob=0.2`.
3. This should provide permanent resolution.

### Phase 3: Evaluation

1. Run full eval suite (`scripts/eval/eval_m2m_v2_all_tasks.py --save-npz --use-rewritten`)
2. Verify FID/MM-Dist/R-Precision metrics improve for caption tasks
3. Visual inspection: different captions should produce visibly different motions

---

## 8. Appendix: Full Diagnostic Output

### A. Raw vs Encoded Similarity (12 captions)

```
Input space (768-dim CLIP-L):
  ‖null_vtxt_feat‖          = 10.125
  ‖clip_embedding‖ (mean)   = 28.180
  cos(clip, null) (mean)    = 0.099
  inter-caption cos (mean)  = 0.343

Output space (1024-dim, after vtxt_encoder):
  ‖encoded_null‖            = 26.072
  ‖encoded_clip‖ (mean)     = 37.561
  cos(enc_clip, enc_null)   = 0.983
  inter-caption cos (mean)  = 0.992
```

### B. Adapter Contribution (timestep vs vtxt)

```
t=0.0:   ‖ts‖=129.3  ‖vtxt‖=37.7  vtxt%=22.7%  cos(adapt_text, adapt_null)=0.9991
t=0.5:   ‖ts‖=150.5  ‖vtxt‖=37.7  vtxt%=20.1%  cos(adapt_text, adapt_null)=0.9993
t=0.98:  ‖ts‖=171.4  ‖vtxt‖=37.7  vtxt%=18.2%  cos(adapt_text, adapt_null)=0.9994
```

### C. MLPEncoder Weight SVD

```
Linear₁ (768→1024):  σ_max=36.70  σ_min=0.046  cond=806
Linear₂ (1024→1024): σ_max=43.51  σ_min=0.000143  cond=303,706  ← RANK COLLAPSE
```

### D. Directional Collapse

```
Average raw angle between caption pairs:     71.0°
Average encoded angle between caption pairs:  7.1°
Ratio: 0.10  → 90% of angular diversity destroyed
```

### E. Bias Dominance

```
vtxt_encoder(zeros):  ‖output‖ = 40.03  (pure bias path)
vtxt_encoder(CLIP):   ‖output‖ = 37.56  (signal barely adds to bias)
cos(encoded, bias_output) = 0.97–0.99 for all captions
```
