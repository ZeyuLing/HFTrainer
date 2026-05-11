# Comprehensive Research Analysis: HyMotion M2M Architecture

**Date**: 2026-05-11  
**Status**: Research-only analysis (no code modifications)  
**Scope**: Complete M2M motion-to-motion editing system

## Executive Summary

HyMotion M2M is a flow-matching diffusion model for motion-to-motion editing that combines:
1. **Dual-stream text conditioning** (token-level ctxt + sentence-level vtxt)
2. **VACE motion conditioning** (video creation/editing with inactive/reactive channels)
3. **Hybrid DiT architecture** (MMDiT: double-stream + single-stream blocks)
4. **Mask-aware training** (MAN: x_t[known]=clean, not noisy)
5. **Universal mask prior** (Rank-K Boolean decomposition)

---

## 1. TEXT CONDITIONING SYSTEM

### 1.1 Two-Stage Text Encoding

**File**: `hftrainer/models/motion/hymotion_m2m/network/text_encoder.py` (lines 73-118)

```python
class HYTextModel(nn.Module):
    # Token-level encoder: Qwen3-8B (4096-dim)
    # Sentence-level encoder: CLIP-L (768-dim)
```

**Dimensions**:
- `vtxt_dim = 768`: Sentence-level embeddings from CLIP-L
- `ctxt_dim = 4096`: Token-level embeddings from Qwen3-8B
- `max_length_llm = 512 + crop_start`: Qwen3 token limit
- `max_length_sentence_emb = 77`: Standard CLIP token limit

**Encoding Path** (lines 119-169):

1. **LLM Encoding** `_encode_llm()`:
   - Input: Text list → Qwen3 tokenizer
   - Template: `f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}\n{{}}"`
   - Padding: `max_length_llm` with `padding_side="right"`
   - Crop: Remove template prefix via `crop_start` to get actual text embeddings
   - Output: `ctxt_raw` (B, orig_max_length_llm, 4096), `ctxt_length` (B)

2. **Sentence Embedding** `_encode_sentence_emb()`:
   - Input: Text list → CLIP-L tokenizer
   - Padding: `max_length_sentence_emb=77`
   - Pooling: Either `pooler_output` (if available) or mean-pooling with attention mask
   - Output: `vtxt_raw` (B, 1, 768) unsqueezed

3. **Return** (line 197-200):
   ```python
   return vtxt_raw, ctxt_raw, ctxt_length
   ```

### 1.2 Null Embeddings & Classifier-Free Guidance

**File**: `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 115-116, 222-252)

**Parameter Initialization**:
```python
self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, vtxt_input_dim), requires_grad=False)
self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, ctxt_input_dim), requires_grad=False)
```

**Critical Fix** (Fixed 2026-03-27):
- Problem: Bundle-level nn.Parameter was invisible to optimizer/checkpoints/DDP sync
- Solution: Added `trainable_parameters()` override to include `self.named_parameters(recurse=False)`
- State dict: Saved under key `'__bundle_params__'` for checkpoint tracking
- Initialization: Changed from `torch.randn` to `torch.zeros` for determinism
- Frozen: `requires_grad=False` prevents accidental training

**CFG Masking** (lines 222-252):
```python
def mask_text_cond(vtxt, ctxt, force_mask=False, cond_mask_prob=0.0):
    """Apply classifier-free guidance masking."""
    if force_mask:
        return null_vtxt_feat.expand(*vtxt.shape), null_ctxt_input.expand(*ctxt.shape)
    if self.training and cond_mask_prob > 0.0:
        mask = torch.bernoulli(torch.ones(bs) * cond_mask_prob).view(bs, 1).bool()
        vtxt = torch.where(mask_vtxt, null_vtxt_feat, vtxt)
        ctxt = torch.where(mask_ctxt, null_ctxt_input, ctxt)
    return vtxt, ctxt
```

**Training Convention** (from pipeline lines 196-212):
- Captioned samples: All text padded to fixed `max_text_len=128`
- Unconditioned samples: Single null token with `ctxt_length=1`, NOT repeated nulls
- OOD prevention: Distribution must match training exactly

---

## 2. MOTION CONDITIONING: VACE SYSTEM

### 2.1 VACE Context Preparation

**File**: `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 307-360)

**Concept**: VACE = Video Creation And Editing framework with complementary channels:

```python
def prepare_vace_input(src_motion, ref_pose=None, src_mask=None):
    """Build VACE conditioning context.
    
    Returns tensor of shape (B, L, 3*D) where D is the motion dim (135).
    """
    B, L_src, D = src_motion.shape
    
    # Channel 1: Inactive (known values in the target)
    inactive = src_motion * (1 - src_mask)  # Zero in generation regions
    
    # Channel 2: Reactive (edit values or zeros)
    if self.vace_condition_mode == 'split_reactive':
        reactive = src_motion * src_mask  # Known motion in generation regions
    elif self.vace_condition_mode == 'clean_zero_mask':
        reactive = torch.zeros_like(src_motion)  # Force zeros for completion
    elif self.vace_condition_mode == 'no_inactive':
        # V2 slim: drop inactive channel, only 2*D total
        reactive = src_motion * src_mask
        vace_context = torch.cat([reactive, src_mask], dim=-1)
        return vace_context
    
    # Channel 3: Mask signal (1=generate, 0=known)
    vace_context = torch.cat([inactive, reactive], dim=-1)  # (B, L, 2*D)
    vace_context = torch.cat([vace_context, src_mask], dim=-1)  # (B, L, 3*D)
    
    # Optional: Prepend reference pose
    if ref_pose is not None:
        ref_pose_padded = torch.cat([ref_pose, torch.zeros_like(ref_pose)], dim=1)
        vace_context = torch.cat([ref_pose_padded, vace_context], dim=1)
    
    return vace_context
```

**Critical Invariant**: src_motion MUST be zeroed in mask=1 regions BEFORE VACE preparation
- Trainer zeroes: `src_motion *= (1-mask)` after normalization (trainer.py lines 90-121)
- Prevents reactive channel leak during completion training

### 2.2 Padding & Sequence Handling

**File**: `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 254-305)

```python
def prepare_padding(src_motion, tgt_motion, tgt_length, src_mask=None, src_length=None, ref_pose=None):
    """Pad src/tgt motions to same length and build tgt_padding_mask.
    
    Returns: (src_motion_padded, src_mask_padded, tgt_motion_padded, 
             src_length_list, tgt_length_list, tgt_padding_mask)
    """
    B, L_s, D = src_motion.shape
    L_t = tgt_motion.shape[1] if tgt_motion is not None else L_s
    L_r = ref_pose.shape[1] if ref_pose is not None else 0
    
    max_len = max(L_s, L_t)
    
    # Pad to max_len
    if L_s < max_len:
        pad = max_len - L_s
        src_motion = F.pad(src_motion, (0, 0, 0, pad))  # Pad frames, keep dims
        src_mask = F.pad(src_mask, (0, 0, 0, pad))
    
    if tgt_motion is not None and L_t < max_len:
        pad = max_len - L_t
        tgt_motion = F.pad(tgt_motion, (0, 0, 0, pad))
    elif tgt_motion is None:
        tgt_motion = torch.zeros(B, max_len, D)
    
    # Build tgt_padding_mask from REAL tgt_length (not padded length)
    tgt_mask = _length_to_mask(torch.tensor(tgt_length), max_len)  # (B, max_len)
    
    if L_r > 0:
        ref_mask = torch.ones(B, L_r, dtype=torch.bool)
        tgt_padding_mask = torch.cat([ref_mask, tgt_mask], dim=1)
    else:
        tgt_padding_mask = tgt_mask
    
    return src_motion, src_mask, tgt_motion, src_length, tgt_length, tgt_padding_mask
```

**Convention**:
- `tgt_length`: List[int] of real frame counts (from 'num_frames' in dataset)
- NOT padded length; used to build mask via `_length_to_mask()`
- Padded frames are zeroed and masked in loss computation

### 2.3 Mask Pattern System (M1-M7 + v3)

**File**: `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` (lines 104-117)

**7 Mask Strategies**:

| Strategy | Type | Probability | Description |
|----------|------|-------------|-------------|
| M1 | Random Cell | 20% | Bernoulli per-element masking |
| M2 | Random Block | 12% | Random rectangular blocks |
| M3 | Temporal Contiguous | 23% | Contiguous time window |
| M4 | Joint Contiguous | 15% | Contiguous joint set |
| M5 | Full Mask (Unconditional) | 5% | All frames masked (T2M mode) |
| M6 | Keyframe Sparse | 15% | Regular frame intervals (e.g., every 5 frames) |
| M7 | Scattered Joint + Dilation | 10% | Random joint subsets with temporal dilation |

**v3 Sampler** (lines 1-99):

**Universal Rank-K Boolean Tensor Prior**:
```
M = ⋁_{k=1..K} (t_k ⊗ d_k)
```

- `K ∈ {0..4}` with weights `(0.10, 0.55, 0.25, 0.07, 0.03)`
- **6 Temporal Primitives**: all, empty, interval, periodic, renewal, markov
- **5 Dimensional Kinds**: rot_only, pos_only, trans_only, mixed, all_dim
- **17 Anatomical Joint Groups**: end_effectors, hands_feet, arms, legs, spine_chain, etc.
- Covers all M1-M7 strategies + new unseen patterns

---

## 3. HYBRID DITRANSFORMER ARCHITECTURE (MMDiT)

### 3.1 Double-Stream Block

**File**: `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py` (lines 50-79)

**Architecture**:
```
Motion Stream:
  - LayerNorm -> Modulation(adapter) -> QKV -> Joint Attn -> Proj
  - LayerNorm -> Modulation(adapter) -> MLP

Text Stream:
  - LayerNorm -> Modulation(adapter) -> QKV -> Joint Attn -> Proj
  - LayerNorm -> Modulation(adapter) -> MLP
```

**Key Components**:

1. **Modulation (DiT-style)**: Each stream gets per-adapter shift/scale/gate parameters
   - File: `network/modulate.py` (lines 1-47)
   - Adapter = timestep embedding ⊕ vtxt embedding
   - Shift/scale modulate LayerNorm output

2. **Joint Attention**: Q/K/V from both streams concatenated
   - Cross-stream interaction while maintaining separate norms/MLPs
   - Allows motion to attend to text and vice versa
   - Uses narrowband attention masking for efficiency

3. **Rotary Position Embedding (RoPE)**:
   - Applied optionally to single stream (motion only typical)
   - Provides relative position awareness without absolute embeddings

### 3.2 Single-Stream Block

**Simplified design**: Motion and text tokens concatenated, processed together
- Sequence: `[motion_tokens] [text_tokens]`
- Shared attention mechanism
- Used in later layers for tighter integration

### 3.3 Forward Pass

**File**: `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 362-394)

```python
def predict_flow(x_input, ctxt_input, vtxt_input, timesteps, 
                 x_mask_temporal=None, ctxt_mask_temporal=None, mask_density=None):
    """Single forward pass through MMDiT.
    
    Args:
        x_input: (B, L, D + 3*D_motion) = [x_t, vace_context]
        ctxt_input: (B, Lc, 4096) token-level text embeddings
        vtxt_input: (B, 1, 768) sentence-level embeddings
        timesteps: (B,) diffusion timesteps
        x_mask_temporal: (B, L) boolean mask for motion (1=valid, 0=pad)
        ctxt_mask_temporal: (B, Lc) boolean mask for text
        mask_density: (B,) optional CDE conditioning [0, 1]
    
    Returns:
        pred: (B, L, D_motion) model prediction
    """
    return self.motion_transformer(
        x=x_input,
        ctxt_input=ctxt_input,
        vtxt_input=vtxt_input,
        timesteps=timesteps,
        x_mask_temporal=x_mask_temporal,
        ctxt_mask_temporal=ctxt_mask_temporal,
        mask_density=mask_density,
    )
```

### 3.4 CRFM v3 Enhancements

**File**: `hftrainer/models/motion/hymotion_m2m/network/condition_routing.py` (lines 1-100)

**Condition Density Embedding (CDE)**:
- Encodes mask density ∈ [0, 1] as continuous embedding
- Sinusoidal positional encoding (like timesteps)
- 4-layer MLP to projection
- Zero-initialized final layer (gradual introduction)
- Purpose: Help model scale text influence based on generation ratio

**Text Attention Preservation (TAP)**:
- Applies gradient scaling (default 0.01x) to text pathway parameters
- Patterns: text_mod, text_norm, text_qkv, text_proj, text_mlp
- Prevents text attention from atrophying during mixed training
- Only applies to double-stream text parameters (single-stream mixes both)

---

## 4. TRAINING FLOW

### 4.1 Trainer Entry Point

**File**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (lines 34-47)

```python
class HyMotionM2MTrainer:
    def __init__(self, ..., mask_aware_noise: bool = False, ...):
        self.mask_aware_noise = mask_aware_noise  # Enable MAN training
        self.max_text_len = 128  # Fixed token length for text
```

### 4.2 Unified Forward Pass

**File**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (lines 49-283)

**`_prepare_and_forward()` Orchestration**:

```python
def _prepare_and_forward(self, data):
    # 1. Load & normalize motions (lines 68-88)
    src_motion = data['src_motion']  # (B, L_s, 135)
    tgt_motion = data['tgt_motion']  # (B, L_t, 135) or None
    tgt_length = data['tgt_length']  # List[int] real frame counts
    src_mask = data['src_mask']  # (B, L_s, 135), 1=generate, 0=known
    
    src_motion, tgt_motion = self.normalize(src_motion, tgt_motion)
    
    # 2. Zero-out mask regions for completion (lines 90-121)
    # CRITICAL: src_motion *= (1-mask) to prevent reactive leak
    if not edit_flags:  # Completion mode
        src_motion = src_motion * (1 - src_mask)
    else:  # Editing mode
        # Keep LQ motion in mask regions
        pass
    
    # 3. Pad sequences (lines 123-134)
    src_motion, src_mask, tgt_motion, src_length, tgt_length, tgt_padding_mask \
        = self.bundle.prepare_padding(src_motion, tgt_motion, tgt_length, src_mask, ...)
    
    # 4. Text embeddings (lines 136-215)
    if 'text_vec_raw' in data:
        # Pre-encoded text
        vtxt = data['text_vec_raw'].to(device)
        ctxt = data['text_ctxt_raw'].to(device)
        ctxt_length = data['text_ctxt_raw_length'].to(device)
    elif 'caption' in data:
        # Online encoding
        text_dict = self.bundle.encode_text(data['caption'])
        vtxt, ctxt, ctxt_length = ...
    else:
        # Null embeddings (unconditioned)
        vtxt = self.bundle.null_vtxt_feat.expand(B, 1, -1)
        ctxt = self.bundle.null_ctxt_input.expand(B, 1, -1)
        ctxt_length = torch.ones(B, dtype=torch.long)
    
    # 5. CFG masking during training
    if self.training:
        vtxt, ctxt = self.bundle.mask_text_cond(vtxt, ctxt, cond_mask_prob=0.1)
    
    # 6. Flow matching setup (lines 217-238)
    x0 = torch.randn_like(tgt_motion)  # Noise
    x1 = tgt_motion  # Clean
    t = torch.rand(B, device=device)
    x_t = (1 - t) * x0 + t * x1
    
    # Mask-aware noise: x_t[known] = x1[known] if enabled
    if self.mask_aware_noise:
        x_t = x_t * src_mask + x1 * (1 - src_mask)
    
    # 7. VACE context preparation (lines 240-256)
    vace_context = self.bundle.prepare_vace_input(src_motion, src_mask=src_mask)
    x_input = torch.cat([x_t, vace_context], dim=-1)  # (B, L, D + 3*D)
    
    # 8. Forward through MMDiT (lines 258-280)
    pred = self.bundle.predict_flow(
        x_input, ctxt, vtxt, timesteps=t,
        x_mask_temporal=tgt_padding_mask,
        ctxt_mask_temporal=ctxt_mask,
    )
    
    return {
        'x0': x0, 'x1': x1, 'x_t': x_t,
        'pred': pred,
        'tgt_padding_mask': tgt_padding_mask,
        'generation_mask': src_mask,  # For mask-aware loss
    }
```

### 4.3 Loss Computation

**File**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py` (lines 106-150+)

```python
class M2MLoss(nn.Module):
    def __init__(self, loss_type='smooth_l1', pred_type='velocity',
                 velocity_weight=1.0, keypoints3d_weight=1.0, ...):
        # loss_type: smooth_l1, l1, mse
        # pred_type: velocity (predict dx1) or x1 (predict clean motion)
        self.velocity_weight = velocity_weight
        self.keypoints3d_weight = keypoints3d_weight
```

**Loss Computation**:

1. **Velocity Loss** (default):
   - GT: `vel_gt = (x1 - x0) / t`  (approx; exact: compute from x0, x1 via reparameterization)
   - Pred: Model predicts velocity
   - Reduction: Element-wise or component-wise (KIMODO-style):
     - Component 0-3: Translation (3D)
     - Component 3-135: Rotation (22×6D)
     - Component 135-198: Position (21×3D, if 198-dim)
     - Each component gets its own mask-weighted mean before averaging

2. **Mask-Aware Loss** (lines 62-104):
   ```python
   def _masked_motion_loss(per_dim, data_mask_temporal, generation_mask=None):
       """Reduce (B, L, D) losses with double masking."""
       if generation_mask is not None:
           combined = generation_mask * data_mask_temporal.unsqueeze(-1)
           return (per_dim * combined).sum() / combined.sum()
       else:
           per_frame = per_dim.mean(dim=-1)
           return (per_frame * data_mask_temporal).sum() / data_mask_temporal.sum()
   ```

3. **3D Keypoint Loss** (optional):
   - Denormalize x1 to get clean motion
   - Run FK through SMPL body model
   - L1/smooth_l1 loss against GT keypoints
   - Applied only in generation regions via generation_mask

4. **Auxiliary Losses** (optional):
   - FK consistency: Reconstruction error from FK
   - Joint position/velocity (KIMODO-style)
   - Motion smoothness (temporal gradient penalty)

### 4.4 Train Step

**File**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (lines 391-398)

```python
def train_step(self, data):
    context = self._prepare_and_forward(data)
    loss_dict = self._compute_base_loss(context, global_step=self.global_step)
    loss_dict_aux = self._compute_kimodo_aux_loss(context)
    
    total_loss = sum(loss_dict.values()) + sum(loss_dict_aux.values())
    return total_loss, {**loss_dict, **loss_dict_aux}
```

---

## 5. INFERENCE / PIPELINE

### 5.1 Pipeline Setup

**File**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` (lines 82-121)

```python
class HyMotionM2MPipeline:
    def __init__(self, bundle, num_steps=50, text_guidance_scale=7.5,
                 replacement_guidance='skip_last', max_text_len=128, ...):
        self.num_steps = num_steps
        self.text_guidance_scale = text_guidance_scale  # CFG scale
        self.replacement_guidance = replacement_guidance
        self.max_text_len = max_text_len  # MUST match trainer!
```

### 5.2 Inference Preparation

**File**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` (lines 131-219)

```python
def prepare_inference_batch(self, src_motion, src_mask, src_length, tgt_length, text, 
                           ref_pose=None, target_motion=None):
    """Prepare batch for inference with exact training conventions."""
    
    # 1. Normalize & pad
    src_motion, tgt_motion, src_mask, ... = self.prepare_padding(...)
    
    # 2. Text handling (CRITICAL: must match training distribution)
    if text is not None:
        # Captioned: encode and pad to max_text_len=128
        text_dict = self.bundle.encode_text(text)
        vtxt = text_dict['text_vec_raw']
        ctxt = text_dict['text_ctxt_raw']  # (B, 512, 4096) from encoder
        ctxt_length = text_dict['text_ctxt_raw_length']
        
        # PAD TO max_text_len=128 to match training
        ctxt_padded = torch.zeros(B, 128, ctxt.shape[-1], device=ctxt.device)
        for b in range(B):
            actual_len = min(ctxt_length[b], 128)
            ctxt_padded[b, :actual_len] = ctxt[b, :actual_len]
        ctxt = ctxt_padded
    else:
        # Unconditioned: SINGLE null token (not 128 repeated!)
        vtxt = self.bundle.null_vtxt_feat.expand(B, 1, -1)
        ctxt = self.bundle.null_ctxt_input.expand(B, 1, -1)  # (B, 1, 4096)
        ctxt_length = torch.ones(B, dtype=torch.long)
    
    # 3. VACE preparation
    vace_context = self.bundle.prepare_vace_input(src_motion, src_mask=src_mask, ...)
    
    return vtxt, ctxt, ctxt_length, vace_context, src_mask, ...
```

**OOD Prevention Rules**:
- Line 105-107: `max_text_len` must exactly match trainer (128)
- Line 196-212: Unconditioned inference MUST use single null token with length=1
- Never use repeated null embeddings or different token padding

### 5.3 Classifier-Free Guidance

**File**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py` (lines 222-282)

```python
def ode_function(self, t, y):
    """ODE solver callback with CFG."""
    # Prepare cond & uncond batches
    y_cond = y[:B]
    y_uncond = y[B:]
    
    # Forward cond
    x_input_cond = torch.cat([y_cond, vace_context], dim=-1)
    pred_cond = self.bundle.predict_flow(x_input_cond, ctxt_cond, vtxt_cond, t)
    
    # Forward uncond with null embeddings
    x_input_uncond = torch.cat([y_uncond, vace_context_uncond], dim=-1)
    pred_uncond = self.bundle.predict_flow(x_input_uncond, ctxt_null, vtxt_null, t)
    
    # CFG scaling
    pred = pred_uncond + text_guidance_scale * (pred_cond - pred_uncond)
    
    # Velocity scaling for ODE solver
    return (pred - y) / (1 - t + eps)
```

### 5.4 Replacement Guidance

**Modes**:

1. **`none`**: No imputation (standard flow-matching inference)
2. **`skip_last`**: Impute only early steps, let model refine final steps
3. **`all`**: Impute clean motion at every step
4. **`flow_interp`**: Interpolate between current step and clean motion

**Application**: Per-step masking based on `src_mask` to keep known regions clean

---

## 6. MOTION REPRESENTATION

### 6.1 Dimensionality

**SMPL-22 Layout** (135-dim or 198-dim):

| Component | Dimensions | Range | Meaning |
|-----------|-----------|-------|---------|
| Translation | 0:3 | 3D | Global root translation (absolute) |
| Rotation | 3:135 | 22×6D | 6D rotations (row-major: [xx, xy, xz, yx, yy, yz]) |
| Position | 135:198 | 21×3D | Joint positions (only in 198-dim, excludes pelvis) |

**Rotation Convention**:
- **Row-major 6D** in training/inference: first 3 elements = first row of rotation matrix
- **File**: `hftrainer/datasets/motion/motionhub/transforms/fk_utils.py`
- **Reordering needed**: Some utilities use column-major; explicit reordering [0,2,4,1,3,5]

### 6.2 Normalization

**File**: `hftrainer/models/motion/hymotion_m2m/bundle.py` (lines 156-170, 457-464)

```python
def _load_mean_std(self, mean_std_dir):
    mean = torch.from_numpy(np.load(f'{mean_std_dir}/Mean.npy')).float()
    std = torch.from_numpy(np.load(f'{mean_std_dir}/Std.npy')).float()
    std = torch.where(std < 1e-3, torch.ones_like(std), std)  # Clamp to avoid div-by-zero
    self.register_buffer('mean', mean)
    self.register_buffer('std', std)

def normalize_motion(self, motion):
    return (motion - self.mean) / self.std

def denormalize_motion(self, motion):
    std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
    return motion * std + self.mean
```

### 6.3 Rotation Spaces

**V5 Ablation** (from CLAUDE.md):
- **`local`** (default): SMPL-convention local rotations (relative to parent joint)
- **`global`**: World-frame global rotations
- Conversion: `global_to_local_rot6d_torch()` before FK to ensure SMPL-compatible output

---

## 7. KNOWN BUGS & FIXES

### 7.1 Bundle-Level Parameters Bug (Critical, Fixed 2026-03-27)

**Problem**:
- `nn.Parameter` and `register_buffer` on ModelBundle were excluded from:
  - Optimizer parameter groups
  - Checkpoint state_dict save/load
  - DDP gradient synchronization
- Symptoms: null embeddings randomly re-initialized on each load, producing invalid rot6d

**Root Cause**:
- `trainable_parameters()` only iterated `_trainable_modules`
- `state_dict_to_save()` only looked at sub-module parameters
- Bundle-level attributes (not sub-modules) were orphaned

**Solution Implemented**:
1. Modified `trainable_parameters()` to include `self.named_parameters(recurse=False)`
2. State dict save/load: Added `'__bundle_params__'` key for bundle-level parameters
3. Added `_sync_orphan_param_grads()` hook for all-reduce in DDP
4. Changed initialization: `torch.randn` → `torch.zeros` for determinism
5. Frozen parameters: Set `requires_grad=False` on null embeddings

**Checkpoint Compatibility**:
- Old checkpoints without `'__bundle_params__'` load with fallback to pretrained T2M null embeddings

### 7.2 VACE Reactive Channel Leak (Fixed 2026-03-25)

**Problem**:
- `src_motion` not zeroed in mask=1 (generation) regions before VACE
- Reactive channel = src_motion * mask contained answer values
- Training loss artificially low (~0.0003)
- Model learned to copy from reactive channel instead of generating

**Solution**:
- Trainer now applies `src_motion *= (1-mask)` after normalization in completion mode
- Ensures inactive and reactive channels have no information in generation regions

### 7.3 Operator Precedence Bug (Fixed 2026-03-23)

**Problem**:
```python
# Wrong (Python: * binds tighter than -)
inactive = src_motion * 1 - src_mask  # Interpreted as (src_motion * 1) - src_mask

# Correct
inactive = src_motion * (1 - src_mask)
```
- All old checkpoints from hymotion_1.0_train affected

### 7.4 Text Length OOD Shift (Fixed 2026-04-20)

**Problem**:
- Early inference used per-sample token length (12-20) instead of fixed max_text_len=128
- Captioned outputs became distorted (severe jitter) because model never saw those distributions

**Solution**:
- Pipeline now pads all `ctxt_input` to fixed 128 tokens matching trainer

### 7.5 Unconditioned CFG Null Distribution Mismatch (Fixed 2026-04-21)

**Problem**:
- Uncond inference used repeated null embeddings: `(B, 128, dim)` with all-False mask
- Training used single null token: `(B, 1, dim)` with length=1
- Catastrophic jitter in uncond outputs

**Solution**:
- Uncond inference now uses `ctxt_length=1`, `ctxt_input shape (B, 1, 4096)`
- Matches training convention exactly for distribution consistency

---

## 8. SUPPLEMENTARY COMPONENTS

### 8.1 Motion Condition Encoder

**File**: `hftrainer/models/motion/hymotion_m2m/network/motion_cond_encoder.py`

Processes VACE context (x_t ⊕ inactive ⊕ reactive ⊕ mask) into hidden dimension

### 8.2 Role Embedding

**File**: `hftrainer/models/motion/hymotion_m2m/network/role_embedding.py`

Differentiates motion and text token roles via learned role embeddings

### 8.3 Token Refiner

**File**: `hftrainer/models/motion/hymotion_m2m/network/token_refiner.py`

Optional post-processing for single tokens (e.g., sentence embedding refinement)

### 8.4 FK/IK Geometry

**File**: `hftrainer/models/motion/hymotion_m2m/network/geometry.py`

Functions for 6D rotation ↔ matrix conversion, used in `decode_motion_from_latent()`

### 8.5 SMPL Body Model

**File**: `hftrainer/models/motion/hymotion_m2m/network/smpl_lite.py`

Lazy-loaded SmplxLiteJ24 for FK during training (keypoint loss) and inference

### 8.6 KIMODO Auxiliary Losses

**File**: `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py`

Post-hoc losses for:
- Joint position consistency
- Joint velocity consistency  
- FK reconstruction accuracy

---

## 9. CONFIGURATION & HYPERPARAMETERS

### 9.1 Key Training Hyperparameters

**File**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py` (lines 34-47)

```python
class HyMotionM2MTrainer:
    max_text_len = 128  # CRITICAL: must match inference
    uncondition_mode = True  # Enable CFG
    cond_mask_prob = 0.1  # CFG dropout probability
    mask_aware_noise = False  # Enable MAN training (optional)
    pred_type = 'velocity'  # or 'x1'
    vace_condition_mode = 'split_reactive'  # or 'clean_zero_mask', 'no_inactive'
```

### 9.2 Bundle Configuration

From typical configs:
```yaml
motion_transformer:
  enable_cde: false  # Enable CDE (CRFM v3)
  feat_dim: 1024
  num_heads: 8
  num_blocks: 24

text_encoder:
  llm_type: "qwen3_embedding"
  sentence_emb_type: "clipl"
  max_length_llm: 512
  max_length_sentence_emb: 77

mean_std_dir: "data/hymotion_m2m_data/normalization/"
body_model_path: "checkpoints/smpl/"

losses_cfg:
  loss_type: "smooth_l1"
  pred_type: "velocity"
  velocity_weight: 1.0
  keypoints3d_weight: 0.1
```

---

## 10. SUMMARY TABLE

| Component | Location | Dimension/Type | Purpose |
|-----------|----------|----------------|---------|
| Text Encoder (vtxt) | text_encoder.py | (B, 1, 768) | Sentence-level CLIP-L embeddings |
| Text Encoder (ctxt) | text_encoder.py | (B, ≤512, 4096) | Token-level Qwen3 embeddings |
| Motion (denorm) | bundle.py | (B, L, 135) | SMPL-22: [transl+rot6d] |
| Motion (198-dim) | bundle.py | (B, L, 198) | Extended: [transl+rot6d+pos] |
| VACE Context | bundle.py | (B, L, 3×135) | [inactive+reactive+mask] |
| Model Input | MMDiT | (B, L, 4×135) | [x_t + vace_context] |
| Null Embeddings | bundle.py | Parameters | torch.zeros, frozen |
| Mask Patterns | condition_sampler_v3.py | (B, L, 135) | Rank-K Boolean decomposition |
| Loss Reduction | m2m_loss.py | scalar | Element-wise or component-wise |

---

## 11. DATA FLOW DIAGRAM

```
Training Data
    ↓
[src_motion, tgt_motion, src_mask, caption, tgt_length]
    ↓
1. Normalize
    ↓ src_motion *= (1-mask) [CRITICAL: prevent reactive leak]
    ↓
2. Pad to max_len
    ↓
3. Encode Text (vtxt, ctxt, ctxt_length)
    ↓
4. CFG Masking (bernoulli dropout on text)
    ↓
5. Flow Matching Setup (x0 ~ N, x1 = clean, x_t = (1-t)*x0 + t*x1)
    ↓ [Optional MAN: x_t[known] = x1[known]]
    ↓
6. VACE Preparation ([inactive, reactive, mask])
    ↓
7. Concatenate: x_input = [x_t, vace_context]
    ↓
8. MMDiT Forward (with adapter = timestep ⊕ vtxt)
    ↓
9. Loss Computation (velocity loss + aux losses, double-masked)
    ↓
10. Backprop & Update
```

---

## 12. CRITICAL INVARIANTS TO MAINTAIN

1. **Bundle-level parameters must be in optimizer**: Check `trainable_parameters()` includes `self.named_parameters(recurse=False)`
2. **src_motion MUST be zeroed in mask=1 regions**: Before VACE, after normalization
3. **max_text_len MUST match trainer↔pipeline**: Both fixed at 128
4. **Unconditioned text must be single token**: ctxt shape (B, 1, dim), not (B, 128, dim)
5. **Null embeddings frozen**: requires_grad=False to prevent accidental updates
6. **Padding mask from real length**: tgt_length comes from 'num_frames', not padded length
7. **VACE context 3×D structure**: [inactive||reactive||mask], not variants without signal
8. **Double masking in loss**: Both data_mask_temporal (padding) AND generation_mask (completion regions)
9. **Rotation space consistency**: Global rotations converted to local before FK/output
10. **Timestep embeddings**: Include in adapter computation for modulation

---

**End of Analysis**
