# HyMotion M2M Architecture - Comprehensive Technical Analysis

## Executive Summary

HyMotion-M2M is a flow-matching-based motion-to-motion (M2M) editing framework that uses:
- **Flow Matching**: Continuous interpolation from noise (x0) to clean motion (x1)
- **Multi-Modal Conditioning**: Text (via T5-XXL CLIP embeddings) + Motion (via VACE)
- **Masked Editing**: Per-sample source motion masking for completion & editing tasks
- **DiT Architecture**: Hunyuan Motion Multi-Modal Diffusion Transformer (MMDiT)

---

## 1. TEXT CONDITIONING SYSTEM

### 1.1 Text Encoding Pipeline

**Location**: `hftrainer/models/motion/hymotion_m2m/bundle.py:193-220`

```python
def encode_text(self, text: List[str]) -> Dict[str, Tensor]:
    """Lazy-load text encoder and encode text to vtxt/ctxt.
    
    Returns:
        - text_vec_raw: (B, 1, 768) - sentence-level CLIP embedding
        - text_ctxt_raw: (B, Lc, 4096) - token-level T5-XXL embeddings
        - text_ctxt_raw_length: (B,) - actual token count per sample
    """
```

**Text Encoder**: HYTextModel (Qwen3-8B + T5-XXL fusion)
- **vtxt_input_dim**: 768 (sentence-level embedding from CLIP)
- **ctxt_input_dim**: 4096 (token-level embeddings from T5-XXL)
- **max_text_len**: Fixed at 128 tokens (padded/truncated in trainer)
- **Device**: CPU (never moved to GPU) — each DDP rank has its own copy

**Key Implementation**: `encode_text()` in bundle.py
- Text encoder lives on CPU to avoid GPU memory explosion (8B LLM)
- Outputs are moved to training device after encoding
- Called in trainer at `_prepare_and_forward()` line 201

### 1.2 Null Embedding Parameters (Classifier-Free Guidance)

**Location**: `hftrainer/models/motion/hymotion_m2m/bundle.py:115-116`

```python
self.null_vtxt_feat = nn.Parameter(torch.zeros(1, 1, 768), requires_grad=False)
self.null_ctxt_input = nn.Parameter(torch.zeros(1, 1, 4096), requires_grad=False)
```

**Critical Bug History (2026-03-27)**:
- **Problem**: Null embeddings saved/loaded as all-zeros, losing trained values from T2M pretraining
- **Symptom**: Inference ODE divergence in M2M/T2M/UMO tasks (random null conditioning)
- **Root Cause**: Bundle-level `nn.Parameter` not tracked by `state_dict_to_save()` 
- **Fix**: Added `'__bundle_params__'` key in checkpoint; `_sync_orphan_param_grads()` for DDP

**Current Status**: ✅ Fixed (commit `9a67a3d`)
- Frozen (requires_grad=False) — values come from T2M pretraining checkpoint
- Automatically saved/loaded via bundle checkpoint mechanism
- DDP gradient sync handled per-step

**Where Used**:
- Line 244-245: Replaced with null embeddings when `mask_vtxt=True` (CFG dropout)
- Line 212-214: Default null conditioning when no text/caption provided

### 1.3 Text Masking (Classifier-Free Guidance)

**Location**: `hftrainer/models/motion/hymotion_m2m/bundle.py:222-252`

```python
def mask_text_cond(self, vtxt, ctxt, force_mask=False, cond_mask_prob=0.0):
    """Apply classifier-free guidance masking during training."""
    if force_mask:
        return (null_vtxt_feat, null_ctxt_input)
    
    if self.training and cond_mask_prob > 0.0:
        mask = torch.bernoulli(ones(bs) * cond_mask_prob)  # (B, 1)
        vtxt = where(mask, null_vtxt_feat, vtxt)  # Replace with null
        ctxt = where(mask, null_ctxt_input, ctxt)
    return vtxt, ctxt
```

**Parameters**:
- `cond_mask_prob`: Default 1.0 (100% of batches use CFG during training)
- **Training vs Inference**: Masking only applied during training

**Used In**:
- Line 180-184: When using pre-extracted text embeddings
- Line 206-210: When online encoding from raw captions
- **Pipeline/Inference**: No masking; always use real text embedding for generation

---

## 2. MOTION CONDITIONING SYSTEM (VACE)

### 2.1 VACE (Vector-Aligned Context Encoding)

**Location**: `hftrainer/models/motion/hymotion_m2m/bundle.py:307-360`

Three conditioning modes implemented:

```python
def prepare_vace_input(self, src_motion, ref_pose=None, src_mask=None) -> (B, L, 3*D):
    """Build VACE conditioning context.
    
    Returns (B, L, motion_dim + vace_context_dim) where vace_context_dim 
    depends on mode.
    """
```

#### Mode 1: `split_reactive` (Default v2)
```python
if vace_condition_mode == 'split_reactive':
    inactive = src_motion * (1 - src_mask)      # Known regions (mask=0)
    reactive = src_motion * src_mask              # Mask regions (mask=1)
    vace_context = [inactive, reactive, src_mask]  # (B, L, 3*D)
```
- **inactive**: Motion values in known regions (to maintain context)
- **reactive**: Motion values in masked regions (LQ for editing, 0 for completion)
- **src_mask**: Binary mask (1=generate, 0=known)

#### Mode 2: `clean_zero_mask`
```python
elif vace_condition_mode == 'clean_zero_mask':
    reactive = torch.zeros_like(src_motion)  # Always zero
    vace_context = [inactive, zeros, src_mask]
```
- Never used in practice; experimental variant

#### Mode 3: `no_inactive` (v2 Slim - Current)
```python
elif vace_condition_mode == 'no_inactive':
    # Under mask-aware noise (MAN), x_t[known] = clean_motion already
    # carries known values, so inactive is redundant.
    reactive = src_motion * src_mask
    vace_context = [reactive, src_mask]  # (B, L, 2*D) only!
    # Model input = x_t + reactive + mask = 3*D
```
- **Rationale**: With MAN training, known regions already clean in x_t
- **Efficiency**: Reduces VACE context from 3D to 2D

### 2.2 Input Concatenation & Padding

**Location**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:248`

```python
# Trainer forward
vace_context = bundle.prepare_vace_input(src_motion, ref_pose, src_mask)
x_input = torch.cat([x_t, vace_context], dim=-1)
# x_input shape: (B, L, motion_dim + 3*motion_dim) = (B, L, 4*motion_dim)
# With 135-dim motion: (B, L, 540)
```

**Sequence Padding**:
```python
# prepare_padding() in bundle.py:254-305
# Pads src/tgt to same length L_max
# Builds tgt_padding_mask: (B, L_ref + L_tgt)
# - L_ref: optional reference pose frames (0 if no ref)
# - L_tgt: actual target motion frames
```

### 2.3 Reference Poses (Optional)

When provided (M2M transition tasks E14/E15/E16):
- Prepended to motion sequence
- Marked as always-valid in padding mask
- VACE context for ref frames: all-zero reactive + zero src_mask

**Line 336-342**: Ref pose handling in `no_inactive` mode

---

## 3. DIFFUSION / FLOW MATCHING

### 3.1 Flow Matching Forward Process

**Location**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:217-238`

```python
# Target motion (clean)
x1 = tgt_motion
if ref_pose is not None:
    x1 = torch.cat([ref_pose, x1], dim=1)

# Noise (random initialization)
x0 = torch.randn_like(x1)

# Sample timesteps
if bundle.pred_type == 'x1':
    z = torch.randn(B) * 0.8 - 0.8  # Biased toward high t
    timesteps = torch.sigmoid(z)     # (B,) in [0, 1]
else:  # 'velocity'
    timesteps = torch.rand(B)        # (B,) uniform [0, 1]

# Linear interpolation
t = timesteps.unsqueeze(-1, -1)     # (B, 1, 1)
x_t = (1 - t) * x0 + t * x1          # Noisy at t=0, clean at t=1
```

**Supported pred_types**:
1. **`velocity`** (Default): Predict velocity v = x1 - x0
2. **`x1`**: Predict target x1 directly

**Loss Computation** (velocity mode):
```python
gt_velocity = x1 - x0
pred_velocity = model(x_t)
loss = smooth_l1_loss(pred_velocity, gt_velocity, mask=data_mask)
```

### 3.2 Mask-Aware Noise (MAN) Training

**Location**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:236-238`

```python
if self.mask_aware_noise and src_mask is not None:
    keep_mask = 1 - src_mask         # (B, L, D), 1=known
    x_t = x_t * src_mask + x1 * keep_mask
    # After: x_t[known] = x1 (clean), x_t[gen] = noisy
```

**Rationale**: 
- During training: Known regions should be clean (no noise)
- Matches inference where replacement guidance uses clean motion
- Creates train-consistent distribution

**Enabled by**:
- Trainer init: `mask_aware_noise=True`
- Requires src_mask in batch

---

## 4. DENOISER ARCHITECTURE (HunyuanMotionMMDiT)

### 4.1 Class Overview

**Location**: `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py:571-690`

```python
class HunyuanMotionMMDiT(nn.Module):
    """Multi-Modal Diffusion Transformer for motion generation.
    
    Combines parallel double-stream blocks (motion + text) with 
    single-stream concatenation blocks for efficiency.
    """
```

**Key Hyperparameters**:
- `input_dim`: Motion feature dimension (default 135)
- `feat_dim`: Hidden dimension throughout (default 512 or 768)
- `ctxt_input_dim`: T5-XXL token embeddings (4096)
- `vtxt_input_dim`: CLIP sentence embedding (768)
- `num_layers`: Total transformer blocks (typically 12 or 16)
  - First 1/3: Double-stream blocks (parallel motion + text)
  - Remaining 2/3: Single-stream blocks (concatenated motion + text)

### 4.2 Forward Pass (High-level)

**Location**: `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py:786-890`

```python
def forward(self, x, ctxt_input, vtxt_input, timesteps, x_mask_temporal, ctxt_mask_temporal):
    """
    Full forward pass:
    1. Encode motion input x → motion_feat (B, L, feat_dim)
    2. Encode timestep + vtxt → adapter (B, 1, feat_dim)
    3. Refine text (optional) via token_refiner
    4. Build attention masks (full/causal/narrowband)
    5. Double-stream blocks: parallel processing
    6. Single-stream blocks: joint processing
    7. Optional long skip connection
    8. Final layer projection → output (B, L, output_dim)
    """
```

**Input Shapes**:
- `x`: (B, L, 135+405) = concatenated [x_t, vace_context]
- `ctxt_input`: (B, 128, 4096) - padded to max_text_len
- `vtxt_input`: (B, 1, 768)
- `timesteps`: (B,) - scalar timestep per sample
- `x_mask_temporal`: (B, L) - boolean, True=valid
- `ctxt_mask_temporal`: (B, 128) - boolean, True=valid token

### 4.3 Adapter Signal (Timestep + Text)

**Lines 859-864**:
```python
timestep_feat = self.timestep_encoder(timesteps)  # (B, feat_dim)
vtxt_feat = self.vtxt_encoder(vtxt_input.float())  # (B, 1, feat_dim)
adapter = timestep_feat.unsqueeze(1) + vtxt_feat   # (B, 1, feat_dim)
# Broadcast to all sequence positions via ModulateDiT
```

**CRFM v3 Addition** (lines 866-872):
```python
if self.enable_cde and 'mask_density' in kwargs:
    mask_density = kwargs['mask_density']  # (B,)
    cde_out = self.cde(mask_density)       # (B, feat_dim)
    adapter = adapter + cde_out.unsqueeze(1)
```
- **CDE**: Condition Density Embedding (CRFM v3)
- Encodes how much motion is masked (density of mask=1)

### 4.4 Double-Stream Blocks

```python
class MMDoubleStreamBlock(nn.Module):
    """
    Parallel processing: motion and text in separate streams.
    
    Flow:
    1. Motion self-attention + cross-attention to text
    2. Text self-attention + cross-attention to motion
    3. MLP for each modality
    4. No weight sharing between streams (modality-specific)
    """
```

**Used for**: Early layers where modality-specific features are rich

### 4.5 Single-Stream Blocks

```python
class MMSingleStreamBlock(nn.Module):
    """
    Joint processing: motion + text concatenated into single sequence.
    
    Flow:
    1. Concatenate [motion, text] → (B, L_motion + L_text, feat_dim)
    2. Self-attention on concatenated sequence
    3. Shared MLP weights
    4. More efficient; enables global motion-text interaction
    """
```

**Used for**: Later layers where fusion is beneficial

### 4.6 Attention Masks

**Lines 874-890**: Three modes supported

1. **`mask_mode=None`** (Full attention):
   - All tokens can attend to all other tokens
   - Motion can attend to text, text can attend to motion

2. **`mask_mode='causal'`** (Autoregressive):
   - Token i can only attend to positions ≤ i
   - Useful for sequential generation

3. **`mask_mode='narrowband'`** (Local attention):
   - Window size: `narrowband_length * 30` frames (assuming 30fps)
   - Each token attends to ~window-sized neighborhood
   - Reduces O(L²) to O(L·window)

---

## 5. LOSS FUNCTION

### 5.1 M2MLoss

**Location**: `hftrainer/models/motion/hymotion_m2m/network/m2m_loss.py:8-52`

```python
class M2MLoss(nn.Module):
    def __init__(self,
        loss_type: str = "smooth_l1",         # L1, MSE, or smooth L1
        velocity_weight: float = 1.0,         # Main loss weight
        x1_weight: float = 1.0,               # Alt: predict x1 directly
        keypoints3d_weight: float = 1.0,      # FK loss for 3D joints
        translation_weight: float = 1.0,      # Separate translation loss
        motion_smoothness_weight: float = 0.0,# Temporal smoothness
        fk_consistency_weight: float = 0.0,   # Joint position consistency
        fk_consistency_warmup_steps: int = 1000,  # Warmup schedule
    )
```

### 5.2 Loss Computation (Velocity Mode)

**Location**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:300-347`

```python
# Compute GT velocity
gt_velocity = x1 - x0

# Predict velocity
pred_velocity = model_output

# FK losses (if body_model available)
if fk_loss_enabled:
    pred_x1_for_smooth = x_t + (1 - t) * pred_velocity
    pred_kp3d, gt_kp3d = compute_fk(pred_x1_for_smooth, x1)
    # Keypoints3D loss: smooth_l1(pred_kp3d, gt_kp3d)

# Main loss
losses = bundle.m2m_loss(
    pred_vel=pred_velocity,
    gt_vel=gt_velocity,
    pred_x1=pred_x1_for_smooth,
    gt_x1=x1,
    pred_keypoints3d=pred_kp3d,
    gt_keypoints3d=gt_kp3d,
    data_mask_temporal=tgt_padding_mask,
    generation_mask=generation_mask,  # src_mask when MAN enabled
)

total_loss = sum(losses.values())
```

### 5.3 Generation Mask (MAN-Aware Loss)

**Lines 259-262**: When mask-aware noise enabled

```python
generation_mask = src_mask if self.mask_aware_noise else None
# generation_mask: (B, L, D), 1=generate, 0=known
# In loss: only masked regions contribute to loss
```

**Purpose**: Focus loss computation on regions model must generate

---

## 6. MOTION REPRESENTATION

### 6.1 Motion Dimensions

```
Total: 135 dimensions (smpl_22 format)
├── [0:3]      Translation (absolute global position)
├── [3:9]      Root rotation (6D repr. of 3×3 rotation matrix)
├── [9:135]    Body rotations (126D = 21 joints × 6D per joint)
```

**rot6d Format**:
- Each 3D rotation ↔ 6D vector (first 2 rows of 3×3 rotation matrix)
- Orthonormal constraint via Gram-Schmidt during FK

### 6.2 Extended Motion (198-dim)

```
Total: 198 dimensions
├── [0:3]      Translation
├── [3:135]    All rotations (132D)
├── [135:198]  Joint positions (63D = 21 joints × 3 coords)
```

**Scheme D** (Current): `XZ_rel + Y_abs`
- X, Z: Relative to root (normalize out height variation)
- Y: Absolute (preserve global height)

### 6.3 Denormalization

**Location**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:421-442`

```python
def _fk(x_norm):
    x = x_norm * std + mean  # Denormalize
    transl = x[..., 0:3]                          # (B, L, 3)
    root_rot6d = x[..., 3:9].reshape(B, L, 1, 6)  # (B, L, 1, 6)
    body6d = x[..., 9:135].reshape(B, L, 21, 6)   # (B, L, 21, 6)
    
    # FK with body model (SMPL)
    kp = body_model(body6d, root_rot6d, transl)   # (B, L, 22, 3)
    return kp
```

**Body Model**: SmplxLiteJ24
- Input: 6D rotations + translation
- Output: 3D joint positions (world-space)

---

## 7. TRAINING FLOW

### 7.1 Complete Training Step (train_step)

**Location**: `hftrainer/trainers/motion/hymotion_m2m_trainer.py:391-398`

```python
def train_step(self, batch):
    ctx = self._prepare_and_forward(batch)  # Steps 1-5
    losses = self._compute_base_loss(ctx)   # Compute loss
    loss = sum(losses.values())
    return {'loss': loss, 'loss_velocity': ..., 'loss_smoothness': ...}
```

### 7.2 Data Preparation (Step 1-5)

**Lines 49-283**: `_prepare_and_forward(batch)`

```python
Step 1: Extract & normalize motion
  - src_motion, tgt_motion → normalize via bundle.normalize_motion()
  - Handle edit_mode flag: keep or zero mask regions
  - Zero-out padding frames based on length lists

Step 2: Prepare text embedding
  Three options (priority order):
  a) Pre-extracted embeddings (text_vec_raw, text_ctxt_raw)
     → batch['text_vec_raw']; text_encoder on CPU
  b) Online encoding from captions
     → batch['caption']; calls bundle.encode_text()
  c) Null embedding
     → uses null_vtxt_feat, null_ctxt_input
  
  Then: mask_text_cond() for CFG dropout

Step 3: Flow matching
  - Sample timesteps: uniform or biased toward high-t
  - Interpolate: x_t = (1-t)*x0 + t*x1
  - Apply MAN if enabled: x_t[known] = x1

Step 4: Build VACE context
  - prepare_vace_input() → [inactive, reactive, mask] or [reactive, mask]
  - Concatenate with x_t: x_input = [x_t, vace_context]

Step 5: Forward through bundle
  - predict_flow(x_input, ctxt_input, vtxt_input, timesteps, masks)
  - Returns: (B, L, output_dim) prediction
```

---

## 8. MOTION REPR GLOBAL VS LOCAL ROTATION SPACE

**Location**: `hftrainer/models/motion/hymotion_m2m/bundle.py:99-101`

```python
self.rotation_space = rotation_space  # 'local' or 'global'
```

### 8.1 Local Rotation Space (Default)

- Rotations relative to parent joint (SMPL convention)
- Standard during training and storage
- FK requires forward pass: parent → child

### 8.2 Global Rotation Space

- Rotations in world frame (absolute)
- Used when `rotation_space='global'`
- Faster inference (no FK chain computation)
- Conversion in `decode_motion_from_latent()`: global → local before NPZ save

---

## 9. PIPELINE INFERENCE

### 9.1 Replacement Guidance Modes

**Location**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:80-120`

Three modes for known-region replacement during ODE:

```python
VALID_REPLACEMENT_MODES = ('none', 'all', 'skip_last', 'flow_interp')

"none":         No per-step replacement. Standard ODE from noise → clean.
"all":          At every ODE step, replace known regions with clean_motion.
"skip_last":    Same as "all" but skip replacement on final ODE step.
"flow_interp":  Flow-based interpolation (experimental).
```

**Recommended**: `"skip_last"` for `_man` variants (mask-aware noise models)
- Matches training distribution where known regions are clean
- Skipping last step prevents over-replacement at final frame

### 9.2 SDEdit Support

**Lines 113-120**: Partial-noise inpainting

```python
sdedit_tau: float = 0.0  # Default: full regeneration
# In flow-matching: t ∈ [0, 1] where t=0 is noise, t=1 is clean
# SDEdit τ: start ODE from t = 1 - τ instead of t = 0
# τ=0: full regen (standard), τ=1: no change (pure imputation)
```

**Use Case**: Motion repair tasks (E9) where defects are subtle

---

## 10. KNOWN ISSUES & BUGS

### 10.1 Null Embedding Bug (2026-03-27) ✅ FIXED

**Status**: Fixed in commit `9a67a3d`

**Symptom**: 
- Inference ODE divergence in M2M/T2M/UMO
- Random null embeddings instead of learned values

**Root Cause**: 
- Bundle-level `nn.Parameter` (`null_vtxt_feat`, `null_ctxt_input`) 
- Not included in `state_dict_to_save()` → saved as all-zeros
- Not synced across DDP ranks
- Not restored by `accelerator.load_state()`

**Solution**:
- `'__bundle_params__'` key in checkpoint
- `_sync_orphan_param_grads()` after backward
- `_BundleOrphanCheckpoint` adapter for `accelerator.register_for_checkpointing()`

### 10.2 Text Length Bug (2026-04-20)

**Context**: `hftrainer/pipelines/motion/hymotion_m2m_pipeline.py:102-107`

**Bug**: Using per-sample token length (12-20) instead of fixed padding (128)
- Context attention mask mismatches training
- Produces distorted outputs for captioned inference

**Fix**: Fixed `max_text_len=128` in pipeline (must match trainer)

### 10.3 Rot6d Rotation Convention

**Risk**: Column-major vs row-major mismatches
- 6D: first 2 rows of rotation matrix (row-major)
- FK requires orthonormal constraint → Gram-Schmidt

**Safeguard**: `rotation_converter.py` has conversion utilities

---

## 11. CRFM v3 EXTENSIONS

### 11.1 Condition Density Embedding (CDE)

**Location**: `hftrainer/models/motion/hymotion_m2m/network/hymotion_mmdit.py:866-872`

```python
if self.enable_cde and 'mask_density' in kwargs:
    mask_density = kwargs['mask_density']  # (B,) in [0, 1]
    cde_out = self.cde(mask_density)       # (B, feat_dim)
    adapter = adapter + cde_out.unsqueeze(1)
```

**Purpose**: Explicit conditioning on how much motion is masked
- Helps model adapt to different editing scenarios
- Optional; requires `enable_cde=True` in config

### 11.2 Text Gradient Scaling

**Location**: `hftrainer/models/motion/hymotion_m2m/bundle.py:84-85, 149-150`

```python
text_grad_scale: float = 1.0  # Config parameter
self._text_grad_scale = text_grad_scale  # Stored in bundle
```

**Purpose**: Anti-forgetting mechanism
- Scale text cross-attention gradients to prevent text-understanding regression
- Used in CRFM training to preserve text conditioning

---

## 12. CONFIGURATION MAPPING

### 12.1 Model Config Example

```yaml
model:
  type: 'HyMotionM2MBundle'
  motion_transformer:
    type: 'HunyuanMotionMMDiT'
    input_dim: 135
    feat_dim: 512
    ctxt_input_dim: 4096
    vtxt_input_dim: 768
    num_layers: 12
    num_heads: 16
  text_encoder:
    type: 'HYTextModel'
    # Qwen3-8B + T5-XXL config
  motion_type: 'smpl_22'
  pred_type: 'velocity'
  uncondition_mode: true
  cond_mask_prob: 1.0
  vace_condition_mode: 'no_inactive'
  enable_cde: false  # Set true for CRFM v3
```

### 12.2 Trainer Config Example

```yaml
trainer:
  type: 'HyMotionM2MTrainer'
  val_num_steps: 10
  max_text_len: 128
  mask_aware_noise: true  # Enable MAN
```

### 12.3 Pipeline Config Example

```yaml
pipeline:
  type: 'HyMotionM2MPipeline'
  num_steps: 50
  text_guidance_scale: 7.5
  replacement_guidance: 'skip_last'  # For _man models
  sdedit_tau: 0.0
```

---

## 13. FILE LOCATIONS (Key Code References)

```
hftrainer/models/motion/hymotion_m2m/
├── bundle.py                      (528 lines) - Main bundle class
│   ├── encode_text()              [Line 193]
│   ├── mask_text_cond()           [Line 222]
│   ├── prepare_vace_input()       [Line 307]
│   ├── prepare_padding()          [Line 254]
│   └── decode_motion_from_latent()[Line 396]
│
├── network/
│   ├── hymotion_mmdit.py          (1000+ lines) - Main transformer
│   │   ├── HunyuanMotionMMDiT     [Line 571]
│   │   └── forward()              [Line 786]
│   │
│   ├── m2m_loss.py                (210 lines) - Loss computation
│   │   ├── M2MLoss                [Line 8]
│   │   └── forward()              [Line 106]
│   │
│   ├── text_encoder.py            (280 lines) - Text model
│   ├── motion_cond_encoder.py     (250 lines) - VACE encoder
│   ├── role_embedding.py          (290 lines) - Stream embedding
│   ├── timestep_gate.py           (150 lines) - Adapter fusion
│   └── kimodo_aux_loss.py         (290 lines) - FK losses
│
hftrainer/trainers/motion/
├── hymotion_m2m_trainer.py        (592 lines) - Training loop
│   ├── class HyMotionM2MTrainer    [Line 22]
│   ├── _prepare_and_forward()     [Line 49]
│   ├── _compute_base_loss()       [Line 285]
│   ├── train_step()               [Line 391]
│   └── _compute_fk_keypoints()    [Line 400]
│
├── hymotion_m2m_v3_trainer.py     (392 lines) - Dual-stream variant
├── hymotion_m2m_soar_trainer.py   (590 lines) - SOAR post-training
└── hymotion_m2m_crfm_trainer.py   (340 lines) - CRFM condition routing
│
hftrainer/pipelines/motion/
├── hymotion_m2m_pipeline.py       (493 lines) - Inference pipeline
│   └── class HyMotionM2MPipeline   [Line 48]
```

---

## 14. QUICK REFERENCE: CONDITIONING FLOW

```
TEXT CONDITIONING
────────────────
Caption → Text Encoder (Qwen3-8B + T5-XXL on CPU)
        → vtxt_raw (B, 1, 768), ctxt_raw (B, L_text, 4096)
        → Pad/truncate ctxt to max_text_len=128
        → CFG masking: maybe replace with null embeddings
        → Final: vtxt_input, ctxt_input

MOTION CONDITIONING (VACE)
──────────────────────────
src_motion, src_mask → prepare_vace_input()
                    → [reactive, mask] or [inactive, reactive, mask]
                    → (B, L, 2*D) or (B, L, 3*D)

DENOISER INPUT
──────────────
x_t (noisy motion) + VACE context = x_input (B, L, 4*D for split_reactive)
ctxt_input (text tokens), vtxt_input (sentence embedding)
timesteps (diffusion timestep)
→ Forward through HunyuanMotionMMDiT
→ Output: (B, L, D) velocity or x1 prediction
```

---

## 15. SUMMARY TABLE: MODEL VARIANTS

| Variant | Pred Type | MAN | VACE Mode | Training |
|---------|-----------|-----|-----------|----------|
| M2M v2 base | velocity | No | split_reactive | Standard |
| M2M v2 _man | velocity | Yes | split_reactive | MAN training |
| M2M v2 _slim | velocity | Yes | no_inactive | MAN + slim VACE |
| M2M v3 dual | velocity | Yes | dual-stream | Dual-stream fusion (sep encoders) |
| CRFM | velocity | Yes | with CDE | + Condition routing + TMCR |
| SOAR | velocity | Yes | any | Post-training: flow correction |

---

## KEY TAKEAWAYS

1. **Text**: T5-XXL tokens + CLIP sentence embedding, lazy-loaded on CPU
2. **Motion Conditioning**: VACE encodes (reactive, mask) or (inactive, reactive, mask)
3. **Denoiser**: HunyuanMotionMMDiT with double-stream + single-stream blocks
4. **Training**: Flow matching with velocity prediction, optional MAN for known-region consistency
5. **Inference**: ODE-based with optional replacement guidance for editing tasks
6. **Null Embeddings**: Critical for CFG; now properly saved/synced/loaded (fixed 2026-03-27)
7. **Extensibility**: CRFM (CDE), SOAR, dual-stream variants all supported via config

