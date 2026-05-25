# KIMODO Implementation: Exact Technical Details

## Executive Summary

KIMODO (NVIDIA, 2026-03-16) is a two-stage transformer diffusion model for controllable motion generation. Its key distinction from similar systems:
- **Global coordinate frame rotations** (vs. SMPL-style local relative rotations)
- **Direct imputation-based constraints** (vs. soft conditioning like VACE)
- **Two-stage denoiser architecture** (root prediction → body prediction)
- **Smooth root trajectory** (animator-friendly, not noisy pelvis)

---

## 1. INPUT CONSTRUCTION: How Motion + Masks → Model Input

### 1.1 Motion Representation (KimodoMotionRep)

**Per-frame feature layout: 333 dimensions total**

```
Dimension ranges and components:

[0:3]       smooth_root_pos              3D position (x smooth, z smooth, y absolute)
[3:5]       global_root_heading          [cos(ψ), sin(ψ)] heading angle
[5:86]      local_joints_positions       27 joints × 3D (relative to smooth root XZ, absolute Y)
[86:248]    global_rot_data              27 joints × 6D (continuous 6D rotation, world frame)
[248:329]   velocities                   27 joints × 3D global velocities
[329:333]   foot_contacts                4D binary flags [L_heel, L_toe, R_heel, R_toe]

Total: 3 + 2 + 81 + 162 + 81 + 4 = 333 dims
```

**Critical distinction from HyMotion M2M (135-dim):**
- KIMODO: 333 dims including explicit positions + global rotations + velocities + foot contacts
- M2M: 135 dims (3 abs translation + 22×6 rot_6d), no explicit positions or foot contacts
- KIMODO uses NATIVE 27-joint Bones Rigplay skeleton (or retarget to SOMA-30/77)

### 1.2 What Gets Predicted vs. Computed

| Channel | Size | Predicted? | Usage |
|---------|------|---|---|
| smooth_root_pos | 3 | ✅ YES | Model output |
| global_root_heading | 2 | ✅ YES | Model output |
| local_joints_positions | 81 | ✅ YES | Model output |
| global_rot_data | 162 | ✅ YES | Model output |
| velocities | 81 | ❌ NO | Computed post-hoc via finite difference |
| foot_contacts | 4 | ❌ NO | Computed from position+velocity via contact detector |

**Inference truth:** Model predicts exactly **248 dims**. Remaining 85 dims are never predicted, only computed post-inference.

### 1.3 Input Construction via Imputation Mechanism

**When constraints provided** (from constraint sets like `FullBodyConstraintSet`, `Root2DConstraintSet`):

1. **Create `observed_motion` tensor** [B, T, 333]:
   - Fill with GT values at constrained dimensions
   - Fill with zeros elsewhere
   
2. **Create `motion_mask` tensor** [B, T, 333]:
   - Binary mask: 1 where constrained, 0 where free
   - Example: for 2D root trajectory constraint at frame 10:
     - motion_mask[0, 10, [0,2]] = 1  (x,z of smooth_root_pos)
     - motion_mask[0, 10, [1,3,4]] = 1  (y, heading)
     - all others = 0

3. **During diffusion forward pass** (TwostageDenoiser.forward):
   ```python
   if motion_mask_mode == "concat":
       # Direct imputation: overwrite noisy x with GT at constrained locations
       x = x * (1 - motion_mask) + observed_motion * motion_mask
       
       # Concatenate mask as auxiliary channel to inform model
       x_extended = torch.cat([x, motion_mask], axis=-1)  # [B, T, 666]
   ```

4. **Model input becomes 666 dimensions:**
   - 333 from imputed motion (noisy except at constraints)
   - 333 from binary mask (which dims are constrained)

### 1.4 Constraint Types and Dimension Coverage

| Constraint Type | Affected Dims | Example |
|---|---|---|
| `Root2DConstraintSet` | [0, 2] | Root XZ trajectory (2D waypoints) |
| Root Y constraint | [1] | Root height only |
| `global_root_heading` | [3:5] | Heading angle [cos(ψ), sin(ψ)] |
| `global_joints_rots` | [86:248] | Full body or per-joint rotations (6D each) |
| `global_joints_positions` | [5:86] | Full body or per-joint positions (3D each) |

**Unique KIMODO feature:** Position constraints are **GLOBAL world-space positions**, not relative/local like SMPL. No IK needed.

---

## 2. TRAINING PROCEDURE: Two-Phase Curriculum

### 2.1 Phase 1: Pure Text-to-Motion (500k steps)

**Objective:** Train the model on text-conditioned motion generation WITHOUT any constraints.

- No `motion_mask` or `observed_motion` used
- Model learns to generate diverse motions from text prompts
- Loss function components:
  - Smooth L1 on all motion dimensions (position, rotation, velocity)
  - FK consistency loss (rotation → position via forward kinematics)
  - Foot contact detection loss

**Configuration:**
- Batch size: 2048 samples per step (16 A100-80GB GPUs)
- Text encoder: LLM2Vec (bidirectional LLaMA)
- Diffusion: DDPM with 1000 steps, DDIM inference at 100 steps
- Optimizer: Adam-atan2, lr=2e-5
- EMA decay: 0.995 (updated every 10 steps)

### 2.2 Phase 2: Constraint-Aware Training (500k steps)

**Objective:** Train the model to accept and respect constraint imputation.

- Randomly sample constraint patterns at each step
- Create `motion_mask` and `observed_motion` tensors
- Concatenate mask as auxiliary input (666 dims)
- Model learns "when I see mask=1, treat that dimension as GT truth"

**Constraint Sampling During Phase 2:**

Typical mask strategies during training:
- Position constraints at keyframes (motion in-betweening)
- Trajectory constraints (2D waypoint following)
- Full-body keyframe constraints
- End-effector position/rotation constraints
- Foot contact pattern constraints

**Why Two-Phase?**
- Phase 1 ensures strong T2M foundation (not diluted by constraint task)
- Phase 2 learns constraint semantics gradually
- Avoids conflicting gradients from generation vs. constraint objectives

### 2.3 Loss Function Components (Paper Claims)

KIMODO uses **weighted smooth L1 loss** across motion dimensions:

```
Total Loss = Σ γ_i * L1(predicted_i, target_i)

where:
  γ_1 (position) = 10
  γ_2 (velocity) = 2
  γ_3 (rotation) = 10
  γ_4 (foot contact) = 3
  γ_5 (FK consistency) = 4
  γ_6 (rotation consistency) = 5
  γ_7 (additional regularization) = 5
```

**FK Loss Specifics:**
- During training, model predicts rotations
- FK is applied: J_pred = FK(local_rot_mats, root_positions)
- Compare J_pred with J_target (from mocap)
- Loss term: ||J_pred - J_target||_L1 with weight 5

**Velocity Loss:**
- Computed via finite difference: v_t = (x_{t+1} - x_{t-1}) / (2Δt)
- Encourages temporal smoothness
- Weight 2 (lower priority than position/rotation at 10)

**Foot Contact Loss:**
- Ground truth foot contacts detected from mocap (position velocity thresholds: 0.15m height, 0.10m/s velocity)
- Model learns to predict foot contact flags [4]
- Cross-entropy or MSE with weight 3

---

## 3. ARCHITECTURE DETAILS

### 3.1 TwostageDenoiser

**Purpose:** Separate root and body denoising for stability.

```
Stage 1: ROOT PREDICTION
  Input:  x_extended [B, T, 666]  (if concat mode) or [B, T, 333] (if no constraints)
  Output: root_motion_pred [B, T, 5]  (smooth_root_pos[3] + heading[2])
  
  Backbone: TransformerEncoderBlock
    - Input dim: 333 or 666
    - Output dim: 5
    - Architecture: 16 layers, 8 heads, latent_dim=1024

Stage 2: BODY PREDICTION
  Input:  x_new [B, T, local_root_dim + body_dim]  (root_motion_local[4] + body_features[326])
  Output: predicted_body [B, T, 328]
  
  Backbone: TransformerEncoderBlock
    - Input dim: local_root_dim + body_dim + (333 if concat) = 4+326+333=663 (with constraints)
    - Output dim: 328 (all dims except global root)

Output: Concatenate [root_pred, body_pred] → [B, T, 333]
```

**Why two stages?**
- Reduces foot skating error from 7.59 → 3.87 cm/s (paper claim)
- Body motion conditioned on predicted root prevents body from fighting root
- Root operates on global frame (simpler than 27-joint body)

**Key detail:** In training mode, `root_motion_local` is **detached** (no gradient flow) to prevent body gradients from corrupting root prediction. At inference, no detach is used for classifier-free guidance.

### 3.2 TransformerEncoderBlock Architecture

**Per-stage configuration:**

```python
TransformerEncoderBlockConfig:
  input_dim: 333 or 666 (stage 1), 663 (stage 2)
  output_dim: 5 (stage 1 root), 328 (stage 2 body)
  latent_dim: 1024
  ff_size: 4096  (feedforward dimension = 4×latent)
  num_layers: 16
  num_heads: 8
  activation: "gelu"
  dropout: 0.1
  pe_dropout: 0.0
  norm_first: False  (post-norm, not pre-norm)
```

**Components:**

1. **Input Linear:** `input_dim → latent_dim (1024)`
2. **Positional Encoding:** PositionalEncoding (learned, dimension=latent_dim)
3. **Timestep Embedding:** TimestepEmbedder
   - Embeds diffusion step t into latent_dim
   - Used for prefix conditioning
4. **Text Encoding:**
   - Input: LLaMA embeddings [B, text_len, 4096]
   - Linear projection: 4096 → 1024
   - Concatenated as prefix to motion tokens
5. **Transformer Stack:**
   - 16 layers of TransformerEncoderLayer
   - MultiheadAttention: 8 heads, d_model=1024
   - FeedForward: latent_dim → ff_size(4096) → latent_dim
   - Dropout: 0.1 between layers
   - Batch-first: True
6. **Output Linear:** `latent_dim → output_dim`

**Prefix Mode Attention:**
- Text tokens + timestep token + register tokens (49 learnable tokens) + motion tokens
- All attend to all (full attention, not causal)
- Motion tokens see text context without autoregressive constraints

### 3.3 Model Size

**Total parameters: 282M**

```
Root model:     ~141M
  - Transformer: 16 layers × 8 heads × 1024 dim
  
Body model:     ~141M
  - Same architecture as root model
  
Text encoder:   ~7B LLaMA (frozen, external)
```

Not huge by modern standards, but optimized for 100-step DDIM inference.

---

## 4. LOSS FUNCTION (Exact Formulation)

### 4.1 Training Loss

KIMODO uses **separate loss terms per component**:

```
L_total = L_position + L_velocity + L_rotation + L_foot_contact + L_fk + L_rotation_consistency

L_position = smooth_L1(pred_positions, target_positions)  [weight: 10]
L_velocity = smooth_L1(pred_velocities, target_velocities)  [weight: 2]
L_rotation = smooth_L1(pred_rot_6d, target_rot_6d)  [weight: 10]
L_foot_contact = BCE(pred_foot_contacts, target_foot_contacts)  [weight: 3]
L_fk = smooth_L1(FK(pred_rotations) - target_positions)  [weight: 5]
L_rotation_consistency = smooth_L1(pred_rot - FK_rot)  [weight: 5]
```

**Smooth L1 function:**
```
smooth_L1(x, y) = {
    0.5 * (x - y)^2           if |x - y| < 1
    |x - y| - 0.5             otherwise
}
```
(Huber loss variant, robust to outliers)

### 4.2 Diffusion Training Objective

KIMODO trains as **DDPM** (not flow matching like M2M):

```
L_diffusion = E_{x_0, t ~ U(1,1000)} [ || ε - ε_θ(x_t, t, c) ||_2^2 ]

where:
  x_0 ~ data distribution (motion)
  t ~ uniform timestep
  x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε  (DDPM noise schedule)
  c ~ conditioning (text + motion_mask)
  ε_θ ~ model prediction of noise
```

**Noise schedule:** Linear schedule, 1000 training steps, similar to DDPM/Imagen.

### 4.3 Text Dropout for Classifier-Free Guidance

- **During training:** 10% of text prompts dropped (replaced with zero embedding)
- Enables inference-time CFG with **separated guidance** (text + constraint)

---

## 5. MASKS/SCENARIOS SUPPORTED

### 5.1 Constraint Set Classes

KIMODO provides 5 main constraint types, each producing different mask patterns:

#### 1. Root2DConstraintSet
```
Controls: Root (x,z) trajectory + optional heading
Frame-level: Constrain frames [10, 25, 50, ...]
Dims affected: [0, 2, 3:5] (x, z, cos(ψ), sin(ψ))

motion_mask[10:11, [0, 2, 3, 4]] = 1
motion_mask[25:26, [0, 2, 3, 4]] = 1
...
```

#### 2. FullBodyConstraintSet
```
Controls: All 27 joints (positions + rotations) + root trajectory
Frame-level: Constrain specific keyframes [50]
Dims affected: [0:5] (smooth root 2d + heading) + [5:86] (all joint positions)

motion_mask[50, [0:5]] = 1
motion_mask[50, [5:86]] = 1
```
Note: `global_rot_data` NOT applied (not used in practice, only global positions)

#### 3. EndEffectorConstraintSet
```
Controls: Specific joints (hands, feet) - position + rotation
Frame-level: Constrain frames [10, 20, 30]
Dims affected: [0:5] (root) + subset of [5:86] (end-effector joint positions)

Example for left hand at frame 10:
  - Identify left hand joint index J_lh (e.g., 10)
  - motion_mask[10, [0:5]] = 1
  - motion_mask[10, [5+3*J_lh : 5+3*(J_lh+1)]] = 1  (position dims)
```

**Subclasses:**
- LeftHandConstraintSet
- RightHandConstraintSet
- LeftFootConstraintSet
- RightFootConstraintSet

#### 4. Root Y Height Constraint
```
Controls: Root vertical position (Y only)
Dims affected: [1]

motion_mask[frame, 1] = 1
```

#### 5. Global Root Heading Constraint
```
Controls: Heading angle ψ
Dims affected: [3:5]

motion_mask[frame, [3, 4]] = 1
```

### 5.2 Supported Tasks via Mask Patterns

| Task | Constraint Type | Mask Pattern | Example |
|------|---|---|---|
| **Text-to-Motion** | None | All zeros | No constraints |
| **In-Betweening** | FullBody | [0:86] at T_start, T_end | Keyframes at frames 0 and 150 |
| **Waypoint Following** | Root2D | [0, 2] at sparse frames | Root XZ at frames [10, 30, 50, ...] |
| **End-Effector IK** | EndEffector | [0:5] + EE dims | Hand/foot tracking |
| **Multi-Prompt Blending** | FullBody | [0:86] in transition zone | Smooth transition between segments |
| **Foot Contact Pattern** | Special | [329:333] | Foot contact flags (if stored) |
| **Trajectory Following** | Root2D + optional heading | [0, 2] + [3:5] | 2D path + rotation |

### 5.3 Mask Application During Inference

**Per-denoising-step:**
```python
# Step t in DDIM (t=49, 48, ..., 0)

# 1. Imputation: overwrite noisy motion with GT at constrained dims
x_t = x_t * (1 - motion_mask) + observed_motion * motion_mask

# 2. Extend input with mask as auxiliary channel
x_extended = cat([x_t, motion_mask], dim=-1)

# 3. Model denoises
pred_x0 = denoiser(x_extended, text, t)

# 4. Compute x_{t-1} using DDIM sampler
x_{t-1} = DDIM_step(x_t, pred_x0, t)

# 5. Repeat: imputation again before next step
x_{t-1} = x_{t-1} * (1 - motion_mask) + observed_motion * motion_mask
```

**Result:** Constrained dimensions are **exactly locked** throughout sampling. Unconstrained dimensions evolve through diffusion.

---

## 6. SUMMARY: KIMODO vs. HyMotion M2M

### Core Technical Differences

| Aspect | KIMODO | HyMotion M2M |
|--------|--------|---|
| **Coordinate frame** | Global (world-space rotations) | Local (SMPL parent-relative) |
| **Motion dimensions** | 333 (pos + rot + vel + foot) | 135 (transl + rot only) |
| **Rotation representation** | 6D continuous (global) | 6D continuous (local) |
| **Root handling** | Smooth root (ADMM-smoothed XZ) | abs_rel translation (6D) |
| **Constraint application** | Imputation (hard replace) | VACE (soft channel concat) |
| **Constraint space** | Per-dim (T×333) binary mask | Per-dim (T×135) binary mask |
| **Position constraints** | ✅ Global positions supported | ❌ No explicit positions |
| **Joint-level control** | ✅ Per-joint position+rotation | ✅ Per-dim mask control |
| **Diffusion framework** | DDPM (1000 steps, DDIM 100) | Flow Matching (rectified flow, 50 steps) |
| **Model architecture** | Two-stage Transformer (282M) | MMDiT backbone (460M-1.5B) |
| **Training phases** | 2-phase (T2M then constraints) | 1-phase (simultaneous T2M + completion) |
| **Text encoder** | LLM2Vec (4096-dim LLaMA) | Dual encoder (Qwen3-8B + CLIP-L) |
| **Data scale** | 700h optical mocap (high quality) | MotionHub (diverse, mixed quality) |
| **Foot contact** | Explicit modeling (4 dims) | Not modeled |

### Key Advantages of Each

**KIMODO advantages:**
- Global rotation frame → direct world-space constraint application (no IK)
- Smooth root → animator-friendly trajectory (straight lines, curves)
- Explicit foot contact → foot lock post-processing
- Two-phase curriculum → strong T2M foundation

**M2M advantages:**
- Per-dim mask granularity → more flexible constraint patterns
- VACE channel concat → no aggressive imputation (softer conditioning)
- Flow Matching → potentially faster convergence
- Dual text encoders → richer semantic understanding

---

## 7. Code Entry Points

### Training Entry Points (Not Open-Sourced)
- Paper claims Phase 1 (500k steps) + Phase 2 (500k steps) on 700h mocap
- Loss: Weighted smooth L1 + FK consistency
- Optimizer: Adam-atan2, batch size 2048 on 16 A100s

### Inference Entry Points
- **Main model class:** `Kimodo` (kimodo_model.py:25-73)
- **Denoiser:** `TwostageDenoiser` (twostage_denoiser.py:15-154)
- **Backbone:** `TransformerEncoderBlock` (backbone.py:101-190)
- **Motion representation:** `KimodoMotionRep` (motion_rep/reps/kimodo_motionrep.py)
- **Constraints:** `constraints.py` (all constraint classes)

### Key Configuration Files
- `model/cfg.py`: ClassifierFreeGuidedModel (separated CFG)
- `model/diffusion.py`: DDPM + DDIMSampler
- `motion_rep/reps/kimodo_motionrep.py`: Forward/inverse transformations

