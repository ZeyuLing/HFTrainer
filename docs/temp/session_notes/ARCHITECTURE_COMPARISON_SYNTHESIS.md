# KIMODO vs HyMotion M2M: Architecture Comparison Synthesis

**Date:** 2026-05-19  
**Purpose:** Consolidated technical comparison for system design decision-making

---

## Executive Summary

Both KIMODO (NVIDIA) and HyMotion M2M are transformer-based diffusion models for motion generation, but they embody fundamentally different design philosophies:

| Dimension | KIMODO Philosophy | M2M Philosophy |
|-----------|------------------|-----------------|
| **Coordinate System** | Global world-space (IK-free) | Local parent-relative (SMPL) |
| **Constraint Strategy** | Aggressive imputation (hard locks) | Soft conditioning (information channel) |
| **Training Approach** | Two-phase curriculum (foundation first) | Unified single-phase (joint training) |
| **Architectural Complexity** | Two-stage denoiser (root/body separation) | Single-stage unified (MMDiT) |
| **Constraint Precision** | Exact (dimensions locked at GT values) | Approximate (soft target) |
| **Data Philosophy** | High-quality single-source (700h optical) | Diverse multi-source (mixed quality) |

**Key Insight:** KIMODO optimizes for **spatial precision** and **animator control**, while M2M optimizes for **temporal flexibility** and **unified task learning**.

---

## 1. MOTION REPRESENTATION: The Foundation Difference

### KIMODO (333 dimensions)

```
Per-frame representation (27-joint Bones Rigplay):

[0:3]         smooth_root_pos              # ADMM-smoothed root in XZ, absolute Y
[3:5]         global_root_heading          # [cos(ψ), sin(ψ)] in world frame
[5:86]        local_joints_positions       # 27×3 relative to smooth root
[86:248]      global_rot_data              # 27×6 WORLD-FRAME rotations (6D continuous)
[248:329]     velocities                   # 27×3 computed post-hoc (not predicted)
[329:333]     foot_contacts                # 4D binary (not predicted)

PREDICTED: 248 dims (smooth_root_pos + heading + positions + rotations)
COMPUTED:  85 dims (velocities + foot contacts)
```

**Why this design?**
- Global rotations = direct world-space constraints without IK
- Smooth root = animator workflow (straight lines, curves)
- Explicit positions = can be constrained independently from rotations
- Foot contacts = explicit modeling enables post-process foot lock

### HyMotion M2M (135 dimensions)

```
Per-frame representation (SMPL-22):

[0:3]         abs_rel translation          # 3 absolute root + 3 relative offset
[3:135]       rot_6d                       # 22×6 LOCAL parent-relative rotations

PREDICTED: 135 dims (all)
COMPUTED:  0 dims
```

**Why this design?**
- SMPL-standard local rotations = easy compatibility with SMPL pipelines
- Compact representation = faster inference
- No explicit positions = reduces dimensionality
- No foot contacts = simpler training

### Comparison Table

| Aspect | KIMODO | M2M |
|--------|--------|-----|
| **Total dims** | 333 | 135 |
| **Position representation** | ✅ Explicit (81 dims) | ❌ Implicit (via rotation + transl) |
| **Rotation coordinate frame** | Global (world-space) | Local (SMPL parent-relative) |
| **Foot contacts** | ✅ Explicit (4 dims) | ❌ Not modeled |
| **Velocity** | ✅ Explicit (81 dims, computed) | ❌ Not tracked |
| **Root trajectory** | Smooth root (special) | abs_rel translation (standard) |
| **Predicted dims** | 248/333 (75%) | 135/135 (100%) |
| **Data efficiency** | Lower (more dims) | Higher (more compact) |
| **Spatial precision** | Higher (explicit positions) | Requires IK for global constraints |

---

## 2. CONSTRAINT INJECTION: Two Different Philosophies

### KIMODO: Direct Imputation (Hard Constraints)

**Mechanism:** At each diffusion step, constrained dimensions are forcibly replaced with ground truth values.

```python
# Per-denoising-step (DDIM)
motion_mask: torch.Tensor       # [B, T, 333] binary (1=constrained, 0=free)
observed_motion: torch.Tensor   # [B, T, 333] ground truth at constrained dims

# Step 1: Direct imputation
x_t = x_t * (1 - motion_mask) + observed_motion * motion_mask

# Step 2: Extend input with mask
x_extended = torch.cat([x_t, motion_mask], dim=-1)  # [B, T, 666]

# Step 3: Denoise
pred_x0 = denoiser(x_extended, text, t)

# Step 4: DDIM update
x_{t-1} = compute_next_step(x_t, pred_x0, t)

# Step 5: Re-impute constraints
x_{t-1} = x_{t-1} * (1 - motion_mask) + observed_motion * motion_mask
```

**Constraint Types Supported:**
1. **Root2DConstraintSet**: 2D trajectory (x,z) + optional heading → dims [0,2,3:5]
2. **FullBodyConstraintSet**: All 27 joints (pos + rot) + root → dims [0:86]
3. **EndEffectorConstraintSet**: Hand/foot (pos + rot) + root → subset of [0:86]
4. **Root Y Constraint**: Vertical position only → dim [1]
5. **Global Root Heading**: Rotation around Y → dims [3:5]

**Characteristics:**
- ✅ Constraints are **exact** (dimensions locked at GT)
- ✅ **Zero diffusion noise** at constrained dimensions
- ✅ Simple to implement and verify
- ✅ Supports position constraints directly
- ❌ **Aggressive** (no compromise with generative process)
- ❌ Can produce discontinuities if mask changes drastically

### HyMotion M2M: VACE Conditioning (Soft Constraints)

**Mechanism:** Constraints are encoded as additional input channels; model learns to respect them through training, not through hard replacement.

```python
# Per-denoising-step (ODE)
src_motion: torch.Tensor        # [B, T, 135] source motion
src_mask: torch.Tensor          # [B, T, 135] binary mask (1=observed, 0=generate)

# Prepare condition channels
inactive = src_motion * (1 - src_mask)    # [B, T, 135] known regions
reactive = src_motion * src_mask          # [B, T, 135] unknown regions (split usage)

# Build model input
model_input = torch.cat([
    x_t,           # [B, T, 135] current noisy state
    inactive,      # [B, T, 135] known regions
    reactive,      # [B, T, 135] split reactive (for special tasks)
    src_mask       # [B, T, 135] binary mask
], dim=-1)  # Total: [B, T, 540]

# Denoise
pred_x = denoiser(model_input, text_embs, t)

# ODE step (rectified flow)
x_{t-1} = compute_velocity_step(x_t, pred_x, t)

# Optional post-process (exact_match mode for M5)
if exact_match:
    x_{t-1} = x_{t-1} * (1 - src_mask) + src_motion * src_mask
```

**Mask Strategies (M1-M6):**
1. **M1 (Random Cell)**: Random (T, D) cells → 25% of training
2. **M2 (Random Block)**: Contiguous temporal blocks → 15%
3. **M3 (Temporal Pattern)**: First N frames → 25%
4. **M4 (Joint-Level)**: All dims of selected joints → 15%
5. **M5 (Full Mask)**: All zeros (pure T2M) → 5%
6. **M6 (Keyframe)**: Start/end frames → 15%

**Characteristics:**
- ✅ **Soft constraints** (model learns respect, not forced)
- ✅ Flexible mask patterns (any (T, 135) binary mask)
- ✅ Per-dim granularity (finest control level)
- ✅ No aggressive imputation artifacts
- ❌ Constraints are **approximate** (soft targets, not exact)
- ❌ Requires training-time exposure to mask patterns
- ❌ Cannot directly constrain positions (not in representation)

### Comparison Table

| Aspect | KIMODO Imputation | M2M VACE |
|--------|-------------------|----------|
| **Constraint precision** | Exact (locked at GT) | Approximate (soft target) |
| **Mechanism** | Hard replace per-step | Channel conditioning (learned) |
| **Noise schedule** | Constrained dims: no noise | All dims: full noise schedule |
| **Supported constraints** | Position + rotation (global) | Rotation only (local) |
| **Constraint types** | 5 semantic types | 6 mask strategies |
| **Training requirement** | Phase 2 only (500k steps) | Full training (M1-M6 mixed) |
| **Inference flexibility** | Determined at inference prep | Can vary per-step (theoretically) |
| **Artifact risk** | Discontinuities from imputation | Soft target drift |
| **Implementation** | Simple per-step logic | Integrated into model conditioning |

---

## 3. TRAINING STRATEGY: Curriculum vs. Unified

### KIMODO: Two-Phase Curriculum

```
Phase 1: Pure Text-to-Motion (500k steps)
├─ Objective: Strong T2M foundation
├─ Data: No constraints (motion_mask = all zeros)
├─ Loss: Position + velocity + rotation + FK consistency
├─ Batch: 2048 (16×A100-80GB)
└─ Result: Model learns diverse T2M distribution without constraint bias

Phase 2: Constraint-Aware (500k steps)
├─ Objective: Learn constraint imputation semantics
├─ Data: Random mask patterns (keyframes, trajectories, end-effectors)
├─ Loss: Same as Phase 1, but with imputation loss
├─ Input: [motion + mask] = 666 dims
└─ Result: Model learns "when mask=1, treat as GT truth"
```

**Rationale:**
- Phase 1 prevents constraint task from diluting T2M quality
- Phase 2 adapts pre-trained model to imputation mechanism
- Avoids conflicting gradients (generate vs. constrain)
- Two-phase total: 1M steps

### HyMotion M2M: Single-Phase Unified

```
Single Training Phase (all steps)
├─ Task mix: M1-M6 masks sampled simultaneously
├─ Distribution:
│  ├─ M1 (random cell): 25%
│  ├─ M2 (random block): 15%
│  ├─ M3 (temporal): 25%
│  ├─ M4 (joint): 15%
│  ├─ M5 (full mask/T2M): 5%
│  └─ M6 (keyframe): 15%
├─ Loss: Position + velocity + rotation (internal loss composition)
└─ Result: Single model trained on all tasks jointly
```

**Rationale:**
- Multi-task training → better generalization (no task boundary)
- M5 (5%) maintains T2M capability while learning completion
- VACE conditioning → no phase transition needed
- Unified modeling → no architectural complexity

### Comparison

| Aspect | KIMODO | M2M |
|--------|--------|-----|
| **Phases** | 2 (T2M then constraint) | 1 (all simultaneous) |
| **T2M purity** | Phase 1 (500k) without constraints | 5% M5 sampling |
| **Constraint exposure** | Phase 2 only (500k) | Entire training |
| **Task mixing** | Sequential phases | Concurrent distribution |
| **Convergence** | Slower (2× training time) | Faster (1× training time) |
| **Task boundary** | Explicit phase transition | Implicit via mask sampling |
| **Risk** | Overfitting to constraints in Phase 2 | T2M dilution from low M5 ratio |

---

## 4. ARCHITECTURE: Two-Stage vs. Single-Stage

### KIMODO: Two-Stage Transformer Denoiser

```
TwostageDenoiser (282M parameters total)
│
├─ Stage 1: ROOT PREDICTION
│  │
│  ├─ Input: [motion_extended (666 dims)] or [motion (333 dims)]
│  ├─ Backbone: TransformerEncoderBlock
│  │  ├─ Input: 333 or 666 → Linear → 1024
│  │  ├─ Positional encoding: learned
│  │  ├─ Timestep embedding: t → latent_dim
│  │  ├─ Text projection: 4096 (LLaMA) → 1024
│  │  ├─ Transformer: 16 layers, 8 heads, FF=4096
│  │  └─ Output linear: 1024 → 5
│  │
│  └─ Output: [smooth_root_pos(3), heading(2)] = 5 dims
│     └─ In training: DETACH (no gradient flow to body)
│
├─ Local Root Conversion
│  └─ Global root → local root (for Stage 2 conditioning)
│
└─ Stage 2: BODY PREDICTION
   │
   ├─ Input: [local_root(4), body_features(326), text, timestep]
   │  └─ Conditioned on Stage 1 root prediction
   │
   ├─ Backbone: Same TransformerEncoderBlock
   │  ├─ Input: 4+326+333(mask) = 663 → 1024
   │  ├─ (Same 16-layer, 8-head structure)
   │  └─ Output: 1024 → 328
   │
   └─ Output: [all dims except global root] = 328 dims

Final output: [root_pred(5) || body_pred(328)] = 333 dims
```

**Why two stages?**
- Root is global (simpler → can be separable)
- Body conditioned on predicted root → prevents overfitting to root
- Paper claims: foot skating reduced from 7.59 → 3.87 cm/s
- Semantic separation → easier to debug

**Key detail:** In training, Stage 2 receives DETACHED root prediction to prevent body gradients from corrupting root learning. At inference, no detach (enables classifier-free guidance).

### HyMotion M2M: Single-Stage MMDiT

```
HunyuanMotion MMDiT (460M-1.5B parameters)
│
├─ Dual-stream architecture
│  ├─ Context stream (Qwen3-8B embeddings)
│  └─ Motion stream (motion features)
│
├─ Input conditioning: [x_t, inactive, reactive, mask] = 540 dims
│  └─ Input encoder: 540 → latent_dim
│
├─ Backbone: Multi-modality Diffusion Transformer
│  ├─ Dual-stream attention (context + motion)
│  ├─ Cross-modality fusion
│  └─ Single unified output
│
└─ Output: x_next = 135 dims (full motion state)
```

**Why single-stage?**
- Simpler architecture (no phase transitions)
- Unified learning of all motion aspects
- MMDiT designed for multi-modality (text + trajectory + style)
- Larger parameter budget (460M-1.5B vs. 282M)

### Comparison

| Aspect | KIMODO | M2M |
|--------|--------|-----|
| **Stages** | 2 (root + body) | 1 (unified) |
| **Parameters** | 282M | 460M-1.5B |
| **Complexity** | Moderate (two transformer copies) | Higher (dual-stream MMDiT) |
| **Root handling** | Explicit stage separation | Implicit in unified model |
| **Foot skating** | 3.87 cm/s (stage separation benefit) | Not reported |
| **Inference latency** | 2 forward passes per step | 1 forward pass per step |
| **Architectural coherence** | Clear separation (debuggable) | Unified (black box) |

---

## 5. INFERENCE PIPELINE: Imputation vs. Soft Conditioning

### KIMODO Inference (100-step DDIM)

```
Setup:
├─ Text encode → LLaMA embeddings
├─ Build constraints → observed_motion + motion_mask
└─ Initialize noise: x_T ~ N(0, 1) [B, T, 333]

DDIM Loop (t = 99 → 0):
│
├─ Step 1: Impute constraints
│  └─ x_t = x_t * (1 - mask) + observed * mask
│
├─ Step 2: Extend with mask
│  └─ x_ext = cat([x_t, mask]) → [B, T, 666]
│
├─ Step 3: Two-stage denoising
│  ├─ root_pred = TwostageDenoiser.stage1(x_ext, text, t)
│  ├─ body_pred = TwostageDenoiser.stage2(root_pred_local, body, text, t)
│  └─ pred_x0 = cat([root_pred, body_pred])
│
├─ Step 4: Classifier-free guidance (SEPARATED)
│  ├─ Three forward passes: text-guided, constraint-guided, unconditioned
│  └─ x̂_0 = D_∅ + w_text*(D_text - D_∅) + w_constr*(D_constr - D_∅)
│
├─ Step 5: DDIM step
│  └─ x_{t-1} = compute_ddim_step(x_t, x̂_0, t)
│
└─ Step 6: Re-impute (optional)
   └─ x_{t-1} = x_{t-1} * (1 - mask) + observed * mask

Post-processing (optional):
├─ Denormalization
├─ Inverse transform (6D rotation → matrices → FK)
├─ Foot lock post-process
└─ Output: local_rot_mats, posed_joints, root_positions, foot_contacts

Total time: ~2-5s on RTX 3090 (100×2-stage forward passes + CFG)
```

### HyMotion M2M Inference (50-step ODE)

```
Setup:
├─ Text encode → Dual encoding (Qwen3-8B ctxt + CLIP-L text)
├─ Build conditioning: [x_t, inactive, reactive, mask]
└─ Initialize noise: x_T ~ N(0, 1) [B, T, 135]

ODE Loop (t = T → 0, 50 steps):
│
├─ Step 1: Build input conditioning
│  ├─ inactive = src_motion * (1 - src_mask)
│  ├─ reactive = src_motion * src_mask
│  └─ input = cat([x_t, inactive, reactive, mask]) → [B, T, 540]
│
├─ Step 2: Single-stage denoising
│  └─ pred_x = MMDiT(input, text_embs, t)
│
├─ Step 3: Classifier-free guidance (if enabled)
│  └─ x̂ = uncond + w*(text_cond - uncond)
│
├─ Step 4: ODE/Euler step
│  └─ x_next = compute_velocity_step(x_t, pred_x, t)
│
└─ Optional: Exact match post-process (M5 mode)
   └─ x = x * (1 - mask) + src_motion * mask

Post-processing:
├─ Denormalization
├─ Inverse transform (6D → FK)
└─ Output: local_rot_mats, posed_joints, root_positions

Total time: ~similar (50×single-stage forward passes, simpler CFG)
```

### Comparison

| Aspect | KIMODO | M2M |
|--------|--------|-----|
| **Diffusion steps** | 100 (DDIM) | 50 (ODE/Euler) |
| **Constraint enforcement** | Per-step imputation | No per-step (soft via conditioning) |
| **CFG type** | Separated (text + constraint) | Unified (single weight) |
| **Forward passes per step** | 2 (two stages) | 1 (unified) |
| **Constraint precision** | Exact (guaranteed locked) | Approximate (learned) |
| **Post-process** | Optional (foot lock, IK) | Optional (exact_match M5) |
| **Latency advantage** | More steps but two-stage | Fewer steps, single-stage |

---

## 6. LOSS FUNCTIONS: Explicit vs. Implicit

### KIMODO: Explicit Multi-Component Loss

```
L_total = γ_pos*L1_pos + γ_vel*L1_vel + γ_rot*L1_rot + γ_contact*L_contact + γ_fk*L_fk

Components:
├─ γ_pos = 10
│  └─ smooth_L1(pred_positions - target_positions)
│
├─ γ_vel = 2
│  └─ smooth_L1(pred_velocities - target_velocities)
│  └─ Velocities via finite difference: v_t = (x_{t+1} - x_{t-1}) / 2Δt
│
├─ γ_rot = 10
│  └─ smooth_L1(pred_rot_6d - target_rot_6d)
│
├─ γ_contact = 3
│  └─ BCE(pred_foot_contacts - target_foot_contacts)
│  └─ Ground truth from mocap: height < 0.15m and velocity < 0.10 m/s
│
├─ γ_fk = 5
│  └─ smooth_L1(FK(pred_rotations) - target_positions)
│  └─ **Critical: ensures rotation-position coherence**
│
└─ γ_rot_cons = 5
   └─ smooth_L1(pred_rotation - FK_rotation)

Smooth L1 (Huber loss):
├─ If |Δ| < 1: 0.5 * Δ²
└─ Otherwise: |Δ| - 0.5
```

**Key insight:** FK loss is unique to KIMODO; forces model to learn physically plausible rotations.

### HyMotion M2M: Internal Loss (Not Open-Sourced)

```
Likely includes:
├─ Position smoothness (L1 or L2)
├─ Rotation smoothness (6D distance)
├─ Jitter penalty
├─ Temporal coherence
└─ [Exact components unknown; internal to Meta/ByteDance]

Diffusion loss: Flow Matching with likely velocity or x1-prediction
```

**Inference truth:** M2M uses **Flow Matching** (rectified flow), different from KIMODO's DDPM.

### Comparison

| Loss Component | KIMODO | M2M |
|---|---|---|
| **Position loss** | ✅ Direct L1 on positions | Likely indirect via rotation |
| **Velocity loss** | ✅ Explicit (γ=2) | Unknown |
| **Rotation loss** | ✅ Direct (γ=10) | Likely direct |
| **Foot contact loss** | ✅ Explicit BCE | Not modeled |
| **FK consistency** | ✅ Unique (γ=5) | Not visible |
| **Temporal smoothness** | Implicit (via velocity) | Likely explicit |
| **Diffusion framework** | DDPM (noise prediction) | Flow Matching (velocity prediction) |
| **Transparency** | Fully documented | Internal/proprietary |

---

## 7. DATA & SCALABILITY

### KIMODO: High-Quality Single-Source

```
Dataset: Bones Rigplay (NVIDIA proprietary)
├─ Scale: 700 hours optical mocap
├─ Quality: Production-level, professional mocap
├─ Annotation: Single human-written description per sequence
├─ Skeleton: Native 27-joint Bones rig
├─ FPS: 20fps (training), 30fps (release)
└─ Retarget: SOMA-30, SOMA-77, Unitree G1 robots

Training data characteristics:
├─ High signal-to-noise ratio
├─ Consistent quality across dataset
├─ Limited diversity (single collection source)
└─ Requires massive capture infrastructure
```

### HyMotion M2M: Diverse Multi-Source

```
Dataset: MotionHub (internal, proprietary)
├─ Scale: Mixed mocap + reconstruction + synthetic
├─ Quality: Variable (mocap, marker-less, synthetic)
├─ Annotation: Multiple descriptions per motion (augmented)
├─ Skeleton: SMPL-22
├─ FPS: Variable
└─ Retarget: Via SMPL pipeline

Training data characteristics:
├─ Mixed quality (requires denoising model)
├─ Higher diversity (multiple sources)
├─ Scalable (can add synthetic data)
└─ More accessible (no expensive mocap infrastructure)
```

### Comparison

| Aspect | KIMODO | M2M |
|--------|--------|-----|
| **Data scale** | 700h (high quality) | Undisclosed (diverse) |
| **Quality guarantee** | High (optical) | Variable (mixed) |
| **Diversity** | Lower (single source) | Higher (multi-source) |
| **Annotation** | 1 description per clip | Multiple per clip |
| **Synthetic data** | Not used | Likely used |
| **Infrastructure requirement** | Expensive (mocap) | Lower (reconstruction-capable) |

---

## 8. SUPPORTED TASKS & CAPABILITIES

### KIMODO: Spatial Control Focus

```
✅ Supported Tasks:
├─ Text-to-Motion (no constraints)
├─ Motion In-Betweening (full-body keyframes)
├─ Waypoint Following (2D root trajectory)
├─ End-Effector IK (hand/foot world-space control)
├─ Full-Body Keyframes (retargeting)
├─ Foot Contact Pattern Control
├─ Multi-Prompt Blending (sequential segments)
└─ Robotics Retargeting (SOMA-30/77, Unitree G1)

❌ Not Supported:
├─ Trajectory hints (no trajectory modality)
├─ Style transfer
├─ In-between editing
└─ Per-joint rotation control (only global)

Core strength: **Spatial constraints** (positions, trajectories, end-effectors)
```

### HyMotion M2M: Temporal Flexibility Focus

```
✅ Supported Tasks:
├─ Text-to-Motion (M5: 5% of training)
├─ Motion Completion (M1-M4: temporal inpainting)
├─ Temporal Inpainting (arbitrary (T, 135) masks)
├─ Joint-Level Editing (M4: select joints to regenerate)
├─ Keyframe Infilling (M6: start/end frames locked)
└─ [Implicit: any binary mask pattern via VACE]

❌ Not Supported:
├─ Direct XYZ position control (no position dims)
├─ Trajectory following (no trajectory dims)
├─ Foot contact constraints
├─ Robotics-specific retargeting
└─ Multi-prompt sequential blending

Core strength: **Temporal patterns** (which frames/joints to regenerate)
```

### Comparison

| Task | KIMODO | M2M |
|------|--------|-----|
| **T2M** | ✅ (Phase 1) | ✅ (5% M5) |
| **In-Betweening** | ✅ (FullBody) | ✅ (M6) |
| **Trajectory** | ✅ (2D waypoints) | ❌ |
| **End-Effector IK** | ✅ (world-space) | ❌ |
| **Position control** | ✅ (global) | ❌ |
| **Foot contact** | ✅ (explicit) | ❌ |
| **Joint editing** | ✅ (EndEffector) | ✅ (M4) |
| **Temporal patterns** | Limited | ✅ (M1-M4 flexible) |
| **Multi-prompt** | ✅ | ❌ |

---

## 9. KEY TECHNICAL TRADE-OFFS

### KIMODO's Choices & Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| **Global coordinates** | IK-free constraint application | Requires rotation matrix conversion |
| **333 dims** | Explicit positions + contacts | Larger model input (slower per-step) |
| **Two-stage** | Reduced foot skating, clear separation | Double forward passes per step |
| **Imputation** | Exact constraint satisfaction | Risk of discontinuities |
| **Two-phase training** | Strong T2M foundation | 2× training time |
| **700h optical mocap** | High quality | Expensive infrastructure, limited diversity |
| **Smooth root** | Animator-friendly | Requires special handling |

### M2M's Choices & Trade-offs

| Choice | Benefit | Cost |
|--------|---------|------|
| **135 dims** | Compact representation, fast | Cannot constrain positions directly |
| **Local SMPL rotations** | Standard compatibility | IK required for global constraints |
| **Single-stage** | Fast inference, simple architecture | Less interpretable |
| **VACE conditioning** | Soft constraints, no discontinuities | Approximate satisfaction (not exact) |
| **Unified training** | Single model, faster convergence | Possible T2M dilution |
| **Per-dim masks** | Fine-grained control | Requires careful mask strategy design |
| **Mixed-quality data** | Scalable, diverse | Requires robustness to noise |

---

## 10. STRATEGIC POSITIONING

### What Each System Optimizes For

**KIMODO: Production Animation Pipeline**
- Animator writes curves/waypoints → KIMODO generates animation following them
- Strong constraints with exact satisfaction
- Supports retargeting to different rigs
- Designed for professional VFX/game production

**M2M: Flexible Content Generation**
- Generate diverse motions with various constraints
- Per-frame control granularity (any (T, 135) mask)
- Unified model for all completion patterns
- Designed for research and general motion synthesis

### Vulnerability Analysis

**KIMODO's vulnerabilities:**
- Global coordinate learning requires more data
- Imputation can create discontinuities if constraints are sparse
- No trajectory modality (cannot hint partial paths)
- Position constraints require smooth root setup

**M2M's vulnerabilities:**
- Cannot directly control end-effector positions (no position dims)
- Soft constraints don't guarantee satisfaction
- Lower T2M quality (M5 only 5% of training)
- No explicit foot contact modeling
- Mixed-quality training data needs robust loss design

---

## 11. IMPLEMENTATION INSIGHTS

### For Adding KIMODO-like Features to M2M

```
1. ADD GLOBAL POSITION DIMS
   Current: 135 dims (no positions)
   Proposed: 135 + 27×3 = 216 dims (add global positions)
   Cost: Larger input encoder, slower inference
   Benefit: Direct position constraints like KIMODO

2. IMPLEMENT FK CONSISTENCY LOSS
   Add: ||FK(predicted_rotations) - predicted_positions||_L1
   Benefit: Better rotation-position coherence
   Cost: FK computation per training step

3. EXPLICIT FOOT CONTACT MODELING
   Add: 4-dim foot contact channel
   Add: Contact loss (BCE) with weight ~3
   Benefit: Better foot lock post-processing
   Cost: Minimal (4 extra dims)

4. TWO-PHASE TRAINING OPTION
   Option: Phase 1 (pure T2M), Phase 2 (constraints)
   Benefit: Stronger T2M foundation
   Cost: Double training time (or compromise with partial Phase 1)

5. SMOOTH ROOT TRAJECTORY POST-PROCESS
   Add: ADMM-based smoothing on root XZ (like KIMODO)
   Benefit: Animator-friendly trajectories
   Cost: Minimal (post-process only)
```

### For Adding M2M-like Features to KIMODO

```
1. ADD TRAJECTORY MODALITY
   Like MotionLab: separate trajectory attention stream
   Benefit: Can hint partial paths (not just waypoints)
   Cost: Larger model, more complex architecture

2. SOFT CONDITIONING OPTION
   Alternative to imputation: VACE-like channel concat
   Benefit: Avoid discontinuities, smoother blending
   Cost: Loss of exact constraint guarantee

3. STYLE TRANSFER SUPPORT
   Add: Style embedding stream (like MotionLab)
   Benefit: More expressive control
   Cost: More training data (paired style-motion)

4. UNIFIED ARCHITECTURE
   Merge Stage 1 and Stage 2 into single MMDiT
   Benefit: Simpler, faster inference
   Cost: Lose two-stage benefits (foot skating reduction)
```

---

## 12. CONCLUSION

KIMODO and M2M represent two valid but different design philosophies:

| Philosophy | KIMODO | M2M |
|-----------|--------|-----|
| **Primary Goal** | Spatial precision + animator control | Temporal flexibility + unified learning |
| **Constraint Model** | Hard locks (exact) | Soft conditioning (learned) |
| **Architecture** | Modular (two-stage) | Unified (single-stage) |
| **Representation** | Global coordinates (333D) | Local SMPL (135D) |
| **Training** | Sequential phases (T2M → constraint) | Concurrent tasks (M1-M6 mixed) |
| **Data Philosophy** | High quality (700h) | Diverse (mixed sources) |
| **Target Use Case** | Production animation | Research + general synthesis |

**For Motion Generation Systems in 2026:**
- **If prioritizing spatial control & exact constraints** → KIMODO's approach
- **If prioritizing flexibility & unified modeling** → M2M's approach
- **If prioritizing both** → Hybrid (KIMODO positions + M2M VACE softness)

---

**Document prepared:** 2026-05-19  
**Sources:** KIMODO code (github.com/nv-tlabs/kimodo), M2M internal documentation, ref_repo comparison files
