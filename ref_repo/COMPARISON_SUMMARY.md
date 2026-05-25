# Executive Summary: UMO vs MotionLab vs HyMotion M2M

## Quick Answer: What's Different at the TECHNICAL Level

### **UMO (Brown/MIT/Meta) — Frame-Level Meta-Ops**

**Control Mechanism**: 3-class per-frame labels (P=Preserve, G=Generate, E=Edit)
- ❌ **CANNOT do per-joint control** (paper limitation §5)
- ✅ Can freeze backbone + add 0.207M adapter (temporal fusion)
- Element-wise add to input embedding: `x'_t = E_in(x_t) + E_ctx(source + Emb(τ))`
- Adapter: Just 2-layer MLP, initialized as copy of pretrained E_in
- **Semantic advantage**: [E] operation explicitly means "edit based on source"

### **MotionLab (SUTD/Singapore) — 5-Modality Path + Curriculum**

**Control Mechanism**: 5 independent modality paths + Joint Attention + Aligned 1D RoPE
- ✅ **Can do trajectory control** via trajectory hint modality (0.0286m error vs 18.78cm for UMO)
- ✅ Can do instruction editing ("speed up", "opposite leg")
- ✅ Can do style transfer
- **Secret weapon**: Curriculum learning (7 stages, 11.7× FID degradation without it)
- Each modality (source, target, text, trajectory, style) has independent adaLN/FFN
- All share 1D RoPE for time alignment → forces source[i] and target[i] frame sync
- 263D representation (includes velocities, contacts, positions)

### **HyMotion M2M (Our Work) — Per-Dimension Binary Mask**

**Control Mechanism**: (T, 138) binary mask + VACE 3-channel conditioning
- ✅ **ONLY one with true per-joint control** (e.g., regen left arm only)
- ❌ No xyz position dims → can't do trajectory control
- ❌ No explicit instruction editing or style transfer
- Channel concat: `[x_t, inactive, reactive, mask]` → 4× input_encoder
- Mask strategies: M1-M6 fixed ratio (25% random cell, 15% block, 25% temporal, 15% joint, 5% full, 15% keyframe)
- **Advantage**: Most granular control—can manipulate any dimension subset
- **Gap**: No curriculum learning, fixed ratio from day 1

---

## Key Technical Differences

### Conditioning Injection (How control enters the model)

| Method | UMO | MotionLab | M2M |
|--------|-----|-----------|-----|
| **Location** | Input embedding level (element-wise add) | Independent modality paths (separate embeddings + shared attention) | Input encoder input (channel concat) |
| **Overhead** | 0.207M params only | Full architecture trained | 4× input_encoder |
| **Backbone** | Frozen (T2M preserved) | Trained from scratch | Trained |
| **Cross-modal interaction** | Through backbone's self-attention | Through JointAttention with QKV concat | Through shared input_encoder |

### Motion Representation

| Dimension | UMO (201D) | MotionLab (263D) | M2M (138D) |
|-----------|-----------|------------------|-----------|
| Global translation | ✅ 3D | ✅ 3D + velocity | ✅ 3D (abs/rel) |
| Joint rotations | ✅ 21×6D local | ✅ 22×6D local | ✅ 22×6D local |
| **Joint positions** | ✅ 22×3D local | ✅ 22×3D local + velocity | ❌ MISSING |
| **Velocity** | — | ✅ 3D root + 22×3D joint | — |
| **Foot contact** | — | ✅ 4D | — |
| **Trajectory hints** | ❌ (text only) | ✅ 66D (independent modality) | ❌ |

### Control Granularity

| Capability | UMO | MotionLab | M2M |
|-----------|-----|-----------|-----|
| **Whole-frame editing** | ✅ (frame-level τ) | ✅ (via instruction + source) | ✅ (all T×138 mask=1) |
| **Per-joint masking** | ❌ Explicitly limited | ⚠️ Via trajectory hint + text | ✅ Native (dims 30-42 for left arm) |
| **Trajectory following** | ❌ Only 18.78cm (slow optimize) | ✅ 0.0286m (Aligned ROPE) | ❌ No xyz dims |
| **Style transfer** | ❌ | ✅ (style modality + MCM encoder) | ❌ |
| **Instruction editing** | ✅ (via [E] + text) | ✅ (task instruction) | ❌ M4 is part-only regen, no semantics |

### Training Strategy

| Aspect | UMO | MotionLab | M2M |
|--------|-----|-----------|-----|
| **Curriculum** | ❌ None — all tasks from day 1 | ✅ 7-stage (pre-train 1000ep + finetune 1400ep) | ❌ Fixed ratio M1-M6 |
| **Task routing** | Implicit (all tasks mixed) | Explicit (CLIP instruction text added to adaLN) | Implicit (mask pattern only) |
| **Multi-task effect** | Not studied | Proven better than expert (unified > specialist) | Unclear (needs ablation) |
| **Anti-forgetting** | None | ✅ FID-weighted importance sampling | None |
| **Ablation impact** | N/A | **Curriculum removal → 11.7× FID degradation** | N/A |

---

## What's GENUINELY Different About M2M

### 1. **Per-Dimension Control is Unique**
Only M2M allows:
```
M4 joint editing example:
mask[:, 30:42] = 1   # Regen LEFT ARM (dims 30-42)
mask[:, 0:30] = 0    # Keep body
mask[:, 42:] = 0     # Keep right arm

UMO cannot do this: [E] is per-frame, not per-joint
MotionLab approximates via trajectory hint but not native
```

### 2. **VACE Three-Channel Split is Novel**
- `inactive = source * (1-mask)`: "Here's what I know"
- `reactive = source * mask`: "Here are the masked regions I'm ignoring"
- `mask`: "These are what you need to generate"

Different from:
- UMO: Element-wise add (simpler but less explicit)
- MotionLab: Independent paths (more parameters but cleaner)

### 3. **No Pretrained Backbone Dependency**
- UMO: Must use HY-Motion-Lite frozen checkpoint
- M2M: Trains from scratch on large foundation model → own optimization path

### 4. **Simplicity of Representation**
- MotionLab: 263D with implicit semantics (velocity channels, contact channels)
- M2M: 138D purely (translation + rotation) → interpretation clearer

---

## M2M's Critical Gaps

| Gap | M2M Currently | UMO Does | MotionLab Does | Effort |
|-----|---------------|----------|-----------------|--------|
| Task awareness | Implicit (mask pattern) | Implicit (data) | Explicit (CLIP text instruction) | Easy (1 week) |
| Curriculum learning | Fixed ratio | None (not needed?) | 7-stage (CRITICAL: 11.7× difference) | Medium (1-2 weeks) |
| xyz trajectory control | ❌ No position dims | ❌ Only 18.78cm via text | ✅ 0.0286m via Aligned ROPE | Hard (2 weeks + retrain) |
| Instruction editing | ❌ M4 only part-regen | ✅ [E] + text instruction | ✅ Task instruction modulation | Medium (1-2 weeks) |
| Style transfer | ❌ | ❌ | ✅ Style modality path | Medium (reuse MCM encoder) |

---

## Top 3 Recommendations to Boost M2M

### **P0: Task Instruction Modulation** (Doable in 1 week)
Add CLIP-encoded task instruction to timestep embedding. Cost: negligible parameters. Benefit: explicit task awareness (helps multi-task focus, preps for instruction editing).

### **P0: Motion Curriculum Learning** (Doable in 1-2 weeks)
Replace fixed M1-M6 ratio with staged schedule. Evidence: MotionLab removed curriculum → **11.7× FID degradation**. M2M likely has similar untapped potential.

### **P1: Position Dimensions + Trajectory Control** (If needed)
Add 3D root xyz to representation (138 → 141D). Implement Aligned 1D RoPE if adding trajectory as separate modality. Enables OmniControl-level trajectory accuracy (0.0286m vs impossible now).

---

## Bottom Line

| Aspect | Winner | Why |
|--------|--------|-----|
| **Parameter efficiency** | UMO | 0.207M on frozen backbone |
| **Training recipe** | MotionLab | Curriculum is secret sauce (11.7× FID gain) |
| **Control granularity** | **M2M** | Only one with per-joint native support |
| **Trajectory precision** | MotionLab | 0.0286m via Aligned 1D RoPE vs M2M's ❌ |
| **Editing semantics** | MotionLab | Instruction text more powerful than mask patterns |
| **Foundation quality** | M2M | Large model (0.46-1.5B) + large data (549k) |

**M2M's edge**: Ultra-fine granular control. M2M's weakness: Lacks training sophistication (curriculum + task awareness) that MotionLab proved critical.

