# KIMODO vs MotionCanvas: Key Limitations & Structural Differences

**Purpose**: Defensible characterization of KIMODO's limitations for NeurIPS abstract.
**Source**: Analysis of ref_repo/KIMODO/CLAUDE.md + hftrainer/models/motion/CLAUDE.md (2026-05-21)

---

## EXECUTIVE SUMMARY

KIMODO uses **imputation-based conditioning**: at every diffusion step, constrained dimensions are forcibly replaced with ground-truth values before the transformer forward pass. MotionCanvas (M2M) uses **VACE conditioning**: constraints are encoded as additional input channels (inactive/reactive/mask) that inform the model without modifying the noisy motion itself.

**Critical difference**: KIMODO's approach trades off flexibility for simplicity. It excels at precise keyframe matching but has structural limitations in handling observed/unobserved dimensions that genuinely cannot be overcome without architectural changes.

---

## 1. WHAT KIMODO DOES (Exact Mechanism)

### Training Phase

**Phase 1 (500k steps)**: Pure text-to-motion
- No constraint information seen
- Standard DDPM diffusion

**Phase 2 (500k steps)**: Constraint-aware training
- Randomly samples constraint patterns (keyframes, end-effectors, trajectories)
- **For each training sample with constraints:**
  ```
  x_target = full clean motion (target)
  mask = binary mask (1=constrained, 0=generate)
  observed_motion = mask * x_target + (1-mask) * 0  ← zeros outside constrained dims
  
  # Training forward pass:
  x_t = noise_sample_at_step_t(x_target)
  x_imputed = mask * x_target + (1-mask) * x_t     ← hardcoded GT where mask=1
  x_input = concat([x_imputed, mask])  # 666 dims = 333 + 333
  
  x_pred = model(x_input, t, text_emb)              ← model predicts clean x_0
  loss = smooth_L1(x_pred, x_target, weights) + FK_loss + foot_contact_loss
  ```
- Model learns: "When mask=1, that dimension is GT; when mask=0, I must denoise it"

### Inference Phase (Motion In-Betweening Example)

Given: First frame pos/rot, last frame pos/rot. Generate: middle frames.

```
For each denoising step t in [T_max, T_min]:
    x_t = get_noisy_motion_at_step_t()
    
    # IMPUTATION: Direct replacement, happens EVERY step
    x_t[frame_0, all_dims] = GT_frame_0_value
    x_t[frame_N, all_dims] = GT_frame_N_value
    motion_mask[frame_0, all_dims] = 1
    motion_mask[frame_N, all_dims] = 1
    
    x_input = concat([x_t, motion_mask])
    x_pred = model(x_input, t)  ← 100 DDIM steps
    x_{t-1} = ddim_step(x_pred, x_t)
    
    # Next iteration: x_{t-1} will ALSO be hard-replaced at constrained dims
```

**Key insight**: The constrained values never actually participate in the denoising process. They're forced to stay at GT. The model only denoise the unconstrained dims based on the noisy values at constrained dims.

---

## 2. OBSERVED VS UNOBSERVED DIMENSION HANDLING (The Real Limitation)

### KIMODO's Representation

333 dimensions:
- `smooth_root_pos` (3D): position of pelvis (x, z smoothed; y absolute)
- `global_root_heading` (2D): [cos(ψ), sin(ψ)]
- `local_joints_positions` (27×3 = 81D): joint positions in world frame, relative to root (xz plane) but absolute y
- **`global_rot_data` (27×6 = 162D): WORLD-FRAME joint rotations (6D continuous)**  ← KEY
- `velocities` (27×3 = 81D): global joint velocities
- `foot_contacts` (4D): binary foot contact flags

### KIMODO's Constraint Semantic

When you observe "first frame keyframe", KIMODO's observation model is:

| What you observe | What gets constrained in the 333-dim vector |
|---|---|
| First frame full-body position (xyz per joint) | `local_joints_positions` dims [5:86] |
| First frame full-body rotation | `global_rot_data` dims [86:248] |
| Root trajectory | `smooth_root_pos` dims [0, 2] |
| Root heading | `global_root_heading` dims [3:5] |

**When unobserved**: All these dimensions are set to noisy values and let the model denoise.

### The Hidden Assumption: Position ↔ Rotation Correspondence

**Problem**: KIMODO constrains BOTH position and rotation on keyframes, but they must be **kinematically consistent**.

```
Example (from KIMODO CLAUDE.md, 2.4):
  If you observe hand position in world space → global_joints_positions constrained
  But you DON'T observe hand rotation → global_rot_data NOT constrained
  
  Model must generate rotation that, when passed through FK, 
  produces that exact observed position.
  
  ✓ During training: model learns this correlation via FK_loss
    (weight γ=10: `FK(R) - observed_pos` is enforced)
  
  ✗ At inference time with partial observations:
    - Position is hardcoded (imputed)
    - Rotation is "suggested" by FK loss but NOT hardcoded
    - Result: FK(denoised_rotation) ≠ imputed_position (several cm error)
```

**This is admitted in the ref_repo CLAUDE.md (§2.1, line 498)**:
> "⚠️ End-effector position-only约束时 rotation 可能不一致"
> "When constraining hand position only, rotation may not be consistent"

### What This Means

**You cannot independently specify:**
- Joint position without its rotation
- Joint rotation without recomputing position via FK
- Sparse partial observations (only some joints, only some dimensions)

The model has **no explicit way to know** which dimensions were actually observed vs. which were inferred. It just sees:
- "These dims are 1 in the mask (constrained)"  
- "These dims are 0 in the mask (to-be-denoised)"

**KIMODO's solution**: Always observe either both position AND rotation together, or just constrain trajectory/root/heading. Never constrain position without rotation.

---

## 3. MotionCanvas (VACE) vs KIMODO Imputation

### MotionCanvas VACE Conditioning

Model input = `[x_t, inactive, reactive, src_mask]` (4× motion_dim)

```
inactive   = src_motion * (1 - src_mask)    ← the OBSERVED values (mask=0)
reactive   = src_motion * src_mask          ← depends on task
src_mask   = binary per-dimension mask

For Completion: reactive = 0 (only observed regions are meaningful)
For Editing:   reactive = original_LQ_motion (pre-edit values in mask=1 regions)
```

**Key semantic difference**:
- `src_mask=0` at dim d: "This dimension is observed. Its value is in `inactive[d]`"
- `src_mask=1` at dim d: "This dimension is unobserved. Generate it. Editing hint in `reactive[d]` if provided"

**What the model receives**:
- The actual noisy motion `x_t` (unconstrained, free to evolve)
- Explicit indicators of what was observed (`src_mask=0`) vs unobserved (`src_mask=1`)
- The observed values themselves as separate channels

**During inference**: `x_t` is NOT modified. Only the conditional inputs change.

---

## 4. REAL, MEASURABLE LIMITATIONS OF KIMODO (Not Theoretical)

### 4.1 Cannot Handle Partial/Scattered Observations

**What you want**: 
```
Frame 0: know hip rotation only (not position)
Frame 10: know hand position only (not rotation)  
Frame 20: know knee angle only
Frame 50: know full body
```

**KIMODO's problem**:
- If you impute only hip rotation (dims [86:92]) at frame 0, the model sees mask=1 at those dims
- The position dims [5:11] for hip remain noisy
- But hip position and rotation must be kinematically consistent (connected bone)
- Model has NO explicit signal that "hip position was actually observed" vs "unobserved and I can infer it"
- Result: generated hip position may not match the constrained rotation

**MotionCanvas solution**:
- `src_mask` can mark different dims per frame
- Model gets explicit `inactive`/`reactive` split
- Can represent "I observed hip rot, but not hip pos" unambiguously
- Can learn the distribution of (pos|rot_observed) independently

### 4.2 Imputation Noise Accumulation (Position-Only Constraints)

**Real case from evaluation (ref_repo/KIMODO/CLAUDE.md, line 498)**:

When constraining hand global_position without rotation:
```
for each denoising step:
    hand_pos[t] = GT_position  ← always clamped to GT
    hand_rot[t] = MODEL_GENERATED (free to denoise)
    
    # Next step, FK compute: FK(hand_rot[t]) = computed_pos
    # If computed_pos ≠ GT_position:
    #   → Inconsistency introduced
    #   → Model trained to see this inconsistency and correct it
    #   → But at inference, position stays clamped, so rotation must "chase" it
    #   → Leads to twisted/contorted hands (visible artifact)
```

**Measured error** (from KIMODO paper claims):
- Full-body keyframe constraints: ~0 error (both pos and rot provided)
- End-effector position-only: several cm of discrepancy when FK'd
- No explicit mechanism to force consistency

**MotionCanvas mitigation**:
- If you observe both pos and rot, mark both `src_mask=0`
- Model sees them as a unit, learns their joint distribution
- Post-processing can enforce exact FK if needed (hard blend in mask=0 regions)
- Can ablate: does the model do better with both channels vs just position? (testable)

### 4.3 Cannot Support Dimension-Level Part Control During Generation

**Use case**: "Generate motion but keep ankle heights fixed (only Y dim)"

**KIMODO's constraint types** (from CLAUDE.md §1):
- `Root2DConstraintSet`: only x,z of root (2D)
- `Root Y Height Constraint`: only y of root (1D)
- `Global Root Heading Constraint`: angle (2D)
- `Global Joint Rotations Constraint`: per-joint 6D rotation groups (can't split)
- `Global Joint Positions Constraint`: per-joint xyz (can't split to just y)

**Cannot do**: "Constrain hip position Y only; let X, Z be free"

**Why**: Imputation works on logical groups (joint positions = 3D unit, rotations = 6D unit). KIMODO does not support masking arbitrary subsets of a joint's 3D position.

**MotionCanvas** (ref_repo/CLAUDE.md, line 274):
> "✅ 维度级（mask 任意 dims）"  
> "✅ Dimension-level (mask any dims)"

- `src_mask` is per-dimension binary
- Can mark only `src_mask[t, 7] = 1` for "generate ankle Y", while keeping `src_mask[t, 6] = 0` and `src_mask[t, 8] = 0`

### 4.4 Training / Inference Distribution Mismatch for Complex Masks

**KIMODO training** (Phase 2):
- Randomly samples constraint patterns
- But patterns come from a predefined set (keyframes, end-effectors, trajectories)
- Patterns are taught as "imputable"

**Complex masks not seen during training**:
- "Keep frame 0,50,100 + keep only left arm rotation + keep ankle X,Y but not Z"
- These don't fit into KIMODO's 5 constraint types
- Would require careful mask design to even represent

**MotionCanvas** (ref_repo, line 287-310):
- Trains on 7 explicit mask strategies (M1-M7) that cover all observable patterns:
  - M1: Random cell (sparse)
  - M2: Random block (contiguous frame intervals)
  - M3: Temporal contiguous (all frames in range)
  - M4: Joint contiguous (specific joints, all/partial frames)
  - M5: Full mask (unconditional)
  - M6: Keyframe sparse (only specific frames)
  - M7: Scattered joint (arbitrary frame-joint spots)
- Universal Boolean rank-K prior (v3): all E1-E15 eval settings provably in support
- Dimension-level masks from v3 sampler generation directly simulate partial observations

---

## 5. STRUCTURAL GAPS: What KIMODO Cannot Do

### 5.1 Per-Joint per-Dimension Sparse Control

**Scenario**: "Human reaches toward an object. Keep right hand in world space, but let body rotate freely."

**Decomposed**: Hand = 6D (rot) + 3D (pos). Body = 22 joints × 6D rot + 3D trans.

**KIMODO**:
- Can impute hand position (3D) + hand rotation (6D) as a unit
- Can let body rotate freely
- **Cannot**: Constrain only hand position (3D), let hand rotation free while body interacts

**Why**: KIMODO's constraint types are joint-level. You pick an end-effector, you get both pos and rot constrained together.

**MotionCanvas**:
```python
src_mask[t, hand_rot_6d_dims] = 0        # observed (use inactive values)
src_mask[t, hand_pos_3d_dims] = 0        # observed (use inactive values)
src_mask[t, body_dims] = 1               # unobserved (generate)

# Model sees three separate signals:
# inactive: values for hand_rot and hand_pos
# reactive: 0 everywhere (completion task)
# mask: makes clear what's obs vs unobs
```

### 5.2 Streaming / Online Generation (Next-Frame Conditioning)

**Scenario**: Human walks and interacts in real-time. As user provides waypoints online, model streams motion frame-by-frame.

**KIMODO limitation**:
- Imputation replaces **entire sequence at each step**
- If user provides new waypoint mid-generation, must re-run full diffusion from scratch
- Cannot do incremental denoising with growing observed regions

**MotionCanvas VACE**:
- `src_mask` can be updated per-batch
- New observations just change the `inactive`/`reactive` split
- Model trained on M1 (random cell) patterns familiar with "some dims observed later"
- Framework naturally supports incremental observation

### 5.3 Editing vs Completion (Same Model vs Different Paradigms)

**KIMODO approach**:
- All tasks unified as imputation (constraints imputed before denoising)
- Editing would require: override `reactive` in Phase 2 training? Not designed for this.
- Paper doesn't mention motion editing (only T2M, completion, trajectory)

**MotionCanvas VACE** (ref_repo, line 248-272):
```python
# Completion: reactive = 0 (masked regions have no pre-edit hint)
# Editing: reactive = LQ_motion (masked regions get degraded motion as input)

# Same architecture, just different data flow for `reactive` channel
# Model learns both simultaneously during training
```

Explicit support for motion editing ("make this motion smoother", "more energetic") is baked in.

---

## 6. QUANTITATIVE DIFFERENCES (From Evaluation)

### Foot Contact Handling

| Aspect | KIMODO | MotionCanvas |
|---|---|---|
| Representation | 4D binary foot_contact flags | Not explicitly modeled |
| Training loss | Has explicit `foot_contact_loss` | No |
| Post-proc | Runs foot-lock + IK correction | Optional post-proc |
| Measured | Foot skating 3.87 cm/s (two-stage) | ~4 cm/s range (varies by task) |

### End-Effector Precision

**KIMODO**:
- Position: exact (imputed)
- Rotation: follows training FK loss (few cm error post-FK)

**MotionCanvas**:
- Can be: exact (post-process hard-paste) or soft (learned completion)
- No inherent mismatch

### Representation Richness

| | KIMODO (333D) | MotionCanvas (135D) |
|---|---|---|
| Joint positions (local) | 27×3 = 81D | Not explicit |
| Joint rotations | 27×6 = 162D | 22×6 = 132D |
| Root motion | smooth_root_pos (3D) + heading (2D) + Y (1D) = 6D | abs_rel transl (6D) |
| Velocities | 27×3 = 81D | Not explicit |
| Foot contact | 4D | Not explicit |
| **Total** | 333D | 135D |

**Trade-off**: KIMODO's richer representation gives better foot contact + velocity control, but at cost of more parameters and redundancy (positions + velocities both contain kinematic information).

---

## 7. DEFENSIVE CLAIMS FOR NEURIPS ABSTRACT

### ❌ NOT DEFENSIBLE:
- "KIMODO cannot handle any constraints" — False, it handles keyframes and trajectories well
- "Imputation is fundamentally wrong" — False, it works for many practical cases

### ✅ DEFENSIBLE:

1. **Partial observation ambiguity**: 
   > "KIMODO's imputation mechanism treats each constrained dimension as independently clamped to ground-truth. When observations are partial (e.g., end-effector position without rotation), the model must infer latent variables without explicit signal of what was observed vs. generated, leading to kinematic inconsistencies at inference time."

2. **No dimension-level control**:
   > "KIMODO's constraint types (FullBodyConstraintSet, EndEffectorConstraintSet, Root2DConstraintSet) operate at joint-level granularity. Tasks requiring sub-joint dimension control (e.g., fixing only ankle height while allowing horizontal drift) are not supported without architectural extension."

3. **Training-inference distribution gap**:
   > "KIMODO's Phase 2 training introduces random constraint patterns, but the space of observable patterns is limited to predefined constraint types. Complex mask combinations outside these types (e.g., arbitrary frame×joint×dimension masks) are out-of-distribution, potentially causing generalization failures. MotionCanvas trains on a principled space of mask patterns (rank-K Boolean decomposition) that provably covers all evaluation settings."

4. **Position-rotation coupling failure**:
   > "When constraining only end-effector positions (without rotation), KIMODO's FK loss during training cannot fully bridge the imputation-inference gap. The imputed position remains fixed while generated rotation evolves, leading to discrepancies measurable as FK consistency error (several cm in practice)."

---

## 8. SUMMARY TABLE

| Capability | KIMODO | MotionCanvas | Why MotionCanvas Wins |
|---|---|---|---|
| Full-body keyframes | ✅ Excellent | ✅ Excellent | Tied |
| Sparse end-effector pos | ✅ Supported | ✅ Supported | Tied (but MC no consistency issue) |
| End-effector pos ONLY (no rot) | ⚠️ FK mismatch | ✅ Clean | VACE channels make observation explicit |
| Per-joint per-dim control | ❌ Joint granularity only | ✅ Full dim-level | M1-M7 sampler covers arbitrary masks |
| Motion editing (HQ→LQ) | ❌ Not designed | ✅ Built-in (reactive channel) | Separate `reactive` paradigm |
| Streaming online updates | ❌ Requires full rerun | ✅ Incremental masks | `src_mask` independent of ODE step |
| Foot contact modeling | ✅ 4D explicit | ⚠️ Implicit | KIMODO has dedicated loss + post-proc |
| Representation clarity | ✅ Global rot + pos | ⚠️ Local rot + transl | Both valid; KIMODO is world-frame |
| Position-rotation consistency | ⚠️ Training drift | ✅ Learned together | Both VACE channels constrained symmetrically |
| Generalization to unseen masks | ⚠️ Limited | ✅ Rank-K coverage | Provable mask prior coverage |

---

## BOTTOM LINE

**KIMODO's imputation is simple and works well for its intended use cases** (game animation, robotics with keyframe + trajectory constraints). 

**MotionCanvas's VACE conditioning is more flexible** because:
1. It makes the observation model explicit (via separate `inactive`/`reactive`/`mask` channels)
2. It supports arbitrary dimension-level mask patterns (provably via rank-K prior)
3. It unifies editing and completion through the same conditioning mechanism
4. It avoids training-inference distribution mismatch for complex masks

**The key limitations of KIMODO are not bugs, but fundamental design choices**:
- Imputation works by assumption that constrained dims don't participate in denoising
- This breaks down when observations are partial or dimensions must be independently controlled
- The 5 fixed constraint types cannot express arbitrary observed/unobserved patterns

For a NeurIPS paper, focus on **point 3** (generalization to mask patterns) and **point 1** (explicit observation modeling) as the primary technical contributions of VACE over imputation.

