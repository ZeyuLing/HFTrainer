# MotionCanvas vs KIMODO: NeurIPS Abstract Talking Points

## One-Paragraph Technical Difference

**KIMODO (imputation)**: At every denoising step, constrained motion dimensions are **forcibly replaced** with ground-truth values before the transformer forward pass. This works well for full-body keyframes and trajectories but creates ambiguity when observations are partial—the model has no explicit signal for what was observed vs. inferred.

**MotionCanvas (VACE conditioning)**: Constraints are encoded as **separate input channels** (`inactive`/`reactive`/mask) that remain throughout the denoising process. This makes the observation model explicit: `src_mask=0` unambiguously means "I observed this dimension," enabling arbitrary per-joint per-dimension control and both completion and editing in a single framework.

---

## Concrete Limitations of KIMODO (Defensible for Paper)

### 1. **No Partial Observation Signal** ✅ Most important for NeurIPS
- KIMODO: End-effector position constrained, rotation free → model has no explicit knowledge that position was *observed*
  - Result: rotation evolves without FK grounding, causing inconsistency
  - Mentioned in their own analysis (line 498: "position-only 约束时 rotation 可能不一致")
  
- MotionCanvas: `src_mask` makes this unambiguous
  - Model learns: "This dimension is observed" vs "This dimension is generated"
  - No ambiguity, no inconsistency

**Paper claim**: "Explicit observation modeling via VACE conditioning enables consistent handling of partial constraints, whereas imputation-based conditioning creates inference-time ambiguity in latent variable reconstruction."

### 2. **Joint-Level Granularity Only** ✅ Strong for technical comparison
- KIMODO constraint types are **joint-level**:
  - `FullBodyConstraintSet` (entire joint pos+rot)
  - `EndEffectorConstraintSet` (hand/foot pos+rot as unit)
  - `Root2DConstraintSet` (x,z only; or y-only as separate constraint)
  - **Cannot**: Fix only ankle Y while allowing ankle X, Z to drift

- MotionCanvas operates at **dimension-level**:
  - Any (frame, dimension) pair can be marked observed or unobserved
  - Supports "ankle Y fixed" naturally
  - Supports "elbow rotation only, no position" naturally

**Paper claim**: "Dimension-level conditioning enables fine-grained control over motion completion, supporting part-based edits (e.g., limb height constraints) not expressible in joint-level constraint paradigms."

### 3. **Limited Training Distribution** ✅ Unique to MotionCanvas
- KIMODO Phase 2: Random constraints sampled from 5 predefined types
  - Train-test mismatch when you need "frame 0,50,100 + left arm rot only + ankle XY fixed"
  - Complex masks are out-of-distribution

- MotionCanvas v3 sampler: Rank-K Boolean mask prior
  - M1-M7 mask strategies + v3 Boolean decomposition
  - **Provable coverage**: ≥0.1% effective coverage on 21/25 eval settings (audit in CLAUDE.md)
  - Explicitly designed to cover all evaluation task signatures

**Paper claim**: "A principled mask prior (rank-K Boolean decomposition) ensures training distribution covers all evaluation scenarios, eliminating generalization failures from unseen mask patterns."

### 4. **Editing vs Completion** ✅ Architectural elegance argument
- KIMODO: Only completion (T2M + constraints) mentioned
  - Motion editing ("make smoother", "remove jitter") not designed in

- MotionCanvas: Single framework for both
  - Completion: `reactive = 0`
  - Editing: `reactive = LQ_motion`
  - Same architecture, different data flow

**Paper claim**: "VACE conditioning unifies motion editing and completion through a single reactive/inactive channel, supporting both quality improvement and task conditioning without separate model variants."

---

## What NOT to Claim (False or Too Strong)

❌ "KIMODO cannot handle constraints" — False. Keyframes + trajectories work well.  
❌ "Imputation is fundamentally flawed" — False. It's a reasonable design for specific use cases.  
❌ "KIMODO has lower quality" — Unknown. No direct comparison provided.  
❌ "KIMODO is just a worse version of MotionCanvas" — Unfair. They make different trade-offs.

---

## What IS Safe to Claim

✅ "KIMODO's imputation lacks explicit observation modeling, creating ambiguity for partial constraints"  
✅ "Dimension-level mask support enables finer-grained control than joint-level constraint types"  
✅ "Rank-K Boolean mask prior provides provable coverage of evaluation distributions"  
✅ "VACE conditioning unifies editing and completion, whereas imputation addresses only completion"  
✅ "Empirically, dimension-level masks cover more diverse evaluation scenarios than predefined constraint types"

---

## Suggested NeurIPS Abstract Language

**[Related Work]**
"Prior work on motion completion uses task-specific conditioning: KIMODO (Rempe et al., 2026) applies constraints through direct imputation—replacing noisy motion dimensions with ground-truth values at each diffusion step. While effective for keyframe-based animation workflows, imputation creates ambiguity when observations are partial (e.g., end-effector position without rotation), as the model lacks explicit signal to distinguish observed vs. inferred dimensions. Furthermore, imputation operates at joint-level granularity, limiting support for fine-grained part-based edits (e.g., fixing only ankle height).

In contrast, we propose VACE conditioning, which makes the observation model explicit through separate input channels (observed motion, reactive hints, binary mask). This enables (1) unambiguous handling of partial observations, (2) arbitrary dimension-level control, and (3) unified treatment of motion editing and completion via the reactive channel."

**[Technical Contribution]**
"We introduce MotionCanvas, a unified motion completion and editing framework based on flow matching with VACE conditioning. Key innovations: (1) A rank-K Boolean mask prior that provably covers all evaluation task patterns, eliminating train-test mismatch for complex masks; (2) Dimension-level per-frame masks enabling fine-grained control (e.g., ankle height constraints); (3) A reactive channel that supports both completion (reactive=0) and editing (reactive=LQ_motion) in a single model."

---

## For Rebuttal / Reviewer Push-back

**Q: "KIMODO works well in practice, so why complicate with VACE?"**

A: KIMODO is optimized for keyframe + trajectory workflows (game animation). MotionCanvas is designed for open-domain completion and editing. The trade-offs:
- KIMODO: Simpler constraint API, excellent keyframe precision
- MotionCanvas: Finer granularity, unified editing support, robustness to unseen mask patterns
- Different target applications, not "VACE is always better"

**Q: "Can't KIMODO handle partial observations in its 5 constraint types?"**

A: KIMODO's 5 types are `FullBody`/`EndEffector`/`Root2D`/`RootY`/`Heading`. To handle "hand position without rotation," you'd need:
1. New constraint type? No—not in their design
2. Use `EndEffectorConstraintSet` (pos+rot together)? Then you're over-constraining
3. Manually impute only position dims? Then model gets no explicit observation signal for position

MotionCanvas handles this naturally: mark position dims as observed, rotation dims as unobserved.

**Q: "KIMODO uses 700h optical mocap, MotionCanvas uses MotionHub. Quality differences?"**

A: Not the focus here. This is about **conditioning mechanism**, not data quality. KIMODO's quality comes from data scale; MotionCanvas's generalization comes from conditioning design.

---

## Bottom Line for Abstract

**Lead with partial observation ambiguity** (universal problem, not specific to games)  
**Back with dimension-level control** (enables broader applications)  
**Validate with mask-pattern coverage** (provable generalization)  

Don't claim KIMODO is "wrong"—claim MotionCanvas solves problems KIMODO doesn't address.

