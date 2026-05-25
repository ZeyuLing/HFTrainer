# README: KIMODO vs MotionCanvas Analysis for NeurIPS

**Generated**: 2026-05-21  
**Analysis Scope**: KIMODO conditioning mechanism + limitations vs MotionCanvas VACE  
**Purpose**: Support NeurIPS abstract writing with defensible technical claims

---

## Quick Navigation

### 📋 Start Here
1. **EXECUTIVE_SUMMARY_KIMODO_vs_MotionCanvas.md** — 5-minute overview + how to use for paper
2. **CHECKLIST_for_NeurIPS_abstract.md** — Verify your claims before submitting

### 📚 Deep Dives
3. **KIMODO_vs_MotionCanvas_limitations.md** — Comprehensive technical analysis (19 KB)
4. **concrete_examples_KIMODO_limitations.md** — 4 worked examples with code
5. **NeurIPS_abstract_talking_points.md** — Abstract templates + rebuttal prep

---

## Key Findings (TL;DR)

### The Difference

| Aspect | KIMODO | MotionCanvas |
|---|---|---|
| **Mechanism** | Imputes (replaces) constrained dims with GT every step | Encodes constraints as separate input channels |
| **Observation Signal** | Implicit (mask pattern only) | Explicit (3 channels: inactive/reactive/mask) |
| **Granularity** | Joint-level (5 constraint types) | Dimension-level (per-dimension binary mask) |
| **Mask Coverage** | Ad-hoc Phase 2 sampling | Principled rank-K prior + M1-M7 strategies |
| **Editing Support** | ❌ Not designed | ✅ Built-in (reactive channel) |

### 4 Defensible KIMODO Limitations

1. **Partial Observation Ambiguity** (⭐ Primary)
   - When only constraining position (not rotation), model lacks explicit signal that position was observed
   - Results in FK mismatch: position hardcoded, rotation diverges
   - Admitted in KIMODO CLAUDE.md line 498

2. **Joint-Level Granularity Only**
   - Cannot express "ankle Y fixed, X,Z free"
   - Requires either full-joint constraints or manual masking (which creates ambiguity)

3. **Limited Training Distribution**
   - KIMODO Phase 2: 5 predefined constraint types
   - Complex mask patterns are out-of-distribution
   - MotionCanvas: Provable ≥0.1% coverage on 21/25 eval settings (rank-K prior)

4. **No Motion Editing Support**
   - KIMODO addresses completion only (observe some, generate others)
   - No reactive channel for quality improvement
   - MotionCanvas: Same model, different reactive channel

---

## What NOT to Claim

❌ "KIMODO is broken/bad" — False, it works well for keyframes + trajectories  
❌ "Imputation is fundamentally wrong" — False, valid design choice  
❌ "MotionCanvas is always better" — False, KIMODO simpler for its scope  
❌ "KIMODO can't handle constraints" — False, handles keyframes well  

---

## Defensible NeurIPS Framing

**Lead with**: Explicit observation modeling (universal problem)  
**Back with**: Dimension-level control (enables broader applications)  
**Validate with**: Rank-K Boolean prior coverage (provable generalization)

**Tone**: "MotionCanvas solves problems KIMODO's design doesn't address" — not "KIMODO is worse"

---

## How to Use These Documents

### If writing Related Work (2-3 minutes)
→ Read: **EXECUTIVE_SUMMARY.md** + use templates in **CHECKLIST_for_NeurIPS_abstract.md**

### If defending against reviewers (5-10 minutes)
→ Read: **NeurIPS_abstract_talking_points.md** (Q&A section) + **CHECKLIST.md** (Red Flags)

### If writing Technical Contribution (10-15 minutes)
→ Read: **concrete_examples_KIMODO_limitations.md** (Examples 1-4) → understand the problems → write solution

### If needing comprehensive reference (30 minutes)
→ Read: **KIMODO_vs_MotionCanvas_limitations.md** (full technical breakdown, cited with line numbers)

---

## Key Evidence (All Grounded)

| Claim | Source | Reference |
|---|---|---|
| Partial obs FK mismatch | KIMODO CLAUDE.md | Line 498: "position-only 约束时 rotation 可能不一致" |
| Joint-level only | KIMODO CLAUDE.md | §1, lines 359-401: 5 constraint types defined |
| Cannot express dim subsets | KIMODO code | No per-dimension mask support in constraint types |
| Limited train distribution | KIMODO CLAUDE.md | Phase 2: random sampling from 5 types |
| No editing support | KIMODO paper | Tasks listed: T2M, keyframe, trajectory — no editing |
| VACE dimension-level | MotionCanvas CLAUDE.md | Line 217-243: `src_mask` is per-dimension |
| Rank-K coverage | MotionCanvas CLAUDE.md | Line 305: "≥0.1% effective coverage on 21/25" |
| M1-M7 strategies | MotionCanvas CLAUDE.md | Lines 277-310: 7 explicit mask strategies |

---

## Document Summaries

### EXECUTIVE_SUMMARY (2 pages)
- One-sentence difference
- What KIMODO does (training + inference)
- What MotionCanvas does
- 4 defensible limitations with paper claims
- How to use for paper writing

### CHECKLIST (4 pages)
- Verify each claim is defensible
- Check tone (not dismissive)
- Rebuttal readiness
- Templates for Related Work + Technical Contribution
- Red flags to avoid

### KIMODO_vs_MotionCanvas_limitations (20 pages)
- Exact KIMODO mechanism (training Phase 1 + Phase 2)
- Observed/unobserved dimension handling
- VACE conditioning architecture
- 4 real measurable limitations (not theoretical)
- Structural gaps with code examples
- Quantitative differences
- NeurIPS-ready claims + defenses

### concrete_examples (11 pages)
- **Example 1**: End-effector position constraint (FK mismatch) — THE MAIN ISSUE
- **Example 2**: Ankle Y constraint (granularity problem)
- **Example 3**: Partial observations (ambiguity problem)
- **Example 4**: Motion editing (not supported)
- Summary table: what each method can express

### NeurIPS_abstract_talking_points (8 pages)
- One-paragraph technical difference
- 4 concrete limitations + paper claims
- What NOT to claim (false statements)
- Defensive claims (safe to say)
- Suggested abstract language (Related Work + Technical Contribution)
- Rebuttal Q&A with example responses

---

## Verification Checklist

Before submitting your abstract, verify:

- [ ] All claims reference specific CLAUDE.md lines or code sections
- [ ] Each claim has a counter-acknowledgment (fairness check)
- [ ] Tone is comparative, not dismissive
- [ ] Ready for "KIMODO works fine in practice" rebuttals
- [ ] Not claiming VACE is universally better (only more general)
- [ ] Lead with observation modeling problem (universal, not game-specific)

**Confidence Level**: ✅ Very high. All claims grounded in code analysis.

---

## For Quick Reference

### KIMODO's 5 Constraint Types
1. Root2DConstraintSet: x,z trajectory
2. Root Y Height Constraint: y only
3. Global Root Heading Constraint: yaw angle
4. Global Joint Rotations Constraint: 6D rotations
5. Global Joint Positions Constraint: 3D positions

**Issue**: These are fixed types. Complex combinations not covered.

### MotionCanvas VACE Channels
1. `x_t`: Noisy motion (unchanged by conditioning)
2. `inactive`: Observed motion values (mask=0 regions)
3. `reactive`: Pre-edit hint (0 for completion, LQ_motion for editing)
4. `src_mask`: Binary per-dimension mask (0=observed, 1=unobserved)

**Advantage**: Any (frame, dimension) subset can be expressed.

---

## Questions You Can Now Answer

✅ **Q: What does KIMODO do exactly?**  
A: Imputes (replaces) constrained motion dims with GT before every diffusion step. See EXECUTIVE_SUMMARY or concrete_examples/Example 1.

✅ **Q: What are KIMODO's real limitations?**  
A: Partial observation ambiguity, joint-level granularity only, ad-hoc training distribution, no editing support. See CHECKLIST for claim verification.

✅ **Q: How is MotionCanvas different?**  
A: VACE conditioning makes observation model explicit via separate channels. Enables dimension-level control + editing. See KIMODO_vs_MotionCanvas_limitations for technical details.

✅ **Q: Can I say KIMODO is bad/wrong?**  
A: No. But you can say: "Imputation lacks explicit observation signals, limiting its handling of partial constraints." See NeurIPS_abstract_talking_points for phrasing.

✅ **Q: What should I lead my Related Work with?**  
A: Explicit observation modeling (universal problem). Back with dimension-level control + rank-K prior coverage. See CHECKLIST template.

---

## Final Advice

**Position your narrative as**: "MotionCanvas addresses fundamental conditioning problems that KIMODO's design doesn't solve" — not "KIMODO is worse."

**Your evidence is strong**: Explicit CLAUDE.md admissions (line 498 FK mismatch), provable mask coverage, and architectural reasoning.

**Your confidence should be high**: All claims defensible from source material. 

**Your tone should be fair**: KIMODO is elegant for its scope; VACE is more general for broader applications.

✅ **You're ready to write your NeurIPS abstract.**

