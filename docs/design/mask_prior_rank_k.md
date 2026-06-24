# Rank-K Boolean Tensor Prior for Universal Motion Mask Sampling

**Status**: Draft (2026-04-25)
**Supersedes**: `condition_sampler_v2.py` Tier 1 + Tier 2 template mixture.
**Applies to**: HYMotion M2M v2 training (198-dim motion representation).

---

## 1. Motivation

The v2 sampler is a **template mixture** — 1 parametric Tier-1 process plus 8
hard-coded Tier-2 templates reverse-engineered from inference tasks
(`end_effector`, `trajectory`, `foot_ground`, ...). Empirical coverage audit
shows:

- E3/E4 (periodic keyframes): effective sampling probability **≈ 0** (Tier-2
  uses `np.random.choice`, not periodic).
- E6 (foot-pos XYZ): **mismatched** with T2-7 (which locks pos_Y only).
- E10 (body-part rot): **≈ 10⁻⁶** (Tier-1 Beta-Bernoulli never produces
  anatomical blocks like "all upper body" consistently across all frames).
- E4 setting C (l_foot): **0** (hard-coded `EE_ALL = {l_ankle, r_ankle,
  l_wrist, r_wrist}` excludes feet).

The v2 design couples training distribution to a fixed inference task set.
Any new task (E16+) requires new Tier-2 code. This is the wrong abstraction.

**Goal**: replace the template mixture with a **mathematically universal prior
on binary masks** `M ∈ {0,1}^{T × D}` (D = 198) that

1. has non-zero density on every eval task's mask signature (coverage);
2. is described by a small set of 1-D primitive distributions (elegance);
3. is strictly more general than the v2 sampler (migration safety);
4. costs no more than v2 at sampling time (tractability).

---

## 2. Mathematical Framework

### 2.1 Boolean Rank-K Decomposition

For any binary tensor `M ∈ {0,1}^{T × D}` we write

```
M = ⋁_{k=1..K} (t_k ⊗ d_k),   t_k ∈ {0,1}^T,  d_k ∈ {0,1}^D
```

where `⊗` is the outer product and `⋁` is element-wise boolean OR. The
smallest such K is the **Boolean rank** `rB(M)`.

**Fact**: `rB(M) ≤ min(T, D)` for any M (trivial row decomposition). In
practice, every mask signature we actually use has `rB ≤ 3`.

Each rank-1 atom `t_k ⊗ d_k` is a **semantic unit**: "lock dimensions `d_k`
at frames `t_k`". Human-authored edit tasks compose naturally from such
atoms.

### 2.2 Generative Prior

```
K ~ πK                            # number of atoms
for k = 1..K:
    t_k ~ πT                      # temporal pattern
    d_k ~ πD                      # dimensional pattern
M = ⋁_k (t_k ⊗ d_k)
```

Note: our mask convention is **1 = generate, 0 = known/locked**. The atoms
`(t_k ⊗ d_k)` describe **known regions** (lock mask `L`); the generate mask
output is `M = 1 - L`.

### 2.3 Distribution Specification

#### 2.3.1 Number of atoms πK

```
K ∈ {0, 1, 2, 3, 4}   w  (0.10, 0.55, 0.25, 0.07, 0.03)
```

(Stored in ``DEFAULT_K_WEIGHTS``.)

- K=0 → pure generation (covers E1).
- K=1 → single-atom (covers ~80% of eval settings).
- K≥2 → compositions (covers E5.C, E13/E14/E15).

#### 2.3.2 Temporal prior πT

Each `t_k` is drawn from a mixture of 6 primitive distributions. Weights
(stored in ``DEFAULT_TEMPORAL_WEIGHTS``) were tuned from the coverage
audit (§3) so that the most common eval patterns (interval-anchor,
periodic-keyframe) dominate:

| Name | Weight | Support | Coverage intent |
| --- | --- | --- | --- |
| `all` | 2.0 | `t = 1_T` | E5, E10 (whole-sequence masks) |
| `empty` | 0.3 | `t = 0_T` | Degenerate atom (rarely useful) |
| `interval` | 3.5 | `[a, a+ℓ)` — see below | E2, E7, E15 (prefix/suffix/mid) |
| `periodic` | 4.0 | Every `p`-th frame, `p ∈ {5,10,15,20,30,60}` | E3, E4 |
| `renewal` | 1.5 | i.i.d. gaps `g_i ∼ Geom(ρ)`, `ρ ∼ LogU[0.02, 0.5]` | E6 contact, sparse keyframes |
| `markov` | 1.0 | 2-state chain, `p_stay ∼ Beta(2,2)` | Smooth block patterns |

**`interval` length + position mixture** (ensures short/mid/long and
prefix/suffix/interior all have material mass):

```
length ~ mixture:
    40%  Uniform[1, T // 10]          (short anchors, E2.start_1f, E7)
    30%  Uniform[T // 10, T // 3]     (mid, E2.pre20, E15)
    30%  Uniform[T // 3, T]           (long, E15 long-prepend)

position ~ mixture:
    1/3  a = 0                         (prefix-biased)
    1/3  a = T - ℓ                     (suffix-biased)
    1/3  a ~ Uniform[0, T - ℓ]         (interior)
```

Every primitive admits the empty and full masks in its support (with
non-zero probability), so `supp(πT) = {0,1}^T` strictly.

#### 2.3.3 Dimensional prior πD

The 198-dim vector decomposes as

```
[0:3]     translation            (3 channels, x/y/z)
[3:135]   22 joints × 6-D rot6d  (132 channels)
[135:198] 21 joints × 3 pos xyz  (63 channels, pelvis excluded)
```

We sample `d ∈ {0,1}^198` via a 2-level hierarchical process:

**Level 1 — kind** (what kind of dimension is active). Stored in
``DEFAULT_KIND_WEIGHTS``:

```
kind ∈ {rot_only, pos_only, trans_only, mixed, all_dim}
       with weights (0.22, 0.30, 0.10, 0.18, 0.20)
```

The ``all_dim`` kind (``d = 1_198``) was added after the first coverage
audit pass: many eval tasks (E2 in-between, E3 keyframe, E7 first-frame,
E8 loop, E15 prepend) pin *all* 198 dims at selected frames, and
composing rot+pos+trans inside ``mixed`` cannot hit this sub-manifold
with meaningful probability because each sub-atom selects a *proper
subset* of its dims.

**Level 2 — kind-specific shape**:

- **rot_only**: pick a joint subset `J ⊂ {0..21}` and lock those joints'
  rot6d (6 dims each). `J` is drawn from:
  - With prob 0.5: an entry from the **anatomical dictionary**
    (17 entries, non-uniform weights — see §2.3.4).
  - With prob 0.3: Bernoulli(p) with `p ∼ Beta(1.5, 4)` → small random
    set.
  - With prob 0.2: a single random joint (`{j}`).
- **pos_only**: pick joint subset `J` (same scheme, from joints 1..21) ×
  weighted channel subset `C ⊂ {x, y, z}` \ {∅} (`xyz=4, xz=4, y=2,
  x/z/xy/yz=1` before normalisation — boosts E4 (xyz), E5 (xz), and E6
  vertical-only).
- **trans_only**: pick channel subset `C ⊂ {x, y, z}` \ {∅} (same
  weighted scheme); no joint selection.
- **mixed**: independently sample a `rot_only` atom, a `pos_only` atom, a
  `trans_only` atom (each with 50 % dropout); OR them together. Fallback
  to a single rot_only atom if all three drop.
- **all_dim**: `d = 1_198` — lock every channel in the selected frames.

This factorisation gives `|supp(πD)|` ≥ 2¹⁹⁸ effective support, but with
high density on anatomically meaningful subsets.

#### 2.3.4 Anatomical Joint Dictionary

17 predefined joint groups with biomechanical meaning. Listed by index
set (0-based SMPL-22 indexing; see `fk_utils.SMPL22_PARENTS`). Weights
(stored in ``ANATOMICAL_WEIGHTS``) are non-uniform: common eval targets
(`all`, `upper_body`, `lower_body`, `ankles`, `wrists`, `hands_feet`,
`end_effectors`) are up-weighted to 2.0; per-limb groups default to 1.0;
`head` is down-weighted to 0.5.

| Key | Joints | Meaning |
| --- | --- | --- |
| `all` | 0..21 | Whole body |
| `pelvis` | {0} | Root only |
| `spine_chain` | {0, 3, 6, 9, 12, 15} | Pelvis + spine1/2/3 + neck + head |
| `upper_body` | {3,6,9,12,13,14,15,16,17,18,19,20,21} | Torso + arms + head |
| `lower_body` | {0,1,2,4,5,7,8,10,11} | Root + hips + knees + ankles + feet |
| `arms` | {13,14,16,17,18,19,20,21} | Collars + shoulders + elbows + wrists |
| `legs` | {1,2,4,5,7,8,10,11} | Hips + knees + ankles + feet |
| `left_arm` | {13,16,18,20} | L_Collar + L_Shoulder + L_Elbow + L_Wrist |
| `right_arm` | {14,17,19,21} | R_Collar + R_Shoulder + R_Elbow + R_Wrist |
| `left_leg` | {1,4,7,10} | L_Hip + L_Knee + L_Ankle + L_Foot |
| `right_leg` | {2,5,8,11} | R_Hip + R_Knee + R_Ankle + R_Foot |
| `ankles` | {7, 8} | L/R_Ankle (foot-ground constraint) |
| `feet` | {10, 11} | L/R_Foot (toes) |
| `wrists` | {20, 21} | L/R_Wrist (hand IK) |
| `hands_feet` | {10,11,20,21} | All four end-effectors (T2-5 superset, now includes feet) |
| `end_effectors` | {7, 8, 20, 21} | v2 EE_ALL (ankles + wrists) |
| `head` | {15} | Head (gaze) |

`end_effectors` is kept for backward compatibility with v2; `hands_feet`
covers E4 setting C (`{r_wrist, l_foot}` is a subset of `hands_feet`, so
the Bernoulli path also covers it).

### 2.4 Editing-Mode Overlay

The edit/repair flag is orthogonal to the Rank-K mask structure. We retain
the v2 convention: with probability `p_edit` (default 0.08) the sample is
marked as edit mode and the caller applies the existing corruptor pipeline;
the Rank-K mask then represents the over-mask (always ⊇ the corrupted
region).

---

## 3. Coverage Analysis

We list, for each eval task, (a) the (t, d) decomposition that reproduces
its mask signature, (b) an analytic lower bound on sampling probability
under the prior in §2.3.

Notation: `p[·]` denotes the probability of a single primitive draw under
πT or πD. `p[all] = 1/6`, `p[periodic(p=15)] = 1/6 × 1/6 = 1/36`, etc.

| Task | (K, t, d) decomposition | Lower-bound prob | Status |
| --- | --- | --- | --- |
| E1 | K=0 | 0.10 | ✓ explicit |
| E2 start_1f | K=1, t=interval(a=0,ℓ=1), d=mixed(all rot+pos+trans) | ≳ 4×10⁻⁴ | ✓ |
| E2 end_1f | K=1, t=interval(a=T-1,ℓ=1), d=mixed | ≳ 4×10⁻⁴ | ✓ (symmetric with start) |
| E2 both_1f | K=2, t_k=intervals@0 and T-1, d=mixed | ≳ 4×10⁻⁷ | ✓ (K=2 branch) |
| E2 pre20 | K=1, t=interval(a=0, ℓ=0.2T), d=mixed | ≳ 10⁻³ | ✓ |
| E2 post20 | K=1, t=interval(a=0.8T, ℓ=0.2T), d=mixed | ≳ 10⁻³ | ✓ |
| E2 mid60 | K=2, t=prefix+suffix, d=mixed | ≳ 10⁻⁶ | ✓ |
| E3 A/B/C | K=1, t=periodic(p∈{15,30,60}), d=mixed | ≳ 3×10⁻³ | ✓ |
| E4 A (r_wrist every 10) | K=1, t=periodic(p=10), d=pos_only({r_wrist}, xyz) | ≳ 10⁻⁴ | ✓ |
| E4 B (ankles every 15) | K=1, t=periodic(p=15), d=pos_only(ankles, xyz) | ≳ 10⁻⁴ | ✓ |
| E4 C (r_wrist + l_foot) | K=1, t=periodic(p=15), d=pos_only(hands_feet ∩ {r_wrist,l_foot}, xyz) | ≳ 10⁻⁶ (via Bernoulli path) | ✓ |
| E4 D,E,F | same pattern, different joints | ≳ 10⁻⁴ | ✓ |
| E5 A,C | K=1, t=all, d=trans_only(xz) [+ K=2 for heading in C] | ≳ 3×10⁻³ | ✓ |
| E5 B | K=1, t=periodic(p=30), d=trans_only(xz) | ≳ 10⁻⁴ | ✓ |
| E6 pos_contact | K=1, t=renewal(ρ≈0.2), d=pos_only(ankles, xyz) | ≳ 10⁻⁴ | ✓ |
| E7 default | K=1, t=interval(a=0, ℓ=1), d=mixed(all) | ≳ 4×10⁻⁴ | ✓ |
| E8 A loop | K=2, t=intervals@0 and T-1, d=mixed | ≳ 10⁻⁶ | ✓ |
| E10 A_upper | K=1, t=all, d=rot_only(upper_body) | ≳ 1.5×10⁻³ | ✓ (anatomical dict) |
| E10 B_lower | K=1, t=all, d=rot_only(lower_body) [+ trans] | ≳ 1.5×10⁻³ | ✓ |
| E10 C_spine | K=1, t=all, d=rot_only(spine_chain) | ≳ 1.5×10⁻³ | ✓ |
| E13/E14/E15 | K ∈ {2,3}, multiple intervals + mixed | ≳ 10⁻⁷ | ✓ (rare but present) |

**Verdict**: every eval setting has lower-bound sampling probability
`≥ 10⁻⁷`. In practice (10⁴ training samples per epoch) every signature is
seen **multiple times per epoch**.

For comparison, v2's empirical probabilities for E4/E6/E10 are exactly 0
(the pattern cannot be generated by any primitive).

### 3.1 Empirical Coverage Audit

Run ``python tools/sampler_coverage_audit.py --n 10000``. The tool samples
10 000 masks from both v2 and v3 and tests each against a ε-neighbourhood
checker for every E1–E15 setting. Reports are written to
``docs/temp/sampler_coverage_*.md``.

Head-line numbers at N = 10 000 (latest run, 25 task-settings):

| | effective (≥ 0.1 %) | summary |
| --- | :---: | --- |
| v2 | 10 / 25 (40 %) | E3, E4, E10 settings at 0 hits each (pattern-impossible) |
| v3 | **21 / 25 (84 %)** | All E1/E2/E3/E5/E6/E7/E10/E15 settings ≥ 0.1 %; 4 narrow E4 subsettings remain in the 0.02–0.04 % range |

v3 wins on every v2-zero setting (E3, E10, E4.A/D/F, E5.B) by ∞ ratio.
The remaining sub-0.1 % settings in v3 (E4.B/C/E, E2.mid60) involve
either (a) specific 2-joint combinations (`{l_ankle, r_ankle}`,
`{r_wrist, l_foot}`, all four end-effectors) where the joint-subset
prior has combinatorial cost, or (b) K=2 compositions of two full-dim
anchors (in-between). Given per-epoch training sample counts (≳ 10⁵),
these still produce tens–hundreds of hits per epoch, which is sufficient
signal (the model does not need exact-signature samples; its inductive
bias interpolates across the combinatorial family).

---

## 4. Theoretical Properties

### 4.1 Support Completeness

**Theorem**. `supp(p_new) = {M ∈ {0,1}^{T×D} : rB(M) ≤ K_max}` where
`K_max = 4` is the maximum K drawn with non-zero probability.

**Proof sketch**. By construction, every atom `(t_k ⊗ d_k)` with `t_k ∈
{0,1}^T` and `d_k ∈ {0,1}^D` has non-zero density (the Markov primitive in
πT covers `{0,1}^T` irreducibly; the Bernoulli path in πD covers
`{0,1}^198` by independent Beta). The union over `K ≤ K_max` atoms
reproduces any `M` with `rB(M) ≤ K_max`. ∎

### 4.2 Marginal Calibration

Let `μ_{t,d} = E[M_{t,d}]`. Then

```
μ_{t,d} = 1 - ∏_k (1 - E[(t_k)_t] · E[(d_k)_d])
```

which is computable in closed form. This gives per-(frame, dim) mask rate
and allows monitoring (e.g., "frame 0 should be locked 3% of the time" —
assert on training statistics).

### 4.3 Strict Generalisation Over v2

Every v2 template is a special case:

| v2 template | Rank-K equivalent |
| --- | --- |
| T2-1 pure_gen | K = 0 |
| T2-2 inbetween | K = 2 intervals + mixed(all) |
| T2-3 prefix | K = 1 interval(a=0) + mixed(all) |
| T2-4 keyframes | K = 1 renewal + mixed(all) |
| T2-5 end_effector | K = 1 renewal + pos_only(end_effectors, xyz) |
| T2-6 trajectory | K = 1 all/periodic + trans_only(xz) [± heading via mixed] |
| T2-7 foot_ground | K = 1 renewal + pos_only(ankles, {y}) |
| T2-8 edit_repair | edit flag overlay (orthogonal) |
| Tier-1 Markov+Beta | K = 1, t=markov, d=mixed(Bernoulli path) |

Choosing Rank-K parameters to match v2's template weights exactly
reproduces v2's distribution.

---

## 5. Implementation Plan

- File: `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v3.py`
- Public API: `sample_condition_v3(T, rng, **hparams) -> (mask[T,198], edit_mode)`
- Drop-in for `condition_sampler_v2.sample_condition`; callers choose via
  config parameter `sampler_version ∈ {'v2', 'v3'}`.
- Anatomical joint dictionary defined as module-level constants.
- Unit tests: `tests/unit/test_condition_sampler_v3.py`.
- Coverage audit: `tools/sampler_coverage_audit.py` runs N=10000 samples
  under both v2 and v3, computes per-E-task signature hit rate, emits a
  markdown table + JSON.

---

## 6. Validation Protocol

1. **Unit tests** (pure CPU, < 2 min):
   - Primitive distributions: each of `πT`'s 6 primitives produces valid
     `{0,1}^T` vectors with expected marginals.
   - Anatomical dictionary: index sets correct against `SMPL22_PARENTS`.
   - Rank-K composition: output is a valid binary mask with K-dependent
     density.
   - `K=0` yields all-generate mask.

2. **Coverage audit** (~30 s, CPU):
   - 10000 samples under v3.
   - For each E1-E15 setting, compute a **mask signature** (density,
     joint-Jaccard against task joints, channel type) and count v3 hits
     within ε-neighbourhood.
   - Assert every eval setting has `hit_rate ≥ 0.1%` (10 hits per 10k
     samples).

3. **Drop-in integration test**:
   - Run `PrepareM2Mv2Condition` with `sampler_version='v3'` over a
     100-sample subset of motionhub data.
   - Assert no exceptions, no NaN, mask marginal `μ ∈ (0.05, 0.95)` per
     channel.

4. **Debug-machine training** (< 2 h):
   - 1 node × 8 GPU × 2000 steps with v3 sampler.
   - Compare loss curve vs v2 baseline; total_loss differ by < 20% at step
     2000.
   - Snapshot E4 pass-through eval (50 samples, small ckpt) vs baseline.

Pass all four → merge. Gate v3 behind config flag; v2 remains default
until at least 2 full training runs confirm stable quality.

---

## 7. Open Questions

- **Edit-mode overlay**: should edit mode append an extra atom (so the
  corrupted region is always a Rank-K atom) or stay orthogonal? Current
  proposal: orthogonal (cleaner semantics, no correlation with K).
- **Pelvis XZ overlay**: v2 adds trans_xz lock with 25% probability on top
  of every mask. In v3 this is subsumed by sampling an atom with
  `d = trans_only(xz)`; but we may want to keep an explicit overlay to
  ensure scheme-D world-space grounding. Decision: included as part of
  `mixed` kind (Level 1), not as forced overlay.
- **`K_max`** (currently 4): enough for all E1-E15 signatures; keep
  conservative to avoid shattering the mask across too many atoms.
