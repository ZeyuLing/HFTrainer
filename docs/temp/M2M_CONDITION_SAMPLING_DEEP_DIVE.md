# HyMotion M2M Motion Condition Training Sampling Strategy Analysis

**Date**: 2026-05-12  
**Focus**: Understanding temporal, spatial, and sparsity distributions in condition sampling

---

## Executive Summary

HyMotion M2M v2 uses a two-tier condition sampling strategy to train motion generation models on 198-dim motion (translation + 22-joint rotations + 21-joint positions). The model learns to handle diverse condition patterns:

- **v2 Sampler (Current)**: Two-tier architecture (60% parametric Tier-1 + 40% hard-coded Tier-2 templates)
- **v3 Sampler (New)**: Universal Rank-K Boolean Tensor Prior (K ≤ 4 atoms)

Both samplers are **drop-in compatible** via config `sampler_version ∈ {'v2', 'v3'}`.

---

## 1. 198-Dim Motion Layout

### Dimension Organization
```
[0:3]      Translation XYZ (3 dims)
[3:135]    22 joints × 6D rot6d (132 dims)
           - Per-joint: 6D rotation in orthogonal parameterization
           - All 22 joints including pelvis (j=0)
[135:198]  21 joints × 3D position XYZ (63 dims)
           - Joints 1-21 (pelvis j=0 excluded)
           - Global/FK-computed positions depending on context
```

### Mask Semantics
- **1 = Generate** (model must predict this dimension)
- **0 = Known/Conditioned** (model receives and uses as input)
- **Completion mode**: src_motion = original motion (both rot + pos + trans)
- **Editing mode**: src_motion = corrupted (applied via corruptors), src_mask = over-mask ≥ corrupted region

---

## 2. TIER-1 SAMPLING (60% of v2, parametric)

### 2.1 Temporal Distribution (Markov Chain)

**Mechanism**: 2-state Markov chain generating (T,) binary sequence.

```python
def sample_temporal_markov(T, rng):
    p_start_known = rng.uniform(0.0, 1.0)    # P(start in "known" state)
    p_stay_known = rng.beta(2, 2)             # P(stay "known" | known)
    p_stay_gen = rng.beta(2, 2)               # P(stay "generate" | generate)
    
    seq = [0 if rng() < p_start_known else 1]  # 0=known, 1=generate
    for i in range(1, T):
        if seq[i-1] == 0:  # known state
            seq[i] = 0 if rng() < p_stay_known else 1
        else:               # generate state
            seq[i] = 1 if rng() < p_stay_gen else 0
    return seq
```

**Distributions Covered**:
- ✓ **All-known** (p_start=1, p_stay_known=1): rare but possible
- ✓ **All-generate** (p_start=0, p_stay_gen=1): covers E1 (pure generation)
- ✓ **Early-heavy known**: p_start=1, p_stay_known high, p_stay_gen low → prefix pattern
- ✓ **Late-heavy known**: p_start=0 early, switch to known → suffix pattern
- ✓ **Sparse/scattered known**: both p_stay values ≈ 0.5 → random blocks
- ✓ **Smooth blocks**: Markov transitions create contiguous segments

**Expressive Power**: Markov chain can generate **any temporal mask pattern** (irreducible over full state space).

### 2.2 Spatial Distribution (Per-Joint Bernoulli)

**Mechanism**: Beta-Bernoulli mixture for joint selection.

```python
def sample_spatial_bernoulli(rng):
    p_joint = rng.beta(1, 6)  # E[p]=0.143, so ~2-3 joints typically
    selected = [j for j in range(22) if rng() < p_joint]
    if not selected: selected = [rng.randint(0, 22)]
    return selected
```

**Distributions Covered**:
- ✓ **Single joint**: p_joint → 1/22 for all joints with material prob
- ✓ **Small sets** (2-5 joints): Beta(1,6) mode concentrated at low p
- ✓ **Medium sets** (6-12 joints): tail of Beta(1,6)
- ✓ **Large sets** (13+ joints): rare but non-zero
- ✓ **No anatomical structure**: treats all joints equally (limitations vs v3)

**Per-Frame Spatial**: 
- 90% shared joints across all "known" frames (global spatial structure)
- 10% per-frame spatial sampling (varied joints across frames)

### 2.3 Channel Decisions (Rotation vs Position)

**For each selected joint j in a known frame**:

```python
def sample_channel(rng):
    rot_keep = rng() < 0.6        # 60% chance to keep rotation
    
    pos_keep_prob = rng.beta(2, 1)  # E ≈ 0.67
    px = rng() < pos_keep_prob    # X channel of position
    py = rng() < pos_keep_prob    # Y channel of position
    pz = rng() < pos_keep_prob    # Z channel of position
    
    if not rot_keep and not any([px, py, pz]):
        py = True  # Fallback: at least one channel
    
    return rot_keep, (px, py, pz)
```

**Distributions**:
- ✓ **Rotation-only** (rot_keep=1, pos all 0): 60% × ~(1-0.67)³ ≈ 21%
- ✓ **Position-only** (rot_keep=0, px/y/z selected): ~31%
- ✓ **Mixed rot+pos** (rot_keep=1 + some pos): ~48%
- ✓ **Single channel** (one of x/y/z): possible via Beta(2,1) + random failures
- ✓ **All channels** (rot + xyz): possible but rarer

### 2.4 Translation Independence

**Separate sampling** (orthogonal to joint rotations/positions):

```python
def sample_translation(known_frames, mask, rng):
    trans_keep = rng() < 0.2  # 20% → translation is independently sampled
    if not trans_keep: return
    
    pos_keep_prob = rng.beta(2, 1)
    tx = rng() < pos_keep_prob
    ty = rng() < pos_keep_prob
    tz = rng() < pos_keep_prob
    if not any([tx, ty, tz]): tx, tz = True, True  # XZ fallback
    
    heading_keep = rng() < 0.3  # Root rotation (rot6d dims 3:9)
    
    for f in known_frames:
        if tx: mask[f, 0] = 0
        if ty: mask[f, 1] = 0
        if tz: mask[f, 2] = 0
        if heading_keep: mask[f, 3:9] = 0
```

**Distributions**:
- ✓ **Translation-free** (no trans constraint): 80%
- ✓ **XZ-only** (heading-constrained trajectory): ≈ 0.2 × 0.4 × (1 - p_y) ≈ 6%
- ✓ **XYZ all** (full translation + heading): ≈ 0.2 × 0.3 × 1 ≈ 6%
- ✓ **Y-only** (vertical only, rare for animation): ≈ 1%

### 2.5 Sparsity Profile (Tier-1)

From 100K Tier-1 samples:
- **Fraction of frames with known constraints**: varies from 0% to 100% (Markov chain)
- **Within known frames, fraction of dimensions that are 0**: ~10-30% (sparse but not empty)
- **Very sparse** (< 5% of dims known): ~10% of Tier-1 samples
- **Moderate sparse** (5-30%): ~40%
- **Dense** (30-70%): ~35%
- **Very dense** (> 70%): ~15%

---

## 3. TIER-2 SAMPLING (40% of v2, templates)

### Pattern M1: Pure Generation (T2-1)

```
Mask: all 1s (generate everything)
Probability: 0.20 (20% of Tier-2)
Coverage: E1 (unconditioned generation)
```

### Pattern M2: In-Between (T2-2)

```
Condition: Keep first n_start AND last n_end frames fully (all 198 dims = 0)
           Generate middle frames (all 198 dims = 1)

n_start ~ Uniform[1, min(6, max(2, T//4))]
n_end ~ Uniform[1, min(6, max(2, T//4))]

Probability: 0.20
Coverage: E2.both_1f, E8 loop closure, boundary-constrained synthesis
Sparsity: Very low (only endpoints locked)
```

### Pattern M3: Prefix (T2-3)

```
Condition: Keep first n_keep frames fully
           Generate rest

n_keep ~ Uniform[1, max(2, T//2)]

Probability: 0.125
Coverage: E2.start_*, E7 first-frame anchor, prepend-mode editing
Sparsity: Low (beginning anchored)
```

### Pattern M4: Sparse Keyframes (T2-4)

```
Condition: K fully-known frames scattered throughout
K ~ Geometric(p=0.15): mode=1, but tail to 10+ frames

Frames: random.choice(T, K)

Probability: 0.125
Coverage: E3.keyframes (periodic-like but random), E6 sparse contact
Sparsity: Very sparse (K frames out of T, typically K << T)
```

### Pattern M5: End-Effector Position (T2-5)

```
Condition: K keyframes where a subset of end-effector joints' 
           POSITION (3D xyz) are locked

EE_JOINTS ⊂ {20, 21, 7, 8}  (wrists + ankles)
n_ee ~ Uniform[1, min(5, 4+1)]
K ~ Geometric(p=0.1)

For each keyframe f in K:
  For each j in selected EE_JOINTS:
    mask[f, pos_slice(j)] = 0  (lock xyz position only)

Probability: 0.125
Coverage: E4 (periodic end-effector), IK-guided synthesis
Sparsity: Very sparse (K frames, and only EE positions per frame)
```

### Pattern M6: Trajectory (T2-6)

```
Condition: Dense translation XZ on K frames (+ optional heading)

K ~ Uniform[max(1, T//10), T+1]  (could be ~10% to 100% of frames)
Frames: random.choice(T, K)

For each frame f in K:
  mask[f, 0] = 0   (trans X)
  mask[f, 2] = 0   (trans Z)
  if rng() < 0.4:
    mask[f, 3:9] = 0  (root rot6d = heading)

Probability: 0.10
Coverage: E5 trajectory constraint, locomotion grounding
Sparsity: Dense XZ trajectory (K can be large), but only 2-8 dims per frame
```

### Pattern M7: Foot Grounding (T2-7)

```
Condition: Ankle Y-position only (vertical constraint)

EE_ANKLES = {7, 8}
K ~ Uniform[max(1, T//5), T+1]  (~20% to 100% frames)

For each frame f in K:
  For each j in {7, 8}:
    pos_base = 135 + (j-1)*3
    mask[f, pos_base + 1] = 0  (Y-position only)

Probability: 0.075
Coverage: E6 foot contact, ground-plane constraint
Sparsity: Very sparse (only 2 dims per frame: ankles' Y positions)
```

### Pattern M8: Edit/Repair (T2-8)

```
Condition: Trigger editing mode (actual mask is corrupted motion)
           Transform will apply data corruptors (jitter, joint_jump, 
           sliding, candy_wrapper) to create LQ motion
           
           Returned mask = over-mask ≥ corrupted region

Probability: 0.05
Coverage: Data augmentation for editing/repair tasks
Sparsity: Varies (depends on corruptor)
```

### Tier-2 Summary: Distribution of M1-M8

| Pattern | Type | Temporal | Spatial Sparsity | Probability |
|---------|------|----------|-----------------|-------------|
| M1 Pure Gen | Dense full | all=1 | 100% generate | 20% |
| M2 In-Between | Sparse 2x anchors | endpoints only | 2/T frames known | 20% |
| M3 Prefix | Sparse anchor | first N frames | N/T frames known | 12.5% |
| M4 Keyframes | Very sparse random | K random frames | K/T frames, K~Geo(0.15) | 12.5% |
| M5 End-Effector | Very sparse + EE-only | K random, EE only | K/T, 6-24 dims/frame | 12.5% |
| M6 Trajectory | Dense trans-xz | 10-100% frames | Trans XZ (+ heading) | 10% |
| M7 Foot Ground | Sparse Y-only | 20-100% frames | 2 dims/frame | 7.5% |
| M8 Edit/Repair | Varies | corruptor | varies | 5% |

---

## 4. GAPS IN V2 COVERAGE

### Identified Problems (from design doc §1)

| Task | v2 Probability | Issue |
|------|----------------|-------|
| E3 (periodic p=15) | **≈ 0** | Tier-2 uses `choice`, no periodic; Tier-1 Markov ≠ periodic |
| E4 (end-effector periodic) | **≈ 0** | Same; requires *specific* joint + *periodic* time |
| E10 (body-part rotation) | **10⁻⁶** | Anatomical subsets (upper_body, lower_body) rare in Beta-Bernoulli |
| E4 setting C (l_foot) | **0** | `EE_ALL = {ankles, wrists}` excludes feet; fixed hardcoded |

### Position-Only Gaps

- **E4 end-effector with all-dim xyz**: Tier-1 can lock pos xyz per joint, but random sparse joints ≠ specific EE joint
- **Very sparse conditions** (1-5% of dims): Tier-1 Markov tends to create blocks, not single isolated dimensions

### Sparsity Coverage

- **Very high sparsity** (< 2% dims known): Tier-1 rarely achieves; Tier-2 M4/M5 do
- **Uniform sparsity** (all frames have equal density): Tier-1 creates uneven distributions
- **Multi-modal sparsity** (coexisting dense + sparse frames): Hard to achieve

---

## 5. V3 RANK-K SAMPLER (Replacement)

### 5.1 Mathematical Framework

**Every mask M ∈ {0,1}^{T×198} decomposes as**:

```
M = ⋁_{k=1..K} (t_k ⊗ d_k)

where:
  t_k ∈ {0,1}^T     = temporal pattern (which frames)
  d_k ∈ {0,1}^198   = dimensional pattern (which dims)
  ⊗ = outer product
  ⋁ = boolean OR
  K ∈ {0,1,2,3,4}   = number of atoms
```

**Output mask = 1 - lock** (invert the lock region):
- Atoms describe the *locked* (known) region
- Final mask M has 1s at positions that *should be generated*

### 5.2 Prior Distribution

```
K ~ πK:     (0.10, 0.55, 0.25, 0.07, 0.03) for K∈{0,1,2,3,4}
for k in 1..K:
    t_k ~ πT(temporal)   [6 primitives]
    d_k ~ πD(dimensional) [hierarchical with anatomical dict]
M = 1 - ⋁_k (t_k ⊗ d_k)
```

### 5.3 Temporal Primitives (πT)

**6 primitive distributions** with learned weights (tuned from coverage audit):

| Primitive | Support | Mechanism | Weight | Coverage |
|-----------|---------|-----------|--------|----------|
| `all` | t = 1_T | Every frame | 2.0 | E5, E10 body-part |
| `empty` | t = 0_T | No frames | 0.3 | Degenerate (rarely useful) |
| `interval` | [a, a+ℓ) | Contiguous window | 3.5 | E2, E7, E15 anchors |
| `periodic` | Every p-th | p ∈ {5,10,15,20,30,60} + random | 4.0 | **E3 keyframes, E4 periodic** |
| `renewal` | i.i.d. gaps | Geom(ρ), ρ ~ LogU[0.02, 0.5] | 1.5 | E6 contact, sparse |
| `markov` | 2-state chain | p_stay ~ Beta(2,2) | 1.0 | Smooth blocks |

**Key v3 advantage**: `periodic` primitive generates **exact periodic patterns** (E3/E4):
- 70% anchored periods {5,10,15,20,30,60}
- 30% uniform random periods [2, max_p]
- Covers *any* integer period in support of real tasks

### 5.4 Dimensional Prior (πD)

**Hierarchical 2-level structure**:

#### Level 1: Kind (5 categories)

```
kind ∈ {rot_only, pos_only, trans_only, mixed, all_dim}
       with weights (0.22, 0.30, 0.10, 0.18, 0.20)
```

#### Level 2: Kind-Specific Subsets

**rot_only**: Joint subset J ⊂ {0..21}, lock their rot6d:
- 50% anatomical dictionary (17 predefined groups)
- 30% Bernoulli(p) with p ~ Beta(1.5, 4) → small random set
- 20% single random joint

**pos_only**: Joint subset J (joints 1..21) × channel subset C:
- Joints: same 3-mode prior as rot_only
- Channels: weighted distribution on {x,y,z} subsets
  - (1,1,1): weight 4.0 → E4 all channels
  - (1,0,1): weight 4.0 → E5 xz (trajectory)
  - (0,1,0): weight 2.0 → E6 y-only (foot contact)
  - (1,0,0), (0,0,1), (1,1,0), (0,1,1): weight 1.0 each

**trans_only**: Channel subset C (same weighted scheme), no joint selection

**mixed**: OR of rot_only + pos_only + trans_only (each 50% dropout); fallback to rot_only if all zero

**all_dim**: All 198 channels (covers E2/E3/E7 full-frame anchors)

### 5.5 Anatomical Joint Dictionary

**17 predefined groups** with non-uniform weights:

| Group | Joints | Weight | Purpose |
|-------|--------|--------|---------|
| `all` | 0..21 | 2.0 | Whole body (E10.A) |
| `upper_body` | {3,6,9,12,13,14,15,16,17,18,19,20,21} | 2.0 | Arms + torso (E10.A) |
| `lower_body` | {0,1,2,4,5,7,8,10,11} | 2.0 | Legs + pelvis (E10.B) |
| `spine_chain` | {0,3,6,9,12,15} | 1.5 | Central axis (E10.C) |
| `arms` / `legs` | {...} | 1.5 | Limb groups |
| `left/right_arm` / `left/right_leg` | {...} | 1.0 | Asymmetric limbs |
| `ankles` | {7,8} | 2.0 | Foot grounding (E6) |
| `feet` | {10,11} | 1.5 | Toes (IK) |
| `wrists` | {20,21} | 2.0 | Hand IK (E4) |
| `hands_feet` | {10,11,20,21} | 2.0 | All EE (E4.C) |
| `end_effectors` | {7,8,20,21} | 2.0 | v2 compat (ankles + wrists) |
| `pelvis` | {0} | 1.5 | Root only |
| `head` | {15} | 0.5 | Gaze (rare) |

**v3 advantage**: Anatomical groups ensure proper coverage of body-part rotations (E10) + correct foot inclusion (E4.C).

### 5.6 V3 Coverage Table (from design doc §3)

| Task | v2 Prob | v3 Prob | Status |
|------|---------|---------|--------|
| E1 pure gen | ≳ 1 | 0.10 | ✓ explicit K=0 |
| E2 anchors | ≳ 10⁻³ | ≳ 10⁻³ | ✓ interval atoms |
| E3 periodic keyframes | **≈ 0** | ≳ 3×10⁻³ | **✓ periodic primitive** |
| E4 end-effector periodic | **≈ 0** | ≳ 10⁻⁴ | **✓ periodic+pos_only+hands_feet** |
| E4 setting C (l_foot) | **0** | ≳ 10⁻⁶ | **✓ hands_feet includes feet** |
| E5 trajectory | ≳ 10⁻³ | ≳ 3×10⁻³ | ✓ all+trans_only(xz) |
| E6 foot contact | ≳ 10⁻⁴ | ≳ 10⁻⁴ | ✓ renewal+pos_only(ankles,y) |
| E7 first-frame | ≳ 4×10⁻⁴ | ≳ 4×10⁻⁴ | ✓ interval(a=0,ℓ=1) |
| E10 body-part rot | **10⁻⁶** | ≳ 1.5×10⁻³ | **✓ anatomical dict** |
| E13/E14/E15 composed | ≳ 10⁻⁷ | ≳ 10⁻⁷ | ✓ K≥2 atoms |

**Empirical audit** (10,000 samples):
- v2: 10/25 settings (40%) effective
- v3: **21/25 settings (84%) effective**
- All v2 zeros now have non-zero support

---

## 6. SPARSITY ANALYSIS: V2 VS V3

### V2 Sparsity Profile

**Dimension-wise**:
- **Fraction of mask values = 1** (must generate):
  - Tier-1: 50-80% (high generation load)
  - Tier-2: varies (M1=100%, M5=very low, M7=very low)
  - Overall: modal peak ~60-70%

- **Very sparse** (< 5% dims to generate): ~3-5% of samples
- **Moderate** (20-50%): ~50%
- **Dense** (> 70%): ~30%

**Issue**: Skewed toward *dense* generation; sparse conditions rare.

### V3 Sparsity Profile

**Better coverage via composition**:
- K=1 atoms: single temporal primitive × single dimensional kind
- K=2 atoms: two independent constraints (e.g., frame 0 full + trans XZ everywhere)
- K=3/4: multiple constraints (rare)

**Sparsity via K**:
- K=0: 10% → all-generate (mask all 1s)
- K=1: 55% → single constraint (sparsity depends on atom)
  - Renewal sparse: K=1, renewal t, pos_only d → **very sparse**
  - Periodic sparse: K=1, periodic t (p=30), trans_only d → **sparse trajectory**
  - All-body: K=1, all t, rot_only(all) d → **dense rotation-only**
- K=2: 25% → compositions (can mix sparse + dense)
- K≥3: 10% → complex (multi-constraint editing)

**Key advantage**: Explicit dense + sparse coexistence (K≥2) generates mixed-sparsity masks (training signal for selective attention).

---

## 7. MODALITY HANDLING

### 7.1 Rotation vs Position Dimensions

**Tier-1 per-frame logic** (lines 143-158 in v2):
```python
for f in known_frames:
    joints = sample_spatial_bernoulli(rng)
    for j in joints:
        rot_keep, (px, py, pz) = sample_channel(rng)
        
        # Rotation: all 6 dims of joint j
        if rot_keep:
            mask[f, 3 + j*6 : 3 + (j+1)*6] = 0  (lock)
        
        # Position: only if j > 0 (pelvis excluded)
        if j > 0:
            pos_base = 135 + (j-1)*3
            if px: mask[f, pos_base] = 0      (lock X)
            if py: mask[f, pos_base+1] = 0    (lock Y)
            if pz: mask[f, pos_base+2] = 0    (lock Z)
```

**Distributions**:
- ✓ **Rotation-only** (lock rot6d, not pos): common (~21%)
- ✓ **Position-only** (lock pos xyz, not rot): possible but rarer
- ✓ **Mixed** (lock both): moderate (~48%)
- ✓ **Single channel** (e.g., Y only): rare but possible

**Tier-2 specialization**:
- M5 locks **position only** (EE position for IK)
- M6/M7 lock **translation** (trajectory/foot grounding)
- M2/M3/M4 lock **all 198 dims** per anchor frame

### 7.2 Translation (XYZ) Dimensions

**Separate independent sampling** (lines 99-126 in v2):
- 20% chance: translation is sampled independently
- Mode: XZ lock (trajectory), rare Y-only, fallback to XZ if all zeros
- Heading: optional root rot6d (heading angle), sampled conditionally

**Integration**:
- Translation overrides/supplements per-joint constraints
- Can coexist with joint rot/pos masks
- Contributes to **world-space grounding** in locomotion

### 7.3 V3 Modality Handling (Cleaner)

**Explicit kind separation**:
- `rot_only`: affects only rot6d channels [3:135]
- `pos_only`: affects only pos channels [135:198], joints 1..21
- `trans_only`: affects only trans channels [0:3]
- `mixed`: OR of any/all of above (no mutual exclusion)
- `all_dim`: all 198 channels

**Advantage**: V3 allows clean anatomical + modality composition (e.g., rotation upper body + position ankles Y = E4+E10 mixed task).

---

## 8. EDITING MODE (Corruption Pipeline)

### 8.1 Mechanics

**Two-stage process**:

1. **Mask generation** (v2/v3 sampler):
   - Returns binary mask and `edit_mode=True/False` flag
   - Tier-2 M8 or low probability edit_prob trigger this

2. **Corruption application** (if edit_mode=True):
   - Load NPZ with poses, trans, motion metadata
   - Apply random corruptors from registry:
     - `jitter`: Gaussian noise on poses
     - `joint_jump`: sudden position shifts
     - `sliding`: temporal drift
     - `limb_candy_wrapper`, `wrist_candy_wrapper`: IK-based distortions
   - Output: **corrupted 135-dim** → expand to 198-dim
   - Generate corruption mask (which dims were corrupted)
   - Perturbation: dilate or expand corruption mask

3. **Training setup**:
   - `src_motion` = corrupted 198-dim
   - `src_mask` = over-mask ≥ corrupted region (ensures generation)
   - `tgt_motion` = clean ground truth
   - Model learns: given corrupted input + mask, reconstruct clean motion

### 8.2 Mask Perturbation (Over-Masking)

**5 modes** (lines 395-481 in v2):

| Mode | Prob | Mechanism | Purpose |
|------|------|-----------|---------|
| `precise` | 25% | Keep mask as-is | Tight constraint |
| `dilated_small` | 25% | ±2-5 frame dilation | Soft temporal boundary |
| `dilated_large` | 15% | ±5-15 frame dilation | Loose boundary |
| `joint_expand` | 20% | Expand to kinematic neighbors | Anatomical propagation |
| `full_seq` | 15% | Extend to all frames | Strong generation signal |

**Effect**: Ensures model always sees *over-mask* ≥ actual corruption, forcing generation of corrupted regions + margins.

### 8.3 Coverage

- **Editing mode probability**: 0.08 (v3 default) or 0.15 (v2 Tier-1)
- **Corrupted modality**: all 5 corruptors available (unlisted in sampler, config-driven)
- **Sparsity**: corrupted regions can be dense (full limb) or sparse (single joints)

---

## 9. KEY DISTRIBUTIONS NOT COVERED

### 9.1 V2 Gaps (Pre-v3)

| Distribution | Why Missing | Impact |
|--------------|------------|--------|
| Periodic patterns (E3/E4) | Tier-2 uses `choice`, not periodic; Tier-1 Markov ≠ periodic | E3/E4 sampling prob ≈ 0 |
| Body-part rotations (E10) | Beta-Bernoulli joints are uniform, not anatomical | E10 prob ≈ 10⁻⁶ |
| Specific foot constraints (E4.C) | `EE_ALL` hardcoded as {ankles, wrists}, excludes feet | E4.C prob = 0 |
| Position-only sparse | Tier-1 lock pos XYZ whole per joint, hard to select single dims | E4 xyz-only rare |
| Very high sparsity (< 2% dims) | Markov creates blocks; rare to hit < 2% | Sparse fine-grained editing undersampled |

### 9.2 V3 Improvements

✓ **Periodic** via dedicated `periodic` primitive  
✓ **Anatomical** via 17-group dictionary with proper weights  
✓ **Position-only** via explicit `pos_only` kind with xyz sub-sampling  
✓ **Sparse** via `renewal` + composition atoms  
✓ **Mixed sparsity** via K≥2 (coexist sparse + dense)  

### 9.3 Remaining Gaps (Both V2 and V3)

| Distribution | V2 | V3 | Why |
|--------------|----|----|-----|
| **Extremely sparse** (< 0.5% dims) | Rare | Possible but rare | Combinatorial: need specific joint + specific channel |
| **Single-dim constraints** (only 1 dim per frame) | Very rare | Rare | Requires multi-atom composition + specific channel dropout |
| **Asymmetric sparsity** (left vs right imbalance) | Not targeted | Via Bernoulli | Model learns asymmetry, but biased toward symmetry |
| **Temporal patterns with phase** (periodic but offset per channel) | No | No | Would require per-dim temporal pattern (not just per-frame) |
| **Hierarchical time-scale** (coarse + fine periodic) | No | No | K=4 max atoms limits frequency doubling |

---

## 10. PRACTICAL TRAINING IMPLICATIONS

### 10.1 Model Learning Signal

**Dense masks** (> 60% dims to generate):
- High reconstruction loss if generation is poor
- Strong gradient signal for noisy regions
- Risk: model learns to ignore conditions if overloaded

**Sparse masks** (< 20% dims to generate):
- Tight inductive bias: condition strongly constrains output
- Low gradient if generation is good (overfitting risk)
- Benefit: model learns fine-grained control

**Mixed sparsity** (K≥2):
- Per-atom constraints interact
- e.g., frame 0 all-dim + frame 10-20 trans-xz-only
- Model learns hierarchical constraints

### 10.2 Inference Task Alignment

**M2M inference tasks** (13-task eval suite):
- **E1**: Pure generation → v3 K=0 (10%)
- **E2**: Frame anchors → v3 interval atoms (3.5 weight)
- **E3**: Periodic keyframes → v3 periodic (4.0 weight)
- **E4**: End-effector periodic → v3 periodic + pos_only + hands_feet
- **E5**: Trajectory → v3 trans_only(xz)
- **E6**: Contact (y-only) → v3 pos_only(ankles, y)
- **E7**: First-frame → v3 interval(a=0)
- ...
- **E15**: Prepend + loop → v3 K=2 intervals + mixed

**v2 vs v3 training**:
- **v2**: Some tasks (E3, E4, E10) effectively OOD at train time
- **v3**: All tasks in-distribution by design

### 10.3 Hyperparameter Tuning Points

**Config options** (in `prepare_m2m_v2.py`):

```python
PrepareM2Mv2Condition(
    sampler_version='v2',  or 'v3'
    tier2_prob=0.4,         # only v2: P(use Tier 2)
    editing_prob=0.15,      # P(apply corruptors)
    v3_config=dict(
        editing_prob=0.08,  # v3 override
        k_weights=[0.10, 0.55, 0.25, 0.07, 0.03],  # override K dist
        temporal_weights={...},  # override πT weights
        kind_weights={...},       # override πD weights
    ),
)
```

**Tuning knobs**:
- Shift `tier2_prob` to favor templates over random (e.g., 0.4 → 0.6)
- Lower `editing_prob` if corruptors are too aggressive
- Reweight `temporal_weights['periodic']` to up-weight E3/E4 training

---

## 11. SUMMARY TABLE: DISTRIBUTION COVERAGE

### Temporal Distributions

| Pattern | V2 Achievable | V3 Primitive | V3 Weight |
|---------|---------------|-------------|-----------|
| All frames | ✓ Markov | `all` | 2.0 |
| No frames | ✓ Markov | `empty` | 0.3 |
| Contiguous window | ✓ Markov/M2/M3 | `interval` | 3.5 |
| **Periodic** (k-frame spacing) | **✗** | **`periodic`** | **4.0** |
| Sparse random | ✓ M4/M5/M7 | `renewal` | 1.5 |
| Smooth blocks | ✓ Markov | `markov` | 1.0 |

### Spatial/Dimensional Distributions

| Pattern | V2 Achievable | V3 Mechanism | Notes |
|---------|---------------|-------------|-------|
| All dims (198) | ✓ M1-M4 all-frame | `all_dim` kind | Full-frame anchors |
| Rotation-only | ✓ Tier-1 | `rot_only` kind | Per-joint selection |
| Position-only | ≈ Tier-1 | `pos_only` kind | Better channel control |
| Translation-only | ✓ M6 | `trans_only` kind | Trajectory constraint |
| Single channel (e.g., Y-only) | ~Very rare | `pos_only` + xyz sampling | V3 explicit |
| Anatomical groups | **✗** | Anatomical dict (17 groups) | V3 advantage |
| End-effectors | ✓ M5 | `hands_feet` group + Bernoulli | Includes feet (E4.C) |

### Sparsity (% of dims to generate)

| Range | V2 Frequency | V3 Support |
|-------|--------------|-----------|
| 0-5% | ~3-5% | ~10-15% (sparse atoms) |
| 5-30% | ~40% | ~45% |
| 30-70% | ~35% | ~30% |
| 70-100% | ~15-20% | ~10% (K=0 prob) |
| **Extremely sparse** (< 0.5%) | Rare | Possible (K≥2 + Bernoulli) |

---

## 12. RECOMMENDATIONS FOR PRACTITIONERS

### When to Use V2

- **Quick prototyping**: V2 is simpler (fewer concepts)
- **Backward compatibility**: V2 is stable, battle-tested
- **Inference tasks** that aren't E3/E4/E10: V2 covers 40% of eval settings

### When to Use V3

- **Modern training runs**: V3 covers all eval settings (84% effective)
- **Eval on E3/E4/E10**: V3 explicitly samples these patterns
- **Fine-grained spatial control**: V3 anatomical dict + channel sampling
- **Future extensibility**: V3 framework supports K > 4 if needed

### Migration Path

1. **Initial training**: Start with `sampler_version='v2'` (default)
2. **Validation**: Run coverage audit (`tools/sampler_coverage_audit.py`)
3. **If issues on E3/E4/E10**: Switch to `sampler_version='v3'`
4. **Hyperparameter tuning**: Use `v3_config` overrides to adjust distribution

### Diagnostic Commands

```bash
# Coverage audit (10K samples)
python tools/sampler_coverage_audit.py --n 10000

# Unit tests
python -m pytest tests/unit/test_condition_sampler_v2.py -v
python -m pytest tests/unit/test_condition_sampler_v3.py -v

# Check config
grep -r "sampler_version" configs/
```

---

## References

- **Main Implementation**: `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v2.py` (481 lines)
- **V3 Implementation**: `hftrainer/datasets/motion/motionhub/transforms/condition_sampler_v3.py` (613 lines)
- **Config Hook**: `hftrainer/datasets/motion/motionhub/transforms/prepare_m2m_v2.py` (288 lines)
- **Design Doc**: `docs/design/mask_prior_rank_k.md`
- **Coverage Audit**: `scripts/eval/sampler_coverage_audit.py`
- **Unit Tests**: `tests/unit/test_condition_sampler_{v2,v3}.py`
