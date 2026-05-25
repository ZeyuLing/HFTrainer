# HyMotion M2M v2 Evaluation Metrics & Baseline Comparison

**Generated**: May 22, 2026  
**Source**: Repository analysis + `hftrainer/evaluation/motion/` + Baseline CLAUDE.md files

---

## Quick Summary

### Our Evaluation Framework (M2M v2)

**Location**: `hftrainer/evaluation/motion/`
- **Main metric file**: `m2m_eval_metrics.py` (838 lines)
- **Task definitions**: `m2m_eval_tasks.py` (1000+ lines)  
- **Physical metrics**: `phys_metrics.py` (body model integration)
- **Evaluation runner**: `scripts/eval/eval_m2m_v2_all_tasks.py` (200+ lines)

**Motion representation**: 135-dim SMPL-22 (3D abs_transl + 22×6D rot_6d)
- Supports tasks: E1-E16 (text-to-motion + 15 completion/editing variants)
- Per-task settings: A-F (usually 6 variants per task)
- Dataset: 220 held-out test motions with rewritten captions

---

## Part 1: Available Metrics in Our Codebase

### Core Metrics (in `m2m_eval_metrics.py`)

#### Position-Based Metrics

| Metric | Formula | Unit | Typical Values | When Used |
|--------|---------|------|-----------------|-----------|
| **MPJPE (mean)** | Mean(√(Σ_j \|\|pred_j - gt_j\|\|²)) | meters | 0.05-0.15m | All tasks with GT |
| **MPJPE (masked)** | Same, but only on generated frames | meters | 0.05-0.20m | MIB, in-betweening |
| **MPJPE (unmasked)** | Same, on all frames | meters | 0.02-0.10m | Imputation tasks |
| **Jitter (positions)** | Mean(\\|d³x/dt³\\|) across all joints | m/s³ | 0.1-0.5 | All tasks |
| **Jitter (raw 135D)** | Mean(\\|d³m/dt³\\|) on 135-dim directly | unitless | 0.01-0.05 | All tasks |
| **Trajectory ADE** | Average Euclidean distance on root XZ | meters | 0.01-0.20m | E5 (trajectory) |
| **Trajectory FDE** | Final euclidean distance on root XZ | meters | 0.02-0.50m | E5 |

#### Rotation-Related Metrics

| Metric | Formula | Unit | When Used |
|--------|---------|------|-----------|
| **Heading error** | Angular difference between pred/gt pelvis forward direction | degrees | Trajectory, spatial tasks |
| **FK consistency** | Consistency between rotation FK and position channels (for 198D reps) | meters | Reps with explicit positions |

#### End-Effector Metrics (E4)

```python
# For each constraint frame/joint pair:
error = ||FK(pred_rot)[joint] - target_position||

Reported statistics:
  - ee_error_mean: Average
  - ee_error_max: Worst case
  - ee_error_p50: Median (robust)
  - ee_error_p95: 95th percentile (tail failures)
  - ee_error_std: Spread
  - ee_hit_rate_2cm / _5cm / _10cm: Fraction within distance threshold
```

#### Ground/Physical Metrics

| Metric | Calculation | Typical Values | Purpose |
|--------|-------------|-----------------|---------|
| **Foot penetration** | Max(ground_y - foot_y, 0) | 0.00-0.10m | Detect clipping into floor |
| **Foot float** | Height above ground during contact | 0.00-0.15m | Detect unnatural floating |
| **Foot skating ratio** | Ratio of contact frames with XZ velocity > 0.5 cm/frame | 0.00-0.30 | Detect sliding motion |
| **Foot avg_skate** | Average velocity during skating frames | 0.02-0.10 m/s | Magnitude of sliding |

#### Boundary Smoothness (E14/E15 stitching)

| Metric | Definition |
|--------|-----------|
| **Boundary accel jump** | Acceleration discontinuity at mask transition |
| **Boundary smoothness** | Gradient of acceleration at splice points |

#### Loop Continuity (E8D)

| Metric | Definition | Unit |
|--------|-----------|------|
| **Loop position error** | ||first_frame - last_frame|| across all joints | meters |
| **Loop velocity error** | ||vel_first - vel_last|| | meters/second |

### Metric Aggregation

```python
# Per-sample: Dict[metric_name -> float]
# Per-dataset: Dict[metric_name -> {mean, std, median, min, max, count}]
```

**Location**: `compute_all_metrics()` (comprehensive runner, ~80 lines)

---

## Part 2: Standard Task Metrics (E1-E16)

Each task has a `default_metrics` list. Here's the breakdown:

### E1: Text-to-Motion (Unconditional)
```
default_metrics=['jitter_pos', 'foot_skating_ratio']
# Pure generation → only measure motion quality, not reconstruction error
```

### E2: Motion In-Betweening (6 settings A-F)
```
E2A (start_1f): ['mpjpe_masked', 'mpjpe_unmasked', 'boundary_accel_jump']
E2B (end_1f): ['mpjpe_masked', 'mpjpe_unmasked', 'boundary_accel_jump']
E2C (both_1f): ['mpjpe_masked', 'mpjpe_unmasked', 'boundary_accel_jump']
E2D (pre20): Same
E2E (post20): Same
E2F (mid60): Same
# All measure: reconstruction error on masked region + boundary smoothness
```

### E3: Keyframe Infilling (6 settings A-F)
```
Settings: every_5f, every_10f, every_15f, every_30f (standard), every_60f, adaptive
default_metrics=['mpjpe_masked', 'mpjpe_unmasked', 'jitter_pos', 'foot_skating_ratio']
# Dense → sparser keyframes with increasing difficulty
```

### E4: End-Effector Position Constraint (6 settings A-F)
```
E4A-E4F: Varying EE joints and frame density
default_metrics=['ee_error_mean', 'ee_error_max', 'ee_error_p50', 'ee_error_p95',
                 'ee_hit_rate_5cm', 'ee_hit_rate_10cm', 'jitter_pos']
# Spatial constraint precision: how close do we get to target position?
```

### E5: Trajectory Following (6 settings A-F)
```
Modes: dense, sparse trajectory waypoints with varying intervals
default_metrics=['trajectory_ade', 'trajectory_fde', 'jitter_pos', 'foot_skating_ratio']
# Root motion following: path adherence + smoothness
```

### E8D: Temporal In-filling (transition task)
```
default_metrics=['mpjpe_masked', 'boundary_accel_jump', 'jitter_pos', 'foot_skating_ratio']
```

### E14 / E15: Complex Stitching Tasks
```
E14 (multi-segment stitching): 
  ['mpjpe_unmasked', 'jitter_pos', 'foot_skating_ratio', 
   'segment_boundary_smoothness']

E15 (loop stitching):
  ['loop_position_error', 'loop_velocity_error', 'jitter_pos']
```

### E16: Tail Prediction
```
default_metrics=['mpjpe_masked', 'jitter_pos', 'foot_skating_ratio']
```

---

## Part 3: What We DON'T Compute (vs Baselines)

### Missing: FID (Fréchet Inception Distance)
- **Why**: Requires pre-trained feature extractor (MotionCLIP / HyMotion text encoder)
- **What it measures**: Distribution distance between generated and ground truth motions
- **Used by**: KIMODO, UMO, MotionLab, SOAR (all report FID)
- **Our alternative**: MPJPE captures reconstruction accuracy; jitter captures smoothness

### Missing: Diversity Metrics
- **Diversity (FID-based)**: Batch-level diversity using feature embeddings
- **MMDist**: Minimum matching distance between generated samples
- **Used by**: UMO (R@3 = 100%), MotionLab
- **Why missing**: Requires reference feature space + expensive batch computation

### Missing: R-Precision / Matching Scores
- **R@k**: Top-k matching accuracy (text-motion alignment)
- **Used by**: T2M papers (HumanML3D benchmark)
- **Why missing**: M2M is completion-focused, not text-matching-focused

### Missing: Ground Contact Explicitly
- **Foot contact classification**: Is foot touching ground (binary prediction)?
- **Contact accuracy**: Compare predicted vs GT contact pattern
- **Used by**: KIMODO (explicitly models 4-dim foot contact in output)
- **Our limitation**: 135-dim has no contact prediction; use velocity threshold heuristic

---

## Part 4: Baseline Comparison Matrix

### KIMODO (NVIDIA, 2024)

**Representation**: 333-dim (27 joints)
- smooth_root_pos (3D)
- global_root_heading (2D)
- local_joints_positions (27×3)
- global_rot_data (27×6, world-frame)
- velocities (27×3)
- **foot_contacts (4D explicit)**

**Metrics KIMODO reports**:
```
Position-based:
  - Position error (mm)
  - MPJPE (meters) — comparable to ours
  - Trajectory error

Rotation-based:
  - Rotation error (degrees)
  - Heading error

Physical:
  - Foot skating (cm/s) — comparable
  - Penetration (cm)
  - Float height (cm)

Constraint-specific:
  - EE error at constraint frames
  - Contact accuracy (if foot contact constraints)

Difference from ours:
  - Reports in mm/cm (we use meters)
  - Has explicit foot contact accuracy
  - Reports rotation error (6D prediction accuracy)
  - No jitter metric (we emphasize it)
```

**Key insight**: KIMODO's smooth root + explicit contact make it better for foot physics; our jitter-focused metrics better for temporal smoothness.

---

### UMO (Brown/MIT/Meta, 2024)

**Representation**: 201-dim (22 joints, SMPL compatible)
- global_transl (3D)
- root_orient (6D)
- 21_local_rot (126D)
- 22_local_pos (66D)

**Metrics UMO reports**:
```
For Temporal Inpainting:
  - MPJPE (mm) on in-between frames
  - [P]-MPJPE (mm): error on preserve frames (should be ~0)
  - Diversity: FID, R@3
  - Trajectory smoothness

Editing-specific:
  - R@3(batch): Top-3 editing accuracy (qualitative → quantitative)
  - Editing success rate (binary: did model follow instruction?)

Reaction-specific:
  - FID: Distribution matching
  - Trajectory correlation: How well does reaction follow lead?
```

**Comparison with ours**:
- UMO [P]-MPJPE ≈ 0.95mm (perfect Preserve); ours can have larger errors without post-processing
- UMO uses FID heavily; we focus on task-specific metrics
- UMO measures editing success (binary); we don't explicitly label editing tasks

---

### MotionLab (SUTD/Lightspeed, 2025)

**Representation**: 263-dim (same as HumanML3D)

**Metrics MotionLab reports**:
```
Generation quality:
  - FID (vs HumanML3D test set)
  - Diversity (batch-level)
  - Precision / Recall

Trajectory control:
  - Trajectory error (meters) — **lowest reported: 0.0286m**
  - Trajectory smoothness

Editing-specific:
  - Style recognition accuracy (SRA)
  - Text alignment score

Physical:
  - Jitter
  - Foot skating ratio
  - Penetration / float
```

**Comparison with ours**:
- MotionLab has trajectory error metrics we could adopt (0.03m is excellent)
- MotionLab computes FID; we don't
- MotionLab has style transfer metrics; we don't support style

---

## Part 5: How to Add Missing Metrics

### Option A: Add FID Computation

**Prerequisites**:
1. Train/load a motion feature extractor (e.g., MotionCLIP, HyMotion text encoder)
2. Compute embeddings for all GT motions → compute Gaussian parameters (μ, Σ)
3. For each generated motion: compute embedding → FID distance

**Implementation path**:
```python
# In m2m_eval_metrics.py:
def compute_fid(pred_motions_emb, gt_motions_emb):
    """Fréchet Inception Distance between motion embeddings."""
    mu_pred = np.mean(pred_motions_emb, axis=0)
    mu_gt = np.mean(gt_motions_emb, axis=0)
    sigma_pred = np.cov(pred_motions_emb.T)
    sigma_gt = np.cov(gt_motions_emb.T)
    # FID = ||mu_pred - mu_gt||² + Tr(sigma_pred + sigma_gt - 2*(sigma_pred @ sigma_gt)^0.5)
    ...
```

**Effort**: Medium (need feature extractor, batch computation)

### Option B: Add Diversity Metrics

**Approach**: Batch-level statistics on generated samples

```python
def compute_diversity(batch_motions_emb):
    """Pairwise distance statistics in embedding space."""
    pairwise_dist = np.linalg.norm(batch_motions_emb[:, None] - batch_motions_emb[None, :], axis=-1)
    diversity = np.mean(pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)])
    return diversity
```

**Effort**: Low (once FID infra exists)

### Option C: Add Contact Accuracy

**For 135D representation** (no explicit contact):
```python
def predict_foot_contact(positions):
    """Predict contact from foot velocity threshold."""
    foot_vel = np.linalg.norm(np.diff(positions[:, [7,8,10,11]], axis=0), axis=-1)
    contact = foot_vel < 0.01  # m/frame threshold
    return contact

def compute_contact_accuracy(pred_pos, gt_pos):
    """Compare predicted vs GT foot contact."""
    pred_contact = predict_foot_contact(pred_pos)
    gt_contact = predict_foot_contact(gt_pos)
    accuracy = (pred_contact == gt_contact).mean()
    return accuracy
```

**Effort**: Low (heuristic-based)

---

## Part 6: Standard Metric Sets by Paper Category

### For T2M (Text-to-Motion) Papers
**Essential metrics**:
- FID (distribution distance)
- MPJPE (reconstruction quality)
- Diversity (batch-level variation)
- Jitter (temporal smoothness)

### For Motion Completion Papers (Our Category)
**Essential metrics**:
- MPJPE (masked): Reconstruction error on completed regions ← **PRIMARY**
- MPJPE (unmasked): Overall error
- Jitter: Temporal smoothness ← **PRIMARY**
- Boundary accel jump: Splice quality (at mask transitions)
- Foot skating: Physical realism

### For Spatial Control Papers
**Essential metrics**:
- Constraint error (e.g., EE position error)
- Constraint satisfaction rate (e.g., hit_rate @ threshold)
- MPJPE (overall quality)
- Jitter (don't sacrifice smoothness for constraint satisfaction)

### For Trajectory Following
**Essential metrics**:
- Trajectory ADE / FDE: Path adherence
- MPJPE: Quality
- Jitter: Smoothness

---

## Part 7: Metric Values for Reference

### Strong Performance Ranges (from literature)

| Metric | Good | Excellent | Source |
|--------|------|-----------|--------|
| MPJPE (generation) | < 0.10m | < 0.05m | T2M baseline |
| MPJPE (in-betweening) | < 0.10m | < 0.05m | MIB papers |
| Jitter | < 0.3 m/s³ | < 0.1 m/s³ | Motion quality |
| Foot skating | < 0.15 m/s | < 0.05 m/s | Physical realism |
| FID | < 6.0 | < 4.0 | HumanML3D |
| Trajectory ADE | < 0.15m | < 0.05m | MotionLab |
| EE error | < 0.10m | < 0.05m | Spatial control |

---

## Part 8: Quick Reference: Which Baseline Uses Which Metrics

| Metric | KIMODO | UMO | MotionLab | M2M (Ours) |
|--------|--------|-----|-----------|-----------|
| MPJPE | ✓ | ✓ | ✓ | ✓ |
| Jitter | — | ✓ | ✓ | ✓ |
| Foot skating | ✓ | — | ✓ | ✓ |
| Penetration/float | ✓ | — | ✓ | ✓ |
| FID | ✓ | ✓ | ✓ | — |
| Diversity | — | ✓ | ✓ | — |
| Trajectory error | — | ✓ | ✓ | ✓ (ADE/FDE) |
| EE error | ✓ | — | — | ✓ |
| Contact accuracy | ✓ | — | — | — |
| Heading error | ✓ | — | — | ✓ |

---

## Summary: What We Have, What We're Missing

### ✓ What We Compute Well
1. **Reconstruction metrics**: MPJPE per region (masked/unmasked)
2. **Temporal smoothness**: Jitter (3rd-order finite difference)
3. **Physical realism**: Foot skating, penetration, float
4. **Boundary quality**: Acceleration jump at splice points
5. **Loop closure**: Position/velocity error for cyclic sequences
6. **Spatial control**: EE position error with hit rates
7. **Trajectory**: ADE/FDE on root motion

### ✗ What We Don't Compute
1. **FID**: Requires feature extractor and batch processing
2. **Diversity**: Batch-level variation (depends on FID)
3. **Contact accuracy**: Implicit in velocity threshold heuristic
4. **Rotation error**: Explicitly (rotation quality isn't directly reported)
5. **Text-motion matching**: Not applicable to M2M (we're not text-evaluating)

### Recommendation
Our metric set is **complete for motion completion tasks**. To match publication standards:
- Add FID computation (Medium effort, high impact for comparisons with T2M papers)
- Keep our focus on task-specific metrics (MPJPE, jitter, constraint satisfaction)
- Use physical metrics (skating, penetration) as quality filters

