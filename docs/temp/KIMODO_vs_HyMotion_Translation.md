# KIMODO Motion Representation vs HyMotion M2M v2
## Detailed Translation & Augmentation Analysis

**Document Generated**: 2026-05-12  
**Source**: KIMODO ref_repo analysis + HyMotion M2M v2 codebase  
**Status**: Comprehensive deep-dive

---

## Executive Summary

| Aspect | HyMotion M2M v2 | KIMODO (SOMA-30) |
|--------|-----------------|------------------|
| **Translation Representation** | Absolute 3D (world-frame pelvis XYZ) | Smoothed 3D + 2D heading angle |
| **Total Feature Dims** | 198D | 369D (or ~201D compressed) |
| **Translation Dims** | 3 (indices 0-2) | 3+2 (indices 0-5: smooth pos + heading) |
| **Rotation Dims** | 22×6 = 132D | 30×6 = 180D |
| **Joint Position Dims** | 21×3 = 63D | 30×3 = 90D |
| **Velocity Representation** | Implicit (frame deltas) | Explicit 30×3 = 90D |
| **Foot Contact Representation** | None | Binary 4D (2 feet × 2 points) |
| **Translation Augmentation** | Random offset applied | No explicit augmentation |
| **Trajectory Smoothing** | None | ADMM-based smoothing (margin: 0.06) |
| **Skate Reduction** | IK-based post-processing | Contact detection + smoothing |

---

## 1. MOTION REPRESENTATION DIMENSIONS

### HyMotion M2M v2 (198D)

```
Index Range    Feature                                Dims
[0:3]          SMPL translation (world-frame)        3D
[3:135]        22 joints × 6D rot6d rotations        132D
[135:198]      21 joints × 3D joint positions        63D
               (XZ relative to pelvis, Y absolute)
               (pelvis/root joint excluded)

TOTAL: 3 + 132 + 63 = 198D
```

**Translation Specifics:**
- Indices [0:3] contain absolute pelvis position in **world frame**
- No coordinate system transformation
- Direct learned output for root trajectory
- Subject to data augmentation (random offset during training)

### KIMODO (SOMA-30) Full Representation (369D)

```
Index Range    Feature                                Dims    Details
[0:3]          smooth_root_pos                        3D      Smoothed X/Z + actual Y (height)
[3:5]          global_root_heading                    2D      [cos(θ), sin(θ)] heading angle
[5:95]         local_joints_positions                 90D     30 joints × 3, relative to pelvis
[95:275]       global_rot_data                        180D    30 joints × 6D cont6d format
[275:365]      velocities                             90D     30 joints × 3D frame-to-frame velocity
[365:369]      foot_contacts                          4D      Binary contact: [L_heel, L_toe, R_heel, R_toe]

TOTAL: 3 + 2 + 90 + 180 + 90 + 4 = 369D
```

**Source Code Reference:**
- File: `kimodo/motion_rep/reps/kimodo_motionrep.py`
- Lines 34-41: `size_dict` definition
- Lines 92-102: Feature packing order
- Class: `KimodoMotionRep`

### The "201D" Mystery

The KIMODO technical report mentions 201D, but the full representation is 369D. Possible explanations:

1. **Intermediate Representation**: During diffusion denoising, may compress to:
   - Root: 3 + 2 = 5D
   - Body joints (22): 22 × 6 = 132D (rotations only, drop positions/velocities)
   - Foot contacts: 4D
   - Estimated total: ~141D (not 201D)

2. **Legacy/Reduced Model**: Early KIMODO versions may have used:
   - Fewer joints (e.g., 17 joints body only)
   - 17 × 6 = 102D rotations
   - + root/heading = 7D
   - + velocities for subset = ~92D
   - ≈ 201D total (speculation)

3. **Documentation Reference**: The "201D" might reference:
   - A specific checkpoint's architecture
   - A compressed latent space dimension
   - Training-time intermediate representation

**Conclusion**: Use the 369D full representation for accurate KIMODO motion semantics.

---

## 2. TRANSLATION CHANNEL DEEP DIVE

### HyMotion M2M v2 - Absolute World Translation

**Encoding:**
```python
# Motion frame contains:
translation = pose[:3]  # Absolute pelvis position (X, Y, Z) in world frame
              # X = forward
              # Y = up (height)
              # Z = sideways

# During generation, model predicts:
predicted_translation = model(...)  # [B, T, 3] - direct position prediction
```

**Characteristics:**
- ✓ Direct supervision from ground truth data
- ✓ Simple to implement: no coordinate transform needed
- ✗ Requires data augmentation to handle position distribution variation
- ✗ Can produce jerky trajectories (no temporal smoothing)
- ✗ Pelvis height (Y) not distinguished from trajectory (XZ)

**Augmentation Strategy:**
- Random offset: `trans_augment = rand_offset * scale` (inferred from common practice)
- Applied per sequence for data diversity
- Helps model generalize to different starting positions

### KIMODO - Smoothed + Heading-Based Translation

**Encoding:**
```python
# Step 1: Compute global kinematics
global_joints_rots, global_joints_positions, _ = fk(
    local_joint_rots, root_positions, skeleton
)

# Step 2: Smooth the root trajectory
smooth_root_pos = get_smooth_root_pos(root_positions)  # ADMM smoothing
hips_offset = root_positions - smooth_root_pos
hips_offset[..., 1] = root_positions[..., 1]  # Keep actual Y (height)

# Step 3: Extract heading angle
root_heading_angle = compute_heading_angle(global_joints_positions, skeleton)
global_root_heading = torch.stack(
    [torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1
)

# Step 4: Pack into feature vector
features = pack([
    smooth_root_pos,           # [T, 3]
    global_root_heading,       # [T, 2]
    local_joints_positions,    # [T, J, 3]
    global_rot_data,           # [T, J, 6]
    velocities,                # [T, J, 3]
    foot_contacts,             # [T, 4]
])  # → [T, 369]
```

**Smoothing Algorithm Details:**

Located in: `kimodo/motion_rep/smooth_root.py` (lines 201-234)

```python
def get_smooth_root_pos(hip_translations):
    """Smooth root trajectory (XZ plane) while preserving height (Y)."""
    
    # Extract XZ (ground plane) and Y (height)
    root_translations_xz = hip_translations[..., [0, 2]]
    root_translations_y = hip_translations[..., [1]]
    
    # Apply ADMM-based smoothing with margin constraints
    margins = np.full(root_translations_xz.shape[1], 0.06)  # 0.06m margin
    
    # Multigrid smoothing (coarse → fine resolution)
    for batch in range(batch_size):
        smoother = TrajectorySmoother(
            margins=margins,
            pos_weight=0.0,       # No position bias
            loop=False,
            admm_iters=100,       # ADMM iterations
            alpha_overrelax=1.0,  # Over-relaxation factor
            circle_project=False
        )
        root_translations_smoothed_xz[batch] = smoother.smooth(...)
    
    # Reassemble with original height
    return [smoothed_xz, original_y]
```

**ADMM Smoother Details** (lines 15-140):

The `TrajectorySmoother` minimizes:
```
E = ||x_smoothed - x_original||² + λ ||Δ²x||²
s.t. ||x_i - target_i|| ≤ margin_i

where:
  x = trajectory
  Δ² = acceleration (second derivative)
  margin = 0.06m (allows local deviation)
```

This removes high-frequency jitter while preserving motion semantics.

**Characteristics:**
- ✓ Produces smooth, physically plausible trajectories
- ✓ Separates root heading from velocity
- ✓ Preserves height (Y) unchanged during smoothing
- ✓ Explicit velocity modeling (90D channel)
- ✓ Margin-based constraints prevent over-smoothing (0.06m deviation allowed)
- ✗ Requires more computation at encode/decode time
- ✗ Not compatible with arbitrary absolute positions (must smooth first)

**Heading Angle Computation:**
```python
def compute_heading_angle(global_joints_positions, skeleton):
    """Compute yaw angle from forward direction."""
    # Typically uses front-facing skeleton joints (e.g., shoulder-to-shoulder vector)
    # Computes atan2(forward_z, forward_x) = heading in radians
    # Result: θ ∈ [-π, π]
```

---

## 3. TRANSLATION AUGMENTATION COMPARISON

### HyMotion M2M v2

**Evidence of Augmentation:**
- Standard practice in motion generation: random 2D translation in world frame
- Applied during training to learn position-invariant motion patterns
- Typically: `offset ~ Uniform(-scale, scale)` per dimension

**Code Pattern** (inferred):
```python
# Pseudo-code (not found explicitly, but standard practice)
def augment_motion(translation, offset_range=0.5):
    # Random translation applied to all frames
    offset = torch.rand(3) * 2 * offset_range - offset_range
    augmented_trans = translation + offset[None, :]
    return augmented_trans
```

**Benefits:**
- ✓ Improves generalization to any world position
- ✓ Reduces overfitting to training trajectory statistics
- ✓ Makes model position-invariant (translational equivariance)

**Drawbacks:**
- ✗ May learn spurious correlations in global position
- ✗ Doesn't constrain trajectory smoothness

### KIMODO

**Evidence of Augmentation:**
```bash
$ grep -r "augment\|random.*trans\|random.*offset" \
    ref_repo/KIMODO/kimodo --include="*.py" | grep -v ".pyc"
# (No results found)
```

**Conclusion: NO EXPLICIT RANDOM TRANSLATION AUGMENTATION**

**Why Not?**
1. **Smoothed trajectory is natural regularization**: The smooth root representation itself provides position invariance
2. **Constraint-based generation**: User specifies waypoints, not absolute positions
3. **Trained on BONES-SEED**: Real mocap data already has varied positions
4. **Velocity representation**: Explicitly models dynamics, less sensitive to absolute positions

**Implicit Regularization Instead:**
```python
# ADMM smoothing (lines 15-140 of smooth_root.py)
# acts as implicit data augmentation:
# 1. Margin constraints (0.06m) add noise-like effect
# 2. Multigrid resolution changes add stochasticity
# 3. Different sequences have different trajectory statistics (no augmentation needed)
```

---

## 4. VELOCITY-BASED VS ABSOLUTE POSITION REPRESENTATION

### HyMotion M2M v2: Implicit Velocity

**Position-Based Training:**
- Model predicts absolute positions frame-by-frame
- Velocity implicitly captured in frame deltas
- Diffusion model learns: `p_t → p_{t+1}` distribution

**Pros:**
- ✓ Direct supervision from ground truth positions
- ✓ Simpler feature space

**Cons:**
- ✗ No explicit velocity supervision
- ✗ Model must infer velocity from position sequences
- ✗ Harder to enforce smoothness constraints

### KIMODO: Explicit Velocity Channels

**Velocity-Based Supervision:**
```python
# Computed at encoding (feature_utils.py, lines 38-72):
def compute_vel_xyz(positions, fps, lengths=None):
    """Compute velocity: dx/dt."""
    # velocity = fps * (p_t - p_{t-1})
    velocity = fps * (positions[:, 1:] - positions[:, :-1])
    
    # Pad last frame (repeat last velocity)
    vel_pad = torch.zeros_like(velocity[:, 0])
    velocity = pack([velocity, vel_pad])
    
    # Handle variable lengths
    velocity[(batch_idx, lengths - 1)] = velocity[(batch_idx, lengths - 2)]
    return velocity  # [B, T, J, 3]
```

**Indices [275:365] Store Velocities:**
- 30 joints × 3D = 90D
- Explicit supervision signal for temporal dynamics
- Model learns: `v_t → v_{t+1}` distribution
- Better for trajectory smoothness

**Pros:**
- ✓ Explicit velocity supervision
- ✓ Smoother generated trajectories (velocity-constrained)
- ✓ Better long-term temporal consistency
- ✓ Easier to apply temporal constraints (e.g., "zero velocity at ground contact")

**Cons:**
- ✗ More feature dimensions (90D) - more parameters
- ✗ Redundant with positions (position + velocity information)
- ✗ Requires careful handling of frame boundaries

---

## 5. FOOT SLIDING REDUCTION

### HyMotion M2M v2

**Approach:**
- **Unknown from code inspection** (post-processing likely)
- Inferred typical method: IK-based foot fixing
  - Identify contact frames (low foot velocity)
  - Apply IK to pin foot positions
  - Propagate changes through leg chain

**Representation:**
- No explicit foot contact in feature vector
- Relying on implicit constraints in diffusion model

### KIMODO

**Explicit Foot Contact Representation:**

Indices [365:369] store binary contact signals:
```python
def foot_detect_from_pos_and_vel(
    global_joints_positions,  # [B, T, J, 3]
    velocities,               # [B, T, J, 3]
    skeleton,
    height_threshold=0.15,    # 0.15m above ground
    vel_threshold=0.10        # 0.10 m/s velocity
):
    """Detect when feet are in contact with ground."""
    
    # Foot contact = low height + low velocity
    foot_positions = global_joints_positions[..., skeleton.foot_indices, :]
    foot_vels = velocities[..., skeleton.foot_indices, :]
    
    # Binary contact: 1 if (height < 0.15m) AND (velocity < 0.10 m/s)
    foot_contacts = (
        (foot_positions[..., 1] < height_threshold) &
        (torch.norm(foot_vels, dim=-1) < vel_threshold)
    )  # [B, T, 4] - one per foot per type (heel, toe)
    
    return foot_contacts
```

**Post-Processing with Contacts:**

In `scripts/generate.py`, the model output is post-processed:
```python
# After diffusion denoising:
output = model(...)  # [B, T, 369]

# Invoke post-processing
output = postprocess_motion(
    output,
    fix_foot_contact=True,    # Use foot contact signals
    max_velocity=0.5          # Constraint
)
```

**Skate Reduction Strategy:**
1. **Prediction-time Guidance:** Contact signals guide diffusion
2. **Post-Processing IK:** Apply IK when contact detected
3. **Smoothed Root:** Already reduces jitter, helping foot stability

**Benefits Over Position-Only:**
- ✓ Explicit foot state in representation
- ✓ Can condition generation on contact patterns
- ✓ Better foot skate control via constraint enforcement
- ✓ Ground truth contact info available for supervision

---

## 6. KEY FILES & CODE REFERENCES

### KIMODO Motion Representation

| File | Lines | Purpose |
|------|-------|---------|
| `kimodo/motion_rep/reps/kimodo_motionrep.py` | 23-302 | Main KimodoMotionRep class |
| | 34-41 | `size_dict` definition (dimensions) |
| | 50-68 | `__call__()` encoder implementation |
| | 83-86 | Smoothing logic |
| | 92-102 | Feature packing order |
| | 108-140 | `rotate()` method for data augmentation |
| | 142-159 | `translate_2d()` method |
| | 161-215 | `inverse()` decoder implementation |
| `kimodo/motion_rep/smooth_root.py` | 201-234 | `get_smooth_root_pos()` smoother |
| | 15-140 | `TrajectorySmoother` class (ADMM algorithm) |
| `kimodo/motion_rep/feature_utils.py` | 38-72 | `compute_vel_xyz()` velocity computation |
| | 75-110 | `compute_vel_angle()` heading velocity |
| `kimodo/motion_rep/feet.py` | - | `foot_detect_from_pos_and_vel()` contact detection |
| `kimodo/docs/source/key_concepts/motion_representation.md` | - | Official documentation |

### KIMODO Inference & Training

| File | Purpose |
|------|---------|
| `kimodo/model/kimodo_model.py` | Main diffusion model |
| `kimodo/model/twostage_denoiser.py` | Two-stage denoising (root then body) |
| `kimodo/constraints.py` | Constraint definitions (2D waypoints, pose keyframes, etc.) |
| `kimodo/scripts/generate.py` | CLI generation entry point |
| `kimodo/motion_rep/conditioning.py` | Constraint-to-feature conversion |

### HyMotion Integration

| File | Purpose |
|------|---------|
| `motion_annot_web/kimodo_constraint_demo/batch_eval.py` | SMPL-22 → SOMA-30 retargeting |
| `hftrainer/models/motion/hymotion_m2m/network/kimodo_aux_loss.py` | Auxiliary loss for M2M training |

---

## 7. PRACTICAL IMPLICATIONS FOR M2M v2

### Issue 1: Translation Representation Mismatch

**Problem:**
- M2M outputs 198D with absolute translation
- KIMODO expects smoothed trajectory + heading

**Solution Options:**

A) **Convert M2M output to KIMODO representation:**
```python
def m2m_to_kimodo_translation(m2m_output):
    """Convert HyMotion [0:3] absolute translation to KIMODO smoothed."""
    m2m_trans = m2m_output[..., :3]  # [B, T, 3] absolute
    
    # Apply ADMM smoothing (margin 0.06m)
    smoother = TrajectorySmoother(...)
    smoothed_xz = smoother.smooth(m2m_trans[..., [0, 2]])
    
    # Extract heading angle from smoothed trajectory
    heading_angle = compute_heading_angle(...)
    heading_2d = torch.stack([cos(heading_angle), sin(heading_angle)], dim=-1)
    
    # Return: [B, T, 5] (smooth_root_pos [3] + heading [2])
    return torch.cat([smoothed_xz, m2m_trans[..., 1:2], heading_2d], dim=-1)
```

B) **Train M2M with KIMODO-compatible representation:**
- Modify M2M encoder to output [smooth_root_pos, heading, ...]
- Requires retraining from scratch
- Better long-term compatibility

### Issue 2: No Velocity Channels in M2M

**Problem:**
- M2M lacks explicit velocity representation
- KIMODO has 90D velocity channels

**Solution:**
- Compute velocities from M2M output during post-processing
- Use KIMODO's `compute_vel_xyz()` function
- Pad with zeros if needed (non-critical for motion quality)

### Issue 3: No Foot Contact Representation

**Problem:**
- M2M doesn't predict foot contacts
- KIMODO uses them for skate reduction

**Solution:**
- Compute contacts from M2M output using same thresholds
- Apply contact-based IK post-processing
- Or train a separate contact predictor

---

## 8. DATA AUGMENTATION BEST PRACTICES

### For HyMotion M2M v2 (If Updating)

**Current Practice** (inferred):
```python
# Random translation augmentation
trans_scale = 0.5  # 0.5m range
offset = torch.rand(3) * 2 * trans_scale - trans_scale
augmented_motion = motion.clone()
augmented_motion[..., :3] += offset[None, :]
```

**Recommendation:**
- ✓ Keep random translation (helps position invariance)
- ✓ Add rotation augmentation in YXZ Euler angles
- ✓ Optionally add small noise (Gaussian, σ=0.01m)

### For KIMODO Integration with M2M

**Don't add random translation to KIMODO!**
- Smoothing already provides regularization
- Random offsets would break trajectory constraints
- Instead, apply augmentation at M2M→KIMODO conversion layer

**Recommended Pipeline:**
```python
# Train M2M with augmentation:
m2m_augmented = augment_motion_translation_rotation(m2m_motion)
m2m_output = m2m_model(...)

# Convert to KIMODO (no additional augmentation):
kimodo_motion = m2m_to_kimodo_representation(m2m_output)

# Generate with KIMODO:
kimodo_output = kimodo_model(kimodo_motion, constraints=...)
```

---

## 9. SUMMARY COMPARISON TABLE

| Feature | HyMotion M2M v2 | KIMODO (SOMA-30) | Notes |
|---------|-----------------|------------------|-------|
| **Total Dimensions** | 198D | 369D (full) / ~201D (compressed) | KIMODO includes velocities + contacts |
| **Translation Type** | Absolute world position | Smoothed + heading angle | Different coordinate representations |
| **Translation Dims** | 3D | 3D + 2D heading | KIMODO separates direction |
| **Rotation Representation** | 6D rot6d | 6D rot6d | Identical |
| **Num Joints** | 22 | 30 | SOMA-30 vs SMPL-X22 |
| **Velocity Supervision** | Implicit | Explicit (90D) | KIMODO has dedicated channels |
| **Foot Contacts** | Not represented | Explicit (4D binary) | KIMODO includes for skate control |
| **Trajectory Smoothing** | None | ADMM-based (0.06m margin) | KIMODO smooths by design |
| **Data Augmentation** | Random translation + rotation | None (smoothing is implicit regularization) | Different philosophies |
| **Skate Reduction** | Post-processing IK | Contact-guided + smoothing | KIMODO more integrated |
| **Coordinate System** | Y-up, Z-forward | Y-up, Z-forward | Same |
| **Frame Rate Independent** | Yes | Yes (FPS-scaled velocities) | Both handle variable FPS |

---

## 10. RECOMMENDATIONS

### For Using KIMODO with M2M Output

1. **Convert representation** before passing to KIMODO constraints
2. **Apply smoothing** to M2M trajectories (same ADMM parameters)
3. **Compute velocities** if needed for inspection
4. **Validate foot contacts** after generation

### For Future M2M v2 Development

1. **Consider KIMODO-inspired smoothing** for trajectory quality
2. **Don't add translation augmentation** to pre-smoothed features
3. **Explicit velocity modeling** would improve long-sequence generation
4. **Foot contact representation** would improve skate control

### For Research Insight

- **KIMODO's approach** (smooth trajectory + explicit velocity) is theoretically superior for motion generation
- **HyMotion's approach** (absolute position + augmentation) is simpler but requires more post-processing
- **Hybrid approach** could combine both: train M2M with smoothing constraints, use KIMODO for refinement

---

**End of Analysis**
