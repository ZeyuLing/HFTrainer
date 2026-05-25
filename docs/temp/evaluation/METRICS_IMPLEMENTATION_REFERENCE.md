# Metrics Implementation Reference for M2M v2

**Last Updated**: May 22, 2026

---

## File Structure

```
hftrainer/evaluation/motion/
├── m2m_eval_metrics.py      # Core metrics (838 lines)
│   ├── Position-based: MPJPE, jitter, bone_length_cv, trajectory
│   ├── Rotation-based: heading_error, fk_consistency
│   ├── Ground metrics: foot_penetration, foot_float, foot_skating_ratio
│   ├── Boundary: boundary_smoothness
│   ├── Loop: loop_continuity
│   ├── End-effector: end_effector_error (hit rates)
│   └── Aggregation: aggregate_metrics()
│
├── m2m_eval_tasks.py        # Task definitions (1000+ lines)
│   ├── Mask builders: build_inbetween_mask(), build_trajectory_mask(), etc.
│   ├── Task registry: TASK_REGISTRY[task_name] = TaskDef
│   │   └── Each task has: mask_builder, default_metrics, settings A-F
│   └── Constraint builders: for E4 (end-effector), E5 (trajectory)
│
├── phys_metrics.py          # Physical error metrics
│   ├── Jerk, joint_pop, wrist_twist
│   ├── Penetration, float, skating (vertex-level)
│   └── Bone length consistency
│
└── __init__.py
```

---

## Core Metric Implementations

### 1. MPJPE (Mean Per-Joint Position Error)

**Location**: `m2m_eval_metrics.py`, lines 145-207

```python
def compute_mpjpe(pred_pos, gt_pos, mask=None, joint_indices=None):
    """
    Mean Per-Joint Position Error.
    
    Args:
        pred_pos: (T, 22, 3) predicted positions from FK
        gt_pos: (T, 22, 3) ground truth positions from FK
        mask: (T, 135) optional mask where mask=1 means "evaluate this frame"
        joint_indices: optional subset of joints [default: all 22]
    
    Returns:
        {
            'mpjpe_mean': float,
            'mpjpe_per_joint': [22] float list
        }
    """
    # Step 1: Determine evaluation frames (if mask provided)
    if mask is not None:
        frame_mask = mask.max(axis=-1) > 0.5  # (T,) boolean
        # Handle length mismatch (e.g., E14/E15 stitched sequences)
        if frame_mask.shape[0] != T:
            # Pad or crop frame_mask to match T
            ...
    else:
        frame_mask = np.ones(T, dtype=bool)
    
    # Step 2: Select frames and optionally joints
    pred_sel = pred_pos[frame_mask]  # (N_eval, 22, 3)
    gt_sel = gt_pos[frame_mask]
    
    # Step 3: Optionally subset joints
    if joint_indices is not None:
        pred_sel = pred_sel[:, joint_indices]
        gt_sel = gt_sel[:, joint_indices]
    
    # Step 4: Compute L2 error per joint
    per_joint_err = np.linalg.norm(pred_sel - gt_sel, axis=-1)  # (N_eval, J)
    mpjpe_mean = float(per_joint_err.mean())
    mpjpe_per_joint = per_joint_err.mean(axis=0).tolist()
    
    return {
        'mpjpe_mean': mpjpe_mean,
        'mpjpe_per_joint': mpjpe_per_joint,
    }
```

**Units**: meters (input coordinates are meters)  
**Typical values**: 0.05-0.20 m for good completion  
**Variants**:
- `mpjpe_all`: All frames
- `mpjpe_masked`: Only generated (mask=1) frames
- `mpjpe_unmasked`: Only known (mask=0) frames (should be ≈0)

---

### 2. Jitter (3rd-Order Finite Difference)

**Location**: `m2m_eval_metrics.py`, lines 213-244

#### Position-Based (Preferred)

```python
def compute_jitter_positions(positions, fps=30.0):
    """
    Jitter = Mean jerk (3rd-order finite difference) of joint positions.
    
    Jerk = d³x/dt³, computed as:
        diff3[t] = x[t+3] - 3*x[t+2] + 3*x[t+1] - x[t]
        jerk[t] = diff3[t] / dt³
        jitter = mean(||jerk|| across all frames & joints)
    
    Args:
        positions: (T, 22, 3) world-space joint positions
        fps: frames per second (default 30)
    
    Returns:
        Jitter value in m/s³
    """
    if positions.shape[0] < 4:
        return 0.0
    
    dt = 1.0 / fps  # 1/30 sec
    
    # 3rd order finite difference: applies to x[t], x[t+1], x[t+2], x[t+3]
    # Yields len(x)-3 differences
    diff3 = positions[3:] - 3*positions[2:-1] + 3*positions[1:-2] - positions[:-3]
    
    # Normalize by dt³
    jerk = diff3 / (dt ** 3)  # (T-3, 22, 3)
    
    # Compute L2 norm per timepoint, then mean
    jerk_norm = np.linalg.norm(jerk.reshape(jerk.shape[0], -1), axis=-1)  # (T-3,)
    return float(np.mean(jerk_norm))
```

**Units**: m/s³  
**Typical values**: 0.05-0.50 m/s³ (lower = smoother)  
**Interpretation**: Captures high-frequency jitter; complements MPJPE

#### Raw 135D Version

```python
def compute_jitter_135(motion):
    """Jitter directly on 135-dim representation (no FK needed)."""
    if motion.shape[0] < 4:
        return 0.0
    diff3 = motion[3:] - 3*motion[2:-1] + 3*motion[1:-2] - motion[:-3]
    return float(np.mean(np.abs(diff3)))
```

**Advantage**: Fast (no FK)  
**Disadvantage**: Unitless; less interpretable

---

### 3. Foot Ground Metrics

**Location**: `m2m_eval_metrics.py`, lines 618-680

```python
def compute_foot_ground_metrics(positions, ground_y=0.0, 
                                 contact_threshold=0.05,
                                 skating_threshold=0.01, fps=30.0):
    """
    Returns:
    {
        'foot_penetration': float,     # avg depth below ground
        'foot_float': float,           # avg height above ground during "contact"
        'foot_skating_ratio': float,   # fraction of contact frames with XZ motion
        'foot_avg_skate': float,       # avg XZ velocity during skating
    }
    """
    T = positions.shape[0]
    foot_pos = positions[:, [7, 8, 10, 11], :]  # (T, 4, 3) L_ankle, R_ankle, L_foot, R_foot
    foot_y = foot_pos[:, :, 1]  # (T, 4)
    
    # 1. Penetration: how much below ground
    penetration = np.maximum(ground_y - foot_y, 0)
    avg_penetration = float(penetration.mean())
    
    # 2. Float: height above ground when in "contact" (low velocity)
    foot_vel = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1) * fps  # (T-1, 4)
    contact = foot_vel < skating_threshold * fps  # (T-1, 4)
    float_heights = []
    for t in range(T-1):
        for j in range(4):
            if contact[t, j] and foot_y[t, j] > ground_y + contact_threshold:
                float_heights.append(foot_y[t, j] - ground_y)
    avg_float = float(np.mean(float_heights)) if float_heights else 0.0
    
    # 3. Skating: XZ velocity during ground contact
    foot_xz_vel = np.diff(foot_pos[:, :, [0, 2]], axis=0) * fps  # (T-1, 4, 2)
    foot_xz_speed = np.linalg.norm(foot_xz_vel, axis=-1)  # (T-1, 4)
    
    contact_mask = foot_y[:-1] < ground_y + contact_threshold  # (T-1, 4)
    skating_frames = contact_mask & (foot_xz_speed > skating_threshold * fps)
    skating_ratio = float(skating_frames.sum()) / max(contact_mask.sum(), 1)
    skating_speeds = foot_xz_speed[skating_frames]
    avg_skate = float(skating_speeds.mean()) if len(skating_speeds) > 0 else 0.0
    
    return {
        'foot_penetration': avg_penetration,
        'foot_float': avg_float,
        'foot_skating_ratio': skating_ratio,
        'foot_avg_skate': avg_skate,
    }
```

**Parameters**:
- `contact_threshold`: 5cm (foot within this distance considered "contacting")
- `skating_threshold`: 1cm/frame (0.01 m/frame at 30fps = 0.3 m/s)

**Typical values**:
- penetration: 0.00-0.05m (good < 2cm)
- float: 0.00-0.10m (good < 5cm)
- skating_ratio: 0.00-0.30 (good < 15%)
- avg_skate: 0.02-0.15 m/s (good < 5cm/s)

---

### 4. Trajectory Metrics (Root Following)

**Location**: `m2m_eval_metrics.py`, lines 292-348

```python
def compute_trajectory_metrics(pred_motion, gt_motion, mask=None):
    """
    ADE (Average Displacement Error) and FDE (Final Displacement Error)
    on root XZ plane (horizontal 2D trajectory).
    
    Returns:
    {
        'trajectory_ade': float,  # meters
        'trajectory_fde': float,  # meters
    }
    """
    # Extract root XZ from translation (dims 0, 2)
    pred_root_xz = pred_motion[:, [0, 2]]
    gt_root_xz = gt_motion[:, [0, 2]]
    
    # Apply mask if provided (align length, center crop/pad if needed)
    if mask is not None:
        frame_mask = mask.max(axis=-1) > 0.5
        # Handle misalignment...
        if frame_mask.shape[0] != T_pred:
            # Pad with False or center crop
            ...
        pred_root_xz = pred_root_xz[frame_mask]
        gt_root_xz = gt_root_xz[frame_mask]
    
    # ADE: mean distance at all frames
    ade = float(np.mean(np.linalg.norm(pred_root_xz - gt_root_xz, axis=-1)))
    
    # FDE: distance at last frame
    fde = float(np.linalg.norm(pred_root_xz[-1] - gt_root_xz[-1]))
    
    return {
        'trajectory_ade': ade,
        'trajectory_fde': fde,
    }
```

**Typical values**:
- ADE: 0.01-0.20m (good < 5cm)
- FDE: 0.02-0.50m (good < 10cm)

---

### 5. End-Effector Error (E4 Spatial Control)

**Location**: `m2m_eval_metrics.py`, lines 538-603

```python
def compute_end_effector_error(pred_pos, constraint_positions, 
                               constraint_frames, constraint_joints):
    """
    For each (frame, joint) constraint pair:
        error = ||FK(pred_rot)[joint] - target_position||
    
    Aggregates over all constraints with: mean, max, p50, p95, std, hit_rates
    
    Args:
        pred_pos: (T, 22, 3) predicted positions
        constraint_positions: (N, 3) target positions
        constraint_frames: (N,) frame indices
        constraint_joints: (N,) joint indices
    
    Returns:
    {
        'ee_error_mean': float,
        'ee_error_max': float,
        'ee_error_p50': float (median),
        'ee_error_p95': float,
        'ee_error_std': float,
        'ee_hit_rate_2cm': float,  # fraction < 0.02m
        'ee_hit_rate_5cm': float,  # fraction < 0.05m
        'ee_hit_rate_10cm': float, # fraction < 0.10m
    }
    """
    errors = []
    for i in range(len(constraint_positions)):
        f = int(constraint_frames[i])
        j = int(constraint_joints[i])
        if f < pred_pos.shape[0]:
            err = np.linalg.norm(pred_pos[f, j] - constraint_positions[i])
            errors.append(err)
    
    if not errors:
        return {
            'ee_error_mean': 0.0, 'ee_error_max': 0.0,
            'ee_error_p50': 0.0, 'ee_error_p95': 0.0,
            'ee_error_std': 0.0,
            'ee_hit_rate_2cm': 0.0, 'ee_hit_rate_5cm': 0.0,
            'ee_hit_rate_10cm': 0.0,
        }
    
    errors_np = np.asarray(errors, dtype=np.float32)
    return {
        'ee_error_mean': float(errors_np.mean()),
        'ee_error_max': float(errors_np.max()),
        'ee_error_p50': float(np.percentile(errors_np, 50)),
        'ee_error_p95': float(np.percentile(errors_np, 95)),
        'ee_error_std': float(errors_np.std()),
        'ee_hit_rate_2cm': float((errors_np < 0.02).mean()),
        'ee_hit_rate_5cm': float((errors_np < 0.05).mean()),
        'ee_hit_rate_10cm': float((errors_np < 0.10).mean()),
    }
```

**Typical values**:
- error_mean: 0.02-0.10m (good < 5cm)
- hit_rate_5cm: 0.30-1.00 (good > 70%)

---

### 6. Boundary Smoothness (Stitching Tasks)

**Location**: `m2m_eval_metrics.py`, lines 398-477

```python
def compute_boundary_smoothness(motion, mask, bone_offsets=None, 
                                 boundary_width=3, fps=30.0):
    """
    Acceleration discontinuity at mask transition (splice point).
    
    Measures: how smooth is the transition from "known" to "generated"?
    
    Args:
        motion: (T, 135) output motion
        mask: (T, 135) mask (0=known, 1=generated)
        bone_offsets: for FK to positions (optional)
        boundary_width: frames around boundary to evaluate (±3)
    
    Returns:
    {
        'boundary_accel_jump': float,  # L2 norm of acceleration difference
    }
    """
    # Find mask transition points
    mask_per_frame = mask.max(axis=-1) > 0.5  # (T,)
    boundary_frames = set()
    for t in range(1, T):
        if mask_per_frame[t] != mask_per_frame[t-1]:
            for dt in range(-boundary_width, boundary_width+1):
                ft = t + dt
                if 0 <= ft < T:
                    boundary_frames.add(ft)
    
    # Compute acceleration
    if bone_offsets is not None:
        data = motion135_to_positions_np(motion, bone_offsets)  # (T, 22, 3)
        data = data.reshape(T, -1)  # (T, 66)
    else:
        data = motion  # (T, 135)
    
    # Acceleration: 2nd order finite difference
    dt = 1.0 / fps
    accel = (data[2:] - 2*data[1:-1] + data[:-2]) / (dt**2)  # (T-2, ...)
    
    # Accel jump at boundaries
    accel_jumps = []
    for bf in sorted(boundary_frames):
        if 1 <= bf < T-1 and 1 <= bf-1 < T-1:
            a1 = accel[bf-1]
            a0 = accel[bf-2] if bf >= 2 else a1
            jump = np.linalg.norm(a1 - a0)
            accel_jumps.append(jump)
    
    return {
        'boundary_accel_jump': float(np.mean(accel_jumps)) if accel_jumps else 0.0,
    }
```

**Typical values**: 0.01-1.00 (lower = smoother transitions)

---

### 7. Loop Continuity (E8D - Loop Tasks)

**Location**: `m2m_eval_metrics.py`, lines 484-520

```python
def compute_loop_continuity(motion, bone_offsets=None, fps=30.0):
    """
    For looping motion: error between first and last frame.
    
    Returns:
    {
        'loop_position_error': float,   # ||first - last|| meters
        'loop_velocity_error': float,   # ||vel_first - vel_last|| m/s
    }
    """
    T = motion.shape[0]
    if T < 3:
        return {'loop_position_error': 0.0, 'loop_velocity_error': 0.0}
    
    if bone_offsets is not None:
        pos = motion135_to_positions_np(motion, bone_offsets)  # (T, 22, 3)
        pos_err = float(np.mean(np.linalg.norm(pos[0] - pos[-1], axis=-1)))
        
        vel_first = (pos[1] - pos[0]) * fps
        vel_last = (pos[-1] - pos[-2]) * fps
        vel_err = float(np.mean(np.linalg.norm(vel_first - vel_last, axis=-1)))
    else:
        pos_err = float(np.mean(np.abs(motion[0] - motion[-1])))
        vel_first = (motion[1] - motion[0]) * fps
        vel_last = (motion[-1] - motion[-2]) * fps
        vel_err = float(np.mean(np.abs(vel_first - vel_last)))
    
    return {
        'loop_position_error': pos_err,
        'loop_velocity_error': vel_err,
    }
```

**Typical values**:
- position_error: 0.00-0.20m (for perfect loop: 0.0)
- velocity_error: 0.00-0.50 m/s

---

## Task Definitions and Default Metrics

### Location: `m2m_eval_tasks.py`

Each task `E1-E16` has:
```python
{
    'mask_builder': callable,          # How to build the mask
    'default_metrics': List[str],      # Which metrics to compute
    'data_file': str,                  # Which eval JSON to load
    'settings': {
        'A': {...},  # Settings A-F
        'B': {...},
        ...
    },
    'needs_gt': bool,                  # Does this task need ground truth?
}
```

### Example: E2 (Motion In-Betweening)

```python
TaskDef(
    name='E2',
    desc='Motion In-Betweening (6 settings: start/end/both 1-frame + temporal fractions)',
    mask_builder=build_inbetween_mask,
    default_metrics=[
        'mpjpe_masked', 'mpjpe_unmasked', 'boundary_accel_jump',
        'jitter_pos', 'foot_skating_ratio'
    ],
    data_file='eval_e2_inbetween_v2_rewritten.json',
    settings={
        'A': {'keep_start': 1, 'keep_end': 0, 'name': 'start_1f'},
        'B': {'keep_start': 0, 'keep_end': 1, 'name': 'end_1f'},
        'C': {'keep_start': 1, 'keep_end': 1, 'name': 'both_1f'},
        'D': {'keep_start_frac': 0.20, 'keep_end_frac': 0, 'name': 'pre20'},
        'E': {'keep_start_frac': 0, 'keep_end_frac': 0.20, 'name': 'post20'},
        'F': {'keep_start_frac': 0.20, 'keep_end_frac': 0.20, 'name': 'mid60'},
    },
    needs_gt=True,
)
```

### Standard Metric Combinations

| Task Category | Primary Metrics | Secondary Metrics |
|---------------|-----------------|-------------------|
| **T2M (E1)** | jitter_pos | foot_skating_ratio |
| **Temporal (E2, E3, E8D, E16)** | mpjpe_masked, mpjpe_unmasked | jitter_pos, boundary_accel_jump |
| **Spatial (E4)** | ee_error_mean, hit_rate_5cm | jitter_pos |
| **Trajectory (E5)** | trajectory_ade, trajectory_fde | jitter_pos |
| **Stitching (E14, E15)** | loop_error / segment_smoothness | jitter_pos, foot metrics |

---

## How Metrics Are Computed

### Single Sample Flow

```python
# In eval_m2m_v2_all_tasks.py:
metrics = compute_all_metrics(
    pred_motion=(T, 135),
    gt_motion=(T, 135),
    mask=(T, 135),
    bone_offsets=(22, 3),
    rotation_space='local',
    fps=30.0,
    compute_fk=True,
)
# Returns: Dict[metric_name -> float]
```

### Batch Aggregation

```python
per_sample_metrics = [metrics1, metrics2, ..., metricsN]
aggregated = aggregate_metrics(per_sample_metrics)

# Returns: Dict[metric_name -> {mean, std, median, min, max, count}]
```

---

## Tips for Adding New Metrics

1. **Location**: Add function to `m2m_eval_metrics.py`
2. **Signature**: `def compute_my_metric(...) -> Dict[str, float]`
3. **Units**: Document clearly (meters, m/s, %, etc.)
4. **Fast path**: Consider FK-free version if possible
5. **Register**: Add to task's `default_metrics` list in `m2m_eval_tasks.py`

Example template:
```python
def compute_my_metric(pred_motion, gt_motion, mask=None, 
                      bone_offsets=None, fps=30.0):
    """
    Description of what this metric measures.
    
    Args:
        pred_motion: (T, 135)
        gt_motion: (T, 135)
        mask: (T, 135) optional
        bone_offsets: (22, 3) optional
        fps: frames per second
    
    Returns:
        Dict with metric values, e.g.:
        {
            'my_metric_value': float,
            'my_metric_std': float,
        }
    """
    # Implementation
    return {...}
```

