# ProtoMotions Velocity Storage Analysis Report

## Executive Summary

**Critical Finding:** ProtoMotions' `gvs` (global body velocities) and `gavs` (global body angular velocities) are **frame-origin velocities** stored as **world-space linear and angular velocities** of rigid bodies in MuJoCo's global coordinate system. These velocities are **NOT** stored at the center-of-mass (COM) but at the rigid body frame origin (xpos in MuJoCo terms).

The velocities are computed using **finite difference methods** from position data during motion loading, with support for multi-horizon noise filtering in pose_lib.py.

---

## 1. Velocity Storage in MotionLib Class

### 1.1 Field Definitions (motion_lib.py, Lines 113-116)

```python
# MotionLib class fields (lines 113-116)
gts: torch.Tensor   # Global rigid body positions [total_frames, num_bodies, 3]
grs: torch.Tensor   # Global rigid body rotations [total_frames, num_bodies, 4] (xyzw)
gvs: torch.Tensor   # Global rigid body velocities [total_frames, num_bodies, 3] ← LINEAR VELOCITIES
gavs: torch.Tensor  # Global rigid body angular velocities [total_frames, num_bodies, 3] ← ANGULAR VELOCITIES
dvs: torch.Tensor   # DOF velocities [total_frames, num_dofs]
dps: torch.Tensor   # DOF positions (joint angles) [total_frames, num_dofs]
```

**Key Insight:**
- `gvs`: Shape `[total_frames, num_bodies, 3]` - **world-space linear velocity vectors**
- `gavs`: Shape `[total_frames, num_bodies, 3]` - **world-space angular velocity vectors**
- Both stored as **float32 tensors** on specified device

### 1.2 Motion Field Mapping (motion_lib.py, Lines 59-66)

The mapping from motion data files to MotionLib fields:

```python
# Lines 59-66
_motion_field_mapping = {
    "gts": "rigid_body_pos",      # Global positions (frame origin, MuJoCo xpos)
    "grs": "rigid_body_rot",      # Global rotations (xyzw format)
    "gavs": "rigid_body_ang_vel", # Angular velocities ← LOADED FROM FILE
    "gvs": "rigid_body_vel",      # Linear velocities ← LOADED FROM FILE
    "dvs": "dof_vel",             # DOF velocities
    "dps": "dof_pos",             # DOF positions
}
```

---

## 2. Velocity Loading from .motion Files

### 2.1 Motion File Loading Pipeline (motion_lib.py, Lines 431-538)

The `_load_motions()` method loads individual motion files:

```python
# Lines 450-458: Load each motion file as RobotState
def _load_motions(self, motion_file):
    # ... setup code ...
    for f in range(num_motion_files):
        curr_file = motion_files[f]
        
        # LINE 450: torch.load() returns dict with velocity fields
        curr_motion = torch.load(curr_file, weights_only=False)
        
        # LINE 451-453: Convert dict to RobotState (applies field mapping)
        curr_motion = RobotState.from_dict(
            curr_motion, state_conversion=StateConversion.COMMON
        )
        
        motions.append(curr_motion)
        # ... store metadata ...
```

**Critical:** The loaded velocities come directly from the saved .motion files via `RobotState.from_dict()`.

### 2.2 Velocity Concatenation (motion_lib.py, Lines 461-473)

```python
# Lines 461-473: Concatenate velocity tensors from all motions
for lib_field, motion_attr in _motion_field_mapping.items():
    tp = (
        torch.bool
        if getattr(motions[0], motion_attr).dtype == torch.bool
        else torch.float32
    )
    setattr(
        self,
        lib_field,
        torch.cat([getattr(m, motion_attr) for m in motions], dim=0).to(
            dtype=tp, device=self.device
        ),
    )
```

**What this does:**
- Concatenates `gvs` (rigid_body_vel) from all motions into single tensor: `[total_frames, num_bodies, 3]`
- Concatenates `gavs` (rigid_body_ang_vel) from all motions into single tensor: `[total_frames, num_bodies, 3]`
- These are stored as **float32** tensors

### 2.3 Packaged .pt File Storage (motion_lib.py, Lines 576-600)

```python
# Lines 588-599: What gets saved in packaged .pt files
def save_to_file(self, file_path):
    save_data = {}
    for field in self._fields:
        if getattr(self, field) is not None:
            save_data[field] = getattr(self, field)
    
    torch.save(save_data, file_path)
```

The .pt file contains:
```python
{
    "gts": torch.Tensor([total_frames, num_bodies, 3]),     # Positions
    "grs": torch.Tensor([total_frames, num_bodies, 4]),     # Rotations
    "gvs": torch.Tensor([total_frames, num_bodies, 3]),     # ← LINEAR VELOCITIES STORED
    "gavs": torch.Tensor([total_frames, num_bodies, 3]),    # ← ANGULAR VELOCITIES STORED
    "dvs": torch.Tensor([total_frames, num_dofs]),
    "dps": torch.Tensor([total_frames, num_dofs]),
    "motion_num_frames": torch.Tensor([num_motions], dtype=long),
    "length_starts": torch.Tensor([num_motions], dtype=long),
    "motion_weights": torch.Tensor([num_motions]),
    "motion_lengths": torch.Tensor([num_motions]),
    "motion_dt": torch.Tensor([num_motions]),
    "contacts": torch.Tensor or None,
    "lrs": torch.Tensor or None,
    "motion_files": tuple of str,
}
```

### 2.4 Loading from .pt Files (motion_lib.py, Lines 602-627)

```python
# Lines 602-616: Load packaged velocities directly
def load_from_file(self, file_path):
    print(f"Loading motion library from {file_path}")
    loaded_data = torch.load(
        file_path, map_location=self.device, weights_only=False
    )
    
    # Line 614-616: Directly set gvs and gavs from loaded data
    for field in loaded_data:
        assert loaded_data[field] is not None, f"Field {field} is None"
        setattr(self, field, loaded_data[field])
```

---

## 3. Velocity Motion State Retrieval

### 3.1 Getting Motion State at Exact Frame (motion_lib.py, Lines 390-429)

```python
# Lines 390-429: Retrieve velocities at exact frame indices
def get_motion_state_exact_frame(
    self,
    motion_ids,
    frame_indices,
) -> RobotState:
    # LINE 407: Get global indices by adding offsets
    fl = frame_indices + self.length_starts[motion_ids]
    
    # LINES 410-414: Retrieve velocity data
    motion_data = {}
    for lib_field, motion_attr in _motion_field_mapping.items():
        field_data = getattr(self, lib_field)
        if field_data is not None:
            # Extract gvs (rigid_body_vel) and gavs (rigid_body_ang_vel) at frame fl
            motion_data[motion_attr] = field_data[fl].clone()
```

**Result:** Returns RobotState with:
- `rigid_body_vel`: velocities at frame index `fl`
- `rigid_body_ang_vel`: angular velocities at frame index `fl`

### 3.2 Interpolating Velocities Between Frames (motion_lib.py, Lines 317-388)

```python
# Lines 332-349: Velocity interpolation
def get_motion_state(
    self, motion_ids, motion_times, joint_3d_format="exp_map"
) -> RobotState:
    frame_idx0, frame_idx1, blend = self._calc_frame_blend_from_id_and_time(
        motion_ids, motion_times
    )
    
    motion_state_0 = self.get_motion_state_exact_frame(motion_ids, frame_idx0)
    motion_state_1 = self.get_motion_state_exact_frame(motion_ids, frame_idx1)
    
    # LINES 332-337: Interpolate velocity fields
    pos_keys = [
        "rigid_body_pos",
        "rigid_body_vel",      # ← LINEAR VELOCITY INTERPOLATION
        "rigid_body_ang_vel",  # ← ANGULAR VELOCITY INTERPOLATION
        "dof_vel",
    ]
    
    # LINES 341-344: Linear interpolation for velocities
    for key in pos_keys:
        motion_state_0[key] = interpolate_pos(
            motion_state_0[key], motion_state_1[key], blend
        )
```

**Critical Detail:** Velocities are interpolated using **linear interpolation** (not spherical):
- This assumes velocities change linearly between frames
- Same interpolation method as positions

---

## 4. Velocity Computation from Position Data

### 4.1 Finite Difference Velocity Computation (pose_lib.py, Lines 1150-1227)

The `compute_cartesian_velocity()` function computes linear velocities from positions:

```python
# Lines 1150-1227: Compute linear velocities from positions
def compute_cartesian_velocity(
    batched_robot_pos: torch.Tensor,  # [T, Nb, 3]
    fps: int,
    velocity_max_horizon: int = 1,
) -> torch.Tensor:
    """
    Computes Cartesian velocity from position data over time.
    
    When velocity_max_horizon=1, uses simple forward difference.
    When velocity_max_horizon>1, uses multi-horizon minimum to filter noise.
    """
    T = batched_robot_pos.shape[0]
    if T < 2:
        return torch.zeros_like(batched_robot_pos)
    
    # LINES 1183-1200: Compute velocities for each horizon
    velocities = []
    for horizon in range(1, velocity_max_horizon + 1):
        dt = horizon / fps
        vel = torch.zeros_like(batched_robot_pos)
        
        if T > horizon:
            # LINE 1190-1192: Forward difference formula
            # v[t] = (pos[t+horizon] - pos[t]) / dt
            vel[:-horizon] = (
                batched_robot_pos[horizon:] - batched_robot_pos[:-horizon]
            ) / dt
            
            # LINES 1193-1194: Fill last frames with repeated last velocity
            vel[-horizon:] = vel[-horizon - 1].unsqueeze(0).expand(horizon, -1, -1)
        
        velocities.append(vel)
```

**Key Points:**
- **Frame-origin velocity computation**: Velocity vectors computed from frame position differences
- **Forward difference formula**: `v[t] = (pos[t+h] - pos[t]) / (h/fps)`
- **Multi-horizon noise filtering option** (lines 1202-1226): Selects velocity with minimum magnitude across multiple horizons to filter noise

### 4.2 Angular Velocity from Rotation Matrices (pose_lib.py, Lines 1230-1328)

```python
# Lines 1230-1328: Compute angular velocities from rotations
def compute_angular_velocity(
    batched_robot_rot_mats: torch.Tensor,  # [T, Nb, 3, 3]
    fps: int,
    velocity_max_horizon: int = 1,
) -> torch.Tensor:
    """
    Computes angular velocity from rotation matrices over time.
    """
    # LINES 1268-1291: For each horizon, compute rotation difference
    for horizon in range(1, velocity_max_horizon + 1):
        dt = horizon / fps
        ang_vel = torch.zeros(..., device=device, dtype=dtype)
        
        if T > horizon:
            # LINES 1279-1280: Get quaternions at t and t+horizon
            quat_t = batched_robot_quats[:-horizon]
            quat_t_plus_h = batched_robot_quats[horizon:]
            
            # LINES 1283-1285: Compute rotation difference quaternion
            # q_diff = q_{t+h} * q_t^{-1}
            quat_t_inv = quat_conjugate(quat_t, w_last=True)
            diff_quat = quat_mul_norm(quat_t_plus_h, quat_t_inv, w_last=True)
            
            # LINES 1287-1291: Extract angle and axis, compute angular velocity
            # ω = axis * angle / dt
            diff_angle, diff_axis = quat_angle_axis(diff_quat, w_last=True)
            ang_vel_valid = diff_axis * diff_angle.unsqueeze(-1) / dt
```

**Key Formula:**
```
ω[t] = axis * angle / dt
where:
  q_diff = q(t+h) * inverse(q(t))
  (axis, angle) = quat_angle_axis(q_diff)
```

### 4.3 Combined Kinematics Velocity Computation (pose_lib.py, Lines 1334-1359)

```python
# Lines 1334-1359: High-level function for computing both velocity types
def compute_kinematics_velocities(
    batched_robot_pos: torch.Tensor,      # [T, Nb, 3]
    batched_robot_rot_mats: torch.Tensor, # [T, Nb, 3, 3]
    fps: int,
    velocity_max_horizon: int = 3,        # Default uses 3-horizon filtering
) -> Tuple[torch.Tensor, torch.Tensor]:
    lin_vel = compute_cartesian_velocity(batched_robot_pos, fps, velocity_max_horizon)
    ang_vel = compute_angular_velocity(
        batched_robot_rot_mats, fps, velocity_max_horizon
    )
    return lin_vel, ang_vel
```

### 4.4 Full FK with Velocities (pose_lib.py, Lines 1405-1463)

```python
# Lines 1405-1463: Main function computing FK and velocities
def fk_from_transforms_with_velocities(
    kinematic_info: KinematicInfo,
    root_pos: torch.Tensor,
    joint_rot_mats: torch.Tensor,
    fps: Optional[int] = None,
    compute_velocities: bool = True,
    velocity_max_horizon: int = 3,
) -> RobotState:
    # LINE 1437-1439: Compute forward kinematics (positions and rotations)
    world_pos, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos, joint_rot_mats
    )
    
    # LINES 1455-1461: Compute velocities from poses
    if compute_velocities and root_pos.shape[0] > 1:
        assert fps is not None, "fps is required when compute_velocities is True"
        
        # LINE 1457-1459: Compute from time series of positions and rotations
        lin_vel, ang_vel = compute_kinematics_velocities(
            world_pos, world_rot_mat, fps, velocity_max_horizon
        )
        result.rigid_body_vel = lin_vel      # [T, Nb, 3]
        result.rigid_body_ang_vel = ang_vel  # [T, Nb, 3]
```

---

## 5. Position Type: Frame-Origin vs COM

### 5.1 Position Origin Definition

From analysis of the code, **ProtoMotions positions are frame-origin positions**, matching MuJoCo's `data.xpos`:

**In motion_lib.py, Line 754:**
```python
# First frame's root position for this motion
first_frame_root_pos = self.gts[start_idx, 0, :]  # [3] - root body position
```

**In pose_lib.py, Lines 1436-1439 (FK computation):**
```python
# Compute forward kinematics from qpos
# Returns world_pos which are frame positions (like MuJoCo xpos)
world_pos, world_rot_mat = compute_forward_kinematics_from_transforms(
    kinematic_info, root_pos, joint_rot_mats
)
```

### 5.2 Velocity Computation Confirmation

**In pose_lib.py, Lines 1190-1192 (Linear velocity from positions):**
```python
# Velocity computed directly from position differences
# These are frame-origin velocities
vel[:-horizon] = (
    batched_robot_pos[horizon:] - batched_robot_pos[:-horizon]
) / dt
```

Since `batched_robot_pos` contains frame-origin positions (from FK), the velocities computed are **frame-origin velocities**, not COM velocities.

---

## 6. Data Flow Summary

```
Motion Generation (T2M Model)
    ↓
Output: rigid_body_pos, rigid_body_vel, rigid_body_ang_vel, ...
    ↓
Save as .motion file via torch.save()
    ↓
MotionLib._load_motions() (Line 450)
    └─ Load .motion file
    └─ Create RobotState from dict (applies field mapping)
    └─ Extract rigid_body_vel → gvs [T, Nb, 3]
    └─ Extract rigid_body_ang_vel → gavs [T, Nb, 3]
    ↓
Concatenate all motions (Lines 461-473)
    └─ gvs: [total_frames, num_bodies, 3]
    └─ gavs: [total_frames, num_bodies, 3]
    ↓
Option A: Use directly in training
Option B: Save to .pt file (save_to_file, Line 599)
    └─ Packaged file contains gvs, gavs tensors
    ↓
During training: get_motion_state(motion_ids, motion_times)
    └─ Retrieve exact frame or interpolate
    └─ Return RobotState with rigid_body_vel, rigid_body_ang_vel
```

---

## 7. Critical Velocity Mismatch Debugging Checklist

Based on the code analysis, if you're debugging velocity mismatches:

### 7.1 **Verify Position Type Match** (CRITICAL)
- ✓ Check if your T2M model outputs **frame-origin positions** (like MuJoCo xpos)
- ✗ If COM positions: Convert to frame-origin using inertial offset
- Reference: motion_lib.py Line 60 mapping `gts` = `rigid_body_pos` (frame-origin)

### 7.2 **Verify Velocity Computation**
- ✓ If loading velocities from file: They must be **frame-origin velocities**
- ✓ If computing from positions: Use finite differences (pose_lib.py Lines 1190-1192)
- Formula: `v[t] = (pos[t+1] - pos[t]) * fps`

### 7.3 **Interpolation Behavior**
- ✓ Velocities interpolated **linearly** between frames (not SLERP)
- See: motion_lib.py Lines 341-344
- May cause discontinuities if velocities don't smoothly vary

### 7.4 **Multi-Horizon Noise Filtering**
- Default `velocity_max_horizon=3` filters noise but changes velocities
- Set to 1 for exact finite differences
- See: pose_lib.py Lines 1203-1226

### 7.5 **Frame Index Boundary Conditions**
- Last frames repeat velocity from frame T-2 (Lines 1193-1194)
- This can cause velocity artifacts near motion end

---

## 8. Exact Code References Summary

| Concept | File | Lines | Key Finding |
|---------|------|-------|------------|
| **Velocity field definition** | motion_lib.py | 113-116 | `gvs`, `gavs` are float32 tensors |
| **Field mapping** | motion_lib.py | 59-66 | Maps file's `rigid_body_vel` → `gvs` |
| **Load from file** | motion_lib.py | 450-453 | `torch.load()` then `RobotState.from_dict()` |
| **Concatenate** | motion_lib.py | 461-473 | Concatenate all motion velocities |
| **Save packaged** | motion_lib.py | 576-600 | Save `gvs`, `gavs` to .pt file |
| **Load packaged** | motion_lib.py | 602-616 | Direct `setattr()` from loaded dict |
| **Extract at frame** | motion_lib.py | 407-414 | Index into `gvs[fl]` array |
| **Interpolate** | motion_lib.py | 341-344 | Linear interpolation via `interpolate_pos()` |
| **Linear velocity computation** | pose_lib.py | 1190-1192 | Forward difference: `(pos[t+h] - pos[t])/dt` |
| **Angular velocity computation** | pose_lib.py | 1287-1291 | `ω = axis * angle / dt` from quaternion diff |
| **Multi-horizon filtering** | pose_lib.py | 1203-1226 | Select minimum magnitude velocity across horizons |

---

## 9. Conclusion

**ProtoMotions stores and uses FRAME-ORIGIN VELOCITIES**, not COM velocities. These are:

1. **Stored as:** Float32 tensors `gvs` [total_frames, num_bodies, 3] and `gavs` [total_frames, num_bodies, 3]
2. **Loaded from:** .motion files via `torch.load()` + `RobotState.from_dict()`
3. **Computed via:** Finite differences from frame-origin positions (if generating new)
4. **Interpolated:** Linearly between frames during training
5. **Origin:** Frame-origin (like MuJoCo xpos), NOT center-of-mass

Any velocity mismatch likely stems from:
- Using COM velocities instead of frame-origin velocities
- Different finite difference horizons (multi-horizon filtering)
- Interpolation artifacts
- Boundary condition handling near motion ends

