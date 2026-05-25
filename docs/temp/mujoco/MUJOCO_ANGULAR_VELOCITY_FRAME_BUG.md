# CRITICAL BUG: MuJoCo Angular Velocity Frame Mismatch During Resets

**Status**: CONFIRMED  
**Severity**: HIGH  
**Affects**: ProtoMotions MuJoCo Simulator Physics Accuracy  

## Summary

The ProtoMotions MuJoCo simulator has a **frame convention mismatch** in root angular velocity during environment resets. Angular velocity is read from MuJoCo in **WORLD FRAME** but assigned to `qvel[3:6]` which expects **BODY-LOCAL FRAME**, causing physically incorrect simulations when the robot root is rotated.

## Root Cause Analysis

### The Bug in 5 Steps

**1. Reading: World Frame (Correct)**
```python
# File: protomotions/simulator/mujoco/simulator.py, line 968
body_ang_vel = self.data.cvel[1 : 1 + nb, 0:3].copy()  # ← WORLD FRAME
```
MuJoCo's `data.cvel` returns angular velocity in **world frame** (documented in deployment/state_utils.py line 70).

**2. Storage: Still World Frame**
```python
# File: protomotions/simulator/mujoco/simulator.py, line 978
return RobotState(
    rigid_body_ang_vel=_to_torch_f32(body_ang_vel).unsqueeze(0),  # Still world frame
    state_conversion=StateConversion.SIMULATOR,
)
```

**3. Extraction to ResetState: Still World Frame**
```python
# File: protomotions/simulator/base_simulator/simulator_state.py, line 345-348
@property
def root_ang_vel(self) -> Optional[torch.Tensor]:
    if self.rigid_body_ang_vel is not None:
        return self.rigid_body_ang_vel[:, 0, :]  # Still world frame
    return None
```

**4. Reset Conversion: NO CONVERSION APPLIED**
```python
# File: protomotions/simulator/base_simulator/simulator_state.py, line 713-720
def convert_to_sim(self, conversion: DataConversionMapping) -> "ResetState":
    if self.state_conversion == StateConversion.COMMON:
        if not conversion.sim_w_last:
            self._convert_helper_rot(rotations.xyzw_to_wxyz, "root_rot")
        self._convert_helper(conversion.dof_convert_to_sim, "dof_pos")
        self._convert_helper(conversion.dof_convert_to_sim, "dof_vel")
    # ⚠️ NOTE: root_ang_vel is NOT handled here!
    return self
```

**5. Reset Assignment: WRONG FRAME**
```python
# File: protomotions/simulator/mujoco/simulator.py, line 695
self.data.qvel[3:6] = root_ang_vel  # ← ASSIGNING WORLD FRAME TO LOCAL FRAME SLOT
```

MuJoCo's `qvel[3:6]` for a free joint expects **body-local frame** angular velocity.

### Why This is a Bug

From MuJoCo documentation and ProtoMotions' own state_utils.py (line 32-44):

| Source | Frame | Action |
|--------|-------|--------|
| `data.cvel` | World | Apply `compute_root_local_ang_vel_np()` to convert |
| `data.qvel[3:6]` | **Local** | Use directly |

**The code reads from `data.cvel` (world frame) but never converts before assigning to `qvel[3:6]` (expects local frame).**

## Impact

### When the Bug Manifests
1. **Non-identity root rotations**: When the robot's root body is rotated relative to the world axes
2. **Mid-motion resets**: Resetting from states where the body is tilted/rotated
3. **Physics divergence**: After reset, the physics becomes unphysical due to misaligned angular velocities

### When the Bug is Hidden
1. **Identity rotations**: When root quaternion ≈ [0, 0, 0, 1], local and world frames align
2. **Upright-only scenarios**: Policies trained on mostly upright walking won't hit rotated states
3. **No explicit angular velocity**: If policies don't use or set angular velocity explicitly

### Observable Symptoms
- Unexpected torques/forces after reset
- Inconsistent rollout vs. reset behavior
- RL agent confusion due to observation discontinuities
- Non-deterministic behavior at high body rotation angles

## Code Locations

| File | Line | Issue |
|------|------|-------|
| `protomotions/simulator/mujoco/simulator.py` | 968 | Reading in world frame |
| `protomotions/simulator/mujoco/simulator.py` | 978 | Storing without conversion |
| `protomotions/simulator/mujoco/simulator.py` | 685 | Extracting from ResetState |
| `protomotions/simulator/mujoco/simulator.py` | 695 | **Assigning to wrong frame** ← BUG |
| `protomotions/simulator/base_simulator/simulator_state.py` | 713-720 | No conversion in `convert_to_sim()` |
| `deployment/state_utils.py` | 130-166 | Conversion function exists but **never called** |

## The Solution

### Recommended Fix (Option 1: Convert During Reset)

In `protomotions/simulator/mujoco/simulator.py`, modify `_set_simulator_env_state()`:

```python
def _set_simulator_env_state(
    self,
    new_states: ResetState,
    new_object_states: Optional[ObjectState] = None,
    env_ids: Optional[torch.Tensor] = None,
) -> None:
    """Set simulator state (qpos/qvel) and recompute FK."""
    root_pos = new_states.root_pos[0].cpu().numpy().copy()
    root_rot = new_states.root_rot[0].cpu().numpy()
    root_vel = new_states.root_vel[0].cpu().numpy()
    root_ang_vel = new_states.root_ang_vel[0].cpu().numpy()  # WORLD FRAME
    dof_pos = new_states.dof_pos[0].cpu().numpy()
    dof_vel = new_states.dof_vel[0].cpu().numpy()

    if self._has_free_joint:
        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_rot  # wxyz
        self.data.qpos[7 : 7 + self._num_actuated_dofs] = dof_pos

        self.data.qvel[0:3] = root_vel
        
        # FIX: Convert angular velocity from world frame to body local frame
        from protomotions.utils import rotations
        root_rot_xyzw = rotations.wxyz_to_xyzw(root_rot)
        root_ang_vel_local = rotations.quat_rotate_inverse(root_rot_xyzw, root_ang_vel)
        self.data.qvel[3:6] = root_ang_vel_local
        
        self.data.qvel[6 : 6 + self._num_actuated_dofs] = dof_vel
    else:
        self.data.qpos[: self._num_actuated_dofs] = dof_pos
        self.data.qvel[: self._num_actuated_dofs] = dof_vel

    # Clear forces
    self.data.ctrl[:] = 0.0
    self.data.qfrc_applied[:] = 0.0

    # Recompute forward kinematics
    mujoco.mj_forward(self.model, self.data)
```

### Alternative Fix (Option 2: Fix at Read Time)

Modify `_get_simulator_bodies_state()` to return local-frame angular velocity:

```python
def _get_simulator_bodies_state(self, env_ids: Optional[torch.Tensor] = None) -> RobotState:
    from protomotions.utils import rotations
    
    nb = self._num_robot_bodies
    body_pos = self.data.xpos[1 : 1 + nb, :].copy()
    body_rot = self.data.xquat[1 : 1 + nb, :].copy()  # wxyz from MuJoCo

    # cvel is [ang_vel(3), lin_vel(3)] - ANG_VEL IS IN WORLD FRAME
    body_ang_vel_world = self.data.cvel[1 : 1 + nb, 0:3].copy()
    body_vel_frame = self.data.cvel[1 : 1 + nb, 3:6].copy()

    # Apply COM offset velocity correction
    body_vel = self._apply_com_velocity_correction(body_vel_frame, body_ang_vel_world, body_rot)

    # Convert angular velocity from world to local frame for each body
    body_rot_xyzw = rotations.wxyz_to_xyzw(body_rot)
    body_ang_vel_local = rotations.quat_rotate_inverse(body_rot_xyzw, body_ang_vel_world)

    return RobotState(
        rigid_body_pos=_to_torch_f32(body_pos).unsqueeze(0),
        rigid_body_rot=_to_torch_f32(body_rot).unsqueeze(0),
        rigid_body_vel=_to_torch_f32(body_vel).unsqueeze(0),
        rigid_body_ang_vel=_to_torch_f32(body_ang_vel_local).unsqueeze(0),  # ← NOW LOCAL
        state_conversion=StateConversion.SIMULATOR,
    )
```

### Alternative Fix (Option 3: Fix in ResetState Conversion)

Add frame conversion to `ResetState.convert_to_sim()`.

## Testing Strategy

Create `protomotions/tests/test_mujoco_angular_velocity_frame.py`:

```python
def test_mujoco_angular_velocity_reset_consistency():
    """Verify angular velocity frame consistency across reset cycle."""
    # 1. Create initial state with rotated root
    # 2. Run simulation
    # 3. Record state (with world-frame angular velocity from data.cvel)
    # 4. Reset to recorded state
    # 5. Run simulation again
    # 6. Verify that the two rollouts diverge minimally (should match)
```

## Cross-Simulator Comparison

### IsaacGym
IsaacGym handles this correctly because:
- Reads `rigid_body_state` tensor from PhysX
- IsaacGym's rigid_body_state already provides body-frame angular velocities
- Reset assignment works because the frame is already correct

### MuJoCo (Current ProtoMotions)
- Reads world-frame from `data.cvel`
- Stores as-is without conversion
- Assigns to local-frame slot without conversion ← **BUG**

## References

1. **MuJoCo Manual**: Free joint dynamics
2. **ProtoMotions Documentation**: `deployment/state_utils.py` lines 32-44
3. **Existing Conversion Function**: `compute_root_local_ang_vel_np()` in `deployment/state_utils.py` lines 130-166

## Checklist

- [ ] Apply frame conversion in `_set_simulator_env_state()` before line 695
- [ ] Add unit test for angular velocity frame correctness
- [ ] Verify with non-identity root rotations
- [ ] Test mid-motion resets (e.g., humanoid tilted at 45°)
- [ ] Compare MuJoCo vs IsaacGym reset behavior
- [ ] Update comments to clarify frame semantics
