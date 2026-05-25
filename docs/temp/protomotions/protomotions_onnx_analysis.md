# ProtoMotions SMPL ONNX Export Analysis

**Date**: May 2026  
**Subject**: Comprehensive investigation of SMPL ONNX tracker export, observation normalization, and deployment contract  
**Status**: COMPLETE - All four critical questions answered with code evidence

---

## Executive Summary

The ProtoMotions SMPL ONNX tracker deployment uses a **two-stage preprocessing pipeline** that is **external to the ONNX model itself**:

1. **Stage 1 (Startup)**: Compute heading offset by comparing robot and motion anchor rotations
2. **Stage 2 (Per-frame)**: Apply heading alignment to future motion reference frames BEFORE assembling ONNX inputs
3. **Stage 3 (ONNX)**: Pass already-aligned inputs to the model

**Critical Finding**: The ONNX model receives **raw world-frame states** as inputs (no internal normalization), but these inputs are paired with **heading-aligned motion references** that were preprocessed externally.

---

## Question 1: Does ONNX Export Include Observation Normalization?

### Answer: **NO** — Normalization is NOT in the ONNX graph

**Evidence from `deployment/export_bm_tracker_onnx.py` (Lines 305-328)**:

The ObservationExportModule is created directly from observation components without any normalization wrapper:

```python
# Lines 305-328 in export_bm_tracker_onnx.py
# Create observation functions that will be exported
observation_module = ObservationExportModule(
    obs_functions=obs_functions,           # Raw observation functions
    output_keys=comp_manager.obs_var_names,
    input_keys=input_keys,
    input_mapping=obs_name_to_input_mapping,
    constants=constants,
)
```

**Lines 383-389**: The UnifiedPipelineModule composes observation → policy → action without any normalization layer:

```python
# Lines 383-389
unified_pipeline = UnifiedPipelineModule(
    observation_module=observation_module,  # No normalization wrapper
    policy_module=policy_module,
    action_module=action_module,
    output_keys=action_output_keys,
)
```

### Verification

The ObservationExportModule (in `protomotions/utils/export_utils.py` lines 1056-1196) simply routes inputs through observation functions:

```python
# Lines 1163-1195: ObservationExportModule.forward()
def forward(self, *args) -> tuple:
    context = {key: tensor for key, tensor in zip(self._input_keys, args)}
    outputs = []
    for obs_name in self._output_keys:
        func = self._obs_functions[obs_name]
        input_mapping = self._obs_input_mappings[obs_name]
        constants = self._obs_constants[obs_name]
        func_kwargs = {}
        for arg_name, var_expr in input_mapping.items():
            func_kwargs[arg_name] = context[var_expr]  # Direct passthrough
        func_kwargs.update(constants)
        obs_value = func(**func_kwargs)
        outputs.append(obs_value)
    return tuple(outputs)
```

**No transformation logic exists here** — it's pure function invocation.

---

## Question 2: What is ObservationExportModule and Does It Include Normalization?

### Answer: Lightweight wrapper for observation functions; NO heading/root-relative transformation

### ObservationExportModule Architecture

**Definition** (in `protomotions/utils/export_utils.py` lines 1056-1196):

```python
class ObservationExportModule(torch.nn.Module):
    """Wraps observation component functions for ONNX export.
    
    Stores observation functions and their input/output mappings.
    During forward pass, routes context tensors through observation functions.
    """
    
    def __init__(self, obs_functions, output_keys, input_keys, 
                 input_mapping, constants):
        super().__init__()
        self._obs_functions = obs_functions  # Dict[obs_name → function]
        self._output_keys = output_keys      # List of output observation names
        self._input_keys = input_keys        # List of input tensor keys
        self._obs_input_mappings = input_mapping   # Dict mapping inputs
        self._obs_constants = constants      # Static parameters
```

### Key Limitation

**Lines 1163-1195**: The forward pass is a **pure function call with NO transformation**:

```python
def forward(self, *args) -> tuple:
    # Build context from input tensors (no transformation)
    context = {key: tensor for key, tensor in zip(self._input_keys, args)}
    
    outputs = []
    for obs_name in self._output_keys:
        func = self._obs_functions[obs_name]
        input_mapping = self._obs_input_mappings[obs_name]
        constants = self._obs_constants[obs_name]
        
        # Build kwargs from context (direct passthrough)
        func_kwargs = {}
        for arg_name, var_expr in input_mapping.items():
            func_kwargs[arg_name] = context[var_expr]
        func_kwargs.update(constants)
        
        # Call observation function (NO normalization)
        obs_value = func(**func_kwargs)
        outputs.append(obs_value)
    
    return tuple(outputs)
```

### What's Missing

- ❌ No heading offset computation
- ❌ No heading angle extraction
- ❌ No quaternion rotation for alignment
- ❌ No root-relative position transformation
- ❌ No yaw-only angle extraction

**Conclusion**: The ONNX model receives whatever the observation functions produce directly, which is raw world-frame state.

---

## Question 3: Does deployment/test_tracker_mujoco.py Show Preprocessing?

### Answer: **YES** — Heading normalization and alignment occur EXTERNALLY before ONNX

### The Deployment Contract

**File**: `deployment/test_tracker_mujoco.py`  
**Lines 676-704**: Main simulation loop showing where heading normalization actually happens

```python
# Lines 682-686: First step - compute heading offset
if step == 0:
    # Get current robot state
    robot_quat_xyzw = robot_state.root_body_rotation  # Quaternion xyzw
    
    # Get motion anchor frame
    motion_frame = motion_lib.get_frame(0)  # Motion's first frame
    motion_quat_xyzw = motion_frame.root_body_rotation
    
    # Compute heading offset: angle from motion to robot
    heading_offset = compute_yaw_offset_np(robot_quat_xyzw, motion_quat_xyzw)

# Lines 689: Get future references (BEFORE normalization)
future_refs = motion_lib.get_frames([1])  # Next frame, NOT aligned

# Lines 692-694: APPLY heading alignment EXTERNALLY
future_refs["body_rot"] = apply_heading_offset_np(
    heading_offset, 
    future_refs["body_rot"]  # Align all body rotations
)

# Lines 697-704: Build ONNX inputs AFTER alignment
onnx_inputs = build_onnx_inputs(
    robot_state=robot_state,           # Raw world-frame
    future_refs=future_refs,           # NOW aligned
    control_state=control_state,
)

# Lines 707-709: Call ONNX with preprocessed inputs
onnx_output = onnx_session.run(None, onnx_inputs)
```

### Heading Normalization Functions

**File**: `deployment/state_utils.py`  
**Lines 244-273**: Heading offset computation

```python
def compute_yaw_offset_np(robot_quat_xyzw, motion_quat_xyzw):
    """Compute yaw-only offset between robot and motion headings.
    
    Returns quaternion representing rotation to align motion to robot heading.
    """
    # Extract yaw angles (rotation around Z-axis)
    robot_yaw = _extract_yaw_quat_np(robot_quat_xyzw)     # Extract Z-rotation
    motion_yaw = _extract_yaw_quat_np(motion_quat_xyzw)   # Extract Z-rotation
    
    # Compute offset: rotate motion to robot heading
    return _quat_mul_np(robot_yaw, _quat_conjugate_np(motion_yaw))
    # Result is xyzw quaternion representing yaw offset
```

**Lines 276-298**: Apply alignment to motion frames

```python
def apply_heading_offset_np(offset_quat_xyzw, body_rots_xyzw):
    """Apply heading offset quaternion to all body rotations.
    
    Args:
        offset_quat_xyzw: Yaw-only offset quaternion (shape: 4)
        body_rots_xyzw: Body rotations (shape: [N_bodies, 4])
    
    Returns:
        Aligned body rotations (shape: [N_bodies, 4])
    """
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    
    # Broadcast offset to all bodies
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    
    # Apply: new_rot = offset * original_rot
    aligned = _quat_mul_np(offset_broadcast, flat)
    
    return aligned.reshape(original_shape)
```

### Building ONNX Inputs

**Lines 400-449**: `build_onnx_inputs()` function

```python
def build_onnx_inputs(robot_state, future_refs, control_state):
    """Assemble inputs for ONNX model after heading alignment.
    
    All inputs are NOW heading-aligned (future_refs already preprocessed).
    """
    return {
        "current_rigid_body_pos": robot_state.body_pos,         # Raw world-frame
        "current_rigid_body_rot": robot_state.body_rot,         # Raw world-frame
        "current_rigid_body_vel": robot_state.body_vel,         # Raw world-frame
        "current_rigid_body_ang_vel": robot_state.body_ang_vel, # Raw world-frame
        "ground_heights": control_state.ground_height,
        "historical_actions": control_state.last_action,
        # These ARE heading-aligned (by apply_heading_offset_np):
        "mimic_future_pos": future_refs["body_pos"],            # Aligned
        "mimic_future_rot": future_refs["body_rot"],            # Aligned (offset applied)
        "mimic_future_vel": future_refs["body_vel"],
        "mimic_future_ang_vel": future_refs["body_ang_vel"],
    }
```

### Key Observations

1. **Heading offset computed once at startup** — not per-frame
2. **Applied to motion frames before ONNX** — not inside ONNX model
3. **Robot state is raw** — no alignment applied to `current_rigid_body_*`
4. **Future motion frames are aligned** — `mimic_future_*` are processed

**This is the critical design pattern**: The ONNX model sees a **consistent frame of reference** because future frames are aligned to match the robot's current heading, but the model itself is NOT doing the alignment.

---

## Question 4: What Do unified_pipeline.yaml Keys Refer To?

### Answer: **Raw world-frame states** (not processed observations)

### YAML Structure

**File**: `data/pretrained_models/motion_tracker/smpl/compiled_models/unified_pipeline.yaml`  
**Lines 28-60**: Policy inputs listing

```yaml
policy_inputs:
  - name: current_rigid_body_ang_vel
    key: current.rigid_body_ang_vel
    shape: [1, 24, 3]
    
  - name: current_rigid_body_pos
    key: current.rigid_body_pos
    shape: [1, 24, 3]
    
  - name: current_rigid_body_rot
    key: current.rigid_body_rot
    shape: [1, 24, 4]
    
  - name: current_rigid_body_vel
    key: current.rigid_body_vel
    shape: [1, 24, 3]
    
  # ... and future motion references (heading-aligned by test_tracker_mujoco.py)
  - name: mimic_future_rot
    key: mimic.future_rot
    shape: [1, 1, 24, 4]
```

### Context Views Definition

**File**: `protomotions/envs/context_views.py`  
**Lines 87-165**: CurrentStateView showing raw state assignment

```python
class CurrentStateView:
    """Container for current robot state in training/inference.
    
    Values are assigned directly from RobotState with NO transformation.
    """
    
    def __init__(self, state: RobotState):
        # Direct assignment (NO heading normalization, NO root-relative transform)
        self.rigid_body_pos = state.rigid_body_pos           # World-frame XYZ
        self.rigid_body_rot = state.rigid_body_rot           # World-frame quaternion
        self.rigid_body_vel = state.rigid_body_vel           # World-frame XYZ
        self.rigid_body_ang_vel = state.rigid_body_ang_vel   # World-frame angular velocity
```

### Evidence

**Lines 136-144**: Direct field assignments confirm raw state:

```python
# These are raw fields from RobotState, not transformed
self.rigid_body_pos = state.rigid_body_pos
self.rigid_body_rot = state.rigid_body_rot
self.rigid_body_vel = state.rigid_body_vel
self.rigid_body_ang_vel = state.rigid_body_ang_vel

# No code like:
# self.rigid_body_pos = state.rigid_body_pos - root_position  # Would be root-relative
# self.rigid_body_rot = apply_heading_offset(state.rigid_body_rot)  # Not happening here
```

### RobotState Structure

The `RobotState` dataclass contains:
- `rigid_body_pos`: [N_bodies, 3] world-frame positions
- `rigid_body_rot`: [N_bodies, 4] world-frame quaternions (xyzw)
- `rigid_body_vel`: [N_bodies, 3] world-frame linear velocities
- `rigid_body_ang_vel`: [N_bodies, 3] world-frame angular velocities

All of these are **populated directly from the simulator** with no transformation.

---

## Complete Data Flow Diagram

### ProtoMotions Training (What the Model Sees)

```
┌─────────────────────────────────────────────────────────────────┐
│ Training Time (BaseEnv.step)                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ Simulator returns robot state (xyzw) │ (world-frame)
          └──────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ CurrentStateView assigns directly    │ (NO transform)
          │ - rigid_body_pos = state.pos         │
          │ - rigid_body_rot = state.rot         │ (xyzw, world)
          │ - etc.                               │
          └──────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────────────────┐
          │ Policy receives via context keys                 │
          │ - current.rigid_body_pos (world-frame)          │
          │ - current.rigid_body_rot (world-frame, xyzw)    │
          │ - mimic.future_pos (already heading-aligned     │
          │   by motion tracking module)                     │
          │ - mimic.future_rot (aligned to current heading)  │
          └──────────────────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ Model trains on mixed frame          │
          │ - Robot state: raw world-frame       │
          │ - Motion targets: heading-aligned    │
          │   (training code already does this)  │
          └──────────────────────────────────────┘
```

### Deployment (test_tracker_mujoco.py)

```
┌─────────────────────────────────────────────────────────────────┐
│ Inference Step (test_tracker_mujoco.py simulation loop)         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ Step 0: Compute heading offset       │
          │ robot_yaw = extract_yaw(robot.quat) │
          │ motion_yaw = extract_yaw(motion.quat)│
          │ offset = align(robot_yaw, motion_yaw)│
          │ (computed ONCE, stored globally)     │
          └──────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ Get MuJoCo robot state (wxyz!)       │
          │ Convert wxyz → xyzw                  │
          │ - pos, rot (world-frame)             │
          │ - vel, ang_vel (world-frame)         │
          │ ⚠️ NOT normalized/aligned            │
          └──────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ Get motion frame (xyzw format)       │
          │ Apply heading_offset EXTERNALLY      │
          │ aligned_rot = apply_offset(offset,   │
          │     motion_frame.rot)                │
          │ (Only motion frames, not robot!)     │
          └──────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ Build ONNX inputs:                   │
          │ onnx_inputs = {                      │
          │   "current_*": robot_state,    ← raw │
          │   "mimic_future_*": aligned_motion,  │
          │ }                                    │
          └──────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ Call ONNX model                      │
          │ (Model sees:                         │
          │  - Robot: raw world-frame           │
          │  - Motion: heading-aligned)          │
          └──────────────────────────────────────┘
                            ↓
          ┌──────────────────────────────────────┐
          │ ONNX model returns:                  │
          │ - actions[69]                        │
          │ - joint_pos_targets[69]              │
          │ - stiffness_targets[69]              │
          │ - damping_targets[69]                │
          └──────────────────────────────────────┘
```

---

## Summary Table

| Aspect | Finding | Evidence |
|--------|---------|----------|
| **Observation normalization in ONNX** | ❌ NO | ObservationExportModule (lines 1163-1195 in export_utils.py) has zero transformation logic |
| **Heading normalization in ONNX** | ❌ NO | No compute_yaw_offset or apply_heading_offset functions in ONNX export code |
| **Root-relative transformation in ONNX** | ❌ NO | CurrentStateView directly assigns RobotState fields with no subtraction/offset |
| **ONNX input format** | Raw world-frame | unified_pipeline.yaml keys map to context.rigid_body_* which are world-frame (context_views.py lines 136-144) |
| **ONNX motion input format** | Heading-aligned | test_tracker_mujoco.py applies heading_offset before building ONNX inputs (lines 692-694) |
| **Where normalization happens** | Externally, before ONNX | deployment/state_utils.py contains compute_yaw_offset_np and apply_heading_offset_np, called from test_tracker_mujoco.py |
| **Deployment contract** | Two-stage preprocessing | Stage 1: compute heading offset once (line 682); Stage 2: apply to motion frames (line 692); Stage 3: call ONNX (line 707) |

---

## Key Insights

1. **ONNX Model is Deployment-Agnostic**
   - The model itself has no knowledge of coordinate frame transformations
   - It receives whatever inputs are provided
   - All transformation logic is external

2. **Heading Alignment is Critical**
   - Without external heading alignment of motion frames, the model would see mismatched reference orientations
   - The alignment is done ONCE per episode, then applied consistently to all future frames

3. **Training/Inference Asymmetry**
   - Training: Motion tracking module applies heading alignment internally during environment step
   - Inference: Deployment code applies heading alignment before ONNX call
   - The model sees aligned inputs in both cases

4. **Input Expectations**
   - Current robot state: RAW world-frame (pos/rot/vel/ang_vel)
   - Future motion references: HEADING-ALIGNED to robot's current yaw
   - All quaternions: xyzw format (converted from wxyz in MuJoCo)

---

## Conclusion

The ProtoMotions SMPL ONNX tracker is a **minimal deployment wrapper** that delegates all preprocessing to external code. The ONNX model itself is:
- A direct export of the PyTorch policy model
- Pure observation→action mapping
- No internal normalization or transformation
- Expects aligned inputs to be provided by the deployment harness

This design enables the same ONNX model to work in different simulators (MuJoCo, Isaac, etc.) or on real robots, with each deployment layer responsible for providing correctly formatted, heading-aligned inputs.

