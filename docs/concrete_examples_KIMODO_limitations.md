# Concrete Examples: KIMODO Limitations vs MotionCanvas Solutions

## Example 1: End-Effector Position Constraint (The FK Mismatch)

### Scenario
User provides: "Person's right hand at world position (1.2, 0.8, 0.5) at frame 10"

What they likely mean: "I want the hand to be AT this position in the final output."

### KIMODO Approach

**Training Phase 2**:
```python
# KIMODO learns from imputation:
x_target = [full motion with hand at (1.2, 0.8, 0.5)]
mask = [... 0s except at hand position dims [global_joints_positions] ...]
observed = mask * x_target + (1-mask) * 0

# At step t:
x_imputed = mask * x_target + (1-mask) * x_t
# = position imputed, rotation noisy

x_input = concat([x_imputed, mask])
x_pred = model(x_input)  # predicts clean x

# Loss includes FK term: FK(x_pred_rotation) vs x_target_position
# Model learns: "if position is constrained, rotation must be consistent"
```

**Inference**:
```python
for step in denoising_steps:
    x_t[hand_pos_dims] = (1.2, 0.8, 0.5)  # HARD SET
    x_t[hand_rot_dims] = noisy_value      # free to evolve
    
    # Problem: Next iteration, FK(evolved_rotation) ≠ (1.2, 0.8, 0.5)
    # Model was trained on examples where pos and rot were BOTH constrained
    # But at inference, only pos is constrained
    # This is a TRAIN-TEST MISMATCH
    
    x_pred = model(concat([x_t, mask]))
    x_{t-1} = denoise_step(x_pred)
```

**Result**: 
- Position stays at (1.2, 0.8, 0.5) ✓
- Rotation evolves to minimize denoise loss
- But FK(rotation) ≠ position after denoise step
- → Hand gets twisted/contorted
- → Several cm error visible in output (admitted in their CLAUDE.md line 498)

### MotionCanvas Approach

**Training**:
```python
# When observing hand position:
src_motion = hand at (1.2, 0.8, 0.5) + rest of motion
src_mask = [0 at hand_pos_dims, 1 elsewhere]

inactive = src_motion * (1 - src_mask)  # hand pos observed
reactive = 0 (completion task)
mask = src_mask

x_input = concat([x_t, inactive, reactive, mask])

# Model learns: "when mask=0, the dimension is observed"
# This is EXPLICIT in the conditioning
```

**Inference**:
```python
for step in ODE_steps:
    x_t = ODE_step(...)  # x_t evolves freely in UNCONSTRAINED dims
    
    # Conditional inputs stay the same:
    inactive[hand_pos_dims] = (1.2, 0.8, 0.5)
    mask[hand_pos_dims] = 0  # "this is observed"
    
    x_input = concat([x_t, inactive, reactive, mask])
    x_pred = model(x_input)
    
    # If hand_pos_dims were marked as observed,
    # model learned they should stay constant
    # No implicit need for rotation to "chase" position
```

**Result**:
- Model output naturally respects observed dimensions
- No train-test mismatch
- Hand position consistent

**Key difference**: MotionCanvas **explicitly signals** which dimensions were observed. KIMODO requires the model to **infer** this from the mask pattern alone.

---

## Example 2: Ankle Height Constraint (Granularity Problem)

### Scenario
User wants: "Generate walking motion, but keep both ankles at ground level (Y=0)"

In terms of motion representation (SMPL-22):
- Each ankle has 3D position: [X, Y, Z]
- Ankle Y should be 0 (ground level)
- Ankle X, Z should be free to generate naturally

### KIMODO Approach

**Can KIMODO do this?**

Looking at KIMODO's 5 constraint types (from CLAUDE.md §1):
1. `Root2DConstraintSet`: Only root X, Z (not ankle)
2. `Root Y Height Constraint`: Only root Y (not ankle)
3. `Global Root Heading Constraint`: Only heading angle
4. `Global Joint Rotations Constraint`: Full 6D rotation per joint
5. `Global Joint Positions Constraint`: Full 3D position per joint

**Option A**: Use `Global Joint Positions Constraint` on ankles
```python
constraint = GlobalJointPositionsConstraint(
    joint_names=["L_Ankle", "R_Ankle"],
    global_joints_positions = [[X?, Y=0, Z?], ...]  # X, Z must still be specified
)
```
**Problem**: You must specify X and Z too. You can't say "generate these freely."

**Option B**: Manually impute only Y?
```python
observed_motion[frame, ankle_y_dim] = 0  # impute Y only
motion_mask[frame, ankle_y_dim] = 1

# But what about X, Z dims of ankle?
# Model sees mask=0 (not constrained)
# Model doesn't know "ankle position was actually observed in training"
# Result: X, Z generation doesn't respect the kinematic model
```

**Verdict**: KIMODO cannot elegantly express "fix Y, generate X,Z" per joint.

### MotionCanvas Approach

```python
# Create mask at dimension level:
src_mask = zeros(num_frames, 135)

# Mark only ankle Y dims as observed (let's say dims 10-11 for left/right ankle Y):
src_mask[:, 10] = 0  # L_Ankle Y: observed
src_mask[:, 11] = 0  # R_Ankle Y: observed
# All other dims (ankle X, Z, everything else) = 1 (generate)

src_motion[:, 10] = 0  # ground level
src_motion[:, 11] = 0  # ground level

# VACE input:
inactive = src_motion * (1 - src_mask)
# → inactive has ankle Y values, zeros everywhere else

reactive = 0  # completion task

# Model input:
x_input = concat([x_t, inactive, reactive, src_mask])

# Model learns: ankle Y is observed (mask=0), so must be stable
# Everything else is free to generate (mask=1)
```

**Result**: Ankle Y stays at 0, X,Z generate freely. ✓

**Key difference**: MotionCanvas operates at dimension level (135D for SMPL-22). KIMODO operates at joint/constraint-type level.

---

## Example 3: Partial Observations (The Ambiguity Problem)

### Scenario
Complex motion capture scenario:
- Frame 0: Full body observed (from captured footage)
- Frame 50: Only hand rotation observed (hand-held sensor)
- Frame 100: Only foot position observed (contact with ground)
- Frame 150: Nothing observed (pure generation)

### KIMODO Approach

**Can't express this cleanly.**

KIMODO's constraint types assume:
- Full-body: all joints pos+rot
- End-effector: specific joints pos+rot together
- Trajectory: root pos

Mixing partial observations across frames is not a first-class design pattern.

**Workaround**: Manually construct masks?
```python
observed_motion = zeros(T, 333)
motion_mask = zeros(T, 333)

# Frame 0: full body
observed_motion[0, :] = frame_0_full
motion_mask[0, :] = 1

# Frame 50: hand rotation only
# Hand rotation is dims [86+hand_joint*6 : 86+hand_joint*6+6]
observed_motion[50, hand_rot_dims] = hand_rot
motion_mask[50, hand_rot_dims] = 1

# Frame 100: foot position only
# Foot position is dims [5+foot_joint*3 : 5+foot_joint*3+3]
observed_motion[100, foot_pos_dims] = foot_pos
motion_mask[100, foot_pos_dims] = 1

# What about hand position at frame 50? It's NOT constrained.
# But model has never seen "hand rotation observed, position free" during training
# → Out-of-distribution
```

**Problem**: This is not a predefined constraint type. It falls outside KIMODO's training distribution.

### MotionCanvas Approach

```python
# Simply create a mask at dimension-level granularity:

src_mask = ones(T, 135)  # default: generate everything

# Frame 0: full body observed
src_mask[0, :] = 0

# Frame 50: hand rotation only (let's say dims 66-71 for one hand's 6D rot)
src_mask[50, 66:72] = 0  # observed

# Frame 100: foot position only (let's say dims 30-32 for one foot's 3D pos)
src_mask[100, 30:33] = 0  # observed

# Frame 150: nothing
# src_mask[150, :] stays 1  (all generate)

# Build VACE inputs:
for frame in range(T):
    inactive[frame] = src_motion[frame] * (1 - src_mask[frame])
    reactive[frame] = 0  # completion
```

**Why this works**: MotionCanvas was **trained on these patterns** (M1: random cell, M7: scattered joint). The rank-K Boolean prior ensures all these patterns are in the training distribution.

**Result**: Works naturally. No out-of-distribution problem.

---

## Example 4: Motion Editing (Not Even Defined in KIMODO)

### Scenario
User has a captured motion that's "jittery" or "too stiff." They want: "Smooth out the motion while keeping the general trajectory."

This is **motion editing**, not completion.

### KIMODO Approach

**Not designed for this.**

KIMODO Phase 2 learns: "Impute constraints, denoise the rest."
- All tasks are: observe some dims, generate others
- No explicit mechanism for "improve quality of observed region"

To support editing, KIMODO would need:
1. A new training phase? (Phase 3: editing with quality improvement)
2. A separate model? (Editing-specific denoiser)
3. A different loss? (Quality-aware loss function)

Not part of the current design.

### MotionCanvas Approach

```python
# Same architecture, different reactive channel:

# For COMPLETION (e.g., inpainting):
reactive = 0  # or zeros_like(src_motion)
# Model learns: "generate only where mask=1"

# For EDITING (e.g., quality improvement):
reactive = LQ_motion  # low-quality motion in mask=1 regions
# Model learns: "improve the quality of this degraded motion"
```

**Training**:
```python
# Same batch, conditional logic switches the reactive channel:

if task == "completion":
    reactive = 0
elif task == "editing":
    reactive = corrupted_motion  # add jitter, remove high-freq, etc.
    
x_input = concat([x_t, inactive, reactive, src_mask])
x_pred = model(x_input)

# Loss computed on the same model
```

**Result**: 
- Single model handles both completion and editing
- Reactive channel carries task-specific information
- No separate architecture needed

---

## Summary Table: Can You Express These?

| Task | KIMODO | MotionCanvas | KIMODO Issue |
|------|--------|-----|---------|
| Full-body keyframe (frames 10, 50) | ✅ | ✅ | Tied |
| End-effector position + rotation | ✅ | ✅ | Tied |
| End-effector position ONLY | ⚠️ | ✅ | FK mismatch + ambiguity |
| Ankle Y constraint, XZ free | ❌ | ✅ | Joint-level granularity |
| Frame 0: full body; Frame 50: hand rot only | ❌ | ✅ | Out-of-dist; not a predefined type |
| Motion quality improvement (editing) | ❌ | ✅ | Not designed; no reactive channel |
| Sparse random missing dims | ❌ | ✅ | No explicit partial obs signal |

---

## Key Takeaway

**KIMODO's imputation**: "Here's the motion you want. Denoise the parts I didn't specify."
- Works great for: keyframes, trajectories, full-body specs
- Breaks down for: partial observations, per-dim control, editing

**MotionCanvas VACE**: "Here's what I observed. Here's what I want you to improve. Here's which dims are which."
- Works great for: everything, because observation model is explicit

The difference isn't "one is better," it's **fundamentally different information flow**:
- Imputation: observation information encoded in mask pattern (implicit)
- VACE: observation information encoded in mask semantics + separate channels (explicit)

For a scientific paper, this is the story to tell.

