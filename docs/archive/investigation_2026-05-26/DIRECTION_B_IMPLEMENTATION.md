# Direction B: Training RL Tracker on T2M-Generated Motions

## Executive Summary

To train a ProtoMotions RL tracker on T2M-generated motions with MuJoCo backend:

### Required Command

```bash
cd ref_repo/ProtoMotions

python protomotions/train_agent.py \
    --robot-name smpl \
    --simulator mujoco \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_t2m_tracker \
    --motion-file /path/to/t2m_motions.pt \
    --num-envs 1 \
    --batch-size 32 \
    --ngpu 1
```

### Motion Data Format Requirement

Your motion file **must be a `.pt` PyTorch dictionary** with these exact fields:

```python
torch.load("t2m_motions.pt") → dict containing:
{
    # Per-frame data (concatenated across all motions):
    "gts":  torch.Tensor([total_frames, 24, 3], dtype=float32),  # Global body positions
    "grs":  torch.Tensor([total_frames, 24, 4], dtype=float32),  # Quaternions (w_last=[x,y,z,w])
    "gvs":  torch.Tensor([total_frames, 24, 3], dtype=float32),  # Global velocities
    "gavs": torch.Tensor([total_frames, 24, 3], dtype=float32),  # Angular velocities
    "dps":  torch.Tensor([total_frames, 69], dtype=float32),     # DOF positions
    "dvs":  torch.Tensor([total_frames, 69], dtype=float32),     # DOF velocities
    
    # Per-motion metadata:
    "motion_num_frames":  torch.Tensor([num_motions], dtype=long),    # [120, 120, 100, ...]
    "motion_lengths":     torch.Tensor([num_motions], dtype=float32), # [4.0, 4.0, 3.33, ...]
    "motion_dt":          torch.Tensor([num_motions], dtype=float32), # [0.0333, 0.0333, ...]
    "length_starts":      torch.Tensor([num_motions], dtype=long),    # [0, 120, 240, 340]
    "motion_weights":     torch.Tensor([num_motions], dtype=float32), # [1.0, 1.0, 1.0, ...]
    "motion_files":       tuple,                                       # ("t2m_1.pt", "t2m_2.pt", ...)
    
    # Optional:
    "contacts":           torch.Tensor([total_frames, 4], dtype=float32)  # Contact labels
}
```

## Technical Specifications

### SMPL Humanoid Geometry
- **Bodies**: 24 (1 root + 23 joints)
- **DOFs**: 69 (23 joints × 3 DOF each, using quaternions/exp_map)
- **Body ordering**: Root (0), then SMPL kinematic chain (1-23)

### Data Constraints
- All position/velocity/rotation tensors must be `float32`
- Quaternions must be **w_last format**: `[x, y, z, w]`
- Cumulative frame offsets in `length_starts` must match actual concatenated data
- FPS: typically 30 (adjust `motion_dt` if different)

### Example: Single 120-frame T2M Motion

```python
import torch

# T2M outputs: [120 frames, 24 joints, 3 DOF] as axis-angle rotations
t2m_motion_aa = torch.randn(120, 24, 3)

# Convert to quaternions (w_last)
from scipy.spatial.transform import Rotation as sRot
motion_quat = torch.tensor(
    sRot.from_rotvec(t2m_motion_aa.numpy().reshape(-1, 3))
    .as_quat().reshape(120, 24, 4)
)  # [120, 24, 4] with w_last

# Compute global positions (requires forward kinematics)
# This depends on your T2M representation and SMPL model
global_pos = compute_fk_from_quat(motion_quat)  # [120, 24, 3]

# Create motion dict for single motion
motion_dict = {
    "gts": global_pos,
    "grs": motion_quat,
    "gvs": torch.zeros_like(global_pos),  # approximate if unavailable
    "gavs": torch.zeros_like(global_pos),
    "dps": motion_quat.reshape(120, 69),  # flatten 24×4 → 96 or similar
    "dvs": torch.zeros(120, 69),
    
    # Metadata for 1 motion
    "motion_num_frames": torch.tensor([120], dtype=torch.long),
    "motion_lengths": torch.tensor([120/30.0], dtype=torch.float32),
    "motion_dt": torch.tensor([1.0/30.0], dtype=torch.float32),
    "length_starts": torch.tensor([0], dtype=torch.long),
    "motion_weights": torch.tensor([1.0], dtype=torch.float32),
    "motion_files": ("t2m_motion_001.pt",),
}

torch.save(motion_dict, "t2m_motions.pt")
```

## Implementation Steps for Direction B

### Step 1: Prepare T2M Motion Data
```python
import torch
from pathlib import Path

# Load T2M generated motions
# Assume: t2m_motions.shape = [num_motions, frames_per_motion, 24, 3]
# (24 SMPL joints, 3 DOF each in axis-angle format)

t2m_motions = torch.load("t2m_outputs.pt")  # [10, 120, 24, 3]
```

### Step 2: Convert to Global Positions/Rotations
```python
from scipy.spatial.transform import Rotation as sRot
import numpy as np

# Convert axis-angle to quaternions
num_motions, num_frames, num_joints, _ = t2m_motions.shape
motion_quats = []

for motion_aa in t2m_motions:
    # motion_aa: [120, 24, 3]
    motion_quat = sRot.from_rotvec(
        motion_aa.numpy().reshape(-1, 3)
    ).as_quat().reshape(num_frames, num_joints, 4)
    motion_quats.append(torch.tensor(motion_quat, dtype=torch.float32))

motion_quats = torch.stack(motion_quats)  # [10, 120, 24, 4]

# Compute forward kinematics for global positions
# (requires SMPL model; example assumes you have compute_fk)
global_positions = compute_fk(motion_quats)  # [10, 120, 24, 3]
```

### Step 3: Package into ProtoMotions Format
```python
import torch

# Flatten motions into single concatenated tensors
total_frames = sum(m.shape[1] for m in motions)  # e.g., 10 × 120 = 1200

gts_list = [compute_fk(motion) for motion in motion_quats]  # List of [120, 24, 3]
gts_concat = torch.cat(gts_list, dim=0)  # [1200, 24, 3]

grs_list = [motion.reshape(120, 24, 4) for motion in motion_quats]
grs_concat = torch.cat(grs_list, dim=0)  # [1200, 24, 4]

# Compute velocities (or use zeros if not available)
gvs_concat = torch.zeros_like(gts_concat)  # [1200, 24, 3]
gavs_concat = torch.zeros_like(gts_concat)

# DOF positions (flatten quaternions)
dps_list = [motion.reshape(120, 96) for motion in motion_quats]  # 24×4 → 96
dps_concat = torch.cat(dps_list, dim=0)  # [1200, 96]
dvs_concat = torch.zeros_like(dps_concat)

# Create metadata
motion_num_frames = torch.tensor([120] * 10, dtype=torch.long)
motion_lengths = torch.tensor([4.0] * 10, dtype=torch.float32)
motion_dt = torch.tensor([1.0/30.0] * 10, dtype=torch.float32)

# Cumulative frame offsets
length_starts = torch.tensor(
    [sum(motion_num_frames[:i].tolist()) for i in range(len(motion_num_frames))],
    dtype=torch.long
)  # [0, 120, 240, 360, ...]

motion_weights = torch.tensor([1.0] * 10, dtype=torch.float32)

# Save
motion_lib_dict = {
    "gts": gts_concat,
    "grs": grs_concat,
    "gvs": gvs_concat,
    "gavs": gavs_concat,
    "dps": dps_concat,
    "dvs": dvs_concat,
    "motion_num_frames": motion_num_frames,
    "motion_lengths": motion_lengths,
    "motion_dt": motion_dt,
    "length_starts": length_starts,
    "motion_weights": motion_weights,
    "motion_files": tuple(f"t2m_{i}.pt" for i in range(10)),
}

torch.save(motion_lib_dict, "t2m_motions_packaged.pt")
print(f"Saved {len(motion_num_frames)} motions, {gts_concat.shape[0]} total frames")
```

### Step 4: Train RL Tracker
```bash
cd ref_repo/ProtoMotions

# Start with MuJoCo for debugging (CPU, slow)
python protomotions/train_agent.py \
    --robot-name smpl \
    --simulator mujoco \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name t2m_mujoco_debug \
    --motion-file /path/to/t2m_motions_packaged.pt \
    --num-envs 1 \
    --batch-size 32 \
    --training-max-steps 10000

# Or switch to IsaacGym for real training (GPU, 1000x faster)
python protomotions/train_agent.py \
    --robot-name smpl \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name t2m_isaacgym_training \
    --motion-file /path/to/t2m_motions_packaged.pt \
    --num-envs 4096 \
    --batch-size 16384 \
    --ngpu 1
```

## Output Directory Structure

After training, `results/<experiment-name>/` contains:

```
results/t2m_mujoco_debug/
├── config.yaml                           # CLI arguments used
├── resolved_configs.pt                   # Full config objects (pickle)
├── resolved_configs.yaml                 # Human-readable configs
├── resolved_configs_inference.pt         # Inference-only configs
├── experiment_config.py                  # Copy of mlp.py
├── last.ckpt                             # Latest model checkpoint
├── score_based.ckpt                      # Best checkpoint (by eval score)
└── events.out.tfevents.*                 # TensorBoard logs
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: expected shape [120, 69]` | DOF shape mismatch | Check `dps` is `[total_frames, 69]` for SMPL |
| `RuntimeError: expected shape [120, 24, 3]` | Body position/velocity shape | Ensure `gts/gvs` are `[T, 24, 3]` |
| `RuntimeError: expected shape [120, 24, 4]` | Quaternion shape | Ensure `grs` is `[T, 24, 4]` in w_last format |
| `IndexError: index out of bounds` | `length_starts` mismatch | Verify cumulative frame offsets match actual concatenation |
| `"All contact labels are zero"` | Missing contact data | Remove `contacts` field or ignore warning |
| Training very slow | MuJoCo CPU backend | Switch to `--simulator isaacgym` |

## References

- **Entry point**: `protomotions/train_agent.py`
- **Experiment config**: `examples/experiments/mimic/mlp.py`
- **Motion loading**: `protomotions/components/motion_lib.py`
- **State format**: `protomotions/simulator/base_simulator/simulator_state.py` (RobotState class)
- **Full guide**: See `PROTOMOTIONS_TRAINING_GUIDE.md`

## Quick Checklist for Direction B Implementation

- [ ] Load T2M motions from your model
- [ ] Convert axis-angle rotations to quaternions (w_last)
- [ ] Compute forward kinematics for global positions
- [ ] Compute velocities (or set to zeros)
- [ ] Flatten and concatenate all motions
- [ ] Create metadata (frame counts, lengths, offsets, weights)
- [ ] Save as `.pt` dictionary
- [ ] Run training command with correct paths
- [ ] Monitor results in `results/<experiment-name>/`
