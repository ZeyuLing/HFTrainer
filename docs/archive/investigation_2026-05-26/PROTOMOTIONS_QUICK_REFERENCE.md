# ProtoMotions Training - Quick Reference for Direction B

## The Command You Need

To train a tracker on T2M-generated motions with MuJoCo backend:

```bash
cd /apdcephfs/AILab_DHA/apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer/ref_repo/ProtoMotions

python protomotions/train_agent.py \
    --robot-name smpl \
    --simulator mujoco \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_t2m_tracker \
    --motion-file /path/to/your/t2m_motions.pt \
    --num-envs 1 \
    --batch-size 32 \
    --ngpu 1
```

## Motion Data Format: Exactly What's Needed

Your motion file **MUST be a `.pt` file** (PyTorch tensor dict) containing:

```python
{
    "gts": torch.Tensor,           # [total_frames, 24, 3] - global positions
    "grs": torch.Tensor,           # [total_frames, 24, 4] - quaternions (w_last)
    "gvs": torch.Tensor,           # [total_frames, 24, 3] - global velocities
    "gavs": torch.Tensor,          # [total_frames, 24, 3] - angular velocities
    "dps": torch.Tensor,           # [total_frames, 69] - DOF positions
    "dvs": torch.Tensor,           # [total_frames, 69] - DOF velocities
    
    # Metadata (critical):
    "motion_num_frames": torch.Tensor,   # [num_motions] - frames per motion
    "motion_lengths": torch.Tensor,      # [num_motions] - length in seconds
    "motion_dt": torch.Tensor,           # [num_motions] - 1/fps
    "length_starts": torch.Tensor,       # [num_motions] - cumulative frame offsets
    "motion_weights": torch.Tensor,      # [num_motions] - sampling weights
    "motion_files": tuple,               # (filename1, filename2, ...) - metadata
    
    # Optional:
    "contacts": torch.Tensor,     # [total_frames, num_bodies] - contact labels
}
```

### Create from T2M motions:

```python
import torch

# Assuming you have T2M data
motion_dict = {
    "gts": global_positions,      # [T, 24, 3] float32
    "grs": quaternions,           # [T, 24, 4] float32, (x,y,z,w)
    "gvs": global_velocities,     # [T, 24, 3] float32
    "gavs": angular_velocities,   # [T, 24, 3] float32
    "dps": dof_positions,         # [T, 69] float32
    "dvs": dof_velocities,        # [T, 69] float32
    
    # Single motion metadata
    "motion_num_frames": torch.tensor([T]),
    "motion_lengths": torch.tensor([T / 30.0]),    # 30 fps
    "motion_dt": torch.tensor([1.0 / 30.0]),
    "length_starts": torch.tensor([0]),
    "motion_weights": torch.tensor([1.0]),
    "motion_files": ("t2m_motion.pt",),
}

torch.save(motion_dict, "t2m_motions.pt")
```

## Key Facts

| Aspect | Details |
|--------|---------|
| **Entry point** | `protomotions/train_agent.py` |
| **Experiment config** | `examples/experiments/mimic/mlp.py` - defines the tracking task |
| **Output directory** | `results/<experiment-name>/` - contains checkpoints and configs |
| **Motion loading** | Done by `MotionLib` class in `protomotions/components/motion_lib.py` |
| **SMPL geometry** | 24 bodies, 69 DOFs (23 joints × 3 per joint) |
| **FPS** | Standard is 30 fps (adjust `motion_dt` and `motion_lengths` if different) |
| **MuJoCo speed** | **Slow!** (CPU). Use `--simulator isaacgym` for 100-1000x speedup with GPU. |

## Common Issues & Fixes

| Problem | Solution |
|---------|----------|
| Shape mismatch error | Verify: `gts/grs/gvs/gavs [T, 24, ...]`, `dps/dvs [T, 69]` |
| "All contact labels are zero" | Don't include `contacts` field, or just ignore the warning |
| MuJoCo is slow | Use `--simulator isaacgym` for GPU acceleration |
| "No motion files found" | Check `--motion-file` path exists and ends with `.pt` |
| Out of memory | Reduce `--num-envs` or `--batch-size` |

## Training Flow Overview

```
train_agent.py
  ├─ Load experiment config (mlp.py)
  ├─ Call config builders:
  │   └─ motion_lib_config() → MotionLib loads your .pt file
  ├─ Initialize simulator, environment, PPO agent
  ├─ Save configs to results/<exp-name>/
  └─ Run training loop (agent.fit())
```

## Files You're Working With

```
ProtoMotions/
├── protomotions/
│   ├── train_agent.py              ← ENTRY POINT
│   ├── components/
│   │   └── motion_lib.py           ← Loads your .pt data
│   ├── simulator/
│   │   ├── mujoco/                 ← MuJoCo backend
│   │   ├── isaacgym/               ← GPU alternative
│   │   └── base_simulator/
│   │       └── simulator_state.py  ← RobotState class
│   └── agents/ppo/                 ← RL trainer
└── examples/experiments/mimic/
    └── mlp.py                      ← YOUR EXPERIMENT CONFIG
```

## Example: T2M to Training Pipeline

```python
# Step 1: Generate T2M motions
# Your T2M model outputs: motion [10, 120, 24, 3] axis-angle rotations

# Step 2: Convert to ProtoMotions format
from scipy.spatial.transform import Rotation as sRot
motion_quat = sRot.from_rotvec(
    motion.reshape(-1, 3)
).as_quat().reshape(120, 24, 4)  # w_last format

# Step 3: Compute forward kinematics to get global positions
# (requires kinematic chain computation)
global_positions = compute_fk(motion_quat)  # [120, 24, 3]

# Step 4: Package into .pt
motion_dict = {
    "gts": global_positions,
    "grs": motion_quat,
    "gvs": torch.zeros_like(global_positions),  # compute or approximate
    "gavs": torch.zeros_like(global_positions),
    "dps": motion_quat.reshape(120, 69),  # flatten to DOFs
    "dvs": torch.zeros(120, 69),
    "motion_num_frames": torch.tensor([120]),
    "motion_lengths": torch.tensor([4.0]),
    "motion_dt": torch.tensor([1/30.0]),
    "length_starts": torch.tensor([0]),
    "motion_weights": torch.tensor([1.0]),
    "motion_files": ("t2m_001.pt",),
}
torch.save(motion_dict, "t2m_motions.pt")

# Step 5: Train!
# python protomotions/train_agent.py \
#     --robot-name smpl \
#     --simulator isaacgym \  ← Use GPU for real training!
#     --experiment-path examples/experiments/mimic/mlp.py \
#     --motion-file t2m_motions.pt \
#     --num-envs 4096 --batch-size 16384
```

## Recommendations for Direction B

1. **Start with MuJoCo** for debugging and verification
2. **Switch to IsaacGym** for real training (1000x faster)
3. **Ensure proper FPS** in motion data (typically 30 fps)
4. **Compute forward kinematics** correctly from T2M joint rotations
5. **Set correct number of bodies** (24 for SMPL) and DOFs (69)
6. **Include velocities** if possible (approximate if needed)

See `PROTOMOTIONS_TRAINING_GUIDE.md` for detailed explanations.
