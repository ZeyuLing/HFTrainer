# ProtoMotions T2M Integration: Complete Code Examples

## Example 1: Minimal T2M → Motion File Integration

```python
import torch
import torch.nn.functional as F
from pathlib import Path

# Assume you have a T2M model that outputs motions
def t2m_model_forward(text_prompt: str, num_frames: int = 100) -> dict:
    """
    Placeholder for your T2M model.
    In reality, this would be your trained T2M model.
    """
    # For demo: generate random valid motion tensors
    num_dofs = 67      # Humanoid DOF count
    num_bodies = 24    # Humanoid body count
    
    # Generate DOF positions (joint angles in radians)
    dof_pos = torch.randn(num_frames, num_dofs) * 0.5
    dof_vel = torch.randn(num_frames, num_dofs) * 0.1
    
    # Generate rigid body positions (starting from origin, moving in space)
    rigid_body_pos = torch.randn(num_frames, num_bodies, 3)
    # Make root position have some coherent motion
    time_steps = torch.linspace(0, 1, num_frames).reshape(-1, 1, 1)
    rigid_body_pos[:, 0, 0] += time_steps.squeeze(-1).squeeze(-1) * 2  # Root moves in X
    
    # Generate rotations as normalized quaternions (xyzw format)
    rigid_body_rot = F.normalize(torch.randn(num_frames, num_bodies, 4), dim=-1)
    
    # Generate velocities
    rigid_body_vel = torch.randn(num_frames, num_bodies, 3) * 0.5
    rigid_body_ang_vel = torch.randn(num_frames, num_bodies, 3) * 0.1
    
    return {
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "rigid_body_pos": rigid_body_pos,
        "rigid_body_rot": rigid_body_rot,
        "rigid_body_vel": rigid_body_vel,
        "rigid_body_ang_vel": rigid_body_ang_vel,
        "fps": 30,
    }


def save_t2m_output_to_motion_file(
    text_prompts: list[str],
    output_dir: str = "./generated_motions",
    fps: int = 30
) -> str:
    """
    Generate motions from text prompts and save as .motion files.
    
    Args:
        text_prompts: List of text descriptions
        output_dir: Directory to save motion files
        fps: Frames per second
    
    Returns:
        Path to YAML manifest file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    yaml_content = "motions:\n"
    
    for i, prompt in enumerate(text_prompts):
        print(f"Generating motion {i+1}/{len(text_prompts)}: {prompt}")
        
        # Generate motion from T2M model
        motion_dict = t2m_model_forward(prompt, num_frames=100)
        
        # Save to .motion file
        motion_file = f"{output_dir}/motion_{i:03d}.motion"
        torch.save(motion_dict, motion_file)
        print(f"  Saved to {motion_file}")
        
        # Add to YAML manifest
        yaml_content += f"  - file: motion_{i:03d}.motion\n"
        yaml_content += f"    weight: 1.0  # Equal weight for all\n"
    
    # Save YAML manifest
    yaml_file = f"{output_dir}/motions.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    
    print(f"\nSaved YAML manifest to {yaml_file}")
    return yaml_file


# Usage
if __name__ == "__main__":
    prompts = [
        "A person walking forward",
        "A person running quickly",
        "A person jumping",
    ]
    
    yaml_file = save_t2m_output_to_motion_file(prompts)
    print(f"✓ Motion files ready at: {yaml_file}")
```

## Example 2: Loading T2M Motions into ProtoMotions

```python
from protomotions.components.motion_lib import MotionLib, MotionLibConfig
from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion

def load_t2m_motions(yaml_file: str, device: str = "cpu") -> MotionLib:
    """
    Load T2M-generated motions into ProtoMotions.
    
    Args:
        yaml_file: Path to motions.yaml manifest
        device: PyTorch device (cpu or cuda)
    
    Returns:
        Loaded MotionLib ready for training
    """
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=yaml_file),
        device=device
    )
    
    print(f"✓ Loaded {motion_lib.num_motions()} motions")
    print(f"  Total duration: {motion_lib.get_total_length():.2f} seconds")
    print(f"  Avg frames per motion: {motion_lib.motion_num_frames.float().mean():.0f}")
    
    return motion_lib


def validate_motions(motion_lib: MotionLib, num_samples: int = 10) -> None:
    """
    Validate that motions can be sampled correctly.
    """
    print(f"\nValidating {num_samples} motion samples...")
    
    for i in range(num_samples):
        # Sample a random motion and time
        motion_ids = motion_lib.sample_motions(num_samples=1)
        motion_times = torch.rand(1) * motion_lib.motion_lengths[motion_ids]
        
        # Get interpolated state
        state = motion_lib.get_motion_state(motion_ids, motion_times)
        
        # Verify all required fields are present
        assert state.dof_pos is not None, "dof_pos is None"
        assert state.dof_vel is not None, "dof_vel is None"
        assert state.rigid_body_pos is not None, "rigid_body_pos is None"
        assert state.rigid_body_rot is not None, "rigid_body_rot is None"
        
        # Verify quaternion is normalized
        quat_norm = torch.norm(state.rigid_body_rot[0, 0, :])
        assert abs(quat_norm.item() - 1.0) < 1e-5, f"Quaternion not normalized: {quat_norm}"
        
        print(f"  Sample {i+1}: ✓ (motion_id={motion_ids.item()}, time={motion_times.item():.2f}s)")
    
    print("✓ All validations passed!")


# Usage
if __name__ == "__main__":
    import torch
    
    # Load motions
    yaml_file = "./generated_motions/motions.yaml"
    motion_lib = load_t2m_motions(yaml_file, device="cpu")
    
    # Validate
    validate_motions(motion_lib)
```

## Example 3: Integrating with RL Training

```python
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.ppo.agent import PPO
from lightning.fabric import Fabric

def setup_training_with_t2m_motions(
    yaml_file: str,
    robot_config,
    simulator_config,
    device: str = "cpu"
):
    """
    Set up RL training environment with T2M-generated motions.
    """
    
    # 1. Load T2M motions
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=yaml_file),
        device=device
    )
    
    # 2. Create environment
    env = BaseEnv(
        config=env_config,
        robot_config=robot_config,
        simulator_config=simulator_config,
        motion_lib=motion_lib,
        device=device
    )
    
    # 3. Create RL agent
    fabric = Fabric(accelerator="cpu", devices=1)
    agent = PPO(
        fabric=fabric,
        env=env,
        config=ppo_config,
    )
    
    return agent, env, motion_lib


def training_loop_with_t2m(
    agent,
    env,
    motion_lib,
    num_steps: int = 100000,
    log_interval: int = 1000
):
    """
    Main training loop using T2M motions.
    """
    agent.setup()
    
    print("Starting RL training with T2M motions...")
    print(f"  Motions available: {motion_lib.num_motions()}")
    print(f"  Training steps: {num_steps}")
    
    for step in range(num_steps):
        # Agent collects experience and updates
        agent.train_step()
        
        if (step + 1) % log_interval == 0:
            print(f"Step {step+1}/{num_steps}")
            print(f"  Motion library: {motion_lib.num_motions()} motions")
            print(f"  Total motion duration: {motion_lib.get_total_length():.2f}s")
    
    print("✓ Training complete!")


# Usage
if __name__ == "__main__":
    yaml_file = "./generated_motions/motions.yaml"
    
    # Setup (assumes robot_config, simulator_config, env_config, ppo_config are defined)
    agent, env, motion_lib = setup_training_with_t2m_motions(
        yaml_file=yaml_file,
        robot_config=robot_config,
        simulator_config=simulator_config,
        device="cpu"
    )
    
    # Train
    training_loop_with_t2m(agent, env, motion_lib, num_steps=100000)
```

## Example 4: Creating Packaged Motion Library

```python
def create_packaged_motion_library(
    yaml_file: str,
    output_pt_file: str = "./packaged_motions.pt",
    device: str = "cpu"
) -> str:
    """
    Convert individual motion files to a single packaged .pt file for faster loading.
    
    This is useful for deployment and distributed training.
    
    Args:
        yaml_file: Path to motions.yaml manifest
        output_pt_file: Output path for packaged .pt file
        device: PyTorch device
    
    Returns:
        Path to packaged motion file
    """
    
    print(f"Loading motions from {yaml_file}...")
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=yaml_file),
        device=device
    )
    
    print(f"Packaging {motion_lib.num_motions()} motions into {output_pt_file}...")
    motion_lib.save_to_file(output_pt_file)
    
    print(f"✓ Packaged motions saved to {output_pt_file}")
    
    # Verify by reloading
    print("Verifying packaged file...")
    motion_lib_check = MotionLib(
        config=MotionLibConfig(motion_file=output_pt_file),
        device=device
    )
    
    assert motion_lib.num_motions() == motion_lib_check.num_motions()
    assert motion_lib.get_total_length() == motion_lib_check.get_total_length()
    
    print(f"✓ Verification passed! File is ready for deployment.")
    
    return output_pt_file


# Usage
if __name__ == "__main__":
    yaml_file = "./generated_motions/motions.yaml"
    pt_file = create_packaged_motion_library(yaml_file)
    print(f"\nNext time, load directly from: {pt_file}")
```

## Example 5: Advanced T2M Output with Contact Detection

```python
def t2m_model_forward_with_contacts(
    text_prompt: str,
    num_frames: int = 100
) -> dict:
    """
    T2M model output including contact information.
    
    This is more realistic as it includes foot contact detection,
    which ProtoMotions can use for contact-matching rewards.
    """
    num_dofs = 67
    num_bodies = 24
    
    # Generate base motion
    motion = t2m_model_forward(text_prompt, num_frames)
    
    # Add contact information for foot bodies
    # Body indices for left and right feet (humanoid specific)
    left_foot_idx = 10  # Example
    right_foot_idx = 17  # Example
    
    # Generate contact labels (binary: in contact or not)
    contacts = torch.zeros(num_frames, num_bodies, dtype=torch.bool)
    
    # Simulate periodic foot contact (walking pattern)
    contact_period = 20  # Frames
    for t in range(num_frames):
        phase = (t % contact_period) / contact_period
        # Left foot contacts in first half of cycle
        if 0.0 <= phase < 0.5:
            contacts[t, left_foot_idx] = True
        # Right foot contacts in second half of cycle
        if 0.5 <= phase < 1.0:
            contacts[t, right_foot_idx] = True
    
    motion["rigid_body_contacts"] = contacts
    
    return motion


def save_t2m_output_with_contacts(
    text_prompts: list[str],
    output_dir: str = "./generated_motions_with_contacts"
) -> str:
    """
    Save T2M motions with contact information.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    yaml_content = "motions:\n"
    
    for i, prompt in enumerate(text_prompts):
        print(f"Generating motion with contacts {i+1}/{len(text_prompts)}: {prompt}")
        
        # Generate motion with contacts
        motion_dict = t2m_model_forward_with_contacts(prompt, num_frames=100)
        
        # Save
        motion_file = f"{output_dir}/motion_{i:03d}.motion"
        torch.save(motion_dict, motion_file)
        
        yaml_content += f"  - file: motion_{i:03d}.motion\n"
        yaml_content += f"    weight: 1.0\n"
    
    yaml_file = f"{output_dir}/motions.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    
    return yaml_file


# Usage
if __name__ == "__main__":
    prompts = [
        "A person walking",
        "A person running",
    ]
    
    yaml_file = save_t2m_output_with_contacts(prompts)
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=yaml_file),
        device="cpu"
    )
    
    # Sample and check contacts
    motion_ids = motion_lib.sample_motions(1)
    motion_times = torch.tensor([10.0])  # 10 seconds
    state = motion_lib.get_motion_state(motion_ids, motion_times)
    
    if state.rigid_body_contacts is not None:
        print(f"✓ Contacts available: shape {state.rigid_body_contacts.shape}")
    else:
        print("! No contacts")
```

## Example 6: Multi-GPU Training with T2M Motions (Distributed)

```python
def setup_distributed_training_with_t2m(
    yaml_file: str,
    num_chunks: int = 4,
    output_dir: str = "./distributed_motions"
) -> list[str]:
    """
    For distributed training, split motion library across ranks.
    
    This is useful for large motion libraries that don't fit in single GPU memory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load full motion library
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=yaml_file),
        device="cpu"
    )
    
    num_motions = motion_lib.num_motions()
    motions_per_chunk = (num_motions + num_chunks - 1) // num_chunks
    
    print(f"Splitting {num_motions} motions into {num_chunks} chunks")
    print(f"  ~{motions_per_chunk} motions per chunk")
    
    # Create per-chunk YAML files (simulate with different weights)
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * motions_per_chunk
        end_idx = min((chunk_id + 1) * motions_per_chunk, num_motions)
        
        yaml_content = "motions:\n"
        for motion_id in range(start_idx, end_idx):
            yaml_content += f"  - file: motion_{motion_id:03d}.motion\n"
            yaml_content += f"    weight: 1.0\n"
        
        chunk_yaml = f"{output_dir}/motions_chunk_{chunk_id:02d}.yaml"
        with open(chunk_yaml, "w") as f:
            f.write(yaml_content)
        
        print(f"  Created chunk {chunk_id}: {chunk_yaml}")
    
    return [f"{output_dir}/motions_chunk_{i:02d}.yaml" for i in range(num_chunks)]


def load_motion_for_rank(rank: int, chunk_yamls: list[str], device: str = "cpu") -> MotionLib:
    """
    Load motion chunk for specific rank.
    """
    yaml_file = chunk_yamls[rank % len(chunk_yamls)]
    
    print(f"Rank {rank} loading motions from {yaml_file}")
    
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=yaml_file),
        device=device
    )
    
    print(f"Rank {rank}: Loaded {motion_lib.num_motions()} motions")
    
    return motion_lib


# Usage in distributed training
if __name__ == "__main__":
    import torch.distributed as dist
    
    yaml_file = "./generated_motions/motions.yaml"
    
    # Split for 4 ranks
    chunk_yamls = setup_distributed_training_with_t2m(
        yaml_file,
        num_chunks=4
    )
    
    # Simulate rank-specific loading
    # In actual distributed training, each rank would call:
    # rank = torch.distributed.get_rank()
    # motion_lib = load_motion_for_rank(rank, chunk_yamls)
    
    for rank in range(4):
        motion_lib = load_motion_for_rank(rank, chunk_yamls)
```

## Example 7: Validation and Testing

```python
def comprehensive_validation(motion_lib: MotionLib) -> bool:
    """
    Comprehensive validation of loaded motion library.
    """
    print("Running comprehensive validation...")
    
    checks = {
        "num_motions > 0": motion_lib.num_motions() > 0,
        "gts shape correct": motion_lib.gts.shape[2] == 3,
        "grs shape correct": motion_lib.grs.shape[2] == 4,
        "motion_weights sum ~1": abs(motion_lib.motion_weights.sum().item() - 1.0) < 1e-5,
        "motion_lengths > 0": (motion_lib.motion_lengths > 0).all(),
        "motion_dt > 0": (motion_lib.motion_dt > 0).all(),
    }
    
    all_pass = True
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
        all_pass = all_pass and result
    
    # Test sampling
    print("\nTesting sampling and interpolation...")
    try:
        for _ in range(5):
            motion_ids = motion_lib.sample_motions(1)
            motion_times = torch.rand(1) * motion_lib.motion_lengths[motion_ids]
            state = motion_lib.get_motion_state(motion_ids, motion_times)
            
            # Verify outputs
            assert state.rigid_body_pos is not None
            assert state.rigid_body_rot is not None
            assert state.dof_pos is not None
            
            # Check quaternion normalization
            quat = state.rigid_body_rot[0, :, :]
            norms = torch.norm(quat, dim=-1)
            assert (abs(norms - 1.0) < 1e-4).all(), "Quaternions not normalized after interpolation"
        
        print("  ✓ Sampling and interpolation working correctly")
    except Exception as e:
        print(f"  ✗ Sampling test failed: {e}")
        all_pass = False
    
    if all_pass:
        print("\n✓ All validations PASSED!")
    else:
        print("\n✗ Some validations FAILED!")
    
    return all_pass


# Usage
if __name__ == "__main__":
    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file="./generated_motions/motions.yaml"),
        device="cpu"
    )
    
    is_valid = comprehensive_validation(motion_lib)
    
    if is_valid:
        print("\nMotions are ready for RL training!")
    else:
        print("\nPlease fix the validation errors before training.")
```

---

## Summary

These examples show:

1. **Basic T2M → Motion Integration**: Generate motions and save as .motion files
2. **Loading into MotionLib**: Load YAML manifest and validate
3. **RL Training Integration**: Connect to ProtoMotions training pipeline
4. **Packaged Motion Files**: Create .pt files for faster loading
5. **Advanced Features**: Contact detection for better rewards
6. **Distributed Training**: Split motions across multiple GPUs/ranks
7. **Comprehensive Validation**: Ensure motions are correct before training

All examples follow ProtoMotions conventions:
- ✓ Quaternions in **xyzw** format
- ✓ Normalized quaternions
- ✓ SI units (meters, radians, m/s)
- ✓ Proper tensor shapes
