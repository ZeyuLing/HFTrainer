#!/usr/bin/env python3
"""End-to-end pipeline: Text → HyMotion T2M → G1 Retarget → Isaac Gym Sim.

This script implements the full pipeline for driving a Unitree G1 humanoid
robot from text descriptions:

  1. Text → SMPL Motion (HyMotion T2M-Lite)
  2. SMPL Motion → G1 Joint Angles (retargeting)
  3. G1 Joint Angles → Isaac Gym Training (motion imitation)
  4. Trained Policy → Sim2Sim Evaluation (MuJoCo)

Usage examples:
    # Step 1+2: Generate motion and retarget
    python tools/robot_sim/text_to_g1.py \\
        --prompt "a person walks forward slowly" \\
        --output output/g1_walk/ \\
        --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \\
        --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt

    # Step 1+2+3: Also generate ASAP training command
    python tools/robot_sim/text_to_g1.py \\
        --prompt "a person does a jumping jack" \\
        --output output/g1_jump/ \\
        --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \\
        --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt \\
        --generate-train-cmd \\
        --asap-root ~/ASAP

    # Step 2 only: Retarget existing motion file
    python tools/robot_sim/text_to_g1.py \\
        --input-npz output/generated_motion.npz \\
        --output output/g1_retarget/

    # Batch: multiple prompts
    python tools/robot_sim/text_to_g1.py \\
        --prompt-file prompts.txt \\
        --output output/g1_batch/ \\
        --config configs/hymotion_t2m/hymotion_t2m_201dim_046b.py \\
        --checkpoint checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt
"""

import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description='Text → HyMotion → G1 Retarget → Isaac Gym pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input modes (mutually exclusive) ---
    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument('--prompt', type=str, help='Text prompt for motion generation')
    input_group.add_argument('--prompt-file', type=str, help='File with one prompt per line')
    input_group.add_argument('--input-npz', type=str, help='Pre-generated motion .npz (skip generation)')

    # --- HyMotion config ---
    p.add_argument('--config', type=str, help='HyMotion T2M config path')
    p.add_argument('--checkpoint', type=str, help='HyMotion T2M checkpoint path')
    p.add_argument('--num-frames', type=int, default=120, help='Number of frames to generate (default: 120 = 4s @ 30fps)')
    p.add_argument('--num-steps', type=int, default=50, help='ODE integration steps')
    p.add_argument('--guidance-scale', type=float, default=5.0, help='CFG guidance scale')
    p.add_argument('--device', type=str, default='cuda', help='Device for inference')

    # --- Output ---
    p.add_argument('--output', type=str, default='output/g1_motion/', help='Output directory')

    # --- Retargeting ---
    p.add_argument('--no-clamp', action='store_true', help='Disable joint limit clamping')
    p.add_argument('--no-calibration', action='store_true', help='Disable rest-pose calibration')
    p.add_argument('--g1-dof', type=int, default=29, choices=[23, 29], help='G1 DOF version')
    p.add_argument('--motion-dim', type=int, default=None, help='Override motion dim (135 or 201)')

    # --- ASAP integration ---
    p.add_argument('--generate-train-cmd', action='store_true', help='Generate ASAP training command')
    p.add_argument('--asap-root', type=str, default=None, help='ASAP repo root directory')
    p.add_argument('--num-envs', type=int, default=4096, help='Number of parallel environments')

    # --- Visualization ---
    p.add_argument('--visualize', action='store_true', help='Visualize retargeted motion (if mujoco available)')
    p.add_argument('--save-video', action='store_true', help='Save visualization as video')

    return p.parse_args()


def generate_motion(args):
    """Step 1: Generate SMPL motion from text using HyMotion T2M."""
    import torch
    from mmengine.config import Config
    import hftrainer  # noqa: trigger auto-imports

    print(f'[Step 1] Generating motion from text...')
    print(f'  Prompt: "{args.prompt}"')
    print(f'  Frames: {args.num_frames}')

    cfg = Config.fromfile(args.config)

    # Load bundle
    from tools.infer import load_bundle_from_checkpoint
    bundle = load_bundle_from_checkpoint(cfg, args.checkpoint, args.device)

    # Create pipeline
    from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
    pipeline = HyMotionT2MPipeline(
        bundle=bundle,
        num_steps=args.num_steps,
        text_guidance_scale=args.guidance_scale,
    )

    # Generate
    batch = {
        'tgt_length': [args.num_frames],
        'caption': [args.prompt],
    }

    with torch.no_grad():
        output = pipeline(batch)

    # Extract motion
    latent = output['latent']
    if isinstance(latent, torch.Tensor):
        latent = latent.cpu().float().numpy()

    # Denormalize
    result = output
    latent_denorm = result.get('latent_denorm')
    if latent_denorm is not None:
        if isinstance(latent_denorm, torch.Tensor):
            latent_denorm = latent_denorm.cpu().float().numpy()
        motion = latent_denorm[0]  # (T, D)
    else:
        # Manually denormalize
        mean = bundle.mean.cpu().numpy()
        std = bundle.std.cpu().numpy()
        std = np.where(std < 1e-3, 1.0, std)
        motion = latent[0] * std + mean

    # Determine motion_dim
    motion_dim = motion.shape[-1]
    print(f'  Generated motion: shape={motion.shape}, dim={motion_dim}')

    return motion, motion_dim


def load_motion_from_npz(path, motion_dim=None):
    """Load pre-generated motion from npz file."""
    data = np.load(path)

    if 'latent_denorm' in data:
        motion = data['latent_denorm']
    elif 'motion' in data:
        motion = data['motion']
    else:
        # Try to reconstruct from rot6d + transl
        rot6d = data.get('rot6d')
        transl = data.get('transl')
        if rot6d is not None and transl is not None:
            if rot6d.ndim == 4:  # (B, T, J, 6)
                rot6d = rot6d[0]
            if transl.ndim == 3:  # (B, T, 3)
                transl = transl[0]
            T = transl.shape[0]
            rot6d_flat = rot6d.reshape(T, -1)
            motion = np.concatenate([transl, rot6d_flat], axis=-1)
        else:
            raise ValueError(f'Cannot find motion data in {path}. Keys: {list(data.keys())}')

    if motion.ndim == 3:
        motion = motion[0]

    if motion_dim is not None and motion.shape[-1] != motion_dim:
        print(f'  Warning: motion dim {motion.shape[-1]} != expected {motion_dim}')

    return motion


def retarget_motion(motion, args):
    """Step 2: Retarget SMPL motion to G1 joint angles."""
    from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter

    print(f'[Step 2] Retargeting SMPL motion to G1 {args.g1_dof}-DOF...')

    retargeter = SMPLToG1Retargeter(
        apply_limits=not args.no_clamp,
        rest_pose_calibration=not args.no_calibration,
        g1_dof=args.g1_dof,
    )

    motion_dim = motion.shape[-1]
    if motion_dim == 135:
        result = retargeter.retarget_from_hymotion(motion)
    elif motion_dim == 201:
        result = retargeter.retarget_from_hymotion_201(motion)
    else:
        raise ValueError(f'Unsupported motion dim: {motion_dim}. Expected 135 or 201.')

    T = result['joint_angles'].shape[0]
    print(f'  Retargeted: {T} frames, {result["dof"]} DOF')
    print(f'  Root position range: {result["root_pos"].min(0)} to {result["root_pos"].max(0)}')

    # Print joint angle statistics
    angles = result['joint_angles']
    print(f'  Joint angle range: [{angles.min():.3f}, {angles.max():.3f}] rad')
    print(f'  Joint angle std:   {angles.std(0).mean():.4f} rad (mean across joints)')

    return result, retargeter


def save_results(motion, retarget_result, retargeter, args):
    """Save all outputs."""
    os.makedirs(args.output, exist_ok=True)

    # Save raw generated motion
    motion_path = os.path.join(args.output, 'smpl_motion.npz')
    np.savez(motion_path, motion=motion)
    print(f'  Saved SMPL motion: {motion_path}')

    # Save retargeted G1 motion
    g1_path = os.path.join(args.output, 'g1_motion.npz')
    np.savez(
        g1_path,
        joint_angles=retarget_result['joint_angles'],
        root_pos=retarget_result['root_pos'],
        root_orient_quat=retarget_result['root_orient_quat'],
        root_orient_euler=retarget_result['root_orient_euler'],
        fps=retarget_result['fps'],
        joint_names=retarget_result['joint_names'],
    )
    print(f'  Saved G1 motion: {g1_path}')

    # Save ASAP-compatible pkl
    pkl_path = os.path.join(args.output, 'g1_motion_asap.pkl')
    retargeter.to_asap_pkl(retarget_result, pkl_path)
    print(f'  Saved ASAP pkl: {pkl_path}')

    # Save MuJoCo qpos
    qpos = retargeter.to_mujoco_qpos(retarget_result)
    qpos_path = os.path.join(args.output, 'g1_mujoco_qpos.npy')
    np.save(qpos_path, qpos)
    print(f'  Saved MuJoCo qpos: {qpos_path}')

    return pkl_path


def generate_asap_commands(pkl_path, args):
    """Step 3: Generate ASAP training commands."""
    from hftrainer.models.motion.components.retarget.isaac_gym_bridge import (
        ASAPConfigGenerator,
    )

    print(f'\n[Step 3] Generating ASAP training configuration...')

    generator = ASAPConfigGenerator(
        asap_root=args.asap_root,
        num_envs=args.num_envs,
    )

    # Check installation
    checks = generator.check_asap_installation()
    print(f'  ASAP root exists: {checks["asap_root"]}')
    print(f'  Isaac Gym installed: {checks["isaac_gym"]}')
    print(f'  MuJoCo installed: {checks["mujoco"]}')

    # Generate training command
    prompt_slug = args.prompt[:30].replace(' ', '_').replace('"', '') if args.prompt else 'motion'
    exp_name = f'hymotion_{prompt_slug}'

    train_cmd = generator.generate_training_command(
        motion_file=os.path.abspath(pkl_path),
        experiment_name=exp_name,
    )

    # Save commands
    cmd_path = os.path.join(args.output, 'asap_commands.sh')
    with open(cmd_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Auto-generated ASAP training commands for G1 motion imitation\n')
        f.write(f'# Generated at: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Prompt: {args.prompt}\n\n')
        f.write('# === Step 1: Train Motion Tracking Policy ===\n')
        f.write(train_cmd + '\n\n')
        f.write('# === Step 2: Evaluate Policy (set checkpoint path) ===\n')
        f.write(f'# {generator.generate_eval_command("<CHECKPOINT_PATH>")}\n\n')
        f.write('# === Step 3: Sim2Sim Deployment ===\n')
        cmds = generator.generate_sim2sim_commands('<POLICY_ONNX_PATH>')
        f.write(f'# Terminal 1 (simulator):\n# {cmds["simulator"]}\n')
        f.write(f'# Terminal 2 (policy):\n# {cmds["policy"]}\n')

    print(f'  Saved ASAP commands: {cmd_path}')
    print(f'\n  === Training Command ===')
    print(f'  {train_cmd}')

    return cmd_path


def visualize_motion(retarget_result, args):
    """Optional: Visualize retargeted motion using MuJoCo."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print('  [Skip] MuJoCo not installed. Install with: pip install mujoco')
        return

    print(f'\n[Viz] Visualizing G1 motion in MuJoCo...')

    # Look for G1 MJCF/URDF
    g1_model_paths = [
        os.path.expanduser('~/ASAP/humanoidverse/data/robots/g1/g1_29dof.xml'),
        os.path.expanduser('~/ASAP/sim2real/models/g1_29dof.xml'),
        'data/robots/g1/g1_29dof.xml',
    ]

    model_path = None
    for p in g1_model_paths:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print('  [Skip] G1 MJCF model not found. Checked:')
        for p in g1_model_paths:
            print(f'    {p}')
        return

    from hftrainer.models.motion.components.retarget import SMPLToG1Retargeter
    retargeter = SMPLToG1Retargeter(g1_dof=args.g1_dof)
    qpos_seq = retargeter.to_mujoco_qpos(retarget_result)

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    if args.save_video:
        # Offscreen render
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []
        for t in range(qpos_seq.shape[0]):
            data.qpos[:] = qpos_seq[t]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            frames.append(renderer.render())

        video_path = os.path.join(args.output, 'g1_motion.mp4')
        try:
            import imageio
            imageio.mimsave(video_path, frames, fps=retarget_result['fps'])
            print(f'  Saved video: {video_path}')
        except ImportError:
            print('  [Skip] imageio not installed for video saving')
    else:
        # Interactive viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            fps = retarget_result['fps']
            dt = 1.0 / fps
            for t in range(qpos_seq.shape[0]):
                data.qpos[:] = qpos_seq[t]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)


def main():
    args = parse_args()

    print('=' * 60)
    print('  HyMotion T2M → Unitree G1 Pipeline')
    print('=' * 60)

    motion = None
    motion_dim = None

    # --- Step 1: Generate or load motion ---
    if args.input_npz:
        print(f'[Step 1] Loading pre-generated motion from: {args.input_npz}')
        motion = load_motion_from_npz(args.input_npz, args.motion_dim)
        motion_dim = motion.shape[-1]
        print(f'  Loaded motion: shape={motion.shape}, dim={motion_dim}')
    elif args.prompt:
        if not args.config or not args.checkpoint:
            print('ERROR: --config and --checkpoint required for text-to-motion generation.')
            sys.exit(1)
        motion, motion_dim = generate_motion(args)
    elif args.prompt_file:
        if not args.config or not args.checkpoint:
            print('ERROR: --config and --checkpoint required for text-to-motion generation.')
            sys.exit(1)
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f'[Batch] Processing {len(prompts)} prompts...')
        for i, prompt in enumerate(prompts):
            args.prompt = prompt
            sub_output = os.path.join(args.output, f'motion_{i:03d}')
            args_copy = argparse.Namespace(**vars(args))
            args_copy.output = sub_output
            args_copy.prompt = prompt

            motion_i, dim_i = generate_motion(args_copy)
            result_i, retargeter_i = retarget_motion(motion_i, args_copy)
            pkl_path_i = save_results(motion_i, result_i, retargeter_i, args_copy)
            print(f'  [{i+1}/{len(prompts)}] Done: {prompt[:50]}...')
        print(f'\nBatch complete. Results in: {args.output}')
        return
    else:
        print('ERROR: Provide --prompt, --prompt-file, or --input-npz.')
        sys.exit(1)

    if motion is None:
        print('ERROR: No motion generated or loaded.')
        sys.exit(1)

    # --- Step 2: Retarget ---
    retarget_result, retargeter = retarget_motion(motion, args)

    # --- Save ---
    pkl_path = save_results(motion, retarget_result, retargeter, args)

    # --- Step 3: ASAP integration ---
    if args.generate_train_cmd:
        generate_asap_commands(pkl_path, args)

    # --- Optional: Visualization ---
    if args.visualize or args.save_video:
        visualize_motion(retarget_result, args)

    # --- Summary ---
    print(f'\n{"=" * 60}')
    print(f'  Pipeline Complete!')
    print(f'  Output directory: {args.output}')
    print(f'  Files:')
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        size = os.path.getsize(fpath)
        print(f'    {f} ({size:,} bytes)')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
