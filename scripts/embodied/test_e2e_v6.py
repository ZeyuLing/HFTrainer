#!/usr/bin/env python3
"""Quick end-to-end V6 test: T2M (guidance_scale=5.0) → PyRoki retarget → .motion.

Runs 3 test prompts through the full pipeline and prints quality metrics.
Usage (on Taiji container):
    cd /apdcephfs_cq11/share_1467498/home/zeyuling/hf_trainer
    python scripts/embodied/test_e2e_v6.py --output-dir data/embodied_debug/v6_e2e_test
"""

import argparse
import os
import sys
import pathlib
import time
import json

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"

# Add ProtoMotions to path so we can load .motion files
if str(PROTOMOTIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOMOTIONS_ROOT))

import numpy as np

TEST_PROMPTS = [
    {"id": "walk_forward", "text": "a person walks forward steadily", "frames": 120},
    {"id": "jump_in_place", "text": "a person jumps in place", "frames": 90},
    {"id": "wave_hand", "text": "a person stands and waves their right hand", "frames": 90},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="Output directory for test results")
    parser.add_argument("--device", default="cuda", help="Device for T2M")
    parser.add_argument("--num-steps", type=int, default=50, help="ODE steps (50 matches official)")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG scale")
    parser.add_argument("--skip-t2m", action="store_true", help="Skip T2M, use existing NPZ")
    parser.add_argument("--skip-retarget", action="store_true", help="Skip retarget pipeline")
    args = parser.parse_args()

    output_root = pathlib.Path(args.output_dir)
    npz_dir = output_root / "npz"
    retarget_dir = output_root / "retarget"
    npz_dir.mkdir(parents=True, exist_ok=True)
    retarget_dir.mkdir(parents=True, exist_ok=True)

    # ====================================================
    # Step 1: T2M Inference with guidance_scale=5.0
    # ====================================================
    if not args.skip_t2m:
        print("=" * 60)
        print("  Step 1: T2M Inference (guidance_scale=5.0)")
        print("=" * 60)

        import torch
        from mmengine.config import Config
        import hftrainer  # noqa

        config_path = str(PROJECT_ROOT / "configs/hymotion_t2m/hymotion_t2m_201dim_046b.py")
        ckpt_path = str(PROJECT_ROOT / "checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt")

        cfg = Config.fromfile(config_path)
        if not cfg.model.get('text_encoder'):
            cfg.model.text_encoder = dict(type='HYTextModel', llm_type='qwen3', max_length_llm=128)
            print("  Injected text_encoder config")

        from tools.infer import load_bundle_from_checkpoint
        bundle = load_bundle_from_checkpoint(cfg, ckpt_path, args.device)

        from hftrainer.pipelines.motion.hymotion_t2m_pipeline import HyMotionT2MPipeline
        pipeline = HyMotionT2MPipeline(
            bundle=bundle,
            num_steps=args.num_steps,
            text_guidance_scale=args.guidance_scale,
        )

        print(f"  Model loaded. num_steps={args.num_steps}, guidance_scale={args.guidance_scale}")
        print(f"  Pipeline CFG enabled: {args.guidance_scale > 1.0}")

        # Import smoothing from batch script
        sys.path.insert(0, str(SCRIPT_DIR))
        from batch_t2m_to_embodied import smooth_motion_135

        for prompt in TEST_PROMPTS:
            pid = prompt["id"]
            npz_path = npz_dir / f"{pid}.npz"

            if npz_path.exists():
                print(f"\n  [{pid}] NPZ exists, skipping")
                continue

            print(f"\n  [{pid}] '{prompt['text']}' ({prompt['frames']} frames)")
            t0 = time.time()

            batch = {"tgt_length": [prompt["frames"]], "caption": [prompt["text"]]}
            with torch.no_grad():
                output = pipeline(batch)

            latent = output.get("latent_denorm")
            if latent is not None:
                if isinstance(latent, torch.Tensor):
                    latent = latent.cpu().float().numpy()
                motion_201 = latent[0]
            else:
                latent_raw = output["latent"]
                if isinstance(latent_raw, torch.Tensor):
                    latent_raw = latent_raw.cpu().float().numpy()
                mean = bundle.mean.cpu().numpy()
                std = bundle.std.cpu().numpy()
                std = np.where(std < 1e-3, 0.0, std)
                motion_201 = latent_raw[0] * std + mean

            motion_135 = motion_201[:, :135]

            # Apply Markley smoothing
            motion_135_smooth = smooth_motion_135(motion_135)

            # Save both raw and smoothed
            np.savez(
                str(npz_path),
                motion_135=motion_135_smooth.astype(np.float32),
                motion_135_raw=motion_135.astype(np.float32),
                fps=np.array(30),
            )

            dt = time.time() - t0
            print(f"    Generated: {motion_135.shape}, saved to {npz_path} ({dt:.1f}s)")

            # Quick sanity: translation range
            transl = motion_135_smooth[:, :3]
            print(f"    Transl range: x=[{transl[:,0].min():.3f}, {transl[:,0].max():.3f}], "
                  f"y=[{transl[:,1].min():.3f}, {transl[:,1].max():.3f}], "
                  f"z=[{transl[:,2].min():.3f}, {transl[:,2].max():.3f}]")
    else:
        print("Skipping T2M inference")

    # ====================================================
    # Step 2: PyRoki Retarget Pipeline (V6)
    # ====================================================
    if not args.skip_retarget:
        print("\n" + "=" * 60)
        print("  Step 2: PyRoki V6 Retarget Pipeline")
        print("=" * 60)

        import subprocess

        for prompt in TEST_PROMPTS:
            pid = prompt["id"]
            npz_path = npz_dir / f"{pid}.npz"
            mot_retarget_dir = retarget_dir / pid

            if not npz_path.exists():
                print(f"\n  [{pid}] No NPZ found, skipping retarget")
                continue

            # Check if .motion already exists
            mot_retarget_dir.mkdir(parents=True, exist_ok=True)
            motion_files = list(mot_retarget_dir.glob("*.motion"))
            if motion_files:
                print(f"\n  [{pid}] .motion exists: {motion_files[0].name}, skipping")
                continue

            print(f"\n  [{pid}] Running pipeline_motion_to_robot.py...")
            t0 = time.time()

            cmd = [
                sys.executable,
                str(SCRIPT_DIR / "pipeline_motion_to_robot.py"),
                "--input", str(npz_path),
                "--output", str(mot_retarget_dir),
                "--keep-intermediates",
            ]

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=3600,  # 60 min for PyRoki (JAX optimization can be slow)
            )

            dt = time.time() - t0
            if result.returncode != 0:
                print(f"    FAILED ({dt:.1f}s)")
                print(f"    stderr (last 30 lines):")
                for line in (result.stderr or "").strip().split("\n")[-30:]:
                    print(f"      {line}")
                print(f"    stdout (last 10 lines):")
                for line in (result.stdout or "").strip().split("\n")[-10:]:
                    print(f"      {line}")
            else:
                print(f"    OK ({dt:.1f}s)")
                # Find output
                motion_files = list(mot_retarget_dir.glob("*.motion"))
                if motion_files:
                    print(f"    Output: {motion_files[0]}")
    else:
        print("Skipping retarget")

    # ====================================================
    # Step 3: Quality Check
    # ====================================================
    print("\n" + "=" * 60)
    print("  Step 3: Quality Metrics")
    print("=" * 60)

    import torch

    for prompt in TEST_PROMPTS:
        pid = prompt["id"]
        mot_retarget_dir = retarget_dir / pid

        motion_files = list(mot_retarget_dir.glob("*.motion")) if mot_retarget_dir.exists() else []
        if not motion_files:
            print(f"\n  [{pid}] No .motion file found")
            continue

        motion_file = motion_files[0]
        try:
            cache = torch.load(str(motion_file), map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"\n  [{pid}] Failed to load .motion file: {e}")
            continue

        body_pos = cache.get("rigid_body_pos")
        dof_pos = cache.get("dof_pos")

        if body_pos is None or dof_pos is None:
            print(f"\n  [{pid}] Missing data in .motion file")
            continue

        if isinstance(body_pos, torch.Tensor):
            body_pos = body_pos.numpy()
        if isinstance(dof_pos, torch.Tensor):
            dof_pos = dof_pos.numpy()

        root_height = body_pos[:, 0, 2]  # Z-up
        num_frames = body_pos.shape[0]

        # DOF velocity
        fps = float(cache.get("fps", 30))
        dt = 1.0 / fps
        if num_frames > 1:
            dof_vel = np.diff(dof_pos, axis=0) / dt
            max_dof_vel = float(np.max(np.abs(dof_vel)))
            mean_dof_vel = float(np.mean(np.abs(dof_vel)))
        else:
            max_dof_vel = mean_dof_vel = 0.0

        # Fall detection
        fell = bool(np.any(root_height < 0.3))

        # Left/right foot heights (find from body_pos)
        num_bodies = body_pos.shape[1]
        # Typically: feet are at specific indices depending on G1 robot
        foot_min_height = float(np.min(body_pos[:, :, 2]))  # min Z across all bodies
        foot_max_height = float(np.max(body_pos[:, :, 2]))  # max Z

        print(f"\n  [{pid}] {motion_file.name}")
        print(f"    Frames:        {num_frames}")
        print(f"    Duration:      {num_frames / fps:.2f}s")
        print(f"    Root height:   mean={np.mean(root_height):.4f}, min={np.min(root_height):.4f}, max={np.max(root_height):.4f}")
        print(f"    DOF velocity:  max={max_dof_vel:.2f}, mean={mean_dof_vel:.2f}")
        print(f"    Body Z range:  [{foot_min_height:.4f}, {foot_max_height:.4f}]")
        print(f"    DOF range:     [{dof_pos.min():.3f}, {dof_pos.max():.3f}]")
        print(f"    Fell:          {fell}")

        # Check contact labels
        contacts = cache.get("rigid_body_contacts")
        if contacts is not None:
            if isinstance(contacts, torch.Tensor):
                contacts = contacts.numpy()
            contact_any = contacts.any(axis=0)
            print(f"    Contacts:      {contacts.shape}, bodies with contacts: {np.where(contact_any)[0].tolist()}")
        else:
            print(f"    Contacts:      None (not saved)")

    # ====================================================
    # Summary
    # ====================================================
    print("\n" + "=" * 60)
    print("  V6 End-to-End Test Complete!")
    print("=" * 60)
    print(f"  Output: {output_root}")


if __name__ == "__main__":
    with __import__('torch').no_grad():
        main()
