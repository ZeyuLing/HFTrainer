#!/usr/bin/env python3
"""Rebuild V4 output from retargeted .motion files.

After running batch_retarget_parallel.py, this script:
1. Converts .motion files -> JSON for Three.js web viewer
2. Converts .motion files -> old-format .pt caches for physics sim / tracker
3. Extracts metrics and saves metadata
4. Generates manifest for the web gallery

Usage:
    python3 scripts/embodied/rebuild_v4_from_motion.py \
        --retarget-dir output/embodied_t2m_v4/data/retarget \
        --v4-dir output/embodied_t2m_v4

    # Dry run to see what would be rebuilt
    python3 scripts/embodied/rebuild_v4_from_motion.py \
        --retarget-dir output/embodied_t2m_v4/data/retarget \
        --v4-dir output/embodied_t2m_v4 \
        --dry-run
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Add ProtoMotions to path for unpickling .motion files
PROTOMOTIONS_ROOT = PROJECT_ROOT / "ref_repo" / "ProtoMotions"
if str(PROTOMOTIONS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROTOMOTIONS_ROOT))


def convert_motion_to_old_cache(motion_path: str, cache_output: str) -> dict:
    """Convert a .motion file to old-format .pt cache.

    The old .pt format has keys:
        body_pos: (T, N_bodies, 3) numpy
        body_rot: (T, N_bodies, 4) numpy (wxyz quaternion)
        dof_pos: (T, N_dof) numpy
        dof_vel: (T, N_dof) numpy
        body_vel: (T, N_bodies, 3) numpy
        body_ang_vel: (T, N_bodies, 3) numpy
        control_dt: float
        num_frames: int

    The .motion format has keys:
        rigid_body_pos: (T, N_bodies, 3) tensor
        rigid_body_rot: (T, N_bodies, 4) tensor (wxyz)
        dof_pos: (T, N_dof) tensor
        rigid_body_vel: (T, N_bodies, 3) tensor (optional)
        rigid_body_ang_vel: (T, N_bodies, 3) tensor (optional)
        fps / motion_dt: scalar
    """
    data = torch.load(motion_path, map_location="cpu", weights_only=False)

    # Get FPS / control_dt
    if "motion_dt" in data:
        control_dt = float(data["motion_dt"])
    elif "fps" in data:
        control_dt = 1.0 / float(data["fps"])
    else:
        control_dt = 1.0 / 30.0

    # body positions and rotations
    body_pos = data["rigid_body_pos"].numpy()  # (T, N, 3)
    body_rot = data["rigid_body_rot"].numpy()  # (T, N, 4) xyzw (COMMON convention)
    dof_pos = data["dof_pos"].numpy()          # (T, N_dof)
    num_frames = body_pos.shape[0]

    # Velocities (compute from finite differences if not present)
    if "rigid_body_vel" in data:
        body_vel = data["rigid_body_vel"].numpy()
    else:
        body_vel = np.zeros_like(body_pos)
        if num_frames > 1:
            body_vel[:-1] = (body_pos[1:] - body_pos[:-1]) / control_dt
            body_vel[-1] = body_vel[-2]

    if "rigid_body_ang_vel" in data:
        body_ang_vel = data["rigid_body_ang_vel"].numpy()
    else:
        body_ang_vel = np.zeros_like(body_pos)

    # dof_vel (finite differences)
    dof_vel = np.zeros_like(dof_pos)
    if num_frames > 1:
        dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / control_dt
        dof_vel[-1] = dof_vel[-2]

    # Save in old format
    cache = {
        "body_pos": body_pos,
        "body_rot": body_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "body_vel": body_vel,
        "body_ang_vel": body_ang_vel,
        "control_dt": control_dt,
        "num_frames": num_frames,
    }
    torch.save(cache, cache_output)

    return {
        "num_frames": num_frames,
        "fps": round(1.0 / control_dt),
        "root_z_mean": float(np.mean(body_pos[:, 0, 2])),
        "root_z_min": float(np.min(body_pos[:, 0, 2])),
    }


def convert_motion_to_json(motion_path: str, json_output: str) -> dict:
    """Convert .motion to JSON for Three.js web viewer.

    Uses existing convert_cache_to_json module.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from convert_cache_to_json import convert_cache_to_json
    return convert_cache_to_json(motion_path, json_output)


def extract_metrics(motion_path: str) -> dict:
    """Extract metrics from a .motion file."""
    data = torch.load(motion_path, map_location="cpu", weights_only=False)

    body_pos = data["rigid_body_pos"].numpy()
    dof_pos = data["dof_pos"].numpy()

    if "motion_dt" in data:
        control_dt = float(data["motion_dt"])
    elif "fps" in data:
        control_dt = 1.0 / float(data["fps"])
    else:
        control_dt = 1.0 / 30.0

    num_frames = body_pos.shape[0]
    root_height = body_pos[:, 0, 2]

    # Joint velocity
    if num_frames > 1:
        dof_vel = np.diff(dof_pos, axis=0) / control_dt
        max_joint_vel = float(np.max(np.abs(dof_vel)))
        mean_joint_vel = float(np.mean(np.abs(dof_vel)))
    else:
        max_joint_vel = 0.0
        mean_joint_vel = 0.0

    fell = bool(np.any(root_height < 0.3))
    fall_frame = int(np.argmax(root_height < 0.3)) if fell else None

    return {
        "num_frames": num_frames,
        "duration_s": round(num_frames * control_dt, 2),
        "fps": round(1.0 / control_dt),
        "root_height_mean": round(float(np.mean(root_height)), 4),
        "root_height_std": round(float(np.std(root_height)), 4),
        "root_height_min": round(float(np.min(root_height)), 4),
        "root_height_max": round(float(np.max(root_height)), 4),
        "max_joint_velocity": round(max_joint_vel, 2),
        "mean_joint_velocity": round(mean_joint_vel, 2),
        "fell": fell,
        "fall_frame": fall_frame,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild V4 output from retargeted .motion files"
    )
    parser.add_argument("--retarget-dir", type=str, required=True,
                        help="Directory containing retarget subdirs (motion_v4_xxx/)")
    parser.add_argument("--v4-dir", type=str, required=True,
                        help="V4 output directory (output/embodied_t2m_v4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already exist")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip old-format .pt cache generation")
    parser.add_argument("--no-json", action="store_true",
                        help="Skip JSON generation")
    args = parser.parse_args()

    retarget_dir = pathlib.Path(args.retarget_dir)
    v4_dir = pathlib.Path(args.v4_dir)

    # Discover .motion files
    motion_files = []
    for subdir in sorted(retarget_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for motion_file in subdir.glob("*.motion"):
            # Extract the original motion ID (e.g., "v4_turn_001" from "motion_v4_turn_001")
            stem = subdir.name  # e.g., "motion_v4_turn_001"
            # The original V4 ID is the stem without "motion_" prefix
            if stem.startswith("motion_"):
                original_id = stem[len("motion_"):]
            else:
                original_id = stem
            motion_files.append({
                "motion_path": str(motion_file),
                "original_id": original_id,
                "stem": stem,
            })

    if not motion_files:
        print(f"ERROR: No .motion files found in {retarget_dir}/*/")
        sys.exit(1)

    # Output dirs
    cache_dir = v4_dir / "data" / "caches"
    motions_dir = v4_dir / "data" / "motions"
    tracked_dir = v4_dir / "data" / "tracked_motions"
    meta_dir = v4_dir / "data" / "meta"
    for d in [cache_dir, motions_dir, tracked_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Rebuild V4 from .motion files")
    print(f"{'='*60}")
    print(f"  Motion files: {len(motion_files)}")
    print(f"  V4 dir:       {v4_dir}")
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Motions dir:  {motions_dir}")
    print(f"{'='*60}\n")

    if args.dry_run:
        for mf in motion_files:
            oid = mf['original_id']
            print(f"  Would rebuild: {oid}")
            print(f"    .motion: {mf['motion_path']}")
            print(f"    .pt:     {cache_dir / (oid + '.pt')}")
            print(f"    .json:   {motions_dir / (oid + '.json')}")
        return

    t_start = time.time()
    successes = 0
    failures = 0

    for i, mf in enumerate(motion_files):
        motion_path = mf["motion_path"]
        oid = mf["original_id"]
        print(f"\n[{i+1}/{len(motion_files)}] {oid}")

        try:
            # 1. Convert to old-format .pt cache
            if not args.no_cache:
                cache_path = cache_dir / f"{oid}.pt"
                if args.skip_existing and cache_path.exists():
                    print(f"  SKIP .pt cache (exists)")
                else:
                    info = convert_motion_to_old_cache(motion_path, str(cache_path))
                    print(f"  .pt cache: {info['num_frames']}f @ {info['fps']}fps, "
                          f"root_z={info['root_z_mean']:.3f}m (min={info['root_z_min']:.3f}m)")

            # 2. Convert to JSON for web viewer
            if not args.no_json:
                json_path = motions_dir / f"{oid}.json"
                if args.skip_existing and json_path.exists():
                    print(f"  SKIP JSON (exists)")
                else:
                    json_info = convert_motion_to_json(motion_path, str(json_path))
                    print(f"  JSON: {json_info['num_frames']}f @ {json_info['fps']}fps")

                # Also copy to tracked_motions
                tracked_path = tracked_dir / f"{oid}.json"
                if not tracked_path.exists():
                    import shutil
                    shutil.copy2(str(json_path), str(tracked_path))

            # 3. Extract metrics and save metadata
            metrics = extract_metrics(motion_path)
            meta = {
                "id": oid,
                "num_frames": metrics["num_frames"],
                "fps": metrics["fps"],
                "duration": metrics["duration_s"],
                "metrics": metrics,
                "motion_file": motion_path,
            }

            # Try to load text prompt from existing metadata
            existing_meta = meta_dir / f"{oid}.json"
            if existing_meta.exists():
                try:
                    with open(existing_meta) as f:
                        old_meta = json.load(f)
                    meta["text"] = old_meta.get("text", "")
                    meta["prompt"] = old_meta.get("prompt", "")
                except Exception:
                    pass

            with open(meta_dir / f"{oid}.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"  Metrics: root_z={metrics['root_height_mean']:.3f}m, "
                  f"fell={metrics['fell']}, "
                  f"max_jvel={metrics['max_joint_velocity']:.1f}")

            successes += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failures += 1

    total_time = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  Rebuild Complete!")
    print(f"{'='*60}")
    print(f"  Total:   {len(motion_files)}")
    print(f"  Success: {successes}")
    print(f"  Failed:  {failures}")
    print(f"  Time:    {total_time:.0f}s ({total_time/60:.1f}min)")

    # Quick summary of root heights
    if successes > 0:
        print(f"\n  Root height stats across all motions:")
        all_fell = 0
        for mf in motion_files:
            try:
                with open(meta_dir / f"{mf['original_id']}.json") as f:
                    m = json.load(f)
                if m.get("metrics", {}).get("fell", False):
                    all_fell += 1
            except Exception:
                pass
        print(f"  Motions that 'fell' (root Z < 0.3m): {all_fell}/{successes}")


if __name__ == "__main__":
    main()
