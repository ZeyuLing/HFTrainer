#!/usr/bin/env python3
"""Convert prepared G1 qpos-style NPZ files to ProtoMotions .motion files.

This is a small argparse wrapper for the jump benchmark path.  It intentionally
avoids ProtoMotions' Typer CLI because the shared workspace currently has an
older Typer version, while the underlying ProtoMotions FK utilities work.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _add_proto_paths(proto_root: Path) -> None:
    for p in (proto_root, proto_root / "data" / "scripts"):
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def _convert_one(npz_path: Path, out_path: Path, kinematic_info, output_fps: int) -> None:
    from protomotions.components.pose_lib import (
        compute_cartesian_velocity,
        extract_qpos_from_transforms,
        extract_transforms_from_qpos,
        fk_from_transforms_with_velocities,
    )
    from contact_detection import compute_contact_labels_from_pos_and_vel

    data = np.load(npz_path, allow_pickle=True)
    root_pos = torch.from_numpy(data["base_frame_pos"].astype(np.float32))
    root_rot_wxyz = torch.from_numpy(data["base_frame_wxyz"].astype(np.float32))
    joint_angles = torch.from_numpy(data["joint_angles"].astype(np.float32))

    if joint_angles.shape[-1] != kinematic_info.num_dofs:
        raise ValueError(
            f"{npz_path}: joint angle columns {joint_angles.shape[-1]} != "
            f"expected {kinematic_info.num_dofs}"
        )

    qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)
    root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(
        kinematic_info, qpos
    )
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos_from_qpos,
        joint_rot_mats=joint_rot_mats,
        fps=output_fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    qpos_wrapped = extract_qpos_from_transforms(
        kinematic_info, root_pos_from_qpos, joint_rot_mats
    )
    motion.dof_pos = qpos_wrapped[:, 7:]
    motion.dof_vel = compute_cartesian_velocity(
        batched_robot_pos=joint_angles.unsqueeze(1), fps=output_fps
    ).squeeze(1)

    translation_vecs = motion.fix_height_per_frame(height_offset=0.02)
    if motion.rigid_body_vel is not None and motion.fps is not None:
        vel_delta = torch.zeros(
            translation_vecs.shape[0],
            1,
            3,
            device=motion.rigid_body_vel.device,
            dtype=motion.rigid_body_vel.dtype,
        )
        vel_delta[:-1] = (
            (translation_vecs[1:] - translation_vecs[:-1]).unsqueeze(1)
            / motion.motion_dt
        )
        motion.rigid_body_vel = motion.rigid_body_vel + vel_delta
    motion.fix_height(height_offset=0.04)
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)
    motion.local_rigid_body_rot = None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(motion.to_dict(), str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--proto-root", type=Path, default=Path("ref_repo/ProtoMotions"))
    ap.add_argument("--output-fps", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    proto_root = args.proto_root.resolve()
    _add_proto_paths(proto_root)
    from protomotions.components.pose_lib import extract_kinematic_info

    mjcf = proto_root / "protomotions" / "data" / "assets" / "mjcf" / "g1_bm_box_feet.xml"
    if not mjcf.exists():
        raise FileNotFoundError(mjcf)
    kinematic_info = extract_kinematic_info(str(mjcf))

    files = sorted(args.input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under {args.input_dir}")

    with torch.no_grad():
        for npz_path in files:
            out_path = (args.output_dir / npz_path.name).with_suffix(".motion")
            if out_path.exists() and not args.force:
                print(f"skip {out_path}")
                continue
            _convert_one(npz_path, out_path, kinematic_info, args.output_fps)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
