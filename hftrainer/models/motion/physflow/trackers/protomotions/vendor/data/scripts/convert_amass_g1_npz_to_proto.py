# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
"""Convert AMASS_Retarged_for_G1 NPZ files to ProtoMotions .motion files.

The public AMASS-G1 release used in PhysFlow stores retargeted trajectories as:
    fps, dof_names, body_names, dof_positions, dof_velocities,
    body_positions, body_rotations, ...

ProtoMotions' G1 tracker expects reference motions in its own 33-body layout.
This converter therefore uses only the source pelvis pose plus 29 G1 DOFs, then
recomputes the 33-body FK with the official ProtoMotions G1 MJCF.
"""

from __future__ import annotations

import glob
import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from tqdm import tqdm

from protomotions.components.pose_lib import (
    compute_cartesian_velocity,
    extract_kinematic_info,
    extract_qpos_from_transforms,
    extract_transforms_from_qpos,
    fk_from_transforms_with_velocities,
)
from protomotions.robot_configs.factory import robot_config
from contact_detection import compute_contact_labels_from_pos_and_vel
from motion_filter import passes_exclude_motion_filter

app = typer.Typer()


def _as_scalar_fps(value) -> int:
    arr = np.asarray(value).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar fps, got shape {np.asarray(value).shape}")
    return int(round(float(arr[0])))


def _quat_to_wxyz(quat: np.ndarray, quat_order: str) -> np.ndarray:
    if quat_order == "wxyz":
        return quat
    if quat_order == "xyzw":
        return quat[:, [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order: {quat_order}")


def _load_amass_g1_npz(
    npz_path: Path,
    output_fps: int,
    expected_dof_names: list[str],
    quat_order: str,
    device: torch.device,
    dtype: torch.dtype,
):
    data = np.load(npz_path, allow_pickle=True)

    source_fps = _as_scalar_fps(data["fps"])
    if source_fps % output_fps != 0:
        raise ValueError(
            f"{npz_path}: source fps {source_fps} must be divisible by output_fps {output_fps}"
        )
    factor = source_fps // output_fps

    source_dof_names = [str(x) for x in data["dof_names"].tolist()]
    missing = [name for name in expected_dof_names if name not in source_dof_names]
    if missing:
        raise ValueError(f"{npz_path}: missing DOFs required by G1 MJCF: {missing}")
    dof_order = [source_dof_names.index(name) for name in expected_dof_names]

    source_body_names = [str(x) for x in data["body_names"].tolist()]
    try:
        root_idx = source_body_names.index("pelvis")
    except ValueError as exc:
        raise ValueError(f"{npz_path}: body_names has no pelvis root") from exc

    root_pos = data["body_positions"][::factor, root_idx].astype(np.float32)
    root_rot_wxyz = _quat_to_wxyz(
        data["body_rotations"][::factor, root_idx].astype(np.float32), quat_order
    )
    joint_angles = data["dof_positions"][::factor][:, dof_order].astype(np.float32)

    root_pos_t = torch.from_numpy(root_pos).to(device=device, dtype=dtype)
    root_rot_t = torch.from_numpy(root_rot_wxyz).to(device=device, dtype=dtype)
    joint_t = torch.from_numpy(joint_angles).to(device=device, dtype=dtype)
    return root_pos_t, root_rot_t, joint_t, source_fps


@app.command()
def main(
    input_dir: Path = typer.Option(
        ..., help="Root directory containing recursive AMASS-G1 *_jpos.npz files."
    ),
    output_dir: Path = typer.Option(..., help="Directory to save .motion files."),
    output_fps: int = typer.Option(30, help="Output motion fps."),
    robot_type: str = typer.Option("g1", help="Robot type, default Unitree G1."),
    quat_order: str = typer.Option(
        "wxyz", help="Quaternion order in source body_rotations: xyzw or wxyz."
    ),
    force_remake: bool = False,
    max_files: Optional[int] = typer.Option(None, help="Optional max files for smoke runs."),
    start_index: int = typer.Option(0, help="Start index after sorted recursive file list."),
    num_rank: int = typer.Option(1, help="Total deterministic shards."),
    slurm_rank: int = typer.Option(0, help="Shard rank in [0, num_rank)."),
    apply_motion_filter: bool = typer.Option(False, help="Apply motion quality filter."),
    min_height_threshold: float = typer.Option(-0.05),
    max_velocity_threshold: float = typer.Option(15.0),
    max_dof_vel_threshold: float = typer.Option(40.0),
    duration_height_filter: float = typer.Option(0.1),
    duration_height_seconds: float = typer.Option(0.6),
):
    if num_rank <= 0 or not (0 <= slurm_rank < num_rank):
        raise ValueError("--num-rank must be positive and --slurm-rank in range")

    device = torch.device("cpu")
    dtype = torch.float32
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_mjcf_mapping = {"g1": "g1_bm_box_feet.xml", "h1_2": "h1_2.xml"}
    mjcf_filename = robot_mjcf_mapping.get(robot_type, f"{robot_type}.xml")
    mjcf_path = f"protomotions/data/assets/mjcf/{mjcf_filename}"
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found at {mjcf_path}")

    kinematic_info = extract_kinematic_info(mjcf_path)
    robot_cfg = robot_config(robot_type)
    expected_dof_names = list(kinematic_info.dof_names)
    print(
        f"Robot type: {robot_type}, expected_dofs={len(expected_dof_names)}, "
        f"output_fps={output_fps}, quat_order={quat_order}"
    )

    files = sorted(Path(p) for p in glob.glob(str(input_dir / "**" / "*.npz"), recursive=True))
    if start_index:
        files = files[start_index:]
    if max_files is not None:
        files = files[:max_files]
    print(f"Found {len(files)} NPZ files after slicing.")

    converted = 0
    skipped = 0
    for npz_path in tqdm(files, desc="Converting AMASS-G1"):
        rel_path = npz_path.relative_to(input_dir)
        file_hash = int(hashlib.sha256(str(rel_path).encode("utf-8")).hexdigest(), 16)
        if file_hash % num_rank != slurm_rank:
            continue

        outpath = (output_dir / str(rel_path).replace(" ", "_")).with_suffix(".motion")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        if outpath.exists() and not force_remake:
            skipped += 1
            continue

        try:
            root_pos, root_rot_wxyz, joint_angles, source_fps = _load_amass_g1_npz(
                npz_path=npz_path,
                output_fps=output_fps,
                expected_dof_names=expected_dof_names,
                quat_order=quat_order,
                device=device,
                dtype=dtype,
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
                kinematic_info, root_pos, joint_rot_mats
            )
            motion.dof_pos = qpos_wrapped[:, 7:]

            delta = (qpos_wrapped[:, 7:] - joint_angles).abs()
            allowed = torch.zeros_like(delta, dtype=torch.bool)
            for d in [0.0, 2 * np.pi, 4 * np.pi]:
                allowed |= (delta - d).abs() < 1e-4
            if not allowed.all():
                max_delta = delta[~allowed].max().item()
                raise ValueError(f"wrapped qpos diverges from source DOFs; max_delta={max_delta}")

            dof_vel = compute_cartesian_velocity(
                batched_robot_pos=joint_angles.unsqueeze(1), fps=output_fps
            )
            motion.dof_vel = dof_vel.squeeze(1)

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

            if apply_motion_filter and not passes_exclude_motion_filter(
                motion,
                min_height_threshold=min_height_threshold,
                max_velocity_threshold=max_velocity_threshold,
                max_dof_vel_threshold=max_dof_vel_threshold,
                duration_height_filter=duration_height_filter,
                duration_height_seconds=duration_height_seconds,
            ):
                skipped += 1
                continue

            torch.save(motion.to_dict(), str(outpath))
            converted += 1
        except Exception as exc:
            skipped += 1
            print(f"Error processing {npz_path}: {exc}")

    print(f"Done. converted={converted}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
