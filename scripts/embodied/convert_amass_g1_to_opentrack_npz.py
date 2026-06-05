#!/usr/bin/env python3
"""Convert AMASS_Retarged_for_G1 files to OpenTrack qpos/qvel NPZ files."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


def _fps(value) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar fps, got {np.asarray(value).shape}")
    return float(arr[0])


def _safe_stem(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return "__".join(rel.parts).replace(" ", "_")


def _quat_to_wxyz(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "wxyz":
        return quat
    if order == "xyzw":
        return quat[:, [3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order: {order}")


def _resample_motion(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    source_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(source_fps - output_fps) < 1e-6:
        return root_pos, root_quat_wxyz, dof_pos

    n = root_pos.shape[0]
    duration = (n - 1) / source_fps
    src_t = np.arange(n, dtype=np.float64) / source_fps
    out_n = int(round(duration * output_fps)) + 1
    dst_t = np.arange(out_n, dtype=np.float64) / output_fps
    dst_t[-1] = min(dst_t[-1], src_t[-1])

    out_root = np.stack(
        [np.interp(dst_t, src_t, root_pos[:, i]) for i in range(3)], axis=-1
    )
    out_dof = np.stack(
        [np.interp(dst_t, src_t, dof_pos[:, i]) for i in range(dof_pos.shape[1])],
        axis=-1,
    )

    # scipy Rotation uses xyzw, while OpenTrack/MuJoCo qpos uses wxyz.
    src_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    out_xyzw = Slerp(src_t, Rotation.from_quat(src_xyzw))(dst_t).as_quat()
    out_wxyz = out_xyzw[:, [3, 0, 1, 2]]
    return (
        out_root.astype(np.float32),
        out_wxyz.astype(np.float32),
        out_dof.astype(np.float32),
    )


def _quat_step_angvel_wxyz(qpos: np.ndarray, fps: float) -> np.ndarray:
    quat = qpos[:, 3:7]
    inv = np.concatenate([quat[:, :1], -quat[:, 1:]], axis=1)
    q1 = inv[:-1]
    q2 = quat[1:]
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    s = 2 * (w**2) - 1
    angle = np.arccos(np.clip(s, -1, 1))
    axis = np.stack([x, y, z], axis=1)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True).clip(min=1e-9)
    return axis * angle[:, None] * fps


def _build_qvel(qpos: np.ndarray, fps: float) -> np.ndarray:
    qvel = np.zeros((qpos.shape[0], qpos.shape[1] - 1), dtype=np.float32)
    qvel[1:, :3] = (qpos[1:, :3] - qpos[:-1, :3]) * fps
    qvel[1:, 3:6] = _quat_step_angvel_wxyz(qpos, fps)
    qvel[1:, 6:] = (qpos[1:, 7:] - qpos[:-1, 7:]) * fps
    return qvel


def _convert_one(
    source: Path,
    source_root: Path,
    output_dir: Path,
    joint_names: list[str],
    quat_order: str,
    output_fps: float,
    force: bool,
) -> str:
    name = _safe_stem(source_root, source)
    out_path = output_dir / f"{name}.npz"
    if out_path.exists() and not force:
        return name

    data = np.load(source, allow_pickle=True)
    source_dofs = [str(x) for x in data["dof_names"].tolist()]
    dof_order = [source_dofs.index(name) for name in joint_names[1:]]
    source_bodies = [str(x) for x in data["body_names"].tolist()]
    root_idx = source_bodies.index("pelvis")

    root_pos = data["body_positions"][:, root_idx].astype(np.float32)
    root_quat = _quat_to_wxyz(
        data["body_rotations"][:, root_idx].astype(np.float32), quat_order
    )
    dof_pos = data["dof_positions"][:, dof_order].astype(np.float32)
    root_pos, root_quat, dof_pos = _resample_motion(
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        dof_pos=dof_pos,
        source_fps=_fps(data["fps"]),
        output_fps=output_fps,
    )

    qpos = np.concatenate([root_pos, root_quat, dof_pos], axis=1).astype(np.float32)
    qvel = _build_qvel(qpos, output_fps)
    split_points = np.array([0, qpos.shape[0]], dtype=np.int32)
    jnt_type = np.array([0] + [3] * (len(joint_names) - 1), dtype=np.int32)
    np.savez_compressed(
        out_path,
        qpos=qpos,
        qvel=qvel,
        split_points=split_points,
        joint_names=np.array(joint_names),
        frequency=np.array(float(output_fps), dtype=np.float32),
        njnt=np.array(len(joint_names), dtype=np.int32),
        jnt_type=jnt_type,
        body_names=np.array(None, dtype=object),
        site_names=np.array(None, dtype=object),
        metadata=np.array({"source": str(source)}, dtype=object),
    )
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output-fps", type=float, default=50.0)
    parser.add_argument("--quat-order", choices=["wxyz", "xyzw"], default="wxyz")
    parser.add_argument("--num-rank", type=int, default=1)
    parser.add_argument("--slurm-rank", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.num_rank <= 0 or not (0 <= args.slurm_rank < args.num_rank):
        raise ValueError("Invalid shard rank")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(p) for p in glob.glob(str(args.input_dir / "**" / "*.npz"), recursive=True))
    if args.max_files is not None:
        files = files[: args.max_files]

    names = []
    for path in tqdm(files, desc="AMASS-G1 to OpenTrack"):
        rel = path.relative_to(args.input_dir)
        h = int(hashlib.sha256(str(rel).encode("utf-8")).hexdigest(), 16)
        if h % args.num_rank != args.slurm_rank:
            continue
        names.append(
            _convert_one(
                source=path,
                source_root=args.input_dir,
                output_dir=args.output_dir,
                joint_names=joint_names,
                quat_order=args.quat_order,
                output_fps=args.output_fps,
                force=args.force,
            )
        )

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(names, indent=2) + "\n")
    print(f"converted_or_existing={len(names)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
