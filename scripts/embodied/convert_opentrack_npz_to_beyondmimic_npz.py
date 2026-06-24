#!/usr/bin/env python3
"""Convert OpenTrack G1 qpos/qvel NPZ files to BeyondMimic motion NPZ files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import mujoco
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML = PROJECT_ROOT / "ref_repo/OpenTrack/storage/assets/unitree_g1/scene_mjx_flat_terrain.xml"
DEFAULT_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def quat_ang_vel_wxyz(quat: np.ndarray, fps: float) -> np.ndarray:
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
    out = np.zeros((quat.shape[0], 3), dtype=np.float32)
    out[1:] = axis * angle[:, None] * fps
    return out


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    out[1:] = (values[1:] - values[:-1]) * fps
    return out


def scalar(value) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got {np.asarray(value).shape}")
    return float(arr[0])


def joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ][1:]


def as_string_list(value: np.ndarray) -> list[str]:
    arr = np.asarray(value, dtype=object).reshape(-1)
    return [str(x) for x in arr]


def expand_to_model_state(data: np.lib.npyio.NpzFile, model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos = data["qpos"].astype(np.float64)
    qvel = data["qvel"].astype(np.float64)
    if qpos.shape[1] == model.nq and qvel.shape[1] == model.nv:
        return qpos, qvel

    qpos_full = np.repeat(model.qpos0[None, :], qpos.shape[0], axis=0).astype(np.float64)
    qvel_full = np.zeros((qvel.shape[0], model.nv), dtype=np.float64)
    qpos_full[:, :7] = qpos[:, :7]
    qvel_full[:, :6] = qvel[:, :6]

    if "joint_names" not in data:
        raise ValueError(
            f"Reduced qpos/qvel shapes {qpos.shape}/{qvel.shape} require joint_names for mapping "
            f"to model nq/nv {model.nq}/{model.nv}."
        )
    src_names = as_string_list(data["joint_names"])
    if src_names and src_names[0] == "root":
        src_names = src_names[1:]
    if len(src_names) != qpos.shape[1] - 7:
        raise ValueError(
            f"joint_names length {len(src_names)} does not match qpos joint dim {qpos.shape[1] - 7}."
        )

    for src_idx, name in enumerate(src_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Source joint {name!r} is missing from target MuJoCo model.")
        qpos_full[:, model.jnt_qposadr[joint_id]] = qpos[:, 7 + src_idx]
        qvel_full[:, model.jnt_dofadr[joint_id]] = qvel[:, 6 + src_idx]
    return qpos_full, qvel_full


def convert_one(src: Path, dst: Path, model: mujoco.MjModel, body_names: list[str], force: bool) -> str:
    if dst.exists() and not force:
        return dst.stem

    data = np.load(src, allow_pickle=True)
    qpos, qvel = expand_to_model_state(data, model)
    fps = scalar(data["frequency"] if "frequency" in data else 50.0)

    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if any(i < 0 for i in body_ids):
        missing = [name for name, idx in zip(body_names, body_ids) if idx < 0]
        raise ValueError(f"Missing body names in MuJoCo model: {missing}")

    mj_data = mujoco.MjData(model)
    body_pos = np.zeros((qpos.shape[0], len(body_names), 3), dtype=np.float32)
    body_quat = np.zeros((qpos.shape[0], len(body_names), 4), dtype=np.float32)
    for i in range(qpos.shape[0]):
        mj_data.qpos[:] = qpos[i]
        mj_data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, mj_data)
        body_pos[i] = mj_data.xpos[body_ids]
        body_quat[i] = mj_data.xquat[body_ids]

    body_lin_vel = finite_difference(body_pos, fps)
    body_ang_vel = np.stack([quat_ang_vel_wxyz(body_quat[:, i], fps) for i in range(len(body_names))], axis=1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst,
        fps=np.array(fps, dtype=np.float32),
        joint_names=np.array(joint_names(model)),
        body_names=np.array(body_names),
        joint_pos=qpos[:, 7:].astype(np.float32),
        joint_vel=qvel[:, 6:].astype(np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        source=np.array(str(src)),
    )
    return dst.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--input-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--body-names", default=",".join(DEFAULT_BODY_NAMES))
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    files = list(args.input)
    for input_dir in args.input_dir:
        files.extend(Path(p) for p in glob.glob(str(input_dir / "**" / "*.npz"), recursive=True))
    files = sorted(dict.fromkeys(files))
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise SystemExit("No input files.")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    body_names = [x.strip() for x in args.body_names.split(",") if x.strip()]
    names = []
    for src in tqdm(files, desc="OpenTrack to BeyondMimic"):
        names.append(convert_one(src, args.output_dir / src.name, model, body_names, args.force))
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"motions": names, "output_dir": str(args.output_dir)}, indent=2) + "\n")
    print(f"converted={len(names)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
