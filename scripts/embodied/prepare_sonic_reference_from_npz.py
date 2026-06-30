#!/usr/bin/env python3
"""Convert one unified G1 qpos NPZ into a SONIC reference-motion folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


MUJOCO_TO_ISAACLAB = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ],
    dtype=np.int64,
)


def _write_csv(path: Path, header: list[str], values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        np.savetxt(f, values, delimiter=",", fmt="%.9f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    pack = np.load(args.npz, allow_pickle=True)
    qpos = np.asarray(pack["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{args.npz}: expected qpos shape (T, 36), got {qpos.shape}")
    fps = float(np.asarray(pack["frequency"]).reshape(-1)[0]) if "frequency" in pack.files else 30.0
    name = args.name or args.npz.stem
    out_dir = args.out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    joint_pos = qpos[:, 7:][:, MUJOCO_TO_ISAACLAB]
    joint_vel = np.zeros_like(joint_pos)
    if len(joint_pos) > 1:
        joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) * fps
        joint_vel[0] = joint_vel[1]

    _write_csv(out_dir / "body_pos.csv", ["root_x", "root_y", "root_z"], qpos[:, :3])
    _write_csv(out_dir / "body_quat.csv", ["root_w", "root_x", "root_y", "root_z"], qpos[:, 3:7])
    _write_csv(out_dir / "joint_pos.csv", [f"joint_{i}" for i in range(29)], joint_pos)
    _write_csv(out_dir / "joint_vel.csv", [f"joint_vel_{i}" for i in range(29)], joint_vel)
    (out_dir / "metadata.txt").write_text(
        f"Metadata for: {name}\n"
        "==============================\n"
        f"source_npz: {args.npz}\n"
        f"num_frames: {len(qpos)}\n"
        f"fps: {fps:.6f}\n"
    )
    print(out_dir)


if __name__ == "__main__":
    main()
