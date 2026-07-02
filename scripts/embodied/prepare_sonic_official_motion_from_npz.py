#!/usr/bin/env python3
"""Convert a unified G1 qpos NPZ into SONIC's official robot-motion pkl."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


DOF_AXIS_INDEX = np.array(
    [
        1,
        0,
        2,
        1,
        1,
        0,
        1,
        0,
        2,
        1,
        1,
        0,
        2,
        0,
        1,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
        1,
        0,
        2,
        1,
        0,
        1,
        2,
    ],
    dtype=np.int64,
)


def _resample_qpos(qpos: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    if abs(source_fps - target_fps) < 1e-6 or qpos.shape[0] <= 1:
        return qpos.astype(np.float64, copy=False)

    src_t = np.arange(qpos.shape[0], dtype=np.float64) / source_fps
    duration = src_t[-1]
    dst_n = int(round(duration * target_fps)) + 1
    dst_t = np.arange(dst_n, dtype=np.float64) / target_fps
    dst_t[-1] = min(dst_t[-1], duration)

    out = np.empty((dst_n, qpos.shape[1]), dtype=np.float64)
    out[:, :3] = np.stack([np.interp(dst_t, src_t, qpos[:, i]) for i in range(3)], axis=-1)
    src_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    out[:, 3:7] = Slerp(src_t, R.from_quat(src_xyzw))(dst_t).as_quat()[:, [3, 0, 1, 2]]
    out[:, 7:] = np.stack(
        [np.interp(dst_t, src_t, qpos[:, i]) for i in range(7, qpos.shape[1])],
        axis=-1,
    )
    return out


def _load_qpos(npz_path: Path, target_fps: float) -> tuple[np.ndarray, float]:
    pack = np.load(npz_path, allow_pickle=True)
    qpos = np.asarray(pack["qpos"], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36:
        raise ValueError(f"{npz_path}: expected qpos shape (T, 36), got {qpos.shape}")
    source_fps = float(np.asarray(pack["frequency"]).reshape(-1)[0]) if "frequency" in pack.files else 30.0
    return _resample_qpos(qpos, source_fps, target_fps), source_fps


def qpos_to_sonic_motion(qpos: np.ndarray, fps: int) -> dict[str, np.ndarray | int]:
    """Build the motion dict consumed by SONIC's released IsaacLab evaluator.

    Unified G1 qpos uses MuJoCo order: root xyz, root quat wxyz, then 29 dofs.
    SONIC's official robot-motion pkl stores root quat as xyzw plus a 30-body
    axis-angle pose tensor where body 0 is the root and bodies 1: map each
    1-DoF joint onto the axis used by the release sample files.
    """

    root_xyzw = qpos[:, 3:7][:, [1, 2, 3, 0]]
    dof = qpos[:, 7:]
    if dof.shape[1] != len(DOF_AXIS_INDEX):
        raise ValueError(f"expected 29 dofs, got {dof.shape[1]}")

    pose_aa = np.zeros((qpos.shape[0], 30, 3), dtype=np.float32)
    pose_aa[:, 0, :] = R.from_quat(root_xyzw).as_rotvec().astype(np.float32)
    for i, axis_idx in enumerate(DOF_AXIS_INDEX):
        pose_aa[:, i + 1, axis_idx] = dof[:, i].astype(np.float32)

    return {
        "root_trans_offset": qpos[:, :3].astype(np.float32),
        "pose_aa": pose_aa,
        "dof": dof.astype(np.float32),
        "root_rot": root_xyzw.astype(np.float32),
        "smpl_joints": np.zeros((qpos.shape[0], 24, 3), dtype=np.float32),
        "fps": int(fps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if abs(args.target_fps - round(args.target_fps)) > 1e-6:
        raise ValueError("SONIC official motion pkl expects integer fps")
    name = args.name or args.npz.stem
    out_path = args.out_dir / f"{name}.pkl"
    if out_path.exists() and not args.force:
        print(out_path)
        return

    qpos, source_fps = _load_qpos(args.npz, args.target_fps)
    motion = qpos_to_sonic_motion(qpos, int(round(args.target_fps)))
    payload = {name: motion}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(
        "{\n"
        f'  "name": "{name}",\n'
        f'  "source_npz": "{args.npz}",\n'
        f'  "source_fps": {source_fps:.6f},\n'
        f'  "target_fps": {args.target_fps:.6f},\n'
        f'  "num_frames": {qpos.shape[0]}\n'
        "}\n"
    )
    print(out_path)


if __name__ == "__main__":
    main()
