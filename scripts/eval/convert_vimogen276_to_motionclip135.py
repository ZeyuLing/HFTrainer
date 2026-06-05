#!/usr/bin/env python3
"""Convert ViMoGen 276D outputs to MotionCLIP-evaluator 135D motions."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hftrainer.models.motion.components.utils.geometry.rotation_convert import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
)

MBENCH_COORD_CONVERSION = torch.tensor(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)


def resample_motion(motion: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"fps must be positive, got src={src_fps}, dst={dst_fps}")
    if abs(src_fps - dst_fps) < 1e-6 or len(motion) < 2:
        return motion.astype(np.float32, copy=False)
    out_len = max(1, int(round(len(motion) * dst_fps / src_fps)))
    src_t = np.linspace(0.0, 1.0, len(motion), dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    out = np.empty((out_len, motion.shape[1]), dtype=np.float32)
    for dim in range(motion.shape[1]):
        out[:, dim] = np.interp(dst_t, src_t, motion[:, dim])
    return out


def import_vimogen_retarget(vimogen_root: Path):
    old_cwd = Path.cwd()
    sys.path.insert(0, str(vimogen_root))
    os.chdir(vimogen_root)
    try:
        from motion_rep.retarget_motion import motion_rep_to_SMPL  # type: ignore
    finally:
        os.chdir(old_cwd)
    return motion_rep_to_SMPL


def to_motionclip135(
    motion276: torch.Tensor,
    motion_rep_to_SMPL,
    *,
    recover_from_velocity: bool = True,
    equal_length: bool = True,
    coord_conversion: str = "mbench",
) -> np.ndarray:
    smpl, _ = motion_rep_to_SMPL(
        motion276.float(),
        recover_from_velocity=recover_from_velocity,
        equal_length=equal_length,
    )
    transl = smpl["transl"].detach().cpu().float().numpy()
    t = transl.shape[0]
    global_orient = smpl["global_orient"].detach().cpu().float().reshape(t, 3)
    body_pose = smpl["body_pose"].detach().cpu().float().reshape(t, 21, 3)
    if coord_conversion == "mbench":
        convert = MBENCH_COORD_CONVERSION.to(global_orient)
        transl_t = torch.from_numpy(transl).to(global_orient)
        transl = torch.einsum("ij,tj->ti", convert, transl_t).cpu().numpy()
        global_rot = axis_angle_to_matrix(global_orient)
        global_orient = matrix_to_axis_angle(
            torch.einsum("ij,tjk->tik", convert, global_rot)
        ).reshape(t, 3)
    elif coord_conversion != "none":
        raise ValueError(f"unsupported coord conversion: {coord_conversion}")
    go6 = matrix_to_rotation_6d(axis_angle_to_matrix(global_orient)).numpy().reshape(t, 6)
    bp6 = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).numpy().reshape(t, 126)
    out = np.concatenate([transl, go6, bp6], axis=-1).astype(np.float32)
    if out.shape[-1] != 135:
        raise ValueError(f"expected 135D output, got {out.shape}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vimogen-root", default="ref_repo/ViMoGen")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pattern", default="step*/**/motion_gen_condition_on_text.pt")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-recover-from-velocity", action="store_true")
    parser.add_argument("--no-equal-length", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--src-fps", type=float, default=20.0,
                        help="Frame rate of ViMoGen motion latents.")
    parser.add_argument("--dst-fps", type=float, default=20.0,
                        help="Frame rate to write for evaluator input; 20 keeps ViMoGen's native rate.")
    parser.add_argument("--coord-conversion", choices=["mbench", "none"], default="mbench")
    args = parser.parse_args()

    vimogen_root = Path(args.vimogen_root).resolve()
    input_root = Path(args.input_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_rep_to_SMPL = import_vimogen_retarget(vimogen_root)

    files = sorted(input_root.glob(args.pattern))
    if args.max_files:
        files = files[:args.max_files]
    ok = failed = skipped = 0
    for idx, path in enumerate(files, 1):
        sample_id = path.parent.name
        out_path = out_dir / f"{sample_id}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            motion = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(motion, dict):
                motion = motion["motion"]
            if motion.ndim == 3:
                motion = motion[0]
            motion135 = to_motionclip135(
                motion,
                motion_rep_to_SMPL,
                recover_from_velocity=not args.no_recover_from_velocity,
                equal_length=not args.no_equal_length,
                coord_conversion=args.coord_conversion,
            )
            motion135 = resample_motion(motion135, args.src_fps, args.dst_fps)
            np.save(out_path, motion135)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            if failed <= 20:
                print(f"[fail] {path}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 200 == 0:
            print(f"[progress] {idx}/{len(files)} ok={ok} skipped={skipped} failed={failed}", flush=True)
    print(f"[done] input={input_root} out={out_dir} files={len(files)} ok={ok} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
