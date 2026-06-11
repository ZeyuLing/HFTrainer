#!/usr/bin/env python3
"""Retarget HumanML3D-263 predictions to SMPL-style motion_135 (thin CLI).

The IK implementation now lives in the public motion library at
``hftrainer.motion.retarget.hml263_smpl``; this script is a thin batch/IO wrapper
around :func:`hftrainer.motion.retarget.hml263_smpl.retarget_hml263_clip`.

    HML3D-263 -> 22 joints -> hierarchical IK on SMPL rest skeleton
              -> global_orient/body_pose/transl + motion_135

The conversion is not mathematically exact: HumanML3D-263 does not uniquely
determine SMPL pose twist, shape, or mesh details. The saved fit MPJPE is a
diagnostic for how well the SMPL skeleton tracks the recovered 22 joints.

NOTE on rot6d: this CLI defaults to ``--rot6d-convention column`` (MotionCLIP
evaluator) for backward compatibility. For the MS272 chain use ``row`` (or just
call ``hftrainer.motion.representation.convert.hml263_to_motion272``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hftrainer.motion.retarget.hml263_smpl import (  # noqa: E402
    load_gmm_pose_prior,
    load_smpl_rest,
    retarget_hml263_clip,
)


def retarget_one(in_path: Path, out_path: Path, smpl_rest, mean, std, gmm_pose_prior, args) -> dict:
    arr = np.load(str(in_path)).astype(np.float32)
    # Joint-native inputs (e.g. CondMDI) arrive as (T,22,3); IK runs directly on
    # the world joints. HML263 inputs arrive as (T,263).
    joints_world = None
    feats = None
    if arr.ndim == 3 and arr.shape[1:] == (22, 3):
        joints_world = arr
    else:
        feats = arr
        if feats.ndim != 2 or feats.shape[-1] != 263:
            raise ValueError(f"expected (T,263) or (T,22,3), got {feats.shape}")
        if mean is not None and std is not None:
            feats = feats * std + mean

    out = retarget_hml263_clip(
        feats,
        target_joints_world=joints_world,
        smpl_rest=smpl_rest,
        device=args.device,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        batch_size=args.batch_size,
        floor_align=args.floor_align,
        refine_iters=args.refine_iters,
        refine_lr=args.refine_lr,
        rotation_init=args.rotation_init,
        orientation_mode=args.orientation_mode,
        parent_ref_weight=args.parent_ref_weight,
        pose_l2_weight=args.pose_l2_weight,
        angle_prior_weight=args.angle_prior_weight,
        smooth_weight=args.smooth_weight,
        joint_accel_weight=args.joint_accel_weight,
        joint_fit_weight_preset=args.joint_fit_weight_preset,
        gmm_pose_prior=gmm_pose_prior,
        gmm_pose_prior_weight=args.gmm_pose_prior_weight,
        rot6d_convention=args.rot6d_convention,
    )
    mpjpe_mm = out["fit_mpjpe_mm"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        motion_135=out["motion_135"],
        transl=out["transl"],
        global_orient=out["global_orient"],
        body_pose=out["body_pose"],
        target_joints=out["target_joints"],
        fitted_joints=out["fitted_joints"],
        fit_mpjpe_mm=mpjpe_mm,
        source_fps=np.array(args.source_fps, dtype=np.float32),
        target_fps=np.array(args.target_fps, dtype=np.float32),
        refine_iters=np.array(args.refine_iters, dtype=np.int32),
        rot6d_convention=np.array(args.rot6d_convention),
    )
    return {
        "sid": in_path.stem,
        "frames": int(out["target_joints"].shape[0]),
        "mpjpe_mm_mean": float(mpjpe_mm.mean()),
        "mpjpe_mm_p95": float(np.percentile(mpjpe_mm, 95)),
    }


def iter_files(
    in_dir: Path,
    ids_file: Path | None,
    limit: int | None,
    num_shards: int,
    shard_index: int,
) -> Iterable[Path]:
    if ids_file is not None:
        files = [in_dir / f"{line.strip()}.npy" for line in ids_file.read_text().splitlines() if line.strip()]
    else:
        files = sorted(in_dir.glob("*.npy"))
    files = [p for p in files if p.exists()]
    if num_shards > 1:
        files = [p for i, p in enumerate(files) if i % num_shards == shard_index]
    if limit:
        files = files[:limit]
    return files


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default="ref_repo/MDM/body_models")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--source-fps", type=float, default=20.0)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--floor-align", action="store_true")
    ap.add_argument("--refine-iters", type=int, default=0)
    ap.add_argument("--refine-lr", type=float, default=2e-2)
    ap.add_argument("--smooth-weight", type=float, default=1e-3)
    ap.add_argument("--joint-accel-weight", type=float, default=0.0)
    ap.add_argument("--pose-l2-weight", type=float, default=0.0)
    ap.add_argument("--angle-prior-weight", type=float, default=0.0)
    ap.add_argument("--gmm-pose-prior-weight", type=float, default=0.0)
    ap.add_argument(
        "--joint-fit-weight-preset",
        choices=["uniform", "relaxed_torso", "relaxed_upper"],
        default="uniform",
    )
    ap.add_argument("--foot-height-align", action="store_true", default=False)
    ap.add_argument("--no-foot-height-align", dest="foot_height_align", action="store_false")
    ap.add_argument(
        "--rot6d-convention",
        choices=["column", "row"],
        default="column",
        help="6D layout used for saved motion_135. MotionCLIP evaluator uses column; "
        "MS272 chain uses row.",
    )
    ap.add_argument(
        "--rotation-init",
        choices=["position", "hml263"],
        default="position",
        help="Initialize SMPL pose from position-only IK or from the HumanML3D 126-D local rotation block.",
    )
    ap.add_argument(
        "--orientation-mode",
        choices=["bone", "parent_frame"],
        default="bone",
        help="Use parent_frame to add a weak joint-to-parent reference that stabilizes unconstrained twist.",
    )
    ap.add_argument("--parent-ref-weight", type=float, default=0.25)
    ap.add_argument("--input-normalized", action="store_true")
    ap.add_argument(
        "--mean-path",
        default="ref_repo/Momask/weights/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/mean.npy",
    )
    ap.add_argument(
        "--std-path",
        default="ref_repo/Momask/weights/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/std.npy",
    )
    args = ap.parse_args()

    args.device = torch.device(args.device)
    smpl_rest = load_smpl_rest(args.model_dir, args.device)
    gmm_pose_prior = load_gmm_pose_prior(args.device) if args.gmm_pose_prior_weight > 0 else None
    if args.input_normalized:
        mean = np.load(args.mean_path).astype(np.float32)
        std = np.load(args.std_path).astype(np.float32)
        if mean.shape != (263,) or std.shape != (263,):
            raise ValueError(f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")
    else:
        mean = std = None
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"invalid shard args: {args.shard_index}/{args.num_shards}")
    files = list(iter_files(
        Path(args.in_dir),
        Path(args.ids) if args.ids else None,
        args.limit,
        args.num_shards,
        args.shard_index,
    ))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] files={len(files)} shard={args.shard_index}/{args.num_shards} "
        f"out={out_dir} device={args.device} target_fps={args.target_fps}",
        flush=True,
    )

    summary = []
    failed = 0
    for i, in_path in enumerate(files, 1):
        out_path = out_dir / f"{in_path.stem}.npz"
        if args.skip_existing and out_path.exists():
            continue
        try:
            item = retarget_one(in_path, out_path, smpl_rest, mean, std, gmm_pose_prior, args)
            summary.append(item)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {in_path.name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 25 == 0 or i == len(files):
            running = np.mean([x["mpjpe_mm_mean"] for x in summary]) if summary else float("nan")
            print(f"[progress] {i}/{len(files)} ok={len(summary)} fail={failed} mean_mpjpe_mm={running:.2f}", flush=True)

    if summary:
        stats = {
            "count": len(summary),
            "failed": failed,
            "mean_mpjpe_mm": float(np.mean([x["mpjpe_mm_mean"] for x in summary])),
            "median_mpjpe_mm": float(np.median([x["mpjpe_mm_mean"] for x in summary])),
            "p95_frame_mpjpe_mm_mean": float(np.mean([x["mpjpe_mm_p95"] for x in summary])),
            "items": summary[:100],
        }
    else:
        stats = {"count": 0, "failed": failed}
    if args.num_shards > 1:
        summary_name = f"_retarget_summary_s{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    else:
        summary_name = "_retarget_summary.json"
    (out_dir / summary_name).write_text(json.dumps(stats, indent=2))
    print(f"[done] {json.dumps({k: v for k, v in stats.items() if k != 'items'}, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
