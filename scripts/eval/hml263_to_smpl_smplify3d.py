#!/usr/bin/env python3
"""Retarget HumanML3D-263 joints with the MDM/FlowMDM SMPLify3D fitter.

This is a diagnostic alternative to ``hml263_to_smpl_ik.py``.  It is much
slower because it optimizes SMPL parameters frame by frame, but it uses the
released joints2smpl objective with a GMM pose prior and can test whether the
lightweight hierarchical IK is the source of the HML263->SMPL quality gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "ref_repo" / "FlowMDM"))

from scripts.eval.hml263_to_smpl_ik import (  # noqa: E402
    N_JOINTS,
    recover_from_ric,
    resample_linear,
    matrix_to_rot6d,
)

from utils.visualize.joints2smpl.src import config as j2s_config  # noqa: E402

j2s_config.SMPL_MODEL_DIR = str(REPO / "ref_repo" / "MDM" / "body_models")
j2s_config.GMM_MODEL_DIR = str(REPO / "ref_repo" / "FlowMDM" / "utils" / "visualize" / "joints2smpl" / "smpl_models")
j2s_config.SMPL_MEAN_FILE = str(Path(j2s_config.GMM_MODEL_DIR) / "neutral_smpl_mean_params.h5")

import smplx  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from utils.visualize.joints2smpl.src.smplify import SMPLify3D  # noqa: E402


def iter_files(in_dir: Path, limit: int | None) -> Iterable[Path]:
    files = sorted(in_dir.glob("*.npy"))
    return files[:limit] if limit else files


def load_mean_params(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(j2s_config.SMPL_MEAN_FILE, "r") as f:
        mean_pose = torch.from_numpy(f["pose"][:]).unsqueeze(0).float().to(device)
        mean_shape = torch.from_numpy(f["shape"][:]).unsqueeze(0).float().to(device)
    return mean_pose, mean_shape


def smpl_forward_22(model, pose72: np.ndarray, transl: np.ndarray, device: torch.device) -> np.ndarray:
    chunks = []
    for i in range(len(pose72)):
        pose = torch.from_numpy(pose72[i:i + 1]).float().to(device)
        tr = torch.from_numpy(transl[i:i + 1]).float().to(device)
        with torch.no_grad():
            out = model(
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                transl=tr,
                betas=torch.zeros(1, 10, device=device),
            )
        chunks.append(out.joints[:, :N_JOINTS].detach().cpu().numpy()[0])
    return np.stack(chunks, axis=0).astype(np.float32)


def retarget_one(
    in_path: Path,
    out_path: Path,
    model,
    smplify: SMPLify3D,
    mean_pose: torch.Tensor,
    mean_shape: torch.Tensor,
    source_fps: float,
    target_fps: float,
    floor_align: bool,
    device: torch.device,
    rot6d_convention: str,
) -> dict:
    feats = np.load(str(in_path)).astype(np.float32)
    if feats.ndim != 2 or feats.shape[-1] != 263:
        raise ValueError(f"expected (T,263), got {feats.shape}")

    target = recover_from_ric(feats, N_JOINTS)
    target = resample_linear(target, source_fps, target_fps)
    if floor_align:
        target = target.copy()
        target[..., 1] -= target[..., 1].min()

    confidence = torch.ones(N_JOINTS, dtype=torch.float32, device=device)
    pred_pose = mean_pose.clone()
    pred_betas = mean_shape.clone()
    pred_cam_t = torch.zeros(1, 3, dtype=torch.float32, device=device)

    poses = []
    trans = []
    for idx, joints_np in enumerate(target):
        keypoints = torch.from_numpy(joints_np[None]).float().to(device)
        _, _, opt_pose, opt_betas, opt_cam_t, _ = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints,
            conf_3d=confidence,
            seq_ind=idx,
        )
        pred_pose = opt_pose.detach().float()
        pred_betas = opt_betas.detach().float()
        pred_cam_t = opt_cam_t.detach().reshape(1, 3).float()
        poses.append(pred_pose.detach().cpu().numpy()[0])
        trans.append(pred_cam_t.detach().cpu().numpy()[0])

    pose72 = np.stack(poses, axis=0).astype(np.float32)
    transl = np.stack(trans, axis=0).astype(np.float32)
    fitted = smpl_forward_22(model, pose72, transl, device)
    mpjpe_mm = np.linalg.norm(fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0

    global_orient = pose72[:, :3]
    body_pose_21 = pose72[:, 3:66]
    local_r = R.from_rotvec(
        np.concatenate([global_orient[:, None], body_pose_21.reshape(len(target), 21, 3)], axis=1)
        .reshape(-1, 3)
    ).as_matrix().reshape(len(target), N_JOINTS, 3, 3).astype(np.float32)
    motion_135 = np.concatenate(
        [transl, matrix_to_rot6d(local_r, rot6d_convention).reshape(len(target), N_JOINTS * 6)],
        axis=-1,
    ).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        motion_135=motion_135,
        transl=transl.astype(np.float32),
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose_21.astype(np.float32),
        target_joints=target.astype(np.float32),
        fitted_joints=fitted.astype(np.float32),
        fit_mpjpe_mm=mpjpe_mm,
        source_fps=np.array(source_fps, dtype=np.float32),
        target_fps=np.array(target_fps, dtype=np.float32),
        smplify3d=np.array(True),
        rot6d_convention=np.array(rot6d_convention),
    )
    return {
        "sid": in_path.stem,
        "frames": int(len(target)),
        "mpjpe_mm_mean": float(mpjpe_mm.mean()),
        "mpjpe_mm_p95": float(np.percentile(mpjpe_mm, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--source-fps", type=float, default=20.0)
    ap.add_argument("--target-fps", type=float, default=30.0)
    ap.add_argument("--num-smplify-iters", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--floor-align", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--rot6d-convention", choices=["column", "row"], default="column")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = smplx.create(
        j2s_config.SMPL_MODEL_DIR,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(device)
    model.eval()
    smplify = SMPLify3D(
        smplxmodel=model,
        batch_size=1,
        joints_category="AMASS",
        num_iters=args.num_smplify_iters,
        device=device,
    )
    mean_pose, mean_shape = load_mean_params(device)

    files = list(iter_files(Path(args.in_dir), args.limit))
    print(f"[setup] files={len(files)} out={out_dir} device={device}", flush=True)
    summary = []
    failed = 0
    for i, in_path in enumerate(files, 1):
        out_path = out_dir / f"{in_path.stem}.npz"
        if args.skip_existing and out_path.exists():
            continue
        try:
            item = retarget_one(
                in_path,
                out_path,
                model,
                smplify,
                mean_pose,
                mean_shape,
                args.source_fps,
                args.target_fps,
                args.floor_align,
                device,
                args.rot6d_convention,
            )
            summary.append(item)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {in_path.name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 4 == 0 or i == len(files):
            mean = np.mean([x["mpjpe_mm_mean"] for x in summary]) if summary else float("nan")
            print(f"[progress] {i}/{len(files)} ok={len(summary)} fail={failed} mean_mpjpe_mm={mean:.2f}", flush=True)

    stats = {
        "count": len(summary),
        "failed": failed,
        "mean_mpjpe_mm": float(np.mean([x["mpjpe_mm_mean"] for x in summary])) if summary else None,
        "median_mpjpe_mm": float(np.median([x["mpjpe_mm_mean"] for x in summary])) if summary else None,
        "items": summary[:100],
    }
    (out_dir / "_retarget_summary.json").write_text(json.dumps(stats, indent=2))
    print(f"[done] {json.dumps({k: v for k, v in stats.items() if k != 'items'}, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
