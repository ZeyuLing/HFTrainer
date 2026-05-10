#!/usr/bin/env python3
"""Verify joints2smpl fit quality.

Pipeline:
    1. Load 263 prediction -> recover joints (T, 22, 3) at 20 fps
    2. lerp -> 30 fps
    3. Load fitted smpl_85 (pose + trans + beta)
    4. Forward kinematics with smplx -> reconstructed joints (T, 22, 3)
    5. Compute MPJPE between input joints and reconstructed joints

A good fit should give MPJPE < 5 cm.  > 10 cm indicates the optimization
hasn't converged enough.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"))

import smplx  # noqa: E402
from utils.motion_process import recover_from_ric  # noqa: E402

from tools.momask263_to_smpl85 import linear_resample_positions  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir_263", required=True)
    p.add_argument("--smpl85_dir", required=True)
    p.add_argument("--ids", default="000000,000019,000021")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a dynamic-batch SMPL model
    smpl_path = REPO_ROOT / "checkpoints" / "smpl_models"

    print(f"{'id':>10s} | {'T':>4s} | {'mpjpe_mm':>10s} | {'trans_mean':>12s} | {'pose_norm':>10s}")
    print("-" * 70)
    for sid in args.ids.split(","):
        sid = sid.strip()
        m263 = np.load(str(Path(args.pred_dir_263) / f"{sid}.npy"))
        smpl85 = np.load(str(Path(args.smpl85_dir) / f"{sid}.npy"))
        joints20 = recover_from_ric(torch.from_numpy(m263).float(), 22).numpy()
        joints30 = linear_resample_positions(joints20, 20.0, 30.0)
        T = len(smpl85)
        T = min(T, len(joints30))
        joints30 = joints30[:T]
        smpl85 = smpl85[:T]

        smpl = smplx.create(
            str(smpl_path), model_type="smpl", gender="neutral", ext="pkl",
            batch_size=T,
        ).to(device)

        pose = torch.from_numpy(smpl85[:, :72]).float().to(device)
        trans = torch.from_numpy(smpl85[:, 72:75]).float().to(device)
        betas = torch.from_numpy(smpl85[:, 75:]).float().to(device)
        with torch.no_grad():
            out = smpl(
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                betas=betas,
                transl=trans,
            )
        recon_joints = out.joints[:, :22].cpu().numpy()  # SMPL joints[0:22] = AMASS-22

        mpjpe = float(np.linalg.norm(recon_joints - joints30, axis=-1).mean())
        trans_mean = float(np.linalg.norm(trans.cpu().numpy(), axis=-1).mean())
        pose_norm = float(np.linalg.norm(pose.cpu().numpy(), axis=-1).mean())
        print(f"{sid:>10s} | {T:4d} | {mpjpe*1000:10.2f} | {trans_mean:12.4f} | {pose_norm:10.4f}")


if __name__ == "__main__":
    main()
