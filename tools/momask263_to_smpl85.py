#!/usr/bin/env python3
"""Convert MoMask HumanML3D-263 (20 fps) outputs to SMPL-85 (30 fps).

Pipeline per sample:
    1. Decode 263 -> joints (T_20, 22, 3) global via ``recover_from_ric``.
    2. Linear-resample 20 fps -> 30 fps.
    3. Fit SMPL params (beta, pose, trans) with SMPLify3D (batch over frames)
       initialized from the SMPL mean pose.
    4. Pack as smpl_85 = [global_orient_3 + body_pose_63 + hand_pose_6 + trans_3 + beta_10]
       where hand_pose_6 = zeros (we only fit body 22-joint AMASS subset).
       Note: MotionStreamer's smpl_85 has 72 (axis-angle) + 3 (trans) + 10 (beta).

Outputs:
    <out_dir>/<id>.npy  shape (T_30, 85)

This bypasses my (buggy) 263->272 kinematic converter and produces a
faithful SMPL representation that can be (a) directly used in Blender, or
(b) passed through MotionStreamer's representation_272 forward path for
fair cross-evaluator comparison.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ref_repo" / "Momask" / "momask-codes"))

# joints2smpl
J2S_ROOT = REPO_ROOT / "ref_repo" / "MDM" / "visualize" / "joints2smpl"
sys.path.insert(0, str(REPO_ROOT / "ref_repo" / "MDM"))  # for `visualize.joints2smpl.src.config`
sys.path.insert(0, str(J2S_ROOT / "src"))

# Override config module paths BEFORE importing smplify (smplify imports config)
import visualize.joints2smpl.src.config as j2s_config  # noqa: E402
j2s_config.SMPL_MODEL_DIR = str(REPO_ROOT / "checkpoints" / "smpl_models")
j2s_config.GMM_MODEL_DIR = str(J2S_ROOT / "smpl_models") + "/"
j2s_config.SMPL_MEAN_FILE = str(J2S_ROOT / "smpl_models" / "neutral_smpl_mean_params.h5")
j2s_config.Part_Seg_DIR = str(J2S_ROOT / "smpl_models" / "smplx_parts_segm.pkl")

import h5py  # noqa: E402
import smplx  # noqa: E402
from smplify import SMPLify3D  # noqa: E402
from utils.motion_process import recover_from_ric  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_resample_positions(pos_src: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    T_src, J, _ = pos_src.shape
    duration = (T_src - 1) / src_fps
    T_dst = int(round(duration * dst_fps)) + 1
    T_dst = max(2, T_dst)
    src_times = np.arange(T_src, dtype=np.float64) / src_fps
    dst_times = np.linspace(0.0, duration, T_dst)
    pos_dst = np.empty((T_dst, J, 3), dtype=np.float64)
    for j in range(J):
        for d in range(3):
            pos_dst[:, j, d] = np.interp(dst_times, src_times, pos_src[:, j, d])
    return pos_dst


# ---------------------------------------------------------------------------
# SMPLify wrapper that fits a whole sequence in one batch
# ---------------------------------------------------------------------------

class SeqFitter:
    """Wraps SMPLify3D for batched per-frame fitting of one motion sequence."""

    def __init__(self, num_iters: int = 60, device: str = "cuda"):
        self.num_iters = num_iters
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._smpl = None  # rebuilt per-call (depends on batch_size)

        # Load mean SMPL pose / shape (used as initialization)
        with h5py.File(j2s_config.SMPL_MEAN_FILE, "r") as f:
            self.mean_pose = torch.from_numpy(f["pose"][:]).float()  # (72,)
            self.mean_shape = torch.from_numpy(f["shape"][:]).float()  # (10,)

    def _build_smpl(self, batch_size: int):
        if self._smpl is not None and self._smpl.batch_size == batch_size:
            return self._smpl
        self._smpl = smplx.create(
            j2s_config.SMPL_MODEL_DIR,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=batch_size,
        ).to(self.device)
        return self._smpl

    def fit(self, joints3d: np.ndarray) -> dict:
        """Fit (T, 22, 3) AMASS-22 joints to per-frame SMPL params.

        Returns dict with ``pose`` (T, 72), ``betas`` (T, 10), ``trans`` (T, 3).
        """
        T = len(joints3d)
        smpl = self._build_smpl(T)
        smplify = SMPLify3D(
            smplxmodel=smpl,
            batch_size=T,
            joints_category="AMASS",
            num_iters=self.num_iters,
            device=self.device,
        )

        # Initialize per-frame from mean pose / mean shape, trans = zero.
        init_pose = self.mean_pose[None].repeat(T, 1).to(self.device)  # (T, 72)
        init_betas = self.mean_shape[None].repeat(T, 1).to(self.device)  # (T, 10)
        init_cam_t = torch.zeros((T, 3), dtype=torch.float32, device=self.device)
        keypoints_3d = torch.from_numpy(joints3d).float().to(self.device)  # (T, 22, 3)
        confidence = torch.ones(22, device=self.device, dtype=torch.float32)

        new_verts, new_joints, new_pose, new_betas, new_cam_t, new_loss = smplify(
            init_pose,
            init_betas,
            init_cam_t,
            keypoints_3d,
            conf_3d=confidence,
            seq_ind=0,
        )
        # SMPLify returns camera_translation shape (T, 1, 3) due to guess_init_3d's
        # ``unsqueeze(1)``.  Flatten to (T, 3) before packing into smpl_85.
        trans_np = new_cam_t.detach().cpu().numpy().reshape(T, 3)
        return {
            "pose": new_pose.detach().cpu().numpy(),  # (T, 72)
            "betas": new_betas.detach().cpu().numpy(),  # (T, 10)
            "trans": trans_np,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir_263", required=True,
                   help="Directory of <id>.npy 263-dim MoMask outputs (20 fps).")
    p.add_argument("--out_dir_smpl85", required=True,
                   help="Output dir for <id>.npy 85-dim SMPL params (30 fps).")
    p.add_argument("--src_fps", type=float, default=20.0)
    p.add_argument("--dst_fps", type=float, default=30.0)
    p.add_argument("--num_iters", type=int, default=60,
                   help="SMPLify3D LBFGS iterations.  60 = balanced; 100 = paper default.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="If set, process only the first N samples (for sanity check).")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    src = Path(args.pred_dir_263)
    dst = Path(args.out_dir_smpl85)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.npy"))
    if args.max_samples is not None:
        files = files[: args.max_samples]
    print(f"[+] {len(files)} input files in {src}")
    print(f"[+] resampling {args.src_fps}->{args.dst_fps} fps")
    print(f"[+] SMPLify3D iters: {args.num_iters}")

    fitter = SeqFitter(num_iters=args.num_iters, device=args.device)

    n_ok = n_err = 0
    for f in tqdm(files, ncols=80):
        out_file = dst / f.name
        if out_file.exists():
            n_ok += 1
            continue
        try:
            m263 = np.load(str(f))
            if m263.ndim != 2 or m263.shape[1] != 263 or len(m263) < 4:
                n_err += 1
                continue
            joints20 = recover_from_ric(torch.from_numpy(m263).float(), 22).numpy()
            joints30 = linear_resample_positions(joints20, args.src_fps, args.dst_fps)
            params = fitter.fit(joints30)
            smpl_85 = np.concatenate(
                [params["pose"], params["trans"], params["betas"]], axis=-1
            ).astype(np.float32)  # (T, 72+3+10)
            assert smpl_85.shape[1] == 85, smpl_85.shape
            np.save(str(out_file), smpl_85)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [!] {f.name}: {e}")

    print(f"[+] wrote {n_ok}/{len(files)} files to {dst}  ({n_err} errors)")


if __name__ == "__main__":
    main()
