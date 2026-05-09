#!/usr/bin/env python3
"""Validate smpl85_to_repr272 against MotionStreamer's official pipeline.

For each input ``smpl_85.npy`` we run two pipelines and compare block-by-block:

    A) MINE (tools/smpl85_to_repr272.py):
       face_z_transform -> smplx.create(model_type='smpl').forward -> repr272 packing

    B) OFFICIAL (ref_repo/MotionStreamer/272-dim-Motion-Representation/*):
       face_z_transform.py -> infer_get_joints.py SMPL-X FK ->
       representation_272.py packing

Block layout (final_x, T,272):
    [0:2]    velocities_root_xy_no_heading
    [2:8]    heading 6D (first 2 rows of heading-delta R_y)
    [8:74]   positions_no_heading                     (22 joints * 3)
    [74:140] velocities_no_heading                    (22 joints * 3)
    [140:272] joint local 6D rotations (first 2 rows) (22 joints * 6)

We expect mine == official for blocks [0:2], [2:8] and the rotation block [140:272]
(those don't depend on FK -- they come from smpl_85 axis-angle).
The position/velocity blocks ([8:140]) will differ slightly because A uses SMPL FK
and B uses SMPL-X FK; we report and document this gap.

Usage:
    python tools/validate_smpl85_to_272.py \
        --smpl85_dir work_dirs/momask_eval/momask_pred_smpl85_test \
        --max_samples 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same imports as smpl85_to_repr272.py
MS_ROOT = REPO_ROOT / "ref_repo" / "MotionStreamer" / "272-dim-Motion-Representation"
sys.path.insert(0, str(MS_ROOT))
sys.path.insert(0, str(MS_ROOT / "utils"))

import smplx  # noqa: E402

# Reuse functions from smpl85_to_repr272.py to avoid drift
sys.path.insert(0, str(REPO_ROOT / "tools"))
from smpl85_to_repr272 import (  # type: ignore  # noqa: E402
    face_z_transform_smpl85,
    smpl_fk,
    representation_272_forward,
)
from face_z_align_util import (  # type: ignore  # noqa: E402
    expmap_to_quaternion,
    qmul_np,
    qrot_np,
    quaternion_to_axis_angle,
    quaternion_to_matrix_np,
    matrix_to_rotation_6d,
)


# ---------------------------------------------------------------------------
# Faithful re-implementations of the OFFICIAL MotionStreamer scripts in-memory
# ---------------------------------------------------------------------------

def official_face_z_transform_smpl85(smpl_85: np.ndarray) -> np.ndarray:
    """Mirror exactly ref_repo/.../face_z_transform.py main loop body."""
    smpl_data = smpl_85
    seq_len = smpl_data.shape[0]
    pose_body = smpl_data[:, :72].reshape(seq_len, -1, 3).copy()
    trans = smpl_data[:, 72:75].copy()
    beta = smpl_data[:, 75:]

    from face_z_transform import calc_heading_quat_inv  # type: ignore

    root_first_frame_root_orient = pose_body[0, 0]
    q_first_wxyz = expmap_to_quaternion(root_first_frame_root_orient)  # (4,) wxyz
    q_first_xyzw = q_first_wxyz[[1, 2, 3, 0]]
    q_first_xyzw_t = torch.from_numpy(q_first_xyzw).float().unsqueeze(0)
    heading_inv, axis = calc_heading_quat_inv(q_first_xyzw_t)
    heading_inv_aa = (heading_inv * axis).numpy()
    q_diff = expmap_to_quaternion(heading_inv_aa)  # (1, 4) wxyz

    result_root_orient_quat = qmul_np(
        q_diff.reshape(1, -1).repeat(seq_len, axis=0),
        expmap_to_quaternion(pose_body[:, 0]),
    )
    result_root_orient_aa = quaternion_to_axis_angle(
        torch.from_numpy(result_root_orient_quat)
    ).numpy()
    trans = qrot_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), trans)
    out = np.concatenate(
        [
            result_root_orient_aa,
            pose_body[:, 1:].reshape(seq_len, -1),
            trans,
            beta,
        ],
        axis=-1,
    ).astype(np.float32)
    return out


def smpl_x_fk(smpl_85_face_z: np.ndarray, smplx_model_path: Path,
              device: torch.device, smplx_model=None) -> np.ndarray:
    """SMPL-X FK with betas, mirroring infer_get_joints.py.

    The official script calls:
        smplx_model(pose_body=data[:,3:66], root_orient=data[:,:3],
                    trans=data[:, 72:72+3], betas=data[:, 75:]).Jtr
    using human_body_prior's BodyModel. We approximate with smplx.create
    with model_type='smplx', which computes the same body skeleton.
    """
    T = len(smpl_85_face_z)
    if smplx_model is None or smplx_model.batch_size != T:
        smplx_model = smplx.create(
            str(smplx_model_path),
            model_type="smplx",
            gender="neutral",
            ext="npz",
            num_betas=10,
            batch_size=T,
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

    pose = torch.from_numpy(smpl_85_face_z[:, :72]).float().to(device)
    trans = torch.from_numpy(smpl_85_face_z[:, 72:75]).float().to(device)
    betas = torch.from_numpy(smpl_85_face_z[:, 75:]).float().to(device)
    with torch.no_grad():
        out = smplx_model(
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:66],
            betas=betas,
            transl=trans,
        )
    joints = out.joints[:, :22].cpu().numpy()
    return joints, smplx_model


def official_representation_272(joints: np.ndarray, smpl_85_face_z: np.ndarray) -> np.ndarray:
    """Mirror exactly representation_272.py main loop body, no foot detection writes."""
    import copy

    position_data = joints.astype(np.float64)[:, :22, :3].copy()
    nfrm, njoint, _ = position_data.shape

    rotation_smpl_axis_angle = smpl_85_face_z[:, :72]
    rotations_wxyz = expmap_to_quaternion(
        rotation_smpl_axis_angle[:, :66].reshape(nfrm, njoint, 3)
    )
    rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)

    ori = copy.deepcopy(position_data[0, 0])
    y_min = np.min(position_data[:, :, 1])
    ori[1] = y_min
    position_data = position_data - ori
    velocities_root = position_data[1:, 0, :] - position_data[:-1, 0, :]

    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    global_heading = -np.arctan2(
        rotations_matrix[:, 0, 0, 2], rotations_matrix[:, 0, 2, 2]
    )

    def rot_yaw(yaw):
        cs = np.cos(yaw)
        sn = np.sin(yaw)
        return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])

    global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1),
        position_data[..., None],
    ).squeeze(-1)
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]
    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1], velocities_root[:, :, None]
    ).squeeze()[..., [0, 2]]

    rotations_matrix[:, 0, ...] = np.matmul(
        global_heading_rot, rotations_matrix[:, 0, ...]
    )

    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame))
    final_x[0, 2] = 1
    final_x[0, 6] = 1
    final_x[1:, 2:8] = (
        matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8 : 8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)
    final_x[1:, 8 + 3 * njoint : 8 + 6 * njoint] = velocities_no_heading.reshape(
        nfrm - 1, -1
    )
    final_x[:, 8 + 6 * njoint : 8 + 12 * njoint] = (
        rotations_matrix[..., :, :2, :].reshape(nfrm, -1)
    )
    return final_x.astype(np.float32)


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

BLOCKS = [
    ("vel_root_xy", 0, 2),
    ("heading_6d", 2, 8),
    ("pos_no_head", 8, 74),
    ("vel_no_head", 74, 140),
    ("rot_local_6d", 140, 272),
]


def block_stats(name, lo, hi, a, b):
    da = (a - b)[:, lo:hi]
    return {
        "block": name,
        "lo": lo,
        "hi": hi,
        "max_abs": float(np.abs(da).max()),
        "mean_abs": float(np.abs(da).mean()),
        "rms": float(np.sqrt((da ** 2).mean())),
        "ref_std": float(b[:, lo:hi].std()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smpl85_dir", required=True)
    p.add_argument("--max_samples", type=int, default=3)
    p.add_argument("--smpl_path", default=str(REPO_ROOT / "checkpoints" / "smpl_models"))
    p.add_argument("--smplx_path", default=str(REPO_ROOT / "checkpoints" / "smpl_models"))
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    src = Path(args.smpl85_dir)
    files = sorted(src.glob("*.npy"))[: args.max_samples]
    print(f"[+] validating {len(files)} files from {src}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    smpl_model = None
    smplx_model = None

    for f in files:
        smpl_85 = np.load(str(f))
        print(f"\n=== {f.name} ({smpl_85.shape}) ===")

        # ---- compare face_z_transform ----
        fz_mine = face_z_transform_smpl85(smpl_85.copy())
        fz_off = official_face_z_transform_smpl85(smpl_85.copy())
        fz_diff = np.abs(fz_mine - fz_off)
        print(f"face_z (smpl_85)  max|d|={fz_diff.max():.3e}  "
              f"mean|d|={fz_diff.mean():.3e}")

        # ---- run mine (now SMPL-X by default): smpl_fk + repr272 ----
        joints_mine, smplx_model = smpl_fk(
            fz_mine, Path(args.smplx_path), device, smplx_model,
            model_type="smplx",
        )
        m272_mine = representation_272_forward(joints_mine, fz_mine)

        # ---- run official: smplx fk + repr272 ----
        joints_off, smplx_model = smpl_x_fk(
            fz_off, Path(args.smplx_path), device, smplx_model
        )
        m272_off = official_representation_272(joints_off, fz_off)

        # ---- compare per block ----
        print(f"  shapes  mine={m272_mine.shape}  off={m272_off.shape}")
        for name, lo, hi in BLOCKS:
            s = block_stats(name, lo, hi, m272_mine, m272_off)
            print(
                f"  {name:14s} [{lo:3d}:{hi:3d}]  "
                f"max|d|={s['max_abs']:8.4f}  "
                f"mean|d|={s['mean_abs']:8.4f}  "
                f"rms={s['rms']:8.4f}  "
                f"ref_std={s['ref_std']:8.4f}"
            )

        # ---- compare joint positions directly ----
        joint_diff = np.abs(joints_mine - joints_off)
        print(f"  joint xyz (mine vs official)  max|d|={joint_diff.max():.4f}  "
              f"mean|d|={joint_diff.mean():.4f}  rms={np.sqrt((joint_diff**2).mean()):.4f}")


if __name__ == "__main__":
    main()
