#!/usr/bin/env python3
"""Convert SMPL-85 (T, 85) to MotionStreamer's 272-dim representation.

Pipeline per file:
    1. Load smpl_85 = [pose_72, trans_3, beta_10] @ 30 fps.
    2. ``face_z_transform``: rotate first frame so root forward (-z axis) faces Z+,
       and rotate trans accordingly.  Mirrors MotionStreamer's ``face_z_transform.py``.
    3. SMPL forward kinematics -> joint positions (T, 22, 3) global.
       (Uses ``smplx`` with model_type='smpl' since our smpl_85 was fit with SMPL.)
    4. ``representation_272`` forward path: pack into (T, 272).  Mirrors
       MotionStreamer's ``representation_272.py``.

Outputs:
    <out_dir>/<id>.npy  shape (T, 272), in native units (NOT pre-standardized).
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

# face_z_align_util has the rotation utilities MotionStreamer uses
MS_ROOT = REPO_ROOT / "ref_repo" / "MotionStreamer" / "272-dim-Motion-Representation"
sys.path.insert(0, str(MS_ROOT))
sys.path.insert(0, str(MS_ROOT / "utils"))

from face_z_align_util import (  # type: ignore  # noqa: E402
    expmap_to_quaternion,
    qmul_np,
    qrot_np,
    quaternion_to_axis_angle,
    quaternion_to_matrix_np,
)
from face_z_transform import calc_heading_quat_inv  # type: ignore  # noqa: E402

import smplx  # noqa: E402


# ---------------------------------------------------------------------------
# face_z_transform: align first-frame root orientation to face +Z
# ---------------------------------------------------------------------------

def face_z_transform_smpl85(smpl_85: np.ndarray) -> np.ndarray:
    """Rotate the entire sequence so that the first frame's root faces +Z.

    Mirrors MotionStreamer's ``face_z_transform.py`` exactly.

    Returns smpl_85 in the same shape (T, 85) but with rotated root_orient and trans.
    Does **not** mutate the input array.
    """
    seq_len = smpl_85.shape[0]
    pose_body = smpl_85[:, :72].reshape(seq_len, -1, 3).copy()  # (T, 24, 3) axis-angle
    trans = smpl_85[:, 72:75].copy()  # (T, 3)
    beta = smpl_85[:, 75:].copy()  # (T, 10)

    # First-frame root orientation -> quaternion (xyzw order in MotionStreamer's convention)
    root_first = pose_body[0, 0]  # (3,) axis-angle
    root_first_quat_wxyz = expmap_to_quaternion(root_first)  # (4,) wxyz
    root_first_quat_xyzw = root_first_quat_wxyz[[1, 2, 3, 0]]  # convert wxyz -> xyzw
    root_first_quat_xyzw_t = torch.from_numpy(root_first_quat_xyzw).float().unsqueeze(0)
    heading_inv, axis = calc_heading_quat_inv(root_first_quat_xyzw_t)
    heading_inv_aa = (heading_inv * axis).numpy()  # (1, 3)

    q_diff = expmap_to_quaternion(heading_inv_aa)  # (1, 4) wxyz

    # Apply heading-inverse to root orientation of every frame
    root_quat_all = expmap_to_quaternion(pose_body[:, 0])  # (T, 4) wxyz
    new_root_quat = qmul_np(np.broadcast_to(q_diff, (seq_len, 4)), root_quat_all)
    new_root_aa = (
        quaternion_to_axis_angle(torch.from_numpy(new_root_quat)).numpy()
    )  # (T, 3) axis-angle
    pose_body[:, 0] = new_root_aa

    # Apply heading-inverse to translation
    trans = qrot_np(np.broadcast_to(q_diff, (seq_len, 4)), trans)

    out = np.concatenate(
        [pose_body.reshape(seq_len, -1), trans, beta], axis=-1
    ).astype(np.float32)
    assert out.shape == smpl_85.shape, (out.shape, smpl_85.shape)
    return out


# ---------------------------------------------------------------------------
# SMPL forward kinematics
# ---------------------------------------------------------------------------

def smpl_fk(smpl_85: np.ndarray, smpl_path: Path, device: torch.device,
            smpl_model=None, model_type: str = "smplx",
            fixed_batch_size: int = 0) -> np.ndarray:
    """SMPL/SMPL-X FK -> (T, 22, 3) joint positions in global frame.

    MotionStreamer's official pipeline (``infer_get_joints.py``) uses a
    SMPL-X ``BodyModel`` with ``pose_body=data[:,3:66]`` and ``betas=data[:,75:]``.
    To stay distribution-faithful with the evaluator we replicate that:
    ``model_type='smplx'`` and pass only the 21 body joints (drop hands).

    When ``fixed_batch_size > 0`` we build the SMPL-X model once with that
    batch size and pad shorter sequences with zeros (saves ~5s per motion).
    """
    T = len(smpl_85)
    bs = fixed_batch_size if fixed_batch_size > 0 else T
    assert bs >= T, f"fixed_batch_size {bs} < seq len {T}"

    if smpl_model is None or smpl_model.batch_size != bs:
        smpl_model = smplx.create(
            str(smpl_path),
            model_type=model_type,
            gender="neutral",
            ext="npz" if model_type == "smplx" else "pkl",
            num_betas=10,
            batch_size=bs,
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)

    # pad to bs
    padded = np.zeros((bs, 85), dtype=np.float32)
    padded[:T] = smpl_85
    pose = torch.from_numpy(padded[:, :72]).float().to(device)
    trans = torch.from_numpy(padded[:, 72:75]).float().to(device)
    betas = torch.from_numpy(padded[:, 75:]).float().to(device)
    with torch.no_grad():
        if model_type == "smplx":
            out = smpl_model(
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:66],
                betas=betas,
                transl=trans,
            )
        else:
            out = smpl_model(
                global_orient=pose[:, :3],
                body_pose=pose[:, 3:],
                betas=betas,
                transl=trans,
            )
    joints = out.joints[:T, :22].cpu().numpy()  # crop padding -> (T, 22, 3)
    return joints, smpl_model


# ---------------------------------------------------------------------------
# representation_272 forward path
# ---------------------------------------------------------------------------

def rot_yaw(yaw: float) -> np.ndarray:
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])


def representation_272_forward(joints: np.ndarray, smpl_85_face_z: np.ndarray) -> np.ndarray:
    """Build the 272-dim representation following MotionStreamer's official code.

    Args:
        joints: (T, 22, 3) global joint positions from SMPL FK on smpl_85_face_z.
        smpl_85_face_z: (T, 85) smpl_85 AFTER face_z_transform.

    Returns:
        (T, 272) representation in native units (no normalization).
    """
    nfrm, njoint, _ = joints.shape
    assert njoint == 22

    position_data = joints.astype(np.float64).copy()
    rotation_smpl_axis_angle = smpl_85_face_z[:, :72]

    # 22 joints * 3 axis-angle components -> (nfrm, 22, 4) wxyz
    rotations_wxyz = expmap_to_quaternion(
        rotation_smpl_axis_angle[:, :66].reshape(nfrm, njoint, 3)
    )
    rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)  # (nfrm, 22, 3, 3)

    # Put on floor + root xz to origin at first frame
    ori = position_data[0, 0].copy()
    y_min = position_data[:, :, 1].min()
    ori[1] = y_min
    position_data = position_data - ori

    velocities_root = position_data[1:, 0, :] - position_data[:-1, 0, :]

    # All-frame xz-anchor
    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    global_heading = -np.arctan2(
        rotations_matrix[:, 0, 0, 2], rotations_matrix[:, 0, 2, 2]
    )
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
    ).squeeze(-1)[..., [0, 2]]

    rotations_matrix[:, 0, ...] = np.matmul(
        global_heading_rot, rotations_matrix[:, 0, ...]
    )

    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame), dtype=np.float64)
    final_x[0, 2] = 1.0
    final_x[0, 6] = 1.0

    # heading delta as 6D (first two rows of R_y(diff))
    final_x[1:, 2:8] = global_heading_diff_rot[:, :2, :].reshape(-1, 6)
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8 : 8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)
    final_x[1:, 8 + 3 * njoint : 8 + 6 * njoint] = velocities_no_heading.reshape(
        nfrm - 1, -1
    )
    # row-major 6D for joint local rotations
    final_x[:, 8 + 6 * njoint : 8 + 12 * njoint] = rotations_matrix[..., :2, :].reshape(
        nfrm, -1
    )
    return final_x.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smpl85_dir", required=True)
    p.add_argument("--out_dir_272", required=True)
    p.add_argument(
        "--smpl_path",
        default=str(REPO_ROOT / "checkpoints" / "smpl_models"),
    )
    p.add_argument(
        "--model_type",
        default="smplx",
        choices=["smpl", "smplx"],
        help=(
            "FK body model. MotionStreamer's official pipeline uses 'smplx' "
            "(infer_get_joints.py), so 'smplx' is the default to stay "
            "distribution-faithful with the evaluator."
        ),
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    src = Path(args.smpl85_dir)
    dst = Path(args.out_dir_272)
    dst.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    files = [f for f in sorted(src.glob("*.npy")) if ".tmp." not in f.name]
    if args.max_samples is not None:
        files = files[: args.max_samples]
    print(f"[+] {len(files)} input smpl_85 files in {src}")

    # Quick first pass: read shapes to find max sequence length, so we can
    # build SMPL-X model exactly once with that batch size.
    max_T = 0
    for f in files:
        try:
            T = int(np.load(str(f), mmap_mode="r").shape[0])
            max_T = max(max_T, T)
        except Exception:
            pass
    print(f"[+] max seq length = {max_T}")

    smpl_model = None
    n_ok = n_err = 0
    for f in tqdm(files, ncols=80):
        out_file = dst / f.name
        if out_file.exists():
            n_ok += 1
            continue
        try:
            smpl_85 = np.load(str(f))
            if smpl_85.shape[1] != 85 or len(smpl_85) < 4:
                n_err += 1
                continue
            smpl_85_fz = face_z_transform_smpl85(smpl_85)
            joints, smpl_model = smpl_fk(
                smpl_85_fz, Path(args.smpl_path), device, smpl_model,
                model_type=args.model_type,
                fixed_batch_size=max_T,
            )
            m272 = representation_272_forward(joints, smpl_85_fz)
            np.save(str(out_file), m272)
            n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"  [!] {f.name}: {e}", flush=True)

    print(f"[+] wrote {n_ok}/{len(files)} files to {dst} ({n_err} errors)")


if __name__ == "__main__":
    main()
