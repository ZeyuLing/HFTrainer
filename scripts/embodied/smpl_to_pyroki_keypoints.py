#!/usr/bin/env python3
"""SMPL/AMASS npz -> PyRoki retargeting keypoints .npy (18-pt + foot contacts).

Reproduces ProtoMotions' `extract_keypoints_from_motion_smpl_skel` output format so
that `ref_repo/ProtoMotions/pyroki/batch_retarget_to_g1_from_keypoints.py` can consume
it directly (source-type smpl). The PyRoki solver only uses keypoint POSITIONS
(plus the root/index-0 orientation for SE3 init), so we compute global joint
positions from SMPL-X FK and global joint rotations from the kinematic tree.

Output per motion (saved with np.save as a 0-d object array -> load with .item()):
    positions            (T, 18, 3)      float32   world frame, z-up
    orientations         (T, 18, 3, 3)   float32   rotation matrices
    left_foot_contacts   (T, 2)          int       (ankle, toebase) binary
    right_foot_contacts  (T, 2)          int       (ankle, toebase) binary

18-pt order: pelvis, L/R hip, L/R knee, L/R ankle, L/R foot(toe),
             L/R shoulder, L/R elbow, L/R wrist, L/R hand_aux, pelvis_aux

Coordinate frame: SMPL-X FK from raw AMASS global_orient is z-up already in our
hymotion_data (verified: head z > foot z). Use --src-up y to apply Rx(-90) if a
given source is y-up.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
GMR_ROOT = PROJECT_ROOT / "ref_repo" / "GMR"
SMPLX_FOLDER = GMR_ROOT / "assets" / "body_models"

# SMPL(-X) body joint indices for the 15 conceptual keypoints.
SMPL_KP_IDX = {
    "pelvis": 0, "left_hip": 1, "right_hip": 2, "left_knee": 4, "right_knee": 5,
    "left_ankle": 7, "right_ankle": 8, "left_foot": 10, "right_foot": 11,
    "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19,
    "left_wrist": 20, "right_wrist": 21,
}
KP_ORDER = ["pelvis", "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_foot", "right_foot",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"]

_BM_CACHE = {}


def get_body_model(gender: str):
    import smplx
    g = str(gender)
    if g not in _BM_CACHE:
        _BM_CACHE[g] = smplx.create(str(SMPLX_FOLDER), "smplx", gender=g, use_pca=False)
    return _BM_CACHE[g]


def aa_to_mat(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices."""
    from scipy.spatial.transform import Rotation as R
    shape = aa.shape[:-1]
    return R.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3)


def smplx_fk(poses, trans, betas, gender):
    """Return (joints[T,22,3] world pos, global_rot[T,22,3,3], src_fps)."""
    import torch
    bm = get_body_model(gender)
    T = poses.shape[0]
    nb = getattr(bm, "num_betas", 10)
    b = betas.reshape(-1).astype(np.float32)
    b = (b[:nb] if b.shape[0] >= nb else np.concatenate([b, np.zeros(nb - b.shape[0], np.float32)]))
    betas_t = torch.tensor(b).float().view(1, -1).expand(T, -1)
    with torch.no_grad():
        out = bm(betas=betas_t,
                 global_orient=torch.tensor(poses[:, :3]).float(),
                 body_pose=torch.tensor(poses[:, 3:66]).float(),
                 transl=torch.tensor(trans).float(),
                 left_hand_pose=torch.zeros(T, 45).float(),
                 right_hand_pose=torch.zeros(T, 45).float(),
                 jaw_pose=torch.zeros(T, 3).float(), leye_pose=torch.zeros(T, 3).float(),
                 reye_pose=torch.zeros(T, 3).float(), expression=torch.zeros(T, 10).float())
    joints = out.joints.detach().numpy()[:, :22, :]  # (T,22,3)

    # Global joint rotations from kinematic tree.
    parents = bm.parents.detach().cpu().numpy()[:22]
    local_aa = np.concatenate([poses[:, :3][:, None, :], poses[:, 3:66].reshape(T, 21, 3)], axis=1)
    local_mat = aa_to_mat(local_aa)  # (T,22,3,3)
    glob = np.zeros_like(local_mat)
    glob[:, 0] = local_mat[:, 0]
    for j in range(1, 22):
        glob[:, j] = glob[:, parents[j]] @ local_mat[:, j]
    return joints, glob


def resample(arr, src_fps, tgt_fps):
    stride = max(1, int(round(src_fps / tgt_fps)))
    return arr[::stride], max(1, round(src_fps / stride))


def detect_contacts(ankle_z, toe_z, vel_ankle, vel_toe, clear=0.07, spd=0.30):
    """Binary contact per foot part: low height AND low horizontal speed."""
    zmin = min(ankle_z.min(), toe_z.min())
    a = ((ankle_z - zmin) < clear) & (vel_ankle < spd)
    t = ((toe_z - zmin) < clear) & (vel_toe < spd)
    return a.astype(int), t.astype(int)


def build_keypoints(pos15, rot15):
    """Apply ProtoMotions SMPL surgeries + aux. pos15/rot15: (T,15,3)/(T,15,3,3)."""
    pos = pos15.copy()
    rot = rot15.copy()
    idx = {n: i for i, n in enumerate(KP_ORDER)}

    def apply(i, off):
        return pos[:, i, :] + np.einsum("tij,j->ti", rot[:, i, :, :], np.asarray(off))

    # root
    pos[:, idx["pelvis"], :] = apply(idx["pelvis"], [-0.04, 0.0, 0.0])
    # elbows
    pos[:, idx["left_elbow"], :] = apply(idx["left_elbow"], [0.0, 0.0, 0.045])
    pos[:, idx["right_elbow"], :] = apply(idx["right_elbow"], [0.0, 0.0, 0.045])
    # flat feet: toe = ankle_orig + R_ankle @ [0.18,0,0]; then ankle += R_ankle @ [0.03,0,0]
    for side in ("left", "right"):
        ai, fi = idx[f"{side}_ankle"], idx[f"{side}_foot"]
        toe = pos[:, ai, :] + np.einsum("tij,j->ti", rot[:, ai, :, :], np.array([0.18, 0.0, 0.0]))
        pos[:, fi, :] = toe
    for side in ("left", "right"):
        ai = idx[f"{side}_ankle"]
        pos[:, ai, :] = pos[:, ai, :] + np.einsum("tij,j->ti", rot[:, ai, :, :], np.array([0.03, 0.0, 0.0]))

    # aux: hand_aux (wrist + R_wrist@[0.2,0,0]), pelvis_aux (pelvis_cur + R_pelvis@[0.16,0,0])
    extra_pos, extra_rot = [], []
    for side in ("left", "right"):
        wi = idx[f"{side}_wrist"]
        ha = pos[:, wi, :] + np.einsum("tij,j->ti", rot[:, wi, :, :], np.array([0.2, 0.0, 0.0]))
        extra_pos.append(ha[:, None, :]); extra_rot.append(rot[:, wi, :, :][:, None])
    pi = idx["pelvis"]
    pa = pos[:, pi, :] + np.einsum("tij,j->ti", rot[:, pi, :, :], np.array([0.16, 0.0, 0.0]))
    extra_pos.append(pa[:, None, :]); extra_rot.append(rot[:, pi, :, :][:, None])

    pos = np.concatenate([pos] + extra_pos, axis=1)   # (T,18,3)
    rot = np.concatenate([rot] + extra_rot, axis=1)   # (T,18,3,3)
    return pos.astype(np.float32), rot.astype(np.float32)


def process_one(npz_path: Path, tgt_fps: int, src_up: str):
    d = np.load(npz_path, allow_pickle=True)
    poses = d["poses"].astype(np.float32)
    trans = d["trans"].astype(np.float32)
    betas = d["betas"].astype(np.float32) if "betas" in d.files else np.zeros((1, 16), np.float32)
    gender = str(d["gender"]) if "gender" in d.files else "neutral"
    src_fps = int(np.asarray(d.get("mocap_framerate", 30)).reshape(-1)[0])
    if poses.shape[0] < 2:
        return None

    joints, glob = smplx_fk(poses, trans, betas, gender)
    joints, _ = resample(joints, src_fps, tgt_fps)
    glob, out_fps = resample(glob, src_fps, tgt_fps)

    if src_up == "y":  # Rx(+90): y-up -> z-up, (x,y,z)->(x,-z,y); rotate pos & frames
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        joints = joints @ Rx.T
        glob = np.einsum("ij,tkjl->tkil", Rx, glob)

    sel = [SMPL_KP_IDX[n] for n in KP_ORDER]
    pos15 = joints[:, sel, :]
    rot15 = glob[:, sel, :, :]
    positions, orientations = build_keypoints(pos15, rot15)

    # Global yaw alignment: the y-up->z-up Rx above lands the heading 90 deg off
    # from the GMR / SMPL-mesh display frame (forward=+x, left=+y). PyRoki faithfully
    # reproduces the keypoint frame, so without this its whole body comes out yawed
    # ~90 deg vs the reference (looks like "lower body facing wrong"). Verified by
    # Procrustes on root trajectory: PyRoki->GMR = +90.7 deg, residual 0.03.
    # Rz(+90): (x,y,z)->(-y,x,z); apply to positions and orientation frames (world basis).
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    positions = (positions @ Rz.T).astype(np.float32)
    orientations = np.einsum("ij,tkjl->tkil", Rz, orientations).astype(np.float32)

    # foot contacts from ankle(7)/toe(10,11) world joints (pre-surgery)
    T = positions.shape[0]
    la, ra = joints[:, 7, :], joints[:, 8, :]
    lt, rt = joints[:, 10, :], joints[:, 11, :]
    def hspeed(p):
        v = np.zeros(p.shape[0]); v[1:] = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1) * out_fps
        return v
    lac, ltc = detect_contacts(la[:, 2], lt[:, 2], hspeed(la), hspeed(lt))
    rac, rtc = detect_contacts(ra[:, 2], rt[:, 2], hspeed(ra), hspeed(rt))
    left_fc = np.stack([lac, ltc], axis=-1)
    right_fc = np.stack([rac, rtc], axis=-1)

    return {
        "positions": positions,
        "orientations": orientations,
        "left_foot_contacts": left_fc,
        "right_foot_contacts": right_fc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "hymotion_data"))
    ap.add_argument("--out-dir", required=True, help="dir to write <id>_keypoints.npy")
    ap.add_argument("--names", nargs="*", required=True, help="rel npz paths under data-dir")
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--src-up", choices=["z", "y"], default="z")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, rel in enumerate(args.names):
        npz = data_dir / rel
        stem = f"{i:02d}_" + Path(rel).stem
        try:
            kp = process_one(npz, args.target_fps, args.src_up)
        except Exception as e:
            import traceback
            print(f"[fail] {stem}: {e}\n{traceback.format_exc()[-500:]}", flush=True)
            continue
        if kp is None:
            print(f"[skip] {stem}: too short", flush=True)
            continue
        out = out_dir / f"{stem}_keypoints.npy"
        np.save(out, kp, allow_pickle=True)
        print(f"[ok] {stem}: pos{kp['positions'].shape} contacts L{kp['left_foot_contacts'].sum()} R{kp['right_foot_contacts'].sum()}", flush=True)


if __name__ == "__main__":
    main()
