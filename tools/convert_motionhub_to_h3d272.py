#!/usr/bin/env python3
"""Convert MotionHub SMPL-X 55-joint axis-angle motions into HumanML3D 272-dim
features for use with MotionStreamer's TMR-272 evaluator.

Pipeline (mirroring MotionStreamer/272-dim-Motion-Representation/):

    MotionHub .npz (poses[T,165], trans[T,3], mocap_framerate)
        |
        |  1) resample to 20 fps (HumanML3D / MotionStreamer convention)
        |  2) downselect to first 22 joints → smpl_85 = [global_orient(3),
        |                                                 body_pose(63),
        |                                                 zeros(6),
        |                                                 trans(3),
        |                                                 betas(10)=0]
        |
        |  3) face_z_transform  : rotate first frame so root faces +Z
        |  4) SMPL-X FK         : (T, 22, 3) joint positions (via SmplxLite)
        |  5) representation_272: 272-dim feature per frame
        v
    <out_root>/motion_data/<id>.npy  (T, 272)
    <out_root>/texts/<id>.txt        HumanML3D-format captions
    <out_root>/split/test.txt        list of <id>
    <out_root>/mean_std/Mean.npy     (272,)
    <out_root>/mean_std/Std.npy      (272,)

Mean/Std are taken from the original HumanML3D-272 release so the
MotionStreamer evaluator sees the same input distribution it was trained on.

Usage::

    python3 tools/convert_motionhub_to_h3d272.py \
        --anno_file data/annotation/test_motionhub_t2m.json \
        --data_dir  data/motionhub \
        --out_root  work_dirs/motionhub_272 \
        --ms_data_root ref_repo/MotionStreamer/MotionStreamer/humanml3d_272 \
        --smpl_model_path checkpoints/smpl_models/smplx \
        --max_frames 300

The output directory layout matches what
``ref_repo/MotionStreamer/eval_with_motionstreamer_evaluator.py`` expects via
``--data_root <out_root>``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndimage
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
MS_REPR_ROOT = REPO_ROOT / "ref_repo" / "MotionStreamer" / "272-dim-Motion-Representation"
sys.path.insert(0, str(MS_REPR_ROOT))
sys.path.insert(0, str(REPO_ROOT))

# Imports from MotionStreamer's 272-dim utility tree.
from utils.face_z_align_util import (  # noqa: E402  (after sys.path mutation)
    expmap_to_quaternion,
    qmul_np,
    qrot_np,
    quaternion_to_matrix_np,
    quaternion_to_axis_angle,
    matrix_to_rotation_6d,
)

from hftrainer.models.motion.components.body_models.smplx_lite import SmplxLite  # noqa: E402


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_anno(path: Path) -> List[Dict]:
    """Load MotionHub-style annotation file (list or dict-with-data_list)."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "data_list" in data:
        dl = data["data_list"]
        if isinstance(dl, dict):
            return [{**v, "_id": k} for k, v in dl.items()]
        return list(dl)
    if isinstance(data, list):
        return data
    raise TypeError(f"unexpected annotation type: {type(data)}")


def _load_caption_pool(cap_path: Path) -> List[str]:
    """Load all captions from a caption file (.txt one-per-line, .json hierarchical, or .npy)."""
    if not cap_path.exists():
        return []
    if cap_path.suffix == ".txt":
        return [l.strip() for l in cap_path.read_text().splitlines() if l.strip()]
    if cap_path.suffix == ".json":
        try:
            o = json.loads(cap_path.read_text())
        except Exception:
            return []
        pool = []
        if isinstance(o, list):
            pool = [str(x) for x in o if isinstance(x, str)]
        elif isinstance(o, dict):
            for k in ("macro", "meso", "micro", "captions"):
                v = o.get(k)
                if isinstance(v, list):
                    pool.extend([str(x) for x in v if isinstance(x, str)])
                elif isinstance(v, str):
                    pool.append(v)
            if not pool:
                for v in o.values():
                    if isinstance(v, list):
                        pool.extend([str(x) for x in v if isinstance(x, str)])
                    elif isinstance(v, str):
                        pool.append(v)
        return [c for c in pool if c]
    if cap_path.suffix == ".npy":
        try:
            arr = np.load(cap_path, allow_pickle=True)
            return [str(c) for c in arr.tolist() if isinstance(c, str)]
        except Exception:
            return []
    return []


# ---------------------------------------------------------------------------
# Stage 1-2: MotionHub.npz -> smpl_85 (resampled to target fps)
# ---------------------------------------------------------------------------

def _linear_resample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linear resample along time axis (T, *)."""
    if abs(src_fps - dst_fps) < 1e-6:
        return arr.astype(np.float32, copy=False)
    T = arr.shape[0]
    new_T = max(2, int(round(T * dst_fps / src_fps)))
    src_t = np.linspace(0, 1, T, dtype=np.float64)
    dst_t = np.linspace(0, 1, new_T, dtype=np.float64)
    flat = arr.reshape(T, -1).astype(np.float64)
    out = np.empty((new_T, flat.shape[1]), dtype=np.float64)
    for i in range(flat.shape[1]):
        out[:, i] = np.interp(dst_t, src_t, flat[:, i])
    return out.reshape((new_T,) + arr.shape[1:]).astype(np.float32)


def _resample_axis_angle(rot_aa: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample axis-angle (T, J*3 or T, J, 3) by going through quaternion slerp.

    For simplicity we use linear interpolation on quaternions then renormalize.
    Good enough for evaluation use; not as principled as true SLERP for large
    angle jumps but representation_272 only consumes per-frame rotations and is
    insensitive to small interpolation noise.
    """
    if abs(src_fps - dst_fps) < 1e-6:
        return rot_aa.astype(np.float32, copy=False)
    shape_in = rot_aa.shape
    rot_aa = rot_aa.reshape(shape_in[0], -1, 3)  # (T, J, 3)
    T, J, _ = rot_aa.shape
    quat_wxyz = expmap_to_quaternion(rot_aa)  # (T, J, 4) [w,x,y,z]
    # enforce continuity (flip sign so dot >= 0)
    for t in range(1, T):
        d = (quat_wxyz[t] * quat_wxyz[t - 1]).sum(axis=-1)
        flip = d < 0
        quat_wxyz[t, flip] *= -1
    new_T = max(2, int(round(T * dst_fps / src_fps)))
    src_t = np.linspace(0, 1, T, dtype=np.float64)
    dst_t = np.linspace(0, 1, new_T, dtype=np.float64)
    out = np.empty((new_T, J, 4), dtype=np.float64)
    for j in range(J):
        for c in range(4):
            out[:, j, c] = np.interp(dst_t, src_t, quat_wxyz[:, j, c].astype(np.float64))
    out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12)
    out_aa = quaternion_to_axis_angle(torch.from_numpy(out)).numpy()
    return out_aa.reshape((new_T,) + shape_in[1:]).astype(np.float32)


def _build_smpl_85(npz_path: Path, target_fps: float) -> Optional[np.ndarray]:
    """Read MotionHub .npz, return smpl_85 array (T, 85) at target fps.

    Layout (matches MotionStreamer's smpl_85 convention):
        [:, :3]    = global_orient axis-angle
        [:, 3:66]  = body_pose axis-angle (21 * 3)
        [:, 66:72] = unused (set to 0)
        [:, 72:75] = translation (m)
        [:, 75:85] = betas (set to 0)
    """
    try:
        data = np.load(str(npz_path), allow_pickle=True)
    except Exception:
        return None
    if "poses" not in data.files or "trans" not in data.files:
        return None
    poses = np.asarray(data["poses"], dtype=np.float32)  # (T, 165) packed axis-angle
    trans = np.asarray(data["trans"], dtype=np.float32)  # (T, 3)
    fps = float(data["mocap_framerate"]) if "mocap_framerate" in data.files else 20.0

    T = poses.shape[0]
    if T < 8:
        return None
    if poses.shape[-1] not in (52 * 3, 55 * 3):
        return None

    if poses.shape[-1] == 52 * 3:
        # SMPL-H: pad jaw/eyes to 55 joints (after first 22 body joints)
        pose_55 = np.concatenate(
            [poses[:, :22 * 3], np.zeros((T, 9), dtype=np.float32), poses[:, 22 * 3:]],
            axis=1,
        )
    else:
        pose_55 = poses

    # take the first 22 joints (global_orient + 21 body)
    pose_22 = pose_55[:, : 22 * 3]  # (T, 66)

    # resample to target fps using quaternion-domain linear interp for rotations
    if fps and abs(fps - target_fps) > 1e-3:
        pose_22 = _resample_axis_angle(pose_22, fps, target_fps)
        trans = _linear_resample(trans, fps, target_fps)

    new_T = pose_22.shape[0]
    smpl_85 = np.zeros((new_T, 85), dtype=np.float32)
    smpl_85[:, : 22 * 3] = pose_22
    smpl_85[:, 72:75] = trans
    return smpl_85


# ---------------------------------------------------------------------------
# Stage 3: face_z_transform (mirror face_z_transform.py)
# ---------------------------------------------------------------------------

def _face_z_transform(smpl_85: np.ndarray) -> np.ndarray:
    """Rotate so first frame's root faces +Z (in-memory, mirrors face_z_transform.py)."""
    T = smpl_85.shape[0]
    pose_body = smpl_85[:, :72].reshape(T, -1, 3)  # (T, 24, 3)  (only first 22 are populated)
    trans = smpl_85[:, 72:75]
    beta = smpl_85[:, 75:]

    root_first = pose_body[0, 0]  # (3,)
    root_first_quat = expmap_to_quaternion(root_first[None])  # (1, 4) wxyz
    root_first_quat_xyzw = root_first_quat[0, [1, 2, 3, 0]]
    root_first_quat_xyzw_t = torch.from_numpy(root_first_quat_xyzw).float().unsqueeze(0)

    # heading_inv = -atan2(rot_dir.x, rot_dir.z); rotate around +y
    def _calc_heading(q_xyzw: torch.Tensor) -> torch.Tensor:
        ref_dir = torch.zeros_like(q_xyzw[..., 0:3])
        ref_dir[..., 2] = 1
        # quat-rotate v by q (xyzw)
        q_w = q_xyzw[:, -1]
        q_vec = q_xyzw[:, :3]
        a = ref_dir * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, ref_dir, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * torch.bmm(
            q_vec.view(q_xyzw.shape[0], 1, 3), ref_dir.view(q_xyzw.shape[0], 3, 1)
        ).squeeze(-1) * 2.0
        rot_dir = a + b + c
        return torch.atan2(rot_dir[..., 0], rot_dir[..., 2])

    heading = _calc_heading(root_first_quat_xyzw_t)
    axis = torch.zeros_like(root_first_quat_xyzw_t[..., 0:3])
    axis[..., 1] = 1
    heading_inv_axis_angle = (-heading).unsqueeze(-1) * axis  # (1, 3)
    heading_inv_axis_angle = heading_inv_axis_angle.numpy()
    q_diff = expmap_to_quaternion(heading_inv_axis_angle)  # (1, 4) wxyz

    q_diff_T = np.repeat(q_diff.reshape(1, -1), T, axis=0)
    pose_body_root_quat = expmap_to_quaternion(pose_body[:, 0])
    new_root_quat = qmul_np(q_diff_T, pose_body_root_quat)
    new_root_aa = quaternion_to_axis_angle(torch.from_numpy(new_root_quat)).numpy()  # (T, 3)
    trans = qrot_np(q_diff_T, trans)

    out = np.zeros_like(smpl_85)
    out[:, :3] = new_root_aa
    out[:, 3:72] = pose_body[:, 1:].reshape(T, -1)  # body + zero pads
    out[:, 72:75] = trans
    out[:, 75:] = beta
    return out


# ---------------------------------------------------------------------------
# Stage 4: SMPL FK (SmplxLite)
# ---------------------------------------------------------------------------

class _FKEngine:
    def __init__(self, smpl_model_path: str, device: torch.device):
        self.smplx = SmplxLite(model_path=smpl_model_path, num_betas=10).to(device).eval()
        for p in self.smplx.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def forward(self, smpl_85_t: np.ndarray) -> np.ndarray:
        """smpl_85 (T, 85) -> joints (T, 22, 3)."""
        T = smpl_85_t.shape[0]
        global_orient = torch.from_numpy(smpl_85_t[:, :3]).float().to(self.device)[None]
        body_pose = torch.from_numpy(smpl_85_t[:, 3:66]).float().to(self.device)[None]
        transl = torch.from_numpy(smpl_85_t[:, 72:75]).float().to(self.device)[None]
        betas = torch.from_numpy(smpl_85_t[:, 75:85]).float().to(self.device)[None]
        joints, _, _ = self.smplx.fk(transl, global_orient, body_pose, betas=betas)
        return joints[0].cpu().numpy()  # (T, 22, 3)


# ---------------------------------------------------------------------------
# Stage 5: 272-dim representation (mirror representation_272.py)
# ---------------------------------------------------------------------------

def _rot_yaw(yaw: np.ndarray) -> np.ndarray:
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])


def _foot_detect(global_positions: np.ndarray, thres: float = 0.0015) -> np.ndarray:
    left_foot, right_foot = 10, 11
    vel = global_positions[1:] - global_positions[:-1]
    sp = np.sqrt(np.sum(vel[:, np.array([left_foot, right_foot])] ** 2, axis=-1))
    contacts = sp < thres
    for ci in range(contacts.shape[1]):
        contacts[:, ci] = ndimage.median_filter(contacts[:, ci], size=6, mode="nearest")
    return contacts


def _representation_272(joint_pos: np.ndarray, smpl_85: np.ndarray) -> Optional[np.ndarray]:
    """
    Args:
        joint_pos: (T, 22, 3)  joint positions in world coords (after face_z + FK)
        smpl_85:   (T, 85)     same SMPL-85 used for FK (face_z applied)
    Returns:
        feat: (T, 272) or None on failure
    """
    position_data = joint_pos[:, :22, :3].astype(np.float64)
    nfrm, njoint, _ = position_data.shape

    rotation_smpl_axis_angle = smpl_85.astype(np.float64)
    rotations_wxyz = expmap_to_quaternion(rotation_smpl_axis_angle[:, :66].reshape(nfrm, njoint, 3))
    rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)  # (T, J, 3, 3)

    # put on floor & origin (first frame)
    ori = copy.deepcopy(position_data[0, 0])
    y_min = np.min(position_data[:, :, 1])
    ori[1] = y_min
    position_data = position_data - ori
    velocities_root = position_data[1:, 0, :] - position_data[:-1, 0, :]

    # local position on xz origin
    position_data[:, :, 0] -= position_data[:, 0:1, 0]
    position_data[:, :, 2] -= position_data[:, 0:1, 2]

    # heading
    global_heading = -np.arctan2(rotations_matrix[:, 0, 0, 2], rotations_matrix[:, 0, 2, 2])
    global_heading_rot = np.array([_rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([_rot_yaw(x) for x in global_heading_diff])

    positions_no_heading = np.matmul(
        np.repeat(global_heading_rot[:, None, :, :], njoint, axis=1), position_data[..., None]
    ).squeeze(-1)

    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]
    velocities_root_xy_no_heading = np.matmul(
        global_heading_rot[:-1], velocities_root[:, :, None]
    ).squeeze(-1)[..., [0, 2]]

    rotations_matrix[:, 0, ...] = np.matmul(global_heading_rot, rotations_matrix[:, 0, ...])

    size_frame = 8 + njoint * 3 + njoint * 3 + njoint * 6
    final_x = np.zeros((nfrm, size_frame), dtype=np.float64)
    final_x[0, 2] = 1
    final_x[0, 6] = 1
    try:
        final_x[1:, 2:8] = matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy()
    except Exception:
        return None
    final_x[1:, :2] = velocities_root_xy_no_heading
    final_x[:, 8 : 8 + 3 * njoint] = positions_no_heading.reshape(nfrm, -1)
    final_x[1:, 8 + 3 * njoint : 8 + 6 * njoint] = velocities_no_heading.reshape(nfrm - 1, -1)
    final_x[:, 8 + 6 * njoint : 8 + 12 * njoint] = rotations_matrix[..., :, :2, :].reshape(nfrm, -1)

    return final_x.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level: convert one anno entry
# ---------------------------------------------------------------------------

def _resolve_motion_path(entry: Dict, data_dir: Path) -> Optional[Path]:
    """Resolve the SMPL-X .npz path from a MotionHub annotation entry.

    Annotations typically have one of the following structures:
        {"smplx55_path": "<subset>/smplx_55/<id>.npz", ...}
        {"motion_path":   "<subset>/smplx_55/<id>.npz", ...}
        {"path":          "...npz", ...}
    """
    for k in ("smplx55_path", "smplx_path", "motion_path", "path"):
        v = entry.get(k)
        if isinstance(v, str) and v.endswith(".npz"):
            p = data_dir / v
            if p.exists():
                return p
            p2 = Path(v)
            if p2.exists():
                return p2
    return None


def _resolve_caption_pool(entry: Dict, data_dir: Path) -> List[str]:
    """Pool captions like LoadCompatibleCaption: macro/meso/micro from json or .txt."""
    pool: List[str] = []
    for k in (
        "caption", "captions", "text", "texts",
        "compatible_caption_path", "hierarchical_caption_path",
        "caption_path", "text_path", "compatible_caption", "hierarchical_caption",
    ):
        v = entry.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            if v.endswith(".txt") or v.endswith(".json") or v.endswith(".npy"):
                p = data_dir / v if not Path(v).is_absolute() else Path(v)
                pool.extend(_load_caption_pool(p))
            else:
                pool.append(v)
        elif isinstance(v, list):
            pool.extend([str(x) for x in v if isinstance(x, str)])
        elif isinstance(v, dict):
            for kk in ("macro", "meso", "micro"):
                vv = v.get(kk)
                if isinstance(vv, list):
                    pool.extend([str(x) for x in vv if isinstance(x, str)])
                elif isinstance(vv, str):
                    pool.append(vv)
    return [c for c in pool if c]


def _entry_id(entry: Dict, fallback_idx: int) -> str:
    for k in ("_id", "id", "name", "key"):
        v = entry.get(k)
        if isinstance(v, str) and v:
            return v.replace("/", "_")
    p = _resolve_motion_path(entry, Path(""))
    if p is not None:
        return p.stem
    return f"sample{fallback_idx:08d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anno_file", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--ms_data_root", required=True,
                   help="MotionStreamer's humanml3d_272 root (for mean_std).")
    p.add_argument("--smpl_model_path", default="checkpoints/smpl_models/smplx")
    p.add_argument("--target_fps", type=float, default=20.0)
    p.add_argument("--min_frames", type=int, default=60)
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    out_root = Path(args.out_root)
    motion_dir = out_root / "motion_data"
    text_dir = out_root / "texts"
    split_dir = out_root / "split"
    mean_std_dir = out_root / "mean_std"
    motion_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    mean_std_dir.mkdir(parents=True, exist_ok=True)

    # symlink mean_std from MS HumanML3D-272 release
    ms_data_root = Path(args.ms_data_root)
    for fname in ("Mean.npy", "Std.npy"):
        src = ms_data_root / "mean_std" / fname
        dst = mean_std_dir / fname
        if not dst.exists() and src.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                import shutil
                shutil.copy(src, dst)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[+] device = {device}")
    print(f"[+] out_root = {out_root}")

    print("[+] Loading SMPL-X FK ...")
    fk = _FKEngine(args.smpl_model_path, device)

    print("[+] Loading anno ...")
    entries = _load_anno(Path(args.anno_file))
    print(f"    {len(entries)} entries")

    if args.max_samples:
        entries = entries[: args.max_samples]

    data_dir = Path(args.data_dir)
    written, skipped, skip_reasons = 0, 0, {}
    split_lines = []
    t0 = time.time()
    for i, entry in enumerate(tqdm(entries, ncols=80)):
        sid = _entry_id(entry, i)
        npz_path = _resolve_motion_path(entry, data_dir)
        if npz_path is None:
            skipped += 1
            skip_reasons["no_npz_path"] = skip_reasons.get("no_npz_path", 0) + 1
            continue

        smpl_85 = _build_smpl_85(npz_path, args.target_fps)
        if smpl_85 is None:
            skipped += 1
            skip_reasons["bad_npz"] = skip_reasons.get("bad_npz", 0) + 1
            continue
        if smpl_85.shape[0] < args.min_frames:
            skipped += 1
            skip_reasons["too_short"] = skip_reasons.get("too_short", 0) + 1
            continue
        if smpl_85.shape[0] >= args.max_frames:
            smpl_85 = smpl_85[: args.max_frames - 1]

        smpl_85_face_z = _face_z_transform(smpl_85)
        try:
            joints = fk.forward(smpl_85_face_z)
        except Exception as e:
            skipped += 1
            skip_reasons[f"fk:{type(e).__name__}"] = skip_reasons.get(f"fk:{type(e).__name__}", 0) + 1
            continue

        feat = _representation_272(joints, smpl_85_face_z)
        if feat is None:
            skipped += 1
            skip_reasons["repr272_fail"] = skip_reasons.get("repr272_fail", 0) + 1
            continue

        captions = _resolve_caption_pool(entry, data_dir)
        if not captions:
            skipped += 1
            skip_reasons["no_caption"] = skip_reasons.get("no_caption", 0) + 1
            continue

        np.save(motion_dir / f"{sid}.npy", feat)
        # HumanML3D-format text file: caption#tokens#start#end (tokens placeholder)
        with (text_dir / f"{sid}.txt").open("w") as f:
            for cap in captions:
                tokens = " ".join(f"{w.lower()}/UNK" for w in cap.split())
                f.write(f"{cap}#{tokens}#0.0#0.0\n")
        split_lines.append(sid)
        written += 1

    (split_dir / "test.txt").write_text("\n".join(split_lines))
    elapsed = time.time() - t0
    print(f"[+] done: written={written}  skipped={skipped}  elapsed={elapsed:.1f}s")
    if skip_reasons:
        print(f"    skip_reasons={skip_reasons}")
    print(f"[+] split file: {split_dir / 'test.txt'}")
    print(f"[+] motion_data: {motion_dir}")
    print(f"[+] texts:       {text_dir}")
    print(f"[+] mean_std:    {mean_std_dir} -> {ms_data_root / 'mean_std'}")


if __name__ == "__main__":
    main()
