#!/usr/bin/env python3
"""Retarget HumanML3D-263 predictions to SMPL-style motion_135.

This script intentionally avoids the repository's existing HumanML3D-to-SMPL
conversion path. It only uses the canonical HumanML3D RIC decoder, scipy's
vector alignment, and the public smplx layer:

    HML3D-263 -> 22 joints -> hierarchical IK on SMPL rest skeleton
              -> global_orient/body_pose/transl + motion_135

The conversion is not mathematically exact: HumanML3D-263 does not uniquely
determine SMPL pose twist, shape, or mesh details. The saved fit MPJPE is a
diagnostic for how well the SMPL skeleton tracks the recovered 22 joints.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R, Slerp


def _patch_numpy_chumpy_aliases() -> None:
    """Keep legacy SMPL/chumpy pickles loadable under newer NumPy releases."""
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
        "int_": np.int64,
        "float_": np.float64,
        "complex_": np.complex128,
        "object_": object,
        "unicode_": str,
        "str_": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


_patch_numpy_chumpy_aliases()

REPO = Path(__file__).resolve().parents[2]
MS272_ROOT = REPO / "ref_repo" / "MotionStreamer" / "272-dim-Motion-Representation"
VENDORED_SMPLX = MS272_ROOT / "utils" / "smplx"
try:
    import smplx  # noqa: E402
except ModuleNotFoundError:
    if MS272_ROOT.exists():
        sys.path.insert(0, str(MS272_ROOT))
    if VENDORED_SMPLX.exists():
        sys.path.insert(0, str(VENDORED_SMPLX))
    import smplx  # noqa: E402


# HumanML3D 22-joint skeleton order follows the first 22 SMPL joints.
N_JOINTS = 22


def _qinv(q: np.ndarray) -> np.ndarray:
    return q * np.array([1, -1, -1, -1], dtype=q.dtype)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def _recover_root_rot_pos(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rot_vel = data[..., 0]
    r_rot_ang = np.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = np.cumsum(r_rot_ang, axis=-1)
    r_rot_quat = np.zeros(data.shape[:-1] + (4,), dtype=data.dtype)
    r_rot_quat[..., 0] = np.cos(r_rot_ang)
    r_rot_quat[..., 2] = np.sin(r_rot_ang)
    r_pos = np.zeros(data.shape[:-1] + (3,), dtype=data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = _qrot(_qinv(r_rot_quat), r_pos)
    r_pos = np.cumsum(r_pos, axis=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data: np.ndarray, joints_num: int = N_JOINTS) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    r_rot_quat, r_pos = _recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.reshape(positions.shape[:-1] + (-1, 3))
    q = _qinv(r_rot_quat)[..., None, :]
    q = np.broadcast_to(q, positions.shape[:-1] + (4,))
    positions = _qrot(q, positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    return np.concatenate([r_pos[..., None, :], positions], axis=-2)


def cont6d_to_matrix_hml(cont6d: np.ndarray) -> np.ndarray:
    """HumanML/MoMask column-major cont6d -> rotation matrix."""
    cont6d = np.asarray(cont6d, dtype=np.float32)
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.maximum(np.linalg.norm(x_raw, axis=-1, keepdims=True), 1e-8)
    z = np.cross(x, y_raw)
    z = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    xyzw = quat[..., [1, 2, 3, 0]]
    return R.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(quat.shape[:-1] + (3, 3)).astype(np.float32)


def recover_hml263_local_rotations(feats: np.ndarray, joints_num: int = N_JOINTS) -> np.ndarray:
    """Recover HumanML canonical-skeleton local rotations from a 263D clip.

    These are not original SMPL rotations. They are the local rotations produced
    by HumanML3D/MoMask inverse kinematics on the canonical 22-joint skeleton,
    and are useful as a twist/orientation prior for SMPL fitting.
    """
    feats = np.asarray(feats, dtype=np.float32)
    if feats.ndim != 2 or feats.shape[-1] != 263:
        raise ValueError(f"expected HML263 features (T,263), got {feats.shape}")
    root_quat, _ = _recover_root_rot_pos(feats)
    root_mat = quat_wxyz_to_matrix(root_quat)
    start = 4 + (joints_num - 1) * 3
    end = start + (joints_num - 1) * 6
    body = feats[:, start:end].reshape(len(feats), joints_num - 1, 6)
    body_mat = cont6d_to_matrix_hml(body)
    return np.concatenate([root_mat[:, None], body_mat], axis=1).astype(np.float32)


def _slerp_length(rot: np.ndarray, target_len: int) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float32)
    if target_len <= 0 or len(rot) == target_len:
        return rot
    if len(rot) < 2:
        return np.repeat(rot[:1], target_len, axis=0).astype(np.float32)
    src_times = np.arange(len(rot), dtype=np.float64)
    dst_times = np.linspace(0.0, len(rot) - 1, int(target_len), dtype=np.float64)
    out = np.empty((int(target_len), rot.shape[1], 3, 3), dtype=np.float32)
    for j in range(rot.shape[1]):
        out[:, j] = Slerp(src_times, R.from_matrix(rot[:, j]))(dst_times).as_matrix().astype(np.float32)
    return out


def resample_rotations(rot: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    if abs(src_fps - dst_fps) < 1e-6 or len(rot) < 2:
        return np.asarray(rot, dtype=np.float32)
    return _slerp_length(rot, max(2, int(round(len(rot) * dst_fps / src_fps))))


def fit_length_rotations(rot: np.ndarray, target_len: int | None) -> np.ndarray:
    if target_len is None:
        return np.asarray(rot, dtype=np.float32)
    return _slerp_length(rot, int(target_len))


def resample_linear(x: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if abs(src_fps - dst_fps) < 1e-6 or len(x) < 2:
        return x
    new_t = max(2, int(round(len(x) * dst_fps / src_fps)))
    grid = np.linspace(0.0, len(x) - 1, new_t)
    lo = np.floor(grid).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (grid - lo).astype(np.float32)
    shape = (new_t,) + (1,) * (x.ndim - 1)
    return x[lo] * (1.0 - w.reshape(shape)) + x[hi] * w.reshape(shape)


def fit_length_linear(x: np.ndarray, target_len: int | None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if target_len is None:
        return x
    target_len = int(target_len)
    if target_len <= 0 or len(x) == target_len:
        return x
    if len(x) < 2:
        if len(x) == 0:
            return x
        return np.repeat(x[:1], target_len, axis=0).astype(np.float32)
    grid = np.linspace(0.0, len(x) - 1, target_len)
    lo = np.floor(grid).astype(np.int64)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (grid - lo).astype(np.float32)
    shape = (target_len,) + (1,) * (x.ndim - 1)
    return (x[lo] * (1.0 - w.reshape(shape)) + x[hi] * w.reshape(shape)).astype(np.float32)


def _load_canonical_meta(meta_dir: Path | None, sid: str) -> dict[str, np.ndarray] | None:
    if meta_dir is None:
        return None
    path = meta_dir / f"{sid}.npz"
    if not path.exists():
        return None
    data = np.load(str(path), allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def _restore_root_translation(
    transl: np.ndarray,
    meta: dict[str, np.ndarray] | None,
    mode: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if mode == "none" or meta is None:
        return transl.astype(np.float32), {"mode": "none", "applied": False}
    if mode not in {"auto", "source_transl"}:
        raise ValueError(f"unsupported root translation restore mode: {mode}")
    if "source_motion135_transl" not in meta:
        return transl.astype(np.float32), {
            "mode": mode,
            "applied": False,
            "reason": "source_motion135_transl_missing",
        }
    source = np.asarray(meta["source_motion135_transl"], dtype=np.float32)
    restored = fit_length_linear(source, len(transl)).astype(np.float32)
    return restored, {
        "mode": mode,
        "applied": True,
        "source_frames": int(len(source)),
        "output_frames": int(len(restored)),
    }


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    valid = n[..., 0] > eps
    return v / np.maximum(n, eps), valid


def estimate_local_rotations(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
) -> np.ndarray:
    """Estimate local rotations by aligning SMPL rest bones to target bones."""
    target_joints = np.asarray(target_joints, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents[:N_JOINTS], dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(N_JOINTS)]
    for j in range(1, N_JOINTS):
        p = int(parents[j])
        if 0 <= p < N_JOINTS:
            children[p].append(j)

    offsets = np.zeros((N_JOINTS, 3), dtype=np.float64)
    for j in range(1, N_JOINTS):
        offsets[j] = rest_joints[j] - rest_joints[int(parents[j])]

    local = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), N_JOINTS, 1, 1))

    for t, joints in enumerate(target_joints):
        for j in range(N_JOINTS):
            child_ids = children[j]
            parent = int(parents[j])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            rest_vecs_list = [offsets[c] for c in child_ids]
            target_vecs_list = [joints[c] - joints[j] for c in child_ids]
            weights = [1.0] * len(rest_vecs_list)
            if orientation_mode == "parent_frame" and parent >= 0:
                # Position-only IK leaves twist around a single bone undefined.
                # A weak joint-to-parent reference chooses a stable local frame
                # without letting the virtual axis dominate child-bone fitting.
                rest_vecs_list.append(rest_joints[parent] - rest_joints[j])
                target_vecs_list.append(joints[parent] - joints[j])
                weights.append(parent_ref_weight)
            if not rest_vecs_list:
                local[t, j] = np.eye(3)
                global_r[t, j] = parent_global @ local[t, j]
                continue

            rest_vecs = np.stack(rest_vecs_list, axis=0)
            target_vecs = np.stack(target_vecs_list, axis=0)
            rest_unit, rest_valid = _safe_normalize(rest_vecs)
            target_unit, target_valid = _safe_normalize(target_vecs)
            valid = rest_valid & target_valid
            if not np.any(valid):
                rot_local = np.eye(3)
            else:
                src = rest_unit[valid]
                dst_world = target_unit[valid]
                dst_local = (parent_global.T @ dst_world.T).T
                valid_weights = np.asarray(weights, dtype=np.float64)[valid]
                try:
                    rot_local = R.align_vectors(dst_local, src, weights=valid_weights)[0].as_matrix()
                except Exception:
                    rot_local = np.eye(3)
            local[t, j] = rot_local
            global_r[t, j] = parent_global @ rot_local
    return local.astype(np.float32)


def matrix_to_rot6d_rowmajor(rotmat: np.ndarray) -> np.ndarray:
    return np.asarray(rotmat[..., :, :2], dtype=np.float32).reshape(*rotmat.shape[:-2], 6)


def load_smpl_rest(model_dir: Path, device: torch.device):
    if model_dir.name == "body_models":
        nochumpy = model_dir.with_name("body_models_nochumpy")
        if (nochumpy / "smpl" / "SMPL_NEUTRAL.pkl").exists():
            model_dir = nochumpy
    model = smplx.create(
        str(model_dir),
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        batch_size=1,
    ).to(device)
    model.eval()
    with torch.no_grad():
        out = model(
            betas=torch.zeros(1, 10, device=device),
            body_pose=torch.zeros(1, 69, device=device),
            global_orient=torch.zeros(1, 3, device=device),
            transl=torch.zeros(1, 3, device=device),
        )
    rest = out.joints[0, :N_JOINTS].detach().cpu().numpy().astype(np.float32)
    parents = model.parents.detach().cpu().numpy().astype(np.int64)
    return model, rest, parents


def smpl_forward_22(
    model,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    transl: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n = len(global_orient)
    chunks = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start
        body_23 = np.zeros((b, 69), dtype=np.float32)
        body_23[:, :63] = body_pose_21[start:end]
        tr = np.zeros((b, 3), dtype=np.float32) if transl is None else transl[start:end]
        with torch.no_grad():
            out = model(
                betas=torch.zeros(b, 10, device=device),
                body_pose=torch.from_numpy(body_23).to(device),
                global_orient=torch.from_numpy(global_orient[start:end]).to(device),
                transl=torch.from_numpy(tr).to(device),
            )
        chunks.append(out.joints[:, :N_JOINTS].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def refine_smpl_fit(
    model,
    target_joints: np.ndarray,
    global_orient: np.ndarray,
    body_pose_21: np.ndarray,
    transl: np.ndarray,
    iters: int,
    lr: float,
    pose_l2_weight: float,
    angle_prior_weight: float,
    device: torch.device,
    smooth_weight: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine IK initialization by optimizing SMPL pose/transl against joints."""
    if iters <= 0:
        fitted = smpl_forward_22(model, global_orient, body_pose_21, transl, 512, device)
        return global_orient, body_pose_21, transl, fitted

    target = torch.from_numpy(target_joints.astype(np.float32)).to(device)
    n = len(target_joints)
    g = torch.tensor(global_orient, dtype=torch.float32, device=device, requires_grad=True)
    b21 = torch.tensor(body_pose_21, dtype=torch.float32, device=device, requires_grad=True)
    tr = torch.tensor(transl, dtype=torch.float32, device=device, requires_grad=True)
    b21_init = b21.detach().clone()
    opt = torch.optim.Adam([g, b21, tr], lr=lr)

    for _ in range(iters):
        body_23 = torch.zeros(n, 69, dtype=torch.float32, device=device)
        body_23[:, :63] = b21
        out = model(
            betas=torch.zeros(n, 10, device=device),
            body_pose=body_23,
            global_orient=g,
            transl=tr,
        )
        joints = out.joints[:, :N_JOINTS]
        data_loss = ((joints - target) ** 2).sum(dim=-1).mean()
        pose_keep = ((b21 - b21_init) ** 2).mean()
        pose_prior = (body_23 ** 2).mean()
        if angle_prior_weight > 0:
            # SMPLify angle prior indices in the 69-dim body-pose vector:
            # left/right knees and elbows are discouraged from bending backward.
            idx = torch.tensor([55, 58, 12, 15], dtype=torch.long, device=device)
            signs = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device=device)
            angle_prior = torch.exp(body_23[:, idx] * signs).pow(2).mean()
        else:
            angle_prior = torch.tensor(0.0, device=device)
        if n >= 3:
            tr_acc = tr[2:] - 2 * tr[1:-1] + tr[:-2]
            pose_acc = b21[2:] - 2 * b21[1:-1] + b21[:-2]
            smooth = (tr_acc ** 2).mean() + 1e-2 * (pose_acc ** 2).mean()
        else:
            smooth = torch.tensor(0.0, device=device)
        loss = (
            data_loss
            + 1e-4 * pose_keep
            + pose_l2_weight * pose_prior
            + angle_prior_weight * angle_prior
            + smooth_weight * smooth
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        body_23 = torch.zeros(n, 69, dtype=torch.float32, device=device)
        body_23[:, :63] = b21
        out = model(
            betas=torch.zeros(n, 10, device=device),
            body_pose=body_23,
            global_orient=g,
            transl=tr,
        )
        fitted = out.joints[:, :N_JOINTS].detach().cpu().numpy().astype(np.float32)
    return (
        g.detach().cpu().numpy().astype(np.float32),
        b21.detach().cpu().numpy().astype(np.float32),
        tr.detach().cpu().numpy().astype(np.float32),
        fitted,
    )


def retarget_one(
    in_path: Path,
    out_path: Path,
    model,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    source_fps: float,
    target_fps: float,
    batch_size: int,
    device: torch.device,
    floor_align: bool,
    refine_iters: int,
    refine_lr: float,
    rotation_init: str,
    orientation_mode: str,
    parent_ref_weight: float,
    pose_l2_weight: float,
    angle_prior_weight: float,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    canonical_meta_dir: Path | None,
    restore_root_translation: str,
    target_len: int | None = None,
) -> dict:
    arr = np.load(str(in_path)).astype(np.float32)
    hml_local_r = None
    use_hml_rot = rotation_init == "hml263"
    if arr.ndim == 3 and arr.shape[1:] == (22, 3):
        target = resample_linear(arr, source_fps, target_fps)
    else:
        feats = arr
        if feats.ndim != 2 or feats.shape[-1] != 263:
            raise ValueError(f"expected (T,263) or (T,22,3), got {feats.shape}")
        if mean is not None and std is not None:
            feats = feats * std + mean
        target = recover_from_ric(feats, N_JOINTS)
        if use_hml_rot:
            hml_local_r = recover_hml263_local_rotations(feats, N_JOINTS)
            hml_local_r = resample_rotations(hml_local_r, source_fps, target_fps)
        target = resample_linear(target, source_fps, target_fps)
    target = fit_length_linear(target, target_len)
    hml_local_r = fit_length_rotations(hml_local_r, target_len) if hml_local_r is not None else None
    if floor_align:
        target = target.copy()
        target[..., 1] -= target[..., 1].min()

    if rotation_init == "hml263" and hml_local_r is None:
        raise ValueError("--rotation-init hml263 requires HML263 feature input")
    if hml_local_r is not None:
        local_r = hml_local_r
        rotation_init_used = "hml263"
    else:
        local_r = estimate_local_rotations(
            target,
            rest_joints,
            parents,
            orientation_mode=orientation_mode,
            parent_ref_weight=parent_ref_weight,
        )
        rotation_init_used = "position_ik"
    aa = R.from_matrix(local_r.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    aa = aa.reshape(len(target), N_JOINTS, 3)
    global_orient = aa[:, 0]
    body_pose = aa[:, 1:].reshape(len(target), 63)

    joints_no_trans = smpl_forward_22(model, global_orient, body_pose, None, batch_size, device)
    transl = (target[:, 0] - joints_no_trans[:, 0]).astype(np.float32)
    global_orient, body_pose, transl, fitted = refine_smpl_fit(
        model,
        target,
        global_orient,
        body_pose,
        transl,
        refine_iters,
        refine_lr,
        pose_l2_weight,
        angle_prior_weight,
        device,
    )
    canonical_transl = transl.copy()
    canonical_fitted = fitted.copy()
    canonical_mpjpe_mm = np.linalg.norm(canonical_fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0
    meta = _load_canonical_meta(canonical_meta_dir, in_path.stem)
    transl, root_restore_info = _restore_root_translation(transl, meta, restore_root_translation)
    if root_restore_info.get("applied"):
        fitted = smpl_forward_22(model, global_orient, body_pose, transl, batch_size, device)
    local_r = R.from_rotvec(
        np.concatenate([global_orient[:, None, :], body_pose.reshape(len(target), 21, 3)], axis=1)
        .reshape(-1, 3)
    ).as_matrix().reshape(len(target), N_JOINTS, 3, 3).astype(np.float32)
    output_target_mpjpe_mm = np.linalg.norm(fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0

    motion_135 = np.concatenate(
        [transl, matrix_to_rot6d_rowmajor(local_r).reshape(len(target), N_JOINTS * 6)],
        axis=-1,
    ).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        motion_135=motion_135,
        transl=transl.astype(np.float32),
        canonical_transl=canonical_transl.astype(np.float32),
        global_orient=global_orient.astype(np.float32),
        body_pose=body_pose.astype(np.float32),
        target_joints=target.astype(np.float32),
        fitted_joints=fitted.astype(np.float32),
        canonical_fitted_joints=canonical_fitted.astype(np.float32),
        fit_mpjpe_mm=canonical_mpjpe_mm,
        output_vs_canonical_target_mpjpe_mm=output_target_mpjpe_mm,
        source_fps=np.array(source_fps, dtype=np.float32),
        target_fps=np.array(target_fps, dtype=np.float32),
        refine_iters=np.array(refine_iters, dtype=np.int32),
        rotation_init=np.array(rotation_init_used),
        root_translation_restore_mode=np.array(str(root_restore_info["mode"])),
        root_translation_restored=np.array(bool(root_restore_info.get("applied", False))),
    )
    return {
        "sid": in_path.stem,
        "frames": int(len(target)),
        "target_len": None if target_len is None else int(target_len),
        "mpjpe_mm_mean": float(canonical_mpjpe_mm.mean()),
        "mpjpe_mm_p95": float(np.percentile(canonical_mpjpe_mm, 95)),
        "rotation_init": rotation_init_used,
        "root_translation_restore": root_restore_info,
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
    ap.add_argument("--pose-l2-weight", type=float, default=0.0)
    ap.add_argument("--angle-prior-weight", type=float, default=0.0)
    ap.add_argument(
        "--rotation-init",
        choices=["position_ik", "hml263"],
        default="position_ik",
        help="Initial local rotations for SMPL fitting. hml263 is experimental and uses the raw HML263 rot block.",
    )
    ap.add_argument(
        "--canonical-meta-dir",
        default=None,
        help="Optional sidecar directory produced by motion135_dir_to_hml263.py --metadata-dir.",
    )
    ap.add_argument(
        "--restore-root-translation",
        choices=["auto", "none", "source_transl"],
        default="auto",
        help="Restore output root translation from sidecar metadata. "
             "'auto' restores source_transl when it is available and otherwise keeps canonical translation.",
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
        "--target-length-anno",
        default=None,
        help="Optional annotation JSON whose data_list entries contain num_frames. "
             "When set, each output is linearly fit to the official target length.",
    )
    ap.add_argument(
        "--mean-path",
        default="ref_repo/Momask/weights/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/mean.npy",
    )
    ap.add_argument(
        "--std-path",
        default="ref_repo/Momask/weights/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2/meta/std.npy",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    model, rest_joints, parents = load_smpl_rest(Path(args.model_dir), device)
    if args.input_normalized:
        mean = np.load(args.mean_path).astype(np.float32)
        std = np.load(args.std_path).astype(np.float32)
        if mean.shape != (263,) or std.shape != (263,):
            raise ValueError(f"expected 263-dim mean/std, got {mean.shape} and {std.shape}")
    else:
        mean = std = None
    target_lengths = {}
    if args.target_length_anno:
        raw = json.loads(Path(args.target_length_anno).read_text())
        data = raw.get("data_list", raw) if isinstance(raw, dict) else raw
        if isinstance(data, dict):
            iterator = data.items()
        else:
            iterator = (
                (str(item.get("motion_id") or item.get("id") or idx), item)
                for idx, item in enumerate(data)
            )
        for key, entry in iterator:
            if isinstance(entry, dict) and entry.get("num_frames") is not None:
                target_lengths[str(key)] = int(entry["num_frames"])
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
    canonical_meta_dir = Path(args.canonical_meta_dir) if args.canonical_meta_dir else None
    print(
        f"[setup] files={len(files)} shard={args.shard_index}/{args.num_shards} "
        f"out={out_dir} device={device} target_fps={args.target_fps} "
        f"restore_root={args.restore_root_translation}",
        flush=True,
    )

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
                rest_joints,
                parents,
                args.source_fps,
                args.target_fps,
                args.batch_size,
                device,
                args.floor_align,
                args.refine_iters,
                args.refine_lr,
                args.rotation_init,
                args.orientation_mode,
                args.parent_ref_weight,
                args.pose_l2_weight,
                args.angle_prior_weight,
                mean,
                std,
                canonical_meta_dir,
                args.restore_root_translation,
                target_lengths.get(in_path.stem),
            )
            summary.append(item)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {in_path.name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 25 == 0 or i == len(files):
            mean = np.mean([x["mpjpe_mm_mean"] for x in summary]) if summary else float("nan")
            print(f"[progress] {i}/{len(files)} ok={len(summary)} fail={failed} mean_mpjpe_mm={mean:.2f}", flush=True)

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
