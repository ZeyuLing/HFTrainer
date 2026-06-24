#!/usr/bin/env python3
"""Audit HumanML3D GT after an SMPL -> SOMA -> SMPL retarget round trip.

This script builds a control set for KIMODO/SOMA evaluation:

    HumanML3D-272 GT -> motion_135(SMPL22) -> SOMA30 FK -> SMPL22 IK -> motion_135

The resulting ``motion_135`` clips can be converted back to MotionStreamer-272
and scored with the same evaluator as generated motions.  This quantifies how
much the SOMA->SMPL fitting step damages FID/R-Precision/MM-Dist when the input
is ground-truth motion rather than a model prediction.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KIMODO_ROOT = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts/eval", PROJECT_ROOT / "scripts/analysis", KIMODO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402
from build_kimodo_skeleton_smpl_ik_viewer import (  # noqa: E402
    _retarget_one,
    _summarize_outputs,
)
from hml263_to_smpl_ik import load_smpl_rest, matrix_to_rot6d_rowmajor  # noqa: E402


SMPLX22_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
SMPL22_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
SOMA30_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
    "RightFoot", "RightToeBase",
]
SOMA30_PARENT_NAMES = [
    None, "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
    "Head", "Head", "Chest", "LeftShoulder", "LeftArm", "LeftForeArm",
    "LeftHand", "LeftHand", "Chest", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHand", "Hips", "LeftLeg", "LeftShin",
    "LeftFoot", "Hips", "RightLeg", "RightShin", "RightFoot",
]
SMPLX_TO_SOMA_NAME = {
    "pelvis": "Hips", "left_hip": "LeftLeg", "right_hip": "RightLeg",
    "spine1": "Spine1", "left_knee": "LeftShin", "right_knee": "RightShin",
    "spine2": "Spine2", "left_ankle": "LeftFoot", "right_ankle": "RightFoot",
    "spine3": "Chest", "left_foot": "LeftToeBase", "right_foot": "RightToeBase",
    "neck": "Neck1", "left_collar": "LeftShoulder", "right_collar": "RightShoulder",
    "head": "Head", "left_shoulder": "LeftArm", "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm", "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand", "right_wrist": "RightHand",
}
_SOMA30_IDX = {name: idx for idx, name in enumerate(SOMA30_NAMES)}
_SMPLX22_IDX = {name: idx for idx, name in enumerate(SMPLX22_NAMES)}
SMPLX22_TO_SOMA30 = [_SOMA30_IDX[SMPLX_TO_SOMA_NAME[name]] for name in SMPLX22_NAMES]
SOMA30_PARENTS = np.array([
    -1 if parent is None else _SOMA30_IDX[parent] for parent in SOMA30_PARENT_NAMES
], dtype=np.int64)
KIMODO_ASSETS = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo" / "kimodo" / "assets" / "skeletons"


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rot6d_to_rotmat_row_major(rot6d: torch.Tensor) -> torch.Tensor:
    x = rot6d.reshape(*rot6d.shape[:-1], 3, 2)
    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - torch.einsum("...i,...i->...", b1, a2).unsqueeze(-1) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def differentiable_fk(
    local_rotmat: torch.Tensor,
    translation: torch.Tensor,
    bone_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_rot_list: list[torch.Tensor] = [None] * 22  # type: ignore[list-item]
    world_pos_list: list[torch.Tensor] = [None] * 22  # type: ignore[list-item]
    for j, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            world_rot_list[j] = local_rotmat[..., j, :, :]
            world_pos_list[j] = translation + bone_offsets[j]
        else:
            world_rot_list[j] = world_rot_list[parent] @ local_rotmat[..., j, :, :]
            offset_rotated = (world_rot_list[parent] @ bone_offsets[j].unsqueeze(-1)).squeeze(-1)
            world_pos_list[j] = world_pos_list[parent] + offset_rotated
    return torch.stack(world_pos_list, dim=-2), torch.stack(world_rot_list, dim=-3)


@lru_cache(maxsize=None)
def _neutral_joints(skeleton_name: str) -> torch.Tensor:
    return torch.load(KIMODO_ASSETS / skeleton_name / "joints.p", map_location="cpu").squeeze().float()


def _smpl22_bone_offsets() -> np.ndarray:
    neutral_np = _as_numpy(_neutral_joints("smplx22")).astype(np.float32)
    offsets = np.zeros((len(SMPL22_PARENTS), 3), dtype=np.float32)
    for j, p in enumerate(SMPL22_PARENTS):
        if p < 0:
            continue
        offsets[j] = neutral_np[j] - neutral_np[p]
    return offsets


def _soma30_offsets() -> torch.Tensor:
    neutral = _neutral_joints("somaskel30")
    offsets = torch.zeros((len(SOMA30_PARENTS), 3), dtype=torch.float32)
    for j, p in enumerate(SOMA30_PARENTS.tolist()):
        if p < 0:
            continue
        offsets[j] = neutral[j] - neutral[p]
    return offsets


def _slerp_rot_matrices(r1: torch.Tensor, r2: torch.Tensor, t: float) -> torch.Tensor:
    r_delta = torch.einsum("...ij,...ik->...jk", r1, r2)
    tr = r_delta[..., 0, 0] + r_delta[..., 1, 1] + r_delta[..., 2, 2]
    cos_angle = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    small = angle.abs() < 1e-6
    sin_angle = torch.sin(angle).clamp(min=1e-8)
    axis = torch.stack([
        r_delta[..., 2, 1] - r_delta[..., 1, 2],
        r_delta[..., 0, 2] - r_delta[..., 2, 0],
        r_delta[..., 1, 0] - r_delta[..., 0, 1],
    ], dim=-1) / (2.0 * sin_angle.unsqueeze(-1))
    axis = torch.nn.functional.normalize(axis, dim=-1)
    scaled_angle = angle * t
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    k = k.reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=r1.device, dtype=r1.dtype).expand_as(k)
    out = eye + torch.sin(scaled_angle)[..., None, None] * k
    out = out + (1.0 - torch.cos(scaled_angle))[..., None, None] * (k @ k)
    out = torch.where(small[..., None, None], eye, out)
    return r1 @ out


def _soma_fk_from_global_rots(
    global_rots: torch.Tensor,
    root_pos: torch.Tensor,
    soma_offsets: torch.Tensor,
) -> torch.Tensor:
    positions = []
    for j, p in enumerate(SOMA30_PARENTS.tolist()):
        if p < 0:
            positions.append(root_pos + soma_offsets[j].to(root_pos.device))
        else:
            offset = soma_offsets[j].to(root_pos.device, root_pos.dtype)
            positions.append(positions[p] + (global_rots[:, p] @ offset[:, None]).squeeze(-1))
    return torch.stack(positions, dim=1)


def _safe_normalize_np(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps), n[..., 0] > eps


def _rotation_between_np(src: np.ndarray, dst: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src = src / max(float(np.linalg.norm(src)), eps)
    dst = dst / max(float(np.linalg.norm(dst)), eps)
    cross = np.cross(src, dst)
    sin = float(np.linalg.norm(cross))
    cos = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if sin < eps:
        if cos > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if np.linalg.norm(axis) < eps:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis = axis / max(float(np.linalg.norm(axis)), eps)
        return R.from_rotvec(np.pi * axis).as_matrix()
    axis = cross / sin
    return R.from_rotvec(np.arctan2(sin, cos) * axis).as_matrix()


def _scaled_rotation_np(rot: np.ndarray, alpha: float) -> np.ndarray:
    return R.from_rotvec(R.from_matrix(rot).as_rotvec() * float(alpha)).as_matrix()


def _neutral_bone_direction_offsets() -> dict[int, np.ndarray]:
    smpl_neutral = _as_numpy(_neutral_joints("smplx22")).astype(np.float64)
    soma_neutral = _as_numpy(_neutral_joints("somaskel30")).astype(np.float64)
    offsets: dict[int, np.ndarray] = {}
    for smpl_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        smpl_parent = int(SMPL22_PARENTS[smpl_idx])
        soma_parent = int(SOMA30_PARENTS[soma_idx])
        if smpl_parent < 0 or soma_parent < 0:
            offsets[soma_idx] = np.eye(3, dtype=np.float64)
            continue
        offsets[soma_idx] = _rotation_between_np(
            smpl_neutral[smpl_idx] - smpl_neutral[smpl_parent],
            soma_neutral[soma_idx] - soma_neutral[soma_parent],
        )
    return offsets


def _global_to_local_np(global_rots: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local = np.zeros_like(global_rots, dtype=np.float32)
    for j, p in enumerate(parents.tolist()):
        if p < 0:
            local[:, j] = global_rots[:, j]
        else:
            local[:, j] = np.einsum("tki,tkl->til", global_rots[:, p], global_rots[:, j])
    return local


def _estimate_soma30_local_rotations(
    target_joints: np.ndarray,
    parent_ref_weight: float = 0.15,
    reference_global_rots: np.ndarray | None = None,
    twist_ref_weight: float = 0.01,
) -> np.ndarray:
    """Estimate SOMA30 local rotations from target joint positions.

    Directly copying SMPL global rotations into SOMA collapses shoulders for
    motions whose SMPL clavicle/shoulder rest directions differ from SOMA.
    This position-aware initializer aligns SOMA rest bones to the mapped SMPL
    target bones while preserving SOMA's own bone lengths in FK.  The optional
    reference rotations are used only as a twist prior; otherwise one-child
    joints and leaves have under-constrained axial rotation, which makes the
    SOMA mesh look collapsed even when joint positions are numerically close.
    """
    target = np.asarray(target_joints, dtype=np.float64)
    ref_global = (
        np.asarray(reference_global_rots, dtype=np.float64)
        if reference_global_rots is not None
        else None
    )
    neutral = _as_numpy(_neutral_joints("somaskel30")).astype(np.float64)
    parents = SOMA30_PARENTS.astype(np.int64)
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for j, p in enumerate(parents.tolist()):
        if p >= 0:
            children[p].append(j)

    mapped_soma = set(SMPLX22_TO_SOMA30)
    local = np.tile(np.eye(3, dtype=np.float64), (len(target), len(parents), 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target), len(parents), 1, 1))
    basis = np.eye(3, dtype=np.float64)
    for t, joints in enumerate(target):
        for j, child_ids in enumerate(children):
            parent = int(parents[j])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            ref_local = None
            if ref_global is not None:
                ref_local = parent_global.T @ ref_global[t, j]
            child_rest_vecs = []
            child_target_vecs = []
            for c in child_ids:
                child_rest_vecs.append(neutral[c] - neutral[j])
                child_target_vecs.append(joints[c] - joints[j])
            child_rest = np.stack(child_rest_vecs, axis=0) if child_rest_vecs else np.zeros((0, 3))
            child_dst = np.stack(child_target_vecs, axis=0) if child_target_vecs else np.zeros((0, 3))
            child_rest_unit, child_rest_valid = _safe_normalize_np(child_rest)
            child_dst_unit, child_dst_valid = _safe_normalize_np(child_dst)
            child_valid = child_rest_valid & child_dst_valid

            if ref_local is not None and int(np.sum(child_valid)) == 0:
                # Leaf joints are invisible to a pure position fit but are
                # still used by SOMA LBS and by SOMA->SMPL rotation mapping.
                rot_local = ref_local
                local[t, j] = rot_local
                global_r[t, j] = parent_global @ rot_local
                continue

            if ref_local is not None and int(np.sum(child_valid)) == 1:
                # Exact one-bone direction fit with reference twist preserved.
                # This is crucial for arms, legs, feet, and neck chains: Kabsch
                # with a single vector chooses an arbitrary axial twist, while
                # soft reference axes can move the child endpoint.  Swing the
                # reference orientation onto the target bone instead.
                k = int(np.where(child_valid)[0][0])
                target_local = parent_global.T @ child_dst_unit[k]
                reference_dir = ref_local @ child_rest_unit[k]
                rot_local = _rotation_between_np(reference_dir, target_local) @ ref_local
                local[t, j] = rot_local
                global_r[t, j] = parent_global @ rot_local
                continue

            rest_vecs = []
            target_vecs = []
            weights = []
            for local_idx, c in enumerate(child_ids):
                rest_vecs.append(neutral[c] - neutral[j])
                target_vecs.append(joints[c] - joints[j])
                weights.append(1.0 if (j in mapped_soma or c in mapped_soma) else 0.1)
            if parent >= 0:
                rest_vecs.append(neutral[parent] - neutral[j])
                target_vecs.append(joints[parent] - joints[j])
                weights.append(parent_ref_weight)
            if ref_local is not None:
                ref_weight = twist_ref_weight
                for axis in basis:
                    rest_vecs.append(axis)
                    target_vecs.append(ref_local @ axis)
                    weights.append(ref_weight)
            if not rest_vecs:
                rot_local = np.eye(3)
            else:
                rest = np.stack(rest_vecs, axis=0)
                dst = np.stack(target_vecs, axis=0)
                rest_unit, rest_valid = _safe_normalize_np(rest)
                dst_unit, dst_valid = _safe_normalize_np(dst)
                valid = rest_valid & dst_valid & (np.asarray(weights) > 0)
                if not np.any(valid):
                    rot_local = np.eye(3)
                else:
                    dst_local = (parent_global.T @ dst_unit[valid].T).T
                    try:
                        rot_local = R.align_vectors(
                            dst_local,
                            rest_unit[valid],
                            weights=np.asarray(weights, dtype=np.float64)[valid],
                        )[0].as_matrix()
                    except Exception:
                        rot_local = np.eye(3)
            local[t, j] = rot_local
            global_r[t, j] = parent_global @ rot_local
    return local.astype(np.float32)


def _complete_soma30_global_rots(soma_global_rots: torch.Tensor) -> torch.Tensor:
    soma_global_rots[:, 5] = _slerp_rot_matrices(soma_global_rots[:, 4], soma_global_rots[:, 6], 0.5)
    soma_global_rots[:, 7] = soma_global_rots[:, 6]
    soma_global_rots[:, 8] = soma_global_rots[:, 6]
    soma_global_rots[:, 9] = soma_global_rots[:, 6]
    soma_global_rots[:, 14] = soma_global_rots[:, 13]
    soma_global_rots[:, 15] = soma_global_rots[:, 13]
    soma_global_rots[:, 20] = soma_global_rots[:, 19]
    soma_global_rots[:, 21] = soma_global_rots[:, 19]
    return soma_global_rots


def _soma30_targets_from_smpl22(
    smplx_world_pos: torch.Tensor,
    direct_soma_pos: torch.Tensor,
) -> np.ndarray:
    target = direct_soma_pos.detach().cpu().numpy().astype(np.float32)
    smpl = smplx_world_pos.detach().cpu().numpy().astype(np.float32)
    for smplx_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        target[:, soma_idx] = smpl[:, smplx_idx]
    target[:, _SOMA30_IDX["Neck2"]] = 0.5 * (
        target[:, _SOMA30_IDX["Neck1"]] + target[:, _SOMA30_IDX["Head"]]
    )
    return target


def _soma30_fk_from_local(
    local_rots: np.ndarray,
    root_pos: torch.Tensor,
    soma_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_t = torch.from_numpy(np.asarray(local_rots, dtype=np.float32)).to(root_pos.device)
    global_rots = []
    positions = []
    for j, p in enumerate(SOMA30_PARENTS.tolist()):
        if p < 0:
            global_rots.append(local_t[:, j])
            positions.append(root_pos)
            continue
        global_rots.append(global_rots[p] @ local_t[:, j])
        offset = soma_offsets[j].to(root_pos.device, root_pos.dtype)
        positions.append(positions[p] + (global_rots[p] @ offset[:, None]).squeeze(-1))
    return torch.stack(global_rots, dim=1), torch.stack(positions, dim=1)


def _smpl22_to_soma30_retarget_position_aware(
    motion_135: np.ndarray,
    smpl_bone_offsets: np.ndarray,
    soma_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32))
    offsets = torch.from_numpy(np.asarray(smpl_bone_offsets, dtype=np.float32))
    t = motion.shape[0]
    translation = motion[:, :3]
    local_rotmat = rot6d_to_rotmat_row_major(motion[:, 3:135].reshape(t, 22, 6))
    smplx_world_pos, _smplx_global_rots = differentiable_fk(local_rotmat, translation, offsets)

    direct_rots, direct_soma = _smpl22_to_soma30_retarget_minimal(
        motion_135, smpl_bone_offsets, soma_offsets)
    target_soma = _soma30_targets_from_smpl22(smplx_world_pos, direct_soma)
    soma_local = _estimate_soma30_local_rotations(
        target_soma,
        reference_global_rots=_as_numpy(direct_rots),
    )

    soma_root_pos = torch.from_numpy(target_soma[:, 0].astype(np.float32))
    soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)

    smplx_foot_indices = [
        _SMPLX22_IDX["left_foot"],
        _SMPLX22_IDX["right_foot"],
        _SMPLX22_IDX["left_ankle"],
        _SMPLX22_IDX["right_ankle"],
    ]
    soma_foot_indices = [
        _SOMA30_IDX["LeftToeBase"],
        _SOMA30_IDX["RightToeBase"],
        _SOMA30_IDX["LeftFoot"],
        _SOMA30_IDX["RightFoot"],
    ]
    smpl_foot_min_y = smplx_world_pos[:, smplx_foot_indices, 1].min(dim=1).values
    soma_foot_min_y = soma_joints[:, soma_foot_indices, 1].min(dim=1).values
    y_delta = soma_foot_min_y - smpl_foot_min_y
    if torch.max(torch.abs(y_delta)) > 1e-4:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 1] -= y_delta
        soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)

    root_delta_xz = translation[:, [0, 2]] - soma_joints[:, 0, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 0] += root_delta_xz[:, 0]
        soma_root_pos[:, 2] += root_delta_xz[:, 1]
        soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)
    return soma_global_rots, soma_joints


def _smpl22_to_soma30_retarget_shoulder_offset(
    motion_135: np.ndarray,
    smpl_bone_offsets: np.ndarray,
    soma_offsets: torch.Tensor,
    shoulder_offset_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror KIMODO direct retarget, with shoulder-only rest-frame correction.

    The direct SMPL->SOMA path preserves torso/head rotations well but can make
    SOMA shoulder skinning collapse because SMPL clavicle and SOMA shoulder rest
    directions differ substantially.  Applying a small rest-direction offset
    only to SOMA LeftShoulder/RightShoulder keeps the rest of the body on the
    official direct path while stabilizing the shoulder mesh.
    """
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32))
    offsets = torch.from_numpy(np.asarray(smpl_bone_offsets, dtype=np.float32))
    t = motion.shape[0]
    translation = motion[:, :3]
    local_rotmat = rot6d_to_rotmat_row_major(motion[:, 3:135].reshape(t, 22, 6))
    smplx_world_pos, smplx_global_rots = differentiable_fk(local_rotmat, translation, offsets)

    eye = torch.eye(3, dtype=local_rotmat.dtype, device=local_rotmat.device)
    soma_global_rots = eye[None, None].expand(t, 30, 3, 3).clone()
    for smplx_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        soma_global_rots[:, soma_idx] = smplx_global_rots[:, smplx_idx]

    rest_offsets = _neutral_bone_direction_offsets()
    for name in ("LeftShoulder", "RightShoulder"):
        soma_idx = _SOMA30_IDX[name]
        offset = _scaled_rotation_np(rest_offsets[soma_idx], shoulder_offset_alpha).astype(np.float32)
        soma_global_rots[:, soma_idx] = soma_global_rots[:, soma_idx] @ torch.from_numpy(offset).to(soma_global_rots)

    soma_global_rots = _complete_soma30_global_rots(soma_global_rots)
    soma_local = _global_to_local_np(_as_numpy(soma_global_rots), SOMA30_PARENTS)
    soma_root_pos = translation.clone()
    soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)

    smplx_foot_indices = [
        _SMPLX22_IDX["left_foot"],
        _SMPLX22_IDX["right_foot"],
        _SMPLX22_IDX["left_ankle"],
        _SMPLX22_IDX["right_ankle"],
    ]
    soma_foot_indices = [
        _SOMA30_IDX["LeftToeBase"],
        _SOMA30_IDX["RightToeBase"],
        _SOMA30_IDX["LeftFoot"],
        _SOMA30_IDX["RightFoot"],
    ]
    smpl_foot_min_y = smplx_world_pos[:, smplx_foot_indices, 1].min(dim=1).values
    soma_foot_min_y = soma_joints[:, soma_foot_indices, 1].min(dim=1).values
    y_delta = soma_foot_min_y - smpl_foot_min_y
    if torch.max(torch.abs(y_delta)) > 1e-4:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 1] -= y_delta
        soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)

    root_delta_xz = translation[:, [0, 2]] - soma_joints[:, 0, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 0] += root_delta_xz[:, 0]
        soma_root_pos[:, 2] += root_delta_xz[:, 1]
        soma_global_rots, soma_joints = _soma30_fk_from_local(soma_local, soma_root_pos, soma_offsets)
    return soma_global_rots, soma_joints


def _soma30_to_smpl22_motion_rotation(
    soma_global_rots: torch.Tensor,
    source_motion_135: np.ndarray,
    smpl_bone_offsets: np.ndarray,
    height_mode: str = "source_root",
) -> dict[str, np.ndarray]:
    source = torch.from_numpy(np.asarray(source_motion_135, dtype=np.float32))
    t = source.shape[0]
    smpl_global = torch.eye(3, dtype=soma_global_rots.dtype, device=soma_global_rots.device)
    smpl_global = smpl_global[None, None].expand(t, 22, 3, 3).clone()
    for smpl_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        smpl_global[:, smpl_idx] = soma_global_rots[:, soma_idx]

    local = torch.empty_like(smpl_global)
    for j, p in enumerate(SMPL22_PARENTS):
        if p < 0:
            local[:, j] = smpl_global[:, j]
        else:
            local[:, j] = torch.einsum("tki,tkl->til", smpl_global[:, p], smpl_global[:, j])

    offsets = torch.from_numpy(np.asarray(smpl_bone_offsets, dtype=np.float32)).to(local.device)
    source_local = rot6d_to_rotmat_row_major(source[:, 3:135].reshape(t, 22, 6)).to(local.device)
    source_pos, _ = differentiable_fk(source_local, source[:, :3].to(local.device), offsets)
    transl = source[:, :3].to(local.device).clone()
    fitted, _ = differentiable_fk(local, transl, offsets)

    if height_mode == "foot_floor":
        foot_indices = [
            _SMPLX22_IDX["left_ankle"],
            _SMPLX22_IDX["right_ankle"],
            _SMPLX22_IDX["left_foot"],
            _SMPLX22_IDX["right_foot"],
        ]
        y_delta = fitted[:, foot_indices, 1].min(dim=1).values - source_pos[:, foot_indices, 1].min(dim=1).values
        if torch.max(torch.abs(y_delta)) > 1e-5:
            transl[:, 1] -= y_delta
            fitted, _ = differentiable_fk(local, transl, offsets)
    elif height_mode != "source_root":
        raise ValueError(f"unknown height_mode: {height_mode}")

    rot6d = torch.from_numpy(
        matrix_to_rot6d_rowmajor(_as_numpy(local)).reshape(t, 22 * 6)
    ).to(transl.device)
    motion_135 = torch.cat(
        [transl, rot6d],
        dim=-1,
    ).detach().cpu().numpy().astype(np.float32)
    aa = R.from_matrix(_as_numpy(local).reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    aa = aa.reshape(t, 22, 3)
    return {
        "motion_135": motion_135,
        "transl": transl.detach().cpu().numpy().astype(np.float32),
        "global_orient": aa[:, 0].astype(np.float32),
        "body_pose": aa[:, 1:].reshape(t, 63).astype(np.float32),
        "fitted_joints": fitted.detach().cpu().numpy().astype(np.float32),
    }


def _smpl22_to_soma30_retarget_minimal(
    motion_135: np.ndarray,
    smpl_bone_offsets: np.ndarray,
    soma_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32))
    offsets = torch.from_numpy(np.asarray(smpl_bone_offsets, dtype=np.float32))
    t = motion.shape[0]
    translation = motion[:, :3]
    local_rotmat = rot6d_to_rotmat_row_major(motion[:, 3:135].reshape(t, 22, 6))
    smplx_world_pos, smplx_global_rots = differentiable_fk(local_rotmat, translation, offsets)

    eye = torch.eye(3, dtype=local_rotmat.dtype, device=local_rotmat.device)
    soma_global_rots = eye[None, None].expand(t, 30, 3, 3).clone()
    for smplx_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        soma_global_rots[:, soma_idx] = smplx_global_rots[:, smplx_idx]

    soma_global_rots = _complete_soma30_global_rots(soma_global_rots)

    smpl_neutral = _neutral_joints("smplx22")
    soma_neutral = _neutral_joints("somaskel30")
    smplx_centered = smpl_neutral - smpl_neutral[0]
    soma_centered = soma_neutral - soma_neutral[0]

    smplx_foot_indices = [
        _SMPLX22_IDX["left_foot"],
        _SMPLX22_IDX["right_foot"],
        _SMPLX22_IDX["left_ankle"],
        _SMPLX22_IDX["right_ankle"],
    ]
    soma_foot_indices = [
        _SOMA30_IDX["LeftToeBase"],
        _SOMA30_IDX["RightToeBase"],
        _SOMA30_IDX["LeftFoot"],
        _SOMA30_IDX["RightFoot"],
    ]
    foot_offset_y = (
        soma_centered[soma_foot_indices, 1].min()
        - smplx_centered[smplx_foot_indices, 1].min()
    )

    soma_root_pos = translation.clone()
    soma_root_pos[:, 1] -= foot_offset_y.to(soma_root_pos.dtype)
    soma_joints = _soma_fk_from_global_rots(soma_global_rots, soma_root_pos, soma_offsets)

    smpl_foot_min_y = smplx_world_pos[:, smplx_foot_indices, 1].min(dim=1).values
    soma_foot_min_y = soma_joints[:, soma_foot_indices, 1].min(dim=1).values
    y_delta = soma_foot_min_y - smpl_foot_min_y
    if torch.max(torch.abs(y_delta)) > 1e-4:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 1] -= y_delta
        soma_joints = _soma_fk_from_global_rots(soma_global_rots, soma_root_pos, soma_offsets)

    root_delta_xz = translation[:, [0, 2]] - soma_joints[:, 0, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 0] += root_delta_xz[:, 0]
        soma_root_pos[:, 2] += root_delta_xz[:, 1]
        soma_joints = _soma_fk_from_global_rots(soma_global_rots, soma_root_pos, soma_offsets)

    return soma_global_rots, soma_joints


def _read_first_full_caption(txt_path: Path) -> str:
    if not txt_path.exists():
        return ""
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        caption = parts[0].strip()
        try:
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
        except ValueError:
            f_tag = t_tag = 0.0
        if caption and f_tag == 0.0 and t_tag == 0.0:
            return caption
    return ""


def _load_eval_ids(data_root: Path, min_len: int, max_len_exclusive: int, limit: int | None) -> list[str]:
    split = data_root / "split" / "test.txt"
    ids = [line.strip() for line in split.read_text().splitlines() if line.strip()]
    kept: list[str] = []
    for sid in ids:
        motion_file = data_root / "motion_data" / f"{sid}.npy"
        text_file = data_root / "texts" / f"{sid}.txt"
        if not motion_file.exists() or not text_file.exists():
            continue
        length = int(np.load(str(motion_file), mmap_mode="r").shape[0])
        if length < min_len or length >= max_len_exclusive:
            continue
        if not _read_first_full_caption(text_file):
            continue
        kept.append(sid)
        if limit is not None and len(kept) >= limit:
            break
    return kept


def _load_ids_from_dir(ids_dir: Path, limit: int | None) -> list[str]:
    files = sorted(ids_dir.glob("*.npz")) or sorted(ids_dir.glob("*.npy"))
    ids = [path.stem for path in files]
    if limit is not None:
        ids = ids[:limit]
    return ids


def _save_roundtrip_item(
    sid: str,
    data_root: Path,
    out_dir: Path,
    source_motion_dir: Path | None,
    smpl_model,
    smpl_rest_joints: np.ndarray,
    smpl_parents: np.ndarray,
    smpl_bone_offsets: np.ndarray,
    soma_offsets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    floor_align: bool,
    refine_iters: int,
    refine_lr: float,
    orientation_mode: str,
    parent_ref_weight: float,
    pose_l2_weight: float,
    angle_prior_weight: float,
    soma_mode: str,
    shoulder_offset_alpha: float,
    smpl_height_mode: str,
) -> dict[str, float | int | str]:
    if source_motion_dir is not None:
        src_npz = source_motion_dir / f"{sid}.npz"
        with np.load(src_npz, allow_pickle=True) as src:
            key = "source_motion_135" if "source_motion_135" in src.files else "motion_135"
            source_m135 = np.asarray(src[key], dtype=np.float32)
    else:
        m272 = np.load(str(data_root / "motion_data" / f"{sid}.npy")).astype(np.float32)
        source_m135 = humanml272_to_motion135(m272)

    if soma_mode == "position_aware":
        soma_rots, soma_pos = _smpl22_to_soma30_retarget_position_aware(
            source_m135, smpl_bone_offsets, soma_offsets)
    elif soma_mode == "shoulder_offset":
        soma_rots, soma_pos = _smpl22_to_soma30_retarget_shoulder_offset(
            source_m135, smpl_bone_offsets, soma_offsets, shoulder_offset_alpha)
    elif soma_mode == "direct":
        soma_rots, soma_pos = _smpl22_to_soma30_retarget_minimal(
            source_m135, smpl_bone_offsets, soma_offsets)
    else:
        raise ValueError(f"unknown soma_mode: {soma_mode}")
    soma_pos_np = _as_numpy(soma_pos).astype(np.float32)
    target_smpl22 = soma_pos_np[:, SMPLX22_TO_SOMA30, :]

    if floor_align:
        target_smpl22 = target_smpl22.copy()
        target_smpl22[..., 1] -= float(target_smpl22[..., 1].min())
    ret = _soma30_to_smpl22_motion_rotation(
        soma_rots,
        source_m135,
        smpl_bone_offsets,
        height_mode=smpl_height_mode,
    )
    ret["target_joints"] = target_smpl22.astype(np.float32)
    ret["fit_mpjpe_mm"] = (
        np.linalg.norm(ret["fitted_joints"] - target_smpl22, axis=-1).mean(axis=1) * 1000.0
    ).astype(np.float32)

    caption = _read_first_full_caption(data_root / "texts" / f"{sid}.txt")
    np.savez_compressed(
        out_dir / f"{sid}.npz",
        **ret,
        source_motion_135=source_m135.astype(np.float32),
        target_joints_soma30=soma_pos_np,
        target_global_rots_soma30=_as_numpy(soma_rots).astype(np.float32),
        caption=np.array(caption, dtype=object),
        source_id=np.array(sid, dtype=object),
        source_fps=np.array(30.0, dtype=np.float32),
        target_fps=np.array(30.0, dtype=np.float32),
        refine_iters=np.array(refine_iters, dtype=np.int32),
        soma_mode=np.array(soma_mode, dtype=object),
        shoulder_offset_alpha=np.array(shoulder_offset_alpha, dtype=np.float32),
        smpl_height_mode=np.array(smpl_height_mode, dtype=object),
    )
    mpjpe = np.asarray(ret["fit_mpjpe_mm"], dtype=np.float32)
    return {
        "sid": sid,
        "frames": int(len(source_m135)),
        "mpjpe_mm_mean": float(mpjpe.mean()),
        "mpjpe_mm_p95": float(np.percentile(mpjpe, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--out-dir", default="outputs/evaluation/humanml3d_gt_soma_smpl_roundtrip/motion135")
    parser.add_argument(
        "--source-motion-dir",
        default=None,
        help="Optional NPZ directory containing source_motion_135 or motion_135; avoids reconverting HumanML3D-272.",
    )
    parser.add_argument("--ids-dir", default=None, help="Optional directory whose npz/npy stems define the eval IDs.")
    parser.add_argument("--model-dir", default="ref_repo/MDM/body_models")
    parser.add_argument("--min-len", type=int, default=60)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--floor-align", action="store_true", default=True)
    parser.add_argument("--no-floor-align", dest="floor_align", action="store_false")
    parser.add_argument("--refine-iters", type=int, default=0)
    parser.add_argument("--refine-lr", type=float, default=2e-2)
    parser.add_argument("--pose-l2-weight", type=float, default=0.0)
    parser.add_argument("--angle-prior-weight", type=float, default=0.0)
    parser.add_argument("--orientation-mode", choices=["bone", "parent_frame"], default="bone")
    parser.add_argument("--parent-ref-weight", type=float, default=0.25)
    parser.add_argument(
        "--soma-mode",
        choices=["direct", "position_aware", "shoulder_offset"],
        default="shoulder_offset",
        help="SMPL->SOMA retarget mode. shoulder_offset preserves direct rotations except a shoulder rest-frame correction.",
    )
    parser.add_argument("--shoulder-offset-alpha", type=float, default=0.75)
    parser.add_argument(
        "--smpl-height-mode",
        choices=["source_root", "foot_floor"],
        default="source_root",
        help="SOMA->SMPL height handling. source_root preserves original SMPL root height; foot_floor matches feet.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_motion_dir = Path(args.source_motion_dir) if args.source_motion_dir else None
    device = torch.device(args.device)
    limit = args.limit if args.limit > 0 else None

    if args.ids_dir:
        ids = _load_ids_from_dir(Path(args.ids_dir), limit)
    else:
        ids = _load_eval_ids(data_root, args.min_len, args.max_len, limit)
    print(f"[ids] {len(ids)} clips", flush=True)
    smpl_model, rest_joints, parents = load_smpl_rest(Path(args.model_dir), device)
    smpl_bone_offsets = _smpl22_bone_offsets()
    soma_offsets = _soma30_offsets()
    print(
        f"[setup] ids={len(ids)} data={data_root} out={out_dir} device={device} "
        f"refine_iters={args.refine_iters}",
        flush=True,
    )

    done = skipped = failed = 0
    per_item = []
    for idx, sid in enumerate(ids, 1):
        dst = out_dir / f"{sid}.npz"
        if args.skip_existing and dst.exists():
            skipped += 1
            continue
        try:
            item = _save_roundtrip_item(
                sid,
                data_root,
                out_dir,
                source_motion_dir,
                smpl_model,
                rest_joints,
                parents,
                smpl_bone_offsets,
                soma_offsets,
                device,
                args.batch_size,
                args.floor_align,
                args.refine_iters,
                args.refine_lr,
                args.orientation_mode,
                args.parent_ref_weight,
                args.pose_l2_weight,
                args.angle_prior_weight,
                args.soma_mode,
                args.shoulder_offset_alpha,
                args.smpl_height_mode,
            )
            per_item.append(item)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
        if idx % 50 == 0 or idx == len(ids):
            print(f"[progress] {idx}/{len(ids)} done={done} skipped={skipped} failed={failed}", flush=True)

    summary = _summarize_outputs(out_dir)
    summary.update({
        "requested_ids": len(ids),
        "newly_done": done,
        "skipped_existing": skipped,
        "new_failures": failed,
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "source_motion_dir": str(source_motion_dir) if source_motion_dir is not None else None,
        "refine_iters": args.refine_iters,
        "soma_mode": args.soma_mode,
        "shoulder_offset_alpha": args.shoulder_offset_alpha,
        "smpl_height_mode": args.smpl_height_mode,
        "items_new": per_item[:100],
    })
    (out_dir / "_roundtrip_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[done] " + json.dumps({
        "count": summary["count"],
        "failed": summary["failed"],
        "mean_mpjpe_mm": summary["mean_mpjpe_mm"],
        "median_mpjpe_mm": summary["median_mpjpe_mm"],
        "p95_frame_mpjpe_mm_mean": summary["p95_frame_mpjpe_mm_mean"],
    }), flush=True)


if __name__ == "__main__":
    main()
