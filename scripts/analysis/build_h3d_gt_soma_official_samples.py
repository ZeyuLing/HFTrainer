#!/usr/bin/env python3
"""Build clean HumanML3D GT -> SOMA samples using the KIMODO eval path.

This script intentionally stops at SOMA.  It does not run SOMA->SMPL fitting
and it does not optimize SOMA joints toward SMPL joints.  The goal is to audit
whether the deterministic KIMODO retarget and SOMA mesh path are healthy before
using any downstream SMPL evaluator.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "scripts/eval"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from h3d_272_to_135 import humanml272_to_motion135  # noqa: E402
from hftrainer.pipelines.motion.differentiable_fk import (  # noqa: E402
    differentiable_fk,
    rot6d_to_rotmat_row_major,
)

SMPLX22_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
SOMA30_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd", "RightShoulder", "RightArm",
    "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin",
    "RightFoot", "RightToeBase",
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
_SMPL22_IDX = {name: idx for idx, name in enumerate(SMPLX22_NAMES)}
SMPLX22_TO_SOMA30 = [_SOMA30_IDX[SMPLX_TO_SOMA_NAME[name]] for name in SMPLX22_NAMES]
SOMA77_TO_SMPL22 = [
    0, 67, 72, 1, 68, 73, 2, 69, 74, 3, 70, 75,
    4, 11, 39, 6, 12, 40, 13, 41, 14, 42,
]


def _bootstrap_kimodo_skeleton() -> None:
    """Load only kimodo.skeleton on Python 3.9 without importing the full package."""
    if "kimodo.skeleton.definitions" in sys.modules:
        return
    kimodo_root = PROJECT_ROOT / "ref_repo" / "KIMODO" / "kimodo"
    if "kimodo" not in sys.modules:
        pkg = types.ModuleType("kimodo")
        pkg.__path__ = [str(kimodo_root / "kimodo")]
        sys.modules["kimodo"] = pkg
    if "kimodo.assets" not in sys.modules:
        assets = types.ModuleType("kimodo.assets")

        def skeleton_asset_path(name: str) -> Path:
            return kimodo_root / "kimodo" / "assets" / "skeletons" / name

        assets.skeleton_asset_path = skeleton_asset_path
        assets.SKELETONS_ROOT = str(kimodo_root / "kimodo" / "assets" / "skeletons")
        sys.modules["kimodo.assets"] = assets
    if "kimodo.skeleton" not in sys.modules:
        skel_pkg = types.ModuleType("kimodo.skeleton")
        skel_pkg.__path__ = [str(kimodo_root / "kimodo" / "skeleton")]
        sys.modules["kimodo.skeleton"] = skel_pkg

    def load(name: str, relpath: str):
        spec = importlib.util.spec_from_file_location(name, kimodo_root / relpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    load("kimodo.skeleton.kinematics", "kimodo/skeleton/kinematics.py")
    load("kimodo.skeleton.transforms", "kimodo/skeleton/transforms.py")
    load("kimodo.skeleton.base", "kimodo/skeleton/base.py")
    load("kimodo.skeleton.definitions", "kimodo/skeleton/definitions.py")


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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


def _load_skeletons():
    _bootstrap_kimodo_skeleton()
    from kimodo.skeleton.definitions import SMPLXSkeleton22, SOMASkeleton30

    return SMPLXSkeleton22(), SOMASkeleton30()


def _global_to_local(global_rots: torch.Tensor, skeleton) -> torch.Tensor:
    _bootstrap_kimodo_skeleton()
    from kimodo.skeleton.transforms import global_rots_to_local_rots

    return global_rots_to_local_rots(global_rots, skeleton)


def _official_smpl22_to_soma(
    motion_135: np.ndarray,
    bone_offsets: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mirror scripts/kimodo/run_kimodo_all_tasks.py::smpl22_to_soma30_retarget."""
    smplx22, soma30 = _load_skeletons()
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32))
    offsets = torch.from_numpy(np.asarray(bone_offsets, dtype=np.float32))
    t = int(motion.shape[0])
    translation = motion[:, :3]
    local_rotmat = rot6d_to_rotmat_row_major(motion[:, 3:135].reshape(t, 22, 6))
    smpl22_pos, smpl22_global_rots = differentiable_fk(local_rotmat, translation, offsets)

    soma30_global = torch.eye(3, dtype=local_rotmat.dtype, device=local_rotmat.device)
    soma30_global = soma30_global[None, None].expand(t, 30, 3, 3).clone()
    for smpl_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        soma30_global[:, soma_idx] = smpl22_global_rots[:, smpl_idx]

    soma30_global[:, 5] = _slerp_rot_matrices(soma30_global[:, 4], soma30_global[:, 6], 0.5)
    soma30_global[:, 7] = soma30_global[:, 6]
    soma30_global[:, 8] = soma30_global[:, 6]
    soma30_global[:, 9] = soma30_global[:, 6]
    soma30_global[:, 14] = soma30_global[:, 13]
    soma30_global[:, 15] = soma30_global[:, 13]
    soma30_global[:, 20] = soma30_global[:, 19]
    soma30_global[:, 21] = soma30_global[:, 19]

    soma30_local = _global_to_local(soma30_global, soma30)

    smplx_centered = smplx22.neutral_joints - smplx22.neutral_joints[smplx22.root_idx]
    soma_centered = soma30.neutral_joints - soma30.neutral_joints[soma30.root_idx]
    smpl_foot_indices = [
        smplx22.bone_index[n] for n in ["left_foot", "right_foot", "left_ankle", "right_ankle"]
    ]
    soma_foot_indices = [
        soma30.bone_index[n] for n in ["LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot"]
    ]
    foot_offset_y = (
        soma_centered[soma_foot_indices, 1].min()
        - smplx_centered[smpl_foot_indices, 1].min()
    ).item()
    soma_root_pos = translation.clone()
    soma_root_pos[:, 1] -= foot_offset_y
    soma30_global_fk, soma30_pos, _ = soma30.fk(soma30_local, soma_root_pos)

    smpl_foot_min_y = smpl22_pos[:, smpl_foot_indices, 1].min(dim=1).values
    soma_foot_min_y = soma30_pos[:, soma_foot_indices, 1].min(dim=1).values
    y_delta = soma_foot_min_y - smpl_foot_min_y
    if torch.max(torch.abs(y_delta)) > 1e-4:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 1] -= y_delta
        soma30_global_fk, soma30_pos, _ = soma30.fk(soma30_local, soma_root_pos)

    root_delta_xz = translation[:, [0, 2]] - soma30_pos[:, soma30.root_idx, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        soma_root_pos = soma_root_pos.clone()
        soma_root_pos[:, 0] += root_delta_xz[:, 0]
        soma_root_pos[:, 2] += root_delta_xz[:, 1]
        soma30_global_fk, soma30_pos, _ = soma30.fk(soma30_local, soma_root_pos)

    # Recompute local rotations from the final global rotations before 30->77.
    soma30_local_fk = _global_to_local(soma30_global_fk, soma30)
    soma77_local = soma30.to_SOMASkeleton77(soma30_local_fk)
    soma77_global, soma77_pos, _ = soma30.somaskel77.fk(soma77_local, soma30_pos[:, 0])

    return {
        "source_joints_smpl22": _as_numpy(smpl22_pos).astype(np.float32),
        "source_global_rots_smpl22": _as_numpy(smpl22_global_rots).astype(np.float32),
        "soma30_local_rots": _as_numpy(soma30_local_fk).astype(np.float32),
        "soma30_global_rots": _as_numpy(soma30_global_fk).astype(np.float32),
        "soma30_posed_joints": _as_numpy(soma30_pos).astype(np.float32),
        "soma77_local_rots": _as_numpy(soma77_local).astype(np.float32),
        "soma77_global_rots": _as_numpy(soma77_global).astype(np.float32),
        "soma77_posed_joints": _as_numpy(soma77_pos).astype(np.float32),
        "soma77_to_smpl22_joints": _as_numpy(soma77_pos[:, SOMA77_TO_SMPL22]).astype(np.float32),
    }


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


def _neutral_bone_offsets(smplx22, soma30) -> dict[int, np.ndarray]:
    smpl_neutral = _as_numpy(smplx22.neutral_joints).astype(np.float64)
    soma_neutral = _as_numpy(soma30.neutral_joints).astype(np.float64)
    smpl_parents = np.asarray(smplx22.joint_parents, dtype=np.int64)
    soma_parents = np.asarray(soma30.joint_parents, dtype=np.int64)
    offsets: dict[int, np.ndarray] = {}
    for smpl_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        smpl_parent = int(smpl_parents[smpl_idx])
        soma_parent = int(soma_parents[soma_idx])
        if smpl_parent == smpl_idx or soma_parent == soma_idx:
            offsets[soma_idx] = np.eye(3, dtype=np.float32)
            continue
        offsets[soma_idx] = _rotation_between_np(
            smpl_neutral[smpl_idx] - smpl_neutral[smpl_parent],
            soma_neutral[soma_idx] - soma_neutral[soma_parent],
        ).astype(np.float32)
    return offsets


def _soma30_parents(soma30) -> np.ndarray:
    parents = np.asarray(soma30.joint_parents, dtype=np.int64)
    return np.asarray([-1 if p == i else p for i, p in enumerate(parents)], dtype=np.int64)


def _estimate_soma30_local_rotations(
    target_joints: np.ndarray,
    reference_global_rots: np.ndarray,
    soma30,
    parent_ref_weight: float = 0.15,
    twist_ref_weight: float = 0.01,
) -> np.ndarray:
    """Estimate SOMA30 local rotations from target directions.

    This is an audit retargeter for shoulder-collapse debugging.  It keeps SOMA
    bone lengths and produces valid local rotations, but lets each parent orient
    its outgoing rest bones toward the mapped SMPL target directions.
    """
    target = np.asarray(target_joints, dtype=np.float64)
    ref_global = np.asarray(reference_global_rots, dtype=np.float64)
    neutral = _as_numpy(soma30.neutral_joints).astype(np.float64)
    parents = _soma30_parents(soma30)
    children: list[list[int]] = [[] for _ in range(len(parents))]
    for j, p in enumerate(parents.tolist()):
        if p >= 0:
            children[p].append(j)

    mapped_soma = set(SMPLX22_TO_SOMA30)
    local = np.tile(np.eye(3, dtype=np.float64), (len(target), len(parents), 1, 1))
    global_rots = np.tile(np.eye(3, dtype=np.float64), (len(target), len(parents), 1, 1))
    basis = np.eye(3, dtype=np.float64)
    for t, joints in enumerate(target):
        for j, child_ids in enumerate(children):
            parent = int(parents[j])
            parent_global = np.eye(3, dtype=np.float64) if parent < 0 else global_rots[t, parent]
            ref_local = parent_global.T @ ref_global[t, j]

            child_rest_vecs = []
            child_target_vecs = []
            for c in child_ids:
                child_rest_vecs.append(neutral[c] - neutral[j])
                child_target_vecs.append(joints[c] - joints[j])
            child_rest = (
                np.stack(child_rest_vecs, axis=0)
                if child_rest_vecs
                else np.zeros((0, 3), dtype=np.float64)
            )
            child_dst = (
                np.stack(child_target_vecs, axis=0)
                if child_target_vecs
                else np.zeros((0, 3), dtype=np.float64)
            )
            child_rest_unit, child_rest_valid = _safe_normalize_np(child_rest)
            child_dst_unit, child_dst_valid = _safe_normalize_np(child_dst)
            child_valid = child_rest_valid & child_dst_valid

            if int(np.sum(child_valid)) == 0:
                rot_local = ref_local
                local[t, j] = rot_local
                global_rots[t, j] = parent_global @ rot_local
                continue

            if int(np.sum(child_valid)) == 1:
                k = int(np.where(child_valid)[0][0])
                target_local = parent_global.T @ child_dst_unit[k]
                reference_dir = ref_local @ child_rest_unit[k]
                rot_local = _rotation_between_np(reference_dir, target_local) @ ref_local
                local[t, j] = rot_local
                global_rots[t, j] = parent_global @ rot_local
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
            for axis in basis:
                rest_vecs.append(axis)
                target_vecs.append(ref_local @ axis)
                weights.append(twist_ref_weight)

            rest = np.stack(rest_vecs, axis=0)
            dst = np.stack(target_vecs, axis=0)
            rest_unit, rest_valid = _safe_normalize_np(rest)
            dst_unit, dst_valid = _safe_normalize_np(dst)
            weights_arr = np.asarray(weights, dtype=np.float64)
            valid = rest_valid & dst_valid & (weights_arr > 0)
            if not np.any(valid):
                rot_local = ref_local
            else:
                dst_local = (parent_global.T @ dst_unit[valid].T).T
                try:
                    rot_local = R.align_vectors(
                        dst_local,
                        rest_unit[valid],
                        weights=weights_arr[valid],
                    )[0].as_matrix()
                except Exception:
                    rot_local = ref_local
            local[t, j] = rot_local
            global_rots[t, j] = parent_global @ rot_local
    return local.astype(np.float32)


def _target_soma30_from_source(
    source_joints: np.ndarray,
    direct_soma30_pos: np.ndarray,
) -> np.ndarray:
    target = np.asarray(direct_soma30_pos, dtype=np.float32).copy()
    source = np.asarray(source_joints, dtype=np.float32)
    for smpl_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        target[:, soma_idx] = source[:, smpl_idx]
    target[:, _SOMA30_IDX["Neck2"]] = 0.5 * (
        target[:, _SOMA30_IDX["Neck1"]] + target[:, _SOMA30_IDX["Head"]]
    )
    return target


def _position_aware_smpl22_to_soma(
    motion_135: np.ndarray,
    bone_offsets: np.ndarray,
) -> dict[str, np.ndarray]:
    direct = _official_smpl22_to_soma(motion_135, bone_offsets)
    _smplx22, soma30 = _load_skeletons()
    source = direct["source_joints_smpl22"]
    target = _target_soma30_from_source(source, direct["soma30_posed_joints"])
    soma30_local = _estimate_soma30_local_rotations(
        target,
        direct["soma30_global_rots"],
        soma30,
    )

    local_t = torch.from_numpy(soma30_local)
    root_pos = torch.from_numpy(np.asarray(motion_135[:, :3], dtype=np.float32)).clone()
    soma30_global, soma30_pos, _ = soma30.fk(local_t, root_pos)

    smpl_foot_indices = [_SMPL22_IDX[n] for n in ["left_foot", "right_foot", "left_ankle", "right_ankle"]]
    soma_foot_indices = [
        _SOMA30_IDX[n] for n in ["LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot"]
    ]
    source_t = torch.from_numpy(source)
    y_delta = soma30_pos[:, soma_foot_indices, 1].min(dim=1).values
    y_delta = y_delta - source_t[:, smpl_foot_indices, 1].min(dim=1).values
    if torch.max(torch.abs(y_delta)) > 1e-4:
        root_pos = root_pos.clone()
        root_pos[:, 1] -= y_delta
        soma30_global, soma30_pos, _ = soma30.fk(local_t, root_pos)

    root_delta_xz = torch.from_numpy(np.asarray(motion_135[:, [0, 2]], dtype=np.float32))
    root_delta_xz = root_delta_xz - soma30_pos[:, soma30.root_idx, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        root_pos = root_pos.clone()
        root_pos[:, 0] += root_delta_xz[:, 0]
        root_pos[:, 2] += root_delta_xz[:, 1]
        soma30_global, soma30_pos, _ = soma30.fk(local_t, root_pos)

    soma30_local_fk = _global_to_local(soma30_global, soma30)
    soma77_local = soma30.to_SOMASkeleton77(soma30_local_fk)
    soma77_global, soma77_pos, _ = soma30.somaskel77.fk(soma77_local, soma30_pos[:, 0])

    updated = dict(direct)
    updated.update({
        "soma30_local_rots": _as_numpy(soma30_local_fk).astype(np.float32),
        "soma30_global_rots": _as_numpy(soma30_global).astype(np.float32),
        "soma30_posed_joints": _as_numpy(soma30_pos).astype(np.float32),
        "soma77_local_rots": _as_numpy(soma77_local).astype(np.float32),
        "soma77_global_rots": _as_numpy(soma77_global).astype(np.float32),
        "soma77_posed_joints": _as_numpy(soma77_pos).astype(np.float32),
        "soma77_to_smpl22_joints": _as_numpy(soma77_pos[:, SOMA77_TO_SMPL22]).astype(np.float32),
    })
    return updated


def _shoulder_offset_smpl22_to_soma(
    motion_135: np.ndarray,
    bone_offsets: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    """Direct retarget plus a conservative shoulder-only rest-direction offset.

    Full position-aware fitting can over-rotate Chest/Head on arm-down clips,
    which is bad for SOMA's shoulder/neck skin weights.  This audit variant
    keeps torso, neck, head, legs, and lower arms on the direct KIMODO path and
    only rotates LeftShoulder/RightShoulder toward SOMA's rest bone direction.
    """
    direct = _official_smpl22_to_soma(motion_135, bone_offsets)
    smplx22, soma30 = _load_skeletons()
    t = int(motion_135.shape[0])
    soma30_global = np.tile(np.eye(3, dtype=np.float32), (t, 30, 1, 1))
    source_global = direct["source_global_rots_smpl22"]
    offsets = _neutral_bone_offsets(smplx22, soma30)
    for smpl_idx, soma_idx in enumerate(SMPLX22_TO_SOMA30):
        soma30_global[:, soma_idx] = source_global[:, smpl_idx]
    for name in ("LeftShoulder", "RightShoulder"):
        soma_idx = _SOMA30_IDX[name]
        offset = _scaled_rotation_np(offsets[soma_idx], alpha).astype(np.float32)
        soma30_global[:, soma_idx] = soma30_global[:, soma_idx] @ offset

    soma30_global_t = torch.from_numpy(soma30_global)
    soma30_global_t[:, 5] = _slerp_rot_matrices(
        soma30_global_t[:, 4], soma30_global_t[:, 6], 0.5
    )
    soma30_global_t[:, 7] = soma30_global_t[:, 6]
    soma30_global_t[:, 8] = soma30_global_t[:, 6]
    soma30_global_t[:, 9] = soma30_global_t[:, 6]
    soma30_global_t[:, 14] = soma30_global_t[:, 13]
    soma30_global_t[:, 15] = soma30_global_t[:, 13]
    soma30_global_t[:, 20] = soma30_global_t[:, 19]
    soma30_global_t[:, 21] = soma30_global_t[:, 19]

    soma30_local = _global_to_local(soma30_global_t, soma30)
    root_pos = torch.from_numpy(np.asarray(motion_135[:, :3], dtype=np.float32)).clone()
    soma30_global_fk, soma30_pos, _ = soma30.fk(soma30_local, root_pos)

    source_t = torch.from_numpy(direct["source_joints_smpl22"])
    smpl_foot_indices = [_SMPL22_IDX[n] for n in ["left_foot", "right_foot", "left_ankle", "right_ankle"]]
    soma_foot_indices = [
        _SOMA30_IDX[n] for n in ["LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot"]
    ]
    y_delta = soma30_pos[:, soma_foot_indices, 1].min(dim=1).values
    y_delta = y_delta - source_t[:, smpl_foot_indices, 1].min(dim=1).values
    if torch.max(torch.abs(y_delta)) > 1e-4:
        root_pos = root_pos.clone()
        root_pos[:, 1] -= y_delta
        soma30_global_fk, soma30_pos, _ = soma30.fk(soma30_local, root_pos)

    root_delta_xz = torch.from_numpy(np.asarray(motion_135[:, [0, 2]], dtype=np.float32))
    root_delta_xz = root_delta_xz - soma30_pos[:, soma30.root_idx, :][:, [0, 2]]
    if torch.max(torch.abs(root_delta_xz)) > 1e-6:
        root_pos = root_pos.clone()
        root_pos[:, 0] += root_delta_xz[:, 0]
        root_pos[:, 2] += root_delta_xz[:, 1]
        soma30_global_fk, soma30_pos, _ = soma30.fk(soma30_local, root_pos)

    soma30_local_fk = _global_to_local(soma30_global_fk, soma30)
    soma77_local = soma30.to_SOMASkeleton77(soma30_local_fk)
    soma77_global, soma77_pos, _ = soma30.somaskel77.fk(soma77_local, soma30_pos[:, 0])

    updated = dict(direct)
    updated.update({
        "soma30_local_rots": _as_numpy(soma30_local_fk).astype(np.float32),
        "soma30_global_rots": _as_numpy(soma30_global_fk).astype(np.float32),
        "soma30_posed_joints": _as_numpy(soma30_pos).astype(np.float32),
        "soma77_local_rots": _as_numpy(soma77_local).astype(np.float32),
        "soma77_global_rots": _as_numpy(soma77_global).astype(np.float32),
        "soma77_posed_joints": _as_numpy(soma77_pos).astype(np.float32),
        "soma77_to_smpl22_joints": _as_numpy(soma77_pos[:, SOMA77_TO_SMPL22]).astype(np.float32),
    })
    return updated


def _read_caption(text_root: Path, sid: str) -> str:
    path = text_root / f"{sid}.txt"
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split("#")
        if len(parts) < 4:
            continue
        try:
            f_tag = float(parts[2]) if parts[2] != "nan" else 0.0
            t_tag = float(parts[3]) if parts[3] != "nan" else 0.0
        except ValueError:
            continue
        if f_tag == 0.0 and t_tag == 0.0 and parts[0].strip():
            return parts[0].strip()
    return ""


def _identity_motion(num_frames: int = 90) -> np.ndarray:
    motion = np.zeros((num_frames, 135), dtype=np.float32)
    for j in range(22):
        motion[:, 3 + j * 6 + 0] = 1.0
        motion[:, 3 + j * 6 + 3] = 1.0
    return motion


def _load_ids(data_root: Path, ids_dir: Path | None, limit: int | None, include_tpose: bool) -> list[str]:
    ids: list[str] = ["TPOSE"] if include_tpose else []
    if ids_dir is not None:
        ids.extend(path.stem for path in sorted(ids_dir.glob("*.npz")))
        ids.extend(path.stem for path in sorted(ids_dir.glob("*.npy")))
    else:
        split = data_root / "split" / "test.txt"
        ids.extend(line.strip() for line in split.read_text().splitlines() if line.strip())
    seen = set()
    out = []
    for sid in ids:
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
        if limit is not None and len(out) >= limit:
            break
    return out


def _load_motion(data_root: Path, sid: str) -> np.ndarray:
    if sid == "TPOSE":
        return _identity_motion()
    m272 = np.load(str(data_root / "motion_data" / f"{sid}.npy")).astype(np.float32)
    return humanml272_to_motion135(m272).astype(np.float32)


def _dist(a: np.ndarray, i: int, j: int) -> float:
    return float(np.linalg.norm(a[:, i] - a[:, j], axis=-1).mean())


def _summarize_item(sid: str, item: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    source = item["source_joints_smpl22"]
    s30 = item["soma30_posed_joints"]
    s77 = item["soma77_posed_joints"]
    s77_names = _load_skeletons()[1].somaskel77.bone_order_names
    idx77 = {name: i for i, name in enumerate(s77_names)}
    shared77 = [idx77[name] for name in SOMA30_NAMES]
    shared_err = float(np.linalg.norm(s77[:, shared77] - s30, axis=-1).mean())
    return {
        "sid": sid,
        "frames": int(source.shape[0]),
        "source_to_soma30_mapped_mpjpe_m": float(
            np.linalg.norm(s30[:, SMPLX22_TO_SOMA30] - source, axis=-1).mean()
        ),
        "source_shoulder_width_m": _dist(source, 16, 17),
        "source_hip_width_m": _dist(source, 1, 2),
        "source_floor_y_m": float(source[:, [7, 8, 10, 11], 1].min()),
        "soma30_arm_width_m": _dist(s30, _SOMA30_IDX["LeftArm"], _SOMA30_IDX["RightArm"]),
        "soma30_clavicle_width_m": _dist(s30, _SOMA30_IDX["LeftShoulder"], _SOMA30_IDX["RightShoulder"]),
        "soma30_hip_width_m": _dist(s30, _SOMA30_IDX["LeftLeg"], _SOMA30_IDX["RightLeg"]),
        "soma30_floor_y_m": float(s30[:, [_SOMA30_IDX["LeftFoot"], _SOMA30_IDX["LeftToeBase"], _SOMA30_IDX["RightFoot"], _SOMA30_IDX["RightToeBase"]], 1].min()),
        "soma30_larm_chest_y_m": float((s30[:, _SOMA30_IDX["LeftArm"], 1] - s30[:, _SOMA30_IDX["Chest"], 1]).mean()),
        "soma30_rarm_chest_y_m": float((s30[:, _SOMA30_IDX["RightArm"], 1] - s30[:, _SOMA30_IDX["Chest"], 1]).mean()),
        "soma30_chest_lshoulder_x_mean": float(
            ((s30[:, _SOMA30_IDX["LeftShoulder"]] - s30[:, _SOMA30_IDX["Chest"]]) /
             np.maximum(
                 np.linalg.norm(
                     s30[:, _SOMA30_IDX["LeftShoulder"]] - s30[:, _SOMA30_IDX["Chest"]],
                     axis=-1,
                     keepdims=True,
                 ),
                 1e-8,
             ))[:, 0].mean()
        ),
        "soma30_chest_rshoulder_x_mean": float(
            ((s30[:, _SOMA30_IDX["RightShoulder"]] - s30[:, _SOMA30_IDX["Chest"]]) /
             np.maximum(
                 np.linalg.norm(
                     s30[:, _SOMA30_IDX["RightShoulder"]] - s30[:, _SOMA30_IDX["Chest"]],
                     axis=-1,
                     keepdims=True,
                 ),
                 1e-8,
             ))[:, 0].mean()
        ),
        "soma77_arm_width_m": _dist(s77, idx77["LeftArm"], idx77["RightArm"]),
        "soma77_floor_y_m": float(s77[:, [idx77["LeftFoot"], idx77["LeftToeBase"], idx77["RightFoot"], idx77["RightToeBase"]], 1].min()),
        "soma30_to_soma77_shared_mpjpe_m": shared_err,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="ref_repo/MotionStreamer/MotionStreamer/humanml3d_272")
    parser.add_argument("--ids-dir", default=None)
    parser.add_argument("--out-dir", default="outputs/evaluation/humanml3d_gt_soma_official_samples/motion135")
    parser.add_argument("--bone-offsets", default="data/hymotion_m2m_data/bone_offsets_22.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-tpose", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--retarget-mode",
        choices=["direct", "position_aware", "shoulder_offset"],
        default="direct",
        help="direct mirrors the KIMODO eval path; other modes are smoke-test retargets for shoulder-chain auditing.",
    )
    parser.add_argument("--shoulder-offset-alpha", type=float, default=0.75)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_root = data_root / "texts"
    ids_dir = Path(args.ids_dir) if args.ids_dir else None
    limit = args.limit if args.limit > 0 else None
    ids = _load_ids(data_root, ids_dir, limit, args.include_tpose)
    bone_offsets = torch.load(args.bone_offsets, map_location="cpu")
    bone_offsets_np = _as_numpy(bone_offsets).astype(np.float32)

    print(f"[setup] ids={len(ids)} out={out_dir} mode={args.retarget_mode}", flush=True)
    rows = []
    failed = 0
    for idx, sid in enumerate(ids, 1):
        dst = out_dir / f"{sid}.npz"
        if args.skip_existing and dst.exists():
            continue
        try:
            motion_135 = _load_motion(data_root, sid)
            if args.retarget_mode == "direct":
                item = _official_smpl22_to_soma(motion_135, bone_offsets_np)
                method = "kimodo_official_direct"
            elif args.retarget_mode == "position_aware":
                item = _position_aware_smpl22_to_soma(motion_135, bone_offsets_np)
                method = "soma30_position_aware_direction_fit"
            else:
                item = _shoulder_offset_smpl22_to_soma(
                    motion_135,
                    bone_offsets_np,
                    alpha=args.shoulder_offset_alpha,
                )
                method = f"soma30_shoulder_offset_alpha_{args.shoulder_offset_alpha:.2f}"
            caption = "identity T-pose" if sid == "TPOSE" else _read_caption(text_root, sid)
            np.savez_compressed(
                dst,
                source_motion_135=motion_135.astype(np.float32),
                caption=np.array(caption, dtype=object),
                source_id=np.array(sid, dtype=object),
                source_fps=np.array(30.0, dtype=np.float32),
                target_fps=np.array(30.0, dtype=np.float32),
                retarget_method=np.array(method, dtype=object),
                **item,
            )
            row = _summarize_item(sid, item)
            rows.append(row)
            print(
                f"[{idx:04d}/{len(ids):04d}] {sid} "
                f"SOMA arm={row['soma30_arm_width_m']:.3f} "
                f"mapped={row['source_to_soma30_mapped_mpjpe_m'] * 1000.0:.1f}mm "
                f"LArm-ChestY={row['soma30_larm_chest_y_m']:.3f} "
                f"floor={row['soma30_floor_y_m']:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[fail] {sid}: {type(exc).__name__}: {exc}", flush=True)
    summary = {"count": len(rows), "failed": failed, "items": rows}
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {json.dumps({'count': len(rows), 'failed': failed})}", flush=True)


if __name__ == "__main__":
    main()
