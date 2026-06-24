#!/usr/bin/env python3
"""Export InterGen / InterMask model-zoo generations for a small web viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciPyRotation

from hftrainer.motion.representation.interhuman262 import interhuman262_to_joints, interhuman262_to_local_rotmat
from hftrainer.motion.representation.rotation import matrix_to_axis_angle


DEFAULT_PROMPTS = [
    "two people shake hands and then walk apart",
    "one person helps another person stand up",
    "two people hug each other and then step back",
    "one person pushes another person gently",
    "two people dance in a circle together",
]


def _to_jsonable_joints(joints: np.ndarray) -> list:
    return np.asarray(joints, dtype=np.float32).round(5).tolist()


def _write_binary_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.ascontiguousarray(array).tofile(tmp)
    tmp.replace(path)


def _ih262_to_pair_joints(motion: np.ndarray) -> np.ndarray:
    """(T,2,262) -> (T,2,22,3)."""
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 3 or motion.shape[1:] != (2, 262):
        raise ValueError(f"Expected (T,2,262), got {motion.shape}")
    p0 = interhuman262_to_joints(motion[:, 0])
    p1 = interhuman262_to_joints(motion[:, 1])
    return np.stack([p0, p1], axis=1).astype(np.float32)


def _write_motion_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _floor_align_vertices(verts: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Set each person/frame mesh on the floor for viewer-only display."""
    verts = np.asarray(verts, dtype=np.float32).copy()
    floor_y = verts[..., 1].min(axis=2)
    verts[..., 1] -= floor_y[..., None]
    return verts, {
        "floor_aligned": True,
        "floor_shift_y_min": float((-floor_y).min()),
        "floor_shift_y_max": float((-floor_y).max()),
    }


def _canonical_floor_shift_y(verts: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Apply one scene-level vertical shift for viewer canonicalization only."""
    verts = np.asarray(verts, dtype=np.float32).copy()
    floor_y = float(verts[..., 1].min())
    verts[..., 1] -= floor_y
    return verts, {
        "floor_aligned": True,
        "canonical_y_shift": float(-floor_y),
        "canonical_alignment": "scene_y_floor_shift_only",
    }


def _foot_slide_stats_from_vertices(verts: np.ndarray) -> dict[str, float]:
    verts = np.asarray(verts, dtype=np.float32)
    T, P = verts.shape[:2]
    centroids = np.zeros((T, P, 2), dtype=np.float32)
    floor_q = np.zeros((T, P), dtype=np.float32)
    for person in range(P):
        person_verts = verts[:, person]
        y = person_verts[..., 1]
        q = np.quantile(y, 0.015, axis=1).astype(np.float32)
        floor_q[:, person] = q
        for t in range(T):
            mask = y[t] <= q[t] + 0.015
            centroids[t, person] = person_verts[t, mask][:, [0, 2]].mean(axis=0)
    speeds = np.linalg.norm(np.diff(centroids, axis=0), axis=-1)
    contacts = floor_q[:-1] < 0.08
    vals = speeds[contacts]
    if vals.size == 0:
        return {
            "foot_slide_contact_frames": 0,
            "foot_slide_mean_mm_per_frame": 0.0,
            "foot_slide_p95_mm_per_frame": 0.0,
            "foot_slide_max_mm_per_frame": 0.0,
        }
    return {
        "foot_slide_contact_frames": int(contacts.sum()),
        "foot_slide_mean_mm_per_frame": float(vals.mean() * 1000.0),
        "foot_slide_p95_mm_per_frame": float(np.percentile(vals, 95) * 1000.0),
        "foot_slide_max_mm_per_frame": float(vals.max() * 1000.0),
    }


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps), n[..., 0] > eps


def _estimate_global_orient_from_joints(source_joints: np.ndarray, target_joints: np.ndarray) -> np.ndarray:
    """Estimate the missing root/global orientation for InterHuman-262 body pose.

    InterHuman-262 stores canonical joint positions, but its rot6d block contains
    only 21 non-root SMPL body rotations.  Aligning the zero-root FK skeleton to
    the stored positions recovers the root orientation needed for mesh export.
    """
    source_joints = np.asarray(source_joints, dtype=np.float32)
    target_joints = np.asarray(target_joints, dtype=np.float32)
    fit_joints = np.array([1, 2, 4, 5, 7, 8, 10, 11, 16, 17, 18, 19, 20, 21], dtype=np.int64)
    rotations: list[np.ndarray] = []
    for source, target in zip(source_joints, target_joints):
        src = (source[fit_joints] - source[0]).astype(np.float64)
        dst = (target[fit_joints] - target[0]).astype(np.float64)
        u, _, vt = np.linalg.svd(src.T @ dst)
        rot = vt.T @ u.T
        if np.linalg.det(rot) < 0:
            vt[-1] *= -1.0
            rot = vt.T @ u.T
        rotations.append(rot.astype(np.float32))
    return matrix_to_axis_angle(np.stack(rotations, axis=0)).astype(np.float32)


def _estimate_local_rotations_from_joints(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    parent_ref_weight: float = 0.25,
) -> np.ndarray:
    """Fit SMPL local rotations to a 22-joint position trajectory.

    InterHuman-262 stores canonical joint positions directly. For viewer meshes
    we prioritize matching those positions, because the 262 local-rotation block
    is not canonicalized by the same root transform as the position block.
    """
    target_joints = np.asarray(target_joints, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)[:22]
    parents = np.asarray(parents, dtype=np.int64)[:22]
    children: list[list[int]] = [[] for _ in range(22)]
    for joint in range(1, 22):
        parent = int(parents[joint])
        if 0 <= parent < 22:
            children[parent].append(joint)

    offsets = np.zeros((22, 3), dtype=np.float64)
    for joint in range(1, 22):
        offsets[joint] = rest_joints[joint] - rest_joints[int(parents[joint])]

    local = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), 22, 1, 1))
    global_r = np.tile(np.eye(3, dtype=np.float64), (len(target_joints), 22, 1, 1))
    for t, joints in enumerate(target_joints):
        for joint in range(22):
            parent = int(parents[joint])
            parent_global = np.eye(3) if parent < 0 else global_r[t, parent]
            rest_vecs_list = [offsets[child] for child in children[joint]]
            target_vecs_list = [joints[child] - joints[joint] for child in children[joint]]
            weights = [1.0] * len(rest_vecs_list)
            if parent >= 0:
                rest_vecs_list.append(rest_joints[parent] - rest_joints[joint])
                target_vecs_list.append(joints[parent] - joints[joint])
                weights.append(parent_ref_weight)
            if not rest_vecs_list:
                global_r[t, joint] = parent_global
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
                    rot_local = SciPyRotation.align_vectors(dst_local, src, weights=valid_weights)[0].as_matrix()
                except Exception:
                    rot_local = np.eye(3)
            local[t, joint] = rot_local
            global_r[t, joint] = parent_global @ rot_local
    return local.astype(np.float32)


def _interx_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """InterMask/InterX official row-drop 6D rotation -> matrix.

    This differs from the InterHuman-262 row-interleaved convention. The
    official InterMask code stores ``matrix[..., :2, :].reshape(..., 6)`` and
    reconstructs matrices by stacking the two normalized vectors as rows.
    """
    d6 = np.asarray(rot6d, dtype=np.float32)
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)


def _interx_person_to_smpl_pose(person_motion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one InterX person ``(T,56,6)`` into body-only SMPL parameters."""
    person_motion = np.asarray(person_motion, dtype=np.float32)
    if person_motion.ndim != 3 or person_motion.shape[1:] != (56, 6):
        raise ValueError(f"Expected (T,56,6), got {person_motion.shape}")
    pose_mats = _interx_rot6d_to_matrix(person_motion[:, :-1])
    pose_aa = matrix_to_axis_angle(pose_mats.reshape(-1, 3, 3)).astype(np.float32).reshape(
        len(person_motion), 55, 3
    )
    global_orient = pose_aa[:, 0]
    body_pose_21 = pose_aa[:, 1:22].reshape(len(person_motion), 63)
    transl = person_motion[:, -1, :3].astype(np.float32)
    return global_orient, body_pose_21, transl


class _SMPLMeshExporter:
    def __init__(self, model_dir: str, device: str, refine_iters: int = 0) -> None:
        import smplx
        from tools.momask263_to_smpl85 import SeqFitter

        self.device = torch.device(device)
        self.refine_iters = int(refine_iters)
        self.interhuman_smplify_fitter = SeqFitter(
            num_iters=max(self.refine_iters, 20),
            device=str(self.device),
        )
        self.model = smplx.create(
            model_dir,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=1,
        ).to(self.device).eval()
        self.faces = np.asarray(self.model.faces, dtype=np.uint32)
        with torch.no_grad():
            out = self.model(
                global_orient=torch.zeros(1, 3, device=self.device),
                body_pose=torch.zeros(1, 69, device=self.device),
                betas=torch.zeros(1, 10, device=self.device),
                transl=torch.zeros(1, 3, device=self.device),
            )
        self.rest_joints = out.joints[0, :22].detach().cpu().numpy().astype(np.float32)
        self.parents = self.model.parents.detach().cpu().numpy().astype(np.int64)

    def _refine_pose_to_joints(
        self,
        target_joints: np.ndarray,
        global_orient: np.ndarray,
        body_pose_21: np.ndarray,
        transl: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target_joints = np.asarray(target_joints, dtype=np.float32)
        n = len(target_joints)
        if self.refine_iters <= 0 or n == 0:
            fitted = self._pose_to_vertices_and_joints(global_orient, body_pose_21, transl, n or 1)[1]
            return global_orient, body_pose_21, transl, fitted

        with torch.enable_grad():
            target = torch.from_numpy(target_joints).to(self.device)
            g = torch.tensor(global_orient, dtype=torch.float32, device=self.device, requires_grad=True)
            b21 = torch.tensor(body_pose_21, dtype=torch.float32, device=self.device, requires_grad=True)
            tr = torch.tensor(transl, dtype=torch.float32, device=self.device, requires_grad=True)
            b21_init = b21.detach().clone()
            opt = torch.optim.Adam([g, b21, tr], lr=2e-2)

            for _ in range(self.refine_iters):
                body_pose = torch.zeros(n, 69, dtype=torch.float32, device=self.device)
                body_pose[:, :63] = b21
                out = self.model(
                    global_orient=g,
                    body_pose=body_pose,
                    betas=torch.zeros(n, 10, device=self.device),
                    transl=tr,
                )
                joints = out.joints[:, :22]
                data_loss = ((joints - target) ** 2).sum(dim=-1).mean()
                pose_keep = ((b21 - b21_init) ** 2).mean()
                pose_prior = (body_pose ** 2).mean()
                if n >= 3:
                    tr_acc = tr[2:] - 2.0 * tr[1:-1] + tr[:-2]
                    pose_acc = b21[2:] - 2.0 * b21[1:-1] + b21[:-2]
                    smooth = (tr_acc ** 2).mean() + 1e-2 * (pose_acc ** 2).mean()
                else:
                    smooth = torch.tensor(0.0, device=self.device)
                loss = data_loss + 1e-4 * pose_keep + 1e-5 * pose_prior + 1e-2 * smooth
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        with torch.no_grad():
            body_pose = torch.zeros(n, 69, dtype=torch.float32, device=self.device)
            body_pose[:, :63] = b21
            out = self.model(
                global_orient=g,
                body_pose=body_pose,
                betas=torch.zeros(n, 10, device=self.device),
                transl=tr,
            )
            fitted = out.joints[:, :22].detach().cpu().numpy().astype(np.float32)
        return (
            g.detach().cpu().numpy().astype(np.float32),
            b21.detach().cpu().numpy().astype(np.float32),
            tr.detach().cpu().numpy().astype(np.float32),
            fitted,
        )

    @torch.no_grad()
    def _pose_to_vertices_and_joints(
        self,
        global_orient: np.ndarray,
        body_pose_21: np.ndarray,
        transl: np.ndarray,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(global_orient)
        body_pose = np.zeros((n, 69), dtype=np.float32)
        body_pose[:, :63] = body_pose_21.astype(np.float32)
        betas = np.zeros((n, 10), dtype=np.float32)

        verts: list[np.ndarray] = []
        joints: list[np.ndarray] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            out = self.model(
                global_orient=torch.from_numpy(global_orient[start:end]).to(self.device),
                body_pose=torch.from_numpy(body_pose[start:end]).to(self.device),
                betas=torch.from_numpy(betas[start:end]).to(self.device),
                transl=torch.from_numpy(transl[start:end]).to(self.device),
            )
            verts.append(out.vertices.detach().cpu().numpy().astype(np.float32))
            joints.append(out.joints[:, :22].detach().cpu().numpy().astype(np.float32))
        return np.concatenate(verts, axis=0), np.concatenate(joints, axis=0)

    @torch.no_grad()
    def _pose_betas_to_vertices_and_joints(
        self,
        global_orient: np.ndarray,
        body_pose_23: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        verts: list[np.ndarray] = []
        joints: list[np.ndarray] = []
        for start in range(0, len(global_orient), chunk_size):
            end = min(start + chunk_size, len(global_orient))
            out = self.model(
                global_orient=torch.from_numpy(global_orient[start:end]).to(self.device),
                body_pose=torch.from_numpy(body_pose_23[start:end]).to(self.device),
                betas=torch.from_numpy(betas[start:end]).to(self.device),
                transl=torch.from_numpy(transl[start:end]).to(self.device),
            )
            verts.append(out.vertices.detach().cpu().numpy().astype(np.float32))
            joints.append(out.joints[:, :22].detach().cpu().numpy().astype(np.float32))
        return np.concatenate(verts, axis=0), np.concatenate(joints, axis=0)

    @torch.no_grad()
    def _joints22_pair_to_vertices(
        self, pair_joints: np.ndarray, chunk_size: int
    ) -> tuple[np.ndarray, dict[str, float]]:
        pair_joints = np.asarray(pair_joints, dtype=np.float32)
        if pair_joints.ndim != 4 or pair_joints.shape[1:] != (2, 22, 3):
            raise ValueError(f"Expected (T,2,22,3), got {pair_joints.shape}")
        T = pair_joints.shape[0]
        verts_people: list[np.ndarray] = []
        fitted_people: list[np.ndarray] = []
        err_people: list[np.ndarray] = []
        for person in range(2):
            target = pair_joints[:, person]
            local_r = _estimate_local_rotations_from_joints(target, self.rest_joints, self.parents)
            aa = matrix_to_axis_angle(local_r.reshape(-1, 3, 3)).astype(np.float32).reshape(T, 22, 3)
            global_orient = aa[:, 0]
            body_pose_21 = aa[:, 1:].reshape(T, 63)
            zero_trans = np.zeros((T, 3), dtype=np.float32)
            _, joints_no_trans = self._pose_to_vertices_and_joints(
                global_orient, body_pose_21, zero_trans, chunk_size
            )
            transl = (target[:, 0] - joints_no_trans[:, 0]).astype(np.float32)
            global_orient, body_pose_21, transl, fitted = self._refine_pose_to_joints(
                target, global_orient, body_pose_21, transl
            )
            verts, fitted = self._pose_to_vertices_and_joints(global_orient, body_pose_21, transl, chunk_size)
            verts_people.append(verts)
            fitted_people.append(fitted)
            err_people.append(np.linalg.norm(fitted - target, axis=-1) * 1000.0)
        verts = np.stack(verts_people, axis=1)
        err = np.stack(err_people, axis=1)
        stats = {
            "fit_mpjpe_mm_mean": float(err.mean()),
            "fit_mpjpe_mm_p95": float(np.percentile(err, 95)),
            "smpl_refine_iters": int(self.refine_iters),
        }
        verts, floor_stats = _floor_align_vertices(verts)
        stats.update(floor_stats)
        return verts, stats

    @torch.no_grad()
    def pair262_to_vertices(self, motion: np.ndarray, chunk_size: int = 64) -> tuple[np.ndarray, dict[str, float]]:
        """Convert ``(T,2,262)`` InterHuman output to ``(T,2,V,3)`` SMPL mesh."""
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 3 or motion.shape[1:] != (2, 262):
            raise ValueError(f"Expected (T,2,262), got {motion.shape}")
        return self._joints22_pair_to_vertices(_ih262_to_pair_joints(motion), chunk_size)

    @torch.no_grad()
    def pair262_to_vertices_smplify(
        self, motion: np.ndarray, chunk_size: int = 64
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Convert InterHuman-262 to SMPL with MDM/MoMask SMPLify3D joints2smpl."""
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 3 or motion.shape[1:] != (2, 262):
            raise ValueError(f"Expected (T,2,262), got {motion.shape}")
        verts_people: list[np.ndarray] = []
        err_people: list[np.ndarray] = []
        beta_std: list[float] = []
        for person in range(2):
            target = interhuman262_to_joints(motion[:, person]).astype(np.float32)
            with torch.enable_grad():
                params = self.interhuman_smplify_fitter.fit(target)
            pose72 = np.asarray(params["pose"], dtype=np.float32)
            transl = np.asarray(params["trans"], dtype=np.float32)
            betas = np.asarray(params["betas"], dtype=np.float32)
            verts, fitted = self._pose_betas_to_vertices_and_joints(
                pose72[:, :3],
                pose72[:, 3:72],
                transl,
                betas,
                chunk_size,
            )
            verts_people.append(verts)
            err_people.append(np.linalg.norm(fitted - target, axis=-1) * 1000.0)
            beta_std.append(float(betas.std()))
        verts = np.stack(verts_people, axis=1)
        err = np.stack(err_people, axis=1)
        stats = {
            "fit_mpjpe_mm_mean": float(err.mean()),
            "fit_mpjpe_mm_p95": float(np.percentile(err, 95)),
            "smplify_iters": int(max(self.refine_iters, 20)),
            "mesh_pose_init": "interhuman262_positions_smplify3d_joints2smpl",
            "rot6d_block_usage": "ignored_for_mesh_endpoint_orientation",
            "betas_std_mean": float(np.mean(beta_std)),
        }
        verts, floor_stats = _canonical_floor_shift_y(verts)
        stats.update(floor_stats)
        return verts, stats

    @torch.no_grad()
    def pair262_to_vertices_position_ik(
        self, motion: np.ndarray, chunk_size: int = 64
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Convert InterHuman-262 to SMPL from stored joint positions.

        Official InterMask/InterHuman visualization is position based.  The
        generated rot6d block is not used here because endpoint mesh twist from
        that block is visibly unstable for ankles, shoulders, and wrists.
        """
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 3 or motion.shape[1:] != (2, 262):
            raise ValueError(f"Expected (T,2,262), got {motion.shape}")
        verts_people: list[np.ndarray] = []
        err_people: list[np.ndarray] = []
        for person in range(2):
            target = interhuman262_to_joints(motion[:, person]).astype(np.float32)
            with torch.enable_grad():
                result = self.interhuman_position_retargeter.retarget_positions(target)
            global_orient = np.asarray(result["global_orient"], dtype=np.float32)
            body_pose_21 = np.asarray(result["body_pose"], dtype=np.float32)
            transl = np.asarray(result["transl"], dtype=np.float32)
            verts, fitted = self._pose_to_vertices_and_joints(global_orient, body_pose_21, transl, chunk_size)
            verts_people.append(verts)
            err_people.append(np.linalg.norm(fitted - target, axis=-1) * 1000.0)
        verts = np.stack(verts_people, axis=1)
        err = np.stack(err_people, axis=1)
        stats = {
            "fit_mpjpe_mm_mean": float(err.mean()),
            "fit_mpjpe_mm_p95": float(np.percentile(err, 95)),
            "smpl_refine_iters": int(max(self.refine_iters, 30)),
            "mesh_pose_init": "interhuman262_positions_validated_smpl_ik",
            "rot6d_block_usage": "ignored_for_mesh_endpoint_orientation",
        }
        verts, floor_stats = _floor_align_vertices(verts)
        stats.update(floor_stats)
        return verts, stats

    @torch.no_grad()
    def pair262_to_vertices_rot_init(
        self, motion: np.ndarray, chunk_size: int = 64
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Convert InterHuman-262 to SMPL using its rot6d block as pose init."""
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 3 or motion.shape[1:] != (2, 262):
            raise ValueError(f"Expected (T,2,262), got {motion.shape}")
        T = motion.shape[0]
        verts_people: list[np.ndarray] = []
        err_people: list[np.ndarray] = []
        for person in range(2):
            target = interhuman262_to_joints(motion[:, person]).astype(np.float32)
            local_r = interhuman262_to_local_rotmat(motion[:, person]).astype(np.float32)
            aa = matrix_to_axis_angle(local_r.reshape(-1, 3, 3)).astype(np.float32).reshape(T, 21, 3)
            body_pose_21 = aa.reshape(T, 63)
            zero_trans = np.zeros((T, 3), dtype=np.float32)
            zero_global = np.zeros((T, 3), dtype=np.float32)
            _, joints_zero_root = self._pose_to_vertices_and_joints(
                zero_global, body_pose_21, zero_trans, chunk_size
            )
            global_orient = _estimate_global_orient_from_joints(joints_zero_root, target)
            _, joints_no_trans = self._pose_to_vertices_and_joints(
                global_orient, body_pose_21, zero_trans, chunk_size
            )
            transl = (target[:, 0] - joints_no_trans[:, 0]).astype(np.float32)
            global_orient, body_pose_21, transl, fitted = self._refine_pose_to_joints(
                target, global_orient, body_pose_21, transl
            )
            verts, fitted = self._pose_to_vertices_and_joints(global_orient, body_pose_21, transl, chunk_size)
            verts_people.append(verts)
            err_people.append(np.linalg.norm(fitted - target, axis=-1) * 1000.0)
        verts = np.stack(verts_people, axis=1)
        err = np.stack(err_people, axis=1)
        stats = {
            "fit_mpjpe_mm_mean": float(err.mean()),
            "fit_mpjpe_mm_p95": float(np.percentile(err, 95)),
            "smpl_refine_iters": int(self.refine_iters),
            "mesh_pose_init": "interhuman262_rot6d_body_pose_with_kabsch_root_orient",
            "root_orient_estimator": "kabsch_core_joints_from_interhuman262_positions",
        }
        verts, floor_stats = _floor_align_vertices(verts)
        stats.update(floor_stats)
        return verts, stats

    @torch.no_grad()
    def interx_to_vertices(self, motion: np.ndarray, chunk_size: int = 64) -> tuple[np.ndarray, dict[str, float]]:
        """Convert InterX ``motion_rep=smpl`` ``(T,56,12)`` to body-only SMPL meshes."""
        motion = np.asarray(motion, dtype=np.float32)
        if motion.ndim != 3 or motion.shape[1:] != (56, 12):
            raise ValueError(f"Expected (T,56,12), got {motion.shape}")
        T = motion.shape[0]
        go0, bp0, tr0 = _interx_person_to_smpl_pose(motion[:, :, :6])
        go1, bp1, tr1 = _interx_person_to_smpl_pose(motion[:, :, 6:12])
        global_orient = np.stack([go0, go1], axis=1).reshape(T * 2, 3)
        body_pose_21 = np.stack([bp0, bp1], axis=1).reshape(T * 2, 63)
        transl = np.stack([tr0, tr1], axis=1).reshape(T * 2, 3)
        verts, _ = self._pose_to_vertices_and_joints(global_orient, body_pose_21, transl, chunk_size)
        verts = verts.reshape(T, 2, -1, 3)
        verts, floor_stats = _canonical_floor_shift_y(verts)
        foot_stats = _foot_slide_stats_from_vertices(verts)
        stats = {
            "hands_ignored": True,
            "translation_postprocess": "y_floor_shift_only",
        }
        stats.update(floor_stats)
        stats.update(foot_stats)
        return verts, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/evaluation/interaction_model_zoo_viewer")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--motion-len", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--intergen-artifact", default="checkpoints/intergen/hftrainer_interhuman")
    parser.add_argument("--intermask-interhuman-artifact", default="checkpoints/intermask/hftrainer_interhuman")
    parser.add_argument("--intermask-interx-artifact", default="checkpoints/intermask/hftrainer_interx")
    parser.add_argument("--prompt", action="append", help="Prompt to include. Repeatable.")
    parser.add_argument("--intermask-time-steps", type=int, default=20)
    parser.add_argument("--mesh-model-dir", default="checkpoints/smpl_models")
    parser.add_argument("--mesh-chunk-size", type=int, default=64)
    parser.add_argument(
        "--mesh-refine-iters",
        type=int,
        default=30,
        help="Viewer-only SMPL pose refinement steps for InterHuman-262 position fitting.",
    )
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    out_root = Path(args.out_root)
    meshes_dir = out_root / "meshes"
    out_root.mkdir(parents=True, exist_ok=True)

    from hftrainer.models.motion.intergen import InterGenBundle
    from hftrainer.models.motion.intermask import InterMaskBundle

    intergen = InterGenBundle.from_pretrained(args.intergen_artifact, device=args.device)
    intermask_ih = InterMaskBundle.from_pretrained(
        args.intermask_interhuman_artifact,
        dataset_name="interhuman",
        device=args.device,
        time_steps=args.intermask_time_steps,
    )
    intermask_ix = InterMaskBundle.from_pretrained(
        args.intermask_interx_artifact,
        dataset_name="interx",
        device=args.device,
        time_steps=args.intermask_time_steps,
    )
    mesh_exporter = _SMPLMeshExporter(args.mesh_model_dir, args.device, refine_iters=args.mesh_refine_iters)
    faces_rel = "meshes/smpl_faces.uint32.bin"
    _write_binary_array(out_root / faces_rel, mesh_exporter.faces)

    methods = [
        {
            "key": "intergen",
            "label": "InterGen",
            "kind": "smpl_mesh_pair",
            "description": "InterHuman native-262 joint positions fitted to body-only SMPL mesh",
        },
        {
            "key": "intermask_interhuman",
            "label": "InterMask InterHuman",
            "kind": "smpl_mesh_pair",
            "description": "InterHuman native-262 joint positions fitted with SMPLify3D joints2smpl",
        },
        {
            "key": "intermask_interx",
            "label": "InterMask InterX",
            "kind": "smpl_mesh_pair",
            "description": "InterX SMPL-X rot6d/trans converted to body-only SMPL mesh",
        },
    ]

    cases = []
    for idx, prompt in enumerate(prompts):
        case_id = f"case_{idx:02d}"
        seed = int(args.seed) + idx
        print(f"[interaction-viewer] {case_id}: {prompt!r} seed={seed}", flush=True)

        intergen_motion = intergen.generate(prompt, motion_len=args.motion_len, seed=seed)[0]
        intermask_ih_motion = intermask_ih.generate(prompt, motion_len=args.motion_len, seed=seed)[0]
        intermask_ix_motion = intermask_ix.generate(prompt, motion_len=args.motion_len, seed=seed)[0]

        outputs: dict[str, Any] = {}
        for method_key, motion, converter, conversion in [
            (
                "intergen",
                intergen_motion,
                mesh_exporter.pair262_to_vertices,
                "interhuman262_positions_to_body_only_smpl_ik",
            ),
            (
                "intermask_interhuman",
                intermask_ih_motion,
                mesh_exporter.pair262_to_vertices_smplify,
                "interhuman262_positions_to_smplify3d_body_only_smpl",
            ),
            (
                "intermask_interx",
                intermask_ix_motion,
                mesh_exporter.interx_to_vertices,
                "interx_smpl_rot6d_trans_to_body_only_smpl",
            ),
        ]:
            verts, extra_meta = converter(motion, chunk_size=args.mesh_chunk_size)
            rel = f"meshes/{case_id}_{method_key}_vertices.float32.bin"
            _write_binary_array(out_root / rel, verts)
            meta_rel = f"meshes/{case_id}_{method_key}.json"
            meta = {
                "case_id": case_id,
                "method": method_key,
                "kind": "smpl_mesh_pair",
                "prompt": prompt,
                "seed": seed,
                "fps": 30,
                "num_frames": int(verts.shape[0]),
                "num_people": int(verts.shape[1]),
                "num_vertices": int(verts.shape[2]),
                "num_faces": int(mesh_exporter.faces.shape[0]),
                "vertices": rel,
                "faces": faces_rel,
                "conversion": conversion,
                "bounds": {
                    "min": verts.reshape(-1, 3).min(axis=0).round(5).tolist(),
                    "max": verts.reshape(-1, 3).max(axis=0).round(5).tolist(),
                },
            }
            meta.update(extra_meta)
            _write_motion_json(
                out_root / meta_rel,
                meta,
            )
            outputs[method_key] = meta_rel

        cases.append(
            {
                "case_id": case_id,
                "prompt": prompt,
                "seed": seed,
                "motion_len": int(args.motion_len),
                "outputs": outputs,
            }
        )

    manifest = {
        "title": "InterGen / InterMask Model Zoo Smoke Cases",
        "generated_by": "tools/export_interaction_model_zoo_viewer.py",
        "device": args.device,
        "body_model": {
            "type": "SMPL",
            "model_dir": args.mesh_model_dir,
            "faces": faces_rel,
            "num_faces": int(mesh_exporter.faces.shape[0]),
        },
        "methods": methods,
        "cases": cases,
    }
    _write_motion_json(out_root / "manifest.json", manifest)
    print(f"[interaction-viewer] wrote {out_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
