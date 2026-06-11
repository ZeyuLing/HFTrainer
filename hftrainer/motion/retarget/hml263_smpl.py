"""HumanML3D-263 -> SMPL ``motion_135`` retargeting (inverse kinematics).

This is the library home for the conversion previously living only in
``scripts/eval/hml263_to_smpl_ik.py``. Pipeline::

    HML263 features -> recover_from_ric -> 22 world joints (20 fps)
                    -> resample to target fps (30)
                    -> floor align
                    -> hierarchical position IK on the SMPL rest skeleton
                       (scipy align_vectors)
                    -> root translation from SMPL root vs target root
                    -> optional differentiable SMPL refine (Adam)
                    -> motion_135 = [transl(3), 22 x rot6d(6)]

The conversion is approximate (HML263 does not uniquely determine SMPL twist /
shape / mesh), so :func:`retarget_hml263_clip` returns a per-frame IK fit error
``fit_mpjpe_mm`` as the main quality diagnostic.

rot6d convention
----------------
:func:`hml263_to_motion135` defaults to **ROW-major** ``motion_135`` (the
``specs.MOTION_135`` convention), so its output feeds
:func:`hftrainer.motion.representation.motion272.motion135_to_272` directly with
NO repack. (The legacy ``scripts/eval/hml263_to_smpl_ik.py`` CLI defaulted to
``column`` for the MotionCLIP evaluator; pass ``rot6d_convention="column"`` to
reproduce that.)

Dependencies: ``smplx`` + a SMPL model dir (resolved via
:func:`hftrainer.motion.skeleton.body_models.resolve_smpl_model_dir`). The
optional GMM pose prior is loaded from ``ref_repo/FlowMDM`` only when
``gmm_pose_prior_weight > 0``.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Optional, Tuple

import numpy as np

N_JOINTS = 22
FOOT_HEIGHT_JOINTS = [7, 8, 10, 11]


def _patch_numpy_chumpy_aliases() -> None:
    """Keep legacy SMPL/chumpy pickles loadable under newer NumPy releases."""
    aliases = {
        "bool": np.bool_, "int": int, "float": float, "complex": complex,
        "object": object, "unicode": str, "str": str, "int_": np.int64,
        "float_": np.float64, "complex_": np.complex128, "object_": object,
        "unicode_": str, "str_": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def _import_smplx():
    _patch_numpy_chumpy_aliases()
    try:
        import smplx  # noqa
        return smplx
    except ModuleNotFoundError:
        # fall back to the MotionStreamer-vendored copy if present
        import sys
        ms_root = os.path.join("ref_repo", "MotionStreamer", "272-dim-Motion-Representation")
        for p in (ms_root, os.path.join(ms_root, "utils", "smplx")):
            if os.path.isdir(p):
                sys.path.insert(0, p)
        import smplx  # noqa
        return smplx


# --------------------------------------------------------------------------- #
# rotation helpers (numpy) — delegate to the unified rotation module
# --------------------------------------------------------------------------- #
def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    valid = n[..., 0] > eps
    return v / np.maximum(n, eps), valid


def _matrix_to_rot6d(rotmat: np.ndarray, convention: str) -> np.ndarray:
    from hftrainer.motion.representation.rotation import matrix_to_rotation_6d

    return matrix_to_rotation_6d(np.asarray(rotmat, dtype=np.float64), convention=convention).astype(np.float32)


def _cont6d_column_to_matrix(d6: np.ndarray) -> np.ndarray:
    from hftrainer.motion.representation.rotation import rotation_6d_to_matrix

    return rotation_6d_to_matrix(np.asarray(d6, dtype=np.float64), convention="column").astype(np.float32)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    mat = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = ww + xx - yy - zz
    mat[..., 0, 1] = 2.0 * (xy - wz)
    mat[..., 0, 2] = 2.0 * (xz + wy)
    mat[..., 1, 0] = 2.0 * (xy + wz)
    mat[..., 1, 1] = ww - xx + yy - zz
    mat[..., 1, 2] = 2.0 * (yz - wx)
    mat[..., 2, 0] = 2.0 * (xz - wy)
    mat[..., 2, 1] = 2.0 * (yz + wx)
    mat[..., 2, 2] = ww - xx - yy + zz
    return mat.astype(np.float32)


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
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


def _resample_rotations(rotmat: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample a ``(T, J, 3, 3)`` rotation sequence with per-joint **Slerp**.

    Spherical linear interpolation on SO(3) — NOT linear interpolation of
    rotation vectors / matrices, which takes chords through the sphere and
    distorts angular velocity (and degenerates for large inter-frame rotations).
    """
    from scipy.spatial.transform import Rotation as R, Slerp

    rotmat = np.asarray(rotmat, dtype=np.float64)
    T = len(rotmat)
    if abs(src_fps - dst_fps) < 1e-6 or T < 2:
        return rotmat.astype(np.float32)
    J = rotmat.shape[1]
    new_t = max(2, int(round(T * dst_fps / src_fps)))
    key_times = np.arange(T, dtype=np.float64)
    new_times = np.linspace(0.0, T - 1, new_t)
    out = np.empty((new_t, J, 3, 3), dtype=np.float32)
    for j in range(J):
        slerp = Slerp(key_times, R.from_matrix(rotmat[:, j]))
        out[:, j] = slerp(new_times).as_matrix().astype(np.float32)
    return out


def recover_hml263_local_rotations(
    feats_263: np.ndarray, source_fps: float, target_fps: float
) -> np.ndarray:
    """Recover HumanML3D canonical-skeleton local rotations (for IK init).

    The 126-D rotation block is not an SMPL pose, but its local orientations make
    a good initialization that reduces position-only IK twist ambiguity.
    """
    from hftrainer.motion.representation.humanml import recover_root_rot_pos
    import torch

    feats = np.asarray(feats_263, dtype=np.float32)
    r_rot_quat, _ = recover_root_rot_pos(torch.from_numpy(feats).float())
    root_mat = _quat_wxyz_to_matrix(r_rot_quat.numpy())
    start = 4 + (N_JOINTS - 1) * 3
    end = start + (N_JOINTS - 1) * 6
    body_cont6d = feats[..., start:end].reshape(len(feats), N_JOINTS - 1, 6)
    body_mat = _cont6d_column_to_matrix(body_cont6d)
    local_r = np.concatenate([root_mat[:, None], body_mat], axis=1)
    return _resample_rotations(local_r, source_fps, target_fps)


# --------------------------------------------------------------------------- #
# hierarchical position IK
# --------------------------------------------------------------------------- #
def estimate_local_rotations(
    target_joints: np.ndarray,
    rest_joints: np.ndarray,
    parents: np.ndarray,
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
) -> np.ndarray:
    """Estimate local rotations by aligning SMPL rest bones to target bones."""
    from scipy.spatial.transform import Rotation as R

    target_joints = np.asarray(target_joints, dtype=np.float64)
    rest_joints = np.asarray(rest_joints, dtype=np.float64)
    parents = np.asarray(parents[:N_JOINTS], dtype=np.int64)
    children = [[] for _ in range(N_JOINTS)]
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


def make_joint_fit_weights(preset: str) -> np.ndarray:
    weights = np.ones(N_JOINTS, dtype=np.float32)
    if preset == "uniform":
        return weights
    if preset == "relaxed_torso":
        for j in [3, 6, 9, 12, 13, 14, 15]:
            weights[j] = 0.15
        for j in [10, 11, 20, 21]:
            weights[j] = 0.35
        return weights
    if preset == "relaxed_upper":
        for j in [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
            weights[j] = 0.08
        for j in [10, 11]:
            weights[j] = 0.25
        return weights
    raise ValueError(f"unknown joint fit weight preset: {preset}")


def load_gmm_pose_prior(device):
    """Load the SMPLify GMM pose prior from ref_repo/FlowMDM (optional)."""
    import torch

    prior_path = "ref_repo/FlowMDM/utils/visualize/joints2smpl/src/prior.py"
    prior_folder = "ref_repo/FlowMDM/utils/visualize/joints2smpl/smpl_models"
    spec = importlib.util.spec_from_file_location("flowmdm_joints2smpl_prior", prior_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prior module from {prior_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prior = module.MaxMixturePrior(prior_folder=prior_folder, num_gaussians=8, dtype=torch.float32).to(device)
    prior.eval()
    for param in prior.parameters():
        param.requires_grad_(False)
    return prior


# --------------------------------------------------------------------------- #
# SMPL forward / refine
# --------------------------------------------------------------------------- #
def load_smpl_rest(model_dir: Optional[str] = None, device="cpu"):
    """Load a neutral SMPL model + rest 22-joint skeleton + parents.

    Returns ``(model, rest_joints (22,3), parents (J,))``.
    """
    import torch

    from hftrainer.motion.skeleton.body_models import resolve_smpl_model_dir

    smplx = _import_smplx()
    model_dir = resolve_smpl_model_dir(model_dir)
    # prefer no-chumpy copy if present (avoids legacy chumpy pickle issues)
    cand = model_dir
    if os.path.basename(model_dir.rstrip("/")) == "body_models":
        nochumpy = model_dir.rstrip("/")[: -len("body_models")] + "body_models_nochumpy"
        if os.path.isfile(os.path.join(nochumpy, "smpl", "SMPL_NEUTRAL.pkl")):
            cand = nochumpy
    model = smplx.create(cand, model_type="smpl", gender="neutral", ext="pkl", batch_size=1).to(device)
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


def smpl_forward_22(model, global_orient, body_pose_21, transl, batch_size, device):
    import torch

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
    model, target_joints, global_orient, body_pose_21, transl, iters, lr,
    pose_l2_weight, angle_prior_weight, device, smooth_weight=1e-3,
    joint_accel_weight=0.0, joint_fit_weights=None, gmm_pose_prior=None,
    gmm_pose_prior_weight=0.0,
):
    """Refine IK initialization by optimizing SMPL pose/transl against joints."""
    import torch

    if iters <= 0:
        fitted = smpl_forward_22(model, global_orient, body_pose_21, transl, 512, device)
        return global_orient, body_pose_21, transl, fitted

    target = torch.from_numpy(target_joints.astype(np.float32)).to(device)
    if joint_fit_weights is None:
        fit_weights = torch.ones((1, N_JOINTS), dtype=torch.float32, device=device)
    else:
        fit_weights = torch.from_numpy(
            np.asarray(joint_fit_weights, dtype=np.float32).reshape(1, N_JOINTS)
        ).to(device)
    fit_weights = fit_weights / fit_weights.mean().clamp_min(1e-6)
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
            body_pose=body_23, global_orient=g, transl=tr,
        )
        joints = out.joints[:, :N_JOINTS]
        data_loss = (((joints - target) ** 2).sum(dim=-1) * fit_weights).mean()
        pose_keep = ((b21 - b21_init) ** 2).mean()
        pose_prior = (body_23 ** 2).mean()
        if gmm_pose_prior is not None and gmm_pose_prior_weight > 0:
            betas = torch.zeros(n, 10, dtype=torch.float32, device=device)
            gmm_prior = gmm_pose_prior(body_23, betas).mean()
        else:
            gmm_prior = torch.tensor(0.0, device=device)
        if angle_prior_weight > 0:
            idx = torch.tensor([55, 58, 12, 15], dtype=torch.long, device=device)
            signs = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=torch.float32, device=device)
            angle_prior = torch.exp(body_23[:, idx] * signs).pow(2).mean()
        else:
            angle_prior = torch.tensor(0.0, device=device)
        if n >= 3:
            tr_acc = tr[2:] - 2 * tr[1:-1] + tr[:-2]
            pose_acc = b21[2:] - 2 * b21[1:-1] + b21[:-2]
            smooth = (tr_acc ** 2).mean() + 1e-2 * (pose_acc ** 2).mean()
            if joint_accel_weight > 0:
                joints_acc = joints[2:] - 2 * joints[1:-1] + joints[:-2]
                target_acc = target[2:] - 2 * target[1:-1] + target[:-2]
                joint_accel = ((joints_acc - target_acc) ** 2).sum(dim=-1).mean()
            else:
                joint_accel = torch.tensor(0.0, device=device)
        else:
            smooth = torch.tensor(0.0, device=device)
            joint_accel = torch.tensor(0.0, device=device)
        loss = (
            data_loss + 1e-4 * pose_keep + pose_l2_weight * pose_prior
            + angle_prior_weight * angle_prior + smooth_weight * smooth
            + joint_accel_weight * joint_accel + gmm_pose_prior_weight * gmm_prior
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        body_23 = torch.zeros(n, 69, dtype=torch.float32, device=device)
        body_23[:, :63] = b21
        out = model(
            betas=torch.zeros(n, 10, device=device),
            body_pose=body_23, global_orient=g, transl=tr,
        )
        fitted = out.joints[:, :N_JOINTS].detach().cpu().numpy().astype(np.float32)
    return (
        g.detach().cpu().numpy().astype(np.float32),
        b21.detach().cpu().numpy().astype(np.float32),
        tr.detach().cpu().numpy().astype(np.float32),
        fitted,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def retarget_hml263_clip(
    feats_263: np.ndarray,
    *,
    smpl_rest=None,
    model_dir: Optional[str] = None,
    device="cpu",
    source_fps: float = 20.0,
    target_fps: float = 30.0,
    batch_size: int = 256,
    floor_align: bool = True,
    foot_height_align: bool = True,
    refine_iters: int = 0,
    refine_lr: float = 2e-2,
    rotation_init: str = "position",
    orientation_mode: str = "bone",
    parent_ref_weight: float = 0.25,
    pose_l2_weight: float = 0.0,
    angle_prior_weight: float = 0.0,
    smooth_weight: float = 1e-3,
    joint_accel_weight: float = 0.0,
    joint_fit_weight_preset: str = "uniform",
    gmm_pose_prior=None,
    gmm_pose_prior_weight: float = 0.0,
    rot6d_convention: str = "row",
    target_joints_world: Optional[np.ndarray] = None,
) -> dict:
    """Retarget one un-normalized HML263 clip ``(T, 263)`` to SMPL ``motion_135``.

    Args mirror the validated ``scripts/eval/hml263_to_smpl_ik.py`` defaults,
    except ``rot6d_convention`` defaults to ``"row"`` (the ``motion_135`` /
    MS272-chain convention) instead of the script's ``"column"``.

    When ``target_joints_world (T,22,3)`` is supplied the HML263->joints decode
    is bypassed and IK runs directly on those world joints (used by joint-native
    baselines such as CondMDI). ``feats_263`` may then be ``None``.
    ``rotation_init='hml263'`` is unsupported in that mode.

    Returns a dict with ``motion_135 (T,135)``, ``transl``, ``global_orient``,
    ``body_pose``, ``target_joints``, ``fitted_joints``, ``fit_mpjpe_mm``.
    """
    from scipy.spatial.transform import Rotation as R

    from hftrainer.motion.representation.humanml import hml263_to_joints

    if smpl_rest is None:
        smpl_rest = load_smpl_rest(model_dir, device)
    model, rest_joints, parents = smpl_rest

    if target_joints_world is not None:
        if rotation_init == "hml263":
            raise ValueError("rotation_init='hml263' requires feats_263, not joints")
        target = np.asarray(target_joints_world, dtype=np.float32)
        if target.ndim != 3 or target.shape[1:] != (N_JOINTS, 3):
            raise ValueError(f"expected (T,{N_JOINTS},3) joints, got {target.shape}")
        feats = None
    else:
        feats = np.asarray(feats_263, dtype=np.float32)
        if feats.ndim != 2 or feats.shape[-1] != 263:
            raise ValueError(f"expected (T,263), got {feats.shape}")
        target = hml263_to_joints(feats, N_JOINTS)
    target = resample_linear(target, source_fps, target_fps)
    if floor_align:
        target = target.copy()
        target[..., 1] -= target[..., 1].min()

    if rotation_init == "hml263":
        local_r = recover_hml263_local_rotations(feats, source_fps, target_fps)
        if len(local_r) != len(target):
            raise ValueError(f"rotation_init length mismatch: {len(local_r)} vs {len(target)}")
    else:
        local_r = estimate_local_rotations(
            target, rest_joints, parents,
            orientation_mode=orientation_mode, parent_ref_weight=parent_ref_weight,
        )
    aa = R.from_matrix(local_r.reshape(-1, 3, 3)).as_rotvec().astype(np.float32)
    aa = aa.reshape(len(target), N_JOINTS, 3)
    global_orient = aa[:, 0]
    body_pose = aa[:, 1:].reshape(len(target), 63)

    joints_no_trans = smpl_forward_22(model, global_orient, body_pose, None, batch_size, device)
    transl = (target[:, 0] - joints_no_trans[:, 0]).astype(np.float32)

    jfw = make_joint_fit_weights(joint_fit_weight_preset)
    global_orient, body_pose, transl, fitted = refine_smpl_fit(
        model, target, global_orient, body_pose, transl, refine_iters, refine_lr,
        pose_l2_weight, angle_prior_weight, device, smooth_weight, joint_accel_weight,
        jfw, gmm_pose_prior, gmm_pose_prior_weight,
    )

    if foot_height_align:
        target_floor_y = target[:, FOOT_HEIGHT_JOINTS, 1].min(axis=1)
        fitted_floor_y = fitted[:, FOOT_HEIGHT_JOINTS, 1].min(axis=1)
        y_delta = (target_floor_y - fitted_floor_y).astype(np.float32)
        transl = transl.copy()
        fitted = fitted.copy()
        transl[:, 1] += y_delta
        fitted[..., 1] += y_delta[:, None]

    local_r = (
        R.from_rotvec(
            np.concatenate(
                [global_orient[:, None, :], body_pose.reshape(len(target), 21, 3)], axis=1
            ).reshape(-1, 3)
        )
        .as_matrix()
        .reshape(len(target), N_JOINTS, 3, 3)
        .astype(np.float32)
    )
    mpjpe_mm = np.linalg.norm(fitted - target, axis=-1).mean(axis=1).astype(np.float32) * 1000.0

    motion_135 = np.concatenate(
        [transl, _matrix_to_rot6d(local_r, rot6d_convention).reshape(len(target), N_JOINTS * 6)],
        axis=-1,
    ).astype(np.float32)

    return {
        "motion_135": motion_135,
        "transl": transl.astype(np.float32),
        "global_orient": global_orient.astype(np.float32),
        "body_pose": body_pose.astype(np.float32),
        "target_joints": target.astype(np.float32),
        "fitted_joints": fitted.astype(np.float32),
        "fit_mpjpe_mm": mpjpe_mm,
        "rot6d_convention": rot6d_convention,
        "source_fps": float(source_fps),
        "target_fps": float(target_fps),
    }


def hml263_to_motion135(feats_263: np.ndarray, **kwargs) -> np.ndarray:
    """Convenience wrapper returning only ``motion_135 (T,135)``.

    Output is ROW-major by default (chain-ready for ``motion135_to_272``). See
    :func:`retarget_hml263_clip` for all options and diagnostics.
    """
    return retarget_hml263_clip(feats_263, **kwargs)["motion_135"]


__all__ = [
    "N_JOINTS",
    "retarget_hml263_clip",
    "hml263_to_motion135",
    "estimate_local_rotations",
    "recover_hml263_local_rotations",
    "refine_smpl_fit",
    "smpl_forward_22",
    "load_smpl_rest",
    "load_gmm_pose_prior",
    "make_joint_fit_weights",
    "resample_linear",
]
