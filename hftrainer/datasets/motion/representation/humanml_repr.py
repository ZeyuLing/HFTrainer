"""HumanML3D / MotionStreamer-272 representation interop.

This module provides a **single, reusable, validated** path to convert a
MotionStreamer ``humanml3d_272`` clip (30 fps) into the official HumanML3D
``263``-dim feature (20 fps) used by the ``text_mot_match`` (Comp_v6) evaluator.

Why a dedicated module
----------------------
The naive route -- decode the 272 *stored joint positions* and feed them to
MoMask ``process_file`` -- lands ~30 mm (rigid-invariant) away from the joints
the official HumanML3D pipeline uses, because the 272 positions come from
MotionStreamer's own SMPL-X / face-z pipeline. The official HumanML3D features
are built from **SMPL-H forward-kinematics** joints. Replicating that exactly
closed the entire GT metric gap (R@3 0.78 -> 0.795, MM-Dist 3.10 -> 2.95,
Diversity 9.24 -> 9.52, matching published "Real" rows).

The validated recipe (this module implements it):

    272 [rot block 140:272] = SMPL local rotations (SMPL pose, 6D)
        -> recover SMPL params (axis-angle) + world root translation
        -> SMPL-H forward kinematics -> 22 joints (30 fps)
        -> linear resample 30 -> 20 fps
        -> MoMask ``process_file`` (uniform_skeleton to the OFFICIAL canonical
           000021 skeleton + IK) -> 263 feature

Important conventions (do NOT "simplify"):
  * The 263 rotation channels are NOT the SMPL pose -- ``process_file`` derives
    them by inverse kinematics on the joint positions (``get_cont6d_params``).
    This module only uses the SMPL rotations as an intermediate to obtain
    correct joint *positions*.
  * The 6D rotation block of 272 is ROW-major (pytorch3d-style first two rows);
    the local ``_rotation_6d_to_matrix`` here is the verified implementation
    (do not swap for a column-major one).
  * ``uniform_skeleton`` MUST target the official canonical 000021 skeleton,
    otherwise the whole body is rescaled (~6%) and the official Mean/Std no
    longer apply.

External dependencies are imported lazily so importing this module is cheap:
  * MoMask ``utils.motion_process`` (``process_file`` / ``recover_from_ric``)
    from ``<momask_root>``.
  * SMPL-H neutral body model ``model.npz`` (only ``J_regressor`` /
    ``v_template`` / ``kintree_table`` are read; no LBS / mesh needed).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NJOINT = 22


# ============================================================================
# Default external paths (relative to repo root; override via HumanMLReprPaths)
# ============================================================================

@dataclass(frozen=True)
class HumanMLReprPaths:
    """Filesystem locations of the external assets the conversion needs."""

    momask_root: str = "ref_repo/Momask/momask-codes"
    smplh_model: str = ("ref_repo/MoGenDiT/motion_process/body_model/"
                        "smplh/neutral/model.npz")
    canonical_ref_joints: str = "ref_repo/TeSMo/dataset/HumanML3D/000021.npy"
    momask_mean_std_dir: str = ("ref_repo/Momask/momask-codes/checkpoints/"
                                "t2m/Comp_v6_KLD005/meta")

    def resolve(self, name: str) -> Path:
        p = Path(getattr(self, name))
        return p if p.is_absolute() else (_REPO_ROOT / p)


DEFAULT_PATHS = HumanMLReprPaths()


# ============================================================================
# Rotation helpers (verified numerics -- keep as-is)
# ============================================================================

def _rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """ROW-major 6D -> rotation matrix (first two rows; pytorch3d convention).

    Matches MotionStreamer ``utils.face_z_align_util.rotation_6d_to_matrix``.
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)  # rows


def _accumulate_rotations(rel: np.ndarray) -> np.ndarray:
    """Left-accumulate relative rotation matrices: ``R_t = R_rel_t @ R_{t-1}``."""
    out = [rel[0]]
    for r in rel[1:]:
        out.append(np.matmul(r, out[-1]))
    return np.asarray(out)


# ============================================================================
# Resampling
# ============================================================================

def linear_resample_positions(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Linearly resample a time-major array ``(T, ...)`` between frame rates.

    Phase-aligned (t=0 maps to t=0). Empirically near-lossless for joints
    (round-trip 20->30->20 rel-RMSE < 0.03 on HumanML3D channels).
    """
    arr = np.asarray(arr)
    T = arr.shape[0]
    if T < 2 or abs(src_fps - dst_fps) < 1e-9:
        return arr.copy()
    duration = (T - 1) / src_fps
    new_T = max(int(round(duration * dst_fps)) + 1, 2)
    src_t = np.arange(T) / src_fps
    dst_t = np.clip(np.arange(new_T) / dst_fps, src_t[0], src_t[-1])
    flat = arr.reshape(T, -1)
    out = np.empty((new_T, flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, flat[:, c])
    return out.reshape((new_T,) + arr.shape[1:])


# ============================================================================
# 272 -> SMPL params -> SMPL-H FK joints
# ============================================================================

_SMPLH_REST_CACHE: dict = {}


def _load_smplh_rest(model_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(j_rest (22,3), parents (22,))`` for SMPL-H neutral, betas=0."""
    key = str(model_path)
    cached = _SMPLH_REST_CACHE.get(key)
    if cached is not None:
        return cached
    d = np.load(str(model_path), allow_pickle=True)
    j_reg = np.asarray(d["J_regressor"], dtype=np.float64)
    v_template = np.asarray(d["v_template"], dtype=np.float64)
    j_rest = (j_reg @ v_template)[:_NJOINT]
    parents = np.asarray(d["kintree_table"][0], dtype=int)[:_NJOINT]
    _SMPLH_REST_CACHE[key] = (j_rest, parents)
    return j_rest, parents


def recover_local_rotations_and_root(m272: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recover per-joint local rotation matrices + world root translation.

    Faithful to MotionStreamer ``representation_272_to_bvh.recover_from_local_rotation``.

    Returns:
        rot:  ``(T, 22, 3, 3)`` local joint rotation matrices (joint 0 = global
              root orientation; joints 1..21 are parent-relative).
        root: ``(T, 3)`` world root translation (xz integrated from the
              heading-removed root velocity; y = stored root height).
    """
    m272 = np.asarray(m272, dtype=np.float64)
    nfrm = m272.shape[0]
    j = _NJOINT
    rot = _rotation_6d_to_matrix(m272[:, 8 + 6 * j:8 + 12 * j].reshape(nfrm, j, 6))
    heading_delta_6d = m272[:, 2:8]
    vel_root_xy = m272[:, :2]
    pos_nh = m272[:, 8:8 + 3 * j].reshape(nfrm, j, 3)
    height = pos_nh[:, 0, 1]

    heading = _accumulate_rotations(_rotation_6d_to_matrix(heading_delta_6d))
    inv_heading = np.transpose(heading, (0, 2, 1))
    rot[:, 0, ...] = np.matmul(inv_heading, rot[:, 0, ...])

    vel = np.zeros((nfrm, 3))
    vel[:, 0] = vel_root_xy[:, 0]
    vel[:, 2] = vel_root_xy[:, 1]
    if nfrm > 1:
        vel[1:] = np.matmul(inv_heading[:-1], vel[1:, :, None]).squeeze(-1)
    root = np.cumsum(vel, axis=0)
    root[:, 1] = height
    return rot, root


def fk_smplh_joints(rot: np.ndarray, root_trans: np.ndarray,
                    smplh_model: Optional[Path] = None) -> np.ndarray:
    """SMPL-H forward kinematics for the 22 body joints.

    Only joint *locations* are computed (rest joints + kinematic-tree FK); no
    LBS / mesh is needed.
    """
    model_path = Path(smplh_model) if smplh_model else DEFAULT_PATHS.resolve("smplh_model")
    j_rest, parents = _load_smplh_rest(model_path)
    T = rot.shape[0]
    Rg = np.zeros((T, _NJOINT, 3, 3))
    pos = np.zeros((T, _NJOINT, 3))
    Rg[:, 0] = rot[:, 0]
    pos[:, 0] = j_rest[0][None] + root_trans
    for jj in range(1, _NJOINT):
        p = parents[jj]
        Rg[:, jj] = np.matmul(Rg[:, p], rot[:, jj])
        off = (j_rest[jj] - j_rest[p])[None, :, None]
        pos[:, jj] = pos[:, p] + np.matmul(Rg[:, p], off).squeeze(-1)
    return pos


def recover_272_to_smplh_joints(m272: np.ndarray,
                                smplh_model: Optional[Path] = None) -> np.ndarray:
    """``humanml3d_272`` clip -> SMPL-H FK global joint positions ``(T, 22, 3)``.

    Recommended joint source for building HumanML3D-263 features (matches the
    official joint source far better than the 272 stored positions).
    """
    rot, root = recover_local_rotations_and_root(m272)
    return fk_smplh_joints(rot, root, smplh_model)


def recover_272_stored_positions(m272: np.ndarray) -> np.ndarray:
    """Decode the 272 *stored* joint positions to world frame ``(T, 22, 3)``.

    Legacy joint source (~30 mm off the official SMPL-H FK joints). Kept for
    A/B comparison; prefer :func:`recover_272_to_smplh_joints`.
    """
    m272 = np.asarray(m272, dtype=np.float64)
    nfrm = m272.shape[0]
    j = _NJOINT
    pos_nh = m272[:, 8:8 + 3 * j].reshape(nfrm, j, 3)
    heading = _accumulate_rotations(_rotation_6d_to_matrix(m272[:, 2:8]))
    inv_heading = np.transpose(heading, (0, 2, 1))
    pd = np.matmul(np.repeat(inv_heading[:, None], j, axis=1), pos_nh[..., None]).squeeze(-1)
    vel_xy = m272[:, :2]
    vel = np.zeros((nfrm, 3))
    vel[:, 0] = vel_xy[:, 0]
    vel[:, 2] = vel_xy[:, 1]
    if nfrm > 1:
        vel[1:] = np.matmul(inv_heading[:-1], vel[1:, :, None]).squeeze(-1)
    root_tr = np.cumsum(vel, axis=0)
    joints = pd.copy()
    joints[:, :, 0] += root_tr[:, 0:1]
    joints[:, :, 2] += root_tr[:, 2:3]
    return joints


# ============================================================================
# MoMask process_file integration (lazy)
# ============================================================================

_MOTION_PROCESS = None
_PROCESS_GLOBALS_READY = False


def _import_motion_process(momask_root: Path):
    """Lazily import MoMask ``utils.motion_process`` and friends."""
    global _MOTION_PROCESS
    if _MOTION_PROCESS is not None:
        return _MOTION_PROCESS
    if str(momask_root) not in sys.path:
        sys.path.insert(0, str(momask_root))
    import utils.motion_process as motion_process  # noqa: E402
    from utils.motion_process import process_file, recover_from_ric  # noqa: E402
    from utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain  # noqa: E402
    from common.skeleton import Skeleton  # noqa: E402
    _MOTION_PROCESS = dict(
        mp=motion_process, process_file=process_file, recover_from_ric=recover_from_ric,
        t2m_raw_offsets=t2m_raw_offsets, t2m_kinematic_chain=t2m_kinematic_chain,
        Skeleton=Skeleton,
    )
    return _MOTION_PROCESS


def setup_process_globals(canonical_ref_joints: Optional[np.ndarray] = None,
                          paths: HumanMLReprPaths = DEFAULT_PATHS) -> None:
    """Configure MoMask ``process_file`` module globals (skeleton, foot ids, ...).

    ``canonical_ref_joints``: ``(>=22, 3)`` first-frame joints that define the
    ``uniform_skeleton`` target. If ``None``, loads the official canonical
    000021 joints from ``paths.canonical_ref_joints``. Using the official
    skeleton is REQUIRED for the official Mean/Std + evaluator to be valid.
    """
    global _PROCESS_GLOBALS_READY
    mod = _import_motion_process(paths.resolve("momask_root"))
    mp = mod["mp"]
    if canonical_ref_joints is None:
        ref = np.load(str(paths.resolve("canonical_ref_joints")))[:, :_NJOINT, :]
        ref_first = np.asarray(ref[0], dtype=np.float64)
    else:
        ref_first = np.asarray(canonical_ref_joints, dtype=np.float64)[:_NJOINT]

    mp.l_idx1, mp.l_idx2 = 5, 8
    mp.fid_l, mp.fid_r = [7, 10], [8, 11]
    mp.face_joint_indx = [2, 1, 17, 16]
    mp.r_hip, mp.l_hip = 2, 1
    mp.joints_num = _NJOINT
    mp.n_raw_offsets = torch.from_numpy(mod["t2m_raw_offsets"])
    mp.kinematic_chain = mod["t2m_kinematic_chain"]
    skel = mod["Skeleton"](mp.n_raw_offsets, mp.kinematic_chain, "cpu")
    mp.tgt_offsets = skel.get_offsets_joints(torch.from_numpy(ref_first).float())
    _PROCESS_GLOBALS_READY = True


def joints_to_humanml263(joints: np.ndarray, *, feet_thre: float = 0.002,
                         paths: HumanMLReprPaths = DEFAULT_PATHS,
                         ensure_globals: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Run MoMask ``process_file`` on 20 fps joints -> ``(m263, joints263)``.

    ``setup_process_globals`` must have been called once (it is auto-invoked
    with the official canonical skeleton when ``ensure_globals`` is True).
    """
    mod = _import_motion_process(paths.resolve("momask_root"))
    if ensure_globals and not _PROCESS_GLOBALS_READY:
        setup_process_globals(paths=paths)
    m263, _, _, _ = mod["process_file"](np.asarray(joints, dtype=np.float32), feet_thre)
    rec = mod["recover_from_ric"](torch.from_numpy(m263).unsqueeze(0).float(), _NJOINT)
    return m263.astype(np.float32), rec.squeeze(0).numpy().astype(np.float32)


# ============================================================================
# Top-level: 272 -> 263
# ============================================================================

def humanml272_to_humanml263(m272: np.ndarray, *, src_fps: float = 30.0,
                             dst_fps: float = 20.0, joints_from: str = "smpl_fk",
                             feet_thre: float = 0.002,
                             paths: HumanMLReprPaths = DEFAULT_PATHS,
                             ensure_globals: bool = True
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert one ``humanml3d_272`` clip to an official HumanML3D-263 feature.

    Args:
        m272: ``(T, 272)`` MotionStreamer representation (30 fps).
        joints_from: ``"smpl_fk"`` (recommended -- SMPL-H FK joints matching the
            official source) or ``"positions"`` (legacy 272 stored positions).
        paths: external asset locations (defaults point at ``ref_repo``).
        ensure_globals: auto-configure ``process_file`` with the official
            canonical skeleton on first call.

    Returns:
        ``(m263 (T', 263), joints263 (T', 22, 3))`` where ``T' = resampled_T - 1``
        (``process_file`` drops one frame).
    """
    if joints_from == "smpl_fk":
        joints30 = recover_272_to_smplh_joints(m272, paths.resolve("smplh_model"))
    elif joints_from == "positions":
        joints30 = recover_272_stored_positions(m272)
    else:
        raise ValueError(f"unknown joints_from={joints_from!r}")
    joints20 = linear_resample_positions(joints30, src_fps, dst_fps)
    return joints_to_humanml263(joints20, feet_thre=feet_thre, paths=paths,
                                ensure_globals=ensure_globals)


# ============================================================================
# M2M model output (135/198-dim, SMPL-22) -> HumanML3D-263
# ============================================================================

# SMPL-22 rest joints (absolute, Y-up) == ``bone_offsets_22.pt`` deltas; loaded
# lazily from the same SMPL-H model.npz used to build the GT, so the predicted
# joints share the EXACT skeleton/up-axis of the GT 272->263 path.
_SMPLH_FK_OFFSETS_CACHE: dict = {}


def _smplh_bone_offsets(model_path=None):
    """Parent-relative SMPL-22 bone offsets ``(22, 3)`` from the SMPL-H rest pose.

    ``offsets[0]`` is the absolute rest-pelvis position; ``offsets[j>0]`` is
    ``j_rest[j] - j_rest[parent[j]]``. Numerically identical (0.0 mm) to the
    repo's ``data/hymotion_m2m_data/bone_offsets_22.pt`` used at train time.
    """
    mp = Path(model_path) if model_path else DEFAULT_PATHS.resolve("smplh_model")
    key = str(mp)
    cached = _SMPLH_FK_OFFSETS_CACHE.get(key)
    if cached is not None:
        return cached
    j_rest, parents = _load_smplh_rest(mp)
    off = np.zeros((_NJOINT, 3), dtype=np.float64)
    off[0] = j_rest[0]
    for j in range(1, _NJOINT):
        off[j] = j_rest[j] - j_rest[parents[j]]
    _SMPLH_FK_OFFSETS_CACHE[key] = off
    return off


def motion198_to_humanml263(motion_denorm, *,
                            rotation_space: str = "local",
                            src_fps: float = 30.0, dst_fps: float = 20.0,
                            feet_thre: float = 0.002,
                            bone_offsets=None,
                            paths: HumanMLReprPaths = DEFAULT_PATHS,
                            ensure_globals: bool = True):
    """Convert a **denormalized** M2M model output to an official HumanML3D-263.

    Prediction-side twin of :func:`humanml272_to_humanml263`: routes the model
    output through the SAME final stages used to build the GT 263 set, so
    skeleton / up-axis / height / orientation are identical:

        model output (>=135-dim: trans3 + 22x6 rot6d, 30 fps)
          -> differentiable FK on the SMPL-H rest skeleton (``motion135_to_fk``,
             which uses the model's own rot6d->matrix convention; do NOT swap in
             the 272 row-major decoder here -- the two 6D conventions differ by
             ~2.0 and would corrupt the pose)   -> world joints (T30, 22, 3), Y-up
          -> linear resample 30 -> 20
          -> ``process_file`` (uniform_skeleton to official canonical 000021 +
             floor on Y + face frame-0 to +Z + IK) -> 263

    Height and orientation are made HumanML3D-consistent by ``process_file``
    itself (floors on Y, rotates frame-0 to +Z), exactly as for the GT clips --
    any constant pelvis offset cancels.

    Args:
        motion_denorm: ``(T, >=135)`` denormalized motion (bundle
            ``denormalize_motion`` output). Only the first 135 dims are used.
        rotation_space: ``"local"`` or ``"global"`` (passed to ``motion135_to_fk``).
        bone_offsets: optional ``(22, 3)`` override; defaults to SMPL-H rest
            offsets (identical to train-time ``bone_offsets_22.pt``).

    Returns:
        ``(m263 (T'', 263), joints263 (T'', 22, 3))``.
    """
    import torch  # local: keep module import cheap
    from hftrainer.pipelines.motion.differentiable_fk import motion135_to_fk

    arr = np.asarray(motion_denorm, dtype=np.float32)
    m135 = torch.from_numpy(arr[:, :135]).float()
    if bone_offsets is None:
        bo = torch.from_numpy(_smplh_bone_offsets(paths.resolve("smplh_model"))).float()
    else:
        bo = torch.as_tensor(bone_offsets).float()
    world_pos, _, _, _ = motion135_to_fk(m135, bo, rotation_space=rotation_space)
    joints30 = world_pos.detach().cpu().numpy().astype(np.float64)
    joints20 = linear_resample_positions(joints30, src_fps, dst_fps)
    return joints_to_humanml263(joints20, feet_thre=feet_thre, paths=paths,
                                ensure_globals=ensure_globals)
