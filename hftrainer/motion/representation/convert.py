"""Top-level motion representation conversion API (the conversion map).

This is the single entry point for converting between the main motion
representations. Each function has an EXPLICIT, documented rot6d convention and
frame rate so you never have to guess which helper to use.

Conversion map
--------------
::

    HML263 (263, 20fps)  --hml263_to_joints-->        joints (T,22,3)
    HML263 (263, 20fps)  --hml263_to_motion135-->     motion_135 (135, 30fps, ROW)   [SMPL IK]
    SMPL-85/raw params   --smpl85_to_motion272-->     MS272 (272, 30fps)             [official SMPL-X FK]
    motion_135 (ROW)     --motion135_to_motion272-->  MS272 (272, 30fps)             [canon272 FK bridge + encode]
    MS272 (272, 30fps)   --motion272_to_hml263-->     HML263 (263, 20fps)            [decode + re-encode]
    MS272 (272, 30fps)   --motion272_to_joints-->     joints (T,22,3)

Recommended MDM/FlowMDM-style 263 -> MS272-eval chain::

    m135 = hml263_to_motion135(m263)          # ROW-major, SMPL IK bridge
    m272 = motion135_to_motion272(m135)       # canon272 skeleton FK + encode
    # -> feed m272 to the MotionStreamer-272 evaluator

No ``repack_col2row`` step is needed: :func:`hml263_to_motion135` emits ROW-major
``motion_135`` and :func:`motion135_to_motion272` consumes ROW-major. See
``docs/motion/representations.md`` for the full rot6d-convention table.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def hml263_to_joints(m263, joints_num: int = 22):
    """HML263 features ``(...,263)`` -> world joints ``(...,22,3)`` (recover_from_ric)."""
    from hftrainer.motion.representation.humanml import hml263_to_joints as _f

    return _f(m263, joints_num)


def hml263_to_motion135(m263: np.ndarray, **kwargs) -> np.ndarray:
    """HML263 ``(T,263)`` -> SMPL ``motion_135 (T,135)`` ROW-major (SMPL IK).

    See :func:`hftrainer.motion.retarget.hml263_smpl.retarget_hml263_clip` for
    all options (fps, refine_iters, rot6d_convention, ...).
    """
    from hftrainer.motion.retarget.hml263_smpl import hml263_to_motion135 as _f

    return _f(m263, **kwargs)


def motion135_to_motion272(
    m135: np.ndarray,
    *,
    rotation_space: str = "local",
    skeleton: str = "canon272",
    bone_offsets: Optional[np.ndarray] = None,
) -> np.ndarray:
    """ROW-major ``motion_135 (T,135)`` -> MS272 ``(T,272)`` bridge.

    Uses the GT-272 canonical skeleton by default (see
    :func:`hftrainer.motion.representation.motion272.motion135_to_272`). This is
    approximate when raw SMPL betas/shape are unavailable; use
    :func:`smpl85_to_motion272` for official raw-SMPL conversion.
    """
    from hftrainer.motion.representation.motion272 import motion135_to_272 as _f

    return _f(m135, rotation_space=rotation_space, skeleton=skeleton, bone_offsets=bone_offsets)


def smpl85_to_motion272(smpl_85: np.ndarray, **kwargs) -> np.ndarray:
    """Raw MotionStreamer-layout SMPL-85 ``(T,85)`` -> MS272 ``(T,272)``.

    This is the official raw-SMPL path: face-Z canonicalization, SMPL-X FK, then
    MotionStreamer-272 encoding. Keyword arguments are forwarded to
    :func:`hftrainer.motion.representation.motion272.smpl85_to_272`.
    """
    from hftrainer.motion.representation.motion272 import smpl85_to_272 as _f

    return _f(smpl_85, **kwargs)


def smpl_params_to_motion272(
    global_orient: np.ndarray,
    body_pose: np.ndarray,
    transl: np.ndarray,
    betas: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Raw SMPL arrays -> MS272 ``(T,272)`` via the official SMPL-X FK path."""
    from hftrainer.motion.representation.motion272 import smpl_params_to_272 as _f

    return _f(global_orient, body_pose, transl, betas, **kwargs)


def motion272_to_hml263(m272: np.ndarray, **kwargs) -> np.ndarray:
    """MS272 ``(T,272)`` @30fps -> HML263 ``(T',263)`` @20fps.

    Delegates to ``humanml272_to_humanml263`` (decode + optional SMPL-H FK +
    MoMask re-encode). The historical default is ``joints_from="smpl_fk"``;
    use ``joints_from="positions"`` when you explicitly want the native MS272
    stored-position decode. Requires the MoMask/SMPL-H assets used by that
    bridge.
    """
    from hftrainer.motion.representation.humanml import humanml272_to_humanml263 as _f

    return _f(m272, **kwargs)


def motion272_to_joints(m272: np.ndarray):
    """MS272 ``(T,272)`` -> 22 joint positions via the stored-position decoder."""
    from hftrainer.motion.representation.humanml import recover_272_stored_positions as _f

    return _f(m272)


def smpl_to_interhuman262(joints_world: np.ndarray, body_pose_aa: np.ndarray, **kwargs):
    """SMPL-X joints ``(T,22,3)`` Y-up + body_pose ``(T,21,3)`` -> IH262 ``(T-1,262)``.

    See :func:`hftrainer.motion.representation.interhuman262.encode_smpl_to_interhuman262`.
    Returns ``(motion, root_quat_init, root_pos_init_xz)``.
    """
    from hftrainer.motion.representation.interhuman262 import encode_smpl_to_interhuman262 as _f

    return _f(joints_world, body_pose_aa, **kwargs)


def smpl_to_interhuman262_pair(joints1, body_pose1, joints2, body_pose2, **kwargs):
    """Two-person SMPL-X -> aligned IH262 pair ``(m1, m2, L)`` (InterGen protocol).

    See :func:`hftrainer.motion.representation.interhuman262.build_pair`.
    """
    from hftrainer.motion.representation.interhuman262 import build_pair as _f

    return _f(joints1, body_pose1, joints2, body_pose2, **kwargs)


def interhuman262_to_joints(m262):
    """IH262 ``(...,262)`` -> canonical joints ``(...,22,3)`` (stored in [0:66])."""
    from hftrainer.motion.representation.interhuman262 import interhuman262_to_joints as _f

    return _f(m262)


def hml263_to_motion272(m263: np.ndarray, *, ik_kwargs: Optional[dict] = None, **enc_kwargs) -> np.ndarray:
    """Compose the full chain HML263 -> motion_135 (IK) -> MS272 (FK+encode).

    Args:
        m263: ``(T,263)`` un-normalized HML263 features.
        ik_kwargs: forwarded to :func:`hml263_to_motion135` (e.g. ``refine_iters``).
        **enc_kwargs: forwarded to :func:`motion135_to_motion272`.
    """
    m135 = hml263_to_motion135(m263, **(ik_kwargs or {}))
    return motion135_to_motion272(m135, **enc_kwargs)


__all__ = [
    "hml263_to_joints",
    "hml263_to_motion135",
    "motion135_to_motion272",
    "smpl85_to_motion272",
    "smpl_params_to_motion272",
    "motion272_to_hml263",
    "motion272_to_joints",
    "hml263_to_motion272",
    "smpl_to_interhuman262",
    "smpl_to_interhuman262_pair",
    "interhuman262_to_joints",
]
