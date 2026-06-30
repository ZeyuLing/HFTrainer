"""Public cross-representation conversion helpers."""

from __future__ import annotations


def hml263_to_joints(m263, joints_num: int = 22):
    from hftrainer.motion.representation.humanml import hml263_to_joints as fn

    return fn(m263, joints_num=joints_num)


def motion135_to_motion272(motion_135, **kwargs):
    from hftrainer.motion.representation.motion272 import motion135_to_272

    return motion135_to_272(motion_135, **kwargs)


def motion272_to_joints(motion_272):
    from hftrainer.motion.representation.motion272 import motion272_to_joints

    return motion272_to_joints(motion_272)


def motion272_to_hml263(motion_272, **kwargs):
    from hftrainer.motion.representation.motion272 import motion272_to_hml263

    return motion272_to_hml263(motion_272, **kwargs)


def hml263_to_motion135(m263, **kwargs):
    from hftrainer.motion.retarget.hml263_smpl import hml263_to_motion135 as fn

    return fn(m263, **kwargs)


def hml263_to_motion272(m263, ik_kwargs: dict | None = None, **kwargs):
    from hftrainer.motion.representation.motion272 import hml263_to_motion272 as fn

    return fn(m263, ik_kwargs=ik_kwargs, **kwargs)


def dart276_to_smpl_params(motion, **kwargs):
    from hftrainer.motion.representation.dart276 import dart276_to_smpl_params as fn

    return fn(motion, **kwargs)


def dart276_to_joints(motion, **kwargs):
    from hftrainer.motion.representation.dart276 import dart276_to_joints as fn

    return fn(motion, **kwargs)


def dart276_to_motion135(motion, **kwargs):
    from hftrainer.motion.representation.dart276 import dart276_to_motion135 as fn

    return fn(motion, **kwargs)


def smpl_params_and_joints_to_dart276(smpl_params, joints, **kwargs):
    from hftrainer.motion.representation.dart276 import smpl_params_and_joints_to_dart276 as fn

    return fn(smpl_params, joints, **kwargs)


def dart276_to_motion272(motion, *, motion135_kwargs: dict | None = None, **kwargs):
    """DART276 -> repository row-major motion_135 -> MotionStreamer/MS272."""

    motion135_kwargs = dict(motion135_kwargs or {})
    motion135_kwargs.setdefault("rotation_convention", "row")
    m135 = dart276_to_motion135(motion, **motion135_kwargs)
    return motion135_to_motion272(m135, **kwargs)


__all__ = [
    "hml263_to_joints",
    "hml263_to_motion135",
    "hml263_to_motion272",
    "motion135_to_motion272",
    "motion272_to_joints",
    "motion272_to_hml263",
    "dart276_to_smpl_params",
    "dart276_to_joints",
    "dart276_to_motion135",
    "dart276_to_motion272",
    "smpl_params_and_joints_to_dart276",
]
