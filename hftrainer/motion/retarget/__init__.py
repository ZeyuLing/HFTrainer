"""Public retargeting APIs.

- ``smpl_soma``: SMPL motion_135 <-> SOMA30 and KIMODO/SOMA positions -> SMPL.
- ``smpl_g1``: SMPL/SMPL-H/SMPL-X -> Unitree G1 via ``GMRSMPLToG1Retargeter``
    (GMR mink inverse kinematics, ground-aligned Z-up output for
    visualization/deployment). The old fast analytic Euler-decomposition backend
    was removed (low quality / broken poses).
- ``hml263_smpl``: HumanML3D-263 -> SMPL motion_135 (inverse kinematics).
"""

from hftrainer.motion.retarget.smpl_soma import *  # noqa: F401,F403
from hftrainer.motion.retarget.smpl_g1 import *  # noqa: F401,F403

try:
    from hftrainer.motion.retarget.hml263_smpl import (  # noqa: F401
        hml263_to_motion135,
        retarget_hml263_clip,
    )
except ModuleNotFoundError:
    pass
