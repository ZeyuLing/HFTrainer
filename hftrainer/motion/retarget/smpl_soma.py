"""KIMODO/SOMA <-> SMPL retargeting public API.

Compatibility note: this module re-exports the current implementation from
``hftrainer.models.motion.components.retarget``. New code should import from
``hftrainer.motion.retarget``.
"""

from hftrainer.models.motion.components.retarget.smpl_soma import *  # noqa: F401,F403
