"""Compatibility shim — implementation moved to the public motion library.

The canonical implementation now lives at
``hftrainer.motion.representation.rotation``. This module re-exports every
public and private symbol so that historical imports such as::

    from hftrainer.models.motion.components.utils.geometry.rotation_convert import (
        rotation_6d_to_matrix, matrix_to_axis_angle, _COL_TO_ROW,
    )

keep working unchanged. Do not add new code here; import from
``hftrainer.motion.representation.rotation`` instead.
"""

from hftrainer.motion.representation.rotation import *  # noqa: F401,F403
from hftrainer.motion.representation import rotation as _rotation

# Mirror *all* names (including private helpers like ``_COL_TO_ROW`` and
# ``_as_numpy``) that some legacy callers import directly.
globals().update(
    {k: getattr(_rotation, k) for k in dir(_rotation) if not k.startswith("__")}
)
del _rotation
