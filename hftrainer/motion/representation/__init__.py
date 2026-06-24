"""Motion representation utilities (import-light public API).

Submodules:

- ``rotation``: rot6d/axis-angle/quaternion/matrix/euler conversion, with an
  explicit ``Rot6DConvention`` and ``repack_6d`` helper.
- ``specs``: the single source of truth for representation layouts
  (HML263, MS272, IH262, motion_135/138/198/201/147/151) — dims, fps, body model,
  rot6d convention, normalization stats, channel slices.

Conversion functions between representations (HML263 <-> motion_135 <-> MS272,
SMPL-X -> IH262) live in :mod:`hftrainer.motion.representation.convert`.
"""

from hftrainer.motion.representation.rotation import *  # noqa: F401,F403
from hftrainer.motion.representation.specs import (  # noqa: F401
    FieldSpec,
    MotionRepr,
    get_spec,
    list_specs,
    infer_spec_from_dim,
    REGISTRY as MOTION_SPECS,
)
