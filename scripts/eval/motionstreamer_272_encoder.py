"""Compatibility shim — MS272 encoder moved to the public motion library.

The canonical implementation now lives at
``hftrainer.motion.representation.motion272``. Import from there in new code::

    from hftrainer.motion.representation.motion272 import (
        encode_smpl_to_272, motion135_to_272,
    )

This shim preserves the historical ``scripts.eval.motionstreamer_272_encoder``
import surface.
"""

from hftrainer.motion.representation.motion272 import (  # noqa: F401
    encode_smpl_to_272,
    motion135_to_272,
    reencode_272_via_stored_positions,
    reencode_272_via_fk,
    _canonical_272_offsets,
    _matrix_to_rotation_6d_rows,
    _rot_yaw,
)
