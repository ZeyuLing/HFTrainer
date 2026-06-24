"""Self-contained encoder networks for the text-to-motion evaluators.

These are framework-internal ports of the public evaluator encoders so that
``hftrainer`` never imports from ``ref_repo`` at runtime. Only the *weights*
(loaded from ``checkpoints/``) are external artifacts.

* :mod:`temos_encoders` — DistilBERT + ACTOR encoders for MotionStreamer-272.
* :mod:`t2m_eval_modules` — MoMask / Guo et al. encoders for HumanML3D-263.
"""

from hftrainer.evaluation.evaluators.networks.temos_encoders import (
    ActorAgnosticEncoder,
    DistilbertActorAgnosticEncoder,
)
from hftrainer.evaluation.evaluators.networks.t2m_eval_modules import (
    MotionEncoderBiGRUCo,
    MovementConvEncoder,
    TextEncoderBiGRUCo,
)

__all__ = [
    "ActorAgnosticEncoder",
    "DistilbertActorAgnosticEncoder",
    "MotionEncoderBiGRUCo",
    "MovementConvEncoder",
    "TextEncoderBiGRUCo",
]
