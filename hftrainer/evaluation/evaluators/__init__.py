"""Reusable text-to-motion evaluators registered in ``EVALUATORS``.

Two public protocols are wired in:

* :class:`HumanML263Evaluator` (key ``humanml3d_263``) — MoMask / Guo et al.
  evaluator in the native HumanML3D-263 feature space (20 fps).
* :class:`MotionStreamer272Evaluator` (key ``motionstreamer_272``) — the public
  MotionStreamer 272-dim evaluator (30 fps), used for the paper HumanML3D rows.
* :class:`MotionCLIP135Evaluator` (key ``MotionCLIP135Evaluator``) — the public
  MotionCLIP SMPL-135 evaluator with raw-projection no-L2 metrics by default.
* :class:`InterHuman262Evaluator` (key ``InterHuman262Evaluator``) — the
  official InterGen / InterMask InterCLIP evaluator for two-person native-262
  packs.
* :class:`InterXText2MotionEvaluator` (key ``InterXText2MotionEvaluator``) —
  the official Inter-X text2motion HHI evaluator layout.
* :class:`TMRHumanML3DEvaluator` (key ``TMRHumanML3DEvaluator``) — a bridge to
  the official TMR HumanML3D retrieval model.
* :class:`HumanMLM2TEvaluator` (key ``HumanMLM2TEvaluator``) — HumanML3D
  motion-to-text caption metrics (BLEU/ROUGE/CIDEr/BERTScore + semantic
  matching).

Both share the metric primitives in :mod:`t2m_metrics` and ship their own encoder
networks under :mod:`networks` (no ``ref_repo`` import at runtime). Only the
*weights* are external, loaded from ``checkpoints/evaluators/``. The neural
encoders are built lazily on first use, so importing this package stays cheap.
"""

from hftrainer.evaluation.evaluators.humanml3d_263 import HumanML263Evaluator
from hftrainer.evaluation.evaluators.humanml3d_m2t import HumanMLM2TEvaluator
from hftrainer.evaluation.evaluators.interhuman_262 import InterHuman262Evaluator
from hftrainer.evaluation.evaluators.interx_text2motion import InterXText2MotionEvaluator
from hftrainer.evaluation.evaluators.motionclip_135 import MotionCLIP135Evaluator
from hftrainer.evaluation.evaluators.motionstreamer_272 import MotionStreamer272Evaluator
from hftrainer.evaluation.evaluators.tmr_humanml3d import TMRHumanML3DEvaluator

__all__ = [
    "HumanML263Evaluator",
    "HumanMLM2TEvaluator",
    "InterHuman262Evaluator",
    "InterXText2MotionEvaluator",
    "MotionCLIP135Evaluator",
    "MotionStreamer272Evaluator",
    "TMRHumanML3DEvaluator",
]
