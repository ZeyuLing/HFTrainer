"""Motion task models: PRISM, PRISM-MCM, VerMo, HyMotion-M2M, HyMotion-T2M, HyMotion-UMO, MotionCLIP."""

try:
    from hftrainer.models.motion.prism.bundle import PrismBundle
except Exception:
    PrismBundle = None
try:
    from hftrainer.models.motion.prism.mcm_bundle import PrismMCMBundle
except Exception:
    PrismMCMBundle = None
try:
    from hftrainer.models.motion.vermo.bundle import VermoBundle
except Exception:
    VermoBundle = None
try:
    from hftrainer.models.motion.hymotion_m2m.bundle import HyMotionM2MBundle
except Exception:
    HyMotionM2MBundle = None
try:
    from hftrainer.models.motion.hymotion_t2m.bundle import HyMotionT2MBundle
except Exception:
    HyMotionT2MBundle = None
try:
    from hftrainer.models.motion.hymotion_umo.bundle import HyMotionUMOBundle
except Exception:
    HyMotionUMOBundle = None
try:
    from hftrainer.models.motion.motion_clip.bundle import MotionCLIPBundle
except Exception:
    MotionCLIPBundle = None

__all__ = [
    'PrismBundle', 'PrismMCMBundle', 'VermoBundle',
    'HyMotionM2MBundle', 'HyMotionT2MBundle', 'HyMotionUMOBundle',
    'MotionCLIPBundle',
]
