"""Motion task model bundles.

This package intentionally keeps bundle imports lazy. Public motion-domain
utilities such as retargeting and rotation conversion should not pay the cost of
loading model bundles, text encoders, or optional training dependencies.
"""

_BUNDLE_IMPORTS = {
    'PrismBundle': ('hftrainer.models.motion.prism.bundle', 'PrismBundle'),
    'PrismMCMBundle': ('hftrainer.models.motion.prism.mcm_bundle', 'PrismMCMBundle'),
    'VermoBundle': ('hftrainer.models.motion.vermo.bundle', 'VermoBundle'),
    'HyMotionM2MBundle': ('hftrainer.models.motion.hymotion_m2m.bundle', 'HyMotionM2MBundle'),
    'HyMotionT2MBundle': ('hftrainer.models.motion.hymotion_t2m.bundle', 'HyMotionT2MBundle'),
    'HyMotionUMOBundle': ('hftrainer.models.motion.hymotion_umo.bundle', 'HyMotionUMOBundle'),
    'MotionCLIPBundle': ('hftrainer.models.motion.motion_clip.bundle', 'MotionCLIPBundle'),
    # Reproduced open-source T2M baselines (Model Zoo; native runtime)
    'MDMBundle': ('hftrainer.models.motion.mdm.bundle', 'MDMBundle'),
    'MotionStreamerBundle': ('hftrainer.models.motion.motionstreamer.bundle', 'MotionStreamerBundle'),
    'FlowMDMBundle': ('hftrainer.models.motion.flowmdm.bundle', 'FlowMDMBundle'),
    'MotionLabBundle': ('hftrainer.models.motion.motionlab.bundle', 'MotionLabBundle'),
    'MotionMillionBundle': ('hftrainer.models.motion.motionmillion.bundle', 'MotionMillionBundle'),
    'T2MGPTBundle': ('hftrainer.models.motion.t2mgpt.bundle', 'T2MGPTBundle'),
    'MotionGPT3Bundle': ('hftrainer.models.motion.motiongpt3.bundle', 'MotionGPT3Bundle'),
    'MoMaskBundle': ('hftrainer.models.motion.momask.bundle', 'MoMaskBundle'),
    'MoGenTSBundle': ('hftrainer.models.motion.mogents.bundle', 'MoGenTSBundle'),
    'MotionLCMBundle': ('hftrainer.models.motion.motionlcm.bundle', 'MotionLCMBundle'),
    'KIMODOBundle': ('hftrainer.models.motion.kimodo.bundle', 'KIMODOBundle'),
    'InterGenBundle': ('hftrainer.models.motion.intergen.bundle', 'InterGenBundle'),
    'InterMaskBundle': ('hftrainer.models.motion.intermask.bundle', 'InterMaskBundle'),
}


def __getattr__(name):
    if name not in _BUNDLE_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _BUNDLE_IMPORTS[name]
    try:
        module = __import__(module_name, fromlist=[attr_name])
        value = getattr(module, attr_name)
    except Exception:
        value = None
    globals()[name] = value
    return value

__all__ = [
    'PrismBundle', 'PrismMCMBundle', 'VermoBundle',
    'HyMotionM2MBundle', 'HyMotionT2MBundle', 'HyMotionUMOBundle',
    'MotionCLIPBundle',
    'MDMBundle', 'MotionStreamerBundle', 'FlowMDMBundle', 'MotionLabBundle',
    'MotionMillionBundle', 'T2MGPTBundle', 'MotionGPT3Bundle',
    'MoMaskBundle', 'MoGenTSBundle', 'MotionLCMBundle', 'KIMODOBundle',
    'InterGenBundle', 'InterMaskBundle',
]
