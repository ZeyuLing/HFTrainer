"""Motion evaluation metrics for hftrainer."""

from hftrainer.evaluation.motion.mbench_physics import (
    MBenchPhysicsConfig,
    aggregate_mbench_physics,
    compute_mbench_physics_for_file,
    compute_mbench_physics_from_joints,
    evaluate_mbench_physics_dir,
    motion135_to_joints,
    motion272_to_joints,
    table_scaled_metrics,
)


_LEGACY_PHYS_EXPORTS = {
    'compute_phys_metrics',
    'load_motion_data',
    'PHYS_METRICS_CACHE',
}

# Music-to-Dance metrics pull in scipy/librosa lazily; keep them off the eager
# import path so lightweight consumers stay cheap.
_M2D_EXPORTS = {
    'M2DFeatures',
    'aggregate_m2d_metrics',
    'feats_from_joints',
    'extract_music_beats',
    'compute_dance_beats',
    'beat_alignment_score',
    'frechet_distance',
    'diversity',
    'canonicalize_skeleton',
    'bone_lengths',
    'SMPL_PARENTS',
}


def __getattr__(name):
    """Lazy-load optional metric modules on demand.

    ``phys_metrics`` imports optional body-model utilities and heavy training
    dependencies; ``m2d_eval_metrics`` imports scipy/librosa. Keeping them lazy
    lets lightweight consumers, such as web viewers that only need
    ``mbench_physics``, start without pulling those in.
    """

    if name in _LEGACY_PHYS_EXPORTS:
        from hftrainer.evaluation.motion import phys_metrics
        return getattr(phys_metrics, name)
    if name in _M2D_EXPORTS:
        from hftrainer.evaluation.motion import m2d_eval_metrics
        return getattr(m2d_eval_metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'compute_phys_metrics',
    'load_motion_data',
    'PHYS_METRICS_CACHE',
    'MBenchPhysicsConfig',
    'aggregate_mbench_physics',
    'compute_mbench_physics_for_file',
    'compute_mbench_physics_from_joints',
    'evaluate_mbench_physics_dir',
    'motion135_to_joints',
    'motion272_to_joints',
    'table_scaled_metrics',
    'M2DFeatures',
    'aggregate_m2d_metrics',
    'feats_from_joints',
    'extract_music_beats',
    'compute_dance_beats',
    'beat_alignment_score',
    'frechet_distance',
    'diversity',
    'canonicalize_skeleton',
    'bone_lengths',
    'SMPL_PARENTS',
]
