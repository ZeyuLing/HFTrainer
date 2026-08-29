"""Registry-driven pipeline construction from MMEngine configs."""

from __future__ import annotations

import copy


def _plain(value):
    if hasattr(value, 'to_dict'):
        value = value.to_dict()
    return copy.deepcopy(dict(value))


def build_pipeline_from_cfg(cfg):
    """Build ``cfg.model`` and ``cfg.pipeline`` through their registries."""

    model_cfg = getattr(cfg, 'model', None)
    pipeline_cfg = getattr(cfg, 'pipeline', None)
    if model_cfg is None or pipeline_cfg is None:
        raise ValueError('Inference configs require both cfg.model and cfg.pipeline.')

    from hftrainer.registry import MODEL_BUNDLES, PIPELINES

    bundle = MODEL_BUNDLES.build(_plain(model_cfg))
    pipeline_data = _plain(pipeline_cfg)
    pipeline_data['bundle'] = bundle
    return PIPELINES.build(pipeline_data)
