"""Registry-driven pipeline construction from MMEngine configs."""

from __future__ import annotations

import copy
from collections.abc import Mapping


def _plain(value, *, opaque_keys: tuple[str, ...] = ()):
    """Copy config data while preserving explicitly injected runtime objects.

    Config values are normally deep-copied so builders cannot mutate the
    caller's configuration.  Dependency-injection objects such as an LTX
    component store deliberately own locks and caches, so copying them is both
    invalid and contrary to their identity-sharing contract.
    """

    if not isinstance(value, Mapping) and hasattr(value, 'to_dict'):
        value = value.to_dict()
    data = dict(value)
    opaque = set(opaque_keys)
    return {
        key: item if key in opaque else copy.deepcopy(item)
        for key, item in data.items()
    }


def build_pipeline_from_cfg(
    cfg,
    *,
    checkpoint_path: str | None = None,
    device: str | None = None,
    merge_lora: bool = False,
    strict_checkpoint: bool = False,
):
    """Build a config-declared inference pipeline.

    Construction is deliberately independent of the training class name:
    ``cfg.model`` selects the local implementation, ``cfg.pipeline`` owns the
    inference graph, and an optional HFTrainer checkpoint is applied at the
    bundle boundary.
    """

    model_cfg = getattr(cfg, 'model', None)
    pipeline_cfg = getattr(cfg, 'pipeline', None)
    if model_cfg is None or pipeline_cfg is None:
        raise ValueError('Inference configs require both cfg.model and cfg.pipeline.')

    from hftrainer.registry import MODEL_BUNDLES, PIPELINES

    model_data = _plain(model_cfg, opaque_keys=('components',))
    if device is not None and 'device' in model_data:
        model_data['device'] = device
    bundle = MODEL_BUNDLES.build(model_data)
    if checkpoint_path is not None:
        from hftrainer.utils.checkpoint_utils import load_checkpoint

        state = load_checkpoint(checkpoint_path, map_location='cpu')
        bundle.load_state_dict_selective(state, strict=strict_checkpoint)
    if merge_lora:
        bundle.merge_lora_weights()
    if device is not None:
        bundle.to(device)
    bundle.eval()
    pipeline_data = _plain(pipeline_cfg)
    pipeline_data['bundle'] = bundle
    return PIPELINES.build(pipeline_data)
