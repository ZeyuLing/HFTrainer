"""Build the HFTrainer runner from a registered task configuration."""

from __future__ import annotations


def build_runner_from_cfg(cfg):
    """Build the shared runner from a framework experiment config."""
    from hftrainer.runner.accelerate_runner import AccelerateRunner

    return AccelerateRunner.from_cfg(cfg)


def resolve_trainer(cfg):
    """Resolve the configured class through the trainer registry."""
    trainer_cfg = getattr(cfg, 'trainer', None)
    if trainer_cfg is None:
        raise ValueError('cfg.trainer is required.')
    if hasattr(trainer_cfg, 'to_dict'):
        trainer_cfg = trainer_cfg.to_dict()
    trainer_type = trainer_cfg.get('type')
    if trainer_type is None:
        raise KeyError('cfg.trainer.type is required.')

    from hftrainer.registry import TRAINERS

    if isinstance(trainer_type, str):
        trainer_cls = TRAINERS.get(trainer_type)
    elif any(trainer_type is value for value in TRAINERS.module_dict.values()):
        trainer_cls = trainer_type
    else:
        raise KeyError(
            f"{trainer_type!r} is not explicitly registered in 'trainer'. "
            'Import and register the repository-owned trainer before using it.'
        )
    if trainer_cls is None:
        raise KeyError(
            f"Unknown trainer type {trainer_type!r}. Import its package through "
            "config.custom_imports before building the runner."
        )
    return trainer_cls
