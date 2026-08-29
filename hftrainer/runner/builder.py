"""Select the appropriate training-loop owner from an HFTrainer config."""

from __future__ import annotations


def build_runner_from_cfg(cfg):
    """Build the HFTrainer-owned runner selected by the implementation config.

    Most implementations use :class:`AccelerateRunner`. Tightly coupled
    algorithms such as LTX-Video 2.5 keep their local loop under the matching
    ``hftrainer.trainers.<implementation>`` package; the registered trainer
    declares ``manages_training_loop=True`` and implements
    ``from_framework_config``.
    """

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
    if getattr(trainer_cls, 'manages_training_loop', False):
        factory = getattr(trainer_cls, 'from_framework_config', None)
        if factory is None:
            raise TypeError(
                f"Managed trainer {trainer_cls.__name__} must implement "
                "from_framework_config(cfg)."
            )
        return factory(cfg)

    from hftrainer.runner.accelerate_runner import AccelerateRunner

    return AccelerateRunner.from_cfg(cfg)
