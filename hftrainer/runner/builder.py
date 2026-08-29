"""Select the appropriate training-loop owner from an HFTrainer config."""

from __future__ import annotations


def build_runner_from_cfg(cfg):
    """Build a native HFTrainer runner or a managed external trainer.

    Most tasks use :class:`AccelerateRunner`.  Algorithm stacks such as LTX-2.5
    already provide a complete, tightly coupled Accelerator loop; their
    registered trainer sets ``manages_training_loop=True`` and implements
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

    trainer_cls = TRAINERS.get(trainer_type) if isinstance(trainer_type, str) else trainer_type
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
