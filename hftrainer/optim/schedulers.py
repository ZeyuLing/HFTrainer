"""Learning-rate schedules implemented with PyTorch primitives."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def _warmup(step: int, warmup_steps: int) -> float | None:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return None


def build_scheduler(
    name: str,
    optimizer: Optimizer,
    *,
    num_warmup_steps: int = 0,
    num_training_steps: int,
    **kwargs: Any,
):
    """Build a schedule without delegating model runtime to another stack."""

    name = str(name).lower()
    warmup_steps = int(num_warmup_steps)
    training_steps = int(num_training_steps)
    if training_steps <= 0:
        raise ValueError('num_training_steps must be positive.')
    if not 0 <= warmup_steps <= training_steps:
        raise ValueError('num_warmup_steps must be between zero and total steps.')

    if name == 'reduce_lr_on_plateau':
        if warmup_steps:
            raise ValueError('reduce_lr_on_plateau does not support warmup.')
        return ReduceLROnPlateau(optimizer, **kwargs)

    decay_steps = max(1, training_steps - warmup_steps)

    if name == 'constant':
        fn = lambda step: 1.0
    elif name == 'constant_with_warmup':
        fn = lambda step: _warmup(step, warmup_steps) or 1.0
    elif name == 'linear':
        def fn(step):
            warm = _warmup(step, warmup_steps)
            if warm is not None:
                return warm
            return max(0.0, float(training_steps - step) / decay_steps)
    elif name in {'cosine', 'cosine_with_min_lr', 'cosine_warmup_with_min_lr'}:
        cycles = float(kwargs.pop('num_cycles', 0.5))
        min_ratio = kwargs.pop('min_lr_rate', kwargs.pop('min_lr_ratio', None))
        min_lr = kwargs.pop('min_lr', None)
        if min_lr is not None:
            base_lrs = [group['lr'] for group in optimizer.param_groups]
            if not base_lrs or min(base_lrs) <= 0:
                raise ValueError('min_lr requires positive optimizer learning rates.')
            inferred = max(float(min_lr) / lr for lr in base_lrs)
            min_ratio = inferred if min_ratio is None else float(min_ratio)
        floor = float(min_ratio or 0.0)

        def fn(step):
            warm = _warmup(step, warmup_steps)
            if warm is not None:
                return warm
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycles * progress))
            return floor + (1.0 - floor) * max(0.0, cosine)
    elif name == 'cosine_with_restarts':
        cycles = float(kwargs.pop('num_cycles', 1.0))

        def fn(step):
            warm = _warmup(step, warmup_steps)
            if warm is not None:
                return warm
            progress = (step - warmup_steps) / decay_steps
            if progress >= 1.0:
                return 0.0
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((cycles * progress) % 1.0))))
    elif name == 'polynomial':
        power = float(kwargs.pop('power', 1.0))
        lr_end = float(kwargs.pop('lr_end', 1e-7))
        initial_lr = min(float(group['lr']) for group in optimizer.param_groups)
        if initial_lr <= lr_end:
            raise ValueError('lr_end must be smaller than the initial learning rate.')
        end_ratio = lr_end / initial_lr

        def fn(step):
            warm = _warmup(step, warmup_steps)
            if warm is not None:
                return warm
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            return (1.0 - progress) ** power * (1.0 - end_ratio) + end_ratio
    elif name == 'inverse_sqrt':
        timescale = float(kwargs.pop('timescale', warmup_steps or 10_000))
        shift = timescale - warmup_steps

        def fn(step):
            warm = _warmup(step, warmup_steps)
            if warm is not None:
                return warm
            return 1.0 / math.sqrt(max(1.0, (step + shift) / timescale))
    elif name == 'warmup_stable_decay':
        stable_steps = int(kwargs.pop('num_stable_steps', 0))
        decay = int(kwargs.pop('num_decay_steps', training_steps - warmup_steps - stable_steps))
        min_ratio = float(kwargs.pop('min_lr_ratio', 0.0))

        def fn(step):
            warm = _warmup(step, warmup_steps)
            if warm is not None:
                return warm
            if step < warmup_steps + stable_steps:
                return 1.0
            progress = min(1.0, max(0.0, (step - warmup_steps - stable_steps) / max(1, decay)))
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        raise ValueError(f'Unknown local scheduler type: {name}')

    if kwargs:
        unknown = ', '.join(sorted(kwargs))
        raise ValueError(f'Unsupported options for scheduler {name!r}: {unknown}')
    return LambdaLR(optimizer, fn)


__all__ = ['build_scheduler']
