"""Training-loop builders.

Keep Accelerate itself lazy so importing unrelated datasets does not require
the optional runtime before a training command is constructed.
"""

from hftrainer.runner.builder import build_runner_from_cfg

__all__ = ['AccelerateRunner', 'build_runner_from_cfg']


def __getattr__(name):
    if name == 'AccelerateRunner':
        from hftrainer.runner.accelerate_runner import AccelerateRunner

        return AccelerateRunner
    raise AttributeError(name)
