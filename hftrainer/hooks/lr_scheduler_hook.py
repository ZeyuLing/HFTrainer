"""LR Scheduler hook (legacy, schedulers are handled by AccelerateRunner)."""

from hftrainer.registry import HOOKS


@HOOKS.register_module()
class LRSchedulerHook:
    """
    Placeholder hook for LR scheduler stepping.
    AccelerateRunner steps schedulers directly after each optimizer step,
    so this hook is mostly a no-op but kept for compatibility.
    """

    priority = 20

    def __init__(self, by_epoch: bool = False):
        self.by_epoch = by_epoch
        self.runner = None
