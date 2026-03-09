"""Checkpoint hook: saves checkpoints at regular intervals."""

from hftrainer.registry import HOOKS


@HOOKS.register_module()
class CheckpointHook:
    """
    Saves checkpoints at regular intervals.

    Calls runner.save_checkpoint() which in turn:
      - Saves model weights via bundle.state_dict_to_save()
      - Saves full accelerator state (optimizer, scheduler) via accelerator.save_state()
      - Saves training meta (global_step, epoch)
    """

    priority = 80  # runs after logger

    def __init__(
        self,
        interval: int = 1000,
        max_keep_ckpts: int = 3,
        save_last: bool = True,
        by_epoch: bool = False,
    ):
        self.interval = interval
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.by_epoch = by_epoch
        self.runner = None  # injected by AccelerateRunner

    def after_train_iter(self, global_step: int, output: dict = None):
        if not self.by_epoch and (global_step + 1) % self.interval == 0:
            if self.runner is not None:
                self.runner.save_checkpoint()

    def after_train_epoch(self, epoch: int):
        if self.by_epoch and (epoch + 1) % self.interval == 0:
            if self.runner is not None:
                self.runner.save_checkpoint()

    def after_run(self):
        if self.save_last and self.runner is not None:
            self.runner.save_checkpoint()
