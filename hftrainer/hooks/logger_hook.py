"""Logger hook: logs training metrics at regular intervals."""

import time
from collections import deque
from hftrainer.registry import HOOKS
from hftrainer.utils.logger import get_logger

logger = get_logger()


def _format_eta(seconds: float) -> str:
    """Format seconds into human-readable ETA string."""
    if seconds < 0:
        return '0:00:00'
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return f'{days}d {hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{hours}:{minutes:02d}:{seconds:02d}'


@HOOKS.register_module()
class LoggerHook:
    """Logs loss and other metrics to console every N iterations.

    Log format (iter-based):
        step [5/10]  lr=2.00e-05  loss=1.45  data_time=0.01s  train_time=0.12s  eta=0:00:01

    Log format (epoch-based):
        epoch [1/100] step [5/200]  lr=2.00e-05  loss=1.45  data_time=0.01s  train_time=0.12s  eta=2:30:00
    """

    priority = 10  # runs early

    def __init__(self, interval: int = 10, by_epoch: bool = False):
        self.interval = interval
        self.by_epoch = by_epoch
        self.runner = None
        self._start_time = None

        # Timing tracking
        self._prev_after_iter_time = None
        self._data_end_time = None
        self._iter_times = deque(maxlen=100)  # rolling window for ETA

    def before_run(self):
        self._start_time = time.time()
        self._prev_after_iter_time = time.time()

    def before_train_iter(self, global_step: int):
        """Called just before train_step. Data loading happened between
        previous after_train_iter and now."""
        self._data_end_time = time.time()

    def after_train_iter(self, global_step: int, output: dict = None):
        now = time.time()

        # Compute timing
        data_time = None
        train_time = None
        if self._data_end_time is not None and self._prev_after_iter_time is not None:
            data_time = self._data_end_time - self._prev_after_iter_time
        if self._data_end_time is not None:
            train_time = now - self._data_end_time

        # Record total iter time for ETA
        if self._prev_after_iter_time is not None:
            self._iter_times.append(now - self._prev_after_iter_time)

        self._prev_after_iter_time = now

        if output is None:
            return
        if not self.by_epoch and (global_step + 1) % self.interval == 0:
            self._log(global_step, output, data_time=data_time, train_time=train_time)

    def after_train_epoch(self, epoch: int):
        pass

    def _log(self, step: int, output: dict, data_time=None, train_time=None):
        if self.runner is not None and not self.runner.accelerator.is_main_process:
            return

        parts = []

        # Step / epoch info
        if self.runner is not None:
            train_cfg = self.runner.train_cfg
            by_epoch = train_cfg.get('by_epoch', False)

            if by_epoch:
                max_epochs = train_cfg.get('max_epochs', '?')
                current_epoch = self.runner.current_epoch + 1
                parts.append(f"epoch [{current_epoch}/{max_epochs}]")
                # Try to compute per-epoch step
                try:
                    steps_per_epoch = len(self.runner.train_dataloader)
                    step_in_epoch = (step % steps_per_epoch) + 1
                    parts.append(f"step [{step_in_epoch}/{steps_per_epoch}]")
                except (TypeError, AttributeError):
                    parts.append(f"step [{step+1}]")
            else:
                max_iters = train_cfg.get('max_iters', '?')
                parts.append(f"step [{step+1}/{max_iters}]")
        else:
            parts.append(f"step [{step+1}]")

        # LR
        scalar_metrics = {}
        if self.runner is not None:
            for key, sched in self.runner.lr_schedulers.items():
                try:
                    lr = sched.get_last_lr()[0]
                    parts.append(f"lr={lr:.2e}")
                    scalar_metrics['lr'] = lr
                except Exception:
                    pass

        # Losses
        if output:
            for k, v in output.items():
                try:
                    if hasattr(v, 'item'):
                        val = v.item()
                        parts.append(f"{k}={val:.4f}")
                        scalar_metrics[k] = val
                    elif isinstance(v, float):
                        parts.append(f"{k}={v:.4f}")
                        scalar_metrics[k] = v
                except Exception:
                    pass

        # Data time
        if data_time is not None:
            parts.append(f"data_time={data_time:.2f}s")

        # Train time
        if train_time is not None:
            parts.append(f"train_time={train_time:.2f}s")

        # ETA
        if self._iter_times and self.runner is not None:
            avg_iter = sum(self._iter_times) / len(self._iter_times)
            train_cfg = self.runner.train_cfg
            if train_cfg.get('by_epoch', False):
                max_epochs = train_cfg.get('max_epochs', 0)
                try:
                    steps_per_epoch = len(self.runner.train_dataloader)
                    total_iters = max_epochs * steps_per_epoch
                except (TypeError, AttributeError):
                    total_iters = step + 1
                remaining = max(0, total_iters - (step + 1))
            else:
                max_iters = train_cfg.get('max_iters', step + 1)
                remaining = max(0, max_iters - (step + 1))
            eta_seconds = remaining * avg_iter
            parts.append(f"eta={_format_eta(eta_seconds)}")

        logger.info("  ".join(parts))

        # Log scalars to tensorboard via accelerator
        if self.runner is not None and scalar_metrics:
            try:
                self.runner.accelerator.log(scalar_metrics, step=step + 1)
            except Exception:
                pass
