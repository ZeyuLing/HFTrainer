"""Logger hook: logs training metrics at regular intervals."""

import time
from collections import deque
import torch
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
    """Logs loss and other metrics to console.

    The ``by_epoch`` flag controls how ``interval`` is interpreted:

      - ``by_epoch=False``: log every ``interval`` iterations.
      - ``by_epoch=True``: log an epoch summary every ``interval`` epochs,
        **and** log per-iteration metrics every ``iter_interval`` iterations
        within each epoch (default 100).  Set ``iter_interval=0`` to disable
        per-iteration logging and only show epoch summaries.
      - ``by_epoch=None`` (default): auto-inherit from ``train_cfg.by_epoch``.

    Log format (iter-based)::

        step [5/10]  lr=2.00e-05  loss=1.45  data_time=0.01s  train_time=0.12s  eta=0:00:01

    Log format (epoch-based, per-iter)::

        epoch [1/100]  step [50/3216]  lr=2.00e-05  loss=1.45  data_time=0.01s  train_time=0.12s  eta=2:30:00

    Log format (epoch-based, epoch summary)::

        epoch [1/100]  lr=2.00e-05  loss=1.45 (avg)  data_time=0.01s  train_time=0.12s  eta=2:30:00
    """

    priority = 10  # runs early

    def __init__(self, interval: int = 10, by_epoch=None, iter_interval: int = 10):
        self.interval = interval
        self.by_epoch = by_epoch  # None = auto-inherit from train_cfg
        self.iter_interval = iter_interval  # per-iter log interval when by_epoch=True
        self.runner = None
        self._start_time = None

        # Timing tracking
        self._prev_after_iter_time = None
        self._data_end_time = None
        self._iter_times = deque(maxlen=100)  # rolling window for ETA

        # Epoch-based accumulators
        self._epoch_losses = {}     # key -> detached scalar sum
        self._epoch_loss_counts = {}  # key -> number of accumulated values
        self._epoch_iter_count = 0
        self._epoch_start_time = None

    def before_run(self):
        self._start_time = time.time()
        self._prev_after_iter_time = time.time()

    def before_train_epoch(self, epoch: int):
        self._epoch_losses = {}
        self._epoch_loss_counts = {}
        self._epoch_iter_count = 0
        self._epoch_start_time = time.time()

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

        if self.by_epoch:
            # Accumulate local metrics on-device for epoch summary. Calling
            # Tensor.item() here would synchronize the CUDA stream every iter.
            for k, v in output.items():
                self._accumulate_epoch_metric(k, v)
            self._epoch_iter_count += 1

            # Also log per-iteration within each epoch so users get timely
            # feedback (especially for large-epoch training).
            if self.iter_interval and self._epoch_iter_count % self.iter_interval == 0:
                output = self._mean_scalar_output_across_ranks(output)
                self._log(global_step, output, data_time=data_time, train_time=train_time)
        else:
            # Iter-based: log every N iters
            if (global_step + 1) % self.interval == 0:
                output = self._mean_scalar_output_across_ranks(output)
                self._log(global_step, output, data_time=data_time, train_time=train_time)

    def _accumulate_epoch_metric(self, key: str, value):
        """Accumulate scalar metrics without forcing a CPU/GPU sync."""
        try:
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                metric = value.detach().float()
            elif isinstance(value, (int, float)):
                device = self.runner.accelerator.device if self.runner is not None else None
                metric = torch.tensor(float(value), device=device)
            else:
                return

            if key in self._epoch_losses:
                self._epoch_losses[key] = self._epoch_losses[key] + metric
                self._epoch_loss_counts[key] += 1
            else:
                self._epoch_losses[key] = metric.clone()
                self._epoch_loss_counts[key] = 1
        except Exception:
            pass

    def _mean_scalar_output_across_ranks(self, output: dict) -> dict:
        """Mean scalar log metrics across all distributed ranks.

        Training uses per-rank losses for backward; this hook runs after the
        optimizer step and only prepares detached values for logging. Without
        this reduction, multi-GPU logs report rank-0's local mini-batch and can
        look overfit while other ranks remain high.
        """
        if self.runner is None:
            return output

        accelerator = self.runner.accelerator
        if getattr(accelerator, "num_processes", 1) <= 1:
            return output

        reduced = {}
        device = accelerator.device
        for k, v in output.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                metric = v.detach().float()
                if metric.device != device:
                    metric = metric.to(device)
                reduced[k] = accelerator.reduce(metric, reduction="mean")
            elif isinstance(v, (int, float)):
                metric = torch.tensor(float(v), device=device)
                reduced[k] = accelerator.reduce(metric, reduction="mean")
            else:
                reduced[k] = v
        return reduced

    def after_train_epoch(self, epoch: int):
        if not self.by_epoch:
            return
        if (epoch + 1) % self.interval == 0:
            self._log_epoch_summary(epoch)
        # Reset epoch accumulators
        self._epoch_losses = {}
        self._epoch_loss_counts = {}
        self._epoch_iter_count = 0
        self._epoch_start_time = None

    def _log_epoch_summary(self, epoch: int):
        """Print epoch-level summary with averaged metrics."""
        is_main = self.runner is None or self.runner.accelerator.is_main_process

        parts = []
        scalar_metrics = {}

        # Epoch info
        if self.runner is not None:
            train_cfg = self.runner.train_cfg
            max_epochs = train_cfg.get('max_epochs', '?')
            parts.append(f"epoch [{epoch + 1}/{max_epochs}]")
        else:
            parts.append(f"epoch [{epoch + 1}]")

        # LR
        if self.runner is not None:
            for key, sched in self.runner.lr_schedulers.items():
                try:
                    lr = sched.get_last_lr()[0]
                    lr_label = 'lr' if key == 'default' else f'lr_{key}'
                    parts.append(f"{lr_label}={lr:.2e}")
                    scalar_metrics[lr_label] = lr
                except Exception:
                    pass

        # Averaged losses. In distributed runs, reduce the accumulated sums and
        # counts once per epoch instead of synchronizing every training iter.
        for k, total in self._epoch_losses.items():
            count = self._epoch_loss_counts.get(k, 0)
            if not count:
                continue
            try:
                if isinstance(total, torch.Tensor):
                    avg_tensor = total
                    count_tensor = torch.tensor(float(count), device=avg_tensor.device)
                    if self.runner is not None and self.runner.accelerator.num_processes > 1:
                        avg_tensor = self.runner.accelerator.reduce(avg_tensor, reduction="sum")
                        count_tensor = self.runner.accelerator.reduce(count_tensor, reduction="sum")
                    avg_tensor = avg_tensor / count_tensor.clamp_min(1.0)
                    if not is_main:
                        continue
                    avg = avg_tensor.item()
                else:
                    if not is_main:
                        continue
                    avg = float(total) / max(1, count)
                parts.append(f"{k}={avg:.4f}")
                scalar_metrics[k] = avg
            except Exception:
                pass

        if not is_main:
            return

        # Epoch timing
        if self._epoch_start_time is not None:
            epoch_time = time.time() - self._epoch_start_time
            parts.append(f"epoch_time={epoch_time:.1f}s")

        # Average data_time and train_time from iter_times
        if self._iter_times:
            avg_iter = sum(self._iter_times) / len(self._iter_times)
            parts.append(f"avg_iter_time={avg_iter:.2f}s")

        # ETA
        if self._iter_times and self.runner is not None:
            avg_iter = sum(self._iter_times) / len(self._iter_times)
            train_cfg = self.runner.train_cfg
            max_epochs = train_cfg.get('max_epochs', 0)
            try:
                steps_per_epoch = len(self.runner.train_dataloader)
                total_iters = max_epochs * steps_per_epoch
            except (TypeError, AttributeError):
                total_iters = self.runner.global_step
            remaining = max(0, total_iters - self.runner.global_step)
            eta_seconds = remaining * avg_iter
            parts.append(f"eta={_format_eta(eta_seconds)}")

        logger.info("  ".join(parts))

        # Log scalars to tensorboard via accelerator
        if self.runner is not None and scalar_metrics:
            try:
                self.runner.accelerator.log(scalar_metrics, step=self.runner.global_step)
            except Exception:
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
                    lr_label = 'lr' if key == 'default' else f'lr_{key}'
                    parts.append(f"{lr_label}={lr:.2e}")
                    scalar_metrics[lr_label] = lr
                except Exception:
                    pass

        # Losses
        if output:
            for k, v in output.items():
                try:
                    if hasattr(v, 'item'):
                        val = v.item()
                    elif isinstance(v, float):
                        val = v
                    else:
                        continue
                    # Adaptive precision: use scientific notation for very small values
                    if val != 0 and abs(val) < 1e-4:
                        parts.append(f"{k}={val:.2e}")
                    else:
                        parts.append(f"{k}={val:.4f}")
                    scalar_metrics[k] = val
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
