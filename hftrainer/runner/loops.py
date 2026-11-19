"""
Training loop abstractions for EpochBased and IterBased training.
"""

from typing import Optional
from torch.utils.data import DataLoader


class EpochBasedLoop:
    """
    Epoch-based training loop.

    Yields batches and signals when to run validation and save checkpoints.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        max_epochs: int,
        val_interval: int = 1,  # in epochs
        save_interval: Optional[int] = None,  # in epochs, default = val_interval
    ):
        self.dataloader = dataloader
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.save_interval = save_interval if save_interval is not None else val_interval

    def iter_epochs(self):
        """
        Yields (epoch, batch_idx, batch) tuples.
        Also yields special signals via should_val / should_save properties.
        """
        for epoch in range(self.max_epochs):
            for batch_idx, batch in enumerate(self.dataloader):
                yield epoch, batch_idx, batch

    @property
    def total_batches_per_epoch(self) -> int:
        return len(self.dataloader)

    @property
    def total_iters(self) -> int:
        return self.max_epochs * len(self.dataloader)


class IterBasedLoop:
    """
    Iteration-based training loop.

    Cycles through the dataloader indefinitely until max_iters is reached.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        max_iters: int,
        val_interval: int = 1000,   # in iters
        save_interval: Optional[int] = None,  # in iters, default = val_interval
    ):
        self.dataloader = dataloader
        self.max_iters = max_iters
        self.val_interval = val_interval
        self.save_interval = save_interval if save_interval is not None else val_interval

    def iter_batches(self, initial_iterator=None):
        """Yield indexed batches, restarting the dataloader when exhausted."""
        batches = iter(self.dataloader) if initial_iterator is None else initial_iterator
        for index in range(self.max_iters):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(self.dataloader)
                try:
                    batch = next(batches)
                except StopIteration as exc:
                    raise ValueError('Training dataloader does not yield any batches.') from exc
            yield index, batch

    @property
    def total_iters(self) -> int:
        return self.max_iters
