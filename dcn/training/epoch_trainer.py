from __future__ import annotations

import math
from collections.abc import Iterator
from itertools import chain
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuralrec.run.callbacks.base import Callback
from neuralrec.run.callbacks.validation import ValidationCallback
from neuralrec.run.train import TrainRunner


class EpochTrainer(TrainRunner):
    """N-epoch trainer that replays one ``train_loader``, resuming mid-run."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        num_epochs: int,
        lr_schedule_horizon_epochs: int | None = None,
        val_loader: DataLoader | None = None,
        val_callback: ValidationCallback | None = None,
        callbacks: list[Callback] | None = None,
        state: dict[str, Any] | None = None,
        prepared_train_iterator: Iterator | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        assert (val_loader is None) == (val_callback is None), (
            "a validation loader needs the callback that consumes it"
        )
        callbacks = list(callbacks or [])
        if val_callback is not None:
            val_callback.val_loader = val_loader
            # First, so the loggers and the best-checkpoint rule read this
            # epoch's metrics rather than the last one's.
            callbacks = [val_callback, *callbacks]
        super().__init__(
            model=model,
            optimizer=optimizer,
            state=state,
            callbacks=callbacks,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.lr_schedule_horizon_epochs = lr_schedule_horizon_epochs
        self._prepared_train_iterator = prepared_train_iterator

    @staticmethod
    def prepare_train_iterator(train_loader: DataLoader) -> Iterator:
        iterator = iter(train_loader)
        try:
            first_batch = next(iterator)
        except StopIteration:
            return iter(())
        return chain((first_batch,), iterator)

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(
            len(self.train_loader) / self.gradient_accumulation_steps
        )

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.num_epochs

    @property
    def lr_schedule_total_steps(self) -> int:
        epochs = self.lr_schedule_horizon_epochs or self.num_epochs
        return self.steps_per_epoch * epochs

    @property
    def next_epoch(self) -> int:
        return self.current_epoch + int(self.global_step > 0)

    def prepare(self) -> None:
        super().prepare()
        if (
            self.next_epoch >= self.num_epochs
            or self._prepared_train_iterator is not None
        ):
            return
        self._prepared_train_iterator = self.prepare_train_iterator(self.train_loader)

    def discard_prepared_resources(self) -> None:
        self._prepared_train_iterator = None

    def train(self) -> None:
        self._fire_callbacks("on_train_begin", self.state)
        for epoch in range(self.next_epoch, self.num_epochs):
            if any(
                getattr(callback, "should_stop", False)
                for callback in self.callbacks
            ):
                break
            for module in self.model.modules():
                set_epoch = getattr(module, "set_epoch", None)
                if set_epoch is not None:
                    set_epoch(epoch)
            loader = (
                self._prepared_train_iterator
                if self._prepared_train_iterator is not None
                else self.train_loader
            )
            self._prepared_train_iterator = None
            self.train_epoch(epoch, loader)
        self._fire_callbacks("on_train_end", self.state)
