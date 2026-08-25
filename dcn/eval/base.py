from abc import abstractmethod
from collections import defaultdict
from typing import Any, Iterable

import torch
from torch import nn

from neuralrec.data.transforms import move_to_device
from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import add_metrics


class EpochEvalCallback(Callback):
    def __init__(
        self, *, model: nn.Module, prefix: str, every_n_epochs: int = 1
    ) -> None:
        assert every_n_epochs >= 1, "every_n_epochs counts epochs, so it starts at 1"
        self.model = model
        self.prefix = prefix
        self.every_n_epochs = every_n_epochs
        self._epochs_seen = 0

    @property
    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    @abstractmethod
    def _evaluate(self) -> dict[str, float] | None:
        """Metrics to log, or ``None`` when this epoch had nothing to measure."""

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        self._epochs_seen += 1
        if self._epochs_seen % self.every_n_epochs:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            metrics = self._evaluate()
        finally:
            self.model.train(was_training)

        if metrics:
            add_metrics(state, self.prefix, metrics)


class ScoredBatchesCallback(EpochEvalCallback):
    """Scores a loader batch by batch, pooling rows rather than averaging batches."""

    def __init__(
        self,
        *,
        model: nn.Module,
        loader: Iterable[dict],
        prefix: str,
        # Scoring runs outside the training model's AutoCast wrapper.
        dtype: torch.dtype = torch.float32,
        max_batches: int | None = None,
        every_n_epochs: int = 1,
    ) -> None:
        super().__init__(model=model, prefix=prefix, every_n_epochs=every_n_epochs)
        self.loader = loader
        self.dtype = dtype
        self.max_batches = max_batches

    @abstractmethod
    def _batch_scores(self, batch: dict) -> dict[str, torch.Tensor]: ...

    def _evaluate(self) -> dict[str, float] | None:
        scores: dict[str, list[torch.Tensor]] = defaultdict(list)
        with (
            torch.inference_mode(),
            torch.autocast(
                self._device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
            ),
        ):
            for index, batch in enumerate(self.loader):
                if index == self.max_batches:
                    break
                batch = move_to_device(batch, self._device)
                for name, score in self._batch_scores(batch).items():
                    scores[name].append(score)

        metrics = {
            name: float(torch.cat(values).mean())
            for name, values in scores.items()
            if sum(value.numel() for value in values)
        }
        return metrics or None
