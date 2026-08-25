from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import EXTRA_METRICS, to_float


class BestWeights(Callback):
    """Keeps a copy of the best-scoring epoch's weights beside the model.

    A run is judged on its best epoch, not its last, and the epoch that scores
    best is rarely the one training ends on. Held here rather than read back
    from a checkpoint so that reporting the best epoch costs no disk: a
    checkpoint carries both optimizer moments as well, and writing one every
    epoch is most of the time a short run spends outside its training loop.

    The copy stays on the model's own device and is written in place, which
    makes it a second set of weights' worth of memory and a memcpy an epoch.
    Sending it to the host instead costs a pageable transfer and a fresh
    allocation of the whole model, which is most of an epoch's own compute.
    """

    _state_fields: tuple[str, ...] = ("best_score", "best_epoch", "_weights")

    def __init__(
        self,
        metric_name: str,
        metric_prefix: str = "epoch/val",
        metric_mode: Literal["min", "max"] = "max",
    ) -> None:
        self.metric_name = metric_name
        self.metric_prefix = metric_prefix
        self.metric_mode = metric_mode
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self._weights: dict[str, torch.Tensor] | None = None

    def _improves_on_best(self, score: float) -> bool:
        if self.best_score is None:
            return True
        return (
            score > self.best_score
            if self.metric_mode == "max"
            else (score < self.best_score)
        )

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        metrics = state.get(EXTRA_METRICS, {}).get(self.metric_prefix, {})
        if self.metric_name not in metrics:
            return

        score = to_float(metrics[self.metric_name])
        if not self._improves_on_best(score):
            return

        self.best_score = score
        self.best_epoch = state["train_runner"].current_epoch
        self._keep(state["model"].state_dict())

    def _keep(self, weights: dict[str, torch.Tensor]) -> None:
        if self._weights is None:
            self._weights = {
                name: tensor.detach().clone() for name, tensor in weights.items()
            }
            return
        for name, tensor in weights.items():
            self._weights[name].copy_(tensor)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        super().load_state_dict(state_dict)

    def restore(self, model: nn.Module) -> bool:
        if self._weights is None:
            return False
        model.load_state_dict(self._weights)
        return True
