from __future__ import annotations

import math
from typing import Any, Literal

from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import EXTRA_METRICS, to_float


class EarlyStopping(Callback):
    _state_fields = ("best_score", "best_epoch", "bad_epochs", "should_stop")

    def __init__(
        self,
        metric_name: str,
        patience: int,
        metric_prefix: str = "epoch/val",
        metric_mode: Literal["min", "max"] = "max",
        min_delta: float = 0.0,
    ) -> None:
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 1:
            raise ValueError("patience must be a positive integer")
        if metric_mode not in {"min", "max"}:
            raise ValueError("metric_mode must be 'min' or 'max'")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        self.metric_name = metric_name
        self.metric_prefix = metric_prefix
        self.metric_mode = metric_mode
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.bad_epochs = 0
        self.should_stop = False

    def _improves_on_best(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.metric_mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        metrics = state.get(EXTRA_METRICS, {}).get(self.metric_prefix, {})
        if self.metric_name not in metrics:
            return

        score = to_float(metrics[self.metric_name])
        if self._improves_on_best(score):
            self.best_score = score
            self.best_epoch = state["train_runner"].current_epoch
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        self.should_stop = self.bad_epochs >= self.patience
