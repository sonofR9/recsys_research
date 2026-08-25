from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import EXTRA_METRICS, to_float

if TYPE_CHECKING:
    from neuralrec.run.train import TrainRunner


def _scalars(values: dict[str, Any], prefix: str = "") -> list[str]:
    parts = []
    for key, value in values.items():
        number = to_float(value)
        if number is not None:
            parts.append(f"{prefix}{key}={number:.4f}")
    return parts


class LoggingCallback(Callback):
    def __init__(self, level: str = "INFO") -> None:
        self._level = level.upper()

    def on_step_end(
        self,
        state: dict[str, Any],
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        runner: TrainRunner = state["train_runner"]
        parts = [f"step={runner.step}", f"epoch={runner.current_epoch}", *_scalars(out)]
        logger.log(self._level, " ".join(parts))

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        runner: TrainRunner = state["train_runner"]
        parts = [f"epoch {runner.current_epoch} finished"]
        for prefix, metrics in state.get(EXTRA_METRICS, {}).items():
            parts.extend(_scalars(metrics, prefix=f"{prefix}."))
        logger.log(self._level, " ".join(parts))
