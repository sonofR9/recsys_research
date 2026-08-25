from __future__ import annotations

import atexit
from typing import Any

from torch.utils.tensorboard import SummaryWriter

from neuralrec.run.callbacks.base import Callback
from neuralrec.run.train import TrainRunner
from neuralrec.utils import EXTRA_METRICS, DeferredScalars, to_float


class TensorBoardCallback(Callback):
    _state_fields: tuple[str, ...] = ("_log_dir",)

    def __init__(
        self,
        log_dir: str = "runs",
        to_ignore: list[str] = [],
    ) -> None:
        self._log_dir = log_dir
        self._writer: Any | None = None

        self._to_ignore = to_ignore

        self._step_scalars = DeferredScalars(to_ignore)

        self._writer = SummaryWriter(log_dir=self._log_dir)

        atexit.register(self.__del__)

    def on_step_end(
        self,
        state: dict[str, Any],
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        runner: TrainRunner = state["train_runner"]
        self._step_scalars.add(runner.global_step, out)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        runner: TrainRunner = state["train_runner"]
        step = runner.global_step
        epoch = runner.current_epoch

        self._writer.add_scalar("epoch/epoch", epoch, step)

        steps = self._step_scalars.drain()
        for logged_step, values in steps:
            for key, value in values.items():
                self._writer.add_scalar(f"train/{key}", value, logged_step)

        for key, mean in DeferredScalars.means(steps).items():
            self._writer.add_scalar(f"epoch/train/{key}", mean, step)

        extra_metrics = state.get(EXTRA_METRICS, {})
        for prefix, metrics in extra_metrics.items():
            self._log_metrics(metrics, prefix, step)

    def _log_metrics(
        self, metrics: dict[str, Any] | None, prefix: str, step: int
    ) -> None:
        if metrics is None:
            return

        for key, value in metrics.items():
            if key in self._to_ignore:
                continue

            v = to_float(value)
            if v is None:
                continue

            self._writer.add_scalar(f"{prefix}/{key}", v, step)

    def __del__(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
