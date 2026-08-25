from __future__ import annotations

import atexit
from typing import Any

from neuralrec.run.callbacks.base import Callback
from neuralrec.run.train import TrainRunner
from neuralrec.utils import EXTRA_METRICS, DeferredScalars, to_float


class WandbCallback(Callback):
    _state_fields: tuple[str, ...] = ()

    def __init__(
        self,
        run_name: str,
        project: str = "ysda_recsys",
        config: dict[str, Any] | None = None,
        to_ignore: list[str] = [],
    ) -> None:
        import wandb

        self._to_ignore = to_ignore
        self._step_scalars = DeferredScalars(to_ignore)
        self._run = wandb.init(project=project, name=run_name, config=config)

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

        steps = self._step_scalars.drain()
        for logged_step, values in steps:
            self._run.log(
                {f"train/{key}": value for key, value in values.items()},
                step=logged_step,
            )

        scalars: dict[str, float] = {"epoch/epoch": float(epoch)}
        for key, mean in DeferredScalars.means(steps).items():
            scalars[f"epoch/train/{key}"] = mean

        extra_metrics = state.get(EXTRA_METRICS, {})
        for prefix, metrics in extra_metrics.items():
            self._collect_metrics(scalars, metrics, prefix)

        self._run.log(scalars, step=step)

    def _collect_metrics(
        self,
        scalars: dict[str, float],
        metrics: dict[str, Any] | None,
        prefix: str,
    ) -> None:
        if metrics is None:
            return

        for key, value in metrics.items():
            if key in self._to_ignore:
                continue

            v = to_float(value)
            if v is None:
                continue

            scalars[f"{prefix}/{key}"] = v

    def __del__(self) -> None:
        if getattr(self, "_run", None) is not None:
            self._run.finish()
            self._run = None
