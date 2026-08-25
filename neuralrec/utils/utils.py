from collections.abc import Container
from typing import Any

import torch


class DeferredScalars:
    """Per-step metrics collected without reading them.

    Reading a CUDA tensor's value blocks until everything queued behind it has
    run, so doing it once a step keeps the GPU from ever running ahead. These
    stay tensors until the epoch is over, and then cross the bus together.
    """

    def __init__(self, to_ignore: Container[str] = ()) -> None:
        self._to_ignore = to_ignore
        self._steps: list[tuple[int, dict[str, torch.Tensor | float]]] = []

    @property
    def pending(self) -> list[tuple[int, dict[str, torch.Tensor | float]]]:
        return self._steps

    @property
    def resolved(self) -> bool:
        return not any(
            isinstance(value, torch.Tensor)
            for _, values in self._steps
            for value in values.values()
        )

    def add(self, step: int, values: dict[str, Any]) -> None:
        kept = {
            key: value.detach() if isinstance(value, torch.Tensor) else float(value)
            for key, value in values.items()
            if key not in self._to_ignore and _is_scalar(value)
        }
        if kept:
            self._steps.append((step, kept))

    def drain(self) -> list[tuple[int, dict[str, float]]]:
        steps, self._steps = self._steps, []
        tensors = [
            value
            for _, values in steps
            for value in values.values()
            if isinstance(value, torch.Tensor)
        ]
        numbers = iter(torch.stack(tensors).tolist() if tensors else ())
        return [
            (
                step,
                {
                    key: next(numbers) if isinstance(value, torch.Tensor) else value
                    for key, value in values.items()
                },
            )
            for step, values in steps
        ]

    @staticmethod
    def means(steps: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
        totals: dict[str, list[float]] = {}
        for _, values in steps:
            for key, value in values.items():
                totals.setdefault(key, []).append(value)
        return {key: sum(values) / len(values) for key, values in totals.items()}


def _is_scalar(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.numel() == 1
    return isinstance(value, (int, float))


def to_float(value: Any) -> float | None:
    """The number behind a metric, or ``None`` when there is none."""
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.numel() == 1 else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


TRAIN_RUNNER = "train_runner"
MODEL = "model"

EXTRA_METRICS = "extra_metrics"
# Mean losses use this reduction count for weighted gradient accumulation.
LOSS_DENOMINATOR = "_loss_denominator"


def add_metrics(state: dict, prefix: str, metrics: dict) -> None:
    state.setdefault(EXTRA_METRICS, {}).setdefault(prefix, {}).update(metrics)
