from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from time import perf_counter
from typing import Any

import torch
from torch.utils.data import DataLoader

from neuralrec.data.transforms import move_to_device
from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import EXTRA_METRICS, LOSS_DENOMINATOR, add_metrics, Stateful


class TrainRunner(Stateful):
    _state_fields: tuple[str, ...] = ("current_epoch", "step", "global_step")

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: dict[str, Any] | None = None,
        callbacks: list[Callback] | None = None,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        if (
            not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
            or gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.current_epoch = 0
        self.step = 0
        self.global_step = 0

        # Not a default argument: one shared dict would carry a finished run's
        # metrics into the next.
        state = {} if state is None else state
        callbacks = [] if callbacks is None else callbacks
        state["train_runner"] = self
        state["model"] = model
        state["optimizer"] = optimizer
        self.callbacks = callbacks
        for cb in self.callbacks:
            state[cb.__class__.__name__] = cb
        self.state = state

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def total_steps(self) -> int | None:
        """Optimizer steps the whole run will take, where that is known before
        it starts. A runner that decides its own length as it goes says None,
        and anything scheduled against a horizon has to cope."""
        return None

    def prepare(self) -> None:
        for callback in self.callbacks:
            callback.prepare()

    def discard_prepared_resources(self) -> None:
        return

    def _model_forward(self, batch: Any) -> dict[str, Any]:
        batch = move_to_device(batch, self.device, non_blocking=True)
        out = self.model(batch)
        if not isinstance(out, dict) or "loss" not in out:
            raise ValueError("Model must return a dict with 'loss' key")
        return out

    def _fire_callbacks(self, method: str, *args):
        for cb in self.callbacks:
            getattr(cb, method)(*args)

    def _accumulation_windows(
        self, train_loader: DataLoader
    ) -> Iterator[tuple[Any, ...]]:
        iterator = iter(train_loader)
        while window := tuple(islice(iterator, self.gradient_accumulation_steps)):
            yield window

    @classmethod
    def _batch_size(cls, batch: Any) -> int:
        if isinstance(batch, torch.Tensor):
            if batch.ndim == 0:
                raise ValueError("cannot infer batch size from a scalar tensor")
            return batch.shape[0]
        if isinstance(batch, Mapping):
            sizes = []
            for value in batch.values():
                try:
                    sizes.append(cls._batch_size(value))
                except ValueError:
                    pass
            if sizes and len(set(sizes)) == 1:
                return sizes[0]
            raise ValueError("cannot infer one batch size from mapping values")
        if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
            sizes = [cls._batch_size(value) for value in batch]
            if sizes and len(set(sizes)) == 1:
                return sizes[0]
        raise ValueError(
            "cannot infer loss denominator; return "
            f"{LOSS_DENOMINATOR!r} with the model loss"
        )

    @classmethod
    def _loss_denominator(cls, batch: Any, out: dict[str, Any]) -> float:
        value = out.get(LOSS_DENOMINATOR)
        if value is None:
            value = cls._batch_size(batch)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"{LOSS_DENOMINATOR} must be scalar")
            value = float(value.detach())
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{LOSS_DENOMINATOR} must be finite and non-negative")
        return float(value)

    def _normalize_accumulated_gradients(self, denominator: float) -> None:
        if denominator == 0:
            return
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(denominator)

    def _accumulation_spec(self, batch: Any) -> dict[str, tuple[float, int]] | None:
        model = self.model
        seen: set[int] = set()
        while id(model) not in seen:
            seen.add(id(model))
            method = getattr(model, "accumulation_spec", None)
            if callable(method):
                return method(batch)
            nested = getattr(model, "module", None)
            if nested is None:
                nested = getattr(model, "_orig_mod", None)
            if nested is None:
                return None
            model = nested
        return None

    @staticmethod
    def _component_totals(
        specs: list[dict[str, tuple[float, int]]],
    ) -> dict[str, tuple[float, int]]:
        keys = set(specs[0])
        if any(set(spec) != keys for spec in specs[1:]):
            raise ValueError("accumulation component names must be stable")
        totals = {}
        for name in keys:
            coefficients = {spec[name][0] for spec in specs}
            denominators = [spec[name][1] for spec in specs]
            if len(coefficients) != 1 or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in denominators
            ):
                raise ValueError("invalid accumulation component specification")
            coefficient = coefficients.pop()
            if not math.isfinite(coefficient):
                raise ValueError("accumulation component coefficient must be finite")
            totals[name] = (coefficient, sum(denominators))
        return totals

    def train_epoch(self, epoch: int, train_loader: DataLoader) -> None:
        self.current_epoch = epoch
        self.step = 0
        self.model.train()
        # An eval that skips this epoch must publish nothing, not last epoch's
        # score under this epoch's name.
        self.state.pop(EXTRA_METRICS, None)

        start_time = perf_counter()
        first_step_time = None

        for batches in self._accumulation_windows(train_loader):
            self._fire_callbacks("on_optimizer_step_begin", self.state, batches)
            self.optimizer.zero_grad(set_to_none=True)
            out: dict[str, Any] | None = None
            callback_loss: torch.Tensor | None = None
            if len(batches) == 1:
                out = self._model_forward(batches[0])
                out["loss"].backward()
            else:
                specs = [self._accumulation_spec(batch) for batch in batches]
                if all(spec is not None for spec in specs):
                    component_specs = [spec for spec in specs if spec is not None]
                    totals = self._component_totals(component_specs)
                    reported_components: dict[str, list[torch.Tensor]] = {
                        name: [] for name in totals
                    }
                    for batch, spec in zip(batches, component_specs, strict=True):
                        out = self._model_forward(batch)
                        scaled_components = []
                        for name, (coefficient, denominator) in spec.items():
                            total = totals[name][1]
                            if denominator == 0:
                                reported_components[name].append(
                                    out[name].detach().new_zeros(())
                                )
                                continue
                            scaled_components.append(
                                out[name] * coefficient * denominator / total
                            )
                            reported_components[name].append(
                                out[name].detach() * coefficient * denominator
                            )
                        if scaled_components:
                            sum(scaled_components).backward()
                        else:
                            sum(
                                parameter.sum() * 0
                                for parameter in self.model.parameters()
                            ).backward()
                    callback_loss = sum(
                        (
                            torch.stack(reported_components[name]).sum() / total
                            if total
                            else torch.stack(reported_components[name]).mean() * 0
                        )
                        for name, (_, total) in totals.items()
                    )
                else:
                    if any(spec is not None for spec in specs):
                        raise ValueError(
                            "accumulation specification must exist for every microbatch"
                        )
                    denominator = 0.0
                    weighted_losses: list[torch.Tensor] = []
                    for batch in batches:
                        out = self._model_forward(batch)
                        weight = self._loss_denominator(batch, out)
                        (out["loss"] * weight).backward()
                        weighted_losses.append(out["loss"].detach() * weight)
                        denominator += weight
                    self._normalize_accumulated_gradients(denominator)
                    if denominator:
                        callback_loss = torch.stack(weighted_losses).sum() / denominator
                    else:
                        callback_loss = torch.stack(weighted_losses).mean()
            assert out is not None
            self._fire_callbacks("on_before_optimizer_step", self.state)
            self.optimizer.step()
            callback_out = out
            if LOSS_DENOMINATOR in out or callback_loss is not None:
                callback_out = dict(out)
                callback_out.pop(LOSS_DENOMINATOR, None)
                if callback_loss is not None:
                    callback_out["loss"] = callback_loss
            self._fire_callbacks("on_step_end", self.state, batches[-1], callback_out)

            self.step += 1
            self.global_step += 1

            if first_step_time is None:
                first_step_time = perf_counter() - start_time

        epoch_time = perf_counter() - start_time

        add_metrics(
            self.state,
            "timing",
            {
                "train_epoch_time": epoch_time,
                "train_first_step_time": first_step_time or 0.0,
            },
        )

        self._fire_callbacks("on_epoch_end", self.state)

    def train(self, num_epochs: int, train_loader: DataLoader) -> None:
        self._fire_callbacks("on_train_begin", self.state)

        for epoch in range(num_epochs):
            self.train_epoch(epoch, train_loader)

        self._fire_callbacks("on_train_end", self.state)
