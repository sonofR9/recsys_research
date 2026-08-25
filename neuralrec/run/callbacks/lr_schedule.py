from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

from neuralrec.run.callbacks.base import Callback
from neuralrec.utils import add_metrics

ScheduleShape = Literal[
    "constant",
    "linear",
    "cosine",
    "inverse_sqrt",
    "step",
    "exponential",
    "polynomial",
    "warmup_stable_decay",
    "power",
]
OptimizerGroupScope = Literal["both", "deep_only"]


class LrSchedule(Callback):
    """Warm up, then decay, every parameter group from the rate it was given.

    Groups are scaled rather than assigned, so an optimizer that runs the item
    table at its own rate keeps that ratio for the whole run.

    ``warmup_fraction`` and ``min_lr_fraction`` are fractions of the schedule
    horizon and of the starting rate. The trainer may keep that horizon fixed
    while extending an early-stopping safety cap; a shape that decays against
    the horizon then ends the run once it is spent, because the schedule it
    declares is finished and the remaining cap would train a different one.
    The ``inverse_sqrt`` timescale can be a fixed number of steps or a fraction
    of the run, and defaults to the length of the warmup.
    """

    _HORIZON_FREE_SHAPES = frozenset({"constant", "inverse_sqrt", "power"})

    def __init__(
        self,
        shape: ScheduleShape = "constant",
        *,
        warmup_fraction: float = 0.0,
        min_lr_fraction: float = 0.0,
        cycles: int = 1,
        timescale_steps: int | None = None,
        timescale_fraction: float | None = None,
        power_exponent: float = -0.51,
        power_transition_tokens: int | None = None,
        optimizer_group_scope: OptimizerGroupScope = "both",
        stop_at_horizon: bool = True,
    ) -> None:
        if timescale_steps is not None and timescale_fraction is not None:
            raise ValueError("set at most one inverse-sqrt timescale")
        if timescale_fraction is not None and not 0 < timescale_fraction <= 1:
            raise ValueError("timescale_fraction must be in (0, 1]")
        if cycles < 1:
            raise ValueError("cycles must be at least 1")
        if shape == "power" and (
            power_transition_tokens is None or power_transition_tokens < 1
        ):
            raise ValueError("a power schedule needs a positive transition token count")
        if shape == "power" and power_exponent >= 0:
            raise ValueError("a power schedule needs a negative exponent")
        if shape == "power" and warmup_fraction:
            raise ValueError("a power schedule does not use fractional warmup")
        if optimizer_group_scope not in {"both", "deep_only"}:
            raise ValueError("optimizer_group_scope must be 'both' or 'deep_only'")
        self._shape = shape
        self._warmup_fraction = warmup_fraction
        self._min_lr_fraction = min_lr_fraction
        self._cycles = cycles
        self._timescale_steps = timescale_steps
        self._timescale_fraction = timescale_fraction
        self._power_exponent = power_exponent
        self._power_transition_tokens = power_transition_tokens
        self._optimizer_group_scope = optimizer_group_scope
        self._stop_at_horizon = stop_at_horizon
        self._power_seen_tokens = 0
        self._warmup_steps = 0
        self._decay_steps = 1
        self._timescale = 1
        self._total_steps = 0
        self._group_lr_trace: dict[str, list[float]] = {}
        self._group_role_base_lr: dict[str, float] = {}
        self.should_stop = False

    @property
    def group_lr_trace(self) -> dict[str, list[float]]:
        return {name: list(values) for name, values in self._group_lr_trace.items()}

    @property
    def resolved_timescale_steps(self) -> int | None:
        return self._timescale if self._shape == "inverse_sqrt" else None

    @property
    def stops_at_horizon(self) -> bool:
        return self._stop_at_horizon

    @property
    def _anneals_over_horizon(self) -> bool:
        return self._shape not in self._HORIZON_FREE_SHAPES

    @property
    def _needs_a_horizon(self) -> bool:
        return (
            self._warmup_fraction > 0
            or self._timescale_fraction is not None
            or self._anneals_over_horizon
        )

    def on_train_begin(self, state: dict[str, Any]) -> None:
        runner = state["train_runner"]
        total = getattr(runner, "lr_schedule_total_steps", None)
        if total is None:
            total = runner.total_steps
        if total is None:
            if self._needs_a_horizon:
                raise ValueError(
                    f"a {self._shape!r} schedule with {self._warmup_fraction:.0%} "
                    "warmup is a fraction of the run, and "
                    f"{type(state['train_runner']).__name__} does not know how "
                    "long it will be. Use 'constant' or 'inverse_sqrt', which "
                    "are defined step by step, or give the trainer a "
                    "`total_steps`"
                )
            total = 0
        self._total_steps = total

        self._warmup_steps = int(total * self._warmup_fraction)
        self._decay_steps = max(1, total - self._warmup_steps - 1)
        fractional_timescale = (
            max(1, int(total * self._timescale_fraction))
            if self._timescale_fraction is not None
            else 0
        )
        self._timescale = (
            self._timescale_steps or fractional_timescale or self._warmup_steps
        )
        if self._shape == "inverse_sqrt" and not self._timescale:
            raise ValueError(
                "an 'inverse_sqrt' schedule decays in units of a timescale and "
                "has none: give it a warmup_fraction, whose length it will "
                "borrow, or a timescale_steps/timescale_fraction of its own"
            )

        groups = state["optimizer"].param_groups
        if self._optimizer_group_scope == "deep_only":
            names = [group.get("schedule_group") for group in groups]
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError(
                    "deep_only scheduling requires every optimizer group to have "
                    "a schedule_group identity"
                )
            if "deep" not in names:
                raise ValueError(
                    "deep_only scheduling requires a 'deep' schedule_group role"
                )
        self._group_lr_trace = {}
        self._group_role_base_lr = {}
        for index, group in enumerate(groups):
            # Where torch's own schedulers keep it, and for the same reason: it
            # rides along in `optimizer.state_dict()`, so a run resumed from a
            # mid-schedule rate can still recover the rate it started from.
            group.setdefault("initial_lr", group["lr"])
            name = group.get("schedule_group", f"group_{index}")
            role = str(name)
            self._group_lr_trace.setdefault(role, [])
            self._group_role_base_lr[role] = max(
                self._group_role_base_lr.get(role, 0.0),
                float(group["initial_lr"]),
            )
        self._power_seen_tokens = max(
            (
                int(group.get("power_seen_tokens", 0))
                for group in state["optimizer"].param_groups
            ),
            default=0,
        )

    def on_step_begin(self, state: dict[str, Any], batch: Any) -> None:
        if self._shape == "power":
            self._power_seen_tokens += self._batch_tokens(batch)
            factor = self._power_factor()
        else:
            factor = self._factor(state["train_runner"].global_step)
        for group in state["optimizer"].param_groups:
            group_factor = factor if self._scheduled(group) else 1.0
            group["lr"] = group["initial_lr"] * group_factor
            if self._shape == "power":
                group["power_seen_tokens"] = self._power_seen_tokens

    def on_optimizer_step_begin(
        self, state: dict[str, Any], batches: Sequence[Any]
    ) -> None:
        if self._shape != "power":
            self.on_step_begin(state, batches[0])
            return
        self._power_seen_tokens += sum(self._batch_tokens(batch) for batch in batches)
        factor = self._power_factor()
        for group in state["optimizer"].param_groups:
            group_factor = factor if self._scheduled(group) else 1.0
            group["lr"] = group["initial_lr"] * group_factor
            group["power_seen_tokens"] = self._power_seen_tokens

    @staticmethod
    def _batch_tokens(batch: Any) -> int:
        try:
            tokens = int(batch["cumulative_lens"][-1])
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError(
                "a power schedule needs a sequence batch with cumulative_lens"
            ) from error
        if tokens < 1:
            raise ValueError("a power schedule received an empty token batch")
        return tokens

    def _power_factor(self) -> float:
        assert self._power_transition_tokens is not None
        ratio = max(1.0, self._power_seen_tokens / self._power_transition_tokens)
        return max(self._min_lr_fraction, ratio**self._power_exponent)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        """Two runs that differ only in their schedule otherwise write identical
        logs, and a schedule that reached no optimizer looks like one that did."""
        groups = state["optimizer"].param_groups
        rates = [group["lr"] for group in groups]
        role_factors: dict[str, list[float]] = {}
        for index, group in enumerate(groups):
            role = str(group.get("schedule_group", f"group_{index}"))
            initial_lr = float(group["initial_lr"])
            current_lr = float(group["lr"])
            factor = current_lr / initial_lr if initial_lr else 1.0
            role_factors.setdefault(role, []).append(factor)
        for role, factors in role_factors.items():
            reference = factors[0]
            if any(
                not math.isclose(factor, reference, rel_tol=1e-9, abs_tol=1e-12)
                for factor in factors[1:]
            ):
                raise RuntimeError(f"optimizer subgroups in {role!r} have diverged rates")
            self._group_lr_trace[role].append(
                self._group_role_base_lr[role] * reference
            )
        add_metrics(state, "epoch/train", {"lr": max(rates, default=0.0)})
        step = state["train_runner"].global_step
        self.should_stop = (
            self._stop_at_horizon
            and self._anneals_over_horizon
            and step + 1 >= self._total_steps
        )

    def _scheduled(self, group: dict[str, Any]) -> bool:
        return (
            self._optimizer_group_scope == "both"
            or group.get("schedule_group") == "deep"
        )

    def _factor(self, step: int) -> float:
        if step < self._warmup_steps:
            return (step + 1) / self._warmup_steps
        decayed = self._decayed(step - self._warmup_steps)
        return self._min_lr_fraction + (1 - self._min_lr_fraction) * decayed

    def _decayed(self, step: int) -> float:
        if self._shape == "constant":
            return 1.0
        if self._shape == "inverse_sqrt":
            return math.sqrt(self._timescale / (self._timescale + step))
        progress = min(1.0, step / self._decay_steps)
        if self._shape == "linear":
            return 1 - progress
        if self._shape == "cosine":
            if progress == 1:
                return 0.0
            cycle_progress = (self._cycles * progress) % 1.0
            return 0.5 * (1 + math.cos(math.pi * cycle_progress))
        if self._shape == "step":
            if progress < 0.5:
                return 1.0
            if progress < 0.75:
                return 0.1
            return 0.01
        if self._shape == "exponential":
            return 0.01**progress
        if self._shape == "polynomial":
            return (1 - progress) ** 2
        if self._shape == "warmup_stable_decay":
            decay_progress = max(0.0, (progress - 0.8) / 0.2)
            return 0.5 * (1 + math.cos(math.pi * decay_progress))
        raise ValueError(f"unknown learning-rate schedule shape {self._shape!r}")
