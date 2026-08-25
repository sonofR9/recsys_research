from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from neuralrec.utils import Stateful


class Callback(Stateful):
    def prepare(self) -> None:
        pass

    def on_train_begin(self, state: dict[str, Any]) -> None:
        pass

    def on_train_end(self, state: dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        pass

    def on_step_begin(self, state: dict[str, Any], batch: Any) -> None:
        pass

    def on_optimizer_step_begin(
        self, state: dict[str, Any], batches: Sequence[Any]
    ) -> None:
        self.on_step_begin(state, batches[0])

    def on_step_end(
        self,
        state: dict[str, Any],
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        pass

    def on_before_optimizer_step(self, state: dict[str, Any]) -> None:
        pass

    def every_n_steps(
        self,
        n: int,
        include_step_zero: bool = False,
    ) -> Callback:
        return EveryNStepsCallback(self, n, include_step_zero)

    def ignore_if(self, condition: bool) -> Callback:
        return IgnoreIfCallback(self, condition)


# FIXME(sonofr): add lambda Callback


class EveryNStepsCallback(Callback):
    def __init__(self, callback: Callback, n: int, include_step_zero: bool = False):
        super().__init__()

        self.callback = callback
        self.n = n
        self.include_step_zero = include_step_zero

    def _run_if_step(
        self, state: dict[str, Any], fn: Callable[..., None], *args: Any
    ) -> None:
        step = state["train_runner"].step
        if (step == 0 and self.include_step_zero) or (step != 0 and step % self.n == 0):
            fn(*args)

    def on_train_begin(self, state: dict[str, Any]) -> None:
        self.callback.on_train_begin(state)

    def prepare(self) -> None:
        self.callback.prepare()

    def on_train_end(self, state: dict[str, Any]) -> None:
        self.callback.on_train_end(state)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        self.callback.on_epoch_end(state)

    def on_step_begin(self, state: dict[str, Any], batch: Any) -> None:
        self._run_if_step(state, self.callback.on_step_begin, state, batch)

    def on_optimizer_step_begin(
        self, state: dict[str, Any], batches: Sequence[Any]
    ) -> None:
        self._run_if_step(
            state, self.callback.on_optimizer_step_begin, state, batches
        )

    def on_step_end(
        self,
        state: dict[str, Any],
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        self._run_if_step(state, self.callback.on_step_end, state, batch, out)

    def on_before_optimizer_step(self, state: dict[str, Any]) -> None:
        self._run_if_step(state, self.callback.on_before_optimizer_step, state)


class IgnoreIfCallback(Callback):
    def __init__(self, callback: Callback, condition: bool):
        super().__init__()

        self.callback = callback
        self.condition = condition

    def _run_unless_ignored(
        self, state: dict[str, Any], fn: Callable[..., None], *args: Any
    ) -> None:
        if not self.condition:
            fn(*args)

    def on_train_begin(self, state: dict[str, Any]) -> None:
        self._run_unless_ignored(state, self.callback.on_train_begin, state)

    def prepare(self) -> None:
        if not self.condition:
            self.callback.prepare()

    def on_train_end(self, state: dict[str, Any]) -> None:
        self._run_unless_ignored(state, self.callback.on_train_end, state)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        self._run_unless_ignored(state, self.callback.on_epoch_end, state)

    def on_step_begin(self, state: dict[str, Any], batch: Any) -> None:
        self._run_unless_ignored(state, self.callback.on_step_begin, state, batch)

    def on_optimizer_step_begin(
        self, state: dict[str, Any], batches: Sequence[Any]
    ) -> None:
        self._run_unless_ignored(
            state, self.callback.on_optimizer_step_begin, state, batches
        )

    def on_step_end(
        self,
        state: dict[str, Any],
        batch: Any,
        out: dict[str, Any],
    ) -> None:
        self._run_unless_ignored(state, self.callback.on_step_end, state, batch, out)

    def on_before_optimizer_step(self, state: dict[str, Any]) -> None:
        self._run_unless_ignored(state, self.callback.on_before_optimizer_step, state)
