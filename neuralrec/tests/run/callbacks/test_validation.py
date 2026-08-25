from time import sleep

import pytest
import torch

from neuralrec.run.callbacks.validation import ValidationCallback
from neuralrec.utils import DeferredScalars, EXTRA_METRICS, LOSS_DENOMINATOR


class _Runner:
    def __init__(self) -> None:
        self.model = torch.nn.Linear(1, 1)
        self.forward_calls = 0

    def _model_forward(self, batch: float) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        return {
            "loss": torch.tensor(batch),
            "rows": torch.tensor([batch, batch]),
            LOSS_DENOMINATOR: 1,
        }


class _Batches:
    def __init__(self) -> None:
        self.yielded = 0

    def __iter__(self):
        for value in (1.0, 2.0, 3.0):
            self.yielded += 1
            yield value


def test_validation_resolves_metrics_after_all_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _Runner()
    original_drain = DeferredScalars.drain
    drained = False

    def drain(scalars: DeferredScalars) -> list[tuple[int, dict[str, float]]]:
        nonlocal drained
        assert runner.forward_calls == 3
        drained = True
        sleep(0.01)
        return original_drain(scalars)

    monkeypatch.setattr(DeferredScalars, "drain", drain)
    state = {"train_runner": runner}

    ValidationCallback(val_loader=[1.0, 2.0, 3.0]).on_epoch_end(state)

    assert drained
    assert state[EXTRA_METRICS]["epoch/val"] == {"loss": 2.0}
    assert state[EXTRA_METRICS]["timing"]["val_inference_time"] >= 0.01


def test_prepare_starts_loading_without_consuming_the_first_batch() -> None:
    runner = _Runner()
    batches = _Batches()
    callback = ValidationCallback(val_loader=batches)

    callback.prepare()

    assert batches.yielded == 1
    assert runner.forward_calls == 0

    callback.on_epoch_end({"train_runner": runner})

    assert batches.yielded == 3
    assert runner.forward_calls == 3


def test_cached_validation_batches_are_loaded_during_prepare_and_reused() -> None:
    runner = _Runner()
    batches = _Batches()
    callback = ValidationCallback(val_loader=batches, cache_batches=True)

    callback.prepare()
    callback.on_epoch_end({"train_runner": runner})
    callback.on_epoch_end({"train_runner": runner})

    assert batches.yielded == 3
    assert runner.forward_calls == 6
