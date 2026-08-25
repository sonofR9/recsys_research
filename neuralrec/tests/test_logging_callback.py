from typing import Any

import torch

from neuralrec.run.callbacks import LoggingCallback
from neuralrec.utils import EXTRA_METRICS


class _Runner:
    step = 7
    current_epoch = 1


def _capture(monkeypatch) -> list[str]:
    lines: list[str] = []
    monkeypatch.setattr(
        "neuralrec.run.callbacks.logging.logger",
        type(
            "Logger",
            (),
            {
                "log": staticmethod(
                    lambda level, message, *args: lines.append(
                        str(message).format(*args)
                    )
                )
            },
        )(),
    )
    return lines


def _state() -> dict[str, Any]:
    return {"train_runner": _Runner()}


def test_step_line_reports_the_scalars(monkeypatch) -> None:
    lines = _capture(monkeypatch)

    LoggingCallback().on_step_end(
        _state(), batch=None, out={"loss": torch.tensor(0.25), "accuracy": 0.5}
    )

    assert lines == ["step=7 epoch=1 loss=0.2500 accuracy=0.5000"]


def test_step_line_leaves_out_what_is_not_a_number(monkeypatch) -> None:
    lines = _capture(monkeypatch)

    LoggingCallback().on_step_end(
        _state(),
        batch=None,
        out={"loss": torch.tensor(0.25), "predictions": object()},
    )

    assert lines == ["step=7 epoch=1 loss=0.2500"]


def test_step_line_leaves_out_the_per_row_tensors(monkeypatch) -> None:
    lines = _capture(monkeypatch)

    LoggingCallback().on_step_end(
        _state(),
        batch=None,
        out={
            "loss": torch.tensor(0.25),
            "predictions": torch.tensor([0.1, 0.9]),
            "item_ids": torch.tensor([3, 4]),
        },
    )

    assert lines == ["step=7 epoch=1 loss=0.2500"]


def test_epoch_line_reports_the_metrics_collected_during_it(monkeypatch) -> None:
    lines = _capture(monkeypatch)
    state = _state()
    state[EXTRA_METRICS] = {"epoch/val": {"loss": 0.5, "like_llp": -0.25}}

    LoggingCallback().on_epoch_end(state)

    assert lines == [
        "epoch 1 finished epoch/val.loss=0.5000 epoch/val.like_llp=-0.2500"
    ]
