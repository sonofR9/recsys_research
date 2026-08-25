from pathlib import Path

import torch
from torch import nn

from neuralrec.run.callbacks import CheckpointCallback
from neuralrec.run.train import TrainRunner
from neuralrec.utils import EXTRA_METRICS


def _runner() -> TrainRunner:
    model = nn.Linear(2, 2)
    return TrainRunner(model=model, optimizer=torch.optim.SGD(model.parameters(), 0.1))


def _callback(tmp_path: Path, **overrides) -> CheckpointCallback:
    arguments = {
        "checkpoint_dir": str(tmp_path),
        "run_name": "run",
        "prefix": "best",
        "save_strategy": "best_n",
        "n_checkpoints": 2,
        "metric_name": "recall@100",
        "metric_mode": "max",
    }
    return CheckpointCallback(**{**arguments, **overrides})


def _saved(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "run").glob("best_epoch*.pt"))


def test_selects_on_a_metric_outside_the_validation_prefix(tmp_path: Path) -> None:
    state = _runner().state
    state[EXTRA_METRICS] = {"epoch/val_true": {"recall@100": 0.5}}

    _callback(tmp_path, metric_prefix="epoch/val_true").on_epoch_end(state)

    assert len(_saved(tmp_path)) == 1


def test_an_epoch_that_reported_no_such_metric_saves_nothing(tmp_path: Path) -> None:
    state = _runner().state
    state[EXTRA_METRICS] = {"epoch/val": {"val_loss": 0.5}}

    _callback(tmp_path, metric_prefix="epoch/val_true").on_epoch_end(state)

    assert _saved(tmp_path) == []


def test_metrics_do_not_survive_into_the_next_epoch(tmp_path: Path) -> None:
    runner = _runner()
    runner.state[EXTRA_METRICS] = {"epoch/val_true": {"recall@100": 0.5}}

    runner.train_epoch(1, [])

    assert EXTRA_METRICS not in runner.state or not runner.state[EXTRA_METRICS].get(
        "epoch/val_true"
    )
