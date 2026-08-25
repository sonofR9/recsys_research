from types import SimpleNamespace

from neuralrec.run.callbacks import EarlyStopping
from neuralrec.utils import EXTRA_METRICS


def _state(score: float | None, epoch: int, metric_name: str) -> dict:
    metrics = {} if score is None else {metric_name: score}
    return {
        "train_runner": SimpleNamespace(current_epoch=epoch),
        EXTRA_METRICS: {"epoch/val_true": metrics},
    }


def test_early_stopping_counts_only_evaluated_non_improving_epochs() -> None:
    stopping = EarlyStopping(
        metric_name="recall@100",
        metric_prefix="epoch/val_true",
        metric_mode="max",
        patience=2,
    )

    stopping.on_epoch_end(_state(None, 0, "recall@100"))
    assert not stopping.should_stop

    for epoch, score in enumerate((0.1, 0.09, 0.08), start=1):
        stopping.on_epoch_end(_state(score, epoch, "recall@100"))

    assert stopping.should_stop
    assert stopping.best_score == 0.1
    assert stopping.best_epoch == 1


def test_early_stopping_resets_patience_after_minimum_improvement() -> None:
    stopping = EarlyStopping(
        metric_name="loss",
        metric_prefix="epoch/val_true",
        metric_mode="min",
        patience=2,
        min_delta=0.1,
    )

    for epoch, score in enumerate((3.0, 3.05, 2.89, 2.95)):
        stopping.on_epoch_end(_state(score, epoch, "loss"))

    assert not stopping.should_stop
    assert stopping.best_score == 2.89
    assert stopping.best_epoch == 2
