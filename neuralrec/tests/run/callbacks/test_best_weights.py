import torch
from torch import nn

from neuralrec.run.callbacks.best_weights import BestWeights
from neuralrec.utils import EXTRA_METRICS


class _Runner:
    def __init__(self) -> None:
        self.current_epoch = 0


def _state(model: nn.Module, score: float | None) -> dict:
    state: dict = {"model": model, "train_runner": _Runner()}
    if score is not None:
        state[EXTRA_METRICS] = {"epoch/val_true": {"recall@100": score}}
    return state


def _weight(model: nn.Module) -> float:
    return float(model.weight.detach().flatten()[0])


def _run(scores: list[float | None]) -> tuple[nn.Module, BestWeights]:
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    keeper = BestWeights(metric_name="recall@100", metric_prefix="epoch/val_true")
    for epoch, score in enumerate(scores):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        state = _state(model, score)
        state["train_runner"].current_epoch = epoch
        keeper.on_epoch_end(state)
    return model, keeper


def test_restores_the_best_scoring_epoch_not_the_last() -> None:
    model, keeper = _run([0.1, 0.9, 0.4])

    assert keeper.restore(model) is True
    assert _weight(model) == 1.0
    assert keeper.best_epoch == 1


def test_min_mode_keeps_the_lowest_score() -> None:
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    keeper = BestWeights(
        metric_name="loss", metric_prefix="epoch/val", metric_mode="min"
    )
    for epoch, score in enumerate([5.0, 1.0, 3.0]):
        with torch.no_grad():
            model.weight.fill_(float(epoch))
        state = {
            "model": model,
            "train_runner": _Runner(),
            EXTRA_METRICS: {"epoch/val": {"loss": score}},
        }
        keeper.on_epoch_end(state)

    keeper.restore(model)
    assert _weight(model) == 1.0


def test_an_epoch_without_the_metric_is_not_a_candidate() -> None:
    model, keeper = _run([0.1, None, None])

    keeper.restore(model)
    assert _weight(model) == 0.0
    assert keeper.best_epoch == 0


def test_restore_reports_when_nothing_was_ever_kept() -> None:
    model, keeper = _run([None])

    assert keeper.restore(model) is False


def test_the_kept_copy_does_not_move_with_the_model() -> None:
    """It is a snapshot, so training on after the best epoch cannot rewrite it."""
    model, keeper = _run([0.9, 0.1])

    with torch.no_grad():
        model.weight.fill_(99.0)
    keeper.restore(model)

    assert _weight(model) == 0.0


def test_state_restores_best_selection_from_before_resume() -> None:
    model, interrupted = _run([0.9, 0.4])
    resumed = BestWeights(
        metric_name="recall@100", metric_prefix="epoch/val_true"
    )
    resumed.load_state_dict(interrupted.state_dict())

    with torch.no_grad():
        model.weight.fill_(2.0)
    state = _state(model, 0.8)
    state["train_runner"].current_epoch = 2
    resumed.on_epoch_end(state)

    assert resumed.best_score == 0.9
    assert resumed.best_epoch == 0
    assert resumed.restore(model) is True
    assert _weight(model) == 0.0


def test_legacy_empty_checkpoint_state_loads_without_selection() -> None:
    keeper = BestWeights(
        metric_name="recall@100", metric_prefix="epoch/val_true"
    )

    keeper.load_state_dict({})

    assert keeper.state_dict() == {
        "best_score": None,
        "best_epoch": None,
        "_weights": None,
    }
