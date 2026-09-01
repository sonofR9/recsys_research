import math

import pytest

from experiments.g4_future_items.report.selection import (
    RecommenderTrial,
    boundary_direction,
    select_recommender_trial,
)


def _trial(
    name: str,
    recall: float,
    loss: float,
    *,
    batch: int = 512,
    period_count: int | None = None,
    embedding_lr: float = 0.01,
    deep_lr: float = 0.02,
    epochs: int = 20,
    horizon: int = 20,
) -> RecommenderTrial:
    parameters = {
        "batch_size": batch,
        "embedding_learning_rate": embedding_lr,
        "deep_learning_rate": deep_lr,
    }
    if period_count is not None:
        parameters["period_count"] = period_count
    return RecommenderTrial(
        row_id=name,
        run_name=name,
        parameters=parameters,
        validation_recall_at_100=recall,
        validation_loss=loss,
        epochs_trained=epochs,
        horizon_epochs=horizon,
    )


def test_selection_requires_complete_finite_runs_and_applies_total_tie_break() -> None:
    trials = [
        _trial("incomplete", 0.9, 0.1, epochs=19),
        _trial("nonfinite", math.nan, 0.1),
        _trial("higher-loss", 0.3000005, 1.2, batch=128),
        _trial("winner", 0.3, 1.1, batch=512),
        _trial("outside-tie", 0.299998, 0.1, batch=128),
    ]

    assert select_recommender_trial(trials, objective="control").run_name == "winner"


def test_treatment_tie_prefers_smaller_period_count_then_rates() -> None:
    trials = [
        _trial("k4", 0.3, 1.0, period_count=4, embedding_lr=0.001, deep_lr=0.001),
        _trial(
            "k1-high-rate", 0.3, 1.0, period_count=1, embedding_lr=0.02, deep_lr=0.02
        ),
        _trial(
            "k1-low-rate", 0.3, 1.0, period_count=1, embedding_lr=0.01, deep_lr=0.01
        ),
    ]

    assert (
        select_recommender_trial(trials, objective="rq3_learned_hard").run_name
        == "k1-low-rate"
    )


def test_recommender_tie_prefers_shorter_horizon_before_treatment_complexity() -> None:
    trials = [
        _trial("short-k4", 0.3, 1.0, period_count=4, horizon=10, epochs=10),
        _trial("long-k1", 0.3, 1.0, period_count=1, horizon=20),
    ]

    assert (
        select_recommender_trial(trials, objective="rq3_learned_hard").run_name
        == "short-k4"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0001, "lower"),
        (0.0002, "lower"),
        (0.01, None),
        (0.2, "upper"),
        (0.256, "upper"),
    ],
)
def test_boundary_direction_uses_inclusive_outer_ten_percent(
    value: float, expected: str | None
) -> None:
    assert boundary_direction(value, 0.0001, 0.256) == expected
