from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Literal, Sequence

BoundaryDirection = Literal["lower", "upper"]


@dataclass(frozen=True)
class RecommenderTrial:
    row_id: str
    run_name: str
    parameters: dict[str, Any]
    validation_recall_at_100: float
    validation_loss: float
    epochs_trained: int
    horizon_epochs: int

    @property
    def usable(self) -> bool:
        return (
            self.epochs_trained == self.horizon_epochs
            and math.isfinite(self.validation_recall_at_100)
            and math.isfinite(self.validation_loss)
        )


def select_recommender_trial(
    trials: Sequence[RecommenderTrial], *, objective: str
) -> RecommenderTrial:
    usable = [trial for trial in trials if trial.usable]
    if not usable:
        raise ValueError("no horizon-complete finite recommender trial")
    best_recall = max(trial.validation_recall_at_100 for trial in usable)
    tied = [
        trial
        for trial in usable
        if best_recall - trial.validation_recall_at_100 <= 1e-6
    ]

    def key(trial: RecommenderTrial) -> tuple[Any, ...]:
        parameters = trial.parameters
        period_count = parameters.get("period_count")
        complexity = period_count if period_count is not None else 0
        embedding_lr = float(parameters["embedding_learning_rate"])
        deep_lr = float(parameters["deep_learning_rate"])
        canonical = json.dumps(
            parameters,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return (
            trial.validation_loss,
            trial.horizon_epochs,
            complexity,
            embedding_lr + deep_lr,
            embedding_lr,
            deep_lr,
            canonical,
            trial.run_name,
        )

    return min(tied, key=key)


def boundary_direction(
    value: float, lower: float, upper: float
) -> BoundaryDirection | None:
    if not all(math.isfinite(candidate) for candidate in (value, lower, upper)):
        raise ValueError("boundary inputs must be finite")
    if not 0 < lower < upper or not lower <= value <= upper:
        raise ValueError("value must lie inside a positive interval")
    normalized = (math.log(value) - math.log(lower)) / (
        math.log(upper) - math.log(lower)
    )
    if normalized <= 0.1:
        return "lower"
    if normalized >= 0.9:
        return "upper"
    return None
