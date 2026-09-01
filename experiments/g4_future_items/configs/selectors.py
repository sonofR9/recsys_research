from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from experiments.g4_future_items.selectors import (
    DAY_SECONDS,
    SelectorConfiguration,
    SelectorMetrics,
)


FAMILIES = ("time", "content", "frequency", "learned")
PERIOD_WIDTHS = (3_600, 21_600, DAY_SECONDS)
MINIMUM_EVENTS = (1, 2, 4)


@dataclass(frozen=True)
class SelectorTrial:
    family: Literal["time", "content", "frequency", "learned"]
    trial_id: int
    configuration: SelectorConfiguration
    boundary_round: int | None = None
    seed: int = 42

    @property
    def run_name(self) -> str:
        suffix = (
            f"boundary_{self.boundary_round}_{self.trial_id:02d}"
            if self.boundary_round is not None
            else f"trial_{self.trial_id:02d}"
        )
        return f"g4_selector_{self.family}_{suffix}_native50m"

    @property
    def sampler_seed(self) -> int:
        return 42 if self.boundary_round is None else 42 + self.boundary_round

    def to_dict(self) -> dict[str, Any]:
        configuration = asdict(self.configuration)
        return {
            "stage": (
                "selector_search_boundary"
                if self.boundary_round is not None
                else "selector_search"
            ),
            "trial_id": self.trial_id,
            "boundary_round": self.boundary_round,
            "run_name": self.run_name,
            **configuration,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SelectorTrialResult:
    trial: SelectorTrial
    metrics: SelectorMetrics
    artifact_sha256: str


def compile_selector_search(seed: int = 42) -> tuple[SelectorTrial, ...]:
    trials: list[SelectorTrial] = []
    for family in FAMILIES:
        generator = np.random.Generator(np.random.PCG64(seed))
        candidates = _categorical_candidates(family)
        selected = generator.choice(len(candidates), size=12, replace=False)
        for trial_id, candidate_index in enumerate(selected, 1):
            values = dict(candidates[int(candidate_index)])
            if family == "learned":
                values["learning_rate"] = _log_uniform(generator, 0.01, 0.2)
                values["l2_regularization"] = _log_uniform(generator, 1e-5, 1.0)
            configuration = SelectorConfiguration(family=family, **values)
            trials.append(
                SelectorTrial(
                    family=family,
                    trial_id=trial_id,
                    configuration=configuration,
                    seed=seed,
                )
            )
    return tuple(trials)


def selector_trial_from_job(job: Mapping[str, Any]) -> SelectorTrial:
    configuration = SelectorConfiguration(
        family=job["family"],
        period_width_seconds=job["period_width_seconds"],
        lookahead_seconds=job["lookahead_seconds"],
        minimum_liked_events=job["minimum_liked_events"],
        time_tolerance_seconds=job["time_tolerance_seconds"],
        frequency_entity=job["frequency_entity"],
        max_leaf_nodes=job["max_leaf_nodes"],
        learning_rate=job["learning_rate"],
        l2_regularization=job["l2_regularization"],
    )
    return SelectorTrial(
        family=job["family"],
        trial_id=job["trial_id"],
        boundary_round=job["boundary_round"],
        seed=job["seed"],
        configuration=configuration,
    )


def compile_learned_boundary(
    entering: SelectorTrial,
    *,
    boundary_round: Literal[1, 2],
) -> tuple[SelectorTrial, ...]:
    if entering.family != "learned":
        raise ValueError("selector boundary applies only to learned trials")
    configuration = entering.configuration
    learning_rate = float(configuration.learning_rate)
    low, high = _expanded_learning_rate_interval(learning_rate, boundary_round)
    generator = np.random.Generator(np.random.PCG64(42 + boundary_round))
    return tuple(
        SelectorTrial(
            family="learned",
            trial_id=trial_id,
            boundary_round=boundary_round,
            seed=42,
            configuration=SelectorConfiguration(
                family="learned",
                period_width_seconds=configuration.period_width_seconds,
                lookahead_seconds=configuration.lookahead_seconds,
                minimum_liked_events=configuration.minimum_liked_events,
                max_leaf_nodes=configuration.max_leaf_nodes,
                learning_rate=_log_uniform(generator, low, high),
                l2_regularization=configuration.l2_regularization,
            ),
        )
        for trial_id in range(1, 5)
    )


def select_family_winner(
    results: Sequence[SelectorTrialResult],
) -> SelectorTrialResult:
    if not results:
        raise ValueError("selector selection requires results")
    families = {result.trial.family for result in results}
    if len(families) != 1:
        raise ValueError("family selection cannot mix selector families")
    usable = [
        result
        for result in results
        if math.isfinite(result.metrics.user_balanced_ndcg_at_10)
    ]
    if not usable:
        raise ValueError("selector family has no finite NDCG result")
    return _select_with_ndcg_tolerance(usable)


def select_strongest_deterministic(
    results: Sequence[SelectorTrialResult],
) -> SelectorTrialResult:
    deterministic = [
        result
        for result in results
        if result.trial.family in {"time", "content", "frequency"}
    ]
    if not deterministic:
        raise ValueError("no deterministic selector results")
    return _select_with_ndcg_tolerance(deterministic, mixed_families=True)


def learning_rate_triggers_boundary(trial: SelectorTrial) -> bool:
    if trial.family != "learned":
        return False
    value = float(trial.configuration.learning_rate)
    position = (math.log(value) - math.log(0.01)) / (math.log(0.2) - math.log(0.01))
    return position <= 0.1 or position >= 0.9


def _categorical_candidates(family: str) -> list[dict[str, Any]]:
    structural: list[tuple[int, int, int]] = []
    for width in PERIOD_WIDTHS:
        if family == "time" and width != DAY_SECONDS:
            lookaheads = (7 * DAY_SECONDS,)
        elif width == DAY_SECONDS:
            lookaheads = (14 * DAY_SECONDS, 28 * DAY_SECONDS)
        else:
            lookaheads = (3 * DAY_SECONDS, 7 * DAY_SECONDS)
        structural.extend(itertools.product((width,), lookaheads, MINIMUM_EVENTS))
    candidates: list[dict[str, Any]] = []
    for width, lookahead, minimum in structural:
        base = {
            "period_width_seconds": width,
            "lookahead_seconds": lookahead,
            "minimum_liked_events": minimum,
        }
        if family == "time":
            for tolerance in (0, 3_600, 7_200):
                candidates.append(base | {"time_tolerance_seconds": tolerance})
        elif family == "frequency":
            for entity in ("item", "artist", "album"):
                candidates.append(base | {"frequency_entity": entity})
        elif family == "learned":
            for leaves in (7, 15, 31):
                candidates.append(base | {"max_leaf_nodes": leaves})
        else:
            candidates.append(base)
    return candidates


def _log_uniform(generator: np.random.Generator, low: float, high: float) -> float:
    return float(math.exp(generator.uniform(math.log(low), math.log(high))))


def _selection_key(
    result: SelectorTrialResult, *, mixed_families: bool = False
) -> tuple[Any, ...]:
    metrics = result.metrics
    configuration = result.trial.configuration
    auroc = metrics.auroc if metrics.auroc is not None else -math.inf
    family_simplicity: tuple[Any, ...]
    if mixed_families:
        family_simplicity = ()
    elif result.trial.family == "time":
        family_simplicity = (configuration.time_tolerance_seconds,)
    elif result.trial.family == "frequency":
        order = {"item": 0, "artist": 1, "album": 2}
        family_simplicity = (order[configuration.frequency_entity],)
    elif result.trial.family == "learned":
        family_simplicity = (
            configuration.max_leaf_nodes,
            configuration.learning_rate,
            configuration.l2_regularization,
        )
    else:
        family_simplicity = ()
    canonical = tuple(sorted(result.trial.to_dict().items(), key=lambda pair: pair[0]))
    return (
        -auroc,
        configuration.lookahead_seconds,
        -configuration.period_width_seconds,
        configuration.minimum_liked_events,
        family_simplicity,
        repr(canonical),
        result.trial.run_name,
    )


def _select_with_ndcg_tolerance(
    results: Sequence[SelectorTrialResult], *, mixed_families: bool = False
) -> SelectorTrialResult:
    best_ndcg = max(result.metrics.user_balanced_ndcg_at_10 for result in results)
    tied = [
        result
        for result in results
        if best_ndcg - result.metrics.user_balanced_ndcg_at_10 <= 1e-12
    ]
    return min(
        tied,
        key=lambda result: _selection_key(result, mixed_families=mixed_families),
    )


def _expanded_learning_rate_interval(
    value: float, boundary_round: int
) -> tuple[float, float]:
    low, high = 0.01, 0.2
    position = (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))
    factor = 4**boundary_round
    if position <= 0.1:
        return low / factor, high
    if position >= 0.9:
        return low, high * factor
    raise ValueError("entering learned winner does not trigger a boundary round")
