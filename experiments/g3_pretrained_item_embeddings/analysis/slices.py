from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Collection, Mapping, Sequence


@dataclass(frozen=True)
class SliceMetrics:
    axis: str
    name: str
    num_users: int
    num_targets: int
    metrics: tuple[tuple[str, float], ...]
    item_ids: tuple[int, ...] = ()
    user_ids: tuple[int, ...] = ()

    def metric(self, name: str) -> float:
        try:
            return dict(self.metrics)[name]
        except KeyError as error:
            raise ValueError(f"slice {self.axis}/{self.name} has no metric {name!r}") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "name": self.name,
            "num_users": self.num_users,
            "num_targets": self.num_targets,
            "metrics": dict(self.metrics),
            "item_ids": list(self.item_ids),
            "user_ids": list(self.user_ids),
        }


@dataclass(frozen=True)
class RankingSliceReport:
    schema_version: int
    cutoffs: tuple[int, ...]
    slices: tuple[SliceMetrics, ...]

    def slice(self, axis: str, name: str) -> SliceMetrics:
        matches = [
            result
            for result in self.slices
            if result.axis == axis and result.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"unknown slice {axis}/{name}")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "cutoffs": list(self.cutoffs),
            "slices": [result.to_dict() for result in self.slices],
        }


def compute_ranking_slices(
    *,
    rankings: Mapping[int, Sequence[int]],
    relevant_items: Mapping[int, Collection[int]],
    training_item_counts: Mapping[int, int],
    training_history_lengths: Mapping[int, int],
    cutoffs: Sequence[int] = (10, 50, 100),
) -> RankingSliceReport:
    normalized_cutoffs = tuple(int(cutoff) for cutoff in cutoffs)
    if (
        not normalized_cutoffs
        or tuple(sorted(set(normalized_cutoffs))) != normalized_cutoffs
        or normalized_cutoffs[0] < 1
    ):
        raise ValueError("cutoffs must be positive, unique, and increasing")
    catalog = set(training_item_counts)
    users = set(rankings)
    if len(catalog) < 3 or len(users) < 3:
        raise ValueError("ranking slices require at least three items and users")
    if users != set(relevant_items) or users != set(training_history_lengths):
        raise ValueError("evaluation user sets differ")
    if any(type(count) is not int or count < 0 for count in training_item_counts.values()):
        raise ValueError("training item counts must be nonnegative integers")
    if any(
        type(length) is not int or length < 0
        for length in training_history_lengths.values()
    ):
        raise ValueError("training history lengths must be nonnegative integers")

    required_length = min(normalized_cutoffs[-1], len(catalog))
    normalized_rankings: dict[int, tuple[int, ...]] = {}
    normalized_relevance: dict[int, frozenset[int]] = {}
    for user_id in sorted(users):
        ranking = tuple(int(item_id) for item_id in rankings[user_id])
        if len(ranking) < required_length:
            raise ValueError("ranking does not retain every requested cutoff")
        if len(ranking) != len(set(ranking)):
            raise ValueError("ranking contains duplicate items")
        if not set(ranking).issubset(catalog):
            raise ValueError("ranking contains an item outside the mapped catalog")
        targets = frozenset(int(item_id) for item_id in relevant_items[user_id])
        if not targets:
            raise ValueError("every evaluation user must have a relevant target")
        if not targets.issubset(catalog):
            raise ValueError("relevance contains an item outside the mapped catalog")
        normalized_rankings[user_id] = ranking
        normalized_relevance[user_id] = targets

    item_groups = _terciles(
        training_item_counts,
        names=("tail", "mid", "head"),
    )
    user_groups = _terciles(
        training_history_lengths,
        names=("low", "mid", "high"),
    )
    results: list[SliceMetrics] = []
    for name in ("tail", "mid", "head"):
        item_ids = item_groups[name]
        selected_items = set(item_ids)
        sliced_relevance = {
            user_id: targets & selected_items
            for user_id, targets in normalized_relevance.items()
            if targets & selected_items
        }
        results.append(
            _score(
                axis="item_frequency",
                name=name,
                rankings=normalized_rankings,
                relevant_items=sliced_relevance,
                cutoffs=normalized_cutoffs,
                catalog_size=len(catalog),
                recall_only=True,
                item_ids=item_ids,
            )
        )
    for name in ("low", "mid", "high"):
        user_ids = user_groups[name]
        sliced_relevance = {
            user_id: normalized_relevance[user_id] for user_id in user_ids
        }
        results.append(
            _score(
                axis="user_history",
                name=name,
                rankings=normalized_rankings,
                relevant_items=sliced_relevance,
                cutoffs=normalized_cutoffs,
                catalog_size=len(catalog),
                recall_only=False,
                user_ids=user_ids,
            )
        )
    return RankingSliceReport(1, normalized_cutoffs, tuple(results))


def _terciles(
    values: Mapping[int, int],
    *,
    names: tuple[str, str, str],
) -> dict[str, tuple[int, ...]]:
    ordered = sorted(values, key=lambda identifier: (values[identifier], identifier))
    groups: dict[str, list[int]] = {name: [] for name in names}
    for rank, identifier in enumerate(ordered):
        groups[names[min(2, 3 * rank // len(ordered))]].append(identifier)
    return {name: tuple(groups[name]) for name in names}


def _score(
    *,
    axis: str,
    name: str,
    rankings: Mapping[int, tuple[int, ...]],
    relevant_items: Mapping[int, Collection[int]],
    cutoffs: tuple[int, ...],
    catalog_size: int,
    recall_only: bool,
    item_ids: tuple[int, ...] = (),
    user_ids: tuple[int, ...] = (),
) -> SliceMetrics:
    totals: dict[str, float] = {}
    metric_names = ("recall",) if recall_only else (
        "recall",
        "capped_recall",
        "ndcg",
        "mrr",
    )
    for metric_name in metric_names:
        for cutoff in cutoffs:
            totals[f"{metric_name}@{cutoff}"] = 0.0
    coverage = {cutoff: set[int]() for cutoff in cutoffs}
    for user_id, targets_collection in relevant_items.items():
        targets = set(targets_collection)
        ranking = rankings[user_id]
        for cutoff in cutoffs:
            prefix = ranking[:cutoff]
            hit_count = len(set(prefix) & targets)
            totals[f"recall@{cutoff}"] += hit_count / len(targets)
            if not recall_only:
                totals[f"capped_recall@{cutoff}"] += hit_count / min(
                    len(targets), cutoff
                )
                totals[f"ndcg@{cutoff}"] += _ndcg(prefix, targets, cutoff)
                totals[f"mrr@{cutoff}"] += _mrr(prefix, targets)
                coverage[cutoff].update(prefix)
    num_users = len(relevant_items)
    metrics = {
        metric_name: total / num_users if num_users else 0.0
        for metric_name, total in totals.items()
    }
    if not recall_only:
        metrics.update(
            {
                f"coverage@{cutoff}": len(coverage[cutoff]) / catalog_size
                for cutoff in cutoffs
            }
        )
    return SliceMetrics(
        axis=axis,
        name=name,
        num_users=num_users,
        num_targets=sum(len(targets) for targets in relevant_items.values()),
        metrics=tuple(metrics.items()),
        item_ids=item_ids,
        user_ids=user_ids,
    )


def _ndcg(ranking: Sequence[int], targets: set[int], cutoff: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item_id in enumerate(ranking[:cutoff])
        if item_id in targets
    )
    ideal_count = min(len(targets), cutoff)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_count))
    return dcg / ideal


def _mrr(ranking: Sequence[int], targets: set[int]) -> float:
    for rank, item_id in enumerate(ranking, start=1):
        if item_id in targets:
            return 1.0 / rank
    return 0.0
