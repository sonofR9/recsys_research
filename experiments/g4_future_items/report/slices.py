from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

from dcn.eval.ranking_metrics import (
    capped_recall_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


_FINAL_INTERVAL_SECONDS = 7 * 24 * 60 * 60
_CUTOFFS = (10, 50, 100)


@dataclass(frozen=True, order=True)
class RelevanceEvent:
    user_id: int
    item_id: int
    timestamp: int


def slice_metrics(
    *,
    rankings: Mapping[int, Sequence[int]],
    relevance_events: Collection[RelevanceEvent],
    training_like_counts: Mapping[int, int],
    cutoff_timestamp: int,
    catalog_size: int,
) -> dict[str, object]:
    users = set(rankings)
    if users != set(training_like_counts):
        raise ValueError("evaluation user sets differ between rankings and activity")
    if not users:
        raise ValueError("slice evidence requires evaluation users")
    if catalog_size < 1:
        raise ValueError("catalog size must be positive")

    required_ranking_length = min(max(_CUTOFFS), catalog_size)
    normalized_rankings: dict[int, tuple[int, ...]] = {}
    for user_id, ranked_items in rankings.items():
        ranking = tuple(int(item_id) for item_id in ranked_items)
        if len(ranking) < required_ranking_length:
            raise ValueError("rankings do not retain every required cutoff")
        if len(ranking) != len(set(ranking)):
            raise ValueError("rankings contain duplicate ranked items")
        normalized_rankings[int(user_id)] = ranking

    by_user: dict[int, list[RelevanceEvent]] = defaultdict(list)
    final_end = cutoff_timestamp + _FINAL_INTERVAL_SECONDS
    for event in relevance_events:
        if event.user_id not in users:
            raise ValueError("relevance event has no evaluation ranking")
        if not cutoff_timestamp < event.timestamp <= final_end:
            raise ValueError("relevance event is outside the final seven-day interval")
        by_user[event.user_id].append(event)
    if set(by_user) != users:
        raise ValueError("an evaluation user has no relevance event")

    selected: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for user_id, events in by_user.items():
        for event_rank, event in enumerate(
            sorted(events, key=lambda value: (value.timestamp, value.item_id)),
            start=1,
        ):
            distance = event.timestamp - cutoff_timestamp
            selected[_distance_slice(distance)][user_id].add(event.item_id)
            selected[_rank_slice(event_rank)][user_id].add(event.item_id)

    activity_quartiles: dict[str, list[int]] = {
        f"user_activity_q{quartile}": [] for quartile in range(1, 5)
    }
    ordered_users = sorted(
        users, key=lambda user_id: (training_like_counts[user_id], user_id)
    )
    for rank, user_id in enumerate(ordered_users):
        name = f"user_activity_q{4 * rank // len(ordered_users) + 1}"
        activity_quartiles[name].append(user_id)
        selected[name][user_id].update(event.item_id for event in by_user[user_id])

    ordered_names = (
        "target_distance_0_6h",
        "target_distance_6_24h",
        "target_distance_1_3d",
        "target_distance_3_7d",
        "target_event_rank_1",
        "target_event_rank_2_5",
        "target_event_rank_6_10",
        "target_event_rank_11_plus",
        "user_activity_q1",
        "user_activity_q2",
        "user_activity_q3",
        "user_activity_q4",
    )
    slices = {
        name: _score_slice(
            normalized_rankings,
            selected[name],
            catalog_size=catalog_size,
        )
        for name in ordered_names
    }
    return {
        "schema_version": 1,
        "cutoff_timestamp": cutoff_timestamp,
        "final_interval_seconds": _FINAL_INTERVAL_SECONDS,
        "activity_quartiles": activity_quartiles,
        "slices": slices,
    }


def _distance_slice(distance: int) -> str:
    if distance <= 6 * 60 * 60:
        return "target_distance_0_6h"
    if distance <= 24 * 60 * 60:
        return "target_distance_6_24h"
    if distance <= 3 * 24 * 60 * 60:
        return "target_distance_1_3d"
    return "target_distance_3_7d"


def _rank_slice(rank: int) -> str:
    if rank == 1:
        return "target_event_rank_1"
    if rank <= 5:
        return "target_event_rank_2_5"
    if rank <= 10:
        return "target_event_rank_6_10"
    return "target_event_rank_11_plus"


def _score_slice(
    rankings: Mapping[int, tuple[int, ...]],
    relevant_by_user: Mapping[int, set[int]],
    *,
    catalog_size: int,
) -> dict[str, object]:
    sums = {
        f"{metric}@{cutoff}": 0.0
        for metric in ("recall", "capped_recall", "ndcg", "mrr")
        for cutoff in _CUTOFFS
    }
    covered = {cutoff: set[int]() for cutoff in _CUTOFFS}
    for user_id, relevant_items in relevant_by_user.items():
        ranking = rankings[user_id]
        for cutoff in _CUTOFFS:
            sums[f"recall@{cutoff}"] += recall_at_k(ranking, relevant_items, cutoff)
            sums[f"capped_recall@{cutoff}"] += capped_recall_at_k(
                ranking, relevant_items, cutoff
            )
            sums[f"ndcg@{cutoff}"] += ndcg_at_k(ranking, relevant_items, cutoff)
            sums[f"mrr@{cutoff}"] += mrr_at_k(ranking, relevant_items, cutoff)
            covered[cutoff].update(ranking[:cutoff])

    user_count = len(relevant_by_user)
    metrics = {
        name: total / user_count if user_count else 0.0 for name, total in sums.items()
    }
    metrics.update(
        {
            f"coverage@{cutoff}": len(covered[cutoff]) / catalog_size
            for cutoff in _CUTOFFS
        }
    )
    return {
        "num_users": user_count,
        "num_targets": sum(map(len, relevant_by_user.values())),
        "metrics": metrics,
    }
