from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from dcn.eval.ranking_evidence import load_ranking_evidence


_CUTOFFS = (10, 50, 100)
_FINAL_INTERVAL_SECONDS = 7 * 24 * 60 * 60


def evaluate_native500m_rank_slices(
    *,
    context_path: Path,
    ranking_path: Path,
    mapped_events_path: Path,
    cutoff_timestamp: int,
) -> dict[str, Any]:
    evidence = load_ranking_evidence(context_path, ranking_path)
    users = [int(value) for value in evidence.user_ids.tolist()]
    if not users:
        raise ValueError("native-500M slice evidence has no users")
    events = (
        pl.scan_parquet(mapped_events_path)
        .filter((pl.col("event_type") == "like") & pl.col("uid").is_in(users))
        .select("uid", "compact_item_id", "timestamp")
        .collect(engine="streaming")
    )
    training_counts = dict(
        events.filter(pl.col("timestamp") < cutoff_timestamp)
        .group_by("uid")
        .len()
        .iter_rows()
    )
    final_events = events.filter(
        (pl.col("timestamp") >= cutoff_timestamp)
        & (pl.col("timestamp") <= cutoff_timestamp + _FINAL_INTERVAL_SECONDS)
    ).sort(["uid", "timestamp", "compact_item_id"], maintain_order=True)
    events_by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for user_id, item_id, timestamp in final_events.iter_rows():
        events_by_user[int(user_id)].append((int(item_id), int(timestamp)))
    if set(users) != set(training_counts) or set(users) != set(events_by_user):
        raise ValueError("native-500M slice user populations differ")

    ranks_by_user = _ranks_by_user(evidence)
    for user_id, final in events_by_user.items():
        if set(ranks_by_user[user_id]) != {item_id for item_id, _ in final}:
            raise ValueError("native-500M relevance events differ from ranking context")

    selected: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for user_id, final in events_by_user.items():
        for event_rank, (item_id, timestamp) in enumerate(final, start=1):
            distance = timestamp - cutoff_timestamp
            selected[_distance_slice(distance)][user_id].add(item_id)
            selected[_event_rank_slice(event_rank)][user_id].add(item_id)

    ordered_users = sorted(
        users, key=lambda user_id: (training_counts[user_id], user_id)
    )
    activity_quartiles = {f"user_activity_q{quartile}": [] for quartile in range(1, 5)}
    for position, user_id in enumerate(ordered_users):
        name = f"user_activity_q{4 * position // len(ordered_users) + 1}"
        activity_quartiles[name].append(user_id)
        selected[name][user_id].update(ranks_by_user[user_id])

    names = (
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
    slices = {name: _score_rank_slice(ranks_by_user, selected[name]) for name in names}
    overall = _score_rank_slice(
        ranks_by_user,
        {user_id: set(ranks) for user_id, ranks in ranks_by_user.items()},
    )
    return {
        "schema_version": 1,
        "cutoff_timestamp": cutoff_timestamp,
        "final_interval_seconds": _FINAL_INTERVAL_SECONDS,
        "activity_quartiles": activity_quartiles,
        "overall": overall,
        "slices": slices,
    }


def _ranks_by_user(evidence) -> dict[int, dict[int, int]]:
    users = evidence.user_ids.tolist()
    items = evidence.relevant_item_ids.tolist()
    ranks = evidence.relevant_ranks.tolist()
    offsets = evidence.relevance_offsets.tolist()
    return {
        int(user_id): {
            int(items[position]): int(ranks[position])
            for position in range(offsets[index], offsets[index + 1])
        }
        for index, user_id in enumerate(users)
    }


def _score_rank_slice(
    ranks_by_user: Mapping[int, Mapping[int, int]],
    relevant_by_user: Mapping[int, set[int]],
) -> dict[str, Any]:
    sums = {
        f"{metric}@{cutoff}": 0.0
        for metric in ("recall", "capped_recall", "ndcg", "mrr")
        for cutoff in _CUTOFFS
    }
    for user_id, relevant_items in relevant_by_user.items():
        item_ranks = ranks_by_user[user_id]
        for cutoff in _CUTOFFS:
            hit_ranks = [
                item_ranks[item_id]
                for item_id in relevant_items
                if 0 < item_ranks[item_id] <= cutoff
            ]
            sums[f"recall@{cutoff}"] += len(hit_ranks) / len(relevant_items)
            sums[f"capped_recall@{cutoff}"] += len(hit_ranks) / min(
                len(relevant_items), cutoff
            )
            ideal = sum(
                1.0 / math.log2(position + 1)
                for position in range(1, min(len(relevant_items), cutoff) + 1)
            )
            sums[f"ndcg@{cutoff}"] += (
                sum(1.0 / math.log2(rank + 1) for rank in hit_ranks) / ideal
            )
            sums[f"mrr@{cutoff}"] += 1.0 / min(hit_ranks) if hit_ranks else 0.0
    user_count = len(relevant_by_user)
    return {
        "num_users": user_count,
        "num_targets": sum(map(len, relevant_by_user.values())),
        "metrics": {
            name: value / user_count if user_count else 0.0
            for name, value in sums.items()
        },
    }


def _distance_slice(distance: int) -> str:
    if distance <= 6 * 60 * 60:
        return "target_distance_0_6h"
    if distance <= 24 * 60 * 60:
        return "target_distance_6_24h"
    if distance <= 3 * 24 * 60 * 60:
        return "target_distance_1_3d"
    return "target_distance_3_7d"


def _event_rank_slice(rank: int) -> str:
    if rank == 1:
        return "target_event_rank_1"
    if rank <= 5:
        return "target_event_rank_2_5"
    if rank <= 10:
        return "target_event_rank_6_10"
    return "target_event_rank_11_plus"
