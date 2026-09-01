from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from .slices import RelevanceEvent, slice_metrics


_SNAPSHOT_KEYS = {
    "schema_version",
    "catalog_sha256",
    "catalog_size",
    "exclude_seen",
    "max_k",
    "rankings",
}


def _ranking_snapshot(path: Path) -> tuple[dict[int, tuple[int, ...]], dict[str, Any]]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or set(document) != _SNAPSHOT_KEYS:
        raise ValueError("ranking snapshot has missing or unknown fields")
    catalog_size = document["catalog_size"]
    max_k = document["max_k"]
    if (
        document["schema_version"] != 1
        or type(catalog_size) is not int
        or type(max_k) is not int
        or not 1 <= max_k <= catalog_size
    ):
        raise ValueError("ranking snapshot dimensions are invalid")
    rankings: dict[int, tuple[int, ...]] = {}
    for row in document["rankings"]:
        if not isinstance(row, dict) or set(row) != {"user_id", "item_ids"}:
            raise ValueError("ranking row has missing or unknown fields")
        user_id = row["user_id"]
        item_ids = row["item_ids"]
        if type(user_id) is not int or not isinstance(item_ids, list):
            raise ValueError("ranking row types are invalid")
        ranking = tuple(item_ids)
        if (
            len(ranking) != max_k
            or any(type(item_id) is not int for item_id in ranking)
            or len(ranking) != len(set(ranking))
            or user_id in rankings
        ):
            raise ValueError("ranking row is invalid or duplicated")
        rankings[user_id] = ranking
    return rankings, document


def evaluate_slices(
    *,
    ranking_snapshot_path: Path,
    mapped_events_path: Path,
    cutoff_timestamp: int,
    final_interval_seconds: int = 7 * 24 * 60 * 60,
) -> dict[str, Any]:
    rankings, snapshot = _ranking_snapshot(ranking_snapshot_path)
    users = sorted(rankings)
    events = (
        pl.scan_parquet(mapped_events_path)
        .filter((pl.col("event_type") == "like") & pl.col("uid").is_in(users))
        .select("uid", "compact_item_id", "timestamp")
        .collect(engine="streaming")
    )
    training = events.filter(pl.col("timestamp") < cutoff_timestamp)
    counts_frame = training.group_by("uid").len()
    training_counts = dict(
        zip(
            counts_frame["uid"].to_list(),
            counts_frame["len"].to_list(),
            strict=True,
        )
    )
    final = events.filter(
        (pl.col("timestamp") > cutoff_timestamp)
        & (pl.col("timestamp") <= cutoff_timestamp + final_interval_seconds)
    )
    relevance = [
        RelevanceEvent(int(uid), int(item_id), int(timestamp))
        for uid, item_id, timestamp in final.iter_rows()
    ]
    if set(training_counts) != set(users) or {
        event.user_id for event in relevance
    } != set(users):
        raise ValueError("ranking and event evidence have different evaluation users")
    slice_document = slice_metrics(
        rankings=rankings,
        relevance_events=relevance,
        training_like_counts=training_counts,
        cutoff_timestamp=cutoff_timestamp,
        catalog_size=snapshot["catalog_size"],
    )
    return {
        **slice_document,
        "identity": {
            "catalog_sha256": snapshot["catalog_sha256"],
            "catalog_size": snapshot["catalog_size"],
            "evaluation_users": len(users),
            "exclude_seen": snapshot["exclude_seen"],
            "max_k": snapshot["max_k"],
        },
    }
