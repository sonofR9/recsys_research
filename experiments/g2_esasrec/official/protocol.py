from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

TEST_INTERVAL_SECONDS = 7 * 24 * 60 * 60
MIN_TRAIN_EVENTS = 2
USER_COLUMN = "uid"
ITEM_COLUMN = "compact_item_id"
TIMESTAMP_COLUMN = "timestamp"


@dataclass(frozen=True)
class Split:
    train: pl.DataFrame
    validation: pl.DataFrame
    cutoff: int
    catalog: np.ndarray

    @property
    def catalog_size(self) -> int:
        return len(self.catalog)


def dataset_dir(generated: Path) -> Path:
    return generated / "datasets" / "yambda" / "50m_like_core5_knownitems"


def load_split(generated: Path) -> Split:
    events = pl.read_parquet(
        dataset_dir(generated) / "events_remapped.parquet",
        columns=[USER_COLUMN, ITEM_COLUMN, TIMESTAMP_COLUMN],
    ).with_columns(pl.col(TIMESTAMP_COLUMN).cast(pl.Int64))
    cutoff = int(events[TIMESTAMP_COLUMN].max()) - TEST_INTERVAL_SECONDS
    catalog = np.sort(events[ITEM_COLUMN].unique().to_numpy())
    return Split(
        train=events.filter(pl.col(TIMESTAMP_COLUMN) < cutoff),
        validation=events.filter(pl.col(TIMESTAMP_COLUMN) >= cutoff),
        cutoff=cutoff,
        catalog=catalog,
    )


def relevance(split: Split) -> dict[int, set[int]]:
    catalog = set(split.catalog.tolist())
    grouped = split.validation.group_by(USER_COLUMN).agg(pl.col(ITEM_COLUMN).unique())
    return {
        int(user): items
        for user, item_ids in zip(
            grouped.get_column(USER_COLUMN).to_list(),
            grouped.get_column(ITEM_COLUMN).to_list(),
        )
        if (items := {int(item) for item in item_ids} & catalog)
    }


def query_histories(split: Split, max_seq_len: int) -> dict[int, list[int]]:
    grouped = (
        split.train.sort([USER_COLUMN, TIMESTAMP_COLUMN], maintain_order=True)
        .group_by(USER_COLUMN, maintain_order=True)
        .agg(pl.col(ITEM_COLUMN))
        .filter(pl.col(ITEM_COLUMN).list.len() >= MIN_TRAIN_EVENTS)
        .with_columns(pl.col(ITEM_COLUMN).list.tail(max_seq_len))
    )
    return {
        int(user): [int(item) for item in items]
        for user, items in zip(
            grouped.get_column(USER_COLUMN).to_list(),
            grouped.get_column(ITEM_COLUMN).to_list(),
        )
    }


def evaluable_users(
    histories: dict[int, list[int]], relevant: dict[int, set[int]]
) -> list[int]:
    return sorted(user for user in histories if user in relevant)


def candidate_catalog_evidence(
    split: Split, candidate_item_ids: np.ndarray
) -> dict[str, int | str]:
    mapped = np.sort(np.asarray(split.catalog, dtype="<i8"))
    candidates = np.sort(np.asarray(candidate_item_ids, dtype="<i8"))
    if candidates.size != np.unique(candidates).size or not np.array_equal(
        candidates, mapped
    ):
        raise ValueError("model candidate catalog differs from the mapped catalog")
    train_catalog_size = split.train.get_column(ITEM_COLUMN).n_unique()
    return {
        "catalog_size": split.catalog_size,
        "model_candidate_catalog_size": candidates.size,
        "train_catalog_size": train_catalog_size,
        "mapped_items_absent_from_training": split.catalog_size - train_catalog_size,
        "candidate_catalog_sha256": hashlib.sha256(mapped.tobytes()).hexdigest(),
    }


def score_rankings(
    rankings: dict[int, list[int]],
    relevant: dict[int, set[int]],
    users: list[int],
    ks: tuple[int, ...],
    catalog_size: int,
) -> dict[str, float]:
    top_n = max(ks)
    discounts = [1.0 / math.log2(rank + 2) for rank in range(top_n)]
    ideal = np.cumsum(discounts)
    sums = {
        f"{metric}@{k}": 0.0
        for metric in ("ndcg", "recall", "capped_recall", "mrr")
        for k in ks
    }
    covered: dict[int, set[int]] = {k: set() for k in ks}

    for user in users:
        relevant_items = relevant[user]
        ranked = rankings[user]
        hits = [item in relevant_items for item in ranked]
        for k in ks:
            prefix = hits[:k]
            hit_count = sum(prefix)
            sums[f"recall@{k}"] += hit_count / len(relevant_items)
            sums[f"capped_recall@{k}"] += hit_count / min(len(relevant_items), k)
            sums[f"ndcg@{k}"] += (
                sum(discount for hit, discount in zip(prefix, discounts) if hit)
                / ideal[min(len(relevant_items), k) - 1]
            )
            sums[f"mrr@{k}"] += next(
                (1.0 / (rank + 1) for rank, hit in enumerate(prefix) if hit),
                0.0,
            )
            covered[k].update(ranked[:k])

    num_users = len(users)
    metrics = {name: total / num_users for name, total in sums.items()}
    metrics.update({f"coverage@{k}": len(covered[k]) / catalog_size for k in ks})
    metrics["num_users"] = float(num_users)
    return metrics
