"""Full-catalog future-day ranking eval (the "true metric")."""

import hashlib
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import polars as pl
import torch

from dcn.data.dataset import bucket_columns_by_dtype, collate_event_batch
from dcn.data.features import FeatureValues

_EMPTY: frozenset[int] = frozenset()


class _EvaluableUser(NamedTuple):
    """A user with at least one unseen relevant item inside the catalog."""

    user_id: int
    row: int
    relevant: set[int]


def build_interaction_sets(
    parquet_files: list[Path],
    *,
    user_column: str,
    item_id_column: str,
    row_filter: pl.Expr | None = None,
) -> dict[int, set[int]]:
    """`{user_id: {item_id, ...}}` over the given days, counting the rows
    `row_filter` keeps -- pass the experiment's own filter so relevance is
    measured on the events it was trained to predict."""
    keep = [user_column, item_id_column]
    if row_filter is not None:
        keep.extend(row_filter.meta.root_names())
    keep = list(dict.fromkeys(keep))

    df = pl.concat([pl.read_parquet(file, columns=keep) for file in parquet_files])
    if row_filter is not None:
        df = df.filter(row_filter)

    grouped = df.group_by(user_column).agg(pl.col(item_id_column).unique())
    return {
        int(user_id): {int(item_id) for item_id in item_ids}
        for user_id, item_ids in zip(
            grouped.get_column(user_column).to_list(),
            grouped.get_column(item_id_column).to_list(),
        )
    }


def build_item_snapshot(
    parquet_files: list[Path],
    *,
    item_id_column: str,
    columns: Sequence[str] = (),
    timestamp_column: str = "timestamp",
) -> dict:
    """One collated row per catalog item, carrying its most recent features."""
    keep = list(dict.fromkeys([item_id_column, *columns, timestamp_column]))
    df = pl.concat([pl.read_parquet(file, columns=keep) for file in parquet_files])
    latest = (
        df.sort(timestamp_column).group_by(item_id_column, maintain_order=True).last()
    )

    feature_columns = [name for name in keep if name != timestamp_column]
    buckets = bucket_columns_by_dtype(latest.schema, feature_columns)
    rows = latest.to_dicts()
    return collate_event_batch(
        [
            {
                "int_columns": {name: row[name] for name in buckets.int_names},
                "float_columns": {name: row[name] for name in buckets.float_names},
                "timestamp": int(row[timestamp_column]),
            }
            for row in rows
        ]
    )


def build_catalog_batch(item_ids: Iterable[int], *, item_id_column: str) -> dict:
    """One row per item carrying nothing but its id -- the whole input a model
    whose item tower is a plain embedding table needs to encode its catalog."""
    ids = torch.tensor(sorted(item_ids), dtype=torch.int64)
    return {
        "int_columns": {
            item_id_column: FeatureValues(
                values=ids, offsets=torch.arange(ids.shape[0] + 1, dtype=torch.int64)
            )
        },
        "float_columns": {},
    }


def _find_evaluable_users(
    query_user_ids: torch.Tensor,
    id_to_position: dict[int, int],
    relevance: dict[int, set[int]],
    train_seen: dict[int, set[int]],
    exclude_seen: bool,
) -> list[_EvaluableUser]:
    evaluable = []
    for row, user_id in enumerate(query_user_ids.tolist()):
        seen = train_seen.get(user_id, _EMPTY)
        relevant = {
            item_id
            for item_id in relevance.get(user_id, _EMPTY)
            if item_id in id_to_position and (not exclude_seen or item_id not in seen)
        }
        if relevant:
            evaluable.append(
                _EvaluableUser(user_id=user_id, row=row, relevant=relevant)
            )
    return evaluable


def _sample_key(user_id: int, seed: int) -> bytes:
    return hashlib.blake2b(f"{seed}:{user_id}".encode(), digest_size=8).digest()


def _subsample(
    users: list[_EvaluableUser], max_users: int, seed: int
) -> list[_EvaluableUser]:
    """A fixed sample keyed on the user id itself.

    Drawing by position would hand back a different set of users whenever the
    population shifted -- a variant that filters one more event, an extra day of
    data -- and two runs sampled that way are not paired on anything. Keyed on
    the id, a population that shifts only moves the sample at its boundary.
    """
    if len(users) <= max_users:
        return users
    return sorted(users, key=lambda user: _sample_key(user.user_id, seed))[:max_users]


class PreparedRanking(NamedTuple):
    """Everything about the scored population that training does not change.

    Which users are evaluable, where each item sits in the catalog and which
    cells the seen mask blanks all follow from the split, not from the weights,
    so an epoch-end eval resolves them once and then only ranks.
    """

    item_id_list: list[int]
    evaluable: list[_EvaluableUser]
    user_chunk: int
    seen_by_chunk: list[tuple[torch.Tensor, torch.Tensor]]
    query_rows_by_chunk: list[torch.Tensor]
    relevant_by_chunk: list[torch.Tensor]

    def chunks(self):
        return zip(
            self.query_rows_by_chunk,
            self.seen_by_chunk,
            self.relevant_by_chunk,
        )


def prepare_ranking(
    query_user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    relevance: dict[int, set[int]],
    train_seen: dict[int, set[int]],
    *,
    device: torch.device,
    user_chunk: int = 256,
    max_users: int | None = None,
    seed: int = 42,
    exclude_seen: bool = True,
) -> PreparedRanking:
    item_id_list = item_ids.tolist()
    id_to_position = {
        item_id: position for position, item_id in enumerate(item_id_list)
    }

    evaluable = _find_evaluable_users(
        query_user_ids, id_to_position, relevance, train_seen, exclude_seen
    )
    if max_users is not None:
        evaluable = _subsample(evaluable, max_users, seed)

    seen_by_chunk = []
    query_rows_by_chunk = []
    relevant_by_chunk = []
    for start in range(0, len(evaluable), user_chunk):
        chunk = evaluable[start : start + user_chunk]
        rows, columns = [], []
        for row, user in enumerate(chunk):
            if not exclude_seen:
                continue
            for item_id in train_seen.get(user.user_id, _EMPTY):
                position = id_to_position.get(item_id)
                if position is not None:
                    rows.append(row)
                    columns.append(position)
        seen_by_chunk.append(
            (
                torch.tensor(rows, dtype=torch.long, device=device),
                torch.tensor(columns, dtype=torch.long, device=device),
            )
        )
        query_rows_by_chunk.append(
            torch.tensor([user.row for user in chunk], dtype=torch.long, device=device)
        )
        relevant_positions = [
            sorted(id_to_position[item_id] for item_id in user.relevant)
            for user in chunk
        ]
        width = max(map(len, relevant_positions))
        relevant_by_chunk.append(
            torch.tensor(
                [
                    positions + [len(item_id_list)] * (width - len(positions))
                    for positions in relevant_positions
                ],
                dtype=torch.long,
                device=device,
            )
        )
    return PreparedRanking(
        item_id_list,
        evaluable,
        user_chunk,
        seen_by_chunk,
        query_rows_by_chunk,
        relevant_by_chunk,
    )


@torch.inference_mode()
def evaluate_true_ndcg(
    query_repr: torch.Tensor,
    query_user_ids: torch.Tensor,
    item_repr: torch.Tensor,
    item_ids: torch.Tensor,
    relevance: dict[int, set[int]],
    train_seen: dict[int, set[int]],
    ks: Iterable[int],
    *,
    device: torch.device | None = None,
    user_chunk: int = 256,
    max_users: int | None = None,
    seed: int = 42,
    prepared: PreparedRanking | None = None,
    exclude_seen: bool = True,
) -> dict[str, float]:
    """Mean full-catalog NDCG@k / Recall@k / MRR@k over evaluable users, plus
    the share of the catalog their top-k lists between them cover.

    Coverage is a property of the scored population, not of a user, so it grows
    with `num_users` and is comparable only between runs that scored the same
    number of them.
    """
    ks = list(dict.fromkeys(ks))
    device = device or item_repr.device
    item_repr = item_repr.to(device)
    query_repr = query_repr.to(device)

    if prepared is None:
        prepared = prepare_ranking(
            query_user_ids,
            item_ids,
            relevance,
            train_seen,
            device=device,
            user_chunk=user_chunk,
            max_users=max_users,
            seed=seed,
            exclude_seen=exclude_seen,
        )
    evaluable = prepared.evaluable

    sums = {
        f"{metric}@{k}": torch.zeros((), dtype=torch.float64, device=device)
        for metric in ("ndcg", "recall", "capped_recall", "mrr")
        for k in ks
    }
    covered = {
        k: torch.zeros(item_repr.shape[0], dtype=torch.bool, device=device) for k in ks
    }
    top_n = min(max(ks), item_repr.shape[0])
    discounts = 1.0 / torch.log2(
        torch.arange(top_n, dtype=torch.float64, device=device) + 2
    )
    ideal_dcg = discounts.cumsum(0)
    reciprocal_ranks = 1.0 / (
        torch.arange(top_n, dtype=torch.float64, device=device) + 1
    )

    for query_rows, (seen_rows, seen_columns), relevant_positions in prepared.chunks():
        scores = query_repr[query_rows] @ item_repr.t()
        scores[seen_rows, seen_columns] = float("-inf")

        top_positions = torch.topk(scores, top_n, dim=1).indices
        insertion_points = torch.searchsorted(relevant_positions, top_positions)
        clamped_points = insertion_points.clamp_max(relevant_positions.shape[1] - 1)
        hits = (insertion_points < relevant_positions.shape[1]) & (
            relevant_positions.gather(1, clamped_points) == top_positions
        )
        cumulative_hits = hits.to(torch.float64).cumsum(1)
        cumulative_dcg = (hits * discounts).cumsum(1)
        reciprocal_hits = hits * reciprocal_ranks
        relevant_counts = (relevant_positions < item_repr.shape[0]).sum(1)

        for k in ks:
            cutoff = min(k, top_n)
            hit_count = cumulative_hits[:, cutoff - 1]
            ideal_length = relevant_counts.clamp_max(k)
            sums[f"ndcg@{k}"] += (
                cumulative_dcg[:, cutoff - 1] / ideal_dcg[ideal_length - 1]
            ).sum()
            sums[f"recall@{k}"] += (hit_count / relevant_counts).sum()
            sums[f"capped_recall@{k}"] += (hit_count / ideal_length).sum()
            sums[f"mrr@{k}"] += reciprocal_hits[:, :cutoff].amax(1).sum()
            covered[k][top_positions[:, :cutoff].flatten()] = True

    num_users = len(evaluable)
    catalog_size = item_repr.shape[0]
    names = [*sums, *(f"coverage@{k}" for k in ks)]
    totals = torch.stack(
        [*sums.values(), *(covered[k].sum(dtype=torch.float64) for k in ks)]
    ).tolist()
    metrics = {}
    for name, total in zip(names, totals):
        denominator = catalog_size if name.startswith("coverage@") else num_users
        metrics[name] = total / denominator if denominator else 0.0
    metrics["num_users"] = float(num_users)
    return metrics
