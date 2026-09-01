from pathlib import Path

import polars as pl
import pytest
import torch

from dcn.eval.ranking_evidence import RankingEvidence, write_ranking_evidence
from experiments.g4_future_items.report.native500m_slices import (
    evaluate_native500m_rank_slices,
)
from experiments.g4_future_items.report.slices import RelevanceEvent, slice_metrics


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[int, list[int]]]:
    cutoff = 1_000_000
    users = [10, 20, 30, 40]
    training_counts = {10: 1, 20: 2, 30: 3, 40: 4}
    target_counts = {10: 1, 20: 2, 30: 6, 40: 11}
    relevant_items: list[int] = []
    relevant_ranks: list[int] = []
    relevance_offsets = [0]
    rows: list[dict[str, int | str]] = []
    rankings: dict[int, list[int]] = {}
    for user_index, user_id in enumerate(users):
        for event_index in range(training_counts[user_id]):
            rows.append(
                {
                    "uid": user_id,
                    "compact_item_id": 1_000 + user_index * 10 + event_index,
                    "timestamp": cutoff - 100 - event_index,
                    "event_type": "like",
                }
            )
        ranking = list(range(10_000 + user_index * 100, 10_100 + user_index * 100))
        for event_index in range(target_counts[user_id]):
            item_id = 20_000 + user_index * 100 + event_index
            rank = (event_index * 13 + user_index * 7) % 101
            if rank:
                ranking[rank - 1] = item_id
            relevant_items.append(item_id)
            relevant_ranks.append(rank)
            rows.append(
                {
                    "uid": user_id,
                    "compact_item_id": item_id,
                    "timestamp": cutoff + 3_600 * (event_index + 1),
                    "event_type": "like",
                }
            )
        relevance_offsets.append(len(relevant_items))
        rankings[user_id] = ranking

    context_path = tmp_path / "context.pt"
    ranking_path = tmp_path / "ranking.pt"
    write_ranking_evidence(
        RankingEvidence(
            user_ids=torch.tensor(users),
            history_item_ids=torch.tensor([1, 2, 3, 4]),
            history_offsets=torch.tensor([0, 1, 2, 3, 4]),
            relevant_item_ids=torch.tensor(relevant_items),
            relevance_offsets=torch.tensor(relevance_offsets),
            relevant_train_frequencies=torch.zeros(
                len(relevant_items), dtype=torch.int64
            ),
            relevant_ranks=torch.tensor(relevant_ranks),
            max_k=100,
        ),
        context_path=context_path,
        ranking_path=ranking_path,
    )
    events_path = tmp_path / "events.parquet"
    pl.DataFrame(rows).write_parquet(events_path)
    return context_path, ranking_path, events_path, rankings


def test_rank_slices_match_full_ranking_slice_metrics(tmp_path: Path) -> None:
    cutoff = 1_000_000
    context, ranking, events, rankings = _write_fixture(tmp_path)
    actual = evaluate_native500m_rank_slices(
        context_path=context,
        ranking_path=ranking,
        mapped_events_path=events,
        cutoff_timestamp=cutoff,
    )
    frame = pl.read_parquet(events)
    final = frame.filter(pl.col("timestamp") > cutoff)
    expected = slice_metrics(
        rankings=rankings,
        relevance_events=[
            RelevanceEvent(int(user), int(item), int(timestamp))
            for user, item, timestamp in final.select(
                "uid", "compact_item_id", "timestamp"
            ).iter_rows()
        ],
        training_like_counts={10: 1, 20: 2, 30: 3, 40: 4},
        cutoff_timestamp=cutoff,
        catalog_size=50_000,
    )

    assert actual["activity_quartiles"] == expected["activity_quartiles"]
    for name, expected_slice in expected["slices"].items():
        actual_slice = actual["slices"][name]
        assert actual_slice["num_users"] == expected_slice["num_users"]
        assert actual_slice["num_targets"] == expected_slice["num_targets"]
        for metric, expected_value in expected_slice["metrics"].items():
            if not metric.startswith("coverage@"):
                assert actual_slice["metrics"][metric] == pytest.approx(expected_value)


def test_rank_slices_reject_changed_relevance(tmp_path: Path) -> None:
    context, ranking, events, _ = _write_fixture(tmp_path)
    frame = pl.read_parquet(events)
    frame = frame.with_columns(
        pl.when(pl.col("timestamp") > 1_000_000)
        .then(pl.col("compact_item_id") + 1)
        .otherwise(pl.col("compact_item_id"))
        .alias("compact_item_id")
    )
    frame.write_parquet(events)

    with pytest.raises(ValueError, match="relevance events differ"):
        evaluate_native500m_rank_slices(
            context_path=context,
            ranking_path=ranking,
            mapped_events_path=events,
            cutoff_timestamp=1_000_000,
        )


def test_rank_slices_include_the_validation_boundary(tmp_path: Path) -> None:
    context, ranking, events, _ = _write_fixture(tmp_path)
    frame = pl.read_parquet(events)
    first_target = frame.filter(pl.col("timestamp") == 1_000_000 + 7 * 3_600).row(
        0, named=True
    )
    frame = frame.with_columns(
        pl.when(
            (pl.col("uid") == first_target["uid"])
            & (pl.col("compact_item_id") == first_target["compact_item_id"])
        )
        .then(pl.lit(1_000_000))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    frame.write_parquet(events)

    result = evaluate_native500m_rank_slices(
        context_path=context,
        ranking_path=ranking,
        mapped_events_path=events,
        cutoff_timestamp=1_000_000,
    )

    assert result["slices"]["target_distance_0_6h"]["num_targets"] == 16
