import json

import polars as pl

from experiments.g4_future_items.report.evaluation import evaluate_slices


def test_evaluate_slices_uses_closed_ranking_snapshot_and_like_occurrences(
    tmp_path,
) -> None:
    rankings = tmp_path / "top_item_rankings.json"
    rankings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_sha256": "catalog",
                "catalog_size": 120,
                "exclude_seen": False,
                "max_k": 100,
                "rankings": [
                    {"user_id": 1, "item_ids": list(range(100))},
                    {"user_id": 2, "item_ids": list(range(20, 120))},
                ],
            }
        )
    )
    events = tmp_path / "events.parquet"
    pl.DataFrame(
        {
            "uid": [1, 1, 1, 2, 2],
            "compact_item_id": [5, 6, 7, 25, 26],
            "timestamp": [1, 2, 11, 3, 12],
            "event_type": ["like"] * 5,
        }
    ).write_parquet(events)

    result = evaluate_slices(
        ranking_snapshot_path=rankings,
        mapped_events_path=events,
        cutoff_timestamp=10,
        final_interval_seconds=10,
    )

    assert result["identity"] == {
        "catalog_sha256": "catalog",
        "catalog_size": 120,
        "evaluation_users": 2,
        "exclude_seen": False,
        "max_k": 100,
    }
    assert result["slices"]["target_distance_0_6h"]["num_users"] == 2


def test_evaluate_slices_rejects_snapshot_user_mismatch(tmp_path) -> None:
    rankings = tmp_path / "top_item_rankings.json"
    rankings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_sha256": "catalog",
                "catalog_size": 100,
                "exclude_seen": False,
                "max_k": 100,
                "rankings": [{"user_id": 1, "item_ids": list(range(100))}],
            }
        )
    )
    events = tmp_path / "events.parquet"
    pl.DataFrame(
        {
            "uid": [2],
            "compact_item_id": [5],
            "timestamp": [11],
            "event_type": ["like"],
        }
    ).write_parquet(events)

    try:
        evaluate_slices(
            ranking_snapshot_path=rankings,
            mapped_events_path=events,
            cutoff_timestamp=10,
            final_interval_seconds=10,
        )
    except ValueError as error:
        assert "evaluation user" in str(error)
    else:
        raise AssertionError("mismatched evaluation population was accepted")
