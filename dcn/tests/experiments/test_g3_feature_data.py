from pathlib import Path

import polars as pl
import pytest

from experiments.g3_pretrained_item_embeddings.data import (
    load_feature_data,
    materialize_feature_data,
)


def _source(tmp_path: Path) -> tuple[Path, Path]:
    events = tmp_path / "events.parquet"
    remap = tmp_path / "item_id_remap.parquet"
    pl.DataFrame(
        {
            "compact_item_id": [1, 1, 2, 2, 3, 3],
            "uid": [10, 10, 10, 11, 12, 12],
            "timestamp": [10, 20, 20, 30, 90, 100],
            "artist_id": [[20, 10], [10], [30], [30, 40], [50], [50]],
            "album_id": [[200], [200], [], [300], [400], [400]],
        }
    ).write_parquet(events)
    pl.DataFrame({"item_id": [101, 102, 103], "compact_id": [1, 2, 3]}).write_parquet(
        remap
    )
    return events, remap


def test_materialization_uses_only_training_rows_and_is_deterministic(
    tmp_path: Path,
) -> None:
    events, remap = _source(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_summary = materialize_feature_data(
        events_path=events,
        remap_path=remap,
        destination=first,
        validation_interval_seconds=50,
    )
    second_summary = materialize_feature_data(
        events_path=events,
        remap_path=remap,
        destination=second,
        validation_interval_seconds=50,
    )

    assert first_summary == second_summary
    assert first_summary.validation_cutoff_timestamp == 50
    assert first_summary.num_items == 3
    assert first_summary.training_rows == 4
    assert first_summary.artist_vocab_size == 4
    assert first_summary.album_vocab_size == 2
    assert (first / "item_features.parquet").read_bytes() == (
        second / "item_features.parquet"
    ).read_bytes()

    loaded = load_feature_data(first / "item_features.parquet")
    assert loaded.training_counts.tolist() == [0, 2, 2, 0]
    assert loaded.training_history_lengths == {10: 3, 11: 1}
    assert loaded.artist_rows == ((), (1, 2), (3, 4), ())
    assert loaded.album_rows == ((), (1,), (2,), ())


def test_materialization_fails_closed_on_noncontiguous_catalog(tmp_path: Path) -> None:
    events, remap = _source(tmp_path)
    pl.DataFrame({"item_id": [101, 103], "compact_id": [1, 3]}).write_parquet(remap)

    with pytest.raises(ValueError, match="contiguous"):
        materialize_feature_data(
            events_path=events,
            remap_path=remap,
            destination=tmp_path / "output",
            validation_interval_seconds=50,
        )
