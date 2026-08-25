import pickle
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import torch

from dcn.data.dataset import (
    EventDataset,
    collate_event_batch,
    collate_sequence_batch,
)

YAMBDA_FEATURE_COLUMNS: list[str] = ["item_id", "uid", "album_id", "artist_id"]

YAMBDA_COLUMNS: list[str] = [
    *YAMBDA_FEATURE_COLUMNS,
    "target_like",
    "target_listen",
    "listen_mask",
]


_EVENT_FIELDS = (
    "item_id",
    "uid",
    "album_id",
    "artist_id",
    "event_type",
    "listen_share",
    "timestamp",
)

_EVENTS = [
    (1, 100, 10, 1000, "like", 0, 1000),
    (2, 100, 10, 1000, "listen", 75, 2000),
    (3, 101, 11, 1001, "like", 0, 3000),
    (4, 101, 11, 1001, "dislike", 0, 4000),
    (5, 102, 12, 1002, "listen", 50, 5000),
]


def _yambda_like_df(events: list[tuple]) -> pl.DataFrame:
    df = pl.DataFrame([dict(zip(_EVENT_FIELDS, event)) for event in events])
    return df.with_columns(
        target_like=(pl.col("event_type") == "like").cast(pl.Float32),
        target_listen=(
            pl.when(pl.col("event_type") == "listen")
            .then(pl.col("listen_share") / 100.0)
            .otherwise(0.0)
        ).cast(pl.Float32),
        listen_mask=(pl.col("event_type") == "listen"),
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_parquet(temp_dir: Path) -> Path:
    path = temp_dir / "test_data.parquet"
    _yambda_like_df(_EVENTS).write_parquet(path)
    return path


def _make_dataset(
    path: Path, extra_columns: list[str] | tuple[str, ...] = ()
) -> EventDataset:
    return EventDataset(parquet_files=[path], columns=[*YAMBDA_COLUMNS, *extra_columns])


class TestEventDataset:
    def test_dataset_length(self, sample_parquet: Path) -> None:
        assert len(_make_dataset(sample_parquet)) == 5

    def test_getitem_buckets_columns_by_dtype(self, sample_parquet: Path) -> None:
        item = _make_dataset(sample_parquet)[0]

        assert set(item.keys()) == {"int_columns", "float_columns", "timestamp"}

        for feat in YAMBDA_FEATURE_COLUMNS:
            assert feat in item["int_columns"]
        assert "listen_mask" in item["int_columns"]
        assert set(item["float_columns"].keys()) == {"target_like", "target_listen"}

    def test_target_like_encoding(self, sample_parquet: Path) -> None:
        dataset = _make_dataset(sample_parquet)
        assert dataset[0]["float_columns"]["target_like"] == 1.0
        assert dataset[1]["float_columns"]["target_like"] == 0.0
        assert dataset[3]["float_columns"]["target_like"] == 0.0

    def test_listen_target_values(self, sample_parquet: Path) -> None:
        dataset = _make_dataset(sample_parquet)
        assert dataset[1]["float_columns"]["target_listen"] == pytest.approx(0.75)
        assert dataset[4]["float_columns"]["target_listen"] == pytest.approx(0.50)

    def test_listen_mask(self, sample_parquet: Path) -> None:
        dataset = _make_dataset(sample_parquet)
        assert dataset[0]["int_columns"]["listen_mask"] == 0
        assert dataset[1]["int_columns"]["listen_mask"] == 1

    def test_float_columns_without_counters(self, sample_parquet: Path) -> None:
        item = _make_dataset(sample_parquet)[0]
        assert set(item["float_columns"].keys()) == {"target_like", "target_listen"}

    def test_counter_columns_join_float_columns(self, temp_dir: Path) -> None:
        df = _yambda_like_df(_EVENTS[:2]).with_columns(
            counter_a=pl.Series([0.5, 0.6]),
            counter_b=pl.Series([1.0, 2.0]),
        )

        path = temp_dir / "test_with_counters.parquet"
        df.write_parquet(path)

        item = _make_dataset(path, extra_columns=["counter_a", "counter_b"])[0]
        assert item["float_columns"]["counter_a"] == pytest.approx(0.5)
        assert item["float_columns"]["counter_b"] == pytest.approx(1.0)

    def test_packed_counters_column_is_read_instead_of_the_raw_ones(
        self, temp_dir: Path
    ) -> None:
        df = _yambda_like_df(_EVENTS[:2]).with_columns(
            counter_a=pl.Series([0.5, 0.6]),
            counter_b=pl.Series([1.0, 2.0]),
            counters=pl.Series([[0.5, 1.0], [0.6, 2.0]], dtype=pl.Array(pl.Float32, 2)),
        )

        path = temp_dir / "test_packed_counters.parquet"
        df.write_parquet(path)

        dataset = _make_dataset(path, extra_columns=["counters"])
        assert "counter_a" not in dataset[0]["float_columns"]

        counters = collate_event_batch([dataset[i] for i in range(2)])["float_columns"][
            "counters"
        ]
        assert counters.offsets.tolist() == [0, 2, 4]
        assert torch.allclose(counters.values, torch.tensor([0.5, 1.0, 0.6, 2.0]))

    def test_multiple_parquet_files(self, temp_dir: Path) -> None:
        df1 = _yambda_like_df(_EVENTS[:2])
        df2 = _yambda_like_df(_EVENTS[2:4])

        path1 = temp_dir / "file1.parquet"
        path2 = temp_dir / "file2.parquet"
        df1.write_parquet(path1)
        df2.write_parquet(path2)

        dataset = EventDataset(parquet_files=[path1, path2], columns=YAMBDA_COLUMNS)
        assert len(dataset) == 4

    def test_list_columns_become_ragged(self, temp_dir: Path) -> None:
        df = pl.DataFrame(
            {
                "item_id": [1, 2],
                "artist_ids": [[10, 11], [12]],
                "scores": [[0.1, 0.2], [0.3]],
                "timestamp": [100, 200],
            }
        )
        path = temp_dir / "ragged.parquet"
        df.write_parquet(path)

        dataset = EventDataset(
            parquet_files=[path], columns=["item_id", "artist_ids", "scores"]
        )
        collated = collate_event_batch([dataset[i] for i in range(2)])

        artists = collated["int_columns"]["artist_ids"]
        assert artists.values.tolist() == [10, 11, 12]
        assert artists.offsets.tolist() == [0, 2, 3]

        scores = collated["float_columns"]["scores"]
        assert scores.values.dtype == torch.float32
        assert torch.allclose(scores.values, torch.tensor([0.1, 0.2, 0.3]))
        assert scores.offsets.tolist() == [0, 2, 3]

    def test_pickling_ships_the_paths_and_not_the_rows(self, temp_dir: Path) -> None:
        """The loader's workers are started with forkserver, so every worker is
        handed a pickle of the dataset."""
        row_count = 20_000
        path = temp_dir / "many_rows.parquet"
        pl.DataFrame(
            {
                "item_id": range(row_count),
                "score": [0.5] * row_count,
                "timestamp": range(row_count),
            }
        ).write_parquet(path)

        dataset = EventDataset(parquet_files=[path], columns=["item_id", "score"])
        _ = dataset[0]

        pickled = pickle.dumps(dataset)
        assert len(pickled) < 4096

        restored: EventDataset = pickle.loads(pickled)
        assert len(restored) == row_count
        assert restored[row_count - 1] == dataset[row_count - 1]


class TestCollateBatch:
    def test_collate_event_batch_shapes(self, sample_parquet: Path) -> None:
        dataset = _make_dataset(sample_parquet)
        collated = collate_event_batch([dataset[i] for i in range(3)])

        for feat in YAMBDA_FEATURE_COLUMNS:
            feature_values = collated["int_columns"][feat]
            assert feature_values.values.shape == (3,)
            assert feature_values.offsets.tolist() == [0, 1, 2, 3]
        assert collated["int_columns"]["listen_mask"].values.shape == (3,)
        assert collated["float_columns"]["target_like"].values.shape == (3,)
        assert collated["float_columns"]["target_listen"].values.shape == (3,)
        assert collated["timestamp"].shape == (3,)

    def test_event_batch_is_a_batch_of_length_one_sequences(
        self, sample_parquet: Path
    ) -> None:
        dataset = _make_dataset(sample_parquet)
        collated = collate_event_batch([dataset[i] for i in range(3)])

        assert collated["cumulative_lens"].tolist() == [0, 1, 2, 3]

    def test_collate_event_batch_dtypes(self, sample_parquet: Path) -> None:
        dataset = _make_dataset(sample_parquet)
        collated = collate_event_batch([dataset[i] for i in range(3)])

        assert collated["int_columns"]["item_id"].values.dtype == torch.int64
        assert collated["int_columns"]["listen_mask"].values.dtype == torch.int64
        assert collated["float_columns"]["target_like"].values.dtype == torch.float32
        assert collated["float_columns"]["target_listen"].values.dtype == torch.float32
        assert collated["timestamp"].dtype == torch.int64

    def test_collate_event_batch_counter_column(self, temp_dir: Path) -> None:
        df = _yambda_like_df(_EVENTS[:3]).with_columns(
            counter_a=pl.Series([0.1, 0.2, 0.3])
        )

        path = temp_dir / "test_counters.parquet"
        df.write_parquet(path)

        dataset = _make_dataset(path, extra_columns=["counter_a"])
        collated = collate_event_batch([dataset[i] for i in range(3)])

        counter_a = collated["float_columns"]["counter_a"]
        assert counter_a.offsets.tolist() == [0, 1, 2, 3]
        assert torch.allclose(counter_a.dense(), torch.tensor([0.1, 0.2, 0.3]))

    def test_collate_event_batch_variable_length_feature(self) -> None:
        def sample(tags: list[int], user_id: int, timestamp: int) -> dict[str, Any]:
            return {
                "int_columns": {"tags": tags, "user_id": user_id},
                "float_columns": {"like": 1.0},
                "timestamp": timestamp,
            }

        batch = [
            sample([10, 11, 12], 1, 100),
            sample([], 2, 200),
            sample([13], 3, 300),
        ]
        collated = collate_event_batch(batch)

        tags = collated["int_columns"]["tags"]
        assert tags.values.tolist() == [10, 11, 12, 13]
        assert tags.offsets.tolist() == [0, 3, 3, 4]
        assert tags.num_rows() == 3

        user_id = collated["int_columns"]["user_id"]
        assert user_id.values.tolist() == [1, 2, 3]
        assert user_id.offsets.tolist() == [0, 1, 2, 3]


class TestCollateBatchWithSequenceSamples:
    def _seq_sample(
        self,
        item_ids: list[int],
        tags: list[list[int]],
        ratings: list[float],
        timestamps: list[int],
    ) -> dict[str, Any]:
        return {
            "int_columns": {"item_id": item_ids, "tags": tags},
            "float_columns": {"rating": ratings},
            "timestamp": timestamps,
        }

    def test_flattens_sequences_and_emits_cumulative_lens(self) -> None:
        batch = [
            self._seq_sample(
                [1, 2, 3], [[10], [11, 12], []], [0.5, 0.6, 0.7], [100, 110, 120]
            ),
            self._seq_sample([4, 5], [[20, 21], [22]], [0.8, 0.9], [200, 210]),
        ]
        collated = collate_sequence_batch(batch)

        assert collated["cumulative_lens"].tolist() == [0, 3, 5]
        item = collated["int_columns"]["item_id"]
        assert item.values.tolist() == [1, 2, 3, 4, 5]
        assert item.offsets.tolist() == [0, 1, 2, 3, 4, 5]

        tags = collated["int_columns"]["tags"]
        assert tags.offsets.tolist() == [0, 1, 3, 3, 5, 6]
        assert tags.values.tolist() == [10, 11, 12, 20, 21, 22]

        rating = collated["float_columns"]["rating"]
        assert torch.allclose(rating.dense(), torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9]))
        assert collated["timestamp"].tolist() == [100, 110, 120, 200, 210]
