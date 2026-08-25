import json
import threading
import time
from itertools import accumulate
from pathlib import Path

import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from dcn.data import BucketShuffleSampler, SequenceDataset, collate_sequence_batch

COLUMNS = ["item_id", "rating", "tags"]
USER_COLUMN = "uid"


def _dataset(
    paths: Path | list[Path], columns: list[str] = COLUMNS, **kwargs
) -> SequenceDataset:
    paths = [paths] if isinstance(paths, Path) else paths
    kwargs.setdefault("max_seq_len", 3)
    kwargs.setdefault("min_seq_len", 2)
    cache = kwargs.pop("cache", None) or paths[0].parent / "cache"
    return SequenceDataset(paths, columns, cache, user_column=USER_COLUMN, **kwargs)


def _sliding(path: Path, *, max_seq_len: int, stride_events: int) -> SequenceDataset:
    return _dataset(
        path,
        cache=path.parent / f"sliding_{path.stem}_{max_seq_len}_{stride_events}",
        max_seq_len=max_seq_len,
        window="sliding",
        stride=stride_events / max_seq_len,
        n_buckets=1,
    )


def _item_sequences(dataset: SequenceDataset) -> list[list[int]]:
    return [dataset[i]["int_columns"]["item_id"] for i in range(len(dataset))]


def _all_samples(dataset: SequenceDataset) -> list[str]:
    return sorted(
        json.dumps(dataset[index], sort_keys=True) for index in range(len(dataset))
    )


def test_concurrent_datasets_build_a_shared_cold_cache_once(
    multi_file_parquets: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SequenceDataset._build_cache
    active = 0
    maximum = 0
    builds = 0
    guard = threading.Lock()

    def build_cache(
        dataset: SequenceDataset,
        parquet_files: list[Path],
        params: dict,
        n_buckets: int | None,
    ) -> dict:
        nonlocal active, maximum, builds
        with guard:
            active += 1
            maximum = max(maximum, active)
            builds += 1
        try:
            time.sleep(0.05)
            return original(dataset, parquet_files, params, n_buckets)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(SequenceDataset, "_build_cache", build_cache)
    cache = multi_file_parquets[0].parent / "concurrent-cache"
    datasets: list[SequenceDataset] = []
    errors: list[BaseException] = []

    def build() -> None:
        try:
            datasets.append(
                _dataset(multi_file_parquets, cache=cache, n_buckets=2)
            )
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=build) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(datasets) == 2
    assert builds == 1
    assert maximum == 1


def _write(path: Path, **columns: list) -> Path:
    pl.DataFrame(columns).write_parquet(path)
    return path


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """uid 100: 3 events out of order; 101: too short; 102: 5 events, truncated."""
    return _write(
        tmp_path / "events.parquet",
        uid=[100, 100, 100, 101, 102, 102, 102, 102, 102],
        timestamp=[200, 100, 300, 999, 50, 10, 40, 30, 20],
        item_id=[12, 11, 13, 99, 5, 1, 4, 3, 2],
        rating=[2.0, 1.0, 3.0, 9.9, 0.5, 0.1, 0.4, 0.3, 0.2],
        tags=[[2, 3], [1], [4], [99], [9], [5], [8], [7], [6]],
    )


@pytest.fixture
def multi_file_parquets(tmp_path: Path) -> list[Path]:
    """Six users, two events each per file; timestamp 30 spans both files."""
    paths = []
    for file_index, timestamps in enumerate([[10, 30], [20, 30]]):
        item_ids = [
            uid * 100 + file_index * 10 + position
            for uid in range(200, 206)
            for position in range(2)
        ]
        paths.append(
            _write(
                tmp_path / f"day_{file_index}.parquet",
                uid=[uid for uid in range(200, 206) for _ in range(2)],
                timestamp=timestamps * 6,
                item_id=item_ids,
                rating=[item_id / 10 for item_id in item_ids],
                tags=[[item_id] for item_id in item_ids],
            )
        )
    return paths


@pytest.fixture
def mixed_action_parquet(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "mixed.parquet",
        uid=[1, 1, 1, 1, 2, 2, 2],
        timestamp=[1, 2, 3, 4, 1, 2, 3],
        item_id=[10, 11, 12, 13, 20, 21, 22],
        rating=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        tags=[[1], [2], [3], [4], [5], [6], [7]],
        action=[1, 0, 1, 0, 1, 0, 0],
    )


def _write_history(tmp_path: Path, num_events: int) -> Path:
    return _write(
        tmp_path / f"history_{num_events}.parquet",
        uid=[7] * num_events,
        timestamp=list(range(num_events)),
        item_id=list(range(num_events)),
        rating=[float(i) for i in range(num_events)],
        tags=[[i] for i in range(num_events)],
    )


class TestSequenceDataset:
    def test_one_sequence_per_user_above_threshold(self, sample_parquet: Path) -> None:
        assert len(_dataset(sample_parquet)) == 2

    def test_buckets_columns_by_dtype(self, sample_parquet: Path) -> None:
        item = _dataset(sample_parquet)[0]
        assert set(item["int_columns"]) == {"item_id", "tags"}
        assert set(item["float_columns"]) == {"rating"}

    def test_events_sorted_by_timestamp(self, sample_parquet: Path) -> None:
        item = _dataset(sample_parquet)[0]
        assert item["timestamp"] == [100, 200, 300]
        assert item["int_columns"]["item_id"] == [11, 12, 13]
        assert item["int_columns"]["tags"] == [[1], [2, 3], [4]]

    def test_truncation_keeps_most_recent(self, sample_parquet: Path) -> None:
        item = _dataset(sample_parquet)[1]
        assert item["timestamp"] == [30, 40, 50]
        assert item["int_columns"]["item_id"] == [3, 4, 5]

    def test_roundtrip_through_collate_sequence_batch(
        self, sample_parquet: Path
    ) -> None:
        ds = _dataset(sample_parquet)
        collated = collate_sequence_batch([ds[0], ds[1]])

        assert collated["cumulative_lens"].tolist() == [0, 3, 6]
        assert collated["timestamp"].tolist() == [100, 200, 300, 30, 40, 50]

        item_id = collated["int_columns"]["item_id"]
        assert item_id.values.tolist() == [11, 12, 13, 3, 4, 5]
        assert item_id.offsets.tolist() == [0, 1, 2, 3, 4, 5, 6]

        tags = collated["int_columns"]["tags"]
        assert tags.values.tolist() == [1, 2, 3, 4, 7, 8, 9]
        assert tags.offsets.tolist() == [0, 1, 3, 4, 5, 6, 7]

        rating = collated["float_columns"]["rating"]
        assert torch.allclose(
            rating.dense(), torch.tensor([1.0, 2.0, 3.0, 0.3, 0.4, 0.5])
        )

    def test_loader_collates_scalar_sequences_in_batched_fetch(
        self, sample_parquet: Path
    ) -> None:
        dataset = _dataset(
            sample_parquet,
            columns=["item_id"],
            emit_user_column=True,
            n_buckets=1,
        )
        batch = next(
            iter(DataLoader(dataset, batch_size=2, collate_fn=collate_sequence_batch))
        )

        assert batch["cumulative_lens"].tolist() == [0, 3, 6]
        assert batch["int_columns"]["item_id"].values.tolist() == [
            11,
            12,
            13,
            3,
            4,
            5,
        ]
        assert batch["int_columns"]["uid"].values.tolist() == [
            100,
            100,
            100,
            102,
            102,
            102,
        ]

    def test_user_column_is_not_emitted_by_default(self, sample_parquet: Path) -> None:
        assert "uid" not in _dataset(sample_parquet)[0]["int_columns"]

    @pytest.mark.parametrize(
        "kwargs", [{"columns": ["uid", *COLUMNS]}, {"emit_user_column": True}]
    )
    def test_an_emitted_user_column_names_the_owner_of_its_own_sequence(
        self, multi_file_parquets: list[Path], kwargs: dict
    ) -> None:
        dataset = _dataset(multi_file_parquets, max_seq_len=4, n_buckets=3, **kwargs)

        for index in range(len(dataset)):
            sample = dataset[index]
            uids = sample["int_columns"]["uid"]
            assert len(uids) == len(sample["timestamp"])
            assert all(
                item // 100 == uid
                for item, uid in zip(sample["int_columns"]["item_id"], uids)
            )


def _reference_sliding_windows(
    items: list[int], max_seq_len: int, stride_events: int
) -> list[list[int]]:
    overlap = max_seq_len - stride_events
    windows = []
    end = len(items)
    while end > min(overlap, len(items) - 1):
        windows.append(items[max(0, end - max_seq_len) : end])
        end -= stride_events
    return windows


def test_next_item_windows_match_homework_history_target_chunks(tmp_path: Path) -> None:
    path = _write_history(tmp_path, 8)
    dataset = _dataset(
        path,
        cache=tmp_path / "next_item",
        max_seq_len=3,
        window="next_item",
        n_buckets=1,
    )

    assert _item_sequences(dataset) == [
        [0, 1, 2, 3],
        [3, 4, 5, 6],
        [6, 7],
    ]


class TestSlidingWindow:
    def test_matches_a_brute_force_reference_over_the_parameter_space(
        self, tmp_path: Path
    ) -> None:
        user_items = {uid: [uid * 1000 + p for p in range(uid)] for uid in range(1, 41)}
        path = _write(
            tmp_path / "histories.parquet",
            uid=[uid for uid, items in user_items.items() for _ in items],
            timestamp=[p for uid in user_items for p in range(uid)],
            item_id=[item for items in user_items.values() for item in items],
        )

        parameter_space = [
            (1, 1),
            (2, 1),
            (2, 2),
            (3, 1),
            (3, 2),
            (3, 3),
            (10, 1),
            (10, 5),
            (10, 9),
            (10, 10),
        ]
        for max_seq_len, stride_events in parameter_space:
                dataset = _dataset(
                    path,
                    ["item_id"],
                    cache=tmp_path / f"cache_{max_seq_len}_{stride_events}",
                    max_seq_len=max_seq_len,
                    min_seq_len=1,
                    window="sliding",
                    stride=stride_events / max_seq_len,
                    n_buckets=1,
                )
                expected = [
                    window
                    for items in user_items.values()
                    for window in _reference_sliding_windows(
                        items, max_seq_len, stride_events
                    )
                ]
                assert _item_sequences(dataset) == expected, (
                    f"max_seq_len={max_seq_len}, stride_events={stride_events}"
                )

    @pytest.mark.parametrize(
        ("num_events", "expected"),
        [
            (8, [[5, 6, 7], [2, 3, 4], [0, 1]]),
            # min_seq_len drops the oldest window once it is down to one event.
            (7, [[4, 5, 6], [1, 2, 3]]),
        ],
    )
    def test_windows_run_most_recent_first(
        self, tmp_path: Path, num_events: int, expected: list[list[int]]
    ) -> None:
        dataset = _sliding(
            _write_history(tmp_path, num_events), max_seq_len=3, stride_events=3
        )
        assert _item_sequences(dataset) == expected

    def test_long_history_is_partitioned_without_copying_it(
        self, tmp_path: Path
    ) -> None:
        dataset = _sliding(
            _write_history(tmp_path, 2000), max_seq_len=10, stride_events=10
        )

        sequences = _item_sequences(dataset)
        assert len(sequences) == 200
        assert sequences[0] == list(range(1990, 2000))
        assert sequences[-1] == list(range(10))

    def test_windows_roundtrip_collate(self, tmp_path: Path) -> None:
        dataset = _sliding(_write_history(tmp_path, 8), max_seq_len=3, stride_events=3)
        collated = collate_sequence_batch([dataset[i] for i in range(len(dataset))])

        assert collated["cumulative_lens"].tolist() == [0, 3, 6, 8]
        assert collated["timestamp"].tolist() == [5, 6, 7, 2, 3, 4, 0, 1]
        tags = collated["int_columns"]["tags"]
        assert tags.values.tolist() == [5, 6, 7, 2, 3, 4, 0, 1]
        assert tags.offsets.tolist() == list(range(9))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"window": "bogus"}, "window"),
            ({"window": "sliding", "stride": 0.0}, "stride"),
            ({"window": "sliding", "stride": 1.5}, "stride"),
        ],
    )
    def test_invalid_args_rejected(
        self, tmp_path: Path, kwargs: dict, message: str
    ) -> None:
        path = _write_history(tmp_path, 4)
        with pytest.raises(ValueError, match=message):
            _dataset(path, **kwargs)

    def test_a_whole_cache_is_not_served_to_a_sliding_run(self, tmp_path: Path) -> None:
        path = _write_history(tmp_path, 8)

        assert _item_sequences(_dataset(path)) == [[5, 6, 7]]
        assert _item_sequences(_dataset(path, window="sliding")) == [
            [5, 6, 7],
            [2, 3, 4],
            [0, 1],
        ]

    def test_short_stride_gets_more_automatic_buckets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = pl.DataFrame(
            {
                "uid": [uid for uid in range(64) for _ in range(12)],
                "timestamp": list(range(12)) * 64,
                "item_id": list(range(12)) * 64,
            }
        )
        path = tmp_path / "events.parquet"
        frame.write_parquet(path)
        monkeypatch.setattr(
            "dcn.data.sequence_dataset._available_ram_bytes",
            lambda: 16 * frame.estimated_size(),
        )

        whole = _dataset(path, ["item_id"], cache=tmp_path / "whole", max_seq_len=10)
        sliding = _dataset(
            path,
            ["item_id"],
            cache=tmp_path / "sliding",
            max_seq_len=10,
            window="sliding",
            stride=0.1,
        )

        assert len(whole.bucket_lengths) == 1
        assert len(sliding.bucket_lengths) > 1


class TestBucketedBuild:
    @pytest.mark.parametrize("window", ["whole", "sliding"])
    def test_multi_bucket_matches_single_bucket(
        self, multi_file_parquets: list[Path], tmp_path: Path, window: str
    ) -> None:
        common = dict(max_seq_len=4, window=window, stride=0.75)
        single = _dataset(
            multi_file_parquets, cache=tmp_path / "single", n_buckets=1, **common
        )
        multi = _dataset(
            multi_file_parquets, cache=tmp_path / "multi", n_buckets=4, **common
        )

        assert sum(length > 0 for length in multi.bucket_lengths) > 1
        assert len(multi) == len(single)
        assert _all_samples(multi) == _all_samples(single)

    def test_cache_survives_missing_raw_inputs(self, sample_parquet: Path) -> None:
        expected = _all_samples(_dataset(sample_parquet))

        sample_parquet.unlink()

        assert _all_samples(_dataset(sample_parquet)) == expected

    def test_invalidate_cache_rebuilds_from_raw_inputs(
        self, sample_parquet: Path
    ) -> None:
        assert len(_dataset(sample_parquet)) == 2

        extra_user = pl.DataFrame(
            {
                "uid": [103, 103],
                "timestamp": [1, 2],
                "item_id": [21, 22],
                "rating": [2.1, 2.2],
                "tags": [[21], [22]],
            }
        )
        pl.concat([pl.read_parquet(sample_parquet), extra_user]).write_parquet(
            sample_parquet
        )

        assert len(_dataset(sample_parquet)) == 2
        assert len(_dataset(sample_parquet, invalidate_cache=True)) == 3

    def test_interrupted_rebuild_invalidates_cache(
        self, sample_parquet: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        expected = _all_samples(_dataset(sample_parquet, max_seq_len=3))

        def interrupt(*args: object, **kwargs: object) -> str:
            raise RuntimeError("interrupted before metadata write")

        monkeypatch.setattr(json, "dumps", interrupt)
        with pytest.raises(RuntimeError):
            _dataset(sample_parquet, max_seq_len=2)
        monkeypatch.undo()

        assert _all_samples(_dataset(sample_parquet, max_seq_len=3)) == expected


class TestBucketShuffleSampler:
    def test_covers_every_index_exactly_once(
        self, multi_file_parquets: list[Path]
    ) -> None:
        dataset = _dataset(multi_file_parquets, n_buckets=3)
        sampler = BucketShuffleSampler(dataset)

        indices = list(sampler)
        assert len(sampler) == len(dataset)
        assert sorted(indices) == list(range(len(dataset)))

    def test_visits_each_bucket_contiguously(
        self, multi_file_parquets: list[Path]
    ) -> None:
        dataset = _dataset(multi_file_parquets, n_buckets=3)
        sampler = BucketShuffleSampler(
            dataset, generator=torch.Generator().manual_seed(0)
        )

        position = {index: order for order, index in enumerate(sampler)}
        offsets = [0, *accumulate(dataset.bucket_lengths)]
        for start, stop in zip(offsets, offsets[1:]):
            if start == stop:
                continue
            positions = sorted(position[index] for index in range(start, stop))
            assert positions == list(range(positions[0], positions[0] + stop - start))


class TestRowFilter:
    def test_it_removes_events_without_emitting_the_column_it_filtered_on(
        self, mixed_action_parquet: Path
    ) -> None:
        dataset = _dataset(
            mixed_action_parquet, row_filter=pl.col("action") == 1, n_buckets=1
        )

        assert _item_sequences(dataset) == [[10, 12]]
        assert "action" not in dataset[0]["int_columns"]

    def test_removing_everything_yields_an_empty_dataset(
        self, mixed_action_parquet: Path
    ) -> None:
        dataset = _dataset(
            mixed_action_parquet, row_filter=pl.col("action") == 99, n_buckets=1
        )

        assert len(dataset) == 0

    def test_changing_the_filter_rebuilds_the_cache(
        self, mixed_action_parquet: Path
    ) -> None:
        def build(action: int) -> SequenceDataset:
            return _dataset(
                mixed_action_parquet,
                row_filter=pl.col("action") == action,
                n_buckets=1,
            )

        assert _item_sequences(build(1)) == [[10, 12]]
        assert _item_sequences(build(0)) == [[11, 13], [21, 22]]
