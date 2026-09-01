import json
import logging
import math
import os
import re
import shutil
from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, Sampler

from .dataset import ColumnBuckets, bucket_columns_by_dtype
from .features import FeatureValues
from .packed import to_cumulative_lens
from utils.locks import hold

logger = logging.getLogger(__name__)

SYNTHETIC_OCCURRENCE_POSITION_COLUMN = "_g4_occurrence_position"

_GROUPING_COPIES = 4


def _available_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo").read_text()
    return int(re.search(r"MemAvailable:\s+(\d+) kB", meminfo).group(1)) * 1024


def _automatic_bucket_count(in_memory_bytes: int) -> int:
    per_bucket_budget = max(_available_ram_bytes() // (2 * _GROUPING_COPIES), 1)
    return max(1, math.ceil(in_memory_bytes / per_bucket_budget))


class SequenceDataset(Dataset):
    def __init__(
        self,
        parquet_files: list[Path],
        columns: list[str],
        cache_dir: Path,
        *,
        user_column: str,
        timestamp_column: str = "timestamp",
        max_seq_len: int = 100,
        min_seq_len: int = 2,
        window: Literal[
            "whole", "sliding", "next_item", "bounded_prefix"
        ] = "whole",
        stride: float = 1.0,
        prefix_length_rule: Literal["truncated", "required"] = "truncated",
        prefix_cap: int | None = None,
        emit_user_column: bool = False,
        row_filter: pl.Expr | None = None,
        n_buckets: int | None = None,
        invalidate_cache: bool = False,
    ):
        if window not in ("whole", "sliding", "next_item", "bounded_prefix"):
            raise ValueError(
                "window must be 'whole', 'sliding', 'next_item', or "
                "'bounded_prefix', "
                f"got {window!r}"
            )
        if window == "sliding" and not 0 < stride <= 1:
            raise ValueError(
                f"sliding window requires 0 < stride <= 1, got stride={stride}"
            )
        if prefix_length_rule not in ("truncated", "required"):
            raise ValueError(
                "prefix_length_rule must be 'truncated' or 'required', "
                f"got {prefix_length_rule!r}"
            )
        if window == "bounded_prefix" and (prefix_cap is None or prefix_cap < 1):
            raise ValueError(
                "bounded_prefix window requires a positive prefix_cap, "
                f"got {prefix_cap!r}"
            )

        # Polars cannot both group by a key and aggregate it per event, so the
        # user column is stripped here and re-broadcast in __getitem__.
        self.columns = [name for name in columns if name != user_column]
        self.occurrence_position_column = (
            SYNTHETIC_OCCURRENCE_POSITION_COLUMN
            if SYNTHETIC_OCCURRENCE_POSITION_COLUMN in self.columns
            else None
        )
        self.emit_user_column = emit_user_column or user_column in columns
        self.row_filter = row_filter
        self.user_column = user_column
        self.timestamp_column = timestamp_column
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.window = window
        self.stride = stride
        self.prefix_length_rule = prefix_length_rule
        self.prefix_cap = prefix_cap
        self._stride_events = max(1, round(max_seq_len * stride))

        self._cache_dir = Path(cache_dir)
        self._buckets_dir = self._cache_dir / "buckets"
        self._metadata_file = self._cache_dir / "metadata.json"
        self._completion_file = self._cache_dir / "complete"

        params = {
            "parquet_files": [str(Path(f).resolve()) for f in parquet_files],
            "columns": self.columns,
            "user_column": user_column,
            "emit_user_column": self.emit_user_column,
            "timestamp_column": timestamp_column,
            "max_seq_len": max_seq_len,
            "min_seq_len": min_seq_len,
            "window": window,
            "stride": stride,
            "row_filter": None if row_filter is None else str(row_filter),
            "n_buckets": n_buckets,
            **(
                {
                    "prefix_length_rule": prefix_length_rule,
                    "prefix_cap": prefix_cap,
                }
                if window == "bounded_prefix"
                else {}
            ),
        }
        cache_lock = self._cache_dir.with_name(f"{self._cache_dir.name}.lock")
        with hold(cache_lock, "sequence cache"):
            metadata = (
                None if invalidate_cache else self._load_cached_metadata(params)
            )
            if metadata is None:
                metadata = self._build_cache(
                    [Path(f) for f in parquet_files], params, n_buckets
                )
                logger.info(
                    "Built %s user sequences at %s in %s bucket(s) from %s parquet file(s)",
                    sum(metadata["bucket_lengths"]),
                    self._cache_dir,
                    len(metadata["bucket_files"]),
                    len(parquet_files),
                )
            else:
                logger.info("Loaded cached user sequences from %s", self._cache_dir)

        self.bucket_lengths: list[int] = metadata["bucket_lengths"]
        self.event_count: int = metadata["event_count"]
        self._bucket_paths = [
            self._buckets_dir / name for name in metadata["bucket_files"]
        ]
        self._cumulative_bucket_lengths = np.cumsum([0, *self.bucket_lengths])
        self._column_buckets = ColumnBuckets(
            int_names=metadata["int_columns"], float_names=metadata["float_columns"]
        )
        self._loaded_bucket_index: int | None = None
        self._loaded_bucket: pl.DataFrame | None = None

    @cached_property
    def original_user_count(self) -> int:
        return sum(
            pl.read_parquet(path, columns=[self.user_column])[
                self.user_column
            ].n_unique()
            for path in self._bucket_paths
        )

    def _load_cached_metadata(self, params: dict) -> dict | None:
        if not self._metadata_file.exists():
            return None
        metadata = json.loads(self._metadata_file.read_text())
        if metadata.get("params") != params:
            logger.info("Sequence cache at %s is stale; rebuilding", self._cache_dir)
            return None
        if not self._completion_file.exists():
            lock = self._completion_file.with_suffix(".lock")
            with hold(lock, "sequence cache migration"):
                if not self._completion_file.exists():
                    if not all(
                        (self._buckets_dir / name).exists()
                        for name in metadata["bucket_files"]
                    ):
                        return None
                    temporary = self._completion_file.with_suffix(
                        f".{os.getpid()}.tmp"
                    )
                    temporary.touch()
                    temporary.replace(self._completion_file)
        if "event_count" not in metadata:
            metadata["event_count"] = self._count_bucket_events(
                metadata["bucket_files"]
            )
            self._write_metadata(metadata)
        return metadata

    def _build_cache(
        self, parquet_files: list[Path], params: dict, n_buckets: int | None
    ) -> dict:
        if n_buckets is None:
            n_buckets = _automatic_bucket_count(
                self._estimate_in_memory_bytes(parquet_files)
            )
            logger.info(
                "Spilling %s parquet file(s) into %s bucket(s)",
                len(parquet_files),
                n_buckets,
            )

        self._completion_file.unlink(missing_ok=True)
        self._metadata_file.unlink(missing_ok=True)
        shards_dir = self._cache_dir / "shards"
        shutil.rmtree(shards_dir, ignore_errors=True)
        shutil.rmtree(self._buckets_dir, ignore_errors=True)
        self._buckets_dir.mkdir(parents=True)

        self._spill_into_bucket_shards(parquet_files, shards_dir, n_buckets)
        bucket_files, bucket_lengths, event_count = self._group_buckets(shards_dir)
        shutil.rmtree(shards_dir, ignore_errors=True)

        source_columns = [
            column
            for column in self.columns
            if column != self.occurrence_position_column
        ]
        buckets = bucket_columns_by_dtype(
            pl.read_parquet_schema(parquet_files[0]), source_columns
        )
        if self.occurrence_position_column is not None:
            buckets.int_names.append(self.occurrence_position_column)
        metadata = {
            "params": params,
            "bucket_files": bucket_files,
            "bucket_lengths": bucket_lengths,
            "event_count": event_count,
            "int_columns": buckets.int_names,
            "float_columns": buckets.float_names,
        }
        self._write_metadata(metadata)
        temporary_completion = self._completion_file.with_suffix(
            f".{os.getpid()}.tmp"
        )
        temporary_completion.touch()
        temporary_completion.replace(self._completion_file)
        return metadata

    def _write_metadata(self, metadata: dict) -> None:
        temporary_metadata = self._metadata_file.with_suffix(
            f".{os.getpid()}.tmp"
        )
        temporary_metadata.write_text(json.dumps(metadata))
        temporary_metadata.replace(self._metadata_file)

    def _count_bucket_events(self, bucket_files: list[str]) -> int:
        return sum(
            int(
                pl.scan_parquet(self._buckets_dir / name)
                .select(pl.col(self.timestamp_column).list.len().sum())
                .collect()
                .item()
                or 0
            )
            for name in bucket_files
        )

    @property
    def _emitted_columns(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    self.user_column,
                    self.timestamp_column,
                    *[
                        column
                        for column in self.columns
                        if column != self.occurrence_position_column
                    ],
                ]
            )
        )

    @property
    def _needed_columns(self) -> list[str]:
        filter_columns = (
            [] if self.row_filter is None else self.row_filter.meta.root_names()
        )
        return list(dict.fromkeys([*self._emitted_columns, *filter_columns]))

    def _estimate_in_memory_bytes(self, parquet_files: list[Path]) -> int:
        """Peak decoded size of the whole input, measured off one file."""
        sample = pl.read_parquet(parquet_files[0], columns=self._needed_columns)
        total = sample.estimated_size() * len(parquet_files)
        if self.window == "sliding":
            total *= math.ceil(self.max_seq_len / self._stride_events)
        elif self.window == "bounded_prefix":
            prefix_cap = self.prefix_cap
            assert prefix_cap is not None
            total *= prefix_cap
        return total

    def _spill_into_bucket_shards(
        self, parquet_files: list[Path], shards_dir: Path, n_buckets: int
    ) -> None:
        emitted_columns = self._emitted_columns

        for file_index, path in enumerate(parquet_files):
            frame = pl.read_parquet(path, columns=self._needed_columns)
            if self.row_filter is not None:
                frame = frame.filter(self.row_filter).select(emitted_columns)
            partitions = frame.with_columns(
                (pl.col(self.user_column).hash() % n_buckets).alias("_bucket")
            ).partition_by("_bucket", as_dict=True, include_key=False)
            for (bucket_index,), partition in partitions.items():
                bucket_dir = shards_dir / f"bucket_{bucket_index:05d}"
                bucket_dir.mkdir(parents=True, exist_ok=True)
                partition.write_parquet(bucket_dir / f"part_{file_index:05d}.parquet")

    def _group_buckets(
        self, shards_dir: Path
    ) -> tuple[list[str], list[int], int]:
        bucket_files: list[str] = []
        bucket_lengths: list[int] = []
        event_count = 0
        if not shards_dir.exists():
            return bucket_files, bucket_lengths, event_count
        for bucket_dir in sorted(shards_dir.iterdir()):
            # Named by input-file position, so the sorted concat restores the order
            # the stable sort breaks timestamp ties by.
            events = pl.concat(
                pl.read_parquet(f) for f in sorted(bucket_dir.glob("*.parquet"))
            )
            grouped = self._build_time_sorted_user_sequences(events)
            file_name = f"{bucket_dir.name}.parquet"
            grouped.write_parquet(self._buckets_dir / file_name)
            bucket_files.append(file_name)
            bucket_lengths.append(len(grouped))
            event_count += int(grouped[self.timestamp_column].list.len().sum() or 0)
        return bucket_files, bucket_lengths, event_count

    def _build_time_sorted_user_sequences(self, df: pl.DataFrame) -> pl.DataFrame:
        source_columns = [
            column
            for column in self.columns
            if column != self.occurrence_position_column
        ]
        events = df.select(
            pl.col(self.user_column),
            pl.col(self.timestamp_column).cast(pl.Int64),
            *[pl.col(c) for c in source_columns],
        ).sort([self.user_column, self.timestamp_column], maintain_order=True)
        if self.occurrence_position_column is not None:
            events = events.with_columns(
                pl.int_range(pl.len())
                .over(self.user_column)
                .alias(self.occurrence_position_column)
            )

        if self.window == "sliding":
            return self._group_into_sliding_windows(events)
        if self.window == "next_item":
            return self._group_into_next_item_windows(events)
        if self.window == "bounded_prefix":
            return self._group_into_bounded_prefix_windows(events)

        list_columns = [self.timestamp_column, *self.columns]
        return (
            events.group_by(self.user_column, maintain_order=True)
            .agg(pl.col(self.timestamp_column), *[pl.col(c) for c in self.columns])
            .filter(pl.col(self.timestamp_column).list.len() >= self.min_seq_len)
            .with_columns(pl.col(list_columns).list.tail(self.max_seq_len))
        )

    def _group_into_next_item_windows(self, events: pl.DataFrame) -> pl.DataFrame:
        position = pl.int_range(pl.len()).over(self.user_column)
        window = position // self.max_seq_len
        begins_window = (position > 0) & (position % self.max_seq_len == 0)

        return (
            events.with_columns(
                pl.int_ranges(
                    window - begins_window.cast(pl.Int64), window + 1
                ).alias("_window")
            )
            .explode("_window")
            .group_by([self.user_column, "_window"], maintain_order=True)
            .agg(
                pl.col(self.timestamp_column),
                *[pl.col(c) for c in self.columns],
            )
            .filter(pl.col(self.timestamp_column).list.len() >= self.min_seq_len)
            .sort([self.user_column, "_window"], maintain_order=True)
            .drop("_window")
        )

    def _group_into_bounded_prefix_windows(
        self, events: pl.DataFrame
    ) -> pl.DataFrame:
        prefix_cap = self.prefix_cap
        assert prefix_cap is not None
        position = pl.int_range(pl.len()).over(self.user_column)
        user_length = pl.len().over(self.user_column).cast(pl.Int64)
        latest_target = user_length - 1

        if self.prefix_length_rule == "truncated":
            earliest_target = pl.max_horizontal(1, user_length - prefix_cap)
        else:
            earliest_target = (
                pl.when(latest_target < self.max_seq_len)
                .then(latest_target)
                .otherwise(
                    pl.max_horizontal(
                        self.max_seq_len, user_length - prefix_cap
                    )
                )
            )

        first_target = pl.max_horizontal(position, earliest_target)
        last_target = pl.min_horizontal(
            position + self.max_seq_len, latest_target
        )
        list_columns = [self.timestamp_column, *self.columns]

        return (
            events.with_columns(
                pl.int_ranges(first_target, last_target + 1).alias("_target")
            )
            .explode("_target")
            .drop_nulls("_target")
            .group_by([self.user_column, "_target"], maintain_order=True)
            .agg(pl.col(self.timestamp_column), *[pl.col(c) for c in self.columns])
            .filter(pl.col(self.timestamp_column).list.len() >= self.min_seq_len)
            .sort(
                [self.user_column, "_target"],
                descending=[False, True],
                maintain_order=True,
            )
            .drop("_target")
            .select(self.user_column, *list_columns)
        )

    def _group_into_sliding_windows(self, events: pl.DataFrame) -> pl.DataFrame:
        """Windows built by labelling events, not by copying histories."""
        stride = self._stride_events
        # Signed: an unsigned length underflows once max_seq_len exceeds it.
        user_length = pl.len().over(self.user_column).cast(pl.Int64)
        # Numbered back from the newest event, so window 0 is always full and
        # only the oldest is truncated.
        from_end = user_length - 1 - pl.int_range(pl.len()).over(self.user_column)

        first_window = (
            pl.when(from_end >= self.max_seq_len)
            .then((from_end - self.max_seq_len) // stride + 1)
            .otherwise(0)
        )
        oldest_window = pl.max_horizontal(
            0, (user_length - self.max_seq_len + stride - 1) // stride
        )
        last_window = pl.min_horizontal(from_end // stride, oldest_window)

        return (
            events.with_columns(
                pl.int_ranges(first_window, last_window + 1).alias("_window")
            )
            .explode("_window")
            .drop_nulls("_window")
            .group_by([self.user_column, "_window"], maintain_order=True)
            .agg(pl.col(self.timestamp_column), *[pl.col(c) for c in self.columns])
            .filter(pl.col(self.timestamp_column).list.len() >= self.min_seq_len)
            .sort([self.user_column, "_window"], maintain_order=True)
            .drop("_window")
        )

    # Workers start with forkserver, so an attached bucket would be pickled
    # to each of them rather than re-read on the other side.
    def __getstate__(self) -> dict:
        return {**self.__dict__, "_loaded_bucket": None, "_loaded_bucket_index": None}

    def _bucket_frame(self, bucket_index: int) -> pl.DataFrame:
        if bucket_index != self._loaded_bucket_index:
            self._loaded_bucket = pl.read_parquet(self._bucket_paths[bucket_index])
            self._loaded_bucket_index = bucket_index
        return self._loaded_bucket

    def __len__(self) -> int:
        return int(self._cumulative_bucket_lengths[-1])

    def __getitem__(self, index: int) -> dict:
        bucket_index = (
            int(np.searchsorted(self._cumulative_bucket_lengths, index, side="right"))
            - 1
        )
        local_index = index - int(self._cumulative_bucket_lengths[bucket_index])
        frame = self._bucket_frame(bucket_index)
        timestamps = frame[self.timestamp_column][local_index].to_list()
        int_columns = {
            name: frame[name][local_index].to_list()
            for name in self._column_buckets.int_names
        }
        if self.emit_user_column:
            user_id = int(frame[self.user_column][local_index])
            int_columns[self.user_column] = [user_id] * len(timestamps)

        return {
            "int_columns": int_columns,
            "float_columns": {
                name: frame[name][local_index].to_list()
                for name in self._column_buckets.float_names
            },
            "timestamp": timestamps,
        }

    def __getitems__(self, indices: list[int]) -> list[dict] | dict:
        if not indices:
            return []
        absolute = np.asarray(indices)
        buckets = (
            np.searchsorted(self._cumulative_bucket_lengths, absolute, side="right") - 1
        )
        unique_buckets = np.unique(buckets)
        single_bucket = (
            self._bucket_frame(int(unique_buckets[0]))
            if len(unique_buckets) == 1
            else None
        )
        scalar_int_columns = single_bucket is not None and all(
            single_bucket.schema[name].inner.is_integer()
            or single_bucket.schema[name].inner == pl.Boolean
            for name in self._column_buckets.int_names
        )
        if (
            len(unique_buckets) == 1
            and not self._column_buckets.float_names
            and scalar_int_columns
        ):
            bucket_index = int(unique_buckets[0])
            local = absolute - self._cumulative_bucket_lengths[bucket_index]
            frame = single_bucket[local.tolist()]
            lengths = torch.from_numpy(
                frame[self.timestamp_column].list.len().to_numpy().astype(np.int64)
            )
            int_columns = {
                name: self._dense_feature_values(frame[name].explode().to_numpy())
                for name in self._column_buckets.int_names
            }
            if self.emit_user_column:
                users = np.repeat(frame[self.user_column].to_numpy(), lengths.numpy())
                int_columns[self.user_column] = self._dense_feature_values(users)
            return {
                "int_columns": int_columns,
                "float_columns": {},
                "timestamp": torch.from_numpy(
                    frame[self.timestamp_column].explode().to_numpy().astype(np.int64)
                ),
                "cumulative_lens": to_cumulative_lens(lengths),
            }
        samples: list[dict | None] = [None] * len(indices)
        for bucket_index in unique_buckets:
            positions = np.flatnonzero(buckets == bucket_index)
            local = absolute[positions] - self._cumulative_bucket_lengths[bucket_index]
            frame = self._bucket_frame(int(bucket_index))[local.tolist()]
            timestamps = frame[self.timestamp_column].to_list()
            int_columns = {
                name: frame[name].to_list() for name in self._column_buckets.int_names
            }
            float_columns = {
                name: frame[name].to_list() for name in self._column_buckets.float_names
            }
            users = frame[self.user_column].to_list() if self.emit_user_column else None
            for row, position in enumerate(positions):
                row_int_columns = {
                    name: values[row] for name, values in int_columns.items()
                }
                if users is not None:
                    row_int_columns[self.user_column] = [users[row]] * len(
                        timestamps[row]
                    )
                samples[int(position)] = {
                    "int_columns": row_int_columns,
                    "float_columns": {
                        name: values[row] for name, values in float_columns.items()
                    },
                    "timestamp": timestamps[row],
                }
        return [sample for sample in samples if sample is not None]

    @staticmethod
    def _dense_feature_values(values: np.ndarray) -> FeatureValues:
        tensor = torch.from_numpy(values.astype(np.int64))
        return FeatureValues(tensor, torch.arange(tensor.shape[0] + 1))


class BucketShuffleSampler(Sampler[int]):
    def __init__(
        self, dataset: SequenceDataset, generator: torch.Generator | None = None
    ):
        self._bucket_offsets = np.cumsum([0, *dataset.bucket_lengths])
        self._generator = generator

    def __len__(self) -> int:
        return int(self._bucket_offsets[-1])

    def __iter__(self) -> Iterator[int]:
        bucket_count = len(self._bucket_offsets) - 1
        for bucket_index in torch.randperm(
            bucket_count, generator=self._generator
        ).tolist():
            start = int(self._bucket_offsets[bucket_index])
            length = int(self._bucket_offsets[bucket_index + 1]) - start
            for local_index in torch.randperm(
                length, generator=self._generator
            ).tolist():
                yield start + local_index
