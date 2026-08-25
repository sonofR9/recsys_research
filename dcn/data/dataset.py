import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from .features import FeatureValues, as_sequence, ragged_from_rows

logger = logging.getLogger(__name__)


class ColumnBuckets(NamedTuple):
    int_names: list[str]
    float_names: list[str]


def bucket_columns_by_dtype(schema: pl.Schema, columns: list[str]) -> ColumnBuckets:
    int_names: list[str] = []
    float_names: list[str] = []
    for name in columns:
        dtype = schema[name]
        value_dtype = dtype.inner if isinstance(dtype, (pl.List, pl.Array)) else dtype
        if value_dtype.is_float():
            float_names.append(name)
        elif value_dtype.is_integer() or value_dtype == pl.Boolean:
            int_names.append(name)
        else:
            raise ValueError(f"Column {name!r} has unsupported dtype {dtype}")
    return ColumnBuckets(int_names=int_names, float_names=float_names)


def _column_to_numpy(series: pl.Series, scalar_dtype: type[pl.DataType]) -> np.ndarray:
    if isinstance(series.dtype, (pl.List, pl.Array)):
        return series.to_numpy()
    return series.cast(scalar_dtype).to_numpy()


class _Rows(NamedTuple):
    timestamps: np.ndarray
    int_columns: dict[str, np.ndarray]
    float_columns: dict[str, np.ndarray]


class EventDataset(Dataset):
    """Rows of a parquet as generic int/float columns, bucketed by parquet dtype.

    Roles (feature/target/mask/counter) live with the consumers, by name.
    """

    def __init__(
        self,
        parquet_files: list[Path],
        columns: list[str],
        timestamp_column: str = "timestamp",
    ):
        self.parquet_files = list(parquet_files)
        self.columns = columns
        self.timestamp_column = timestamp_column
        self._rows: _Rows | None = None

    @property
    def _loaded_rows(self) -> _Rows:
        if self._rows is None:
            self._rows = self._read()
        return self._rows

    def _read(self) -> _Rows:
        frame = pl.concat([pl.read_parquet(f) for f in self.parquet_files])
        logger.debug(
            "Loaded %s samples from %s parquet file(s)",
            len(frame),
            len(self.parquet_files),
        )
        buckets = bucket_columns_by_dtype(frame.schema, self.columns)
        return _Rows(
            timestamps=frame[self.timestamp_column].cast(pl.Int64).to_numpy(),
            int_columns={
                name: _column_to_numpy(frame[name], pl.Int64)
                for name in buckets.int_names
            },
            float_columns={
                name: _column_to_numpy(frame[name], pl.Float32)
                for name in buckets.float_names
            },
        )

    # Workers start with forkserver, so the parquet is re-read on the other
    # side rather than shipped decoded.
    def __getstate__(self) -> dict:
        return {**self.__dict__, "_rows": None}

    def __len__(self) -> int:
        return len(self._loaded_rows.timestamps)

    def __getitem__(self, index: int) -> dict:
        rows = self._loaded_rows
        return {
            "int_columns": {
                name: values[index] for name, values in rows.int_columns.items()
            },
            "float_columns": {
                name: values[index] for name, values in rows.float_columns.items()
            },
            "timestamp": int(rows.timestamps[index]),
        }


def _collate(batch: list[dict], *, is_sequence_batch: bool) -> dict:
    def rows(columns_key: str, name: str) -> list[list]:
        values = [sample[columns_key][name] for sample in batch]
        if is_sequence_batch:
            return [as_sequence(cell) for value in values for cell in value]
        return [as_sequence(value) for value in values]

    def to_ragged(columns_key: str, dtype: torch.dtype) -> dict[str, FeatureValues]:
        return {
            name: ragged_from_rows(rows(columns_key, name), dtype)
            for name in batch[0][columns_key]
        }

    sequence_lengths = [len(as_sequence(sample["timestamp"])) for sample in batch]

    return {
        "int_columns": to_ragged("int_columns", torch.int64),
        "float_columns": to_ragged("float_columns", torch.float32),
        "timestamp": torch.tensor(
            [t for sample in batch for t in as_sequence(sample["timestamp"])],
            dtype=torch.int64,
        ),
        "cumulative_lens": torch.tensor(
            [0, *sequence_lengths], dtype=torch.int64
        ).cumsum(0),
    }


def collate_event_batch(batch: list[dict]) -> dict:
    return _collate(batch, is_sequence_batch=False)


def collate_sequence_batch(batch: list[dict]) -> dict:
    """One block of rows per sample, as :class:`SequenceDataset` produces them."""
    if isinstance(batch, dict):
        return batch
    return _collate(batch, is_sequence_batch=True)
