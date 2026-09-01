from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch


@dataclass(frozen=True)
class FeatureDataSummary:
    validation_cutoff_timestamp: int
    num_items: int
    training_rows: int
    training_users: int
    artist_vocab_size: int
    album_vocab_size: int


@dataclass(frozen=True)
class LoadedFeatureData:
    training_counts: torch.Tensor
    training_history_lengths: dict[int, int]
    artist_rows: tuple[tuple[int, ...], ...]
    album_rows: tuple[tuple[int, ...], ...]
    artist_vocab_size: int
    album_vocab_size: int


def materialize_feature_data(
    *,
    events_path: Path,
    remap_path: Path,
    destination: Path,
    validation_interval_seconds: int,
) -> FeatureDataSummary:
    if validation_interval_seconds < 1:
        raise ValueError("validation_interval_seconds must be positive")
    remap = pl.read_parquet(remap_path, columns=["compact_id"]).sort("compact_id")
    expected = pl.Series("compact_id", range(1, remap.height + 1), dtype=pl.Int64)
    compact_ids = remap["compact_id"].cast(pl.Int64)
    if compact_ids.n_unique() != remap.height or not compact_ids.equals(expected):
        raise ValueError("item remap compact IDs must be contiguous 1..N")
    num_items = remap.height

    events = pl.scan_parquet(events_path)
    maximum = events.select(pl.col("timestamp").max()).collect().item()
    if maximum is None:
        raise ValueError("events are empty")
    cutoff = int(maximum) - validation_interval_seconds
    train = events.filter(pl.col("timestamp") < cutoff)
    bounds = train.select(
        pl.col("compact_item_id").min().alias("minimum"),
        pl.col("compact_item_id").max().alias("maximum"),
        pl.len().alias("rows"),
    ).collect()
    training_rows = int(bounds["rows"].item())
    if training_rows == 0:
        raise ValueError("training split is empty")
    if int(bounds["minimum"].item()) < 1 or int(bounds["maximum"].item()) > num_items:
        raise ValueError("training data contains an item outside the compact catalog")

    item_counts = (
        train.group_by("compact_item_id")
        .agg(pl.len().cast(pl.Int64).alias("training_count"))
        .collect()
    )
    histories = (
        train.group_by("uid")
        .agg(pl.len().cast(pl.Int64).alias("training_history_length"))
        .sort("uid")
        .collect()
    )
    artists, artist_vocab = _compact_feature(train, "artist_id")
    albums, album_vocab = _compact_feature(train, "album_id")

    item_features = (
        pl.DataFrame({"compact_item_id": range(1, num_items + 1)})
        .join(item_counts, on="compact_item_id", how="left")
        .join(artists, on="compact_item_id", how="left")
        .join(albums, on="compact_item_id", how="left")
        .with_columns(
            pl.col("training_count").fill_null(0),
            pl.col("artist_compact_ids").fill_null(pl.lit([], dtype=pl.List(pl.Int64))),
            pl.col("album_compact_ids").fill_null(pl.lit([], dtype=pl.List(pl.Int64))),
        )
        .sort("compact_item_id")
    )

    destination.mkdir(parents=True, exist_ok=True)
    _write_parquet(item_features, destination / "item_features.parquet")
    _write_parquet(histories, destination / "training_user_histories.parquet")
    _write_parquet(artist_vocab, destination / "artist_vocab.parquet")
    _write_parquet(album_vocab, destination / "album_vocab.parquet")
    return FeatureDataSummary(
        validation_cutoff_timestamp=cutoff,
        num_items=num_items,
        training_rows=training_rows,
        training_users=histories.height,
        artist_vocab_size=artist_vocab.height,
        album_vocab_size=album_vocab.height,
    )


def load_feature_data(path: Path) -> LoadedFeatureData:
    frame = pl.read_parquet(path).sort("compact_item_id")
    expected = pl.Series("compact_item_id", range(1, frame.height + 1), dtype=pl.Int64)
    if not frame["compact_item_id"].cast(pl.Int64).equals(expected):
        raise ValueError("feature data compact IDs must be contiguous 1..N")
    parent = path.parent
    histories = pl.read_parquet(parent / "training_user_histories.parquet").sort("uid")
    artist_vocab = pl.read_parquet(parent / "artist_vocab.parquet")
    album_vocab = pl.read_parquet(parent / "album_vocab.parquet")
    return LoadedFeatureData(
        training_counts=torch.tensor(
            [0, *frame["training_count"].to_list()], dtype=torch.int64
        ),
        training_history_lengths=dict(
            zip(
                histories["uid"].to_list(),
                histories["training_history_length"].to_list(),
                strict=True,
            )
        ),
        artist_rows=((), *map(tuple, frame["artist_compact_ids"].to_list())),
        album_rows=((), *map(tuple, frame["album_compact_ids"].to_list())),
        artist_vocab_size=artist_vocab.height,
        album_vocab_size=album_vocab.height,
    )


def _compact_feature(
    train: pl.LazyFrame, column: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    raw_column = f"raw_{column}"
    compact_column = column.removesuffix("_id") + "_compact_id"
    pairs = (
        train.select("compact_item_id", pl.col(column).alias(raw_column))
        .explode(raw_column)
        .filter(pl.col(raw_column).is_not_null() & (pl.col(raw_column) > 0))
        .select("compact_item_id", pl.col(raw_column).cast(pl.Int64))
        .unique()
    )
    values = pairs.select(raw_column).unique().sort(raw_column).collect()
    vocab = values.with_row_index(compact_column, offset=1).select(
        raw_column, pl.col(compact_column).cast(pl.Int64)
    )
    grouped = (
        pairs.join(vocab.lazy(), on=raw_column, how="inner")
        .group_by("compact_item_id")
        .agg(
            pl.col(compact_column)
            .sort()
            .alias(f"{column.removesuffix('_id')}_compact_ids")
        )
        .collect()
    )
    return grouped, vocab


def _write_parquet(frame: pl.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(destination)
