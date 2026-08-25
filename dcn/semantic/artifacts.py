from __future__ import annotations

from pathlib import Path

import polars as pl
import torch


def load_item_embeddings(
    parquet_path: Path,
    *,
    compact_id_column: str = "compact_id",
    embedding_column: str = "normalized_embed",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(item_ids, embeddings)`` from a compact embedding table."""
    frame = pl.read_parquet(parquet_path).sort(compact_id_column)
    item_ids = torch.from_numpy(
        frame[compact_id_column].cast(pl.Int64).to_numpy(writable=True)
    )
    flat = frame[embedding_column].explode().cast(pl.Float32).to_numpy(writable=True)
    embeddings = torch.from_numpy(flat).reshape(len(frame), -1)
    return item_ids, embeddings
