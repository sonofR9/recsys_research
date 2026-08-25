from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from dcn.data.features import FeatureValues

from .types import ModuleWithDim


class PrecomputedEmbeddingLookup(ModuleWithDim):
    """Embedding table indexed by compact, contiguous ids 1..N; row 0 is unknown."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        learnable_default: bool,
        strict: bool,
    ):
        super().__init__()
        assert embeddings.dim() == 2, "embeddings must be 2D (N, D)"
        num_known, dimension = embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self._embedding_dim = dimension
        self._num_known_ids = num_known
        self._strict = strict

        if learnable_default:
            initial = torch.randn(dimension)
            initial = initial / initial.norm()
            self.default = nn.Parameter(initial)
        else:
            self.default = nn.Buffer(torch.zeros(dimension))

    @property
    def out_dim(self) -> int:
        return self._embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def num_known_ids(self) -> int:
        return self._num_known_ids

    def _per_id_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        valid = (ids >= 1) & (ids <= self._num_known_ids)
        if self._strict:
            assert bool(valid.all()), (
                f"out-of-range ids in strict mode: "
                f"{ids[~valid].unique().tolist()} (known range 1..{self._num_known_ids})"
            )
        safe_indices = torch.where(valid, ids - 1, torch.zeros_like(ids))
        known = self.embedding(safe_indices)
        return torch.where(valid.unsqueeze(-1), known, self.default.expand_as(known))

    def forward(self, item_ids: FeatureValues) -> torch.Tensor:
        return segment_sum(self._per_id_embeddings(item_ids.values), item_ids.offsets)

    @classmethod
    def from_parquet(
        cls,
        parquet_path: str | Path,
        learnable_default: bool,
        strict: bool,
        embedding_column: str = "normalized_embed",
        compact_id_column: str = "compact_id",
    ) -> "PrecomputedEmbeddingLookup":
        frame = pl.read_parquet(parquet_path)
        if compact_id_column in frame.columns:
            frame = frame.sort(compact_id_column)
            expected = pl.arange(1, frame.height + 1, eager=True)
            assert (frame[compact_id_column] == expected).all(), (
                f"{parquet_path} compact_id column must be contiguous 1..N"
            )
        # Flattening the list column and reshaping stays in numpy; the obvious
        # torch.tensor(series) walks it a row at a time and costs seconds here.
        flat = frame[embedding_column].explode().to_numpy()
        embeddings = torch.from_numpy(
            np.ascontiguousarray(flat.reshape(frame.height, -1), dtype=np.float32)
        )
        return cls(
            embeddings=embeddings, learnable_default=learnable_default, strict=strict
        )


def segment_sum(values: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Sum-pool ``values`` along dim 0 by EmbeddingBag-style offsets."""
    num_rows = offsets.shape[0] - 1
    lengths = offsets.diff()
    row_ids = torch.repeat_interleave(
        torch.arange(num_rows, device=values.device), lengths
    )
    output = values.new_zeros(num_rows, values.shape[1])
    output.index_add_(0, row_ids, values)
    return output
