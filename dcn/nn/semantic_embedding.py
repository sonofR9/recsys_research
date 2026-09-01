from __future__ import annotations

from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dcn.data.features import FeatureValues
from dcn.semantic import ResidualCodebooks, SemanticCodes

from .precomputed_embeddings import segment_sum
from .types import ModuleWithDim

Aggregation = Literal["concat", "sum"]


class _FrozenCodebookWithSuffix(nn.Module):
    def __init__(
        self,
        weights: torch.Tensor,
        suffix_start: int,
        suffix_size: int,
    ) -> None:
        super().__init__()
        self.register_buffer("weights", weights)
        self.suffix = nn.Embedding(suffix_size, weights.shape[1], padding_idx=0)
        self.suffix_start = suffix_start
        self.embedding_dim = weights.shape[1]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        suffix_ranks = (tokens - self.suffix_start).clamp(
            min=0, max=self.suffix.num_embeddings - 1
        )
        is_suffix = tokens >= self.suffix_start
        return F.embedding(tokens, self.weights) + self.suffix(suffix_ranks) * (
            is_suffix.unsqueeze(-1)
        )


class SemanticIdEmbedding(ModuleWithDim):
    """Describes an item by the codes it was quantized to."""

    def __init__(
        self,
        item_tokens: torch.Tensor,
        embedding: nn.Module,
        aggregation: Aggregation = "concat",
    ):
        super().__init__()
        assert aggregation in ("concat", "sum"), f"unknown aggregation {aggregation!r}"
        self.register_buffer("item_tokens", item_tokens)
        self.embedding = embedding
        self.aggregation = aggregation

    @classmethod
    def learned(
        cls,
        codes: SemanticCodes,
        num_items: int,
        embedding_dim: int,
        aggregation: Aggregation = "concat",
    ) -> SemanticIdEmbedding:
        vocabulary = codes.vocabulary
        embedding = nn.Embedding(vocabulary.size, embedding_dim)
        return cls(cls._item_tokens(codes, num_items), embedding, aggregation)

    @classmethod
    def from_codebooks(
        cls,
        codes: SemanticCodes,
        codebooks: ResidualCodebooks,
        num_items: int,
        *,
        train_collision_suffix: bool = False,
    ) -> SemanticIdEmbedding:
        """Frozen table holding the centroid each code stands for."""
        vocabulary = codes.vocabulary
        assert codes.num_levels - codebooks.num_levels in (0, 1), (
            f"{codes.num_levels} code levels cannot come from"
            f" {codebooks.num_levels} codebooks"
        )
        weights = torch.zeros(vocabulary.size, codebooks.dim)
        for level in range(codebooks.num_levels):
            first, last = vocabulary.level_range(level)
            weights[first:last] = codebooks.centroids[level]
        if train_collision_suffix:
            if codes.num_levels != codebooks.num_levels + 1:
                raise ValueError(
                    "a trainable collision suffix requires one code level after the"
                    " residual codebooks"
                )
            suffix_start, _ = vocabulary.level_range(codebooks.num_levels)
            embedding: nn.Module = _FrozenCodebookWithSuffix(
                weights,
                suffix_start=suffix_start,
                suffix_size=codes.codes_per_level[-1],
            )
        else:
            embedding = nn.Embedding.from_pretrained(weights, freeze=True)
        return cls(cls._item_tokens(codes, num_items), embedding, aggregation="sum")

    @staticmethod
    def _item_tokens(codes: SemanticCodes, num_items: int) -> torch.Tensor:
        return codes.vocabulary.tokens(codes.lookup_table(num_items))

    @property
    def num_levels(self) -> int:
        return self.item_tokens.shape[1]

    @property
    def level_dim(self) -> int:
        return self.embedding.embedding_dim

    @property
    def out_dim(self) -> int:
        dim = self.level_dim
        return dim * self.num_levels if self.aggregation == "concat" else dim

    def tokens(self, item_ids: torch.Tensor) -> torch.Tensor:
        """``[..., levels]`` vocabulary token of each item's code at each level."""
        known = item_ids < len(self.item_tokens)
        # Row 0, not a clamp: an id past the end is unknown, not the last item.
        return self.item_tokens[torch.where(known, item_ids, 0)]

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    def per_level(self, item_ids: torch.Tensor) -> torch.Tensor:
        """``[..., levels, level_dim]``, before the levels are pooled."""
        return self.embed_tokens(self.tokens(item_ids))

    def forward(self, item_ids: FeatureValues) -> torch.Tensor:
        per_level = self.per_level(item_ids.values)
        if self.aggregation == "concat":
            per_id = per_level.flatten(start_dim=1)
        else:
            per_id = per_level.sum(dim=1)
        return segment_sum(per_id, item_ids.offsets)


class CombinedSemanticIdEmbedding(nn.Module):
    """Several views of one code tuple, laid side by side at every level."""

    def __init__(self, embeddings: Sequence[SemanticIdEmbedding]):
        super().__init__()
        assert embeddings, "nothing to combine"
        assert len({embedding.num_levels for embedding in embeddings}) == 1, (
            "combined views must describe the same code tuple"
        )
        self.embeddings = nn.ModuleList(embeddings)

    @property
    def num_levels(self) -> int:
        return self.embeddings[0].num_levels

    @property
    def level_dim(self) -> int:
        return sum(embedding.level_dim for embedding in self.embeddings)

    def tokens(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[0].tokens(item_ids)

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [embedding.embed_tokens(tokens) for embedding in self.embeddings], dim=-1
        )
