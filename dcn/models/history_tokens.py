"""Turning a user's events into the token sequence a causal model reads."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn

from dcn.data.features import FeatureValues

from dcn.nn.semantic_embedding import CombinedSemanticIdEmbedding, SemanticIdEmbedding
from dcn.nn.types import ModuleWithDim


@dataclass
class TokenizedHistory:
    embeddings: torch.Tensor
    cumulative_lens: torch.Tensor
    # Tokens the model must predict. An action token, or a semantic id's later
    # levels, are context it reads but never has to produce.
    is_target: torch.Tensor
    item_ids: torch.Tensor
    timestamps: torch.Tensor
    # None when the tokens span more than one vocabulary: no single id names
    # both an item and an action. Only a decoder needs these.
    token_ids: torch.Tensor | None = None
    is_query: torch.Tensor | None = None


class EventTokenizer(ModuleWithDim):
    """Expands a packed batch of events into a packed batch of tokens."""

    def __init__(self, item_id_column: str):
        super().__init__()
        self.item_id_column = item_id_column

    @property
    @abstractmethod
    def tokens_per_event(self) -> int: ...

    @abstractmethod
    def forward(self, batch: dict) -> TokenizedHistory: ...

    def _expand(
        self,
        batch: dict,
        embeddings: torch.Tensor,
        is_target: torch.Tensor,
        token_ids: torch.Tensor | None = None,
    ) -> TokenizedHistory:
        item_ids = self._column(batch, self.item_id_column).dense()
        repeats = self.tokens_per_event
        return TokenizedHistory(
            embeddings=embeddings,
            cumulative_lens=batch["cumulative_lens"] * repeats,
            is_target=is_target,
            item_ids=item_ids.repeat_interleave(repeats),
            timestamps=batch["timestamp"].repeat_interleave(repeats),
            token_ids=token_ids,
        )

    @staticmethod
    def _column(batch: dict, name: str) -> FeatureValues:
        return batch["int_columns"][name]


class ItemTokenizer(EventTokenizer):
    """One token per event: the item it happened on."""

    def __init__(
        self,
        item_embedding: nn.Module,
        item_id_column: str,
        projection: nn.Linear | None = None,
    ):
        super().__init__(item_id_column)
        self.item_embedding = item_embedding
        self.projection = projection

    @property
    def tokens_per_event(self) -> int:
        return 1

    @property
    def out_dim(self) -> int:
        return (
            self.item_embedding.embedding_dim
            if self.projection is None
            else self.projection.out_features
        )

    def forward(self, batch: dict) -> TokenizedHistory:
        item_ids = self._column(batch, self.item_id_column).dense()
        embeddings = self.item_embedding(item_ids)
        if self.projection is not None:
            embeddings = self.projection(embeddings)
        return self._expand(
            batch,
            embeddings,
            torch.ones_like(item_ids, dtype=torch.bool),
            token_ids=item_ids,
        )


class TimestampDeltaTokenizer(EventTokenizer):
    def __init__(
        self,
        tokenizer: EventTokenizer,
        *,
        kind: Literal["plain", "log", "bins"],
        combination: Literal["add", "concat"] = "add",
        num_bins: int = 32,
        max_delta_seconds: float = 30 * 86_400,
    ) -> None:
        super().__init__(tokenizer.item_id_column)
        if tokenizer.tokens_per_event != 1:
            raise ValueError("timestamp deltas require one token per event")
        self.tokenizer = tokenizer
        self.kind = kind
        self.combination = combination
        self.max_delta_seconds = max_delta_seconds
        dim = tokenizer.out_dim
        if kind == "bins":
            self.time_encoder: nn.Module = nn.Embedding(num_bins, dim)
            bin_boundaries = torch.linspace(0, 1, num_bins - 1)
            if num_bins > 2:
                bin_boundaries[-1] = torch.nextafter(
                    bin_boundaries[-1], bin_boundaries.new_zeros(())
                )
            self.register_buffer("bin_boundaries", bin_boundaries)
        else:
            self.time_encoder = nn.Sequential(
                nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim)
            )
            self.register_buffer("bin_boundaries", None)
        self.fusion = (
            nn.Sequential(
                nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
            )
            if combination == "concat"
            else None
        )

    @property
    def tokens_per_event(self) -> int:
        return 1

    @property
    def out_dim(self) -> int:
        return self.tokenizer.out_dim

    def _normalized_deltas(self, tokens: TokenizedHistory) -> torch.Tensor:
        timestamps = tokens.timestamps
        deltas = torch.zeros_like(timestamps)
        if timestamps.numel() > 1:
            deltas[1:] = timestamps[1:] - timestamps[:-1]
        deltas[tokens.cumulative_lens[:-1].long()] = 0
        deltas = deltas.clamp(min=0, max=self.max_delta_seconds).float()
        if self.kind in {"log", "bins"}:
            return torch.log1p(deltas) / torch.log1p(
                deltas.new_tensor(self.max_delta_seconds)
            )
        return deltas / self.max_delta_seconds

    def forward(self, batch: dict) -> TokenizedHistory:
        tokens = self.tokenizer(batch)
        values = self._normalized_deltas(tokens)
        time_embedding = (
            self.time_encoder(torch.bucketize(values, self.bin_boundaries))
            if self.kind == "bins"
            else self.time_encoder(values.unsqueeze(1))
        )
        embeddings = (
            tokens.embeddings + time_embedding
            if self.fusion is None
            else self.fusion(torch.cat([tokens.embeddings, time_embedding], dim=1))
        )
        return replace(tokens, embeddings=embeddings)


class BosTokenizer(EventTokenizer):
    """Prepends one learned token to every sequence.

    It is a token like any other rather than something the transformer inserts,
    because the model has to be able to predict the item that follows it -- and
    a position the target layout cannot see is a position nothing is trained
    on.
    """

    def __init__(self, tokenizer: EventTokenizer):
        super().__init__(tokenizer.item_id_column)
        if tokenizer.tokens_per_event != 1:
            raise ValueError(
                "a bos token would shift every slot of a multi-token event, and"
                f" {type(tokenizer).__name__} emits"
                f" {tokenizer.tokens_per_event} tokens per event"
            )
        self.tokenizer = tokenizer
        self.bos = nn.Parameter(torch.zeros(tokenizer.out_dim))
        nn.init.trunc_normal_(self.bos, std=0.02)

    @property
    def tokens_per_event(self) -> int:
        return self.tokenizer.tokens_per_event

    @property
    def out_dim(self) -> int:
        return self.tokenizer.out_dim

    def forward(self, batch: dict) -> TokenizedHistory:
        tokens = self.tokenizer(batch)
        starts = tokens.cumulative_lens[:-1]
        grown, is_bos = _insert_at(tokens.embeddings, starts, self.bos)
        return TokenizedHistory(
            embeddings=grown,
            cumulative_lens=tokens.cumulative_lens
            + torch.arange(
                tokens.cumulative_lens.shape[0],
                device=starts.device,
                dtype=tokens.cumulative_lens.dtype,
            ),
            is_target=_widen(tokens.is_target, is_bos, starts, False),
            # The ids and the timestamp repeat the token the bos precedes. It
            # sits first in its sequence and is not a target, so nothing scores
            # against it; repeating keeps every column a valid lookup.
            item_ids=_widen(tokens.item_ids, is_bos, starts, None),
            timestamps=_widen(tokens.timestamps, is_bos, starts, None),
            token_ids=(
                None
                if tokens.token_ids is None
                else _widen(tokens.token_ids, is_bos, starts, None)
            ),
        )


class EndOfHistoryQueryToken(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def for_training(self, tokens: TokenizedHistory) -> TokenizedHistory:
        lengths = tokens.cumulative_lens.diff()
        eligible = lengths >= 2
        insertions = tokens.cumulative_lens[1:][eligible] - 1
        query_mask = (
            torch.ones_like(tokens.is_target)
            if tokens.is_query is None
            else tokens.is_query.clone()
        )
        query_mask[insertions - 1] = False
        return self._insert(tokens, insertions, eligible, query_mask)

    def for_inference(self, tokens: TokenizedHistory) -> TokenizedHistory:
        lengths = tokens.cumulative_lens.diff()
        eligible = lengths > 0
        insertions = tokens.cumulative_lens[1:][eligible]
        query_mask = (
            torch.ones_like(tokens.is_target)
            if tokens.is_query is None
            else tokens.is_query
        )
        return self._insert(tokens, insertions, eligible, query_mask)

    def _insert(
        self,
        tokens: TokenizedHistory,
        insertions: torch.Tensor,
        eligible: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> TokenizedHistory:
        grown, is_query_token = _insert_at(
            tokens.embeddings, insertions, self.embedding
        )
        previous = insertions - 1
        return TokenizedHistory(
            embeddings=grown,
            cumulative_lens=tokens.cumulative_lens
            + torch.cat(
                [
                    tokens.cumulative_lens.new_zeros(1),
                    eligible.to(tokens.cumulative_lens.dtype).cumsum(0),
                ]
            ),
            is_target=_widen(tokens.is_target, is_query_token, insertions, False),
            item_ids=_widen(tokens.item_ids, is_query_token, insertions, 0),
            timestamps=_widen(
                tokens.timestamps,
                is_query_token,
                insertions,
                tokens.timestamps[previous],
            ),
            token_ids=(
                None
                if tokens.token_ids is None
                else _widen(tokens.token_ids, is_query_token, insertions, 0)
            ),
            is_query=_widen(query_mask, is_query_token, insertions, True),
        )


class InterleavedQueryToken(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def forward(self, tokens: TokenizedHistory) -> TokenizedHistory:
        insertions = tokens.is_target.nonzero(as_tuple=True)[0] + 1
        grown, is_query_token = _insert_at(
            tokens.embeddings, insertions, self.embedding
        )
        inserted_by_end = torch.searchsorted(
            insertions, tokens.cumulative_lens[1:], right=True
        )
        return TokenizedHistory(
            embeddings=grown,
            cumulative_lens=tokens.cumulative_lens
            + torch.cat([tokens.cumulative_lens.new_zeros(1), inserted_by_end]),
            is_target=_widen(tokens.is_target, is_query_token, insertions, False),
            item_ids=_widen(tokens.item_ids, is_query_token, insertions, 0),
            timestamps=_widen(
                tokens.timestamps,
                is_query_token,
                insertions,
                tokens.timestamps[insertions - 1],
            ),
            token_ids=(
                None
                if tokens.token_ids is None
                else _widen(tokens.token_ids, is_query_token, insertions, 0)
            ),
            is_query=_widen(
                torch.zeros_like(tokens.is_target),
                is_query_token,
                insertions,
                True,
            ),
        )


def _insert_at(
    values: torch.Tensor, starts: torch.Tensor, filler: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grown_total = values.shape[0] + starts.shape[0]
    is_bos = torch.zeros(grown_total, dtype=torch.bool, device=values.device)
    is_bos[starts + torch.arange(starts.shape[0], device=values.device)] = True

    grown = values.new_empty((grown_total, *values.shape[1:]))
    grown[is_bos] = filler.to(dtype=values.dtype)
    grown[~is_bos] = values
    return grown, is_bos


def _widen(
    values: torch.Tensor, is_bos: torch.Tensor, starts: torch.Tensor, fill
) -> torch.Tensor:
    grown = values.new_empty(is_bos.shape[0])
    grown[~is_bos] = values
    grown[is_bos] = values[starts] if fill is None else fill
    return grown


class ActionTokenizer(EventTokenizer):
    """Two tokens per event: the item, then the action taken on it."""

    def __init__(
        self,
        item_embedding: nn.Module,
        action_embedding: nn.Embedding,
        item_id_column: str,
        action_column: str,
    ):
        super().__init__(item_id_column)
        assert item_embedding.embedding_dim == action_embedding.embedding_dim, (
            "item and action tokens share one sequence, so they share a width"
        )
        self.item_embedding = item_embedding
        self.action_embedding = action_embedding
        self.action_column = action_column

    @property
    def tokens_per_event(self) -> int:
        return 2

    @property
    def out_dim(self) -> int:
        return self.item_embedding.embedding_dim

    def forward(self, batch: dict) -> TokenizedHistory:
        item_ids = self._column(batch, self.item_id_column).dense()
        actions = self._column(batch, self.action_column).dense()

        interleaved = torch.stack(
            [self.item_embedding(item_ids), self.action_embedding(actions)], dim=1
        )
        is_target = torch.tensor([True, False], device=item_ids.device).repeat(
            item_ids.shape[0]
        )
        return self._expand(batch, interleaved.flatten(0, 1), is_target)


class SemanticIdTokenizer(EventTokenizer):
    """One token per semantic-id level, coarse to fine."""

    def __init__(
        self,
        semantic_embedding: SemanticIdEmbedding | CombinedSemanticIdEmbedding,
        item_id_column: str,
        projection: nn.Linear | None = None,
    ):
        super().__init__(item_id_column)
        self.semantic_embedding = semantic_embedding
        self.projection = projection

    @property
    def tokens_per_event(self) -> int:
        return self.semantic_embedding.num_levels

    @property
    def out_dim(self) -> int:
        if self.projection is None:
            return self.semantic_embedding.level_dim
        return self.projection.out_features

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Vectors for raw vocabulary tokens, as a decoder needs mid-generation."""
        embedded = self.semantic_embedding.embed_tokens(tokens)
        return embedded if self.projection is None else self.projection(embedded)

    def forward(self, batch: dict) -> TokenizedHistory:
        item_ids = self._column(batch, self.item_id_column).dense()
        tokens = self.semantic_embedding.tokens(item_ids)
        per_level = self.embed_tokens(tokens)

        # A code tuple is predicted from its first level; the rest continue that
        # prediction rather than being targets of their own.
        is_target = torch.zeros_like(tokens, dtype=torch.bool)
        is_target[:, 0] = True
        return self._expand(
            batch,
            per_level.flatten(0, 1),
            is_target.flatten(),
            token_ids=tokens.flatten(),
        )
