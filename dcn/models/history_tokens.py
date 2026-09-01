"""Turning a user's events into the token sequence a causal model reads."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn as nn

from dcn.data.features import FeatureValues

from dcn.nn.densenet import DenseNet
from dcn.nn.layer_item_features import ConcatenatedItemFeatureResidual
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
        is_query: torch.Tensor | None = None,
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
            is_query=is_query,
        )

    @staticmethod
    def _column(batch: dict, name: str) -> FeatureValues:
        return batch["int_columns"][name]


def item_encoder_dim(encoder: nn.Module) -> int:
    for name in ("out_dim", "embedding_dim"):
        dimension = getattr(encoder, name, None)
        if isinstance(dimension, int) and not isinstance(dimension, bool):
            return dimension
    raise TypeError("item encoder must declare out_dim or embedding_dim")


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
            item_encoder_dim(self.item_embedding)
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
        return self.tokenizer.tokens_per_event

    @property
    def out_dim(self) -> int:
        return self.tokenizer.out_dim

    def _normalized_deltas(self, tokens: TokenizedHistory) -> torch.Tensor:
        repeats = self.tokens_per_event
        if bool((tokens.cumulative_lens % repeats).any()):
            raise ValueError(
                "timestamp features must wrap event tokens before sequence-level"
                " tokens are added"
            )
        event_timestamps = tokens.timestamps[::repeats]
        deltas = torch.zeros_like(event_timestamps)
        if event_timestamps.numel() > 1:
            deltas[1:] = event_timestamps[1:] - event_timestamps[:-1]
        event_starts = (tokens.cumulative_lens[:-1] // repeats).long()
        deltas[event_starts] = 0
        deltas = deltas.clamp(min=0, max=self.max_delta_seconds).float()
        if self.kind in {"log", "bins"}:
            normalized = torch.log1p(deltas) / torch.log1p(
                deltas.new_tensor(self.max_delta_seconds)
            )
        else:
            normalized = deltas / self.max_delta_seconds
        return normalized.repeat_interleave(repeats)

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
            is_query=(
                None
                if tokens.is_query is None
                else _widen(tokens.is_query, is_bos, starts, True)
            ),
        )


class EndOfHistoryQueryToken(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.embedding, std=0.02)

    def for_training(self, tokens: TokenizedHistory) -> TokenizedHistory:
        starts = tokens.cumulative_lens[:-1]
        ends = tokens.cumulative_lens[1:]
        targets = tokens.is_target.nonzero(as_tuple=True)[0]
        if targets.numel() == 0:
            return tokens
        target_ranks = torch.searchsorted(targets, ends, right=False) - 1
        last_targets = targets[target_ranks.clamp_min(0)]
        has_target = (target_ranks >= 0) & (last_targets >= starts)
        eligible = has_target & (last_targets > starts)
        insertions = last_targets[eligible]
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


@dataclass(frozen=True)
class PackedQueryMemory:
    embeddings: torch.Tensor
    cumulative_lens: torch.Tensor
    is_query: torch.Tensor


class EndQuerySlots(nn.Module):
    def __init__(self, dim: int, *, num_slots: int = 4, shared: bool) -> None:
        super().__init__()
        if num_slots < 1:
            raise ValueError("num_slots must be positive")
        self.num_slots = num_slots
        self.embeddings = nn.Parameter(torch.zeros(1 if shared else num_slots, dim))
        nn.init.trunc_normal_(self.embeddings, std=0.02)

    def forward(self, tokens: TokenizedHistory) -> TokenizedHistory:
        lengths = tokens.cumulative_lens.diff()
        batch_size = lengths.shape[0]
        device = tokens.embeddings.device
        sequence_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device), lengths
        )
        history_positions = (
            torch.arange(tokens.embeddings.shape[0], device=device)
            + sequence_indices * self.num_slots
        )
        cumulative_lens = (
            tokens.cumulative_lens
            + torch.arange(
                batch_size + 1,
                device=device,
                dtype=tokens.cumulative_lens.dtype,
            )
            * self.num_slots
        )
        query_positions = (
            cumulative_lens[1:, None]
            - self.num_slots
            + torch.arange(self.num_slots, device=device)
        )
        total = tokens.embeddings.shape[0] + batch_size * self.num_slots

        embeddings = tokens.embeddings.new_empty(total, tokens.embeddings.shape[1])
        embeddings[history_positions] = tokens.embeddings
        embeddings[query_positions.flatten()] = (
            self.embeddings.expand(self.num_slots, -1)
            .repeat(batch_size, 1)
            .to(dtype=tokens.embeddings.dtype)
        )

        is_target = torch.zeros(total, dtype=torch.bool, device=device)
        is_target[history_positions] = tokens.is_target
        is_query = torch.zeros(total, dtype=torch.bool, device=device)
        is_query[query_positions.flatten()] = True

        item_ids = tokens.item_ids.new_zeros(total)
        item_ids[history_positions] = tokens.item_ids
        timestamps = tokens.timestamps.new_zeros(total)
        timestamps[history_positions] = tokens.timestamps
        nonempty = lengths > 0
        if tokens.timestamps.numel() > 0:
            last_timestamps = tokens.timestamps[
                (tokens.cumulative_lens[1:] - 1).clamp_min(0)
            ]
            query_timestamps = torch.where(
                nonempty, last_timestamps, torch.zeros_like(last_timestamps)
            )
            timestamps[query_positions.flatten()] = query_timestamps.repeat_interleave(
                self.num_slots
            )

        token_ids = None
        if tokens.token_ids is not None:
            token_ids = tokens.token_ids.new_zeros(total)
            token_ids[history_positions] = tokens.token_ids

        return TokenizedHistory(
            embeddings=embeddings,
            cumulative_lens=cumulative_lens,
            is_target=is_target,
            item_ids=item_ids,
            timestamps=timestamps,
            token_ids=token_ids,
            is_query=is_query,
        )

    def extract_memory(
        self,
        hidden: torch.Tensor,
        tokens: TokenizedHistory,
        *,
        include_history: bool,
        retained_query_slots: tuple[int, ...] | None = None,
    ) -> PackedQueryMemory:
        if hidden.shape[0] != tokens.embeddings.shape[0]:
            raise ValueError("hidden states and tokens must have matching lengths")
        if tokens.is_query is None:
            raise ValueError("tokens do not identify query slots")
        retained = (
            tuple(range(self.num_slots))
            if retained_query_slots is None
            else retained_query_slots
        )
        if (
            not retained
            or len(set(retained)) != len(retained)
            or any(slot < 0 or slot >= self.num_slots for slot in retained)
        ):
            raise ValueError("retained query slots must be unique valid slot indices")
        query_count = int(tokens.is_query.sum())
        batch_size = tokens.cumulative_lens.shape[0] - 1
        if query_count != batch_size * self.num_slots:
            raise ValueError("tokens do not contain the declared query slots")
        retain_lookup = torch.zeros(
            self.num_slots, dtype=torch.bool, device=hidden.device
        )
        retain_lookup[list(retained)] = True
        retained_queries = torch.zeros_like(tokens.is_query)
        retained_queries[tokens.is_query] = retain_lookup.repeat(batch_size)
        selected = retained_queries | (~tokens.is_query & include_history)
        lengths = tokens.cumulative_lens.diff()
        sequence_indices = torch.repeat_interleave(
            torch.arange(lengths.shape[0], device=hidden.device), lengths
        )
        memory_lengths = torch.zeros_like(lengths)
        memory_lengths.scatter_add_(
            0,
            sequence_indices,
            selected.to(dtype=memory_lengths.dtype),
        )
        cumulative_lens = torch.cat(
            [memory_lengths.new_zeros(1), memory_lengths.cumsum(0)]
        )
        return PackedQueryMemory(
            embeddings=hidden[selected],
            cumulative_lens=cumulative_lens,
            is_query=tokens.is_query[selected],
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
        assert (
            item_embedding.embedding_dim == action_embedding.embedding_dim
        ), "item and action tokens share one sequence, so they share a width"
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


class SemanticHistoryTokenizer(EventTokenizer):
    """RQ-KMeans history representations for concrete-item retrieval."""

    def __init__(
        self,
        semantic_embedding: SemanticIdEmbedding | CombinedSemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
        layout: Literal["event", "levels", "item_levels"],
        encoder: nn.Module,
        item_embedding: nn.Embedding | None = None,
        item_encoder: nn.Module | None = None,
        level_tags: nn.Embedding | None = None,
        residual_semantic_embedding: SemanticIdEmbedding | None = None,
        event_residual: ConcatenatedItemFeatureResidual | None = None,
    ) -> None:
        super().__init__(item_id_column)
        self.semantic_embedding = semantic_embedding
        self.model_dim = model_dim
        self.layout = layout
        self.encoder = encoder
        self.item_embedding = item_embedding
        self.item_encoder = item_encoder
        self.level_tags = level_tags
        self.residual_semantic_embedding = residual_semantic_embedding
        self.event_residual = event_residual

    @classmethod
    def learned_sid_event(
        cls,
        semantic_embedding: SemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
        encoder_hidden_dim: int,
    ) -> SemanticHistoryTokenizer:
        level_tags = nn.Embedding(
            semantic_embedding.num_levels, semantic_embedding.level_dim
        )
        return cls(
            semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="event",
            encoder=DenseNet(
                2 * semantic_embedding.num_levels * semantic_embedding.level_dim,
                model_dim,
                hidden_dim=encoder_hidden_dim,
            ),
            level_tags=level_tags,
        )

    @classmethod
    def item_frozen_sid_event(
        cls,
        item_embedding: nn.Embedding,
        semantic_embedding: SemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
        encoder_hidden_dim: int,
    ) -> SemanticHistoryTokenizer:
        return cls(
            semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="event",
            encoder=DenseNet(
                item_embedding.embedding_dim
                + semantic_embedding.num_levels * semantic_embedding.level_dim,
                model_dim,
                hidden_dim=encoder_hidden_dim,
            ),
            item_embedding=item_embedding,
        )

    @classmethod
    def item_learned_frozen_sid_event(
        cls,
        item_embedding: nn.Embedding,
        semantic_embedding: CombinedSemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
        encoder_hidden_dim: int,
    ) -> SemanticHistoryTokenizer:
        return cls(
            semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="event",
            encoder=DenseNet(
                item_embedding.embedding_dim
                + semantic_embedding.num_levels * semantic_embedding.level_dim,
                model_dim,
                hidden_dim=encoder_hidden_dim,
            ),
            item_embedding=item_embedding,
        )

    @classmethod
    def item_frozen_sid_learned_residual_event(
        cls,
        item_embedding: nn.Embedding,
        frozen_semantic_embedding: SemanticIdEmbedding,
        learned_semantic_embedding: SemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
        encoder_hidden_dim: int,
        residual_max_scale: float | None = None,
    ) -> SemanticHistoryTokenizer:
        if (
            frozen_semantic_embedding.num_levels
            != learned_semantic_embedding.num_levels
        ):
            raise ValueError("frozen and learned SID levels must match")
        learned_semantic_embedding.initializer_rng_nonadvancing = True
        encoder = DenseNet(
            item_embedding.embedding_dim
            + frozen_semantic_embedding.num_levels
            * frozen_semantic_embedding.level_dim,
            model_dim,
            hidden_dim=encoder_hidden_dim,
        )
        with torch.random.fork_rng(devices=[]):
            event_residual = ConcatenatedItemFeatureResidual(
                model_dim,
                learned_semantic_embedding.num_levels
                * learned_semantic_embedding.level_dim,
                max_scale=residual_max_scale,
            )
        return cls(
            frozen_semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="event",
            encoder=encoder,
            item_embedding=item_embedding,
            residual_semantic_embedding=learned_semantic_embedding,
            event_residual=event_residual,
        )

    @classmethod
    def learned_sid_tokens(
        cls,
        semantic_embedding: SemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
    ) -> SemanticHistoryTokenizer:
        return cls(
            semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="levels",
            encoder=_width_adapter(semantic_embedding.level_dim, model_dim),
        )

    @classmethod
    def learned_frozen_sid_tokens(
        cls,
        semantic_embedding: CombinedSemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
        encoder_hidden_dim: int,
    ) -> SemanticHistoryTokenizer:
        return cls(
            semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="levels",
            encoder=DenseNet(
                semantic_embedding.level_dim,
                model_dim,
                hidden_dim=encoder_hidden_dim,
            ),
        )

    @classmethod
    def frozen_sid_tokens(
        cls,
        semantic_embedding: SemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
        encoder_hidden_dim: int,
    ) -> SemanticHistoryTokenizer:
        return cls(
            semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="levels",
            encoder=DenseNet(
                semantic_embedding.level_dim,
                model_dim,
                hidden_dim=encoder_hidden_dim,
            ),
        )

    @classmethod
    def interleaved_item_sid_tokens(
        cls,
        item_embedding: nn.Embedding,
        semantic_embedding: SemanticIdEmbedding,
        item_id_column: str,
        *,
        model_dim: int,
    ) -> SemanticHistoryTokenizer:
        return cls(
            semantic_embedding,
            item_id_column,
            model_dim=model_dim,
            layout="item_levels",
            encoder=_width_adapter(semantic_embedding.level_dim, model_dim),
            item_embedding=item_embedding,
            item_encoder=_width_adapter(item_embedding.embedding_dim, model_dim),
        )

    @property
    def tokens_per_event(self) -> int:
        if self.layout == "event":
            return 1
        if self.layout == "levels":
            return self.semantic_embedding.num_levels
        return self.semantic_embedding.num_levels + 1

    @property
    def out_dim(self) -> int:
        return self.model_dim

    def forward(self, batch: dict) -> TokenizedHistory:
        item_ids = self._column(batch, self.item_id_column).dense()
        per_level = self.semantic_embedding.embed_tokens(
            self.semantic_embedding.tokens(item_ids)
        )
        embeddings = self._embeddings(item_ids, per_level)
        if self.event_residual is not None:
            assert self.residual_semantic_embedding is not None
            residual_per_level = self.residual_semantic_embedding.per_level(item_ids)
            embeddings = self.event_residual(
                embeddings,
                embeddings,
                residual_per_level.flatten(start_dim=1),
            )
        target_pattern = torch.zeros(
            self.tokens_per_event, dtype=torch.bool, device=item_ids.device
        )
        target_pattern[0] = True
        query_pattern = torch.zeros_like(target_pattern)
        query_pattern[-1] = True
        return self._expand(
            batch,
            embeddings,
            target_pattern.repeat(item_ids.shape[0]),
            is_query=query_pattern.repeat(item_ids.shape[0]),
        )

    def _embeddings(
        self, item_ids: torch.Tensor, per_level: torch.Tensor
    ) -> torch.Tensor:
        if self.layout == "event":
            parts = []
            if self.item_embedding is not None:
                parts.append(self.item_embedding(item_ids))
            parts.append(per_level.flatten(start_dim=1))
            if self.level_tags is not None:
                tags = self.level_tags.weight.unsqueeze(0).expand(
                    item_ids.shape[0], -1, -1
                )
                parts.append(tags.flatten(start_dim=1))
            return self.encoder(torch.cat(parts, dim=-1))
        semantic = self.encoder(per_level)
        if self.layout == "levels":
            return semantic.flatten(0, 1)
        assert self.item_embedding is not None and self.item_encoder is not None
        item = self.item_encoder(self.item_embedding(item_ids)).unsqueeze(1)
        return torch.cat([item, semantic], dim=1).flatten(0, 1)


def _width_adapter(input_dim: int, output_dim: int) -> nn.Module:
    return nn.Linear(input_dim, output_dim)
