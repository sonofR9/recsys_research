from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .history_tokens import (
    EndOfHistoryQueryToken,
    EventTokenizer,
    InterleavedQueryToken,
    TokenizedHistory,
)
from dcn.nn.types import ModuleWithDim
from dcn.nn.layer_item_features import DirectAddItemFeature, LayerItemFeatureFusion

ClsTokenMode = Literal["none", "end_only", "interleaved"]


class SequenceRetrievalModel(nn.Module):
    """SASRec: a causal transformer over the history, scoring items by dot product."""

    def __init__(
        self,
        tokenizer: EventTokenizer,
        sequence_model: ModuleWithDim,
        item_embedding: nn.Embedding,
        item_id_column: str,
        layer_item_embeddings: list[nn.Embedding] | None = None,
        layer_item_feature_fusions: list[LayerItemFeatureFusion] | None = None,
        query_projection: nn.Linear | None = None,
        query_multiplier: float = 1.0,
        cls_token: bool = False,
        cls_token_mode: ClsTokenMode = "none",
        training_reverse_position_offset: int = 0,
    ):
        super().__init__()
        assert tokenizer.out_dim == sequence_model.out_dim, (
            f"tokenizer emits {tokenizer.out_dim}-wide tokens but the sequence"
            f" model reads {sequence_model.out_dim}"
        )
        if query_projection is None:
            assert (
                sequence_model.out_dim == item_embedding.embedding_dim
            ), "unprojected queries and items must share a width"
        else:
            assert query_projection.in_features == sequence_model.out_dim
            assert query_projection.out_features == item_embedding.embedding_dim
        self.tokenizer = tokenizer
        self.sequence_model = sequence_model
        self.item_embedding = item_embedding
        self.item_id_column = item_id_column
        self.layer_item_embeddings = nn.ModuleList(layer_item_embeddings or [])
        self.layer_item_feature_fusions = nn.ModuleList(
            layer_item_feature_fusions
            or [
                DirectAddItemFeature(embedding.embedding_dim)
                for embedding in self.layer_item_embeddings
            ]
        )
        if len(self.layer_item_embeddings) != len(self.layer_item_feature_fusions):
            raise ValueError(
                "layer item embeddings and fusions must have matching depths"
            )
        for embedding, fusion in zip(
            self.layer_item_embeddings, self.layer_item_feature_fusions
        ):
            if embedding.embedding_dim != fusion.feature_dim:
                raise ValueError(
                    "layer item embedding width must match its fusion feature width"
                )
        self.query_projection = query_projection
        self.query_multiplier = query_multiplier
        self.training_reverse_position_offset = training_reverse_position_offset
        if cls_token and cls_token_mode == "interleaved":
            raise ValueError(
                "cls_token=True selects end_only and conflicts with interleaved mode"
            )
        if cls_token_mode not in {"none", "end_only", "interleaved"}:
            raise ValueError(f"unknown CLS token mode {cls_token_mode!r}")
        effective_cls_token_mode = "end_only" if cls_token else cls_token_mode
        if effective_cls_token_mode != "none" and tokenizer.tokens_per_event != 1:
            raise ValueError("CLS queries require one token per event")
        self.end_of_history_query = (
            EndOfHistoryQueryToken(tokenizer.out_dim)
            if effective_cls_token_mode == "end_only"
            else None
        )
        self.interleaved_query = (
            InterleavedQueryToken(tokenizer.out_dim)
            if effective_cls_token_mode == "interleaved"
            else None
        )

    def _encode_tokens(
        self, tokens: TokenizedHistory, *, reverse_position_offset: int = 0
    ) -> torch.Tensor:
        layer_item_features = None
        layer_item_feature_mask = None
        if self.layer_item_embeddings:
            layer_item_features = [
                embedding(tokens.item_ids) for embedding in self.layer_item_embeddings
            ]
            if tokens.is_query is not None:
                special = tokens.is_query & ~tokens.is_target
                layer_item_feature_mask = ~special
        position_kwargs = (
            {"reverse_position_offset": reverse_position_offset}
            if reverse_position_offset
            else {}
        )
        hidden = self.query_multiplier * self.sequence_model(
            tokens.embeddings,
            tokens.cumulative_lens,
            tokens.timestamps,
            layer_item_features=layer_item_features,
            layer_item_feature_fusions=self.layer_item_feature_fusions or None,
            layer_item_feature_mask=layer_item_feature_mask,
            **position_kwargs,
        )
        return (
            hidden if self.query_projection is None else self.query_projection(hidden)
        )

    def encode_queries(self, batch: dict) -> torch.Tensor:
        tokens = self._tokens_for_inference(batch)
        return self._encode_tokens(tokens)

    def encode_cutoff_queries(self, batch: dict) -> torch.Tensor:
        tokens = self._tokens_for_inference(batch)
        hidden = self._encode_tokens(tokens)
        return hidden[tokens.cumulative_lens[1:] - 1]

    def _tokens_for_inference(self, batch: dict) -> TokenizedHistory:
        tokens = self.tokenizer(batch)
        if self.end_of_history_query is not None:
            return self.end_of_history_query.for_inference(tokens)
        if self.interleaved_query is not None:
            return self.interleaved_query(tokens)
        return tokens

    def encode_items(self, batch: dict) -> torch.Tensor:
        return self.item_embedding(batch["int_columns"][self.item_id_column].dense())

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(batch)
        if self.end_of_history_query is not None:
            tokens = self.end_of_history_query.for_training(tokens)
        elif self.interleaved_query is not None:
            tokens = self.interleaved_query(tokens)
        cumulative_lens = tokens.cumulative_lens
        output = {
            "query_repr": self._encode_tokens(
                tokens,
                reverse_position_offset=self.training_reverse_position_offset,
            ),
            "item_repr": self.item_embedding(tokens.item_ids),
            "item_ids": tokens.item_ids,
            "lengths": cumulative_lens.diff(),
            "is_target": tokens.is_target,
            "timestamps": tokens.timestamps,
        }
        if tokens.is_query is not None:
            output["is_query"] = tokens.is_query
        return output
