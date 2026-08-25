"""Ranking variants: DCNv2 over counters and side features, with and without a
history transformer.

The first three read the same sequence batches and predict the same two
targets; they differ only in what the shared trunk is given. The last swaps the
data and the targets for the ranking homework's, to check the framework reaches
the numbers that setup is known to reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import polars as pl
import torch.nn as nn

from dcn.config.networks import build_causal_transformer, build_multi_head_dcn
from dcn.config.semantic import SemanticExperiment
from dcn.config.sequence import SequenceExperiment
from dcn.config.settings import TRANSFORMER, DayRangeConfig, EmbeddingConfig
from dcn.config.yambda_base import (
    NUM_HASHES,
    LikeFullPlayTargets,
    LikeListenTargets,
    YambdaSourceExperiment,
    like_field,
)
from dcn.data import DecayConfig, EmaCounter, FieldConfig
from dcn.datasets.yambda import HomeworkYambdaDatasetSource
from dcn.eval import PairwiseAccuracyCallback
from dcn.models import MultiHeadNetwork
from dcn.nn import ResNet1D
from dcn.nn.history_encoder import HistoryEncoder
from dcn.nn.semantic_embedding import SemanticIdEmbedding
from dcn.nn.types import ModuleWithDim

_SEMANTIC_EMBEDDING_DIM = 32


@dataclass
class RankingExperiment(LikeListenTargets, YambdaSourceExperiment, SequenceExperiment):
    """Multi-target ranking on yambda: like/not-like and listen ratio."""

    run_name: str = "ranking_dcn"

    embedding: EmbeddingConfig = field(
        default_factory=lambda: EmbeddingConfig(
            num_embeddings=2**20, dim=64, sparse=False
        )
    )

    def settings_defaults(self) -> dict[str, Any]:
        return {
            **super().settings_defaults(),
            "day_range": DayRangeConfig(start_day=0, end_day=299),
        }

    def create_counters(self) -> list[EmaCounter]:
        return [
            self.counter(keys=["uid"], fields=[like_field([7, 30])]),
            self.counter(keys=["item_id"], fields=[like_field([7, 30])]),
            self.counter(
                keys=["uid", "artist_id"],
                fields=[like_field([7])],
                aggregations=("mean", "max"),
            ),
        ]

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return (self.item_id_column, self.user_column, "album_id", "artist_id")

    @property
    def sequence_columns(self) -> list[str]:
        return [
            *self.categorical_features,
            *self.target_columns,
            *self.dataset_manager.dense_columns,
        ]

    def create_history_encoder(self, token_dim: int) -> HistoryEncoder | None:
        return None

    @cached_property
    def counter_encoder(self) -> ModuleWithDim | None:
        return self.fit_counter_encoder(self.dataset_manager.counter_columns)

    def create_feature_encoders(self) -> list[tuple[str, ModuleWithDim]]:
        return [(self.item_id_column, self.item_embeddings)]

    def extra_callbacks(self, train_days: list[int], val_days: list[int]) -> list[Any]:
        return [
            PairwiseAccuracyCallback(
                model=self.base_model,
                loader=self.make_sequence_loader(
                    val_days,
                    split="val",
                    batch_size=self.dataloader.val_batch_size,
                    shuffle=False,
                ),
                targets=self.pairwise_targets(),
                dtype=self.runtime.dtype,
            )
        ]

    def _create_model(self) -> MultiHeadNetwork:
        return build_multi_head_dcn(
            categorical_features=self.categorical_features,
            embedding=self.embedding,
            split_ratios=self.head_split_ratios,
            num_hashes=NUM_HASHES,
            task_names=self.task_names,
            feature_encoders=self.create_feature_encoders(),
            dense_feature_names=self.dataset_manager.dense_columns,
            num_counters=self.num_counters,
            dense_encoder=self.counter_encoder,
            history_encoder_factory=self.create_history_encoder,
        )


@dataclass
class RankingWithHistoryExperiment(RankingExperiment):
    """Ranking plus a causal transformer over the user's history."""

    run_name: str = "ranking_dcn_history"

    def create_history_encoder(self, token_dim: int) -> HistoryEncoder:
        return HistoryEncoder(
            projection=ResNet1D(
                input_dim=token_dim,
                hidden_dims=[TRANSFORMER.dim],
                norm_factory=nn.LayerNorm,
                dropout=TRANSFORMER.dropout,
            ),
            sequence_model=build_causal_transformer(
                TRANSFORMER, max_seq_len=self.max_seq_len
            ),
        )


@dataclass
class SemanticRankingExperiment(SemanticExperiment, RankingWithHistoryExperiment):
    """Ranking with history, where each event also carries its semantic id."""

    run_name: str = "ranking_dcn_semantic"

    def create_feature_encoders(self) -> list[tuple[str, ModuleWithDim]]:
        return [
            *super().create_feature_encoders(),
            (
                self.item_id_column,
                SemanticIdEmbedding.learned(
                    self.semantic_codes,
                    num_items=self.item_embeddings.num_known_ids,
                    embedding_dim=_SEMANTIC_EMBEDDING_DIM,
                ),
            ),
        ]


_FEEDBACK_CONDITIONS: dict[str, pl.Expr | None] = {
    "listen": None,
    "like": pl.col("target_like") == 1.0,
    "full_play": pl.col("target_full_play") == 1.0,
    "skip": pl.col("is_skip"),
}


def _feedback_fields(half_life_days: list[float]) -> list[FieldConfig]:
    """How each kind of feedback has been arriving lately."""
    return [
        FieldConfig(
            name=name,
            decays=[DecayConfig(half_life_days=days) for days in half_life_days],
            condition=condition,
        )
        for name, condition in _FEEDBACK_CONDITIONS.items()
    ]


@dataclass
class HomeworkRankingExperiment(LikeFullPlayTargets, RankingExperiment):
    """R1's model on the ranking homework's data: recommended listens only, both
    targets binary, and only the listens whose feedback differs from a
    neighbour's."""

    run_name: str = "ranking_dcn_homework"
    num_epochs: int = 1
    validation_days: int = 30

    dataset_source_class = HomeworkYambdaDatasetSource

    @property
    def row_filter(self) -> pl.Expr:
        return pl.col("is_preference_pair")

    def create_counters(self) -> list[EmaCounter]:
        return [
            self.counter(keys=["uid"], fields=_feedback_fields([7, 30])),
            self.counter(keys=["item_id"], fields=_feedback_fields([7, 30])),
            self.counter(keys=["uid", "item_id"], fields=_feedback_fields([7])),
        ]
