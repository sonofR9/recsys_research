from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from dcn.config.networks import build_multi_head_dcn
from dcn.config.settings import DayRangeConfig, EmbeddingConfig, LoggingConfig
from dcn.config.yambda_base import (
    MULTI_TASK_SPLIT_RATIOS,
    NUM_HASHES,
    TASK_NAMES,
    LikeListenTargets,
    YambdaSourceExperiment,
    like_field,
)
from dcn.data import EmaCounter
from dcn.models import MultiHeadNetwork
from dcn.training import CombinedOptimizer

_CATEGORICAL_FEATURES = (
    "item_id",
    "compact_item_id",
    "uid",
    "album_id",
    "artist_id",
)
_PRECOMPUTED_FEATURE = "compact_item_id"


@dataclass
class YambdaLoggingConfig(LoggingConfig):
    prediction_int_columns: dict[str, str] = field(
        default_factory=lambda: {
            "uid": "int_columns.uid",
            "item_id": "int_columns.item_id",
            "timestamp": "timestamp",
        }
    )
    prediction_float_columns: dict[str, str] = field(
        default_factory=lambda: {
            "like_probability": "like_pred",
            "listen_pred": "listen_pred",
        }
    )


@dataclass
class YambdaDayRangeConfig(DayRangeConfig):
    end_day: int = 299


@dataclass
class YambdaExperiment(LikeListenTargets, YambdaSourceExperiment):
    """The original yambda run: multi-target DCNv2 over loose events, day by day."""

    run_name: str = "yambda_dcn"

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    embedding_learning_rate: float = 1e-2
    deep_learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def settings_defaults(self) -> dict[str, Any]:
        return {
            **super().settings_defaults(),
            "logging": YambdaLoggingConfig(),
            "day_range": YambdaDayRangeConfig(),
        }

    def create_counters(self) -> list[EmaCounter]:
        return [
            self.counter(keys=["uid"], fields=[like_field([7, 30])]),
            self.counter(keys=["item_id"], fields=[like_field([7, 30])]),
            self.counter(
                keys=["uid", "artist_id"],
                fields=[like_field([7, 30])],
                aggregations=("mean", "max"),
            ),
        ]

    def create_optimizers(self) -> torch.optim.Optimizer:
        # Its own optimizer, not a second parameter group: a sparse (or torchrec)
        # table needs one that can step the rows a batch touched.
        embedding = self.base_model.multi_task_embedding
        _, deep_params = self.split_parameters(self.base_model, self.embedding_types)
        return CombinedOptimizer(
            [
                torch.optim.Adam(
                    deep_params,
                    lr=self.deep_learning_rate,
                    weight_decay=self.weight_decay,
                    fused=self.runner_build_device.type == "cuda",
                ),
                self._create_embedding_optimizer(
                    embedding,
                    lr=self.embedding_learning_rate,
                    sparse=self.embedding.sparse,
                    weight_decay=self.weight_decay,
                ),
            ]
        )

    def _create_model(self) -> MultiHeadNetwork:
        return build_multi_head_dcn(
            categorical_features=_CATEGORICAL_FEATURES,
            embedding=self.embedding,
            split_ratios=MULTI_TASK_SPLIT_RATIOS,
            num_hashes=NUM_HASHES,
            task_names=TASK_NAMES,
            feature_encoders=[(_PRECOMPUTED_FEATURE, self.item_embeddings)],
            dense_feature_names=self.dataset_manager.dense_columns,
            num_counters=self.num_counters,
        )
